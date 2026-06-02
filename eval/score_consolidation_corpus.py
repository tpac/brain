#!/usr/bin/env python3
"""Score a consolidation prompt against the ground-truth corpus (eval/corpus/labels.json).

No LLM grader — decisions are classified MECHANICALLY from the emitted ops and
compared to Anchor's hand-judged labels. The only cost is running the encoder.

Primary axis = MERGE confusion: a cluster either should-merge (absorb/split) or
should-keep. We count:
  - under_merge  : should-merge, didn't  → leaves duplicates (the silent miss)
  - over_merge   : should-keep, merged    → destroys distinct knowledge (irreversible)
  - correct      : matched the merge/keep call
Plus exact-action accuracy (absorb/split/keep) and, on absorbs, lossless rate
(content override + (id:) ref present). Borderline-labeled clusters never count
as wrong.

Runs BOTH arms (production baseline + candidate) × N samples — the encoder is
non-deterministic, so report the distribution.

Usage:
    ./dev python3 eval/score_consolidation_corpus.py --samples 3
"""
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CLUSTERS = os.path.join(ROOT, 'eval', 'corpus', 'clusters.json')
LABELS = os.path.join(ROOT, 'eval', 'corpus', 'labels.json')
CANDIDATE = os.path.join(ROOT, 'eval', 'candidate_prompts', 's2_consolidation_absorb.md')
INTERACTION = 's2_consolidation_enrichment'


def classify(ops):
    """Classify the action a set of cluster-ops represents."""
    absorbs = [o for o in ops if o.get('op') == 'absorb']
    archives = [o for o in ops if o.get('op') == 'archive']
    similar = [o for o in ops if o.get('op') == 'connect'
               and o.get('relation') == 'similar_to']
    merged = bool(absorbs or archives)
    if merged:
        # distinct survivors: absorb.survivor_id; revise+archive → the non-archived node
        survivors = {o.get('survivor_id') for o in absorbs if o.get('survivor_id')}
        action = 'split' if len(survivors) >= 2 else 'absorb'
    elif similar:
        action = 'keep'
    else:
        action = 'none'
    # lossless signal on absorbs
    loss = None
    if absorbs:
        good = sum(1 for o in absorbs
                   if (o.get('content') or '').strip() and '(id:' in (o.get('content') or ''))
        loss = (good, len(absorbs))
    return action, merged, loss


def ops_for(ops, cluster_ids):
    s = set(cluster_ids)
    out = []
    for o in ops:
        ids = {o.get('survivor_id'), o.get('absorbed_id'), o.get('node_id'),
               o.get('source_id'), o.get('target_id')}
        if ids & s:
            out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--arm', choices=['both', 'candidate', 'baseline'], default='both')
    ap.add_argument('--save')
    args = ap.parse_args()

    from tests.isolated_brain import IsolatedBrain
    from eval.s2_consolidation_eval import run_decoder, run_capture_variant

    clusters_meta = json.load(open(CLUSTERS))
    labels = json.load(open(LABELS))['labels']
    # map sorted node_ids tuple -> (corpus cluster_id, label)
    by_ids = {tuple(sorted(c['node_ids'])): (str(c['cluster_id']), c) for c in clusters_meta}

    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        baseline_prompt = brain.get_interaction_prompt(INTERACTION)
        candidate_prompt = open(CANDIDATE).read().strip()
        decode = run_decoder(brain)
        # keep only clusters that are in our labeled corpus
        live = [c for c in decode.get('clusters', [])
                if tuple(sorted(c.get('nodes', []))) in by_ids]
        print('Matched %d/%d labeled clusters in the live decode.' % (len(live), len(labels)))

        arms = {'baseline': baseline_prompt, 'candidate': candidate_prompt}
        if args.arm != 'both':
            arms = {args.arm: arms[args.arm]}

        results = {}
        for arm, prompt in arms.items():
            # accumulate per-corpus-cluster the actions seen across samples
            seen = {cid: [] for cid in labels}
            for s in range(args.samples):
                var = run_capture_variant(brain, live, prompt)
                for c in live:
                    cid, _meta = by_ids[tuple(sorted(c.get('nodes', [])))]
                    act, merged, loss = classify(ops_for(var['ops'], c.get('nodes', [])))
                    seen[cid].append({'action': act, 'merged': merged, 'loss': loss})
            results[arm] = seen

    # ── score ──
    def score_arm(seen):
        # majority action per cluster across samples (mode); merge if majority merged
        rows = []
        conf = Counter()  # under_merge / over_merge / correct / borderline_ok
        action_hits = 0
        action_total = 0
        lossless_good = lossless_total = 0
        for cid, lab in labels.items():
            samples = seen.get(cid, [])
            if not samples:
                continue
            merged_frac = sum(1 for s in samples if s['merged']) / len(samples)
            merged_majority = merged_frac >= 0.5
            # dominant action
            act = Counter(s['action'] for s in samples).most_common(1)[0][0]
            exp_action = lab['action']
            exp_merge = lab['merge_expected']
            borderline = lab.get('confidence') == 'borderline'
            # lossless on absorbs
            for s in samples:
                if s['loss']:
                    lossless_good += s['loss'][0]; lossless_total += s['loss'][1]
            # merge-axis verdict
            if borderline:
                verdict = 'borderline_ok'
            elif exp_merge and not merged_majority:
                verdict = 'UNDER_merge'
            elif (not exp_merge) and merged_majority:
                verdict = 'OVER_merge'
            else:
                verdict = 'correct'
            conf[verdict] += 1
            # exact action accuracy (split vs absorb vs keep), borderline excluded
            if not borderline:
                action_total += 1
                # treat split-expected satisfied by split; absorb by absorb; keep by keep/none
                ok = (act == exp_action) or (exp_action == 'keep' and act in ('keep', 'none'))
                if ok:
                    action_hits += 1
            rows.append((cid, exp_action + ('*' if borderline else ''),
                         act, '%.0f%%' % (merged_frac * 100), verdict))
        return rows, conf, action_hits, action_total, (lossless_good, lossless_total)

    out = {}
    for arm, seen in results.items():
        rows, conf, ah, at, loss = score_arm(seen)
        out[arm] = {'conf': dict(conf), 'action_acc': '%d/%d' % (ah, at),
                    'lossless': '%d/%d' % loss}
        print('\n=== %s  (samples=%d) ===' % (arm.upper(), args.samples))
        print('  %-4s %-10s %-8s %-7s %s' % ('clu', 'expected', 'got', 'merged', 'verdict'))
        for cid, exp, act, mf, v in rows:
            flag = '' if v in ('correct', 'borderline_ok') else '  <<'
            print('  %-4s %-10s %-8s %-7s %s%s' % (cid, exp, act, mf, v, flag))
        print('  merge-axis: %s' % dict(conf))
        print('  exact-action acc (non-borderline): %d/%d   absorb-lossless: %d/%d' % (
            ah, at, loss[0], loss[1]))

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        json.dump(out, open(args.save, 'w'), indent=2)
        print('\nsaved', args.save)


if __name__ == '__main__':
    main()
