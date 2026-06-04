#!/usr/bin/env python3
"""Score a consolidation prompt against the ground-truth corpus (eval/corpus/labels.json).

No LLM grader — decisions are classified MECHANICALLY from the emitted ops and
compared to Anchor's hand-judged labels. The only cost is running the encoder.

Clusters are INJECTED by node_id: each labeled cluster is rebuilt via the decoder's
own `_enrich_clusters` (+ computed cosines + `_pre_classify`), NOT matched against a
live decode. This makes the corpus a FROZEN oracle — independent of decoder ranking
and the scan's suppression filter — so augmented hard clusters (locked, contradiction)
that the live scan would skip can still be scored, and a decoder-config change
(e.g. expanding suppression_relations) can't silently drop clusters from scoring.

Primary axis = MERGE confusion: a cluster either should-merge (absorb/split) or
should-keep. We count:
  - under_merge  : should-merge, didn't  → leaves duplicates (the silent miss)
  - over_merge   : should-keep, merged    → destroys distinct knowledge (irreversible)
  - correct      : matched the merge/keep call
Plus exact-action accuracy (absorb/split/keep) and, on absorbs, lossless rate
(content override + (id:) ref present). Borderline-labeled clusters never count
as wrong.

Tiers: each label carries tier ∈ {active, solved}; missing = active. By default only
'active' clusters run — 'solved' = both arms already nailed it (no A/B signal), frozen
to save cost. `--tier all` re-scores solved clusters as a regression check.

Runs BOTH arms (production baseline + candidate) × N samples — the encoder is
non-deterministic, so report the distribution. Emits a per-cluster × per-arm matrix
and a both-arms-correct freeze list (the clusters that could move to tier:solved).

Usage:
    ./dev python3 eval/score_consolidation_corpus.py --samples 3
    ./dev python3 eval/score_consolidation_corpus.py --tier all --samples 3
    ./dev python3 eval/score_consolidation_corpus.py --arm candidate --save eval/reports/corpus_score.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

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


def _pair_cosines(brain, ids):
    """Max pairwise content (_primary) and title cosine for a set of node ids.

    Replicates the decoder's embedding-load so injected/augmented clusters get the
    SAME cosines the decoder would compute. Returns (content_max, title_max)."""
    if len(ids) < 2:
        return 0.0, 0.0
    ph = ','.join('?' * len(ids))

    def load(vtype):
        vecs = {}
        for nid, emb in brain.conn.execute(
                "SELECT node_id, embedding FROM node_enrichments "
                "WHERE node_id IN (%s) AND vector_type = ? "
                "AND embedding IS NOT NULL AND typeof(embedding) = 'blob'" % ph,
                (*ids, vtype)).fetchall():
            v = np.frombuffer(emb, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                vecs[nid] = v / n
        return vecs

    def maxcos(vecs):
        present = [i for i in ids if i in vecs]
        m = 0.0
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                m = max(m, float(vecs[present[i]] @ vecs[present[j]]))
        return m

    return maxcos(load('_primary')), maxcos(load('title'))


def build_cluster(decoder, brain, meta):
    """Rebuild a labeled corpus cluster as the enriched cluster dict the encoder
    expects — by node_id, bypassing the live decode + scan suppression."""
    ids = sorted(meta['node_ids'])
    cc = meta.get('content_cosine')
    tc = meta.get('title_cosine')
    if cc is None or tc is None:
        cc, tc = _pair_cosines(brain, ids)
    cluster = {
        'nodes': ids, 'size': len(ids),
        'content_cosine_max': cc, 'content_cosine_avg': cc,
        'title_cosine_max': tc, 'title_cosine_avg': tc,
        'pair_scores': {},
    }
    decoder._enrich_clusters([cluster])      # mutates in place
    cluster['pre_class'] = decoder._pre_classify(cluster)
    return cluster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--arm', choices=['both', 'candidate', 'baseline'], default='both')
    ap.add_argument('--tier', choices=['active', 'solved', 'all'], default='active',
                    help="which corpus tier to score (default: active)")
    ap.add_argument('--save')
    args = ap.parse_args()

    from tests.isolated_brain import IsolatedBrain
    from eval.s2_consolidation_eval import run_capture_variant
    from servers.scales.s2.consolidation_decoder import ConsolidationDecoder

    clusters_meta = {str(c['cluster_id']): c for c in json.load(open(CLUSTERS))}
    labels = json.load(open(LABELS))['labels']

    def in_scope(cid):
        tier = labels[cid].get('tier', 'active')
        return args.tier == 'all' or tier == args.tier

    scoped = [cid for cid in labels if cid in clusters_meta and in_scope(cid)]

    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        baseline_prompt = brain.get_interaction_prompt(INTERACTION)
        candidate_prompt = open(CANDIDATE).read().strip()

        decoder = ConsolidationDecoder(brain)
        built = {}
        for cid in scoped:
            try:
                built[cid] = build_cluster(decoder, brain, clusters_meta[cid])
            except Exception as e:
                print('  ⚠ skip cluster %s (%s): %r' % (
                    cid, clusters_meta[cid].get('node_ids'), e))
        cluster_list = list(built.values())
        print('Built %d/%d scoped clusters (tier=%s) by injection.' % (
            len(built), len(scoped), args.tier))

        arms = {'baseline': baseline_prompt, 'candidate': candidate_prompt}
        if args.arm != 'both':
            arms = {args.arm: arms[args.arm]}

        results = {}
        for arm, prompt in arms.items():
            seen = {cid: [] for cid in built}
            for _ in range(args.samples):
                var = run_capture_variant(brain, cluster_list, prompt)
                for cid, cl in built.items():
                    act, merged, loss = classify(ops_for(var['ops'], cl['nodes']))
                    seen[cid].append({'action': act, 'merged': merged, 'loss': loss})
            results[arm] = seen

    # ── score ──
    def score_arm(seen):
        rows = []
        conf = Counter()
        verdicts = {}
        action_hits = action_total = 0
        lossless_good = lossless_total = 0
        for cid in built:
            lab = labels[cid]
            samples = seen.get(cid, [])
            if not samples:
                continue
            merged_frac = sum(1 for s in samples if s['merged']) / len(samples)
            merged_majority = merged_frac >= 0.5
            act = Counter(s['action'] for s in samples).most_common(1)[0][0]
            exp_action = lab['action']
            exp_merge = lab['merge_expected']
            borderline = lab.get('confidence') == 'borderline'
            for s in samples:
                if s['loss']:
                    lossless_good += s['loss'][0]
                    lossless_total += s['loss'][1]
            if borderline:
                verdict = 'borderline_ok'
            elif exp_merge and not merged_majority:
                verdict = 'UNDER_merge'
            elif (not exp_merge) and merged_majority:
                verdict = 'OVER_merge'
            else:
                verdict = 'correct'
            conf[verdict] += 1
            verdicts[cid] = verdict
            if not borderline:
                action_total += 1
                ok = (act == exp_action) or (exp_action == 'keep' and act in ('keep', 'none'))
                if ok:
                    action_hits += 1
            aug = '*' if lab.get('augmented') else ''
            rows.append({'cluster': cid + aug, 'pre_class': built[cid].get('pre_class', '?'),
                         'expected': exp_action + ('?' if borderline else ''),
                         'got': act, 'merged_pct': round(merged_frac * 100),
                         'verdict': verdict})
        return rows, conf, verdicts, action_hits, action_total, (lossless_good, lossless_total)

    out = {}
    verdicts_by_arm = {}
    for arm, seen in results.items():
        rows, conf, verdicts, ah, at, loss = score_arm(seen)
        verdicts_by_arm[arm] = verdicts
        out[arm] = {'conf': dict(conf), 'action_acc': '%d/%d' % (ah, at),
                    'lossless': '%d/%d' % loss, 'per_cluster': rows}
        print('\n=== %s  (samples=%d) ===' % (arm.upper(), args.samples))
        print('  %-5s %-18s %-10s %-7s %-7s %s' % (
            'clu', 'pre_class', 'expected', 'got', 'merged', 'verdict'))
        for r in rows:
            flag = '' if r['verdict'] in ('correct', 'borderline_ok') else '  <<'
            print('  %-5s %-18s %-10s %-7s %3d%%    %s%s' % (
                r['cluster'], r['pre_class'], r['expected'], r['got'],
                r['merged_pct'], r['verdict'], flag))
        print('  merge-axis: %s' % dict(conf))
        print('  exact-action acc (non-borderline): %d/%d   absorb-lossless: %d/%d' % (
            ah, at, loss[0], loss[1]))

    # ── both-arms-correct freeze list (no A/B signal → candidates for tier:solved) ──
    if len(verdicts_by_arm) == 2:
        b, c = verdicts_by_arm['baseline'], verdicts_by_arm['candidate']
        both_correct = sorted((cid for cid in b
                               if b[cid] == 'correct' and c.get(cid) == 'correct'),
                              key=int)
        out['both_correct_freeze_candidates'] = both_correct
        print('\n── FREEZE CANDIDATES (both arms correct, no A/B signal) ──')
        print('  %s' % (both_correct or '(none)'))
        print('  → move to tier:solved to skip in future runs (keep augmented/borderline active).')

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        json.dump(out, open(args.save, 'w'), indent=2)
        print('\nsaved', args.save)


if __name__ == '__main__':
    main()
