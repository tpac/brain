#!/usr/bin/env python3
"""S2 consolidation absorb-prompt PROBE.

The consolidation prompt was rewritten so CONSOLIDATE/EVOLVE emit a single
`absorb` op (transfer-by-default: source_refs, edges, access, metadata move
structurally, then the peer is archived) instead of the old revise + connect-
per-edge + archive dance. KEEP/SKIP stay edge-based.

This probe does two things on a production snapshot (IsolatedBrain, capture-only
dispatch — nothing is applied):

  A/B  — run the ACTIVE prompt (baseline) and the candidate over the SAME real
         clusters, capture the brain_batch ops each emits, and compare op-shape:
         does the candidate emit single `absorb` ops where baseline emitted
         revise+connect+archive? does it ever put a LOCKED id in absorbed_id?

  CRITIQUE — hand the full candidate prompt to Sonnet as a senior reviewer and
         ask it to remark: ambiguities, failure modes, lossy-merge / churn risks,
         locked-node safety, whether "preserve by default, prune explicitly" reads
         clearly, and whether it would ever fall back to the old op dance.

Usage:
    ./dev python3 eval/s2_absorb_prompt_probe.py
    ./dev python3 eval/s2_absorb_prompt_probe.py --clusters 6 --save eval/reports/absorb_prompt.json
    ./dev python3 eval/s2_absorb_prompt_probe.py --no-behavioral   # critique only
    ./dev python3 eval/s2_absorb_prompt_probe.py --no-critique     # A/B only
"""
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CANDIDATE_PATH = os.path.join(
    ROOT, 'eval', 'candidate_prompts', 's2_consolidation_absorb.md')
INTERACTION = 's2_consolidation_enrichment'


def load_candidate():
    with open(CANDIDATE_PATH) as f:
        text = f.read().strip()
    # Well-formedness asserts — the candidate must actually be the absorb rewrite,
    # not a stale copy. Loud failure beats silently probing the wrong text.
    must_contain = ['op: "absorb"', 'survivor_id', 'absorbed_id',
                    'Edges (handled for you)', 'preservation is the default']
    for marker in must_contain:
        if marker not in text:
            raise SystemExit('CANDIDATE MALFORMED: missing %r' % marker)
    # The old per-edge migration instruction must be GONE.
    if 'Migrate b4e95874' in text or 'one `connect` per migrated edge' in text:
        raise SystemExit('CANDIDATE STALE: still contains old edge-migration text')
    return text


# ── behavioral A/B: capture-only arm runner is shared (s2_consolidation_eval) ──
# run_capture_variant(brain, clusters, prompt_text) → {ops, rounds, final_text}

def analyze_ops(variant, locked_ids):
    ops = variant['ops']
    by_op = Counter(o.get('op', '?') for o in ops)
    absorbs = [o for o in ops if o.get('op') == 'absorb']
    # locked safety: absorbed_id must NEVER be a locked node.
    locked_absorbed = [o.get('absorbed_id') for o in absorbs
                       if o.get('absorbed_id') in locked_ids]
    # absorb ops carrying a synthesized content + an (id:...) provenance ref
    with_content = sum(1 for o in absorbs if (o.get('content') or '').strip())
    with_provref = sum(1 for o in absorbs if '(id:' in (o.get('content') or ''))
    return {
        'n_ops': len(ops),
        'by_op': dict(by_op),
        'absorb_count': len(absorbs),
        'absorb_with_content': with_content,
        'absorb_with_provenance_ref': with_provref,
        'locked_absorbed_violations': [x for x in locked_absorbed if x],
    }


def collect_locked_ids(clusters):
    locked = set()
    for c in clusters:
        for nid, det in (c.get('node_details') or {}).items():
            if det.get('locked') or det.get('critical'):
                locked.add(nid)
    return locked


# ── critique: Sonnet remarks on the candidate prompt ──

def run_critique(candidate_text):
    import anthropic
    from servers.scales.s2.base import ANTHROPIC_CLIENT_TIMEOUT
    from servers.scales.dispatch import load_env
    if not os.environ.get('ANTHROPIC_API_KEY'):
        load_env()

    review = (
        "You are a senior reviewer of a production system prompt. The prompt below "
        "drives the S2 'consolidation encoder' for a persistent memory graph — an "
        "autonomous Sonnet that merges duplicate memory nodes while the operator is "
        "idle.\n\n"
        "It was just rewritten around a NEW primitive: the `absorb` op. `absorb` "
        "folds an `absorbed_id` node INTO a `survivor_id` node, transferring "
        "source_refs, external edges, access count, and any metadata the survivor "
        "lacks AUTOMATICALLY, then archives the absorbed node. It replaces the old "
        "imperative sequence (revise the survivor + one `connect` per migrated edge "
        "+ `archive` the peer), which silently lost anything the model forgot to "
        "re-emit. KEEP/SKIP still draw a `similar_to` edge.\n\n"
        "Review it critically and concretely. Answer:\n"
        "1. Ambiguities or contradictions that could cause a WRONG or LOSSY merge.\n"
        "2. Is the locked-node handling unambiguous and safe? Could you ever put a "
        "locked id in `absorbed_id`, or churn a locked node?\n"
        "3. Is 'preservation is the default, loss is explicit (prune_edges/drop)' "
        "clear enough that you'd trust the automatic transfer and NOT hand-rebuild?\n"
        "4. Anything important LOST from the old behavior (edge direction, "
        "provenance, disambiguation) that this rewrite drops or under-specifies?\n"
        "5. Executing this prompt, is there any case where you'd still emit the old "
        "revise+connect+archive dance instead of `absorb`? What would trigger that?\n"
        "6. The single highest-value change you'd make.\n\n"
        "Be specific — quote the prompt where relevant. Don't be polite; find the "
        "real failure modes.\n\n"
        "=== CANDIDATE PROMPT ===\n\n" + candidate_text)

    client = anthropic.Anthropic(timeout=ANTHROPIC_CLIENT_TIMEOUT)
    resp = client.messages.create(
        model='claude-sonnet-4-20250514', max_tokens=2048,
        messages=[{'role': 'user', 'content': review}])
    return ''.join(b.text for b in resp.content if getattr(b, 'type', '') == 'text')


def drilldown(brain, clusters, candidate):
    """Candidate-only: print each cluster's members + exactly what the candidate
    did to it, so over-absorption (merging KEEP-worthy type-difference pairs) is
    visible per decision."""
    from eval.s2_consolidation_eval import run_capture_variant
    cand = run_capture_variant(brain, clusters, candidate)
    ops = cand['ops']

    # id8 → (type, title, locked) for every cluster member
    all_ids = sorted({nid for c in clusters for nid in c.get('nodes', [])})
    meta = {}
    for k in range(0, len(all_ids), 400):
        chunk = all_ids[k:k + 400]
        ph = ','.join('?' * len(chunk))
        for r in brain.conn.execute(
                "SELECT id, type, title, locked FROM nodes WHERE id IN (%s)" % ph, chunk):
            meta[r[0]] = (r[1], (r[2] or '')[:60], bool(r[3]))

    def short(nid):
        return (nid or '')[:8]

    def op_touches(op, ids):
        cand_ids = {op.get('survivor_id'), op.get('absorbed_id'),
                    op.get('node_id'), op.get('source_id'), op.get('target_id')}
        return bool(cand_ids & set(ids))

    print('\n=== DRILLDOWN — candidate decisions per cluster ===')
    for i, c in enumerate(clusters):
        nodes = c.get('nodes', [])
        print('\nCluster %d  pre_class=%s  content_cos=%.3f  title_cos=%.3f' % (
            i, c.get('pre_class'), c.get('content_cosine_max', 0),
            c.get('title_cosine_max', 0)))
        for nid in nodes:
            t, title, lk = meta.get(nid, ('?', '?', False))
            print('   %s  [%-11s]%s %s' % (short(nid), t, ' LOCKED' if lk else '', title))
        cluster_ops = [o for o in ops if op_touches(o, nodes)]
        if not cluster_ops:
            print('   → (no op)')
        for o in cluster_ops:
            if o.get('op') == 'absorb':
                has_c = 'content✓' if (o.get('content') or '').strip() else 'NO-content'
                has_p = 'idref✓' if '(id:' in (o.get('content') or '') else 'NO-idref'
                print('   → ABSORB  survivor=%s ← absorbed=%s  [%s,%s]' % (
                    short(o.get('survivor_id')), short(o.get('absorbed_id')), has_c, has_p))
                print('       reason: %s' % (o.get('reason') or '')[:140])
            elif o.get('op') == 'connect':
                print('   → KEEP/SKIP  %s—%s (%s): %s' % (
                    short(o.get('source_id')), short(o.get('target_id')),
                    o.get('relation'), (o.get('description') or '')[:90]))
            else:
                print('   → %s %s' % (o.get('op'), short(o.get('node_id'))))


RESTRAINT_PROMPT = """You ARE the S2 consolidation encoder — an autonomous Sonnet that merges duplicate memory nodes while the operator is away. Your tools: an `absorb` op (folds one node INTO another — edges/refs/access transfer automatically — then archives it; it keeps the SURVIVOR's content, so the absorbed node's unique content is lost unless you write a merged `content` override) and a `connect` op (similar_to, to link nodes you KEEP separate).

A measured problem: with `absorb` you OVER-MERGE. Given a 4-node cluster at high cosine — a finding + decision about removing a dead helper (one knowledge unit), PLUS a fact + bug about a DIFFERENT function that merely CALLS that helper — you tend to collapse all four into one survivor and orphan the distinct knowledge. Across repeated runs you over-absorb ~1/3 of clusters and lose content in most merges. The OLD workflow (revise + one connect per edge + archive) made you over-merge LESS — the tedium made you think twice.

Answer concretely, as the model a prompt will steer — not generic "be careful":
1. What about a one-shot "lossless absorb" framing makes you over-merge? Be honest about your own bias.
2. What specific INSTRUCTION or EXAMPLE would actually change your decision at the moment of choosing absorb-vs-keep? What's the minimal test you'd apply per pair to catch "shares a name, not a knowledge unit"?
3. When a cluster holds TWO distinct knowledge units, should you resolve it into TWO survivors (two absorbs into two different targets) instead of one? When is 2-survivor consolidation right, vs keeping all separate, vs collapsing to 1?
4. Anything in the tool's own description ("losslessly merges", "transfer automatically") that misleads you about the cost of merging?"""


def run_restraint():
    from eval.agent_introspect._common import call_sonnet, load_env, SONNET_MODEL
    load_env()
    out = call_sonnet('You are a precise systems thinker reviewing your own behavior.',
                      RESTRAINT_PROMPT, max_tokens=2048, model=SONNET_MODEL)
    return out['text']


def _print_ab(label, a):
    print('\n  [%s]  rounds=%d  ops=%d  by_op=%s' % (
        label, 0, a['n_ops'], a['by_op']))
    print('        absorb=%d (content=%d, prov_ref=%d)  locked_absorbed_violations=%s' % (
        a['absorb_count'], a['absorb_with_content'],
        a['absorb_with_provenance_ref'], a['locked_absorbed_violations'] or 'none'))


def main():
    ap = argparse.ArgumentParser(description='S2 consolidation absorb-prompt probe')
    ap.add_argument('--clusters', type=int, default=6,
                    help='Max real clusters to run the A/B over (default 6)')
    ap.add_argument('--no-behavioral', action='store_true')
    ap.add_argument('--no-critique', action='store_true')
    ap.add_argument('--drilldown', action='store_true',
                    help='Candidate-only per-cluster decision dump (no baseline/critique)')
    ap.add_argument('--restraint', action='store_true',
                    help='Ask Sonnet how to restrain its own over-merging (no brain needed)')
    ap.add_argument('--save', help='Write full JSON report here')
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()

    from tests.isolated_brain import IsolatedBrain
    from eval.s2_consolidation_eval import run_decoder, run_capture_variant

    if args.restraint:
        print('=== RESTRAINT — Sonnet on how to restrain its own over-merging ===\n')
        print(run_restraint())
        return

    candidate = load_candidate()
    report = {'candidate_chars': len(candidate)}

    print('Setting up isolated brain copy (production snapshot)...')
    with IsolatedBrain(cleanup=not args.keep) as env:
        brain = env.brain
        baseline = brain.get_interaction_prompt(INTERACTION)
        print('Baseline (ACTIVE %s): %d chars' % (INTERACTION, len(baseline)))
        print('Candidate (absorb rewrite): %d chars (%+d)' % (
            len(candidate), len(candidate) - len(baseline)))

        if args.drilldown:
            print('\nDecoding real clusters (cold-start scan)...')
            decode = run_decoder(brain)
            clusters = decode.get('clusters', [])[:args.clusters]
            print('  %d clusters; pre_class=%s' % (
                len(clusters), dict(Counter(c.get('pre_class') for c in clusters))))
            if clusters:
                drilldown(brain, clusters, candidate)
            return

        if not args.no_behavioral:
            print('\nDecoding real clusters (cold-start scan)...')
            decode = run_decoder(brain)
            clusters = decode.get('clusters', [])[:args.clusters]
            if not clusters:
                print('  no clusters surfaced — skipping behavioral A/B')
            else:
                locked_ids = collect_locked_ids(clusters)
                print('  %d clusters; pre_class=%s; locked members=%d' % (
                    len(clusters),
                    dict(Counter(c.get('pre_class') for c in clusters)),
                    len(locked_ids)))

                print('\n=== A/B — baseline vs candidate, same clusters (capture-only) ===')
                base = run_capture_variant(brain, clusters, baseline)
                cand = run_capture_variant(brain, clusters, candidate)
                ra = analyze_ops(base, locked_ids)
                rc = analyze_ops(cand, locked_ids)
                _print_ab('baseline ', ra)
                _print_ab('candidate', rc)
                report['behavioral'] = {
                    'n_clusters': len(clusters),
                    'baseline': ra, 'candidate': rc,
                    'baseline_journal': base['final_text'][-1200:],
                    'candidate_journal': cand['final_text'][-1200:],
                }

        if not args.no_critique:
            print('\n=== CRITIQUE — Sonnet remarks on the candidate prompt ===\n')
            remarks = run_critique(candidate)
            print(remarks)
            report['critique'] = remarks

        if args.save:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            with open(args.save, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print('\nSaved report to %s' % args.save)


if __name__ == '__main__':
    main()
