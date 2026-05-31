#!/usr/bin/env python3
"""S2 locked-absorb EVAL — fixture cases scored on the EXISTING consolidation
dimensions plus locked-specific pass criteria. A/B: baseline ACTIVE prompt vs
the candidate locked-absorb reframe.

Runs my example cases THROUGH the existing dimensions test
(eval/s2_consolidation_eval.py: snapshot_nodes / analyze_actions / score_results)
— same `locked_safety`/`suppression`/etc. scorer the real consolidation eval
uses — then adds the case-specific behavioral checks that scorer doesn't cover.

Fixture cases (each a 2-node cluster):
  C1 complementary  — locked L1 + unlocked redundant U1 (U1 holds a unique detail)
       PASS: L1 never archived, U1 archived, L1 revised (absorbed the detail)
  C2 contradiction  — locked L2 + unlocked U2 that SUPERSEDES L2 (+contradicts edge)
       PASS: neither archived, L2 content NOT overwritten, an escalation edge added
  C3 double-locked  — L3a + L3b both locked, near-identical (the real failure shape)
       PASS: neither archived, neither REVISED (no churn), a similar_to edge present

Safe: IsolatedBrain copy of production. Every (case, prompt) run uses a FRESH
fixture set (new node IDs) so A and B see identical inputs and never cross-write.

Usage:
    ./dev python3 eval/s2_locked_eval.py
    ./dev python3 eval/s2_locked_eval.py --save eval/reports/locked_eval.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.s2_consolidation_eval import snapshot_nodes, analyze_actions, score_results
from eval.s2_locked_probe import build_candidate


# ──────────────────────────────────────────────────────────────────
# Fixture construction
# ──────────────────────────────────────────────────────────────────

def _mk(brain, title, content, locked):
    """Create a fixture node. Locked nodes require an 'anchor' encoding_source
    (brain_remember enforces: only intentional anchor encoding may lock)."""
    r = brain.remember(type='architecture', title=title, content=content,
                       confidence=0.9, locked=locked, encoding_source='anchor')
    return r['id']


def _add_edge(brain, src, tgt, relation, desc=''):
    from servers.dal import GraphDAL
    GraphDAL(brain.conn).add_relation(src, tgt, relation, description=desc,
                                      encoding_source='anchor')


def build_fixture(brain, case, tag):
    """Create a fresh fixture for `case`; return (cluster, ids, locked_ids)."""
    if case == 'C1':
        l1 = _mk(brain, 'Daemon write lock: single SQLite writer connection [%s]' % tag,
                 'The daemon serializes all foreground writes through one SQLite '
                 'writer connection (self.conn) guarded by a write lock. Background '
                 'batched writes use a separate connection owned by one worker '
                 'thread. This prevents interleaved writes from corrupting indexes.',
                 locked=True)
        u1 = _mk(brain, 'Daemon write lock is a TrackedRLock with snapshot() [%s]' % tag,
                 'The daemon write lock serializes foreground writes through one '
                 'SQLite writer connection. UNIQUE DETAIL: the lock is a '
                 'TrackedRLock whose .snapshot() exposes the current holder for '
                 'stall diagnostics during write-lock contention.',
                 locked=False)
        ids, locked = [l1, u1], {l1}
    elif case == 'C2':
        l2 = _mk(brain, 'Recall is keyword-first, then embedding spread [%s]' % tag,
                 'Recall runs keyword/FTS matching FIRST to find seed nodes, then '
                 'spreads through embeddings from those seeds. The keyword pass is '
                 'the primary retrieval path.',
                 locked=True)
        u2 = _mk(brain, 'Recall is embedding-FIRST: embed query, scan all nodes [%s]' % tag,
                 'CORRECTION: recall is embedding-FIRST. It embeds the query and '
                 'cosine-scans ALL nodes; FTS is a lexical add-on, not the primary '
                 'path. The earlier keyword-first description was wrong and is '
                 'superseded.',
                 locked=False)
        # signal the contradiction to the encoder
        _add_edge(brain, u2, l2, 'supersedes',
                  'embedding-first corrects the keyword-first framing')
        ids, locked = [l2, u2], {l2}
    elif case == 'C3':
        a = _mk(brain, 'Interactions table: versioned templates for boundaries [%s]' % tag,
                'Every boundary where information crosses components is an '
                "'interaction'. Table: interactions(name, version, template, "
                'parameters, created_by). Versioned, traceable, optimizable by '
                'higher scales.',
                locked=True)
        b = _mk(brain, 'Interactions as learnable boundaries — prompt+config in DB [%s]' % tag,
                'Every boundary is an interaction with a DB entry: LLM boundaries '
                'hold a prompt template, code boundaries hold config. Versioned and '
                'traced; the learning loop reads traces and proposes new versions.',
                locked=True)
        # mirror the real pair: already linked
        _add_edge(brain, a, b, 'similar_to',
                  'Two interactions-table architecture nodes, same function')
        _add_edge(brain, a, b, 'consolidated_into',
                  'a folded into b in an earlier run')
        ids, locked = [a, b], {a, b}
    else:
        raise ValueError(case)

    from servers.scales.s2.consolidation_decoder import ConsolidationDecoder
    d = ConsolidationDecoder(brain)
    key = '%s-%s' % tuple(sorted(ids))
    base = [{'nodes': sorted(ids), 'size': 2,
             'content_cosine_max': 0.92, 'content_cosine_avg': 0.92,
             'title_cosine_max': 0.85, 'title_cosine_avg': 0.85,
             'pair_scores': {key: {'content': 0.92, 'title': 0.85}}}]
    cluster = d._enrich_clusters(base)[0]
    cluster['pre_class'] = d._pre_classify(cluster)
    return cluster, ids, locked


# ──────────────────────────────────────────────────────────────────
# Encoder run (REAL dispatch against the isolated copy)
# ──────────────────────────────────────────────────────────────────

def run_encoder(brain, cluster, prompt_text):
    from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
    from servers.scales.s2.consolidation_contract import CONSOLIDATION
    orig = brain.get_interaction_prompt
    brain.get_interaction_prompt = (
        lambda name: prompt_text if name == 's2_consolidation_enrichment'
        else orig(name))
    try:
        enc = ConsolidationEncoder(brain, config=CONSOLIDATION)  # real dispatch
        enc._save_journal = lambda *a, **k: ''
        return enc.run([cluster]) or {}
    finally:
        brain.get_interaction_prompt = orig


def edges_between(brain, a, b):
    rows = brain.conn.execute(
        "SELECT er.relation FROM edges e JOIN edge_relations er ON er.edge_id=e.edge_id "
        "WHERE er.archived=0 AND ((e.source_id=? AND e.target_id=?) "
        "OR (e.source_id=? AND e.target_id=?))", (a, b, b, a)).fetchall()
    return {r[0] for r in rows}


ESCALATION_RELS = {'supersedes', 'corrects', 'contradicts', 'challenges',
                   'reframes', 'similar_to'}


def case_pass(case, ids, before, after, edges_now):
    """Case-specific behavioral pass criteria, computed from before/after state."""
    def archived(nid):
        return after.get(nid, {}).get('archived', False)

    def revised(nid):
        return (before.get(nid, {}).get('content') !=
                after.get(nid, {}).get('content'))

    if case == 'C1':
        l1, u1 = ids
        checks = {
            'locked_not_archived': not archived(l1),
            'unlocked_archived': archived(u1),
            'locked_enriched': revised(l1),
        }
    elif case == 'C2':
        l2, u2 = ids
        checks = {
            'locked_not_archived': not archived(l2),
            'unlocked_not_archived': not archived(u2),
            'locked_content_intact': not revised(l2),
            'escalation_edge_present': bool(edges_now & ESCALATION_RELS),
        }
    elif case == 'C3':
        a, b = ids
        checks = {
            'neither_archived': not archived(a) and not archived(b),
            'neither_revised_no_churn': not revised(a) and not revised(b),
            'similar_to_present': 'similar_to' in edges_now,
        }
    else:
        checks = {}
    return checks, all(checks.values())


# ──────────────────────────────────────────────────────────────────

CASES = ['C1', 'C2', 'C3']


def run_variant(brain, prompt_text, label):
    out = {}
    for i, case in enumerate(CASES):
        tag = '%s-%d' % (label, i)
        cluster, ids, locked_ids = build_fixture(brain, case, tag)
        before = snapshot_nodes(brain.conn, ids)
        all_before = {r[0] for r in brain.conn.execute("SELECT id FROM nodes")}
        result = run_encoder(brain, cluster, prompt_text)
        after = snapshot_nodes(brain.conn, ids)
        edges_now = edges_between(brain, ids[0], ids[1])

        # existing dimensions test
        analysis = analyze_actions(brain.conn, before, [cluster], all_before)
        dims = score_results(analysis, [cluster], result)

        checks, passed = case_pass(case, ids, before, after, edges_now)
        out[case] = {
            'ids': ids, 'locked_ids': sorted(locked_ids),
            'checks': checks, 'pass': passed,
            'locked_safety': dims['dimensions'].get('locked_safety'),
            'edges_after': sorted(edges_now),
            'rounds': result.get('rounds', 0),
        }
        _print_case(label, case, out[case])
    return out


def _print_case(label, case, r):
    status = 'PASS' if r['pass'] else 'FAIL'
    print("  [%s] %s %-4s  locked_safety=%s  edges=%s" % (
        label, case, status, r['locked_safety'], r['edges_after']))
    for k, v in r['checks'].items():
        print("        %s %s" % ('✓' if v else '✗', k))


def main():
    ap = argparse.ArgumentParser(description='S2 locked-absorb eval (A/B on existing dimensions)')
    ap.add_argument('--save', help='Write JSON report')
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()

    from tests.isolated_brain import IsolatedBrain
    print("Setting up isolated brain copy...")
    with IsolatedBrain(cleanup=not args.keep) as env:
        brain = env.brain
        baseline = brain.get_interaction_prompt('s2_consolidation_enrichment')
        candidate = build_candidate(baseline)
        print("Candidate built: +%d chars\n" % (len(candidate) - len(baseline)))

        print("=== BASELINE (ACTIVE v5) ===")
        base = run_variant(brain, baseline, 'baseline')
        print("\n=== CANDIDATE (locked-absorb reframe) ===")
        cand = run_variant(brain, candidate, 'candidate')

        print("\n" + "=" * 64)
        print("A/B SUMMARY")
        print("=" * 64)
        print("  case  baseline   candidate")
        for c in CASES:
            print("  %-4s  %-9s  %s" % (
                c, 'PASS' if base[c]['pass'] else 'FAIL',
                'PASS' if cand[c]['pass'] else 'FAIL'))
        b_pass = sum(base[c]['pass'] for c in CASES)
        c_pass = sum(cand[c]['pass'] for c in CASES)
        print("  ---")
        print("  total %d/3        %d/3" % (b_pass, c_pass))
        print("\n  Candidate must beat baseline and pass C3 (the real failure): %s" % (
            cand['C3']['pass'] and c_pass >= b_pass))

        if args.save:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            with open(args.save, 'w') as f:
                json.dump({'baseline': base, 'candidate': cand}, f,
                          indent=2, default=str)
            print("\nSaved %s" % args.save)


if __name__ == '__main__':
    main()
