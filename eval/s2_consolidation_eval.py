#!/usr/bin/env python3
"""S2 Consolidation Eval — measures encoder quality on a brain copy.

Runs decoder + encoder against an IsolatedBrain (copy of production).
Never touches live data. Produces a scored report across quality dimensions.

Usage:
    python3 eval/s2_consolidation_eval.py                    # 10 clusters
    python3 eval/s2_consolidation_eval.py --clusters 5       # 5 clusters
    python3 eval/s2_consolidation_eval.py --all              # All clusters
    python3 eval/s2_consolidation_eval.py --keep             # Keep temp dir
    python3 eval/s2_consolidation_eval.py --save report.json # Save report
"""

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_decoder(brain):
    """Run consolidation decoder, force cold start."""
    from servers.scales.s2.consolidation_decoder import ConsolidationDecoder

    decoder = ConsolidationDecoder(brain)

    # Force cold start scan (bypass trace gate)
    t0 = time.time()
    candidates, stats = decoder._scan_embeddings('', True)
    scan_ms = (time.time() - t0) * 1000

    if not candidates:
        return {'clusters': [], 'stats': stats, 'scan_ms': scan_ms}

    clusters = decoder._cluster_pairs(candidates)
    enriched = decoder._enrich_clusters(clusters)
    for c in enriched:
        c['pre_class'] = decoder._pre_classify(c)

    return {
        'clusters': enriched,
        'stats': {**stats, 'clusters': len(enriched),
                  'class_counts': dict(Counter(c['pre_class'] for c in enriched))},
        'scan_ms': scan_ms,
    }


def run_encoder(brain, clusters):
    """Run consolidation encoder on clusters."""
    from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
    from servers.scales.s2.consolidation_contract import CONSOLIDATION

    encoder = ConsolidationEncoder(brain, config=CONSOLIDATION)

    t0 = time.time()
    result = encoder.run(clusters)
    encode_ms = (time.time() - t0) * 1000

    if result:
        result['encode_ms'] = encode_ms
    return result


def _op_key(op):
    """Canonical identity of a brain_batch op — for deduping the capture-only
    re-emission artifact (see run_capture_variant)."""
    return (op.get('op'), op.get('survivor_id'), op.get('absorbed_id'),
            op.get('node_id'), op.get('source_id'), op.get('target_id'),
            op.get('relation'))


def run_capture_variant(brain, clusters, prompt_text):
    """Run the consolidation encoder over `clusters` with `prompt_text` swapped
    in for the s2_consolidation_enrichment interaction. CAPTURE-ONLY: brain_batch
    ops are recorded, never applied (dry-run dispatch), so multiple prompt arms
    see byte-identical input and we observe the DECISION, not the mutation.

    Shared by the absorb-prompt probe (behavioral A/B) and the consolidation
    contract eval (dimensions scoring) so both score the same captured decisions.

    Returns {ops, rounds, final_text}. Ops are de-duplicated by canonical
    identity: dry-run dispatch never changes state, so the encoder can re-emit
    the same ops across rounds (it can't see that the first batch "took"). In
    real dispatch the first batch archives the peer and round 2 can't re-absorb,
    so this de-dup matches production intent.
    """
    from servers.daemon_dispatch import COMMAND_TABLE
    from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
    from servers.scales.s2.consolidation_contract import CONSOLIDATION

    captured = []

    def dispatch(cmd, cmd_args):
        if cmd == 'brain_batch':
            captured.append(cmd_args)
            ops = cmd_args.get('operations', []) if isinstance(cmd_args, dict) else []
            return {'ok': True, 'result': {'dry_run': True, 'ops_seen': len(ops)}}
        entry = COMMAND_TABLE.get(cmd)
        if entry:
            return entry.handler(brain, cmd_args, [])
        return {'ok': True, 'result': {}}

    orig = brain.get_interaction_prompt
    brain.get_interaction_prompt = (
        lambda name: prompt_text if name == 's2_consolidation_enrichment'
        else orig(name))
    try:
        enc = ConsolidationEncoder(brain, dispatch_fn=dispatch, config=CONSOLIDATION)
        enc._save_journal = lambda *a, **k: ''   # don't contaminate other arms
        result = enc.run(clusters) or {}
    finally:
        brain.get_interaction_prompt = orig

    ops, seen = [], set()
    for cmd_args in captured:
        for op in (cmd_args.get('operations', []) if isinstance(cmd_args, dict) else []):
            if not isinstance(op, dict):
                continue
            k = _op_key(op)
            if k in seen:
                continue
            seen.add(k)
            ops.append(op)
    return {'ops': ops, 'rounds': result.get('rounds', 0),
            'final_text': result.get('final_text', '') or ''}


def snapshot_nodes(conn, node_ids):
    """Capture node state before encoding for diff analysis."""
    snapshots = {}
    placeholders = ','.join('?' * len(node_ids))
    for row in conn.execute(
            "SELECT id, title, type, content, confidence, archived, locked "
            "FROM nodes WHERE id IN (%s)" % placeholders,
            list(node_ids)).fetchall():
        snapshots[row[0]] = {
            'title': row[1], 'type': row[2], 'content': row[3],
            'confidence': row[4], 'archived': bool(row[5]),
            'locked': bool(row[6]),
        }
    return snapshots


def analyze_actions(brain, before_snapshot, clusters, all_ids_before=None):
    """Analyze what the encoder did by comparing before/after state."""
    conn = brain.conn
    all_ids = set()
    for c in clusters:
        all_ids.update(c['nodes'])

    after_snapshot = snapshot_nodes(conn, all_ids)

    # Find newly created nodes — IDs that didn't exist before
    all_ids_after = {r[0] for r in conn.execute(
        "SELECT id FROM nodes").fetchall()}
    new_ids = all_ids_after - (all_ids_before or set(before_snapshot.keys()))
    new_nodes = []
    if new_ids:
        ph = ','.join('?' * len(new_ids))
        new_nodes = conn.execute(
            "SELECT id, title, type, content, confidence "
            "FROM nodes WHERE id IN (%s)" % ph,
            list(new_ids)).fetchall()

    # Find ALL suppression edges (not just from this run — encoding_source
    # may not be set on connect operations routed through brain_batch)
    # v25: forensic eval — see all suppression edges including archived
    # (archive_node archives edges pointing at archived nodes).
    # Suppression verbs derive from the settlement aspect, same source the
    # decoder reads — the eval must not drift from the live set.
    settlement_rels = sorted(brain.aspects.settlement.edge_relations)
    suppression_edges = conn.execute("""
        SELECT e.source_id, e.target_id, er.relation, er.description
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE er.relation IN (%s)
    """ % ','.join('?' * len(settlement_rels)), settlement_rels).fetchall()

    # Detect revised nodes (content changed)
    revised_ids = set()
    for nid in all_ids:
        b = before_snapshot.get(nid, {})
        a = after_snapshot.get(nid, {})
        if b.get('content') and a.get('content') and b['content'] != a['content']:
            revised_ids.add(nid)

    new_node_ids = {n[0] for n in new_nodes}

    # Classify actions per cluster
    actions = []
    for cluster in clusters:
        nids = set(cluster['nodes'])
        archived = [nid for nid in nids
                    if not before_snapshot.get(nid, {}).get('archived', False)
                    and after_snapshot.get(nid, {}).get('archived', False)]
        locked_archived = [nid for nid in archived
                           if before_snapshot.get(nid, {}).get('locked', False)]
        revised = [nid for nid in nids if nid in revised_ids]

        # Check for suppression edges involving cluster nodes
        cluster_supp = [e for e in suppression_edges
                        if e[0] in nids or e[1] in nids]
        # Also check for new nodes connected to cluster nodes
        cluster_new = [n for n in new_nodes
                       if any(e[1] in nids or e[0] in nids
                              for e in suppression_edges if e[0] == n[0])]

        # Determine action type
        if cluster_new or (archived and new_node_ids):
            action_type = 'CONSOLIDATED'
        elif archived:
            action_type = 'EVOLVED'
        elif cluster_supp:
            action_type = 'KEPT' if any(e[2] == 'similar_to' for e in cluster_supp) else 'SKIPPED'
        elif revised:
            action_type = 'REVISED'  # Encoder revised nodes without archiving
        else:
            action_type = 'NO_ACTION'

        actions.append({
            'cluster_nodes': list(nids),
            'pre_class': cluster['pre_class'],
            'action_type': action_type,
            'archived': archived,
            'revised': revised,
            'locked_violated': locked_archived,
            'suppression_edges': len(cluster_supp),
        })

    return {
        'actions': actions,
        'new_nodes': [{'id': n[0], 'title': n[1], 'type': n[2],
                       'content': n[3] or '', 'confidence': n[4]}
                      for n in new_nodes],
        'suppression_edges': len(suppression_edges),
    }


def score_results(analysis, clusters, encode_result):
    """Score encoder output across quality dimensions."""
    scores = {}
    details = {}

    actions = analysis['actions']
    if not actions:
        return {'composite': 0, 'dimensions': {}, 'details': {}}

    # ── 1. Acceptance rate ──
    acted = sum(1 for a in actions if a['action_type'] != 'NO_ACTION')
    scores['acceptance_rate'] = acted / len(actions) if actions else 0
    details['acted'] = acted
    details['total'] = len(actions)

    # ── 2. Locked safety ──
    locked_violations = sum(len(a['locked_violated']) for a in actions)
    scores['locked_safety'] = 1.0 if locked_violations == 0 else 0.0
    details['locked_violations'] = locked_violations

    # ── 3. Suppression completeness ──
    with_suppression = sum(1 for a in actions if a['suppression_edges'] > 0)
    scores['suppression'] = with_suppression / len(actions) if actions else 0
    details['with_suppression'] = with_suppression

    # ── 4. Action distribution ──
    action_counts = Counter(a['action_type'] for a in actions)
    details['action_counts'] = dict(action_counts)
    # Penalize if everything is the same action (no judgment nuance)
    unique_actions = len([k for k, v in action_counts.items() if v > 0 and k != 'NO_ACTION'])
    scores['action_diversity'] = min(1.0, unique_actions / 3) if acted > 3 else 1.0

    # ── 5. New node quality (for CONSOLIDATED actions) ──
    new_nodes = analysis['new_nodes']
    if new_nodes:
        content_lengths = [len(n['content']) for n in new_nodes]
        avg_content = sum(content_lengths) / len(content_lengths)
        scores['synthesis_content_length'] = min(1.0, avg_content / 200)

        # Check for ID references in content
        import re
        id_refs = sum(len(re.findall(r'id:[a-z0-9]{6,8}', n['content']))
                      for n in new_nodes)
        scores['synthesis_references'] = min(1.0, id_refs / len(new_nodes))
    else:
        scores['synthesis_content_length'] = 0.5  # No consolidations = neutral
        scores['synthesis_references'] = 0.5

    # ── 6. Pre-class alignment ──
    # Did the encoder's action align with the decoder's pre-classification?
    aligned = 0
    for a in actions:
        pre = a['pre_class']
        act = a['action_type']
        if pre == 'likely_consolidate' and act == 'CONSOLIDATED':
            aligned += 1
        elif pre == 'likely_evolve' and act == 'EVOLVED':
            aligned += 1
        elif pre == 'likely_keep' and act in ('KEPT', 'SKIPPED'):
            aligned += 1
        elif pre == 'needs_judgment' and act != 'NO_ACTION':
            aligned += 1  # Any decision on needs_judgment is good
    scores['preclass_alignment'] = aligned / len(actions) if actions else 0

    # ── 7. Efficiency ──
    rounds = encode_result.get('rounds', 0) if encode_result else 0
    expected_rounds = 2 * ((len(clusters) + 9) // 10)  # 2 rounds per batch
    if rounds > 0 and expected_rounds > 0:
        scores['efficiency'] = min(1.0, expected_rounds / rounds)
    else:
        scores['efficiency'] = 0.5

    # ── Composite ──
    weights = {
        'acceptance_rate': 0.15,
        'locked_safety': 0.20,         # Critical — wrong here = data loss
        'suppression': 0.15,
        'action_diversity': 0.10,
        'synthesis_content_length': 0.10,
        'synthesis_references': 0.10,
        'preclass_alignment': 0.10,
        'efficiency': 0.10,
    }

    composite = sum(scores.get(k, 0) * w for k, w in weights.items())

    return {
        'composite': round(composite, 3),
        'dimensions': {k: round(v, 3) for k, v in scores.items()},
        'details': details,
    }


def print_report(decode_stats, encode_result, analysis, scores, elapsed_s):
    """Pretty-print eval report."""
    print("=" * 70)
    print("S2 CONSOLIDATION EVAL REPORT")
    print("=" * 70)

    # Decoder
    ds = decode_stats
    print("\nDecoder: %d nodes scanned, %d pairs, %d clusters (%.0fms)" % (
        ds.get('nodes_scanned', 0), ds.get('pairs_found', 0),
        ds.get('clusters', 0), decode_stats.get('scan_ms', 0)))
    cc = ds.get('class_counts', {})
    if cc:
        print("  Classes: %s" % ', '.join('%s=%d' % (k, v) for k, v in
                                           sorted(cc.items(), key=lambda x: -x[1])))

    # Encoder
    if encode_result:
        print("\nEncoder: %d actions (%d writes) in %d rounds (%.1fs)" % (
            encode_result.get('actions', 0),
            encode_result.get('write_actions', 0),
            encode_result.get('rounds', 0),
            encode_result.get('encode_ms', 0) / 1000))

    # Actions
    print("\nActions:")
    for a in analysis.get('actions', []):
        pre = a['pre_class']
        act = a['action_type']
        titles = []
        for nid in a['cluster_nodes'][:3]:
            titles.append(nid[:8])
        violation = ' ⚠ LOCKED VIOLATED' if a['locked_violated'] else ''
        supp = ' ✓supp' if a['suppression_edges'] > 0 else ' ✗NO_SUPP'
        print("  [%s → %s] %s%s%s" % (
            pre[:15].ljust(15), act[:12].ljust(12),
            ', '.join(titles), supp, violation))

    # New nodes
    if analysis.get('new_nodes'):
        print("\nSynthesized nodes:")
        for n in analysis['new_nodes']:
            print("  [%s] %s (conf=%.2f)" % (n['type'], n['title'][:60], n['confidence']))
            print("    %s..." % n['content'][:120])

    # Scores
    print("\nScores:")
    print("  COMPOSITE: %.3f" % scores['composite'])
    for dim, val in sorted(scores['dimensions'].items()):
        bar = '█' * int(val * 20)
        print("  %-30s %.3f  %s" % (dim, val, bar))

    # Details
    d = scores.get('details', {})
    if d.get('locked_violations', 0) > 0:
        print("\n  ⚠ LOCKED VIOLATIONS: %d — encoder archived locked nodes!" %
              d['locked_violations'])
    if d.get('action_counts'):
        print("\n  Action counts: %s" % d['action_counts'])

    print("\n  Total time: %.1fs" % elapsed_s)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='S2 Consolidation Eval')
    parser.add_argument('--clusters', type=int, default=10,
                        help='Number of clusters to process (default: 10)')
    parser.add_argument('--indices', help='JSON file with cluster indices to process')
    parser.add_argument('--all', action='store_true',
                        help='Process all clusters')
    parser.add_argument('--keep', action='store_true',
                        help='Keep temp directory for inspection')
    parser.add_argument('--save', help='Save report to JSON file')
    args = parser.parse_args()

    from tests.isolated_brain import IsolatedBrain

    t_start = time.time()

    print("Setting up isolated brain copy...")
    with IsolatedBrain(cleanup=not args.keep) as env:
        brain = env.brain

        if args.keep:
            print("Isolated brain at: %s" % env.db_dir)

        # Run decoder
        print("\nRunning decoder (cold start)...")
        decode_result = run_decoder(brain)
        clusters = decode_result['clusters']
        print("  Found %d clusters" % len(clusters))

        if not clusters:
            print("No clusters found. Nothing to eval.")
            return

        # Select clusters to process
        if args.all:
            selected = clusters
        elif args.indices:
            import json as _json
            with open(args.indices) as _f:
                indices = _json.load(_f)
            selected = [clusters[i] for i in indices if i < len(clusters)]
        else:
            selected = clusters[:args.clusters]
        print("  Processing %d clusters" % len(selected))

        # Snapshot before — cluster members AND all node IDs
        cluster_member_ids = set()
        for c in selected:
            cluster_member_ids.update(c['nodes'])
        before = snapshot_nodes(brain.conn, cluster_member_ids)
        # Track ALL node IDs to detect new nodes created by encoder
        all_ids_before = {r[0] for r in brain.conn.execute(
            "SELECT id FROM nodes").fetchall()}

        # Run encoder
        print("\nRunning encoder...")
        encode_result = run_encoder(brain, selected)

        if not encode_result:
            print("Encoder returned None — check prompt registration")
            return

        # Show what Sonnet actually sent
        for ad in encode_result.get('action_details', []):
            print("\n  Tool call: %s" % ad.get('tool', '?'))
            inp = ad.get('input', {})
            ops = inp.get('operations', [])
            for j, op in enumerate(ops):
                print("    [%d] op=%s" % (j, op.get('op', '?')))
                if op.get('op') == 'remember':
                    print("      title: %s" % op.get('title', '?')[:70])
                    conns = op.get('connections', [])
                    if conns:
                        print("      connections: %d" % len(conns))
                        for c in conns:
                            print("        → %s (%s)" % (c.get('target_id', '?')[:8], c.get('relation', '?')))
                elif op.get('op') == 'revise':
                    print("      node_id: %s archived=%s" % (op.get('node_id', '?')[:8], op.get('archived', '?')))
                elif op.get('op') == 'connect':
                    print("      %s → %s (%s)" % (op.get('source_id', '?')[:8], op.get('target_id', '?')[:8], op.get('relation', '?')))
                elif op.get('op') == 'archive':
                    print("      node_id: %s" % op.get('node_id', '?')[:8])

        # Analyze
        print("\nAnalyzing results...")
        analysis = analyze_actions(brain, before, selected, all_ids_before)
        scores = score_results(analysis, selected, encode_result)

        elapsed = time.time() - t_start

        # Report
        print_report(decode_result['stats'], encode_result,
                     analysis, scores, elapsed)

        # Save
        if args.save:
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'clusters_processed': len(selected),
                'clusters_total': len(clusters),
                'decode_stats': decode_result['stats'],
                'encode_result': {
                    'actions': encode_result.get('actions', 0),
                    'write_actions': encode_result.get('write_actions', 0),
                    'rounds': encode_result.get('rounds', 0),
                },
                'analysis': analysis,
                'scores': scores,
                'elapsed_s': elapsed,
            }
            with open(args.save, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print("\nSaved to %s" % args.save)


if __name__ == '__main__':
    main()
