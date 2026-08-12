#!/usr/bin/env python3
"""S2 Community Decoder Eval — production decoder, multi-run convergence.

Runs the SHIPPED CommunityDecoder against an isolated copy of the brain and
measures backlog convergence, fingerprint suppression, and proposal mix
across simulated encoder cycles. All decode and suppression logic is
imported from servers/ — this file owns only the harness (isolated brain,
simulated acceptance, cross-run metrics, report).

The decode side mirrors CommunityDetection._run_pipeline: decoder.run()
(incremental decode with the unplaceable rest gate and the Step 9c batch
contract) followed by filter_rejected on the proposals. The dead-community
auto-archive and the unplaceable marking are orchestrator writes
(community.py) and are not simulated here.

Usage:
    ./dev python3 eval/s2_community_decoder_eval.py                   # 3 runs, 60% accept
    ./dev python3 eval/s2_community_decoder_eval.py --runs 5
    ./dev python3 eval/s2_community_decoder_eval.py --accept-rate 0.8
    ./dev python3 eval/s2_community_decoder_eval.py --keep            # keep temp dir
    ./dev python3 eval/s2_community_decoder_eval.py --save report.json
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.s2.rejection_table import (  # noqa: E402
    filter_rejected, record_rejections)
from servers.scales.s2.community_contract import COMMUNITY_DETECTION  # noqa: E402

# Types the encoder acts on; the rest (node_affinities, cross_cutting) are
# context and never accepted/rejected/fingerprinted.
ACTIONABLE_TYPES = ('new_community', 'add_to_existing', 'drift',
                    'health_update', 'merge_communities')


# ═══════════════════════════════════════════════════════════════
# DECODE — the production pipeline's decode side
# ═══════════════════════════════════════════════════════════════

def run_decoder(brain, config=None):
    """Run the production CommunityDecoder + the pipeline's rejection filter.

    Returns {proposals, community_state, stats, unplaced_count,
    proposed_node_ids, skipped}. `proposals` are post-filter_rejected —
    what the encoder would see.
    """
    from servers.scales.s2.community_decoder import CommunityDecoder

    decoder = CommunityDecoder(brain, config=config)
    t0 = time.time()
    result = decoder.run()
    decode_ms = (time.time() - t0) * 1000

    community_state = result.get('community_state')
    if community_state is None:
        community_state = decoder._read_community_state()

    raw = result.get('proposals', [])
    proposals, suppressed_count = filter_rejected(brain, raw)
    survived = {id(p) for p in proposals}

    proposed_node_ids = set()
    for p in proposals:
        if p.get('node_id'):
            proposed_node_ids.add(p['node_id'])
        for m in p.get('members', []):
            proposed_node_ids.add(m)

    stats = dict(result.get('stats') or {})
    stats.update({
        'unplaced': result.get('unplaced_count', 0),
        'communities': len(community_state),
        'raw_proposals': len(raw),
        'suppressed_count': suppressed_count,
        'suppressed_by_type': dict(Counter(
            p.get('type', '?') for p in raw if id(p) not in survived)),
        'total_proposals': len(proposals),
        'by_type': dict(Counter(p.get('type', '?') for p in proposals)),
        'decode_ms': round(decode_ms, 1),
    })

    return {
        'proposals': proposals,
        'community_state': community_state,
        'stats': stats,
        'unplaced_count': result.get('unplaced_count', 0),
        'proposed_node_ids': proposed_node_ids,
        'skipped': result.get('skipped'),
    }


# ═══════════════════════════════════════════════════════════════
# SIMULATED ENCODER — accept/reject without LLM calls
# ═══════════════════════════════════════════════════════════════

def simulate_acceptance(brain, proposals, accept_rate=0.6, run_seed=42):
    """Simulate encoder acceptance/rejection over the actionable proposals.

    Accepted: apply the placement to the DB (community nodes / member
    edges / archives). Rejected: fingerprint via the production
    record_rejections, so the next run_decoder suppresses them — the loop
    under test.
    """
    from servers.dal_graph import GraphDAL

    rng = random.Random(run_seed)
    graph_dal = GraphDAL(brain.conn)
    ts = datetime.now(timezone.utc).isoformat()  # clock-ok — eval bookkeeping

    accepted = []
    rejected = []
    members_placed = 0
    communities_created = 0
    community_sizes = []

    for p in proposals:
        if p.get('type') not in ACTIONABLE_TYPES:
            continue
        if rng.random() >= accept_rate:
            rejected.append(p)
            continue
        accepted.append(p)

        if p['type'] == 'new_community':
            node_id = uuid.uuid4().hex[:8]
            member_ids = p.get('members', [])
            title = 'Community: %s + %d more' % (
                p.get('representatives', [{}])[0].get('title', '?')[:40]
                if p.get('representatives') else '?',
                max(0, len(member_ids) - 1))
            brain.conn.execute(
                "INSERT INTO nodes (id, type, title, content, confidence, "
                "encoding_source, created_at, updated_at, archived) "
                "VALUES (?, 'community', ?, '', 0.7, ?, ?, ?, 0)",
                (node_id, title, 's2:community_detection', ts, ts))
            for mid in member_ids:
                graph_dal.add_relation(
                    node_id, mid, 'community_member', weight=0.3,
                    encoding_source='s2:community_detection')
            members_placed += len(member_ids)
            communities_created += 1
            community_sizes.append(len(member_ids))

        elif p['type'] == 'add_to_existing':
            # Production shape: candidate communities in `communities`,
            # sorted by affinity desc — the encoder connects the top one.
            comms = p.get('communities') or []
            nid = p.get('node_id')
            if comms and nid:
                graph_dal.add_relation(
                    comms[0]['id'], nid, 'community_member', weight=0.3,
                    encoding_source='s2:community_detection')
                members_placed += 1

        elif p['type'] == 'drift':
            nid = p.get('node_id')
            foreign = p.get('foreign', [])
            if nid and foreign:
                graph_dal.add_relation(
                    foreign[0]['id'], nid, 'community_member', weight=0.3,
                    encoding_source='s2:community_detection')

        elif p['type'] == 'health_update' and p.get('signal') == 'dead':
            brain.conn.execute(
                "UPDATE nodes SET archived = 1 WHERE id = ?",
                (p['community_id'],))

        elif p['type'] == 'merge_communities':
            larger_id = p.get('larger_id')
            smaller_id = p.get('smaller_id')
            if larger_id and smaller_id:
                smaller_members = set(r[0] for r in brain.conn.execute("""
                    SELECT CASE WHEN e.source_id = ? THEN e.target_id
                           ELSE e.source_id END
                    FROM edges e
                    JOIN edge_relations er ON er.edge_id = e.edge_id
                    WHERE (e.source_id = ? OR e.target_id = ?)
                    AND er.relation = 'community_member'
                    AND er.archived = 0
                """, (smaller_id, smaller_id, smaller_id)).fetchall())
                for mid in smaller_members:
                    graph_dal.add_relation(
                        larger_id, mid, 'community_member', weight=0.3,
                        encoding_source='s2:community_detection')
                brain.conn.execute(
                    "UPDATE nodes SET archived = 1 WHERE id = ?",
                    (smaller_id,))

    record_rejections(brain, rejected)
    brain.conn.commit()

    return {
        'accepted': len(accepted),
        'rejected': len(rejected),
        'accepted_by_type': dict(Counter(p['type'] for p in accepted)),
        'rejected_by_type': dict(Counter(p['type'] for p in rejected)),
        'members_placed': members_placed,
        'communities_created': communities_created,
        'community_sizes': community_sizes,
    }


# ═══════════════════════════════════════════════════════════════
# MULTI-RUN LOOP + CROSS-RUN METRICS
# ═══════════════════════════════════════════════════════════════

def run_multi(brain, n_runs=3, accept_rate=0.6, seed=42, config=None):
    """decoder → simulate → repeat, n_runs times."""
    runs = []
    for i in range(n_runs):
        result = run_decoder(brain, config)
        if result.get('skipped'):
            runs.append({
                'run': i + 1,
                'stats': result['stats'],
                'skipped': result['skipped'],
                'simulation': {'accepted': 0, 'rejected': 0,
                               'members_placed': 0, 'communities_created': 0},
                'proposed_node_ids': set(),
                'rejection_table_size': brain.conn.execute(
                    "SELECT COUNT(*) FROM s2_rejections").fetchone()[0],
            })
            continue

        sim = simulate_acceptance(
            brain, result['proposals'],
            accept_rate=accept_rate, run_seed=seed + i)

        rejection_count = brain.conn.execute(
            "SELECT COUNT(*) FROM s2_rejections").fetchone()[0]

        runs.append({
            'run': i + 1,
            'stats': result['stats'],
            'simulation': sim,
            'proposed_node_ids': result['proposed_node_ids'],
            'rejection_table_size': rejection_count,
        })
    return runs


def compute_cross_run_metrics(runs):
    """Trajectories across runs: backlog, suppression, re-proposal rate."""
    if not runs:
        return {}

    backlog = [r['stats'].get('unplaced', 0) for r in runs]
    raw_proposals = [r['stats'].get('raw_proposals', 0) for r in runs]
    proposals = [r['stats'].get('total_proposals', 0) for r in runs]
    suppressed = [r['stats'].get('suppressed_count', 0) for r in runs]
    rejection_table = [r.get('rejection_table_size', 0) for r in runs]

    re_proposal_rates = [0]
    for i in range(1, len(runs)):
        prev_ids = runs[i - 1]['proposed_node_ids']
        curr_ids = runs[i]['proposed_node_ids']
        if curr_ids:
            re_proposal_rates.append(
                round(len(prev_ids & curr_ids) / len(curr_ids), 3))
        else:
            re_proposal_rates.append(0)

    convergence = round(1 - (proposals[-1] / proposals[0]), 3) \
        if proposals and proposals[0] else 0

    return {
        'backlog_trajectory': backlog,
        'raw_proposal_trajectory': raw_proposals,
        'proposal_trajectory': proposals,
        'suppression_trajectory': suppressed,
        'rejection_table_trajectory': rejection_table,
        're_proposal_rates': re_proposal_rates,
        'convergence_score': convergence,
        'total_members_placed': sum(
            r['simulation']['members_placed'] for r in runs),
        'total_communities_created': sum(
            r['simulation']['communities_created'] for r in runs),
    }


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════

def print_report(runs, cross_metrics, brain):
    SEP = '=' * 70
    node_count = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
    comm_count = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE type = 'community' AND archived = 0"
    ).fetchone()[0]

    print(SEP)
    print('S2 COMMUNITY DECODER EVAL — production decoder, simulated encoder')
    print(SEP)
    print('\nBrain: %d nodes, %d live communities' % (node_count, comm_count))

    for r in runs:
        s = r['stats']
        sim = r['simulation']
        print('\n--- RUN %d ---' % r['run'])
        if r.get('skipped'):
            print('  decode rested: %s' % r['skipped'])
            continue
        print('  Unplaced: %d | Communities: %d | decode %.0fms' % (
            s.get('unplaced', 0), s.get('communities', 0),
            s.get('decode_ms', 0)))
        print('  Clusters: %d valid (%d seeded, %d dissolved, %d absorbed)' % (
            s.get('valid_clusters', 0), s.get('clusters_seeded', 0),
            s.get('fragments_dissolved', 0), s.get('subsets_absorbed', 0)))
        print('  Proposals: %d raw → %d suppressed → %d surviving' % (
            s.get('raw_proposals', 0), s.get('suppressed_count', 0),
            s.get('total_proposals', 0)))
        for ptype, count in sorted(s.get('by_type', {}).items(),
                                   key=lambda x: -x[1]):
            print('    %-20s %d' % (ptype, count))
        if s.get('suppressed_by_type'):
            print('  Suppressed by type: %s' % '  '.join(
                '%s=%d' % (t, c) for t, c in
                sorted(s['suppressed_by_type'].items(), key=lambda x: -x[1])))
        print('  Simulated: %d accepted / %d rejected '
              '(placed %d members, created %d communities)' % (
                  sim['accepted'], sim['rejected'],
                  sim['members_placed'], sim['communities_created']))
        print('  Rejection table: %d entries' % r.get('rejection_table_size', 0))

    if cross_metrics:
        print('\n' + SEP)
        print('CROSS-RUN')
        print(SEP)
        print('  Backlog:          %s' % cross_metrics['backlog_trajectory'])
        print('  Raw proposals:    %s' % cross_metrics['raw_proposal_trajectory'])
        print('  Surviving:        %s' % cross_metrics['proposal_trajectory'])
        print('  Suppressed:       %s' % cross_metrics['suppression_trajectory'])
        print('  Rejection table:  %s' % cross_metrics['rejection_table_trajectory'])
        print('  Re-proposal rate: %s' % cross_metrics['re_proposal_rates'])
        print('  Convergence:      %.1f%%' % (
            cross_metrics['convergence_score'] * 100))
        print('  Placed %d members, created %d communities across runs' % (
            cross_metrics['total_members_placed'],
            cross_metrics['total_communities_created']))


def save_report(path, runs, cross_metrics):
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),  # clock-ok — report stamp
        'runs': [{k: v for k, v in r.items() if k != 'proposed_node_ids'}
                 for r in runs],
        'cross_run': cross_metrics,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print('\nSaved: %s' % path)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--accept-rate', type=float, default=0.6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--keep', action='store_true',
                    help='keep the isolated temp dir for inspection')
    ap.add_argument('--save', help='write JSON report to this path')
    args = ap.parse_args()

    from tests.isolated_brain import IsolatedBrain

    with IsolatedBrain(cleanup=not args.keep) as env:
        brain = env.brain
        runs = run_multi(brain, n_runs=args.runs,
                         accept_rate=args.accept_rate, seed=args.seed,
                         config=dict(COMMUNITY_DETECTION))
        cross = compute_cross_run_metrics(runs)
        print_report(runs, cross, brain)
        if args.save:
            save_report(args.save, runs, cross)
        if args.keep:
            print('\nInspect at: %s' % env.db_dir)


if __name__ == '__main__':
    main()
