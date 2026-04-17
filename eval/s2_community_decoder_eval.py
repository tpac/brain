#!/usr/bin/env python3
"""S2 Community Decoder Eval — Inside-Out + Fingerprint Suppression.

Tests a new decoder approach against a brain copy across multiple simulated runs.
Measures backlog convergence, suppression effectiveness, and proposal quality.

Suppression mechanism: rejected proposals are fingerprinted and stored in a table.
The decoder generates all proposals, then filters out any whose fingerprint matches
a previous rejection. When graph changes cause different parameters (different
affinity, different members, different internal fraction), the fingerprint changes
and the old rejection no longer applies.

Usage:
    python3 eval/s2_community_decoder_eval.py                    # 3 runs, 60% accept
    python3 eval/s2_community_decoder_eval.py --runs 5           # 5 runs
    python3 eval/s2_community_decoder_eval.py --accept-rate 0.8  # 80% acceptance
    python3 eval/s2_community_decoder_eval.py --compare-only     # Just old vs new
    python3 eval/s2_community_decoder_eval.py --keep             # Keep temp dir
    python3 eval/s2_community_decoder_eval.py --save report.json
"""

import hashlib
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


# ═══════════════════════════════════════════════════════════════
# REJECTION TABLE + FINGERPRINTS
# ═══════════════════════════════════════════════════════════════

def create_rejection_table(brain):
    """Create the s2_rejections table if it doesn't exist."""
    brain.conn.execute("""
        CREATE TABLE IF NOT EXISTS s2_rejections (
            fingerprint TEXT PRIMARY KEY,
            integration_unit TEXT NOT NULL,
            proposal_type TEXT NOT NULL,
            proposed_ids TEXT,
            created_at TEXT NOT NULL
        )
    """)
    brain.conn.commit()


def compute_fingerprint(proposal):
    """Compute a stable fingerprint for a proposal.

    Captures what the encoder actually judges — not implementation artifacts.
    The fingerprint should only change when the encoder would genuinely
    reconsider its decision.

    - new_community: top 40% of representatives (structural hubs define identity)
    - add_to_existing: node + community + affinity tier (borderline/moderate/strong)
    - drift: node + destination community
    - health_update: community + signal type (dead/degrading/maturing)
    - merge: the two community IDs
    """
    ptype = proposal['type']

    if ptype == 'add_to_existing':
        # Coarse affinity tier: the encoder's judgment changes across
        # these regimes, not within them.
        aff = proposal.get('affinity', 0)
        if aff >= 0.65:
            tier = 'strong'
        elif aff >= 0.40:
            tier = 'moderate'
        else:
            tier = 'borderline'
        raw = 'add:%s:%s:%s' % (
            proposal.get('node_id', ''),
            proposal.get('community_id', ''),
            tier)

    elif ptype == 'new_community':
        # Top 40% of representatives sorted by internal edges — the structural
        # hubs that define cluster identity. Peripheral members can change
        # without invalidating the rejection.
        reps = proposal.get('representatives', [])
        members = sorted(proposal.get('members', []))
        if reps:
            # Representatives are already sorted by internal_edges (descending)
            rep_ids = sorted(r['id'] for r in reps)
            n_keep = max(2, int(len(rep_ids) * 0.4 + 0.5))
            core = rep_ids[:n_keep]
        else:
            # Fallback: top 40% of sorted member IDs
            n_keep = max(2, int(len(members) * 0.4 + 0.5))
            core = members[:n_keep]
        raw = 'new:' + ':'.join(core)

    elif ptype == 'drift':
        foreign = proposal.get('foreign', [{}])
        foreign_id = foreign[0].get('id', '') if foreign else ''
        raw = 'drift:%s:%s' % (proposal.get('node_id', ''), foreign_id)

    elif ptype == 'health_update':
        raw = 'health:%s:%s' % (
            proposal.get('community_id', ''),
            proposal.get('signal', ''))

    elif ptype == 'merge_communities':
        raw = 'merge:%s:%s' % (
            proposal.get('larger_id', ''),
            proposal.get('smaller_id', ''))

    else:
        raw = '%s:%s' % (ptype, proposal.get('node_id', ''))

    return hashlib.md5(raw.encode()).hexdigest()[:16]


def get_proposed_ids(proposal):
    """Extract the node IDs involved in a proposal."""
    ids = []
    if proposal.get('node_id'):
        ids.append(proposal['node_id'])
    if proposal.get('members'):
        ids.extend(proposal['members'])
    if proposal.get('community_id'):
        ids.append(proposal['community_id'])
    if proposal.get('larger_id'):
        ids.append(proposal['larger_id'])
    if proposal.get('smaller_id'):
        ids.append(proposal['smaller_id'])
    for f in proposal.get('foreign', []):
        if f.get('id'):
            ids.append(f['id'])
    return ids


def filter_rejected(brain, proposals):
    """Filter out proposals whose fingerprint matches a previous rejection.

    Returns (surviving_proposals, suppressed_count).
    """
    if not proposals:
        return proposals, 0

    # Compute fingerprints for all proposals
    fps = [(p, compute_fingerprint(p)) for p in proposals]

    # Batch-check against rejection table
    all_fp = [fp for _, fp in fps]
    rejected_set = set()
    for chunk_start in range(0, len(all_fp), 999):
        chunk = all_fp[chunk_start:chunk_start + 999]
        ph = ','.join('?' * len(chunk))
        rows = brain.conn.execute(
            "SELECT fingerprint FROM s2_rejections WHERE fingerprint IN (%s)" % ph,
            chunk).fetchall()
        for row in rows:
            rejected_set.add(row[0])

    surviving = []
    suppressed = 0
    for p, fp in fps:
        if fp in rejected_set:
            suppressed += 1
        else:
            surviving.append(p)

    return surviving, suppressed


def record_rejections(brain, proposals, integration_unit='s2:community_detection'):
    """Write rejected proposal fingerprints to the rejection table."""
    ts = datetime.now(timezone.utc).isoformat()
    for p in proposals:
        fp = compute_fingerprint(p)
        ids = json.dumps(get_proposed_ids(p))
        brain.conn.execute(
            "INSERT OR IGNORE INTO s2_rejections "
            "(fingerprint, integration_unit, proposal_type, proposed_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (fp, integration_unit, p['type'], ids, ts))
    brain.conn.commit()


# ═══════════════════════════════════════════════════════════════
# NEW DECODER — Inside-Out + Fingerprint Suppression
# ═══════════════════════════════════════════════════════════════

def run_new_decoder(brain, config=None):
    """Run the inside-out decoder with fingerprint-based suppression.

    Phases generate all proposals freely. Fingerprint filtering happens
    once at the end — one mechanism for all proposal types.
    """
    from servers.scales.s2.community_decoder import CommunityDecoder
    from servers.scales.s2.community_contract import COMMUNITY_DETECTION

    config = config or COMMUNITY_DETECTION
    decoder = CommunityDecoder(brain, config=config)

    t0 = time.time()

    # Read graph state
    community_state = decoder._read_community_state()

    already_placed = set()
    node_to_community = {}
    for comm in community_state:
        for mid in comm.get('members', []):
            already_placed.add(mid)
            node_to_community[mid] = comm

    # Load edge families + build adjacency
    edge_families_config = decoder._get_interaction_config('s2_edge_families') or {}
    rel_to_fam = {}
    for fam, members in edge_families_config.items():
        if isinstance(members, list):
            for m in members:
                rel_to_fam[m] = fam
    skip_fams = {'generic_relation', 'noise'}

    t_adj = time.time()
    edges_by_node, typed_neighbors = decoder._build_typed_adjacency(
        rel_to_fam, skip_fams)
    adj_ms = (time.time() - t_adj) * 1000

    all_active = set(r[0] for r in brain.conn.execute(
        "SELECT id FROM nodes WHERE archived = 0 AND type != 'community'"
    ).fetchall())
    unplaced = all_active - already_placed

    if not unplaced:
        return {
            'proposals': [], 'stats': {'unplaced': 0, 'total_proposals': 0,
                                        'suppressed_count': 0},
            'community_state': community_state,
            'unplaced_count': 0, 'edges_by_node': edges_by_node,
            'proposed_node_ids': set(),
        }

    # Phase 1: Inside-out
    t1 = time.time()
    p1_result = _phase1_inside_out(
        community_state, typed_neighbors, unplaced, config)
    phase1_ms = (time.time() - t1) * 1000

    # Phase 2: Orphan clustering
    t2 = time.time()
    p2_result = _phase2_cluster_orphans(
        decoder, edges_by_node, typed_neighbors, unplaced,
        p1_result['reached'], community_state, config)
    phase2_ms = (time.time() - t2) * 1000

    # Phase 3: Maintenance
    t3 = time.time()
    p3_result = _phase3_maintenance(
        brain, decoder, community_state, edges_by_node,
        typed_neighbors, already_placed, node_to_community, config)
    phase3_ms = (time.time() - t3) * 1000

    # Combine all proposals from all phases
    raw_proposals = p1_result['proposals'] + p2_result['proposals'] + p3_result['proposals']
    raw_count = len(raw_proposals)

    # ── Unified fingerprint suppression ──
    proposals, suppressed_count = filter_rejected(brain, raw_proposals)

    total_ms = (time.time() - t0) * 1000

    # Count suppressed by type
    suppressed_proposals = [p for p in raw_proposals if p not in proposals]
    suppressed_by_type = dict(Counter(p['type'] for p in suppressed_proposals)) \
        if suppressed_proposals else {}

    # Collect proposed node IDs for cross-run tracking
    proposed_node_ids = set()
    for p in proposals:
        if p.get('node_id'):
            proposed_node_ids.add(p['node_id'])
        for m in p.get('members', []):
            proposed_node_ids.add(m)

    stats = {
        'unplaced': len(unplaced),
        'communities': len(community_state),
        'raw_proposals': raw_count,
        'suppressed_count': suppressed_count,
        'suppressed_by_type': suppressed_by_type,
        'total_proposals': len(proposals),
        'adjacency_build_ms': round(adj_ms, 1),
        'phase1_ms': round(phase1_ms, 1),
        'phase2_ms': round(phase2_ms, 1),
        'phase3_ms': round(phase3_ms, 1),
        'total_ms': round(total_ms, 1),
        # Phase 1
        'phase1_raw': len(p1_result['proposals']),
        'phase1_reached': len(p1_result['reached']),
        'phase1_coverage': round(len(p1_result['reached']) / len(unplaced), 3)
        if unplaced else 0,
        'phase1_affinities': p1_result.get('affinities', []),
        # Phase 2
        'phase2_raw': len(p2_result['proposals']),
        'orphan_pool_size': p2_result['orphan_pool_size'],
        'clusters_seeded': p2_result.get('clusters_seeded', 0),
        'clusters_valid': p2_result.get('clusters_valid', 0),
        # Phase 3
        'phase3_raw': len(p3_result['proposals']),
        'drift_proposals': p3_result.get('drift_count', 0),
        'health_proposals': p3_result.get('health_count', 0),
        'merge_proposals': p3_result.get('merge_count', 0),
    }

    return {
        'proposals': proposals,
        'stats': stats,
        'community_state': community_state,
        'unplaced_count': len(unplaced),
        'edges_by_node': edges_by_node,
        'proposed_node_ids': proposed_node_ids,
    }


# ── Phase 1: Inside-Out ──

def _phase1_inside_out(community_state, typed_neighbors, unplaced, config):
    """Scan from community members outward to find unplaced neighbors.

    No suppression here — generates all eligible proposals.
    Fingerprint filtering happens after all phases complete.
    """
    min_affinity = config.get('add_to_existing_min_affinity', 0.25)
    reached = set()
    proposals = []
    affinities = []

    for comm in community_state:
        comm_members = comm['members']
        if not comm_members:
            continue

        # Find unplaced nodes in the neighborhood of this community
        frontier = set()
        for mid in comm_members:
            nbrs = typed_neighbors.get(mid, set())
            for nbr in nbrs:
                if nbr in unplaced:
                    frontier.add(nbr)

        # Compute affinity for each frontier node
        for nid in frontier:
            nbrs = typed_neighbors.get(nid, set())
            if not nbrs:
                continue
            shared = len(nbrs & comm_members)
            aff = shared / len(nbrs)
            affinities.append(aff)

            if aff >= min_affinity:
                reached.add(nid)
                proposals.append({
                    'type': 'add_to_existing',
                    'node_id': nid,
                    'source': 'inside_out',
                    'community_id': comm['id'],
                    'community_title': comm['title'],
                    'affinity': round(aff, 3),
                })

    # Deduplicate: keep only the highest-affinity proposal per node
    best_by_node = {}
    for p in proposals:
        nid = p['node_id']
        if nid not in best_by_node or p['affinity'] > best_by_node[nid]['affinity']:
            best_by_node[nid] = p
    proposals = list(best_by_node.values())

    return {
        'proposals': proposals,
        'reached': reached,
        'affinities': affinities,
    }


# ── Phase 2: Orphan Clustering ──

def _phase2_cluster_orphans(decoder, edges_by_node, typed_neighbors,
                            unplaced, phase1_reached, community_state, config):
    """Cluster orphan nodes (unplaced nodes not reached by Phase 1).

    No suppression here — generates all proposals from clustering.
    Fingerprint filtering happens after all phases complete.
    """
    orphan_pool = unplaced - phase1_reached

    if not orphan_pool:
        return {
            'proposals': [], 'orphan_pool_size': 0,
            'clusters_seeded': 0, 'clusters_valid': 0,
        }

    # Compute pair scores on FULL graph (for z-score normalization),
    # then filter to pairs with at least one orphan
    pair_zscores, degrees, bucket_stats = decoder._compute_pair_scores(
        typed_neighbors, edges_by_node)

    pair_zscores = {
        (a, b): z for (a, b), z in pair_zscores.items()
        if a in orphan_pool or b in orphan_pool
    }

    direct_pairs = decoder._get_direct_pairs(edges_by_node)
    direct_pairs = {
        (a, b) for a, b in direct_pairs
        if a in orphan_pool or b in orphan_pool
    }

    clusters = decoder._seed_clusters(pair_zscores, direct_pairs)
    valid_clusters, corridors, dissolved = decoder._validate_clusters(
        clusters, edges_by_node, typed_neighbors)
    valid_clusters, absorbed = decoder._absorb_subsets(valid_clusters)

    # Build proposals
    proposals = decoder._build_proposals(
        valid_clusters, corridors,
        decoder._compute_affinities(valid_clusters, typed_neighbors),
        {},  # No orphan embedding affinities
        set(),  # No cross-cutting detection
        {'genuine_overlaps': 0, 'possible_splits': 0},
        edges_by_node, typed_neighbors, community_state)

    # Filter to actionable types only
    proposals = [p for p in proposals
                 if p['type'] in ('new_community', 'add_to_existing')]

    return {
        'proposals': proposals,
        'orphan_pool_size': len(orphan_pool),
        'clusters_seeded': len(clusters),
        'clusters_valid': len(valid_clusters),
    }


# ── Phase 3: Maintenance ──

def _phase3_maintenance(brain, decoder, community_state, edges_by_node,
                        typed_neighbors, already_placed, node_to_community,
                        config):
    """Drift, health, and merge detection — reuses production logic.

    No suppression here — fingerprint filtering happens after all phases.
    """
    proposals = []
    drift_count = health_count = merge_count = 0

    # Drift detection (production Step 5c)
    default_drift_ratio = config.get('drift_ratio', 1.5)
    min_foreign_aff = config.get('drift_min_foreign_affinity', 0.15)

    node_drift_thresholds = {}
    drift_rows = brain.conn.execute(
        "SELECT node_id, value FROM node_metadata_kv "
        "WHERE key = '_sys_drift_threshold'").fetchall()
    for _nid, _val in drift_rows:
        try:
            node_drift_thresholds[_nid] = float(_val)
        except (ValueError, TypeError):
            pass

    titles = {}
    types_map = {}
    for row in brain.conn.execute(
            "SELECT id, title, type FROM nodes WHERE archived = 0"):
        titles[row[0]] = row[1][:60]
        types_map[row[0]] = row[2]

    drift_candidates = {}
    for nid in already_placed:
        if nid not in typed_neighbors:
            continue
        nbrs = typed_neighbors[nid]
        if not nbrs:
            continue
        drift_ratio = node_drift_thresholds.get(nid, default_drift_ratio)
        home = node_to_community.get(nid)
        if not home:
            continue
        home_aff = len(nbrs & home['members']) / len(nbrs)

        for comm in community_state:
            if comm['id'] == home['id']:
                continue
            foreign_aff = len(nbrs & comm['members']) / len(nbrs)
            if foreign_aff > home_aff * drift_ratio and foreign_aff > min_foreign_aff:
                if nid not in drift_candidates:
                    drift_candidates[nid] = {
                        'home': home, 'home_aff': home_aff,
                        'drift_ratio': drift_ratio, 'foreign': []}
                drift_candidates[nid]['foreign'].append(
                    (comm['id'], comm['title'], foreign_aff))

    for nid, drift in drift_candidates.items():
        proposals.append({
            'type': 'drift',
            'node_id': nid,
            'node_title': titles.get(nid, nid[:8]),
            'node_type': types_map.get(nid, '?'),
            'home_community': drift['home']['title'],
            'home_affinity': drift['home_aff'],
            'current_drift_threshold': drift.get('drift_ratio', default_drift_ratio),
            'foreign': [
                {'id': cid, 'title': ctitle, 'affinity': aff}
                for cid, ctitle, aff in drift['foreign'][:3]],
        })
        drift_count += 1

    # Health updates (production Step 5d)
    from servers.scales.s2.community_decoder import read_community_meta

    for comm in community_state:
        if not comm['members']:
            continue
        ms = comm['members']
        internal = sum(1 for n in ms
                       for nbr, _, _ in edges_by_node.get(n, [])
                       if nbr in ms) // 2
        external = sum(1 for n in ms
                       for nbr, _, _ in edges_by_node.get(n, [])
                       if nbr not in ms)
        int_frac = internal / (internal + external) if (internal + external) else 0

        old_frac = read_community_meta(
            brain.conn, comm['id'], 'community_internal_fraction', type='float')
        old_maturity = read_community_meta(
            brain.conn, comm['id'], 'community_maturity', type='str')

        signal = None
        if int_frac < 0.05 and len(ms) > 0:
            signal = 'dead'
        elif old_frac > 0 and int_frac < old_frac * 0.7:
            signal = 'degrading'
        elif old_maturity == 'corridor' and int_frac > 0.3:
            signal = 'corridor_maturing'

        if signal:
            proposals.append({
                'type': 'health_update',
                'community_id': comm['id'],
                'community_title': comm['title'],
                'signal': signal,
                'old_fraction': old_frac,
                'new_fraction': int_frac,
            })
            health_count += 1

    # Merge detection (production Step 5e)
    merge_candidates = decoder._detect_merge_candidates(community_state)
    for merge in merge_candidates:
        proposals.append({
            'type': 'merge_communities',
            'larger_id': merge['larger']['id'],
            'larger_title': merge['larger']['title'],
            'larger_size': len(merge['larger']['members']),
            'smaller_id': merge['smaller']['id'],
            'smaller_title': merge['smaller']['title'],
            'smaller_size': len(merge['smaller']['members']),
            'shared_count': merge['shared_count'],
            'overlap_pct': merge['overlap_pct'],
            'unique_in_smaller': merge['unique_in_smaller'],
        })
        merge_count += 1

    return {
        'proposals': proposals,
        'drift_count': drift_count,
        'health_count': health_count,
        'merge_count': merge_count,
    }


# ═══════════════════════════════════════════════════════════════
# ACCEPTANCE SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_acceptance(brain, proposals, edges_by_node, accept_rate=0.6,
                        run_seed=42):
    """Simulate encoder acceptance/rejection without LLM calls.

    Accepted: create community nodes/edges in the DB.
    Rejected: write fingerprint to s2_rejections table.
    """
    from servers.dal import GraphDAL

    rng = random.Random(run_seed)
    graph_dal = GraphDAL(brain.conn)
    ts = datetime.now(timezone.utc).isoformat()

    accepted = []
    rejected = []
    members_placed = 0
    communities_created = 0
    community_sizes = []

    for p in proposals:
        decision = rng.random() < accept_rate

        if decision:
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
                    try:
                        graph_dal.add_relation(
                            node_id, mid, 'community_member', weight=0.3,
                            encoding_source='s2:community_detection')
                    except ValueError:
                        pass
                members_placed += len(member_ids)
                communities_created += 1
                community_sizes.append(len(member_ids))

            elif p['type'] == 'add_to_existing':
                comm_id = p.get('community_id')
                nid = p.get('node_id')
                if comm_id and nid:
                    try:
                        graph_dal.add_relation(
                            comm_id, nid, 'community_member', weight=0.3,
                            encoding_source='s2:community_detection')
                        members_placed += 1
                    except ValueError:
                        pass

            elif p['type'] == 'drift':
                nid = p.get('node_id')
                foreign = p.get('foreign', [])
                if nid and foreign:
                    try:
                        graph_dal.add_relation(
                            foreign[0]['id'], nid, 'community_member',
                            weight=0.3,
                            encoding_source='s2:community_detection')
                    except ValueError:
                        pass

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
                    """, (smaller_id, smaller_id, smaller_id)).fetchall())
                    for mid in smaller_members:
                        try:
                            graph_dal.add_relation(
                                larger_id, mid, 'community_member',
                                weight=0.3,
                                encoding_source='s2:community_detection')
                        except ValueError:
                            pass
                    brain.conn.execute(
                        "UPDATE nodes SET archived = 1 WHERE id = ?",
                        (smaller_id,))
        else:
            rejected.append(p)

    # Record all rejections as fingerprints
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
# MULTI-RUN LOOP
# ═══════════════════════════════════════════════════════════════

def run_multi(brain, n_runs=3, accept_rate=0.6, seed=42, config=None):
    """Run decoder -> simulate -> repeat for n_runs."""
    runs = []

    for i in range(n_runs):
        result = run_new_decoder(brain, config)
        proposed_ids = result.get('proposed_node_ids', set())

        sim = simulate_acceptance(
            brain, result['proposals'], result['edges_by_node'],
            accept_rate=accept_rate, run_seed=seed + i)

        # Count rejection table size
        rejection_count = brain.conn.execute(
            "SELECT COUNT(*) FROM s2_rejections").fetchone()[0]

        runs.append({
            'run': i + 1,
            'stats': result['stats'],
            'simulation': sim,
            'proposed_node_ids': proposed_ids,
            'rejection_table_size': rejection_count,
        })

    return runs


# ═══════════════════════════════════════════════════════════════
# OLD DECODER (for comparison)
# ═══════════════════════════════════════════════════════════════

def run_old_decoder(brain):
    """Run production CommunityDecoder for comparison."""
    from servers.scales.s2.community_decoder import CommunityDecoder

    decoder = CommunityDecoder(brain)
    t0 = time.time()
    result = decoder.run()
    elapsed = (time.time() - t0) * 1000

    proposals = result.get('proposals', [])
    return {
        'proposals': proposals,
        'proposal_count': len(proposals),
        'by_type': dict(Counter(p['type'] for p in proposals)),
        'unplaced': result.get('unplaced_count', 0),
        'stats': result.get('stats', {}),
        'elapsed_ms': round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════

def run_comparison(brain, config=None):
    """Run old vs new decoder on same brain state."""
    old = run_old_decoder(brain)
    new_result = run_new_decoder(brain, config)

    new = {
        'proposal_count': new_result['stats']['total_proposals'],
        'by_type': dict(Counter(p['type'] for p in new_result['proposals'])),
        'unplaced': new_result['unplaced_count'],
        'phase1_raw': new_result['stats']['phase1_raw'],
        'phase1_coverage': new_result['stats']['phase1_coverage'],
        'phase2_raw': new_result['stats']['phase2_raw'],
        'phase3_raw': new_result['stats']['phase3_raw'],
        'suppressed_count': new_result['stats']['suppressed_count'],
        'elapsed_ms': new_result['stats']['total_ms'],
    }

    return {'old': old, 'new': new}


# ═══════════════════════════════════════════════════════════════
# CROSS-RUN METRICS
# ═══════════════════════════════════════════════════════════════

def compute_cross_run_metrics(runs):
    """Analyze trajectory across multiple runs."""
    if not runs:
        return {}

    backlog = [r['stats']['unplaced'] for r in runs]
    raw_proposals = [r['stats']['raw_proposals'] for r in runs]
    proposals = [r['stats']['total_proposals'] for r in runs]
    suppressed = [r['stats']['suppressed_count'] for r in runs]
    rejection_table = [r.get('rejection_table_size', 0) for r in runs]

    phase1_share = []
    for r in runs:
        total = r['stats']['total_proposals']
        # Count surviving Phase 1 proposals (we know raw, but not post-filter by phase)
        # Use raw as proxy — suppression is uniform across phases
        p1_raw = r['stats']['phase1_raw']
        total_raw = r['stats']['raw_proposals']
        phase1_share.append(round(p1_raw / total_raw, 3) if total_raw else 0)

    # Re-proposal rate
    re_proposal_rates = [0]
    for i in range(1, len(runs)):
        prev_ids = runs[i - 1]['proposed_node_ids']
        curr_ids = runs[i]['proposed_node_ids']
        if curr_ids:
            overlap = len(prev_ids & curr_ids)
            re_proposal_rates.append(round(overlap / len(curr_ids), 3))
        else:
            re_proposal_rates.append(0)

    convergence = round(1 - (proposals[-1] / proposals[0]), 3) if proposals[0] else 0

    members_placed_total = sum(r['simulation']['members_placed'] for r in runs)
    communities_created_total = sum(r['simulation']['communities_created'] for r in runs)

    return {
        'backlog_trajectory': backlog,
        'raw_proposal_trajectory': raw_proposals,
        'proposal_trajectory': proposals,
        'suppression_trajectory': suppressed,
        'rejection_table_trajectory': rejection_table,
        'phase1_share_trajectory': phase1_share,
        're_proposal_rates': re_proposal_rates,
        'convergence_score': convergence,
        'total_members_placed': members_placed_total,
        'total_communities_created': communities_created_total,
    }


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════

def print_report(comparison, runs, cross_metrics, brain):
    """Print comprehensive eval report."""
    SEP = '=' * 70

    node_count = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
    comm_count = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE type = 'community' AND archived = 0"
    ).fetchone()[0]

    print(SEP)
    print('S2 COMMUNITY DECODER EVAL — Fingerprint Suppression')
    print(SEP)

    if comparison:
        old = comparison['old']
        new = comparison['new']

        print('\nBrain: %d nodes, %d communities, %d unplaced' % (
            node_count, comm_count, old['unplaced']))

        print('\n--- COMPARISON (same brain state, no prior rejections) ---\n')
        print('  OLD decoder: %d proposals in %.0fms' % (
            old['proposal_count'], old['elapsed_ms']))
        for ptype, count in sorted(old['by_type'].items(), key=lambda x: -x[1]):
            print('    %-20s %d' % (ptype, count))

        print()
        print('  NEW decoder: %d proposals in %.0fms (suppressed: %d)' % (
            new['proposal_count'], new['elapsed_ms'], new['suppressed_count']))
        print('    Phase 1 (inside-out):  %d raw  (coverage: %.0f%%)' % (
            new['phase1_raw'], new['phase1_coverage'] * 100))
        print('    Phase 2 (clustering):  %d raw' % new['phase2_raw'])
        print('    Phase 3 (maintenance): %d raw' % new['phase3_raw'])
        for ptype, count in sorted(new['by_type'].items(), key=lambda x: -x[1]):
            print('      %-20s %d' % (ptype, count))

        ratio = new['elapsed_ms'] / old['elapsed_ms'] if old['elapsed_ms'] else 0
        print('\n  Latency: old=%.0fms new=%.0fms (%.1fx)' % (
            old['elapsed_ms'], new['elapsed_ms'], ratio))

    # Per-run details
    if runs:
        print('\n' + SEP)
        print('MULTI-RUN SIMULATION (accept_rate=%.0f%%)' % (
            runs[0]['simulation']['accepted'] /
            max(1, runs[0]['simulation']['accepted'] + runs[0]['simulation']['rejected']) * 100))
        print(SEP)

        for r in runs:
            s = r['stats']
            sim = r['simulation']
            print('\n--- RUN %d ---\n' % r['run'])

            # Performance
            print('  Performance: total=%.0fms (adj=%.0f, p1=%.0f, p2=%.0f, p3=%.0f)' % (
                s['total_ms'], s['adjacency_build_ms'],
                s['phase1_ms'], s['phase2_ms'], s['phase3_ms']))

            # Decoder
            print('  Decoder:')
            print('    Unplaced: %d | Communities: %d' % (
                s['unplaced'], s['communities']))
            print('    Phase 1: %d raw (coverage: %.0f%%, reached: %d)' % (
                s['phase1_raw'], s['phase1_coverage'] * 100,
                s['phase1_reached']))
            print('    Phase 2: %d raw (orphan_pool: %d, clusters: %d/%d seeded/valid)' % (
                s['phase2_raw'], s['orphan_pool_size'],
                s['clusters_seeded'], s['clusters_valid']))
            print('    Phase 3: %d raw (drift=%d, health=%d, merge=%d)' % (
                s['phase3_raw'], s['drift_proposals'],
                s['health_proposals'], s['merge_proposals']))
            print('    Suppressed: %d (rejection table: %d entries)' % (
                s['suppressed_count'], r.get('rejection_table_size', 0)))
            if s.get('suppressed_by_type'):
                print('      by type: %s' % '  '.join(
                    '%s=%d' % (t, c) for t, c in
                    sorted(s['suppressed_by_type'].items(), key=lambda x: -x[1])))
            print('    Surviving: %d proposals' % s['total_proposals'])

            # Affinity distribution
            affs = s.get('phase1_affinities', [])
            if affs:
                bins = {'0.25-0.35': 0, '0.35-0.50': 0, '0.50-0.75': 0, '0.75+': 0}
                for a in affs:
                    if a >= 0.75:
                        bins['0.75+'] += 1
                    elif a >= 0.50:
                        bins['0.50-0.75'] += 1
                    elif a >= 0.35:
                        bins['0.35-0.50'] += 1
                    elif a >= 0.25:
                        bins['0.25-0.35'] += 1
                print('    Affinity distribution (>= 0.25): %s' %
                      '  '.join('%s=%d' % (k, v) for k, v in bins.items() if v))

            # Simulation
            print('  Simulation:')
            print('    Accepted: %d | Rejected: %d' % (
                sim['accepted'], sim['rejected']))
            print('    Placed: %d members | Created: %d communities' % (
                sim['members_placed'], sim['communities_created']))
            if sim['community_sizes']:
                print('    Community sizes: %s' % sim['community_sizes'])
            if sim.get('accepted_by_type'):
                print('    Accepted: %s' % '  '.join(
                    '%s=%d' % (t, c) for t, c in
                    sorted(sim['accepted_by_type'].items(), key=lambda x: -x[1])))
            if sim.get('rejected_by_type'):
                print('    Rejected: %s' % '  '.join(
                    '%s=%d' % (t, c) for t, c in
                    sorted(sim['rejected_by_type'].items(), key=lambda x: -x[1])))

    # Cross-run trajectory
    if cross_metrics:
        cm = cross_metrics
        print('\n' + SEP)
        print('CROSS-RUN TRAJECTORY')
        print(SEP)

        def _trend(vals):
            if len(vals) < 2:
                return ''
            delta = vals[-1] - vals[0]
            pct = delta / vals[0] * 100 if vals[0] else 0
            arrow = '\u2193' if delta < 0 else '\u2191' if delta > 0 else '\u2192'
            return '%s %.1f%%' % (arrow, abs(pct))

        print()
        print('  %-22s %-35s %s' % ('Metric', 'Trajectory', 'Trend'))
        print('  ' + '-' * 72)
        print('  %-22s %-35s %s' % (
            'Backlog (unplaced)',
            ' \u2192 '.join(str(v) for v in cm['backlog_trajectory']),
            _trend(cm['backlog_trajectory'])))
        print('  %-22s %-35s %s' % (
            'Raw proposals',
            ' \u2192 '.join(str(v) for v in cm['raw_proposal_trajectory']),
            _trend(cm['raw_proposal_trajectory'])))
        print('  %-22s %-35s %s' % (
            'After suppression',
            ' \u2192 '.join(str(v) for v in cm['proposal_trajectory']),
            _trend(cm['proposal_trajectory'])))
        print('  %-22s %-35s %s' % (
            'Suppressed',
            ' \u2192 '.join(str(v) for v in cm['suppression_trajectory']),
            _trend(cm['suppression_trajectory'])))
        print('  %-22s %-35s %s' % (
            'Rejection table',
            ' \u2192 '.join(str(v) for v in cm['rejection_table_trajectory']),
            _trend(cm['rejection_table_trajectory'])))
        print('  %-22s %-35s' % (
            'Phase 1 share (raw)',
            ' \u2192 '.join('%.0f%%' % (v * 100) for v in cm['phase1_share_trajectory'])))
        print('  %-22s %s' % (
            'Re-proposal rate',
            ' \u2192 '.join('%.0f%%' % (v * 100) for v in cm['re_proposal_rates'])))
        print()
        print('  Convergence score: %.2f  (0=none, 1=full)' % cm['convergence_score'])
        print('  Total placed: %d members in %d new communities' % (
            cm['total_members_placed'], cm['total_communities_created']))

        # Risk assessment
        print('\n' + SEP)
        print('RISK ASSESSMENT')
        print(SEP)

        risks = []
        bt = cm['backlog_trajectory']
        if len(bt) >= 2 and bt[-1] >= bt[0]:
            risks.append('BACKLOG NOT SHRINKING \u2014 unplaced count is flat or growing')

        st = cm['suppression_trajectory']
        if len(st) >= 3 and st[-1] > 0:
            suppress_pct = st[-1] / cm['raw_proposal_trajectory'][-1] * 100 \
                if cm['raw_proposal_trajectory'][-1] else 0
            if suppress_pct > 80:
                risks.append(
                    'OVER-SUPPRESSION \u2014 %.0f%% of raw proposals suppressed. '
                    'Rejection table may need expiry.' % suppress_pct)

        p1s = cm['phase1_share_trajectory']
        if len(p1s) >= 2 and all(v > 0.90 for v in p1s):
            risks.append('COMMUNITY MONOPOLIZATION \u2014 Phase 1 dominates (>90%), '
                         'new communities may not form')

        if cm['convergence_score'] < 0.05:
            risks.append('NO CONVERGENCE \u2014 proposals not decreasing across runs')

        rr = cm['re_proposal_rates']
        if len(rr) >= 2 and rr[-1] > 0.5:
            risks.append('HIGH RE-PROPOSAL \u2014 %.0f%% of proposed node IDs were in previous run '
                         '(note: fingerprint suppression handles exact duplicates, '
                         'this tracks node-level overlap)' % (rr[-1] * 100))

        rt = cm['rejection_table_trajectory']
        if len(rt) >= 2:
            growth = rt[-1] - rt[0]
            if growth > 2000:
                risks.append(
                    'REJECTION TABLE GROWTH \u2014 %d entries after %d runs. '
                    'May need periodic expiry for stale fingerprints.' % (
                        rt[-1], len(rt)))

        if not risks:
            print('\n  No structural risks detected.')
        else:
            for risk in risks:
                print('\n  [!] %s' % risk)

    print('\n' + SEP)


def save_report(path, comparison, runs, cross_metrics):
    """Save full report as JSON."""
    data = {
        'timestamp': datetime.utcnow().isoformat(),
        'comparison': comparison,
        'runs': [{
            'run': r['run'],
            'stats': r['stats'],
            'simulation': r['simulation'],
            'rejection_table_size': r.get('rejection_table_size', 0),
        } for r in runs] if runs else [],
        'cross_metrics': cross_metrics,
    }
    for r in data.get('runs', []):
        r.get('stats', {}).pop('phase1_affinities', None)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print('Saved to %s' % path)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='S2 Community Decoder Eval — Fingerprint Suppression')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of decoder+simulate runs (default: 3)')
    parser.add_argument('--accept-rate', type=float, default=0.6,
                        help='Simulated acceptance rate (default: 0.6)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only run old vs new comparison, no multi-run')
    parser.add_argument('--keep', action='store_true',
                        help='Keep temp directory for inspection')
    parser.add_argument('--save', help='Save report to JSON file')
    args = parser.parse_args()

    from tests.isolated_brain import IsolatedBrain

    print('Setting up isolated brain copy...')
    with IsolatedBrain(cleanup=not args.keep) as env:
        brain = env.brain
        print('Isolated brain at: %s' % env.db_dir)

        # Create rejection table
        create_rejection_table(brain)

        # Comparison: old vs new on same state (no prior rejections)
        print('\nRunning comparison (old vs new decoder)...')
        comparison = run_comparison(brain, config=None)

        runs = None
        cross_metrics = None

        if not args.compare_only:
            print('\nRunning %d decoder+simulation iterations...' % args.runs)
            runs = run_multi(
                brain, n_runs=args.runs,
                accept_rate=args.accept_rate, seed=args.seed)
            cross_metrics = compute_cross_run_metrics(runs)

        print_report(comparison, runs, cross_metrics, brain)

        if args.save:
            save_report(args.save, comparison, runs, cross_metrics)

        if args.keep:
            print('\nTemp dir preserved: %s' % env.db_dir)


if __name__ == '__main__':
    main()
