"""S2CD — Community Decoder.

Reads graph structure → produces community proposals. Pure algorithmic,
no LLM calls. Runs in <1s on a 2500-node graph.

Algorithm:
1. Build typed adjacency (edge families, skip noise)
2. Z-score pair scoring within degree buckets
3. Seed clusters from high-z pairs + direct edges
4. Validate: dissolve fragments, flag corridors
5. Compute affinities (new clusters + existing communities)
6. Detect cross-cutting nodes (high-degree, thin spread)
7. Embedding placement for orphans
8. Tie analysis (genuine overlap vs split)
9. Build proposals for encoder

Incremental: placed nodes excluded from seeding. New nodes matched
against existing communities. Drift detected when foreign affinity
exceeds home affinity × 1.5.
"""

import json
import base64
from collections import defaultdict, Counter

import numpy as np

from .base import IntegrationUnit
from .community_contract import (
    COMMUNITY_DETECTION,
    COMMUNITY_METADATA_KEYS,
)
from servers.embedder import cosine_similarity


# ── Module-level helpers ──

def _safe_float(value, default=0.0):
    """Parse a metadata value to float, handling %, commas, text."""
    if not value:
        return default
    s = str(value).strip().rstrip('%').replace(',', '')
    try:
        result = float(s)
        if '%' in str(value):
            result /= 100.0
        return result
    except (ValueError, TypeError):
        return default


def _safe_int(value, default=0):
    """Parse a metadata value to int, handling text."""
    if not value:
        return default
    s = str(value).strip().replace(',', '')
    s = s.split()[0] if ' ' in s else s
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def read_community_meta(conn, node_id, key, type='str'):
    """Read a single community metadata value with safe parsing.

    type: 'str', 'float', 'int', 'bool'
    """
    row = conn.execute(
        "SELECT value FROM node_metadata_kv WHERE node_id=? AND key=?",
        (node_id, key)).fetchone()
    if not row or not row[0]:
        return '' if type == 'str' else 0.0 if type == 'float' else 0 if type == 'int' else False
    val = row[0]
    if type == 'float':
        return _safe_float(val)
    elif type == 'int':
        return _safe_int(val)
    elif type == 'bool':
        return str(val).lower() in ('true', '1', 'yes')
    return val


class CommunityDecoder(IntegrationUnit):
    NAME = 'community_detection'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:community_detection'

    O_SOURCES = ['s1_delta', 'graph_structure']
    K_SOURCES = ['semantic_similarity', 'relational_signals', 'usage_traces']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or COMMUNITY_DETECTION

    def run(self):
        """Decode only — returns proposals for the encoder.

        Returns:
            dict with keys: proposals, stats, community_state, s1_delta,
                            is_cold_start, skipped (if no new traces)
        """
        # v23: Always incremental — find unplaced nodes, not trace deltas.
        # Survives partial failures: if encoder placed 50/100 before crash,
        # next run finds the remaining 50.
        community_state = self._read_community_state()

        # Count unplaced nodes for the trace
        already_placed = set()
        for comm in community_state:
            for mid in comm.get('members', []):
                already_placed.add(mid)
        all_active = set(r[0] for r in self.brain.conn.execute(
            "SELECT id FROM nodes WHERE archived = 0 AND type != 'community'"
        ).fetchall())
        unplaced_count = len(all_active - already_placed)

        if unplaced_count == 0:
            return {'proposals': [], 'skipped': 'all nodes placed'}

        # Build s1_delta for compat with _decode interface
        s1_delta = {
            'encoding_runs': [],
            'new_node_ids': all_active - already_placed,
        }

        self.trace('O', 's1_delta',
                   '%d unplaced nodes, %d communities' % (
                       unplaced_count, len(community_state)))

        decode_result = self._decode(s1_delta, community_state, is_cold_start=False)
        proposals = decode_result['proposals']
        proposals.extend(self._detect_recall_signals(s1_delta, community_state))

        if not proposals:
            self.trace('K', 'community_proposals', 'No proposals generated')

        # Rich trace with decode state
        if proposals:
            self.trace('K', 'community_proposals',
                       '%d proposals: %d clusters, %d affinities, %d cross-cutting' % (
                           len(proposals),
                           decode_result['stats'].get('valid_clusters', 0),
                           decode_result['stats'].get('nodes_with_affinities', 0),
                           decode_result['stats'].get('cross_cutting', 0)),
                       metadata={
                           'decode_stats': decode_result['stats'],
                           'cluster_summaries': decode_result.get('cluster_summaries', []),
                       })

        return {
            'proposals': proposals,
            'stats': decode_result.get('stats', {}),
            'community_state': community_state,
            's1_delta': s1_delta,
            'unplaced_count': unplaced_count,
        }

    # ══════════════════════════════════════════════════════════
    # _read_s1_delta
    # ══════════════════════════════════════════════════════════

    def _read_s1_delta(self, since_ts):
        encoding_runs = self._read_traces_since(
            's1', since_ts or '', ref_types=['encoding_run'])
        surface_selections = self._read_traces_since(
            's1', since_ts or '', ref_types=['surface_selected'])

        new_node_ids = set()
        for trace in encoding_runs:
            meta = trace.get('metadata', {})
            if isinstance(meta, dict):
                for nid in meta.get('created', []):
                    if nid:
                        new_node_ids.add(nid)
                for nid in meta.get('revised', []):
                    if nid:
                        new_node_ids.add(nid)

        co_surface_pairs = []
        for trace in surface_selections:
            ref_id = trace.get('ref_id', '')
            if ref_id:
                try:
                    ids = json.loads(ref_id)
                    if isinstance(ids, list) and len(ids) > 1:
                        for i in range(len(ids)):
                            for j in range(i + 1, len(ids)):
                                co_surface_pairs.append((ids[i], ids[j]))
                except (json.JSONDecodeError, TypeError):
                    pass

        return {
            'encoding_runs': encoding_runs,
            'surface_selections': surface_selections,
            'new_node_ids': new_node_ids,
            'co_surface_pairs': co_surface_pairs,
        }

    # ══════════════════════════════════════════════════════════
    # _read_community_state
    # ══════════════════════════════════════════════════════════

    def _read_community_state(self):
        communities = []
        rows = self.brain.conn.execute(
            "SELECT id, title, content, keywords, confidence "
            "FROM nodes WHERE type = 'community' AND archived = 0 "
            "AND encoding_source = ?",
            (self.ENCODING_SOURCE,)).fetchall()

        for nid, title, content, keywords, conf in rows:
            meta = {}
            for k, v in self.brain.conn.execute(
                    "SELECT key, value FROM node_metadata_kv WHERE node_id = ?",
                    (nid,)).fetchall():
                if k in COMMUNITY_METADATA_KEYS:
                    try:
                        meta[k] = json.loads(v) if v else None
                    except (json.JSONDecodeError, TypeError):
                        meta[k] = v

            centroid = None
            centroid_b64 = meta.get('community_centroid')
            if centroid_b64 and isinstance(centroid_b64, str):
                try:
                    centroid = base64.b64decode(centroid_b64)
                except Exception:
                    pass

            # Member IDs via GraphDAL (archived=0 default, v25).
            from servers.dal import GraphDAL
            member_dicts = GraphDAL(self.brain.conn).get_community_members(
                nid, require_active_member=False)
            members = {m['id'] for m in member_dicts}

            communities.append({
                'id': nid, 'title': title, 'content': content,
                'keywords': keywords, 'confidence': conf,
                'members': members,
                'centroid': centroid,
                'edge_signature': meta.get('community_edge_signature', {}),
                'health': meta.get('community_health', {}),
            })

        return communities

    # ══════════════════════════════════════════════════════════
    # _decode — the 9-step pipeline
    # ══════════════════════════════════════════════════════════

    def _decode(self, s1_delta, community_state, is_cold_start=False):
        # Load edge families
        edge_families_config = self._get_interaction_config(
            's2_edge_families') or {}
        rel_to_fam = {}
        for fam, members in edge_families_config.items():
            if isinstance(members, list):
                for m in members:
                    rel_to_fam[m] = fam
        skip_fams = {'generic_relation', 'noise'}

        # Step 1: Typed adjacency (ALL nodes — placed nodes' edges matter)
        edges_by_node, typed_neighbors = self._build_typed_adjacency(
            rel_to_fam, skip_fams)

        if len(edges_by_node) < self.config['min_community_size']:
            return {'proposals': [], 'stats': {}}

        # Identify already-placed nodes (members of existing communities)
        already_placed = set()
        node_to_community = {}
        for comm in community_state:
            for member_id in comm['members']:
                already_placed.add(member_id)
                node_to_community[member_id] = comm

        unplaced = set(edges_by_node.keys()) - already_placed

        # Step 2: Z-score pairs — only pairs with at least one UNPLACED node
        pair_zscores, degrees, bucket_stats = self._compute_pair_scores(
            typed_neighbors, edges_by_node)

        # Always filter to pairs with at least one unplaced node
        pair_zscores = {
            (a, b): z for (a, b), z in pair_zscores.items()
            if a in unplaced or b in unplaced
        }

        # Step 3: Seed clusters from unplaced nodes
        direct_pairs = self._get_direct_pairs(edges_by_node)
        direct_pairs = {
            (a, b) for a, b in direct_pairs
            if a in unplaced or b in unplaced
        }
        clusters = self._seed_clusters(pair_zscores, direct_pairs)

        # Step 4: Validate
        valid_clusters, corridors, dissolved_count = self._validate_clusters(
            clusters, edges_by_node, typed_neighbors)

        # Step 4b: Absorb subsets — when one cluster is fully contained
        # in another, merge the smaller into the larger. The larger cluster
        # keeps all members (no information loss). The smaller is dissolved.
        valid_clusters, absorbed_count = self._absorb_subsets(valid_clusters)

        # Step 5: Affinities for unplaced nodes to NEW clusters
        node_affinities = self._compute_affinities(
            valid_clusters, typed_neighbors)

        # Step 5b: Affinities for unplaced nodes to EXISTING communities
        add_min_affinity = self.config.get('add_to_existing_min_affinity', 0.25)
        existing_additions = {}
        if community_state:
            for nid in unplaced:
                nbrs = typed_neighbors.get(nid, set())
                if not nbrs:
                    continue
                for comm in community_state:
                    shared = len(nbrs & comm['members'])
                    if shared > 0:
                        aff = shared / len(nbrs)
                        if aff >= add_min_affinity:
                            if nid not in existing_additions:
                                existing_additions[nid] = []
                            existing_additions[nid].append(
                                (comm['id'], comm['title'], aff))

            for nid in existing_additions:
                existing_additions[nid].sort(key=lambda x: -x[2])

        # Step 5c: Drift detection for PLACED nodes
        # Per-node threshold: encoder raises _sys_drift_threshold on rejection
        default_drift_ratio = self.config.get('drift_ratio', 1.5)
        min_foreign_aff = self.config.get('drift_min_foreign_affinity', 0.15)

        # Bulk-load per-node drift thresholds
        node_drift_thresholds = {}
        drift_rows = self.brain.conn.execute(
            "SELECT node_id, value FROM node_metadata_kv "
            "WHERE key = '_sys_drift_threshold'").fetchall()
        for _nid, _val in drift_rows:
            try:
                node_drift_thresholds[_nid] = float(_val)
            except (ValueError, TypeError):
                pass

        drift_candidates = {}
        if community_state:
            for nid in already_placed:
                if nid not in typed_neighbors:
                    continue
                nbrs = typed_neighbors[nid]
                if not nbrs:
                    continue

                drift_ratio = node_drift_thresholds.get(nid, default_drift_ratio)
                home = node_to_community[nid]
                home_aff = len(nbrs & home['members']) / len(nbrs) if nbrs else 0

                for comm in community_state:
                    if comm['id'] == home['id']:
                        continue
                    foreign_aff = len(nbrs & comm['members']) / len(nbrs)
                    if foreign_aff > home_aff * drift_ratio and foreign_aff > min_foreign_aff:
                        if nid not in drift_candidates:
                            drift_candidates[nid] = {
                                'home': home, 'home_aff': home_aff,
                                'drift_ratio': drift_ratio,
                                'foreign': []}
                        drift_candidates[nid]['foreign'].append(
                            (comm['id'], comm['title'], foreign_aff))

                # Also check affinity to NEW clusters
                for cid, members in valid_clusters.items():
                    foreign_aff = len(nbrs & members) / len(nbrs)
                    if foreign_aff > home_aff * drift_ratio and foreign_aff > min_foreign_aff:
                        if nid not in drift_candidates:
                            drift_candidates[nid] = {
                                'home': home, 'home_aff': home_aff,
                                'drift_ratio': drift_ratio,
                                'foreign': []}
                        drift_candidates[nid]['foreign'].append(
                            ('new_cluster_%d' % cid, 'new cluster', foreign_aff))

        # Step 5d: Community health
        community_health_updates = []
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
                self.brain.conn, comm['id'],
                'community_internal_fraction', type='float')
            old_maturity = read_community_meta(
                self.brain.conn, comm['id'],
                'community_maturity', type='str')

            if int_frac < 0.05 and len(ms) > 0:
                # Community's members have no internal edges — it's dead.
                # Encoder should archive it.
                community_health_updates.append({
                    'community': comm,
                    'old_fraction': old_frac,
                    'new_fraction': int_frac,
                    'signal': 'dead',
                })
            elif old_frac > 0 and int_frac < old_frac * 0.7:
                community_health_updates.append({
                    'community': comm,
                    'old_fraction': old_frac,
                    'new_fraction': int_frac,
                    'signal': 'degrading',
                })
            elif old_maturity == 'corridor' and int_frac > 0.3:
                community_health_updates.append({
                    'community': comm,
                    'old_fraction': old_frac,
                    'new_fraction': int_frac,
                    'signal': 'corridor_maturing',
                })

        # Step 5e: Community merge detection
        merge_candidates = self._detect_merge_candidates(community_state)

        # Step 6: Cross-cutting
        cross_cutting = self._detect_cross_cutting(
            node_affinities, degrees)

        # Step 7: Orphan embedding affinities
        orphan_affinities = self._compute_orphan_affinities(
            valid_clusters, typed_neighbors)

        # Step 8: Tie analysis
        tie_analysis = self._analyze_ties(
            node_affinities, cross_cutting, valid_clusters,
            edges_by_node)

        # Step 9: Build proposals
        proposals = self._build_proposals(
            valid_clusters, corridors, node_affinities,
            orphan_affinities, cross_cutting, tie_analysis,
            edges_by_node, typed_neighbors, community_state)

        # Step 9b: Add incremental proposals
        titles = {}
        types_map = {}
        for row in self.brain.conn.execute(
                "SELECT id, title, type FROM nodes WHERE archived = 0"):
            titles[row[0]] = row[1][:60]
            types_map[row[0]] = row[2]

        for nid, additions in existing_additions.items():
            proposals.append({
                'type': 'add_to_existing',
                'node_id': nid,
                'node_title': titles.get(nid, nid[:8]),
                'node_type': types_map.get(nid, '?'),
                'communities': [
                    {'id': cid, 'title': ctitle, 'affinity': aff}
                    for cid, ctitle, aff in additions[:3]],
            })

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

        for update in community_health_updates:
            proposals.append({
                'type': 'health_update',
                'community_id': update['community']['id'],
                'community_title': update['community']['title'],
                'signal': update['signal'],
                'old_fraction': update['old_fraction'],
                'new_fraction': update['new_fraction'],
            })

        # Merge candidates
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

        # Cluster summaries for trace (S3 consumption)
        cluster_summaries = []
        for cid, members in sorted(valid_clusters.items(),
                                    key=lambda x: -len(x[1]))[:30]:
            ms = set(members)
            internal = sum(1 for n in ms
                           for nbr, _, _ in edges_by_node.get(n, [])
                           if nbr in ms) // 2
            external = sum(1 for n in ms
                           for nbr, _, _ in edges_by_node.get(n, [])
                           if nbr not in ms)
            int_frac = internal / (internal + external) if (internal + external) else 0
            edge_sig = self._compute_edge_signature(members, edges_by_node)

            cluster_summaries.append({
                'cluster_id': cid,
                'size': len(members),
                'internal_edges': internal,
                'external_edges': external,
                'internal_fraction': round(int_frac, 3),
                'edge_signature': edge_sig,
                'is_corridor': cid in corridors,
            })

        stats = {
            'nodes_with_typed_edges': len(edges_by_node),
            'total_pairs_scored': len(pair_zscores),
            'bucket_stats': {str(k): {'mean': round(v['mean'], 2),
                                       'std': round(v['std'], 2)}
                             for k, v in bucket_stats.items()},
            'clusters_seeded': len(clusters),
            'fragments_dissolved': dissolved_count,
            'subsets_absorbed': absorbed_count,
            'valid_clusters': len(valid_clusters),
            'corridors': len(corridors),
            'nodes_with_affinities': len(node_affinities) - len(cross_cutting),
            'cross_cutting': len(cross_cutting),
            'orphans_placed': len(orphan_affinities),
            'genuine_overlaps': tie_analysis['genuine_overlaps'],
            'possible_splits': tie_analysis['possible_splits'],
        }

        return {
            'proposals': proposals,
            'stats': stats,
            'cluster_summaries': cluster_summaries,
        }

    # ── Step 1 ──

    def _build_typed_adjacency(self, rel_to_fam, skip_fams):
        # TODO(v25-dal): graph-wide typed-edge scan has no existing DAL
        # method and fits community detection's needs specifically. Keep
        # raw with archived=0; consider GraphDAL.iter_semantic_edges()
        # if a second caller appears.
        edges_by_node = defaultdict(list)
        rows = self.brain.conn.execute("""
            SELECT e.source_id, e.target_id, er.relation
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes ns ON ns.id = e.source_id AND ns.archived = 0
                AND ns.type != 'community'
            JOIN nodes nt ON nt.id = e.target_id AND nt.archived = 0
                AND nt.type != 'community'
            WHERE er.archived = 0
            AND er.relation NOT IN (
                'co_accessed', 'emergent_bridge', 'community_member')
        """).fetchall()

        for src, tgt, rel in rows:
            fam = rel_to_fam.get(rel, 'unclassified')
            if fam in skip_fams:
                continue
            edges_by_node[src].append((tgt, fam, rel))
            edges_by_node[tgt].append((src, fam, rel))

        typed_neighbors = {
            nid: set(nbr for nbr, _, _ in el)
            for nid, el in edges_by_node.items()
        }
        return edges_by_node, typed_neighbors

    # ── Step 2 ──

    def _compute_pair_scores(self, typed_neighbors, edges_by_node):
        degrees = {nid: len(nbrs) for nid, nbrs in typed_neighbors.items()}

        neighbor_to_nodes = defaultdict(set)
        for nid, nbrs in typed_neighbors.items():
            for nbr in nbrs:
                neighbor_to_nodes[nbr].add(nid)

        raw_shared = defaultdict(int)
        for shared_node, node_set in neighbor_to_nodes.items():
            nodes = list(node_set)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    key = (min(nodes[i], nodes[j]), max(nodes[i], nodes[j]))
                    raw_shared[key] += 1

        for nid, edge_list in edges_by_node.items():
            for nbr, fam, rel in edge_list:
                key = (min(nid, nbr), max(nid, nbr))
                raw_shared[key] += 2

        def _bucket(degree):
            if degree <= 1: return 1
            if degree <= 2: return 2
            if degree <= 4: return 3
            if degree <= 7: return 5
            if degree <= 12: return 8
            return 13

        bucket_scores = defaultdict(list)
        pair_to_bucket = {}
        for (a, b), raw in raw_shared.items():
            bucket = _bucket(min(degrees.get(a, 1), degrees.get(b, 1)))
            bucket_scores[bucket].append(raw)
            pair_to_bucket[(a, b)] = bucket

        bucket_stats = {}
        for bucket, scores in bucket_scores.items():
            arr = np.array(scores, dtype=float)
            bucket_stats[bucket] = {
                'mean': float(arr.mean()),
                'std': max(float(arr.std()), 0.5),
            }

        pair_zscores = {}
        for (a, b), raw in raw_shared.items():
            stats = bucket_stats[pair_to_bucket[(a, b)]]
            pair_zscores[(a, b)] = (raw - stats['mean']) / stats['std']

        return pair_zscores, degrees, bucket_stats

    # ── Step 3 ──

    def _get_direct_pairs(self, edges_by_node):
        direct = set()
        for nid, edge_list in edges_by_node.items():
            for nbr, fam, rel in edge_list:
                direct.add((min(nid, nbr), max(nid, nbr)))
        return direct

    def _seed_clusters(self, pair_zscores, direct_pairs):
        """Direct edges always seed. Z>=1.0 for neighbor-found pairs."""
        z_threshold = self.config.get('z_seed_threshold', 1.0)

        seed_pairs = []
        for (a, b), z in pair_zscores.items():
            is_direct = (a, b) in direct_pairs
            if is_direct or z >= z_threshold:
                seed_pairs.append((a, b, z, is_direct))
        seed_pairs.sort(key=lambda x: (-x[2], -x[3]))

        clusters = {}
        node_home = {}
        cluster_counter = 0
        used = set()

        for a, b, z, is_direct in seed_pairs:
            if a in used and b in used:
                continue
            a_c = node_home.get(a)
            b_c = node_home.get(b)
            if a_c and not b_c:
                clusters[a_c].add(b); node_home[b] = a_c
            elif b_c and not a_c:
                clusters[b_c].add(a); node_home[a] = b_c
            elif not a_c and not b_c:
                cluster_counter += 1
                clusters[cluster_counter] = {a, b}
                node_home[a] = cluster_counter
                node_home[b] = cluster_counter
            used.add(a); used.add(b)

        return clusters

    # ── Step 4 ──

    def _validate_clusters(self, clusters, edges_by_node, typed_neighbors):
        valid = {}
        corridors = set()
        dissolved = 0
        min_size = self.config['min_community_size']

        for cid, members in clusters.items():
            if len(members) < min_size:
                continue

            ms = set(members)
            internal = sum(1 for n in ms
                           for nbr, _, _ in edges_by_node.get(n, [])
                           if nbr in ms) // 2
            external = sum(1 for n in ms
                           for nbr, _, _ in edges_by_node.get(n, [])
                           if nbr not in ms)

            if internal < 2:
                dissolved += 1
                continue

            valid[cid] = ms

            int_frac = internal / (internal + external) if (internal + external) else 0
            if int_frac < 0.20 and len(members) > 3:
                corridors.add(cid)

        return valid, corridors, dissolved

    # ── Step 4b ──

    def _absorb_subsets(self, valid_clusters):
        """When one cluster is fully contained in another, absorb it.

        The larger cluster keeps all its members (no information loss).
        The smaller cluster is dissolved — its members are already
        covered by the larger one.

        Handles chains: if A ⊂ B ⊂ C, both A and B dissolve into C.

        Returns: (filtered_clusters, absorbed_count)
        """
        absorbed = set()
        cluster_ids = sorted(valid_clusters.keys(),
                             key=lambda k: -len(valid_clusters[k]))

        for i in range(len(cluster_ids)):
            if cluster_ids[i] in absorbed:
                continue
            larger = valid_clusters[cluster_ids[i]]
            for j in range(i + 1, len(cluster_ids)):
                if cluster_ids[j] in absorbed:
                    continue
                smaller = valid_clusters[cluster_ids[j]]
                if smaller <= larger:  # set subset check
                    absorbed.add(cluster_ids[j])

        filtered = {cid: members for cid, members in valid_clusters.items()
                    if cid not in absorbed}
        return filtered, len(absorbed)

    # ── Step 5 ──

    def _compute_affinities(self, valid_clusters, typed_neighbors):
        node_affinities = {}
        for nid, nbrs in typed_neighbors.items():
            if not nbrs:
                continue
            affinities = []
            for cid, members in valid_clusters.items():
                shared = len(nbrs & members)
                if shared > 0:
                    affinities.append((cid, shared / len(nbrs)))
            if affinities:
                affinities.sort(key=lambda x: -x[1])
                node_affinities[nid] = affinities
        return node_affinities

    # ── Step 6 ──

    # ── Step 5e ──

    def _detect_merge_candidates(self, community_state):
        """Detect community pairs with high member overlap.

        Merge when:
        - overlap >= threshold (% of smaller community's members)
        - AND smaller community has < min_unique unique members

        This adaptive condition prevents merging in young brains where
        small communities naturally overlap due to few nodes.
        """
        threshold = self.config.get('merge_overlap_threshold', 0.80)
        min_unique = self.config.get('merge_min_unique_members', 3)
        candidates = []

        # Only check communities with members
        active = [c for c in community_state if c['members']]

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                shared = a['members'] & b['members']
                if not shared:
                    continue

                # Overlap relative to the smaller community
                smaller, larger = (a, b) if len(a['members']) <= len(b['members']) else (b, a)
                overlap_pct = len(shared) / len(smaller['members'])
                unique_in_smaller = len(smaller['members'] - shared)

                if overlap_pct >= threshold and unique_in_smaller < min_unique:
                    candidates.append({
                        'larger': larger,
                        'smaller': smaller,
                        'shared_count': len(shared),
                        'overlap_pct': round(overlap_pct, 3),
                        'unique_in_smaller': unique_in_smaller,
                    })

        candidates.sort(key=lambda c: -c['overlap_pct'])
        return candidates

    # ── Step 6 ──

    def _detect_cross_cutting(self, node_affinities, degrees):
        cross_cutting = set()
        min_deg = self.config.get('cross_cutting_min_degree', 15)
        max_top = self.config.get('cross_cutting_max_top_affinity', 0.35)

        for nid, affs in node_affinities.items():
            deg = degrees.get(nid, 0)
            top = affs[0][1] if affs else 0
            if deg >= min_deg and top < max_top:
                cross_cutting.add(nid)
        return cross_cutting

    # ── Step 7 ──

    def _compute_orphan_affinities(self, valid_clusters, typed_neighbors):
        centroids = {}
        for cid, members in valid_clusters.items():
            blobs = []
            for nid in members:
                row = self.brain.conn.execute(
                    "SELECT embedding FROM node_enrichments "
                    "WHERE node_id = ? AND vector_type = '_primary' AND embedding IS NOT NULL",
                    (nid,)).fetchone()
                if row and row[0]:
                    blobs.append(row[0])
            if blobs:
                vecs = [np.frombuffer(b, dtype=np.float32) for b in blobs]
                c = np.mean(vecs, axis=0)
                n = np.linalg.norm(c)
                if n > 0:
                    c = c / n
                centroids[cid] = c.astype(np.float32).tobytes()

        if not centroids:
            return {}

        all_nodes = set(r[0] for r in self.brain.conn.execute(
            "SELECT id FROM nodes WHERE archived = 0 "
            "AND type != 'community'").fetchall())
        orphans = all_nodes - set(typed_neighbors.keys())
        threshold = self.config.get('embedding_placement_threshold', 0.50)

        orphan_affinities = {}
        for row in self.brain.conn.execute(
                "SELECT node_id, embedding FROM node_enrichments "
                "WHERE vector_type = '_primary' AND embedding IS NOT NULL"):
            nid, emb = row
            if nid not in orphans:
                continue
            affinities = []
            for cid, centroid in centroids.items():
                sim = cosine_similarity(emb, centroid)
                if sim >= threshold:
                    affinities.append((cid, sim))
            if affinities:
                affinities.sort(key=lambda x: -x[1])
                orphan_affinities[nid] = affinities

        return orphan_affinities

    # ── Step 8 ──

    def _analyze_ties(self, node_affinities, cross_cutting,
                      valid_clusters, edges_by_node):
        genuine = 0
        splits = 0

        for nid, affs in node_affinities.items():
            if nid in cross_cutting:
                continue
            if len(affs) < 2 or abs(affs[0][1] - affs[1][1]) >= 0.03:
                continue

            c1, c2 = affs[0][0], affs[1][0]
            fams1 = set()
            fams2 = set()
            for nbr, fam, rel in edges_by_node.get(nid, []):
                if c1 in valid_clusters and nbr in valid_clusters[c1]:
                    fams1.add(fam)
                if c2 in valid_clusters and nbr in valid_clusters[c2]:
                    fams2.add(fam)

            if (fams1 - fams2) or (fams2 - fams1):
                genuine += 1
            else:
                splits += 1

        return {'genuine_overlaps': genuine, 'possible_splits': splits}

    # ── Step 9 ──

    def _build_proposals(self, valid_clusters, corridors,
                         node_affinities, orphan_affinities,
                         cross_cutting, tie_analysis,
                         edges_by_node, typed_neighbors,
                         community_state=None):
        proposals = []

        titles = {}
        types_map = {}
        for row in self.brain.conn.execute(
                "SELECT id, title, type FROM nodes WHERE archived = 0"):
            titles[row[0]] = row[1][:60]
            types_map[row[0]] = row[2]

        # Build community membership lookup for overlap check
        comm_members = {}  # comm_id -> set of member ids
        if community_state:
            for comm in community_state:
                comm_members[comm['id']] = comm['members']

        overlap_threshold = self.config.get('cluster_overlap_threshold', 0.60)
        converted_to_add = 0

        # ── Community proposals ──
        for cid, members in sorted(valid_clusters.items(),
                                    key=lambda x: -len(x[1])):
            ms = set(members)
            edge_sig = self._compute_edge_signature(members, edges_by_node)

            internal = sum(1 for n in ms
                           for nbr, _, _ in edges_by_node.get(n, [])
                           if nbr in ms) // 2
            external = sum(1 for n in ms
                           for nbr, _, _ in edges_by_node.get(n, [])
                           if nbr not in ms)
            int_frac = internal / (internal + external) if (internal + external) else 0

            # Timeline
            member_dates = []
            for nid in members:
                row = self.brain.conn.execute(
                    "SELECT created_at FROM nodes WHERE id = ?",
                    (nid,)).fetchone()
                if row and row[0]:
                    member_dates.append((nid, row[0]))
            member_dates.sort(key=lambda x: x[1])

            origin = member_dates[0] if member_dates else None
            latest = member_dates[-1] if member_dates else None

            transitions = []
            seen_trans = set()
            for nid in ms:
                for nbr, fam, rel in edges_by_node.get(nid, []):
                    if nbr in ms and fam == 'replacement_correction':
                        pair = tuple(sorted([nid, nbr]))
                        if pair in seen_trans:
                            continue
                        seen_trans.add(pair)
                        nid_d = next((d for n, d in member_dates if n == nid), '')
                        nbr_d = next((d for n, d in member_dates if n == nbr), '')
                        corrector = nid if nid_d > nbr_d else nbr
                        corrected = nbr if corrector == nid else nid
                        transitions.append({
                            'corrector_id': corrector,
                            'corrected_id': corrected,
                            'corrector_title': titles.get(corrector, '?'),
                            'corrected_title': titles.get(corrected, '?'),
                            'date': max(nid_d, nbr_d)[:10],
                            'relation': rel,
                        })
            transitions.sort(key=lambda t: t['date'])

            timeline = {
                'origin': {'id': origin[0], 'date': origin[1][:10],
                           'title': titles.get(origin[0], '?'),
                           'type': types_map.get(origin[0], '?')}
                if origin else None,
                'latest': {'id': latest[0], 'date': latest[1][:10],
                           'title': titles.get(latest[0], '?'),
                           'type': types_map.get(latest[0], '?')}
                if latest else None,
                'transitions': transitions[:5],
                'date_range': '%s to %s' % (
                    member_dates[0][1][:10] if member_dates else '?',
                    member_dates[-1][1][:10] if member_dates else '?'),
            }

            # Structural hubs
            rep_scores = {}
            for nid in members:
                rep_scores[nid] = sum(
                    1 for nbr in typed_neighbors.get(nid, set())
                    if nbr in ms)
            reps = sorted(rep_scores.items(), key=lambda x: -x[1])[:3]
            representatives = [
                {'id': nid, 'title': titles.get(nid, nid[:8]),
                 'type': types_map.get(nid, '?'),
                 'internal_edges': score}
                for nid, score in reps]

            sample_edges = self._sample_internal_edges(members, limit=5)

            # All members with titles (chronological)
            all_member_info = [
                {'id': nid, 'title': titles.get(nid, nid[:8]),
                 'type': types_map.get(nid, '?'),
                 'date': next((d[:10] for n, d in member_dates if n == nid), '?')}
                for nid in [n for n, _ in member_dates]]

            latest_id = latest[0] if latest else None
            hub_ids = {r['id'] for r in representatives}
            render_latest = latest_id and latest_id not in hub_ids

            # ── Overlap check: would this cluster duplicate an existing community? ──
            # For each existing community, count how many cluster members have
            # neighbors in that community. If >= threshold, convert to add_to_existing.
            best_overlap_comm = None
            best_overlap_frac = 0.0
            if comm_members:
                for comm_id, comm_mids in comm_members.items():
                    connecting = sum(
                        1 for nid in ms
                        if typed_neighbors.get(nid, set()) & comm_mids)
                    frac = connecting / len(ms) if ms else 0
                    if frac > best_overlap_frac:
                        best_overlap_frac = frac
                        best_overlap_comm = comm_id

            if best_overlap_frac >= overlap_threshold and best_overlap_comm:
                # Convert: route unplaced members to the overlapping community
                comm_title = next(
                    (c['title'] for c in community_state
                     if c['id'] == best_overlap_comm), '?')
                existing_members = comm_members.get(best_overlap_comm, set())
                for nid in ms:
                    if nid not in existing_members:
                        proposals.append({
                            'type': 'add_to_existing',
                            'node_id': nid,
                            'node_title': titles.get(nid, nid[:8]),
                            'node_type': types_map.get(nid, '?'),
                            'source': 'overlap_check',
                            'overlap_frac': round(best_overlap_frac, 2),
                            'communities': [
                                {'id': best_overlap_comm,
                                 'title': comm_title,
                                 'affinity': best_overlap_frac}],
                        })
                converted_to_add += 1
                continue

            proposals.append({
                'type': 'new_community',
                'cluster_id': cid,
                'members': list(members),
                'member_count': len(members),
                'all_members': all_member_info,
                'internal_edges': internal,
                'external_edges': external,
                'internal_fraction': round(int_frac, 3),
                'is_corridor': cid in corridors,
                'timeline': timeline,
                'edge_signature': edge_sig,
                'representatives': representatives,
                'render_latest': render_latest,
                'sample_edges': sample_edges,
            })

        # ── Node affinity proposals ──
        for nid, affinities in node_affinities.items():
            if nid in cross_cutting or not affinities:
                continue
            proposals.append({
                'type': 'node_affinities',
                'node_id': nid,
                'node_title': titles.get(nid, nid[:8]),
                'node_type': types_map.get(nid, '?'),
                'affinities': [{'cluster_id': c, 'affinity': a}
                                for c, a in affinities[:5]],
                'method': 'structural',
            })

        # ── Cross-cutting proposals ──
        for nid in cross_cutting:
            affs = node_affinities.get(nid, [])
            proposals.append({
                'type': 'cross_cutting',
                'node_id': nid,
                'node_title': titles.get(nid, nid[:8]),
                'node_type': types_map.get(nid, '?'),
                'cluster_count': len(affs),
                'top_affinity': affs[0][1] if affs else 0,
            })

        # ── Orphan proposals ──
        for nid, affinities in orphan_affinities.items():
            proposals.append({
                'type': 'node_affinities',
                'node_id': nid,
                'node_title': titles.get(nid, nid[:8]),
                'node_type': types_map.get(nid, '?'),
                'affinities': [{'cluster_id': c, 'affinity': sim}
                                for c, sim in affinities[:5]],
                'method': 'embedding',
            })

        if converted_to_add:
            print('[s2cd] Overlap check: converted %d clusters to add_to_existing '
                  '(threshold %.0f%%)' % (converted_to_add, overlap_threshold * 100),
                  flush=True)

        return proposals

    # ── Decode helpers ──

    def _compute_edge_signature(self, members, edges_by_node):
        counts = Counter()
        ms = set(members)
        for nid in members:
            for nbr, fam, rel in edges_by_node.get(nid, []):
                if nbr in ms:
                    counts[fam] += 1
        total = sum(counts.values())
        if total == 0:
            return {}
        return {fam: round(cnt / total, 2)
                for fam, cnt in counts.most_common(10)}

    def _sample_internal_edges(self, members, limit=5):
        ms = set(members)
        # TODO(v25-dal): internal-edge rendering with descriptions. Shape
        # is close to has_edge_between but returns metadata, not a bool.
        # Keep raw with archived=0 until a second caller surfaces.
        placeholders = ','.join('?' * len(ms))
        id_list = list(ms)
        rows = self.brain.conn.execute("""
            SELECT ns.title, nt.title, er.relation, er.description
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes ns ON ns.id = e.source_id
            JOIN nodes nt ON nt.id = e.target_id
            WHERE e.source_id IN (%s) AND e.target_id IN (%s)
            AND er.archived = 0
            AND er.relation NOT IN (
                'co_accessed', 'emergent_bridge', 'community_member',
                'related_to', 'related')
            AND er.description IS NOT NULL AND er.description != ''
            ORDER BY e.weight DESC LIMIT ?
        """ % (placeholders, placeholders),
            id_list * 2 + [limit]).fetchall()
        return [{'source': r[0][:50], 'target': r[1][:50],
                 'relation': r[2], 'description': r[3][:100]}
                for r in rows]

    # ── Recall signals (stub) ──

    def _detect_recall_signals(self, s1_delta, community_state):
        return []
