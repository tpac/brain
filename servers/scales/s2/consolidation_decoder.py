"""S2 Consolidation Decoder — finds and characterizes convergent node clusters.

Phase 1: detection + behavioral enrichment only. No LLM, no writes.
Produces proposals with enough evidence for human review (now)
and future encoder (phase 2).

O = graph embeddings + S1 behavioral traces
K = similarity thresholds + edge families + community membership
Δ = consolidation proposals (written to traces, not to graph)
"""

import json
from collections import defaultdict, Counter

import numpy as np

from .base import IntegrationUnit
from .consolidation_contract import CONSOLIDATION
from servers.embedder import cosine_similarity


class ConsolidationDecoder(IntegrationUnit):
    NAME = 'consolidation'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:consolidation'

    O_SOURCES = ['graph_embeddings', 's1_recall_traces', 's1_encode_traces']
    K_SOURCES = ['similarity_threshold', 'edge_families', 'community_membership']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or CONSOLIDATION

    def run(self):
        """Find convergent clusters, enrich with behavioral evidence.

        Returns:
            dict with: clusters (list), stats (dict), skipped (str or None)
        """
        if not self._has_new_traces('s1', ref_type='encoding_run'):
            return {'clusters': [], 'stats': {}, 'skipped': 'no new S1 traces'}

        # Step 0: Graph health — archive broken artifacts
        healed = self._heal_graph()

        last_ts = self._last_run_timestamp()
        is_cold_start = not last_ts

        # Step 1: Find candidate pairs by embedding similarity
        candidates, scan_stats = self._scan_embeddings(last_ts, is_cold_start)

        self.trace('O', 'consolidation_candidates',
                   'Scanned %d nodes, found %d pairs above %.2f%s' % (
                       scan_stats['nodes_scanned'],
                       scan_stats['pairs_found'],
                       self.config['similarity_threshold'],
                       ' (cold start)' if is_cold_start else ''),
                   metadata=scan_stats)

        if not candidates:
            return {'clusters': [], 'stats': scan_stats}

        # Step 2: Cluster connected components
        clusters = self._cluster_pairs(candidates)

        # Step 3: Enrich with behavioral evidence
        enriched = self._enrich_clusters(clusters)

        # Step 4: Pre-classify
        for cluster in enriched:
            cluster['pre_class'] = self._pre_classify(cluster)

        # Write K trace with full proposal data
        class_counts = Counter(c['pre_class'] for c in enriched)
        self.trace('K', 'consolidation_proposals',
                   '%d clusters: %s' % (
                       len(enriched),
                       ', '.join('%d %s' % (v, k) for k, v in class_counts.most_common())),
                   metadata={'clusters': self._serialize_clusters(enriched)})

        return {
            'clusters': enriched,
            'stats': {**scan_stats, 'clusters_formed': len(enriched),
                      'class_counts': dict(class_counts),
                      'healed': healed},
        }

    # ══════════════════════════════════════════════════════════
    # Step 0: Graph health
    # ══════════════════════════════════════════════════════════

    def _heal_graph(self):
        """Archive broken graph artifacts. Runs before every scan.

        Currently detects:
        - Community nodes with 0 members (failed edge creation)
        """
        archived = []

        # Find community nodes with no community_member edges
        orphan_communities = self.brain.conn.execute("""
            SELECT n.id, n.title FROM nodes n
            WHERE n.type = 'community' AND n.archived = 0
            AND NOT EXISTS (
                SELECT 1 FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE (e.source_id = n.id OR e.target_id = n.id)
                AND er.relation = 'community_member'
            )
        """).fetchall()

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for nid, title in orphan_communities:
            self.brain.conn.execute(
                'UPDATE nodes SET archived = 1, updated_at = ? WHERE id = ?',
                (now, nid))
            archived.append({'id': nid, 'title': title, 'reason': 'community with 0 members'})
            print('[consolidation] Healed: archived orphan community "%s" (%s)' % (
                title[:50], nid[:8]), flush=True)

        if archived:
            self.brain.conn.commit()
            self.trace('delta', 'consolidated',
                       'Healed %d broken artifacts: %s' % (
                           len(archived),
                           ', '.join(a['title'][:30] for a in archived[:5])),
                       metadata={'healed': archived})

        return archived

    # ══════════════════════════════════════════════════════════
    # Step 1: Embedding scan
    # ══════════════════════════════════════════════════════════

    def _scan_embeddings(self, last_ts, is_cold_start):
        """Find all node pairs above similarity threshold.

        Uses two similarity dimensions:
        - content_cosine: blend embedding (title + content) from node_embeddings
        - title_cosine: title-only embedding from node_enrichments

        A pair is a candidate if EITHER dimension exceeds threshold.
        Both scores are reported per pair for the encoder.

        Cold start: full pairwise scan.
        Incremental: new + revised nodes vs all existing.
        """
        threshold = self.config['similarity_threshold']
        suppression = self.config['suppression_relations']

        # Load content (blend) embeddings from node_embeddings
        content_rows = self.brain.conn.execute("""
            SELECT ne.node_id, ne.embedding
            FROM node_embeddings ne
            JOIN nodes n ON n.id = ne.node_id
            WHERE n.archived = 0 AND n.type != 'community'
            AND ne.embedding IS NOT NULL AND typeof(ne.embedding) = 'blob'
        """).fetchall()

        ids = []
        content_vecs = []
        for nid, emb in content_rows:
            try:
                v = np.frombuffer(emb, dtype=np.float32)
                if len(v) > 0:
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        ids.append(nid)
                        content_vecs.append(v / norm)
            except Exception:
                pass

        if len(content_vecs) < 2:
            return [], {'nodes_scanned': len(content_vecs), 'pairs_found': 0, 'mode': 'empty'}

        id_to_idx = {nid: i for i, nid in enumerate(ids)}

        # Load title embeddings from node_enrichments
        title_vecs_by_id = {}
        for row in self.brain.conn.execute("""
            SELECT node_id, embedding FROM node_enrichments
            WHERE vector_type = 'title'
            AND embedding IS NOT NULL AND typeof(embedding) = 'blob'
        """).fetchall():
            if row[0] in id_to_idx:
                try:
                    v = np.frombuffer(row[1], dtype=np.float32)
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        title_vecs_by_id[row[0]] = v / norm
                except Exception:
                    pass

        # Build title matrix aligned with ids (zeros for missing)
        dim = len(content_vecs[0])
        title_vecs = []
        has_title = set()
        for nid in ids:
            if nid in title_vecs_by_id:
                title_vecs.append(title_vecs_by_id[nid])
                has_title.add(nid)
            else:
                title_vecs.append(np.zeros(dim, dtype=np.float32))

        # Load suppression edges
        suppressed = set()
        if suppression:
            placeholders = ','.join('?' * len(suppression))
            for row in self.brain.conn.execute("""
                SELECT e.source_id, e.target_id
                FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE er.relation IN (%s)
            """ % placeholders, list(suppression)).fetchall():
                suppressed.add((min(row[0], row[1]), max(row[0], row[1])))

        content_mat = np.stack(content_vecs)
        title_mat = np.stack(title_vecs)

        def _find_pairs(content_sim, title_sim, row_ids, col_ids):
            """Find pairs where either content or title similarity exceeds threshold."""
            pairs = []
            # Content matches
            yi, xi = np.where(content_sim > threshold)
            pair_set = set()
            for y, x in zip(yi, xi):
                a, b = row_ids[y], col_ids[x]
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key not in suppressed and key not in pair_set:
                    pair_set.add(key)
                    # Get both scores
                    t_sim = float(title_sim[y, x]) if (a in has_title and b in has_title) else 0.0
                    pairs.append((a, b, float(content_sim[y, x]), t_sim))

            # Title matches (may find pairs content missed)
            if title_sim is not None:
                yi, xi = np.where(title_sim > threshold)
                for y, x in zip(yi, xi):
                    a, b = row_ids[y], col_ids[x]
                    if a == b:
                        continue
                    key = (min(a, b), max(a, b))
                    if key not in suppressed and key not in pair_set:
                        if a in has_title and b in has_title:
                            pair_set.add(key)
                            c_sim = float(content_sim[y, x])
                            pairs.append((a, b, c_sim, float(title_sim[y, x])))
            return pairs

        if is_cold_start:
            content_sim = content_mat @ content_mat.T
            np.fill_diagonal(content_sim, 0)
            title_sim = title_mat @ title_mat.T
            np.fill_diagonal(title_sim, 0)

            # Only keep upper triangle
            content_sim_upper = np.triu(content_sim, k=1)
            title_sim_upper = np.triu(title_sim, k=1)

            pairs_raw = _find_pairs(content_sim_upper, title_sim_upper, ids, ids)
            mode = 'cold_start'
        else:
            changed_ids = self._get_changed_node_ids(last_ts)
            if not changed_ids:
                return [], {'nodes_scanned': len(content_vecs), 'pairs_found': 0,
                            'mode': 'incremental', 'changed_nodes': 0}

            changed_indices = [id_to_idx[nid] for nid in changed_ids if nid in id_to_idx]
            if not changed_indices:
                return [], {'nodes_scanned': len(content_vecs), 'pairs_found': 0,
                            'mode': 'incremental', 'changed_nodes': len(changed_ids)}

            changed_content = content_mat[changed_indices]
            changed_title = title_mat[changed_indices]
            changed_id_list = [ids[i] for i in changed_indices]

            content_sim = changed_content @ content_mat.T
            title_sim = changed_title @ title_mat.T

            # Zero out self-matches
            for ci, full_idx in enumerate(changed_indices):
                content_sim[ci, full_idx] = 0
                title_sim[ci, full_idx] = 0

            pairs_raw = _find_pairs(content_sim, title_sim, changed_id_list, ids)
            mode = 'incremental'

        # Deduplicate
        seen = set()
        unique_pairs = []
        for a, b, c_score, t_score in pairs_raw:
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                unique_pairs.append((a, b, c_score, t_score))

        stats = {
            'nodes_scanned': len(content_vecs),
            'title_embeddings': len(has_title),
            'pairs_found': len(unique_pairs),
            'suppressed_pairs': len(suppressed),
            'mode': mode,
        }
        if mode == 'incremental':
            stats['changed_nodes'] = len(changed_ids) if not is_cold_start else 0

        return unique_pairs, stats

    def _get_changed_node_ids(self, since_ts):
        """Get node IDs created or revised since last run, from S1E traces."""
        traces = self._read_traces_since(
            's1', since_ts, ref_types=['encoding_run'])
        changed = set()
        for t in traces:
            meta = t.get('metadata')
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            if isinstance(meta, dict):
                for nid in meta.get('created', []):
                    if nid:
                        changed.add(nid)
                for nid in meta.get('revised', []):
                    if nid:
                        changed.add(nid)
        return changed

    # ══════════════════════════════════════════════════════════
    # Step 2: Cluster connected components
    # ══════════════════════════════════════════════════════════

    def _cluster_pairs(self, pairs):
        """Group pairs into connected components, capped at max_cluster_size.

        Each pair carries (a, b, content_cosine, title_cosine).
        """
        max_size = self.config['max_cluster_size']

        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        pair_scores = {}  # (a, b) → {content: float, title: float}
        for a, b, c_score, t_score in pairs:
            union(a, b)
            pair_scores[(min(a, b), max(a, b))] = {
                'content': c_score, 'title': t_score}

        groups = defaultdict(set)
        all_nodes = set()
        for a, b, _, _ in pairs:
            all_nodes.add(a)
            all_nodes.add(b)
        for nid in all_nodes:
            groups[find(nid)].add(nid)

        clusters = []
        for members in groups.values():
            if len(members) > max_size:
                continue
            member_list = sorted(members)
            content_scores = []
            title_scores = []
            for i in range(len(member_list)):
                for j in range(i + 1, len(member_list)):
                    key = (min(member_list[i], member_list[j]),
                           max(member_list[i], member_list[j]))
                    if key in pair_scores:
                        content_scores.append(pair_scores[key]['content'])
                        title_scores.append(pair_scores[key]['title'])

            clusters.append({
                'nodes': member_list,
                'size': len(member_list),
                'content_cosine_max': max(content_scores) if content_scores else 0,
                'content_cosine_avg': sum(content_scores) / len(content_scores) if content_scores else 0,
                'title_cosine_max': max(title_scores) if title_scores else 0,
                'title_cosine_avg': sum(title_scores) / len(title_scores) if title_scores else 0,
                'pair_scores': {
                    '%s-%s' % k: v for k, v in pair_scores.items()
                    if k[0] in members and k[1] in members
                },
            })

        # Sort by highest similarity (max of either dimension)
        clusters.sort(key=lambda c: -max(c['content_cosine_max'], c['title_cosine_max']))
        return clusters

    # ══════════════════════════════════════════════════════════
    # Step 3: Enrich with behavioral evidence
    # ══════════════════════════════════════════════════════════

    def _enrich_clusters(self, clusters):
        """Add S1 behavioral traces, edge data, and community membership."""
        if not clusters:
            return []

        # Collect all node IDs across clusters
        all_ids = set()
        for c in clusters:
            all_ids.update(c['nodes'])

        # Batch-load node data
        node_data = self._load_node_data(all_ids)

        # Load S1 behavioral data
        recall_data = self._load_recall_data(all_ids)
        catalog_data = self._load_catalog_data(all_ids)

        # Load community membership
        community_map = self._load_community_membership(all_ids)

        # Load edge data per node
        edge_data = self._load_edge_data(all_ids)

        for cluster in clusters:
            nids = cluster['nodes']

            # Node details
            cluster['node_details'] = {
                nid: node_data.get(nid, {}) for nid in nids
            }

            # Recall behavioral signals
            cluster['co_recall_count'] = recall_data['co_recall'].get(
                tuple(sorted(nids)), 0) if len(nids) == 2 else 0
            cluster['judge_preference'] = {
                nid: recall_data['selections'].get(nid, 0) for nid in nids
            }
            cluster['recall_counts'] = {
                nid: recall_data['candidates'].get(nid, 0) for nid in nids
            }
            cluster['query_coverage'] = {
                nid: recall_data['queries'].get(nid, []) for nid in nids
            }

            # Catalog blindness
            cluster['catalog_blind'] = {}
            for nid in nids:
                # Was this node created without seeing the others?
                blind = catalog_data.get(nid, {}).get('blind_to', set())
                cluster['catalog_blind'][nid] = bool(blind & set(nids) - {nid})

            # Edge comparison
            shared_neighbors = set()
            all_neighbors = {}
            for nid in nids:
                neighbors = set(edge_data.get(nid, {}).keys())
                all_neighbors[nid] = neighbors
                if not shared_neighbors:
                    shared_neighbors = neighbors
                else:
                    shared_neighbors &= neighbors
            # Remove cluster members from neighbor sets
            for nid in nids:
                all_neighbors[nid] -= set(nids)
                shared_neighbors -= set(nids)

            cluster['shared_edge_count'] = len(shared_neighbors)
            cluster['unique_edges'] = {
                nid: len(all_neighbors[nid] - shared_neighbors) for nid in nids
            }
            cluster['edge_details'] = {
                nid: edge_data.get(nid, {}) for nid in nids
            }

            # Community membership
            cluster['communities'] = {
                nid: community_map.get(nid, []) for nid in nids
            }
            # Check if any pair shares a community
            comm_sets = {nid: set(c['id'] for c in comms)
                         for nid, comms in cluster['communities'].items()}
            shared_comms = set()
            for nid in nids:
                if not shared_comms:
                    shared_comms = comm_sets.get(nid, set())
                else:
                    shared_comms &= comm_sets.get(nid, set())
            cluster['same_community'] = bool(shared_comms)
            cluster['shared_community_ids'] = list(shared_comms)

            # Correction relationship
            cluster['has_correction_edge'] = self._has_correction_edge(nids)

        return clusters

    def _load_node_data(self, node_ids):
        """Batch-load title, type, content, confidence, encoding_source, timestamps."""
        data = {}
        placeholders = ','.join('?' * len(node_ids))
        for row in self.brain.conn.execute("""
            SELECT id, title, type, content, confidence, encoding_source,
                   keywords, locked, critical, created_at, updated_at
            FROM nodes WHERE id IN (%s)
        """ % placeholders, list(node_ids)).fetchall():
            data[row[0]] = {
                'title': row[1], 'type': row[2], 'content': row[3],
                'confidence': row[4], 'encoding_source': row[5] or '',
                'keywords': row[6] or '', 'locked': bool(row[7]),
                'critical': bool(row[8]),
                'created_at': row[9] or '', 'updated_at': row[10] or '',
            }

        # Load raw quotes from metadata
        for row in self.brain.conn.execute("""
            SELECT node_id, key, value FROM node_metadata_kv
            WHERE node_id IN (%s)
            AND key IN ('user_raw_quote', 'anchor_raw_quote', 'situation',
                        'reasoning', 'correction_of')
        """ % placeholders, list(node_ids)).fetchall():
            if row[0] in data:
                data[row[0]][row[1]] = row[2]

        return data

    def _load_recall_data(self, node_ids):
        """Load S1R trace data: co-recall, judge preference, queries."""
        lookback = self.config['co_recall_lookback_hours']
        candidates_count = defaultdict(int)  # node_id → times appeared as candidate
        selections_count = defaultdict(int)  # node_id → times selected by judge
        queries_by_node = defaultdict(list)  # node_id → [query strings]
        co_recall = defaultdict(int)         # (sorted node pair) → co-occurrence count

        # Read recall O traces (candidates)
        recall_traces = self._read_traces_since(
            's1', '', hours=lookback, ref_types=['recall'])
        for t in recall_traces:
            meta = t.get('metadata')
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(meta, dict):
                continue

            query = meta.get('query', '')
            cands = meta.get('candidates', [])
            # Candidates are "id|title|score|type" strings
            cand_ids = set()
            for c in cands:
                if isinstance(c, str) and '|' in c:
                    cid = c.split('|')[0]
                    if cid in node_ids:
                        cand_ids.add(cid)
                        candidates_count[cid] += 1
                        if query:
                            queries_by_node[cid].append(query[:100])

            # Co-recall: pairs of our target nodes that appeared together
            target_cands = sorted(cand_ids & node_ids)
            for i in range(len(target_cands)):
                for j in range(i + 1, len(target_cands)):
                    co_recall[tuple(sorted([target_cands[i], target_cands[j]]))] += 1

        # Read surface_selected K traces (judge selections)
        selected_traces = self._read_traces_since(
            's1', '', hours=lookback, ref_types=['surface_selected'])
        for t in selected_traces:
            ref_id = t.get('ref_id', '')
            if not ref_id:
                continue
            try:
                selected_ids = json.loads(ref_id)
                if isinstance(selected_ids, list):
                    for sid in selected_ids:
                        if sid in node_ids:
                            selections_count[sid] += 1
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            'candidates': dict(candidates_count),
            'selections': dict(selections_count),
            'queries': {nid: list(set(qs))[:5] for nid, qs in queries_by_node.items()},
            'co_recall': dict(co_recall),
        }

    def _load_catalog_data(self, node_ids):
        """Check S1E catalog traces for blindness — was node B created without A in catalog?"""
        lookback = self.config['encoding_lookback_hours']
        catalog_data = {}

        catalog_traces = self._read_traces_since(
            's1', '', hours=lookback, ref_types=['node_catalog'])
        encoding_traces = self._read_traces_since(
            's1', '', hours=lookback, ref_types=['encoding_run'])

        # Catalog ref_id contains 8-char prefix IDs (comma-separated).
        # Build prefix→full_id lookup for our target nodes.
        prefix_to_full = {}
        for nid in node_ids:
            prefix_to_full[nid[:8]] = nid

        # Build: for each encoding run chain, what full node IDs were in the catalog?
        catalogs_by_chain = {}
        for t in catalog_traces:
            chain = t.get('chain_id', '')
            ref_id = t.get('ref_id', '')
            if ref_id:
                catalog_prefixes = set(ref_id.split(','))
                # Resolve prefixes to full IDs where they match our targets
                catalog_full = set()
                for prefix in catalog_prefixes:
                    prefix = prefix.strip()
                    if prefix in prefix_to_full:
                        catalog_full.add(prefix_to_full[prefix])
                catalogs_by_chain[chain] = catalog_full

        for t in encoding_traces:
            chain = t.get('chain_id', '')
            meta = t.get('metadata')
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(meta, dict):
                continue

            created = meta.get('created', [])
            catalog = catalogs_by_chain.get(chain, set())

            for nid in created:
                if nid in node_ids:
                    # Which of our target nodes were NOT in the catalog when this was created?
                    blind_to = node_ids - catalog - {nid}
                    if blind_to:
                        catalog_data[nid] = {'blind_to': blind_to, 'catalog_size': len(catalog)}

        return catalog_data

    def _load_community_membership(self, node_ids):
        """Load community membership for target nodes."""
        membership = defaultdict(list)
        placeholders = ','.join('?' * len(node_ids))

        rows = self.brain.conn.execute("""
            SELECT
                CASE WHEN e.source_id IN (%s) THEN e.source_id ELSE e.target_id END as member,
                CASE WHEN e.source_id IN (%s) THEN e.target_id ELSE e.source_id END as community,
                n.title
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE WHEN e.source_id IN (%s) THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id IN (%s) OR e.target_id IN (%s))
            AND er.relation = 'community_member'
            AND n.type = 'community' AND n.archived = 0
        """ % (placeholders, placeholders, placeholders, placeholders, placeholders),
            list(node_ids) * 5).fetchall()

        for member_id, comm_id, comm_title in rows:
            membership[member_id].append({'id': comm_id, 'title': comm_title})

        return dict(membership)

    def _load_edge_data(self, node_ids):
        """Load typed edges per node (excluding noise relations)."""
        edges = defaultdict(dict)
        placeholders = ','.join('?' * len(node_ids))

        rows = self.brain.conn.execute("""
            SELECT e.source_id, e.target_id, er.relation, er.description,
                   n.title, n.type
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE WHEN e.source_id IN (%s) THEN e.target_id
                                        ELSE e.source_id END
            WHERE (e.source_id IN (%s) OR e.target_id IN (%s))
            AND er.relation NOT IN ('co_accessed', 'emergent_bridge', 'community_member')
            AND n.archived = 0
        """ % (placeholders, placeholders, placeholders),
            list(node_ids) * 3).fetchall()

        for src, tgt, rel, desc, nbr_title, nbr_type in rows:
            member = src if src in node_ids else tgt
            neighbor = tgt if member == src else src
            if neighbor not in edges[member]:
                edges[member][neighbor] = []
            edges[member][neighbor].append({
                'relation': rel,
                'description': (desc or '')[:80],
                'title': nbr_title[:60],
                'type': nbr_type,
            })

        return dict(edges)

    def _has_correction_edge(self, node_ids):
        """Check if any correction edge exists between cluster members."""
        correction_rels = self.brain.get_relations_for_families(
            'correction_improvement', 'hierarchical_structure')
        if not correction_rels:
            correction_rels = {'corrects', 'corrected_by', 'supersedes', 'superseded_by'}

        node_ph = ','.join('?' * len(node_ids))
        rel_ph = ','.join('?' * len(correction_rels))

        rows = self.brain.conn.execute("""
            SELECT 1 FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE e.source_id IN (%s) AND e.target_id IN (%s)
            AND er.relation IN (%s) LIMIT 1
        """ % (node_ph, node_ph, rel_ph),
            list(node_ids) * 2 + list(correction_rels)).fetchall()

        return len(rows) > 0

    # ══════════════════════════════════════════════════════════
    # Step 4: Pre-classify
    # ══════════════════════════════════════════════════════════

    def _pre_classify(self, cluster):
        """Algorithmic pre-classification to guide future encoder.

        Uses both title and content similarity dimensions:
        - High title sim = likely same thing encoded twice (catalog blindness)
        - High content sim = same knowledge, possibly different framing
        - Both high = strongest consolidation signal
        - Type mismatch = likely distinct perspectives (KEEP)
        """
        likely_cosine = self.config.get('likely_consolidate_cosine', 0.90)
        content_max = cluster['content_cosine_max']
        title_max = cluster['title_cosine_max']

        # Correction edge → evolution, not consolidation
        if cluster['has_correction_edge']:
            return 'likely_evolve'

        # Type mismatch between cluster members → likely distinct
        types = set()
        for nid in cluster['nodes']:
            t = cluster.get('node_details', {}).get(nid, {}).get('type', '')
            if t:
                types.add(t)
        if len(types) > 1:
            # Different types but high similarity → distinct perspectives
            type_mismatch = True
        else:
            type_mismatch = False

        # Both dimensions agree (high title + high content) → strong signal
        if title_max >= 0.95 and content_max >= 0.50:
            return 'likely_keep' if type_mismatch else 'likely_consolidate'

        # High title but content diverges significantly → same name, different knowledge
        # Could be evolution, naming collision, or shallow vs rich
        if title_max >= 0.95 and content_max < 0.50:
            return 'likely_evolve'

        # Very high content similarity (regardless of title)
        if content_max >= 0.95:
            return 'likely_keep' if type_mismatch else 'likely_consolidate'

        # High similarity + structural signal
        if content_max >= likely_cosine or title_max >= likely_cosine:
            if (cluster['shared_edge_count'] > 0
                    or cluster['same_community']
                    or any(cluster['catalog_blind'].values())):
                return 'likely_keep' if type_mismatch else 'likely_consolidate'

        return 'needs_judgment'

    # ══════════════════════════════════════════════════════════
    # Serialization (for traces)
    # ══════════════════════════════════════════════════════════

    def _serialize_clusters(self, clusters):
        """Serialize clusters for trace metadata (JSON-safe, within 4000 chars)."""
        serialized = []
        for c in clusters:
            entry = {
                'nodes': c['nodes'],
                'size': c['size'],
                'content_cosine': round(c['content_cosine_max'], 3),
                'title_cosine': round(c['title_cosine_max'], 3),
                'pre_class': c['pre_class'],
                'co_recall_count': c.get('co_recall_count', 0),
                'judge_preference': c.get('judge_preference', {}),
                'catalog_blind': c.get('catalog_blind', {}),
                'shared_edge_count': c.get('shared_edge_count', 0),
                'unique_edges': c.get('unique_edges', {}),
                'same_community': c.get('same_community', False),
                'has_correction_edge': c.get('has_correction_edge', False),
                'node_titles': {
                    nid: c.get('node_details', {}).get(nid, {}).get('title', '')[:60]
                    for nid in c['nodes']
                },
            }
            serialized.append(entry)
        return serialized
