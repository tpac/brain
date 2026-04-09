"""Community Detection — first S2 integration unit.

O: Graph structure (non-archived nodes + edges)
K: Leiden algorithm with resolution parameter
Δ: Community labels in node_communities table + community nodes in graph

Community nodes are regular nodes — same fields, same enrichment,
same everything. They participate in recall, get embeddings, get
connected. First-class graph citizens.
"""

import json
from collections import Counter

from .base import IntegrationUnit
from .community_contract import COMMUNITY_DETECTION

try:
    import igraph
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


class CommunityDetection(IntegrationUnit):
    NAME = 'community_detection'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:community_detection'

    O_SOURCES = ['graph_nodes', 'graph_edges']
    K_SOURCES = ['leidenalg', 'resolution_param', 'edge_weights']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or COMMUNITY_DETECTION

    def run(self):
        """Detect communities and write results.

        Returns:
            {actions: int, communities: int, details: [...]}
        """
        if not HAS_LEIDEN:
            return {'actions': 0, 'error': 'leidenalg/igraph not installed'}

        # Load graph
        graph, node_ids, id_to_idx = self._load_graph()
        if graph is None:
            return {'actions': 0, 'skipped': 'graph too small'}

        # Trace O: what we observed
        self.trace('O', 'graph_structure',
                   '%d nodes, %d edges' % (graph.vcount(), graph.ecount()),
                   metadata={'node_count': graph.vcount(),
                             'edge_count': graph.ecount()})

        # Detect communities
        partition = self._detect_communities(graph)
        community_members = self._partition_to_members(partition, node_ids)

        # Filter small communities
        min_size = self.config['min_community_size']
        community_members = {
            cid: members for cid, members in community_members.items()
            if len(members) >= min_size
        }

        if not community_members:
            self.trace('K', 'community_partition',
                       'No communities above min_size=%d' % min_size)
            return {'actions': 0, 'communities': 0, 'details': ['no communities above threshold']}

        # Trace K: what the algorithm produced
        self.trace('K', 'community_partition',
                   '%d communities (min_size=%d)' % (len(community_members), min_size),
                   metadata={'community_sizes': {
                       str(cid): len(m) for cid, m in community_members.items()
                   }})

        # Diff against existing
        diff = self._diff_communities(community_members)

        # Check stability
        if diff['total_changed'] == 0:
            self.trace('K', 'community_diff', 'No changes from previous run')
            return {'actions': 0, 'communities': len(community_members),
                    'details': ['stable — no changes']}

        total_nodes = sum(len(m) for m in community_members.values())
        change_pct = (diff['total_changed'] / total_nodes * 100) if total_nodes else 0
        threshold = self.config['stability_threshold_pct']

        if change_pct < threshold and not diff['new'] and not diff['removed']:
            self.trace('K', 'community_diff',
                       'Below stability threshold: %.1f%% changed (threshold=%d%%)' % (
                           change_pct, threshold))
            return {'actions': 0, 'communities': len(community_members),
                    'details': ['stable — %.1f%% changed, below %d%% threshold' % (
                        change_pct, threshold)]}

        # Write results
        actions = self._write_results(community_members, diff)

        # Trace delta
        self.trace('delta', 'community_assignments',
                   '%d communities written (%d new, %d updated, %d removed)' % (
                       len(community_members),
                       len(diff['new']), len(diff['updated']), len(diff['removed'])),
                   metadata={'new': list(diff['new']),
                             'updated': list(diff['updated']),
                             'removed': list(diff['removed'])})

        return {
            'actions': actions,
            'communities': len(community_members),
            'details': diff,
        }

    def _load_graph(self):
        """Load non-archived nodes + weighted edges into igraph.

        Returns:
            (igraph.Graph, node_ids list, id_to_idx dict) or (None, None, None) if too small
        """
        min_nodes = self.config['min_graph_nodes']
        weight_threshold = self.config['edge_weight_threshold']

        # Get all non-archived, non-community node IDs
        rows = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE archived = 0 AND type != 'community'"
        ).fetchall()
        node_ids = [r[0] for r in rows]

        if len(node_ids) < min_nodes:
            return None, None, None

        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        node_set = set(node_ids)

        # Get edges between these nodes, above weight threshold
        edges_raw = self.brain.conn.execute(
            'SELECT source_id, target_id, weight FROM edges WHERE weight >= ?',
            (weight_threshold,)
        ).fetchall()

        # Filter to edges where both endpoints are in our node set
        # Also deduplicate bidirectional edges (keep one direction)
        seen_pairs = set()
        edges = []
        weights = []
        for src, tgt, w in edges_raw:
            if src not in node_set or tgt not in node_set:
                continue
            pair = tuple(sorted([src, tgt]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append((id_to_idx[src], id_to_idx[tgt]))
            weights.append(w or 0.5)

        g = igraph.Graph(n=len(node_ids), edges=edges, directed=False)
        g.es['weight'] = weights

        return g, node_ids, id_to_idx

    def _detect_communities(self, graph):
        """Run Leiden algorithm on the graph."""
        resolution = self.config['resolution']
        return leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            seed=42,  # deterministic for stability
        )

    def _partition_to_members(self, partition, node_ids):
        """Convert leidenalg partition to {community_id: [node_ids]}."""
        communities = {}
        for idx, cid in enumerate(partition.membership):
            if cid not in communities:
                communities[cid] = []
            communities[cid].append(node_ids[idx])
        return communities

    def _diff_communities(self, new_members):
        """Compare new partition against existing node_communities table.

        Returns:
            {new: set, updated: set, removed: set, unchanged: set, total_changed: int}
        """
        # Load existing assignments
        existing = {}
        rows = self.brain.conn.execute(
            'SELECT node_id, community_id FROM node_communities'
        ).fetchall()
        existing_by_community = {}
        for node_id, cid in rows:
            existing[node_id] = cid
            if cid not in existing_by_community:
                existing_by_community[cid] = set()
            existing_by_community[cid].add(node_id)

        existing_cids = set(existing_by_community.keys())
        new_cids = set(new_members.keys())

        # Compare membership sets
        new_sets = {cid: set(members) for cid, members in new_members.items()}
        removed = existing_cids - new_cids
        brand_new = new_cids - existing_cids
        potentially_updated = new_cids & existing_cids

        updated = set()
        unchanged = set()
        for cid in potentially_updated:
            if new_sets[cid] != existing_by_community.get(cid, set()):
                updated.add(cid)
            else:
                unchanged.add(cid)

        # Count total nodes that changed community
        total_changed = 0
        for node_id, old_cid in existing.items():
            # Find new community for this node
            new_cid = None
            for cid, members in new_members.items():
                if node_id in new_sets[cid]:
                    new_cid = cid
                    break
            if new_cid != old_cid:
                total_changed += 1
        # Also count nodes in brand_new communities
        for cid in brand_new:
            total_changed += len(new_members[cid])

        return {
            'new': brand_new,
            'updated': updated,
            'removed': removed,
            'unchanged': unchanged,
            'total_changed': total_changed,
        }

    def _name_community(self, member_ids):
        """Generate community title and content from member keywords/titles.

        Returns:
            (title, content, keywords, situation, confidence)
        """
        max_kw = self.config['max_community_name_keywords']
        max_titles = self.config['max_member_titles_in_content']

        # Fetch member data
        placeholders = ','.join('?' * len(member_ids))
        rows = self.brain.conn.execute(
            'SELECT id, title, keywords, confidence, type FROM nodes WHERE id IN (%s)' % placeholders,
            member_ids
        ).fetchall()

        all_keywords = Counter()
        titles = []
        confidences = []
        types = Counter()

        for nid, title, kw, conf, ntype in rows:
            titles.append(title or '')
            confidences.append(conf or 0.7)
            if ntype:
                types[ntype] += 1
            if kw:
                for word in kw.split():
                    word = word.strip().lower()
                    if len(word) > 2:
                        all_keywords[word] += 1

        # Remove very common stop-words from keywords
        stop_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this',
                      'not', 'are', 'was', 'but', 'has', 'had', 'have',
                      'its', 'can', 'will', 'how', 'what', 'when', 'where',
                      'use', 'used', 'using', 'node', 'nodes', 'brain'}
        for sw in stop_words:
            all_keywords.pop(sw, None)

        top_keywords = [kw for kw, _ in all_keywords.most_common(max_kw)]
        top_titles = sorted(titles, key=len)[:max_titles]  # shortest titles tend to be most descriptive
        top_type = types.most_common(1)[0][0] if types else 'mixed'
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.7

        # Build title
        if top_keywords:
            title = ', '.join(top_keywords[:3]).title()
        else:
            title = 'Community %d' % hash(tuple(sorted(member_ids))) % 1000

        # Build content
        content_parts = [
            '%d members.' % len(member_ids),
            'Dominant type: %s.' % top_type,
        ]
        if top_keywords:
            content_parts.append('Key themes: %s.' % ', '.join(top_keywords))
        if top_titles:
            content_parts.append('Representative nodes:')
            for t in top_titles:
                content_parts.append('  - %s' % t)

        content = '\n'.join(content_parts)
        keywords = ' '.join(top_keywords)

        # Situation — when is this community relevant
        situation = 'When working on topics related to %s' % ', '.join(top_keywords[:3]) if top_keywords else ''

        return title, content, keywords, situation, avg_confidence

    def _write_results(self, community_members, diff):
        """Write community assignments + community nodes + edges.

        Returns number of write actions performed.
        """
        actions = 0

        # 1. Update node_communities table
        self.brain.conn.execute('DELETE FROM node_communities')
        for cid, members in community_members.items():
            for node_id in members:
                self.brain.conn.execute(
                    'INSERT INTO node_communities (node_id, community_id) VALUES (?, ?)',
                    (node_id, cid))
        self.brain.conn.commit()
        actions += 1

        # 2. Find existing community nodes
        existing_community_nodes = {}
        rows = self.brain.conn.execute(
            "SELECT id, title FROM nodes WHERE encoding_source = ? AND archived = 0",
            (self.ENCODING_SOURCE,)
        ).fetchall()

        # Load metadata to match by community_id
        for nid, title in rows:
            meta_row = self.brain.conn.execute(
                "SELECT value FROM node_metadata_kv WHERE node_id = ? AND key = 'community_id'",
                (nid,)
            ).fetchone()
            if meta_row:
                try:
                    stored_cid = int(meta_row[0])
                    existing_community_nodes[stored_cid] = nid
                except (ValueError, TypeError):
                    pass

        # 3. Archive removed community nodes
        for cid in diff['removed']:
            old_node_id = existing_community_nodes.get(cid)
            if old_node_id:
                self.brain.conn.execute(
                    'UPDATE nodes SET archived = 1 WHERE id = ?', (old_node_id,))
                actions += 1
                self.trace('delta', 'community_removed',
                           'Archived community %d node %s' % (cid, old_node_id[:8]),
                           ref_id=old_node_id)
        self.brain.conn.commit()

        # 4. Create/update community nodes
        for cid, members in community_members.items():
            title, content, keywords, situation, confidence = self._name_community(members)
            existing_node_id = existing_community_nodes.get(cid)

            if existing_node_id and cid not in diff['new']:
                # Update existing community node
                self.brain.revise(
                    node_id=existing_node_id,
                    title=title,
                    content=content,
                    keywords=keywords,
                    situation=situation,
                    confidence=confidence,
                    reason='S2 community detection — membership changed',
                )
                community_node_id = existing_node_id
                self.trace('delta', 'community_updated',
                           'Updated community %d: %s (%d members)' % (cid, title, len(members)),
                           ref_id=existing_node_id)
            else:
                # Create new community node (auto_connect=False — we manage edges explicitly)
                result = self.brain.remember(
                    type='community',
                    title=title,
                    content=content,
                    keywords=keywords,
                    situation=situation,
                    confidence=confidence,
                    encoding_source=self.ENCODING_SOURCE,
                    auto_connect=False,
                )
                community_node_id = result['id']

                # Store community_id in metadata for future diff matching
                self.brain.conn.execute(
                    "INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) VALUES (?, 'community_id', ?)",
                    (community_node_id, str(cid)))
                self.brain.conn.commit()

                self.trace('delta', 'community_created',
                           'Created community %d: %s (%d members)' % (cid, title, len(members)),
                           ref_id=community_node_id)

            actions += 1

            # 5. Create bidirectional edges between community node and members
            self._sync_member_edges(community_node_id, members)
            actions += 1

        return actions

    def _sync_member_edges(self, community_node_id, member_ids):
        """Ensure bidirectional community_member edges between community node and all members.

        Removes edges to nodes no longer in the community.
        Creates edges to new members.
        """
        target_members = set(member_ids)

        # Get existing community_member edges from this community node
        rows = self.brain.conn.execute("""
            SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as member_id
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE (e.source_id = ? OR e.target_id = ?)
            AND er.relation = 'community_member'
        """, (community_node_id, community_node_id, community_node_id)).fetchall()
        existing_outgoing = {r[0] for r in rows}

        # Remove community_member edges to nodes no longer in community
        to_remove = existing_outgoing - target_members
        for old_member in to_remove:
            from servers.dal import GraphDAL
            GraphDAL(self.brain.conn).remove_relation(community_node_id, old_member, 'community_member')

        # Create edges to new members (connect() creates both directions)
        to_add = target_members - existing_outgoing
        for member_id in to_add:
            self.brain.connect(community_node_id, member_id, relation='community_member', weight=0.3)

        if to_remove or to_add:
            self.brain.conn.commit()
