"""
brain — BrainConnections Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import random
from .brain_constants import (
    EDGE_TYPES,
    LEARNING_RATE,
    MAX_HOPS,
    MAX_NEIGHBORS,
    MAX_WEIGHT,
    SPREAD_DECAY,
)


class BrainConnectionsMixin:
    """Connections methods for Brain."""

    def connect(self, source_id: str, target_id: str, relation: str = 'related', weight: float = 0.5):
        """Add a relation between two nodes (idempotent upsert).

        Stage 1B: add_relation is field-preserving — repeated calls do NOT
        auto-strengthen weight (use GraphDAL.strengthen_relation() for Hebbian
        bumps), and unspecified fields preserve existing values on update.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relation type (e.g., 'related', 'co_accessed')
            weight: Edge weight (0-1) — set on create; replaces existing weight on update
        """
        from .dal import GraphDAL
        graph_dal = GraphDAL(self.conn)
        # description omitted so add_relation's sentinel default kicks in
        # (preserves existing on update; defaults to '' on create).
        graph_dal.add_relation(source_id, target_id, relation, weight=weight)

    def connect_typed(self, source_id: str, target_id: str, relation: str = 'related',
                     weight: Optional[float] = None, edge_type: Optional[str] = None,
                     description: Optional[str] = None,
                     encoding_source: Optional[str] = None):
        """Add a typed relation (idempotent upsert).

        Stage 1B: field-preserving upsert. None defaults mean 'preserve existing
        on update' — pass empty string '' explicitly if you want to clear a
        field. Repeated calls do NOT auto-strengthen weight; use
        GraphDAL.strengthen_relation() for Hebbian bumps.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relation name (open text — any string)
            weight: Edge weight (uses EDGE_TYPES default if None — always passed)
            edge_type: DEPRECATED — ignored, kept for backward compat
            description: Why this relation exists. None preserves existing on update.
            encoding_source: Who created this edge. None preserves existing on update.
        """
        # Known types get configured weight; unknown types get 0.5 default
        edge_def = EDGE_TYPES.get(relation)
        actual_weight = weight if weight is not None else (
            edge_def.get('defaultWeight', 0.5) if edge_def else 0.5)

        from .dal import GraphDAL
        graph_dal = GraphDAL(self.conn)
        # Build kwargs: only pass explicitly-provided fields so add_relation's
        # sentinel-based field-preservation propagates cleanly through this
        # layer. weight is always passed (resolved above).
        kwargs = {'weight': actual_weight}
        if description is not None:
            kwargs['description'] = description
        if encoding_source is not None:
            kwargs['encoding_source'] = encoding_source
        graph_dal.add_relation(source_id, target_id, relation, **kwargs)

        # v23: edge_context vectors deferred to idle backfill.
        # Previously recomputed inline here, causing ONNX multi-thread spinning.

    def _random_walk(self, start_id: str, steps: int) -> List[str]:
        """
        Weighted random walk along edges.
        Avoids loops (don't revisit nodes).
        Returns list of node IDs in path.
        """
        path = [start_id]
        current = start_id

        for _ in range(steps):
            neighbors = self.conn.execute('''
                SELECT CASE WHEN source_id = ? THEN target_id ELSE source_id END, weight
                FROM edges WHERE source_id = ? OR target_id = ?
                ORDER BY RANDOM() LIMIT 10
            ''', (current, current, current)).fetchall()

            if not neighbors:
                break

            # Weighted random selection
            total_weight = sum(w for _, w in neighbors)
            if total_weight <= 0:
                break

            roll = random.random() * total_weight
            next_id = neighbors[0][0]
            for nid, w in neighbors:
                roll -= w
                if roll <= 0:
                    next_id = nid
                    break

            # Avoid loops
            if next_id not in path:
                path.append(next_id)
                current = next_id

        return path

    def _get_node_title(self, node_id: str) -> str:
        """Get title of a node by ID."""
        try:
            row = self.conn.execute('SELECT title FROM nodes WHERE id = ?', (node_id,)).fetchone()
            return row[0] if row else node_id
        except:
            return node_id

    def _find_bridge_candidates(self, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find bridge candidates: 2-hop shared neighbor analysis.
        Returns nodes that share >= threshold neighbors but no direct edge.
        """
        threshold = self.get_config('bridge_threshold', 2)
        max_per_node = self.get_config('bridge_max_per_node', 5)

        # Check existing bridge count via GraphDAL (archived=0 default).
        from .dal import GraphDAL
        current_bridge_count = GraphDAL(self.conn).count_node_edges(
            node_id, min_weight=0.0, relations={'emergent_bridge'})

        if current_bridge_count >= max_per_node:
            return []

        slots_left = max_per_node - current_bridge_count

        # Find 2-hop neighbors
        candidates = self.conn.execute(f'''
            SELECT second_hop.id, COUNT(DISTINCT mid.id) as shared_count,
                   second_hop.title, second_hop.type,
                   GROUP_CONCAT(mid.title, ' | ') as shared_titles
            FROM (
              SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as id
              FROM edges e
              WHERE (e.source_id = ? OR e.target_id = ?) AND e.weight >= 0.1
            ) AS neighbor
            JOIN nodes mid ON mid.id = neighbor.id AND mid.archived = 0
            JOIN edges e2 ON (e2.source_id = neighbor.id OR e2.target_id = neighbor.id) AND e2.weight >= 0.1
            JOIN nodes second_hop ON second_hop.id = CASE WHEN e2.source_id = neighbor.id THEN e2.target_id ELSE e2.source_id END
              AND second_hop.id != ?
              AND second_hop.archived = 0
            WHERE second_hop.id NOT IN (
              SELECT CASE WHEN e3.source_id = ? THEN e3.target_id ELSE e3.source_id END
              FROM edges e3
              WHERE e3.source_id = ? OR e3.target_id = ?
            )
            GROUP BY second_hop.id
            HAVING shared_count >= ?
            ORDER BY shared_count DESC
            LIMIT ?
        ''', (node_id, node_id, node_id, node_id, node_id, node_id, node_id, threshold, min(limit, slots_left))).fetchall()

        return [
            {
                'targetId': r[0],
                'sharedCount': r[1],
                'targetTitle': r[2],
                'targetType': r[3],
                'sharedTitles': r[4] or ''
            }
            for r in candidates
        ]

    def _create_bridge(self, source_id: str, target_id: str, shared_titles: str = '') -> Optional[Dict[str, Any]]:
        """
        Create a bridge edge between source and target.
        Returns created edge info or None if bridge already exists.
        """
        # Check no direct edge already exists
        existing = self.conn.execute(
            'SELECT weight FROM edges WHERE source_id = ? AND target_id = ?',
            (source_id, target_id)
        ).fetchone()

        if existing:
            return None

        # Get titles
        src_title = self._get_node_title(source_id) or source_id
        tgt_title = self._get_node_title(target_id) or target_id

        # Description: just the structural fact. LLM-generated "why" is a future consolidation improvement.
        description = 'shares %d neighbors' % max(2, shared_titles.count('|') + 1) if shared_titles else ''

        self.connect_typed(source_id, target_id, 'emergent_bridge', 0.15, 'emergent_bridge', description)

        return {'sourceId': source_id, 'targetId': target_id, 'description': description, 'weight': 0.15}

