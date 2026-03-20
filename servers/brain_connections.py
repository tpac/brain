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
    STABILITY_BOOST,
)


class BrainConnectionsMixin:
    """Connections methods for Brain."""

    def connect(self, source_id: str, target_id: str, relation: str = 'related', weight: float = 0.5):
        """
        Create or strengthen an edge between two nodes.
        Bidirectional Hebbian learning.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relation type (e.g., 'related', 'inspired_by')
            weight: Edge weight (0-1)
        """
        ts = self.now()

        # Check if edge already exists
        cursor = self.conn.execute(
            'SELECT weight, co_access_count FROM edges WHERE source_id = ? AND target_id = ?',
            (source_id, target_id)
        )
        existing = cursor.fetchone()

        if existing:
            old_weight, count = existing
            new_weight = min(MAX_WEIGHT, old_weight + LEARNING_RATE * 0.5)
            self.conn.execute(
                'UPDATE edges SET weight = ?, co_access_count = ?, last_strengthened = ?, relation = ?, edge_type = ? WHERE source_id = ? AND target_id = ?',
                (new_weight, count + 1, ts, relation, relation, source_id, target_id)
            )
        else:
            # Create bidirectional edge
            self.conn.execute(
                'INSERT OR IGNORE INTO edges (source_id, target_id, weight, relation, edge_type, co_access_count, stability, last_strengthened, created_at) VALUES (?, ?, ?, ?, ?, 1, 1.0, ?, ?)',
                (source_id, target_id, weight, relation, relation, ts, ts)
            )
            self.conn.execute(
                'INSERT OR IGNORE INTO edges (source_id, target_id, weight, relation, edge_type, co_access_count, stability, last_strengthened, created_at) VALUES (?, ?, ?, ?, ?, 1, 1.0, ?, ?)',
                (target_id, source_id, weight, relation, relation, ts, ts)
            )

        self.conn.commit()

    def connect_typed(self, source_id: str, target_id: str, relation: str = 'related',
                     weight: Optional[float] = None, edge_type: Optional[str] = None,
                     description: str = ''):
        """
        Create or strengthen typed edge with description.
        Edge types can be any string; EDGE_TYPES defines decay behavior for known types.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relation name
            weight: Edge weight (optional; uses EDGE_TYPES default if not provided)
            edge_type: Edge type (optional; defaults to relation)
            description: Human-readable description of connection
        """
        if not edge_type:
            edge_type = relation

        # Known types get configured weight; unknown types get 0.5 default
        edge_def = EDGE_TYPES.get(edge_type)
        actual_weight = weight if weight is not None else (edge_def.get('defaultWeight', 0.5) if edge_def else 0.5)

        ts = self.now()

        # Check if edge already exists
        cursor = self.conn.execute(
            'SELECT weight, co_access_count FROM edges WHERE source_id = ? AND target_id = ?',
            (source_id, target_id)
        )
        existing = cursor.fetchone()

        if existing:
            old_weight, count = existing
            new_weight = min(MAX_WEIGHT, old_weight + LEARNING_RATE * 0.5)
            self.conn.execute(
                'UPDATE edges SET weight = ?, co_access_count = ?, last_strengthened = ?, relation = ?, edge_type = ?, description = ? WHERE source_id = ? AND target_id = ?',
                (new_weight, count + 1, ts, relation, edge_type, description, source_id, target_id)
            )
        else:
            # Create bidirectional edge
            self.conn.execute(
                'INSERT OR IGNORE INTO edges (source_id, target_id, weight, relation, edge_type, description, co_access_count, stability, last_strengthened, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, 1.0, ?, ?)',
                (source_id, target_id, actual_weight, relation, edge_type, description, ts, ts)
            )
            self.conn.execute(
                'INSERT OR IGNORE INTO edges (source_id, target_id, weight, relation, edge_type, description, co_access_count, stability, last_strengthened, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, 1.0, ?, ?)',
                (target_id, source_id, actual_weight, relation, edge_type, description, ts, ts)
            )

        self.conn.commit()

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
                SELECT target_id, weight FROM edges WHERE source_id = ? ORDER BY RANDOM() LIMIT 10
            ''', (current,)).fetchall()

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

        # Check existing bridge count
        existing = self.conn.execute('''
            SELECT COUNT(*) FROM edges
            WHERE (source_id = ? OR target_id = ?) AND edge_type = 'emergent_bridge'
        ''', (node_id, node_id)).fetchone()
        current_bridge_count = existing[0] if existing else 0

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

        # Auto-generate description
        shared_part = f' via shared neighbors: {shared_titles[:150]}' if shared_titles else ''
        description = f'Emergent bridge: "{src_title[:60]}" ↔ "{tgt_title[:60]}"{shared_part}'

        self.connect_typed(source_id, target_id, 'emergent_bridge', 0.15, 'emergent_bridge', description)

        return {'sourceId': source_id, 'targetId': target_id, 'description': description, 'weight': 0.15}

    def _bridge_at_consolidation(self) -> List[Dict[str, Any]]:
        """
        Broader bridging sweep during consolidation.
        Scans recent + random nodes, creates bridges, spawns thoughts for high-shared-count pairs.
        """
        max_bridges = self.get_config('bridge_max_per_consolidation', 5)
        scan_size = 20
        created = []
        max_thoughts = 2

        # Recency-biased node selection
        recent = self.conn.execute('''
            SELECT id FROM nodes WHERE archived = 0 AND locked = 0
            AND last_accessed > datetime('now', '-48 hours')
            ORDER BY RANDOM() LIMIT ?
        ''', (max(1, scan_size // 2),)).fetchall()

        random_nodes = self.conn.execute('''
            SELECT id FROM nodes WHERE archived = 0 AND locked = 0
            ORDER BY RANDOM() LIMIT ?
        ''', (scan_size,)).fetchall()

        # Include locked engineering/vocabulary nodes — they need organic connections too
        locked_eng = self.conn.execute('''
            SELECT id FROM nodes WHERE archived = 0 AND locked = 1
            AND type IN ('vocabulary', 'purpose', 'mechanism', 'impact',
                         'constraint', 'convention', 'lesson', 'mental_model')
            ORDER BY RANDOM() LIMIT ?
        ''', (max(1, scan_size // 4),)).fetchall()

        # Merge with dedup
        seen_ids = set()
        merged = []
        for source in [recent, random_nodes, locked_eng]:
            for (node_id,) in source:
                if node_id not in seen_ids and len(merged) < scan_size + len(locked_eng):
                    seen_ids.add(node_id)
                    merged.append(node_id)

        thoughts = 0
        for node_id in merged:
            if len(created) >= max_bridges:
                break
            candidates = self._find_bridge_candidates(node_id, limit=2)
            for c in candidates:
                if len(created) >= max_bridges:
                    break
                bridge = self._create_bridge(node_id, c['targetId'], c.get('sharedTitles', ''))
                if bridge:
                    created.append(bridge)
                    # Spawn thought for high shared-count
                    if c['sharedCount'] >= 4 and thoughts < max_thoughts:
                        try:
                            src_title = self._get_node_title(node_id) or node_id
                            tgt_title = c['targetTitle'] or c['targetId']
                            self._spawn_thought(
                                f'Cluster forming: "{src_title}" and "{tgt_title}" share {c["sharedCount"]} neighbors ({c.get("sharedTitles", "")[:100]}). These areas are converging.',
                                [node_id, c['targetId']],
                                'cluster_observation'
                            )
                            thoughts += 1
                        except Exception as _e:
                            self._log_error("_bridge_at_consolidation", _e, "spawning cluster observation thought")

        return created

    def _propose_bridge(self, source_id: str, target_id: str, shared_titles: str = '',
                       dream_session_id: str = '') -> Optional[Dict[str, Any]]:
        """
        Propose a bridge into maturation queue (used by dream).
        Bridge won't be created until maturation timer expires.
        """
        max_pending = self.get_config('bridge_dream_max_pending', 10)

        # Check current pending count
        try:
            count_row = self.conn.execute(
                "SELECT COUNT(*) FROM bridge_proposals WHERE status = 'pending'"
            ).fetchone()
            current = count_row[0] if count_row else 0
            if current >= max_pending:
                return None
        except:
            return None

        # Check no direct edge exists
        existing = self.conn.execute(
            'SELECT weight FROM edges WHERE source_id = ? AND target_id = ?',
            (source_id, target_id)
        ).fetchone()
        if existing:
            return None

        # Check not already proposed
        try:
            dup = self.conn.execute(
                "SELECT id FROM bridge_proposals WHERE source_id = ? AND target_id = ? AND status = 'pending'",
                (source_id, target_id)
            ).fetchone()
            if dup:
                return None
        except:
            return None

        maturation_minutes = self.get_config('bridge_dream_maturation_minutes', 120)
        ts = self.now()

        try:
            self.conn.execute('''
                INSERT INTO bridge_proposals
                  (source_id, target_id, shared_context, dream_session_id, status, proposed_at, matures_at)
                VALUES (?, ?, ?, ?, 'pending', ?, datetime(?, '+' || ? || ' minutes'))
            ''', (source_id, target_id, shared_titles[:300], dream_session_id, ts, ts, maturation_minutes))
        except:
            return None

        return {'sourceId': source_id, 'targetId': target_id, 'maturationMinutes': maturation_minutes}

    def _mature_bridge_proposals(self) -> int:
        """
        Promote pending bridge proposals that have matured.
        Returns count of bridges created.
        """
        matured = 0
        try:
            ready = self.conn.execute('''
                SELECT id, source_id, target_id, shared_context FROM bridge_proposals
                WHERE status = 'pending' AND matures_at <= datetime('now')
            ''').fetchall()

            for row_id, src, tgt, ctx in ready:
                bridge = self._create_bridge(src, tgt, ctx or '')
                if bridge:
                    self.conn.execute(
                        "UPDATE bridge_proposals SET status = 'created' WHERE id = ?",
                        (row_id,)
                    )
                    matured += 1
                else:
                    self.conn.execute(
                        "UPDATE bridge_proposals SET status = 'expired' WHERE id = ?",
                        (row_id,)
                    )
        except Exception as _e:
            self._log_error("_mature_bridge_proposals", _e, "maturing or expiring bridge proposals")

        return matured
