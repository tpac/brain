"""
brain — BrainAbsorb Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from typing import Any, Dict


class BrainAbsorbMixin:
    """Absorb methods for Brain."""

    def absorb(self, source_brain: 'Brain',
               auto_merge_locked: bool = True,
               auto_merge_unlocked: bool = False,
               fuzzy_threshold: int = 3,
               dry_run: bool = False) -> Dict[str, Any]:
        """
        Absorb knowledge from another brain into this one.

        Finds nodes in source_brain that don't exist here, and merges them in.
        Designed for the Hindsight cycle: run relearning simulation → absorb
        new discoveries back into the active brain.

        Strategy:
          1. For each source node, fuzzy-match against existing titles
          2. Exact/fuzzy match → skip (already known) or flag (possible update)
          3. No match + locked → auto-absorb (high-confidence new knowledge)
          4. No match + unlocked → skip by default (context/task decay anyway)
          5. Transfer connections where both endpoints exist in target brain

        Args:
            source_brain: Brain to absorb knowledge from
            auto_merge_locked: Auto-absorb new locked nodes (default True)
            auto_merge_unlocked: Auto-absorb new unlocked nodes (default False)
            fuzzy_threshold: Min word overlap for fuzzy title matching (default 3)
            dry_run: If True, report what would happen without making changes

        Returns:
            Dict with absorbed, skipped, flagged, connections_created counts
            and detailed lists of each category
        """
        report = {
            'absorbed': [],
            'skipped': [],
            'flagged': [],
            'connections_created': 0,
            'summary': {},
        }

        # Build title index of existing nodes for fast matching
        existing = {}
        cursor = self.conn.execute('SELECT id, title, type, locked FROM nodes WHERE title IS NOT NULL')
        for row in cursor.fetchall():
            nid, title, ntype, locked = row
            existing[title] = {
                'id': nid,
                'type': ntype,
                'locked': bool(locked),
                'words': set(title.lower().split()),
            }

        # Map source node IDs → new node IDs (for connection transfer)
        id_map = {}

        # Get all source nodes
        source_nodes = source_brain.conn.execute(
            '''SELECT id, type, title, content, keywords, locked, emotion,
                      emotion_label, project, access_count
               FROM nodes WHERE title IS NOT NULL
               ORDER BY locked DESC, access_count DESC'''
        ).fetchall()

        for src_row in source_nodes:
            (src_id, src_type, src_title, src_content, src_keywords,
             src_locked, src_emotion, src_emotion_label, src_project,
             src_access) = src_row

            if not src_title:
                continue

            # ── Step 1: Match against existing nodes ──
            match_type, match_node = self._match_existing(
                src_title, existing, fuzzy_threshold
            )

            if match_type == 'exact':
                # Already have this exact title — skip
                report['skipped'].append({
                    'title': src_title[:80],
                    'type': src_type,
                    'reason': 'exact_duplicate',
                    'existing_id': match_node['id'],
                })
                id_map[src_id] = match_node['id']
                continue

            if match_type == 'fuzzy':
                # Close match — flag for review (might be update or duplicate)
                report['flagged'].append({
                    'source_title': src_title[:80],
                    'source_type': src_type,
                    'source_locked': bool(src_locked),
                    'existing_title': match_node.get('_matched_title', '')[:80],
                    'existing_id': match_node['id'],
                    'reason': 'fuzzy_match',
                })
                id_map[src_id] = match_node['id']
                continue

            # ── Step 2: New knowledge — decide whether to absorb ──
            should_absorb = False
            if src_locked and auto_merge_locked:
                should_absorb = True
            elif not src_locked and auto_merge_unlocked:
                should_absorb = True

            if not should_absorb:
                report['skipped'].append({
                    'title': src_title[:80],
                    'type': src_type,
                    'locked': bool(src_locked),
                    'reason': 'policy_skip' if not src_locked else 'auto_merge_disabled',
                })
                continue

            # ── Step 3: Absorb the node ──
            if dry_run:
                report['absorbed'].append({
                    'title': src_title[:80],
                    'type': src_type,
                    'locked': bool(src_locked),
                    'emotion': src_emotion or 0,
                    'dry_run': True,
                })
                continue

            try:
                result = self.remember(
                    type=src_type,
                    title=src_title,
                    content=src_content or '',
                    keywords=src_keywords or '',
                    locked=bool(src_locked),
                    emotion=float(src_emotion or 0),
                    emotion_label=src_emotion_label or 'neutral',
                    project=src_project or '',
                )
                new_id = result.get('id') if isinstance(result, dict) else result
                id_map[src_id] = new_id

                report['absorbed'].append({
                    'title': src_title[:80],
                    'type': src_type,
                    'locked': bool(src_locked),
                    'new_id': new_id,
                    'emotion': src_emotion or 0,
                })

                # Register in existing index for subsequent matching
                existing[src_title] = {
                    'id': new_id,
                    'type': src_type,
                    'locked': bool(src_locked),
                    'words': set(src_title.lower().split()),
                }

            except Exception as e:
                report['flagged'].append({
                    'source_title': src_title[:80],
                    'source_type': src_type,
                    'reason': f'remember_error: {e}',
                })

        # ── Step 4: Transfer connections ──
        if not dry_run:
            report['connections_created'] = self._absorb_connections(
                source_brain, id_map
            )

        # ── Summary ──
        report['summary'] = {
            'source_nodes': len(source_nodes),
            'absorbed': len(report['absorbed']),
            'skipped': len(report['skipped']),
            'flagged': len(report['flagged']),
            'connections_created': report['connections_created'],
            'dry_run': dry_run,
        }

        if not dry_run:
            self.conn.commit()

        return report

    def _match_existing(self, title: str, existing: Dict,
                        fuzzy_threshold: int = 3) -> tuple:
        """
        Match a title against existing nodes.

        Returns:
            ('exact', node_dict) — exact title match
            ('fuzzy', node_dict) — fuzzy word overlap match
            ('none', None) — no match found
        """
        # Exact match
        if title in existing:
            return ('exact', existing[title])

        # Fuzzy match: check word overlap
        title_words = set(title.lower().split())
        best_overlap = 0
        best_match = None
        best_title = ''

        for ex_title, ex_node in existing.items():
            overlap = len(title_words & ex_node['words'])
            if overlap > best_overlap and overlap >= fuzzy_threshold:
                best_overlap = overlap
                best_match = ex_node
                best_title = ex_title

        if best_match:
            best_match['_matched_title'] = best_title
            return ('fuzzy', best_match)

        return ('none', None)

    def _absorb_connections(self, source_brain: 'Brain',
                            id_map: Dict[str, str]) -> int:
        """
        Transfer edges from source brain where both endpoints exist in target.

        Args:
            source_brain: Brain to read edges from
            id_map: Mapping of source_id → target_id

        Returns:
            Number of connections created
        """
        created = 0
        source_edges = source_brain.conn.execute(
            '''SELECT source_id, target_id, weight, relation, edge_type
               FROM edges
               WHERE edge_type IN ('related', 'part_of', 'corrected_by',
                                   'exemplifies', 'depends_on', 'produced')'''
        ).fetchall()

        for src_source, src_target, weight, relation, edge_type in source_edges:
            # Both endpoints must exist in target brain
            target_source = id_map.get(src_source)
            target_target = id_map.get(src_target)

            if not target_source or not target_target:
                continue
            if target_source == target_target:
                continue

            # Check if edge already exists
            existing = self.conn.execute(
                'SELECT 1 FROM edges WHERE source_id = ? AND target_id = ?',
                (target_source, target_target)
            ).fetchone()

            if existing:
                continue

            try:
                self.connect_typed(
                    source_id=target_source,
                    target_id=target_target,
                    relation=relation or 'related',
                    weight=weight or 0.5,
                    edge_type=edge_type or 'related',
                )
                created += 1
            except Exception as _e:
                self._log_error("_absorb_connections", _e, "absorbing edge connection from source brain")

        return created
