"""
brain — BrainConnections Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
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

    def _maybe_embed_edge_relation(self, edge_id: str, relation: str,
                                    add_result: dict) -> None:
        """Compute + store the edge's enriched-text embedding when a write
        actually changed the (relation, description, family meaning) tuple.

        Stored on `edge_relations.embedding` (schema v26+). Read by
        `_build_edge_coeffs` and `select_edges` instead of running fastembed
        at recall time. The 'edges have stored embeddings like nodes'
        symmetry — see docs/aspects.py compose_edge_text.

        Skipped when:
          - The write was a no-op (no fields changed) — existing embedding
            is still valid.
          - The 'description' field wasn't part of the deltas — only changes
            that affect `compose_edge_text(relation, description)` invalidate
            the stored embedding.
          - Embedding fails (logged; row keeps NULL or stale; spread falls
            back to live compose for that edge).

        Concurrency: assumes the caller holds `brain.write_lock` (the
        daemon dispatch wrapper acquires it for the whole `fn()`, so
        connect_typed → add_relation → this method all serialize on
        the same lock). Two concurrent revisions are fully ordered;
        last writer wins, embedding always matches the description it
        was computed from. A reader running concurrently sees snapshot
        consistency (WAL) — there's a brief window between
        add_relation's commit (description=new) and the embedding
        UPDATE+commit landing where the reader observes `description=new`
        with `embedding=embed(old)`. That's transient and harmless: the
        cosine score is slightly off for one recall, self-heals next call.

        Durability: explicit commit at the end. Without it, the embedding
        UPDATE stayed uncommitted until the next writer's
        `add_relation.commit()` or the autosave loop — meaning the LAST
        embedding in a brain_batch (or any single connect_typed call
        that wasn't followed by another writer) had a gap where a daemon
        crash would lose the embedding. add_relation already commits its
        own writes; this method now matches that contract.

        Layer note: lives in BrainConnectionsMixin (not GraphDAL) because
        the embedding work needs `brain.aspects` (for family meaning) and
        the embedder — neither belongs in DAL. GraphDAL stays storage-only.
        """
        # Skip relations that are never read by surface_spread or
        # select_edges (DEFAULT_EXCLUDED_RELATIONS = {co_accessed,
        # emergent_bridge}). Hebbian fires on every recall; embedding
        # those edges is pure wasted work.
        from .dal import DEFAULT_EXCLUDED_RELATIONS
        if relation in DEFAULT_EXCLUDED_RELATIONS:
            return
        # Skip if write was a no-op
        if not (add_result.get('created') or add_result.get('updated') or
                add_result.get('revived_from_archive')):
            return
        # Skip if description wasn't part of the change set (and this is an
        # update, not a fresh insert). New rows always need embedding.
        if add_result.get('updated') and not add_result.get('created'):
            text_changed = any(
                d.get('field') in ('description', 'relation')
                for d in add_result.get('deltas') or [])
            if not text_changed:
                return

        try:
            row = self.conn.execute(
                'SELECT description FROM edge_relations '
                'WHERE edge_id = ? AND relation = ?',
                (edge_id, relation)).fetchone()
            if row is None:
                return
            description = row[0] or ''
            text = self.aspects.compose_edge_text(relation, description)
            if not text:
                return
            from . import embedder as _embedder
            # 'document' kind matches the prefix used at recall time
            # (`_desc_vecs_batched` calls `embed_batch(kind='document')`).
            # Mismatched prefixes produce different vectors for the same
            # text — would break the read path's cosine score.
            blobs = _embedder.embed_batch([text], kind='document')
            blob = blobs[0] if blobs else None
            if not blob:
                return
            model = (_embedder.stats.get('model_name') or '') if hasattr(
                _embedder, 'stats') else ''
            self.conn.execute(
                'UPDATE edge_relations SET embedding = ?, embedding_model = ? '
                'WHERE edge_id = ? AND relation = ?',
                (blob, model, edge_id, relation))
            # Explicit commit so the embedding write is durable on its own
            # — the caller holds brain.write_lock, so this is a fast WAL
            # commit without contention. add_relation commits the
            # description; this commits the embedding. Symmetric with
            # node_enrichments writes which commit per node.
            self._maybe_commit()
        except Exception as e:
            # Embedding failure must NOT fail the connect — the row exists,
            # spread will fall through to the on-demand embed path. Log
            # loudly so a systemic embed failure is visible.
            self._log_error(
                'edge_embedding_write', e,
                'compute+store edge embedding for %s/%s' %
                (edge_id[:12] if edge_id else '?', relation))

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

        Returns:
            The result dict from GraphDAL.add_relation() (edge_id, created,
            revived_from_archive, updated, deltas, warnings) — used by callers
            that want to emit trace events.
        """
        graph_dal = self._graph
        # description omitted so add_relation's sentinel default kicks in
        # (preserves existing on update; defaults to '' on create).
        result = graph_dal.add_relation(source_id, target_id, relation, weight=weight)
        self._maybe_embed_edge_relation(result.get('edge_id'), relation, result)
        return result

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

        Returns:
            The result dict from GraphDAL.add_relation() (edge_id, created,
            revived_from_archive, updated, deltas, warnings) — used by callers
            that want to emit trace events.
        """
        # Known types get configured weight; unknown types get 0.5 default
        edge_def = EDGE_TYPES.get(relation)
        actual_weight = weight if weight is not None else (
            edge_def.get('defaultWeight', 0.5) if edge_def else 0.5)

        graph_dal = self._graph
        # Build kwargs: only pass explicitly-provided fields so add_relation's
        # sentinel-based field-preservation propagates cleanly through this
        # layer. weight is always passed (resolved above).
        kwargs = {'weight': actual_weight}
        if description is not None:
            kwargs['description'] = description
        if encoding_source is not None:
            kwargs['encoding_source'] = encoding_source
        result = graph_dal.add_relation(source_id, target_id, relation, **kwargs)
        self._maybe_embed_edge_relation(result.get('edge_id'), relation, result)
        return result

    def revise_edge(self, source_id, target_id, relation,
                    new_relation=None, description=None, weight=None,
                    encoding_source=None, reason=''):
        """Revise an existing edge relation IN PLACE. Mirrors revise()'s contract:
        identify the edge-relation row by (source_id, target_id, relation), then
        update only the fields you pass — omit a field to preserve it.

          - new_relation: rename the relation via GraphDAL.rename_relation (in
            place — keeps the same row, its weight, and created_at; no
            delete+recreate). The relation string is part of compose_edge_text,
            so the edge embedding is refreshed below — a bare rename would
            otherwise leave a stale embedding.
          - description / weight: field-preserving update via add_relation.

        Loud (ok=False) on a missing edge / missing relation / rename collision,
        rather than a silent no-op. Returns {ok, edge_id, relation, deltas}.
        """
        gdal = self._graph
        edge_id = gdal.get_edge_id(source_id, target_id)
        if not edge_id:
            return {'ok': False, 'error': 'no edge between %s and %s' % (
                str(source_id)[:8], str(target_id)[:8])}
        active = {r['relation'] for r in gdal.get_relations(edge_id)}
        if relation not in active:
            return {'ok': False, 'error': 'edge has no active relation %r (has: %s)' % (
                relation, sorted(active))}

        deltas = []
        src = encoding_source or 'anchor'
        final_relation = relation
        if new_relation and new_relation != relation:
            if new_relation in active:
                return {'ok': False, 'error': 'edge already has active relation %r '
                        '— rename would collide' % new_relation}
            gdal.rename_relation(edge_id, relation, new_relation, src)
            deltas.append({'field': 'relation', 'old': relation, 'new': new_relation})
            final_relation = new_relation

        if description is not None or weight is not None:
            kwargs = {}
            if description is not None:
                kwargs['description'] = description
            if weight is not None:
                kwargs['weight'] = weight
            if encoding_source is not None:
                kwargs['encoding_source'] = encoding_source
            res = gdal.add_relation(source_id, target_id, final_relation, **kwargs)
            deltas.extend(res.get('deltas') or [])

        # Refresh the edge embedding when the embedded (relation, description)
        # tuple changed. rename_relation does NOT re-embed on its own, so without
        # this a rename leaves edge_relations.embedding computed from the OLD
        # relation string. The gate fires on a 'relation' OR 'description' delta.
        if any(d.get('field') in ('relation', 'description') for d in deltas):
            self._maybe_embed_edge_relation(
                edge_id, final_relation, {'updated': True, 'deltas': deltas})

        return {'ok': True, 'edge_id': edge_id, 'relation': final_relation,
                'deltas': deltas}

    # _random_walk removed 2026-05-30 (DAL cleanup Phase 0) — dead (0 callers);
    # the random-walk neighbor path is retired (GraphDAL.get_random_walk_neighbors
    # was also removed).

    def _get_node_title(self, node_id: str) -> str:
        """Get title of a node by ID, falling back to the id if absent."""
        return self._nodes.get_title(node_id) or node_id

    def _find_bridge_candidates(self, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find bridge candidates: 2-hop shared neighbor analysis.
        Returns nodes that share >= threshold neighbors but no direct edge.
        """
        threshold = self.get_config('bridge_threshold', 2)
        max_per_node = self.get_config('bridge_max_per_node', 5)

        # Check existing bridge count via GraphDAL (archived=0 default).
        current_bridge_count = self._graph.count_node_edges(
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

