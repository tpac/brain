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

    def backfill_edge_embeddings(self, edge_ids) -> int:
        """Re-embed edge relations whose stored embedding is NULL — the async,
        centralized counterpart of the old synchronous per-write embed.

        Edge writes only INVALIDATE: GraphDAL.add_relation / rename_relation NULL
        the embedding and enqueue_edge. This pass — driven by the embed_queue
        worker, the SAME mechanism that embeds nodes and traces — recomputes off
        the write hot path. Idempotent: skips rows already embedded. Nothing
        outside the brain layer ever touches embedding (S2 units just mutate the
        graph and stay embedding-ignorant).

        Stored on edge_relations.embedding (v26+), read by select_edges /
        _build_edge_coeffs; recall falls back to live compose_edge_text for any
        row still NULL, so the brief pre-drain window is harmless. Returns the
        number of relations embedded.

        Layer note: lives here (not GraphDAL) because the compute needs the
        embedder + brain.aspects.compose_edge_text. GraphDAL stays storage-only.
        """
        ids = [e for e in (edge_ids or []) if e]
        if not ids:
            return 0
        from .dal import DEFAULT_EXCLUDED_RELATIONS
        from . import embedder as _embedder
        if not _embedder.is_ready():
            # Embedder still loading (e.g. at boot before warmup). Don't drop
            # these — re-enqueue so a later drain embeds them. Rows stay NULL
            # meanwhile and recall falls back to live compose, so this is a perf
            # safety net, not data loss.
            from . import embed_queue
            for e in ids:
                embed_queue.enqueue_edge(e)
            return 0
        model = _embedder.stats.get('model_name') or ''
        excl = sorted(DEFAULT_EXCLUDED_RELATIONS)
        ph = ','.join('?' * len(ids))
        excl_ph = ','.join('?' * len(excl))
        # Re-embed rows that are unembedded OR embedded by a DIFFERENT model
        # (a model swap makes the stored blob unreadable — the read path filters
        # embedding_model = active — so stale-model rows must re-embed; matches
        # node find_missing semantics). Exclude co_accessed/emergent_bridge in
        # SQL — they're never read by recall, so embedding them is pure waste.
        rows = self.conn.execute(
            'SELECT edge_id, relation, description FROM edge_relations '
            'WHERE edge_id IN (%s) AND archived = 0 '
            'AND relation NOT IN (%s) '
            'AND (embedding IS NULL OR embedding_model IS NOT ?)'
            % (ph, excl_ph),
            [*ids, *excl, model]).fetchall()
        if not rows:
            return 0
        # Compute OUTSIDE the write lock — fastembed is CPU-heavy. 'document'
        # kind matches the prefix used at recall time (_desc_vecs_batched);
        # a mismatched prefix would break the read path's cosine score. Keep the
        # description each blob was computed from for the concurrency guard below.
        # Compose all texts, then embed in ONE batch — fastembed batches far
        # better than N single-text calls (mirrors backfill_vectors._store_batch;
        # ~10x on a bulk re-embed). Keep the description each blob was computed
        # from for the concurrency guard on write.
        triples = []  # (edge_id, relation, description, text)
        for edge_id, relation, description in rows:
            text = self.aspects.compose_edge_text(relation, description or '')
            if text:
                triples.append((edge_id, relation, description, text))
        if not triples:
            return 0
        try:
            blobs = _embedder.embed_batch([t[3] for t in triples], kind='document')
        except Exception as e:
            self._log_error('edge_embedding_backfill', e, 'batch of %d' % len(triples))
            return 0
        pending = [(eid, rel, desc, blob)            # (edge_id, relation, description, blob)
                   for (eid, rel, desc, _t), blob in zip(triples, blobs)
                   if blob]
        if not pending:
            return 0
        # Write under write_lock (fast) — mirrors backfill_vectors' self-locking.
        # Optimistic guard: only write if the description STILL matches what we
        # embedded (`description IS ?`). A concurrent connect/revise that changed
        # the description has already NULLed + re-enqueued the row; without this
        # guard we'd clobber that fresh-NULL row with the stale blob, and the
        # next drain's filter would no longer see it as needing re-embed —
        # permanently stale geometry. On a description change we match 0 rows and
        # leave the row for the next drain to embed against the new text.
        written = 0
        with self.write_lock:
            for edge_id, relation, description, blob in pending:
                cur = self.conn.execute(
                    'UPDATE edge_relations SET embedding = ?, embedding_model = ? '
                    'WHERE edge_id = ? AND relation = ? AND description IS ?',
                    (blob, model, edge_id, relation, description))
                written += max(cur.rowcount, 0) if cur.rowcount is not None else 0
            self._maybe_commit()
        return written

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
        # Embedding is async: add_relation invalidates + enqueues; the embed_queue
        # worker re-embeds via backfill_edge_embeddings. No sync embed here.
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
        # Embedding is async (add_relation invalidates + enqueues; worker re-embeds).
        return result

    def revise_edge(self, source_id, target_id, relation,
                    new_relation=None, description=None, weight=None,
                    encoding_source=None, reason=''):
        """Revise an existing edge relation IN PLACE. Mirrors revise()'s contract:
        identify the edge-relation row by (source_id, target_id, relation), then
        update only the fields you pass — omit a field to preserve it.

          - new_relation: rename the relation via GraphDAL.rename_relation (in
            place — keeps the same row, its weight, and created_at; no
            delete+recreate). rename_relation NULLs the stale embedding +
            enqueues the edge; the embed_queue worker re-embeds async (this
            method does no embedding work — embedding is a brain-layer concern).
          - description / weight: field-preserving update via add_relation
            (which likewise invalidates + enqueues on a description change).

        Loud (ok=False) on a missing edge / missing relation / rename collision,
        rather than a silent no-op. Returns {ok, edge_id, relation, deltas}.
        """
        gdal = self._graph
        edge_id = gdal.get_edge_id(source_id, target_id)
        if not edge_id:
            return {'ok': False, 'error': 'no edge between %s and %s' % (
                str(source_id)[:8], str(target_id)[:8])}
        active = {r['relation']: r for r in gdal.get_relations(edge_id)}
        if relation not in active:
            return {'ok': False, 'error': 'edge has no active relation %r (has: %s)' % (
                relation, sorted(active))}

        deltas = []
        final_relation = relation
        if new_relation and new_relation != relation:
            # Collision must consider ARCHIVED rows too: the (edge_id, relation)
            # primary key spans active + archived, so renaming onto an archived
            # relation name would violate the PK (uncaught IntegrityError).
            all_relations = {r['relation']
                             for r in gdal.get_relations(edge_id, include_archived=True)}
            if new_relation in all_relations:
                return {'ok': False, 'error': 'edge already has relation %r (active or '
                        'archived) — rename would collide on the (edge_id, relation) '
                        'primary key' % new_relation}
            # rename_relation ALWAYS writes encoding_source. Preserve the row's
            # existing provenance when the caller didn't pass one — defaulting to
            # 'anchor' would silently clobber a field the caller never asked to
            # change (e.g. an 's2:reclassify' edge becoming 'anchor' on rename).
            rename_src = (encoding_source if encoding_source is not None
                          else active[relation].get('encoding_source') or '')
            gdal.rename_relation(edge_id, relation, new_relation, rename_src)
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

        # Embedding is handled async: rename_relation and add_relation both NULL
        # the stored embedding + enqueue_edge, and the embed_queue worker
        # re-embeds via backfill_edge_embeddings. revise_edge does no embed work.
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

