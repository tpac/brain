"""
brain — BrainRecall Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import math
import struct
import sys
import time
# Recall result cache (2026-05-08) — short-TTL dedup on top of the
# single-flight gate. 10s window covers multi-hook bursts (pre_edit +
# pre_bash_safety + hook_recall fire on every tool call) without
# letting suggestions go stale on natural pauses. Tom: "let's increase
# pre_edit and bash cache to 10 seconds" — both go through this single
# recall layer now, so one TTL knob covers both.
_RECALL_CACHE_TTL_S = 10.0
_RECALL_CACHE_MAX_ENTRIES = 100


from .brain_constants import (
    CRITICAL_BOOST,
    CRITICAL_SIMILARITY_THRESHOLD,
    DECAY_HALF_LIFE,
    EDGE_TYPES,
    EMBEDDING_PRIMARY_WEIGHT,
    GRAPH_AUGMENT_TOP_N,
    INTENTIONAL_EDGE_TYPES,
    KEYWORD_FALLBACK_WEIGHT,
    LEARNING_RATE,
    MAX_PAGE_SIZE,
    MAX_WEIGHT,
    PRUNE_THRESHOLD,
    RELEVANCE_FLOOR_ENRICHED,
    RELEVANCE_FLOOR_PRIMARY,
    TRAVERSE_DAMPEN,
    TRAVERSE_LIMITS,
    TRAVERSE_SEMANTIC_BONUS,
    TRAVERSE_SEMANTIC_THRESHOLD,
    TRAVERSE_CONVERGENCE_BOOST,
    FRESHNESS_MULTIPLIERS,
    EXCLUDED_EDGE_TYPES,
    SITUATION_WEIGHT,
    SITUATION_THRESHOLD,
    NOISE_FLOOR_THRESHOLD,
    FTS5_CANDIDATE_LIMIT,
    FTS5_SEARCH_LIMIT,
    FTS5_PASSTHROUGH_SCORE,
    RETRIEVAL_LOW_CONFIDENCE,
    ZSCORE_ENABLED,
    ZSCORE_MIN_STD,
    ZSCORE_DEFAULT_MEAN,
    ZSCORE_DEFAULT_STD,
    ZSCORE_STATS_KEY_MEAN,
    ZSCORE_STATS_KEY_STD,
)
from .dal import GraphDAL, NodeDAL, TfIdfDAL, LogsDAL, Fts5DAL, VectorDAL


# ── Lexical bridge — Haiku-generated query expansion ───────────────
# Cosine in our embedding space is flat (top-25 spread ~0.09) and doesn't
# bridge synonyms (feed/scratch grains) or contrastive cases (uncle/niece).
# This helper asks Haiku for 2-3 alternate phrasings — synonyms, related
# entities, and explicit contrasts (for abstention queries). Each phrasing
# gets embedded; downstream cosine takes max across all phrasings.
#
# Opt-in via env var BRAIN_QUERY_EXPANSION=on. Failure modes are non-fatal:
# Haiku error → skip expansion, recall continues with primary query only.

_EXPANSION_PROMPT = """Generate 2-3 alternate query phrasings that bridge LEXICAL GAPS — vocabulary differences between how the user asks and how the memory was originally stored. Don't paraphrase. Don't say the same thing in different words.

Each alternate MUST drop or replace at least one specific term from the original query, choosing one of these strategies:

1. STRIP the specific entity, keep the activity:
   "What did I bake for my uncle's birthday party?" → "what I baked for a birthday party"
   "Where did I attend study abroad?" → "country I studied in", "university I went to"

2. REPLACE the specific entity with a category or sibling entity (in case the memory is about a related entity):
   "uncle's birthday" → "family member's birthday", "niece's birthday"
   "feed" → "feed for chickens", "scratch grains for chickens"
   "Memrise" → "language learning apps with mnemonics", "apps for memorization"

3. BROADEN to the category the original is in:
   "siblings count" → "brothers and sisters family"
   "gym time" → "evening workout schedule"

The original query gets searched separately. Your alternates must reach memories the original would NOT.

Return ONLY a JSON array of 2-3 strings, no prose, no explanation.

Query: "{query}"
"""


def _expand_query_via_haiku(query: str) -> List[str]:
    """Ask Haiku for 2-3 alternate phrasings to bridge lexical gaps in cosine.

    Returns list of strings (may be empty on any failure). Cost: 1 Haiku
    call (~1s, ~300 tokens). Caller is expected to embed each separately.
    """
    if not query or len(query.strip()) < 3:
        return []
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception:
        return []
    try:
        resp = client.messages.create(
            model='claude-haiku-4-5',
            max_tokens=200,
            messages=[{'role': 'user',
                       'content': _EXPANSION_PROMPT.format(query=query)}],
        )
        text = ''.join(b.text for b in resp.content if hasattr(b, 'text')).strip()
    except Exception:
        return []
    # Tolerate Haiku wrapping the array in code fences or extra prose.
    if '```' in text:
        # Strip markdown code fences
        parts = text.split('```')
        for p in parts:
            p = p.strip()
            if p.startswith('json'):
                p = p[4:].strip()
            if p.startswith('['):
                text = p
                break
    # Find JSON array bounds
    start = text.find('[')
    end = text.rfind(']')
    if start < 0 or end < 0 or end <= start:
        return []
    try:
        arr = json.loads(text[start:end + 1])
        return [s for s in arr if isinstance(s, str) and s.strip()][:3]
    except Exception:
        return []


# Node table columns — filter checks these on the result dict.
# Anything not in this set is looked up as a metadata key.
_NODE_COLUMNS = frozenset({
    'id', 'type', 'title', 'content', 'keywords', 'activation', 'stability',
    'access_count', 'locked', 'archived', 'critical', 'recency_score',
    'emotion', 'emotion_label', 'emotion_source', 'project', 'confidence',
    'personal', 'personal_context', 'evolution_status', 'resolved_at',
    'resolved_by', 'due_date', 'content_summary', 'source_attribution',
    'scope', 'encoding_version', 'encoding_source', 'revised_at',
    'last_accessed', 'created_at',
})


def _apply_filter(nodes: list, filter_dict: dict, conn) -> list:
    """Apply dict filter to a list of node dicts.

    Filter format:
        {"type": {"in": ["moment", "reflection"]}}
        {"anchor_raw_quote": {"exists": True}}
        {"content": {"contains": "Anchor"}}
        {"confidence": {"gte": 0.9}}
        {"locked": {"equals": 1}}

    Keys in _NODE_COLUMNS are read from the node dict.
    Other keys are looked up in node_metadata_kv.
    """
    if not filter_dict:
        return nodes

    from .dal_metadata import MetadataDAL
    mdal = MetadataDAL(conn)

    def _matches(node):
        for key, conditions in filter_dict.items():
            # Get the value
            if key in _NODE_COLUMNS:
                val = node.get(key)
            else:
                val = mdal.get_field(node['id'], key)

            # Apply conditions
            for op, expected in conditions.items():
                if op == 'exists':
                    if expected and not val:
                        return False
                    if not expected and val:
                        return False
                elif op == 'equals':
                    if val != expected:
                        return False
                elif op == 'in':
                    if val not in expected:
                        return False
                elif op == 'contains':
                    if not val or expected not in str(val):
                        return False
                elif op == 'gte':
                    if val is None or val < expected:
                        return False
                elif op == 'lte':
                    if val is None or val > expected:
                        return False
                else:
                    print('[brain] WARNING: unknown filter operator %r for key %r' % (op, key), file=sys.stderr)
        return True

    out = []
    for n in nodes:
        try:
            if _matches(n):
                out.append(n)
        except Exception as e:
            print('[brain] ERROR in _apply_filter for node %s: %s' % (n.get('id', '?'), e), file=sys.stderr)
    return out


def _rerank_by_relevance(conn, rich_nodes: list, query_text: str, limit: int,
                         vector_type: str = '_primary') -> list:
    """Re-rank a list of rich nodes by cosine similarity to the query embedding.

    Frame Phase 2.5 helper. Hybrid output: top half by relevance, remainder by
    the order the caller passed in (which is the structural sort — recency,
    typically). Lets relevance lift the most-meaningful candidates while
    preserving the structural recency floor.

    vector_type: which `node_enrichments.vector_type` to score against.
    Default `_primary` (canonical single-vector encoding, full coverage). v2
    callers may pass other types or, eventually, request multi-field
    z-weighted scoring (mirroring surface).

    Nodes without an embedding fall through to the recency tail. Cost: 1
    embed call + N cosine ops + 1 SQL query for embeddings (single batch).
    """
    if not query_text or not rich_nodes:
        return rich_nodes[:limit]
    try:
        import numpy as np
        from . import embedder as _emb
        if not _emb.is_ready():
            return rich_nodes[:limit]
        query_blob = _emb.embed_query(query_text)
        if not query_blob:
            return rich_nodes[:limit]
        query_vec = np.frombuffer(query_blob, dtype=np.float32) \
            if isinstance(query_blob, bytes) else np.array(query_blob, dtype=np.float32)
        norm = float(np.linalg.norm(query_vec))
        if norm < 1e-9:
            return rich_nodes[:limit]
        query_vec = query_vec / norm

        # Batch fetch _primary embeddings for all rich nodes
        ids = [n['id'] for n in rich_nodes if n.get('id')]
        if not ids:
            return rich_nodes[:limit]
        placeholders = ','.join('?' * len(ids))
        rows = conn.execute(
            'SELECT node_id, embedding FROM node_enrichments '
            'WHERE node_id IN (%s) AND vector_type = ?' % placeholders,
            ids + [vector_type]).fetchall()
        emb_by_id = {nid: blob for nid, blob in rows}

        # Score
        scored = []
        unscored = []
        for n in rich_nodes:
            blob = emb_by_id.get(n.get('id'))
            if not blob:
                unscored.append(n)
                continue
            nv = np.frombuffer(blob, dtype=np.float32)
            nnorm = float(np.linalg.norm(nv))
            if nnorm < 1e-9:
                unscored.append(n)
                continue
            score = float(np.dot(query_vec, nv / nnorm))
            scored.append((score, n))

        # Hybrid: top half by relevance, remainder by original (structural) order
        relevance_slots = max(1, limit // 2)
        relevant = [n for _, n in sorted(scored, key=lambda x: -x[0])[:relevance_slots]]
        relevant_ids = {n.get('id') for n in relevant}
        # Remainder preserves the input order (which is the structural sort)
        remainder = [n for n in rich_nodes if n.get('id') not in relevant_ids]
        return (relevant + remainder)[:limit]
    except Exception as e:
        # Loud but not fatal — rerank failure falls back to structural order
        print('[brain] _rerank_by_relevance failed: %s' % e, file=sys.stderr)
        return rich_nodes[:limit]


class BrainRecallMixin:
    """Recall methods for Brain."""

    def get_node(self, node_id_or_ids):
        """Fully assembled node(s): content + metadata + correction chain + connections.

        Accepts a single node_id (str) or a list of node_ids.
        - Single ID → returns one rich node dict, or None if not found.
        - List of IDs → returns dict {node_id: rich_node_dict}. Missing nodes omitted.

        When given a list, uses batched queries (5 queries total instead of N×4).
        This is the canonical way to get a node. For the bare DB row, use NodeDAL.get_naked_node().
        """
        from .dal_metadata import MetadataDAL
        from .scales.s1.surface_contract import correction_enrich

        conn = self.conn
        ndal = NodeDAL(conn)

        # ── Dispatch: single vs batch ──
        single = isinstance(node_id_or_ids, str)
        raw_ids = [node_id_or_ids] if single else list(node_id_or_ids)

        if not raw_ids:
            return None if single else {}

        # Resolve short IDs
        full_ids = []
        for nid in raw_ids:
            full = ndal.resolve_id(nid) if len(str(nid)) < 16 else nid
            if full:
                full_ids.append(full)

        if not full_ids:
            return None if single else {}

        # ── 1. Batch fetch all nodes ──
        ph = ','.join('?' for _ in full_ids)
        cols = [desc[0] for desc in conn.execute('SELECT * FROM nodes LIMIT 0').description]
        rows = conn.execute(
            'SELECT * FROM nodes WHERE id IN (%s)' % ph, full_ids
        ).fetchall()

        nodes = {}
        for row in rows:
            d = dict(zip(cols, row))
            for bf in ('locked', 'archived', 'critical'):
                d[bf] = d.get(bf) == 1
            d['emotion'] = d.get('emotion') or 0
            d['emotion_label'] = d.get('emotion_label') or 'neutral'
            nodes[d['id']] = d

        if not nodes:
            return None if single else {}

        found_ids = list(nodes.keys())
        ph = ','.join('?' for _ in found_ids)

        # ── 2. Batch fetch all metadata (includes situation as of v24) ──
        meta_rows = conn.execute(
            'SELECT node_id, key, value FROM node_metadata_kv WHERE node_id IN (%s)' % ph,
            found_ids
        ).fetchall()
        meta_by_node = {}
        for nid, key, value in meta_rows:
            meta_by_node.setdefault(nid, {})[key] = value
        for nid in found_ids:
            if nid in meta_by_node:
                nodes[nid]['_metadata'] = meta_by_node[nid]
                # Promote situation to top-level node dict — it's a first-class
                # field consumers expect alongside title/content. Previously
                # a separate SELECT from node_enrichments; v24 collapsed it
                # into the same kv fetch above.
                sit = meta_by_node[nid].get('situation')
                if sit:
                    nodes[nid]['situation'] = sit

        # ── 4. Batch corrections (already set-based) ──
        all_ids_for_corrections = set()
        for fid in found_ids:
            all_ids_for_corrections.add(fid)
            all_ids_for_corrections.add(fid[:8])
        corrections = correction_enrich(all_ids_for_corrections, conn)
        for nid in found_ids:
            node_corrs = corrections.get(nid, []) or corrections.get(nid[:8], [])
            if node_corrs:
                for corr in node_corrs:
                    corr_full = ndal.resolve_id(corr['id']) or corr['id']
                    corr_node = ndal.get_naked_node(corr_full)
                    if corr_node:
                        corr['content'] = corr_node.get('content', '')
                        corr['type'] = corr_node.get('type', '')
                nodes[nid]['_corrections'] = node_corrs

        # ── 5. Batch fetch all connections via GraphDAL (v25) ──
        # DAL centralizes: archived=0 default, noise-relation exclusion,
        # direction detection, per-neighbor relation grouping.
        from .dal import GraphDAL
        connections_by_owner = GraphDAL(conn).get_connections_bulk(found_ids)

        for nid in found_ids:
            conns = connections_by_owner.get(nid, [])
            # Sort by aggregate weight, set 'relation' to highest-weight relation for compat
            for c in conns:
                rels = sorted(c['relations'], key=lambda r: r.get('weight', 0), reverse=True)
                c['relations'] = rels
                c['relation'] = rels[0]['relation'] if rels else 'related'
                c['description'] = rels[0]['description'] if rels else ''
            conns.sort(key=lambda x: x.get('weight', 0), reverse=True)
            nodes[nid]['connections'] = conns

        # ── Return ──
        if single:
            return nodes.get(full_ids[0])
        return nodes

    def filter_nodes(self, field: str, include=None, exclude=None,
                     lt=None, gt=None, limit: int = 50,
                     sort_by: str = 'created_at', sort_order: str = 'desc',
                     rich: bool = True,
                     relevance_query: str = None,
                     relevance_pool_multiplier: int = 3,
                     relevance_vector_type: str = '_primary'):
        """Structured query: filter nodes by any structural field.

        rich=True (default): full content, metadata, corrections, connections
        via batched get_node() — 5 queries regardless of N. The consumer is a
        reasoner; richness is the advantage (see node 9b938b91).
        rich=False: skinny shape (id/title/type/confidence/created_at), for
        discovery scans or feeding IDs to other ops.

        relevance_query (Frame Phase 2.5): when provided, the result is
        re-ranked by semantic relevance to the query text. The DAL pulls a
        wider candidate pool (limit * relevance_pool_multiplier), the brain
        embeds the query once and scores each candidate by cosine similarity
        against its `_primary` embedding, then returns the top-`limit`
        candidates as a hybrid: top half by relevance + remainder by the
        original sort order (deduped).

        Lets consumers (Frame, future agentic recall) ask for "nodes of type
        X most relevant to my current arc" without owning the embedding
        mechanics. Cost: 1 embed call (~50ms) + N cosine ops (cheap matrix
        math). When relevance_query is empty/None, behavior is unchanged.
        """
        node_dal = NodeDAL(self.conn)
        # Widen the pool when relevance ranking is requested
        dal_limit = limit * relevance_pool_multiplier if relevance_query else limit
        result = node_dal.filter_nodes(
            field=field, include=include, exclude=exclude,
            lt=lt, gt=gt, limit=dal_limit, sort_by=sort_by, sort_order=sort_order)
        if not rich or 'error' in result or not result.get('nodes'):
            # rich=False or no results — apply pool truncation if relevance was
            # requested but skipped (still want `limit` results, not pool size)
            if relevance_query and result.get('nodes'):
                result['nodes'] = result['nodes'][:limit]
            return result
        ids = [n['id'] for n in result['nodes']]
        rich_map = self.get_node(ids)
        rich_nodes = [rich_map[i] for i in ids if i in rich_map]

        # Optional relevance re-rank
        if relevance_query and rich_nodes:
            rich_nodes = _rerank_by_relevance(
                self.conn, rich_nodes, relevance_query, limit,
                vector_type=relevance_vector_type)

        result['nodes'] = rich_nodes
        return result

    def query_logs(self, source: str = 'all', hours: int = 24,
                   level: str = 'all', hook_name: str = '',
                   limit: int = 50):
        """Query brain logs: errors, debug events, and signals."""
        return self._logs_dal.query_logs(
            source=source, hours=hours, level=level,
            hook_name=hook_name, limit=limit)

    def query_traces(self, scale: str = '', hours: int = 24,
                     event_type: str = '', chain_id: str = '',
                     session_id: str = '', ref_type: str = '',
                     grouped: bool = False, limit: int = 100):
        """Query trace events — the fractal learning loop data.

        Modes:
        - chain_id set: return single chain with all events
        - ref_type set: filter events by ref_type
        - grouped=True + session_id: return chains grouped with nested events
        - default: return flat recent events
        """
        if chain_id:
            return {'chain': self._trace_dal.get_chain(chain_id)}
        if ref_type:
            return {'events': self._trace_dal.get_by_ref_type(
                ref_type=ref_type, scale=scale, hours=hours, limit=limit)}
        if grouped and session_id:
            return {'chains': self._trace_dal.get_chains(
                session_id=session_id, scale=scale, hours=hours, limit=limit)}
        if session_id:
            return {'events': self._trace_dal.get_recent(
                scale=scale, hours=hours, event_type=event_type, limit=limit)}
        return {'events': self._trace_dal.get_recent(
            scale=scale, hours=hours, event_type=event_type, limit=limit)}

    def query_outcomes(self, chain_id: str = '', scale: str = '',
                       hours: int = 168):
        """Query outcome events — the learning signal."""
        return self._trace_dal.get_outcomes(
            chain_id=chain_id, scale=scale, hours=hours)

    def count_traces(self, field: str, scale: str = '', hours: int = 24):
        """Count trace events grouped by a field."""
        return self._trace_dal.count_by(field=field, scale=scale, hours=hours)

    def list_interactions(self):
        """List all registered interactions with latest versions."""
        return self._interaction_dal.list_all()

    def get_interaction(self, name: str, version: int = 0):
        """Get an interaction by name, optionally a specific version."""
        if version:
            return self._interaction_dal.get_version(name, version)
        return self._interaction_dal.get_latest(name)

    def semantic_recall(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Pure embedding-based search (brute-force cosine scan).
        Embed query, compute cosine similarity against all stored embeddings.

        Args:
            query: Query text
            limit: Max results

        Returns:
            List of {'id': str, 'similarity': float} dicts, sorted by similarity
        """
        if not embedder.is_ready():
            return []

        t0 = time.time()
        query_vec = embedder.embed_query(query)
        if not query_vec:
            return []

        # Load all primary embeddings (excluding archived nodes). Filter by
        # active model in SQL so recall doesn't score against stale-model rows.
        _vdal = self._vec_dal
        _active_model = embedder.stats.get('model_name') or ''
        emb_rows = [{'node_id': r['node_id'], 'embedding': r['embedding']}
                    for r in _vdal.get_all_vectors(
                        vector_types=['_primary'],
                        model=_active_model or None)]

        if not emb_rows:
            return []

        # Score every node
        scored = []
        for row in emb_rows:
            node_id, blob = row['node_id'], row['embedding']
            if not blob:
                continue
            similarity = embedder.cosine_similarity(query_vec, blob)
            scored.append({'id': node_id, 'similarity': similarity})

        # Sort and take top-k
        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:limit]

    def backfill_embeddings(self, batch_size: int = 20) -> int:
        """Legacy wrapper — calls backfill_vectors()."""
        result = self.backfill_vectors(batch_size)
        return result.get('total', 0) if isinstance(result, dict) else 0

    def backfill_vectors(self, batch_size: int = 20,
                          node_ids=None) -> dict:
        """Backfill ALL missing vectors for nodes — batched for throughput.

        Primary path: called by embed_queue worker, scoped to a set of
        recently-written node_ids so we don't rescan the graph every tick.
        Safety path: called by S2 Heal during idle, full scan (node_ids=None).

        Each vector type is resolved with one `embed_batch` call instead of N
        sequential embeds — on ORT 1.24.4 + fastembed gives ~5-10× speedup.
        Handles: _primary, _situation, title, high_meta, other_meta, edge_context.

        Returns dict with counts per vector type.
        """
        if not embedder.is_ready():
            return {'error': 'embedder not ready'}

        import time as _time
        _t0 = _time.time()
        print('[backfill_vectors] started (batch_size=%d)' % batch_size, flush=True)

        from .dal import VectorDAL
        from .pipeline_contract import EMBEDDING_GROUPS, EMBEDDING_SKIP_FIELDS, EMBEDDING_FIELD_CHAR_LIMIT
        from .dal_metadata import MetadataDAL

        vdal = self._vec_dal
        mdal = MetadataDAL(self.conn)
        model = embedder.stats.get('model_name', '')
        result = {}

        def _store_batch(items, vector_type):
            """items = list of (node_id, text). One embed_batch + one
            executemany store. Returns count of rows written.

            For vector_type='_situation', the text column is deprecated — kv
            is the single source of truth. Text is used to generate the
            embedding, then discarded (empty string stored). Other vector_types
            keep the text column (used by dashboard, debugging, recall group
            reconstruction)."""
            if not items:
                return 0
            texts = [t for _, t in items]
            blobs = embedder.embed_batch(texts, kind='document')
            store_text = '' if vector_type == '_situation' else None
            rows = [(nid, vector_type,
                     store_text if store_text is not None else text,
                     blob)
                    for (nid, text), blob in zip(items, blobs)]
            stored = vdal.store_batch(rows, model=model)
            if stored:
                self.conn.commit()
            return stored

        # 1. Primary vectors — nodes missing _primary for the active model
        try:
            missing = vdal.find_missing('_primary', batch_size,
                                        model=model, node_ids=node_ids)
            items = [(n['id'], '%s %s' % (n['title'], n['content'])) for n in missing]
            stored = _store_batch(items, '_primary')
            if stored:
                result['_primary'] = stored
        except Exception as e:
            self._log_error('backfill_primary_scan', e, 'scanning for missing primaries')

        # 2. Situation vectors — nodes with situation in kv but no _situation
        # embedding for the active model. Stale-model rows count as missing.
        # Source: node_metadata_kv['situation'] (canonical, single source of
        # truth). The enrichments row stores the embedding only; its text
        # column is deprecated and written empty for _situation.
        try:
            sit_sql = '''
                SELECT n.id, kv.value as situation_text
                FROM nodes n
                JOIN node_metadata_kv kv
                  ON kv.node_id = n.id AND kv.key = 'situation'
                WHERE n.archived = 0
                  AND kv.value IS NOT NULL AND kv.value != ''
                  AND n.id NOT IN (
                      SELECT node_id FROM node_enrichments
                      WHERE vector_type = '_situation'
                        AND embedding IS NOT NULL
                        AND model = ?
                  )
            '''
            sit_params = [model]
            if node_ids:
                ids = list(node_ids)
                sit_sql += ' AND n.id IN (%s)' % ','.join('?' * len(ids))
                sit_params.extend(ids)
            sit_sql += ' LIMIT ?'
            sit_params.append(batch_size)
            sit_missing = self.conn.execute(sit_sql, sit_params).fetchall()
            items = [(nid, txt) for nid, txt in sit_missing if txt]
            stored = _store_batch(items, '_situation')
            if stored:
                result['_situation'] = stored
        except Exception as e:
            self._log_error('backfill_situation_scan', e, 'scanning for missing situations')

        # 3. Group vectors — title, high_meta, other_meta, edge_context, field cohort
        # Derive which fields live in node_metadata_kv (vs the nodes table) so
        # find_missing can pre-filter to nodes that actually have at least one
        # source field populated. Without this filter, the top-N batch fills
        # with nodes lacking the field, the backfill skips them, and they stay
        # in the missing pool while older nodes-WITH-the-field never reach the
        # front (last_accessed DESC ordering). Result: incomplete coverage even
        # after many passes.
        from .contract import STRUCTURAL_FIELDS as _STRUCT_FIELDS
        nodes_table_fields = set(_STRUCT_FIELDS.keys())

        for group_name, group_config in EMBEDDING_GROUPS.items():
            vector_type = group_config.get('vector_type')
            if not vector_type or vector_type == '_primary':
                continue  # blend is the primary, already handled

            # Identify which fields in this group are kv-stored. Skip:
            #  - fields on the nodes table (always present, no filter needed)
            #  - special markers like `_emergent` and `_edge_descriptions`
            kv_source_keys = [
                f for f in group_config.get('fields', [])
                if f not in nodes_table_fields and not f.startswith('_')
            ]

            try:
                missing = vdal.find_missing(
                    vector_type, batch_size, model=model, node_ids=node_ids,
                    source_kv_keys=kv_source_keys if kv_source_keys else None)
                if not missing:
                    continue

                items = []
                for node in missing:
                    try:
                        field_values = {'title': node['title'], 'content': node['content']}
                        kv = mdal.get(node['id'])
                        for k, v in kv.items():
                            if k not in EMBEDDING_SKIP_FIELDS and v and str(v).strip():
                                field_values[k] = str(v)

                        parts = []
                        for field in group_config.get('fields', []):
                            if field == '_emergent':
                                continue  # Emergent: any KV field not in other groups
                            val = field_values.get(field)
                            if val and val.strip():
                                parts.append(val[:EMBEDDING_FIELD_CHAR_LIMIT])
                        if not parts:
                            continue
                        items.append((node['id'], '. '.join(parts)))
                    except Exception as e:
                        self._log_error('backfill_group_%s' % vector_type, e,
                                        'node %s' % node['id'][:8])

                stored = _store_batch(items, vector_type)
                if stored:
                    result[vector_type] = stored
            except Exception as e:
                self._log_error('backfill_%s_scan' % vector_type, e,
                                'scanning for missing %s vectors' % vector_type)

        elapsed = _time.time() - _t0
        total = sum(v for v in result.values() if isinstance(v, int))
        if total > 0:
            parts = ['%s:%d' % (k, v) for k, v in result.items() if isinstance(v, int) and v > 0]
            print('[backfill_vectors] %d vectors in %.1fs (%s)' % (total, elapsed, ', '.join(parts)), flush=True)
        else:
            print('[backfill_vectors] no missing vectors (%.1fs)' % elapsed, flush=True)

        return result

    def _keyword_recall(self, query: str, filter: Optional[Dict[str, Any]] = None, limit: int = 20,
                        offset: int = 0, include_archived: bool = False, min_recency: float = 0,
                        project: Optional[str] = None, session_id: Optional[str] = None,
                        _skip_log: bool = False) -> Dict[str, Any]:
        """INTERNAL: TF-IDF keyword recall. Used by recall() for keyword blending.
        Do NOT call directly — use recall() (embeddings + graph traversal) instead.

        Args:
            query: Search query
            filter: Dict filter (same format as recall())
            limit: Max results to return
            offset: Pagination offset
            include_archived: Include archived nodes
            min_recency: Minimum recency score threshold
            project: Filter to specific project
            session_id: Optional session ID for logging

        Returns:
            Dict with results (list of nodes), recall_ref, intent
        """
        limit = min(limit, MAX_PAGE_SIZE)

        expanded_query = query

        # v5 Step 0: Intent detection
        intent_data = self._classify_intent(query)
        intent = intent_data['intent']
        type_boosts = intent_data['typeBoosts']
        temporal_filter = intent_data['temporalFilter']

        # Step 1: Keyword search for seeds
        seeds = self._search_keywords(expanded_query, 30)

        all_seeds = {}
        for seed in seeds:
            if seed['id'] not in all_seeds:
                all_seeds[seed['id']] = seed

        # v5: Also find seeds via TF-IDF
        tfidf_query_terms = self._tfidf_tokenize(query)
        if tfidf_query_terms:
            try:
                unique_terms = list(set(tfidf_query_terms))
                tfidf_dal = TfIdfDAL(self.conn)
                tfidf_node_ids = tfidf_dal.get_nodes_matching_terms(unique_terms)
                _node_dal = NodeDAL(self.conn)
                for nid in tfidf_node_ids[:50]:
                    if nid not in all_seeds:
                        node = _node_dal.get_naked_node(nid)
                        if node and not node.get('archived'):
                            all_seeds[nid] = node
            except Exception as _e:
                self._log_error("recall", _e, "fetching seed node details from database")

        if not all_seeds:
            # Return recent nodes if no seeds found
            return {
                'results': _apply_filter(self._get_recent(limit), filter, self.conn),
                'intent': intent
            }

        # Step 1b: Compute direct keyword match strength per seed
        query_terms = [w.replace('[^a-z0-9]', '', ) for w in query.lower().split()
                       if len(w) > 2]
        query_terms = [w for w in query_terms if w]

        direct_match_scores = {}
        for seed_id, seed in all_seeds.items():
            kw = (seed.get('keywords') or '').lower()
            title = (seed.get('title') or '').lower()
            content = (seed.get('content') or '').lower()
            match_count = 0
            for term in query_terms:
                if term in kw or term in title or term in content:
                    match_count += 1
            direct_match_scores[seed_id] = (match_count / len(query_terms)) if query_terms else 0

        # Step 2: Score seeds directly (spread_activation removed 2026-04-14)
        activated = list(all_seeds.values())
        activated_ids = [n['id'] for n in activated]
        tfidf_scores = self._batch_tfidf_scores(tfidf_query_terms, activated_ids)

        now_ms = time.time() * 1000  # milliseconds

        scored = []
        for node in activated:
            # Keyword-based relevance from direct match strength
            keyword_relevance = direct_match_scores.get(node['id'], 0)

            direct_match = direct_match_scores.get(node['id'], 0)
            if direct_match == 0 and query_terms:
                nkw = (node.get('keywords') or '').lower()
                ntitle = (node.get('title') or '').lower()
                ncontent = (node.get('content') or '').lower()
                mc = 0
                for term in query_terms:
                    if term in nkw or term in ntitle or term in ncontent:
                        mc += 1
                direct_match = mc / len(query_terms)

            if direct_match > 0:
                keyword_relevance = min(1.0, keyword_relevance + direct_match * 0.5)

            # v5: TF-IDF semantic relevance
            semantic_score = tfidf_scores.get(node['id'], 0)

            # v5: Blend keyword and semantic scores (tunable blend)
            blend = self._get_tunable('embedding_blend', {
                'embedding': EMBEDDING_PRIMARY_WEIGHT, 'keyword': KEYWORD_FALLBACK_WEIGHT
            })
            emb_w = blend.get('embedding', EMBEDDING_PRIMARY_WEIGHT) if isinstance(blend, dict) else EMBEDDING_PRIMARY_WEIGHT
            kw_w = blend.get('keyword', KEYWORD_FALLBACK_WEIGHT) if isinstance(blend, dict) else KEYWORD_FALLBACK_WEIGHT
            relevance = (kw_w * keyword_relevance + emb_w * semantic_score)

            # v4: Hub dampening (nodes with 20+ connections get reduced relevance)
            _graph_dal = GraphDAL(self.conn)
            edge_count = _graph_dal.count_node_edges(node['id'], min_weight=0)
            hub = self._get_tunable('hub_dampening', {'threshold': 40, 'penalty': 0.5})
            hub_threshold = hub.get('threshold', 40) if isinstance(hub, dict) else 40
            if edge_count > hub_threshold:
                relevance *= hub_threshold / edge_count

            # v4: Type dampening
            if node.get('type') in ('project', 'person'):
                relevance *= 0.5

            # v5: Intent-based type boosting
            type_boost = type_boosts.get(node.get('type'), 1.0)
            relevance *= type_boost

            # v5.2: Critical node boost
            if node.get('critical'):
                relevance *= CRITICAL_BOOST

            # v10: Keyword path produces raw relevance score.
            # Unified scoring is applied ONCE in STEP 6 of recall() after
            # embedding and keyword scores merge. Applying it here would
            # double-modulate. Keep keyword path output as raw relevance.
            emotion_intensity = abs(node.get('emotion', 0))
            effective = relevance

            scored.append({
                **node,
                'relevance_score': relevance,
                'semantic_score': semantic_score,
                'keyword_relevance': keyword_relevance,
                'emotion_intensity': emotion_intensity,
                'effective_activation': effective
            })

        # Step 4: Filter
        filtered = scored
        if not include_archived:
            filtered = [n for n in filtered if not n.get('archived')]
        filtered = _apply_filter(filtered, filter, self.conn)
        if min_recency > 0:
            filtered = [n for n in filtered if n.get('recency_score', 0) >= min_recency]

        # v5: Project filter
        if project:
            filtered.sort(key=lambda n: (1 if n.get('project') == project else 0, -n.get('effective_activation', 0)))

        # v5: Temporal filter
        if temporal_filter:
            after = temporal_filter.get('after')
            before = temporal_filter.get('before')
            filtered = [n for n in filtered if self._matches_temporal_filter(n.get('created_at'), after, before)]

        # Step 5: Sort by effective activation (if no project filter)
        if not project:
            filtered.sort(key=lambda n: -n.get('effective_activation', 0))

        # Step 6: Pagination
        page = filtered[offset:offset + limit]

        # Step 7: Mark accessed + Hebbian
        if not session_id:
            session_id = self.session_id

        for node in page:
            self._mark_accessed(node['id'], session_id)

        # v10: Hebbian co_accessed edge creation DISABLED.
        # Previously: every recall created co_accessed edges between all top-25 results.
        # This produced 71K noise edges (90% of graph) that destroyed topology.
        # Biology: neurons that fire together wire together — but our "firing together"
        # was just "scored similarly on cosine," not meaningful co-activation.
        #
        # Re-enable when: surface-selected node IDs are available in hook_post_response_track.
        # Then: only strengthen between nodes the surfacer selected AND the assistant used.
        # That's real co-activation — two memories genuinely contributing to the same response.

        # v4: Auto-instrument (skipped when called from recall
        # or hooks — they log via the precision module instead)
        returned_ids = [n['id'] for n in page]
        # recall_log writes REMOVED 2026-04-05 — S1 traces capture all recall data.

        # v6: Attach reasoning chains when intent is reasoning_chain
        reasoning_chains = []
        if intent == 'reasoning_chain':
            # 1. Pull chains for decision nodes in results
            decision_nodes = [n for n in page if n.get('type') == 'decision']
            for dn in decision_nodes:
                # Note: reasoning methods not yet implemented, skipping for now
                pass

        # recall_log writes removed 2026-04-05 — traces are source of truth
        result = {
            'results': page,
            'intent': intent,
        }

        if reasoning_chains:
            result['reasoning_chains'] = reasoning_chains

        return result

    def recall(self, query: str, filter: Optional[Dict[str, Any]] = None,
               limit: int = 20, offset: int = 0,
               include_archived: bool = False,
               min_recency: float = 0, project: Optional[str] = None,
               session_id: Optional[str] = None,
               situation_vec=None, source: str = 'unknown') -> Dict[str, Any]:
        """Recall: embeddings + 3-degree graph traversal + keyword blending + situation matching.

        Args:
            query: Search query
            filter: Dict filter on node/metadata fields. Examples:
                {"type": {"in": ["moment", "reflection"]}}
                {"anchor_raw_quote": {"exists": True}}
                {"content": {"contains": "Anchor"}}
                {"confidence": {"gte": 0.9}}
            limit: Max results
            offset: Pagination offset
            include_archived: Include archived
            min_recency: Min recency threshold
            project: Optional project filter
            session_id: Optional session ID

        Returns:
            Dict with results, recall_ref, _embedding_stats, intent, _recall_mode

        Two layers of dedup (2026-05-08):
        1. Result cache (5s TTL) — repeat identical recalls return cached
           result without re-running. Deepcopy on read so callers are
           isolated.
        2. Single-flight gate — concurrent identical recalls share work:
           one becomes leader, others wait for its result.

        Both keyed by (query, filter, limit, offset, include_archived,
        min_recency, project, session_id, situation_vec). session_id is
        in the key because synaptic fatigue is per-session.

        Replaces the dispatch + brain.pre_edit caches: every recall caller
        (pre_edit, pre_bash_safety, hook_recall, MCP) now benefits.
        """
        # Build dedup key from result-affecting params.
        try:
            filter_key = json.dumps(filter, sort_keys=True, default=str) if filter else None
            sit_key = bytes(situation_vec) if situation_vec is not None else None
            dedup_key = (
                query, int(min(limit, MAX_PAGE_SIZE)), int(offset),
                bool(include_archived), float(min_recency or 0),
                project, session_id, filter_key, sit_key,
            )
        except Exception:
            # If key construction fails, skip dedup — better to do
            # redundant work than crash the recall path.
            dedup_key = None

        # Layer 1: result cache fast path
        if dedup_key is not None:
            cached = self._recall_cache_get(dedup_key)
            if cached is not None:
                return cached

        # Layer 2: single-flight gate
        if dedup_key is not None:
            inflight, leader = self._recall_inflight_acquire(dedup_key)
            if not leader:
                # Wait for leader. .result() blocks until set_result/
                # set_exception fires; deepcopy keeps callers isolated.
                import copy as _copy
                return _copy.deepcopy(inflight.result())

            # Leader path — compute, populate cache, fan out, clean up.
            try:
                result = self._run_recall_with_commit(
                    query=query, filter=filter, limit=limit, offset=offset,
                    include_archived=include_archived, min_recency=min_recency,
                    project=project, session_id=session_id,
                    situation_vec=situation_vec, source=source)
                self._recall_cache_put(dedup_key, result)
                inflight.set_result(result)
                return result
            except Exception as e:
                inflight.set_exception(e)
                raise
            finally:
                self._recall_inflight_release(dedup_key, inflight)

        # Dedup disabled (key construction failed) — fall through.
        return self._run_recall_with_commit(
            query=query, filter=filter, limit=limit, offset=offset,
            include_archived=include_archived, min_recency=min_recency,
            project=project, session_id=session_id,
            situation_vec=situation_vec, source=source)

    def _run_recall_with_commit(self, **kwargs):
        """Wrapper around _recall_impl that commits ONCE at the end of any
        path (success, early return, or exception).

        Why this exists: _mark_accessed accumulates UPDATEs without
        committing (the commit storm was the root cause of spinning CPU
        under concurrent recall load — see commit log 2026-05-08). The
        single end-of-recall commit holds the SQLite write lock briefly
        once instead of N times. Try/finally ensures we don't leak an
        open transaction on early returns or exceptions.
        """
        try:
            return self._recall_impl(**kwargs)
        finally:
            try:
                self.conn.commit()
            except Exception as _e:
                self._log_error('recall_final_commit', _e,
                                'committing post-recall writes')

    # ─── recall result cache (5s TTL) ─────────────────────────────────

    def _recall_cache_get(self, key):
        """Return cached recall result if present + within TTL, else None.

        Returns deepcopy so callers can mutate freely.
        """
        import time as _time
        import copy as _copy
        if not hasattr(self, '_recall_cache'):
            return None
        now = _time.time()
        with self._recall_cache_lock:
            entry = self._recall_cache.get(key)
            if entry is None:
                return None
            result, ts = entry
            if now - ts > _RECALL_CACHE_TTL_S:
                del self._recall_cache[key]
                return None
            return _copy.deepcopy(result)

    def _recall_cache_put(self, key, result) -> None:
        """Cache a recall result with current timestamp. Lazy-init + LRU cap."""
        import time as _time
        import threading
        if not hasattr(self, '_recall_cache'):
            self._recall_cache = {}
            self._recall_cache_lock = threading.Lock()
        now = _time.time()
        with self._recall_cache_lock:
            self._recall_cache[key] = (result, now)
            if len(self._recall_cache) > _RECALL_CACHE_MAX_ENTRIES:
                oldest = min(self._recall_cache,
                             key=lambda k: self._recall_cache[k][1])
                del self._recall_cache[oldest]

    def _recall_inflight_acquire(self, key):
        """Return (future, is_leader). Lazy-inits inflight state on first call.

        is_leader=True → caller computes the result + sets the future.
        is_leader=False → caller waits on the existing future for the result.
        """
        import threading
        from concurrent.futures import Future
        if not hasattr(self, '_recall_inflight'):
            self._recall_inflight = {}
            self._recall_inflight_lock = threading.Lock()
        with self._recall_inflight_lock:
            existing = self._recall_inflight.get(key)
            if existing is not None:
                return existing, False
            fut = Future()
            self._recall_inflight[key] = fut
            return fut, True

    def _recall_inflight_release(self, key, fut) -> None:
        """Drop the inflight slot. Defensive: ensure future is set so any
        waiters that arrived AFTER set_result/set_exception don't deadlock
        (shouldn't happen — they'd see existing fut and wait — but the
        cost of guarding is one cheap check)."""
        with self._recall_inflight_lock:
            # Only delete if it's still the same future (defensive against
            # a different recall having taken the slot, which shouldn't
            # be possible but guard anyway).
            if self._recall_inflight.get(key) is fut:
                del self._recall_inflight[key]
        if not fut.done():
            fut.set_exception(RuntimeError(
                'recall leader exited without setting result'))

    def _recall_impl(self, query: str, filter=None, limit: int = 20,
                     offset: int = 0, include_archived: bool = False,
                     min_recency: float = 0, project=None,
                     session_id=None, situation_vec=None,
                     source: str = 'unknown') -> Dict[str, Any]:
        """Actual recall implementation — hot path. Single-flight wrapper
        in recall() ensures only one of these runs per (query, scope) at
        a time across the daemon."""
        t0 = time.time()
        limit = min(limit, MAX_PAGE_SIZE)

        # ── FALLBACK: If embedder not ready, degrade to keyword-only ──
        if not embedder.is_ready():
            result = self._keyword_recall(query, filter, limit, offset, include_archived,
                               min_recency, project, session_id, _skip_log=True)
            result['_recall_mode'] = 'keyword_only_DEGRADED'
            result['_embedding_stats'] = {
                'embedder_ready': False,
                'embedder_status': embedder.get_model_status(),
                'warning': 'Recall is keyword-only. Semantic understanding disabled.',
            }
            print(f'[brain] WARNING: keyword-only recall (embedder not ready)', file=sys.stderr)
            # recall_log writes REMOVED 2026-04-05 — S1 traces capture all recall data.
            # recall_ref no longer returned (consumers use stop counter instead).
            return result

        # ── PRIMARY PATH: Embeddings-first ──

        expanded_query = query
        _active_model = embedder.stats.get('model_name') or None

        # STEP 1: Embed the query
        try:
            query_vec = embedder.embed_query(expanded_query)
            if not query_vec:
                # Embedding failed for this query — fall back
                result = self._keyword_recall(query, filter, limit, offset, include_archived,
                                   min_recency, project, session_id)
                result['_recall_mode'] = 'keyword_only_DEGRADED'
                return result
        except Exception as e:
            result = self._keyword_recall(query, filter, limit, offset, include_archived,
                               min_recency, project, session_id)
            result['_recall_mode'] = 'keyword_only_DEGRADED'
            return result

        # Lexical bridge alternates — populated conditionally AFTER primary
        # cosine completes (see post-STEP-3 expansion gate). Empty here so
        # primary cosine runs unmodified. Both lists feed STEP 4.5 so
        # alternates are searched via BOTH cosine (alternate_vecs) AND FTS5
        # keyword (alternate_strings) — the latter catches lexical matches
        # cosine collapses on (e.g. "scratch grains" vs "feed").
        alternate_vecs = []
        alternate_strings = []

        # STEP 2: Intent classification — DEPRECATED 2026-04-12.
        # Regex patterns fire on 12% of queries and miscalibrate scores when they do
        # (how_to boosted irrelevant rules to 0.943). Replaced by z-score contrastive
        # scoring which naturally handles type relevance through per-node statistics.
        # Type boosts removed from STEP 6 scoring.
        intent = 'general'
        type_boosts = {}

        # STEP 2.5: Wire situation matching — query IS the situation context.
        # 1085 nodes have situation embeddings describing WHEN they're relevant.
        # Scoring logic exists in STEP 3.5b but situation_vec was never passed.
        if situation_vec is None:
            situation_vec = query_vec

        # STEP 3: Brute-force cosine similarity against ALL stored embeddings
        # This is the core change: embeddings drive retrieval, not keywords.
        # For 600 nodes this is fast (<50ms). At 10k+ nodes, switch to sqlite-vec.
        # v8.7: Primary scores stored separately. Enrichments boost but don't replace (ENRICHMENT_CAP).
        embedding_scores = {}  # node_id → final cosine score (primary + capped enrichment)
        primary_scores = {}    # node_id → primary embedding similarity (content match)
        enrichment_hits = {}   # node_id → vector_type that matched best (for telemetry)
        node_personal_data = {}  # node_id → (personal, personal_context) for pre-sort penalty
        node_confidence = {}    # node_id → confidence (0-1, None=default)
        nodes_with_embeddings = 0
        nodes_without_embeddings = 0

        # v9: Brain size for retrieval stats (computed once)
        try:
            _brain_size = self.conn.execute("SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
        except Exception:
            _brain_size = 0

        # Pre-compute query terms for contextual qualifier matching (applied in STEP 6)
        _query_terms_set = set(query.lower().split()) if query else set()

        try:
            node_critical = {}  # node_id → critical flag
            node_titles = {}    # node_id → title (for title-match boost)
            node_types = {}     # node_id → type (for vocab detection)
            _vec_dal = self._vec_dal
            # Extract type filter from dict filter for embedding scan pre-filter
            _types_filter = None
            if filter and 'type' in filter and 'in' in filter['type']:
                _types_filter = filter['type']['in']
            emb_rows = _vec_dal.get_all_with_context(
                exclude_archived=not include_archived,
                types=_types_filter, project=project,
                model=_active_model)
            # Per-node data collected here for unified_score() in STEP 6.
            # created_at/emotion/access_count feed the modulator formula.
            node_created_at = {}    # node_id → ISO timestamp (for freshness)
            node_emotion = {}       # node_id → float (for emotional amplification)
            node_access_count = {}  # node_id → int (for hub penalty)

            for row in emb_rows:
                node_id = row['node_id']
                blob = row['embedding']
                node_personal_data[node_id] = (row['personal'], row['personal_context'])
                node_confidence[node_id] = row['confidence']
                node_critical[node_id] = row['critical']
                node_titles[node_id] = row['title']
                node_types[node_id] = row['type']
                node_created_at[node_id] = row.get('created_at')
                node_emotion[node_id] = row.get('emotion', 0)
                node_access_count[node_id] = row.get('access_count', 0)
                if blob:
                    sim = embedder.cosine_similarity(query_vec, blob)
                    # Lexical bridge: take max cosine across all phrasings
                    # (primary + Haiku-expanded alternates). This is the
                    # mechanism that makes "uncle's birthday party" reach
                    # "niece's birthday party" nodes — Haiku generated
                    # the contrastive phrasing, embedding bridged the gap.
                    for _av in alternate_vecs:
                        _alt_sim = embedder.cosine_similarity(_av, blob)
                        if _alt_sim > sim:
                            sim = _alt_sim

                    # v11: Z-score contrastive normalization (BEFORE fatigue).
                    # Measures SURPRISE: how unusual is this cosine for this node?
                    # Hub nodes (high mean) get flattened. Specialized nodes
                    # (low mean, high cosine on this query) get amplified.
                    # Stats precomputed by scripts/compute_zscore_stats.py.
                    # v11 NOTE: Z-score contrastive normalization was tested here
                    # (2026-04-12) but reverted. Z-score is a monotonic per-node
                    # transform — it changes scores but NOT the ranking within a
                    # single query. It only helps when nodes from different queries
                    # compete in a global pool (eval artifact, not production).
                    # The real lever is V5 enrichment coverage (S2 enrichment unit).

                    # v10: Synaptic fatigue — AFTER z-score normalization.
                    # Dampens the surprise-score for nodes recalled repeatedly
                    # this session. Resets between sessions.
                    # K (fatigue resistance) scales with structural degree:
                    #   Hub (30 edges): K=2.5, fatigues fast
                    #   Peripheral (3 edges): K=7.7, fatigues slow
                    #   New node (0 edges): K=10, barely fatigues
                    if not hasattr(self, '_session_fatigue'):
                        _fatigue_sid = session_id or self.session_id
                        try:
                            _ctx = self.get_or_create_session(_fatigue_sid) if _fatigue_sid else None
                            self._session_fatigue = _ctx.fatigue if _ctx else {}
                            self._fatigue_ctx = _ctx
                        except Exception:
                            self._session_fatigue = {}
                            self._fatigue_ctx = None
                    if not hasattr(self, '_structural_degree_cache'):
                        self._structural_degree_cache = {}
                        try:
                            for _row in self.conn.execute("""
                                SELECT node_id, COUNT(*) FROM (
                                    SELECT e.source_id as node_id FROM edges e
                                    JOIN edge_relations er ON er.edge_id = e.edge_id
                                    WHERE er.relation NOT IN ('co_accessed','emergent_bridge')
                                    UNION ALL
                                    SELECT e.target_id as node_id FROM edges e
                                    JOIN edge_relations er ON er.edge_id = e.edge_id
                                    WHERE er.relation NOT IN ('co_accessed','emergent_bridge')
                                ) GROUP BY node_id"""):
                                self._structural_degree_cache[_row[0]] = \
                                    self._structural_degree_cache.get(_row[0], 0) + _row[1]
                        except Exception as _e:
                            self._log_error('fatigue_degree_cache', _e, 'building structural degree cache')

                    _fatigue_count = self._session_fatigue.get(node_id, 0)
                    if _fatigue_count > 0:
                        _degree = self._structural_degree_cache.get(node_id, 0)
                        _K = 10.0 / (1.0 + _degree / 10.0)
                        _fatigue = _fatigue_count / (_fatigue_count + _K)
                        sim *= (1.0 - _fatigue)

                    embedding_scores[node_id] = sim
                    primary_scores[node_id] = sim
                    enrichment_hits[node_id] = 'primary'
                    nodes_with_embeddings += 1

            # STEP 3.1: Conditional lexical bridge — Haiku query expansion
            # gated on cosine flatness. The cost of expansion is dominated
            # by the Haiku call (~800ms). Most queries have a clear winner
            # in primary cosine and don't need it. Only when scores are
            # genuinely flat does the lexical bridge add value.
            #
            # Modes (BRAIN_QUERY_EXPANSION env var):
            #   off (default)     — no expansion, no Haiku call
            #   on_flat           — expand only when top1 - top10 < gate
            #   on                — expand always (research / eval; expensive)
            #
            # When expansion fires: re-iterate emb_rows with alternate
            # vectors, taking max cosine. primary_scores updated in place
            # so STEP 3.5 sees the boosted scores.
            import os as _os
            _expansion_mode = _os.environ.get(
                'BRAIN_QUERY_EXPANSION', 'off').lower()
            _do_expand = False
            if _expansion_mode == 'on':
                _do_expand = True
            elif _expansion_mode == 'on_flat' and primary_scores:
                _sorted = sorted(primary_scores.values(), reverse=True)
                if len(_sorted) >= 10:
                    _spread = _sorted[0] - _sorted[9]
                    _gate_str = _os.environ.get('BRAIN_EXPANSION_GATE', '0.05')
                    try:
                        _gate = float(_gate_str)
                    except Exception:
                        _gate = 0.05
                    _do_expand = _spread < _gate
                    # Telemetry: log every gate decision so we can tune.
                    # Cheap (one stderr line per recall, no Haiku call).
                    print('[recall] on_flat gate: top1=%.3f top10=%.3f '
                          'spread=%.3f gate=%.3f → %s' % (
                            _sorted[0], _sorted[9], _spread, _gate,
                            'expand' if _do_expand else 'skip'),
                          file=sys.stderr)
                else:
                    _do_expand = True  # too few candidates → expand

            if _do_expand:
                try:
                    _alts = _expand_query_via_haiku(expanded_query)
                    alternate_strings = list(_alts)  # for FTS5 path in STEP 4.5
                    for _alt in _alts:
                        _av = embedder.embed_query(_alt)
                        if _av:
                            alternate_vecs.append(_av)
                    if _alts:
                        print('[recall] query-expansion (mode=%s): %r → %s' % (
                            _expansion_mode, expanded_query[:60],
                            [a[:50] for a in _alts]), file=sys.stderr)
                except Exception as _expand_e:
                    self._log_error('query_expansion', _expand_e,
                                    'Haiku query expansion failed — '
                                    'proceeding with primary query only')

                # Re-iterate primary cosine with alternates, take max
                if alternate_vecs:
                    for row in emb_rows:
                        nid = row['node_id']
                        blob = row['embedding']
                        if blob:
                            current = primary_scores.get(nid, 0)
                            for _av in alternate_vecs:
                                _alt_sim = embedder.cosine_similarity(_av, blob)
                                if _alt_sim > current:
                                    current = _alt_sim
                            if current != primary_scores.get(nid, 0):
                                primary_scores[nid] = current
                                embedding_scores[nid] = current

            # v10: STEP 3.5: Z-weighted multi-vector scoring
            # Each node has 2-4 group vectors (title, high_meta, other_meta) in
            # node_enrichments table. Primary is _primary, groups are title/high_meta/etc.
            # Old enrichment types (question, anchor, bridge, keywords) still exist
            # and participate with the other_meta weight.
            #
            # Scoring: weight × cosine_sim per vector, average top 2.
            # Requires 2 vectors to agree — prevents noisy single-field matches.
            # Tested 2026-04-02: +20pts R@8, +22pts R@25 vs old enrichment cap.
            # See pipeline_contract.EMBEDDING_GROUPS for weight definitions.
            try:
                from .pipeline_contract import get_group_weight, EMBEDDING_GROUPS
            except ImportError:
                from pipeline_contract import get_group_weight, EMBEDDING_GROUPS

            # Group vector types that get their own weight (from contract)
            _known_group_types = {g['vector_type'] for g in EMBEDDING_GROUPS.values()
                                  if g.get('vector_type') != '_primary'}
            _other_meta_weight = EMBEDDING_GROUPS['other_meta']['weight']
            _blend_weight = EMBEDDING_GROUPS['blend']['weight']

            enrichment_count = 0
            enrichment_used = 0
            # Collect ALL weighted scores per node: [(weighted_score, vector_type), ...]
            node_vector_scores = {}  # node_id → list of (weighted_sim, type)

            # Primary (blend) scores — already computed, add with blend weight
            for nid, prim_sim in primary_scores.items():
                if nid not in node_vector_scores:
                    node_vector_scores[nid] = []
                node_vector_scores[nid].append((_blend_weight * prim_sim, '_primary'))

            try:
                _enrich_rows = _vec_dal.get_all_vectors(
                    exclude_archived=not include_archived,
                    model=_active_model or None)
                for erow in _enrich_rows:
                    e_node_id = erow['node_id']
                    e_type = erow['vector_type']
                    e_blob = erow['embedding']
                    if not e_blob:
                        continue
                    enrichment_count += 1
                    # Skip if node not in our filtered set
                    if e_node_id not in node_types:
                        continue

                    e_sim = embedder.cosine_similarity(query_vec, e_blob)
                    # Lexical bridge: max over query + alternate phrasings
                    for _av in alternate_vecs:
                        _alt = embedder.cosine_similarity(_av, e_blob)
                        if _alt > e_sim:
                            e_sim = _alt

                    # Get z-index weight: known group types get their contract weight,
                    # old enrichment types (question, anchor, etc.) get other_meta weight
                    if e_type in _known_group_types:
                        weight = get_group_weight(e_type)
                    else:
                        weight = _other_meta_weight

                    if e_node_id not in node_vector_scores:
                        node_vector_scores[e_node_id] = []
                    node_vector_scores[e_node_id].append((weight * e_sim, e_type))

                # Z-weighted top2-avg: for each node, sort weighted scores, avg top 2
                for nid, scores in node_vector_scores.items():
                    if len(scores) < 1:
                        continue
                    scores.sort(reverse=True)
                    if len(scores) >= 2:
                        final = (scores[0][0] + scores[1][0]) / 2
                    else:
                        final = scores[0][0]

                    # Update embedding_scores if this beats the current
                    if final > embedding_scores.get(nid, 0):
                        embedding_scores[nid] = final
                        enrichment_hits[nid] = scores[0][1]  # best vector type
                        enrichment_used += 1

            except Exception as e:
                if 'no such table' not in str(e):
                    self._log_error("recall_enrichment_scan", e, "STEP 3.5 z-weighted scoring")

            # v9.2: Apply fatigue AFTER z-weighted scoring (not before).
            # Fatigue was previously applied to primary sim only, but STEP 3.5
            # could overwrite with unfatigued z-weighted scores. Now fatigue
            # dampens the final embedding score regardless of which vector won.
            if hasattr(self, '_session_fatigue') and self._session_fatigue:
                for nid in list(embedding_scores.keys()):
                    _fc = self._session_fatigue.get(nid, 0)
                    if _fc > 0:
                        _deg = self._structural_degree_cache.get(nid, 0) if hasattr(self, '_structural_degree_cache') else 0
                        _K = 10.0 / (1.0 + _deg / 10.0)
                        _fat = _fc / (_fc + _K)
                        embedding_scores[nid] *= (1.0 - _fat)

        except Exception as e:
            print(f'[brain] Embedding scan error: {e}', file=sys.stderr)

        # STEP 3.5: Situation scan — boost nodes whose situation matches current context
        situation_scores = {}
        if situation_vec:
            try:
                sit_rows = _vec_dal.get_all_situations(model=_active_model)
                for row in sit_rows:
                    nid = row['node_id']
                    svec = struct.unpack('%df' % (len(row['situation_embedding']) // 4),
                                         row['situation_embedding'])
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(situation_vec, svec))
                    mag_a = sum(a * a for a in situation_vec) ** 0.5
                    mag_b = sum(b * b for b in svec) ** 0.5
                    if mag_a > 0 and mag_b > 0:
                        sim = dot / (mag_a * mag_b)
                        if sim >= SITUATION_THRESHOLD:
                            situation_scores[nid] = sim
            except Exception as e:
                self._log_error("recall_situation_scan", e, "situation scan failed")

        # STEP 4: Also run keyword recall to catch nodes WITHOUT embeddings
        # and to get keyword precision scores for exact-match tiebreaking
        # _skip_log=True: precision module handles logging via hooks, not here.
        keyword_result = self._keyword_recall(query, filter, limit * 3, offset, include_archived,
                                    min_recency, project, session_id, _skip_log=True)
        keyword_scores = {}  # node_id → keyword_effective_activation
        keyword_nodes = {}   # node_id → full node dict
        for node in keyword_result.get('results', []):
            nid = node['id']
            keyword_scores[nid] = node.get('effective_activation', 0)
            keyword_nodes[nid] = node
            if nid not in embedding_scores:
                nodes_without_embeddings += 1

        # STEP 4.5: FTS5 independent candidate net
        # FTS5 catches nodes where words match but embeddings didn't connect.
        # These go to the surfacer as fts5_only candidates (no blended score needed).
        # Lexical bridge: when alternate_strings is non-empty (gate fired),
        # also run FTS5 for each alternate. Cheap (~10ms per query) and
        # specifically catches token-level matches the cosine collapses on.
        fts5_only_ids = set()
        fts5_all_ids = set()
        try:
            fts5_dal = Fts5DAL(self.conn)
            _fts5_queries = [query] + alternate_strings
            for _fq in _fts5_queries:
                if not _fq or not _fq.strip():
                    continue
                fts5_hits = fts5_dal.search(_fq, FTS5_SEARCH_LIMIT)
                fts5_all_ids.update(fts5_hits)
                for nid in fts5_hits:
                    if nid not in embedding_scores and nid not in keyword_scores:
                        fts5_only_ids.add(nid)
                        if len(fts5_only_ids) >= FTS5_CANDIDATE_LIMIT:
                            break
                if len(fts5_only_ids) >= FTS5_CANDIDATE_LIMIT:
                    break
            if alternate_strings:
                print('[recall] FTS5 lexical bridge: %d alternates → %d total fts5_only ids' % (
                    len(alternate_strings), len(fts5_only_ids)), file=sys.stderr)
        except Exception as e:
            self._log_error('recall_fts5', e, 'FTS5 candidate search')

        # STEP 5: Build unified candidate set (all nodes seen by any path)
        all_candidate_ids = set(embedding_scores.keys()) | set(keyword_scores.keys()) | fts5_only_ids

        # STEP 6: Score each candidate — embeddings primary, keywords fallback
        scored_results = []
        for nid in all_candidate_ids:
            emb_score = embedding_scores.get(nid, 0)
            kw_score = keyword_scores.get(nid, 0)

            # Determine discovery source and compute blended score
            if nid in fts5_only_ids:
                # v9: FTS5-only — word match, no embedding match.
                # Passthrough score above noise floor. Judge decides relevance.
                blended = FTS5_PASSTHROUGH_SCORE
                discovery = 'fts5_only'
            elif emb_score > 0 and kw_score > 0:
                # Both signals available — blend with embeddings primary
                blended = (EMBEDDING_PRIMARY_WEIGHT * emb_score +
                          KEYWORD_FALLBACK_WEIGHT * kw_score)
                discovery = 'embedding+keyword'
                # v9: Tag 'both' if also found by FTS5
                if nid in fts5_all_ids:
                    discovery = 'both'
            elif emb_score > 0:
                # Embedding only — use embedding score directly
                blended = emb_score
                discovery = 'embedding_only'
                # v9: Tag 'both' if also found by FTS5
                if nid in fts5_all_ids:
                    discovery = 'both'
            else:
                # Keyword only (node has no embedding) — use keyword but PENALIZE.
                # Keyword-only results lack the primary signal. They should never
                # outrank a strong embedding match. Scale by KEYWORD_FALLBACK_WEIGHT
                # so a perfect keyword match (1.0) scores at most 0.10.
                blended = KEYWORD_FALLBACK_WEIGHT * kw_score
                discovery = 'keyword_only_fallback'

            # Intent-based type boosting — DEPRECATED 2026-04-12.
            # Z-score contrastive scoring replaces type boosts.
            # type_boosts is always {} now (set in STEP 2).
            node = keyword_nodes.get(nid)

            # v8.8: Title-match boost — proportional to query/title word overlap.
            # If query terms appear in the node's title, strong relevance signal.
            from .brain_constants import TITLE_MATCH_BOOST
            title = node_titles.get(nid, '').lower()
            if title and _query_terms_set:
                matched = sum(1 for t in _query_terms_set if t in title)
                title_fraction = matched / len(_query_terms_set)
                if title_fraction > 0:
                    blended += title_fraction * TITLE_MATCH_BOOST

            # v5.2: Critical node boost — safety-important nodes always surface
            is_critical = node_critical.get(nid, 0) if 'node_critical' in dir() else 0
            if not is_critical and node:
                is_critical = node.get('critical', 0)
            if is_critical:
                blended *= CRITICAL_BOOST

            # v4 FIX: Contextual qualifier penalty — nodes marked 'contextual' with
            # a personal_context that doesn't overlap query terms get penalized.
            _context_mismatch = False
            node_personal_pair = node_personal_data.get(nid)
            if not node_personal_pair and node:
                _np = node.get('personal')
                _npc = node.get('personal_context', '')
                node_personal_pair = (_np, _npc)
            if node_personal_pair:
                _np, _npc = node_personal_pair
                if _np == 'contextual' and _npc:
                    qualifier_terms = set(_npc.lower().split())
                    overlap = qualifier_terms & _query_terms_set
                    if not overlap:
                        blended *= 0.7
                        _context_mismatch = True

            # Situation boost — additive, never subtractive
            sit_score = situation_scores.get(nid, 0)
            if sit_score > 0:
                blended += SITUATION_WEIGHT * sit_score

            # v10: unified_score integration DEFERRED.
            # Testing showed that applying modulator formula to the embedding path
            # regresses R@8 by -10pts because the modulators dampen scores that were
            # previously passing the relevance floor. The z-weighted top2-avg embedding
            # groups (STEP 3.5) already provide the R@25 improvement.
            #
            # Next step: investigate why the frequency penalty and hub dampening
            # cause regressions — likely need per-query-type adaptive weights
            # rather than one fixed formula.
            #
            # The recall_scoring.py module is ready but not wired in yet.
            # Data collection (node_created_at, node_emotion, node_access_count)
            # is in place for when we're ready to integrate.

            # Minimum threshold — don't return noise
            # v9: Raised from 0.05 to NOISE_FLOOR_THRESHOLD (0.15)
            min_threshold = CRITICAL_SIMILARITY_THRESHOLD if is_critical else NOISE_FLOOR_THRESHOLD
            if blended < min_threshold:
                continue

            scored_results.append({
                'node_id': nid,
                'blended_score': blended,
                'embedding_similarity': round(emb_score * 1000) / 1000 if emb_score else None,
                'keyword_score': round(kw_score * 1000) / 1000 if kw_score else None,
                '_source': discovery,
                '_context_mismatch': _context_mismatch,
            })

        # Sort by blended score descending
        scored_results.sort(key=lambda x: -x['blended_score'])
        scored_results = scored_results[:limit]

        # STEP 6.5: Graph traversal MOVED to Layer 3 (post-surface).
        # Previously: traversed from top-5 cosine results (often hubs).
        # Now: traversal happens in pre_response_recall.py AFTER the surfacer
        # selects relevant nodes. The surfacer's selections are the right seeds.
        # The _traverse_graph() function still exists for Layer 3 to call
        # via the 'graph_expand' daemon command.
        graph_neighborhoods = {}
        try:
            pass  # Traversal deferred to Layer 3
        except Exception as e:
            self._log_error("recall", e, "STEP 6.5 graph traversal")

        # STEP 6.9: Per-result relevance floor.
        # v8.7: Changed from all-or-nothing (top result gates everything) to per-result.
        # Each result must meet its own floor based on how it was discovered.
        # v9: FTS5-only candidates bypass the relevance floor — they go straight to surfacer.
        scored_results = [
            sr for sr in scored_results
            if sr['_source'] == 'fts5_only'  # FTS5-only: always pass to surfacer
            or sr['blended_score'] >= (
                RELEVANCE_FLOOR_ENRICHED
                if enrichment_hits.get(sr['node_id'], 'primary') != 'primary'
                else RELEVANCE_FLOOR_PRIMARY
            )
        ]

        # STEP 7: Hydrate full node data for top results
        # v8.8: Vocab nodes go to separate list — they're connectors, not primary results
        final_results = []
        vocab_context = []
        for sr in scored_results:
            nid = sr['node_id']
            node = keyword_nodes.get(nid)
            if not node:
                # Node came from embedding-only path — fetch from DB via DAL
                try:
                    _node_dal = NodeDAL(self.conn)
                    node = _node_dal.get_naked_node(nid)
                except Exception as e:
                    self._log_error("recall_hydrate", e, "Failed to hydrate node %s" % nid[:8])
                    continue

            if node:
                node['effective_activation'] = sr['blended_score']
                node['embedding_similarity'] = sr['embedding_similarity']
                node['_keyword_score'] = sr['keyword_score']
                node['_source'] = sr['_source']
                if sr.get('_context_mismatch'):
                    node['_context_mismatch'] = True

                # v4: Brain→Host communication dimensions.
                # The brain expresses WHAT it needs to communicate. The host adapter
                # translates HOW. Four dimensions (all 0-1):
                #   priority: how important (locked=high, evolution=high, regular=medium)
                #   confidence: how certain (locked=1.0, hypothesis=its confidence, regular=0.7)
                #   action_expected: should host act on it? (rule/constraint=yes, context=no)
                #   feedback_needed: does brain need a response? (evolution=yes, fact=no)
                ntype = node.get('type', '')
                is_locked = node.get('locked', False)
                is_evolution = ntype in ('tension', 'hypothesis', 'pattern', 'catalyst', 'aspiration')
                is_rule = ntype in ('rule', 'arch_constraint', 'bug_lesson', 'failure_mode',
                                    'constraint', 'lesson', 'convention')  # v5 engineering memory
                is_cognitive = ntype in ('mental_model', 'reasoning_trace', 'uncertainty',
                                         'correction', 'validation')  # v5 cognitive layer
                is_engineering = ntype in ('purpose', 'mechanism', 'impact', 'concept')  # v5 engineering

                # Temporal freshness — how old is this info?
                _freshness = 'unknown'
                _created = node.get('created_at', '')
                if _created:
                    try:
                        from datetime import timezone
                        _cdt = datetime.fromisoformat(_created.replace('Z', '+00:00'))
                        _age_hours = (datetime.now(timezone.utc) - _cdt).total_seconds() / 3600
                        if _age_hours < 1:
                            _freshness = 'just_now'
                        elif _age_hours < 24:
                            _freshness = 'today'
                        elif _age_hours < 168:
                            _freshness = 'this_week'
                        elif _age_hours < 720:
                            _freshness = 'this_month'
                        else:
                            _freshness = 'older'
                    except Exception as e:
                        self._log_error('freshness_parse', e, 'parsing created_at for freshness classification')

                node['_brain_to_host'] = {
                    'priority': 0.9 if is_locked or is_evolution else (
                        0.8 if is_engineering or is_cognitive else (0.7 if is_rule else 0.5)),
                    'confidence': node.get('confidence') or (1.0 if is_locked else 0.7),
                    'action_expected': is_rule or is_locked or ntype == 'impact',
                    'feedback_needed': is_evolution or ntype in ('failure_mode', 'uncertainty', 'correction'),
                    'freshness': _freshness,
                }

                # Attach graph neighborhood from traversal
                node['_graph'] = graph_neighborhoods.get(nid, {})
                node['_discovery'] = sr.get('_source', 'embedding')

                # v4: Contextual qualifier matching.
                # Penalty is applied to blended_score BEFORE sorting in STEP 6.
                # Here we only apply confidence reduction and set the qualifier label.
                node_personal = node.get('personal')
                if node_personal == 'contextual':
                    pctx = node.get('personal_context', '')
                    if pctx and query:
                        qualifier_terms = set(pctx.lower().split())
                        query_terms_set = set(query.lower().split())
                        overlap = qualifier_terms & query_terms_set
                        if not overlap:
                            node['_context_mismatch'] = True
                            node['_context_qualifier'] = pctx
                            # Score penalty already applied in STEP 6 — only reduce confidence here
                            node['_brain_to_host']['confidence'] *= 0.6

                # v9: All nodes are primary results (vocab→concept migration complete)
                final_results.append(node)

        # STEP 7.5a: Apply dict filter on final results
        final_results = _apply_filter(final_results, filter, self.conn)

        # STEP 7.5: Enrich top 3 results with metadata + neighbors
        # All results keep full content (no truncation). Top 3 also get
        # metadata and neighbor context for richer understanding.
        self._enrich_results(final_results[:3])

        # STEP 8: Mark accessed (for Hebbian learning)
        # v9.2: Use the same resolved session_id as fatigue (consistency)
        # v9.2: Unified session_id — single source, no timestamp fallbacks
        sid = getattr(self, '_fatigue_session_id', None) or session_id or self.session_id
        for node in final_results:
            try:
                self._mark_accessed(node['id'], sid)
            except Exception as _e:
                self._log_error("recall", _e, "marking node as accessed for Hebbian learning")

        # STEP 9: Log recall to recall_log (single source of truth)
        recall_ms = (time.time() - t0) * 1000
        # recall_log writes REMOVED 2026-04-05 — S1 traces capture all recall data.
        # Previously inserted into recall_log here. Traces (O/K/Δ) in daemon_hooks
        # are the single source of truth for recall events.

        # Build result
        result = {
            'results': final_results,
            'vocab_context': vocab_context,  # v8.8: vocab nodes as connectors, not results
            'intent': intent,
            '_recall_mode': 'embeddings_first',
            '_embedding_stats': {
                'embedder_ready': True,
                'nodes_with_embeddings': nodes_with_embeddings,
                'nodes_without_embeddings': nodes_without_embeddings,
                'embedding_primary_weight': EMBEDDING_PRIMARY_WEIGHT,
                'keyword_fallback_weight': KEYWORD_FALLBACK_WEIGHT,
                'recall_ms': round(recall_ms, 1),
                'results_by_source': {
                    'embedding+keyword': sum(1 for r in final_results if r.get('_discovery') == 'embedding+keyword'),
                    'embedding_only': sum(1 for r in final_results if r.get('_discovery') == 'embedding_only'),
                    'keyword_only_fallback': sum(1 for r in final_results if r.get('_discovery') == 'keyword_only_fallback'),
                    'fts5_only': sum(1 for r in final_results if r.get('_discovery') == 'fts5_only'),
                    'both': sum(1 for r in final_results if r.get('_discovery') == 'both'),
                    'graph_d1': sum(1 for r in final_results if r.get('_discovery') == 'graph_d1'),
                    'graph_d2': sum(1 for r in final_results if r.get('_discovery') == 'graph_d2'),
                    'graph_d3': sum(1 for r in final_results if r.get('_discovery') == 'graph_d3'),
                    'convergence': sum(1 for r in final_results if r.get('_discovery') == 'convergence'),
                },
                # v6: Enrichment scan stats
                'enrichment_vectors_scanned': enrichment_count if 'enrichment_count' in dir() else 0,
                'enrichment_vectors_used': enrichment_used if 'enrichment_used' in dir() else 0,
                'results_via_enrichment': sum(1 for r in final_results if enrichment_hits.get(r.get('id', ''), 'primary') != 'primary'),
            },
            # v9: Retrieval stats for surfacer — distribution-aware context
            '_retrieval_stats': {
                'brain_size': _brain_size,
                'candidates_after_floor': len(scored_results) if 'scored_results' in dir() else 0,
                'top_score': round(scored_results[0]['blended_score'], 3) if scored_results else 0,
                'median_score': round(scored_results[len(scored_results)//2]['blended_score'], 3) if scored_results else 0,
                'source_breakdown': {
                    'embedding_only': sum(1 for sr in scored_results if sr['_source'] == 'embedding_only'),
                    'embedding+keyword': sum(1 for sr in scored_results if sr['_source'] == 'embedding+keyword'),
                    'fts5_only': sum(1 for sr in scored_results if sr['_source'] == 'fts5_only'),
                    'both': sum(1 for sr in scored_results if sr['_source'] == 'both'),
                    'keyword_only_fallback': sum(1 for sr in scored_results if sr['_source'] == 'keyword_only_fallback'),
                } if 'scored_results' in dir() else {},
            },
        }

        # Gap detection: when no results pass the relevance floor,
        # flag it so the voice layer can prompt encoding
        if not final_results and query:
            result['_gap'] = {
                'query': query,
                'top_score': max_score if 'max_score' in locals() else 0,
            }

        # Carry over reasoning chains from keyword result
        if keyword_result.get('reasoning_chains'):
            result['reasoning_chains'] = keyword_result['reasoning_chains']

        # v5.1: Return query embedding for segment boundary detection
        # Zero cost — already computed in STEP 1
        result['_query_embedding'] = query_vec

        return result

    def _load_zscore_stats(self) -> None:
        """Load precomputed z-score stats (mean, std) from node_metadata_kv.

        Called once per session on first recall. Stats computed by
        scripts/compute_zscore_stats.py and stored via MetadataDAL.
        """
        from .dal_metadata import MetadataDAL
        try:
            mdal = MetadataDAL(self.conn)
            self._zscore_stats = mdal.get_paired_keys(
                ZSCORE_STATS_KEY_MEAN, ZSCORE_STATS_KEY_STD)
        except Exception as e:
            self._log_error('zscore_load', e, 'loading z-score stats from metadata')
            self._zscore_stats = {}

    def _enrich_results(self, results: List[Dict[str, Any]], neighbor_limit: int = 3) -> None:
        """Enrich recall results with metadata and neighbor context.

        This is the ONE place that makes a node result 'complete'.
        Called by both recall (text search) and recall_node (ID lookup).
        The caller decides WHICH results to enrich — this method enriches all it receives.
        Mutates results in place.
        """
        graph_dal = GraphDAL(self.conn)
        for node in results:
            nid = node.get('id')
            if not nid:
                continue

            # Attach metadata from KV store
            try:
                from .dal_metadata import MetadataDAL
                _meta_dal = MetadataDAL(self.conn)
                meta = _meta_dal.get(nid)
                if meta:
                    node['_metadata'] = meta
            except Exception as e:
                self._log_error('recall_node_meta', e, 'fetching node metadata for enrichment')

            # Attach neighbor context — reuse existing get_neighbors_with_context,
            # filter to intentional edges in Python
            if neighbor_limit > 0:
                try:
                    all_neighbors = graph_dal.get_neighbors(
                        nid, limit=neighbor_limit * 3  # fetch extra, filter down
                    )
                    node['_neighbors'] = [
                        {'id': nb['id'], 'type': nb['type'], 'title': nb['title'],
                         'relation': nb['relation'], 'weight': nb['weight']}
                        for nb in all_neighbors
                        if nb.get('relation') in INTENTIONAL_EDGE_TYPES
                    ][:neighbor_limit]
                except Exception as e:
                    self._log_error('recall_node_neighbors', e, 'fetching neighbor context')
                    node['_neighbors'] = []

    # _traverse_graph removed 2026-04-14 — dead code, 0 callers.
    # S1R uses _graph_expand() in surface.py instead.

    def recall_node(self, node_id: str, neighbor_limit: int = 3) -> Dict[str, Any]:
        """Recall a specific node by ID with full enrichment.

        Returns same shape as recall() so callers get a
        consistent interface regardless of how the node was found.
        """
        from .dal import NodeDAL
        node_dal = NodeDAL(self.conn)
        node = node_dal.get_naked_node(node_id)
        if not node:
            return {'results': [], 'intent': 'direct_lookup'}

        # Set display fields for format compatibility
        node['effective_activation'] = node.get('activation', 0)
        node['embedding_similarity'] = None
        node['_keyword_score'] = None
        node['_source'] = 'direct_lookup'

        results = [node]
        self._enrich_results(results, neighbor_limit=neighbor_limit)

        return {
            'results': results,
            'intent': 'direct_lookup',
            '_recall_mode': 'by_id',
        }

    # spread_activation removed 2026-04-14 — keyword search finds matches directly.

    def _search_keywords(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search nodes by full-text search (FTS5).

        v9: Replaced LIKE-based search with FTS5. Porter stemming,
        BM25 ranking, title weighted 10x. TF-IDF scoring layer
        (_keyword_recall) stays unchanged — it scores, FTS5 finds.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching nodes
        """
        try:
            fts5_dal = Fts5DAL(self.conn)
            node_ids = fts5_dal.search(query, limit)
            _ndal = NodeDAL(self.conn)
            results = []
            for nid in node_ids:
                node = _ndal.get_naked_node(nid)
                if node and not node.get('archived'):
                    results.append(node)
            return results
        except Exception as e:
            self._log_error('search_keywords_fts5', e, 'FTS5 search failed, falling back to empty')
            return []

    def _mark_accessed(self, node_id: str, session_id: str):
        """Mark a node as accessed, increment synaptic fatigue.

        Commit-batched (2026-05-08): the per-node `self.conn.commit()`
        was removed. Each recall returns N nodes and used to commit N
        times, which serialized concurrent recalls on the SQLite write
        lock and was the root cause of sustained 100% CPU spinning under
        hook-driven load. The pending UPDATE now batches into either:
          - the _hebbian_strengthen commit at the end of recall, OR
          - the explicit final commit in _recall_impl when Hebbian skipped
        Both produce one commit per recall instead of N.
        """
        node_dal = NodeDAL(self.conn)
        node_dal.mark_accessed(node_id)
        # access_log write removed 2026-04-05 — table dropped
        # commit deferred — see docstring

        # Increment session fatigue counter — next recall will dampen this node's cosine
        # Fatigue lives on SessionContext, persisted via ctx.save() on stop
        _ctx = getattr(self, '_fatigue_ctx', None)
        if _ctx:
            _ctx.increment_fatigue(node_id)
        else:
            # Fallback: in-memory only (no session context available)
            if not hasattr(self, '_session_fatigue'):
                self._session_fatigue = {}
            self._session_fatigue[node_id] = self._session_fatigue.get(node_id, 0) + 1

    def _hebbian_strengthen(self, node_ids: List[str], segment_node_ids: Optional[List[str]] = None):
        """
        Strengthen connections between co-accessed nodes (Hebbian learning).

        If two nodes are co-recalled but have no edge, CREATE a co_accessed edge.
        If they already have an edge, strengthen it.
        This is how the brain auto-discovers relationships from usage patterns.

        v5.1: When segment_node_ids is provided, only create NEW co_accessed edges
        between nodes that are both in the same segment. Existing edges are always
        strengthened regardless (if they co-fire across segments, the edge earned it).
        """
        if len(node_ids) < 2:
            return

        segment_set = set(segment_node_ids) if segment_node_ids else None

        ts = self.now()

        # Cap pairwise work: only top N nodes to avoid O(n^2) explosion
        ids = node_ids[:15]

        graph_dal = GraphDAL(self.conn)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                nid_i = ids[i]
                nid_j = ids[j]

                # Check if edge exists (either direction) — strengthen via DAL
                if graph_dal.strengthen_edge(nid_i, nid_j, amount=LEARNING_RATE * 0.1):
                    continue
                if graph_dal.strengthen_edge(nid_j, nid_i, amount=LEARNING_RATE * 0.1):
                    continue
                else:
                    # NO edge — create co_accessed (v5.1: only within same segment)
                    if segment_set and (nid_i not in segment_set or nid_j not in segment_set):
                        continue
                    graph_dal.create_edge(
                        nid_i, nid_j,
                        weight=EDGE_TYPES['co_accessed']['defaultWeight'],
                        relation='co_accessed')

        self.conn.commit()

    # _log_recall REMOVED 2026-04-05 — recall_log writes deprecated, traces are source of truth

    def _get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently accessed nodes."""
        sql = 'SELECT id FROM nodes WHERE archived = 0'
        params = []
        sql += ' ORDER BY last_accessed DESC LIMIT ?'
        params.append(limit)

        _ndal = NodeDAL(self.conn)
        results = []
        for row in self.conn.execute(sql, params).fetchall():
            node = _ndal.get_naked_node(row[0])
            if node:
                node['spread_activation'] = node.get('activation', 0)
                node['effective_activation'] = node.get('activation', 0)
                results.append(node)
        return results

    def _matches_temporal_filter(self, created_at: Optional[str], after: Optional[str], before: Optional[str]) -> bool:
        """Check if a node creation date matches temporal filter."""
        if not created_at:
            return False
        if after and created_at < after:
            return False
        if before and created_at > before:
            return False
        return True
