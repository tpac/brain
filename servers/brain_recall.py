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
# letting suggestions go stale on natural pauses. Both paths go through
# this single recall layer, so one TTL knob covers both.
_RECALL_CACHE_TTL_S = 10.0
_RECALL_CACHE_MAX_ENTRIES = 100


from .brain_constants import (
    CRITICAL_BOOST,
    CRITICAL_SIMILARITY_THRESHOLD,
    DECAY_HALF_LIFE,
    EMBEDDING_PRIMARY_WEIGHT,
    KEYWORD_FALLBACK_WEIGHT,
    MAX_PAGE_SIZE,
    PRUNE_THRESHOLD,
    RECALL_EXPANSION_TIMEOUT_S,
    RELEVANCE_FLOOR_ENRICHED,
    RELEVANCE_FLOOR_PRIMARY,
    _TITLE_BOOST_STOPWORDS,
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
from .db_backends.sqlite import commit_unless_batched


# ── Lexical bridge — LLM-generated query expansion ─────────────────
# Cosine in our embedding space is flat (top-25 spread ~0.09) and doesn't
# bridge synonyms (feed/scratch grains) or contrastive cases (uncle/niece).
# This helper asks a small model for 2-3 alternate phrasings — synonyms,
# related entities, and explicit contrasts (for abstention queries). Each
# phrasing gets embedded; downstream cosine takes max across all phrasings.
#
# Prompt, model and max_tokens live in the `recall_query_expansion`
# interaction (learnable boundary; code default in
# servers/recall_expansion_prompt.py).
#
# Opt-in via env var BRAIN_QUERY_EXPANSION=on. Failure modes are non-fatal:
# LLM error → skip expansion, recall continues with primary query only.


def _expand_query_via_llm(brain, query: str) -> List[str]:
    """Ask a small model for 2-3 alternate phrasings to bridge lexical gaps.

    Prompt template ({query} slot), model and max_tokens come from the
    `recall_query_expansion` interaction. Returns list of strings (may be
    empty on any failure). Cost: 1 small-model call (~1s, ~300 tokens).
    Caller is expected to embed each separately.
    """
    if not query or len(query.strip()) < 3:
        return []
    # Resolved through the override model — template and config fall back to
    # the code defaults in recall_expansion_prompt.py when no row exists.
    template = brain.get_interaction_prompt('recall_query_expansion')
    cfg = brain.get_interaction_config('recall_query_expansion')
    model = cfg['model']
    max_tokens = cfg['max_tokens']
    try:
        import anthropic
        # Bounded, and no SDK retries. This is a best-effort call on the recall
        # hot path: the `except` below catches errors, not hangs, so only the
        # timeout stops a stalled socket from holding a recall worker thread.
        # The encoder lane's 600s ceiling is the wrong shape here — a recall
        # that waits ten minutes has already failed. max_retries=0 keeps the
        # bound hard (the SDK default of 2 would triple the worst case); recall
        # proceeds on the primary query when expansion misses, which is the
        # same best-effort posture scouts/base.py takes.
        client = anthropic.Anthropic(
            timeout=RECALL_EXPANSION_TIMEOUT_S, max_retries=0)
    except Exception:
        return []
    try:
        # Effective-model line: the resolved config decides the model — this
        # print is the in-run proof of what actually gets called.
        print('[recall] query-expansion model=%s' % model, file=sys.stderr)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{'role': 'user',
                       'content': template.format(query=query)}],
            # Anthropic Structured Outputs — guarantees a JSON array of
            # strings. Closes the drift class where Haiku returns prose
            # or markdown-fenced JSON on long-context queries.
            output_config={
                'format': {
                    'type': 'json_schema',
                    'schema': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                },
            },
        )
        text = ''.join(b.text for b in resp.content if hasattr(b, 'text')).strip()
    except Exception:
        return []
    # With Structured Outputs the response is a guaranteed JSON array, but
    # keep the fence/bounds fallback in case output_config is ever rejected.
    if '```' in text:
        parts = text.split('```')
        for p in parts:
            p = p.strip()
            if p.startswith('json'):
                p = p[4:].strip()
            if p.startswith('['):
                text = p
                break
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
# `project` removed 2026-07-03: provenance lives in node_metadata_kv, so
# filter {'project': ...} routes to the KV lookup like other promoted fields.
_NODE_COLUMNS = frozenset({
    'id', 'type', 'title', 'content', 'activation', 'stability',
    'access_count', 'locked', 'archived', 'critical', 'recency_score',
    'emotion', 'emotion_label', 'emotion_source', 'confidence',
    'personal', 'personal_context', 'evolution_status', 'resolved_at',
    'resolved_by', 'due_date', 'content_summary', 'source_attribution',
    'scope', 'encoding_version', 'encoding_source', 'revised_at',
    'last_accessed', 'created_at',
})


def _apply_filter(nodes: list, filter_dict: dict, conn) -> list:
    """Apply dict filter to a list of node dicts.

    Filter format:
        {"type": {"in": ["moment", "reflection"]}}
        {"my_raw_quote": {"exists": True}}
        {"content": {"contains": "daemon"}}
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

        # Hybrid: top half by relevance, remainder by original (structural)
        # order. limit=None (unbounded relevance rank) → every scored node is
        # a slot, so the whole matching set comes back relevance-first.
        relevance_slots = len(scored) if limit is None else max(1, limit // 2)
        relevant = [n for _, n in sorted(scored, key=lambda x: -x[0])[:relevance_slots]]
        relevant_ids = {n.get('id') for n in relevant}
        # Remainder preserves the input order (which is the structural sort)
        remainder = [n for n in rich_nodes if n.get('id') not in relevant_ids]
        return (relevant + remainder)[:limit]
    except Exception as e:
        # Loud but not fatal — rerank failure falls back to structural order
        print('[brain] _rerank_by_relevance failed: %s' % e, file=sys.stderr)
        return rich_nodes[:limit]


# The failure modes _empty_recall can return in `_recall_mode` — an empty
# result under one of these is an infrastructure failure, NOT "the brain
# knows nothing". Single source: consumers (hook exits, eval validity
# marking) import this instead of copying the strings, so a new failure
# mode cannot silently stop being flagged downstream.
RECALL_FAILURE_MODES = ('embedder_unavailable', 'embed_failed')


class BrainRecallMixin:
    """Recall methods for Brain."""

    def _ensure_structural_degree_cache(self) -> None:
        """Build the per-node structural degree cache used by fatigue.

        Idempotent — returns immediately if already built. Called by:
          - the recall hot path (lazy on first use)
          - Brain.warm_up()  (eager at boot, off the user's critical path)

        Reads `edges` × `edge_relations`. The result feeds the fatigue formula
        `K = 10 / (1 + degree/10)` — hubs fatigue fast, peripherals slow.

        Hoisted out of the per-node cosine loop (was an `if not hasattr`
        check inside an N-row loop) so the lazy guard is paid once, not
        N times. Stores the dict on `self._structural_degree_cache`.
        Failures are logged and result in an empty cache — fatigue then
        defaults to maximum K (no dampening), which degrades gracefully.
        """
        if hasattr(self, '_structural_degree_cache'):
            return
        cache: Dict[str, int] = {}
        try:
            for row in self.conn.execute("""
                SELECT node_id, COUNT(*) FROM (
                    SELECT e.source_id as node_id FROM edges e
                    JOIN edge_relations er ON er.edge_id = e.edge_id
                    UNION ALL
                    SELECT e.target_id as node_id FROM edges e
                    JOIN edge_relations er ON er.edge_id = e.edge_id
                ) GROUP BY node_id"""):
                cache[row[0]] = cache.get(row[0], 0) + row[1]
        except Exception as e:
            self._log_error(
                'fatigue_degree_cache', e,
                'building structural degree cache')
        self._structural_degree_cache = cache

    def get_source_refs(self, node_id: str) -> list:
        """Source refs (trace event ids) for one node — the public door to
        SourceRefDAL. Readers (eval, tooling) route here, never _source_refs."""
        return self._source_refs.get_source_refs(node_id) if node_id else []

    def resolve_live(self, ids, *, on_orphan: str = 'drop'):
        """Owner door for the survivor walk (NodeDAL.resolve_live) — id-SET
        conversion for callers that transform an id collection BEFORE any
        node fetch (candidate feeds, ref lists). Node-returning reads never
        need it: get_node / filter_nodes(field='id') resolve themselves."""
        return self._nodes.resolve_live(ids, on_orphan=on_orphan)

    def get_node(self, node_id_or_ids, follow_absorbed: bool = True):
        """Fully assembled node(s): content + metadata + correction chain + connections.

        Accepts a single node_id (str) or a list of node_ids.
        - Single ID → returns one rich node dict, or None if not found.
        - List of IDs → returns dict keyed by the REQUESTED id. Missing nodes omitted.

        Absorbed ids resolve to their live survivor (the canonical-pull
        contract): an absorb is the store's own identity claim, so an
        id-keyed read of the absorbed id returns the survivor node, keyed
        by the requested id, carrying `_redirected_from` so the renderer
        marks the redirect — the consumer always learns the id moved. A
        RETIRED node (archived, no survivor) returns itself, honestly
        archived. `follow_absorbed=False` is the audit hatch: return the
        absorbed node's own record instead of walking the pointer.
        Writes never follow this redirect — the write doors refuse with
        the pointer instead.

        When given a list, uses batched queries (5 queries total instead of N×4).
        This is the canonical way to get a node. For the bare DB row, use NodeDAL.get_naked_node().
        """
        ndal = self._nodes

        # ── Dispatch: single vs batch ──
        single = isinstance(node_id_or_ids, str)
        raw_ids = [node_id_or_ids] if single else list(node_id_or_ids)

        if not raw_ids:
            return None if single else {}

        full_ids = [nid for nid in raw_ids if nid]

        if not full_ids:
            return None if single else {}

        # ── 1. Batch fetch all nodes (via NodeDAL — single SQL source) ──
        nodes = ndal.get_bulk(full_ids)

        if not nodes:
            return None if single else {}

        # ── 1b. Absorbed-id redirect. Zero extra queries when every hit is
        # live (the common case): detection rides get_bulk's own archived
        # flag; only actual archived hits pay the survivor walk. The stamp
        # is applied per-REQUEST at return time (copy-on-stamp), never onto
        # the shared survivor row — a survivor the caller also asked for
        # directly must not carry another request's redirect banner.
        alias = {}  # requested id -> live terminal id (redirects only)
        if follow_absorbed:
            archived_hits = [nid for nid, n in nodes.items()
                             if n.get('archived')]
            if archived_hits:
                redirected = ndal.resolve_live(archived_hits)['redirected']
                missing = [t for t in set(redirected.values())
                           if t not in nodes]
                if missing:
                    nodes.update(ndal.get_bulk(missing))
                alias = {src: t for src, t in redirected.items()
                         if t in nodes}
                for src in alias:
                    # The corpse leaves the working set — its requested id
                    # re-enters at return time, keyed onto the survivor.
                    nodes.pop(src, None)
                # Orphans (retired / dead chain) keep their corpse row —
                # honest archived render, nothing to resolve to.

        found_ids = list(nodes.keys())

        # ── 2. Batch fetch all metadata via MetadataDAL.get_all_bulk.
        # Single SQL source for KV reads — same DAL the correction-enrich
        # pipeline uses for its scoped fetch.
        meta_by_node = self._meta_kv.get_all_bulk(found_ids)
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
                # Promote project the same way — provenance is first-class.
                # KV wins over the legacy column value (migration transition).
                proj = meta_by_node[nid].get('project')
                if proj:
                    nodes[nid]['project'] = proj

        # ── 4. Batch corrections via aspect-edge walk ──
        # Renderer slices the heavy payload per consumer (HAIKU_FORMAT
        # balanced, ENCODER_FORMAT heavy).
        corrections = self.correction_enrich(found_ids)
        for nid in found_ids:
            node_corrs = corrections.get(nid) or []
            if node_corrs:
                nodes[nid]['_corrections'] = node_corrs

        # ── 5. Batch fetch all connections via GraphDAL (v25) ──
        # DAL centralizes: archived=0 default, direction detection,
        # per-neighbor relation grouping. No default relation exclusion —
        # noise hiding for the encoder view lives in encode_contract's
        # _filter_noise_relations (aspect-owned read-exclusion for get_node
        # is a deferred design, not implemented).
        connections_by_owner = self._graph.get_connections_bulk(found_ids)

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

        # ── Return: keyed by the REQUESTED id. A redirected request gets a
        # per-request SHALLOW COPY of the survivor carrying its own
        # REDIRECTED_FROM_KEY (`node['id']` names the survivor); the
        # survivor's own entry, if also requested, stays unstamped. Nested
        # structures (connections, _metadata) are shared across copies —
        # renderers replace keys, they don't mutate in place. ──
        from .contract import REDIRECTED_FROM_KEY
        if single:
            fid = full_ids[0]
            t = alias.get(fid)
            if t is None:
                return nodes.get(fid)
            node = nodes.get(t)
            return (dict(node, **{REDIRECTED_FROM_KEY: [fid]})
                    if node else None)
        if not alias:
            return nodes
        out = {}
        for fid in full_ids:
            t = alias.get(fid)
            if t is None:
                if fid in nodes:
                    out[fid] = nodes[fid]
            elif t in nodes:
                out[fid] = dict(nodes[t], **{REDIRECTED_FROM_KEY: [fid]})
        return out

    def canonicalize_results(self, results, session_id: str = '') -> None:
        """Overlay the canonical pull onto recall results, in place.

        The ONE door that makes a recall result complete. get_node is the
        canonical pull — metadata, situation, corrections, connections — and
        this lays exactly its attachments (CANONICAL_ATTACHMENT_KEYS) over
        each result, leaving recall's own scoring fields untouched. Then the
        session veil scrubs those attachments: a connection or correction
        line carries a neighbor's id and title, and a correction its full
        text, which is the payload a wall exists to stop.

        Every recall door routes through here — by-query, by-id and batch —
        so no caller can hand back a node with its corrections missing.
        Bypassing it is how `recall(node_id=…)` used to return a superseded
        claim with no correction marker attached.

        EVERY result, not a slice. A cap here reads as a cost saving and acts
        as a correctness boundary: the renderer draws whatever it is handed,
        so capped-off results render as authoritative with their correction
        chain silently absent — the exact failure this door exists to remove.
        Measured at ~2.5ms/node, bounded per call by recall's MAX_PAGE_SIZE.
        recall_batch multiplies that by its query count; if that ever needs
        bounding, the lever is the recall LIMIT — how many nodes a caller
        asked for — never a cap on how many of them come back whole.

        A failed pull RAISES rather than degrading. Returning a node whose
        corrections could not be read is indistinguishable, downstream, from
        a node that has none — so silence here would reintroduce the hazard
        in its worst form. The dispatcher turns the raise into a loud error.

        `session_id` falls back to the brain's ambient session, matching the
        by-query door: the doors resolving the veil differently is the
        divergence this method was written to end.
        """
        from .contract import CANONICAL_ATTACHMENT_KEYS
        from .scopes import scrub_node
        rich = self.get_node([r['id'] for r in results
                              if isinstance(r, dict) and r.get('id')])
        if not rich:
            return
        veil = self.scope_veil(session_id or self.session_id)
        for r in results:
            node = rich.get(r.get('id')) if isinstance(r, dict) else None
            if not node:
                continue
            if node.get('id') and node['id'] != r.get('id'):
                # The requested id was absorbed — get_node returned its live
                # survivor (marked). The row BECOMES the survivor wholesale:
                # overlaying only the attachments would dress the corpse in
                # the survivor's corrections and edges, the exact chimera the
                # one-door contract exists to prevent. Recall's own scoring
                # fields aren't node fields, so they survive the update.
                r.update(node)
            for key in CANONICAL_ATTACHMENT_KEYS:
                if key in node:
                    r[key] = node[key]
            scrub_node(r, veil)

    def filter_nodes(self, field: str, include=None, exclude=None,
                     lt=None, gt=None, contains=None, prefix=None, limit: int = 50,
                     sort_by: str = 'created_at', sort_order: str = 'desc',
                     rich: bool = True,
                     relevance_query: str = None,
                     relevance_pool_multiplier: int = 3,
                     relevance_vector_type: str = '_primary',
                     session_id: str = ''):
        """Structured query: filter nodes by any structural field.

        rich=True (default): full content, metadata, corrections, connections
        via batched get_node() — 5 queries regardless of N. The consumer is a
        reasoner; richness is the advantage (see node 9b938b91).
        rich=False: skinny shape (id/title/type/confidence/created_at), for
        discovery scans or feeding IDs to other ops.

        contains: substring LIKE match on `field` (recall's dict-filter speaks
        the same op). prefix: prefix LIKE match (prefix is new here — recall's
        dict-filter has no prefix). limit=None returns every match unbounded
        (e.g. every s2:* node for an internal id-set scan); a number is an
        honest page. The agent-facing cap lives at the dispatch door. Pair
        limit=None with rich=False: with the default rich=True it enriches
        every match (get_node per row), which for a large set is the firehose
        the cap exists to prevent.

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
        from .contract import truncation_payload, REDIRECTED_FROM_KEY
        node_dal = self._nodes

        def _stamp_redirects(node_list):
            if not _absorbed_from:
                return
            for n in node_list:
                srcs = _absorbed_from.get(n.get('id'))
                if srcs:
                    n[REDIRECTED_FROM_KEY] = list(srcs)
        # Widen the pool when relevance ranking is requested. The structural
        # path's truncation detection rides the DAL's exact total_count (an
        # unclamped COUNT(*) on the same WHERE) — no +1 probe: a probe row is
        # eaten by the veil cut and capped by the DAL clamp, both of which
        # silently un-flag saturated results (2026-08-07 review, finding 3).
        # The relevance path is ranked top-k — truncation is its contract.
        # limit=None → unbounded pull (all matches); the pool-widening below
        # (relevance pool, veil backfill) only applies to a finite page.
        dal_limit = limit
        if relevance_query and dal_limit is not None:
            dal_limit = dal_limit * relevance_pool_multiplier
        # Scope veil: filter_nodes is an ambient ENUMERATION surface (a
        # structural sweep, not a reach for a known id) — rich=True would
        # otherwise hand out the walled corpus's full content in one call,
        # and skinny results feed boot's standing items. Over-fetch so
        # walled drops backfill. Sessionless callers ('' — never the
        # ambient last-seen session, whose borrowed INWARD veil would be
        # the complement of a wall) get the default-deny outward veil.
        # The veil cut does NOT slice on the relevance path — the widened
        # pool belongs to _rerank_by_relevance, which owns the trim.
        _veil = self.scope_veil(session_id or '')
        if _veil and dal_limit is not None:
            dal_limit = dal_limit * 2
        result = node_dal.filter_nodes(
            field=field, include=include, exclude=exclude,
            lt=lt, gt=gt, contains=contains, prefix=prefix,
            limit=dal_limit, sort_by=sort_by, sort_order=sort_order)
        # Absorbed-id redirect for the id-keyed lookup shape (the canonical-
        # pull contract) — POST-HOC and missing-only, so the all-live case
        # (every boot/Stop ref pull) pays zero extra queries: the DAL's
        # archived=0 already said which ids are gone, and only those pay the
        # survivor walk. Retired/missing ids stay dropped, as before. This
        # door returns a LIST, so a stamp means "these requested ids resolve
        # here" — a survivor already in the page is stamped in place rather
        # than duplicated.
        _absorbed_from = {}  # survivor id -> [requested ids]
        if field == 'id' and include and result.get('nodes') is not None:
            _returned = {n.get('id') for n in result['nodes']}
            _gone = [i for i in include if i and i not in _returned]
            if _gone:
                for _src, _dst in node_dal.resolve_live(
                        _gone)['redirected'].items():
                    _absorbed_from.setdefault(_dst, []).append(_src)
                _need = [t for t in dict.fromkeys(_absorbed_from)
                         if t not in _returned]
                if _need:
                    _rows = node_dal.filter_nodes(
                        field='id', include=_need,
                        limit=len(_need)).get('nodes') or []
                    result['nodes'].extend(_rows)
                    result['total_count'] = (
                        result.get('total_count', 0) + len(_rows))
        if _veil and result.get('nodes'):
            kept = [n for n in result['nodes'] if n.get('id') not in _veil]
            # The relevance path's trim belongs to _rerank_by_relevance; the
            # structural trim happens in the truncation block below.
            result['nodes'] = kept
        if 'error' in result or not result.get('nodes'):
            return result

        # Truncation contract (contract.py): trim the structural pool to
        # `limit`; saturation detection is the DAL's exact, unclamped
        # total_count — veil-independent and alive at every limit. When a
        # veil hides rows, total_count still counts them, so a veiled page
        # can over-flag (report truncated when the hidden remainder is all
        # walled) — conservative in the right direction: isolation governs
        # what rises, not what the count admits exists.
        if not relevance_query and limit is not None:
            result['nodes'] = result['nodes'][:limit]
            if result.get('total_count', 0) > limit:
                result['truncated'] = truncation_payload(
                    limit, result['nodes'])
        # limit is None → unbounded: every matching row is kept and total_count
        # == len(nodes), so there is nothing to trim or flag.

        # Relevance ranking happens BEFORE enrichment. _rerank_by_relevance
        # scores by embedding (looked up by id) + structural order — it needs
        # only ids, not rich content — so we rank the skinny candidate pool and
        # trim to `limit` here, then enrich only the winners below. This is the
        # decoupling: correction-enrich `limit` nodes, not the whole
        # (limit × relevance_pool_multiplier) candidate pool we discard most of.
        nodes = result['nodes']
        if relevance_query:
            nodes = _rerank_by_relevance(
                self.conn, nodes, relevance_query, limit,
                vector_type=relevance_vector_type)
        # No relevance: the truncation block above already trimmed to `limit`.

        if not rich:
            _stamp_redirects(nodes)
            result['nodes'] = nodes
            return result

        # Rich: enrich only the (already-ranked, <= limit) winners.
        ids = [n['id'] for n in nodes]
        rich_map = self.get_node(ids)
        result['nodes'] = [rich_map[i] for i in ids if i in rich_map]
        # After enrichment — get_node returns fresh dicts, so the skinny
        # rows' stamps don't survive it.
        _stamp_redirects(result['nodes'])
        return result

    def query_logs(self, source: str = 'all', hours: int = 24,
                   level: str = 'all', hook_name: str = '',
                   limit: int = 50):
        """Query brain logs: errors, debug events, and signals.

        Windowed read → truncation contract (contract.py): the DAL already
        counts each source's TRUE window total alongside the limited fetch,
        so saturation detection is exact with no extra query — flagged
        loudly instead of buried in `counts` for the caller to notice.
        """
        result = self._logs_dal.query_logs(
            source=source, hours=hours, level=level,
            hook_name=hook_name, limit=limit)
        from .contract import truncation_payload
        from .dal_logs import LOG_QUERY_MAX_LIMIT
        # Report the EFFECTIVE limit — the DAL clamps at LOG_QUERY_MAX_LIMIT,
        # and a note advising "raise limit" past the cap prescribes an
        # impossible remedy (2026-08-07 review, finding 10).
        effective = min(max(limit, 1), LOG_QUERY_MAX_LIMIT)
        entries = result.get('entries', [])
        total = sum(result.get('counts', {}).values())
        if total > len(entries):
            result['truncated'] = truncation_payload(
                effective, entries,
                reason='%d of %d matching rows returned (effective limit=%d,'
                       ' hard cap %d)' % (len(entries), total, effective,
                                          LOG_QUERY_MAX_LIMIT))
        return result

    def list_interactions(self):
        """List all registered interactions with latest versions."""
        return self._interaction_dal.list_all()

    def list_interaction_versions(self, name: str):
        """Every registered version of one interaction: version + created_by."""
        return self._interaction_dal.list_versions(name)

    def backfill_embeddings(self, batch_size: int = 20) -> int:
        """Legacy wrapper — calls backfill_vectors()."""
        result = self.backfill_vectors(batch_size)
        return result.get('total', 0) if isinstance(result, dict) else 0

    def vector_coverage_sweep(self, batch_size: int = 30) -> dict:
        """Repair vector gaps the enqueue path missed; report what it couldn't.

        ONE door, because repair and detection must ask the same question.
        Split across two callers they drifted: the repair passed `model=` and a
        separate probe did not, and `find_missing` counts a stale-model row as
        present without it and missing with it. A model swap — the bulk case,
        the whole corpus — therefore read as "nothing left to do" while
        thousands of rows waited.

        `remaining` is derived from the repair's own per-type counts rather than
        a second query: a type that filled its batch has more behind it. The two
        halves then cannot disagree, and a node that can never be embedded —
        which repairs nothing — can never hold a caller in a re-sweep loop.

        Returns {repaired, by_type, remaining, stuck}. `stuck` is only probed
        when repair achieved nothing, which is the one state worth waking for:
        `_primary` is the LAF-visibility invariant, and a node without it is
        invisible to the field entirely.
        """
        result = self.backfill_vectors(batch_size=batch_size) or {}
        counts = {k: v for k, v in result.items() if isinstance(v, int)}
        repaired = sum(counts.values())
        remaining = any(v >= batch_size for v in counts.values())
        stuck = []
        if not repaired:
            stuck = self._vec_dal.find_missing(
                '_primary', 1, model=embedder.stats.get('model_name', ''))
        return {'repaired': repaired, 'by_type': result,
                'remaining': remaining, 'stuck': stuck}

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
        from .pipeline_contract import (EMBEDDING_GROUPS, EMBEDDING_SKIP_FIELDS,
                                        EMBEDDING_FIELD_CHAR_LIMIT,
                                        EMBEDDING_DEAD_HANDLER_MIN_CANDIDATES)
        from .dal_metadata import MetadataDAL

        vdal = self._vec_dal
        mdal = self._meta_kv
        model = embedder.stats.get('model_name', '')
        result = {}

        def _store_batch(items, vector_type):
            """items = list of (node_id, text). One embed_batch + one
            executemany store. Returns count of rows written.

            Embedding runs OUTSIDE brain.write_lock — `embedder.embed_batch`
            is CPU-heavy ONNX work that historically held the lock for
            seconds-to-minutes on large batches, blocking every other
            writer (multi-session contention). The lock is acquired
            only around the actual DB write + commit, which is brief.
            write_lock is an RLock, so callers that already hold it
            (none today) wouldn't deadlock.

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
            with self.write_lock:
                stored = vdal.store_batch(rows, model=model)
                if stored:
                    # Gate on conn.in_batch — NOT an unconditional commit. A bare
                    # self.conn.commit() here releases ALL open SAVEPOINTs, which
                    # silently broke a content-rewriting absorb run inside a
                    # brain_batch: the embed drain backfilled the survivor's
                    # re-embedded vector mid-merge and its commit killed
                    # `absorb_sp`, so the later RELEASE threw `no such savepoint`
                    # and the merge no-op'd while the batch reported ok. When a
                    # batch/savepoint envelope owns self.conn (in_batch=True), the
                    # owner commits; standalone backfill commits as before.
                    commit_unless_batched(self.conn)
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

            # edge_context's only source (_edge_descriptions) lives on edges,
            # not kv — gate find_missing on edge existence so edgeless nodes
            # don't clog the batch and starve the edged ones (see find_missing).
            needs_edge_filter = '_edge_descriptions' in group_config.get('fields', [])

            try:
                missing = vdal.find_missing(
                    vector_type, batch_size, model=model, node_ids=node_ids,
                    source_kv_keys=kv_source_keys if kv_source_keys else None,
                    require_described_edge=needs_edge_filter)
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
                            if field == '_edge_descriptions':
                                # Edge context: descriptions live on edges, not the
                                # node. Mirrors the write-time handler in
                                # _compute_group_vectors (brain_remember.py) — same
                                # helper, same defaults (both directions; noise/
                                # archived/short-desc filtering centralized in the DAL).
                                for desc in self._graph.get_edge_descriptions_for(node['id']):
                                    parts.append(desc[:EMBEDDING_FIELD_CHAR_LIMIT])
                                continue
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

                # Loud-by-default dead-handler trip. Two gates keep it precise so
                # it only fires on the real silent-partial that hid edge_context's
                # 0-row bug, not on healthy edge cases:
                #  (a) the group must have an ELIGIBILITY FILTER (kv-key or
                #      described-edge) — only then is a returned candidate a
                #      guarantee it SHOULD yield text. An unfiltered group like
                #      field_content can return nodes with empty content that
                #      legitimately build nothing.
                #  (b) ZERO text built (`not items`), not merely zero stored — a
                #      transient embed failure (embed_batch returns []) builds
                #      items but stores 0; that's an embedder problem, not a dead
                #      handler, and would mis-fire the alarm.
                # Pass a real exception (not None): _log_error reads
                # error.__traceback__, so None never reaches debug_log.
                has_eligibility_filter = bool(kv_source_keys) or needs_edge_filter
                if (has_eligibility_filter
                        and len(missing) >= EMBEDDING_DEAD_HANDLER_MIN_CANDIDATES
                        and not items):
                    self._log_error(
                        'embedding_handler_dead',
                        RuntimeError(
                            'group %s: %d eligible candidates but built no embed '
                            'text — handler is dead/missing'
                            % (vector_type, len(missing))),
                        'embedding backfill dead-handler check')
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
                        session_id: Optional[str] = None,
                        _skip_log: bool = False,
                        mark_accessed: bool = True) -> Dict[str, Any]:
        """INTERNAL: TF-IDF keyword recall. Used by recall() for keyword blending.
        Do NOT call directly — use recall() (embeddings + graph traversal) instead.

        `project` param removed 2026-07-03 — project is provenance in
        node_metadata_kv now: filter with {'project': {...}} (routes to the KV
        lookup) or lean on the LAF proj lane, which scores it per session.

        Args:
            query: Search query
            filter: Dict filter (same format as recall())
            limit: Max results to return
            offset: Pagination offset
            include_archived: Include archived nodes
            min_recency: Minimum recency score threshold
            session_id: Optional session ID for logging

        Returns:
            Dict with results (list of nodes).
        """
        limit = min(limit, MAX_PAGE_SIZE)

        expanded_query = query

        # Pure FTS5/keyword-precision net — no intent classification, no type
        # boosts, no date windowing.

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
                tfidf_dal = self._tfidf
                tfidf_node_ids = tfidf_dal.get_nodes_matching_terms(unique_terms)
                _node_dal = self._nodes
                # Batch-fetch the not-yet-seen seed candidates in one query
                # instead of N get_naked_node calls (C1 / H2).
                _cand_ids = [nid for nid in tfidf_node_ids[:50] if nid not in all_seeds]
                _bulk = _node_dal.get_bulk(_cand_ids)
                for nid in _cand_ids:
                    node = _bulk.get(nid)
                    if node and not node.get('archived'):
                        all_seeds[nid] = node
            except Exception as _e:
                self._log_error("recall", _e, "fetching seed node details from database")

        if not all_seeds:
            # Return recent nodes if no seeds found — behind the scope veil
            # (an isolated project's nodes are recent too). Over-fetch when
            # a veil is active so the "give them SOMETHING" floor survives
            # the drops, then re-cut to limit.
            _veil = self.scope_veil(session_id or '')
            _fetch = limit * 2 if _veil else limit
            _recent = [n for n in
                       _apply_filter(self._get_recent(_fetch), filter, self.conn)
                       if n.get('id') not in _veil][:limit]
            return {
                'results': _recent,
            }

        # Step 1b: Compute direct keyword match strength per seed
        query_terms = [w.replace('[^a-z0-9]', '', ) for w in query.lower().split()
                       if len(w) > 2]
        query_terms = [w for w in query_terms if w]

        direct_match_scores = {}
        for seed_id, seed in all_seeds.items():
            title = (seed.get('title') or '').lower()
            content = (seed.get('content') or '').lower()
            match_count = 0
            for term in query_terms:
                if term in title or term in content:
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
                ntitle = (node.get('title') or '').lower()
                ncontent = (node.get('content') or '').lower()
                mc = 0
                for term in query_terms:
                    if term in ntitle or term in ncontent:
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
            _graph_dal = self._graph
            edge_count = _graph_dal.count_node_edges(node['id'], min_weight=0)
            hub = self._get_tunable('hub_dampening', {'threshold': 40})
            hub_threshold = hub.get('threshold', 40) if isinstance(hub, dict) else 40
            if edge_count > hub_threshold:
                relevance *= hub_threshold / edge_count

            # v4: Type dampening
            if node.get('type') in ('project', 'person'):
                relevance *= 0.5

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
            # Per-session recency_score when available (post 2026-05-25
            # parallel-session work) — fall back to the global field on the
            # node payload if this session hasn't touched the node yet. The
            # gate is disabled in prod today (callers pass min_recency=0),
            # so this is the correct shape for when it turns on, not a
            # behavior change at the current call sites.
            ctx = self._session_contexts.get(session_id) if session_id else None
            if ctx is not None:
                def _rec(n):
                    rec = ctx.node_activity.get(n['id'], {})
                    return float(rec.get('recency_score',
                                         n.get('recency_score', 0)))
                filtered = [n for n in filtered if _rec(n) >= min_recency]
            else:
                filtered = [n for n in filtered if n.get('recency_score', 0) >= min_recency]

        # Step 5: Sort by effective activation
        filtered.sort(key=lambda n: -n.get('effective_activation', 0))

        # Step 5.5: Scope veil — BEFORE pagination, so walled drops backfill
        # from lower-ranked eligible nodes instead of under-filling the page
        # (and offset pagination stays monotonic). Before access marks: a
        # gated node was never surfaced, so it must not strengthen.
        _veil = self.scope_veil(session_id or '')
        if _veil:
            filtered = [n for n in filtered if n.get('id') not in _veil]

        # Step 6: Pagination
        page = filtered[offset:offset + limit]

        # Step 7: Mark accessed (keyword-only fallback path).
        # No ctx loaded — keyword recall doesn't use fatigue for scoring, so
        # we skip ctx for the access marks too. access_count still updates.
        #
        # mark_accessed must be honored HERE as well as in _recall_impl: this
        # function is not only the keyword-only fallback, it also runs as
        # _recall_impl's STEP 4 on every single recall (with limit*3), so a
        # read-only caller that guarded just the outer loop would still mark
        # three times the requested nodes through this path.
        if mark_accessed:
            if not session_id:
                session_id = self.session_id
            for node in page:
                self._mark_accessed(node['id'], session_id, ctx=None)

        # No edge creation here — the co_accessed family is retired
        # (node ab56d25a); surface_selected traces are the co-access substrate.

        # v4: Auto-instrument (skipped when called from recall
        # or hooks — they log via the precision module instead)
        returned_ids = [n['id'] for n in page]
        # recall_log writes REMOVED 2026-04-05 — S1 traces capture all recall data.

        # recall_log writes removed 2026-04-05 — traces are source of truth
        result = {
            'results': page,
        }

        return result

    def _trace_chain_candidates(self, query_vec, exclude_ids):
        """Episodic dual-store rescue lane (flag: BRAIN_TRACE_CHAIN). Two-hop cosine:
        query -> top-T s0 DIALOGUE traces -> each trace's STORED vector -> top-N nodes.
        Returns {node_id: combined tcos*ncos} for the top TRACE_CHAIN_RESERVE nodes NOT already
        found by embedding/keyword/fts5 (exclude_ids). The trace de-dilutes a buried query: it is
        specific, un-pooled conversation text, so it pulls EX.CO nodes the diluted query cosine missed.

        Design: docs/RECALL-DUAL-STORE-DESIGN.md §3.3 form 1 (the semantic chain — the burial FIX).
        Hygiene (§4): s0 user/assistant only; tool_result dropped (the 82% recall-echo poison).
        Cost note (§8): ~1 trace-vector scan + T node-vector scans per call. Flag-gated; the lane is
        OFF by default so this never touches the live hot path until eval-gated activation.
        """
        from .brain_constants import TRACE_CHAIN_T, TRACE_CHAIN_N, TRACE_CHAIN_RESERVE
        from .trace_contract import CONVERSATIONAL_REF_TYPES
        try:
            # rows: (chain_id, session_id, created_at, vector) — indexed pull of
            # exactly the embedded conversational traces (same door recall_laf
            # uses); the conversational dial lives in trace_contract.
            trows = self._trace_dal.event_vector_rows(
                scale='s0', ref_types=list(CONVERSATIONAL_REF_TYPES))
            tr = [(embedder.cosine_similarity(query_vec, r[3]), r[3])
                  for r in trows]
            if not tr:
                return {}
            tr.sort(key=lambda x: -x[0])
            model = embedder.stats.get('model_name') or None
            node_rows = self._vec_dal.get_all_vectors(vector_types=['_primary'], model=model)
            node_vecs = [(nr['node_id'], nr['embedding']) for nr in node_rows if nr['embedding']]
            best = {}   # node_id -> max combined tcos*ncos (strongest puller wins)
            for tcos, tvec in tr[:TRACE_CHAIN_T]:
                nn = sorted(((embedder.cosine_similarity(tvec, b), nid) for nid, b in node_vecs),
                            key=lambda x: -x[0])[:TRACE_CHAIN_N]
                for ncos, nid in nn:
                    if nid in exclude_ids:
                        continue
                    comb = tcos * ncos
                    if nid not in best or comb > best[nid]:
                        best[nid] = comb
            return dict(sorted(best.items(), key=lambda x: -x[1])[:TRACE_CHAIN_RESERVE])
        except Exception as e:
            self._log_error('recall_trace_chain', e, 'trace-chain lane')
            return {}

    def recall(self, query: str, filter: Optional[Dict[str, Any]] = None,
               limit: int = 20, offset: int = 0,
               include_archived: bool = False,
               min_recency: float = 0,
               session_id: Optional[str] = None,
               situation_vec=None, source: str = 'unknown',
               ctx=None, as_of: Optional[str] = None,
               mark_accessed: bool = True) -> Dict[str, Any]:
        """Recall: embeddings + 3-degree graph traversal + keyword blending + situation matching.

        `project` param removed 2026-07-03 (it had zero production callers) —
        project is provenance in node_metadata_kv now: hard-scope with
        filter {"project": {"equals": ...}} (routes to the KV lookup), or let
        the LAF proj lane score the session's project as a soft field.

        Args:
            query: Search query
            filter: Dict filter on node/metadata fields. Examples:
                {"type": {"in": ["moment", "reflection"]}}
                {"my_raw_quote": {"exists": True}}
                {"content": {"contains": "daemon"}}
                {"confidence": {"gte": 0.9}}
            limit: Max results
            offset: Pagination offset
            include_archived: Include archived
            min_recency: Min recency threshold
            session_id: Optional session ID

        Returns:
            Dict with results, _embedding_stats, _recall_mode

        Two layers of dedup (2026-05-08):
        1. Result cache (5s TTL) — repeat identical recalls return cached
           result without re-running. Deepcopy on read so callers are
           isolated.
        2. Single-flight gate — concurrent identical recalls share work:
           one becomes leader, others wait for its result.

        Both keyed by (query, filter, limit, offset, include_archived,
        min_recency, session_id, situation_vec). session_id is
        in the key because synaptic fatigue is per-session.

        Replaces the dispatch + brain.pre_edit caches: every recall caller
        (pre_edit, pre_bash_safety, hook_recall, MCP) now benefits.
        """
        # Resolve session_id from ctx when caller passed an object; ctx wins
        # over session_id if both supplied (the convention is to pass the object).
        if ctx is not None and not session_id:
            session_id = ctx.session_id

        # Build dedup key from result-affecting params.
        try:
            filter_key = json.dumps(filter, sort_keys=True, default=str) if filter else None
            sit_key = bytes(situation_vec) if situation_vec is not None else None
            dedup_key = (
                query, int(min(limit, MAX_PAGE_SIZE)), int(offset),
                bool(include_archived), float(min_recency or 0),
                session_id, filter_key, sit_key, as_of,
                # mark_accessed is in the key even though it can't change the
                # RESULT: without it an observer's read-only recall could win
                # the single-flight race and a real recall waiting behind it
                # would silently lose its access + fatigue marking.
                bool(mark_accessed),
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
            # Phase 5 (2026-05-18): _run_recall_with_commit removed — recall
            # is now read-only at SQLite (writes deferred to recall_write_queue
            # drained by the bg_writer worker), so no commit step is needed.
            try:
                result = self._recall_impl(
                    query=query, filter=filter, limit=limit, offset=offset,
                    include_archived=include_archived, min_recency=min_recency,
                    session_id=session_id, ctx=ctx,
                    situation_vec=situation_vec, source=source, as_of=as_of,
                    mark_accessed=mark_accessed)
                self._recall_cache_put(dedup_key, result)
                inflight.set_result(result)
                return result
            except Exception as e:
                inflight.set_exception(e)
                raise
            finally:
                self._recall_inflight_release(dedup_key, inflight)

        # Dedup disabled (key construction failed) — fall through.
        return self._recall_impl(
            query=query, filter=filter, limit=limit, offset=offset,
            include_archived=include_archived, min_recency=min_recency,
            session_id=session_id, ctx=ctx,
            situation_vec=situation_vec, source=source, as_of=as_of,
            mark_accessed=mark_accessed)

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

    def _recall_cache_purge(self) -> None:
        """Drop every cached recall result — called by
        Brain.invalidate_interaction_caches when a recall-shaping interaction
        (recall_laf, recall_query_expansion) flips or clears. The dedup key
        carries no config fingerprint, so without the purge an identical query
        re-asked within the TTL returns the pre-flip result verbatim."""
        if not hasattr(self, '_recall_cache'):
            return
        with self._recall_cache_lock:
            self._recall_cache.clear()

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

    def _empty_recall(self, mode: str) -> Dict[str, Any]:
        """The one empty-result shape for recall's no-embedding exits.

        Every exit carries `_embedding_stats` so the MCP diagnostic footer
        (brain_mcp renders it only when stats is non-empty) shows the failure
        mode inline — an exit without it reads as "brain knows nothing" instead
        of "embedding is broken"."""
        return {'results': [], '_recall_mode': mode,
                '_embedding_stats': {'embedder_ready': embedder.is_ready(),
                                     'embedder_status': embedder.get_model_status()}}

    def _recall_impl(self, query: str, filter=None, limit: int = 20,
                     offset: int = 0, include_archived: bool = False,
                     min_recency: float = 0,
                     session_id=None, situation_vec=None,
                     source: str = 'unknown', ctx=None,
                     as_of: Optional[str] = None,
                     mark_accessed: bool = True) -> Dict[str, Any]:
        """Actual recall implementation — hot path. Single-flight wrapper
        in recall() ensures only one of these runs per (query, scope) at
        a time across the daemon."""
        t0 = time.time()
        limit = min(limit, MAX_PAGE_SIZE)

        # ── No embedder → no recall. Deliberately NO keyword fallback: a silent
        #    keyword substitute masks that semantic recall is down. Report the
        #    condition (dashboard errors view + boot unsurfaced-errors notice)
        #    and return empty — the recall hook degrades gracefully on empty
        #    results, and the embedder is warmed at daemon boot so this window
        #    is small.
        if not embedder.is_ready():
            self._log_error('recall_embedder_unavailable', None,
                            'embedder not ready — returning empty (no keyword fallback)')
            return self._empty_recall('embedder_unavailable')

        # ── PRIMARY PATH: Embeddings-first ──

        expanded_query = query
        _active_model = embedder.stats.get('model_name') or None

        # STEP 1: Embed the query. No fallback — a failed embed reports the
        # condition and returns empty (see the not-ready branch above).
        try:
            query_vec = embedder.embed_query(expanded_query)
        except Exception as e:
            self._log_error('recall_embed_failed', e,
                            'embed_query raised — returning empty (no keyword fallback)')
            return self._empty_recall('embed_failed')
        if not query_vec:
            self._log_error('recall_embed_failed', None,
                            'embed_query returned empty — returning empty (no keyword fallback)')
            return self._empty_recall('embed_failed')

        # Lexical bridge alternates — populated conditionally AFTER primary
        # cosine completes (see post-STEP-3 expansion gate). Empty here so
        # primary cosine runs unmodified. Both lists feed STEP 4.5 so
        # alternates are searched via BOTH cosine (alternate_vecs) AND FTS5
        # keyword (alternate_strings) — the latter catches lexical matches
        # cosine collapses on (e.g. "scratch grains" vs "feed").
        alternate_vecs = []
        alternate_strings = []

        # STEP 2.5: Wire situation matching — query IS the situation context.
        # 1085 nodes have situation embeddings describing WHEN they're relevant.
        # Scoring logic exists in STEP 3.5b but situation_vec was never passed.
        if situation_vec is None:
            situation_vec = query_vec

        # STEP 2.7: LAF v1 challenger scorer — flag-gated (§19 P1).
        # BRAIN_RECALL_VARIANT=laf_v1 → recall_laf computes the full field score
        # per node (maxsim + episodic pick/enc + idf + situation lanes, sigmoid-
        # squashed to (0,1)) and this loop INJECTS it as each node's `sim` below,
        # so filters, fatigue, critical boost, floors, hydration and tracing are
        # inherited from the champion path. The channels the field REPLACES
        # (z-weighted groups, situation scan, FTS5 net, keyword blend, idf2 title
        # boost, trace-chain lane) are gated off under the flag at their sites.
        # Flag unset / scorer failure → _laf_scores is None → champion unchanged.
        # Session context — resolved ONCE for the whole recall: the proj
        # lane's query-side project (STEP 2.7) and the fatigue snapshot
        # (STEP 3) both read it. Previously the ctx-less MCP path loaded the
        # SessionContext twice (session_env_for here + get_or_create_session
        # in STEP 3), a redundant logs-db blob parse per recall.
        _recall_ctx = ctx
        if _recall_ctx is None and session_id:
            try:
                _recall_ctx = self.get_or_create_session(session_id)
            except Exception as _ferr:
                self._log_error('recall_fatigue_ctx', _ferr,
                                'loading session ctx for fatigue dampening — '
                                'falling back to empty fatigue')

        _laf_scores = None
        _laf_fields = None
        import os as _os_laf
        _laf_on = (_os_laf.environ.get('BRAIN_RECALL_VARIANT', '')
                   .strip().lower() == 'laf_v1')
        # as_of (§20.11 replay time-travel) is a LAF-only capability: the
        # champion channels have no masks, so an as_of recall that fell back
        # to champion would silently score against TODAY's corpus and corrupt
        # the replay. Refuse loudly instead (Leg B's harness treats this as a
        # broken-run signal, never a data point).
        if as_of is not None and (not _laf_on or include_archived):
            raise ValueError('as_of recall requires BRAIN_RECALL_VARIANT='
                             'laf_v1 and include_archived=False — the champion '
                             'path cannot time-travel')
        # include_archived falls back to champion: the LAF engine's universe is
        # deliberately live-only (vectors_since / get_all_vectors exclude
        # archived — a performance choice, don't widen it), so archived
        # candidates would all score _laf_scores.get(id, 0.0) = dead-last.
        # Champion cosines them correctly, and these calls are rare and
        # not latency-critical.
        if _laf_on and not include_archived:
            try:
                try:
                    from .recall_laf import get_engine as _laf_get_engine
                except ImportError:
                    from recall_laf import get_engine as _laf_get_engine
                # Session project — the query-side source for the proj lane
                # (deterministic provenance from ctx, never from query text).
                _session_project = (_recall_ctx.project
                                    if _recall_ctx is not None else None)
                _laf_scores, _laf_fields = _laf_get_engine(self).scores(
                    self, query, query_vec, model=_active_model,
                    session_project=_session_project,
                    as_of=as_of, session_id=session_id)
                if not _laf_scores:
                    _laf_scores = None      # empty field → champion, not zeros
                if as_of is not None and _laf_scores is None:
                    # under time-travel there is no champion to fall back to
                    raise ValueError('as_of recall got no field scores '
                                     '(empty masked universe or scorer '
                                     'failure) — refusing champion fallback')
            except ValueError:
                raise
            except Exception as _laf_e:
                self._log_error('recall_laf', _laf_e,
                                'laf_v1 scoring failed — champion fallback')
                _laf_scores = None
                _laf_fields = None

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
            _brain_size = self._nodes.count()
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
                types=_types_filter,
                model=_active_model)
            # Hoisted out of the per-row loop — these were `if not hasattr`
            # guards behind the cosine inner loop, so paid N times for no
            # reason. Now paid once. Idempotent if Brain.warm_up() already
            # built the structural cache during daemon boot.
            self._ensure_structural_degree_cache()
            # Per-session fatigue snapshot — _recall_ctx was resolved once
            # before STEP 2.7 (shared with the proj lane's session project).
            # ctx is saved at end of _recall_impl so fatigue increments (via
            # _mark_accessed below) persist for the next recall this session.
            _recall_fatigue: Dict[str, int] = {}
            if _recall_ctx is not None:
                _recall_fatigue = _recall_ctx.fatigue

            for row in emb_rows:
                node_id = row['node_id']
                blob = row['embedding']
                node_personal_data[node_id] = (row['personal'], row['personal_context'])
                node_confidence[node_id] = row['confidence']
                node_critical[node_id] = row['critical']
                node_titles[node_id] = row['title']
                node_types[node_id] = row['type']
                if blob:
                    if _laf_scores is not None:
                        # laf_v1: the field score IS the similarity — cosine and
                        # the lexical bridge are subsumed by the field's lanes.
                        # RANGE CONTRACT: values are sigmoid ∈ (0,1) (validated
                        # in recall_laf.scores) — the fatigue multiply below and
                        # STEP 6's floors/boosts assume cosine-like magnitudes.
                        sim = _laf_scores.get(node_id, 0.0)
                    else:
                        sim = embedder.cosine_similarity(query_vec, blob)
                        # Lexical bridge: take max cosine across all phrasings
                        # (primary + LLM-expanded alternates). This is the
                        # mechanism that makes "uncle's birthday party" reach
                        # "niece's birthday party" nodes — the model generated
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
                    # this session. _recall_fatigue is the per-call snapshot
                    # of this session's ctx.fatigue (loaded above). Resets
                    # between sessions.
                    # K (fatigue resistance) scales with structural degree:
                    #   Hub (30 edges): K=2.5, fatigues fast
                    #   Peripheral (3 edges): K=7.7, fatigues slow
                    #   New node (0 edges): K=10, barely fatigues
                    _fatigue_count = _recall_fatigue.get(node_id, 0)
                    if _fatigue_count > 0:
                        _degree = self._structural_degree_cache.get(node_id, 0)
                        _K = 10.0 / (1.0 + _degree / 10.0)
                        _fatigue = _fatigue_count / (_fatigue_count + _K)
                        sim *= (1.0 - _fatigue)

                    embedding_scores[node_id] = sim
                    primary_scores[node_id] = sim
                    enrichment_hits[node_id] = 'primary'
                    nodes_with_embeddings += 1

            # STEP 3.1: Conditional lexical bridge — LLM query expansion
            # gated on cosine flatness. The cost of expansion is dominated
            # by the LLM call (~800ms). Most queries have a clear winner
            # in primary cosine and don't need it. Only when scores are
            # genuinely flat does the lexical bridge add value.
            #
            # Modes (BRAIN_QUERY_EXPANSION env var):
            #   off (default)     — no expansion, no LLM call
            #   on_flat           — expand only when top1 - top10 < gate
            #   on                — expand always (research / eval; expensive)
            #
            # When expansion fires: re-iterate emb_rows with alternate
            # vectors, taking max cosine. primary_scores updated in place
            # so STEP 3.5 sees the boosted scores.
            import os as _os
            _expansion_mode = _os.environ.get(
                'BRAIN_QUERY_EXPANSION', 'off').lower()
            if _laf_scores is not None:
                _expansion_mode = 'off'   # laf_v1: field scores replace cosine —
                #                           alternates would max() against them
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
                    # Cheap (one stderr line per recall, no LLM call).
                    print('[recall] on_flat gate: top1=%.3f top10=%.3f '
                          'spread=%.3f gate=%.3f → %s' % (
                            _sorted[0], _sorted[9], _spread, _gate,
                            'expand' if _do_expand else 'skip'),
                          file=sys.stderr)
                else:
                    _do_expand = True  # too few candidates → expand

            # Availability gate. Every other LLM lane checks this before
            # spending a call that can only 401 — S1 Scribe and S2 (brain.py),
            # surface (daemon_hooks), keepalive (daemon_server), voice
            # (brain_voice). Expansion was the one site that didn't, so a
            # keyless brain — or one whose key the provider has refused and the
            # rejection latch has paused — still fired here, into a bare
            # `except` that swallowed the 401 silently. The gates above are
            # QUALITY gates (is expansion worth it); this is the AVAILABILITY
            # gate (can it run at all).
            #
            # Operand order is load-bearing: `_do_expand` first means
            # self.llm_available — which re-reads the key file on every access
            # — is touched ONLY when expansion is actually enabled. Reversing
            # the operands would put a stat + read on every recall, which
            # resolve_api_key's contract explicitly excludes ("not the recall
            # hot path"). tests/test_recall_query_expansion.py pins the order.
            if _do_expand and not self.llm_available:
                # note_llm_unavailable records the keyless state ONCE per
                # daemon lifetime, so it cannot explain this particular skip —
                # and the on_flat block above has already printed its
                # "→ expand" decision. Without this line the tuning log shows
                # an expansion that silently never happened.
                print('[recall] query-expansion skipped: no usable API key',
                      file=sys.stderr)
                self.note_llm_unavailable('query expansion')
                _do_expand = False

            if _do_expand:
                try:
                    _alts = _expand_query_via_llm(self, expanded_query)
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
                                    'LLM query expansion failed — '
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
            # (laf_v1: STEP 3.5 gated off — the field score already fuses the
            # views; empty inputs make every loop below a no-op.)
            for nid, prim_sim in (primary_scores.items()
                                  if _laf_scores is None else ()):
                if nid not in node_vector_scores:
                    node_vector_scores[nid] = []
                node_vector_scores[nid].append((_blend_weight * prim_sim, '_primary'))

            try:
                _enrich_rows = _vec_dal.get_all_vectors(
                    exclude_archived=not include_archived,
                    model=_active_model or None) if _laf_scores is None else []
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
                # (A/B'd vs max/maxbonus 2026-06-09: gold-neutral on the control
                # corpus — burial lives in the STEP 6 title boost, not here.)
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
            # (laf_v1: skipped — 3.5 never overwrites, STEP 3 already fatigued
            # the injected field score once.)
            if _recall_fatigue and _laf_scores is None:
                for nid in list(embedding_scores.keys()):
                    _fc = _recall_fatigue.get(nid, 0)
                    if _fc > 0:
                        _deg = self._structural_degree_cache.get(nid, 0) if hasattr(self, '_structural_degree_cache') else 0
                        _K = 10.0 / (1.0 + _deg / 10.0)
                        _fat = _fc / (_fc + _K)
                        embedding_scores[nid] *= (1.0 - _fat)

        except Exception as e:
            print(f'[brain] Embedding scan error: {e}', file=sys.stderr)

        # STEP 3.5: Situation scan — boost nodes whose situation matches current context
        # (laf_v1: gated off — situation is a gain-weighted lane inside the field)
        situation_scores = {}
        if situation_vec and _laf_scores is None:
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
                                    min_recency, session_id, _skip_log=True,
                                    mark_accessed=mark_accessed)
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
            fts5_dal = self._fts
            # laf_v1: FTS5 net gated off — the field scores the full universe
            # (no discovery needed) and the fts lane measured harmful in-stack
            # at static gains (eval/laf/composition_probe.md); P4's per-query
            # gate is the path back in.
            _fts5_queries = ([query] + alternate_strings
                             if _laf_scores is None else [])
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

        # STEP 4.6: Trace-chain lane (episodic dual-store rescue) — flag-gated, additive, default OFF.
        # docs/RECALL-DUAL-STORE-DESIGN.md §3.3 form 1. Off -> trace_chain_scores empty -> zero impact.
        # exclude_ids = fts5_only only (its own reserved lane). We deliberately do NOT exclude
        # embedding/keyword hits: the buried EX.CO nodes ARE in embedding_scores (scored but below the
        # cut) — rescuing them from below the cut is the whole point. Dedup vs the main TOP is at merge.
        import os as _os_tc
        trace_chain_scores = {}
        if (_os_tc.environ.get('BRAIN_TRACE_CHAIN', '') == '1'
                and _laf_scores is None):   # laf_v1: episodic lanes supersede
            trace_chain_scores = self._trace_chain_candidates(query_vec, exclude_ids=fts5_only_ids)

        # STEP 5: Build unified candidate set (all nodes seen by any path)
        all_candidate_ids = set(embedding_scores.keys()) | set(keyword_scores.keys()) | fts5_only_ids
        if as_of is not None and _laf_scores is not None:
            # as_of chokepoint: the masked field's score map IS the as-of
            # universe — nodes created after the cue (reachable here only via
            # the keyword fallback for embedding-less nodes) must not leak
            # into a replay candidate set.
            all_candidate_ids &= set(_laf_scores)

        # STEP 6: Score each candidate — embeddings primary, keywords fallback
        #
        # BRAIN_TITLE_BOOST A/B knob (2026-06-09, see eval/oracle_audit/ab_boost_decomp.py):
        #   'add' — production: boost = (matched/|terms|) × TITLE_MATCH_BOOST, raw
        #           whitespace terms (punctuation kept), substring containment.
        #           Verified failure mode on episodic queries: flood terms ('on'
        #           hits 98/100 titles, 'session' 82/100) lift low-cosine nodes
        #           +0.18 while the discriminative term ('ex.co?') matches nothing
        #           — buried gold dabb3078 at rank 92 with rank-12 cosine.
        #   'off' — no title boost (null arm).
        #   'idf' — punctuation-stripped terms, each weighted by rarity across
        #           node titles (log idf); flood terms ≈ 0, rare terms dominate.
        #   'idf2' — idf + three calibration fixes from the TO1/TO4/TO6 decomp
        #           (ab_topic_decomp.py): real tokenization (keeps 'ex.co',
        #           'spread_activation'; kills the em-dash df=2303 pseudo-term),
        #           stopword floor (df-over-titles misprices conversational
        #           words — 'does' df=58 looked as rare as 'fatigue' df=41),
        #           and word-boundary matching ('do' no longer hits 'docs').
        # Default flipped add → idf2 (2026-06-09) after: control-corpus A/B
        # (fails 15→13, top5 59→67%, top25 81→88%), held-out recall_corpus_v2
        # (zero regressions, terse+rich), 43/43 recall+contract tests green.
        import os as _os_tb
        _title_boost_mode = _os_tb.environ.get('BRAIN_TITLE_BOOST', 'idf2').strip().lower()
        if _laf_scores is not None:
            _title_boost_mode = 'off'   # laf_v1: idf2 is a gain-weighted lane inside the field
        _title_idf = None
        _idf_total = 1.0
        _title_tok = None
        if _title_boost_mode == 'idf' and _query_terms_set:
            import string as _string
            _clean_terms = {t.strip(_string.punctuation) for t in _query_terms_set}
            _clean_terms.discard('')
            _titles_l = [t.lower() for t in node_titles.values() if t]
            _n_titles = max(len(_titles_l), 1)
            _title_idf = {}
            for _t in _clean_terms:
                _df = sum(1 for _ti in _titles_l if _t in _ti)
                _title_idf[_t] = math.log((_n_titles + 1) / (_df + 1))
            _idf_total = sum(_title_idf.values()) or 1.0
        elif _title_boost_mode == 'idf2' and query:
            import re as _re_tb
            # Dots/underscores join identifiers (ex.co, spread_activation, v15.2);
            # hyphens join prose words ("Scouts-in-examples") and must SPLIT, or
            # compound title words match none of their parts.
            _tok = _re_tb.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")
            _q_tokens = {t for t in _tok.findall(query.lower())
                         if len(t) >= 2 and t not in _TITLE_BOOST_STOPWORDS}
            if _q_tokens:
                _title_tok = {nid: frozenset(_tok.findall(t.lower()))
                              for nid, t in node_titles.items() if t}
                _n_titles = max(len(_title_tok), 1)
                _title_idf = {}
                for _t in _q_tokens:
                    _df = sum(1 for _ts in _title_tok.values() if _t in _ts)
                    _title_idf[_t] = math.log((_n_titles + 1) / (_df + 1))
                _idf_total = sum(_title_idf.values()) or 1.0
        scored_results = []
        for nid in all_candidate_ids:
            emb_score = embedding_scores.get(nid, 0)
            kw_score = keyword_scores.get(nid, 0)

            # Determine discovery source and compute blended score
            if _laf_scores is not None and emb_score > 0:
                # laf_v1: the field score stands alone — no keyword blending
                # (lexical signal is the idf lane inside the field).
                blended = emb_score
                discovery = 'laf_v1'
            elif nid in fts5_only_ids:
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

            node = keyword_nodes.get(nid)

            # v8.8: Title-match boost — proportional to query/title word overlap.
            # If query terms appear in the node's title, strong relevance signal.
            # Gated by BRAIN_TITLE_BOOST (see knob comment at STEP 6 head).
            from .brain_constants import TITLE_MATCH_BOOST
            title = node_titles.get(nid, '').lower()
            if title and _query_terms_set and _title_boost_mode != 'off':
                if _title_tok is not None:  # 'idf2' — word-boundary token match
                    _ts = _title_tok.get(nid)
                    if _ts:
                        matched_idf = sum(w for t, w in _title_idf.items() if t in _ts)
                        if matched_idf > 0:
                            blended += (matched_idf / _idf_total) * TITLE_MATCH_BOOST
                elif _title_idf is not None:  # 'idf' — substring match
                    matched_idf = sum(w for t, w in _title_idf.items() if t in title)
                    if matched_idf > 0:
                        blended += (matched_idf / _idf_total) * TITLE_MATCH_BOOST
                else:  # 'add' — production
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

        # Scope veil — BEFORE the [:limit] cut, so eligible lower-ranked
        # candidates backfill the slots walled nodes vacate (gating after
        # truncation starves isolated sessions — the seen_dedup_headroom
        # class, a98143f finding 1). One set-membership check per candidate;
        # empty frozenset (no isolation configured) costs nothing.
        _veil = self.scope_veil(session_id or '')
        _iso_dropped = []
        if _veil:
            _kept = [r for r in scored_results if r['node_id'] not in _veil]
            _iso_dropped = [r['node_id'] for r in scored_results
                            if r['node_id'] in _veil]
            scored_results = _kept

        if trace_chain_scores:
            # Reserved tail (§4): rescue trace-chain nodes that did NOT make the main top. Additive —
            # never reorders the main top. A buried node (scored low here, below the cut) is PROMOTED
            # via its trace-chain combined score; one already in the main top is left alone (dedup).
            # Fresh candidate dicts so the rescue survives the [:limit] cut (the bug the fts5 lane hits,
            # finding 703a9402 — here the reserved slots are guaranteed before truncation).
            from .brain_constants import TRACE_CHAIN_RESERVE
            _k = TRACE_CHAIN_RESERVE
            _main_top = scored_results[:max(0, limit - _k)]
            _have = {r['node_id'] for r in _main_top}
            _rescues = []
            for _nid, _comb in sorted(trace_chain_scores.items(), key=lambda x: -x[1]):
                if _nid in _have or _nid in _veil:
                    continue
                _rescues.append({'node_id': _nid, 'blended_score': _comb, '_source': 'trace_chain',
                                 'embedding_similarity': None, 'keyword_score': None,
                                 '_context_mismatch': False})
                if len(_rescues) >= _k:
                    break
            scored_results = _main_top + _rescues
        else:
            scored_results = scored_results[:limit]

        # STEP 6.9: Per-result relevance floor.
        # v8.7: Changed from all-or-nothing (top result gates everything) to per-result.
        # Each result must meet its own floor based on how it was discovered.
        # v9: FTS5-only candidates bypass the relevance floor — they go straight to surfacer.
        scored_results = [
            sr for sr in scored_results
            if sr['_source'] in ('fts5_only', 'trace_chain')  # reserved lanes: always pass to surfacer
            or sr['blended_score'] >= (
                RELEVANCE_FLOOR_ENRICHED
                if enrichment_hits.get(sr['node_id'], 'primary') != 'primary'
                else RELEVANCE_FLOOR_PRIMARY
            )
        ]

        # STEP 7: Hydrate full node data for top results
        final_results = []
        # Pre-batch the embedding-only nodes (those not already cached in
        # keyword_nodes) into one query instead of N get_naked_node calls
        # (C1 / H2). The per-node post-processing below is unchanged.
        _hydrate_ids = [sr['node_id'] for sr in scored_results
                        if sr['node_id'] not in keyword_nodes]
        try:
            _hydrated = self._nodes.get_bulk(_hydrate_ids) if _hydrate_ids else {}
        except Exception as e:
            self._log_error("recall_hydrate", e,
                            "Failed to bulk-hydrate %d nodes" % len(_hydrate_ids))
            _hydrated = {}
        # Default type exclusions (pipeline_contract registry). An explicit
        # dict filter on type overrides — a deliberate community query must
        # not come back empty because of the default.
        from .pipeline_contract import get_excluded_types
        _excluded_types = get_excluded_types('recall') \
            if not (filter and 'type' in filter) else set()
        for sr in scored_results:
            nid = sr['node_id']
            node = keyword_nodes.get(nid)
            if not node:
                # Node came from embedding-only path — pre-batched above.
                node = _hydrated.get(nid)

            if node and (node.get('type') or '') in _excluded_types:
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

        # (Scope veil already applied pre-[:limit] — candidates were gated
        # before truncation, and _iso_dropped carries the walled ids.)

        # No enrichment here: recall scores and returns, and each door
        # canonicalizes what it hands out (canonicalize_results). The hook
        # path runs its own get_node pass over the capped candidate pool, so
        # enriching a top-3 slice here only ever produced a shape a
        # downstream consumer immediately overwrote.

        # STEP 8: Mark accessed (recognition signal + fatigue)
        # Per-session: _recall_ctx (loaded at top of this function) is passed
        # to _mark_accessed so fatigue increments land on the right session.
        #
        # mark_accessed=False makes this a pure READ. It exists for observers
        # (the dashboard's recall probe) that need the real pipeline's answer
        # without becoming part of the brain's own history: access_count,
        # last_accessed and fatigue are what the graph renders as recall heat
        # and what LAF scores against, so an observer that marked them would
        # be measuring its own looking. Every in-brain caller leaves this True.
        if mark_accessed:
            sid = session_id or self.session_id
            for node in final_results:
                try:
                    self._mark_accessed(node['id'], sid, ctx=_recall_ctx)
                except Exception as _e:
                    self._log_error("recall", _e, "marking node as accessed")

        # Fatigue increments live on the cached SessionContext (mutations
        # in memory). Persistence happens via the daemon autosave loop
        # every AUTOSAVE_INTERVAL_SECONDS — fatigue is approximate and
        # self-healing, so per-recall saves are unnecessary churn.

        # STEP 9: Log recall to recall_log (single source of truth)
        recall_ms = (time.time() - t0) * 1000
        # recall_log writes REMOVED 2026-04-05 — S1 traces capture all recall data.
        # Previously inserted into recall_log here. Traces (O/K/Δ) in daemon_hooks
        # are the single source of truth for recall events.

        # Build result
        result = {
            'results': final_results,
            '_recall_mode': 'laf_v1' if _laf_scores is not None else 'embeddings_first',
            '_embedding_stats': {
                'embedder_ready': True,
                'nodes_with_embeddings': nodes_with_embeddings,
                'nodes_without_embeddings': nodes_without_embeddings,
                'embedding_primary_weight': EMBEDDING_PRIMARY_WEIGHT,
                'keyword_fallback_weight': KEYWORD_FALLBACK_WEIGHT,
                'recall_ms': round(recall_ms, 1),
                'results_by_source': {
                    'laf_v1': sum(1 for r in final_results if r.get('_discovery') == 'laf_v1'),
                    'embedding+keyword': sum(1 for r in final_results if r.get('_discovery') == 'embedding+keyword'),
                    'embedding_only': sum(1 for r in final_results if r.get('_discovery') == 'embedding_only'),
                    'keyword_only_fallback': sum(1 for r in final_results if r.get('_discovery') == 'keyword_only_fallback'),
                    'fts5_only': sum(1 for r in final_results if r.get('_discovery') == 'fts5_only'),
                    'trace_chain': sum(1 for r in final_results if r.get('_discovery') == 'trace_chain'),
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
                    'laf_v1': sum(1 for sr in scored_results if sr['_source'] == 'laf_v1'),
                    'embedding_only': sum(1 for sr in scored_results if sr['_source'] == 'embedding_only'),
                    'embedding+keyword': sum(1 for sr in scored_results if sr['_source'] == 'embedding+keyword'),
                    'fts5_only': sum(1 for sr in scored_results if sr['_source'] == 'fts5_only'),
                    'both': sum(1 for sr in scored_results if sr['_source'] == 'both'),
                    'keyword_only_fallback': sum(1 for sr in scored_results if sr['_source'] == 'keyword_only_fallback'),
                } if 'scored_results' in dir() else {},
            },
        }

        # Scope-veil observability — present only when the wall dropped
        # something, so the legacy result shape is unchanged for unscoped
        # brains.
        if _iso_dropped:
            result['_scope_isolated_dropped'] = len(_iso_dropped)

        # laf_v1 telemetry: per-candidate field z-scores for the top nodes —
        # rides the result into the S1R trace so the P2 dataset walker accretes
        # (query, per-field scores, outcome) rows in production for free.
        if _laf_fields is not None:
            result['_laf_fields'] = _laf_fields

        # Gap detection: when no results pass the relevance floor,
        # flag it so the voice layer can prompt encoding
        if not final_results and query:
            result['_gap'] = {
                'query': query,
                'top_score': max_score if 'max_score' in locals() else 0,
            }

        # v5.1: Return query embedding for segment boundary detection
        # Zero cost — already computed in STEP 1
        result['_query_embedding'] = query_vec

        return result

    # _traverse_graph removed 2026-04-14 — dead code, 0 callers.
    # S1R uses _graph_expand() in surface.py instead.

    def recall_node(self, node_id: str, session_id: str = '') -> Dict[str, Any]:
        """Recall a specific node by ID with full enrichment.

        Returns same shape as recall() so callers get a
        consistent interface regardless of how the node was found.

        By-id reach is the veil's sanctioned open door — the NODE returns
        regardless of scope. Its ATTACHMENTS are still scrubbed, inside
        canonicalize_results: reaching for a known id says nothing about
        the neighbors and correctors it happens to be linked to.
        """
        node = self._nodes.get_naked_node(node_id)
        if not node:
            return {'results': []}
        # An absorbed id needs no handling here: canonicalize_results below
        # swaps the row to the live survivor (marked) — the one door owns
        # the redirect. A retired corpse passes through, honestly archived.

        # Set display fields for format compatibility
        node['effective_activation'] = node.get('activation', 0)
        node['embedding_similarity'] = None
        node['_keyword_score'] = None
        node['_source'] = 'direct_lookup'

        results = [node]
        self.canonicalize_results(results, session_id)

        return {
            'results': results,
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
            fts5_dal = self._fts
            node_ids = fts5_dal.search(query, limit)
            _ndal = self._nodes
            _bulk = _ndal.get_bulk(node_ids)
            results = []
            for nid in node_ids:
                node = _bulk.get(nid)
                if node and not node.get('archived'):
                    results.append(node)
            return results
        except Exception as e:
            self._log_error('search_keywords_fts5', e, 'FTS5 search failed, falling back to empty')
            return []

    def _mark_accessed(self, node_id: str, session_id: str, ctx=None):
        """Enqueue a recognition signal for this node + session pair.

        Architecture (2026-05-18, Phase 5 of bg_writer migration):
        - No DB I/O on the hot path. Enqueues (node_id, session_id, ts)
          into `recall_write_queue`. The background worker drains every
          EMBED_DRAIN_INTERVAL seconds via `brain.conn_bg_writer`,
          producing one atomic +1 UPDATE per unique (node, session)
          pair (dedup'd by Dict semantics within the drain window).
        - Recall is now read-only at SQLite. No `conn_recall_write`,
          no per-recall commit, no busy_timeout exposure on the hot
          path.

        Parallel-session correctness: fatigue increments go to the
        SessionContext passed in by the caller. When ctx is None,
        fatigue is dropped for this access — the access mark still
        enqueues, just no per-session fatigue feedback.
        """
        try:
            from . import recall_write_queue
            recall_write_queue.enqueue_access(node_id, session_id, self.now())
        except Exception as e:
            self._log_error('mark_accessed_enqueue', e,
                            'enqueue failed for node=%s session=%s' %
                            (node_id[:12] if node_id else '', session_id))

        # Increment session fatigue counter — next recall will dampen this
        # node's cosine. ctx is the per-call SessionContext; mutations are
        # saved by the caller at end of recall (see _recall_impl).
        if ctx is not None:
            try:
                ctx.increment_fatigue(node_id)
            except Exception as e:
                self._log_error('mark_accessed_fatigue', e,
                                'fatigue increment failed for node=%s' %
                                (node_id[:12] if node_id else ''))
            # Per-session node activity — parallel-session replacement for
            # global nodes.{activation,recency_score,last_accessed,access_count}
            # in reads that should be session-scoped (spreading-activation
            # kernel, recency filtering, live-session Frame composition).
            # In-memory; persisted with the SessionContext save at end of
            # recall. Global nodes columns still bumped by recall_write_queue
            # drain for S2 maintenance + dashboard analytics.
            try:
                ctx.bump_node_activity(node_id, self.now())
            except Exception as e:
                self._log_error('mark_accessed_activity', e,
                                'node_activity bump failed for node=%s' %
                                (node_id[:12] if node_id else ''))

    # _hebbian_strengthen REMOVED 2026-05-18 (Phase 5); the surface-picks
    # successor was retired with the whole co_accessed family 2026-08-17
    # (node ab56d25a) — surface_selected traces are the co-access substrate.

    # _log_recall REMOVED 2026-04-05 — recall_log writes deprecated, traces are source of truth

    def _get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently accessed nodes."""
        sql = 'SELECT id FROM nodes WHERE archived = 0'
        params = []
        sql += ' ORDER BY last_accessed DESC LIMIT ?'
        params.append(limit)

        _ndal = self._nodes
        _ids = [row[0] for row in self.conn.execute(sql, params).fetchall()]
        _bulk = _ndal.get_bulk(_ids)
        results = []
        for nid in _ids:
            node = _bulk.get(nid)
            if node:
                node['spread_activation'] = node.get('activation', 0)
                node['effective_activation'] = node.get('activation', 0)
                results.append(node)
        return results

