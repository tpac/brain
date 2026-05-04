"""Surface Contract — S1 surface (Haiku) prompt building, candidate formatting, output formatting.

S1 Surface pushes relevant memories into awareness. This contract defines:
- What the surfacer sees (SURFACE config, CANDIDATES_FILE, neighbor fields)
- How candidates are formatted (format_candidate_for_surface, enrich_candidate_metadata)
- How the prompt is assembled (build_surface_prompt)
- How output is formatted for Claude (format_surface_output)
- Correction enrichment (correction_enrich — shared with encoding)

Interaction: 'surface' in interactions table. Prompt is learnable.
"""

import threading
from datetime import datetime, timezone
from collections import OrderedDict

# Hoisted: any embedder API drift fails at daemon boot, not 16s into hook_recall.
from servers.embedder import embed_document, embed_batch
from servers import embedder as _embedder

# Bounded cache for edge-description embeddings. Same descriptions recur
# across turns and across sessions — caching eliminates redundant embed calls.
#
# Key: (model_name, description_text). Model-scoped so a model swap doesn't
# hand out vectors from the wrong geometry. Values are L2-normalized blobs.
# LRU eviction via OrderedDict; capped at 5000 entries (~15MB at 3KB/vector).
# All reads/writes guarded by _DESC_CACHE_LOCK so concurrent embed callers
# (future multi-threaded recall/backfill) can't corrupt LRU order.
_DESC_VEC_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_DESC_VEC_CACHE_MAX = 5000
_DESC_CACHE_LOCK = threading.Lock()
_DESC_CACHE_HITS = 0
_DESC_CACHE_MISSES = 0


def get_desc_cache_stats() -> dict:
    """Cache diagnostics — surfaced through embedder stats / logs."""
    with _DESC_CACHE_LOCK:
        total = _DESC_CACHE_HITS + _DESC_CACHE_MISSES
        hit_rate = (_DESC_CACHE_HITS / total) if total else 0.0
        return {
            'size': len(_DESC_VEC_CACHE),
            'max': _DESC_VEC_CACHE_MAX,
            'hits': _DESC_CACHE_HITS,
            'misses': _DESC_CACHE_MISSES,
            'hit_rate': round(hit_rate, 3),
        }


def _desc_vecs_batched(descs):
    """Resolve descriptions → normalized blobs, using cache + one batched embed.

    Returns list of blobs aligned with descs input. Empty strings return None.
    Keys the cache by active model name so model swaps don't hand out stale
    geometry. Dedupes within a single call so a batch with duplicates only
    pays one embed per unique text.
    """
    global _DESC_CACHE_HITS, _DESC_CACHE_MISSES
    if not descs:
        return []

    model = _embedder.stats.get('model_name') or ''
    out = [None] * len(descs)

    # Partition under lock: cached → out[i]; uncached → unique_texts + index map
    # (unique_texts is deduped so duplicate descriptions in one call only embed once)
    unique_texts: list = []
    text_to_unique_idx: dict = {}
    indices_for_unique: list = []  # list of lists of original indices per unique text
    hits = 0
    misses = 0
    with _DESC_CACHE_LOCK:
        for i, d in enumerate(descs):
            if not d:
                continue
            key = (model, d)
            blob = _DESC_VEC_CACHE.get(key)
            if blob is not None:
                out[i] = blob
                _DESC_VEC_CACHE.move_to_end(key)
                hits += 1
                continue
            misses += 1
            u_idx = text_to_unique_idx.get(d)
            if u_idx is None:
                u_idx = len(unique_texts)
                text_to_unique_idx[d] = u_idx
                unique_texts.append(d)
                indices_for_unique.append([i])
            else:
                indices_for_unique[u_idx].append(i)

    # Embed outside the lock — embedder call may be slow; we hold no state.
    if unique_texts:
        fresh = embed_batch(unique_texts, kind='document')
        # Populate out[] + cache under lock
        with _DESC_CACHE_LOCK:
            for u_idx, blob in enumerate(fresh):
                if blob is None:
                    continue
                for i in indices_for_unique[u_idx]:
                    out[i] = blob
                _DESC_VEC_CACHE[(model, unique_texts[u_idx])] = blob
            while len(_DESC_VEC_CACHE) > _DESC_VEC_CACHE_MAX:
                _DESC_VEC_CACHE.popitem(last=False)

    if hits or misses:
        with _DESC_CACHE_LOCK:
            _DESC_CACHE_HITS += hits
            _DESC_CACHE_MISSES += misses
    return out


def _relative_time(iso_str):
    """Convert UTC ISO timestamp to relative time label.

    Returns human-readable age: 'just now', 'today', 'yesterday', '3d ago', '2w ago', '1mo ago'.
    Both surface and Anchor see this instead of raw UTC timestamps.
    """
    if not iso_str:
        return None
    try:
        ts_str = str(iso_str).replace('Z', '+00:00')
        if '+' not in ts_str and ts_str.count('-') <= 2:
            ts_str += '+00:00'
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - ts
        hours = delta.total_seconds() / 3600
        days = delta.days

        if hours < 1:
            return "just now"
        elif hours < 24:
            return "today"
        elif hours < 48:
            return "yesterday"
        elif days < 7:
            return "%dd ago" % days
        elif days < 30:
            return "%dw ago" % (days // 7)
        else:
            months = days // 30
            return "%dmo ago" % months
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════
# SURFACE CONFIG
# ═══════════════════════════════════════════════════════════════

# Candidates file (written by daemon, read by surface + encoding agent)
CANDIDATES_FILE = {
    'content_limit': 1000,
    'max_candidates': 25,
    'include_graph': True,      # _graph with degree 1/2/3 neighbors
    'include_metadata': True,   # situation, reasoning, user_raw_quote, correction_of
    'metadata_fields': ['situation', 'reasoning', 'user_raw_quote', 'correction_of'],
    'max_edges_described': 3,   # top edges with descriptions per candidate
}

# Judge (Haiku) — selects relevant nodes with reasoning
# v9: max_candidates 25→20, Anchor truncation 150→400, recent_messages 5→7
SURFACE = {
    'content_limit': 300,           # shorter per node since more candidates
    'max_candidates': 30,           # 2026-05-01: was 20. FTS5 adds up to 5 more = 35 max total.
                                    # Bumped to give Haiku more axis-of-context room while
                                    # multi-axis candidate generation is being designed.
    'max_selected': 5,              # Haiku picks at most this many (was 8 — reduced for 10K hook cap)
    'user_message_limit': 300,
    'anchor_message_limit': 400,    # v9: was 150. Anchor responses carry design context
    'recent_messages': 7,           # v9: was 5. Deeper conversation window
    'recent_recalls_messages': 10,  # look back 10 messages for previously surfaced nodes
    'session_context_limit': 800,   # shared with ENCODING_AGENT — full session journey
    'session_context_tail': 800,  # 2026-05-02 (Frame Phase 1): was 200. Surface now gets
                                  # the full session_context blob, not just the tail.
                                  # Encoder writes ~768 chars; surface was seeing ~25%.
                                  # See docs/FRAME-DESIGN.md Phase 1.
    'max_tokens': 600,              # Haiku output cap
}


# ═══════════════════════════════════════════════════════════════
# NEIGHBOR FIELDS — what to show for graph neighbors
# ═══════════════════════════════════════════════════════════════

# Degree 1 neighbors (closest — richest display)
NEIGHBOR_D1_FIELDS = {
    'id', 'type', 'title', 'relation', 'confidence',
    'locked', 'revised_at', 'content_summary', 'weight',
}

# Degree 2 neighbors (breadcrumbs — moderate display)
NEIGHBOR_D2_FIELDS = {
    'id', 'type', 'title', 'relation',
}

# Degree 3 neighbors (hints — minimal display)
NEIGHBOR_D3_FIELDS = {
    'id', 'title',
}

# Truncation per neighbor field
NEIGHBOR_TRUNCATION = {
    'd1_title': 60,
    'd1_id': 12,
    'd1_content_summary': 150,
    'd2_title': 35,
    'd2_id': 8,
    'd3_title': 30,
    'd3_id': 8,
}


# DEPRECATED: use PIPELINE['recall_log_*'] instead.
PRECISION = {
    'title_limit': 100,       # → PIPELINE['recall_log_title']
    'snippet_limit': 150,     # → PIPELINE['recall_log_snippet']
    'query_limit': 500,       # → PIPELINE['recall_log_query']
}


# ═══════════════════════════════════════════════════════════════
# CANDIDATE ENRICHMENT — DEPRECATED 2026-04-07
# Replaced by get_rich_node() in pipeline_contract.py.
# Candidates now use unified get_rich_node shape everywhere.
# ═══════════════════════════════════════════════════════════════

def enrich_candidate_metadata(brain, node_id, node_data, config):
    """DEPRECATED 2026-04-07: Use brain.get_node() instead.

    Kept as stub for backward compatibility with eval scripts.
    """
    rich = brain.get_node(node_id)
    if rich:
        node_data.update({k: v for k, v in rich.items() if k not in node_data})


# ═══════════════════════════════════════════════════════════════
# CORRECTION ENRICHMENT — shared by surface and encoding
# ═══════════════════════════════════════════════════════════════

def correction_enrich(node_ids, db_conn):
    """Find corrections for a set of nodes. Both directions.

    Returns dict: {node_id: [{"id", "title", "direction"}]}
    - direction "corrected_by": this node was superseded by another
    - direction "corrects": this node corrects another

    Checks two data sources:
    1. edges table: relation='corrected_by' (bidirectional pairs)
    2. node_metadata: correction_of field (node → what it corrects)
    """
    if not node_ids or not db_conn:
        return {}

    corrections = {}  # node_id → list of correction info dicts
    try:
        placeholders = ','.join('?' for _ in node_ids)

        # 1. REMOVED: Edge-based correction lookup.
        # In an open relation system, corrections are tracked via the correction_of
        # metadata field, not hardcoded edge relation types. The encoder can use
        # any relation name (corrects, supersedes, challenges, replaces, etc.)
        # and those show up naturally in edge rendering.
        # Correction_of metadata (below) is the authoritative signal.

        # 2. node_metadata_kv: correction_of field (forward: which of our nodes correct something)
        from servers.dal import NodeDAL
        dal = NodeDAL(db_conn)

        meta_rows = db_conn.execute(
            """SELECT node_id, value FROM node_metadata_kv
               WHERE key = 'correction_of' AND node_id IN (%s)
               AND value IS NOT NULL AND value != ''""" % placeholders,
            list(node_ids)
        ).fetchall()

        for nid, corrects_id in meta_rows:
            title = dal.get_title(corrects_id[:8])
            if title:
                corrections.setdefault(nid, []).append({
                    "id": corrects_id[:8], "title": title, "direction": "corrects"})

        # 3. Reverse: find nodes that correct OUR nodes (via correction_of field)
        meta_reverse = db_conn.execute(
            """SELECT node_id, value FROM node_metadata_kv
               WHERE key = 'correction_of'
               AND value IS NOT NULL AND value != ''"""
        ).fetchall()
        for nid, corrects_id in meta_reverse:
            for our_id in node_ids:
                if our_id.startswith(corrects_id[:8]) or corrects_id.startswith(our_id[:8]):
                    title = db_conn.execute(
                        "SELECT title FROM nodes WHERE id = ?", (nid,)).fetchone()
                    if title:
                        corrections.setdefault(our_id, []).append({
                            "id": nid[:8], "title": title[0], "direction": "corrected_by"})

    except Exception:
        pass

    # Deduplicate per node
    for nid in corrections:
        seen = set()
        deduped = []
        for c in corrections[nid]:
            key = (c["id"], c["direction"])
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        corrections[nid] = deduped

    return corrections


# ═══════════════════════════════════════════════════════════════
# FORMATTERS
# ═══════════════════════════════════════════════════════════════

def format_candidate_for_surface(c, index):
    """Format a single candidate for the surface prompt.

    Thin wrapper around render_rich_node(HAIKU_FORMAT) — adds recall-specific
    fields (score, discovery) that aren't in the node itself.

    Candidates must be in get_rich_node() shape (with _metadata, _corrections, connections).
    """
    from servers.contract import render_rich_node

    # Recall-specific header (score + discovery — not part of the node)
    score_parts = []
    score = c.get("score", 0)
    if score:
        display_score = min(score, 1.0)
        score_str = "match:%.2f" % display_score
        if score > 1.0:
            score_str += ",boosted"
        score_parts.append(score_str)
    discovery = c.get("discovery", "")
    if discovery and discovery not in ("embedding", "embedding_only", "embedding+keyword"):
        score_parts.append("via:%s" % discovery)

    header = "#%d" % index
    if score_parts:
        header += " (%s)" % ", ".join(score_parts)

    return header + "\n" + render_rich_node(c, HAIKU_FORMAT)


def _dedup_candidates(candidates):
    """Remove near-duplicate candidates by title similarity.
    Keeps the higher-scored candidate when two titles share >80% words."""
    if len(candidates) <= 1:
        return candidates
    seen_titles = {}  # normalized title words → candidate
    result = []
    for c in candidates:
        title_words = set(c.get("title", "").lower().split())
        duplicate = False
        for seen_key, seen_c in list(seen_titles.items()):
            seen_words = set(seen_key.split())
            if not title_words or not seen_words:
                continue
            overlap = len(title_words & seen_words) / max(len(title_words), len(seen_words))
            if overlap > 0.8:
                # Keep the higher scored one
                if (c.get("score", 0) or 0) > (seen_c.get("score", 0) or 0):
                    result.remove(seen_c)
                    seen_titles[" ".join(sorted(title_words))] = c
                    result.append(c)
                duplicate = True
                break
        if not duplicate:
            key = " ".join(sorted(title_words))
            seen_titles[key] = c
            result.append(c)
    return result


def build_surface_prompt(candidates, user_message,
                       recent_messages=None, recently_recalled=None,
                       retrieval_stats=None, intent=None, frame=""):
    """Build the S1 recall surface USER message — per-turn delta only.

    v11 (2026-05-03, Frame Phase 2.5 / surface prompt v2): instructions
    moved to the cached system block (in `_call_surface`). This function
    now builds ONLY the per-turn user content: Frame, conversation,
    recently surfaced, retrieval stats, intent, candidates. The registered
    `surface` interaction template is the system block; this is the user
    block. Two parts → two-block API call → caching becomes possible.

    Operator name is rendered generically as "Operator:" — the brain plugin
    ships to different operators, prompts must not hardcode personal names.

    Args:
        candidates: List of candidate node dicts (enriched with metadata)
        user_message: The operator's latest message
        recent_messages: List of {"role": str, "content": str}
        recently_recalled: List of {"id": str, "title": str} from last N recalls
        retrieval_stats: Dict with brain_size, top_score, median_score, source_breakdown
        intent: Query intent from STEP 2 classification
        frame: Markdown Frame (Anchor's prior). When non-empty becomes the
            "Partnership context:" block. When empty, explicit degraded marker.

    Returns: (user_content_string, max_tokens)
    """
    cfg = SURFACE

    # v9: Deduplicate candidates (remove near-identical titles)
    candidates = _dedup_candidates(candidates[:cfg['max_candidates']])

    # Format conversation context (both roles, asymmetric truncation).
    # 2026-05-03: "Operator:" is generic — prompts ship to different operators.
    conversation = ""
    if recent_messages:
        for msg in recent_messages[-(cfg['recent_messages']):]:
            role = msg.get("role", "?")
            if role == "user":
                label = "Operator"
                content = (msg.get("content") or "")[:cfg['user_message_limit']]
            else:
                label = "Anchor"
                content = (msg.get("content") or "")[:cfg['anchor_message_limit']]
            conversation += "%s: %s\n" % (label, content)

    # Append current user message (not yet in message_stream — stored on Stop)
    if user_message:
        conversation += "Operator: %s\n" % (user_message or "")[:cfg['user_message_limit']]

    # 2026-05-02 (Frame Phase 2): Frame is the canonical session prior.
    # When non-empty, it's the "Partnership context:" block — rich content
    # carrying operator + partnership + active-threads + current focus +
    # recent moves. When empty (Frame Constructor failed), surface runs
    # without any partnership context — explicit degraded mode, logged
    # upstream by daemon_hooks. No silent fallback to a different layout.
    partnership_block = ""
    if frame:
        partnership_block = "Partnership context (your prior — what's currently in awareness):\n" + frame

    # Format recently recalled (lightweight — id + title only)
    recalled_text = ""
    if recently_recalled:
        for r in recently_recalled:
            recalled_text += "%s \"%s\"\n" % (str(r.get("id", ""))[:8], r.get("title", "")[:60])

    # v9: Build retrieval context block (always present when stats available)
    retrieval_context = ""
    if retrieval_stats:
        rs = retrieval_stats
        top = rs.get('top_score', 0)
        median = rs.get('median_score', 0)
        brain_sz = rs.get('brain_size', 0)
        n_candidates = rs.get('candidates_after_floor', 0)
        breakdown = rs.get('source_breakdown', {})

        retrieval_context = "Retrieval: %d candidates from %d memories. Top: %.2f, median: %.2f." % (
            n_candidates, brain_sz, top, median)

        # Source breakdown (non-zero only)
        src_parts = []
        for src, count in breakdown.items():
            if count > 0:
                src_parts.append("%d %s" % (count, src))
        if src_parts:
            retrieval_context += " Sources: %s." % ", ".join(src_parts)

        # v9: Dynamic guidance based on distribution
        from servers.brain_constants import RETRIEVAL_LOW_CONFIDENCE
        if top < RETRIEVAL_LOW_CONFIDENCE:
            retrieval_context += (
                "\nNOTE: Top score %.2f is low for %d memories — "
                "brain likely has nothing relevant. Prefer selecting 0." % (top, brain_sz))

    # v9: Intent context (from STEP 2 classification). 2026-05-03: generalized
    # — no operator name hardcoded.
    intent_context = ""
    if intent and intent != 'general':
        _intent_guidance = {
            'decision_lookup': 'Operator is looking for a past decision — prioritize decision, rule, and correction nodes.',
            'reasoning_chain': 'Design/reasoning task — architecture, mechanism, and pattern nodes most helpful.',
            'correction_lookup': 'Looking for a correction — prioritize correction and lesson nodes.',
            'how_to': 'How-to question — mechanism, convention, and lesson nodes most helpful.',
            'temporal': 'Time-based query — check created_at dates, prioritize session and milestone nodes.',
            'state_query': 'Checking current state — recent decisions and open items most relevant.',
        }
        if intent in _intent_guidance:
            intent_context = "Query type: %s. %s" % (intent, _intent_guidance[intent])

    # Format candidates
    candidates_text = ""
    for i, c in enumerate(candidates, 1):
        candidates_text += format_candidate_for_surface(c, i) + "\n\n"

    # User content — per-turn delta only. Instructions live in the cached
    # system block (the registered `surface` interaction template), assembled
    # in _call_surface. This function builds ONLY what changes per turn.
    user_content = """%s

Conversation (recent, oldest first):
%s
Recently surfaced (deprioritize — only select if the current message specifically needs them):
%s
%s
%s
%d candidates follow. Select 0-%d.

Candidates:

%s""" % (
        partnership_block or "(no partnership context — fresh session or Frame unavailable)",
        conversation or "(no recent messages)",
        recalled_text or "(none)",
        retrieval_context,
        intent_context,
        len(candidates),
        cfg['max_selected'],
        candidates_text,
    )

    return user_content, cfg['max_tokens']


SURFACE_FORMAT = {
    'content_limit': 400, 'edge_limit': 3, 'metadata_limit': 150, 'time_format': 'relative',
    # Surface strips decision/debug fields — keeps content, situation, edges
    # (what Anchor needs to respond). Judge reasoning, question, keywords, model
    # provenance, and confidence score aren't action-taking context.
    'show_confidence': False,
    'show_encoding_source': False,
    'show_keywords': False,
    'extra_skip_keys': ('question', 'reasoning'),
}
HAIKU_FORMAT = {'content_limit': 300, 'edge_limit': 3, 'metadata_limit': 120, 'time_format': 'relative'}


# ═══════════════════════════════════════════════════════════════
# SPREAD_ACTIVATION — the kernel that decides what surfaces
#
# One mechanism, three granularities:
#   • Nodes receive activation  (max over per-field cosines with query)
#   • Edges transmit activation (cosine of query to enriched edge text)
#   • Fields within a node hold per-field activation (for render budget)
#
# Depth is emergent — high-coefficient edges keep propagating, low-
# coefficient edges stop on the first hop. Saturation (tanh) bounds
# accumulation. The per-hop transmission gate is self-calibrating:
# when no edges in the batch exceed the brain's noise floor, propagation
# halts — the brain is allowed to be quiet.
#
# No static blend weights. No multiplicative family boost (family meaning
# rides inside the enriched edge text). No count-based fatigue. The
# activation map IS the ranking, the tiebreaker, and the render-budget
# allocator, all in one.
# ═══════════════════════════════════════════════════════════════

import numpy as np

# Safety cap — if activation hasn't settled by 3 hops, force stop.
# Real termination comes from the transmission gate + saturation, not this.
_SPREAD_MAX_STEPS = 3

# Per-source neighbor cap during spread expansion. Held at 50 after
# 2026-05-01 partial-rollback: lim=30 was tried (variant C, iter_M
# 14/15 on seed=42) but iter_O on a different sample (seed=1) revealed
# the narrowing starves muster recall during ingestion → encoder
# misses specific facts → quality drops 80%→60% on that sample. The
# narrower lim hurt 5 items on seed=1, helped 0. Reverted.
#
# The latency win lives entirely in HOP_SCRUTINY_DEFAULT (variant D in
# the bench): scrutiny alone reduces avg recall 11.2s → 6.3s (44%) by
# narrowing only at hop 3, where the work doesn't bite encoder context.
# Lim stays at production-original 50.
SPREAD_NEIGHBOR_LIMIT_DEFAULT = 50

# Per-hop scrutiny: at hop 3+ (step >= 2), only the top half of currently
# active nodes by activation propagate further. Distribution-derived —
# the threshold is the median of the current activation set, not a static
# number. Forces narrowing as depth increases.
#
# Production default = OFF (2026-05-01): scrutiny was shipped on the
# strength of LongMemEval iter_M (14/15) but the eval is N=15 with ~20pp
# variance, and scrutiny was never A/B'd in isolation against baseline.
# Real-world conversational use (where ambient context lives 2-3 hops out)
# regressed — see the ex.co class of failure: the brain held rich operator
# context but it lived past the scrutiny cut and stopped surfacing.
# Override BRAIN_SPREAD_HOP_SCRUTINY=on to re-enable for benchmarking.
HOP_SCRUTINY_DEFAULT = False

# Outer propagation gate: if the max transmission coefficient in the batch
# falls below the brain's noise floor, nothing meaningful to spread —
# halt rather than forcing below-noise edges through. Imported from
# brain_constants to stay in sync with recall()'s own noise definition.
from servers.brain_constants import NOISE_FLOOR_THRESHOLD as _SPREAD_NOISE_FLOOR


def _batch_load_field_vectors(brain_conn, node_ids):
    """One SQL round-trip: all field-cohort + legacy vectors for given nodes.

    Returns: {full_node_id: {vector_type: numpy.ndarray}}
    Short-prefix matching handled by caller if needed; here we read full IDs.
    """
    if not brain_conn or not node_ids:
        return {}
    ids = list(node_ids)
    placeholders = ','.join('?' * len(ids))
    rows = brain_conn.execute(
        "SELECT node_id, vector_type, embedding FROM node_enrichments "
        "WHERE node_id IN (%s) AND embedding IS NOT NULL" % placeholders,
        ids).fetchall()
    out = {}
    for nid, vtype, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        out.setdefault(nid, {})[vtype] = vec
    return out


def _field_cosines_for_node(query_vec, node_vectors, norm_q=None):
    """Cosines against query, per semantic field.

    `node_vectors` = {vector_type: vec} from _batch_load_field_vectors.
    For each field type in the field-cohort stable order, try the per-field
    vector first; fall back to the blended legacy vector via
    FIELD_VECTOR_FALLBACK. Missing field → not in the result (no cosine
    fabricated).

    Returns: {field_name: cosine} — order matches field_vector_types().
    """
    from servers.pipeline_contract import field_vector_types, FIELD_VECTOR_FALLBACK
    out = {}
    for field_type in field_vector_types():
        # Field-cohort vector available?
        vec = node_vectors.get(field_type)
        # Fall back through legacy chain
        if vec is None:
            for fb in FIELD_VECTOR_FALLBACK.get(field_type, []):
                vec = node_vectors.get(fb)
                if vec is not None:
                    break
        if vec is None:
            continue
        out[field_type] = _cosine_nonneg(query_vec, vec, norm_a=norm_q)
    return out


def _build_edge_coeffs(brain, brain_conn, activated_nodes, query_vec,
                      rel_to_family, meaning_by_family, cached_edge_coeffs):
    """Collect outgoing edges from all currently activated nodes; compute
    their transmission coefficients.

    Returns list of (source_id, target_id, coeff, edge_dict).
    Uses cached_edge_coeffs as an in-kernel memo so repeat edges in later
    hops don't re-embed.
    """
    from servers.dal import GraphDAL
    from servers.pipeline_contract import TRAVERSE_EXCLUDED_EDGES

    gdal = GraphDAL(brain_conn)
    excluded = set(TRAVERSE_EXCLUDED_EDGES)

    # Gather ALL edges from activated sources in one pass
    edges_out = []
    enriched_to_embed = []  # texts needing embedding
    enriched_keys = []      # parallel: (source, target, edge, enriched_text)

    # Per-source neighbor cap. Default from contract (SPREAD_NEIGHBOR_LIMIT_DEFAULT).
    # Env override: BRAIN_SPREAD_NEIGHBOR_LIMIT — used by eval variants.
    import os as _os
    _SPREAD_LIMIT = int(_os.environ.get(
        'BRAIN_SPREAD_NEIGHBOR_LIMIT', str(SPREAD_NEIGHBOR_LIMIT_DEFAULT)))

    for source_id in activated_nodes:
        rows = gdal.get_neighbors(
            source_id, limit=_SPREAD_LIMIT,
            exclude_relations=excluded,
            content_preview_chars=0,
        )
        for r in rows:
            target_id = r.get('id', '')
            # Compose edge's semantic identity text
            enriched = _compose_enriched_edge_text(
                {'title': r.get('title', ''),
                 'relation': r.get('relation', ''),
                 'description': r.get('edge_description') or ''},
                rel_to_family, meaning_by_family)
            cache_key = enriched

            cached = cached_edge_coeffs.get(cache_key)
            if cached is not None:
                edges_out.append((source_id, target_id, cached, r))
            else:
                # Queue for batch embedding
                enriched_to_embed.append(enriched)
                enriched_keys.append((source_id, target_id, r, enriched))

    # Batch-embed queued enriched texts (hits _desc_vecs_batched cache)
    if enriched_to_embed:
        blobs = _desc_vecs_batched(enriched_to_embed)
        norm_q = float(np.linalg.norm(query_vec))
        for (src, tgt, edge, text), blob in zip(enriched_keys, blobs):
            if blob is None:
                coeff = 0.0
            else:
                vec = np.frombuffer(blob, dtype=np.float32)
                coeff = _cosine_nonneg(query_vec, vec, norm_a=norm_q)
            cached_edge_coeffs[text] = coeff
            edges_out.append((src, tgt, coeff, edge))

    return edges_out


def spread_activation(seed_ids, query_vec, brain, prior_vecs=None):
    """Spreading activation from seed nodes through the graph.

    One mechanism does what select_edges + _graph_expand + compute_shared_context
    + fatigue were doing separately:
      • Activation originates at each seed from its max field-cosine with query
      • It flows through edges, weighted by the edge's own enriched-text cosine
      • Nodes receiving activation from multiple paths accumulate (tanh-saturated)
      • Depth is emergent — a strong-matching edge propagates far, a weak one
        stops on hop 1. Mutual traversal is automatic: two seeds whose paths
        meet at a neighbor boost that neighbor above singleton reach.

    Args:
        seed_ids: List of full node IDs to start activated (typically the set
            Haiku selected from the 25 candidates).
        query_vec: numpy array (768d) — current query embedding.
        brain: Brain instance — needed for interaction_config (family map)
            and conn access.
        prior_vecs: Optional list of prior-turn query embeddings for
            multi-turn blending (dampens one-word query ambiguity).

    Returns:
        dict with:
          'node_activation':  {node_id: float in [0,1]}
          'field_activation': {node_id: {field_name: float in [0,1]}}
          'trace': list of per-step {step, new_nodes, edges_considered,
                                     edges_transmitted, max_act}
    """
    if not seed_ids or query_vec is None:
        return {'node_activation': {}, 'field_activation': {}, 'trace': []}

    # ── Multi-turn query blend (existing pattern, kept) ──
    if prior_vecs:
        ws = TURN_WEIGHTS[:len([query_vec] + prior_vecs)]
        total = sum(ws)
        ws = [w / total for w in ws]
        blended = sum(w * v for w, v in zip(ws, [query_vec] + prior_vecs))
        nm = np.linalg.norm(blended)
        if nm > 0:
            blended = blended / nm
    else:
        blended = query_vec
    norm_q = float(np.linalg.norm(blended))

    # ── Load aspect map (aspect meaning composes into edge enriched text) ──
    # AspectRegistry replaced the old s2_edge_families interaction lookup
    # (Step 7 of unified-aspects). Same data, single source of truth.
    rel_to_family = {}
    meaning_by_family = {}
    try:
        for name, aspect in brain.aspects.all().items():
            if aspect.meaning:
                meaning_by_family[name] = aspect.meaning
            for r in aspect.edge_relations:
                rel_to_family[r] = name
    except Exception as _e:
        # Aspect data is optional but loading failure is worth noticing —
        # if registry exists and we fail to read it, family_boost dies silently.
        brain._log_error('spread_aspect_config', _e,
                         'loading brain.aspects in spread_activation')

    # ── Step 0: seeds' own activations (per-field cosines vs query) ──
    all_touched_ids = list(seed_ids)
    node_vectors = _batch_load_field_vectors(brain.conn, all_touched_ids)
    # Silent-failure guard: if we got NO vectors for a seed, the kernel can't
    # produce activation for it. This happened at N=15 once (seed had no
    # vectors at all), which is the failure mode worth surfacing.
    seeds_missing_vectors = [sid for sid in seed_ids if sid not in node_vectors]
    if seeds_missing_vectors:
        brain._log_error('spread_seed_no_vectors',
                         RuntimeError('seeds without any vectors'),
                         'seeds=%s (kernel will produce zero activation for these)' %
                         ','.join(s[:8] for s in seeds_missing_vectors[:5]))
    node_activation = {}
    field_activation = {}

    for nid in seed_ids:
        vecs = node_vectors.get(nid, {})
        field_cos = _field_cosines_for_node(blended, vecs, norm_q=norm_q)
        field_activation[nid] = field_cos
        # MAX over signals, per TRIZ — best discriminating field wins
        node_activation[nid] = max(field_cos.values()) if field_cos else 0.0

    # ── Spread loop ──
    trace_steps = []
    cached_edge_coeffs = {}  # memo across hops: enriched_text → coeff

    # Variant 'thickness' (eval): multiply cosine by accumulated edge weight
    # before any gating. Edges already track confirmation count via Hebbian
    # strengthening on each repeat write (servers/dal.py:add_relation —
    # weight grows by LEARNING_RATE × 0.5 per repeat, capped at MAX_WEIGHT).
    # Today that signal is unused in spread; this variant uses it.
    import os as _os
    _THICKNESS = 'thickness' in _os.environ.get('BRAIN_RECALL_VARIANT', '').lower()

    # Hop scrutiny: at hop 3+ (step >= 2), only the top half of currently-
    # active nodes by activation propagate further. Distribution-derived —
    # the threshold is the median of the current activation set, not a
    # static number. Default from contract (HOP_SCRUTINY_DEFAULT = True).
    # Opt-out via BRAIN_SPREAD_HOP_SCRUTINY=off for A/B comparisons.
    _scrutiny_env = _os.environ.get('BRAIN_SPREAD_HOP_SCRUTINY')
    if _scrutiny_env is None:
        _HOP_SCRUTINY = HOP_SCRUTINY_DEFAULT
    else:
        _HOP_SCRUTINY = _scrutiny_env.lower() == 'on'

    for step in range(_SPREAD_MAX_STEPS):
        # Source nodes at this hop: whichever are currently activated above zero
        raw_active = [(n, a) for n, a in node_activation.items() if a > 0]
        if not raw_active:
            break

        # Per-hop scrutiny on the jump to hop 3+. The median of current
        # activations becomes the floor — only above-median sources
        # propagate further. The cost of going deeper is that you only
        # follow strong activations, not all weak ones.
        if _HOP_SCRUTINY and step >= 2 and len(raw_active) > 4:
            sorted_acts = sorted([a for _, a in raw_active], reverse=True)
            scrutiny_floor = sorted_acts[len(sorted_acts) // 2]
            active_sources = [n for n, a in raw_active if a >= scrutiny_floor]
        else:
            active_sources = [n for n, _ in raw_active]

        edges = _build_edge_coeffs(
            brain, brain.conn, active_sources, blended,
            rel_to_family, meaning_by_family, cached_edge_coeffs)

        if not edges:
            break

        # Apply thickness: cosine × edge_weight. A confirmed edge (weight=1.0)
        # transmits at full cosine; a single-mention edge (weight=0.5)
        # transmits at half. Weakened/contradicted edges (weight<0.3) drop out
        # of the gate naturally without needing a separate suppression rule.
        if _THICKNESS:
            edges = [
                (s, t, c * float(e.get('weight') or 0.5), e)
                for (s, t, c, e) in edges
            ]

        # Outer gate: any meaningfully-matching edges at all?
        max_coeff = max(e[2] for e in edges)
        if max_coeff < _SPREAD_NOISE_FLOOR:
            trace_steps.append({'step': step, 'new_nodes': 0,
                               'edges_considered': len(edges),
                               'edges_transmitted': 0,
                               'max_act': max(node_activation.values())
                                   if node_activation else 0.0,
                               'halted': 'below_noise_floor'})
            break

        # Per-hop self-calibrating threshold: median of positive coeffs.
        # Ensures only better-than-typical edges in THIS batch transmit.
        threshold = float(np.percentile([e[2] for e in edges], 50))

        # Variant 'lineage' (eval): lineage-family edges bypass the median
        # gate. The relation type itself encodes structural meaning the
        # enriched-text cosine can't capture (corrects, extends, supersedes).
        # This is opt-in via env var so production behavior is unchanged.
        import os as _os
        _LINEAGE_PASS = 'lineage' in _os.environ.get(
            'BRAIN_RECALL_VARIANT', '').lower()

        # Accumulate contributions for each target
        contributions = {}
        transmitted_count = 0
        for src, tgt, coeff, _edge in edges:
            if coeff < threshold:
                if _LINEAGE_PASS:
                    relation = (_edge.get('relation') or '').strip()
                    family = rel_to_family.get(relation, '')
                    if family not in LINEAGE_FAMILIES:
                        continue
                    # Else fall through — lineage edge bypasses the gate
                else:
                    continue
            transmitted_count += 1
            source_act = node_activation.get(src, 0)
            # Target hasn't been computed yet? Skip source-act lookup failures.
            transferred = source_act * coeff
            contributions[tgt] = contributions.get(tgt, 0.0) + transferred

        if not contributions:
            trace_steps.append({'step': step, 'new_nodes': 0,
                               'edges_considered': len(edges),
                               'edges_transmitted': 0,
                               'max_act': max(node_activation.values()),
                               'halted': 'all_below_threshold'})
            break

        # Load field vectors for any newly-touched targets (one round-trip)
        new_target_ids = [t for t in contributions.keys() if t not in node_activation]
        if new_target_ids:
            fresh_vecs = _batch_load_field_vectors(brain.conn, new_target_ids)
            node_vectors.update(fresh_vecs)
            all_touched_ids.extend(new_target_ids)

        # Apply saturation — tanh blends old activation + incoming contribution.
        # Also seed target's own field-cosines (against query) so rendering has
        # field-level signal for newly-activated targets.
        pre_activation = dict(node_activation)
        new_nodes_added = 0
        for target, incoming in contributions.items():
            prior = node_activation.get(target, 0.0)
            saturated = float(np.tanh(prior + incoming))
            node_activation[target] = saturated
            if target not in field_activation:
                new_nodes_added += 1
                vecs = node_vectors.get(target, {})
                field_activation[target] = _field_cosines_for_node(
                    blended, vecs, norm_q=norm_q)

        trace_steps.append({'step': step, 'new_nodes': new_nodes_added,
                           'edges_considered': len(edges),
                           'edges_transmitted': transmitted_count,
                           'max_act': max(node_activation.values()),
                           'threshold': threshold})

        # No-change check: if nothing moved meaningfully, stop early
        delta = max((node_activation[n] - pre_activation.get(n, 0.0)
                    for n in node_activation), default=0.0)
        if delta < 0.01:
            break

    return {
        'node_activation': node_activation,
        'field_activation': field_activation,
        'trace': trace_steps,
    }


# ═══════════════════════════════════════════════════════════════
# SPREAD_ACTIVATION_CLUSTER — variant under eval (2026-04-30)
#
# Three principled changes from spread_activation, each derived from a
# distribution or taxonomy already present in the brain:
#
#  1. Distribution-derived per-hop gate. Replace the median-threshold
#     ("always 50% pass") and earlier draft's hardcoded fractions with a
#     z-score gate within this hop's edge coefficients: pass iff
#     coeff > μ_hop + k_hop·σ_hop. k narrows per hop (more strict deeper).
#     If the distribution is tight (everything similar), the gate
#     naturally lets more through; if wide (clear winners), it cuts hard.
#     The cut is the derivative of the situation, per c2730676.
#
#  2. Aspect-aware ride-along. Instead of a hardcoded relation allowlist,
#     read brain.aspects (the AspectRegistry) and treat edges whose aspect
#     is in LINEAGE_FAMILIES as structural — they ride along even when
#     their enriched-text cosine is weak. New aspects emerging via
#     AspectIntegration inherit the behavior automatically (when they map
#     to one of the lineage names). Floor for lineage transmission is the
#     per-hop 25th percentile of edge coefficients — distribution-derived,
#     not 0.4.
#
#  3. Convergence as a tag, not a multiplier. When a target is reached by
#     ≥2 distinct sources, mark it convergent and increment a per-node
#     convergence count. Activation stays scalar (sum + tanh as before).
#     The render layer can read the convergence map and prioritize cluster
#     boundaries — that's qualitative information, not amplitude.
#
# Same return shape as spread_activation + an additional 'convergence'
# field. Render layer can ignore it for compat.
# ═══════════════════════════════════════════════════════════════

# Aspect names whose semantic role is structural-lineage rather than
# topical-similarity. These edges ride along even with weak enriched-text
# cosine, because their meaning is carried by the relation type itself,
# not by the description embedding.
#
# The classification mirrors what the brain already has in
# brain.aspects (AspectRegistry); if the taxonomy evolves via
# AspectIntegration, this allowlist may need updating. Kept narrow on
# purpose — broader is "everything rides," which is the current
# spread's behavior we're trying to bound.
LINEAGE_FAMILIES = frozenset({
    'correction_improvement',          # corrects, corrected_by — anti-staleness
    'extension_refinement',            # extends, refines, evolves
    'evolution_and_change',            # evolves_from, evolved_into
    'version_and_replacement',         # supersedes, replaces
    'composition_and_structure',       # part_of, contains, consolidated_into
    'dependency_and_prerequisite',     # depends_on, requires
    'hierarchical_structure',          # supersedes, contains, includes
    'refinement_and_correction',       # refines, corrects, corrected_by
})

# Per-hop z-gate strictness. k controls how many σ above hop mean an edge
# must be to transmit. Larger k = stricter cut = narrower spread. Hop 0
# casts wide (k=0, anything above mean), hop 1 narrows (k=0.5), hop 2
# only outliers (k=1.0). This is the only place a "tuning constant"
# survives — and it's a strictness schedule, not a static threshold.
# The actual cut value is computed from the hop's distribution.
_CLUSTER_K_SCHEDULE = (0.0, 0.5, 1.0)

# Safety cap on hops. Real termination comes from "no convergence + no
# new high-activation" check inside the loop.
_CLUSTER_MAX_STEPS = 3


def spread_activation_cluster(seed_ids, query_vec, brain, prior_vecs=None):
    """Cluster-completion variant of spread_activation.

    See module-level comment above for the design rationale. Returns the
    same shape as spread_activation plus a 'convergence' map that the
    render layer can use to prioritize cluster-boundary nodes.
    """
    if not seed_ids or query_vec is None:
        return {'node_activation': {}, 'field_activation': {},
                'convergence': {}, 'trace': []}

    # Multi-turn query blend (same pattern as the original kernel)
    if prior_vecs:
        ws = TURN_WEIGHTS[:len([query_vec] + prior_vecs)]
        total = sum(ws)
        ws = [w / total for w in ws]
        blended = sum(w * v for w, v in zip(ws, [query_vec] + prior_vecs))
        nm = np.linalg.norm(blended)
        if nm > 0:
            blended = blended / nm
    else:
        blended = query_vec
    norm_q = float(np.linalg.norm(blended))

    # Aspect map — single source of truth via brain.aspects.
    rel_to_family = {}
    meaning_by_family = {}
    try:
        for name, aspect in brain.aspects.all().items():
            if aspect.meaning:
                meaning_by_family[name] = aspect.meaning
            for r in aspect.edge_relations:
                rel_to_family[r] = name
    except Exception as _e:
        brain._log_error('cluster_spread_aspect_config', _e,
                         'loading brain.aspects in spread_activation_cluster')

    # Seed activation (same as original)
    all_touched_ids = list(seed_ids)
    node_vectors = _batch_load_field_vectors(brain.conn, all_touched_ids)
    seeds_missing_vectors = [sid for sid in seed_ids if sid not in node_vectors]
    if seeds_missing_vectors:
        brain._log_error('cluster_seed_no_vectors',
                         RuntimeError('seeds without any vectors'),
                         'seeds=%s' % ','.join(s[:8] for s in seeds_missing_vectors[:5]))
    node_activation = {}
    field_activation = {}
    for nid in seed_ids:
        vecs = node_vectors.get(nid, {})
        field_cos = _field_cosines_for_node(blended, vecs, norm_q=norm_q)
        field_activation[nid] = field_cos
        node_activation[nid] = max(field_cos.values()) if field_cos else 0.0

    # Spread loop with distribution-gated narrowing + family-aware lineage
    # + convergence tagging
    trace_steps = []
    cached_edge_coeffs = {}
    convergence_count = {}  # node_id → number of distinct sources that reached it

    for step in range(_CLUSTER_MAX_STEPS):
        active_sources = [n for n, a in node_activation.items() if a > 0]
        if not active_sources:
            break

        edges = _build_edge_coeffs(
            brain, brain.conn, active_sources, blended,
            rel_to_family, meaning_by_family, cached_edge_coeffs)

        if not edges:
            break

        # Classify: lineage = ride-along by family; semantic = subject to
        # distribution-derived gate.
        lineage = []
        semantic = []
        for src, tgt, coeff, edge in edges:
            relation = (edge.get('relation') or '').strip()
            family = rel_to_family.get(relation, '')
            if family in LINEAGE_FAMILIES:
                lineage.append((src, tgt, coeff, edge, family))
            else:
                semantic.append((src, tgt, coeff, edge, family))

        # Distribution-derived gate on semantic edges. Compute mean+std
        # from this hop's coefficients and require coeff > μ + k·σ.
        # If σ is tiny (everything similar), only just-above-mean pass —
        # which is fine, those are the cluster's coherent fringe. If σ is
        # wide (mixed quality), the cut is harder.
        if semantic:
            sem_coeffs = np.array([c for _, _, c, _, _ in semantic])
            mu = float(np.mean(sem_coeffs))
            sigma = float(np.std(sem_coeffs))
            k = _CLUSTER_K_SCHEDULE[min(step, len(_CLUSTER_K_SCHEDULE) - 1)]
            cut = mu + k * sigma
            transmitting_sem = [(s, t, c, e, f) for s, t, c, e, f in semantic
                                if c >= cut]
        else:
            mu = sigma = 0.0
            transmitting_sem = []

        # Distribution-derived floor for lineage: per-hop p25 of all edge
        # coefficients (semantic + lineage). Lineage with raw coeff below
        # this floor transmits AT the floor — the family carries the
        # meaning. With no semantic edges, lineage transmits at its own
        # coeff or a nominal small value, whichever is greater.
        all_coeffs = [c for _, _, c, _, _ in edges]
        floor = float(np.percentile(all_coeffs, 25)) if all_coeffs else 0.0
        transmitting_lin = [
            (s, t, max(c, floor), e, f)
            for s, t, c, e, f in lineage
        ]

        # Outer halt: if neither path will transmit anything meaningful,
        # the brain's allowed to be quiet.
        if not transmitting_sem and not transmitting_lin:
            trace_steps.append({'step': step, 'new_nodes': 0,
                               'edges_considered': len(edges),
                               'edges_transmitted': 0,
                               'mu': round(mu, 3), 'sigma': round(sigma, 3),
                               'cut': round(mu + (_CLUSTER_K_SCHEDULE[min(step, len(_CLUSTER_K_SCHEDULE)-1)] * sigma), 3),
                               'max_act': max(node_activation.values())
                                   if node_activation else 0.0,
                               'halted': 'gate_closed'})
            break

        transmitting = transmitting_sem + transmitting_lin

        # Accumulate contributions per target. Track distinct sources for
        # convergence tagging (NOT amplification).
        contributions = {}
        sources_per_target = {}
        families_per_target = {}
        for src, tgt, coeff, _edge, family in transmitting:
            source_act = node_activation.get(src, 0)
            transferred = source_act * coeff
            contributions[tgt] = contributions.get(tgt, 0.0) + transferred
            sources_per_target.setdefault(tgt, set()).add(src)
            if family:
                families_per_target.setdefault(tgt, set()).add(family)

        # Convergence: count distinct sources reaching each target. Tag,
        # don't amplify — render layer reads this map.
        for tgt, srcs in sources_per_target.items():
            if len(srcs) >= 2:
                convergence_count[tgt] = convergence_count.get(tgt, 0) + len(srcs)

        # Load field vectors for new targets (one round-trip)
        new_target_ids = [t for t in contributions.keys() if t not in node_activation]
        if new_target_ids:
            fresh_vecs = _batch_load_field_vectors(brain.conn, new_target_ids)
            node_vectors.update(fresh_vecs)
            all_touched_ids.extend(new_target_ids)

        # Apply with tanh saturation; seed targets' field activations
        pre_activation = dict(node_activation)
        new_nodes_added = 0
        for target, incoming in contributions.items():
            prior = node_activation.get(target, 0.0)
            saturated = float(np.tanh(prior + incoming))
            node_activation[target] = saturated
            if target not in field_activation:
                new_nodes_added += 1
                vecs = node_vectors.get(target, {})
                field_activation[target] = _field_cosines_for_node(
                    blended, vecs, norm_q=norm_q)

        n_converged = sum(1 for s in sources_per_target.values() if len(s) >= 2)
        trace_steps.append({'step': step,
                           'new_nodes': new_nodes_added,
                           'edges_considered': len(edges),
                           'edges_transmitted': len(transmitting),
                           'edges_lineage': len(transmitting_lin),
                           'edges_semantic': len(transmitting_sem),
                           'mu': round(mu, 3), 'sigma': round(sigma, 3),
                           'cut': round(mu + (_CLUSTER_K_SCHEDULE[min(step, len(_CLUSTER_K_SCHEDULE)-1)] * sigma), 3),
                           'floor_p25': round(floor, 3),
                           'targets_converged': n_converged,
                           'max_act': max(node_activation.values())})

        # Convergence-stop: cluster boundary reached when no target this
        # hop saw ≥2 sources AND no new high-activation node added.
        any_new_high = any(node_activation[t] > 0.3 and t not in pre_activation
                           for t in contributions)
        if n_converged == 0 and not any_new_high and step > 0:
            break

    return {
        'node_activation': node_activation,
        'field_activation': field_activation,
        'convergence': convergence_count,
        'trace': trace_steps,
    }


# ═══════════════════════════════════════════════════════════════
# EDGE SELECTION — per-node edge ranking for DISPLAY
#
# Used when a single activated node is being rendered and we need to pick
# which of its edges to show (typically 3). Scoring is the same MAX formula
# as the spreading-activation kernel uses for transmission — one rule at
# two granularities (node-level and edge-level).
#
#   edge_score = max(
#       cos(query, target_node_embedding),
#       cos(query, enriched_edge_text)
#   )
#
# No blend weights. No family_boost term (family meaning is inlined into
# the enriched text, so it rides inside the second signal). No fatigue
# discount (novelty is handled at the activation-set level, not per edge).
# No weight tiebreaker (edge weights are static ~0.6 across the graph
# today — they're not signal yet).
#
# Multi-turn query blending is still applied: when a prior turn's query
# vector is available, the effective query is a weighted average that
# smooths single-word follow-ups.
# ═══════════════════════════════════════════════════════════════

# Multi-turn query blending (current turn, previous, two-back)
TURN_WEIGHTS = [0.6, 0.3, 0.1]


def _compose_enriched_edge_text(conn, rel_to_family, meaning_by_family):
    """Compose the text embedded as the edge's semantic identity.

    Pattern: "<target_title> [<relation>] <description>. family: <meaning>"
    Parts that are missing are dropped silently — an edge with bare title
    and no description still gets an embedding based on title + relation.
    """
    title = (conn.get('title') or conn.get('target_title') or '').strip()
    rel = (conn.get('relation') or '').strip()
    desc = (conn.get('description') or '').strip()
    family = rel_to_family.get(rel, '') if rel_to_family else ''
    meaning = meaning_by_family.get(family, '') if (family and meaning_by_family) else ''

    parts = []
    if title:
        parts.append(title)
    if rel:
        parts.append('[%s]' % rel)
    if desc:
        parts.append(desc)
    if meaning:
        parts.append('family: ' + meaning)
    return ' '.join(parts)


def _cosine_nonneg(a, b, norm_a=None):
    """Non-negative cosine. Computes norm_b on the fly; caller may pre-pass
    norm_a since it's invariant across edges in a single select_edges call."""
    import numpy as np
    if a is None or b is None or len(a) != len(b):
        return 0.0
    if norm_a is None:
        norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, float(np.dot(a, b)) / (norm_a * norm_b))


def select_edges(connections, query_vec, session=None, limit=3, prior_vecs=None,
                 brain_conn=None, brain=None):
    """Pick the top-N edges of a single node for display rendering.

    Used inside the activation renderer when laying out an activated node —
    we have space for ~3 edges under the node header, and this function
    picks the 3 that best fit the query.

    Scoring: max(cos(query, target_node_embedding), cos(query, enriched_edge_text)).
    Family meaning is composed into the enriched text via _compose_enriched_edge_text,
    so it contributes automatically to the second term without a separate signal.

    Args:
        connections: list of edge dicts (from get_rich_node().connections OR
            from GraphDAL.get_neighbors()). Required keys: id, relation;
            useful keys: title/target_title, description.
        query_vec: numpy array (768d) — current query embedding.
        session: kept for signature compat; no longer used (fatigue dissolved).
        limit: max edges to return.
        prior_vecs: prior-turn query embeddings for multi-turn blending.
        brain_conn: sqlite3 connection — used to load stored target node
            embeddings. Without it, node_rel falls back to 0 and scoring
            leans on enriched_edge only.
        brain: Brain instance — used to load `s2_edge_families` config so
            family meaning can be composed into enriched edge text.

    Returns:
        list of edge dicts (top N by score), same shape as input.
        On empty input or missing query_vec, returns the input unchanged
        (truncated to limit) — no graceful reordering.
    """
    import numpy as np

    if not connections or query_vec is None:
        return connections[:limit]

    # Multi-turn blend — fold prior queries into effective query vector
    if prior_vecs:
        ws = TURN_WEIGHTS[:len([query_vec] + prior_vecs)]
        total = sum(ws)
        ws = [w / total for w in ws]
        blended = sum(w * v for w, v in zip(ws, [query_vec] + prior_vecs))
        nm = np.linalg.norm(blended)
        if nm > 0:
            blended = blended / nm
    else:
        blended = query_vec
    norm_q = float(np.linalg.norm(blended))

    # Load family context — family meaning is composed INTO the enriched text,
    # not applied as a separate multiplicative signal.
    rel_to_family = {}
    meaning_by_family = {}
    if brain is not None:
        try:
            for name, aspect in brain.aspects.all().items():
                if aspect.meaning:
                    meaning_by_family[name] = aspect.meaning
                for r in aspect.edge_relations:
                    rel_to_family[r] = name
        except Exception as _e:
            brain._log_error('select_edges_aspect_config', _e,
                             'loading brain.aspects in select_edges')

    # Batch-load target node embeddings (one SQL round-trip)
    target_ids = [c.get('id', '')[:8] for c in connections]
    stored_embeddings = {}
    if brain_conn is not None and target_ids:
        like_clauses = ' OR '.join(['node_id LIKE ?'] * len(target_ids))
        params = [tid + '%' for tid in target_ids]
        rows = brain_conn.execute(
            "SELECT node_id, embedding FROM node_enrichments "
            "WHERE vector_type = '_primary' AND (%s)" % like_clauses,
            params).fetchall()
        for full_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            stored_embeddings[full_id[:8]] = vec

    # Batch-embed composed edge texts (cache-aware)
    enriched_texts = [
        _compose_enriched_edge_text(c, rel_to_family, meaning_by_family)
        for c in connections
    ]
    enriched_blobs = _desc_vecs_batched(enriched_texts)

    # Score each edge: MAX of node-target and enriched-edge cosines
    scored = []
    for c, enriched_blob in zip(connections, enriched_blobs):
        tid = c.get('id', '')[:8]

        target_vec = stored_embeddings.get(tid)
        node_rel = (_cosine_nonneg(blended, target_vec, norm_a=norm_q)
                    if target_vec is not None else 0.0)

        if enriched_blob is not None:
            enriched_vec = np.frombuffer(enriched_blob, dtype=np.float32)
            enriched_rel = _cosine_nonneg(blended, enriched_vec, norm_a=norm_q)
        else:
            enriched_rel = 0.0

        scored.append((max(node_rel, enriched_rel), c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:limit]]


# ═══════════════════════════════════════════════════════════════
# ACTIVATION-DRIVEN RENDERING
#
# Once spread_activation has produced {node_activation, field_activation},
# format_surface_output_activation renders nodes by activation rank, with
# each node's token budget proportional to its activation and each field
# appearing only when its per-field activation clears a minimum threshold.
#
# No SURFACE_FORMAT whitelist. No hardcoded "show content but not reasoning."
# Fields with high activation surface; fields with low activation don't.
# Tom's framing: "fading means less and less data surfaced."
# ═══════════════════════════════════════════════════════════════

# Minimum field activation to render the field at all. Below this, the field
# is masked out so render_rich_node drops it naturally.
_FIELD_RENDER_THRESHOLD = 0.3

# Minimum per-node budget — below this, we stop rendering further nodes
# rather than emit a stub that can't carry meaning.
_MIN_NODE_BUDGET_CHARS = 150


def _allocate_budget_softmax(activations, total_budget):
    """Softmax-weighted budget per node. High-activation nodes get more share;
    saturated-at-1.0 nodes split evenly; weak nodes still get minimum viable.
    Returns list of int budgets aligned with activations input.
    """
    if not activations:
        return []
    arr = np.array(activations, dtype=np.float64)
    # Subtract max for numerical stability, then softmax
    exps = np.exp(arr - arr.max())
    weights = exps / exps.sum()
    return [max(_MIN_NODE_BUDGET_CHARS, int(w * total_budget)) for w in weights]


def _mask_node_by_field_activation(node, field_activation, threshold=_FIELD_RENDER_THRESHOLD):
    """Return a copy of `node` with low-activation fields zeroed out.

    Fields that score below threshold are removed so render_rich_node drops
    them. This is how activation drives WHICH fields appear — no config
    whitelist, just the field's own query-match deciding its visibility.
    """
    masked = dict(node)

    # Top-level fields: content, situation
    if field_activation.get('content', 1.0) < threshold:
        masked['content'] = ''
    if field_activation.get('situation', 1.0) < threshold:
        masked['situation'] = ''

    # Metadata-KV-backed fields: reasoning, user_raw_quote, anchor_raw_quote,
    # question. render_rich_node reads these from node['_metadata'].
    meta = dict(node.get('_metadata') or {})
    for field_name in ('reasoning', 'user_raw_quote', 'anchor_raw_quote', 'question'):
        if field_activation.get(field_name, 1.0) < threshold:
            meta.pop(field_name, None)
    masked['_metadata'] = meta

    return masked


def _render_node_activation(node, field_activation, budget, activation,
                             is_seed=False, why='', query_vec=None, brain=None,
                             session=None):
    """Render a single activated node within a char budget.

    • Fields below activation threshold are masked out — they simply don't
      appear. This is the "fade" mechanism.
    • Budget scales content / metadata / edges limits proportionally.
    • Edges are picked query-aware via select_edges (top-3 by MAX formula).
    """
    from servers.contract import render_rich_node

    # Budget allocation within a node (heuristics, tunable later)
    content_budget = max(50, int(budget * 0.50))
    meta_budget = max(30, int(budget * 0.10))  # per-field metadata limit

    masked = _mask_node_by_field_activation(node, field_activation)

    # Query-aware edge picking for the node's own connections (display order)
    connections = masked.get('connections') or []
    if query_vec is not None and connections:
        masked = dict(masked)
        masked['connections'] = select_edges(
            connections, query_vec, session=session, limit=10,
            brain_conn=brain.conn if brain is not None else None,
            brain=brain)

    cfg = {
        'content_limit': content_budget,
        'metadata_limit': meta_budget,
        'edge_limit': 3,
        'time_format': 'relative',
        'show_confidence': False,
        'show_encoding_source': False,
        'show_keywords': field_activation.get('content', 0) > _FIELD_RENDER_THRESHOLD,
    }

    body = render_rich_node(masked, cfg)

    # Annotate seeds (Haiku-selected) with their "why"
    prefix_parts = []
    prefix_parts.append('act=%.2f' % activation)
    if is_seed:
        prefix_parts.append('SEED')
        if why:
            prefix_parts.append('why: %s' % why[:80])
    prefix_line = '  ↑ %s' % ' | '.join(prefix_parts)

    return body + '\n' + prefix_line


def format_surface_output_activation(node_activation, field_activation,
                                      rich_nodes, selected_why=None,
                                      query_vec=None, brain=None,
                                      session=None, total_budget=4000):
    """Render activated nodes as additionalContext, driven by activation.

    Args:
        node_activation:  {node_id: float} from spread_activation
        field_activation: {node_id: {field_name: float}} from spread_activation
        rich_nodes:       {node_id: rich_node_dict} from brain.get_node(ids)
        selected_why:     {node_id: str} — Haiku's "why" annotation (seeds only)
        query_vec:        query embedding, used to re-rank each node's own edges
        brain:            Brain instance — for select_edges family lookup
        session:          SessionContext — for select_edges fatigue (not used in
                          new kernel but passed through for now)
        total_budget:     int char budget for full output
    """
    if not node_activation:
        return ""

    selected_why = selected_why or {}

    # Rank by (activation, mean_field_activation) — the second key breaks
    # ties cleanly when many nodes hit saturation at 1.0.
    def sort_key(item):
        nid, act = item
        fa = field_activation.get(nid, {})
        mean_fa = (sum(fa.values()) / len(fa)) if fa else 0.0
        return (act, mean_fa, nid)  # nid as stable final tiebreaker

    ranked = sorted(node_activation.items(), key=sort_key, reverse=True)

    # Filter to nodes we have full rich data for
    ranked = [(nid, act) for nid, act in ranked if nid in rich_nodes]

    if not ranked:
        return ""

    # Softmax budget allocation
    acts = [a for _, a in ranked]
    budgets = _allocate_budget_softmax(acts, total_budget)

    lines = ['Brain activated %d memories:' % len(ranked), '']
    remaining = total_budget - len(lines[0])

    for (nid, activation), budget in zip(ranked, budgets):
        if remaining < _MIN_NODE_BUDGET_CHARS:
            break

        node = rich_nodes[nid]
        fa = field_activation.get(nid, {})
        is_seed = nid in selected_why
        why = selected_why.get(nid, '')

        effective_budget = min(budget, remaining)
        rendered = _render_node_activation(
            node, fa, effective_budget, activation,
            is_seed=is_seed, why=why, query_vec=query_vec,
            brain=brain, session=session)

        lines.append(rendered)
        lines.append('')  # blank line between nodes
        remaining -= len(rendered) + 2

    return '\n'.join(lines)


def format_surface_output(selected, candidates, graph_neighbors=None):
    """LEGACY: Format surfaced selections into structured additionalContext.

    Kept temporarily for callers that haven't migrated to
    format_surface_output_activation. New work should use the activation
    renderer which replaces this function's combination of
    SURFACE_FORMAT + inline-neighbor rendering.

    Per-node rendering delegates to render_rich_node() with SURFACE_FORMAT.
    Candidates must be in get_rich_node() shape (_metadata, _corrections, connections present).
    This function adds: collection header, relevance reasoning, graph neighbors.
    """
    from servers.contract import render_rich_node

    cfg = SURFACE
    if not selected:
        return ""

    # Build a lookup from candidate ID (first 8 chars) to full candidate data
    candidates_by_id = {}
    for c in candidates:
        short_id = str(c.get("id", ""))[:8]
        candidates_by_id[short_id] = c

    lines = ["Brain recalled %d memories:\n" % len(selected)]

    # Track all IDs in selected nodes + their connections for neighbor dedup
    seen_ids = set()

    for s in selected[:cfg['max_selected']]:
        sid = str(s.get("id", ""))[:8]
        why = s.get("why", "")
        c = candidates_by_id.get(sid)

        if not c:
            continue

        seen_ids.add(c.get("id", ""))
        seen_ids.add(sid)
        # Track connection IDs for dedup
        for conn in c.get("connections", []):
            seen_ids.add(conn.get("id", ""))
            seen_ids.add(conn.get("id", "")[:8])

        # Per-node rendering — single formatter
        # _corrections and connections already present from get_rich_node
        is_locked_rule = (c.get('type') == 'rule' and c.get('locked'))
        if is_locked_rule:
            lines.append("━━━ ACTIVE RULE (locked, applies to this response) ━━━")
            lines.append(render_rich_node(c, SURFACE_FORMAT))
            lines.append("Before finalizing your response, check: does this rule apply? "
                         "If you're about to do what the rule forbids or skip what it requires, stop and correct.")
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        else:
            lines.append(render_rich_node(c, SURFACE_FORMAT))

        # Surfacer's relevance reasoning (S1-specific, not in render_rich_node)
        if why:
            lines.append("Relevance: %s" % why)

        lines.append("")  # blank line between nodes

    # Graph neighbors — connected knowledge from surface-selected seeds
    # Dedup: skip nodes already shown as selected nodes or their connections
    if graph_neighbors:
        deduped = [nb for nb in graph_neighbors
                   if nb.get("id", "") not in seen_ids and nb.get("id", "")[:8] not in seen_ids]
        if deduped:
            # Map seed IDs → titles so we can say "corrects 'S1 Surface...'" by
            # name instead of leaving the relation dangling.
            seed_titles = {}
            for c in candidates:
                cid = c.get('id', '')
                if cid:
                    seed_titles[cid] = c.get('title', '')
                    seed_titles[cid[:8]] = c.get('title', '')

            lines.append("Related knowledge (via graph):")
            for nb in deduped[:6]:
                nid = nb.get('id', '') or ''
                # Time marker: "1w ago" for unrevised, "created Xw ago, revised Yd ago" when revised
                created = _relative_time(nb.get('created_at') or '')
                revised = _relative_time(nb.get('revised_at') or '')
                if revised and created and revised != created:
                    time_str = "created %s, revised %s" % (created, revised)
                elif created:
                    time_str = created
                else:
                    time_str = ""
                header_parts = ["id:%s" % nid[:8]]
                if time_str:
                    header_parts.append(time_str)
                # Full title — no 60-char cap (readability > density)
                lines.append('[%s] "%s" (%s)' % (
                    nb.get("type", "?"),
                    nb.get("title", "?"),
                    ", ".join(header_parts)))

                # Direction line — unambiguous who acts on whom. Matches the
                # convention render_rich_node uses for in-node edges:
                #   outgoing from seed (seed→rel→this):  "seed" rel this
                #   incoming to seed (this→rel→seed):    this rel "seed"
                seed_id = nb.get('seed_id', '') or ''
                seed_title = seed_titles.get(seed_id) or seed_titles.get(seed_id[:8]) or '(seed)'
                edge_type = nb.get('edge_type', 'related')
                edge_desc = nb.get('edge_description', '')
                direction = nb.get('direction', '')
                if direction == 'outgoing':
                    # Edge was seed → this neighbor; seed is the subject
                    rel_line = '  "%s" %s this' % (seed_title, edge_type)
                else:
                    # Edge was this neighbor → seed; neighbor is the subject
                    rel_line = '  this %s "%s"' % (edge_type, seed_title)
                if edge_desc:
                    rel_line += ' — ' + edge_desc
                lines.append(rel_line)

                # Content snippet last
                content = (nb.get("content") or "")[:200]
                if content:
                    lines.append("  %s" % content)
                lines.append("")  # blank line between neighbors

    return "\n".join(lines)
