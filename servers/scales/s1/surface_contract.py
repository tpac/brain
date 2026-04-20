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
    'metadata_fields': ['situation_text', 'reasoning', 'user_raw_quote', 'correction_of'],
    'max_edges_described': 3,   # top edges with descriptions per candidate
}

# Judge (Haiku) — selects relevant nodes with reasoning
# v9: max_candidates 25→20, Anchor truncation 150→400, recent_messages 5→7
SURFACE = {
    'content_limit': 300,           # shorter per node since more candidates
    'max_candidates': 20,           # v9: was 25. FTS5 adds up to 5 more = 25 max total
    'max_selected': 5,              # Haiku picks at most this many (was 8 — reduced for 10K hook cap)
    'user_message_limit': 300,
    'anchor_message_limit': 400,    # v9: was 150. Anchor responses carry design context
    'recent_messages': 7,           # v9: was 5. Deeper conversation window
    'recent_recalls_messages': 10,  # look back 10 messages for previously surfaced nodes
    'session_context_limit': 800,   # shared with ENCODING_AGENT — full session journey
    'session_context_tail': 200,  # v9: surface gets tail of session context (current focus)
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


def build_surface_prompt(candidates, user_message, session_context="",
                       recent_messages=None, recently_recalled=None,
                       retrieval_stats=None, intent=None,
                       prompt_instructions=None):
    """Build the S1 recall surface prompt. Single entry point.

    v9: Added retrieval_stats, intent, score normalization, conversation
    context expansion, session context tail, candidate dedup, discovery tags.
    v10: prompt_instructions from interactions table (learnable boundary).

    Args:
        candidates: List of candidate node dicts (enriched with metadata)
        prompt_instructions: Optional surface instructions from interactions table.
            If provided, replaces the hardcoded prompt text. Data assembly
            (conversation, candidates, etc.) stays in code.
        user_message: The user's latest message
        session_context: Encoder's session summary (from brain_meta)
        recent_messages: List of {"role": str, "content": str}
        recently_recalled: List of {"id": str, "title": str} from last N recalls
        retrieval_stats: Dict with brain_size, top_score, median_score, source_breakdown
        intent: Query intent from STEP 2 classification (e.g. 'reasoning_chain', 'how_to')

    Returns: (prompt_string, max_tokens)
    """
    cfg = SURFACE

    # v9: Deduplicate candidates (remove near-identical titles)
    candidates = _dedup_candidates(candidates[:cfg['max_candidates']])

    # Format conversation context (both roles, asymmetric truncation)
    # v9: Anchor responses now 400 chars (was 150), 7 messages (was 5)
    conversation = ""
    if recent_messages:
        for msg in recent_messages[-(cfg['recent_messages']):]:
            role = msg.get("role", "?")
            if role == "user":
                label = "Tom"
                content = (msg.get("content") or "")[:cfg['user_message_limit']]
            else:
                label = "Anchor"
                content = (msg.get("content") or "")[:cfg['anchor_message_limit']]
            conversation += "%s: %s\n" % (label, content)

    # Append current user message (not yet in message_stream — stored on Stop, not Submit)
    if user_message:
        conversation += "Tom: %s\n" % (user_message or "")[:cfg['user_message_limit']]

    # v9: Session context — use tail for current focus, not full changelog
    surface_session_context = ""
    if session_context:
        tail_limit = cfg.get('session_context_tail', 200)
        if len(session_context) > tail_limit:
            surface_session_context = "Current focus: ..." + session_context[-tail_limit:]
        else:
            surface_session_context = session_context

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

    # v9: Intent context (from STEP 2 classification)
    intent_context = ""
    if intent and intent != 'general':
        _intent_guidance = {
            'decision_lookup': 'Tom is looking for a past decision — prioritize decision, rule, and correction nodes.',
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

    # Instructions: from interactions table (learnable) or hardcoded default
    if not prompt_instructions:
        prompt_instructions = (
            "You surface relevant memories from a shared AI brain. The brain stores "
            "memories from conversations between an operator (Tom) and an AI assistant "
            "(Anchor). You decide which memories help Anchor respond to Tom's next message.\n\n"
            "Field guide:\n"
            "- match: similarity to query (0-1). High match = topically close, but topic alone ≠ relevant. "
            "'boosted' means score was artificially raised (critical node).\n"
            "- conf: system confidence (0-1). Higher = more established.\n"
            "- locked: operator-confirmed important.\n"
            "- via:fts5_only: found by word match only — no semantic similarity. May be coincidence. Verify carefully.\n"
            "- via:both: found by word match AND semantic similarity. Strong convergence signal.\n"
            "- Situation: WHEN this memory applies — match to current context.\n"
            "- Reasoning: WHY stored. Corrects: replaces this ID. Edges: connections (type tells HOW related).\n\n"
            "Selection rules:\n"
            "- Short confirmations (\"yes\", \"ok\", \"thanks\") → select 0.\n"
            "- Word coincidence without meaning overlap → select 0. (\"React hooks\" ≠ \"brain hooks\")\n"
            "- Unsure? Don't select. No context > wrong context. Silence is better than noise.\n\n"
            "Return ONLY JSON:\n"
            "{\"selected\":[{\"id\":\"...\",\"why\":\"one phrase\"}]}\n"
            "If nothing relevant: {\"selected\":[],\"reason\":\"brief reason\"}")

    prompt = """%s

Session: %s

Conversation (recent, oldest first):
%s
Recently surfaced (deprioritize — only select if the current message specifically needs them):
%s
%s
%s
%d candidates follow. Select 0-%d.

Candidates:

%s""" % (
        prompt_instructions,
        surface_session_context or "(first messages)",
        conversation or "(no recent messages)",
        recalled_text or "(none)",
        retrieval_context,
        intent_context,
        len(candidates),
        cfg['max_selected'],
        candidates_text,
    )

    return prompt, cfg['max_tokens']


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
# EDGE SELECTION — query-aware scoring for S1 surface
#
# Strategy D: 70% node embedding + 30% description embedding.
# Weight as tiebreaker only (currently static ~0.60 for most edges).
# Session fatigue rotates edges across repeated queries.
# 3-message query blending for multi-turn context.
#
# FUTURE: When S2 makes weight dynamic, restore weight's role
# in the score formula (not just tiebreaker).
# FUTURE: When description coverage exceeds 80%, switch to
# strategy F (description-first) — see eval/EDGE_SELECTION_EVAL_SPEC.md.
# FUTURE: When encoder uses real edge types (not 'related'),
# edge_type becomes a scoring signal.
# ═══════════════════════════════════════════════════════════════

# Fatigue constants (session-scoped edge rotation)
K_EDGE_FATIGUE = 0.25   # Gentler than node fatigue — rotation, not suppression
# Node fatigue K: 10.0 base, 10.0 scale — hardcoded in brain_recall.py:684
# TODO: Extract node fatigue K here too.

# Relevance blend weights
EDGE_NODE_WEIGHT = 0.7    # Stored node embedding (title+content)
EDGE_DESC_WEIGHT = 0.3    # Edge description embedding (when available)

# Weight role (tiebreaker until S2 makes weight dynamic)
WEIGHT_TIEBREAKER = 0.01

# Multi-turn query blending
TURN_WEIGHTS = [0.6, 0.3, 0.1]  # current, previous, two_back


def select_edges(connections, query_vec, session=None, limit=3, prior_vecs=None,
                 brain_conn=None):
    """Select the best edges for a query from a node's full connection list.

    This is S1's edge intelligence. It scores each edge by:
      relevance × fatigue_discount + weight × tiebreaker

    Where relevance = 0.7 × cosine(query, stored_node_embedding)
                    + 0.3 × cosine(query, embed(description))  [when desc exists]

    Args:
        connections: list of edge dicts from get_rich_node().connections
        query_vec: numpy array (768d) — current query embedding
        session: SessionContext — for edge fatigue tracking (optional)
        limit: max edges to return
        prior_vecs: list of previous query embeddings for multi-turn blend (optional)
        brain_conn: sqlite3 connection — for loading stored embeddings

    Returns:
        list of edge dicts (top N by score), same shape as input
    """
    import numpy as np

    if not connections or query_vec is None:
        # No query context — fall back to weight order
        return sorted(connections, key=lambda c: c.get('weight', 0), reverse=True)[:limit]

    # Multi-turn blend
    if prior_vecs:
        weights = TURN_WEIGHTS[:len([query_vec] + prior_vecs)]
        total = sum(weights)
        weights = [w / total for w in weights]
        blended = sum(w * v for w, v in zip(weights, [query_vec] + prior_vecs))
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm
    else:
        blended = query_vec

    # Batch-load all stored target embeddings in one SQL round-trip (v23:
    # from node_enrichments). Short-prefix match kept for backward compat
    # with 8-char connection IDs. Falls back silently if brain_conn missing.
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
            stored_embeddings[full_id] = vec

    # Batch-embed edge descriptions in one embedder call (cache-aware).
    descs = [c.get('description', '') for c in connections]
    desc_blobs = _desc_vecs_batched(descs)

    # Pre-compute the blended-query norm once — it's invariant across edges.
    norm_q = float(np.linalg.norm(blended))

    # Score each edge
    scored = []
    for c, desc_blob in zip(connections, desc_blobs):
        tid = c.get('id', '')[:8]
        weight = c.get('weight', 0.5)

        # Node relevance (stored embedding = title + content)
        node_rel = 0.3  # default when embedding missing
        target_vec = stored_embeddings.get(tid)
        if target_vec is not None and len(target_vec) == len(blended) and norm_q > 0:
            norm_t = float(np.linalg.norm(target_vec))
            if norm_t > 0:
                dot = float(np.dot(blended, target_vec))
                node_rel = max(0, dot / (norm_q * norm_t))

        # Description relevance (when available)
        relevance = node_rel
        if desc_blob is not None:
            desc_vec = np.frombuffer(desc_blob, dtype=np.float32)
            if len(desc_vec) == len(blended) and norm_q > 0:
                norm_d = float(np.linalg.norm(desc_vec))
                if norm_d > 0:
                    dot = float(np.dot(blended, desc_vec))
                    desc_rel = max(0, dot / (norm_q * norm_d))
                    relevance = EDGE_NODE_WEIGHT * node_rel + EDGE_DESC_WEIGHT * desc_rel

        # Fatigue discount (session-scoped rotation)
        fatigue_count = 0
        if session is not None:
            fatigue_count = session.get_edge_fatigue(tid)
        fatigue_discount = 1.0 / (1.0 + K_EDGE_FATIGUE * fatigue_count)

        # Final score: relevance-primary, weight as tiebreaker
        score = relevance * fatigue_discount + weight * WEIGHT_TIEBREAKER
        scored.append((score, c))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [c for _, c in scored[:limit]]

    # Update fatigue for selected edges
    if session is not None:
        for c in selected:
            session.increment_edge_fatigue(c.get('id', '')[:8])

    return selected


def format_surface_output(selected, candidates, graph_neighbors=None):
    """Format surfaced selections into structured additionalContext for Claude.

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
