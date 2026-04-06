"""Judge Contract — S1 recall judge (Haiku) prompt building, candidate formatting, output formatting.

The judge selects relevant nodes from recall candidates. This contract defines:
- What the judge sees (JUDGE config, CANDIDATES_FILE, neighbor fields)
- How candidates are formatted (format_candidate_for_judge, enrich_candidate_metadata)
- How the prompt is assembled (build_judge_prompt)
- How output is formatted for Claude (format_judge_output)
- Correction enrichment (correction_enrich — shared with encoding)

Interaction: 'judge' in interactions table. Prompt is learnable.
"""

from datetime import datetime, timezone


def _relative_time(iso_str):
    """Convert UTC ISO timestamp to relative time label.

    Returns human-readable age: 'just now', 'today', 'yesterday', '3d ago', '2w ago', '1mo ago'.
    Both judge and Anchor see this instead of raw UTC timestamps.
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
# JUDGE CONFIG
# ═══════════════════════════════════════════════════════════════

# Candidates file (written by daemon, read by judge + encoding agent)
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
JUDGE = {
    'content_limit': 300,           # shorter per node since more candidates
    'max_candidates': 20,           # v9: was 25. FTS5 adds up to 5 more = 25 max total
    'max_selected': 8,              # Haiku picks at most this many
    'user_message_limit': 300,
    'anchor_message_limit': 400,    # v9: was 150. Anchor responses carry design context
    'recent_messages': 7,           # v9: was 5. Deeper conversation window
    'recent_recalls_messages': 10,  # look back 10 messages for previously surfaced nodes
    'session_context_limit': 800,   # shared with ENCODING_AGENT — full session journey
    'judge_session_context_tail': 200,  # v9: judge gets tail of session context (current focus)
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
# CANDIDATE ENRICHMENT
# ═══════════════════════════════════════════════════════════════

try:
    from servers.contract import METADATA_KEYS as _CONTRACT_METADATA_KEYS
except ImportError:
    from contract import METADATA_KEYS as _CONTRACT_METADATA_KEYS

_SITUATION_QUERY = 'SELECT situation_text FROM node_embeddings WHERE node_id = ?'

_EDGES_QUERY = (
    'SELECT n.title, e.edge_type, e.description, e.weight '
    'FROM edges e JOIN nodes n ON e.target_id = n.id '
    'WHERE e.source_id = ? AND e.edge_type != "co_accessed" '
    'ORDER BY e.weight DESC LIMIT ?'
)


def enrich_candidate_metadata(brain, node_id, node_data, config):
    """Add metadata fields to a candidate dict for the Layer 2 judge.

    Reads from MetadataDAL (KV store), node_embeddings, and edges tables.
    Only surfaces keys listed in the contract's METADATA_KEYS.
    """
    if not node_id:
        return

    # Metadata from KV store — only contract-defined keys
    try:
        try:
            from servers.dal_metadata import MetadataDAL
        except ImportError:
            from dal_metadata import MetadataDAL
        dal = MetadataDAL(brain.conn)
        meta = dal.get_fields(node_id, _CONTRACT_METADATA_KEYS)
        for key, value in meta.items():
            node_data[key] = value
    except Exception:
        pass

    # Situation text
    try:
        sit = brain.conn.execute(_SITUATION_QUERY, (node_id,)).fetchone()
        if sit and sit[0]:
            node_data["situation"] = sit[0]
    except Exception:
        pass

    # Top intentional edges with descriptions
    try:
        max_edges = config.get('max_edges_described', 3)
        edges = brain.conn.execute(_EDGES_QUERY, (node_id, max_edges)).fetchall()
        if edges:
            node_data["top_edges"] = [
                {"title": e[0][:60], "type": e[1], "why": e[2] or "", "weight": e[3]}
                for e in edges
            ]
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# CORRECTION ENRICHMENT — shared by judge and encoding
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

        # 1. Edges: find corrected_by edges where our nodes are source or target
        edge_rows = db_conn.execute(
            """SELECT source_id, target_id FROM edges
               WHERE relation = 'corrected_by'
               AND (source_id IN (%s) OR target_id IN (%s))""" % (placeholders, placeholders),
            list(node_ids) + list(node_ids)
        ).fetchall()

        for src, tgt in edge_rows:
            if src in node_ids:
                # src was corrected_by tgt
                title = db_conn.execute(
                    "SELECT title FROM nodes WHERE id = ?", (tgt,)).fetchone()
                if title:
                    corrections.setdefault(src, []).append({
                        "id": tgt[:8], "title": title[0], "direction": "corrected_by"})
            if tgt in node_ids:
                # tgt was corrected_by src (reverse — tgt corrects src)
                title = db_conn.execute(
                    "SELECT title FROM nodes WHERE id = ?", (src,)).fetchone()
                if title:
                    corrections.setdefault(tgt, []).append({
                        "id": src[:8], "title": title[0], "direction": "corrects"})

        # 2. node_metadata: correction_of field
        meta_rows = db_conn.execute(
            """SELECT node_id, correction_of FROM node_metadata
               WHERE node_id IN (%s) AND correction_of IS NOT NULL
               AND correction_of != ''""" % placeholders,
            list(node_ids)
        ).fetchall()

        from servers.dal import NodeDAL
        dal = NodeDAL(db_conn)
        for nid, corrects_id in meta_rows:
            title = dal.get_title(corrects_id[:8])
            if title:
                corrections.setdefault(nid, []).append({
                    "id": corrects_id[:8], "title": title, "direction": "corrects"})

        # 3. Reverse: find nodes that correct OUR nodes (via correction_of field)
        meta_reverse = db_conn.execute(
            """SELECT node_id, correction_of FROM node_metadata
               WHERE correction_of IS NOT NULL AND correction_of != ''"""
        ).fetchall()
        for nid, corrects_id in meta_reverse:
            # Check if corrects_id matches any of our node_ids (prefix match)
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

def format_candidate_for_judge(c, index):
    """Format a single candidate for the judge prompt. Compact, metadata-rich."""
    cfg = JUDGE
    # Header: index, type, title, id, score, confidence, locked, created, discovery
    parts = ["id:%s" % str(c.get("id", ""))[:8]]
    score = c.get("score", 0)
    if score:
        # v9: Cap displayed score at 1.0 — critical boost inflates past 1.0
        # which misleads the judge. Show 'boosted' flag if capped.
        display_score = min(score, 1.0)
        score_str = "match:%.2f" % display_score
        if score > 1.0:
            score_str += ",boosted"
        parts.append(score_str)
    conf = c.get("confidence")
    if conf:
        parts.append("conf:%.1f" % conf)
    if c.get("locked"):
        parts.append("locked")
    # v9: Discovery source — how this candidate was found
    discovery = c.get("discovery", "")
    if discovery and discovery not in ("embedding", "embedding_only", "embedding+keyword"):
        parts.append("via:%s" % discovery)
    # v9.1: Relative time instead of raw UTC — judge and Anchor both think in relative time
    created_rel = _relative_time(c.get("created_at"))
    revised_rel = _relative_time(c.get("revised_at"))
    if revised_rel and created_rel and revised_rel != created_rel:
        parts.append("created %s, revised %s" % (created_rel, revised_rel))
    elif created_rel:
        parts.append(created_rel)

    header = "#%d [%s] \"%s\" (%s)" % (
        index, c.get("type", "?"), c.get("title", "?")[:70], ", ".join(parts))

    lines = [header]

    # Content (truncated)
    content = (c.get("content") or "")[:cfg['content_limit']]
    if content:
        lines.append("  %s" % content)

    # Metadata — only if present (no empty fields)
    situation = c.get("situation", "")
    if situation:
        lines.append("  Situation: %s" % situation[:120])

    reasoning = c.get("reasoning", "")
    if reasoning:
        lines.append("  Reasoning: %s" % reasoning[:120])

    quote = c.get("user_raw_quote", "")
    if quote:
        lines.append("  Quote: \"%s\"" % quote[:120])

    corrects = c.get("correction_of", "")
    if corrects:
        lines.append("  Corrects: %s" % corrects[:30])

    # Top edges (intentional only, not co_accessed)
    edges = c.get("top_edges", [])
    if edges:
        edge_parts = []
        for e in edges[:3]:
            desc = " — %s" % e["why"] if e.get("why") else ""
            edge_parts.append("\"%s\" (%s%s)" % (e["title"][:40], e["type"], desc))
        lines.append("  Edges: " + ", ".join(edge_parts))

    return "\n".join(lines)


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


def build_judge_prompt(candidates, user_message, session_context="",
                       recent_messages=None, recently_recalled=None,
                       retrieval_stats=None, intent=None,
                       prompt_instructions=None):
    """Build the S1 recall judge prompt. Single entry point.

    v9: Added retrieval_stats, intent, score normalization, conversation
    context expansion, session context tail, candidate dedup, discovery tags.
    v10: prompt_instructions from interactions table (learnable boundary).

    Args:
        candidates: List of candidate node dicts (enriched with metadata)
        prompt_instructions: Optional judge instructions from interactions table.
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
    cfg = JUDGE

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
    judge_session_context = ""
    if session_context:
        tail_limit = cfg.get('judge_session_context_tail', 200)
        if len(session_context) > tail_limit:
            judge_session_context = "Current focus: ..." + session_context[-tail_limit:]
        else:
            judge_session_context = session_context

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
        candidates_text += format_candidate_for_judge(c, i) + "\n\n"

    # Instructions: from interactions table (learnable) or hardcoded default
    if not prompt_instructions:
        prompt_instructions = (
            "You are a memory relevance judge for a shared AI brain. The brain stores "
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
        judge_session_context or "(first messages)",
        conversation or "(no recent messages)",
        recalled_text or "(none)",
        retrieval_context,
        intent_context,
        len(candidates),
        cfg['max_selected'],
        candidates_text,
    )

    return prompt, cfg['max_tokens']


def format_judge_output(selected, candidates, graph_neighbors=None,
                        corrections=None):
    """Format the judge's selections into structured additionalContext for Claude.

    Takes Haiku's selected nodes (with "why" reasoning) and the full candidates
    list (with content, metadata, edges). Produces a clean text block that Claude
    reads as its memory context.

    Args:
        corrections: dict from correction_enrich() — {node_id: [{"id", "title", "direction"}]}
    """
    cfg = JUDGE
    if not selected:
        return ""

    # Build a lookup from candidate ID (first 8 chars) to full candidate data
    candidates_by_id = {}
    for c in candidates:
        short_id = str(c.get("id", ""))[:8]
        candidates_by_id[short_id] = c

    lines = ["Brain recalled %d memories:\n" % len(selected)]

    for s in selected[:cfg['max_selected']]:
        sid = str(s.get("id", ""))[:8]
        why = s.get("why", "")
        c = candidates_by_id.get(sid)

        if not c:
            continue

        # Header — v9.1: includes relative time so Anchor knows when this memory is from
        parts = ["id:%s" % sid]
        conf = c.get("confidence")
        if conf:
            parts.append("conf:%.1f" % conf)
        if c.get("locked"):
            parts.append("locked")
        created_rel = _relative_time(c.get("created_at"))
        revised_rel = _relative_time(c.get("revised_at"))
        if revised_rel and created_rel and revised_rel != created_rel:
            parts.append("created %s, revised %s" % (created_rel, revised_rel))
        elif created_rel:
            parts.append(created_rel)
        header = "[%s] \"%s\" (%s)" % (c.get("type", "?"), c.get("title", "?")[:70], ", ".join(parts))
        lines.append(header)

        # Haiku's relevance reasoning
        if why:
            lines.append("Relevance: %s" % why)

        # Content (truncated for context budget)
        content = (c.get("content") or "")[:400]
        if content:
            lines.append("Content: %s" % content)

        # Metadata (only if present)
        situation = c.get("situation", "")
        if situation:
            lines.append("Situation: %s" % situation[:150])

        quote = c.get("user_raw_quote", "")
        if quote:
            lines.append("Quote: \"%s\"" % quote[:150])

        corrects = c.get("correction_of", "")
        if corrects:
            lines.append("Corrects: %s" % corrects[:30])

        # Top edges (from candidate data, not re-queried)
        edges = c.get("top_edges", [])
        if edges:
            edge_parts = []
            for e in edges[:3]:
                desc = " — %s" % e["why"] if e.get("why") else ""
                edge_parts.append("\"%s\" (%s%s)" % (e["title"][:40], e["type"], desc))
            lines.append("Connected: " + ", ".join(edge_parts))

        # Correction chain — show if this node was corrected or corrects another
        if corrections:
            node_corrections = corrections.get(c.get("id", ""), [])
            if not node_corrections:
                # Try short ID match
                node_corrections = corrections.get(c.get("id", "")[:8], [])
            for corr in node_corrections:
                if corr["direction"] == "corrected_by":
                    lines.append("⚠ Updated by: \"%s\" (%s)" % (corr["title"][:50], corr["id"]))
                elif corr["direction"] == "corrects":
                    lines.append("Corrects: \"%s\" (%s)" % (corr["title"][:50], corr["id"]))

        lines.append("")  # blank line between nodes

    # Graph neighbors — connected knowledge from judge-selected seeds
    if graph_neighbors:
        lines.append("Related knowledge (via graph):")
        for nb in graph_neighbors[:6]:  # Cap at 6 neighbors total
            edge_desc = " — %s" % nb["edge_description"] if nb.get("edge_description") else ""
            lines.append("[%s] \"%s\" (%s%s)" % (
                nb.get("type", "?"),
                nb.get("title", "?")[:60],
                nb.get("edge_type", "related"),
                edge_desc))
            content = (nb.get("content") or "")[:200]
            if content:
                lines.append("  %s" % content)
        lines.append("")

    return "\n".join(lines)
