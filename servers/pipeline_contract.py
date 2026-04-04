"""Pipeline Contract — single source of truth for what data flows through recall and encoding.

contract.py defines what fields a node HAS.
pipeline_contract.py defines what fields FLOW at each stage.

Stages:
  RECALL → CANDIDATES FILE → JUDGE (Haiku) → additionalContext → Claude
  RECALL → CANDIDATES FILE → ENCODING AGENT (Sonnet) → brain writes
  RECALL → MCP TOOL OUTPUT → Claude (direct recall)
  RECALL → PRE-EDIT SUGGESTIONS → Claude

To change what the judge sees: edit JUDGE config + build_judge_prompt().
To change what the encoding agent sees: edit ENCODING_AGENT config.
Truncation limits: PIPELINE dict (single source of truth for all stages).
"""


# ═══════════════════════════════════════════════════════════════
# NODE FIELDS — what to include at each stage
# ═══════════════════════════════════════════════════════════════

# Core fields present in every pipeline stage
NODE_CORE_FIELDS = {
    'id', 'type', 'title', 'content', 'confidence', 'locked',
    'revised_at', 'created_at',
}

# Additional fields for richer contexts
NODE_EXTENDED_FIELDS = NODE_CORE_FIELDS | {
    'access_count', 'encoding_source', 'content_summary',
    'emotion', 'emotion_label', 'updated_at',
}


# ═══════════════════════════════════════════════════════════════
# EMBEDDING GROUPS — multi-vector architecture for recall
# ═══════════════════════════════════════════════════════════════
#
# Each node gets 2-4 embedding vectors, stored in node_enrichments table.
# At recall time, each vector's cosine sim is multiplied by its group weight,
# then the top-2 weighted scores are averaged. This requires two vectors to
# "agree" — prevents noisy single-field matches from dominating.
#
# Tested 2026-04-02: z-weighted top2-avg = +20pts R@8, +22pts R@25 vs baseline.
# Title-only was +19/+15. Adding metadata groups adds +1 R@8 and +7 R@25.
# The big win is R@25 — metadata surfaces nodes that title alone misses.
#
# To add a new metadata field: put it in the right group's 'fields' list.
# To add a new group: add an entry here. Recall reads these dynamically.
# Emergent KV fields (not in any group) auto-flow into 'other_meta'.

EMBEDDING_GROUPS = {
    # Group 1: Title — the diagnostic pointer. Always exists. Highest weight.
    'title': {
        'weight': 1.00,
        'fields': ['title'],
        'vector_type': 'title',       # stored as this type in node_enrichments
        'always_compute': True,        # embed even if field is empty (title always exists)
    },
    # Group 2: Blend — existing title+content embedding. Lives in node_embeddings.
    # Not stored in node_enrichments — uses the primary embedding from node_embeddings.
    'blend': {
        'weight': 0.85,
        'fields': ['title', 'content'],
        'vector_type': '_primary',     # special: reads from node_embeddings.embedding
        'always_compute': True,
    },
    # Group 3: High-priority metadata — when is this relevant + who said it.
    # Concatenates situation + quotes into one vector.
    'high_meta': {
        'weight': 0.70,
        'fields': ['situation', 'user_raw_quote', 'anchor_raw_quote'],
        'vector_type': 'high_meta',
        'always_compute': False,       # only if at least one field has data
    },
    # Group 4: Other metadata — why was this stored + behavioral patterns + emergent.
    # Emergent KV fields (not in groups 1-3) auto-flow here.
    'other_meta': {
        'weight': 0.40,
        'fields': ['reasoning', 'correction_pattern', 'source_context', '_emergent'],
        'vector_type': 'other_meta',
        'always_compute': False,
    },
}

# Scoring method for combining group vectors
# 'top2_avg': z-weight each vector, average the top 2 scores
# Requires 2 vectors to agree — prevents noisy single-field matches
EMBEDDING_SCORING_METHOD = 'top2_avg'

# KV fields to skip when building embedding text (not semantic content)
EMBEDDING_SKIP_FIELDS = {
    'metadata_created_at', 'validation_count', 'last_validated',
    'alternatives', 'change_impacts',
}

# Max chars per field when building group embedding text
EMBEDDING_FIELD_CHAR_LIMIT = 300


def get_group_fields(group_name):
    """Get the field names for an embedding group. Used by remember() and revise()."""
    group = EMBEDDING_GROUPS.get(group_name, {})
    return [f for f in group.get('fields', []) if f != '_emergent']


def get_group_weight(vector_type):
    """Get the z-index weight for a vector type. Used by recall scoring."""
    for group in EMBEDDING_GROUPS.values():
        if group.get('vector_type') == vector_type:
            return group['weight']
    return EMBEDDING_GROUPS['other_meta']['weight']  # default for unknown types


# ═══════════════════════════════════════════════════════════════
# TRUNCATION LIMITS — per stage, per field
# Single source of truth. Never hardcode a limit in application code.
# ═══════════════════════════════════════════════════════════════

# Pipeline-wide limits (used across multiple stages)
PIPELINE = {
    'user_message_store': 500,       # user message stored as config (last_user_message)
    'user_message_query': 500,       # user message used as recall/priming query input
    'assistant_response_store': 4000, # assistant response stored to message_stream
    'recent_message_content': 300,   # recent message content for judge conversation context
    'recall_log_query': 500,         # query text stored in recall_log
    'recall_log_title': 80,          # node title in recall_log
    'recall_log_snippet': 150,       # content snippet in recall_log
    'encoding_state_compat': 2000,   # encoding_agent_state backward compat config
}

# Candidates file (written by daemon, read by judge + encoding agent)
CANDIDATES_FILE = {
    'content_limit': 1000,
    'max_candidates': 25,
    'include_graph': True,      # _graph with degree 1/2/3 neighbors
    'include_metadata': True,   # situation, reasoning, user_raw_quote, correction_of
    'metadata_fields': ['situation_text', 'reasoning', 'user_raw_quote', 'correction_of'],
    'max_edges_described': 3,   # top edges with descriptions per candidate
}

# Judge (Haiku) — selects relevant nodes with reasoning (replaces distiller)
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

# Encoding agent v3.2 (Sonnet) — split node catalog + timeline with references
ENCODING_AGENT = {
    'message_content_limit': 2500,    # per message stored in message_stream (both roles equally)
    'message_display_limit': 2500,    # per message in timeline (both roles — shared learnings, not just Tom's words)
    'max_messages': 10,               # last N exchanges
    'recall_candidates_limit': 5,     # candidates per turn (pre-attached)
    'max_rounds': 5,                  # Sonnet API round limit (target: 2-3)
    'journal_max_chars': 8000,        # encoding journal truncation limit
    'max_d1': 3,                      # degree 1 neighbors shown
    'max_d2': 3,                      # degree 2
    'max_d3': 3,                      # degree 3
    'recall_on_create_limit': 5,      # max related_nodes returned per remember()
    'recall_on_create_content_limit': 500,  # chars of content per related node
    'recall_on_create_query_limit': 200,    # chars of content used in recall query
    'journal_entry_limit': 2000,      # max chars per journal entry
    'max_tokens': 4096,               # Sonnet API output cap
    'timeline_snippet_limit': 500,    # chars of recalled content shown in timeline (fallback only)
    'session_context_limit': 800,     # session context chars (additive within session)

    # Node catalog: full rich nodes shown once at top, referenced by ID in timeline
    'node_content_limit': None,       # full content — no truncation for encoder
    'node_edge_limit': 5,             # structural edges per node (with descriptions)
}


def format_node_for_encoder(node_id, db_conn):
    """Format a single node with full rich metadata for the encoding agent.

    The encoder needs everything to make good decisions:
    - Full content (not truncated) — to judge quality and decide revisions
    - Situation — to prevent cross-project encoding mistakes
    - Reasoning — to understand WHY the node was created
    - Keywords — for connection discovery
    - Edge descriptions — to understand the graph neighborhood
    - Confidence/locked/type — to know what kind of node it is

    Excluded: relevance reasoning (that's for Claude, not encoder).
    """
    cfg = ENCODING_AGENT
    try:
        # Core node data
        row = db_conn.execute(
            "SELECT id, type, title, content, keywords, confidence, locked, "
            "emotion, encoding_source, created_at, personal, personal_context "
            "FROM nodes WHERE id LIKE ?", (node_id + '%',)).fetchone()
        if not row:
            return None

        nid = row[0]
        lines = ['[%s] "%s" (id:%s, conf:%s%s)' % (
            row[1] or '?', row[2] or '?', nid[:8],
            ('%.1f' % row[5]) if row[5] else '?',
            ', locked' if row[6] else '')]

        # Content — full, not truncated
        content = row[3] or ''
        content_limit = cfg.get('node_content_limit')
        if content_limit:
            content = content[:content_limit]
        if content:
            lines.append('  Content: %s' % content)

        # Situation
        sit = db_conn.execute(
            "SELECT situation_text FROM node_embeddings WHERE node_id = ?",
            (nid,)).fetchone()
        if sit and sit[0]:
            lines.append('  Situation: %s' % sit[0])

        # All metadata KV — don't filter by key, schema evolves
        meta = db_conn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id = ?",
            (nid,)).fetchall()
        for m in meta:
            if m[1] and m[0] not in ('metadata_created_at',):  # skip purely operational
                lines.append('  %s: %s' % (m[0].replace('_', ' ').title(), m[1][:300]))

        # Keywords
        if row[4]:
            lines.append('  Keywords: %s' % row[4])

        # Personal context (cross-project guard)
        if row[10] and row[11]:
            lines.append('  Context: %s (%s)' % (row[10], row[11]))

        # Structural edges with descriptions
        edge_limit = cfg.get('node_edge_limit', 5)
        edges = db_conn.execute(
            "SELECT e.relation, e.weight, n2.title, n2.type, e.description "
            "FROM edges e JOIN nodes n2 ON n2.id = e.target_id "
            "WHERE e.source_id = ? AND e.relation NOT IN ('co_accessed', 'emergent_bridge') "
            "ORDER BY e.weight DESC LIMIT ?",
            (nid, edge_limit)).fetchall()
        if edges:
            edge_parts = []
            for e in edges:
                desc = ' — %s' % e[4] if e[4] else ''
                edge_parts.append('"%s" [%s] (%s%s)' % (
                    (e[2] or '')[:50], e[3] or '?', e[0], desc))
            lines.append('  Edges: %s' % ', '.join(edge_parts))

        return '\n'.join(lines)
    except Exception:
        return None


def build_encoder_node_catalog(judge_outputs, db_conn):
    """Build deduplicated node catalog from judge outputs across multiple turns.

    Args:
        judge_outputs: list of judge_output strings (one per turn, may be None)
        db_conn: brain.db connection for rich metadata lookup

    Returns:
        (catalog_text, node_id_set) — formatted catalog + set of IDs for reference
    """
    import re
    # Extract all node IDs from judge outputs (pattern: id:XXXXXXXX)
    seen_ids = set()
    for jo in judge_outputs:
        if not jo or jo == '(no selection)':
            continue
        # Match id:8-char-hex pattern
        for match in re.finditer(r'id:([a-f0-9]{8})', jo):
            seen_ids.add(match.group(1))

    if not seen_ids:
        return '', set()

    lines = ['=== BRAIN NODES SURFACED THIS SESSION (%d unique) ===' % len(seen_ids), '']
    formatted_ids = set()
    for nid in seen_ids:
        formatted = format_node_for_encoder(nid, db_conn)
        if formatted:
            lines.append(formatted)
            lines.append('')
            formatted_ids.add(nid)

    return '\n'.join(lines), formatted_ids


# MCP tool output (direct recall by Claude)
MCP_OUTPUT = {
    'content_limit': None,      # no truncation — Claude asked, show everything
    'max_results': 20,
    'enrich_top_n': 3,          # top N get metadata + neighbors
}

# Pre-edit suggestions
PRE_EDIT = {
    'title_limit': 80,
    'content_limit_engineering': 350,
    'content_limit_code': 350,
    'content_limit_other': 250,
    'content_limit_impact': 300,
}


# ═══════════════════════════════════════════════════════════════
# NEIGHBOR FIELDS — what to show for graph neighbors at each stage
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


# ═══════════════════════════════════════════════════════════════
# PRECISION — DEPRECATED: use PIPELINE['recall_log_*'] instead.
# Kept for backward compat with callers that import PRECISION directly.
PRECISION = {
    'title_limit': 100,       # → PIPELINE['recall_log_title']
    'snippet_limit': 150,     # → PIPELINE['recall_log_snippet']
    'query_limit': 500,       # → PIPELINE['recall_log_query']
}


# ═══════════════════════════════════════════════════════════════
# FORMATTERS — standard string builders that read from this contract
# ═══════════════════════════════════════════════════════════════

def format_node_header(node, id_length=8):
    """Standard one-line node header used across all stages."""
    locked = "LOCKED " if node.get("locked") else ""
    return "[%s] %s%s (id:%s, conf:%.2f, revised:%s, created:%s)" % (
        node.get("type", "?"),
        locked,
        node.get("title", "?"),
        str(node.get("id", ""))[:id_length],
        node.get("confidence") or 0,
        node.get("revised_at") or "never",
        str(node.get("created_at") or "")[:10],
    )


def format_neighbor_d1(nb):
    """Standard degree-1 neighbor line."""
    t = NEIGHBOR_TRUNCATION
    locked = "LOCKED " if nb.get("locked") else ""
    line = "  → %s: %s\"%s\" (%s, id:%s, conf:%.2f, revised:%s)" % (
        nb.get("relation", "related"),
        locked,
        str(nb.get("title", ""))[:t['d1_title']],
        nb.get("type", "?"),
        str(nb.get("id", ""))[:t['d1_id']],
        nb.get("confidence") or 0,
        nb.get("revised_at") or "never",
    )
    summary = nb.get("content_summary") or ""
    if summary:
        line += "\n      %s" % summary[:t['d1_content_summary']]
    return line


def format_neighbor_d2(nb):
    """Standard degree-2 neighbor breadcrumb."""
    t = NEIGHBOR_TRUNCATION
    return "\"%s\" (%s, id:%s)" % (
        str(nb.get("title", ""))[:t['d2_title']],
        nb.get("type", "?"),
        str(nb.get("id", ""))[:t['d2_id']],
    )


# ═══════════════════════════════════════════════════════════════
# JUDGE (Layer 2) — relevance judgment prompt
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
    created = str(c.get("created_at") or "")[:19]  # trim to seconds, UTC
    if created:
        parts.append(created + "Z")

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
                       retrieval_stats=None, intent=None):
    """Build the Layer 2 judge prompt. Single entry point.

    v9: Added retrieval_stats, intent, score normalization, conversation
    context expansion, session context tail, candidate dedup, discovery tags.

    Args:
        candidates: List of candidate node dicts (enriched with metadata)
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
        from .brain_constants import RETRIEVAL_LOW_CONFIDENCE
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

    prompt = """You are a memory relevance judge for a shared AI brain. The brain stores memories from conversations between an operator (Tom) and an AI assistant (Anchor). You decide which memories help Anchor respond to Tom's next message.

Session: %s

Conversation (recent, oldest first):
%s
Recently surfaced (deprioritize — only select if the current message specifically needs them):
%s
Field guide:
- match: similarity to query (0-1). High match = topically close, but topic alone ≠ relevant. 'boosted' means score was artificially raised (critical node).
- conf: system confidence (0-1). Higher = more established.
- locked: operator-confirmed important.
- via:fts5_only: found by word match only — no semantic similarity. May be coincidence. Verify carefully.
- via:both: found by word match AND semantic similarity. Strong convergence signal.
- Situation: WHEN this memory applies — match to current context.
- Reasoning: WHY stored. Corrects: replaces this ID. Edges: connections (type tells HOW related).

%s
%s
%d candidates follow. Select 0-%d.
- Short confirmations ("yes", "ok", "thanks") → select 0.
- Word coincidence without meaning overlap → select 0. ("React hooks" ≠ "brain hooks")
- Unsure? Don't select. No context > wrong context. Silence is better than noise.

Return ONLY JSON:
{"selected":[{"id":"...","why":"one phrase"}]}
If nothing relevant: {"selected":[],"reason":"brief reason"}

Candidates:

%s""" % (
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


def format_judge_output(selected, candidates, graph_neighbors=None):
    """Format the judge's selections into structured additionalContext for Claude.

    Takes Haiku's selected nodes (with "why" reasoning) and the full candidates
    list (with content, metadata, edges). Produces a clean text block that Claude
    reads as its memory context.

    Example output:
        Brain recalled 3 memories:

        [rule] "Be proactive about learning" (id:54132e56, conf:0.95, locked)
        Relevance: directly applies — Tom is asking about proactive brain behavior
        Content: Tom: 'Make sure you are proactive about asking questions...'
        Situation: When brain detects gap
        Connected: → "Learn EX.CO" (depends_on) → "Recall precision crisis" (addresses)

        [correction] "Project field exists but recall doesn't use it" (id:ab12cd34)
        Relevance: Tom caught this gap before — don't repropose
        Content: Tom caught Claude proposing a solution when the infrastructure already existed...
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

        # Header
        parts = ["id:%s" % sid]
        conf = c.get("confidence")
        if conf:
            parts.append("conf:%.1f" % conf)
        if c.get("locked"):
            parts.append("locked")
        header = "[%s] \"%s\" (%s)" % (c.get("type", "?"), c.get("title", "?")[:70], ", ".join(parts))
        lines.append(header)

        # Haiku's relevance reasoning — this is the key addition over the old distiller
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

        lines.append("")  # blank line between nodes

    # Layer 3: Graph neighbors — connected knowledge from judge-selected seeds
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


# ═══════════════════════════════════════════════════════════════
# CANDIDATE ENRICHMENT — metadata fields for Layer 2 judge
# ═══════════════════════════════════════════════════════════════

# Metadata fields to surface to the judge (from contract)
try:
    from .contract import METADATA_KEYS as _CONTRACT_METADATA_KEYS
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
            from .dal_metadata import MetadataDAL
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
