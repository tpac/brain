"""Pipeline Contract — single source of truth for what data flows through recall and encoding.

contract.py defines what fields a node HAS.
pipeline_contract.py defines what fields FLOW at each stage.

Without this, truncation limits, field selections, and neighbor formats were
hardcoded in 6+ files. Changing a limit required grep + manual edits + surprises.

Stages:
  RECALL → CANDIDATES FILE → DISTILLER (Haiku) → [BRAIN] context → Claude
  RECALL → CANDIDATES FILE → ENCODING AGENT (Sonnet) → brain writes
  RECALL → MCP TOOL OUTPUT → Claude (direct recall)
  RECALL → PRE-EDIT SUGGESTIONS → Claude

To change what the distiller sees: edit DISTILLER_*.
To change what the encoding agent sees: edit ENCODING_*.
To add a field everywhere: add to NODE_CORE_FIELDS + update stage configs.
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

# Judge (Haiku) — selects relevant nodes with reasoning (replaces distiller)
JUDGE = {
    'content_limit': 300,           # shorter per node since more candidates
    'max_candidates': 25,           # wide net
    'max_selected': 8,              # Haiku picks at most this many
    'user_message_limit': 300,
    'recent_messages': 5,           # last 5 user messages for context
    'recent_recalls_messages': 10,  # look back 10 messages for previously surfaced nodes
    'session_context_limit': 400,   # encoder's session summary (evolves, carries multiple topics)
    'max_tokens': 600,              # Haiku output cap
}

# DEPRECATED — kept for backward compat during migration
DISTILLER = {
    'content_limit': 500,
    'max_candidates': 8,
    'user_message_limit': 500,
    'budget_base': 400,
    'budget_per_relevant': 100,
    'budget_long_query_bonus': 100,
    'budget_max': 1200,
    'max_tokens': 500,
}

# Encoding agent v3 (Sonnet) — timeline with pre-attached recall
ENCODING_AGENT = {
    'message_content_limit': 2000,    # per message from message_stream
    'message_display_limit': 800,     # per message in formatted timeline (increased for v3)
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
    'timeline_snippet_limit': 500,    # chars of recalled content shown in timeline
}

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
# PRECISION LOGGING — what to capture for evaluation
# ═══════════════════════════════════════════════════════════════

PRECISION = {
    'title_limit': 100,
    'snippet_limit': 150,
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


def format_candidates_for_distiller(candidates, user_message):
    """DEPRECATED 2026-04-01 — replaced by format_candidate_for_judge + build_judge_prompt.
    Kept for backward compat during migration. Remove when distiller code is fully gone.
    """
    cfg = DISTILLER
    text = ""
    relevant_count = 0

    for c in candidates[:cfg['max_candidates']]:
        # Node header
        text += format_node_header(c) + "\n"
        # Content
        text += "  %s\n" % (c.get("content") or "")[:cfg['content_limit']]

        # Graph neighborhood
        graph = c.get("_graph", {})
        d1 = graph.get("degree_1", [])
        for nb in d1[:3]:
            text += format_neighbor_d1(nb) + "\n"

        d2 = graph.get("degree_2", [])
        if d2:
            d2_items = ", ".join(format_neighbor_d2(n) for n in d2[:3])
            text += "  →→ %s\n" % d2_items

        text += "\n"
        if (c.get("confidence") or 0) > 0.3:
            relevant_count += 1

    return text, relevant_count


def compute_distiller_budget(user_message, relevant_count):
    """DEPRECATED 2026-04-01 — replaced by JUDGE config. Remove with distiller."""
    cfg = DISTILLER
    query_len = len(user_message or "")
    budget = cfg['budget_base'] + min(
        cfg['budget_max'] - cfg['budget_base'],
        relevant_count * cfg['budget_per_relevant'] +
        (cfg['budget_long_query_bonus'] if query_len > 100 else 0)
    )
    max_tokens = min(cfg['max_tokens'], budget // 2)
    return budget, max_tokens


def build_distiller_prompt(candidates, user_message, recent_messages=None):
    """DEPRECATED 2026-04-01 — replaced by build_judge_prompt.
    Kept for backward compat. Remove when pre_response_recall.py migration is confirmed stable.
    """
    cfg = DISTILLER
    candidates_text, relevant_count = format_candidates_for_distiller(
        candidates, user_message)
    budget, max_tokens = compute_distiller_budget(user_message, relevant_count)

    # Format recent messages
    recent_text = ""
    if recent_messages:
        for msg in recent_messages[-5:]:
            role = (msg.get("role") or "?").upper()
            content = (msg.get("content") or "")[:200]
            recent_text += "[%s]: %s\n" % (role, content)

    prompt = """You are the awareness layer of a persistent AI brain.
Filter memory candidates to what's relevant for the main AI's next response.

RECENT MESSAGES (last 5):
%s

USER'S LATEST MESSAGE: %s

CANDIDATES:
%s

Rules:
- Only include what's DIRECTLY relevant to the user's message and current session topic
- Preserve node IDs like (id:abc123)
- Include graph connections when they add useful context
- Lead with corrections/rules if they apply
- If nothing is relevant, return EMPTY
- Max %d characters""" % (
        recent_text or "(no recent messages)",
        (user_message or "")[:cfg['user_message_limit']],
        candidates_text,
        budget,
    )

    return prompt, budget, max_tokens


# ═══════════════════════════════════════════════════════════════
# JUDGE (Layer 2) — relevance judgment prompt
# ═══════════════════════════════════════════════════════════════

def format_candidate_for_judge(c, index):
    """Format a single candidate for the judge prompt. Compact, metadata-rich."""
    cfg = JUDGE
    # Header: index, type, title, id, score, confidence, locked, created
    parts = ["id:%s" % str(c.get("id", ""))[:8]]
    score = c.get("score", 0)
    if score:
        parts.append("match:%.2f" % score)
    conf = c.get("confidence")
    if conf:
        parts.append("conf:%.1f" % conf)
    if c.get("locked"):
        parts.append("locked")
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


def build_judge_prompt(candidates, user_message, session_context="",
                       recent_messages=None, recently_recalled=None):
    """Build the Layer 2 judge prompt. Single entry point.

    Args:
        candidates: List of candidate node dicts (enriched with metadata)
        user_message: The user's latest message
        session_context: Encoder's session summary (from brain_meta)
        recent_messages: List of {"role": str, "content": str}
        recently_recalled: List of {"id": str, "title": str} from last N recalls

    Returns: (prompt_string, max_tokens)
    """
    cfg = JUDGE

    # Format conversation context (both roles, asymmetric truncation)
    conversation = ""
    if recent_messages:
        for msg in recent_messages[-(cfg['recent_messages']):]:
            role = msg.get("role", "?")
            if role == "user":
                label = "Tom"
                content = (msg.get("content") or "")[:300]
            else:
                label = "Anchor"
                content = (msg.get("content") or "")[:150]
            conversation += "%s: %s\n" % (label, content)

    # Append current user message (not yet in message_stream — stored on Stop, not Submit)
    if user_message:
        conversation += "Tom: %s\n" % (user_message or "")[:300]

    # Format recently recalled (lightweight — id + title only)
    recalled_text = ""
    if recently_recalled:
        for r in recently_recalled:
            recalled_text += "%s \"%s\"\n" % (str(r.get("id", ""))[:8], r.get("title", "")[:60])

    # Format candidates
    candidates_text = ""
    for i, c in enumerate(candidates[:cfg['max_candidates']], 1):
        candidates_text += format_candidate_for_judge(c, i) + "\n\n"

    prompt = """You are a memory relevance judge for a shared AI brain. The brain stores memories from conversations between an operator (Tom) and an AI assistant (Anchor). You decide which memories help Anchor respond to Tom's next message.

Session: %s

Conversation (recent, oldest first):
%s
Recently surfaced (deprioritize — only select if the current message specifically needs them):
%s
%d candidate memories follow. Each has a type, title, and content snippet.

Field guide:
- match: how similar this memory is to the current message (0-1, from embedding search). High match means topically close, but topic alone doesn't mean relevant.
- conf: system confidence in this memory (0-1). Higher = more established knowledge.
- locked: manually confirmed as important by the operator.
- Situation: describes WHEN this memory is relevant — use this to judge if it applies now.
- Reasoning: WHY this memory was stored — helps you understand its purpose.
- Quote: the operator's exact words when this was learned.
- Corrects: ID of a memory this one replaces or updates. If you surface the original, surface the correction too.
- Edges: connections to other memories. The type (extends, corrects, depends_on, addresses) tells you HOW they're related.

Not all memories have all fields — older memories may only have title and content. Judge by what's available.

Example: Tom is working on a web app project and asks "why is the daemon crashing on startup?" Candidates include "DAEMON_HOST 127.0.0.1 breaks macOS: localhost resolves to IPv6" (match:0.78, situation: debugging daemon persistence, created 2026-03-18), "Refactoring strategy: one layer per session" (match:0.65, locked), and "Tom generalizes specific fixes into structural principles" (match:0.42).
Good judgment: reject ALL three. "DAEMON_HOST 127.0.0.1" is about the brain plugin's daemon, not the web app's daemon — situation says "debugging daemon persistence" which is brain-specific. "Refactoring strategy" is a general principle but doesn't help debug a crash. "Tom generalizes fixes" is a pattern about Tom, not about daemons. Return {"selected":[],"reason":"all candidates are from brain-plugin context, none relate to the web app daemon crash"}.

Critical rules:
- If the user's message is a short confirmation ("yes", "ok", "got it", "thanks", "continue"), select 0.
- If the candidates are only tangentially related — they share a word but not the meaning — select 0. Example: a query about "React hooks" matching a memory about "brain hooks" is a word coincidence, not relevance.
- If you're unsure whether a candidate helps, don't select it. The assistant is better off without context than with misleading context.

Select 0-%d memories. Return ONLY this JSON, no other text:
{"selected":[{"id":"...","why":"one phrase for Anchor explaining relevance"}]}

If nothing is relevant, return: {"selected":[],"reason":"brief reason"}
Silence is better than noise.

Candidates:

%s""" % (
        session_context or "(first messages — no session context yet)",
        conversation or "(no recent messages)",
        recalled_text or "(none)",
        len(candidates[:cfg['max_candidates']]),
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
