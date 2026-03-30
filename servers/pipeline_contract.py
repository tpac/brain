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
# TRUNCATION LIMITS — per stage, per field
# ═══════════════════════════════════════════════════════════════

# Candidates file (written by daemon, read by distiller + encoding agent)
CANDIDATES_FILE = {
    'content_limit': 1000,
    'max_candidates': 8,
    'include_graph': True,      # _graph with degree 1/2/3 neighbors
}

# Distiller (Haiku) — compact, focused
DISTILLER = {
    'content_limit': 500,       # chars of node content shown
    'max_candidates': 8,
    'user_message_limit': 500,
    'budget_base': 400,         # min output chars
    'budget_per_relevant': 100, # added per relevant candidate
    'budget_long_query_bonus': 100,  # added if query > 100 chars
    'budget_max': 1200,         # cap
    'max_tokens': 500,          # Haiku output cap
}

# Encoding agent (Sonnet) — needs rich context for revision decisions
ENCODING_AGENT = {
    'message_content_limit': 2000,    # per message from message_stream
    'message_display_limit': 600,     # per message in formatted prompt
    'max_messages': 10,               # last N exchanges
    'recall_candidates_limit': 5,     # candidates for brain context
    'max_d1': 3,                      # degree 1 neighbors shown
    'max_d2': 3,                      # degree 2
    'max_d3': 3,                      # degree 3
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
    """Build the full candidates text block for the Haiku distiller.

    This is the single source of truth for how candidates are formatted
    before being sent to the distiller. No other code should build this string.
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
    """Compute dynamic character budget for distiller output."""
    cfg = DISTILLER
    query_len = len(user_message or "")
    budget = cfg['budget_base'] + min(
        cfg['budget_max'] - cfg['budget_base'],
        relevant_count * cfg['budget_per_relevant'] +
        (cfg['budget_long_query_bonus'] if query_len > 100 else 0)
    )
    max_tokens = min(cfg['max_tokens'], budget // 2)
    return budget, max_tokens


def build_distiller_prompt(candidates, user_message):
    """Build the complete distiller prompt. Single entry point.

    Returns: (prompt_string, budget, max_tokens)
    """
    cfg = DISTILLER
    candidates_text, relevant_count = format_candidates_for_distiller(
        candidates, user_message)
    budget, max_tokens = compute_distiller_budget(user_message, relevant_count)

    prompt = """You are the awareness layer of a persistent AI brain.
Distill these memory candidates into focused context for the main AI.

USER MESSAGE: %s

CANDIDATES:
%s
Rules:
- Only include what's DIRECTLY relevant to the user's message
- Preserve node IDs like (id:abc123) so the AI can pull full details
- Include graph connections when they add context (→ related nodes)
- If a correction or rule applies, lead with it
- If nothing is relevant, return just the word EMPTY. No explanation.
- Max %d characters. Be surgical, like a colleague whispering context.
- If this seems like the start of a conversation, be more generous.
- NEVER add your own opinions or analysis. You are a filter, not an advisor.""" % (
        (user_message or "")[:cfg['user_message_limit']],
        candidates_text,
        budget,
    )

    return prompt, budget, max_tokens
