"""Encoding Contract — S1 turn encoder (Sonnet) config and catalog building.

The encoding agent reads conversation turns and creates/revises brain nodes.
This contract defines:
- What the encoder sees (ENCODING_AGENT config)
- How the node catalog is built (build_node_catalog)

Node formatting uses the system contract: servers.contract.format_node()
Interaction: 'encoding_agent' in interactions table. Prompt is learnable.
"""

from servers.contract import format_node

# ═══════════════════════════════════════════════════════════════
# ENCODING AGENT CONFIG
# ═══════════════════════════════════════════════════════════════

# Encoding agent v3.2 (Sonnet) — split node catalog + timeline with references
ENCODING_AGENT = {
    'message_content_limit': 2500,    # per message stored in message_stream (both roles equally)
    'message_display_limit': 2500,    # per message in timeline (both roles — shared learnings, not just Tom's words)
    'max_messages': 20,               # last N messages (~10 turns)
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
    'max_tokens': 12288,              # Sonnet API output cap (raised from 4096)
    'timeline_snippet_limit': 500,    # chars of recalled content shown in timeline (fallback only)
    'session_context_limit': 800,     # session context chars (additive within session, editable by S2)

    # Node catalog: full rich nodes shown once at top, referenced by ID in timeline
    'node_content_limit': None,       # full content — no truncation for encoder
    'node_edge_limit': 5,             # structural edges per node (with descriptions)
}


# ═══════════════════════════════════════════════════════════════
# NODE CATALOG — uses system format_node() with S1 config
# ═══════════════════════════════════════════════════════════════

# S1 encoder node config — full depth, no truncation
S1_NODE_CONFIG = {
    'content_limit': ENCODING_AGENT.get('node_content_limit'),
    'edge_limit': ENCODING_AGENT.get('node_edge_limit', 5),
}


def build_node_catalog(judge_outputs, db_conn):
    """Build deduplicated node catalog from surface outputs across multiple turns.

    Uses system format_node() with S1 config for full rich nodes.
    Adds correction chain annotations on top.

    Args:
        judge_outputs: list of surface_output strings (one per turn, may be None)
        db_conn: brain.db connection for rich metadata lookup

    Returns:
        (catalog_text, node_id_set) — formatted catalog + set of IDs for reference
    """
    import re
    # Extract all node IDs from surface outputs (pattern: id:XXXXXXXX)
    # Supports both hex IDs (d7d1ddfa) and typed-prefix IDs (con_1c0v)
    seen_ids = set()
    for jo in judge_outputs:
        if not jo or jo == '(no selection)':
            continue
        for match in re.finditer(r'id:([a-z0-9_]{6,8})', jo):
            seen_ids.add(match.group(1))

    if not seen_ids:
        return '', set()

    # Skip community nodes — S2CE manages communities, S1E encodes from conversation.
    # S1E still sees "SURFACED: community node" in the timeline but doesn't get
    # the full content in the catalog. This prevents S1E from revising, correcting,
    # or connecting to community nodes instead of their members.
    community_ids = set()
    if seen_ids:
        placeholders = ','.join('?' * len(seen_ids))
        for row in db_conn.execute(
                "SELECT id FROM nodes WHERE id IN (%s) AND type = 'community'" % placeholders,
                list(seen_ids)):
            community_ids.add(row[0])

    # format_node → get_rich_node + render_rich_node includes corrections
    catalog_ids = seen_ids - community_ids
    lines = ['Node Catalog (%d nodes surfaced this session)' % len(catalog_ids), '']
    formatted_ids = set()
    for nid in catalog_ids:
        formatted = format_node(nid, db_conn, config=S1_NODE_CONFIG)
        if formatted:
            lines.append(formatted)
            lines.append('')
            formatted_ids.add(nid)

    return '\n'.join(lines), formatted_ids


# Backward compat aliases
format_node_for_encoder = lambda node_id, db_conn: format_node(node_id, db_conn, config=S1_NODE_CONFIG)
build_encoder_node_catalog = build_node_catalog
