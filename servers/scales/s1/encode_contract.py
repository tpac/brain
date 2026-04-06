"""Encoding Contract — S1 turn encoder (Sonnet) config and catalog building.

The encoding agent reads conversation turns and creates/revises brain nodes.
This contract defines:
- What the encoder sees (ENCODING_AGENT config)
- How the node catalog is built (build_node_catalog)

Node formatting uses the system contract: servers.contract.format_node()
Interaction: 'encoding_agent' in interactions table. Prompt is learnable.
"""

from servers.scales.s1.recall_contract import correction_enrich
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
    'max_tokens': 4096,               # Sonnet API output cap
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
    """Build deduplicated node catalog from judge outputs across multiple turns.

    Uses system format_node() with S1 config for full rich nodes.
    Adds correction chain annotations on top.

    Args:
        judge_outputs: list of judge_output strings (one per turn, may be None)
        db_conn: brain.db connection for rich metadata lookup

    Returns:
        (catalog_text, node_id_set) — formatted catalog + set of IDs for reference
    """
    import re
    # Extract all node IDs from judge outputs (pattern: id:XXXXXXXX)
    # Supports both hex IDs (d7d1ddfa) and typed-prefix IDs (con_1c0v)
    seen_ids = set()
    for jo in judge_outputs:
        if not jo or jo == '(no selection)':
            continue
        for match in re.finditer(r'id:([a-z0-9_]{6,8})', jo):
            seen_ids.add(match.group(1))

    if not seen_ids:
        return '', set()

    # Enrich with correction chains so encoder can revise stale nodes
    corrections = correction_enrich(seen_ids, db_conn)

    lines = ['Node Catalog (%d nodes surfaced this session)' % len(seen_ids), '']
    formatted_ids = set()
    for nid in seen_ids:
        formatted = format_node(nid, db_conn, config=S1_NODE_CONFIG)
        if formatted:
            # Append correction annotations
            node_corrs = corrections.get(nid, [])
            for corr in node_corrs:
                if corr["direction"] == "corrected_by":
                    formatted += '\n  ⚠ UPDATED BY: "%s" (id:%s) — consider revising' % (
                        corr["title"][:60], corr["id"])
                elif corr["direction"] == "corrects":
                    formatted += '\n  CORRECTS: "%s" (id:%s)' % (corr["title"][:60], corr["id"])
            lines.append(formatted)
            lines.append('')
            formatted_ids.add(nid)

    return '\n'.join(lines), formatted_ids


# Backward compat aliases
format_node_for_encoder = lambda node_id, db_conn: format_node(node_id, db_conn, config=S1_NODE_CONFIG)
build_encoder_node_catalog = build_node_catalog
