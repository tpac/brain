"""Encoding Contract — S1 turn encoder (Sonnet) config, node formatting, catalog building.

The encoding agent reads conversation turns and creates/revises brain nodes.
This contract defines:
- What the encoder sees (ENCODING_AGENT config)
- How nodes are formatted for the encoder (format_node_for_encoder)
- How the node catalog is built (build_encoder_node_catalog)

Interaction: 'encoding_agent' in interactions table. Prompt is learnable.
"""

from servers.judge_contract import correction_enrich

# ═══════════════════════════════════════════════════════════════
# ENCODING AGENT CONFIG
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# NODE FORMATTING FOR ENCODER
# ═══════════════════════════════════════════════════════════════

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

    # Enrich with correction chains so encoder can revise stale nodes
    corrections = correction_enrich(seen_ids, db_conn)

    lines = ['=== BRAIN NODES SURFACED THIS SESSION (%d unique) ===' % len(seen_ids), '']
    formatted_ids = set()
    for nid in seen_ids:
        formatted = format_node_for_encoder(nid, db_conn)
        if formatted:
            # Append correction annotations
            node_corrs = corrections.get(nid, [])
            for corr in node_corrs:
                if corr["direction"] == "corrected_by":
                    formatted += '\n  ⚠ UPDATED BY: "%s" (%s) — consider revising this node' % (
                        corr["title"][:50], corr["id"])
                elif corr["direction"] == "corrects":
                    formatted += '\n  CORRECTS: "%s" (%s)' % (corr["title"][:50], corr["id"])
            lines.append(formatted)
            lines.append('')
            formatted_ids.add(nid)

    return '\n'.join(lines), formatted_ids
