"""Brain Data Contract — the single source of truth for node fields.

Every layer reads from this file:
  - schema.py → DDL generation
  - dal.py → what to SELECT and validate
  - dispatch → what fields are valid on write
  - recall → which fields embed and score
  - voice → what to render
  - mcp → tool schema generation
  - encoding prompt → what fields the agent can use

Field categories:
  STRUCTURAL: columns on 'nodes' table. Typed, validated, drive SQL/math.
  PROMOTED: additional storage (node_embeddings, node_metadata). Have behavior.
  FREE: anything else. Stored in nodes.metadata JSON blob. Watched for promotion.

To add a new field:
  1. Add it to PROMOTED_FIELDS here
  2. Run tests
  3. That's it. All layers pick it up.
"""


# ── STRUCTURAL FIELDS ──
# These are columns on the 'nodes' table.
# Changing these requires schema migration.

STRUCTURAL_FIELDS = {
    "id":         {"store": "nodes", "type": "str", "required": True, "immutable": True},
    "type":       {"store": "nodes", "type": "str", "required": True},
    "title":      {"store": "nodes", "type": "str", "required": True},
    "content":    {"store": "nodes", "type": "str", "replace_on_revise": True, "history": "revision_history in metadata_kv (last 5)"},
    "keywords":   {"store": "nodes", "type": "str"},
    "confidence": {"store": "nodes", "type": "float", "range": (0.0, 1.0), "default": 1.0},
    "locked":     {"store": "nodes", "type": "bool", "default": False},
    "archived":   {"store": "nodes", "type": "bool", "default": False},
    "critical":   {"store": "nodes", "type": "bool", "default": False},
    "emotion":    {"store": "nodes", "type": "float"},
    "emotion_label": {"store": "nodes", "type": "str", "default": "neutral"},
    "project":    {"store": "nodes", "type": "str"},
    "personal":   {"store": "nodes", "type": "str"},
    "personal_context": {"store": "nodes", "type": "str"},
    "evolution_status":  {"store": "nodes", "type": "str"},
    "source_turn_id":   {"store": "nodes", "type": "str", "description": "message_stream ID that produced this node (episode linkage)"},
    "encoding_source":  {"store": "nodes", "type": "str", "description": "Who created this node. Convention: category:process. anchor = direct MCP, encoder:sonnet = encoding agent, idle:dreams/redistribution/etc, hook:boot/compaction. Only anchor can lock."},
}


# ── PROMOTED FIELDS ──
# These live in secondary tables but have system behavior.
# "embeds": True → gets its own embedding vector, scored during recall.
# "surfaces_in": where the field is shown (engineering, boot, distiller, etc.)

PROMOTED_FIELDS = {
    "situation": {
        "store": "node_embeddings",
        "column": "situation_text",
        "type": "str",
        "embeds": True,
        "description": "When is this knowledge relevant? One sentence.",
    },
    "reasoning": {
        "store": "metadata_kv",
        "type": "str",
        "description": "Why this was encoded — decision rationale.",
    },
    "user_raw_quote": {
        "store": "metadata_kv",
        "type": "str",
        "description": "Operator's exact words.",
    },
    "anchor_raw_quote": {
        "store": "metadata_kv",
        "type": "str",
        "description": "Anchor's exact words — reflections, realizations, insights.",
    },
    "correction_of": {
        "store": "metadata_kv",
        "type": "str",
        "description": "Node ID this corrects.",
    },
    "correction_pattern": {
        "store": "metadata_kv",
        "type": "str",
        "description": "Behavioral pattern behind the correction.",
    },
    "source_context": {
        "store": "metadata_kv",
        "type": "str",
        "description": "Session/context when this was encoded.",
    },
}

# Metadata field names (for generic read/write through MetadataDAL)
METADATA_KEYS = [k for k, v in PROMOTED_FIELDS.items() if v.get("store") == "metadata_kv"]


# ── COMBINED VIEW ──

ALL_FIELDS = {**STRUCTURAL_FIELDS, **PROMOTED_FIELDS}


# ── HELPER FUNCTIONS ──

def get_writable_fields():
    """Fields that can be set via remember() or revise()."""
    return {k: v for k, v in ALL_FIELDS.items()
            if not v.get("immutable") and k != "archived"}


def get_remember_fields():
    """Fields that brain.remember() accepts — ALL writable fields.
    Structural fields go to nodes table, promoted fields go to their
    respective stores (node_metadata, node_embeddings)."""
    return get_writable_fields()


def get_embeddable_fields():
    """Fields that have their own embedding vector."""
    return {k: v for k, v in ALL_FIELDS.items() if v.get("embeds")}


def get_fields_for_store(store_name):
    """Fields stored in a specific table."""
    return {k: v for k, v in ALL_FIELDS.items() if v.get("store") == store_name}


def validate_field(name, value):
    """Validate a field value against the contract. Returns (ok, error_msg)."""
    if name not in ALL_FIELDS:
        return True, None  # Unknown field — free field, no validation

    spec = ALL_FIELDS[name]
    if value is None:
        return True, None  # NULL is always valid

    expected_type = spec.get("type")
    if expected_type == "float":
        try:
            value = float(value)
        except (ValueError, TypeError):
            return False, "%s must be a number, got %s" % (name, type(value).__name__)
        range_check = spec.get("range")
        if range_check and not (range_check[0] <= value <= range_check[1]):
            return False, "%s must be between %s and %s" % (name, range_check[0], range_check[1])
    elif expected_type == "bool":
        if not isinstance(value, (bool, int)):
            return False, "%s must be boolean, got %s" % (name, type(value).__name__)
    elif expected_type == "str":
        if not isinstance(value, str):
            return False, "%s must be string, got %s" % (name, type(value).__name__)

    return True, None


# ── NODE FORMATTING ──
# The standard way any LLM consumer sees a node.
# A node is never naked — it always includes edges, corrections, metadata.
# Config controls depth (edge_limit, content_limit), not shape.

# Default config — full depth, no truncation
NODE_FORMAT_DEFAULTS = {
    'content_limit': None,      # None = full content
    'edge_limit': 5,            # structural edges per node
    'metadata_limit': 300,      # chars per metadata value
    'edge_filter': ('co_accessed', 'emergent_bridge'),  # relations to exclude
}


def format_node(node_id, db_conn, config=None):
    """Format a node for LLM consumption — always complete.

    Returns: formatted string or None if node not found.

    A node includes: type, title, id, confidence, locked status,
    full content, situation, all metadata KV, keywords, personal context,
    encoding source, created_at, and structural edges with descriptions.

    Config overrides (from NODE_FORMAT_DEFAULTS):
        content_limit: truncate content (None = full)
        edge_limit: max edges shown (default 5)
        metadata_limit: chars per metadata value (default 300)
        edge_filter: edge relations to exclude (default: co_accessed, emergent_bridge)
    """
    cfg = {**NODE_FORMAT_DEFAULTS, **(config or {})}
    try:
        from servers.dal import NodeDAL
        full_id = NodeDAL(db_conn).resolve_id(node_id)
        if not full_id:
            return None
        row = db_conn.execute(
            "SELECT id, type, title, content, keywords, confidence, locked, "
            "emotion, encoding_source, created_at, personal, personal_context, "
            "revised_at "
            "FROM nodes WHERE id = ?", (full_id,)).fetchone()
        if not row:
            return None

        nid = row[0]
        lines = ['--- [%s] "%s" (id:%s, conf:%s%s%s) ---' % (
            row[1] or '?', row[2] or '?', nid[:8],
            ('%.1f' % row[5]) if row[5] else '?',
            ', locked' if row[6] else '',
            ', src:%s' % row[8] if row[8] else '')]

        # Content (indented to separate from header)
        content = row[3] or ''
        if cfg['content_limit']:
            content = content[:cfg['content_limit']]
        if content:
            lines.append('  Content: %s' % content)

        # Situation (own embedding — key for recall)
        sit = db_conn.execute(
            "SELECT situation_text FROM node_embeddings WHERE node_id = ?",
            (nid,)).fetchone()
        if sit and sit[0]:
            lines.append('  Situation: %s' % sit[0])

        # Metadata KV — resolve correction_of to readable reference
        meta = db_conn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id = ?",
            (nid,)).fetchall()
        meta_limit = cfg['metadata_limit']
        skip_keys = ('metadata_created_at', 'revision_history')
        for m in meta:
            if not m[1] or m[0] in skip_keys:
                continue
            key, val = m[0], m[1]
            # Resolve correction_of to title + short ID
            if key == 'correction_of':
                corr_row = db_conn.execute(
                    "SELECT id, title FROM nodes WHERE id LIKE ?",
                    (val[:8] + '%',)).fetchone()
                if corr_row:
                    lines.append('  ⚠ Corrects: "%s" (id:%s)' % (corr_row[1][:60], corr_row[0][:8]))
                else:
                    lines.append('  ⚠ Corrects: id:%s' % val[:8])
                continue
            lines.append('  %s: %s' % (key.replace('_', ' ').title(), val[:meta_limit]))

        # Keywords
        if row[4]:
            lines.append('  Keywords: %s' % row[4])

        # Personal context (cross-project guard)
        if row[10] and row[11]:
            lines.append('  Context: %s (%s)' % (row[10], row[11]))

        # Dates
        if row[9]:
            date_str = 'Created: %s' % row[9][:10]
            if row[12]:  # revised_at
                date_str += ' | Revised: %s' % row[12][:10]
            lines.append('  %s' % date_str)

        # Structural edges — nested, one per line
        edge_limit = cfg['edge_limit']
        edge_filter = cfg['edge_filter']
        placeholders = ','.join('?' for _ in edge_filter)
        edges = db_conn.execute(
            "SELECT e.relation, e.weight, n2.title, n2.type, e.description, "
            "n2.id, n2.created_at, n2.revised_at "
            "FROM edges e JOIN nodes n2 ON n2.id = e.target_id "
            "WHERE e.source_id = ? AND e.relation NOT IN (%s) "
            "ORDER BY e.weight DESC LIMIT ?" % placeholders,
            (nid, *edge_filter, edge_limit)).fetchall()
        if edges:
            lines.append('  Edges:')
            for e in edges:
                desc = ' — %s' % e[4] if e[4] else ''
                target_id = e[5][:8] if e[5] else '?'
                dates = e[6][:10] if e[6] else '?'
                if e[7]:
                    dates += ', rev:%s' % e[7][:10]
                lines.append('    → "%s" (id:%s, %s) [%s] %s%s' % (
                    (e[2] or '')[:60], target_id, dates, e[3] or '?', e[0], desc))

        return '\n'.join(lines)
    except Exception:
        return None


def generate_field_summary():
    """Generate a human-readable field summary for the encoding agent prompt."""
    lines = []
    for name, spec in get_writable_fields().items():
        parts = [name]
        parts.append("(%s)" % spec.get("type", "any"))
        if spec.get("required"):
            parts.append("REQUIRED")
        if spec.get("embeds"):
            parts.append("— gets its own embedding for recall matching")
        if spec.get("description"):
            parts.append("— %s" % spec["description"])
        elif spec.get("replace_on_revise"):
            parts.append("— replaced on revise (old content saved to revision_history)")
        lines.append("  ".join(parts))
    lines.append("")
    lines.append("RETURNS: remember() and remember_batch() return related_nodes — "
                 "the top 5 most similar existing nodes with full content. "
                 "Use these to connect() immediately without a separate recall round.")
    return "\n".join(lines)
