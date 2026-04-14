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
    "created_at":   {"store": "nodes", "type": "str", "immutable": True, "description": "ISO 8601 timestamp, auto-set on insert. Filterable with lt/gt for date-range queries."},
    "updated_at":   {"store": "nodes", "type": "str", "immutable": True, "description": "ISO 8601 timestamp, auto-updated on revise. Filterable with lt/gt for date-range queries."},
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
    'time_format': 'date',      # 'date' = YYYY-MM-DD, 'relative' = "2d ago"
}


def render_rich_node(node, config=None):
    """Render a get_rich_node() dict as a formatted string.

    This is the single formatter. Different configs produce different views:
    - Encoder: full content, all metadata, 5 edges
    - Surface: truncated content, key metadata, 3 edges
    - Boot: same as surface (for now)
    """
    cfg = {**NODE_FORMAT_DEFAULTS, **(config or {})}
    nid = node.get('id', '?')
    use_relative = cfg.get('time_format') == 'relative'

    def _fmt_time(ts):
        if not ts:
            return None
        if use_relative:
            from servers.pipeline_contract import _relative_time
            return _relative_time(ts)
        return str(ts)[:10]

    # Header
    parts = ["id:%s" % nid[:8]]
    conf = node.get('confidence')
    if conf:
        parts.append("conf:%.1f" % conf)
    if node.get('locked'):
        parts.append("locked")
    if node.get('encoding_source'):
        parts.append("src:%s" % node['encoding_source'])
    created_rel = _fmt_time(node.get('created_at'))
    revised_rel = _fmt_time(node.get('revised_at'))
    if revised_rel and created_rel and revised_rel != created_rel:
        parts.append("created %s, revised %s" % (created_rel, revised_rel))
    elif created_rel:
        parts.append(created_rel)

    lines = ['[%s] "%s" (%s)' % (
        node.get('type', '?'), node.get('title', '?'), ", ".join(parts))]

    # Content
    content = node.get('content', '')
    if cfg.get('content_limit'):
        content = content[:cfg['content_limit']]
    if content:
        lines.append('  Content: %s' % content)

    # Situation
    situation = node.get('situation', '')
    if situation:
        lines.append('  Situation: %s' % situation)

    # Metadata KV
    meta = node.get('_metadata', {})
    meta_limit = cfg.get('metadata_limit', 300)
    skip_keys = (
        'metadata_created_at', 'revision_history',
        # S2 community structural metrics — useful for S2CD/S3, not for Anchor
        'community_internal_edges', 'community_external_edges',
        'community_internal_fraction', 'community_is_corridor',
        'community_centroid', 'community_size', 'community_run_count',
        'community_growth_rate', 'community_edge_signature',
        'community_last_change',
    )
    for key, val in meta.items():
        if not val or key in skip_keys:
            continue
        if key == 'correction_of':
            # Corrections are in _corrections with full context
            continue
        lines.append('  %s: %s' % (key.replace('_', ' ').title(), str(val)[:meta_limit]))

    # Keywords
    if node.get('keywords'):
        lines.append('  Keywords: %s' % node['keywords'])

    # Personal context
    if node.get('personal') and node.get('personal_context'):
        lines.append('  Context: %s (%s)' % (node['personal'], node['personal_context']))

    # Dates (already in header when using relative time)
    if not use_relative:
        created = str(node.get('created_at', ''))[:10]
        revised = str(node.get('revised_at', '') or '')[:10]
        if created:
            date_str = 'Created: %s' % created
            if revised:
                date_str += ' | Revised: %s' % revised
            lines.append('  %s' % date_str)

    # Corrections
    for corr in node.get('_corrections', []):
        if corr.get('direction') == 'corrected_by':
            lines.append('  ⚠ Updated by: "%s" (id:%s)' % (corr.get('title', '')[:60], corr.get('id', '')[:8]))
        elif corr.get('direction') == 'corrects':
            lines.append('  ⚠ Corrects: "%s" (id:%s)' % (corr.get('title', '')[:60], corr.get('id', '')[:8]))

    # Edges — direction as natural language for contextless LLM understanding
    edge_limit = cfg.get('edge_limit', 5)
    connections = node.get('connections', [])[:edge_limit]
    if connections:
        lines.append('  Edges:')
        for e in connections:
            target_id = e.get('id', '?')[:8]
            time_str = _fmt_time(e.get('created_at')) or '?'
            title = e.get('title', '')[:60]
            ntype = e.get('type', '?')
            incoming = e.get('direction') == 'incoming'

            relations = e.get('relations', [])
            if relations and len(relations) > 1:
                rel_strs = []
                for r in relations:
                    rel = r.get('relation', '')
                    desc = ' — %s' % r['description'] if r.get('description') else ''
                    if incoming:
                        rel_strs.append('"%s" %s this%s' % (title, rel, desc))
                    else:
                        rel_strs.append('this %s "%s"%s' % (rel, title, desc))
                lines.append('    [%s id:%s %s] %s' % (
                    ntype, target_id, time_str, ' | '.join(rel_strs)))
            else:
                rel = e.get('relation', '')
                desc = ' — %s' % e.get('description', '') if e.get('description') else ''
                if incoming:
                    lines.append('    [%s id:%s %s] "%s" %s this%s' % (
                        ntype, target_id, time_str, title, rel, desc))
                else:
                    lines.append('    [%s id:%s %s] this %s "%s"%s' % (
                        ntype, target_id, time_str, rel, title, desc))

    return '\n'.join(lines)


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
