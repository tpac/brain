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
    "content":    {"store": "nodes", "type": "str", "append_on_revise": True},
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
        "store": "node_metadata",
        "column": "reasoning",
        "type": "str",
        "description": "Why this was encoded — decision rationale.",
    },
    "user_raw_quote": {
        "store": "node_metadata",
        "column": "user_raw_quote",
        "type": "str",
        "description": "Operator's exact words.",
    },
    "correction_of": {
        "store": "node_metadata",
        "column": "correction_of",
        "type": "str",
        "description": "Node ID this corrects.",
    },
    "correction_pattern": {
        "store": "node_metadata",
        "column": "correction_pattern",
        "type": "str",
        "description": "Behavioral pattern behind the correction.",
    },
    "source_context": {
        "store": "node_metadata",
        "column": "source_context",
        "type": "str",
        "description": "Session/context when this was encoded.",
    },
}


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
        elif spec.get("append_on_revise"):
            parts.append("— appended on revise, not replaced")
        lines.append("  ".join(parts))
    lines.append("")
    lines.append("RETURNS: remember() and remember_batch() return related_nodes — "
                 "the top 5 most similar existing nodes with full content. "
                 "Use these to connect() immediately without a separate recall round.")
    return "\n".join(lines)
