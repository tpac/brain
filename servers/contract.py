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
  PROMOTED: additional storage (node_enrichments, node_metadata_kv). Have behavior.
  FREE: anything else. Stored in nodes.metadata JSON blob. Watched for promotion.

To add a new field:
  1. Add it to PROMOTED_FIELDS here
  2. Run tests
  3. That's it. All layers pick it up.
"""


# ── BRAIN_BATCH OP VOCABULARY ──
# The closed set of operation names brain_batch accepts. Single source of
# truth for the three sites that must agree (CLAUDE.md "closed vocabulary"):
#   - dispatch_write._handle_brain_batch  (the if/elif dispatcher + guard)
#   - brain_mcp brain_batch inputSchema   (the JSON-schema enum)
#   - s2 rejection_table                  (detecting invalid-op attempts so a
#                                           dropped op isn't mistaken for a SKIP)
# Adding an op means adding it here and wiring the dispatcher branch — the
# enum and the S2 detector pick it up automatically.

VALID_BATCH_OPS = frozenset({
    "remember", "revise", "connect", "disconnect", "archive", "absorb",
})


# ── STRUCTURAL FIELDS ──
# These are columns on the 'nodes' table.
# Changing these requires schema migration.

STRUCTURAL_FIELDS = {
    "id":         {"store": "nodes", "type": "str", "required": True, "immutable": True},
    "type":       {"store": "nodes", "type": "str", "required": True},
    "title":      {"store": "nodes", "type": "str", "required": True},
    "content":    {"store": "nodes", "type": "str", "replace_on_revise": True, "history": "revision_history in metadata_kv (last 5)"},
    # keywords column dropped in schema v28 — auto-extractor produced
    # near-duplicate noise; FTS5 indexes title+content directly via
    # porter stemming.
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
        "store": "metadata_kv",
        "type": "str",
        "embeds": True,
        "derived_vector": "_situation",
        "description": "When is this knowledge relevant? One sentence. Stored in node_metadata_kv (canonical); a derived _situation embedding row in node_enrichments provides recall scoring. Enrichment text column is deprecated for _situation — kv is the single source of truth.",
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
    # `correction_of` removed 2026-05-17 — corrections are tracked via
    # correction_improvement-aspect edges (corrects, supersedes, reframes,
    # ...), walked by correction_enrich() and rendered by render_corrections().
    # The legacy metadata field's 19 live rows were migrated to `corrects`
    # edges; the SQLite column drop is deferred to a follow-up schema
    # migration. See community node 0769ccec for the full kill plan.
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
    respective stores (node_metadata_kv, node_enrichments)."""
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


# ── get_nodes BATCH-AWARE FORMATTING ──
# Prevents tool_result context explosion when encoders call get_nodes
# on large batches of IDs (the community encoder bug: 30-50 nodes returned
# as raw JSON dumps = 100-200K tokens, blowing past context window).
#
# Callers and their typical batch sizes:
#   Anchor (MCP):           1-5 nodes   → want full detail
#   S1E encoder:            5-15 nodes  → need content + edges for context
#   S2 consolidation:       2-8 nodes   → per-cluster inspection
#   S2 community encoder:   30-50 nodes → coherence check, gist > depth
#
# Strategy: render_rich_node() with scaled config. Small batches stay rich;
# large batches compress content/edges/metadata but keep structure intact.

# Threshold: below this, return raw JSON (no trimming) — preserves Anchor's
# single-node drill-downs and targeted lookups.
GET_NODES_SMALL_MAX = 3

# Threshold: up to this, use balanced config — more room than S2CE but bounded.
GET_NODES_MEDIUM_MAX = 10

# Balanced: for 4-10 node batches (S1E, consolidation, Anchor multi-node)
GET_NODES_BALANCED_FORMAT = {
    'content_limit': 600,       # full enough for encoding decisions
    'edge_limit': 6,            # relations matter — keep top 6
    'metadata_limit': 250,
    'time_format': 'relative',
}

# Compact: for 11+ node batches (S2 community encoder coherence checks)
GET_NODES_COMPACT_FORMAT = {
    'content_limit': 400,       # gist only — enough to judge fit
    'edge_limit': 4,
    'metadata_limit': 200,
    'time_format': 'relative',
}


def _truncate(s: str, limit: int) -> str:
    """Cap `s` at `limit` chars, ending in '…' when truncation occurred.

    Mid-word cuts without ellipsis are confusing — readers don't know if
    the original text ended naturally or got chopped. The single-char
    ellipsis is unambiguous and respects the limit (returned len ≤ limit).
    """
    if not s or limit <= 0 or len(s) <= limit:
        return s
    return s[:max(1, limit - 1)] + '…'


# ── Corrector K/V allowlist for render_corrections heavy mode ──
# Three keys carry meaningful correction context: the corrector's stored
# reasoning, the operator's words, and Anchor's words. Anything else on
# the corrector's metadata is either bookkeeping or downstream-of-surface
# context the heavy render shouldn't replay.
_CORRECTION_HEAVY_KV_KEYS = ('reasoning', 'user_raw_quote', 'anchor_raw_quote')


def render_corrections(corrections, mode='lean',
                       content_limit_balanced=150,
                       content_limit_heavy=400,
                       meta_limit_heavy=300,
                       indent='  '):
    """Render a node's `_corrections` list as formatted lines.

    Single rendering path — both render_rich_node and consumer-specific
    formatters (HealerEncoder._format_batch) call this. Per the data/format
    separation contract (decision id:3c3a3046): one formatter, configs drive
    verbosity.

    Args:
        corrections: list of correction dicts (output of correction_enrich).
            Each carries: id, title, type, direction, relation,
            edge_description, content, reasoning, user_raw_quote.
        mode: 'none' | 'lean' | 'balanced' | 'heavy'.
              none     → no lines emitted
              lean     → header line only (title + id) — legacy default
              balanced → + relation verb + edge_description + content excerpt
              heavy    → + full content + corrector K/V (reasoning,
                         user_raw_quote, anchor_raw_quote) honouring the
                         noise filter.
        content_limit_balanced: char cap for content excerpt in balanced mode
        content_limit_heavy: char cap for content in heavy mode
        meta_limit_heavy: char cap per K/V value in heavy mode
        indent: line prefix for nested values (default '  ')

    Returns list[str] of lines (no leading section header — caller can
    prepend 'CORRECTIONS:' or similar).
    """
    if mode == 'none' or not corrections:
        return []

    lines = []
    sub_indent = indent + '   '
    for corr in corrections:
        direction = corr.get('direction')
        title = (corr.get('title') or '')[:60]
        corr_id = (corr.get('id') or '')[:8]
        verb = corr.get('relation') or direction or 'corrects'
        edge_desc = corr.get('edge_description') or ''

        if direction == 'corrected_by':
            header = '%s⚠ Updated by: "%s" (id:%s)' % (indent, title, corr_id)
        elif direction == 'corrects':
            header = '%s⚠ Corrects: "%s" (id:%s)' % (indent, title, corr_id)
        else:
            header = '%s⚠ Correction: "%s" (id:%s)' % (indent, title, corr_id)

        if mode == 'lean':
            lines.append(header)
            continue

        # balanced/heavy share the relation + edge_desc preamble
        preamble_bits = []
        if verb and verb not in ('corrects', 'corrected_by'):
            preamble_bits.append('relation=%s' % verb)
        if edge_desc:
            preamble_bits.append('why: %s' % _truncate(edge_desc, 200))
        if preamble_bits:
            lines.append('%s — %s' % (header, '  '.join(preamble_bits)))
        else:
            lines.append(header)

        if mode == 'balanced':
            content = (corr.get('content') or '').strip()
            if content:
                lines.append('%s%s' % (sub_indent, _truncate(content, content_limit_balanced)))
            continue

        # heavy
        content = (corr.get('content') or '').strip()
        if content:
            lines.append('%sContent: %s' % (sub_indent, _truncate(content, content_limit_heavy)))
        for kv_key in _CORRECTION_HEAVY_KV_KEYS:
            val = corr.get(kv_key)
            if not val:
                continue
            label = kv_key.replace('_', ' ').title()
            lines.append('%s%s: %s' % (sub_indent, label, _truncate(str(val), meta_limit_heavy)))

    return lines


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

    # Header — individual parts are opt-out via cfg flags (defaults preserve
    # current behavior for callers that don't set them, e.g. Anchor's MCP queries).
    parts = ["id:%s" % nid[:8]]
    # 2026-05-31: confidence display defaults OFF. The field is dormant — set
    # from TYPE_CONFIDENCE at creation, never maintained, read by no ranking
    # path. Showing it to Anchor/Haiku/encoders is false authority (a number
    # nobody updates). Rooted out until it earns a justification; opt back in
    # explicitly via show_confidence=True if a real consumer appears.
    if cfg.get('show_confidence', False):
        conf = node.get('confidence')
        if conf:
            parts.append("conf:%.1f" % conf)
    if node.get('locked'):
        parts.append("locked")
    if cfg.get('show_encoding_source', True):
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

    # Content (None = no truncation, 0 = hide, N = truncate to N chars)
    content_limit = cfg.get('content_limit')
    if content_limit != 0:
        content = node.get('content', '')
        if content_limit and content_limit > 0:
            content = _truncate(content, content_limit)
        if content:
            lines.append('  Content: %s' % content)

    # Situation
    situation = node.get('situation', '')
    if situation:
        lines.append('  Situation: %s' % situation)

    # Metadata KV
    meta = node.get('_metadata', {})
    meta_limit = cfg.get('metadata_limit', 300)
    skip_keys = set((
        'metadata_created_at',
        # situation is rendered at top-level (line above) — skip here to
        # avoid double-display. kv is canonical; promotion to top-level
        # keeps code callers ergonomic.
        'situation',
        # S2 community structural metrics — useful for S2CD/S3, not for Anchor
        'community_internal_edges', 'community_external_edges',
        'community_internal_fraction', 'community_is_corridor',
        'community_centroid', 'community_size', 'community_run_count',
        'community_growth_rate', 'community_edge_signature',
        'community_last_change',
    ))
    # Caller-supplied extra skips (e.g. surface drops 'question')
    skip_keys.update(cfg.get('extra_skip_keys', ()))
    # Voice fields bypass meta_limit — operator and Anchor verbatim
    # quotes are high-signal-per-char and naturally short. Truncating
    # them at 150 chars loses the actual words. Cap defensively at 600.
    _VOICE_KEYS = ('user_raw_quote', 'anchor_raw_quote')
    _VOICE_LIMIT = 600
    if meta_limit > 0:
        for key, val in meta.items():
            if not val or key in skip_keys:
                continue
            # _sys_ prefix = system/infrastructure fields, never shown to LLMs
            if key.startswith('_sys_'):
                continue
            limit = _VOICE_LIMIT if key in _VOICE_KEYS else meta_limit
            lines.append('  %s: %s' % (key.replace('_', ' ').title(), _truncate(str(val), limit)))

    # Keywords column dropped in schema v28 — render block removed.
    # The `show_keywords` config flag is now an inert no-op; callers can
    # be cleaned up incrementally.

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

    # Corrections — unified rendering via render_corrections().
    # Config knob: 'correction_render' ∈ {'none','lean','balanced','heavy'}.
    # Heavy content cap respects metadata_limit so consumers with tight
    # token budgets stay consistent.
    lines.extend(render_corrections(
        node.get('_corrections', []),
        mode=cfg.get('correction_render', 'lean'),
        content_limit_heavy=max(meta_limit, 400),
        meta_limit_heavy=meta_limit))

    # Edges — direction as natural language for contextless LLM understanding.
    # Title gets 100 chars (was 60) — the "why" description is the load-bearing
    # signal, and a 60-char title truncation often dropped the meaningful tail
    # ("Always used together: 'Tom correction: don't mak..." vs full).
    edge_limit = cfg.get('edge_limit', 5)
    connections = node.get('connections', [])[:edge_limit]
    if connections:
        lines.append('  Edges:')
        for e in connections:
            target_id = e.get('id', '?')[:8]
            time_str = _fmt_time(e.get('created_at')) or '?'
            title = e.get('title', '')[:100]
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
