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


# ── BRAIN_BATCH PER-OP CONTRACT ──
# Single source of truth for brain_batch's discriminated op schemas. Three
# sites derive from this dict and cannot drift (CLAUDE.md "closed vocabulary"):
#   - brain_mcp brain_batch inputSchema   (oneOf branch per op — const
#     discriminator + required list + property fragments)
#   - dispatch_write._handle_brain_batch  (per-op required-field pre-check +
#     the if/elif dispatcher + invalid-op guard)
#   - s2 rejection_table                  (detecting invalid-op attempts so a
#                                           dropped op isn't mistaken for a SKIP)
# Adding an op means adding an entry here and wiring the dispatcher branch.
#
# `properties` are the op-specific fields worth signaling at generation time;
# ops accepting open-ended node fields (remember, revise, absorb overrides)
# intentionally leave additionalProperties open. Probe-validated 2026-06-12:
# the oneOf shape took the reason/reasoning incident class from 0/10 to 10/10
# with NO prose support (eval/mcp_variants/probe_v1_oneof_prefix.*); see
# eval/mcp_batch_probe.py. Dict order = probed branch order — keep it.

# Shared schema for revise's patch-mode content field — one source for the
# revise/revise_batch tool schemas AND brain_batch's revise branch; two
# hand-maintained copies of a wire contract drift.
CONTENT_EDITS_SCHEMA = {
    "type": "array",
    "description": (
        "Surgical content patches, applied in order: each item replaces ONE "
        "exact occurrence of `old` with `new` in the stored content. `old` is "
        "copied VERBATIM from the node's current content and must match "
        "exactly once — a missing or ambiguous match fails this op loudly "
        "with guidance. This is how a falsified claim gets fixed without "
        "re-authoring — and risking — everything else the node holds. "
        "Mutually exclusive with `content` (a full rewrite is for "
        "restructures)."),
    "items": {
        "type": "object",
        "required": ["old", "new"],
        "properties": {
            "old": {"type": "string", "description":
                    "Exact, unique substring of the current content"},
            "new": {"type": "string", "description": "Replacement text"},
        },
    },
}

# Shared item schema for connect_to entries — one source for the
# remember/remember_batch schemas AND brain_batch's remember branch
# (BATCH_OP_SPECS below). Carries the {title, relation, why} shape and
# the BAD/GOOD `why` examples; without it brain_batch callers had no
# generation-time signal for entry shape (2026-06-12 review finding #1).
CONNECT_TO_ITEM_SCHEMA = {
    "type": "object",
    "required": ["title"],
    "properties": {
        "title": {
            "type": "string",
            "description": (
                "Target: for an EXISTING node, its 8-char hex id copied from any "
                "visible id: surface (the expected form — a hex-shaped value is "
                "always treated as an id: resolved by unique id prefix, and on a "
                "miss dropped loudly, never matched as a title); for a node "
                "created in this same batch, its exact title (siblings resolve "
                "before catalog matches, any declaration order). NEW wins on "
                "title collision — "
                "to update an existing catalog node use `revise` on its id, not "
                "a duplicate-title remember. Unresolved targets are logged and "
                "skipped, never failing the batch."
            ),
        },
        "relation": {
            "type": "string",
            "description": (
                "Edge relation, open text — e.g. refines, grounds, corrects, "
                "depends_on, supersedes, triggers, implements, anchored_to, "
                "during. Invent a specific verb when none fits better. NEVER "
                "`related`/`related_to`/empty — generic relations pollute the "
                "activation kernel and match no query."
            ),
        },
        "why": {
            "type": "string",
            "description": (
                "What the edge MEANS — the insight living between the two nodes, "
                "not a summary of either. Embedded for query matching; >=30 "
                "chars, or drop the edge.\n"
                "BAD: \"example of the principle\" — generic gloss, no insight "
                "about which example or why.\n"
                "GOOD: \"the assumption treated concurrent access as a "
                "thread-safety question; the correction reframes it as wal-index "
                "contention — different failure mode, different fix\"."
            ),
        },
        "relations": {
            "type": "array",
            "description": (
                "Alternative to relation+why when the same pair carries multiple "
                "distinct relationships. Each item is {relation, why}."
            ),
            "items": {
                "type": "object",
                "required": ["relation", "why"],
                "properties": {
                    "relation": {"type": "string"},
                    "why": {"type": "string"},
                },
            },
        },
    },
}


BATCH_OP_SPECS = {
    "remember": {
        "required": ["type", "title", "content"],
        # creates_node: this op mints a node, so provenance stamping
        # (stamp_scope_provenance) FORCE-stamps the session's scope fields
        # (project, counterpart) onto its payload; ops without the flag get
        # agent-supplied values STRIPPED. Derived, not enumerated — a future
        # node-creating op added here inherits the stamp automatically.
        "creates_node": True,
        "description": ("Create a node. Accepts all remember() fields "
                        "(situation, reasoning, quotes, ...)."),
        "properties": {
            "type": {"type": "string", "description": "Node type"},
            "title": {"type": "string", "description": "Specific, scannable title"},
            "content": {"type": "string", "description": "Rich content"},
            "connect_to": {"type": "array", "description":
                           "Typed edges to siblings/catalog — see tool description",
                           "items": CONNECT_TO_ITEM_SCHEMA},
        },
    },
    "revise": {
        "required": ["node_id", "reason"],
        "description": ("Update node fields. Any other key is a field update "
                        "(content, situation, reasoning, ...) — specified "
                        "fields are REPLACED. For content, prefer "
                        "`content_edits` when fixing specific claims: "
                        "surgical patches that leave the rest of the content "
                        "untouched. A full `content` rewrite is for "
                        "restructures; the two are mutually exclusive."),
        "properties": {
            "node_id": {"type": "string", "description": "Node to revise"},
            "reason": {"type": "string", "description":
                       "Audit note for this revision — recorded in trace "
                       "events, NOT stored on the node. Distinct from the "
                       "node FIELD `reasoning`, which a revise op updates "
                       "like any other field."},
            "content_edits": CONTENT_EDITS_SCHEMA,
        },
    },
    "connect": {
        "required": ["source_id", "target_id"],
        "description": "Create/update an edge between two EXISTING catalog nodes.",
        "properties": {
            "source_id": {"type": "string", "description":
                          "Actor node id (must already exist)"},
            "target_id": {"type": "string", "description":
                          "Acted-upon node id (must already exist)"},
            "relation": {"type": "string", "description": "Open-text verb"},
            "description": {"type": "string", "description":
                            "What the edge MEANS (>=30 chars)"},
            "weight": {"type": "number"},
        },
    },
    "disconnect": {
        "required": ["source_id", "target_id", "relation"],
        "description": ("Soft-archive one relation on an edge; other "
                        "relations on the same edge survive."),
        "properties": {
            "source_id": {"type": "string", "description": "Edge source id"},
            "target_id": {"type": "string", "description": "Edge target id"},
            "relation": {"type": "string", "description": "Relation to archive"},
        },
    },
    "archive": {
        "required": ["node_id"],
        "description": "Soft-archive a node.",
        "properties": {
            "node_id": {"type": "string", "description": "Node to soft-archive"},
            "reason": {"type": "string", "description": "Why (audit note)"},
            "survivor_id": {"type": "string", "description":
                            "ONLY when a live node REPLACES this one "
                            "(supersession): its id. Records the redirect "
                            "lineage (absorbed_into edge) recall walks to "
                            "the successor. Omit for plain retirement; for "
                            "merging content use absorb instead."},
        },
    },
    "absorb": {
        "required": ["survivor_id", "absorbed_id"],
        "description": ("Lossless merge: fold absorbed INTO survivor. "
                        "Accepts revise-shape field overrides (content, "
                        "title, confidence, situation)."),
        "properties": {
            "survivor_id": {"type": "string", "description":
                            "Node that remains (may be locked)"},
            "absorbed_id": {"type": "string", "description":
                            "Node folded in + archived (must be archivable)"},
            "content": {"type": "string", "description":
                        "Merged content override — REQUIRED for losslessness "
                        "unless survivor already states the absorbed claim"},
        },
    },
}

# Derived — kept as the cheap membership check used across dispatch + S2.
VALID_BATCH_OPS = frozenset(BATCH_OP_SPECS)


# ── STRUCTURAL FIELDS ──
# These are columns on the 'nodes' table.
# Changing these requires schema migration.

STRUCTURAL_FIELDS = {
    "id":         {"store": "nodes", "type": "str", "required": True, "immutable": True},
    "type":       {"store": "nodes", "type": "str", "required": True},
    "title":      {"store": "nodes", "type": "str", "required": True},
    "content":    {"store": "nodes", "type": "str", "replace_on_revise": True, "history": "trace events (node_revised deltas); legacy _sys_revision_history blobs dropped by migration"},
    "confidence": {"store": "nodes", "type": "float", "range": (0.0, 1.0), "default": 1.0},
    "locked":     {"store": "nodes", "type": "bool", "default": False},
    "archived":   {"store": "nodes", "type": "bool", "default": False},
    "critical":   {"store": "nodes", "type": "bool", "default": False},
    "emotion":    {"store": "nodes", "type": "float"},
    "emotion_label": {"store": "nodes", "type": "str", "default": "neutral"},
    # `project` lives in PROMOTED_FIELDS (metadata_kv) — the nodes.project
    # column was dropped in schema v30. Provenance is system-stamped at the
    # write boundary, never agent-authored.
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
    "project": {
        "store": "metadata_kv",
        "type": "str",
        # system_stamped: excluded from the agent-facing MCP schemas
        # (get_writable_fields) — the write boundary stamps it from
        # SessionContext.project and overrides/drops agent-supplied values,
        # so advertising it as an input would only train drift.
        "system_stamped": True,
        "description": (
            "Repo provenance — WHERE this was learned (the session's main-repo "
            "directory name, derived from cwd). System-stamped at the write "
            "boundary from SessionContext.project; agent-supplied values are "
            "overridden or dropped, and a revise never moves it (only "
            "migration does). Read by the LAF proj lane and dict filters."),
    },
    "counterpart": {
        "store": "metadata_kv",
        "type": "str",
        # system_stamped: same contract as project — scope provenance,
        # stamped at the write boundary (stamp_scope_provenance), never
        # agent-authored. Today the value is the install-default operator
        # (constant); it becomes per-session when the speaker arc's F4
        # (counterpart on SessionContext) lands.
        "system_stamped": True,
        "description": (
            "Counterpart provenance — WHO the session was with when this was "
            "learned. System-stamped at the write boundary; agent-supplied "
            "values are overridden or dropped. Read by scope_marks for "
            "differential exposure; a future speaker lane reads it the way "
            "the proj lane reads project."),
    },
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
        "description": ("Why this was encoded — decision rationale, stored "
                        "on the node. NOT revise()'s `reason` param — that is "
                        "the audit note for a revision, recorded in trace "
                        "events and never stored on the node."),
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
    """Fields an AGENT can set via remember() or revise() — feeds the MCP
    schemas. Excludes immutable fields and system_stamped ones (project:
    the write boundary derives it from the session; advertising it as an
    input would only train drift)."""
    return {k: v for k, v in ALL_FIELDS.items()
            if not v.get("immutable") and not v.get("system_stamped")
            and k != "archived"}


def get_remember_fields():
    """Fields that brain.remember() accepts — ALL writable fields.
    Structural fields go to nodes table, promoted fields go to their
    respective stores (node_metadata_kv, node_enrichments)."""
    return get_writable_fields()


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

# Small-batch default (<=3 nodes, rich=false): the de-stuffed drill view.
# Content is the signal you fetched for, so it stays full; the edge tail
# (40-76% of a well-connected node's payload, weight-sorted) and the heavy
# correction K/V are what get bounded. This replaces the old <=3 raw-JSON
# escape hatch — small pulls stay readable without the firehose.
GET_NODES_SMALL_FORMAT = {
    'content_limit': None,      # full content — content is signal, not stuffing
    'edge_limit': 8,            # top-8 by weight: the meaningful constellation
    'metadata_limit': 300,
    'correction_render': 'balanced',
    'time_format': 'relative',
}

# Full: the rich=true opt-in for get_node/get_nodes — the deliberate
# "give me everything" drill. NOT the old raw dict dump: still curated
# through render_rich_node (drops _sys_ fields and raw relation sub-dicts),
# but uncapped on the dimensions that carry meaning — full content, all edges
# (weight-sorted), and heavy correction K/V (reasoning + raw quotes).
GET_NODES_FULL_FORMAT = {
    'content_limit': None,      # full content
    'edge_limit': None,         # all edges (None → no slice; see render_rich_node)
    'metadata_limit': 400,
    'correction_render': 'heavy',
    'time_format': 'relative',
}

# The skinny node shape returned by NodeDAL.filter_nodes (dal.py:2038-2040)
# when rich=False — id/title/type/confidence/created_at, plus the filtered
# column. render_skinny_node uses this to tell the standard columns from the
# filtered field it should surface. Keep in sync with the DAL's SELECT.
SKINNY_NODE_FIELDS = ('id', 'title', 'type', 'confidence', 'created_at')

# connect_to catalog-title matching (write path) — deterministic token
# matching, NO vectors (decision 2026-07-30: a wrong edge outlives a missing
# one; cosine near-misses put a wrong node 0.003 below a right one). A title
# resolves when its normalized token sequence is within MAX_OPS Levenshtein
# token-edits of the query — distance 0 is the exact match (any length);
# distance 1..MAX_OPS additionally requires the query to carry at least
# MIN_TOKENS distinct tokens (short strings false-positive on containment)
# and the runner-up candidate to sit at least MARGIN ops further out
# (a photo-finish is a reason to refuse, not a tiebreak to win).
NEAR_TITLE_MAX_OPS = 2
NEAR_TITLE_MIN_TOKENS = 5
NEAR_TITLE_MARGIN = 2

# Candidate-pool ceiling for the FTS5 title probe (_title_candidate_rows).
# The pigeonhole recall guarantee only holds while the pool fits — at the
# limit the write path REFUSES (never guesses), so this bound is what
# decides how often a legitimate connect_to is dropped as "saturated".
# It must clear the OR of MAX_OPS+MARGIN probe tokens on the real corpus:
# at 8.4k live nodes a single common probe (`encoder*` 851, `recall*` 649)
# already blew the old 500, so every edge whose longest tokens included a
# hot word was silently refused. Measured worst real 4-probe pool ≈ 950.
# Cost is one indexed IN-hydrate plus a length-rejected Levenshtein per row,
# and it stays far under SQLite's 32766 bound variables.
TITLE_CANDIDATE_POOL_LIMIT = 4000

# Survivor-pointer walk budget for NodeDAL.resolve_live — how many
# archived→survivor redirects an id may take before it's declared orphaned.
# Supersession chains (session-opener handoffs) grow one hop per generation;
# the 2026-07-30 backfill compressed the existing chains to depth 1, so this
# is headroom for organic growth, not a working depth. Cost of a higher cap
# is per-LEVEL queries (trivial); the real ceiling guard is cycle detection,
# not this number. If real chains approach it, add an S2 pointer-compression
# pass rather than raising it again.
RESOLVE_LIVE_MAX_HOPS = 12


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


# ── Bounded-read truncation contract ──
#
# A windowed read (hours / older_than / gt / lt bounds) that hits its row
# limit covers only the most recent slice of the requested window while
# LOOKING complete — a 168h aggregate that silently spanned 2 days
# (2026-08-06 cost tally). Every windowed read door attaches this payload
# when saturated; ranked top-k doors (recall, semantic recall_episodes,
# relevance-mode filter_nodes) are exempt — there, truncation IS the
# contract. The MCP render layer prepends a ⚠ banner for any result dict
# carrying 'truncated' (single chokepoint in brain_mcp._format_result's
# caller). tests/test_truncation_contract.py pins flag-or-exempt for every
# read tool.

def truncation_payload(limit, rows, reason=''):
    """Build the standard 'truncated' payload for a saturated bounded read.

    rows: the rows actually RETURNED (post-trim). The payload reports the
    COVERED [coverage_start .. coverage_end] created_at range (dug out of
    nested 'events' lists for grouped chains) and never claims which SIDE
    was dropped — a DESC read drops older rows, an ASC read drops newer
    ones, and a directional claim was factually inverted for one of them
    (2026-08-07 review, finding 6). reason overrides the default cause.
    """
    times = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get('created_at'):
            times.append(row['created_at'])
        for ev in (row.get('events') or []):
            if isinstance(ev, dict) and ev.get('created_at'):
                times.append(ev['created_at'])
    coverage_start = min(times) if times else ''
    coverage_end = max(times) if times else ''
    cause = reason or ('hit limit=%d before exhausting the requested window'
                       % limit)
    covered = (('%s .. %s' % (coverage_start, coverage_end))
               if times else 'an unknown slice')
    return {
        'limit': limit,
        'coverage_start': coverage_start,
        'coverage_end': coverage_end,
        'note': ('%s — the result covers only %s; matching rows outside '
                 'that slice were dropped. Raise limit or narrow filters '
                 'before trusting any aggregate over this result.'
                 % (cause, covered)),
    }


def flag_truncation(result, fetched, limit, key):
    """+1-probe convenience: `fetched` was pulled with limit+1; an extra row
    is PROOF the window holds more than the (pre-trimmed) result carries.
    `key` names the rows entry in `result` — explicit, never inferred from
    dict order. Attaches 'truncated' and returns `result`."""
    if len(fetched) > limit:
        result['truncated'] = truncation_payload(limit, result[key])
    return result


def truncation_banner(payload):
    """The one render form for a truncation payload — every surface that
    shows a saturated result to a reasoning consumer prepends this."""
    return '⚠ TRUNCATED — %s' % (
        (payload or {}).get('note') or 'result hit the row limit')


# ── Scope dimensions — differential exposure ──
# The dimension SET has one source: PROMOTED_FIELDS entries flagged
# `system_stamped` (the registry the MCP schema exclusion already consumes).
# The write boundary (scales/dispatch.stamp_scope_provenance) and the render
# marks below both derive from it — adding a dimension is the PROMOTED_FIELDS
# entry plus its label here; the module-load assert refuses a half-landed one.
SCOPE_PROVENANCE_FIELDS = tuple(
    k for k, v in PROMOTED_FIELDS.items() if v.get('system_stamped'))

# The session's declared side travels as ONE `scope` dict ({dimension:
# current value}, built by brain.session_scope) so adding a dimension never
# re-threads render signatures. Only truthy-declared dimensions participate
# (an unscoped session applies no pressure — unknown is neutral, matching
# the scope lane semantics).
SCOPE_MARK_LABELS = {
    'project': 'From another project',
    'counterpart': 'Learned with another counterpart',
}
assert set(SCOPE_MARK_LABELS) == set(SCOPE_PROVENANCE_FIELDS), (
    'scope dimension registries diverged: PROMOTED_FIELDS system_stamped=%r '
    'vs SCOPE_MARK_LABELS=%r — a dimension stamped but never marked (or '
    'marked but never stamped) fails silently everywhere else, so refuse to '
    'import' % (SCOPE_PROVENANCE_FIELDS, tuple(SCOPE_MARK_LABELS)))


def scope_marks(node, scope, meta=None):
    """Mismatch marks for every declared scope dimension — the single place
    differential-exposure logic lives. A node value that is absent stays
    unmarked (never punish missing provenance); a match renders nothing
    (same-value lines are noise); only foreign values mark. Mark, don't
    hide — ranking pressure is the scope lane's job, not the render's.

    Node value resolution: promoted top-level → caller-supplied meta →
    '_metadata' (the canonical get_node attachment; callers that build meta
    from other keys, e.g. background mode's metadata_kv, still resolve
    non-promoted dimensions). Comparison is case-insensitive: the producers
    aren't case-coordinated (a marker file may say 'Brain' where git says
    'brain') and a case slip must not mark the whole corpus foreign."""
    meta = meta if meta is not None else {}
    node_meta = node.get('_metadata') or {}
    lines = []
    for dim, label in SCOPE_MARK_LABELS.items():
        current = (scope or {}).get(dim)
        if not current:
            continue
        value = node.get(dim) or meta.get(dim) or node_meta.get(dim) or ''
        if value and value.strip().lower() != current.strip().lower():
            lines.append('  ⚠ %s: %s' % (label, value))
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
            # time_now: the as-of instant (replay-safe callers pass conversation
            # time); time_fine: sub-day '3h ago' steps (the encoder catalog).
            return _relative_time(ts, now=cfg.get('time_now'),
                                  fine=cfg.get('time_fine', False))
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
    # Ownership mark: caller-supplied set of node ids this session WROTE
    # (encoder catalog — created/revised by a prior run or Anchor mid-session;
    # reads deliberately don't qualify). Renders in the header so recency and
    # ownership sit together.
    if nid in (cfg.get('this_session_ids') or ()):
        parts.append("this session")

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

    # Differential scope exposure: when the caller declares its session
    # scope (cfg['scope']), each scope dimension renders ONLY on mismatch —
    # a same-value line on a mostly-uniform corpus is noise consumers learn
    # to skip; a line that only appears when foreign is signal (mark, don't
    # hide — ranking pressure is the scope lane's job). Callers that don't
    # declare keep the legacy generic KV render below.
    scope = cfg.get('scope') or None
    if scope:
        lines.extend(scope_marks(node, scope, meta=meta))
    skip_keys = set((
        'metadata_created_at',
        # Nothing writes `keywords` any more, but ~648 nodes still carry the
        # KV row, and the generic loop below renders every stored key. Without
        # this skip they would each show a stray "Keywords: ..." line. Remove
        # it only together with a purge of those rows, never before.
        'keywords',
        # situation is rendered at top-level (line above) — skip here to
        # avoid double-display. kv is canonical; promotion to top-level
        # keeps code callers ergonomic.
        'situation',
        # counterpart is differential-only: today its value is the install
        # default — identical on every node — so a generic 'Counterpart: X'
        # line is one row of pure noise per node on every undeclared render
        # (S2 encoder prompts, MCP get_node, 25-candidate menus). The
        # scope_marks path above renders it on mismatch; nothing else should.
        # (project stays generically visible for undeclared callers — its
        # values genuinely vary, so the line carries information.)
        'counterpart',
        # S2 community structural metrics — useful for S2CD/S3, not for Anchor
        'community_internal_edges', 'community_external_edges',
        'community_internal_fraction', 'community_is_corridor',
        'community_centroid', 'community_size', 'community_run_count',
        'community_growth_rate', 'community_edge_signature',
        'community_last_change',
    ))
    # Caller-supplied extra skips (e.g. surface drops 'question')
    skip_keys.update(cfg.get('extra_skip_keys', ()))
    # Differential mode owns the scope-dimension renders (mismatch marks
    # above) — suppress the generic KV lines for declared dimensions.
    if scope:
        skip_keys.update(k for k in SCOPE_MARK_LABELS if scope.get(k))
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
    all_conns = node.get('connections', [])
    connections = all_conns[:edge_limit]
    # Edge-total indicator (opt-in via show_edge_total — encoder catalog): when
    # the limit truncates, say so — '(5 of 23)' tells the reader how connected
    # the node really is (the DAL pull is uncapped and noise-excluded, so the
    # total is honest). Default off: legacy renders keep the bare header.
    edges_header = '  Edges:'
    if cfg.get('show_edge_total') and len(all_conns) > len(connections):
        edges_header = '  Edges (%d of %d):' % (len(connections), len(all_conns))
    if cfg.get('show_edge_total') and all_conns and not connections:
        # edge_limit=0 with edges present: say they exist. Rendering nothing
        # made a well-connected node read as isolated — the reader cannot tell
        # "no edges" from "edges not shown here", and only one of those is a
        # reason to go look.
        lines.append('  Edges (%d, not shown — get_nodes for them):'
                     % len(all_conns))
    if connections and cfg.get('edge_style') == 'oneline':
        # Selection-grade edge render: direction + relation + target title only.
        # No description, no id, no timestamps — those are injection payload
        # (the full style below). One line per edge, top relation only.
        lines.append(edges_header)
        for e in connections:
            title = e.get('title', '')[:80]
            rels = e.get('relations') or []
            rel = (rels[0].get('relation') if rels else e.get('relation', '')) or 'related'
            if e.get('direction') == 'incoming':
                lines.append('    "%s" %s this' % (title, rel))
            else:
                lines.append('    this %s "%s"' % (rel, title))
    elif connections:
        lines.append(edges_header)
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


def render_skinny_node(node, extra_value_limit=120):
    """One-line render of a skinny node (id/title/type — no content, edges, or
    corrections). Used for filter_nodes discovery scans.

    Surfaces any NON-standard field (the filtered column, e.g. encoding_source)
    so the scan shows the value being filtered on — bounded by
    `extra_value_limit` so a long-valued filter field (content, reasoning,
    situation) can't turn a 50-row scan into a firehose.
    """
    extra = ' '.join(
        '%s=%s' % (k, _truncate(str(v), extra_value_limit))
        for k, v in node.items()
        if k not in SKINNY_NODE_FIELDS and v is not None)
    return '[%s] "%s" (id:%s)%s' % (
        node.get('type', '?'), node.get('title', '?'),
        (node.get('id') or '')[:8], ('  ' + extra) if extra else '')


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
            parts.append("— replaced on revise; revision history lives in trace events")
        lines.append("  ".join(parts))
    lines.append("")
    lines.append("RETURNS: remember() and remember_batch() return related_nodes — "
                 "the top 5 most similar existing nodes with full content. "
                 "Use these to connect() immediately without a separate recall round. "
                 "related_nodes is NOT the outcome of connect_to — when you pass "
                 "connect_to, the response carries a separate connect_to_result "
                 "{created:[...], failed:[{title, reason}]} reporting which edges "
                 "formed and why any didn't.")
    return "\n".join(lines)
