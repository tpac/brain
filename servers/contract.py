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

import json
import re

from servers.loud_truncation import cap_text_at_boundary


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

# The revise rule (docs/REVISE-SHAPE-SPEC.md §1): on revise a field takes
# either its NEW VALUE — the whole field replaced — or a SWAP `{old, new}`
# (a list of swaps for several spots) that changes only what is stale. One
# source for the revise/revise_batch tool schemas AND brain_batch's revise
# branch; two hand-maintained copies of a wire contract drift.
REVISE_RULE = (
    "On revise a field takes its NEW VALUE (the whole field replaced) or a "
    "swap `{old, new}` — a list of swaps for several spots — that changes "
    "only what is stale; `old` is copied VERBATIM from the node as shown and "
    "must occur exactly once, or the op fails loudly with the count and "
    "nothing is written. Fields not named are untouched. Edges ride as "
    "`connect_to` exactly as on remember: on revise an entry changes the "
    "edge this node already has to that `target` (its `why`, or its "
    "`relation` — value or swap), or creates it if there is none.")

# The swap item. No prose of its own: REVISE_RULE states the rule once per
# op description, and this object is inlined at every swappable field, so a
# sentence here would ride the tool blob many times over.
SWAP_SCHEMA = {
    "type": "object",
    "required": ["old", "new"],
    "properties": {
        "old": {"type": "string", "description":
                "Exact, unique span of the current value"},
        "new": {"type": "string", "description": "Replacement text"},
    },
}

# The reference forms. A tool schema states each shared shape ONCE under its
# root `$defs` and points at it from every field that takes it — forty inline
# copies of the swap object cost ~8K chars per encoder round for information
# the agent has after the first, and the generation-shape probe scores the
# reference form identical to inline. attach_defs() (below, after the shapes
# it registers) adds to a tool the definitions it actually references.
REF_SWAP = {"$ref": "#/$defs/swap"}
REF_SWAP_LIST = {"type": "array", "items": REF_SWAP}
REF_CONNECT_TO_ITEM = {"$ref": "#/$defs/connect_to_item"}
REF_REVISE_CONNECT_TO_ITEM = {"$ref": "#/$defs/revise_connect_to_item"}


def is_swap(value):
    """`{old, new}` — one in-place swap (REVISE_RULE)."""
    return isinstance(value, dict) and set(value) == {'old', 'new'}


def is_swap_list(value):
    """A non-empty list of swaps."""
    return (isinstance(value, list) and bool(value)
            and all(is_swap(v) for v in value))


def validate_swaps(value, field):
    """Shape check for a swap or swap list on `field` — string `old`
    (non-empty) and `new`, `old != new`. Returns (ok, error)."""
    swaps = value if isinstance(value, list) else [value]
    for i, e in enumerate(swaps):
        if (not is_swap(e) or not isinstance(e['old'], str) or not e['old']
                or not isinstance(e['new'], str)):
            return False, ("%s swap[%d] must be {old: <non-empty string>, "
                           "new: <string>}" % (field, i))
        if e['old'] == e['new']:
            return False, ("%s swap[%d]: old and new are identical — a no-op "
                           "swap is a mistake, not a patch" % (field, i))
    return True, None


def apply_swaps(stored, value, field):
    """Apply a swap or swap list to `field`'s stored value, in order, each
    `old` matching exactly once against the running result (REVISE_RULE).
    Zero or ambiguous matches fail loudly with the count — a swap that lands
    in the wrong place silently corrupts a surface recall reads. Errors are
    written to teach the calling agent the fix. Returns (new_value, None) or
    (None, error)."""
    ok, err = validate_swaps(value, field)
    if not ok:
        return None, err
    swaps = value if isinstance(value, list) else [value]
    current = stored
    for i, e in enumerate(swaps):
        old, new = e['old'], e['new']
        n = current.count(old)
        if n == 0:
            return None, ("%s swap[%d]: `old` not found in the node's current "
                          "%s (%d chars). `old` must be copied VERBATIM from "
                          "the value as shown — if your view of this node was "
                          "truncated or aged, expand it with get_nodes first, "
                          "or send the field's full new value instead."
                          % (field, i, field, len(current)))
        if n > 1:
            return None, ("%s swap[%d]: `old` matches %d places — extend it "
                          "with surrounding context until it is unique."
                          % (field, i, n))
        current = current.replace(old, new, 1)
    return current, None


def swappable(prop):
    """A text field's revise-time schema: `string | swap | swap[]`, keeping
    the field's own description. Applied to every get_swap_fields() spec
    wherever a revise surface is generated. The swap is a `$ref` — the tool
    carrying the field gets the definition from attach_defs()."""
    out = {"anyOf": [{"type": "string"}, REF_SWAP, REF_SWAP_LIST]}
    if prop.get("description"):
        out["description"] = prop["description"]
    return out


# `content_edits` is the ALIAS of `content: [swaps]` — the pre-rule name for
# the swap list, content-only. Accepted and normalized at the write boundary;
# retire (drop this schema + the alias handling, add the name to
# tests/test_retired_fields.py) once the encoder's op dumps show zero uses
# across a full A/B round.
CONTENT_EDITS_SCHEMA = {
    **REF_SWAP_LIST,
    "description": ("Deprecated alias of `content: [{old, new}, ...]`; "
                    "passing both is an error."),
}
# Deprecated revise-field aliases, alias → the field it stands for. The
# vocabulary guardrail (tests/test_teaching_vocabulary_sync.py) exempts these
# keys from the taught set — an alias is advertised and described as
# deprecated, and taught nowhere on purpose; brain.revise carries the
# write-side normalization of each (content_edits → content swaps). Retiring
# one = drop its schema + its normalization + its entry here, and add the name
# to tests/test_retired_fields.RETIRED_NODE_FIELDS.
REVISE_FIELD_ALIASES = {'content_edits': 'content'}

# Shared item schema for connect_to entries — one source for the
# remember/remember_batch/revise schemas AND brain_batch's remember + revise
# branches (BATCH_OP_SPECS below). Carries the {target, relation, why} shape
# and the BAD/GOOD `why` examples; without it brain_batch callers had no
# generation-time signal for entry shape (2026-06-12 review finding #1).
# `target` is the key; `title` is its deprecated alias, accepted for one
# window (the field is an id in the usual case — a name that says "title"
# lies about its content, and on revise would be actively misleading).
CONNECT_TO_ITEM_SCHEMA = {
    "type": "object",
    "anyOf": [{"required": ["target"]}, {"required": ["title"]}],
    "properties": {
        "target": {
            "type": "string",
            "description": (
                "The other node: for an EXISTING node, its exact 8-char hex id "
                "copied verbatim from any visible id: surface (a hex-shaped "
                "value is always treated as an id — never matched as a title "
                "— and on a miss dropped loudly). On remember only, a node "
                "created in this same batch may be named by its exact title "
                "(siblings resolve before catalog matches, any declaration "
                "order; NEW wins on title collision — to update an existing "
                "catalog node use `revise` on its id, not a duplicate-title "
                "remember). On revise the target must be an id. Unresolved "
                "targets are logged and skipped, never failing the batch."
            ),
        },
        "title": {
            "type": "string",
            "description": "Deprecated alias of `target`.",
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

def connect_to_target(entry):
    """The target a connect_to entry names — `target`, or its deprecated alias
    `title`. The one place the alias is known; every reader of an entry's
    target goes through here, so dropping the alias is a one-line change."""
    if not isinstance(entry, dict):
        return entry
    return entry.get('target') or entry.get('title', '')


def connect_to_why(entry):
    """The why a connect_to entry (or one of its `relations` items) carries —
    `why`, or its alias `description` (the name the standalone `connect` op
    and the edge table use). The one place the alias is known; the write
    path and every scorer read it through here."""
    if not isinstance(entry, dict):
        return None
    return entry.get('why', entry.get('description'))


def _revise_connect_to_item_schema():
    """connect_to on REVISE, derived from the remember item schema: same
    keys, the same relation vocabulary and `why` exemplars, plus what revise
    adds — `relation` and `why` take a swap (rename the relation in place /
    patch the description) because the entry addresses an edge that may
    already exist, and `target` must be an id (there are no siblings on a
    revise). Built from CONNECT_TO_ITEM_SCHEMA so an edit there reaches both."""
    base = CONNECT_TO_ITEM_SCHEMA["properties"]
    props = dict(base)
    props["target"] = {"type": "string", "description": (
        "The other node's exact 8-char hex id — an existing node; sibling "
        "titles are not valid on revise.")}
    props["relation"] = swappable({"description": (
        base["relation"]["description"] + " On revise a bare relation "
        "identifies the edge row (required when the pair carries more than "
        "one relation); a swap {old, new} renames it in place — weight and "
        "history survive. Also the relation to create when the pair has no "
        "edge yet.")})
    props["why"] = swappable({"description": (
        base["why"]["description"] + " On revise a bare string replaces the "
        "description and a swap {old, new} patches it; required, bare, when "
        "the edge is being created.")})
    rel_items = base["relations"]["items"]
    props["relations"] = {**base["relations"], "items": {
        **rel_items, "properties": {k: swappable(v) for k, v in
                                    rel_items["properties"].items()}}}
    return {**CONNECT_TO_ITEM_SCHEMA, "properties": props}


REVISE_CONNECT_TO_ITEM_SCHEMA = _revise_connect_to_item_schema()

# The shapes a tool schema may point at with `$ref` — keyed by the name after
# `#/$defs/`. The registry attach_defs() draws from; a pointer to a name not
# here is a build-time error, never a dangling reference the model meets.
SCHEMA_DEFS = {
    "swap": SWAP_SCHEMA,
    "connect_to_item": CONNECT_TO_ITEM_SCHEMA,
    "revise_connect_to_item": REVISE_CONNECT_TO_ITEM_SCHEMA,
}


def referenced_defs(schema):
    """Names of every `#/$defs/<name>` a schema points at, transitively
    through the definitions themselves (revise_connect_to_item points at
    swap). Raises on a pointer with no registry entry — a dangling `$ref`
    would otherwise ship silently and leave the model a field it cannot
    fill."""
    found, todo = set(), [schema]
    while todo:
        node = todo.pop()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                name = ref[len("#/$defs/"):] if ref.startswith("#/$defs/") else None
                if name not in SCHEMA_DEFS:
                    raise ValueError("schema points at an unregistered definition %r" % ref)
                if name not in found:
                    found.add(name)
                    todo.append(SCHEMA_DEFS[name])
            todo.extend(node.values())
        elif isinstance(node, list):
            todo.extend(node)
    return found


def attach_defs(schema):
    """A tool's inputSchema with `$defs` holding exactly the shapes it
    references — unchanged when it references none. Applied once per tool
    when the tool list is built."""
    names = referenced_defs(schema)
    if not names:
        return schema
    return {**schema, "$defs": {n: SCHEMA_DEFS[n] for n in sorted(names)}}


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
                           "items": REF_CONNECT_TO_ITEM},
        },
    },
    "revise": {
        "required": ["node_id", "reason"],
        "description": ("Update an existing node. Any other key is a field "
                        "(content, title, situation, question, reasoning, "
                        "...). " + REVISE_RULE + " Non-text fields "
                        "(confidence, type, event_time, ...) take bare values. "
                        "`content_edits` is the alias of `content: [swaps]`."),
        "properties": {
            "node_id": {"type": "string", "description": "Node to revise"},
            "reason": {"type": "string", "description":
                       "Audit note for this revision — recorded in trace "
                       "events, NOT stored on the node. Distinct from the "
                       "node FIELD `reasoning`, which a revise op updates "
                       "like any other field."},
            # One text field shown with the value-or-swap type so the batch
            # schema carries the shape; every other swap field follows the
            # same type (additionalProperties stays open).
            "content": swappable({"description":
                                  "New content, or swaps into the stored content"}),
            "connect_to": {"type": "array", "description":
                           "This node's edges to change or add — an entry per "
                           "target; see the item shape",
                           "items": REF_REVISE_CONNECT_TO_ITEM},
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
                        "Merging two communities preserves their live members; "
                        "ordinary node merges do not inherit community placement. "
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


def unwrap_operations(operations):
    """A brain_batch `operations` value that arrived as a JSON-encoded
    string, unwrapped to the list it encodes — None if it isn't one.
    The single definition of the recovery dispatch performs before
    executing such a batch; readers of recorded action_details (S2
    rejection_table) must interpret the same shape the same way.
    """
    try:
        parsed = json.loads(operations)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) else None


def normalize_connect_to(value):
    """Normalize the optional edge list shared by writes and action readers.

    Null means no requested edges. JSON-encoded lists are accepted by the
    write boundary; malformed values return None so it can report them.
    Entry validation and target resolution remain with the write owner.
    """
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return None
    return value if isinstance(value, list) else None


# ── STRUCTURAL FIELDS ──
# These are columns on the 'nodes' table.
# Changing these requires schema migration.

STRUCTURAL_FIELDS = {
    "id":         {"store": "nodes", "type": "str", "required": True, "immutable": True},
    # Every writable str field takes `{old, new}` swaps on revise (REVISE_RULE,
    # get_swap_fields) — the same default an open KV key has. `bare_only`
    # marks the exceptions: vocabulary and ISO-valued strings (type,
    # evolution_status, event_time, emotion_label) and ids, where nothing is
    # a span. Exclusion is the deliberate act, so a new text field is
    # swappable without anyone remembering a flag.
    "type":       {"store": "nodes", "type": "str", "required": True, "bare_only": True,
                   "description": "Node type (decision, lesson, mechanism, correction, moment, open, ... — open vocabulary, use what fits)."},
    "title":      {"store": "nodes", "type": "str", "required": True,
                   "description": "Specific and scannable — the title is itself an embedded recall vector; specificity is findability."},
    "content":    {"store": "nodes", "type": "str", "replace_on_revise": True, "history": "trace events (node_revised deltas); legacy _sys_revision_history blobs dropped by migration",
                   "description": ("Rich content — reasoning, tradeoffs, specifics. On revise: "
                                   "its new value, or `{old, new}` swaps into what is stored "
                                   "(`content_edits` is the alias of the swap list) — a full "
                                   "rewrite must re-author everything the node holds, and "
                                   "dropped details are silent losses.")},
    "confidence": {"store": "nodes", "type": "float", "range": (0.0, 1.0), "default": 1.0,
                   "description": ("0.0-1.0. Set below 1.0 when the claim is hedged, contested, "
                                   "or inferred — recall exposes it and filters select on it. "
                                   "Don't fabricate precision.")},
    "locked":     {"store": "nodes", "type": "bool", "default": False,
                   "description": ("Protect from casual revision. Belongs to the interactive "
                                   "session: a write from any automated source (encoder, S2, "
                                   "hooks) has its locked:true demoted at the write boundary. "
                                   "Locking is a rare act.")},
    # agent_writable=False: retired from the agent-facing write surface
    # (get_writable_fields) — the column and its read paths stay (recall's
    # critical boost / personal penalty), but no agent is offered the field:
    # near-zero live use, dead options tax every write decision.
    "archived":   {"store": "nodes", "type": "bool", "default": False, "agent_writable": False},
    "critical":   {"store": "nodes", "type": "bool", "default": False, "agent_writable": False},
    "emotion":    {"store": "nodes", "type": "float",
                   "description": "Emotional charge of the moment — signed; recall reads the magnitude. Pair with emotion_label."},
    "emotion_label": {"store": "nodes", "type": "str", "default": "neutral", "bare_only": True,
                      "description": "Name of the felt register ('satisfaction', 'frustration', ...)."},
    # `project` lives in PROMOTED_FIELDS (metadata_kv) — the nodes.project
    # column was dropped in schema v30. Provenance is system-stamped at the
    # write boundary, never agent-authored.
    "personal":   {"store": "nodes", "type": "str", "agent_writable": False},
    "personal_context": {"store": "nodes", "type": "str", "agent_writable": False},
    "evolution_status":  {"store": "nodes", "type": "str", "bare_only": True,
                          "description": "Claim lifecycle once settled: active | resolved | validated | confirmed | disproven | dismissed."},
    "source_turn_id":   {"store": "nodes", "type": "str", "bare_only": True, "description": "message_stream ID that produced this node (episode linkage)"},
    # system_stamped: excluded from the agent-facing schemas — MCP writes
    # default to 'anchor' at the write boundary; scale agents get theirs
    # force-stamped by apply_encoder_attribution. Never agent-authored
    # (the `project` precedent: advertising it as an input only trains drift).
    "encoding_source":  {"store": "nodes", "type": "str", "system_stamped": True, "description": "Who created this node. Convention: category:process. anchor = direct MCP, encoder:sonnet = encoding agent, s2:<unit> = graph units, hook:boot/compaction, migration:*. Only anchor can lock."},
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
    "question": {
        "store": "metadata_kv",
        "type": "str",
        "embeds": True,
        "derived_vector": "question",
        "description": ("The question this node answers, as the other side "
                        "would ask it — gets its own recall embedding, "
                        "bridging how it's stored and how it's asked for. "
                        "Skip when the title already asks it."),
    },
    "event_time": {
        "store": "metadata_kv",
        "type": "str",
        "bare_only": True,
        "description": ("When the remembered thing HAPPENED — ISO 8601, "
                        "distinct from created_at (when it was written). "
                        "Resolve relative dates to absolute; leave absent "
                        "rather than guess. Read by the temporal lane at "
                        "recall."),
    },
    "reasoning": {
        "store": "metadata_kv",
        "type": "str",
        "description": ("What this claim rests on — how it was established "
                        "(measured, reported, inferred), how strongly, and "
                        "what would change it. Written for a reader who has "
                        "never seen this prompt. NOT revise()'s `reason` "
                        "param — that is the audit note for a revision, "
                        "recorded in trace events and never stored on the "
                        "node."),
    },
    "thought": {
        "store": "metadata_kv",
        "type": "str",
        "description": ("My own read on the memory — a hunch, a connection, "
                        "a take the content doesn't carry. Delivered: "
                        "rendered beside the node at recall and in encoder "
                        "catalogs. A living field — update it when a re-read "
                        "moves it; most nodes carry none, and empty is "
                        "correct."),
    },
    "their_raw_quote": {
        "store": "metadata_kv",
        "type": "str",
        "description": "Their exact words — my counterpart's, verbatim.",
    },
    "my_raw_quote": {
        "store": "metadata_kv",
        "type": "str",
        "description": "My own exact words — reflections, realizations, insights.",
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
    schemas and the encoder's field summary. Excludes immutable fields,
    system_stamped ones (project, counterpart, encoding_source: the write
    boundary derives them; advertising them as inputs would only train
    drift), and agent_writable=False ones (retired from the write surface;
    columns and read paths stay)."""
    return {k: v for k, v in ALL_FIELDS.items()
            if not v.get("immutable") and not v.get("system_stamped")
            and v.get("agent_writable", True)}


def get_swap_fields():
    """{name: spec} of the writable fields that take `{old, new}` swaps on
    revise (REVISE_RULE): every str field not marked `bare_only`. Open KV keys
    (not in ALL_FIELDS) are swappable too — callers treat an unknown key as
    text. Same shape as get_writable_fields so a generator can `swappable(spec)`
    straight from it."""
    return {k: v for k, v in get_writable_fields().items()
            if v.get("type") == "str" and not v.get("bare_only")}


def get_remember_fields():
    """Fields that brain.remember() accepts — ALL writable fields.
    Structural fields go to nodes table, promoted fields go to their
    respective stores (node_metadata_kv, node_enrichments)."""
    return get_writable_fields()


def validate_field(name, value, revising=False):
    """Validate a field value against the contract. Returns (ok, error_msg).
    `revising=True` admits the swap shape (REVISE_RULE) — swaps patch a stored
    value, so they exist only on revise; on remember a swap is a type error."""
    if name not in ALL_FIELDS:
        return True, None  # Unknown field — free field, no validation

    spec = ALL_FIELDS[name]
    if value is None:
        return True, None  # NULL is always valid

    expected_type = spec.get("type")
    # Value-or-swap: a swap is valid on any text field that is not bare_only;
    # on anything else it is a schema error, said plainly.
    if revising and (is_swap(value) or is_swap_list(value)):
        if expected_type == "str" and not spec.get("bare_only"):
            return validate_swaps(value, name)
        return False, ("%s takes a bare value, not a swap {old, new} — swaps "
                       "are for text fields" % name)
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


# The `encoding_source` convention declared above, made executable for the
# doors that key BEHAVIOUR off provenance rather than just recording it.
# Today that is the Thalamus door, where the same string is both the
# per-producer budget key and the withdraw-ownership key, so free text forks
# silently (a typo'd source gets a fresh budget and orphans its own items).
# Shape only: the category set stays open so a new producer needs no
# registration.
_SOURCE_RE = re.compile(r'[a-z][a-z0-9_.-]*(:[a-z0-9_.-]+)?')


def validate_encoding_source(value):
    """Validate a `category:process` provenance string. Returns
    (ok, error_msg) — validate_field's convention."""
    if not value or not isinstance(value, str):
        return False, ("source is required — 'category' or 'category:process' "
                       "(e.g. 'anchor', 'encoder:sonnet', 's2:consolidation')")
    if not _SOURCE_RE.fullmatch(value):
        return False, ("source %r is not 'category[:process]' — lowercase "
                       "letters, digits and _ . - , at most one colon "
                       "(e.g. 'anchor', 'encoder:sonnet', 's2:consolidation')"
                       % (value,))
    return True, None


# Node id = uuid4().hex[:8]. Trace ids share the shape, different namespace —
# their predicates stay their own.
_NODE_ID_RE = re.compile(r'[0-9a-f]{8}')


def looks_like_node_id(value):
    """True when `value` has the node-id shape. Shape only — existence is the
    graph's question."""
    return isinstance(value, str) and _NODE_ID_RE.fullmatch(value) is not None


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


# ── NODE-FETCH RENDER: two views, one selector ──
# Every tool that hands nodes back by id or by recall — MCP get_nodes, the
# recall tool's results, an encoder's own get_nodes results — renders through
# node_format_for(n, rich). Up to GET_NODES_DETAIL_MAX nodes get the DETAIL
# view: the reader is going to act on these nodes. More get the SCAN view:
# the reader is judging fit across many, and edges dominate the cost
# (measured 2026-06: generous content is nearly free, an extra edge is not).
# One selector for every door is what keeps a recall of three nodes and a
# get_nodes of the same three reading identically.
GET_NODES_DETAIL_MAX = 10

# The agent-facing node-count cap for filter_nodes. The DAL read is unbounded
# (limit=None → all matches — internal id-set scans need every row); this bound
# lives at the dispatch door, where results render into the caller's context
# and a flood is the real risk. Pairs with dal_logs.LOG_QUERY_MAX_LIMIT and
# brain_constants.EPISODE_MAX_LIMIT — same pattern (named per-door cap), each
# valued for its own cost. Internal Python callers bypass dispatch → uncapped.
NODE_QUERY_MAX_LIMIT = 200
# Default page an agent gets when it names no limit (mirrors EPISODE_DEFAULT_LIMIT).
NODE_QUERY_DEFAULT_LIMIT = 50

# DETAIL — the reader will act on these nodes (revise, link, answer from
# them). Content whole: it is the signal the pull was for. The edge tail
# (40-76% of a well-connected node's payload) and the heavy correction K/V
# are what get bounded; the total says when the cut dropped edges.
GET_NODES_DETAIL_FORMAT = {
    'content_limit': None,
    'edge_limit': 8,
    'metadata_limit': 300,
    'correction_render': 'balanced',
    'time_format': 'relative',
    'communities': 'ref',       # Anchor follows ids: "title" (id:xxxxxxxx)
    'show_edge_total': True,
}

# SCAN — the reader is judging fit across many nodes. 800 chars of content
# is enough to judge a claim and costs little; the edges are the budget.
GET_NODES_SCAN_FORMAT = {
    'content_limit': 800,
    'edge_limit': 5,
    'metadata_limit': 200,
    'correction_render': 'balanced',
    'time_format': 'relative',
    'communities': 'ref',
    'show_edge_total': True,
}

# rich=True (the MCP opt-in on get_node/get_nodes): DETAIL lifted to every
# edge and the heavy correction K/V (reasoning + raw quotes) — the deliberate
# "give me everything" drill, still curated through render_rich_node.
GET_NODES_RICH_LIFT = {
    'edge_limit': None,
    'metadata_limit': 400,
    'correction_render': 'heavy',
}


def node_format_for(n, rich=False, explicit=None):
    """The render config for a fetch of `n` nodes — the one selector behind
    MCP get_nodes, the recall tool's results and an encoder's get_nodes
    results. An `explicit` caller config (run_llm_loop's get_nodes_config —
    how a consumer declares its own representation) wins outright and is
    never second-guessed; otherwise `rich` lifts DETAIL to the full view,
    and the count picks DETAIL (act on these) or SCAN (judge across many)."""
    if explicit is not None:
        return explicit
    if rich:
        return {**GET_NODES_DETAIL_FORMAT, **GET_NODES_RICH_LIFT}
    return (GET_NODES_DETAIL_FORMAT if n <= GET_NODES_DETAIL_MAX
            else GET_NODES_SCAN_FORMAT)

# The skinny node shape returned by NodeDAL.filter_nodes (dal.py:2038-2040)
# when rich=False — id/title/type/confidence/created_at, plus the filtered
# column. render_skinny_node uses this to tell the standard columns from the
# filtered field it should surface. Keep in sync with the DAL's SELECT.
SKINNY_NODE_FIELDS = ('id', 'title', 'type', 'confidence', 'created_at')

# ── Absorbed-id redirect vocabulary — ONE owner for the key and the marker.
# The canonical pull stamps REDIRECTED_FROM_KEY (a list of the requested ids
# that resolved to this node); every render marks it through these two
# formatters, never a hand-written glyph. Node-body renders use the sentence
# form; one-line ref renders use the compact form.
REDIRECTED_FROM_KEY = '_redirected_from'

# What the canonical pull (Brain.get_node) attaches on top of the bare DB row:
# the KV block, the two fields promoted out of it to top-level, the correction
# chain, the edges, the community membership, and — when the requested id was
# absorbed — the redirect marker. `canonicalize_results` overlays exactly
# these onto a recall result, so a result carries the same shape whichever
# recall door the caller came through. Keep in sync with get_node's assembly
# — the parity test (tests/test_recall_door_parity.py) fails if get_node
# grows an attachment this tuple doesn't name.
CANONICAL_ATTACHMENT_KEYS = ('_metadata', 'situation', 'project',
                             '_corrections', 'connections', 'communities',
                             REDIRECTED_FROM_KEY)


def redirect_marker(src_id, dst_id):
    """Sentence-form marker for node-body renders."""
    return ('⚠ %s ↦ %s — requested id was absorbed into this node'
            % (str(src_id)[:8], str(dst_id)[:8]))


def redirect_ref_line(src_id, dst_id, title):
    """Compact one-line marker for ref-list renders."""
    return '%s ↦ %s · %s (absorbed)' % (
        str(src_id)[:8], str(dst_id)[:8], title or '')


def absorbed_refusal(verb, node_id, survivor_id):
    """The write-door refusal message — writes never redirect; every door
    refuses an absorbed id with the SAME sentence and the pointer, so a
    producer holding a stale alias can re-aim regardless of which door it
    hit. Dict-shaped doors also return survivor_id as a structured field."""
    return ('Cannot %s %s — absorbed into %s; %s that node instead'
            % (verb, str(node_id)[:8], str(survivor_id)[:8], verb))

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


# The inject's content cut: a sentence-boundary cap whose marker points the
# reader at the whole node. The mechanism is loud_truncation's; only the
# marker wording is this consumer's.
CONTENT_CUT_MARKER = '… (+%d chars: get_nodes)'

# How many trace ids the Conversation line names before it counts the rest.
SOURCE_REF_RENDER_LIMIT = 8


# ── Corrector K/V allowlist for render_corrections heavy mode ──
# Three keys carry meaningful correction context: the corrector's stored
# reasoning, the operator's words, and Anchor's words. Anything else on
# the corrector's metadata is either bookkeeping or downstream-of-surface
# context the heavy render shouldn't replay.
_CORRECTION_HEAVY_KV_KEYS = ('reasoning', 'their_raw_quote', 'my_raw_quote')


def render_corrections(corrections, mode='lean',
                       content_limit_balanced=150,
                       content_limit_heavy=400,
                       meta_limit_heavy=300,
                       indent='  ', limit=None):
    """Render a node's `_corrections` list as formatted lines.

    Single rendering path — both render_rich_node and consumer-specific
    formatters (HealerEncoder._format_batch) call this. Per the data/format
    separation contract: one formatter, configs drive
    verbosity.

    Args:
        corrections: list of correction dicts (output of correction_enrich).
            Each carries: id, title, type, direction, relation,
            edge_description, content, reasoning, their_raw_quote.
        mode: 'none' | 'lean' | 'balanced' | 'heavy'.
              none     → no lines emitted
              lean     → header line only (title + id) — legacy default
              balanced → + relation verb + edge_description + content excerpt
              heavy    → + full content + corrector K/V (reasoning,
                         their_raw_quote, my_raw_quote) honouring the
                         noise filter.
        content_limit_balanced: char cap for content excerpt in balanced mode
        content_limit_heavy: char cap for content in heavy mode
        meta_limit_heavy: char cap per K/V value in heavy mode
        indent: line prefix for nested values (default '  ')
        limit: render at most this many corrections, then one line
               counting the rest (None = all). The inject caps here: a
               node with seven corrections spent 2,753 chars on them
               while its own content got 114.

    Returns list[str] of lines (no leading section header — caller can
    prepend 'CORRECTIONS:' or similar).
    """
    if mode == 'none' or not corrections:
        return []

    shown = corrections[:limit] if limit and len(corrections) > limit else corrections
    hidden = len(corrections) - len(shown)
    lines = []
    sub_indent = indent + '   '
    for corr in shown:
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

    if hidden:
        lines.append('%s+%d more correction%s' % (indent, hidden,
                                                  '' if hidden == 1 else 's'))
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
#
# A door can saturate without a window: get_traces serves a capped slice of
# an explicit id list. Same payload shape, same banner — truncation_payload_ids
# carries that cause, because "covers only [t0..t1], raise limit" is the wrong
# thing to tell a caller whose remedy is "ask for the remaining ids".

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


TRUNCATION_IDS_LISTED = 25   # ids named in a note before it only counts them


def truncation_payload_ids(requested, cap):
    """The 'truncated' payload for a saturated ID-LIST read (get_traces).

    Same dict shape and same ⚠ banner as the windowed sibling, different
    cause: there is no window to under-cover and no +1 probe to run — the
    caller named the ids, so the overflow is exact and so is the remedy
    (call again with the rest). `requested` is the DEDUPED id list.

    The note NAMES the dropped ids, it doesn't just count them — the banner
    renders only `note`, so a count would make 'call again with them'
    unactionable. The caller can't re-derive them either: the cap applies
    after dedupe, so slicing their own input at `cap` gives the wrong set.
    Long overflows are listed up to TRUNCATION_IDS_LISTED (the full set
    always stays in `dropped_ids` for programmatic readers).
    """
    dropped = list(requested[cap:])
    shown = dropped[:TRUNCATION_IDS_LISTED]
    more = ('' if len(dropped) <= len(shown)
            else ' +%d more (paginate)' % (len(dropped) - len(shown)))
    return {
        'limit': cap,
        'dropped_ids': dropped,
        'note': ('%d distinct ids requested, %d served — this door fetches '
                 '%d per call. Not looked up, call again with these: %s%s'
                 % (len(requested), cap, cap, ', '.join(shown), more)),
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


def _fmt_node_time(ts, cfg):
    """A timestamp the way a render config asks for it: relative ('3w ago';
    time_now = the as-of instant for replay-safe callers, time_fine = sub-day
    steps for the encoder catalog) or the bare date. None when empty."""
    if not ts:
        return None
    if cfg.get('time_format') == 'relative':
        from servers.pipeline_contract import _relative_time
        return _relative_time(ts, now=cfg.get('time_now'),
                              fine=cfg.get('time_fine', False))
    return str(ts)[:10]


def relation_age(rel, conn=None):
    """When a relation's CLAIM last changed — the one recency every reader
    agrees on: `updated_at` (description, weight or verb repaired in place),
    else `created_at` (never repaired), else the pair's `created_at`. The
    edge line renders it and get_node's ordering ties break on it, so a
    freshly repaired claim both reads as recent and sorts as recent."""
    return (rel.get('updated_at') or rel.get('created_at')
            or (conn or {}).get('edge_created_at') or '')


def render_edge_lines(conn, cfg=None, indent='    '):
    """The one edge render for LLM readers — the lines under a node's Edges.
    Every reader that shows a node's edges (the encoder catalog, the recall
    surface, Anchor's get_node, the S2 units) calls this, so the grammar
    cannot drift between them.

    `conn` is a get_connections_bulk entry — one neighbor, grouped: id, type,
    title, direction, edge_created_at, relations: [{relation, description,
    created_at}]. One line per relation:

        [type id:xxxxxxxx <age>] this <relation> "<neighbor title>" — <description>
        [type id:xxxxxxxx <age>] "<neighbor title>" <relation> this — <description>

    the second form for an incoming edge (the neighbor is the actor). The age
    is `relation_age` — when the RELATION's claim last changed — so a reader
    can tell a fresh claim from one older than the node it hangs on. The
    description is
    never truncated: a reader may copy it verbatim as a swap's `old`, and a
    cut copy fails the exactly-once match. The neighbor title is cut at 100
    chars — it is the neighbor's own field, not something this line is for
    editing.

    `cfg` merges over NODE_FORMAT_DEFAULTS like render_rich_node's, so a
    direct caller (the S2 units) and a rich-node render agree on defaults.
    edge_style='oneline' is the selection-grade surface: direction, relation
    and title only; no description or age, and the neighbor's id only when
    `edge_ids` is set (the inject's seed render, so the reader can fetch it).
    """
    cfg = {**NODE_FORMAT_DEFAULTS, **(cfg or {})}
    title = (conn.get('title') or '')[:100]
    incoming = conn.get('direction') == 'incoming'
    rels = conn.get('relations') or [{'relation': conn.get('relation', ''),
                                      'description': conn.get('description', ''),
                                      'created_at': None}]
    if cfg.get('edge_style') == 'oneline':
        r = rels[0]
        rel = r.get('relation') or 'related'
        short = title[:80]
        # edge_ids (the inject): the neighbor's short id after the title, so
        # the reader can fetch it. Off for the picker, whose lean render is
        # ablation-measured without ids.
        tail = ' (id:%s)' % (conn.get('id') or '?')[:8] if cfg.get('edge_ids') else ''
        return [indent + ('"%s" %s this' % (short, rel) if incoming
                          else 'this %s "%s"' % (rel, short)) + tail]
    lines = []
    tag_head = '[%s id:%s' % (conn.get('type', '?'), (conn.get('id') or '?')[:8])
    for r in rels:
        rel = r.get('relation') or 'related'
        age = _fmt_node_time(relation_age(r, conn), cfg) or '?'
        desc = r.get('description') or ''
        head = ('"%s" %s this' % (title, rel)) if incoming else ('this %s "%s"' % (rel, title))
        lines.append('%s%s %s] %s%s' % (indent, tag_head, age, head,
                                        ' — %s' % desc if desc else ''))
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

    # Header — individual parts are opt-out via cfg flags (defaults preserve
    # current behavior for callers that don't set them, e.g. Anchor's MCP queries).
    parts = ["id:%s" % nid[:8]]
    # A retired node renders honestly — without this flag an archived hit
    # (audit hatch, orphaned pointer) reads as current.
    if node.get('archived'):
        parts.append("⚠ ARCHIVED")
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
    # A claim the brain has since disproven or dismissed must read as such
    # where it is shown as knowledge — opt-in per consumer (the inject sets
    # it), so catalog and MCP renders keep their current header.
    if cfg.get('mark_status'):
        _status = str((node.get('_metadata') or {}).get('evolution_status')
                      or node.get('evolution_status') or '').strip().lower()
        if _status in ('disproven', 'dismissed'):
            parts.append('⚠ %s' % _status.upper())
    if cfg.get('show_encoding_source', True):
        if node.get('encoding_source'):
            parts.append("src:%s" % node['encoding_source'])
    created_rel = _fmt_node_time(node.get('created_at'), cfg)
    revised_rel = _fmt_node_time(node.get('revised_at'), cfg)
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

    # Canonical-pull redirect: the consumer asked for an absorbed id and got
    # this survivor — say so, always, with both ids (a silent swap would breed
    # the confusion class the redirect exists to prevent).
    for _src in node.get(REDIRECTED_FROM_KEY) or ():
        lines.append('  ' + redirect_marker(_src, nid))

    # Content (None = no truncation, 0 = hide, N = truncate to N chars)
    content_limit = cfg.get('content_limit')
    if content_limit != 0:
        content = node.get('content', '')
        if content_limit and content_limit > 0:
            # 'sentence': cut on a sentence boundary and name the remainder
            # (the inject) — a reader acting on the text must know it ends
            # early. Default: the bare ellipsis cut.
            if cfg.get('content_cut') == 'sentence':
                content = cap_text_at_boundary(content, content_limit,
                                               marker=CONTENT_CUT_MARKER)
            else:
                content = _truncate(content, content_limit)
        if content:
            lines.append('  Content: %s' % content)

    # Situation — a top-level field (the canonical pull promotes it out of
    # KV), so a caller that skips 'situation' must be honored here, not only
    # in the KV loop below.
    situation = node.get('situation', '')
    if situation and 'situation' not in (cfg.get('extra_skip_keys') or ()):
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
    _VOICE_KEYS = ('their_raw_quote', 'my_raw_quote')
    _VOICE_LIMIT = 600
    # Caller-supplied key-prefix skips (the inject drops every community_*
    # bookkeeping field) and per-key labels (the inject renders the
    # counterpart's quote as '<name> said:' — a label the reader reads
    # through, not a record label to parse).
    skip_prefixes = tuple(cfg.get('skip_key_prefixes') or ())
    voice_labels = cfg.get('voice_labels') or {}
    if meta_limit > 0:
        for key, val in meta.items():
            if not val or key in skip_keys:
                continue
            # _sys_ prefix = system/infrastructure fields, never shown to LLMs
            if key.startswith('_sys_') or (skip_prefixes and key.startswith(skip_prefixes)):
                continue
            limit = _VOICE_LIMIT if key in _VOICE_KEYS else meta_limit
            label = voice_labels.get(key) or key.replace('_', ' ').title()
            lines.append('  %s: %s' % (label, _truncate(str(val), limit)))

    # Conversation pointer — the trace ids behind this memory, written as
    # the call that opens them. Rendered only when the caller attached
    # `source_refs` to the node AND asked for the line (the inject does for
    # Haiku's picks; a bulk fetch on every canonical pull would tax every
    # reader for a line only some of them want).
    if cfg.get('show_source_refs'):
        _refs = [str(r)[:8] for r in (node.get('source_refs') or []) if r]
        if _refs:
            _more = ('  +%d more' % (len(_refs) - SOURCE_REF_RENDER_LIMIT)
                     if len(_refs) > SOURCE_REF_RENDER_LIMIT else '')
            lines.append('  Conversation: get_traces(%s)%s' % (
                json.dumps(_refs[:SOURCE_REF_RENDER_LIMIT]), _more))

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
        meta_limit_heavy=meta_limit,
        limit=cfg.get('correction_limit')))

    # Community membership — its own line, never an edge line: community
    # edges are noise-excluded from `connections` (they carry no claim and
    # were taking 27% of top-5 slots). cfg `communities`: 'title' shows the
    # placement as context (most readers), 'ref' adds the id for a reader
    # that follows it with a pull (Anchor's tools), unset/off renders nothing
    # (the consolidation block prints its own line; the picker is
    # ablation-measured).
    comm_mode = cfg.get('communities')
    if comm_mode and node.get('communities'):
        lines.append('  Communities: %s' % ', '.join(
            ('"%s" (id:%s)' % ((c.get('title') or '?')[:100], (c.get('id') or '?')[:8])
             if comm_mode == 'ref' else '"%s"' % (c.get('title') or '?')[:100])
            for c in node['communities']))

    # Edges — one grammar for every reader, owned by render_edge_lines.
    edge_limit = cfg.get('edge_limit', 5)
    all_conns = node.get('connections', [])
    connections = all_conns[:edge_limit]
    # Edge-total indicator (opt-in via show_edge_total — the encoder catalog
    # and Anchor's small/balanced pulls): when the limit truncates, say so —
    # '(5 of 23)' tells the reader how connected the node really is (the DAL
    # pull is uncapped and noise-excluded, so the total is honest). Default
    # off: legacy renders keep the bare header.
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
    if connections:
        lines.append(edges_header)
        for e in connections:
            lines.extend(render_edge_lines(e, cfg))

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
        if k not in SKINNY_NODE_FIELDS and k != REDIRECTED_FROM_KEY
        and v is not None)
    # Redirect marker, not a raw list dump — same owner as the rich render.
    for _src in node.get(REDIRECTED_FROM_KEY) or ():
        extra = (redirect_marker(_src, node.get('id') or '')
                 + (('  ' + extra) if extra else ''))
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
    # source_refs is a join-table field, not a contract column — appended
    # here explicitly so the agent-facing list is actually complete.
    lines.append("source_refs  (array)  — 8-char hex trace ids anchoring the node "
                 "to its originating moments; sparse (1-3 load-bearing turns), "
                 "copied verbatim from the input's trace markers")
    # The revise rule, once, in the contract's own words — this summary is
    # injected after the prompt, so it is the surface that wins on a
    # disagreement (E10); it must state the same rule the tools do.
    lines.append("")
    lines.append(REVISE_RULE)
    lines.append("")
    lines.append("RETURNS: every remember (single or batch) returns related_nodes — "
                 "the top 5 most similar existing nodes with full content. "
                 "Use these to draw edges immediately (a connect op, or connect_to) "
                 "without a separate recall round. "
                 "related_nodes is NOT the outcome of connect_to — when you pass "
                 "connect_to, the response carries a separate connect_to_result "
                 "{created:[...], failed:[{title, reason}]} reporting which edges "
                 "formed and why any didn't.")
    return "\n".join(lines)
