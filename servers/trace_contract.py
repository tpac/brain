"""Trace Contract — single source of truth for the fractal trace system.

Defines: scales, event types, ref types, and validation.
All trace writers MUST validate against this contract.
All trace readers can rely on these guarantees.

Architecture: docs/ARCHITECTURE-FRACTAL.md
"""

from servers.loud_truncation import cap_text_loud, cap_list_loud


# ── SCALES ──
# The fractal hierarchy. Each scale observes the one below.

SCALES = {
    "s0": {
        "name": "Exchange",
        "description": "Raw partnership interaction — messages, tool calls, tool results",
        "triggers": "Every turn (Stop hook) + every tool call (PostToolUse hook)",
    },
    "s1": {
        "name": "Turn",
        "description": "Brain's first processing pass — surface, encode",
        "triggers": "UserPromptSubmit (surface) + Stop every 5th (encode)",
    },
    "s2": {
        "name": "Graph",
        "description": "Graph-wide operations on S1's accumulated output — communities, dedup, confidence, corrections",
        "triggers": "Idle hook (between sessions)",
    },
    "s3": {
        "name": "Reasoning",
        "description": "Cross-cluster patterns, abstract insights, resolved uncertainties",
        "triggers": "Periodic / scheduled",
        "status": "NOT BUILT",
    },
    "s4": {
        "name": "Growth",
        "description": "External knowledge and long-term evolution",
        "triggers": "Periodic / weekly",
        "status": "NOT BUILT",
    },
}


# ── EVENT TYPES ──
# O/K/Δ/outcome — structurally identical at every scale.

EVENT_TYPES = {
    "O": "Observation — everything available at this moment",
    "K": "Knowledge — what was selected as relevant from O",
    "delta": "Changes — what was produced (the response, encoding, reorganization)",
    "outcome": "What happened next — added retrospectively (corrections, future recalls)",
}


# ── REF TYPES ──
# What ref_type values are valid per scale + event_type.
# ref_type tells you WHAT the event is about. ref_id points to it.

REF_TYPES = {
    # Scale 0: raw exchange
    ("s0", "K"):       ["user_message",
                         "self_message",      # incoming turn from a stream of thought (self↔self),
                                              # not the operator — same exchange, different correspondent
                         "heartbeat"],        # a /watch wakeup re-arm with no real input (no operator
                                              # prompt, empty inbox). Recorded for observability, but
                                              # NOT a conversational turn — see S0 TURN CLASSIFICATION.
    ("s0", "delta"):   ["assistant_message", "tool_result",
                         "node_revised", "edge_relation_revised"],
    ("s0", "outcome"): ["correction", "follow_up"],

    # Scale 1: turn integration
    # Surface path (chain prefix: s1r-): O=candidates, K=surfaced picks, delta=context sent to Anchor
    # Encode path (chain prefix: s1e-): O=prompt given, K=node catalog, delta=actions+reasoning
    ("s1", "O"):       ["recall",            # candidates with scores
                         "encoding_prompt",    # what the encoder was given
                         "scout_input"],       # muster scouts: what they saw
    ("s1", "K"):       ["surface_selected",  # what the surfacer picked
                         "node_catalog",       # which nodes available to encoder
                         "scout_findings"],    # muster scouts: their candidates
    ("s1", "delta"):   ["additionalContext",       # what reached Anchor
                         "encoding_run",            # what the encoder produced
                         "node_revised",            # field-level revise emitted by S1 encoder
                         "edge_relation_revised"],  # connect upsert / archive emitted by S1 encoder
    ("s1", "outcome"): ["correction",         # Tom corrected something that was recalled
                         "recall_hit"],        # node was recalled in a future turn

    # Scale 2: graph integration
    # Fires during idle hook. Operates on S1's accumulated output (the graph).
    # Multiple integration units, each with own O/K/Δ.
    ("s2", "O"):       ["graph_structure",      # nodes + edges observed (community detection)
                         "graph_stats",          # node/edge counts, density
                         "s1_delta",             # S1 encoding/surfacing traces since last run
                         "consolidation_candidates",  # embedding scan + behavioral evidence
                         "heal_archive",         # decoder-level archive of broken artifacts (e.g. 0-member communities)
                         "correction_chains",    # brain-wide correction chain traversal
                         "healer_scan",          # S2 Healer: gaps + flags scanned
                         "aspect_scan"],         # S2 AspectIntegration: distinct types/relations vs aspects_v1.json
    ("s2", "K"):       ["community_proposals",  # S2CD proposals (placements, overlaps, splits, seeds)
                         "community_partition",  # algorithm output (communities + membership)
                         "community_diff",       # comparison with previous run
                         "consolidation_proposals",   # enriched clusters with pre-classification
                         "stale_nodes",          # nodes not accessed recently
                         "healer_proposals",     # S2 Healer: nodes to heal (fill missing fields)
                         "aspect_proposals"],    # S2 AspectIntegration: candidate strings + example records
    ("s2", "delta"):   ["community_enriched",   # S2CE enrichment results (accepted, rejected, placed)
                         "community_created",    # new community node
                         "community_updated",    # revised community node
                         "community_removed",    # stale community archived
                         "community_assignments",# membership edges updated
                         "recall_quality_signal",# recall diagnostic (false positive, redundancy, gap)
                         "consolidated",         # new node from smart merge
                         "evolved",              # evolution edge added
                         "kept_distinct",        # similar_to edge, no merge
                         "confidence_adjust",       # adjusted confidence scores
                         "healer_generated",        # S2 Healer: missing fields generated + stored
                         "aspect_classified",       # S2 AspectIntegration: candidates merged into aspects_v1.json
                         "node_revised",            # field-level revise emitted by S2 units (healer, consolidation)
                         "edge_relation_revised"],  # connect upsert / archive emitted by S2 units
    ("s2", "outcome"): ["recall_improved",      # community nodes improved recall
                         "operator_reviewed"],   # Tom reviewed S2 output

    # Scale 3: reasoning integration
    # Operates on S2's output (clusters, trajectories, landscapes).
    ("s3", "O"):       ["cluster_patterns",     # S2 clusters across parameters
                         "correction_trajectories",  # how understanding evolved
                         "confidence_landscapes"],    # stable vs turbulent areas
    ("s3", "K"):       ["cross_cluster",        # nodes appearing across multiple clusters
                         "learning_curves"],     # correction trajectories over time
    ("s3", "delta"):   ["abstract_insight",     # cross-cluster pattern recognized
                         "resolved_question",    # uncertainty answered
                         "meta_optimization"],   # S2 prompt/config improvement
    ("s3", "outcome"): ["adopted",              # insight used by Tom/Anchor
                         "rejected"],            # Tom rejected the insight

    # Scale 4: growth integration
    # Fires periodically (weekly). Sees full graph + external sources.
    ("s4", "O"):       ["uncertainty_nodes",   # brain's open questions
                         "external_research"],  # web search results, papers
    ("s4", "K"):       ["stale_decisions",     # decisions that may be outdated
                         "open_questions"],     # unresolved uncertainties
    ("s4", "delta"):   ["research_finding",    # new knowledge from outside
                         "decision_update",     # stale decision refreshed
                         "cross_project"],      # bridge between projects
    ("s4", "outcome"): ["adopted",             # finding was used by Tom/Anchor
                         "rejected"],           # Tom rejected the finding
}


# ── S0 TURN CLASSIFICATION ──
# Every stop produces one S0 turn, classified by its incoming-side (s0,"K")
# ref_type. NOT every turn is a conversation worth encoding. This is the single
# source of truth for "what's filtered and what's not" — consumers read these
# constants instead of re-deciding inline.
#
#   incoming ref_type   what it is                                conversational?
#   ──────────────────────────────────────────────────────────────────────────
#   user_message        real operator prompt (hook_recall ran;     YES
#                        last_user_activity reset this turn)
#   self_message        inbound msg from another stream            no  (planned)
#                        (anchor↔anchor)
#   heartbeat           /watch wakeup re-arm, no real input        no  (never)
#                        (no prompt + empty inbox)
#
# "conversational" means BOTH: (a) the turn counts toward the S1 Scribe's
# integration CADENCE — derived live from these traces by
# turns_since_last_encode() (counts s0 user_message turns since the last encode),
# which the Scribe gates on (>= ENCODE_EVERY). This is distinct from stop_counter,
# the per-stop SEQUENCE number that advances on EVERY stop (incl. heartbeats) so
# chain IDs stay unique. And (b) the encoder reads it via get_session_turns.
# Non-conversational turns are still written to S0 (for observability) but never
# drive or feed encoding.
#
# anchor↔anchor encoding is a PLANNED capability, switched OFF today. The single
# dial to enable it is below: flip self_message to True. heartbeat stays False
# forever.
S0_CONVERSATIONAL_INCOMING = {
    "user_message": True,
    "self_message": False,   # recorded today; flip to True to encode anchor↔anchor turns
    "heartbeat":    False,   # a wakeup re-arm is never a turn
}

# Flat ref_type set the encoder's conversation window (dal.get_session_turns)
# selects: the conversational incoming types + the assistant response side.
# DERIVED from S0_CONVERSATIONAL_INCOMING so there is exactly one dial — flipping
# a type there updates both the Scribe counter gate and the encoder whitelist.
CONVERSATIONAL_REF_TYPES = tuple(
    rt for rt, conv in S0_CONVERSATIONAL_INCOMING.items() if conv
) + ("assistant_message",)

# A wakeup ignite (e.g. a background-task notification) arrives as turn CONTENT,
# not a distinct ref_type: it runs recall, so it's recorded as a `user_message`
# (conversational) even though it's an ENVELOPE, not work. Presence focus skips
# any conversational turn whose summary starts with this marker. One constant so
# the skip is defined ONCE, not reproduced as a scattered SQL literal.
WAKE_ENVELOPE_MARKER = "<task-notification>"


# ── CHAIN ID CONVENTIONS ──
# chain_id groups related O/K/Δ/outcome events.
#
# One chain per stop at S0. Everything between stop N-1 and stop N
# (messages, tool calls) belongs to the same S0 chain.
# S1 chains reference the S0 chain via parent_chain in metadata.

CHAIN_PREFIXES = {
    "s0":         "s0-{session_short}-{stop}",        # one chain per stop — messages + tools
    "s1_recall":  "s1r-{session_short}-{stop}",       # surface for this stop
    "s1_encode":  "s1e-{session_short}-{stop}",       # encoding run triggered at this stop
    "s2":         "s2-{date}-{operation}",              # date=YYYYMMDD, operation=community/dedup/etc
    "s3":         "s3-{date}-{operation}",             # date=YYYYMMDD, operation=synthesis/meta/etc
    "s4":         "s4-{date}-{topic}",                 # date=YYYYMMDD, topic=what was researched
}


# ── DELTA METADATA SHAPE ──
# Agentic encoders (S1E, S2 community, S2 consolidation, S2 healer) all
# have the same structural shape: an LLM loop that processes inputs, runs
# N rounds, produces write actions, writes a journal entry, and may record
# rejection fingerprints. One schema, unit-specific vocab in `outcomes`.

DELTA_METADATA_SHAPE = {
    'actions':           int,     # total tool calls
    'write_actions':     int,     # successful writes to the graph
    'rounds':            int,     # LLM conversation rounds
    'inputs_processed':  int,     # clusters / proposals / nodes seen
    'outcomes':          dict,    # unit-specific vocab: {action_name: count}
    'rejection_skipped': int,     # fingerprints recorded this run
    'journal_entry':     str,     # THIS RUN's journal contribution (extracted)
    'action_details':    list,    # per-action records (truncated if huge)
    'final_text':        str,     # raw agent text, first 2KB
    'errors':            list,    # first 5 errors
    # Op-attributed node-id lists — the structured Δ that S2 reads (community
    # detection, consolidation catalog-blindness). Authoritative split returned
    # by the dispatch write handlers (`affected`), aggregated here over
    # action_details. NODE lifecycle only — edges are NOT in this delta.
    #
    # Edges: every FIRST-CLASS typed edge — explicit connect / connect_batch /
    # revise_edge / disconnect / connect_to / co_anchored — is its own
    # directional `edge_relation_revised` event carrying
    # source_id/target_id/relation, so a flat directionless `connected` node-id
    # list (a two-sided-era vestige that couldn't represent a v22
    # single-direction edge) was removed. SOFT/derived edges (co_accessed,
    # emergent_bridge) are intentionally NOT traced — they're recomputable and
    # excluded from the graph views (dashboard, S2 decisions). So "reconstruct
    # the graph from traces" means the first-class typed graph, not the soft layer.
    'created':           list,    # node ids created this run
    'revised':           list,    # node ids revised this run (incl. absorb survivors — content rewritten)
    'archived':          list,    # node ids archived this run (incl. absorb's folded-in originals)
    # AspectIntegration's structured Δ. Aspect mutates aspects_v1.json, not the
    # graph, so created/revised/archived don't apply — the real change record is
    # WHICH string routed to WHICH aspect(s): [{category, value, aspects}, ...].
    # First-class (validated + dashboard-known + capped) rather than smuggled
    # through **extras. Empty [] for every non-aspect delta.
    'classifications':   list,
    # Cost & provenance of producing this Δ (all int, default 0). elapsed_ms +
    # token counts let you trend encoder latency/cost over time — and, paired
    # with interaction_version, compare cost across prompt versions — straight
    # from traces. truncated flags silent data loss (a max_tokens cut mid
    # tool-call corrupts the write). interaction_version records WHICH K version
    # produced this Δ (the FK `interaction_id` is stamped on the trace row
    # itself; this mirror keeps it human-readable in the payload).
    'elapsed_ms':            int,
    'input_tokens':          int,
    'output_tokens':         int,
    'cache_read_tokens':     int,
    'cache_creation_tokens': int,
    'truncated':             int,
    'interaction_version':   int,
}

DELTA_FINAL_TEXT_LIMIT = 2000
DELTA_ERROR_LIST_LIMIT = 5
DELTA_CLASSIFICATIONS_LIMIT = 200  # cap aspect's per-item Δ (cold-start runs can be large)


def build_delta_metadata(*,
                         actions=0, write_actions=0, rounds=0,
                         inputs_processed=0, outcomes=None,
                         rejection_skipped=0, journal_entry='',
                         action_details=None, read_calls=None,
                         final_text='',
                         errors=None,
                         created=None, revised=None, archived=None,
                         classifications=None,
                         elapsed_ms=0, input_tokens=0, output_tokens=0,
                         cache_read_tokens=0, cache_creation_tokens=0,
                         truncated=0, interaction_version=0,
                         **extras):
    """Build a unified delta trace metadata dict.

    All agentic encoders (S1E, S2 units) should call this to build the
    metadata payload for their `delta` trace event. Standardizes field
    names, applies truncation, and lets each unit pass additional keys
    via **extras (e.g. clusters_processed, batches).

    read_calls captures non-write tool invocations (recall_batch, get_nodes,
    etc.). Useful for observability — answering "what did the encoder ask
    for that the catalog didn't already give it?" without parsing logs.

    created/revised/archived default to an aggregation over action_details
    (each write action carries its own op-attributed split — the `affected`
    dict the dispatch handler returned, copied onto the action by the runner).
    Pass them explicitly only to override. This is the structured Δ S2 reads
    — `revised` includes absorb survivors and `archived` their folded-in
    originals, so a merge-only consolidation run is no longer invisible. Edges
    are out of scope here (see the shape comment) — they live in directional
    `edge_relation_revised` events.

    Returns a dict ready to pass as the metadata kwarg to a trace writer.
    """
    ad = list(action_details or [])

    def _agg(key, explicit):
        if explicit is not None:
            return list(explicit)
        out = []
        for a in ad:
            if isinstance(a, dict):
                out.extend(a.get(key) or [])
        return out

    def _cap(items, limit):
        # Loud-in-data truncation for a dict-list (can't append a string marker
        # like cap_list_loud): keep `limit`, append a sentinel naming the drop.
        items = list(items or [])
        if len(items) <= limit:
            return items
        return items[:limit] + [{'_truncated': len(items) - limit}]

    metadata = {
        'actions':           int(actions or 0),
        'write_actions':     int(write_actions or 0),
        'rounds':            int(rounds or 0),
        'inputs_processed':  int(inputs_processed or 0),
        'outcomes':          dict(outcomes or {}),
        'rejection_skipped': int(rejection_skipped or 0),
        'journal_entry':     cap_text_loud(journal_entry, DELTA_FINAL_TEXT_LIMIT),
        'action_details':    ad,
        'read_calls':        list(read_calls or []),
        'final_text':        cap_text_loud(final_text, DELTA_FINAL_TEXT_LIMIT),
        'errors':            cap_list_loud(errors, DELTA_ERROR_LIST_LIMIT),
        'created':           _agg('created', created),
        'revised':           _agg('revised', revised),
        'archived':          _agg('archived', archived),
        'classifications':   _cap(classifications, DELTA_CLASSIFICATIONS_LIMIT),
        'elapsed_ms':            int(elapsed_ms or 0),
        'input_tokens':          int(input_tokens or 0),
        'output_tokens':         int(output_tokens or 0),
        'cache_read_tokens':     int(cache_read_tokens or 0),
        'cache_creation_tokens': int(cache_creation_tokens or 0),
        'truncated':             int(truncated or 0),
        'interaction_version':   int(interaction_version or 0),
    }
    # Extras preserved for per-unit fields (can't collide with shared keys).
    for k, v in extras.items():
        if k not in metadata:
            metadata[k] = v
    return metadata


# ── METADATA PAYLOAD VALIDATION (the chokepoint guard) ──
# validate_trace_event() checks the (scale, event_type, ref_type) envelope.
# It historically said nothing about the metadata PAYLOAD — which is exactly
# how two writers emitted two different shapes for the same `encoding_run`
# ref_type, undetected, for weeks. This closes that hole: a ref_type with a
# declared schema must carry every required key with the right type.
# Keyed by ref_type (the unit of shape divergence). Covers every delta built by
# build_delta_metadata — the S1 Scribe plus the four S2 units — so a malformed
# payload on any of them is caught, not just encoding_run. (reclassify's
# `community_assignments` is excluded: it only ever writes a bare summary marker,
# with no build_delta_metadata payload to shape-check.)
METADATA_REQUIRED_BY_REF_TYPE = {
    'encoding_run':       DELTA_METADATA_SHAPE,  # S1 Scribe
    'consolidated':       DELTA_METADATA_SHAPE,  # S2 consolidation
    'community_enriched': DELTA_METADATA_SHAPE,  # S2 community
    'healer_generated':   DELTA_METADATA_SHAPE,  # S2 healer
    'aspect_classified':  DELTA_METADATA_SHAPE,  # S2 aspect integration
}


def validate_trace_metadata(event_type, ref_type, metadata):
    """Validate a trace event's metadata payload against its ref_type schema.

    Returns (ok, error_message). Two ways to pass:
      • ref_types without a declared schema (permissive — we only lock shapes
        that have a builder);
      • a bare marker with NO metadata (None) — the delta ref_types double as
        early-out/error markers (`self.trace('delta','consolidated','No clusters
        to process')`), which legitimately carry no payload.
    A PRESENT payload, though, must match the schema. The contract HELPS (catches
    a malformed delta dict) without BLOCKING a no-op marker or dropping anything —
    the caller logs loud and writes the full payload regardless.
    """
    schema = METADATA_REQUIRED_BY_REF_TYPE.get(ref_type or '')
    if not schema:
        return True, ""
    if metadata is None:
        return True, ""   # bare marker — no payload to shape-check
    if not isinstance(metadata, dict):
        return False, "metadata for ref_type '%s' must be a dict or None, got %s" % (
            ref_type, type(metadata).__name__)
    missing = [k for k in schema if k not in metadata]
    if missing:
        return False, "metadata for ref_type '%s' missing required keys: %s" % (
            ref_type, missing)
    bad = [k for k in schema
           if not isinstance(metadata[k], schema[k])]
    if bad:
        return False, "metadata for ref_type '%s' wrong types on keys: %s" % (
            ref_type, bad)
    return True, ""


# ── SELECTION METADATA SHAPE ──
# Decode-style units (S1R) don't have LLM rounds or write actions — they
# select from candidates. Sibling shape keeps them typed correctly and
# gives the dashboard/S3 a second vocabulary to read.

SELECTION_METADATA_SHAPE = {
    'candidates_considered': int,    # how many inputs scored
    'selected':              list,   # IDs/tags of picks
    'dropped':               list,   # IDs/tags of rejects
    'outcomes_per_candidate': dict,  # {candidate_id: 'selected'|'dropped'|...}
    'content':               str,    # the delta output (e.g. additionalContext), truncated
}

SELECTION_CONTENT_LIMIT = 4000


def build_selection_metadata(*,
                             candidates_considered=0, selected=None,
                             dropped=None, outcomes_per_candidate=None,
                             content='', **extras):
    """Build a unified selection-style trace metadata dict (S1R-like)."""
    metadata = {
        'candidates_considered':  int(candidates_considered or 0),
        'selected':               list(selected or []),
        'dropped':                list(dropped or []),
        'outcomes_per_candidate': dict(outcomes_per_candidate or {}),
        'content':                (content or '')[:SELECTION_CONTENT_LIMIT],
    }
    for k, v in extras.items():
        if k not in metadata:
            metadata[k] = v
    return metadata


# ── REVISE METADATA SHAPE ──
# Field-level revise events (event_type='delta', ref_type='node_revised')
# carry per-field deltas + warnings instead of the LLM-loop shape. Used by
# every caller of revise() — direct MCP, S1 encoder, S2 units. Same shape
# whether the caller is dispatch, an encoder, or the operator via MCP.
#
# Warnings carry attempts that didn't land (immutable field passed,
# archive blocked on locked/critical node). The trace event is emitted
# even when deltas is empty as long as warnings is non-empty — so that
# audit history captures attempted-but-rejected operations, not just
# successful changes.

REVISE_METADATA_SHAPE = {
    'node_id':         str,    # which node was revised
    'reason':          str,    # human-readable reason (required at API)
    'encoding_source': str,    # who made the change (anchor, encoder:sonnet, s2:healer, ...)
    'deltas':          list,   # [{'field': str, 'old': any, 'new': any}, ...]
    'warnings':        list,   # ['immutable field skipped: id', 'archive blocked (locked/critical): archived', ...]
}


def build_revise_metadata(*, node_id, reason, encoding_source='',
                          deltas=None, warnings=None):
    """Build trace metadata for a node revise event.

    Caller responsibility: collect (old, new) pairs for each field that
    actually changed, pass them as `deltas`. Pass `warnings` for fields
    that were rejected (immutable, locked-archive). The trace event is
    worth emitting whenever EITHER deltas or warnings is non-empty.

    Used by daemon_dispatch._handle_revise / _handle_revise_batch and any
    direct caller of brain.revise(). Returns a dict ready to pass as the
    metadata kwarg to a trace writer.
    """
    return {
        'node_id':         node_id,
        'reason':          reason or '',
        'encoding_source': encoding_source or '',
        'deltas':          list(deltas or []),
        'warnings':        list(warnings or []),
    }


# ── EDGE REVISE METADATA SHAPE (Stage 1B) ──
# Edge-level revise events (event_type='delta', ref_type='edge_relation_revised')
# carry the same delta+warnings shape as node revises but identified by
# (edge_id, relation) tuple. ref_id encoding: f"{edge_id}:{relation}".
#
# Single ref_type covers both create-via-upsert and update-via-upsert from
# `connect()` / `connect_to`, plus archive via polymorphic `archive` op. Empty
# `old` in a delta means the field was just created; populated `old` = update.
#
# source_id/target_id make the edge SELF-DESCRIBING: the directional pair is in
# the trace itself, so the graph's edges are reconstructable from the trace
# substrate alone — without joining the live edges table to invert edge_id.

EDGE_REVISE_METADATA_SHAPE = {
    'edge_id':         str,    # physical edge id (deterministic from source+target)
    'source_id':       str,    # edge actor (directional — source acts on target)
    'target_id':       str,    # edge acted-upon
    'relation':        str,    # which specific relation on that edge
    'reason':          str,    # human-readable reason (required at API)
    'encoding_source': str,    # who made the change
    'deltas':          list,   # [{'field': str, 'old': any, 'new': any}, ...]
    'warnings':        list,   # any skipped/blocked operations
}


def build_edge_revise_metadata(*, edge_id, relation, reason, encoding_source='',
                               source_id='', target_id='',
                               deltas=None, warnings=None):
    """Build trace metadata for an edge_relation revise event.

    Mirrors build_revise_metadata for nodes; same delta shape captures
    connect-upsert outcomes (empty `old` = create, populated `old` = update)
    and polymorphic archive (deltas show archived flag flipping).

    source_id/target_id carry the directional pair so the edge is
    reconstructable from the trace alone (edge_id is a one-way hash of the
    pair — not invertible without the live edges table).

    Used by daemon_dispatch handlers for `connect`, `connect_batch`,
    `revise_edge`, `disconnect`, the `connect_to` and `co_anchored` paths
    (via _emit_edge_traces), and polymorphic `archive` (when archive targets
    an edge_relation).
    """
    return {
        'edge_id':         edge_id,
        'source_id':       source_id or '',
        'target_id':       target_id or '',
        'relation':        relation,
        'reason':          reason or '',
        'encoding_source': encoding_source or '',
        'deltas':          list(deltas or []),
        'warnings':        list(warnings or []),
    }


def validate_trace_event(scale, event_type, ref_type=""):
    """Validate a trace event against the contract.

    Returns (ok, error_message).
    """
    if scale not in SCALES:
        return False, "Unknown scale '%s'. Valid: %s" % (scale, ', '.join(SCALES.keys()))

    if event_type not in EVENT_TYPES:
        return False, "Unknown event_type '%s'. Valid: %s" % (event_type, ', '.join(EVENT_TYPES.keys()))

    if ref_type:
        key = (scale, event_type)
        if key in REF_TYPES and ref_type not in REF_TYPES[key]:
            return False, "Invalid ref_type '%s' for (%s, %s). Valid: %s" % (
                ref_type, scale, event_type, REF_TYPES[key])

    return True, ""
