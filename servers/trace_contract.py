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
# O/K/Δ — structurally identical at every scale.

EVENT_TYPES = {
    "O": "Observation — everything available at this moment",
    "K": "Knowledge — what was selected as relevant from O",
    "delta": "Changes — what was produced (the response, encoding, reorganization)",
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
                         "node_revised", "edge_relation_revised",
                         "anchor_touched"],   # per-turn aggregate of what Anchor's own
                                              # MCP tools touched this turn (created/revised/
                                              # archived/recalled/endo) — the S0 mirror of the
                                              # S1 encoding_run delta; feeds the encoder's
                                              # widened catalog (trace_links). See
                                              # build_anchor_touched_metadata.

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
                         "encoding_run_failed",     # LLM loop died — no writes; NOT read by
                                                    # trace_links (coverage joins on encoding_run
                                                    # only, so a failed run never claims turns)
                         "node_revised",            # field-level revise emitted by S1 encoder
                         "edge_relation_revised",   # connect upsert / archive emitted by S1 encoder
                         "journal_note"],           # S1 Scribe residue — one note (subject=ref_id) per row

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
                         "edge_relation_revised",   # connect upsert / archive emitted by S2 units
                         "journal_note"],           # S2 unit residue (consolidation, community) — one note per row

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

    # Scale 4: growth integration
    # Fires periodically (weekly). Sees full graph + external sources.
    ("s4", "O"):       ["uncertainty_nodes",   # brain's open questions
                         "external_research"],  # web search results, papers
    ("s4", "K"):       ["stale_decisions",     # decisions that may be outdated
                         "open_questions"],     # unresolved uncertainties
    ("s4", "delta"):   ["research_finding",    # new knowledge from outside
                         "decision_update",     # stale decision refreshed
                         "cross_project"],      # bridge between projects
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

# The "said + did" timeline: conversation plus tool activity. What the S1
# encoder's lived timeline reads and what the embed queue eagerly embeds —
# the two must stay in lockstep (an unembedded timeline row can't anchor
# recall), so the set is defined once here.
SAID_AND_DID_REF_TYPES = CONVERSATIONAL_REF_TYPES + ("tool_result",)


def is_machine_turn(op_text) -> bool:
    """Harness-injected machine turn — a background-task completion packaged
    as a prompt through UserPromptSubmit. NOT an operator turn: the operator
    side is dropped wherever turns feed scoring (the LAF moment stack, the
    walker's labeling — v6's 778-mislabel lesson), while Anchor's response to
    it is real and kept as history. ONE definition shared by production and
    eval (eval/laf/walker/extract.py imports this) so the filter can't drift;
    production's recall hook routes these register_only (node b2953766)."""
    return '<task-notification>' in (op_text or '')

# A wakeup ignite (e.g. a background-task notification) arrives as turn CONTENT,
# not a distinct ref_type: it runs recall, so it's recorded as a `user_message`
# (conversational) even though it's an ENVELOPE, not work. Presence focus skips
# any conversational turn whose summary starts with this marker. One constant so
# the skip is defined ONCE, not reproduced as a scattered SQL literal.
WAKE_ENVELOPE_MARKER = "<task-notification>"


# ── CHAIN ID CONVENTIONS ──
# chain_id groups related O/K/Δ events.
#
# One chain per stop at S0. Everything between stop N-1 and stop N
# (messages, tool calls) belongs to the same S0 chain.
# S1 chains reference the S0 chain via parent_chain in metadata.

CHAIN_PREFIXES = {
    "s0":         "s0-{session_short}-{stop}",        # one chain per stop — messages + tools
    "s1_recall":  "s1r-{session_short}-{stop}",       # surface for this stop
    "s1_encode":  "s1e-{session_short}-{stop}",       # encoding run triggered at this stop
    "s2":         "s2-{datetime}-{operation}",          # datetime=YYYYMMDDHHMMSS (seconds, per-run — see s2/base.py chain_id), operation=community/consolidation/etc
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


# ── AGENT-RUN TELEMETRY (the shared cost+loop field-set) ──
# The cost of producing ONE agent run — an encoder Δ OR a Surface selection.
# Every agent that drives an LLM loop spends the same currency: wall-clock,
# rounds, output truncation, and the four token counts. Defined ONCE here so
# the encoder delta (build_delta_metadata) and the Surface K trace build their
# cost block through the same builder and can never drift into two field-sets.
#
# Kept FLAT on purpose (not a nested sub-object): the dashboard cost lane, the
# loud telemetry guards, and DELTA_METADATA_SHAPE already read these as
# top-level keys, so flat = zero consumer migration. This unifies the
# DEFINITION (one builder) the way runner.USAGE_FIELDS unified the SDK attribute
# NAMES — two different concerns, each single-sourced. (USAGE_FIELDS stays in
# runner.py next to read_usage, the SDK mapper; this is the trace-payload set.)
RUN_TELEMETRY_FIELDS = (
    'elapsed_ms', 'rounds', 'truncated',
    'input_tokens', 'output_tokens',
    'cache_read_tokens', 'cache_creation_tokens',
)


def build_run_telemetry(*, elapsed_ms=0, rounds=0, truncated=0,
                        input_tokens=0, output_tokens=0,
                        cache_read_tokens=0, cache_creation_tokens=0):
    """Build the shared agent-run cost block (a flat dict of RUN_TELEMETRY_FIELDS).

    Used by build_delta_metadata (encoders) and the Surface K-trace writer.
    All int, default 0 — `truncated` is a count of rounds cut at max_tokens,
    `rounds` the number of LLM calls, the rest wall-clock + token spend. Spread
    flat into the surrounding metadata dict; never nest it.
    """
    return {
        'elapsed_ms':            int(elapsed_ms or 0),
        'rounds':                int(rounds or 0),
        'truncated':             int(truncated or 0),
        'input_tokens':          int(input_tokens or 0),
        'output_tokens':         int(output_tokens or 0),
        'cache_read_tokens':     int(cache_read_tokens or 0),
        'cache_creation_tokens': int(cache_creation_tokens or 0),
    }


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
        # 'rounds' is emitted by build_run_telemetry below (shared cost block).
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
        # Shared cost+loop block (elapsed_ms/rounds/truncated + token counts) —
        # one builder so the encoder Δ and the Surface K trace can't drift.
        **build_run_telemetry(
            elapsed_ms=elapsed_ms, rounds=rounds, truncated=truncated,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens),
        'interaction_version':   int(interaction_version or 0),
    }
    # Extras preserved for per-unit fields (can't collide with shared keys).
    for k, v in extras.items():
        if k not in metadata:
            metadata[k] = v
    return metadata


# ── ANCHOR-TOUCHED METADATA SHAPE ──
# The S0 mirror of the S1 encoding_run delta: one per-turn aggregate of the
# nodes Anchor's OWN MCP tools touched this turn. Deliberately reuses the encode
# delta's `created`/`revised`/`archived` keys so trace_links reads BOTH the S1
# encode delta and this S0 delta through ONE parser (no second parsing path) —
# plus read-side keys S1 has no analog for (`recalled` = deliberate get_node(s);
# `endo` = endo-surface ids, empty until endo lands). NODE ids only (full),
# never edges or the encoding_source field (technical, encoder-invisible).
ANCHOR_TOUCHED_KEYS = ('created', 'revised', 'archived', 'recalled', 'endo')
# Shape for validate_trace_metadata — every key an id-list. Derived from the
# constant (not a second hand-written list) so the keyset can't drift.
ANCHOR_TOUCHED_SHAPE = {k: list for k in ANCHOR_TOUCHED_KEYS}


def build_anchor_touched_metadata(**ids):
    """Build the anchor_touched delta metadata — a flat dict of id-lists, ONE per
    ANCHOR_TOUCHED_KEYS (the constant drives the keyset; pass any subset). Order-
    preserving dedup per key; never the raw encoding_source. Reuses the encode
    delta's `created`/`revised`/`archived` key names so one reader serves both.
    Unknown kwargs are ignored (the flush passes `**ctx.touched`, which is itself
    keyed by ANCHOR_TOUCHED_KEYS — so they always agree)."""
    return {k: list(dict.fromkeys(ids.get(k) or [])) for k in ANCHOR_TOUCHED_KEYS}


# ── JOURNAL NOTE METADATA SHAPE ──
# A journal note is a Δ written as its OWN trace event (event_type='delta',
# ref_type='journal_note'): the residue of a run's integrate() — the why, the
# friction, the doubt, the surprise. SEPARATE from the run's objective ops-delta
# (encoding_run / consolidated / ...): the ops-delta records what the hands did,
# the note records what the mind did. Per run = 1 ops-delta + 0..N notes, ALL
# sharing the run's chain_id — which is per-run-unique at BOTH scales (S1
# `s1e-{session}-{stop}`, S2 `s2-{YYYYMMDDHHMMSS}-{unit}`), so the read groups
# runs by chain_id with no separate run_id field.
#
# The SUBJECT lives in the trace's ref_id, NOT here — a node id / cluster / tool /
# input the note is about. It's the load-bearing index (N notes on one ref_id =
# a hotspot) and the quality gate (can't name a subject → not a note). Notes are
# s1/s2 scale → never embedded (EAGER_TRACE_SCALES=('s0',)) → unreachable by
# recall()/recall_episodes(); the only door is the traces-module notes() query.

JOURNAL_NOTE_METADATA_SHAPE = {
    'note': str,    # the prose: the why / friction / doubt / surprise (required)
    'tag':  str,    # one open word for the KIND of thing (friction, doubt, ...); '' when absent
}

JOURNAL_NOTE_LIMIT = 600   # a note is terse residue, not an essay — capped loud like other delta text
JOURNAL_TAG_LIMIT = 40     # 'one word' — cap drift loud rather than let a sentence become a grouping key


def build_journal_note_metadata(*, note, tag=''):
    """Build trace metadata for one journal note (ref_type='journal_note').

    The SUBJECT is the trace's ref_id, supplied by the writer — not here.
    `note` is the prose (required, non-empty — the same gate the parser
    applies); `tag` is one open word, '' when the encoder gave none. Both are
    capped loud via cap_text_loud, like every other delta text field. Raises
    ValueError on an empty note so the builder and parser AGREE on validity —
    the write path only feeds parser-validated notes and isolates per-note
    write errors, so this fires solely on direct misuse, never normal flow.
    """
    note = (note or '').strip()
    if not note:
        raise ValueError('journal note requires non-empty prose (got empty note)')
    return {
        'note': cap_text_loud(note, JOURNAL_NOTE_LIMIT),
        'tag':  cap_text_loud((tag or '').strip(), JOURNAL_TAG_LIMIT),
    }


# ── JOURNAL REVIEW BLOCK + PARSER (single source for all journaling encoders) ──
# §7.1/§7.3: one shared instruction block (roles-free — a bar to clear, not
# buckets to fill); each encoder appends ONLY its own examples + subject
# vocabulary via render_journal_review_block(). The encoder emits one note per
# line as `tag · subject · note`; the write path calls parse_journal_notes() to
# split them into rows. Single source here so the five encoders can't re-diverge
# into the five reinventions this redesign is removing.

JOURNAL_NOTE_DELIMITER = '·'

# ── Journal lifecycle verbs (2026-07-28, audit finding #6) ──
# Read-time only — traces stay append-only. `resolved`/`retire` drops older
# same-subject notes from the continuity prefix; `open` pins the newest note
# per subject beyond the K-run window until resolved. Matching is normalized
# (casefold+strip) EXACT subject equality — the corpus showed paraphrase
# references never match, so the instruction teaches exact-copy.
JOURNAL_RESOLVE_TAGS = ('resolved', 'retire')
JOURNAL_OPEN_TAGS = ('open', 'still-open')   # still-open: pre-existing wild alias
JOURNAL_OPEN_PIN_CAP = 10        # max pinned subjects carried beyond the window
JOURNAL_OPEN_NUDGE_RUNS = 5      # open ×N at/past this → render the promote nudge
# The escalation type is boot-visible: render_standing_items (frame.py) injects
# all live nodes of the types in BRAIN_BOOT_INJECT_TYPES at session boot.
JOURNAL_ESCALATION_TYPE = 'journals-escalation'

# Self-grounding by design (no `brain`/`trace`/`operator`/agent-verb/identity
# tokens): the block means the same dropped into any host prompt or standing
# alone, so a host-prompt edit can't silently shift the journal, and the block
# is testable in isolation. EAGER by intent (2026-06-23, Tom): no value-filter
# gate — capture residue freely; dedupe/mine later. The earlier "two tests"
# (reconstruction/successor) were removed as over-correction against the OLD
# journal's restatement disease, not an evidenced need. Iterate from LIVE
# results, not synthetic probes (which can't reproduce the encoder's lived run).
JOURNAL_REVIEW_INSTRUCTION = (
    "A review — a short note to the next run of this work, about anything "
    "noticed here that won't be visible in the actions taken.\n"
    "The changes made are already recorded automatically; don't restate them. "
    "This note is only for what the actions don't capture — a doubt, a "
    "friction, a surprise, a pattern forming.\n\n"
    "`tag` — one word for the kind of thing (friction, doubt, surprise, "
    "dead-end — examples, not a list).\n"
    "`subject` — what the note is about: the specific thing touched (its id), "
    "a tool or input handed in, or the run itself.\n\n"
    "To clear a handled note, write `resolved %s <its exact subject> %s why` "
    "— one line per subject.\n"
    "Mark a persisting item once: `open %s subject %s note` — it stays "
    "visible until resolved; don't re-assert it each run.\n\n"
    "Put the notes under a `## Review` heading, inside a fenced code block — "
    "one note per line as `tag %s subject %s note`. A clean run is an empty "
    "fence — leave it empty rather than saying there's nothing to note.\n\n"
    "Time is precious — actions are already logged automatically; no need "
    "to rephrase. Stay sharp."
) % ((JOURNAL_NOTE_DELIMITER,) * 6)


def render_journal_review_block(examples=''):
    """The shared review block — self-contained (output structure + close folded
    in), identical for every encoder.

    `examples` is optional and ships empty by default: a positive example
    anchors *what to notice*, which for residue we deliberately leave open. A
    per-encoder caller may pass its own `tag · subject · note` examples later if
    a unit proves to need them, appended as a fenced block.
    """
    block = JOURNAL_REVIEW_INSTRUCTION
    if examples and examples.strip():
        block += "\n\n```\n" + examples.strip() + "\n```\n"
    return block


# The arc — the SECOND closing act (§7.2: Encode → Arc → Review), a journal-
# mechanism component distinct from the review: the review is residue notes
# (traces, per-note rows); the arc is ONE line of session orientation
# (accumulated onto a running per-session digest that downstream readers rank
# and orient against). Never merged into the review — different shape,
# different reader. Self-grounding like the review block (no host-coupled
# tokens); placement is stated HERE, not in the closure, so the closure stays
# shared with encoders that never emit an arc. Per-encoder opt-in: injected
# only by encoders that write a session arc (S1 Scribe today).
JOURNAL_ARC_INSTRUCTION = (
    "The arc — ONE line: what progressed in this stretch of work, this run.\n"
    "It accumulates onto a running digest of the whole conversation, so write "
    "only the new movement — never a recap of what the digest already says.\n\n"
    "Put it under a `## Arc` heading, inside a fenced code block — a single "
    "line, on the same final reply as the review, just before it. If nothing "
    "meaningfully progressed, leave the fence empty.\n\n"
    "Example: `judge reliability crisis found — 85% timeout rate`"
)


def render_journal_arc_block():
    """The shared arc block — the write-side instruction for the session arc,
    identical for every encoder that opts in. Single-sourced here (never baked
    into a registered prompt) for the same reason as the review block: it
    iterates in one place and every opted-in encoder gets it live.
    """
    return JOURNAL_ARC_INSTRUCTION


def render_prompt_closure():
    """The run's CLOSURE — separate concern from the review block. Defines the
    terminal turn the way the runner does (a reply with no tool call IS the
    final one), places the review on it whether the encoder acted or not, and
    carries the `DONE` stop signal. Injected as the LAST block of the prompt,
    independent of the review block — so removing or relocating the review never
    drags the closure with it. References the `## Review` artifact by name; it
    does NOT define it (that's render_journal_review_block's job).

    The no-tool-call branch is the fix for the no-action batch: an all-reject /
    nothing-to-change reply terminates the loop on its first turn, and that turn
    must still carry the review (an empty fence on a clean run).
    """
    return (
        "## Finishing\n\n"
        "The run is done when a reply makes no tool call — that final reply is the "
        "only place the review goes. Two ways to get there, both ending the same:\n"
        "- After tool calls: once the results come back, the next reply is the final one.\n"
        "- A reply with no tool call at all (nothing needed changing): that reply is "
        "already the final one.\n\n"
        'End the final reply with the `## Review`, then write "DONE".'
    )


def render_journal_notes_prefix(notes, label='RECENT REVIEW NOTES'):
    """Render journal_notes() output into a prompt prefix — the READ side of
    the journal (residue continuity). Shared single source so every encoder
    (S2 units now, S1E later) feeds continuity the same way.

    `notes` is the list of {tag, subject, note, ...} dicts journal_notes()
    returns (newest first, already bounded to the last K note-bearing runs).
    Returns '' when there are none, so a clean history adds nothing to the
    prompt — no "first run, no notes" filler. Each line mirrors the write
    format `tag · subject · note`.
    """
    if not notes:
        return ''
    lines = ['%s — residue your recent runs flagged, for continuity (not a '
             'to-do list):' % label]
    for n in notes:
        tag = (n.get('tag') or '').strip()
        head = ('%s · ' % tag) if tag else ''
        line = '- %s%s · %s' % (head, n.get('subject', ''), n.get('note', ''))
        # Open items render their persistence: the loader computed ×N (distinct
        # runs mentioning the subject) and pins the newest note beyond the
        # window. Past the threshold, the nudge appears ON the item, in the run
        # that should act — zero standing prompt cost.
        runs = n.get('open_runs') or 0
        if runs:
            since = (n.get('first_seen') or '')[5:10]
            line = '- %s ×%d%s · %s · %s' % (
                tag or 'open', runs,
                (' since %s' % since) if since else '',
                n.get('subject', ''), n.get('note', ''))
            if runs >= JOURNAL_OPEN_NUDGE_RUNS:
                # Tool-neutral phrasing: encoders write nodes through different
                # doors (brain_batch remember op, remember_batch) — name the
                # node type, not a tool signature.
                line += (
                    "\n  ⚠ long-lived — resolve it, or promote it out of the "
                    "journal: create a `%s`-type node carrying it, then write "
                    "`resolved · %s · promoted to <id>`"
                    % (JOURNAL_ESCALATION_TYPE, n.get('subject', '')))
        lines.append(line)
    return '\n'.join(lines) + '\n\n'


def parse_journal_notes(text):
    """Parse an encoder's review section into notes.

    One note per line: `tag · subject · note`, split on '·' with maxsplit=2 so
    a '·' inside the prose is safe (it all stays in `note`).
      • 3 fields → (tag, subject, note)
      • 2 fields → (subject, note) with tag='' — tag is optional
      • no delimiter, or empty subject/note → MALFORMED
    Blank lines and markdown headers (`#`-prefixed, e.g. the `## Review`
    header) are skipped silently. Malformed lines are NOT silently dropped —
    they're returned so the caller logs loud (loud-by-default).

    Returns (notes, malformed): notes is a list of {'tag','subject','note'};
    malformed is a list of the raw offending lines.
    """
    notes, malformed = [], []
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0] in '-*•':       # tolerate a leading markdown bullet (LLMs
            line = line[1:].lstrip()   # list-format their review); keep the tag clean
            if not line:
                continue
        if JOURNAL_NOTE_DELIMITER not in line:
            # No delimiter: a markdown header (e.g. the `## Review` title) is
            # structural — skip silently. Anything else is a malformed note,
            # surfaced loud (never silently dropped). A delimiter-bearing line
            # is ALWAYS a note candidate even if it starts with '#', so a
            # subject like a `#1234` issue id isn't eaten by the header skip.
            if line.startswith('#'):
                continue
            malformed.append(raw)
            continue
        parts = [p.strip() for p in line.split(JOURNAL_NOTE_DELIMITER, 2)]
        if len(parts) == 3:
            tag, subject, note = parts
        else:  # delimiter present + maxsplit=2 → exactly 2 parts here
            tag, subject, note = '', parts[0], parts[1]
        if not subject or not note:
            malformed.append(raw)
            continue
        notes.append({'tag': tag, 'subject': subject, 'note': note})
    return notes, malformed


JOURNAL_REVIEW_MARKER = '## Review'   # the section heading the encoder emits; the
                                      # write path keys on it. Kept in sync with the
                                      # prompt structure (§7.2) — #8 wires the prompt.

JOURNAL_ARC_MARKER = '## Arc'         # the arc heading (§7.2: "Arc — ONE line: what
                                      # progressed this run"). Its write path
                                      # (write_session_arc) keys on it. A journal-
                                      # mechanism component, per-encoder opt-in —
                                      # S1 Scribe today; any S2 unit later.


def _fenced_section_scan(text, marker):
    """The ONE scanner for a journal section — finds the `marker` heading and
    its own fenced ``` block, returning `(section_start, section_end, content)`
    with the section span as [start, end) character offsets and `content` the
    fence body (language tag skipped, stripped). None when the section is
    absent or its fence is malformed. Both the extractors and the strip below
    ride this, so the section-ownership rules can't drift apart.
    """
    if not text:
        return None
    idx = text.find(marker)
    if idx == -1:
        return None
    after = text[idx + len(marker):]
    open_fence = after.find('```')
    if open_fence == -1:
        return None
    # The fence must belong to THIS section. If a new `## ` heading starts
    # before the opening fence, this section has no fence of its own and the
    # fence we found belongs to a LATER section — return None (drift) rather
    # than capturing the wrong section's content. Without this, a fenceless
    # `## Arc` reaches forward into the `## Review` fence (§7.2 orders Arc
    # before Review) and review notes get written as the session arc, silently.
    # Checking position (heading-before-fence) — not blunt truncation — leaves
    # legit fence content that itself contains a `## ` line intact.
    next_heading = after.find('\n## ')
    if next_heading != -1 and next_heading < open_fence:
        return None
    rest = after[open_fence + 3:]
    nl = rest.find('\n')           # skip an optional language tag on the fence line
    if nl != -1:
        rest = rest[nl + 1:]
    close_fence = rest.find('```')
    if close_fence == -1:
        return None
    content_end_in_after = (open_fence + 3
                            + (nl + 1 if nl != -1 else 0)
                            + close_fence + 3)
    return (idx, idx + len(marker) + content_end_in_after,
            rest[:close_fence].strip())


def _extract_fenced_block(text, marker):
    """Pull a fenced block out of an encoder's final text: find the `marker`
    section heading and return the content of its first fenced ``` block.

    Three-valued so writers can tell the cases apart and stay loud:
      • **None** — no `marker` section, or a marker with no parseable fence
        (missing open/close fence). The caller distinguishes "no section" from
        "format drift" by re-checking `marker in text`.
      • **''** — a fenced block that's empty (a legit clean run), distinct
        from drift.
      • **str** — the fence content.
    Extracting ONLY the fence (not the whole section) keeps surrounding prose
    from being mis-parsed as content.
    """
    scan = _fenced_section_scan(text, marker)
    return scan[2] if scan else None


def strip_journal_sections(text):
    """Remove the journal sections (`## Arc` / `## Review` heading + fenced
    block) from an encoder's final text, returning the payload remainder.

    This is harvest's envelope rule for single-shot agents: their response
    carries a JSON payload AND the journal fence in one text, and
    `extract_json`'s rfind-based scan would be corrupted by a `]`/`}` inside
    a fence that follows the payload — so the journal is stripped first.
    Sections that are absent or malformed (drift) are left untouched.
    """
    if not text:
        return text
    stripped_any = False
    for marker in (JOURNAL_ARC_MARKER, JOURNAL_REVIEW_MARKER):
        scan = _fenced_section_scan(text, marker)
        if scan:
            text = text[:scan[0]] + text[scan[1]:]
            stripped_any = True
    # Only tidy whitespace when a section was actually removed — a text with
    # no journal sections passes through byte-identical.
    return text.strip() if stripped_any else text


def extract_review_block(text):
    """The `## Review` fence — content ready for `parse_journal_notes`.
    Three-valued; see `_extract_fenced_block`."""
    return _extract_fenced_block(text, JOURNAL_REVIEW_MARKER)


def extract_arc_block(text):
    """The `## Arc` fence — the run's one-line arc delta, ready for
    `write_session_arc`. Three-valued; see `_extract_fenced_block`."""
    return _extract_fenced_block(text, JOURNAL_ARC_MARKER)


def salvage_review_fence(text):
    """Drift salvage for the write door: notes the encoder fenced WITHOUT the
    `## Review` heading. Observed on Haiku community runs — a perfectly formed
    notes fence loses its heading and the whole batch's residue was dropped on
    the strict marker match.

    Strict all-or-nothing gate, so a code/table fence can never be harvested:
    a fence qualifies only when `parse_journal_notes` accepts EVERY non-blank
    line (>=1 note, zero malformed). Well-formed journal sections are stripped
    first so an `## Arc` fence is never mistaken for notes. Multiple qualifying
    fences → the LAST one (the closure puts the review at the end of the final
    reply). Returns the fence content, or None when nothing qualifies — an
    empty heading-less fence does NOT qualify (indistinguishable from a stray
    code block, unlike a fenced `## Review` where empty means a clean run).
    """
    remainder = strip_journal_sections(text)
    if not remainder:
        return None
    salvaged = None
    pos = 0
    while True:
        open_fence = remainder.find('```', pos)
        if open_fence == -1:
            break
        rest = remainder[open_fence + 3:]
        nl = rest.find('\n')          # skip an optional language tag
        if nl == -1:
            break
        body = rest[nl + 1:]
        close_fence = body.find('```')
        if close_fence == -1:
            break
        pos = open_fence + 3 + nl + 1 + close_fence + 3
        content = body[:close_fence].strip()
        if not content:
            continue
        notes, malformed = parse_journal_notes(content)
        if notes and not malformed:
            salvaged = content
    return salvaged


# Per-encoder continuity window: how many of an encoder's most recent
# note-bearing runs the "where things stand" read pulls into the next run's
# prompt. Bounds the READ, never storage — notes are append-only and retained
# (§2.7); a 9th run simply doesn't read the 1st's note, which still exists for
# the operator + future miner. A contract constant, NOT interaction-tunable
# (Tom's call): continuity depth is a structural property of each encoder's
# cadence, not a knob S2 should self-tune. Keys are the encoder identity used by
# notes() (S1 chain prefix `s1e`; S2 unit NAME). Unlisted encoders use DEFAULT.
JOURNAL_CONTINUITY_RUNS = {
    's1e':                 5,   # S1 Scribe — every 5th Stop; a session spans several runs
    'consolidation':       3,   # S2 idle units run far apart; 3 is enough for cross-run escalation
    'community_detection': 3,
}
JOURNAL_CONTINUITY_RUNS_DEFAULT = 3


# Residue ref_types: encoder *notes*, not integration deltas. Consumers that
# read per-run integration deltas (S2 idle-gating `_last_run_timestamp`, the
# dashboard run-card queries) must EXCLUDE these — a journal_note shares the
# run's chain_id + event_type='delta', so an unfiltered `event_type='delta'`
# pull would otherwise scoop notes and miscount them as runs. Single source for
# the ops-delta-vs-residue partition; exclusion-style so it stays
# behavior-preserving (everything that isn't residue still counts) and
# forward-compatible (add a residue type here, every consumer excludes it).
RESIDUE_REF_TYPES = ('journal_note',)


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
    'journal_note':       JOURNAL_NOTE_METADATA_SHAPE,  # encoder residue (one note per row)
    'anchor_touched':     ANCHOR_TOUCHED_SHAPE,  # S0 per-turn Anchor action aggregate
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


# ── LLM-ENCODER TELEMETRY GUARD (loud at the write boundary) ──
# A delta produced by an agent that actually called an LLM MUST carry the
# cost/latency telemetry build_delta_metadata accepts (elapsed_ms + token
# counts). These are the ref_types of those deltas — one per LLM encoder.
# (Selection deltas, node/edge_revised, and bare early-out markers are NOT
# here: they have no LLM round to measure.)
LLM_ENCODER_DELTA_REF_TYPES = (
    'encoding_run',        # S1 Scribe
    'consolidated',        # S2 consolidation
    'community_enriched',  # S2 community
    'healer_generated',    # S2 healer
    'aspect_classified',   # S2 aspect integration
)


def check_delta_telemetry(ref_type, metadata):
    """Detect an LLM-encoder delta that ran the model AND did work, yet
    recorded output_tokens==0 — the silent telemetry-threading gap where an
    encoder built its delta without passing run_llm_loop's / the API response's
    token counts to build_delta_metadata (the 2026-06-24 fleet-wide S2 gap).

    Returns a one-line warning string for the caller to log via
    brain._log_error / _log_warning, or None when there's nothing to flag. Pure
    — no logger here (this and build_delta_metadata are contract functions; the
    WRITE boundary owns logging, per "loud at the write boundary"; TraceDAL, the
    other chokepoint, can't reach the errors table mid-append).

    Returns None (no flag), by design, for:
      • non-LLM-encoder ref_types (selection deltas, node/edge_revised, markers);
      • bare markers / no payload (metadata None or not a dict — the early-out
        "No clusters to process" traces);
      • no-work runs (actions==0). actions>0 — not rounds>0 alone — is the
        load-bearing guard. The model can't emit a tool call or a parsed JSON
        result without spending output tokens, so actions>0 with
        output_tokens==0 is an UNAMBIGUOUS wiring gap. Gating on actions>0 also
        excludes the all-LLM-calls-failed case (e.g. healer, whose `rounds`
        counts batches ATTEMPTED — a run where every call raised has rounds>0
        but actions==0; that's an LLM failure, already logged, not a telemetry
        gap, so it must not cry wolf here).
    """
    if ref_type not in LLM_ENCODER_DELTA_REF_TYPES:
        return None
    if not isinstance(metadata, dict):
        return None
    rounds = metadata.get('rounds') or 0
    actions = metadata.get('actions') or 0
    output_tokens = metadata.get('output_tokens') or 0
    if rounds > 0 and actions > 0 and output_tokens == 0:
        return ('%s delta ran %d round(s) with %d action(s) but recorded '
                'output_tokens=0 — LLM telemetry not threaded into '
                'build_delta_metadata' % (ref_type, rounds, actions))
    return None


# Surface (the S1 decoder) is the one LLM agent that is NOT a delta encoder: it
# spends tokens selecting from candidates and writes its cost into the K trace
# (ref_type 'surface_selected'), not a delta. Same silent-regression risk as the
# encoders had pre-2026-06-24 — this is its guard. Unlike the encoder case there
# is no `actions` gate: Haiku ALWAYS emits output (the selection JSON, even an
# empty {"selected":[]}), so rounds>0 with output_tokens==0 is an unambiguous
# wiring gap on its own.
SURFACE_TELEMETRY_REF_TYPE = 'surface_selected'


def check_surface_telemetry(metadata):
    """Detect a Surface K trace that ran Haiku yet recorded output_tokens==0 —
    the surface-side analog of check_delta_telemetry (the cost telemetry was
    not threaded from read_usage into build_run_telemetry into the K trace).

    Returns a one-line warning string for the caller to log via
    brain._log_error, or None when there's nothing to flag. Pure — the write
    boundary owns logging (same contract as check_delta_telemetry).
    """
    if not isinstance(metadata, dict):
        return None
    rounds = metadata.get('rounds') or 0
    output_tokens = metadata.get('output_tokens') or 0
    if rounds > 0 and output_tokens == 0:
        return ('surface_selected K trace ran %d Haiku round(s) but recorded '
                'output_tokens=0 — surface cost telemetry not threaded into '
                'the K trace metadata' % rounds)
    return None


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


# ── TRACE RENDERING ──
# Mirrors contract.py's node-render layer. brain.query_traces / get_trace /
# get_traces return full rows (the data layer — S2 units read them
# programmatically); the MCP layer renders bounded text via render_trace +
# these configs, never a raw json.dumps. recall_episodes shares this renderer.
#
# The heavy field is `metadata` — s2 K/delta rows reach ~140KB. Bounding it is
# the lever here, exactly as the edge tail was for get_nodes. `rich=true` opts
# into the full row.

TRACE_BODY_CHARS = 280          # body cap (was brain_constants.EPISODE_RENDER_BODY_CHARS)
TRACE_BULK_BODY_CHARS = 200     # tighter body cap for bulk pulls (>TRACE_BULK_MAX rows)
TRACE_GIST_VALUE_CHARS = 80     # per-key value cap in gist metadata
TRACE_GIST_MAX_KEYS = 8         # keys shown in gist before "+N more"
TRACE_BULK_MAX = 20             # above this many rows, default drops to summary-only

# Default for a focused pull (get_trace, small query): body + metadata gist
# (key=value, big values elided to "<N chars>" so a 140KB blob can't leak).
TRACE_COMPACT_FORMAT = {'body_limit': TRACE_BODY_CHARS, 'metadata_mode': 'gist',
                        'show_scale': True}
# Many rows (large query_traces/get_traces): summary only, no metadata.
TRACE_BULK_FORMAT = {'body_limit': TRACE_BULK_BODY_CHARS, 'metadata_mode': 'none', 'show_scale': True}
# rich=true opt-in: the complete row — full body + full metadata.
TRACE_FULL_FORMAT = {'body_limit': None, 'metadata_mode': 'full', 'show_scale': True}
# recall_episodes (conversational): matches its historic render — body only, no
# scale/event_type chrome (it's always s0 conversation).
TRACE_EPISODE_FORMAT = {'body_limit': TRACE_BODY_CHARS, 'metadata_mode': 'none',
                        'show_scale': False}


def _render_trace_metadata(meta, mode):
    """Render a trace row's metadata dict. 'gist' = key=value with big values
    elided to "<N chars>" (kills the blob, keeps the shape); 'full' = complete.
    'content' is rendered as the body, never repeated here."""
    items = [(k, v) for k, v in meta.items()
             if k != 'content' and v not in (None, '', [], {})]
    if not items:
        return []
    if mode == 'full':
        import json as _json
        out = ['  metadata:']
        for k, v in items:
            sval = v if isinstance(v, str) else _json.dumps(v, default=str)
            out.append('    %s: %s' % (k, sval))
        return out
    # gist
    bits = []
    for k, v in items[:TRACE_GIST_MAX_KEYS]:
        sval = v if isinstance(v, str) else str(v)
        bits.append('%s=<%d chars>' % (k, len(sval)) if len(sval) > TRACE_GIST_VALUE_CHARS
                    else '%s=%s' % (k, sval))
    line = '  ' + '  '.join(bits)
    if len(items) > TRACE_GIST_MAX_KEYS:
        line += '  +%d more' % (len(items) - TRACE_GIST_MAX_KEYS)
    return [line]


def render_trace(row, config=None):
    """Render one trace_event row to text — the single trace renderer.

    The MCP trace tools (query_traces / get_traces / get_trace) and
    recall_episodes all route through here, mirroring how render_rich_node is
    the one node renderer. Body source: metadata['content'] (conversational
    episodes) falls back to summary (structural traces). `metadata` is bounded
    per config.metadata_mode.
    """
    from servers.contract import _truncate
    cfg = {**TRACE_COMPACT_FORMAT, **(config or {})}
    meta = row.get('metadata')
    if not isinstance(meta, dict):
        meta = {}

    sid = (row.get('session_id') or '')[:8]
    score = row.get('_score')
    score_str = ' %.2f' % score if isinstance(score, (int, float)) else ''
    ref_type = row.get('ref_type') or ''
    if ref_type == 'assistant_message':
        label = meta.get('agent_identity') or 'Anchor'
    elif ref_type == 'user_message':
        label = meta.get('human_identity') or 'Operator'
    elif ref_type == 'tool_result':
        label = meta.get('tool') or 'tool_result'
    else:
        label = ref_type or '?'
    when = (row.get('created_at') or '')[:16].replace('T', ' ')

    # Middle segments: [scale event_type] then ref_type (unless it IS the label)
    mids = []
    if cfg.get('show_scale'):
        mids.append('%s %s' % (row.get('scale') or '?', row.get('event_type') or '?'))
    if ref_type and ref_type != label:
        mids.append(ref_type)
    mid_str = (' · ' + ' '.join(mids)) if mids else ''
    # The leading [sid score] bracket is omitted entirely when a trace has
    # neither (session-less S2 system traces, grouped events) — no empty "[]".
    inner = (sid + score_str).strip()
    bracket = '[%s] ' % inner if inner else ''
    tid = row.get('id') or ''
    tid_str = ' (trace:%s)' % tid if tid else ''
    header = '%s%s · %s%s%s' % (bracket, label, when, mid_str, tid_str)

    lines = [header]
    body = (meta.get('content') or row.get('summary') or '').strip()
    if body:
        blim = cfg.get('body_limit')
        body = _truncate(body, blim) if blim else body
        lines.append('  ' + body.replace('\n', '\n  '))
    if cfg.get('metadata_mode', 'gist') != 'none':
        lines.extend(_render_trace_metadata(meta, cfg['metadata_mode']))
    return '\n'.join(lines)
