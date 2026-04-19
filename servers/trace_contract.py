"""Trace Contract — single source of truth for the fractal trace system.

Defines: scales, event types, ref types, and validation.
All trace writers MUST validate against this contract.
All trace readers can rely on these guarantees.

Architecture: docs/ARCHITECTURE-FRACTAL.md
"""


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
    ("s0", "K"):       ["user_message"],
    ("s0", "delta"):   ["assistant_message", "tool_result"],
    ("s0", "outcome"): ["correction", "follow_up"],

    # Scale 1: turn integration
    # Surface path (chain prefix: s1r-): O=candidates, K=surfaced picks, delta=context sent to Anchor
    # Encode path (chain prefix: s1e-): O=prompt given, K=node catalog, delta=actions+reasoning
    ("s1", "O"):       ["recall",            # candidates with scores
                         "encoding_prompt"],   # what the encoder was given
    ("s1", "K"):       ["surface_selected",  # what the surfacer picked
                         "node_catalog"],      # which nodes available to encoder
    ("s1", "delta"):   ["additionalContext",  # what reached Anchor
                         "encoding_run"],      # what the encoder produced
    ("s1", "outcome"): ["correction",         # Tom corrected something that was recalled
                         "recall_hit"],        # node was recalled in a future turn

    # Scale 2: graph integration
    # Fires during idle hook. Operates on S1's accumulated output (the graph).
    # Multiple integration units, each with own O/K/Δ.
    ("s2", "O"):       ["graph_structure",      # nodes + edges observed (community detection)
                         "graph_stats",          # node/edge counts, density
                         "s1_delta",             # S1 encoding/surfacing traces since last run
                         "consolidation_candidates",  # embedding scan + behavioral evidence
                         "correction_chains",    # brain-wide correction chain traversal
                         "healer_scan"],         # S2 Healer: gaps + flags scanned
    ("s2", "K"):       ["community_proposals",  # S2CD proposals (placements, overlaps, splits, seeds)
                         "community_partition",  # algorithm output (communities + membership)
                         "community_diff",       # comparison with previous run
                         "consolidation_proposals",   # enriched clusters with pre-classification
                         "stale_nodes",          # nodes not accessed recently
                         "healer_proposals"],    # S2 Healer: nodes to heal (fill missing fields)
    ("s2", "delta"):   ["community_enriched",   # S2CE enrichment results (accepted, rejected, placed)
                         "community_created",    # new community node
                         "community_updated",    # revised community node
                         "community_removed",    # stale community archived
                         "community_assignments",# membership edges updated
                         "recall_quality_signal",# recall diagnostic (false positive, redundancy, gap)
                         "consolidated",         # new node from smart merge
                         "evolved",              # evolution edge added
                         "kept_distinct",        # similar_to edge, no merge
                         "confidence_adjust",    # adjusted confidence scores
                         "healer_generated"],    # S2 Healer: missing fields generated + stored
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
}

DELTA_FINAL_TEXT_LIMIT = 2000
DELTA_ERROR_LIST_LIMIT = 5


def build_delta_metadata(*,
                         actions=0, write_actions=0, rounds=0,
                         inputs_processed=0, outcomes=None,
                         rejection_skipped=0, journal_entry='',
                         action_details=None, final_text='',
                         errors=None, **extras):
    """Build a unified delta trace metadata dict.

    All agentic encoders (S1E, S2 units) should call this to build the
    metadata payload for their `delta` trace event. Standardizes field
    names, applies truncation, and lets each unit pass additional keys
    via **extras (e.g. clusters_processed, batches).

    Returns a dict ready to pass as the metadata kwarg to a trace writer.
    """
    metadata = {
        'actions':           int(actions or 0),
        'write_actions':     int(write_actions or 0),
        'rounds':            int(rounds or 0),
        'inputs_processed':  int(inputs_processed or 0),
        'outcomes':          dict(outcomes or {}),
        'rejection_skipped': int(rejection_skipped or 0),
        'journal_entry':     (journal_entry or '')[:DELTA_FINAL_TEXT_LIMIT],
        'action_details':    list(action_details or []),
        'final_text':        (final_text or '')[:DELTA_FINAL_TEXT_LIMIT],
        'errors':            list(errors or [])[:DELTA_ERROR_LIST_LIMIT],
    }
    # Extras preserved for per-unit fields (can't collide with shared keys).
    for k, v in extras.items():
        if k not in metadata:
            metadata[k] = v
    return metadata


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
