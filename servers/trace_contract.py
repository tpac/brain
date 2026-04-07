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
        "name": "Session",
        "description": "Patterns across accumulated turns — journey, not moments",
        "triggers": "Every ~15 stops",
        "status": "NOT BUILT",
    },
    "s3": {
        "name": "Sleep",
        "description": "Graph-wide operations — communities, bridges, dedup, corrections",
        "triggers": "Between sessions / idle",
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

    # Scale 2: session integration
    # Fires every ~15 stops. Sees all S0+S1 traces from this session.
    ("s2", "O"):       ["session_turns",       # accumulated S0+S1 chains as observation
                         "session_patterns"],    # detected patterns across turns
    ("s2", "K"):       ["session_nodes",       # all nodes touched this session
                         "correction_chains"],   # corrections accumulated this session
    ("s2", "delta"):   ["journey_arc",         # session narrative encoding
                         "consolidation"],       # merged or revised shallow encodings
    ("s2", "outcome"): ["cross_session",       # pattern validated in a future session
                         "correction"],          # session encoding was later corrected

    # Scale 3: sleep integration
    # Fires between sessions. Sees full graph + S1/S2 traces across sessions.
    ("s3", "O"):       ["graph_structure",     # community detection, bridge analysis
                         "dedup_scan",          # cosine similarity findings
                         "correction_chains"],  # brain-wide correction chain traversal
    ("s3", "K"):       ["community_members",   # nodes in a community
                         "bridge_nodes",        # cross-community connectors
                         "stale_nodes"],        # nodes not accessed recently
    ("s3", "delta"):   ["community_label",     # named a community
                         "merge",               # merged duplicate nodes
                         "schema_node",         # extracted pattern into schema
                         "confidence_adjust"],  # lowered superseded confidence
    ("s3", "outcome"): ["recall_improved",     # recall quality measurably improved
                         "operator_approved"],  # Tom reviewed and approved the change

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
    "s2":         "s2-{session_short}-{run}",          # session encoder run number
    "s3":         "s3-{date}-{operation}",             # date=YYYYMMDD, operation=community/dedup/etc
    "s4":         "s4-{date}-{topic}",                 # date=YYYYMMDD, topic=what was researched
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
