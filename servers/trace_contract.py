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
        "description": "Brain's first processing pass — recall, judge, encode",
        "triggers": "UserPromptSubmit (recall/judge) + Stop every 5th (encode)",
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
    # Recall path (chain prefix: recall-): O=candidates, K=judge picks, delta=context sent to Anchor
    # Encode path (chain prefix: encode-): O=prompt given, K=node catalog, delta=actions+reasoning
    ("s1", "O"):       ["recall",            # candidates with scores
                         "encoding_prompt"],   # what the encoder was given
    ("s1", "K"):       ["judge_selected",    # what the judge picked
                         "node_catalog"],      # which nodes available to encoder
    ("s1", "delta"):   ["additionalContext",  # what reached Anchor
                         "encoding_run"],      # what the encoder produced
    ("s1", "outcome"): ["correction",         # Tom corrected something that was recalled
                         "recall_hit"],        # node was recalled in a future turn
    # Scale 2-4: to be defined when built
}


# ── CHAIN ID CONVENTIONS ──
# chain_id groups related O/K/Δ/outcome events.

CHAIN_PREFIXES = {
    "s0": "s0-{session_id_short}-{stop_counter}",
    "s0_tool": "s0-{session_id_short}-tool",
    "s1_recall": "recall-{session_id_short}-{recall_log_id}",
    "s1_encode": "encode-{session_id_short}-{counter}",
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
