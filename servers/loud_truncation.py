"""Loud truncation — Tom's standing truncation contract (node 8178593a).

Every truncation point in the brain must be LOUD: never a silent slice, always a
marker naming what was dropped. These are the shared primitives. Callers pass a
domain-specific marker (e.g. one pointing at where the full text still lives) —
the mechanism (rstrip head + dropped-count marker) is the same everywhere.

Used by the trace system (trace_contract.build_delta_metadata) and the
self-channel delivery render (self_contract._render_one). Imports nothing from
servers — it sits below both, so neither domain owns the other's primitive.
"""


def cap_text_loud(s, limit, marker="…[+%d chars truncated]"):
    """Truncate `s` to `limit` chars LOUDLY: rstrip the head and append `marker`
    (a %d format string receiving the dropped char count). Returns `s` unchanged
    when it fits. The dropped tail is genuinely lost at call sites that keep only
    a head, so the marker is the only record it happened."""
    s = s or ''
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + " " + (marker % (len(s) - limit))


def cap_list_loud(items, limit, marker="…[+%d more truncated]"):
    """Keep the first `limit` items LOUDLY: append a marker element naming how
    many were dropped, vs a silent slice. Stays a list (shape-valid)."""
    items = list(items or [])
    if len(items) <= limit:
        return items
    return items[:limit] + [marker % (len(items) - limit)]
