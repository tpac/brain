"""Loud truncation — the standing truncation contract.

Every truncation point in the brain must be LOUD: never a silent slice, always a
marker naming what was dropped. These are the shared primitives. Callers pass a
domain-specific marker (e.g. one pointing at where the full text still lives) —
the mechanism (rstrip head + dropped-count marker) is the same everywhere.

Used wherever the brain shapes text for a bounded destination — trace stores,
delta metadata, and the channel blocks injected into a session. Imports
nothing from servers — it sits below every domain that caps, so none of them
owns another's primitive.
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


def one_line(s):
    """Collapse all whitespace (incl. newlines) to single spaces — the shape
    for text that must occupy ONE line of a rendered block (a row in a
    prompt table, a summary). Pairs with cap_text_loud."""
    return ' '.join(str(s or '').split())


def cap_list_loud(items, limit, marker="…[+%d more truncated]"):
    """Keep the first `limit` items LOUDLY: append a marker element naming how
    many were dropped, vs a silent slice. Stays a list (shape-valid)."""
    items = list(items or [])
    if len(items) <= limit:
        return items
    return items[:limit] + [marker % (len(items) - limit)]


def compose_block_loud(items, render, cap, reserved=0, sep="\n\n"):
    """Fit rendered items under `cap` chars LOUDLY: render in order, stop at
    the first item that would overflow (always keeping one, so a single
    oversize item still shows), and report how many were dropped so the
    caller can name them at the tail. `reserved` is what the caller's head
    already spent from the budget. Items past the cut are never rendered.
    `sep` joins the kept items — a blank line for multi-line items (the
    default), a newline for one-line rows. Returns (body, kept, dropped).
    The caller owns head and tail wording: what a dropped item MEANS differs
    per domain (a thalamus item stays due; a courier message is already
    spent)."""
    parts, used, dropped = [], reserved, 0
    for i, item in enumerate(items):
        rendered = render(item).strip()
        if parts and used + len(rendered) + len(sep) > cap:
            dropped = len(items) - i
            break
        parts.append(rendered)
        used += len(rendered) + len(sep)
    return sep.join(parts), len(parts), dropped
