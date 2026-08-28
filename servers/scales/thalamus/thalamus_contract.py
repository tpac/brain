"""Thalamus contract — the brain speaking to its streams.

The Thalamus is a durable store of STANDING INTENTS with delivery policy: one
item can spawn N deliveries over its life (once per session inside a window),
knows who already got it (the ledger), and can carry an answer back to its
producer. It sits BESIDE the self-channel, not inside it — the msgs layer is
streams speaking to each other (ephemeral, consume-once, TTL); the Thalamus is
the brain speaking to its streams (durable, windowed, ledgered). A live-now
fire-and-forget FYI never enters the queue: `file()` delegates it to the
courier broadcast, where irrelevance-at-send is already the contract.

This file owns: states, audiences, caps and default windows, the `when` /
`for_whom` resolution grammar, the delivery render, and the trace-contract
drift guard. Mechanics live in thalamus.py; DDL in servers/schema.py
(thalamus_items / thalamus_deliveries, logs DB).

Time is WALL-CLOCK (iso_now / iso_after) — delivery windows are courier-class
real-elapsed deadlines, the same documented exemption as the self-channel;
nothing here is on the eval-replay conversation-time path.

Design: docs/THALAMUS-DESIGN.md
"""

import re

from servers.trace_contract import REF_TYPES as _REF_TYPES
from servers.loud_truncation import cap_text_loud
from servers.clock import iso_after


# ═══════════════════════════════════════════════════════════════
# STATES  —  the item lifecycle (delivery state, NOT producer-condition state)
# ═══════════════════════════════════════════════════════════════
# Whether the underlying condition is still true stays the producer's business
# (two orthogonal state machines — brain node e63c41dd). Retry, when it ships
# (Phase 3), gates on unacked delivery, never on a still-open condition.
STATE_OPEN = 'open'            # active: pullable inside its window
STATE_ANSWERED = 'answered'    # closed by Anchor with an answer payload
STATE_DISMISSED = 'dismissed'  # closed by Anchor without an answer
STATE_WITHDRAWN = 'withdrawn'  # closed by its OWN producer (retraction)
STATE_EXPIRED = 'expired'      # window ended — LOUD for an unanswered ask
                               # (the dead-letter fix), natural for a notice
STATE_SENT = 'sent'            # terminal at file(): delegated live-now
                               # broadcast — the courier owns its death

TERMINAL_STATES = (STATE_ANSWERED, STATE_DISMISSED, STATE_WITHDRAWN,
                   STATE_EXPIRED, STATE_SENT)


# ═══════════════════════════════════════════════════════════════
# AUDIENCES  —  how many sessions one item reaches (the pull predicate)
# ═══════════════════════════════════════════════════════════════
AUDIENCE_ONCE = 'once'   # first session that pulls after due — reminders/notices
AUDIENCE_ALL = 'all'     # once per session inside the window — standing notices
                         # and asks ("push once per session, stay pullable")

def default_audience(needs_answer):
    """An ask renders at each new session's boot until answered/expired; a
    reminder/notice fires once. Both overridable via for_whom."""
    return AUDIENCE_ALL if needs_answer else AUDIENCE_ONCE


# ═══════════════════════════════════════════════════════════════
# CAPS & WINDOWS  —  policy as data. Volume is owned HERE, not by producer
# discretion (v1's fatal finding #2 — brain node 6789e133).
# ═══════════════════════════════════════════════════════════════
MAX_OPEN_PER_SOURCE = 8    # file() REJECTS (loudly, synchronously) at the cap
PULL_MAX_ITEMS = 5         # per render moment (boot / stop), overflow named
BLOCK_MAX = 4000           # whole injected block — loud cap, mirror of the
                           # self-channel RECEIVED_BLOCK_MAX discipline
BODY_MAX = 1500            # per-item body at render — loud cap; storage is FULL
RENDER_REFS_MAX = 3        # refs resolved inline per item, rest named

ASK_EXPIRES_DAYS = 14      # needs_answer window; expiry past it is LOUD
NOTICE_EXPIRES_DAYS = 7    # audience-all notice window
REMIND_GRACE_DAYS = 7      # once-items: expiry = (deliver_at or now) + grace


# ═══════════════════════════════════════════════════════════════
# `when` RESOLUTION  —  the door's time grammar (a presentation concern,
# kept out of the mechanics; the table only ever sees resolved ISO)
# ═══════════════════════════════════════════════════════════════
_SHORTHAND_RE = re.compile(r'(\d+)\s*([mhdw])', re.IGNORECASE)


def resolve_when(value):
    """Resolve a producer's `when` to a future ISO deliver_at, or None for
    "next opportunity". Accepts relative shorthand ('30m', '2h', '3d', '1w'),
    an ISO timestamp literal (normalized via fromisoformat), or ''/None/'now'.
    Anything else raises ValueError — a malformed deadline must fail loud at
    the door, not silently become an immediate delivery."""
    if not value or str(value).strip().lower() == 'now':
        return None
    s = str(value).strip()
    m = _SHORTHAND_RE.fullmatch(s)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        kw = {'m': {'minutes': n}, 'h': {'hours': n},
              'd': {'days': n}, 'w': {'days': 7 * n}}[unit]
        return iso_after(**kw)
    from datetime import datetime as _dt, timezone as _tz
    try:
        parsed = _dt.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError(
            "thalamus: when=%r is neither shorthand ('30m','2h','3d','1w') "
            "nor an ISO timestamp" % (value,))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_tz.utc)
    return parsed.astimezone(_tz.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# `for_whom` RESOLUTION  —  producer intent → route
# ═══════════════════════════════════════════════════════════════
FOR_WHOM_LIVE = ('live', 'live-now', 'all-live')  # → courier broadcast, no queue row lifecycle
FOR_WHOM_ALL = 'all'                              # → audience 'all' (window)


def resolve_for_whom(for_whom, needs_answer):
    """Map a producer's `for_whom` to (route, audience, target_session).

    route is 'live' (delegate to the courier broadcast — fire-and-forget) or
    'queue'. A full session UUID targets one session (directed). None/'' means
    the default audience for the item's kind. Unknown strings raise — a typo'd
    audience must not silently become a default."""
    fw = (for_whom or '').strip()
    if not fw:
        return 'queue', default_audience(needs_answer), ''
    if fw.lower() in FOR_WHOM_LIVE:
        return 'live', '', ''
    if fw.lower() == FOR_WHOM_ALL:
        return 'queue', AUDIENCE_ALL, ''
    from servers.scales.self_channel.self_contract import is_session_id
    if is_session_id(fw):
        # Directed: the session-keyed correction (b474ccbd) — full UUID only;
        # an 8-char short is a display convention, not a key.
        return 'queue', AUDIENCE_ONCE, fw
    raise ValueError(
        "thalamus: for_whom=%r is not 'live', 'all', or a full session UUID"
        % (for_whom,))


# ═══════════════════════════════════════════════════════════════
# TRACE  —  the s0 delivery marker
# ═══════════════════════════════════════════════════════════════
# A Thalamus delivery is an incoming K to the receiving session, next to
# self_message. Untraced delivery IS the visibility problem this system
# exists to fix. The Stop hook writes it (the caller owns tracing — it holds
# the chain); at boot the ledger + boot_renders row are the record.
REF_THALAMUS_DELIVERY = 'thalamus_delivery'

# Loud-by-default: fail at import, not at the first delivery.
if REF_THALAMUS_DELIVERY not in _REF_TYPES.get(('s0', 'K'), ()):
    raise RuntimeError(
        "thalamus_contract ↔ trace_contract drift: %r is missing from "
        "REF_TYPES[('s0','K')]. Add it (next to 'self_message')."
        % REF_THALAMUS_DELIVERY)


# ═══════════════════════════════════════════════════════════════
# RENDER  —  how due items surface. Answerable without a fetch: body +
# resolved refs + the pre-filled resolve call, inline (brain node 70016ed3).
# Single truncation point, always loud; storage keeps everything.
# ═══════════════════════════════════════════════════════════════

def render_item(item):
    """One due item. The verb points at Anchor and the exit is inline —
    items arrive as asks/notices with affordances, not status to read past.
    Formats what it is handed: refs arrive pre-resolved as 'ref_lines'
    (pull() owns resolution — batched, veil-aware; the contract never
    reaches into the brain)."""
    body = cap_text_loud(
        (item.get('body') or '').strip(), BODY_MAX,
        marker='…[+%d chars — thalamus_list shows it in full]')
    verb = ('asks' if item.get('needs_answer') else
            ('reminds' if item.get('deliver_at') else 'notes'))
    head = '• %s · %s %s' % (item.get('id', '?'), item.get('source', '?'), verb)
    lines = [head, '  "%s"' % body]
    for ref_line in item.get('ref_lines') or []:
        lines.append('  ↳ %s' % ref_line)
    if item.get('needs_answer'):
        lines.append('  → thalamus_resolve("%s", answer=…) · defer_until=… · '
                     'dismiss=true' % item.get('id', '?'))
    else:
        lines.append('  → thalamus_resolve("%s", dismiss=true) when handled'
                     % item.get('id', '?'))
    return '\n'.join(lines)


def render_block(items, overflow=0, cap=BLOCK_MAX):
    """Compose due items into ONE budgeted block ('' when empty). Two loud
    caps — per item (BODY_MAX, in render_item) and whole block (`cap`);
    overflow items are named at the tail, never silently cut. Items are from
    the brain's own machinery (not a stream) — rendered under a brain head so
    origin is honest."""
    if not items:
        return ''
    head = '🧠 from the brain (thalamus) — %d item(s)' % len(items)
    parts, used, dropped = [], len(head), 0
    for i, item in enumerate(items):
        rendered = render_item(item).strip()
        if parts and used + len(rendered) + 2 > cap:  # always keep one
            dropped = len(items) - i
            break
        parts.append(rendered)
        used += len(rendered) + 2
    body = '\n\n'.join(parts)
    tail = dropped + max(0, overflow)
    if tail:
        body += '\n\n(+%d more due — thalamus_list shows them)' % tail
    return '%s\n\n%s' % (head, body)
