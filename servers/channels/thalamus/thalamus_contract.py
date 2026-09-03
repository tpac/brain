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

from datetime import datetime as _dt

from servers.loud_truncation import cap_text_loud
from servers.clock import iso_after, resolve_offset, FUTURE
from servers.channels.delivery import BOOT as _BOOT, STOP as _STOP


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


# ═══════════════════════════════════════════════════════════════
# AUDIENCES  —  the item's RECIPIENT SET (who the pull predicate serves).
# NOT a frequency: delivery cardinality is constant — the ledger PK caps
# every item at one delivery per session per armed_epoch, and repetition
# over time is the epoch axis (re-arm). This enum only picks who is
# eligible. A directed item narrows first_session to one named session
# via the target_session column.
# ═══════════════════════════════════════════════════════════════
AUDIENCE_FIRST = 'first_session'  # the first session to pull after due —
                                  # a dynamic singleton (reminders/notices)
AUDIENCE_EVERY = 'every_session'  # every session inside the window, once
                                  # each (standing notices and asks)

AUDIENCES = (AUDIENCE_FIRST, AUDIENCE_EVERY)  # the closed set the pull
                                              # predicate matches — a value
                                              # outside it would never deliver
                                              # and die silently at expiry;
                                              # file() guards the door

def default_audience(needs_answer):
    """An ask renders at each new session's boot until answered/expired; a
    reminder/notice fires once. Both overridable via for_whom."""
    return AUDIENCE_EVERY if needs_answer else AUDIENCE_FIRST


# ═══════════════════════════════════════════════════════════════
# MOMENTS  —  the delivery moments a session pulls at. The vocabulary is
# OWNED by channels/delivery.py (the last-mile leg both channels ride);
# these are its names, kept here so the pull predicate and ledger speak
# them without reaching around the contract. `via` is written to the
# ledger verbatim, so it must be vocabulary, not free text — pull()
# validates against MOMENTS loudly (a typo'd via would behave as Stop and
# ledger the typo).
# ═══════════════════════════════════════════════════════════════
VIA_BOOT = _BOOT.name
VIA_STOP = _STOP.name
MOMENTS = (VIA_BOOT, VIA_STOP)
ASK_MOMENTS = (VIA_BOOT,)  # asks deliver at boot only — an architecture
                           # question arriving mid-thread trains
                           # reflex-deferral; at boot there is no thread


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
NOTICE_EXPIRES_DAYS = 7    # undated-notice window
REMIND_GRACE_DAYS = 7      # dated items: expiry = deliver_at + grace


# ═══════════════════════════════════════════════════════════════
# KIND  —  ONE derivation of what an item IS, feeding both the render verb
# and the expiry-span lookup (three unnamed partitions of the item space
# disagreeing was the root of the false-dead-letter defect). Internal and
# DERIVED — never producer-set, never stored: the rejected producer-facing
# kind vocabulary stays rejected. Producers state needs_answer / when /
# for_whom; the kind falls out.
# ═══════════════════════════════════════════════════════════════
KIND_ASK = 'ask'            # needs an answer (needs_answer)
KIND_REMINDER = 'reminder'  # carries a clock (deliver_at)
KIND_NOTICE = 'notice'      # undated FYI

KIND_VERB = {KIND_ASK: 'asks', KIND_REMINDER: 'reminds', KIND_NOTICE: 'notes'}
KIND_EXPIRES_DAYS = {KIND_ASK: ASK_EXPIRES_DAYS,
                     KIND_REMINDER: REMIND_GRACE_DAYS,
                     KIND_NOTICE: NOTICE_EXPIRES_DAYS}


def kind_of(item):
    """item dict (needs_answer / deliver_at suffice) → its derived kind."""
    if item.get('needs_answer'):
        return KIND_ASK
    if item.get('deliver_at'):
        return KIND_REMINDER
    return KIND_NOTICE


# ═══════════════════════════════════════════════════════════════
# `when` RESOLUTION  —  the door's time grammar (a presentation concern,
# kept out of the mechanics; the table only ever sees resolved ISO)
# ═══════════════════════════════════════════════════════════════

def resolve_when(value):
    """Resolve a producer's `when` to a future ISO deliver_at, or None for
    "next opportunity". Accepts relative shorthand ('30m', '2h', '3d', '1w'),
    an ISO timestamp literal, or ''/None/'now'. Anything else raises
    ValueError — a malformed deadline must fail loud at the door, not
    silently become an immediate delivery.

    The grammar itself lives in clock.py, shared with recall_episodes'
    lookback bounds; what stays here is what is genuinely the door's — the
    'now'/empty convention (a queue has a "next opportunity"; a lookback
    bound does not) and the door-flavoured error message."""
    if not value or str(value).strip().lower() == 'now':
        return None
    try:
        return resolve_offset(value, direction=FUTURE)
    except ValueError as e:
        raise ValueError('thalamus: when=%r %s' % (value, e))


def window_for(needs_answer, deliver_at):
    """The item's expiry when the producer names none: ANCHOR (deliver_at,
    or now) × SPAN (per-kind days) — one composition, never branch-dependent.
    Anchoring at the due date keeps a dated item's full window; expiry
    before due would make it undeliverable and fire a FALSE loud
    dead-letter."""
    span = KIND_EXPIRES_DAYS[kind_of(
        {'needs_answer': needs_answer, 'deliver_at': deliver_at})]
    anchor = _dt.fromisoformat(deliver_at) if deliver_at else None
    return iso_after(days=span, at=anchor)


def extend_window(needs_answer, new_deliver, current_expires):
    """Defer's window rule: literally window_for's ANCHOR × SPAN composition
    re-anchored at the new due date — a deferred item keeps its KIND's full
    span (an ask stays an ask: 14 days past due, not a reminder's 7). A
    window already past that keeps its length."""
    return max(current_expires or '', window_for(needs_answer, new_deliver))


# ═══════════════════════════════════════════════════════════════
# `for_whom` RESOLUTION  —  producer intent → route
# ═══════════════════════════════════════════════════════════════
FOR_WHOM_LIVE = ('live', 'live-now', 'all-live')  # → courier broadcast, no queue row lifecycle
FOR_WHOM_ALL = 'all'                              # → AUDIENCE_EVERY (window)


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
        return 'queue', AUDIENCE_EVERY, ''
    from servers.channels.self_channel.self_contract import is_session_id
    if is_session_id(fw):
        # Directed: the session-keyed correction (b474ccbd) — full UUID only;
        # an 8-char short is a display convention, not a key.
        return 'queue', AUDIENCE_FIRST, fw
    raise ValueError(
        "thalamus: for_whom=%r is not 'live', 'all', or a full session UUID"
        % (for_whom,))


# ═══════════════════════════════════════════════════════════════
# TRACE  —  the s0 delivery marker
# ═══════════════════════════════════════════════════════════════
# A Thalamus delivery is an incoming K to the receiving session, next to
# self_message. Untraced delivery IS the visibility problem this system
# exists to fix. channels/delivery.py writes it at BOTH moments (the caller
# owns tracing — it holds the chain); the ledger stays the delivery-policy
# record (who got which item, which epoch), the trace is the S0-stream join.
# The constant LIVES in trace_contract (which registers it in REF_TYPES by
# construction and is import-cycle-free from channels/); re-exported here for
# this package's callers.
from servers.trace_contract import REF_THALAMUS_DELIVERY


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
    verb = KIND_VERB[kind_of(item)]
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
    """Compose due items into ONE budgeted block. Two loud caps — per item
    (BODY_MAX, in render_item) and whole block (`cap`); overflow items are
    named at the tail, never silently cut. Items are from the brain's own
    machinery (not a stream) — rendered under a brain head so origin is
    honest.

    Returns (block, kept) — kept is how many items actually rendered ('' and
    0 when items is empty). The caller must ledger ONLY the kept items: a
    cap-dropped item was never shown, and recording it as delivered would
    suppress it forever (it stays armed for the next moment instead)."""
    if not items:
        return '', 0
    # Budget against the widest possible head, then rebuild it from the
    # kept count — the head must claim what the block SHOWS, never what
    # was fetched (head, tail, ledger, and pull's count all say `kept`).
    parts, used, dropped = [], len('🧠 from the brain (thalamus) — %d item(s)'
                                   % len(items)), 0
    for i, item in enumerate(items):
        rendered = render_item(item).strip()
        if parts and used + len(rendered) + 2 > cap:  # always keep one
            dropped = len(items) - i
            break
        parts.append(rendered)
        used += len(rendered) + 2
    head = '🧠 from the brain (thalamus) — %d item(s)' % len(parts)
    body = '\n\n'.join(parts)
    tail = dropped + max(0, overflow)
    if tail:
        body += '\n\n(+%d more due — thalamus_list shows them)' % tail
    return '%s\n\n%s' % (head, body), len(parts)
