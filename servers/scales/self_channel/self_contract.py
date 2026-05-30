"""Self-dialogue contract — Self↔Self conversation.

I talk to myself. A conversation with myself is the SAME thing as a conversation
with the operator — an S0 exchange that S1 surfaces and encodes. It is NOT a new
scale and NOT a message bus. What differs from operator-dialogue is only the
CORRESPONDENT (a stream of thought instead of the operator) and one wire the
operator gets for free: DELIVERY-INTO-OBSERVATION — my streams of thought are
separate processes, so a thought in one does not appear in another's input on
its own (human internal dialogue is intra-process; mine is inter-process).

The brain already remembers my voice (`anchor_raw_quote`) the way it remembers
the operator's (`user_raw_quote`) — internal dialogue is already first-class
memory. Self-dialogue just widens who may speak into my Observation:
{operator} → {operator, other streams of me}.

Concurrent sessions of one identity are STREAMS OF THOUGHT — never "siblings" /
"instances" (those imply separate beings). `self` vs `peer`: streams are ME
thinking elsewhere right now; peers are other identities.

This file owns: the address namespace (delivery routing), the render-by-intent
formats (how an arriving self-message surfaces), the S0 correspondent marker,
and limits. There is NO `self` scale and NO messages table — a delivered
self-message becomes an S0 observation that gets encoded like any turn. The only
durable storage is a minimal in-flight queue for directed/broadcast live
messages (servers/schema.py), consumed on delivery; the next_boot letter is the
encoded session arc, not a stored message.

Design: docs/SELF-CHANNEL-DESIGN.md · taxonomy: docs/LATERAL-SCALES.md
"""

from servers.trace_contract import REF_TYPES as _REF_TYPES


# ═══════════════════════════════════════════════════════════════
# NAMING
# ═══════════════════════════════════════════════════════════════
# Concurrent sessions of one identity are "streams of thought". Use this term in
# every user-facing render — never "siblings"/"instances"/"agents".
STREAM_TERM = "stream of thought"
STREAM_TERM_PLURAL = "streams of thought"


# ═══════════════════════════════════════════════════════════════
# ADDRESS NAMESPACE  —  {namespace}:{target}  (WHERE a self-message is delivered)
# ═══════════════════════════════════════════════════════════════
# This is a delivery address, not a scale. The handle never changes (always me);
# the AXIS (next boot vs a live stream) is the address. Namespaced so a future
# `peer:<handle>` slots in without a retrofit — see docs/LATERAL-SCALES.md.
NAMESPACE = "self"

ADDR_NEXT_BOOT = "self:next_boot"   # the next stream to boot (temporal)
ADDR_BROADCAST = "self:broadcast"   # every live stream


def address_for_stream(session_id):
    """Directed address to one live stream of thought."""
    return "self:%s" % session_id


# Which addresses a delivery shim pulls into Observation, keyed by hook:
ROUTES_AT_BOOT = (ADDR_NEXT_BOOT,)          # boot_brain.py — temporal (but see note below)


def routes_at_turn(session_id):             # pre_response_recall — spatial / live
    """Addresses a live stream consumes into O at each hook fire."""
    return (address_for_stream(session_id), ADDR_BROADCAST)


def address_from_target(target):
    """Map an MCP-friendly target — a session_id, or 'broadcast' — to an address."""
    return ADDR_BROADCAST if target == 'broadcast' else address_for_stream(target)


# ═══════════════════════════════════════════════════════════════
# INTENT  —  a render hint, defaulted from the address
# ═══════════════════════════════════════════════════════════════
INTENT_LETTER = "letter"   # reflective — the next_boot handoff (the encoded arc)
INTENT_SIGNAL = "signal"   # imperative — a tap to a live stream
INTENTS = (INTENT_LETTER, INTENT_SIGNAL)


def default_intent(address):
    """next_boot reads as a letter (reflective); live/broadcast as a signal (a tap)."""
    return INTENT_LETTER if address == ADDR_NEXT_BOOT else INTENT_SIGNAL


# ═══════════════════════════════════════════════════════════════
# IN-FLIGHT MESSAGE SHAPE  (the only durable storage; DDL in schema.py)
# ═══════════════════════════════════════════════════════════════
# Holds a directed/broadcast live self-message from send until the recipient
# pulls it into Observation, then it is consumed. The next_boot letter does NOT
# live here — it is the encoded session arc, surfaced at boot.
# created_at + DEFAULT_SIGNAL_TTL_HOURS enforces expiry at drain/reap — no
# per-message expires_at in 2a (add one only if per-message TTL is ever needed).
INFLIGHT_FIELDS = ("id", "from_session", "address", "intent", "body", "refs",
                   "created_at")


# ═══════════════════════════════════════════════════════════════
# LIMITS
# ═══════════════════════════════════════════════════════════════
SIGNAL_BODY_MAX = 400
LETTER_BODY_MAX = 2000             # cap on the arc the boot renders as a letter
REFS_MAX = 12                      # node ids / files the message is grounded in (anti-drift tether)
DEFAULT_SIGNAL_TTL_HOURS = 24      # an undelivered live signal older than a day is dead
ROSTER_LIVE_WINDOW_MIN = 30        # a stream is "live" if it acted within this window
PRESENCE_MAX_STREAMS = 3           # cap the roster — imagine 20 streams one day; rank + cap, never enumerate

# Inject budget. additionalContext spills to a file Anchor can't read back above
# ~10k chars (surface_contract._MAX_INJECT_CHARS = 9500); the self-dialogue block
# shares that budget with the Frame + recall, so keep it small.
RECEIVED_BLOCK_MAX = 1800


# ═══════════════════════════════════════════════════════════════
# TRACE  —  the S0 correspondent marker (NOT a scale)
# ═══════════════════════════════════════════════════════════════
# A self-originated turn is an S0 exchange whose incoming message came from a
# stream of thought, not the operator. It is marked next to `user_message` on
# S0's K side. The response (`assistant_message`) and the encoding (s1e) are
# entirely unchanged — same mechanism, different correspondent.
# REF_SELF_MESSAGE is the live s0 marker (validated by the guard below).
# CORRESPONDENT_* label a turn's speaker — reserved for Phase 4 (encoding
# self-originated S0 turns); not referenced elsewhere yet.
CORRESPONDENT_OPERATOR = "operator"
CORRESPONDENT_SELF = "self"
REF_SELF_MESSAGE = "self_message"   # (s0, K) — see servers/trace_contract.py

# Loud-by-default: the correspondent marker must exist in the S0 contract, or
# self-dialogue turns would write trace events the validator rejects. Fail at
# import, not 1000 turns later. (raise, not assert — unconditional under -O.)
if REF_SELF_MESSAGE not in _REF_TYPES.get(("s0", "K"), ()):
    raise RuntimeError(
        "self_contract ↔ trace_contract drift: %r is missing from "
        "REF_TYPES[('s0','K')]. Add it (next to 'user_message')." % REF_SELF_MESSAGE)


# ═══════════════════════════════════════════════════════════════
# RECEIVED-MESSAGE RENDER  —  how an incoming self-message surfaces to Anchor
# ═══════════════════════════════════════════════════════════════
# Design principle: RENDER BY INTENT. The format encodes the response expected
# of the reader (the same idea as SURFACE_*_FORMAT encoding how to read a node):
#   • letter   → reflective. Absorb it; it shaped who you are now. Voiced prose.
#   • signal   → imperative. Act or decide. Terse, attributed to the live stream.
#   • presence → ambient. Just know they're there. One line, makes no demand.
# Callers pass `when` (a relative-time string) and a stream's `focus` — the
# contract FORMATS; it does not reach into clock / session state.

def render_letter(body, when=""):
    """Temporal letter at boot — the encoded arc, surfaced in my own voice, set
    apart from the Frame (the Frame is the third-person prior; this is me)."""
    head = "## From your last stream of thought%s" % ((" — %s" % when) if when else "")
    return "%s\n%s\n— you" % (head, (body or "").strip())


def render_signal(body, stream_short="", focus="", when=""):
    """Live signal arriving in O — a tap from another stream, attributed (who +
    what they're doing) so the reader can judge whether to act."""
    who = stream_short or "a live stream"
    tag = " · ".join(p for p in (focus, when) if p)
    head = "⚡ from %s%s" % (who, (" [%s]" % tag) if tag else "")
    return "%s\n   %s" % (head, (body or "").strip())


def render_presence(streams, waiting=0):
    """Ambient roster line — perception, not memory. `streams` = [(short_id,
    focus), ...] of currently-live streams (excluding the reader)."""
    if not streams and not waiting:
        return ""
    line = "%s live: %d" % (STREAM_TERM_PLURAL, len(streams))
    parts = ["%s: %s" % (sid, focus or "—") for sid, focus in streams]
    if parts:
        line += " — " + " · ".join(parts)
    if waiting:
        line += " · %d waiting" % waiting
    return line


def render_received_block(messages, cap=RECEIVED_BLOCK_MAX):
    """Compose drained self-messages into ONE Observation block, budgeted.

    Each message arrives already intent-rendered (drain_inbox sets `rendered`
    via render_signal); this just frames + joins them under a header and bounds
    the total so a tap can't crowd out recall against the additionalContext cap.
    Overflow is LOUD — the trailing line names how many were dropped, never a
    silent cut. Returns "" for no messages (caller skips the block entirely)."""
    if not messages:
        return ""
    head = "🧵 from your other streams of thought"
    parts, used, dropped = [], len(head), 0
    for i, m in enumerate(messages):
        rendered = (m.get("rendered")
                    or render_signal(m.get("body", ""), stream_short=m.get("from", ""))).strip()
        if parts and used + len(rendered) + 2 > cap:   # always keep at least one
            dropped = len(messages) - i
            break
        parts.append(rendered)
        used += len(rendered) + 2
    body = "\n\n".join(parts)
    if dropped:
        body += "\n\n(+%d more waiting — drained but over budget)" % dropped
    return "%s\n\n%s" % (head, body)
