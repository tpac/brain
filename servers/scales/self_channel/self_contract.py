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
# TRUNCATION CONTRACT  —  one truncation point, always loud
# ═══════════════════════════════════════════════════════════════
# Tom's standing rule (node 8178593a): every truncation point must have an
# EXPLICIT, documented contract — never a bare magic number doing a silent
# slice. This section IS that contract for the self-channel.
#
#   SEND      (signal.send)            stores body + refs IN FULL. No
#                                      truncation. Only guard: a non-empty body.
#   DELIVERY  (render_received_block)  the SINGLE truncation point. Two caps,
#                                      both LOUD — never a silent cut:
#                                        • per message → DELIVERED_BODY_MAX:
#                                          the body is cut with an inline marker
#                                          naming the dropped char count.
#                                        • whole block → RECEIVED_BLOCK_MAX:
#                                          overflow messages are named at the tail.
#   INVARIANT                          the full message ALWAYS survives in the
#                                      courier (self_inflight) — the dashboard
#                                      Streams tab shows it untruncated.
#                                      Truncation only ever shapes what is
#                                      INJECTED into Observation, never storage.
#
# Why a delivery cap exists at all (and a send cap does not): additionalContext
# spills to a file Anchor can't read back past ~9500 chars
# (surface_contract._MAX_INJECT_CHARS = 9500), and the self-block shares that
# budget with the Frame + recall. So the cap is a real downstream delivery
# constraint — NOT a stylistic limit on how much I'm allowed to say. There is
# deliberately no SIGNAL_BODY_MAX / REFS_MAX: those were arbitrary silent slices
# (removed 2026-05-30), exactly the anti-pattern this contract forbids.
DELIVERED_BODY_MAX = 1000          # per-message body, capped LOUDLY at delivery render
RECEIVED_BLOCK_MAX = 1800          # whole injected self-block, overflow named LOUDLY

# ── POLICY DEFAULTS (tunable knobs; NOT truncation points) ──────────────
DEFAULT_SIGNAL_TTL_HOURS = 24      # an undelivered live signal older than a day is dead (drain/reap filter)
ROSTER_LIVE_WINDOW_MIN = 30        # a stream counts as "live" if it acted within this window
PRESENCE_MAX_STREAMS = 3           # roster shows a count + top-K ranked, never enumerates all

# ── Phase 3 forward placeholder (NOT yet enforced) ──────────────────────
# The boot-letter budget — designed when Phase 3 (the first-person letter)
# lands. Defined so the design-doc reference resolves; nothing enforces it today.
LETTER_BODY_MAX = 2000


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
    """Live signal arriving in O — a tap from ANOTHER stream, rendered as REPORTED
    speech: `who` is a grammatical subject and the body is QUOTED as their claim,
    never your own first-person assertion. That re-voicing is the containment
    barrier — you cannot absorb '<who> says: "I did X"' as something YOU did
    without a visible grammatical error. (The boot letter stays first-person; see
    render_letter — continuity across time is the point there. This is for live,
    concurrent streams, where two agencies coexist in one moment and a verb is
    about to get mis-owned.) `who` is the stream's label, falling back to its
    short id."""
    who = stream_short or "a live stream"
    tag = " · ".join(p for p in (focus, when) if p)
    head = "⚡ %s%s says:" % (who, (" [%s]" % tag) if tag else "")
    return '%s\n   "%s"' % (head, (body or "").strip())


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


def _render_one(m):
    """Render ONE drained message for injection, capping its body LOUDLY at
    DELIVERED_BODY_MAX (the per-message half of the truncation contract). Renders
    from the raw body — not a pre-baked string — so the cap actually applies; the
    full body always remains in the courier. Letters render reflective; anything
    else as a signal tap."""
    body = m.get("body", "") or ""
    if len(body) > DELIVERED_BODY_MAX:
        dropped = len(body) - DELIVERED_BODY_MAX
        body = (body[:DELIVERED_BODY_MAX].rstrip()
                + " …[+%d chars — full message in the dashboard Streams tab]" % dropped)
    if m.get("intent") == INTENT_LETTER:
        return render_letter(body)
    return render_signal(body, stream_short=m.get("from", ""))


def render_received_block(messages, cap=RECEIVED_BLOCK_MAX):
    """Compose drained self-messages into ONE budgeted Observation block.

    This is the SINGLE truncation point of the self-channel (see the TRUNCATION
    CONTRACT above). Two LOUD caps, never a silent cut:
      • per message → DELIVERED_BODY_MAX (in _render_one), body cut with a marker
      • whole block → `cap` (RECEIVED_BLOCK_MAX), overflow named at the tail
    The full message always survives in the courier (self_inflight); the
    dashboard Streams tab shows it untruncated. Returns "" for no messages
    (caller skips the block entirely)."""
    if not messages:
        return ""
    head = "🧵 from your other streams of thought"
    note = ("   — what they did is theirs; you know it, you didn't do it. "
            "Attribute accordingly if you encode.")
    parts, used, dropped = [], len(head) + len(note), 0
    for i, m in enumerate(messages):
        rendered = _render_one(m).strip()
        if parts and used + len(rendered) + 2 > cap:   # always keep at least one
            dropped = len(messages) - i
            break
        parts.append(rendered)
        used += len(rendered) + 2
    body = "\n\n".join(parts)
    if dropped:
        body += ("\n\n(+%d more waiting — over the injection budget; "
                 "full text in the dashboard Streams tab)" % dropped)
    return "%s\n%s\n\n%s" % (head, note, body)
