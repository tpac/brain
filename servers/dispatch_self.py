"""Dispatch handlers for the self channel — presence (pull) + signal (reach).

Thin command surface over servers/scales/self_channel/{presence,signal}.py.
- presence (self_presence / self_peek): read-only look at other streams.
- signal   (self_send / self_inbox):    directed message + consume-once drain.
Signal's brain_logs.db writes go through brain.write_lock inside signal.py;
none of these touch brain.db, so all register is_write=False.

Handler contract (shared by all daemon commands):
    handler(brain, args, graph_changes) -> {"ok": True, "result": <payload>}

Every handler MUST return the {"ok", "result"} envelope, like every other
dispatch_*.py handler. The table dispatch in daemon_server sends the return
verbatim — a raw, un-enveloped dict reaches the MCP client (brain_mcp) as a
falsy `ok` with no `error`, surfacing as the misleading "Unknown daemon error"
even though the handler succeeded. test_self_dispatch.py locks this.
"""

from servers.scales.self_channel import presence, signal


def _handle_self_presence(brain, args, graph_changes):
    """Roster of streams of thought awake right now + the rendered presence line.

    args.session_id = the caller's session (excluded from its own roster).
    args.limit      = optional cap (default PRESENCE_MAX_STREAMS).
    """
    return {"ok": True, "result": presence.build_presence(
        brain,
        my_session_id=args.get('session_id', '') or '',
        limit=args.get('limit'))}


def _handle_self_peek(brain, args, graph_changes):
    """Look into one stream of thought — its current focus. Read-only.

    args.stream_id = the TARGET stream to peek (distinct from the caller's
    session_id, so peeking never collides with the caller identity).
    """
    return {"ok": True, "result": presence.peek(brain, args.get('stream_id', '') or '')}


def _handle_self_send(brain, args, graph_changes):
    """Send a directed/broadcast self-message into the courier — the deliberate reach.

    args.to           = target: label, id-prefix, full session id, or 'broadcast'.
    args.body         = the message.
    args.from_session = caller's session id for attribution (falls back to session_id).
    args.from_label   = optional display name to send as (persisted).
    args.intent/refs  = optional.

    `to` resolves gracefully (signal.resolve_to): canonical id / broadcast pass
    through; a label or id-prefix matches the live roster; ambiguous or no match
    is a LOUD error so silence is never mistaken for delivery.
    """
    address, error = signal.resolve_to(brain, args.get('to', '') or '')
    if error:
        return {"ok": False, "error": error}
    return {"ok": True, "result": signal.send(
        brain,
        from_session=args.get('from_session', '') or args.get('session_id', '') or '',
        address=address,
        body=args.get('body', '') or '',
        intent=args.get('intent'),
        refs=args.get('refs'),
        from_label=args.get('from_label'))}


def _handle_self_inbox(brain, args, graph_changes):
    """Drain the caller's inbox — consume-once pending self-messages.

    args.session_id = the caller's own session, to fetch messages addressed to it.
    """
    return {"ok": True, "result": {'messages': signal.drain_inbox(brain, to_session=args.get('session_id', '') or '')}}


def _handle_self_outbox(brain, args, graph_changes):
    """Delivery status of the caller's SENT messages — who's drained each, and
    whether a directed target is still pending. Read-only (sender-side receipt).

    args.from_session = caller's session id (falls back to session_id).
    args.limit        = optional cap (default 20).
    """
    return {"ok": True, "result": signal.outbox(
        brain,
        from_session=args.get('from_session', '') or args.get('session_id', '') or '',
        limit=args.get('limit', 20))}
