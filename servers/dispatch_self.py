"""Dispatch handlers for the self channel — presence (pull) + signal (reach).

Thin command surface over servers/scales/self_channel/{presence,signal}.py.
- presence (self_presence / self_peek): read-only look at other streams.
- signal   (self_send / self_inbox):    directed message + consume-once drain.
Signal's brain_logs.db writes go through brain.write_lock inside signal.py;
none of these touch brain.db, so all register is_write=False.

Handler contract (shared by all daemon commands):
    handler(brain, args, graph_changes) -> dict
"""

from servers.scales.self_channel import presence, signal, self_contract


def _handle_self_presence(brain, args, graph_changes):
    """Roster of streams of thought awake right now + the rendered presence line.

    args.session_id = the caller's session (excluded from its own roster).
    args.limit      = optional cap (default PRESENCE_MAX_STREAMS).
    """
    return presence.build_presence(
        brain,
        my_session_id=args.get('session_id', '') or '',
        limit=args.get('limit'))


def _handle_self_peek(brain, args, graph_changes):
    """Look into one stream of thought — its current focus. Read-only.

    args.stream_id = the TARGET stream to peek (distinct from the caller's
    session_id, so peeking never collides with the caller identity).
    """
    return presence.peek(brain, args.get('stream_id', '') or '')


def _handle_self_send(brain, args, graph_changes):
    """Send a directed/broadcast self-message into the courier — the deliberate reach.

    args.to           = target stream id, or 'broadcast'.
    args.body         = the message.
    args.from_session = caller's session id for attribution (falls back to session_id).
    args.intent/refs  = optional.
    """
    return signal.send(
        brain,
        from_session=args.get('from_session', '') or args.get('session_id', '') or '',
        address=self_contract.address_from_target(args.get('to', '') or ''),
        body=args.get('body', '') or '',
        intent=args.get('intent'),
        refs=args.get('refs'))


def _handle_self_inbox(brain, args, graph_changes):
    """Drain the caller's inbox — consume-once pending self-messages.

    args.session_id = the caller's own session, to fetch messages addressed to it.
    """
    return {'messages': signal.drain_inbox(brain, to_session=args.get('session_id', '') or '')}
