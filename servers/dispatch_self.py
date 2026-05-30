"""Dispatch handlers for the self channel — presence layer (read-only).

Thin command surface over servers/scales/self_channel/presence.py. Presence is
PULL, not push (docs/SELF-CHANNEL-DESIGN.md, BOOT-REIGNITION.md), so both
handlers are reads — no lock, no graph mutation.

Handler contract (shared by all daemon commands):
    handler(brain, args, graph_changes) -> dict
"""

from servers.scales.self_channel import presence


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
