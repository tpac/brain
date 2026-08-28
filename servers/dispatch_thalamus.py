"""Dispatch handlers for the Thalamus — Anchor's MCP entrance to the door.

Thin command surface over servers/scales/thalamus/thalamus.py:
- remind           → thalamus.file()  (THE producer verb: a notice, a reminder,
                     and an ask are the same call with different params)
- thalamus_list    → the pullable view
- thalamus_resolve → answer / defer / dismiss

All storage is brain_logs.db serialized on brain.logs_write_lock inside the
module, so every entry registers is_write=False (the dispatch_self pattern).
Handlers return the {"ok", "result"} envelope like every dispatch_*.py.
"""

from servers.scales.thalamus import thalamus
from servers.dispatch_common import caller_session


def _handle_remind(brain, args, graph_changes):
    """File a Thalamus item from Anchor.

    args.what         = the body (required).
    args.when         = shorthand ('2h','3d','1w'), ISO, or ''/now (immediate).
    args.for_whom     = ''/None (default), 'live' (broadcast to live streams
                        now via the courier), 'all' (every session in window),
                        or a full session UUID (directed).
    args.needs_answer = ask semantics: boot-only delivery, renders per session
                        until answered, loud expiry.
    args.refs         = node ids, resolved at render.
    args.dedup_key    = producer-owned identity; repeat updates, not inserts.
    """
    result = thalamus.file(
        brain,
        source='anchor',
        body=args.get('what', '') or '',
        needs_answer=bool(args.get('needs_answer', False)),
        when=args.get('when'),
        for_whom=args.get('for_whom'),
        refs=args.get('refs'),
        dedup_key=args.get('dedup_key'),
        session_id=caller_session(args))
    if not result.get('filed'):
        return {"ok": False, "error": result.get('error', 'thalamus.file failed')}
    return {"ok": True, "result": result}


def _handle_thalamus_list(brain, args, graph_changes):
    """The pullable view — open items with delivery counts (include_closed for
    the audit trail)."""
    return {"ok": True, "result": thalamus.list_items(
        brain,
        include_closed=bool(args.get('include_closed', False)),
        limit=args.get('limit', 50))}


def _handle_thalamus_resolve(brain, args, graph_changes):
    """Close or defer one item — exactly one of answer / defer_until / dismiss."""
    result = thalamus.resolve(
        brain,
        item_id=args.get('id', '') or '',
        answer=args.get('answer'),
        defer_until=args.get('defer_until'),
        dismiss=bool(args.get('dismiss', False)))
    if not result.get('ok'):
        return {"ok": False, "error": result.get('error', 'thalamus_resolve failed')}
    return {"ok": True, "result": result}
