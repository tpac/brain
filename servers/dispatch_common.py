"""Daemon dispatch — shared helpers.

Imported by every dispatch_* handler module AND by daemon_dispatch (the
registry). Holds the CmdEntry contract, the unknown-key guard, id resolution
and session-context popping. No dependency on the handler modules, so the
import graph stays acyclic: common <- {write,read,observability,ops} <- daemon_dispatch.
"""

from typing import Any, Dict, Callable, Optional, NamedTuple


def _resolve_id(brain, node_id):
    """Resolve a node ID — exact match first, then prefix match.

    Handles both new 8-char IDs and old 32-char IDs gracefully.
    Returns the full ID if found, or the original input if not.
    """
    if not node_id:
        return node_id
    dal = brain._nodes
    full_id = dal.resolve_id(node_id)
    return full_id if full_id else node_id  # Not found — let the caller handle the error


# Reserved arg key: the calling session's identity, stamped by the MCP proxy
# (brain_mcp.daemon_send) on EVERY tool call under its OWN name — never as
# `session_id`. Identity handlers (attribution / per-session state) resolve the
# caller via caller_session(); cross-session FILTER reads (recall_episodes,
# query_traces) read the caller-supplied `session_id` only, so an absent scope
# means "all streams" by design — never the calling session.
CALLER_SESSION_KEY = "_caller_session"


def caller_session(args):
    """The calling session's identity, for attribution / per-session state.

    Explicit caller-supplied `session_id` wins (the few write/self tools that
    surface it in their schema); otherwise the ambient session the MCP proxy
    stamped under `_caller_session`. Returns '' when neither is present.

    Do NOT use for cross-session FILTER reads — those read
    `args.get('session_id')` directly so an omitted scope means all streams.
    """
    return args.get('session_id') or args.get(CALLER_SESSION_KEY) or ''


def _pop_session_ctx(brain, args):
    """Resolve the calling session's ctx, popping BOTH identity keys.

    Handlers that pass `**args` into a brain method must call this first:
    identity arrives under `_caller_session` (stamped by the MCP proxy) or an
    explicit caller-supplied `session_id`; left in args, either cascades into
    the brain method's `**extra_fields` and is silently stored as KV on every
    node. Returns the resolved SessionContext (or None), ready to pass as an
    explicit `ctx=` kwarg.
    """
    sid = caller_session(args)
    args.pop('session_id', None)
    args.pop(CALLER_SESSION_KEY, None)
    if not sid:
        return None, args
    try:
        return brain.get_or_create_session(sid), args
    except Exception:
        return None, args


class CmdEntry(NamedTuple):
    handler: Callable
    is_write: bool
    marks_dirty: bool = False
    # Optional contract: the set of top-level keys this handler knows how to
    # consume. When set, the dispatcher logs an error for any arg key not in
    # this set — surfaces silent drops (e.g. encoding_source being passed but
    # not forwarded to the brain method). None = no check, opt-in per entry.
    accepts: Optional[frozenset] = None


def check_unknown_keys(cmd: str, entry: 'CmdEntry', args: Dict[str, Any], brain) -> None:
    """Log an error if args contains keys the handler doesn't list as accepted.

    Defensive: any `accepts` contract must allow all keys the handler reads —
    if `accepts` is declared but the handler reads a field not in the set,
    this function will (incorrectly) flag legitimate inputs as unknown. Update
    `accepts` when you change what a handler reads.
    """
    if entry.accepts is None or not args:
        return
    # _caller_session is the ambient identity the MCP proxy stamps on every
    # call — it is never declared in a handler's `accepts` set, so exempt it
    # here rather than mis-flagging it as a dropped key.
    unknown = set(args.keys()) - entry.accepts - {CALLER_SESSION_KEY}
    if not unknown:
        return
    try:
        brain._log_error(
            'dispatch_unknown_keys',
            ValueError('cmd=%s dropped keys=%s' % (cmd, sorted(unknown))),
            'accepted=%s' % sorted(entry.accepts))
    except Exception:
        pass
