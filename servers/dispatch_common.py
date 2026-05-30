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
    from servers.dal import NodeDAL
    dal = brain._nodes
    full_id = dal.resolve_id(node_id)
    return full_id if full_id else node_id  # Not found — let the caller handle the error


def _pop_session_ctx(brain, args):
    """Pop session_id from args, return (ctx, args).

    Handlers that pass `**args` into a brain method should call this first.
    session_id is auto-injected by daemon_send() from CLAUDE_CODE_SESSION_ID;
    without popping it, it cascades into the brain method's `**extra_fields`
    and is silently stored as a `session_id` KV on every node — a real leak
    from the auto-injection convenience. Returns the resolved SessionContext
    (or None), ready to pass as an explicit `ctx=` kwarg.
    """
    sid = args.pop('session_id', None)
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
    unknown = set(args.keys()) - entry.accepts
    if not unknown:
        return
    try:
        brain._log_error(
            'dispatch_unknown_keys',
            ValueError('cmd=%s dropped keys=%s' % (cmd, sorted(unknown))),
            'accepted=%s' % sorted(entry.accepts))
    except Exception:
        pass
