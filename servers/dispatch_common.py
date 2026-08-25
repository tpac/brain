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


def sender_id(args):
    """The caller's OWN canonical id for attributing a self-message it SENDS.

    The authoritative id (`caller_session` — the proxy-stamped full session) WINS
    over an explicit `from_session` arg. That ordering is the fix, not an accident:
    a caller can pass `from_session` as the 8-char SHORT (the form Anchor sees in
    rendered messages and presence lines), and storing a short corrupts attribution
    and seeds the self_send resolver's false-ambiguity — the courier ends up
    holding one stream under two id formats (db79e0c1 / brain node 41c6ebed). The
    explicit arg is honored only HEADLESS, when the proxy stamped no caller session.

    Distinct from `caller_session`, which keeps the raw `session_id`-first
    precedence the read tools that scope BY an explicit id rely on."""
    return caller_session(args) or args.get('from_session', '') or ''


def _agent_limit(req, default, ceiling):
    """Clamp an agent-facing read limit at the dispatch door.

    Absent/None → the default page; no request exceeds `ceiling`. The MCP
    surface renders results into the caller's context, so the agent path stays
    bounded here — while the underlying DAL read is unbounded (limit=None) for
    internal id-set / window scans, a path that never comes through dispatch.
    One clamp, parameterized per door: filter_nodes (50, NODE_QUERY_MAX_LIMIT)
    and recall_episodes (EPISODE_DEFAULT_LIMIT, EPISODE_MAX_LIMIT).
    """
    req = default if req is None else req
    return min(max(int(req), 1), ceiling)


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


def log_failed_batch_ops(brain, source: str, cmd: str, result) -> None:
    """Loud-at-the-write-boundary scan for per-op failures inside an ok=True
    batch result (docs/TRACE-MODES-DESIGN.md §Failed-run residue, gap 1): a
    brain_batch can return ok=True while individual operations carry ok=False —
    without this, a per-op error string never reaches the errors table and error
    scans miss it entirely.

    Called from `daemon_dispatch.dispatch_command`, so it covers EVERY caller
    (daemon TCP / MCP, the S1+S2 encoder closure, IsolatedBrain) rather than the
    encoder path alone. `source` is the attribution label — the call's
    `encoding_source`, e.g. 's2:consolidation' or 'anchor'.

    Never raises. Runs inside the caller's write lock (it writes brain_logs.db
    via _log_error); see dispatch_command's docstring for why that matters.
    """
    try:
        if not (isinstance(result, dict) and result.get('ok')):
            return  # whole-call failures are already loud at the caller
        inner = result.get('result')
        per_op = inner.get('results') if isinstance(inner, dict) else None
        if not isinstance(per_op, list):
            return
        failed = [r for r in per_op
                  if isinstance(r, dict) and r.get('ok') is False]
        if not failed:
            return
        heads = '; '.join(
            '#%s %s: %s' % (r.get('index', '?'), r.get('op', '?'),
                            str(r.get('error', ''))[:200])
            for r in failed[:5])
        brain._log_error(
            'batch_op_failed',
            RuntimeError('%d/%d op(s) failed inside an ok=True %s'
                         % (len(failed), len(per_op), cmd)),
            'source=%s; %s' % (source, heads))
    except Exception as e:
        # Never break the write path — but the loudness mechanism dying
        # silently would reopen the exact gap it closes, so leave a trace
        # in daemon.log (stderr) even when _log_error itself is what broke.
        import sys as _sys
        print('[dispatch:%s] per-op failure scan broke: %s' % (source, e),
              file=_sys.stderr, flush=True)
