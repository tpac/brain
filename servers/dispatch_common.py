"""Daemon dispatch — shared helpers.

Imported by every dispatch_* handler module AND by daemon_dispatch (the
registry). Holds the CmdEntry contract, the unknown-key guard and
session-context popping. No dependency on the handler modules, so the
import graph stays acyclic: common <- {write,read,observability,ops} <- daemon_dispatch.

Node ids are exact 8-char hex everywhere — there is no prefix resolution
step at the dispatch layer; misses are the owning brain method's to report.
"""

import hashlib
import hmac
import os
import re
from typing import Any, Dict, Callable, Optional, NamedTuple


# Reserved arg key: the calling session's identity, stamped by the MCP proxy
# (brain_mcp.daemon_send) on every tool call it can attribute, under its OWN
# name — never as `session_id`. Identity handlers (attribution / per-session
# state) resolve the caller via caller_session(); cross-session FILTER reads
# (recall_episodes, query_traces) read the caller-supplied `session_id` only,
# so an absent scope means "all streams" by design — never the calling session.
CALLER_SESSION_KEY = "_caller_session"

# Reserved arg key: the HMAC that lets the proxy trust a `_caller_session` it
# did not write itself. On a host that hands the MCP proxy no session identity
# (Codex passes no thread id to stdio servers), a PreToolUse hook — which does
# receive `session_id` — signs it and rewrites the tool input with both keys
# (hooks/scripts/stamp_caller_session.py). The proxy verifies and strips the
# signature before dispatch (brain_mcp._stamp_caller_session); _pop_session_ctx
# drops and logs a stray one so it never lands in a node's KV.
#
# Threat model: the model writing identity keys by accident or on a whim. The
# secret is a 0600 file under the same uid the agent's shell runs as, so the
# pair raises the cost of a forgery, it does not make one impossible — the
# once-per-process note in the proxy is the control that matters. Both reserved
# keys share the `_caller_` prefix; hook_common.strip_caller_stamp leans on it
# to keep the pair out of tool traces without importing this module.
CALLER_SIG_KEY = "_caller_sig"

# The name every host adapter registers the proxy's MCP server under (the
# tests pin the adapters to it — the service layer never reads a manifest).
# Hosts name its tools `mcp__<server>__<tool>`; Claude Code prefixes a plugin's
# servers as `plugin_<plugin>_<server>`.
BRAIN_MCP_SERVER = "brain"
_BRAIN_TOOL_RE = re.compile(r"^mcp__(?:.*_)?%s__" % re.escape(BRAIN_MCP_SERVER))


def is_brain_tool(tool_name):
    """True for one of the brain's own MCP tools on either host — the only
    tools a hook may approve and rewrite; never a user's tool."""
    return isinstance(tool_name, str) and _BRAIN_TOOL_RE.match(tool_name) is not None


def hook_secret_path():
    """`<user config dir>/brain/hook-secret`, beside the user env knob. The dir
    is daemon_config's; imported lazily so its import-time work stays off the
    hook's hot path until a stamp is actually signed."""
    from servers.daemon_config import user_config_dir
    return os.path.join(user_config_dir(), "brain", "hook-secret")


def _read_secret(path):
    try:
        with open(path, "rb") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def _hook_secret():
    """The per-install signing secret: 32 random bytes as hex, mode 0600,
    created on first use by whichever side asks first. Published by hard-linking
    a fully written temp file onto the path, so the name never exists with
    partial content and the loser of a hook/proxy race on a fresh install reads
    the winner's key. An existing empty file is a broken install, not a key."""
    path = hook_secret_path()
    data = _read_secret(path)
    if data is None:
        import secrets
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        tmp = "%s.%d.tmp" % (path, os.getpid())
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_hex(32).encode("ascii") + b"\n")
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass  # lost the race — the winner's key is the key
        finally:
            os.unlink(tmp)
        data = _read_secret(path)
    if not data:
        raise RuntimeError("hook secret at %s is empty — delete it to regenerate" % path)
    return data


def sign_caller_session(session_id):
    """HMAC-SHA256 (hex) of the session id under the install's hook secret.
    Raises when the secret cannot be read or made — signing has nothing to
    degrade to."""
    return hmac.new(_hook_secret(), session_id.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_caller_session(session_id, signature):
    """True only for a (session_id, signature) pair this install's hook produced.
    Malformed inputs are False; a broken secret file raises, which the proxy
    turns into an unattributed call plus a note."""
    if not (isinstance(session_id, str) and session_id and isinstance(signature, str)):
        return False
    return hmac.compare_digest(sign_caller_session(session_id).encode("ascii"),
                               signature.encode("utf-8"))


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
    if args.pop(CALLER_SIG_KEY, None) is not None:
        # The proxy strips the signature before dispatch; a stray one means a
        # client bypassed it. Loud, and never into a node's KV.
        try:
            brain._log_error('caller_sig_leaked',
                             ValueError('%s reached the daemon' % CALLER_SIG_KEY),
                             'a client sent the hook signature past the MCP proxy')
        except Exception:
            pass
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
