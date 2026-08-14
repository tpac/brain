"""Common setup for all brain hook Python scripts.

Eliminates repeated boilerplate: path setup, Brain import, input parsing,
daemon connection helpers, error logging. Every hook .py file imports this.
"""
import sys, os, json, traceback, sqlite3
from datetime import datetime, timezone

# ── Path setup ──
server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db") if db_dir else ""

if server_dir:
    parent = os.path.dirname(server_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
else:
    # BRAIN_SERVER_DIR is exported by resolve-brain-db.sh, which only the
    # `bash <script>.sh` hooks source — a hook invoked as a bare `python3
    # <script>.py` inherits none of it, and `import servers.*` then raises
    # ModuleNotFoundError inside helpers that swallow exceptions. Derive the
    # plugin root from this file's own location instead of trusting the env:
    # hooks/scripts/hook_common.py → <plugin root>.
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isdir(os.path.join(_root, "servers")) and _root not in sys.path:
        sys.path.insert(0, _root)


def _get_hook_name():
    """Infer the calling hook name from the call stack."""
    import inspect
    for frame_info in inspect.stack():
        fname = os.path.basename(frame_info.filename)
        if fname.endswith(".py") and fname != "hook_common.py":
            return fname.replace(".py", "")
    return "unknown_hook"


# ── Debug mode ──
_debug_mode_cache = None


def is_debug_mode():
    """Check if brain debug mode is on.

    Resolution order:
      1. BRAIN_DEBUG env var (fastest, set at boot)
      2. brain_meta.debug_enabled in brain.db (persistent config)

    Debug mode shows all brain activity to Claude: recalls, injections,
    errors, encoding, telemetry. Toggle with brain.set_config('debug_enabled', '1'|'0').
    """
    global _debug_mode_cache
    if _debug_mode_cache is not None:
        return _debug_mode_cache

    # Env var (fast path)
    env = os.environ.get("BRAIN_DEBUG", "")
    if env:
        _debug_mode_cache = env == "1"
        return _debug_mode_cache

    # Read from brain_meta
    if db_path and os.path.isfile(db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            row = conn.execute(
                "SELECT value FROM brain_meta WHERE key = 'debug_enabled'"
            ).fetchone()
            conn.close()
            _debug_mode_cache = row is not None and row[0] == "1"
            return _debug_mode_cache
        except Exception:
            pass

    _debug_mode_cache = False
    return False


def brain_debug(msg):
    """Log hook activity to brain_logs.db for later inspection.

    Rows go to the debug_log table with event_type='hook_debug'. They are
    forensic — read on demand via the `query_logs` MCP tool (source='debug'),
    not pushed into Claude's context. The dashboard's Logs tab filters to
    event_type IN ('error','warning'), so these rows do not appear there.

    Only logs when debug mode is on. Falls back to stderr if DB write fails.
    """
    if not is_debug_mode():
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    source = _get_hook_name()
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if logs_db and os.path.isdir(db_dir):
        try:
            conn = sqlite3.connect(logs_db, timeout=10)
            conn.execute(
                "INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) "
                "VALUES (?, 'hook_debug', ?, ?, ?)",
                ('current', source, json.dumps({"message": msg}), ts),
            )
            conn.commit()
            conn.close()
            return
        except Exception:
            pass
    # Fallback: stderr (won't reach Claude but at least not lost)
    print("[BRAIN DEBUG] %s" % msg, file=sys.stderr)


def brain_error(msg):
    """Print error visible to Claude. Always prints (not gated by debug).

    Errors should never be silent — this is the lesson from 1,740 blind recalls.
    """
    print("[BRAIN ERROR] %s" % msg, file=sys.stderr)


def log_hook_error(source, error, context="", level="error"):
    """Log a hook error to brain_logs.db AND stderr.

    This is the ONLY place hook errors should be logged. Never swallow silently.
    Uses direct SQLite (not Brain) so it works even when Brain fails to import.

    Not every caller is inside a live `except`: a failure handed over as a
    RETURN VALUE (daemon_client.send_command's transport classes, the
    daemon-down handlers) has no sys.exc_info() to read, and those rows used to
    persist an empty traceback column. Falling back to format_stack() records
    the CALLER chain — which is the part worth reading anyway; the exception's
    own frames are usually one socket call the error string already names.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if sys.exc_info()[2]:
        tb = traceback.format_exc()
    else:
        # INNERMOST frames, not the whole stack: format_stack() is oldest-first,
        # so the column's 2000-char truncation would keep the interpreter's
        # bootstrap and cut the frames that name what actually failed. -1 drops
        # this function.
        tb = "".join(traceback.format_stack()[-9:-1])
    msg = "hook_error [%s] %s: %s" % (source, error, context)

    # Always print to stderr — never silent
    print("brain: %s" % msg, file=sys.stderr)
    if tb and tb.strip() != "NoneType: None":
        print("  traceback: %s" % tb.strip()[:500], file=sys.stderr)

    # Try to log to brain_logs.db
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if logs_db and os.path.isdir(db_dir):
        try:
            conn = sqlite3.connect(logs_db, timeout=10)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hook_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    hook_name TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'error',
                    error TEXT NOT NULL,
                    context TEXT DEFAULT '',
                    traceback TEXT DEFAULT '',
                    surfaced INTEGER DEFAULT 0
                )
            """)
            conn.execute(
                "INSERT INTO hook_errors (created_at, hook_name, level, error, context, traceback) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, source, level, str(error), context[:500], tb[:2000]),
            )
            # Prune old entries (keep last 200)
            conn.execute("DELETE FROM hook_errors WHERE id NOT IN (SELECT id FROM hook_errors ORDER BY id DESC LIMIT 200)")
            conn.commit()
            conn.close()
        except Exception:
            pass  # Last resort — stderr was already printed


def run_hook(name, fn, on_error=None):
    """Single error boundary for a hook script — the standard every hook runs through.

    Runs fn() (the hook body). If it raises, logs ONCE to hook_errors (via
    log_hook_error, the canonical hook-side sink) and invokes on_error() for the
    hook's fail-safe output — e.g. a PreToolUse hook printing its `approve`
    decision so the tool isn't blocked by the hook's own crash.

    Catches Exception only — SystemExit/KeyboardInterrupt propagate, so a hook's
    own `sys.exit(0)` skip-path (and Ctrl-C) still work. Never re-raises: a
    crashing hook must not break the host. Does NOT impose an exit code — each
    hook owns its own output contract (approve/block/additionalContext/none),
    which is why output lives in fn()/on_error(), not here.
    """
    try:
        fn()
    except Exception as e:
        log_hook_error(name, e, "hook exception")
        if on_error is not None:
            try:
                on_error()
            except Exception as e2:
                log_hook_error(name, e2, "hook on_error fallback also failed")


def get_unsurfaced_hook_errors(limit=10):
    """Read unsurfaced hook errors from brain_logs.db. Returns list of dicts."""
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if not logs_db or not os.path.isfile(logs_db):
        return []
    try:
        conn = sqlite3.connect(logs_db, timeout=10)
        # Check if table exists
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hook_errors'").fetchall()
        if not tables:
            conn.close()
            return []
        rows = conn.execute(
            "SELECT id, created_at, hook_name, level, error, context FROM hook_errors WHERE surfaced = 0 ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [{"id": r[0], "created_at": r[1], "hook_name": r[2], "level": r[3], "error": r[4], "context": r[5]} for r in rows]
    except Exception:
        return []


def mark_hook_errors_surfaced(error_ids):
    """Mark hook errors as surfaced so they are not shown again."""
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if not logs_db or not error_ids:
        return
    try:
        conn = sqlite3.connect(logs_db, timeout=10)
        placeholders = ",".join("?" * len(error_ids))
        conn.execute("UPDATE hook_errors SET surfaced = 1 WHERE id IN (%s)" % placeholders, error_ids)
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_hook_input():
    """Parse HOOK_INPUT from environment (set by bash shim).

    Backfills `session_id` from CLAUDE_CODE_SESSION_ID (the env var Claude Code
    exports in every hook process) when the payload omits it. Some events —
    notably WorktreeCreate — don't carry session_id in their JSON, and the daemon
    is a separate launchd process that can't read the hook's env; without this the
    daemon would fall back to the last-writer-wins singleton and mis-attribute
    per-session state to the wrong stream."""
    try:
        data = json.loads(os.environ.get("HOOK_INPUT", "{}"))
    except Exception as e:
        log_hook_error(_get_hook_name(), e, "Failed to parse HOOK_INPUT", level="warning")
        return {}
    if isinstance(data, dict) and not data.get("session_id"):
        env_sid = os.environ.get("CLAUDE_CODE_SESSION_ID", "")
        if env_sid:
            data["session_id"] = env_sid
    return data


def daemon_unavailable_error(hook_name=None):
    """Return a CRITICAL error when daemon is unreachable.

    Does three things:
    1. Returns CRITICAL additionalContext that Claude MUST relay to user
    2. Logs to dashboard DB directly (no daemon needed)
    3. Attempts to restart daemon via ensure_daemon()

    Exception — unconfigured install (no brain.db anywhere AND no daemon):
    that is not a crash, it's Anchor never having come up. Logs ANCHOR OFFLINE
    and exits 1 (real, non-blocking hook error) WITHOUT attempting recovery —
    boot creates brains, recovery must not, and recover-storms on machines
    with nothing to recover are how first installs get flaky.
    """
    name = hook_name or _get_hook_name()

    if not (db_path and os.path.isfile(db_path)):
        log_hook_error(
            name,
            "ANCHOR OFFLINE - no brain.db at %s and the daemon is not answering"
            % (db_dir or "unresolved"),
            "plugin enabled but Anchor never came up: recall+encode+traces dead. "
            "Likely causes in order: stale BRAIN_DB_DIR override in "
            "~/.config/brain/env; first install interrupted; daemon crash "
            "before bind. (A missing ANTHROPIC_API_KEY does NOT cause this — "
            "boot creates the brain and daemon keyless since a1a620e.)",
            level="critical")
        sys.exit(1)

    # 1. Format CRITICAL message for Claude
    msg = (
        "[CRITICAL — RELAY TO USER IMMEDIATELY]\n"
        "⚠️ Brain daemon is unreachable. Hook: %s.\n"
        "Brain memories are safe (on disk). Recall and encoding are disabled.\n"
        "Attempting automatic restart.\n"
        "DO NOT continue without informing the user.\n"
        "[/CRITICAL]"
    ) % name

    print("[brain] CRITICAL: %s — daemon unavailable" % name, file=sys.stderr)

    # 2. Persist to brain_logs.db directly (SQLite, no daemon needed) so the
    #    outage shows in the dashboard + query_logs even while the daemon is
    #    down. The old log_hook_output path was a silent no-op (removed
    #    2026-05-30), which had quietly broken this guarantee for hooks here.
    try:
        log_hook_error("DAEMON_DOWN", "daemon unreachable (detected by %s)" % name,
                       "recall + encoding disabled until restart; relayed to operator",
                       level="critical")
    except Exception:
        pass

    # 3. A hung daemon keeps the port bound, so launchd's crash-respawn never
    # fires and the daemon can't SIGTERM itself. Hand off to the single
    # recovery path — it kills the corpse + lets launchd respawn, guarded by
    # maintenance-lock + cooldown + circuit breaker. (The MCP health monitor
    # calls the same function; shared state keeps them from fighting.)
    try:
        from servers.daemon_client import recover_daemon
        recover_daemon(db_path)
    except Exception:
        pass

    return msg


# ── Daemon helpers ──
# No host/port here: servers.daemon_config owns the address (env-first, formula
# as fallback) and daemon_client owns the wire.


def daemon_available(timeout=2.0):
    """Liveness check — True only if the daemon answers a ping.

    Delegates to daemon_client.is_daemon_responsive(), the single source of
    truth. A hung "corpse" that holds the port but never replies reads as
    unavailable here, so recovery can fire; a slow-but-alive daemon still
    answers and reads as available."""
    try:
        from servers.daemon_client import is_daemon_responsive
        return is_daemon_responsive(timeout=timeout)
    except Exception:
        return False


def daemon_call(cmd, args=None, timeout=10.0):
    """Send a command to the daemon and return the result.
    Returns the result dict on success, empty dict on failure.

    The quiet sibling of daemon_call_raw: same wire (daemon_client owns it),
    no logging — for callers that treat a daemon-down as a non-event.
    """
    try:
        from servers.daemon_client import send_command
        resp = send_command(cmd, args, timeout=timeout)
        return resp.get("result", {}) if resp.get("ok") else {}
    except Exception:
        return {}


def daemon_call_raw(cmd, args=None, timeout=10.0):
    """Send a command to the daemon and return the full response.
    Returns the raw response dict including 'ok' field.

    The WIRE belongs to daemon_client.send_command; this wrapper owns the
    hook-side OBSERVABILITY — the [BRAIN ERROR] line Claude sees and the
    hook_errors row the dashboard and boot read. Failure classes come from
    send_command's `transport` key, never from matching its prose.

    On error: always prints [BRAIN ERROR] to stderr so Claude sees it.
    """
    try:
        from servers.daemon_client import send_command
        resp = send_command(cmd, args, timeout=timeout)
    except Exception as e:
        # The client module itself is unreachable (import/path failure) — a
        # hook must degrade, never crash the session.
        brain_error("%s: %s" % (cmd, e))
        log_hook_error(cmd, e, "daemon_call_raw failed")
        return {"ok": False, "error": str(e)}

    transport = resp.get("transport")
    if transport == "timeout":
        brain_error("%s timed out after %.0fs" % (cmd, timeout))
        log_hook_error(cmd, "timeout", "%.0fs" % timeout)
        return {"ok": False, "error": "timeout"}
    if transport == "refused":
        brain_error("%s: daemon not running" % cmd)
        return {"ok": False, "error": "daemon not running"}
    if transport:
        # Empty read, garbled JSON, socket reset. Logged, not just returned:
        # a failure that reaches Claude but never reaches hook_errors is
        # invisible in the dashboard and at boot.
        err = resp.get("error", "unknown error")
        brain_error("%s: %s" % (cmd, err))
        log_hook_error(cmd, err, "daemon_call_raw failed")
        return {"ok": False, "error": err}

    if not resp.get("ok"):
        err = resp.get("error", "unknown error")
        brain_error("%s failed: %s" % (cmd, err))
        log_hook_error(cmd, err, "daemon returned ok=false")

    return resp
