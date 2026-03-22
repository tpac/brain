"""Common setup for all brain hook Python scripts.

Eliminates repeated boilerplate: path setup, Brain import, input parsing,
daemon connection helpers, error logging. Every hook .py file imports this.
"""
import sys, os, json, socket, traceback, sqlite3
from datetime import datetime, timezone

# ── Path setup ──
server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db") if db_dir else ""

if server_dir:
    parent = os.path.dirname(server_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)


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
            conn = sqlite3.connect(db_path, timeout=2)
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
    """Log debug info to brain_logs.db for draining into Claude's context.

    Messages are written to the debug_log table with event_type='hook_debug'.
    The pre-response recall hook drains these and includes them in
    additionalContext, making them visible to Claude on the next prompt.

    Only logs when debug mode is on. Falls back to stderr if DB write fails.
    """
    if not is_debug_mode():
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    source = _get_hook_name()
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if logs_db and os.path.isdir(db_dir):
        try:
            conn = sqlite3.connect(logs_db, timeout=2)
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
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    tb = traceback.format_exc() if sys.exc_info()[2] else ""
    msg = "hook_error [%s] %s: %s" % (source, error, context)

    # Always print to stderr — never silent
    print("brain: %s" % msg, file=sys.stderr)
    if tb and tb.strip() != "NoneType: None":
        print("  traceback: %s" % tb.strip()[:500], file=sys.stderr)

    # Try to log to brain_logs.db
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if logs_db and os.path.isdir(db_dir):
        try:
            conn = sqlite3.connect(logs_db, timeout=3)
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


def get_unsurfaced_hook_errors(limit=10):
    """Read unsurfaced hook errors from brain_logs.db. Returns list of dicts."""
    logs_db = os.path.join(db_dir, "brain_logs.db") if db_dir else ""
    if not logs_db or not os.path.isfile(logs_db):
        return []
    try:
        conn = sqlite3.connect(logs_db, timeout=3)
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
        conn = sqlite3.connect(logs_db, timeout=3)
        placeholders = ",".join("?" * len(error_ids))
        conn.execute("UPDATE hook_errors SET surfaced = 1 WHERE id IN (%s)" % placeholders, error_ids)
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_hook_input():
    """Parse HOOK_INPUT from environment (set by bash shim)."""
    try:
        return json.loads(os.environ.get("HOOK_INPUT", "{}"))
    except Exception as e:
        log_hook_error(_get_hook_name(), e, "Failed to parse HOOK_INPUT", level="warning")
        return {}


def get_brain():
    """Import and instantiate Brain. Returns None on failure (logged)."""
    try:
        from servers.brain import Brain
        return Brain(db_path)
    except Exception as e:
        log_hook_error(_get_hook_name(), e, "Failed to import/init Brain")
        return None


def close_brain(brain):
    """Safely close brain."""
    if brain:
        try:
            brain.close()
        except Exception as e:
            log_hook_error(_get_hook_name(), e, "Failed to close brain", level="warning")


# ── Daemon helpers ──
SOCKET_PATH = f"/tmp/brain-daemon-{os.getuid()}.sock"


def daemon_available():
    """Check if daemon socket exists."""
    return os.path.exists(SOCKET_PATH) and stat_is_socket(SOCKET_PATH)


def stat_is_socket(path):
    """Check if path is a Unix socket."""
    import stat
    try:
        return stat.S_ISSOCK(os.stat(path).st_mode)
    except Exception:
        return False


def daemon_call(cmd, args=None, timeout=10.0):
    """Send a command to the daemon and return the result.
    Returns the result dict on success, empty dict on failure.
    """
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(SOCKET_PATH)
        msg = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        sock.sendall(msg.encode())
        data = b""
        while b"\n" not in data:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
        sock.close()
        resp = json.loads(data.decode().strip())
        return resp.get("result", {}) if resp.get("ok") else {}
    except Exception:
        return {}


def daemon_call_raw(cmd, args=None, timeout=10.0):
    """Send a command to the daemon and return the full response.
    Returns the raw response dict including 'ok' field.

    On error: always prints [BRAIN ERROR] to stderr so Claude sees it.
    On success + debug mode: prints [BRAIN DEBUG] summary.
    """
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(SOCKET_PATH)
        msg = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        sock.sendall(msg.encode())
        data = b""
        while b"\n" not in data:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
        sock.close()
        resp = json.loads(data.decode().strip())

        if not resp.get("ok"):
            err = resp.get("error", "unknown error")
            brain_error("%s failed: %s" % (cmd, err))
            log_hook_error(cmd, err, "daemon returned ok=false")

        return resp
    except socket.timeout:
        brain_error("%s timed out after %.0fs" % (cmd, timeout))
        log_hook_error(cmd, "timeout", "%.0fs" % timeout)
        return {"ok": False, "error": "timeout"}
    except ConnectionRefusedError:
        brain_error("%s: daemon not running" % cmd)
        return {"ok": False, "error": "daemon not running"}
    except Exception as e:
        brain_error("%s: %s" % (cmd, e))
        return {"ok": False, "error": str(e)}


def store_pending_message(brain_or_daemon, message):
    """Store a message for surfacing on next UserPromptSubmit.
    Accepts either a Brain instance or 'daemon' string.
    """
    try:
        if brain_or_daemon == "daemon":
            existing = daemon_call("get_config", {"key": "pending_hook_messages", "default": "[]"})
            pending = json.loads(existing) if isinstance(existing, str) else []
        else:
            existing = brain_or_daemon.get_config("pending_hook_messages", "[]")
            pending = json.loads(existing) if existing else []
    except Exception:
        pending = []

    pending.append(message)
    pending = pending[-5:]  # cap at 5

    try:
        if brain_or_daemon == "daemon":
            daemon_call("set_config", {"key": "pending_hook_messages", "value": json.dumps(pending)}, timeout=3.0)
        else:
            brain_or_daemon.set_config("pending_hook_messages", json.dumps(pending))
    except Exception:
        pass


def drain_pending_messages(brain_or_daemon):
    """Read and clear pending messages. Returns list of strings."""
    try:
        if brain_or_daemon == "daemon":
            existing = daemon_call("get_config", {"key": "pending_hook_messages", "default": "[]"})
            pending = json.loads(existing) if isinstance(existing, str) else []
            if pending:
                daemon_call("set_config", {"key": "pending_hook_messages", "value": "[]"}, timeout=3.0)
        else:
            existing = brain_or_daemon.get_config("pending_hook_messages", "[]")
            pending = json.loads(existing) if existing else []
            if pending:
                brain_or_daemon.set_config("pending_hook_messages", "[]")
    except Exception:
        pending = []
    return pending
