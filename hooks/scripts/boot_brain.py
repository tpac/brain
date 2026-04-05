"""SessionStart hook — boots brain context via daemon.

Uses daemon for all brain access (single model, single process).
Falls back to direct Brain() ONLY if daemon is completely unavailable.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import db_path, daemon_available, daemon_call, daemon_call_raw, log_hook_output

db_dir = os.environ.get("BRAIN_DB_DIR", "")

# Session ID from hook input — extracted by boot-brain.sh, passed as env var
_hook_session_id = os.environ.get("BRAIN_HOOK_SESSION_ID", "")


def _boot_via_daemon():
    """Boot context through daemon — the normal path.
    Skips full boot if already booted this session (once: true not honored for settings hooks).
    """
    # Check if already booted — use Claude's session_id for continuity.
    current_sid = daemon_call("get_config", {"key": "session_id", "default": ""})
    current_sid = current_sid.get("value", current_sid) if isinstance(current_sid, dict) else (current_sid or "")
    booted_sid = daemon_call("get_config", {"key": "last_booted_session", "default": ""})
    booted_sid = booted_sid.get("value", booted_sid) if isinstance(booted_sid, dict) else (booted_sid or "")

    # Use Claude's session_id if available, otherwise check existing
    _sid = _hook_session_id or current_sid
    if _sid and booted_sid and _sid == booted_sid:
        print("[BRAIN] (session resumed — brain already loaded)", file=sys.stderr)
        return True

    # Reset session activity with Claude's session_id (not a random UUID)
    daemon_call("reset_session", {"session_id": _hook_session_id} if _hook_session_id else {})

    # Get debug mode
    debug = daemon_call("get_config", {"key": "debug_enabled", "default": "0"})
    if isinstance(debug, str) and debug == "1":
        os.environ["BRAIN_DEBUG"] = "1"
    elif isinstance(debug, dict) and debug.get("value") == "1":
        os.environ["BRAIN_DEBUG"] = "1"

    # Resolve user/project
    user = os.environ.get("BRAIN_USER", "User")
    project = os.environ.get("BRAIN_PROJECT", "default")
    if user == "User":
        stored = daemon_call("get_config", {"key": "default_user", "default": "User"})
        if isinstance(stored, str) and stored != "User":
            user = stored
        elif isinstance(stored, dict) and stored.get("value", "User") != "User":
            user = stored["value"]
    if project == "default":
        stored = daemon_call("get_config", {"key": "default_project", "default": "default"})
        if isinstance(stored, str) and stored != "default":
            project = stored
        elif isinstance(stored, dict) and stored.get("value", "default") != "default":
            project = stored["value"]

    # Get formatted boot context
    result = daemon_call("context_boot", {"user": user, "project": project})
    if isinstance(result, dict):
        text = result.get("for_claude", "") or result.get("text", "")
        if text:
            log_hook_output("boot", output_text=text)
            print(text)
            # Mark session as booted (skip re-boots on resume)
            # Mark this session as booted (skip re-boots on resume)
            sid = daemon_call("get_config", {"key": "session_id", "default": ""})
            sid_val = sid.get("value", sid) if isinstance(sid, dict) else (sid or "")
            if sid_val:
                daemon_call("set_config", {"key": "last_booted_session", "value": sid_val})
            return True
    elif isinstance(result, str) and result:
        log_hook_output("boot", output_text=result)
        print(result)
        daemon_call("set_config", {"key": "session_booted", "value": "1"})
        return True

    return False


def _boot_via_direct():
    """Fallback: direct Brain() — only if daemon is completely dead."""
    server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
    if server_dir:
        parent = os.path.dirname(server_dir)
        if parent not in sys.path:
            sys.path.insert(0, parent)

    try:
        from servers.brain import Brain
    except ImportError as e:
        print("brain: Failed to import: " + str(e), file=sys.stderr)
        return

    try:
        brain = Brain(db_path)
    except Exception as e:
        print("brain: Failed to init: " + str(e), file=sys.stderr)
        return

    try:
        brain.reset_session_activity()
        user = os.environ.get("BRAIN_USER", "User")
        project = os.environ.get("BRAIN_PROJECT", "default")
        rendered = brain.format_boot_context(user=user, project=project, db_dir=db_dir)
        if isinstance(rendered, dict):
            print(rendered.get("for_claude", ""))
        else:
            print(rendered)
    finally:
        brain.close()


# ── Main ──
if daemon_available():
    if not _boot_via_daemon():
        print("[brain-boot] Daemon returned empty context, falling back to direct", file=sys.stderr)
        _boot_via_direct()
else:
    print("[brain-boot] Daemon not available, using direct Brain()", file=sys.stderr)
    _boot_via_direct()
