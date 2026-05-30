"""SessionStart hook — boots brain context via daemon.

Uses daemon for all brain access (single model, single process).
Falls back to direct Brain() ONLY if daemon is completely unavailable.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import db_path, daemon_available, daemon_call, daemon_call_raw, run_hook, get_unsurfaced_hook_errors, mark_hook_errors_surfaced

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

    # Get formatted boot context (retry once if daemon just started)
    # 2026-05-02 (Frame Phase 2.5): pass session_id so render_boot_v2 can
    # build the Frame for THIS session via ctx.get_frame(brain).
    boot_args = {"user": user, "project": project}
    if _sid:
        boot_args["session_id"] = _sid
    result = daemon_call("context_boot", boot_args)
    if not result:
        import time
        time.sleep(1)
        result = daemon_call("context_boot", boot_args)
    if isinstance(result, dict):
        text = result.get("for_claude", "") or result.get("text", "")
        if text:
            print(text)
            # Mark session as booted (skip re-boots on resume)
            # Mark this session as booted (skip re-boots on resume)
            sid = daemon_call("get_config", {"key": "session_id", "default": ""})
            sid_val = sid.get("value", sid) if isinstance(sid, dict) else (sid or "")
            if sid_val:
                daemon_call("set_config", {"key": "last_booted_session", "value": sid_val})
            return True
    elif isinstance(result, str) and result:
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
def main():
    if daemon_available():
        if not _boot_via_daemon():
            print("[brain-boot] Daemon returned empty context, falling back to direct", file=sys.stderr)
            _boot_via_direct()
    else:
        print("[brain-boot] Daemon not available, using direct Brain()", file=sys.stderr)
        _boot_via_direct()


def _surface_unsurfaced_errors():
    """Boot-surfacing (#1): proactively show hook errors that accumulated since
    the last session. Resilient by design — runs even if boot failed, because
    errors matter most exactly when the brain is too broken to boot. Reads the
    hook_errors sink directly (no daemon needed), prints a banner Claude relays
    to the operator, then marks them surfaced. Never breaks boot."""
    try:
        errs = get_unsurfaced_hook_errors(limit=5)
        if not errs:
            return
        lines = ["[BRAIN] ⚠️ %d unsurfaced hook error(s) since last session:" % len(errs)]
        for e in errs:
            lines.append("  • [%s] %s: %s (%s)" % (
                e.get("level", "error"), e.get("hook_name", "?"),
                (e.get("error") or "")[:120], (e.get("created_at") or "")[:19]))
        lines.append("  → full detail in the dashboard errors view.")
        print("\n".join(lines))
        mark_hook_errors_surfaced([e["id"] for e in errs])
    except Exception as e:
        # Must never break boot — but don't swallow silently either (that's the
        # bug this whole change fought). stderr is the last-resort channel when
        # the hook_errors sink itself may be the thing that's failing.
        print("[brain-boot] error-surfacing failed: %s" % e, file=sys.stderr)


run_hook("boot_brain", main)
_surface_unsurfaced_errors()
