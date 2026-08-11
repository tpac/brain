"""Idle maintenance — S2 graph integration, backfill, cleanup.
Fires on Notification(idle_prompt). Notification stdout is NOT visible.
Thin client: sends hook_idle_maintenance to daemon, falls back to direct Python.
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, run_hook

# Observable: log every invocation so we can verify the notification fires.
# Lives in the resolved brain dir — the wrapper (idle-maintenance.sh) sources
# resolve-brain-db.sh and exits before us unless BRAIN_DB_DIR is set.
def _idle_log_path():
    db_dir = os.environ.get("BRAIN_DB_DIR")
    return os.path.join(db_dir, "idle_fires.log") if db_dir else None

def _log_fire(msg):
    """Append to idle fire log — proves the notification reached us."""
    path = _idle_log_path()
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg))
    except Exception:
        pass

hook_input = get_hook_input()
_log_fire("FIRE session=%s" % hook_input.get("session_id", "unknown"))

def main():
    if daemon_available():
        _log_fire("daemon_available=True, calling hook_idle_maintenance")
        resp = daemon_call_raw("hook_idle_maintenance", {"session_id": hook_input.get("session_id", "")}, timeout=60.0)
        _log_fire("daemon responded: %s" % str(resp)[:200])
    else:
        _log_fire("daemon_available=False")
        daemon_unavailable_error("idle_maintenance")

run_hook("idle_maintenance", main, on_error=lambda: _log_fire("ERROR — logged to hook_errors"))
