"""Idle maintenance — S2 graph integration, backfill, cleanup.
Fires on Notification(idle_prompt). Notification stdout is NOT visible.
Thin client: sends hook_idle_maintenance to daemon, falls back to direct Python.
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, log_hook_error

# Observable: log every invocation so we can verify the notification fires
_IDLE_LOG = os.path.expanduser("~/AgentsContext/brain/idle_fires.log")

def _log_fire(msg):
    """Append to idle fire log — proves the notification reached us."""
    try:
        with open(_IDLE_LOG, "a") as f:
            f.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), msg))
    except Exception:
        pass

hook_input = get_hook_input()
_log_fire("FIRE session=%s" % hook_input.get("session_id", "unknown"))

try:
    if daemon_available():
        _log_fire("daemon_available=True, calling hook_idle_maintenance")
        resp = daemon_call_raw("hook_idle_maintenance", {"session_id": hook_input.get("session_id", "")}, timeout=60.0)
        _log_fire("daemon responded: %s" % str(resp)[:200])
    else:
        _log_fire("daemon_available=False")
        daemon_unavailable_error("idle_maintenance")
except Exception as e:
    _log_fire("ERROR: %s" % e)
    log_hook_error("idle_maintenance", e, "idle_maintenance hook failed")
    print("brain: idle_maintenance error: %s" % e, file=sys.stderr)
