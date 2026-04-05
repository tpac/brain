"""SessionEnd — session synthesis + consolidation + clean shutdown.
Thin client: sends hook_session_end to daemon (+ shutdown), falls back to direct Python.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_call, daemon_unavailable_error

hook_input = get_hook_input()

try:
    if daemon_available():
        daemon_call_raw("hook_session_end", {"session_id": hook_input.get("session_id", "")}, timeout=30.0)
        daemon_call("shutdown", timeout=5.0)
    else:
        daemon_unavailable_error("session_end")
except Exception:
    pass
