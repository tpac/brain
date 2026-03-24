"""SessionEnd — session synthesis + consolidation + clean shutdown.
Thin client: sends hook_session_end to daemon (+ shutdown), falls back to direct Python.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import daemon_available, daemon_call_raw, daemon_call, daemon_unavailable_error

try:
    if daemon_available():
        daemon_call_raw("hook_session_end", {}, timeout=30.0)
        daemon_call("shutdown", timeout=5.0)
    else:
        daemon_unavailable_error("session_end")
except Exception:
    pass
