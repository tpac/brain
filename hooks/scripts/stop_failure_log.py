"""StopFailure — logs API failures to brain for pattern detection.
Thin client: sends hook_stop_failure_log to daemon, falls back to direct Python.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, log_hook_error

hook_input = get_hook_input()

try:
    if daemon_available():
        daemon_call_raw("hook_stop_failure_log", hook_input, timeout=5.0)
    else:
        daemon_unavailable_error("stop_failure_log")
except Exception as e:
    log_hook_error("stop_failure_log", e, "hook exception")
