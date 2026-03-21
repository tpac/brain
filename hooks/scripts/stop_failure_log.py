"""StopFailure — logs API failures to brain for pattern detection.
Thin client: sends hook_stop_failure_log to daemon, falls back to direct Python.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, get_brain, close_brain

hook_input = get_hook_input()

try:
    if daemon_available():
        daemon_call_raw("hook_stop_failure_log", hook_input, timeout=5.0)
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_stop_failure_log
        brain = get_brain()
        if brain:
            try:
                hook_stop_failure_log(brain, hook_input, [])
            except Exception:
                pass
            finally:
                close_brain(brain)
except Exception:
    pass
