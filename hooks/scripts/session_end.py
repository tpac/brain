"""SessionEnd — session synthesis + consolidation + clean shutdown.
Thin client: sends hook_session_end to daemon (+ shutdown), falls back to direct Python.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import daemon_available, daemon_call_raw, daemon_call, get_brain, close_brain

try:
    if daemon_available():
        daemon_call_raw("hook_session_end", {}, timeout=30.0)
        daemon_call("shutdown", timeout=5.0)
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_session_end
        brain = get_brain()
        if brain:
            try:
                hook_session_end(brain, {}, [])
            except Exception:
                pass
            finally:
                close_brain(brain)
except Exception:
    pass
