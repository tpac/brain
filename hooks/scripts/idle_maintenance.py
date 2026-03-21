"""Idle maintenance — dream, consolidate, heal, tune, reflect.
Fires on Notification(idle_prompt). Notification stdout is NOT visible.
Thin client: sends hook_idle_maintenance to daemon, falls back to direct Python.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import daemon_available, daemon_call_raw, get_brain, close_brain

try:
    if daemon_available():
        resp = daemon_call_raw("hook_idle_maintenance", {}, timeout=60.0)
        # Output is invisible anyway; stored as pending message inside daemon_hooks
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_idle_maintenance
        brain = get_brain()
        if brain:
            try:
                hook_idle_maintenance(brain, {}, [])
            except Exception:
                pass
            finally:
                close_brain(brain)
except Exception:
    pass
