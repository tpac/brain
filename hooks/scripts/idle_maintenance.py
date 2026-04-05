"""Idle maintenance — dream, consolidate, heal, tune, reflect.
Fires on Notification(idle_prompt). Notification stdout is NOT visible.
Thin client: sends hook_idle_maintenance to daemon, falls back to direct Python.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error

hook_input = get_hook_input()

try:
    if daemon_available():
        resp = daemon_call_raw("hook_idle_maintenance", {"session_id": hook_input.get("session_id", "")}, timeout=60.0)
        # Output is invisible anyway; stored as pending message inside daemon_hooks
    else:
        daemon_unavailable_error("idle_maintenance")
except Exception:
    pass
