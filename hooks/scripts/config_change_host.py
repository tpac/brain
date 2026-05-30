"""ConfigChange — detects host environment changes.
ConfigChange stdout is NOT visible. Stores output as pending message.
Thin client: sends hook_config_change_host to daemon, falls back to direct Python.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, run_hook

hook_input = get_hook_input()

def main():
    if daemon_available():
        daemon_call_raw("hook_config_change_host", hook_input, timeout=5.0)
    else:
        daemon_unavailable_error("config_change_host")

run_hook("config_change_host", main)
