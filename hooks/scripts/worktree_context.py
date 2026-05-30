"""WorktreeCreate — tracks git branch/worktree info in brain.
Thin client: sends hook_worktree_context to daemon, falls back to direct Python.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, log_hook_error

hook_input = get_hook_input()

try:
    if daemon_available():
        resp = daemon_call_raw("hook_worktree_context", hook_input, timeout=5.0)
        if resp.get("ok"):
            output = resp.get("result", {}).get("output", "")
            if output:
                print(output)
    else:
        daemon_unavailable_error("worktree_context")
except Exception as e:
    log_hook_error("worktree_context", e, "hook exception")
