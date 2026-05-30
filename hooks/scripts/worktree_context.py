"""WorktreeCreate — tracks git branch/worktree info in brain.
Thin client: sends hook_worktree_context to daemon, falls back to direct Python.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, run_hook

hook_input = get_hook_input()

def main():
    if daemon_available():
        resp = daemon_call_raw("hook_worktree_context", hook_input, timeout=5.0)
        if resp.get("ok"):
            output = resp.get("result", {}).get("output", "")
            if output:
                print(output)
    else:
        daemon_unavailable_error("worktree_context")

run_hook("worktree_context", main)
