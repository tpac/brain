"""WorktreeCreate — tracks git branch/worktree info in brain.
Thin client: sends hook_worktree_context to daemon, falls back to direct Python.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, get_brain, close_brain

hook_input = get_hook_input()

try:
    if daemon_available():
        resp = daemon_call_raw("hook_worktree_context", hook_input, timeout=5.0)
        if resp.get("ok"):
            output = resp.get("result", {}).get("output", "")
            if output:
                print(output)
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_worktree_context
        brain = get_brain()
        if brain:
            try:
                result = hook_worktree_context(brain, hook_input, [])
                output = result.get("output", "")
                if output:
                    print(output)
            except Exception:
                pass
            finally:
                close_brain(brain)
except Exception:
    pass
