"""Post-response tracker: vocab gap detection + encoding checkpoints.
Fires on UserPromptSubmit AND Stop.
Thin client: sends hook_post_response_track to daemon, falls back to direct Python.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, get_brain, close_brain

hook_input = get_hook_input()

# UserPromptSubmit provides "prompt", Stop provides "last_assistant_message"
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")
event_name = hook_input.get("hook_event_name", "")
has_user_message = user_message and len(user_message) >= 10

# If no user message and not a Stop event, nothing to do
if not has_user_message and event_name != "Stop":
    sys.exit(0)

try:
    if daemon_available():
        resp = daemon_call_raw("hook_post_response_track", {
            "prompt": hook_input.get("prompt", ""),
            "message": hook_input.get("message", ""),
            "hook_event_name": event_name,
            "last_assistant_message": hook_input.get("last_assistant_message", ""),
        }, timeout=3.0)
        if resp.get("ok"):
            output = resp.get("result", {}).get("output", "")
            if output:
                print(output)
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_post_response_track
        brain = get_brain()
        if brain:
            try:
                result = hook_post_response_track(brain, hook_input, [])
                output = result.get("output", "")
                if output:
                    print(output)
            except Exception:
                pass
            finally:
                close_brain(brain)
except Exception:
    pass
