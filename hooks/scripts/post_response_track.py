"""Post-response tracker: vocab gap detection + encoding checkpoints.
Fires on Stop only (removed from UserPromptSubmit — it fired before Claude
responded, so last_assistant_message was always empty, breaking precision loop).
Thin client: sends hook_post_response_track to daemon, falls back to direct Python.
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, brain_debug, is_debug_mode, log_hook_output

hook_input = get_hook_input()

# UserPromptSubmit provides "prompt", Stop provides "last_assistant_message"
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")
event_name = hook_input.get("hook_event_name", "")
has_user_message = user_message and len(user_message) >= 10

# If no user message and not a Stop event, nothing to do
if not has_user_message and event_name != "Stop":
    brain_debug("track: skipped (no message, not Stop)")
    sys.exit(0)

t0 = time.time()
try:
    if daemon_available():
        last_msg = hook_input.get("last_assistant_message", "")
        brain_debug("track: event=%s, user_msg=%d chars, assistant_msg=%d chars" % (
            event_name or "UserPromptSubmit", len(user_message), len(last_msg)))
        resp = daemon_call_raw("hook_post_response_track", {
            "prompt": hook_input.get("prompt", ""),
            "message": hook_input.get("message", ""),
            "hook_event_name": event_name,
            "last_assistant_message": last_msg,
        })
        latency = (time.time() - t0) * 1000
        if resp.get("ok"):
            output = resp.get("result", {}).get("output", "")
            brain_debug("track: completed in %dms%s" % (latency, ", output=%d chars" % len(output) if output else ""))
            log_hook_output("stop", output_text=output or "(precision + auto-encode ran, no visible output)")
            if output:
                print(output)
        else:
            brain_debug("track: daemon returned ok=false")
    else:
        daemon_unavailable_error("post_response_track")
except Exception:
    pass
