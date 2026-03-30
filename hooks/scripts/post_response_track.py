"""Post-response tracker: store exchange + gate encoding agent.
Fires on Stop. Stores conversation to message stream and sets stop_agent_prompt
config every 5th stop so the Stop agent hook runs encoding.
Thin client: sends hook_post_response_track to daemon.
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, brain_debug, is_debug_mode, log_hook_output

hook_input = get_hook_input()

# Stop hook does NOT include user's message — only last_assistant_message.
# Extract user message from transcript file if available.
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")
if not user_message and hook_input.get("transcript_path"):
    try:
        import json as _json
        with open(os.path.expanduser(hook_input["transcript_path"])) as _tf:
            lines = _tf.readlines()
        # Walk backwards to find last user message
        for _line in reversed(lines):
            try:
                entry = _json.loads(_line.strip())
                if entry.get("type") in ("human", "user"):
                    # Extract text from message content
                    msg = entry.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            texts = [
                                p.get("text", "") for p in content
                                if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
                            ]
                            if texts:
                                user_message = " ".join(texts)
                                break  # Found actual text, stop looking
                            # If no text parts, this is a tool result — keep looking
                            continue
                        elif isinstance(content, str) and content:
                            user_message = content
                            break
                    elif isinstance(msg, str) and msg:
                        user_message = msg
                        break
            except Exception as e:
                print('[brain] ERROR track_transcript_entry_parse: %s' % e, file=sys.stderr)
                continue
    except Exception as e:
        print('[brain] ERROR track_transcript_read: %s' % e, file=sys.stderr)
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
            "prompt": user_message,
            "message": hook_input.get("message", ""),
            "hook_event_name": event_name,
            "last_assistant_message": last_msg,
        }, timeout=55.0)  # Encoding agent runs ~49s on every 5th stop
        latency = (time.time() - t0) * 1000
        if resp.get("ok"):
            output = resp.get("result", {}).get("output", "")
            brain_debug("track: completed in %dms%s" % (latency, ", output=%d chars" % len(output) if output else ""))
            log_hook_output("stop", output_text=output or "(store_exchange + encoding gate ran)")
            if output:
                print(output)
        else:
            brain_debug("track: daemon returned ok=false")
    else:
        daemon_unavailable_error("post_response_track")
except Exception as e:
    print('[brain] ERROR post_response_track: %s' % e, file=sys.stderr)
