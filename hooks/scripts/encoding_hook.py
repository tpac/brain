"""Encoding hook — thin client that sends Stop event to daemon via TCP.

The daemon handles encoding in a background thread (non-blocking).
This script returns immediately so the Stop hook doesn't timeout.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, brain_debug

hook_input = get_hook_input()

if not daemon_available():
    brain_debug("encoding: daemon unavailable, skipping")
    sys.exit(0)

# Send to daemon's hook_post_response_track (which handles counter + encoding)
# The daemon already increments stop_counter and fires encoding every 5th stop.
# We just need to make sure the exchange is stored and the counter increments.
resp = daemon_call_raw("hook_post_response_track", {
    "prompt": hook_input.get("prompt", "") or hook_input.get("message", ""),
    "last_assistant_message": hook_input.get("last_assistant_message", ""),
}, timeout=5.0)

if not resp.get("ok"):
    brain_debug("encoding: daemon error: %s" % resp.get("error", "?"))
