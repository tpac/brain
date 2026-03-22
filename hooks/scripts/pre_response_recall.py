"""Pre-response recall — surfaces brain context before Claude responds.
Fires on UserPromptSubmit. Output: JSON {"additionalContext":"..."}.
Thin client: sends hook_recall to daemon, falls back to direct Python.
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, get_brain, close_brain, brain_debug, is_debug_mode

APPROVE = json.dumps({"decision": "approve"})

hook_input = get_hook_input()
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")

# Skip short, slash, or bang messages
if not user_message or len(user_message) < 5 or user_message.startswith("/") or user_message.startswith("!"):
    brain_debug("recall: skipped (short/slash/bang)")
    print(APPROVE)
    sys.exit(0)

def _debug_recall_result(result, latency_ms):
    """Log recall details when debug mode is on."""
    if not is_debug_mode():
        return
    j = result.get("json", {})
    ctx = j.get("additionalContext", "") or j.get("reason", "")
    decision = j.get("decision", "approve")
    # Count recalled items — each node line starts with "  [type]"
    import re
    node_count = len(re.findall(r'^\s+\[(?:rule|decision|concept|correction|lesson|procedure|person|project|task|mental_model|vocabulary|pattern|uncertainty)\]', ctx, re.MULTILINE))
    chars = len(ctx)
    brain_debug("recall: %d nodes, %d chars injected, %dms" % (node_count, chars, latency_ms))
    if reason:
        # Show first 200 chars of what was injected
        brain_debug("recall preview: %s" % reason[:200].replace("\n", " "))

t0 = time.time()
try:
    if daemon_available():
        resp = daemon_call_raw("hook_recall", {
            "prompt": hook_input.get("prompt", ""),
            "message": hook_input.get("message", ""),
        }, timeout=4.5)
        latency = (time.time() - t0) * 1000
        if resp.get("ok"):
            result = resp.get("result", {})
            _debug_recall_result(result, latency)
            if "json" in result:
                print(json.dumps(result["json"]))
            elif "output" in result:
                print(result["output"])
            else:
                print(APPROVE)
        else:
            print(APPROVE)
    else:
        # Direct Python fallback — import and run hook logic directly
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_recall
        brain = get_brain()
        if not brain:
            print(APPROVE)
            sys.exit(0)
        try:
            result = hook_recall(brain, hook_input, [])
            latency = (time.time() - t0) * 1000
            _debug_recall_result(result, latency)
            if "json" in result:
                print(json.dumps(result["json"]))
            elif "output" in result:
                print(result["output"])
            else:
                print(APPROVE)
        except Exception:
            print(APPROVE)
        finally:
            close_brain(brain)
except Exception:
    print(APPROVE)
