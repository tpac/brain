"""Pre-response recall — thin wrapper that calls daemon for recall + judge.

The daemon handles everything: Layer 1 recall, Layer 2 Haiku judge, Layer 3 graph expansion.
This script just passes the user message to the daemon and prints the result.

Flow:
1. Send user message to daemon via "hook_recall" command
2. Daemon does recall → judge → graph expand → formats additionalContext
3. This script prints the result (additionalContext or approve)
"""
import sys, os, json, time

_t0 = time.time()
sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw,
                         daemon_unavailable_error, brain_debug, log_hook_output)
from datetime import datetime as _dt
def _ts(): return _dt.now().strftime("%H:%M:%S.%f")[:-3]
sys.stderr.write("[recall-hook %s] import: %dms\n" % (_ts(), (time.time() - _t0) * 1000))

APPROVE = json.dumps({"decision": "approve"})

hook_input = get_hook_input()
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")

# Skip short, slash, or bang messages
if not user_message or len(user_message) < 5 or user_message.startswith("/") or user_message.startswith("!"):
    brain_debug("recall: skipped (short/slash/bang)")
    print(APPROVE)
    sys.exit(0)

t0 = time.time()
try:
    if not daemon_available():
        err = daemon_unavailable_error("recall")
        log_hook_output("recall", output_text="(daemon unavailable)", user_prompt=user_message)
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": err}}))
        sys.exit(0)

    # Call daemon — it handles Layer 1 + Layer 2 judge + Layer 3 graph expand
    resp = daemon_call_raw("hook_recall", {
        "prompt": hook_input.get("prompt", ""),
        "message": hook_input.get("message", ""),
        "session_id": hook_input.get("session_id", ""),
    }, timeout=20.0)  # Slightly under hook timeout (21s) to avoid race
                       # 2026-05-02: bumped 14→20 to cover Haiku tail latency
                       # under load. See FRAME-DESIGN.md and node 2340b053.

    if not resp.get("ok"):
        err_msg = resp.get("error", "unknown error")
        log_hook_output("recall", output_text="(daemon error: %s)" % err_msg, user_prompt=user_message)
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext":
            "[BRAIN]\n⚠️ RECALL FAILED: %s\nThe brain could not search for relevant memories.\n[/BRAIN]" % err_msg}}))
        sys.exit(0)

    result = resp.get("result", {})
    elapsed = int((time.time() - t0) * 1000)

    # The daemon returns either additionalContext (judge completed) or approve (no results/judge failed)
    result_json = result.get("json", {})
    if "additionalContext" in result_json:
        context = result_json["additionalContext"]
        log_hook_output("recall", output_text=context, user_prompt=user_message)
        brain_debug("recall: daemon returned context (%d chars) in %dms" % (len(context), elapsed))
        sys.stderr.write("[recall-hook %s] total: %dms, printing and exiting\n" % (_ts(), (time.time() - _t0) * 1000))
        sys.stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}))
        sys.stdout.flush()
        os._exit(0)  # Fast exit — skip Python cleanup
    else:
        log_hook_output("recall", output_text="(approve: %s)" % result_json.get("decision", "?"),
                       user_prompt=user_message)
        brain_debug("recall: daemon returned approve in %dms" % elapsed)
        sys.stdout.write(json.dumps(result_json))
        sys.stdout.flush()
        os._exit(0)

except Exception as e:
    log_hook_output("recall", output_text="(exception) %s" % e, user_prompt=user_message)
    brain_debug("recall: exception: %s" % e)
    print(APPROVE)
