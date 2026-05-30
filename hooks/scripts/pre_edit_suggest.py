"""PreToolUse(Edit|Write) — surfaces brain rules/suggestions before edits.
Thin client: sends hook_pre_edit to daemon, falls back to direct Python.
Output: JSON {"decision":"approve","reason":"..."}.
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, brain_debug, is_debug_mode, log_hook_error

APPROVE = json.dumps({"decision": "approve"})

hook_input = get_hook_input()
tool_input = hook_input.get("tool_input", {})
file_path = tool_input.get("file_path", "")
tool_name = hook_input.get("tool_name", "Edit")

if not file_path:
    print(APPROVE)
    sys.exit(0)

filename = os.path.basename(file_path)

# Skip non-source files
skip_exts = [".log", ".map", ".lock", ".json"]
if any(filename.endswith(ext) for ext in skip_exts) and filename != "package.json":
    brain_debug("suggest: skipped %s (non-source)" % filename)
    print(APPROVE)
    sys.exit(0)

t0 = time.time()
try:
    if daemon_available():
        resp = daemon_call_raw("hook_pre_edit", {
            "filename": filename,
            "tool_name": tool_name,
            "session_id": hook_input.get("session_id", ""),
        }, timeout=7.0)
        latency = (time.time() - t0) * 1000
        if resp.get("ok"):
            result = resp.get("result", {})
            if "json" in result:
                j = result["json"]
                reason = j.get("reason", "")
                brain_debug("suggest: %s → %d chars, %dms" % (filename, len(reason), latency))
                print(json.dumps(j))
            else:
                brain_debug("suggest: %s → no rules, %dms" % (filename, latency))
                print(APPROVE)
        else:
            print(APPROVE)
    else:
        print(json.dumps({"decision": "approve", "reason": daemon_unavailable_error("pre_edit_suggest")}))
except Exception as e:
    log_hook_error("pre_edit_suggest", e, "hook exception")
    print(APPROVE)
