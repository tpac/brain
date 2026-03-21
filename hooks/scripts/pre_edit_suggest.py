"""PreToolUse(Edit|Write) — surfaces brain rules/suggestions before edits.
Thin client: sends hook_pre_edit to daemon, falls back to direct Python.
Output: JSON {"decision":"approve","reason":"..."}.
"""
import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, get_brain, close_brain

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
    print(APPROVE)
    sys.exit(0)

try:
    if daemon_available():
        resp = daemon_call_raw("hook_pre_edit", {
            "filename": filename,
            "tool_name": tool_name,
        }, timeout=7.0)
        if resp.get("ok"):
            result = resp.get("result", {})
            if "json" in result:
                print(json.dumps(result["json"]))
            else:
                print(APPROVE)
        else:
            print(APPROVE)
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_pre_edit
        brain = get_brain()
        if not brain:
            print(APPROVE)
            sys.exit(0)
        try:
            result = hook_pre_edit(brain, {"filename": filename, "tool_name": tool_name}, [])
            if "json" in result:
                print(json.dumps(result["json"]))
            else:
                print(APPROVE)
        except Exception:
            print(APPROVE)
        finally:
            close_brain(brain)
except Exception:
    print(APPROVE)
