"""PreToolUse(Bash) — catches destructive commands before execution.
Fast regex pre-screen stays in client (avoids daemon round-trip for safe commands).
Only calls daemon/brain for destructive commands.
Output: JSON {"decision":"approve"|"block","reason":"..."}.
"""
import sys, os, json, re

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, get_brain, close_brain

APPROVE = json.dumps({"decision": "approve"})

hook_input = get_hook_input()
tool_input = hook_input.get("tool_input", {})
command = tool_input.get("command", "")

if not command:
    print(APPROVE)
    sys.exit(0)

# ── Fast regex pre-screen (stays in client — no daemon round-trip for safe commands) ──
DESTRUCTIVE_REGEXES = [
    r"rm\s+(-[rf]+\s+|.*--force)",
    r"git\s+worktree\s+remove",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[fd]",
    r"git\s+checkout\s+--\s",
    r"git\s+push\s+.*--force",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
    r"TRUNCATE",
    r"\brmdir\b",
    r"xargs\s+rm",
]

is_destructive = any(re.search(pat, command, re.IGNORECASE) for pat in DESTRUCTIVE_REGEXES)

if not is_destructive:
    print(APPROVE)
    sys.exit(0)

# ── Destructive command detected — call daemon/brain for safety check ──
try:
    if daemon_available():
        resp = daemon_call_raw("hook_pre_bash_safety", {"command": command}, timeout=7.0)
        if resp.get("ok"):
            result = resp.get("result", {})
            if "json" in result:
                print(json.dumps(result["json"]))
            else:
                print(json.dumps({
                    "decision": "approve",
                    "reason": "\u26a0\ufe0f Destructive command detected. Proceed carefully.",
                }))
        else:
            print(json.dumps({
                "decision": "approve",
                "reason": "\u26a0\ufe0f Destructive command detected. Safety check unavailable — proceed carefully.",
            }))
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_pre_bash_safety
        brain = get_brain()
        if not brain:
            print(json.dumps({
                "decision": "approve",
                "reason": "\u26a0\ufe0f Destructive command detected. Brain unavailable — proceed carefully.",
            }))
            sys.exit(0)
        try:
            result = hook_pre_bash_safety(brain, {"command": command}, [])
            if "json" in result:
                print(json.dumps(result["json"]))
            else:
                print(APPROVE)
        except Exception:
            print(json.dumps({
                "decision": "approve",
                "reason": "\u26a0\ufe0f Destructive command detected. Safety check failed — proceed carefully.",
            }))
        finally:
            close_brain(brain)
except Exception:
    print(json.dumps({
        "decision": "approve",
        "reason": "\u26a0\ufe0f Destructive command detected. Safety check error — proceed carefully.",
    }))
