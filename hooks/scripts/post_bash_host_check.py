"""PostToolUse(Bash) — detects env changes after pip/brew/etc.
PostToolUse stdout is NOT visible. Stores output as pending message.
Regex pre-screen stays in client to avoid daemon round-trip for non-env commands.
Thin client: sends hook_post_bash_host_check to daemon, falls back to direct Python.
"""
import sys, os, re

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, get_brain, close_brain

hook_input = get_hook_input()

# Only scan after commands that could change the environment
tool_input = hook_input.get("tool_input", {})
command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
if not command:
    sys.exit(0)

ENV_CHANGE_PATTERNS = [
    r"\bpip\b.*\binstall\b", r"\bpip\b.*\buninstall\b",
    r"\bbrew\b.*\binstall\b", r"\bbrew\b.*\buninstall\b",
    r"\bapt\b.*\binstall\b", r"\bnpm\b.*\binstall\b",
    r"\bcargo\b.*\binstall\b", r"\bgem\b.*\binstall\b",
    r"\bpyenv\b", r"\bnvm\b.*\buse\b", r"\bnvm\b.*\binstall\b",
    r"\bconda\b.*\binstall\b", r"\bconda\b.*\bactivate\b",
]

if not any(re.search(pat, command, re.IGNORECASE) for pat in ENV_CHANGE_PATTERNS):
    sys.exit(0)

try:
    if daemon_available():
        daemon_call_raw("hook_post_bash_host_check", {"command": command}, timeout=5.0)
    else:
        # Direct fallback
        parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from servers.daemon_hooks import hook_post_bash_host_check
        brain = get_brain()
        if brain:
            try:
                hook_post_bash_host_check(brain, {"command": command}, [])
            except Exception:
                pass
            finally:
                close_brain(brain)
except Exception:
    pass
