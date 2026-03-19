#!/bin/bash
# brain v4 — PostToolUse/Bash hook (lightweight host check)
# Hook: PostToolUse matcher=Bash — fires after bash commands complete.
# Detects environment changes from commands like pip install, brew, etc.
#
# Input: JSON on stdin with { session_id, tool_name, tool_input, tool_response }
# Output: stdout with host diff if environment changed

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SERVER_DIR="$PLUGIN_ROOT/servers"

# ── Resolve brain DB ──
DB_DIR=""
if [ -n "$BRAIN_DB_DIR" ] && [ -d "$BRAIN_DB_DIR" ]; then
  DB_DIR="$BRAIN_DB_DIR"
fi
if [ -z "$DB_DIR" ]; then
  for candidate in /sessions/*/mnt/AgentsContext/brain; do
    if [ -f "$candidate/brain.db" ]; then
      DB_DIR="$candidate"
      break
    fi
  done
fi
if [ -z "$DB_DIR" ] && [ -f "$HOME/AgentsContext/brain/brain.db" ]; then
  DB_DIR="$HOME/AgentsContext/brain"
fi
if [ -z "$DB_DIR" ] || [ ! -f "$DB_DIR/brain.db" ]; then
  exit 0
fi

export BRAIN_DB_DIR="$DB_DIR"
export BRAIN_SERVER_DIR="$SERVER_DIR"
export HOOK_INPUT=$(cat)

python3 -c '
import sys, os, json, re

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
except Exception:
    sys.exit(0)

# Only scan after commands that could change the environment
tool_input = hook_input.get("tool_input", {})
command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
if not command:
    sys.exit(0)

# Pattern match for environment-changing commands
ENV_CHANGE_PATTERNS = [
    r"\bpip\b.*\binstall\b", r"\bpip\b.*\buninstall\b",
    r"\bbrew\b.*\binstall\b", r"\bbrew\b.*\buninstall\b",
    r"\bapt\b.*\binstall\b", r"\bnpm\b.*\binstall\b",
    r"\bcargo\b.*\binstall\b", r"\bgem\b.*\binstall\b",
    r"\bpyenv\b", r"\bnvm\b.*\buse\b", r"\bnvm\b.*\binstall\b",
    r"\bconda\b.*\binstall\b", r"\bconda\b.*\bactivate\b",
]

is_env_change = any(re.search(pat, command, re.IGNORECASE) for pat in ENV_CHANGE_PATTERNS)
if not is_env_change:
    sys.exit(0)

try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    sys.exit(0)

try:
    env_result = brain.scan_host_environment()
    changes = env_result.get("changes", {}) if env_result else {}

    if changes:
        output_lines = ["HOST ENVIRONMENT CHANGED (after bash command):"]
        for key, change in changes.items():
            old_val = change.get("old", "?")
            new_val = change.get("new", "?")
            output_lines.append(f"  {key}: {old_val} → {new_val}")
        output_lines.append(f"  Command: {command[:100]}")
        output_lines.append("")
        print("\n".join(output_lines))

    brain.close()

except Exception:
    try:
        brain.close()
    except Exception:
        pass
' 2>/dev/null

exit 0
