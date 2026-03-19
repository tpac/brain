#!/bin/bash
# brain v4 — ConfigChange hook (host awareness)
# Hook: ConfigChange — fires when settings/skills/config files change.
# This IS the "host changed" signal — brain detects environment changes.
#
# Input: JSON on stdin with { session_id, source, file_path }
# Output: stdout with host diff (injected into context if significant)

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
import sys, os, json

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

try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    sys.exit(0)

try:
    source = hook_input.get("source", "unknown")
    file_path = hook_input.get("file_path", "")

    # Scan host environment and diff against last known state
    env_result = brain.scan_host_environment()
    changes = env_result.get("changes", {}) if env_result else {}

    output_lines = []

    if changes:
        output_lines.append("HOST ENVIRONMENT CHANGED (detected by brain):")
        for key, change in changes.items():
            old_val = change.get("old", "?")
            new_val = change.get("new", "?")
            output_lines.append(f"  {key}: {old_val} → {new_val}")
        output_lines.append(f"  Trigger: config change in {source}")
        if file_path:
            output_lines.append(f"  File: {file_path}")
        output_lines.append("")
        output_lines.append("Review arch_constraint and capability nodes that may be affected.")

    brain.close()

    if output_lines:
        print("\n".join(output_lines))

except Exception:
    try:
        brain.close()
    except Exception:
        pass
' 2>/dev/null

exit 0
