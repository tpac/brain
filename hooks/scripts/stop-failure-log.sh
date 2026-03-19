#!/bin/bash
# brain v4 — StopFailure hook (failure mode logging)
# Hook: StopFailure — fires when a turn ends due to API error.
# Logs failures to brain for pattern detection via failure_mode consciousness.
#
# Input: JSON on stdin with { session_id, error, error_details, last_assistant_message }
# Output: none (logging only)

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
from datetime import datetime, timezone

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
    error_type = hook_input.get("error", "unknown")
    error_details = hook_input.get("error_details", "")
    session_id = hook_input.get("session_id", "")

    # Log as a miss event for failure pattern detection
    brain.log_miss(
        session_id=session_id,
        signal="api_failure",
        query=f"API error: {error_type}",
        expected_node_id=None,
        context=str(error_details)[:500]
    )

    brain.save()
    brain.close()

except Exception:
    try:
        brain.close()
    except Exception:
        pass
' 2>/dev/null

exit 0
