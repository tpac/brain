#!/bin/bash
# brain v4 — SessionEnd hook
# Hook: SessionEnd — fires when session terminates.
# Ensures clean shutdown: WAL flush, bridge detection, DB close.
#
# Input: JSON on stdin with { session_id, reason, cwd, transcript_path }
# Output: none required (session is ending)

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

python3 -c '
import sys, os, json

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    from servers.brain import Brain
    brain = Brain(db_path)

    # Consolidate: strengthen frequent memories, detect bridges
    try:
        brain.consolidate()
    except Exception:
        pass

    # Flush WAL + close
    brain.save()
    brain.close()
except Exception as e:
    print(f"brain: session-end error: {e}", file=sys.stderr)
' 2>/dev/null
