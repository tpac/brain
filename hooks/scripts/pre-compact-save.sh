#!/bin/bash
# brain v14 (serverless) — PreCompact hook
#
# Pre-compact just SAVES. No HTTP server needed.
# The post-compact log reader (extract-session-log.py, called at SessionStart)
# handles retroactive encoding by reading the JSONL chat logs.
#
# This hook does exactly two things:
#   1. Save brain state to disk (commit WAL)
#   2. Write a compaction boundary node (lightweight marker)

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
  echo '{"decision":"approve"}'
  exit 0
fi

export BRAIN_DB_DIR="$DB_DIR"
export BRAIN_SERVER_DIR="$SERVER_DIR"

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
    from servers.brain import Brain
    brain = Brain(db_path)

    # Write compaction boundary marker
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    brain.remember(
        type="context",
        title=f"Compaction boundary at {ts}",
        content="Context compacted. The post-compact log reader will handle retroactive encoding at next session start.",
        keywords="compaction boundary session handoff",
        locked=False,
    )

    # Save brain state (critical operation)
    brain.save()
    brain.close()
except Exception as e:
    print(f"brain: pre-compact error: {e}", file=sys.stderr)
' 2>/dev/null

# Approve compaction — don't fight it
echo '{"decision":"approve"}'
