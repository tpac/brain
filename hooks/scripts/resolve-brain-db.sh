#!/bin/bash
# Shared brain DB resolution — sourced by all hook scripts.
# Boot sets BRAIN_DB_DIR once; other hooks reuse it or re-resolve.
#
# Usage: source "$(dirname "$0")/resolve-brain-db.sh"
# After sourcing: BRAIN_DB_DIR, BRAIN_SERVER_DIR, and PLUGIN_ROOT are set.
# If no brain.db found, BRAIN_DB_DIR is empty — caller should exit 0.

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BRAIN_SERVER_DIR="$PLUGIN_ROOT/servers"

# If BRAIN_DB_DIR already set and valid (e.g. from boot), skip resolution
if [ -n "$BRAIN_DB_DIR" ] && [ -f "$BRAIN_DB_DIR/brain.db" ]; then
  export BRAIN_DB_DIR BRAIN_SERVER_DIR PLUGIN_ROOT
  return 0 2>/dev/null || true
fi

# Full resolution chain (runs at boot or if env not set)
DB_DIR=""

# 1. Explicit override
if [ -n "$BRAIN_DB_DIR" ] && [ -d "$BRAIN_DB_DIR" ]; then
  DB_DIR="$BRAIN_DB_DIR"
fi

# 2. Cowork: search mounted AgentsContext directories
if [ -z "$DB_DIR" ]; then
  for candidate in /sessions/*/mnt/AgentsContext/brain; do
    if [ -f "$candidate/brain.db" ]; then
      DB_DIR="$candidate"
      break
    fi
  done
fi

# 3. Local Claude Code (symlink to Google Drive)
if [ -z "$DB_DIR" ] && [ -f "$HOME/AgentsContext/brain/brain.db" ]; then
  DB_DIR="$HOME/AgentsContext/brain"
fi

# 4. Cowork first-run: create in mounted AgentsContext
if [ -z "$DB_DIR" ]; then
  for ac_dir in /sessions/*/mnt/AgentsContext; do
    if [ -d "$ac_dir" ]; then
      DB_DIR="$ac_dir/brain"
      mkdir -p "$DB_DIR" 2>/dev/null
      break
    fi
  done
fi

BRAIN_DB_DIR="$DB_DIR"
export BRAIN_DB_DIR BRAIN_SERVER_DIR PLUGIN_ROOT
