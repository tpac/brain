#!/bin/bash
# Shared brain DB resolution — sourced by all hook scripts.
# Boot sets BRAIN_DB_DIR once; other hooks reuse it or re-resolve.
#
# Usage: source "$(dirname "$0")/resolve-brain-db.sh"
# After sourcing: BRAIN_DB_DIR, BRAIN_SERVER_DIR, PLUGIN_ROOT, BRAIN_PYTHON are set.
# If no brain.db found, BRAIN_DB_DIR is empty — caller should exit 0.

# Ensure the isolated runtime is installed and BRAIN_PYTHON / PATH point at it.
# First hook invocation on a fresh install blocks ~15s downloading uv + Python
# + deps. Every subsequent invocation is ~8ms (sentinel fast path).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/brain-env.sh"

# Always resolve from script location — never use CLAUDE_PLUGIN_ROOT cache.
# The cache can be stale (old code, old socket protocol). Working dir is truth.
PLUGIN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BRAIN_SERVER_DIR="$PLUGIN_ROOT/servers"

# If BRAIN_DB_DIR already set and valid (e.g. from boot), skip resolution
if [ -n "$BRAIN_DB_DIR" ] && [ -f "$BRAIN_DB_DIR/brain.db" ]; then
  export BRAIN_DB_DIR BRAIN_SERVER_DIR PLUGIN_ROOT
  return 0 2>/dev/null || true
fi

# Full resolution chain (runs at boot or if env not set).
#
# Priority is: existing-brain-found > standard-location > Cowork > legacy.
# Auto-create only happens at the standard or Cowork location, never at the
# legacy path (existing users who never migrated must opt in explicitly).
DB_DIR=""

# 1. Explicit override
if [ -n "$BRAIN_DB_DIR" ] && [ -d "$BRAIN_DB_DIR" ]; then
  DB_DIR="$BRAIN_DB_DIR"
fi

# 2. Cowork: search mounted AgentsContext directories for an existing brain
if [ -z "$DB_DIR" ] && [ -d "/sessions" ]; then
  for candidate in /sessions/*/mnt/AgentsContext/brain; do
    [ -f "$candidate/brain.db" ] 2>/dev/null && DB_DIR="$candidate" && break
  done 2>/dev/null
fi

# 3. Standard Claude Code plugin data location ($CLAUDE_PLUGIN_DATA is set
#    by Claude Code per-plugin and survives plugin updates — the documented
#    convention for plugin-owned runtime state).
if [ -z "$DB_DIR" ] && [ -n "$CLAUDE_PLUGIN_DATA" ] && [ -f "$CLAUDE_PLUGIN_DATA/brain/brain.db" ]; then
  DB_DIR="$CLAUDE_PLUGIN_DATA/brain"
fi

# 4. Legacy local path (~/AgentsContext/brain/) — supported for
#    pre-CLAUDE_PLUGIN_DATA installs. New installs land at $CLAUDE_PLUGIN_DATA.
if [ -z "$DB_DIR" ] && [ -f "$HOME/AgentsContext/brain/brain.db" ]; then
  DB_DIR="$HOME/AgentsContext/brain"
fi

# 5. Standard first-run: create at $CLAUDE_PLUGIN_DATA/brain
if [ -z "$DB_DIR" ] && [ -n "$CLAUDE_PLUGIN_DATA" ]; then
  DB_DIR="$CLAUDE_PLUGIN_DATA/brain"
  mkdir -p "$DB_DIR" 2>/dev/null
fi

# 6. Cowork first-run: create in mounted AgentsContext
if [ -z "$DB_DIR" ] && [ -d "/sessions" ]; then
  for ac_dir in /sessions/*/mnt/AgentsContext; do
    if [ -d "$ac_dir" ] 2>/dev/null; then
      DB_DIR="$ac_dir/brain"
      mkdir -p "$DB_DIR" 2>/dev/null
      break
    fi
  done 2>/dev/null
fi

BRAIN_DB_DIR="$DB_DIR"
export BRAIN_DB_DIR BRAIN_SERVER_DIR PLUGIN_ROOT
