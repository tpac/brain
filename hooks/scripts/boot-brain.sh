#!/bin/bash
# brain — SessionStart hook: boots brain, prints context + consciousness signals.
# Output: full brain state for Claude's context (injected via SessionStart stdout)
#
# Brain DB resolution order:
# 1. BRAIN_DB_DIR env var (explicit override)
# 2. /sessions/*/mnt/AgentsContext/brain/ (Cowork mounted paths)
# 3. $HOME/AgentsContext/brain/ (local Claude Code via symlink)
# If none found, boot fails cleanly (no /tmp fallback — silent data loss is worse).


# Save stdin early — inline python commands below would consume it.
# Extract session_id from hook input and pass as env var.
HOOK_STDIN=$(cat)
BRAIN_HOOK_SESSION_ID=$(echo "$HOOK_STDIN" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('session_id',''))" 2>/dev/null)
export BRAIN_HOOK_SESSION_ID

source "$(dirname "$0")/resolve-brain-db.sh"

# ── Validate hooks.json schema ──
# Claude Code expects { "hooks": { ... } } wrapper but silently drops hooks
# if the schema is wrong (Zod parse failure caught and swallowed in pluginLoader).
# Catch this ourselves so a format mistake doesn't silently kill all hooks.
HOOKS_FILE="$PLUGIN_ROOT/hooks/hooks.json"
if [ -f "$HOOKS_FILE" ]; then
  if ! python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if 'hooks' not in data or not isinstance(data['hooks'], dict):
    print('ERROR: hooks.json missing top-level \"hooks\" key.', file=sys.stderr)
    print('Expected: {\"hooks\": {\"SessionStart\": [...], ...}}', file=sys.stderr)
    print('Got top-level keys: ' + ', '.join(data.keys()), file=sys.stderr)
    sys.exit(1)
" "$HOOKS_FILE" 2>&1; then
    echo "⚠️  hooks.json schema validation FAILED — hooks will NOT fire"
    echo "   Claude Code silently drops hooks when the wrapper format is wrong."
    echo "   Fix: wrap all hook events inside a top-level \"hooks\" key."
  fi
fi

# No DB found — guide the user
if [ -z "$BRAIN_DB_DIR" ]; then
  echo ""
  echo "brain: No brain.db found."
  echo ""
  echo "Two options:"
  echo ""
  echo "  1. CONNECT TO EXISTING BRAIN — Set the path to your brain folder:"
  echo "     In Claude Code settings or .claude/settings.json, add to env:"
  echo '       "BRAIN_DB_DIR": "/path/to/your/brain/folder"'
  echo "     The folder should contain (or will contain) brain.db."
  echo ""
  echo "  2. START FRESH — Create a new brain:"
  echo "     mkdir -p ~/AgentsContext/brain"
  echo "     Then restart this session. The brain will initialize automatically."
  echo ""
  echo "Searched locations:"
  echo "  - \$BRAIN_DB_DIR env var (not set)"
  echo "  - /sessions/*/mnt/AgentsContext/brain/ (Cowork — not found)"
  echo "  - \$HOME/AgentsContext/brain/ (not found)"
  echo ""
  exit 0
fi

# ── Start daemon via ensure_daemon() — fcntl-locked singleton ──
# No inline spawning. ensure_daemon() handles: lock, ping, spawn, code-change restart.
PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)" python3 -c "
from servers.daemon_client import ensure_daemon; import os, sys
db = os.path.join(os.environ.get('BRAIN_DB_DIR', ''), 'brain.db')
sys.stderr.write('[brain-boot] Daemon %s\n' % ('ready' if ensure_daemon(db) else 'FAILED'))
"

# ── Verify MCP server can start ──
# Claude Code starts the MCP server as a separate process. If it crashes (e.g. import error),
# Claude silently gets no brain tools. Catch that here and scream.
MCP_CRASH=""
PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)" python3 -c "
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('$PLUGIN_ROOT')), ''))
try:
    from servers.brain_mcp import TOOLS
    sys.stderr.write('[brain-boot] MCP server OK — %d tools available\n' % len(TOOLS))
except Exception as e:
    # Write to stdout so it appears in session context
    print('⚠️  MCP SERVER BROKEN — Anchor will have NO direct brain tools')
    print('   Error: %s' % e)
    print('   The brain hooks still work, but recall/remember/connect MCP tools are dead.')
    print('   Fix the import error in servers/brain_mcp.py and restart the session.')
    sys.exit(1)
" 2>&1
MCP_EXIT=$?
if [ $MCP_EXIT -ne 0 ]; then
  # Also check for crash sentinel from a previous failed startup
  if [ -f /tmp/brain-mcp-crash.txt ]; then
    echo ""
    echo "Last MCP crash details:"
    head -5 /tmp/brain-mcp-crash.txt
  fi
fi

# Mark as booted, then run
touch "$BOOT_FLAG"
exec python3 "$(dirname "$0")/boot_brain.py"
