#!/bin/bash
# brain — SessionStart hook: boots brain, prints context + consciousness signals.
# Output: full brain state for Claude's context (injected via SessionStart stdout)
#
# Brain DB resolution order:
# 1. BRAIN_DB_DIR env var (explicit override)
# 2. /sessions/*/mnt/AgentsContext/brain/ (Cowork mounted paths)
# 3. $HOME/AgentsContext/brain/ (local Claude Code via symlink)
# If none found, boot fails cleanly (no /tmp fallback — silent data loss is worse).

source "$(dirname "$0")/resolve-brain-db.sh"

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

# ── Start persistent daemon for fast subsequent hooks ──
# The daemon keeps Brain + embedder loaded in memory.
# Boot still runs direct Python (needs full output formatting),
# but all subsequent hooks (recall, track, edit-suggest) use the daemon.
python3 -c "
import sys, os
parent = os.path.dirname(os.environ.get('BRAIN_SERVER_DIR', ''))
if parent:
    sys.path.insert(0, parent)
try:
    from servers.daemon import ensure_daemon
    ensure_daemon(os.path.join(os.environ.get('BRAIN_DB_DIR', ''), 'brain.db'))
except Exception:
    pass  # Daemon is optional — hooks fall back to direct Python
" &

exec python3 "$(dirname "$0")/boot_brain.py"
