#!/bin/bash
# brain v4 — WorktreeRemove hook (git context cleanup)
# Hook: WorktreeRemove — fires when a worktree is removed.
# Clears worktree context from brain config.
#
# Input: JSON on stdin with { session_id, worktree_path, cwd, transcript_path }
# Output: none

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  exit 0
fi

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

    worktree_path = hook_input.get("worktree_path", "")

    # Clear worktree context
    brain.set_config("current_worktree", "")
    brain.set_config("current_branch", "")
    brain.set_config("current_cwd", "")

    brain.save()
    brain.close()
except Exception:
    try:
        brain.close()
    except Exception:
        pass
'

exit 0
