#!/bin/bash
# brain — WorktreeCreate hook: tracks git branch/worktree info.
# Output: git context info (injected into context)
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && exit 0
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/worktree_context.py"
