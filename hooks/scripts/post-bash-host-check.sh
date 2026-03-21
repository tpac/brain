#!/bin/bash
# brain — PostToolUse(Bash) hook: detects env changes after pip/brew/etc.
# Output: none (stores pending message for next recall)
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && exit 0
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/post_bash_host_check.py"
