#!/bin/bash
# brain — PreToolUse(Bash) hook: catches destructive commands before execution.
# Output: JSON {"decision":"approve"|"block","reason":"..."}
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && { echo '{"decision":"approve"}'; exit 0; }
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/pre_bash_safety.py"
