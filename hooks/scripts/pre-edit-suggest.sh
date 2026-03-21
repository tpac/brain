#!/bin/bash
# brain — PreToolUse(Edit|Write) hook: surfaces brain rules before edits.
# Output: JSON {"decision":"approve","reason":"..."} with brain suggestions
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && { echo '{"decision":"approve"}'; exit 0; }
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/pre_edit_suggest.py"
