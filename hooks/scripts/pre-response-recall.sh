#!/bin/bash
# brain — Pre-response recall: surfaces brain context before Claude responds.
# Fires on UserPromptSubmit.
# Output: JSON {"additionalContext":"..."} (injected into context)
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && { echo '{"decision":"approve"}'; exit 0; }
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/pre_response_recall.py"
