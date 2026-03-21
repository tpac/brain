#!/bin/bash
# brain — Post-response tracker: vocab gap detection + encoding checkpoints.
# Fires on UserPromptSubmit and Stop events.
# Output: encoding checkpoint text (visible on UserPromptSubmit, pending on Stop)
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && exit 0
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/post_response_track.py"
