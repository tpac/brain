#!/bin/bash
# Encoding hook — fires on Stop via command hook.
# Sends hook input to daemon via TCP for encoding agent processing.
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] && exit 0

export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/encoding_hook.py"
