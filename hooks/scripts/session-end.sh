#!/bin/bash
# brain — SessionEnd hook: session synthesis + consolidation + clean shutdown.
# Output: none (session is ending)
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && exit 0
exec python3 "$(dirname "$0")/session_end.py"
