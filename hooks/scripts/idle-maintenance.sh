#!/bin/bash
# brain — Idle maintenance: dream, consolidate, heal, tune, reflect.
# Fires on Notification(idle_prompt).
# Output: none (stores pending message for next recall)
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] && exit 0
exec python3 "$(dirname "$0")/idle_maintenance.py"
