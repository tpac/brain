#!/bin/bash
# brain — PostCompact hook: re-inject brain context after compaction.
# Output: brain state re-injection (injected into context)
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && exit 0
exec python3 "$(dirname "$0")/post_compact_reboot.py"
