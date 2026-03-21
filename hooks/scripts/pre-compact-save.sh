#!/bin/bash
# brain — PreCompact hook: synthesize session + compaction boundary + save.
# Output: JSON {"decision":"approve"} (must not block compaction)
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && { echo '{"decision":"approve"}'; exit 0; }
exec python3 "$(dirname "$0")/pre_compact_save.py"
