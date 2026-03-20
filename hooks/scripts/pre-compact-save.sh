#!/bin/bash
# brain v5.1 — PreCompact hook
#
# Fires before context compaction. Time-constrained (10s timeout).
# Does THREE things as fast as possible:
#   1. Synthesize session (harvests from DB, includes confidence recalibration)
#   2. Write compaction boundary marker
#   3. Save brain state to disk
#
# NOTE: This hook may NOT fire during context overflow (as opposed to
# explicit /compact). The post-compact reboot and transcript rehydration
# are the safety nets for that case.

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  echo '{"decision":"approve"}'
  exit 0
fi

# ── Try daemon first (fast path) ──
source "$(dirname "$0")/daemon-client.sh"
if daemon_available; then
  daemon_send '{"cmd":"synthesize_session","args":{}}' 8 2>/dev/null
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  daemon_send '{"cmd":"remember","args":{"type":"context","title":"Compaction boundary at '"$TS"'","content":"Context compacted. Synthesis ran. Post-compact reboot will re-inject context.","keywords":"compaction boundary session handoff","locked":false}}' 3 2>/dev/null
  daemon_send '{"cmd":"save","args":{}}' 3 2>/dev/null
  echo '{"decision":"approve"}'
  exit 0
fi

# ── Direct Python fallback ──
# IMPORTANT: Do NOT suppress stderr — errors must be visible for debugging.
python3 -c '
import sys, os, json
from datetime import datetime, timezone

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    from servers.brain import Brain
    brain = Brain(db_path)

    # Synthesize session (includes confidence recalibration)
    try:
        synthesis = brain.synthesize_session()
        parts = []
        for key in ("decisions", "corrections", "teaching_arcs", "open_questions"):
            val = synthesis.get(key)
            if val:
                parts.append("%s %s" % (val, key))
        if parts:
            print("brain: pre-compact synthesis: " + ", ".join(parts), file=sys.stderr)
    except Exception as e:
        print("brain: synthesis error (non-fatal): %s" % e, file=sys.stderr)

    # Write compaction boundary marker
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    brain.remember(
        type="context",
        title="Compaction boundary at %s" % ts,
        content="Context compacted. Synthesis ran. Post-compact reboot will re-inject context.",
        keywords="compaction boundary session handoff",
        locked=False,
    )

    brain.save()
    brain.close()
except Exception as e:
    print("brain: pre-compact error: %s" % e, file=sys.stderr)
'

# Always approve — don't block compaction even if brain ops fail
echo '{"decision":"approve"}'
