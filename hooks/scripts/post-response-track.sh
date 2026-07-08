#!/bin/bash
# brain — Post-response tracker: vocab gap detection + encoding checkpoints.
# Fires on UserPromptSubmit and Stop events.
# Output: encoding checkpoint text (visible on UserPromptSubmit, pending on Stop)
source "$(dirname "$0")/resolve-brain-db.sh"
if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  # Real hook error (see pre-response-recall.sh) — logged + non-blocking exit 1.
  BRAIN_HOOK_DIR="$(dirname "$0")" python3 -c '
import os, sys
sys.path.insert(0, os.environ["BRAIN_HOOK_DIR"])
from hook_common import log_hook_error
p = os.environ.get("BRAIN_DB_DIR") or "unresolved"
log_hook_error("post-response-track", "MEMORY OFFLINE - no brain.db at %s" % p,
               context="turn not recorded")
' || echo "[brain-stop] MEMORY OFFLINE — no brain.db at '\''${BRAIN_DB_DIR:-unresolved}'\''" >&2
  exit 1
fi
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/post_response_track.py"
