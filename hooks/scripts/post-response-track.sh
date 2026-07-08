#!/bin/bash
# brain — Post-response tracker: vocab gap detection + encoding checkpoints.
# Fires on UserPromptSubmit and Stop events.
# Output: encoding checkpoint text (visible on UserPromptSubmit, pending on Stop)
source "$(dirname "$0")/resolve-brain-db.sh"
if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  # Hook can't see a brain — but the daemon may still own a live one (the
  # daemon, not this hook, writes traces). If it answers: proceed normally —
  # the turn IS recorded and recall works through the daemon — and log a
  # path-mismatch WARNING to hook_errors. If it doesn't: nothing can record
  # this turn — ANCHOR OFFLINE, real hook error, non-blocking exit 1.
  if BRAIN_HOOK_DIR="$(dirname "$0")" python3 -c '
import os, sys
sys.path.insert(0, os.environ["BRAIN_HOOK_DIR"])
from hook_common import daemon_available
sys.exit(0 if daemon_available() else 1)
' 2>/dev/null; then
    BRAIN_HOOK_DIR="$(dirname "$0")" python3 -c '
import os, sys
sys.path.insert(0, os.environ["BRAIN_HOOK_DIR"])
from hook_common import log_hook_error
p = os.environ.get("BRAIN_DB_DIR") or "unresolved"
log_hook_error("post-response-track",
               "ANCHOR PATH MISMATCH - hook resolved %r (no brain.db) but the daemon is alive" % p,
               context="hooks and daemon disagree on BRAIN_DB_DIR - check the ~/.config/brain/env "
                       "override vs the daemon spawn env. Proceeding through the daemon; turn is recorded.",
               level="warning")
'
  else
    BRAIN_HOOK_DIR="$(dirname "$0")" python3 -c '
import os, sys
sys.path.insert(0, os.environ["BRAIN_HOOK_DIR"])
from hook_common import log_hook_error
p = os.environ.get("BRAIN_DB_DIR") or "unresolved"
log_hook_error("post-response-track",
               "ANCHOR OFFLINE - no brain.db at %s and the daemon is not answering" % p,
               context="turn not recorded")
' || echo "[post-response-track] ANCHOR OFFLINE — no brain.db at '"'"'${BRAIN_DB_DIR:-unresolved}'"'"' and no daemon" >&2
    exit 1
  fi
fi

export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/post_response_track.py"
