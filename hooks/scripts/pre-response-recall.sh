#!/bin/bash
# brain — Pre-response recall: surfaces brain context before Claude responds.
# Fires on UserPromptSubmit.
# Output: JSON {"additionalContext":"..."} (injected into context)
source "$(dirname "$0")/resolve-brain-db.sh"
if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  # Plugin enabled but no brain — a REAL hook error, not simulated context.
  # exit 1 = non-blocking hook error: Claude Code surfaces it, the prompt
  # still goes through. Also logged to hook_errors via hook_common (Layer 1,
  # works without Brain; lands in brain_logs.db next to where the brain
  # should be, when the dir exists).
  BRAIN_HOOK_DIR="$(dirname "$0")" python3 -c '
import os, sys
sys.path.insert(0, os.environ["BRAIN_HOOK_DIR"])
from hook_common import log_hook_error
p = os.environ.get("BRAIN_DB_DIR") or "unresolved"
log_hook_error("pre-response-recall",
               "MEMORY OFFLINE - no brain.db at %s" % p,
               context="plugin enabled but brain missing: recall+encode dead this session. "
                       "Likely causes in order: ANTHROPIC_API_KEY missing (boot exits before "
                       "creating the brain); stale BRAIN_DB_DIR override in ~/.config/brain/env; "
                       "first install interrupted.")
' || echo "[brain-recall] MEMORY OFFLINE — no brain.db at '\''${BRAIN_DB_DIR:-unresolved}'\''" >&2
  exit 1
fi
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/pre_response_recall.py"
