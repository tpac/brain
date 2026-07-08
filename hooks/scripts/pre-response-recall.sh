#!/bin/bash
# brain — Pre-response recall: surfaces brain context before Claude responds.
# Fires on UserPromptSubmit.
# Output: JSON {"additionalContext":"..."} (injected into context)
source "$(dirname "$0")/resolve-brain-db.sh"
if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  # LOUD bail — this used to be a silent approve, which made "plugin enabled
  # but memory dead" invisible (first friend install: prompts never traced,
  # nobody knew). Tell Claude via additionalContext so it can tell the operator.
  echo "[brain-recall] no brain.db at '${BRAIN_DB_DIR:-unresolved}' — memory OFFLINE, turn not recorded" >&2
  _BRAIN_MISSING_PATH="${BRAIN_DB_DIR:-unresolved}" python3 -c '
import json, os
p = os.environ.get("_BRAIN_MISSING_PATH", "unresolved")
msg = ("[BRAIN] MEMORY OFFLINE — the brain plugin is enabled but no brain database "
       "exists (looked for brain.db under: %s). Recall and encoding are dead; nothing from "
       "this session is being remembered. Likely causes, in order: (1) ANTHROPIC_API_KEY "
       "missing — boot exits before creating the brain (it printed setup instructions at "
       "session start); (2) a stale BRAIN_DB_DIR override in ~/.config/brain/env pointing at "
       "the wrong place; (3) first boot was interrupted mid-install. Tell the operator "
       "plainly once — do not act as if memory works." % p)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                         "additionalContext": msg}}))
' 2>/dev/null || echo '{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"[BRAIN] MEMORY OFFLINE - brain plugin enabled but no brain.db found. Recall/encoding dead. Check ANTHROPIC_API_KEY and ~/.config/brain/env. Tell the operator."}}'
  exit 0
fi
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/pre_response_recall.py"
