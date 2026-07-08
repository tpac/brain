#!/bin/bash
# brain — Pre-response recall: surfaces brain context before Claude responds.
# Fires on UserPromptSubmit.
# Output: JSON {"additionalContext":"..."} (injected into context)
#
# Pure shim: env setup + exec. ALL policy (daemon liveness, unconfigured-install
# ANCHOR OFFLINE, recovery) lives in the python client / hook_common — the
# daemon owns everything behind it. Do not add gates here.
source "$(dirname "$0")/resolve-brain-db.sh"
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/pre_response_recall.py"
