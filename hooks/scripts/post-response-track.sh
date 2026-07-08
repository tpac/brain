#!/bin/bash
# brain — Post-response tracker: vocab gap detection + encoding checkpoints.
# Fires on UserPromptSubmit and Stop events.
# Output: encoding checkpoint text (visible on UserPromptSubmit, pending on Stop)
#
# Pure shim: env setup + exec. ALL policy (daemon liveness, unconfigured-install
# ANCHOR OFFLINE, recovery) lives in the python client / hook_common — the
# daemon owns everything behind it. Do not add gates here.
source "$(dirname "$0")/resolve-brain-db.sh"
export HOOK_INPUT=$(cat)
exec python3 "$(dirname "$0")/post_response_track.py"
