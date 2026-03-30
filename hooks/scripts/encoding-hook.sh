#!/bin/bash
# Encoding hook — fires on Stop, POSTs to daemon's /encoding-hook endpoint.
# Uses type:"command" (proven to work) instead of type:"http" (untested).
# The daemon returns immediately (encoding runs in background thread).
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] && exit 0

PORT=$((47200 + $(id -u) % 100 + 1))
export HOOK_INPUT=$(cat)

# 3s curl timeout — daemon responds in <100ms, encoding runs async
curl -s --max-time 3 -X POST "http://127.0.0.1:${PORT}/encoding-hook" \
  -H "Content-Type: application/json" \
  -d "$HOOK_INPUT" 2>/dev/null || true
