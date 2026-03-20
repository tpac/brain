#!/bin/bash
# brain v4 — SessionEnd hook
# Hook: SessionEnd — fires when session terminates.
# Ensures clean shutdown: WAL flush, bridge detection, DB close.
#
# Input: JSON on stdin with { session_id, reason, cwd, transcript_path }
# Output: none required (session is ending)

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  exit 0
fi

# ── Try daemon first (fast path) ──
source "$(dirname "$0")/daemon-client.sh"
if daemon_available; then
  # Synthesize session before shutdown
  daemon_send '{"cmd":"synthesize_session","args":{}}' 10 >/dev/null 2>&1
  # Consolidate via daemon
  daemon_send '{"cmd":"consolidate","args":{}}' 30 >/dev/null 2>&1
  # Save and shutdown daemon (saves + closes brain + removes socket/PID)
  daemon_send '{"cmd":"save","args":{}}' 5 >/dev/null 2>&1
  daemon_send '{"cmd":"shutdown","args":{}}' 5 >/dev/null 2>&1
  exit 0
fi

# ── Direct Python fallback ──
python3 -c '
import sys, os, json

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    from servers.brain import Brain
    brain = Brain(db_path)

    # Synthesize session before shutdown
    try:
        synthesis = brain.synthesize_session()
        syn_id = synthesis.get("id")
        if syn_id:
            parts = []
            if synthesis.get("decisions"): parts.append(str(synthesis["decisions"]) + " decisions")
            if synthesis.get("corrections"): parts.append(str(synthesis["corrections"]) + " corrections")
            if synthesis.get("node_summary"): parts.append(synthesis["node_summary"])
            if parts:
                print("brain: session synthesis: " + ", ".join(parts), file=sys.stderr)
    except Exception as e:
        print("brain: synthesis error (non-fatal): " + str(e), file=sys.stderr)

    # Consolidate: strengthen frequent memories, detect bridges
    try:
        brain.consolidate()
    except Exception:
        pass

    # Flush WAL + close
    brain.save()
    brain.close()
except Exception as e:
    print("brain: session-end error: " + str(e), file=sys.stderr)
'
