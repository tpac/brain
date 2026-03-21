#!/bin/bash
# brain v4 — ConfigChange hook (host awareness)
# Hook: ConfigChange — fires when settings/skills/config files change.
# This IS the "host changed" signal — brain detects environment changes.
#
# Input: JSON on stdin with { session_id, source, file_path }
# Output: stdout with host diff (injected into context if significant)

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  exit 0
fi
export HOOK_INPUT=$(cat)

python3 -c '
import sys, os, json

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
except Exception:
    sys.exit(0)

try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    sys.exit(0)

try:
    source = hook_input.get("source", "unknown")
    file_path = hook_input.get("file_path", "")

    # Scan host environment and diff against last known state
    env_result = brain.scan_host_environment()
    changes = env_result.get("changes", {}) if env_result else {}

    output_lines = []

    if changes:
        output_lines.append("HOST ENVIRONMENT CHANGED (detected by brain):")
        for key, change in changes.items():
            old_val = change.get("old", "?")
            new_val = change.get("new", "?")
            output_lines.append(f"  {key}: {old_val} → {new_val}")
        output_lines.append(f"  Trigger: config change in {source}")
        if file_path:
            output_lines.append(f"  File: {file_path}")
        output_lines.append("")
        output_lines.append("Review arch_constraint and capability nodes that may be affected.")

    # ConfigChange stdout is NOT injected into context.
    # Store output as pending message for next UserPromptSubmit recall.
    if output_lines:
        summary = "\n".join(output_lines)
        try:
            existing = brain.get_config("pending_hook_messages", "[]")
            pending = json.loads(existing) if existing else []
        except Exception:
            pending = []
        pending.append(summary)
        pending = pending[-5:]
        brain.set_config("pending_hook_messages", json.dumps(pending))
        brain.save()

    brain.close()

except Exception:
    try:
        brain.close()
    except Exception:
        pass
' 2>/dev/null

exit 0
