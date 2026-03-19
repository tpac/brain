#!/bin/bash
# brain v4 — WorktreeCreate hook (git context tracking)
# Hook: WorktreeCreate — fires when a worktree is created.
# Stores git branch/worktree info as brain context for host awareness.
#
# Input: JSON on stdin with { session_id, name, cwd, transcript_path }
# Output: stdout with git context info (injected into context)

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
    worktree_name = hook_input.get("name", "unknown")
    cwd = hook_input.get("cwd", "")

    # Detect git branch from cwd
    branch = "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
    except Exception:
        pass

    # Store as brain config for host awareness
    brain.set_config("current_worktree", worktree_name)
    brain.set_config("current_branch", branch)
    brain.set_config("current_cwd", cwd)

    # Scan host environment to pick up the change
    brain.scan_host_environment()

    output_lines = [
        "GIT CONTEXT (brain is tracking):",
        "  Worktree: " + worktree_name,
        "  Branch: " + branch,
        "  CWD: " + cwd,
    ]

    brain.save()
    brain.close()
    print("\n".join(output_lines))

except Exception:
    try:
        brain.close()
    except Exception:
        pass
'

exit 0
