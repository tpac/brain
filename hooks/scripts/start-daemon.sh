#!/bin/bash
# brain — Daemon launcher for macOS LaunchAgent.
#
# Called by ~/Library/LaunchAgents/com.brain.daemon.plist as the single
# entry point. exec's the daemon process directly so launchd's KeepAlive
# sees the real daemon PID.
#
# Lives in the repo so the Python version + boot sequence follow the
# codebase, not a hand-edited plist. To change how the daemon starts,
# edit this file — not ~/Library/LaunchAgents/.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source the same env every hook uses. Sets BRAIN_PYTHON + prepends
# $PLUGIN_DIR/venv/bin to PATH. Also runs ensure-runtime.sh on fresh
# installs (idempotent + fast-path via sentinel).
# shellcheck disable=SC1091
source "$SCRIPT_DIR/brain-env.sh"

# Fail fast with a specific error if the venv Python is missing. Without
# this, launchd would respawn every 10s in a confusing loop.
if [ ! -x "$BRAIN_PYTHON" ]; then
    echo "[start-daemon] FATAL: venv python not executable at $BRAIN_PYTHON" >&2
    echo "[start-daemon] Run: $SCRIPT_DIR/ensure-runtime.sh" >&2
    exit 1
fi

# BRAIN_DB_DIR must be set (by launchd plist EnvironmentVariables, or
# externally). No /tmp fallback — silent data loss is worse.
if [ -z "$BRAIN_DB_DIR" ]; then
    echo "[start-daemon] FATAL: BRAIN_DB_DIR not set" >&2
    exit 1
fi

DB_PATH="$BRAIN_DB_DIR/brain.db"

# exec so launchd supervises the daemon directly (not this shell).
# KeepAlive restarts the daemon, not the shell, when it exits.
exec "$BRAIN_PYTHON" -c "
import sys, os
sys.path.insert(0, '$PLUGIN_DIR')
from servers.daemon_server import BrainDaemon
BrainDaemon('$DB_PATH').start()
"
