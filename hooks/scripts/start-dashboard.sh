#!/bin/bash
# brain — Dashboard launcher for macOS LaunchAgent.
#
# Called by ~/Library/LaunchAgents/com.brain.dashboard.plist as the single
# entry point. exec's the dashboard process directly so launchd's KeepAlive
# sees the real dashboard PID — which makes the dashboard a global SINGLETON:
# exactly one process, ever, on a fixed port, shared by every Claude Code
# session (the dashboard's "All sessions" view is global anyway).
#
# The dashboard is the daemon's companion, NOT a per-chat preview server. It
# was previously spawned per-session via .claude/launch.json, which leaked an
# orphaned process + port for every session. Do NOT re-add it there.
#
# Lives in the repo so the Python version + boot sequence follow the codebase,
# not a hand-edited plist. To change how the dashboard starts, edit this file.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Same env every hook uses. Sets BRAIN_PYTHON + PLUGIN_DIR, prepends the venv
# to PATH, runs ensure-runtime.sh on fresh installs (idempotent + fast-path).
# shellcheck disable=SC1091
source "$SCRIPT_DIR/brain-env.sh"

# Fail fast with a specific error if the venv Python is missing. Without this,
# launchd would respawn every 10s in a confusing loop.
if [ ! -x "$BRAIN_PYTHON" ]; then
    echo "[start-dashboard] FATAL: venv python not executable at $BRAIN_PYTHON" >&2
    echo "[start-dashboard] Run: $SCRIPT_DIR/ensure-runtime.sh" >&2
    exit 1
fi

# Port + DB dir come from the plist EnvironmentVariables (DASHBOARD_PORT,
# BRAIN_DB_DIR). exec so launchd supervises the dashboard directly — KeepAlive
# restarts the dashboard, not this shell, when it exits.
exec "$BRAIN_PYTHON" "$PLUGIN_DIR/dashboard/brain_dashboard_standalone.py"
