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

# Resolve the brain location through the SAME ladder every hook uses
# (resolve-brain-db.sh sources brain-env.sh: BRAIN_PYTHON, PATH, user env,
# ensure-runtime.sh on fresh installs). The plist-baked BRAIN_DB_DIR arrives
# in our environment and becomes the ladder's fast-path HINT — confirmed only
# if brain.db is actually there — not a verdict. A daemon relaunched after
# the user adopted a new brain location follows the ladder to the new path
# instead of writing the frozen plist snapshot forever (D-13 step 4a).
# `|| true`: the resolver is not errexit-safe (guarded mkdirs); a resolution
# failure surfaces below as empty BRAIN_DB_DIR / missing BRAIN_PYTHON, with
# the resolver's own stderr kept.
# NO_PERSIST: the daemon consumes the resolution, it is not an authority on
# it — a relaunch running off baked plist state must not overwrite the
# resolved.env record hooks maintain.
export BRAIN_RESOLVE_NO_PERSIST=1
# shellcheck disable=SC1091
source "$SCRIPT_DIR/resolve-brain-db.sh" || true

# Fail fast with a specific error if the venv Python is missing. Without
# this, launchd would respawn every 10s in a confusing loop.
if [ ! -x "$BRAIN_PYTHON" ]; then
    echo "[start-daemon] FATAL: venv python not executable at $BRAIN_PYTHON" >&2
    echo "[start-daemon] Run: $SCRIPT_DIR/ensure-runtime.sh" >&2
    exit 1
fi

# The ladder found nothing (no baked env, no existing brain, no persisted
# record). No /tmp fallback — silent data loss is worse.
if [ -z "$BRAIN_DB_DIR" ]; then
    echo "[start-daemon] FATAL: BRAIN_DB_DIR unresolved" >&2
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
