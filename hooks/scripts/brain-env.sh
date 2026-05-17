#!/bin/bash
# brain — shared environment setup, sourced by every hook .sh
#
# After sourcing:
#   $PLUGIN_DIR    resolves to plugin root
#   $BRAIN_PYTHON  points to the venv's python (the ONLY python hooks use)
#   $PATH          has $PLUGIN_DIR/venv/bin prepended so `python3` resolves there too
#
# First invocation triggers ensure-runtime.sh (blocks ~60-90s on fresh install).
# Subsequent invocations are instant — just PATH + env var wiring.

# Resolve plugin dir from whichever .sh sourced us
_BRAIN_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PLUGIN_DIR="$(cd "$_BRAIN_ENV_DIR/../.." && pwd)"

# Ensure runtime is installed (idempotent, fast-path on sentinel)
if ! "$_BRAIN_ENV_DIR/ensure-runtime.sh"; then
    echo "[brain-env] FATAL: runtime bootstrap failed — brain disabled" >&2
    # Don't `exit` — we're sourced. Let the calling hook handle it.
    return 1 2>/dev/null || exit 1
fi

# Wire the venv as the authoritative Python
export BRAIN_PYTHON="$PLUGIN_DIR/venv/bin/python"
export PATH="$PLUGIN_DIR/venv/bin:$PATH"

# Ensure nothing in the shell environment overrides venv resolution
unset PYTHONHOME

# Surface variant — v5_agentic enables the Haiku tool-use loop (recall_*,
# expand_node, etc.) plus the final-round force-select code path. Without
# this, the registered surface prompt runs under the legacy v4 single-shot
# path and tools never fire. Rollback: unset this var and restart the daemon.
export BRAIN_SURFACE_VARIANT="v5_agentic"
