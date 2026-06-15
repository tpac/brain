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

# Source the canonical user config (~/.config/brain/env) so secrets and
# identity tokens (ANTHROPIC_API_KEY, BRAIN_OPERATOR_NAME, BRAIN_AGENT_NAME, ...)
# propagate into both the hook scripts and the launchd-spawned daemon
# launcher. set -a exports each loaded variable; explicit shell-level
# values still win (we don't override an already-set var).
_BRAIN_USER_ENV="${XDG_CONFIG_HOME:-$HOME/.config}/brain/env"
if [ -f "$_BRAIN_USER_ENV" ]; then
    set -a
    . "$_BRAIN_USER_ENV"
    set +a
fi

# Additive userConfig fallback: if the env file / shell didn't supply the key,
# take it from the plugin-config value CC injects as CLAUDE_PLUGIN_OPTION_<KEY>
# (per plugins-reference). Env file / shell still win. Both casings checked —
# the doc doesn't pin <KEY>'s case and a wrong name would be a silent no-op.
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    if [ -n "${CLAUDE_PLUGIN_OPTION_API_KEY:-}" ]; then
        export ANTHROPIC_API_KEY="$CLAUDE_PLUGIN_OPTION_API_KEY"
    elif [ -n "${CLAUDE_PLUGIN_OPTION_api_key:-}" ]; then
        export ANTHROPIC_API_KEY="$CLAUDE_PLUGIN_OPTION_api_key"
    fi
fi

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
