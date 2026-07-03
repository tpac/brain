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

# Recall variant — laf_v1 is the LAF challenger scorer (§19 P1): maxsim +
# episodic pick/enc + idf + situation lanes (servers/recall_laf.py). Gate:
# eval/laf/p1_gate.md (2026-07-02) — 16%/23% need@5/@25 vs champion 11%/17%,
# 2.3× faster p50. Read by the DAEMON (brain_recall._recall_impl) — takes
# effect at daemon restart. Rollback: remove this line and restart.
export BRAIN_RECALL_VARIANT="laf_v1"

# S1 Scribe lived-sequence input — ON activates the v28/v29 encoder rebuild:
# XML lived-sequence timeline (<other>/<me> + tool actions + provenance),
# widened catalog, facts-only scout (temporal+quote retired), inline scout
# notes, `## Arc`/`## Review` residue. Paired with s1e active=v29 (medium
# effort). Gate: LongMemEval do-no-harm A/B 2026-07-03 — raw pass 70%→77%,
# encode-miss 6→0, temporal held 1.0 (brain finding bab8d86a). Read by the
# DAEMON's S1 Scribe (encode._lived_sequence_enabled) — takes effect at
# daemon restart. Rollback: set to "" (or remove) + set_interaction_active
# s1e 25, then restart.
export BRAIN_S1E_LIVED_SEQUENCE="1"
