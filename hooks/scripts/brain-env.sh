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

# Resolve plugin dir from whichever .sh sourced us.
# ${BASH_SOURCE:-$0}, NOT ${BASH_SOURCE[0]}: the subscripted form resolves to
# the CWD under zsh and is a fatal "Bad substitution" under dash, so a sourcer
# in either shell silently loaded the wrong tree — or died. Bare $BASH_SOURCE
# is element 0 in bash, and $0 is the correct fallback everywhere else.
# resolve-brain-db.sh carries the same idiom for the same reason.
_BRAIN_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
export PLUGIN_DIR="$(cd "$_BRAIN_ENV_DIR/../.." && pwd)"

# API-key + user-config resolution — owned by api-key-env.sh, shared with
# boot-brain.sh (which runs before this file is reachable).
# Readability-guarded: `.` on a missing file is a special-builtin failure that
# `|| true` cannot rescue — it exits the shell outright under dash and under
# `set -e` (brain-daemon sets it), before any resolution can happen. A damaged
# install must degrade, not take the daemon down.
if [ -r "$_BRAIN_ENV_DIR/api-key-env.sh" ]; then
    . "$_BRAIN_ENV_DIR/api-key-env.sh"
    # Sourced and used in the same branch: an undefined function is a 127 that
    # `set -e` (brain-daemon sets it) turns into a dead daemon, and a flag
    # recording "the source worked" is just this branch, spelled twice.
    brain_source_user_env
    brain_api_key_from_plugin_option
else
    echo "[brain-env] WARN: api-key-env.sh missing or unreadable (damaged install)" >&2
    echo "[brain-env] — user config and API key will NOT be loaded this run" >&2
fi

# Source the canonical user config (~/.config/brain/env) so secrets and
# identity tokens (ANTHROPIC_API_KEY, BRAIN_OPERATOR_NAME, BRAIN_AGENT_NAME, ...)
# propagate into both the hook scripts and the launchd-spawned daemon
# launcher. Unconditional here (unlike boot-brain.sh's key-only read): every
# variable in the file is wanted. A value in the file wins over the process
# env for everything downstream of this line — so a BRAIN_DB_DIR knob line
# re-points even a plist-baked daemon env; the resolver's ladder then
# re-confirms the same choice.
# Daemon rendezvous port — set EARLY, before the runtime-bootstrap guard below,
# since it depends only on the uid (not the venv). The ONE shell source of the
# per-user port; shell scripts + hook Python read $BRAIN_DAEMON_PORT (the formula
# survives only as a resilience fallback). An explicit value (shell / the user
# env above) wins. Python inside servers/ uses daemon_config.DAEMON_PORT.
export BRAIN_DAEMON_PORT="${BRAIN_DAEMON_PORT:-$((47200 + $(id -u) % 100))}"

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
