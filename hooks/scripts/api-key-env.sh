#!/bin/bash
# brain — the ONE definition of "where does ANTHROPIC_API_KEY come from?".
#
# Sourced by brain-env.sh (every hook) and boot-brain.sh (which runs BEFORE
# brain-env is reachable — the bootstrap ordering that forced the copy in the
# first place). Both used to carry their own copy of the env-file source and
# the dual-casing CLAUDE_PLUGIN_OPTION fallback; a casing fix landing in only
# one of them re-creates the 2026-07-15 failure where a user filled in the
# plugin's key field and the daemon still ran keyless.
#
# Three functions, because the callers legitimately differ on WHEN they read:
#   brain_user_env_file               — the canonical config path
#   brain_source_user_env             — source it (set -a: every var exported)
#   brain_api_key_from_plugin_option  — userConfig fallback, both casings
#
# POLICY stays with the callers: brain-env.sh sources the file unconditionally
# because it needs every variable in it, not just the key; boot-brain.sh reads
# it only when the key is missing and mirrors a userConfig-resolved key back
# into the file for the launchd-spawned daemon.
#
# Written with `if` blocks and no bare `&&` statements: brain-daemon runs under
# `set -e` and sources this transitively, where a failing test as a standalone
# command would abort the whole resolver.

# The canonical user config file. One definition — every consumer that names
# this path (resolvers, installers, the boot notices) is naming the same file.
brain_user_env_file() {
    printf '%s' "${XDG_CONFIG_HOME:-$HOME/.config}/brain/env"
}

# Source the user config so secrets and identity tokens (ANTHROPIC_API_KEY,
# BRAIN_OPERATOR_NAME, BRAIN_AGENT_NAME, ...) propagate to everything
# downstream. NOTE: plain sourcing OVERWRITES an already-set shell value — a
# variable in this file always wins over the process env from here on.
brain_source_user_env() {
    local _brain_env_f
    _brain_env_f="${1:-$(brain_user_env_file)}"
    if [ -f "$_brain_env_f" ]; then
        set -a
        . "$_brain_env_f"
        set +a
    fi
    return 0
}

# Additive userConfig fallback: if the env file / shell didn't supply the key,
# take it from the plugin-config value CC injects as CLAUDE_PLUGIN_OPTION_<KEY>
# (per plugins-reference). Env file / shell still win. Both casings checked —
# the doc doesn't pin <KEY>'s case and a wrong name would be a silent no-op.
brain_api_key_from_plugin_option() {
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        if [ -n "${CLAUDE_PLUGIN_OPTION_API_KEY:-}" ]; then
            export ANTHROPIC_API_KEY="$CLAUDE_PLUGIN_OPTION_API_KEY"
        elif [ -n "${CLAUDE_PLUGIN_OPTION_api_key:-}" ]; then
            export ANTHROPIC_API_KEY="$CLAUDE_PLUGIN_OPTION_api_key"
        fi
    fi
    return 0
}
