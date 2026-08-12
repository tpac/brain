#!/bin/bash
# brain — the ONE definition of "how a LaunchAgent gets materialized, loaded,
# and reloaded on drift". Sourced by install-daemon-service.sh and
# ensure-dashboard.sh; the label is an ARGUMENT (D-11: the service layer never
# renames, and a helper that bakes one label is a helper for one service).
#
# The ritual was written twice and had already drifted: only the daemon side
# re-verified after bootstrap, and the dashboard's first-install branch
# rendered straight over the target instead of through the compared temp file.
#
# MECHANISM here, POLICY in the callers — they legitimately differ:
#   install-daemon-service.sh  install-only; verifies with launchctl afterwards
#   ensure-dashboard.sh        also probes liveness, kickstarts a loaded-but-down
#                              service, and verifies by waiting for :PORT
#
# Requires resolve-brain-db.sh to have been sourced first: the identity guard
# reads the two durable adoption channels through that file's
# brain_config_knob_db_dir / brain_config_option_db_dir. If it was not, both
# read empty and the guard keeps the INSTALLED brain dir — the safe direction
# (never hijack a service onto another brain), never a silent re-point.

brain_launchd_domain() {
    printf 'gui/%s' "$(id -u)"
}

brain_launchd_target() {
    printf '%s/Library/LaunchAgents/%s.plist' "$HOME" "$1"
}

# True iff launchd has this label loaded in the user's GUI domain.
brain_launchd_managed() {
    launchctl print "$(brain_launchd_domain)/$1" >/dev/null 2>&1
}

# Materialize the template for THIS machine into $6 (real paths, no tokens).
# Rendering is deterministic, so the caller can diff the result against the
# installed copy to detect drift without touching launchd.
#
# Identity preservation — an already-installed plist names the tree that OWNS
# the service and the brain it serves; re-materialization converges the
# TEMPLATE (shape, env set, launchd timing) but must not silently re-point
# either:
#   PLUGIN_DIR    hooks run from the installed plugin copy while a dev
#                 machine's daemon is launchd-pinned to the repo — rendering
#                 with the caller's own tree would flip service ownership every
#                 boot (and ping-pong back on repo-side runs). Keep the
#                 installed tree while its launcher still exists; a vanished
#                 tree (uninstall) falls back to the caller's.
#   BRAIN_DB_DIR  only the durable adoption channels (the ~/.config/brain/env
#                 knob or the userConfig brain path) may re-point an installed
#                 service's brain. An ephemeral shell BRAIN_DB_DIR (eval runs,
#                 isolated copies) must never hijack the shared service onto a
#                 temp brain.
# The installed-tree extraction is launcher-name-agnostic (anchored on
# /hooks/scripts/ — only ProgramArguments carries that path shape) so a plist
# written BEFORE a launcher rename still yields its tree; validity is then
# checked against the launcher the template ships, because that is what the
# re-materialized plist will exec.
brain_launchd_render() {
    # $1 label  $2 template  $3 launcher  $4 plugin_dir  $5 db_dir  $6 out
    local _label="$1" _template="$2" _launcher="$3"
    local _plugin_dir="$4" _db_dir="$5" _out="$6"
    local _target _installed_dir _installed_db _knob _opt
    _target="$(brain_launchd_target "$_label")"

    if [ -f "$_target" ]; then
        _installed_dir="$(sed -n 's|.*<string>\(.*\)/hooks/scripts/[^<]*</string>.*|\1|p' "$_target" | head -1)"
        if [ -n "$_installed_dir" ] && [ -x "$_installed_dir/hooks/scripts/$_launcher" ]; then
            _plugin_dir="$_installed_dir"
        fi
        _installed_db="$(sed -n '/<key>BRAIN_DB_DIR<\/key>/{n;s|.*<string>\(.*\)</string>.*|\1|p;}' "$_target" | head -1)"
        if [ -n "$_installed_db" ] && [ "$_installed_db" != "$_db_dir" ]; then
            _knob="$(brain_config_knob_db_dir 2>/dev/null)"
            _opt="$(brain_config_option_db_dir 2>/dev/null)"
            if [ "$_db_dir" != "$_knob" ] && [ "$_db_dir" != "$_opt" ]; then
                _db_dir="$_installed_db"
            fi
        fi
    fi

    sed -e "s|__PLUGIN_DIR__|$_plugin_dir|g" \
        -e "s|__BRAIN_DB_DIR__|$_db_dir|g" \
        "$_template" > "$_out"
}

# Replace an ALREADY-LOADED service with the rendered plist. `kickstart` is not
# enough: launchd keeps the EnvironmentVariables it read at bootstrap time, so
# the plist must be re-bootstrapped for a template change or a brain-location
# adoption to reach the running service.
#
# Unload FIRST and VERIFY it unloaded before touching the file — a still-loaded
# service keeps its bootstrap-time env, so overwriting the plist under it would
# leave "file current, launchd stale", where every future diff blesses the
# divergence and it never heals. Returns 1 in exactly that case (installed
# plist untouched, drift retries next run); the caller decides how loud to be.
brain_launchd_reinstall() {
    # $1 label  $2 rendered  $3 log prefix
    local _label="$1" _rendered="$2" _prefix="$3"
    local _domain _target _bootstrapped=""
    _domain="$(brain_launchd_domain)"
    _target="$(brain_launchd_target "$_label")"

    launchctl bootout "$_domain/$_label" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5; do
        launchctl print "$_domain/$_label" >/dev/null 2>&1 || break
        sleep 1
    done
    if launchctl print "$_domain/$_label" >/dev/null 2>&1; then
        echo "$_prefix WARN: bootout did not unload $_label — keeping installed plist (drift retries next run)" >&2
        return 1
    fi

    cp "$_rendered" "$_target"
    for _ in 1 2 3; do
        launchctl bootstrap "$_domain" "$_target" >/dev/null 2>&1 && { _bootstrapped=1; break; }
        sleep 1
    done
    [ -n "$_bootstrapped" ] || launchctl load -w "$_target" >/dev/null 2>&1 || true
    return 0
}

# First install: no service loaded, so there is nothing to unload and nothing
# to preserve. `load -w` is the fallback for launchd versions/contexts where
# bootstrap is unavailable; both are swallowed, so callers that care must
# VERIFY afterwards rather than assume.
brain_launchd_install_fresh() {
    # $1 label  $2 rendered
    local _label="$1" _rendered="$2"
    local _domain _target
    _domain="$(brain_launchd_domain)"
    _target="$(brain_launchd_target "$_label")"

    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$_rendered" "$_target"
    launchctl bootstrap "$_domain" "$_target" >/dev/null 2>&1 \
        || launchctl load -w "$_target" >/dev/null 2>&1 || true
}
