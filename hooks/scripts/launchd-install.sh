#!/bin/bash
# brain — the ONE definition of "how a LaunchAgent gets materialized, loaded,
# and reloaded on drift". Sourced by install-daemon-service.sh and
# ensure-dashboard.sh; the label is an ARGUMENT (D-11: the service layer never
# renames, and a helper that bakes one label is a helper for one service).
#
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

# Computed once at source time, not per call: `$(id -u)` is a fork + exec, and
# install-daemon-service.sh runs SYNCHRONOUSLY on the SessionStart budget.
_BRAIN_LAUNCHD_DOMAIN="gui/$(id -u)"

brain_launchd_domain() {
    printf '%s' "$_BRAIN_LAUNCHD_DOMAIN"
}

brain_launchd_target() {
    printf '%s/Library/LaunchAgents/%s.plist' "$HOME" "$1"
}

# True iff launchd has this label loaded in the user's GUI domain.
brain_launchd_managed() {
    launchctl print "$_BRAIN_LAUNCHD_DOMAIN/$1" >/dev/null 2>&1
}

# Entities (BRAIN_INSTANCE set — eval brains, install smoke runs) never use
# launchd: they run daemon_server as a child process on their own port. The
# loaded service is uid-wide while the plist path is $HOME-relative, so an
# entity under a scratch $HOME reads "no plist at my path" as drift and would
# bootout + re-bootstrap PRODUCTION's job onto its own tree — persistent
# across reboots. Every caller checks this first; the mutators below refuse
# regardless, so a caller that forgets cannot do the damage.
brain_launchd_entity() {
    [ -n "${BRAIN_INSTANCE:-}" ]
}
_brain_launchd_refuse_entity() {
    # $1 what  $2 log prefix — true (and loud) when the caller must stop
    brain_launchd_entity || return 1
    echo "${2:-[launchd-install]} refusing to $1 under BRAIN_INSTANCE=$BRAIN_INSTANCE — entities never touch launchd" >&2
    return 0
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
    # $1 label  $2 template  $3 launcher  $4 plugin_dir  $5 db_dir  $6 out  $7 log prefix
    local _label="$1" _template="$2" _launcher="$3"
    local _plugin_dir="$4" _db_dir="$5" _out="$6" _prefix="$7"
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
            # The XDG service dir counts as a durable target alongside the
            # two config channels: it is the D-13 service-owned default, so
            # a resolver that landed there did so via a real brain.db — the
            # relocation flow deliberately needs no pointer, and without
            # this arm the guard would pin both plists to the vacated path
            # forever (cmp then blesses the divergence every boot).
            if [ "$_db_dir" != "$_knob" ] && [ "$_db_dir" != "$_opt" ] \
               && [ "$_db_dir" != "${XDG_DATA_HOME:-$HOME/.local/share}/brain" ]; then
                _db_dir="$_installed_db"
            fi
        fi
    fi

    # A `|` in either path makes sed exit non-zero having written a PARTIAL
    # file, and an `&` makes it succeed while re-emitting the token literally.
    # Either way the result is not an installable plist, and an unchecked
    # failure is indistinguishable from drift — the caller would boot out a
    # working service to install the broken file. So: check both, and say which
    # path shape did it, here where that knowledge belongs rather than in two
    # callers' error strings.
    if ! sed -e "s|__PLUGIN_DIR__|$_plugin_dir|g" \
             -e "s|__BRAIN_DB_DIR__|$_db_dir|g" \
             "$_template" > "$_out"; then
        echo "$_prefix FATAL: could not render $_label.plist (a '|' in a path?)" >&2
        return 1
    fi
    # `case` on a builtin read, not `grep`: one fewer fork on the SessionStart
    # path, and it dodges grep's exit-2-on-unreadable-file, which a `! grep -q`
    # would have inverted into "render fine".
    case "$(<"$_out")" in
        *__PLUGIN_DIR__*|*__BRAIN_DB_DIR__*)
            echo "$_prefix FATAL: $_label.plist still holds unsubstituted tokens (an '&' in a path?)" >&2
            return 1 ;;
    esac
    return 0
}

# Replace an ALREADY-LOADED service with the rendered plist. `kickstart` is not
# enough: launchd keeps the EnvironmentVariables it read at bootstrap time, so
# the plist must be re-bootstrapped for a template change or a brain-location
# adoption to reach the running service.
#
# Unload a service and VERIFY it actually unloaded. Bootout alone is
# fire-and-forget; a service that survives it keeps its bootstrap-time env
# and its open files. Returns 1 when the service is still loaded — the
# caller decides how loud to be. Also the door for out-of-band consumers
# (relocate-brain.sh) that must stop a service without copying this ritual.
brain_launchd_unload() {
    # $1 label  $2 log prefix
    local _label="$1" _prefix="$2"
    _brain_launchd_refuse_entity "unload $_label" "$_prefix" && return 1
    launchctl bootout "$_BRAIN_LAUNCHD_DOMAIN/$_label" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5; do
        brain_launchd_managed "$_label" || break
        sleep 1
    done
    if brain_launchd_managed "$_label"; then
        echo "$_prefix WARN: bootout did not unload $_label" >&2
        return 1
    fi
    return 0
}

# Unload FIRST and VERIFY it unloaded before touching the file — a still-loaded
# service keeps its bootstrap-time env, so overwriting the plist under it would
# leave "file current, launchd stale", where every future diff blesses the
# divergence and it never heals. Returns 1 in exactly that case (installed
# plist untouched, drift retries next run); the caller decides how loud to be.
brain_launchd_reinstall() {
    # $1 label  $2 rendered  $3 log prefix
    local _label="$1" _rendered="$2" _prefix="$3"
    _brain_launchd_refuse_entity "reinstall $_label" "$_prefix" && return 1

    if ! brain_launchd_unload "$_label" "$_prefix"; then
        echo "$_prefix WARN: keeping installed plist (drift retries next run)" >&2
        return 1
    fi

    _brain_launchd_load "$_label" "$_rendered"
    return 0
}

# First install: no service loaded, so there is nothing to unload and nothing
# to preserve.
brain_launchd_install_fresh() {
    # $1 label  $2 rendered
    _brain_launchd_refuse_entity "install $1" && return 1
    _brain_launchd_load "$1" "$2"
}

# Install the rendered plist and hand it to launchd — the tail both paths share.
# `load -w` is the fallback for contexts where `bootstrap` is unavailable; both
# are swallowed, so callers that care must VERIFY afterwards rather than assume.
# Retried because a just-booted-out label can take a moment to free up; a first
# install does not need the patience, but one loading routine that is
# occasionally patient beats two that drift apart.
_brain_launchd_load() {
    local _label="$1" _rendered="$2" _target _bootstrapped=""
    _target="$(brain_launchd_target "$_label")"

    mkdir -p "$(dirname "$_target")"
    cp "$_rendered" "$_target"
    for _ in 1 2 3; do
        launchctl bootstrap "$_BRAIN_LAUNCHD_DOMAIN" "$_target" >/dev/null 2>&1 && { _bootstrapped=1; break; }
        sleep 1
    done
    [ -n "$_bootstrapped" ] || launchctl load -w "$_target" >/dev/null 2>&1 || true
}
