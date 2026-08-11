#!/bin/bash
# brain — ensure the daemon's launchd service is INSTALLED (macOS). Idempotent.
#
# Division of labor: the daemon's LIVENESS is owned by ensure_daemon() (Python)
# + launchd KeepAlive + the ping nets. This script does the ONE thing they can't
# — provision the LaunchAgent on a fresh install, so launchd manages the daemon
# from the first boot instead of it coming up as ensure_daemon's detached
# no-launchd fallback (no KeepAlive, no boot persistence). It is a mirror of
# ensure-dashboard.sh, narrowed to install-only. See docs/DAEMON-LIFECYCLE-ARCH-PLAN.md
# Step 7.
#
# Called from boot-brain.sh BEFORE ensure_daemon(), so once installed,
# ensure_daemon sees launchd managing the daemon (manages() → True) and routes
# (re)starts through `launchctl kickstart -k` rather than direct-spawning a rival.
#
#   managed, installed plist current      → no-op (every non-fresh boot)
#   managed, installed plist DRIFTED      → re-materialize + re-bootstrap (the
#                                           installed plist is a frozen snapshot;
#                                           a template change or a brain-location
#                                           adoption must reach it — D-13 step 4c)
#   not managed, macOS                    → materialize the plist template for THIS
#                                           machine + launchctl bootstrap it
#                                           (RunAtLoad starts it; ensure_daemon
#                                           converges on it next in boot)
#   non-macOS                             → no-op (no launchd; ensure_daemon's
#                                           detached spawn is the only off-macOS path)
#
# NOTE: no `set -u` — resolve-brain-db.sh (sourced below) is not nounset-safe.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LABEL="com.brain.daemon"
TEMPLATE="$SCRIPT_DIR/$LABEL.plist"

# No launchd off macOS — ensure_daemon's detached spawn owns that path.
[ "$(uname -s)" = "Darwin" ] || exit 0

DOMAIN="gui/$(id -u)"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"

# Resolve BRAIN_DB_DIR the same way every launch path does, so the installed
# plist points at the right brain. boot-brain.sh already resolved + exported it
# before calling us; this re-source fast-paths (runtime sentinel is set by now,
# so no ensure-runtime recursion) and just confirms the value.
source "$SCRIPT_DIR/resolve-brain-db.sh" >/dev/null 2>&1 || true

[ -f "$TEMPLATE" ] || { echo "[install-daemon-service] FATAL: plist template missing: $TEMPLATE" >&2; exit 1; }
[ -n "${BRAIN_DB_DIR:-}" ] || { echo "[install-daemon-service] FATAL: BRAIN_DB_DIR unresolved — not installing" >&2; exit 1; }

# Identity preservation: an already-installed plist names the tree that OWNS
# the service (ProgramArguments) and the brain it serves. Re-materialization
# converges the TEMPLATE (shape, env set, launchd timing) but must not
# silently re-point identity:
#  - PLUGIN_DIR: hooks run from the installed plugin copy while a dev
#    machine's daemon is launchd-pinned to the repo — rendering with the
#    caller's own tree would flip service ownership every boot (and ping-pong
#    back on repo-side runs). Keep the installed tree while its launcher
#    still exists; a vanished tree (uninstall) falls back to ours.
#  - BRAIN_DB_DIR: only the durable adoption channels (the ~/.config/brain/env
#    knob or userConfig brain_path) may re-point an installed service's brain.
#    An ephemeral shell BRAIN_DB_DIR (eval runs, isolated copies) must never
#    hijack the shared daemon onto a temp brain.
RENDER_PLUGIN_DIR="$PLUGIN_DIR"
RENDER_DB_DIR="$BRAIN_DB_DIR"
if [ -f "$TARGET" ]; then
  _installed_plugin_dir="$(sed -n 's|.*<string>\(.*\)/hooks/scripts/start-daemon.sh</string>.*|\1|p' "$TARGET" | head -1)"
  if [ -n "$_installed_plugin_dir" ] && [ -x "$_installed_plugin_dir/hooks/scripts/start-daemon.sh" ]; then
    RENDER_PLUGIN_DIR="$_installed_plugin_dir"
  fi
  _installed_db_dir="$(sed -n '/<key>BRAIN_DB_DIR<\/key>/{n;s|.*<string>\(.*\)</string>.*|\1|p;}' "$TARGET" | head -1)"
  if [ -n "$_installed_db_dir" ] && [ "$_installed_db_dir" != "$BRAIN_DB_DIR" ]; then
    # Same subshell-source read the resolver's step 4b uses — one grammar.
    _knob="$(BRAIN_DB_DIR=''; . "${XDG_CONFIG_HOME:-$HOME/.config}/brain/env" 2>/dev/null; printf '%s' "$BRAIN_DB_DIR")"
    _opt="${CLAUDE_PLUGIN_OPTION_BRAIN_PATH:-${CLAUDE_PLUGIN_OPTION_brain_path:-}}"
    if [ "$BRAIN_DB_DIR" != "$_knob" ] && [ "$BRAIN_DB_DIR" != "$_opt" ]; then
      RENDER_DB_DIR="$_installed_db_dir"
    fi
  fi
fi

# Materialize the template (real paths, no tokens) to a temp file FIRST —
# rendering is deterministic, so comparing it against the installed copy
# detects drift (template evolved, or the brain legitimately moved) without
# touching launchd on the no-drift common case.
RENDERED="$(mktemp "${TMPDIR:-/tmp}/$LABEL.plist.XXXXXX")"
trap 'rm -f "$RENDERED"' EXIT
sed -e "s|__PLUGIN_DIR__|$RENDER_PLUGIN_DIR|g" \
    -e "s|__BRAIN_DB_DIR__|$RENDER_DB_DIR|g" \
    "$TEMPLATE" > "$RENDERED"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  # Already managed (every non-fresh boot). The installed plist is a frozen
  # snapshot launchd read at bootstrap time — if it no longer matches what we'd
  # render today, re-materialize and re-bootstrap so the daemon adopts the
  # current template + brain location. `kickstart` is NOT enough here: launchd
  # keeps the old EnvironmentVariables until the plist is re-bootstrapped.
  if cmp -s "$RENDERED" "$TARGET" 2>/dev/null; then
    exit 0
  fi
  echo "[install-daemon-service] installed plist drifted from template — re-materializing" >&2
  # Unload FIRST and VERIFY it unloaded before touching the file: a
  # still-loaded service keeps its bootstrap-time env, so overwriting the
  # plist under it would leave "file current, launchd stale" — every future
  # cmp blesses the divergence and it never heals.
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  for _ in 1 2 3 4 5; do
    launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || break
    sleep 1
  done
  if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    echo "[install-daemon-service] WARN: bootout did not unload $LABEL — keeping installed plist (drift retries next run)" >&2
    exit 1
  fi
  cp "$RENDERED" "$TARGET"
  _bootstrapped=""
  for _ in 1 2 3; do
    launchctl bootstrap "$DOMAIN" "$TARGET" >/dev/null 2>&1 && { _bootstrapped=1; break; }
    sleep 1
  done
  [ -n "$_bootstrapped" ] || launchctl load -w "$TARGET" >/dev/null 2>&1 || true
else
  echo "[install-daemon-service] first run — installing launchd service $LABEL" >&2
  mkdir -p "$HOME/Library/LaunchAgents"
  cp "$RENDERED" "$TARGET"
  launchctl bootstrap "$DOMAIN" "$TARGET" >/dev/null 2>&1 \
    || launchctl load -w "$TARGET" >/dev/null 2>&1 || true
fi
# VERIFY instead of assuming: bootstrap/load failures were swallowed above (the
# fallback chain needs that), so re-probe launchd for the truth. A false
# "installed" here would mask exactly the state this script exists to prevent —
# the daemon silently running detached (no KeepAlive) forever.
if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  echo "[install-daemon-service] installed $TARGET (launchd now owns the daemon)" >&2
else
  echo "[install-daemon-service] WARN: wrote $TARGET but launchctl bootstrap failed —" >&2
  echo "[install-daemon-service] daemon will run DETACHED via ensure_daemon (no KeepAlive," >&2
  echo "[install-daemon-service] no boot persistence). Retry: launchctl bootstrap $DOMAIN $TARGET" >&2
  exit 1
fi
