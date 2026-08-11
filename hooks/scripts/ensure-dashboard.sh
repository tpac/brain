#!/bin/bash
# brain — ensure the dashboard singleton is running, installing its launchd
# service on FIRST use. Idempotent. Called by the /dashboard skill (and safe to
# run directly).
#
#   already up         → no-op
#   down, not installed (macOS, first run) → materialize the plist template for
#                        THIS machine + launchctl bootstrap it. launchd's
#                        RunAtLoad + KeepAlive keep it up across reboots/crashes
#                        from then on — "install once, then it just works".
#   down, installed    → launchctl kickstart it
#   non-macOS          → detached fallback (per-session, no boot persistence;
#                        systemd deferred — docs/DISTRIBUTION-READINESS.md D-3)
#
# NOTE: no `set -u` — resolve-brain-db.sh (sourced below) is not nounset-safe.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LABEL="com.brain.dashboard"
TEMPLATE="$SCRIPT_DIR/$LABEL.plist"

# Resolve BRAIN_DB_DIR (+ DASHBOARD_PORT if the user set it) the same way every
# launch path does — so the installed plist points at the right brain.
# stdout suppressed (bootstrap noise); stderr kept — a resolution failure must
# say WHY, not surface later as a bare "BRAIN_DB_DIR unresolved".
source "$SCRIPT_DIR/resolve-brain-db.sh" >/dev/null || true
PORT="${DASHBOARD_PORT:-47303}"

_up() { [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null)" = "200" ]; }

# Detect installed-plist drift BEFORE the up fast-path: an "up" dashboard whose
# plist points at a moved brain is serving the WRONG data — exactly the state
# this check exists to fix (D-13 step 4c). Render the template deterministically
# to a temp file and compare; no drift costs one sed + cmp.
_DRIFTED=""
RENDERED=""
if [ "$(uname -s)" = "Darwin" ] && [ -f "$TEMPLATE" ] && [ -n "${BRAIN_DB_DIR:-}" ]; then
  TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
  # Identity preservation — mirror of install-daemon-service.sh (see the
  # rationale there): keep the installed tree while its launcher exists, and
  # let only the durable adoption channels re-point BRAIN_DB_DIR; an
  # ephemeral shell override must not hijack the singleton dashboard.
  RENDER_PLUGIN_DIR="$PLUGIN_DIR"
  RENDER_DB_DIR="$BRAIN_DB_DIR"
  if [ -f "$TARGET" ]; then
    _installed_plugin_dir="$(sed -n 's|.*<string>\(.*\)/hooks/scripts/brain-dashboard</string>.*|\1|p' "$TARGET" | head -1)"
    if [ -n "$_installed_plugin_dir" ] && [ -x "$_installed_plugin_dir/hooks/scripts/brain-dashboard" ]; then
      RENDER_PLUGIN_DIR="$_installed_plugin_dir"
    fi
    _installed_db_dir="$(sed -n '/<key>BRAIN_DB_DIR<\/key>/{n;s|.*<string>\(.*\)</string>.*|\1|p;}' "$TARGET" | head -1)"
    if [ -n "$_installed_db_dir" ] && [ "$_installed_db_dir" != "$BRAIN_DB_DIR" ]; then
      _knob="$(BRAIN_DB_DIR=''; . "${XDG_CONFIG_HOME:-$HOME/.config}/brain/env" 2>/dev/null; printf '%s' "$BRAIN_DB_DIR")"
      _opt="${CLAUDE_PLUGIN_OPTION_BRAIN_PATH:-${CLAUDE_PLUGIN_OPTION_brain_path:-}}"
      if [ "$BRAIN_DB_DIR" != "$_knob" ] && [ "$BRAIN_DB_DIR" != "$_opt" ]; then
        RENDER_DB_DIR="$_installed_db_dir"
      fi
    fi
  fi
  RENDERED="$(mktemp "${TMPDIR:-/tmp}/$LABEL.plist.XXXXXX")"
  trap 'rm -f "$RENDERED"' EXIT
  sed -e "s|__PLUGIN_DIR__|$RENDER_PLUGIN_DIR|g" \
      -e "s|__BRAIN_DB_DIR__|$RENDER_DB_DIR|g" \
      "$TEMPLATE" > "$RENDERED"
  if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 \
     && ! cmp -s "$RENDERED" "$TARGET" 2>/dev/null; then
    _DRIFTED=1
  fi
fi

if [ -z "$_DRIFTED" ] && _up; then
  echo "[ensure-dashboard] already up on :$PORT"
  exit 0
fi

case "$(uname -s)" in
  Darwin)
    DOMAIN="gui/$(id -u)"
    TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
    if [ -n "$_DRIFTED" ]; then
      # Re-materialize + re-bootstrap: launchd keeps the plist's
      # EnvironmentVariables as read at bootstrap time — kickstart alone
      # would restart the dashboard with the STALE env. Unload first and
      # VERIFY before touching the file (see install-daemon-service.sh:
      # "file current, launchd stale" never heals).
      echo "[ensure-dashboard] installed plist drifted from template — re-materializing"
      launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
      for _ in 1 2 3 4 5; do
        launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || break
        sleep 1
      done
      if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
        echo "[ensure-dashboard] WARN: bootout did not unload $LABEL — keeping installed plist (drift retries next run)" >&2
        exit 1
      fi
      cp "$RENDERED" "$TARGET"
      _bootstrapped=""
      for _ in 1 2 3; do
        launchctl bootstrap "$DOMAIN" "$TARGET" >/dev/null 2>&1 && { _bootstrapped=1; break; }
        sleep 1
      done
      [ -n "$_bootstrapped" ] || launchctl load -w "$TARGET" >/dev/null 2>&1 || true
    elif launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
      echo "[ensure-dashboard] service loaded but down — kickstarting"
      launchctl kickstart -k "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
    else
      echo "[ensure-dashboard] first run — installing launchd service $LABEL"
      [ -f "$TEMPLATE" ] || { echo "[ensure-dashboard] FATAL: plist template missing: $TEMPLATE" >&2; exit 1; }
      [ -n "${BRAIN_DB_DIR:-}" ] || { echo "[ensure-dashboard] FATAL: BRAIN_DB_DIR unresolved" >&2; exit 1; }
      mkdir -p "$HOME/Library/LaunchAgents"
      # Materialize the template for this machine (real paths, no tokens).
      sed -e "s|__PLUGIN_DIR__|$PLUGIN_DIR|g" \
          -e "s|__BRAIN_DB_DIR__|$BRAIN_DB_DIR|g" \
          "$TEMPLATE" > "$TARGET"
      launchctl bootstrap "$DOMAIN" "$TARGET" >/dev/null 2>&1 \
        || launchctl load -w "$TARGET" >/dev/null 2>&1 || true
    fi
    ;;
  *)
    echo "[ensure-dashboard] $(uname -s): no launchd — starting detached (no boot persistence)"
    nohup "$PLUGIN_DIR/hooks/scripts/brain-dashboard" >/dev/null 2>&1 &
    ;;
esac

# Wait for it to answer.
for _ in $(seq 1 15); do
  _up && { echo "[ensure-dashboard] up on :$PORT"; exit 0; }
  sleep 1
done
echo "[ensure-dashboard] WARN: not up on :$PORT after 15s — see ${BRAIN_DB_DIR:-?}/dashboard.log" >&2
exit 1
