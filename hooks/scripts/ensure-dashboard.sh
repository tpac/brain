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
# The plist ritual is owned by launchd-install.sh and shared with
# install-daemon-service.sh; the liveness policy below is this script's own.
source "$SCRIPT_DIR/launchd-install.sh"
PORT="${DASHBOARD_PORT:-47303}"

_up() { [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null)" = "200" ]; }

# Detect installed-plist drift BEFORE the up fast-path: an "up" dashboard whose
# plist points at a moved brain is serving the WRONG data — exactly the state
# this check exists to fix (D-13 step 4c). Render the template deterministically
# to a temp file and compare; no drift costs one sed + cmp.
_DRIFTED=""
RENDERED=""
if [ "$(uname -s)" = "Darwin" ] && [ -f "$TEMPLATE" ] && [ -n "${BRAIN_DB_DIR:-}" ]; then
  TARGET="$(brain_launchd_target "$LABEL")"
  RENDERED="$(mktemp "${TMPDIR:-/tmp}/$LABEL.plist.XXXXXX")"
  trap 'rm -f "$RENDERED"' EXIT
  brain_launchd_render "$LABEL" "$TEMPLATE" "brain-dashboard" \
                       "$PLUGIN_DIR" "$BRAIN_DB_DIR" "$RENDERED"
  if brain_launchd_managed "$LABEL" \
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
      # would restart the dashboard with the STALE env.
      echo "[ensure-dashboard] installed plist drifted from template — re-materializing"
      brain_launchd_reinstall "$LABEL" "$RENDERED" "[ensure-dashboard]" || exit 1
    elif brain_launchd_managed "$LABEL"; then
      echo "[ensure-dashboard] service loaded but down — kickstarting"
      launchctl kickstart -k "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
    else
      echo "[ensure-dashboard] first run — installing launchd service $LABEL"
      [ -f "$TEMPLATE" ] || { echo "[ensure-dashboard] FATAL: plist template missing: $TEMPLATE" >&2; exit 1; }
      [ -n "${BRAIN_DB_DIR:-}" ] || { echo "[ensure-dashboard] FATAL: BRAIN_DB_DIR unresolved" >&2; exit 1; }
      [ -n "$RENDERED" ] || { echo "[ensure-dashboard] FATAL: plist not rendered" >&2; exit 1; }
      brain_launchd_install_fresh "$LABEL" "$RENDERED"
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
