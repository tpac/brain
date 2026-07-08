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

if _up; then
  echo "[ensure-dashboard] already up on :$PORT"
  exit 0
fi

case "$(uname -s)" in
  Darwin)
    DOMAIN="gui/$(id -u)"
    TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
    if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
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
    nohup "$PLUGIN_DIR/bin/brain-dashboard" >/dev/null 2>&1 &
    ;;
esac

# Wait for it to answer.
for _ in $(seq 1 15); do
  _up && { echo "[ensure-dashboard] up on :$PORT"; exit 0; }
  sleep 1
done
echo "[ensure-dashboard] WARN: not up on :$PORT after 15s — see ${BRAIN_DB_DIR:-?}/dashboard.log" >&2
exit 1
