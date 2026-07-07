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
#   already managed (launchctl print ok) → no-op (every non-fresh boot)
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
# Already loaded (installed on a prior boot, or hand-installed) → nothing to do.
# This is the common case; only a genuinely fresh machine falls through.
if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  exit 0
fi

# Resolve BRAIN_DB_DIR the same way every launch path does, so the installed
# plist points at the right brain. boot-brain.sh already resolved + exported it
# before calling us; this re-source fast-paths (runtime sentinel is set by now,
# so no ensure-runtime recursion) and just confirms the value.
source "$SCRIPT_DIR/resolve-brain-db.sh" >/dev/null 2>&1 || true

[ -f "$TEMPLATE" ] || { echo "[install-daemon-service] FATAL: plist template missing: $TEMPLATE" >&2; exit 1; }
[ -n "${BRAIN_DB_DIR:-}" ] || { echo "[install-daemon-service] FATAL: BRAIN_DB_DIR unresolved — not installing" >&2; exit 1; }

echo "[install-daemon-service] first run — installing launchd service $LABEL" >&2
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents"
# Materialize the template for this machine (real paths, no tokens).
sed -e "s|__PLUGIN_DIR__|$PLUGIN_DIR|g" \
    -e "s|__BRAIN_DB_DIR__|$BRAIN_DB_DIR|g" \
    "$TEMPLATE" > "$TARGET"
launchctl bootstrap "$DOMAIN" "$TARGET" >/dev/null 2>&1 \
  || launchctl load -w "$TARGET" >/dev/null 2>&1 || true
echo "[install-daemon-service] installed $TARGET (launchd now owns the daemon)" >&2
