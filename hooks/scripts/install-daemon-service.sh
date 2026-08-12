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
# The plist ritual (render + identity preservation + bootstrap/reload) is
# owned by launchd-install.sh and shared with ensure-dashboard.sh; the policy
# below — install-only, then verify — is this script's own.
source "$SCRIPT_DIR/launchd-install.sh"

[ -f "$TEMPLATE" ] || { echo "[install-daemon-service] FATAL: plist template missing: $TEMPLATE" >&2; exit 1; }
[ -n "${BRAIN_DB_DIR:-}" ] || { echo "[install-daemon-service] FATAL: BRAIN_DB_DIR unresolved — not installing" >&2; exit 1; }

# Render to a temp file FIRST — rendering is deterministic, so comparing it
# against the installed copy detects drift (template evolved, or the brain
# legitimately moved) without touching launchd on the no-drift common case.
# LAUNCHER is what the CURRENT template's ProgramArguments points at.
RENDERED="$(mktemp "${TMPDIR:-/tmp}/$LABEL.plist.XXXXXX")"
trap 'rm -f "$RENDERED"' EXIT
brain_launchd_render "$LABEL" "$TEMPLATE" "brain-daemon" \
                     "$PLUGIN_DIR" "$BRAIN_DB_DIR" "$RENDERED"

if brain_launchd_managed "$LABEL"; then
  # Already managed (every non-fresh boot). The installed plist is a frozen
  # snapshot launchd read at bootstrap time — if it no longer matches what we'd
  # render today, re-materialize and re-bootstrap so the daemon adopts the
  # current template + brain location.
  if cmp -s "$RENDERED" "$TARGET" 2>/dev/null; then
    exit 0
  fi
  echo "[install-daemon-service] installed plist drifted from template — re-materializing" >&2
  brain_launchd_reinstall "$LABEL" "$RENDERED" "[install-daemon-service]" || exit 1
else
  echo "[install-daemon-service] first run — installing launchd service $LABEL" >&2
  brain_launchd_install_fresh "$LABEL" "$RENDERED"
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
