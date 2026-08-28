#!/bin/bash
# Relocate the brain from a host-owned dir (a plugins-data root that
# `claude plugin uninstall` deletes) to the service-owned XDG dir (D-13).
# USER-INVOKED ONLY — the boot notices print this path; boot itself never
# creates, moves, or deletes brains.
#
#   Usage: relocate-brain.sh [source-dir]
#   Default source: the resolved BRAIN_DB_DIR (the live brain).
#
# Safety design — nothing is ever deleted:
#   1. maintenance lock ON for the whole window, so ensure_daemon /
#      recover_daemon / the MCP health monitor cannot respawn a writer
#      mid-copy (a bare `launchctl bootout` flips manages() to False, which
#      AUTHORIZES the direct-spawn fallback — the lock is what actually
#      holds recovery off)
#   2. portable stop: bootout where launchd manages, then the pid file —
#      covers Linux (D-3) and macOS detached daemons; dashboard stopped too
#      (its read-only WAL connection keeps shm read-marks alive)
#   3. copy to a staging dir, PRAGMA quick_check on both DBs, then one
#      same-volume rename into place — an interrupted copy leaves the
#      original untouched and only ever costs the partial staging copy
#   4. the source is RENAMED beside its original path as an inert spare
#      (it is the backup), never removed
#   5. stale pointers at the old path are commented out (env-file knob) or
#      loudly named (plugin setting, which this script cannot edit)
#   6. services re-rendered via install-daemon-service.sh so the plists
#      carry the new path
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
say() { printf '%s\n' "[relocate-brain] $*"; }
die() { printf '%s\n' "[relocate-brain] FATAL: $*" >&2; exit 1; }

SRC="${1:-}"
if [ -z "$SRC" ]; then
  . "$SCRIPT_DIR/resolve-brain-db.sh"
  SRC="${BRAIN_DB_DIR:-}"
fi
SRC="${SRC%/}"
[ -n "$SRC" ] || die "no brain directory resolved and none given"
[ -f "$SRC/brain.db" ] || die "no brain.db at $SRC"
TARGET="${BRAIN_XDG_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/brain}"
if [ "$SRC" = "$TARGET" ]; then
  say "brain already lives at $TARGET — nothing to do"
  exit 0
fi
PY="$(command -v python3 || true)"
[ -n "$PY" ] || die "python3 not found on PATH"

# Target must be free. A leftover empty dir is cleared with rmdir (which by
# definition cannot touch a non-empty dir); anything else is a real occupant
# and this refuses LOUDLY rather than guessing which brain is current.
if [ -e "$TARGET" ]; then
  if [ -f "$TARGET/brain.db" ]; then
    die "target $TARGET already holds a brain.db — a second brain. Not moving anything. Inspect both and decide which is current before retrying."
  fi
  rmdir "$TARGET" 2>/dev/null \
    || die "target $TARGET exists and is not empty — not moving. Inspect it first."
fi

# Overridable for sandboxed tests only — production paths are the contract
# (daemon_config.get_maintenance_path / get_pid_path).
LOCK="${BRAIN_MAINTENANCE_LOCK:-/tmp/brain-maintenance-$(id -u).lock}"
touch "$LOCK" || die "cannot write $LOCK"
trap 'rm -f "$LOCK"' EXIT
say "maintenance lock on — daemon auto-recovery paused for the move"

# Stop the writers, then wait for the daemon to actually exit (it has a 15s
# shutdown budget; moving before it finishes flushing defeats the point).
launchctl bootout "gui/$(id -u)/com.brain.daemon" 2>/dev/null
launchctl bootout "gui/$(id -u)/com.brain.dashboard" 2>/dev/null
PIDFILE="${BRAIN_DAEMON_PIDFILE:-/tmp/brain-daemon-$(id -u).pid}"
if [ -f "$PIDFILE" ]; then
  _pid="$(cat "$PIDFILE" 2>/dev/null)"
  if [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null; then
    kill "$_pid" 2>/dev/null
    _waited=0
    while kill -0 "$_pid" 2>/dev/null && [ "$_waited" -lt 20 ]; do
      sleep 1; _waited=$((_waited + 1))
    done
    kill -0 "$_pid" 2>/dev/null && { kill -9 "$_pid" 2>/dev/null; sleep 1; }
  fi
fi
say "daemon and dashboard stopped"

# Copy → verify → switch. The staging dir is the only thing ever removed on
# failure, and it is a partial DUPLICATE — the original is intact throughout.
STAGE="$TARGET.incoming.$$"
mkdir -p "$(dirname "$TARGET")" || die "cannot create $(dirname "$TARGET")"
say "copying $SRC -> staging (this can take a minute on large brains)"
cp -a "$SRC" "$STAGE" \
  || { rm -rf "$STAGE"; die "copy failed — original untouched at $SRC"; }
for _db in brain.db brain_logs.db; do
  [ -f "$STAGE/$_db" ] || continue
  if ! "$PY" -c 'import sqlite3,sys; r=sqlite3.connect(sys.argv[1]).execute("PRAGMA quick_check").fetchone()[0]; sys.exit(0 if r=="ok" else 1)' "$STAGE/$_db"; then
    rm -rf "$STAGE"
    die "integrity check failed on the $_db copy — original untouched at $SRC"
  fi
done
mv "$STAGE" "$TARGET" || { rm -rf "$STAGE"; die "could not finalize $TARGET"; }
RETIRED="$SRC.relocated-$(date +%Y%m%dT%H%M%S)"
mv "$SRC" "$RETIRED" \
  || die "brain copied to $TARGET but the source could not be renamed — TWO live copies exist. Rename or remove $SRC yourself before the next session, or the resolver may still pick it."
say "moved: $TARGET"
say "spare copy kept at $RETIRED — delete it whenever you like; nothing references it"

# Retire stale pointers. The env-file knob is commented out in place (with a
# backup); the plugin brain-path setting can only be named, not edited.
ENVF="${XDG_CONFIG_HOME:-$HOME/.config}/brain/env"
if [ -f "$ENVF" ]; then
  # python -c, not a heredoc-in-$(): bash 3.2 (macOS /bin/bash) miscounts
  # parens across a heredoc inside command substitution.
  _knob_out="$("$PY" -c "
import re, shutil, sys
path, src = sys.argv[1], sys.argv[2].rstrip(\"/\")
lines = open(path).read().splitlines(True)
out, hit = [], False
for ln in lines:
    m = re.match(r\"\s*BRAIN_DB_DIR=(['\\\"]?)(.*?)\1\s*\$\", ln)
    if m and m.group(2).rstrip(\"/\") == src:
        out.append(\"# relocated to the standard location: \" + ln)
        hit = True
    else:
        out.append(ln)
if hit:
    shutil.copy2(path, path + \".bak-relocate\")
    open(path, \"w\").writelines(out)
print(\"commented\" if hit else \"none\")
" "$ENVF" "$SRC")"
  if [ "$_knob_out" = "commented" ]; then
    say "commented out the BRAIN_DB_DIR knob in $ENVF (backup: $ENVF.bak-relocate)"
  elif grep -q '^[[:space:]]*BRAIN_DB_DIR=' "$ENVF" 2>/dev/null; then
    say "NOTE: $ENVF carries a BRAIN_DB_DIR line that does not match the old path — check it points where you intend"
  fi
fi
if [ -n "${CLAUDE_PLUGIN_OPTION_BRAIN_PATH:-}${CLAUDE_PLUGIN_OPTION_brain_path:-}" ]; then
  say "NOTE: the plugin's brain-path setting still names the old dir — clear it in the plugin settings, or every hook will keep recreating an empty $SRC"
fi

# Lock off BEFORE services return, then re-render the plists so launchd
# carries the new path (the render guard accepts the XDG service dir as a
# durable target).
rm -f "$LOCK"; trap - EXIT
say "maintenance lock released"
bash "$SCRIPT_DIR/install-daemon-service.sh" \
  || say "WARN: daemon service re-install reported a problem — the next session boot retries it"

# Prove the ladder now lands on the new home (and persist it for non-hook
# consumers via resolved.env).
_check="$(BRAIN_DB_DIR= bash -c ". '$SCRIPT_DIR/resolve-brain-db.sh'; printf %s \"\$BRAIN_DB_DIR\"")"
if [ "$_check" = "$TARGET" ]; then
  say "verified: the resolver finds the brain at $TARGET"
else
  say "WARN: the resolver resolved '$_check' (expected $TARGET) — a stale pointer above still wins; fix it before relying on this brain"
fi

# Other brains still parked under a plugins-data root deserve a mention —
# the resolver only ever inspects the one it resolved.
for _root in "$HOME/.claude/plugins/data" "${CLAUDE_PLUGIN_DATA:+$(dirname "$CLAUDE_PLUGIN_DATA")}"; do
  [ -n "$_root" ] && [ -d "$_root" ] || continue
  find "$_root" -mindepth 3 -maxdepth 3 -path '*/brain/brain.db' 2>/dev/null \
    | while IFS= read -r _other; do
        say "NOTE: another brain remains at $(dirname "$_other") — rerun this script with that path to rescue it too"
      done
done
say "done — start a new session"
