#!/bin/bash
# 5.7 — the sandbox a stranger's machine is simulated in. Sourced by
# install-smoke.sh and upgrade-smoke.sh; never run directly, never ships.
#
# A stranger has: no ~/.config/brain, no brain, no API key, no launchd
# service, no Python runtime. Isolation from THIS machine's production brain:
# scratch $HOME + XDG dirs, an `env -i` environment (nothing of the caller's
# shell leaks in), BRAIN_INSTANCE (keys every /tmp rendezvous path; the
# launchd installers refuse under it), an ephemeral daemon port, and a COPY
# of the tree — the bootstrap writes venv/ py/ bin/ INTO the plugin dir. The
# uv download cache is the one thing borrowed from the real home: a package
# cache changes no boot decision, and a cold one is ~200MB per run.
#
# Caller sets REPO (the dev checkout — its venv answers the questions asked
# before the staged tree has a runtime) and `set -euo pipefail`. Provides:
#   smoke_stage_new TREE   copy TREE into a fresh stage, arm teardown
#   smoke_install          a stranger's first run, steps 1–7; leaves the daemon up
#   smoke_node_count       the daemon's own node count (its status JSON)
#   smoke_remember TITLE   one lived node, written through the daemon; prints its id
#   smoke_node_present ID  the daemon still serves that node
#   smoke_fingerprint      the code fingerprint the running daemon reports
#   smoke_overlay TREE     install TREE over the staged plugin the way a
#                          marketplace update does — everything replaced except
#                          the runtime the first boot bootstrapped
#   smoke_converge         after an overlay: the daemon must reload onto the new
#                          code and the next session must boot clean
# Every helper that asks the daemon something logs its stderr under the stage
# and returns non-zero on failure — callers `|| _die`, so a red always names
# its cause. Env: SMOKE_BOOTSTRAP_TIMEOUT (default 1200) · SMOKE_UV_CACHE_DIR
# (default the real home's ~/.cache/uv) · SMOKE_COLD_CACHE=1 (borrow nothing)
# · SMOKE_KEEP=1 (keep the stage on success too).

say()  { printf '%s\n' "[$SMOKE_NAME] $*"; }
fail() { printf '%s\n' "[$SMOKE_NAME] FAILED: $*" >&2; exit 1; }

smoke_stage_new() {
  local tree="$1"
  [ -f "$tree/.claude-plugin/plugin.json" ] || fail "not a plugin tree: $tree"
  DEV_PY="$REPO/venv/bin/python"
  [ -x "$DEV_PY" ] || fail "dev venv missing at $DEV_PY — run hooks/scripts/ensure-runtime.sh"
  BOOT_TIMEOUT="${SMOKE_BOOTSTRAP_TIMEOUT:-1200}"
  REAL_HOME="$HOME"
  STAGE="$(mktemp -d "${TMPDIR:-/tmp}/entity-smoke.XXXXXX")"
  PLUGIN="$STAGE/plugin"; FAKE_HOME="$STAGE/home"; LOGS="$STAGE/logs"
  mkdir -p "$FAKE_HOME" "$LOGS"
  cp -R "$tree" "$PLUGIN"
  # Instance name ≤ 32 chars of [A-Za-z0-9_-] (daemon_config validates).
  INSTANCE="smoke$$"
  PORT="$(_free_port)"; DPORT="$(_free_port)"
  XDG_CFG="$FAKE_HOME/.config"; XDG_DATA="$FAKE_HOME/.local/share"
  # D-13 birthplace. It must NOT exist before the warm boot: the ladder
  # creates it. BRAIN_DB_DIR is nevertheless handed to the daemon's processes
  # because daemon_config refuses an instance without an explicit one (an
  # inherited dir would be production's) — which also means those processes
  # never walk the ladder. Steps 6 and 7 drop the variable again so BOTH
  # resolvers, the shell one and the Python one, are exercised for real.
  BRAIN_HOME="$XDG_DATA/brain"
  UV_CACHE="${SMOKE_UV_CACHE_DIR:-$REAL_HOME/.cache/uv}"
  [ "${SMOKE_COLD_CACHE:-}" = "1" ] && UV_CACHE="$FAKE_HOME/.cache/uv"
  trap _teardown EXIT
  say "tree=$tree stage=$STAGE instance=$INSTANCE port=$PORT"
}

# An ephemeral port the kernel just handed out. A dev sandbox choosing its own
# port is outside the daemon/client socket concern — the containment gate
# names this file as the one dev-tool owner of that construction.
_free_port() { "$DEV_PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; }

# Every process the smoke launches sees ONLY this environment.
_run() {
  env -i \
    HOME="$FAKE_HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    USER="$(id -un)" LOGNAME="$(id -un)" TMPDIR="${TMPDIR:-/tmp}" LANG="${LANG:-en_US.UTF-8}" \
    XDG_CONFIG_HOME="$XDG_CFG" XDG_DATA_HOME="$XDG_DATA" XDG_CACHE_HOME="$FAKE_HOME/.cache" \
    UV_CACHE_DIR="$UV_CACHE" \
    BRAIN_INSTANCE="$INSTANCE" BRAIN_DAEMON_PORT="$PORT" BRAIN_DB_DIR="$BRAIN_HOME" \
    DASHBOARD_PORT="$DPORT" CLAUDE_PLUGIN_ROOT="$PLUGIN" \
    "$@"
}
_py() {  # the tree's own venv python, from the tree, as the hooks run it
  (cd "$PLUGIN" && _run env PYTHONPATH="$PLUGIN" "$PLUGIN/venv/bin/python" "$@")
}
_hook() {  # a SessionStart hook invocation, stdin shaped as Claude Code sends it
  printf '{"session_id":"smoke-%s","cwd":"%s","source":"startup"}' "$$" "$STAGE" \
    | _run bash "$PLUGIN/hooks/scripts/boot-brain.sh" >"$LOGS/$1.out" 2>"$LOGS/$1.err"
}
# Which stream a message lands on is boot-brain.sh's business, not ours.
_has()  { grep -q -- "$2" "$LOGS/$1.out" "$LOGS/$1.err" 2>/dev/null; }
_show() { for f in "$@"; do printf '%s\n' "--- $f ---" >&2; tail -40 "$LOGS/$f" >&2 2>/dev/null || true; done; }
_die()  { local msg="$1"; shift; _show "$@"; fail "$msg"; }

_teardown() {
  local rc=$?
  if [ -x "$PLUGIN/venv/bin/python" ]; then
    _py -c 'from servers.daemon_client import stop_daemon; stop_daemon()' >"$LOGS/teardown.log" 2>&1 || true
    # A daemon that outlives its stage keeps a port and a deleted tree; say so
    # and keep the evidence rather than delete the tree from under it.
    if _py -c 'from servers.daemon_client import is_daemon_running; import sys; sys.exit(0 if is_daemon_running() else 1)' 2>/dev/null; then
      say "WARN: the smoke daemon (instance $INSTANCE, port $PORT) did not stop — stage kept"
      rc=1
    fi
    # the instance's rendezvous files, named by their owner — never a glob
    # that could brush production's
    _py -c '
import os, sys
from servers import daemon_config as c
paths = [c.get_socket_path(), c.get_pid_path(), c.get_lock_path(), c.get_startup_lock_path(),
         c.get_status_path(), c.get_maintenance_path(), c.get_recovery_state_path(),
         c.get_db_lock_path(sys.argv[1])]
for p in paths:
    try: os.remove(p)
    except FileNotFoundError: pass
' "$BRAIN_HOME/brain.db" >>"$LOGS/teardown.log" 2>&1 || true
  fi
  if [ "$rc" -ne 0 ] || [ "${SMOKE_KEEP:-0}" = "1" ]; then
    say "stage kept at $STAGE (logs in $STAGE/logs)"
  else
    rm -rf "$STAGE"
  fi
}

# A warm SessionStart must come back with the daemon up and the MCP server
# importable, and without any of the notices a broken install prints.
_assert_boot() {
  local name="$1"
  for bad in "MCP SERVER BROKEN" "No brain.db found" "needs one pointer" "Daemon FAILED"; do
    ! _has "$name" "$bad" || _die "$name boot: $bad" "$name.out" "$name.err"
  done
  _has "$name" "Daemon ready"  || _die "$name boot did not report the daemon ready" "$name.out" "$name.err"
  _has "$name" "MCP server OK" || _die "$name boot did not import the MCP server" "$name.out" "$name.err"
  [ -s "$LOGS/$name.out" ]     || _die "$name boot injected no context" "$name.err"
}
_assert_entrypoints() {
  _py -c 'import servers.brain_mcp, servers.daemon_server' >"$LOGS/imports.log" 2>&1 \
    || _die "an entrypoint failed to import" imports.log
}
_assert_roundtrip() {
  _py -c '
from servers.daemon_client import is_daemon_responsive, send_command
assert is_daemon_responsive(5.0), "daemon not responsive"
r = send_command("get_config", {"key": "debug_enabled", "default": "0"})
assert r.get("ok"), r
' >"$LOGS/daemon.log" 2>&1 || _die "daemon round trip failed" daemon.log
}

smoke_install() {
  local t0=$SECONDS t1
  # ── 1. cold boot: the hook must answer fast and hand off the bootstrap ────
  _hook cold || _die "cold SessionStart hook exited non-zero" cold.out cold.err
  _has cold "first-run install in progress" || _die "cold boot did not announce the detached bootstrap" cold.out cold.err
  _has cold "setup in progress" || _die "keyless boot did not present the API-key setup step" cold.out cold.err
  say "1/7 cold boot: answered in $((SECONDS - t0))s, bootstrap detached"

  # ── 2. the bootstrap the cold boot detached ──────────────────────────────
  . "$PLUGIN/hooks/scripts/runtime-state.sh"   # brain_runtime_ready — the ONE readiness predicate
  t1=$SECONDS
  until brain_runtime_ready "$PLUGIN"; do
    if [ -f "$PLUGIN/.bootstrap-failed" ]; then
      tail -40 "$PLUGIN/.bootstrap.log" >&2 || true
      fail "runtime bootstrap failed: $(cat "$PLUGIN/.bootstrap-failed")"
    fi
    if [ $((SECONDS - t1)) -ge "$BOOT_TIMEOUT" ]; then
      tail -40 "$PLUGIN/.bootstrap.log" >&2 || true
      fail "runtime bootstrap not finished after ${BOOT_TIMEOUT}s"
    fi
    sleep 2
  done
  say "2/7 runtime bootstrap: ready in $((SECONDS - t1))s"

  # ── 3. warm boot: birth at the XDG dir, daemon up keyless, MCP imports ───
  _hook warm || _die "warm SessionStart hook exited non-zero" warm.out warm.err
  _assert_boot warm
  [ -f "$BRAIN_HOME/brain.db" ] || _die "no brain.db at the D-13 service dir $BRAIN_HOME" warm.err
  say "3/7 warm boot: brain born at $BRAIN_HOME, daemon ready, MCP imports"

  # ── 4. both entrypoints import from the installed tree (redeploy.sh's check)
  _assert_entrypoints
  say "4/7 entrypoints: brain_mcp + daemon_server import"

  # ── 5. the daemon answers a real command ─────────────────────────────────
  _assert_roundtrip
  say "5/7 daemon: responsive, get_config round trip ok"

  # ── 6. the shell ladder, as a non-hook shell walks it ────────────────────
  # No BRAIN_DB_DIR: the resolver must FIND the brain (the XDG rung) rather
  # than be told, and persist resolved.env for the skills and the /watch
  # listener. Both values come out of the resolver's own reader.
  _run env -u BRAIN_DB_DIR bash -c '
    . "$1/hooks/scripts/resolve-brain-db.sh" || exit 1
    printf "%s\n%s\n" "$BRAIN_DB_DIR" "$(_brain_db_dir_from "$XDG_CONFIG_HOME/brain/resolved.env")"
  ' _ "$PLUGIN" >"$LOGS/resolve.out" 2>"$LOGS/resolve.err" || _die "shell resolution failed" resolve.err
  [ "$(sed -n 1p "$LOGS/resolve.out")" = "$BRAIN_HOME" ] \
    || _die "shell ladder landed on '$(sed -n 1p "$LOGS/resolve.out")', not $BRAIN_HOME" resolve.out resolve.err
  [ "$(sed -n 2p "$LOGS/resolve.out")" = "$BRAIN_HOME" ] \
    || _die "resolved.env does not record $BRAIN_HOME" resolve.out resolve.err
  say "6/7 shell ladder: found the brain unaided, resolved.env persisted"

  # ── 7. the Python ladder, as the daemon and MCP server walk it ───────────
  # Without the instance keys, daemon_config imports read-only (no rendezvous
  # file is touched by a resolve), and resolve_db_dir must find the same brain.
  (cd "$PLUGIN" && _run env -u BRAIN_DB_DIR -u BRAIN_INSTANCE -u BRAIN_DAEMON_PORT \
      PYTHONPATH="$PLUGIN" "$PLUGIN/venv/bin/python" -c \
      'from servers.daemon_config import resolve_db_dir; print(resolve_db_dir())') \
    >"$LOGS/resolve-py.out" 2>"$LOGS/resolve-py.err" || _die "python resolution failed" resolve-py.err
  [ "$(cat "$LOGS/resolve-py.out")" = "$BRAIN_HOME" ] \
    || _die "python ladder landed on '$(cat "$LOGS/resolve-py.out")', not $BRAIN_HOME" resolve-py.out resolve-py.err
  say "7/7 python ladder: found the brain unaided"
}

# The daemon's own count, from the status JSON it writes for the statusline —
# never a second connection to its database. It rewrites that file on the
# HOOK path and on its autosave tick, not after a plain command, so a
# read-only hook that marks nothing dirty is asked first to refresh it.
smoke_node_count() {
  _py -c '
import json
from servers.daemon_client import send_command
from servers import daemon_config as c
r = send_command("hook_pre_bash_safety", {"command": "true"})
assert r.get("ok"), r
print(json.load(open(c.get_status_path()))["nodes"])
' 2>"$LOGS/count.err"
}
smoke_remember() {
  _py -c '
import sys
from servers.daemon_client import send_command
r = send_command("remember", {"type": "moment", "title": sys.argv[1],
                              "content": "Written by the upgrade smoke before the upgrade; must survive it."})
assert r.get("ok"), r
print(r["result"]["id"])
' "$1" 2>"$LOGS/remember.err"
}
smoke_node_present() {
  _py -c '
import sys
from servers.daemon_client import send_command
r = send_command("get_node", {"node_id": sys.argv[1]})
assert r.get("ok") and r.get("result"), r
' "$1" 2>"$LOGS/present.err"
}
smoke_fingerprint() {
  _py -c 'from servers.daemon_client import send_command; r = send_command("ping"); assert r.get("ok"), r; print(r["result"]["code_fingerprint"])' 2>"$LOGS/fingerprint.err"
}

# What a marketplace update does to an installed plugin, as far as this
# harness models it: the tree is replaced by the new release's — files the
# release dropped disappear — while the runtime the first boot bootstrapped
# stays (its artifacts are the ones ensure-runtime.sh lays down), and the
# brain lives outside the tree (D-13) and is not touched at all. An update
# that also discards the runtime is a cold re-bootstrap, not modelled here.
smoke_overlay() {
  local tree="$1" entry base
  local keep=' venv py bin .runtime-ready .bootstrap.log .bootstrap-failed .runtime-bootstrap.lock '
  [ -f "$tree/.claude-plugin/plugin.json" ] || fail "not a plugin tree: $tree"
  for entry in "$PLUGIN"/* "$PLUGIN"/.[!.]*; do
    [ -e "$entry" ] || continue
    base="$(basename "$entry")"
    case "$keep" in *" $base "*) continue ;; esac
    rm -rf "$entry"
  done
  cp -R "$tree/." "$PLUGIN/"
}

# The first SessionStart after an update asks the healthy-but-stale daemon to
# reload in place. ensure_daemon gives that ~20s before it defers to the
# booting successor and reports FAILED for THIS session; the daemon converges
# regardless. So: fire the update's session, wait (bounded) for the daemon to
# serve the installed tree's code, then judge the NEXT session's boot — the
# one a user actually sees after an update.
smoke_converge() {
  local want have t0=$SECONDS
  _hook upgraded || _die "post-upgrade SessionStart hook exited non-zero" upgraded.out upgraded.err
  want="$(_py -c 'from servers.daemon_config import _CODE_FINGERPRINT; print(_CODE_FINGERPRINT)' 2>"$LOGS/fingerprint.err")" \
    || _die "could not fingerprint the installed tree" fingerprint.err
  until have="$(smoke_fingerprint)" && [ "$have" = "$want" ]; do
    [ $((SECONDS - t0)) -lt 120 ] \
      || _die "daemon did not converge in 120s: serving '${have:-nothing}', the installed tree is '$want'" upgraded.out upgraded.err fingerprint.err
    sleep 2
  done
  _hook upgraded2 || _die "second post-upgrade SessionStart hook exited non-zero" upgraded2.out upgraded2.err
  _assert_boot upgraded2
  _assert_entrypoints
  _assert_roundtrip
}
