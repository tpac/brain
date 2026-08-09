"""
brain — Daemon Client

Client-side functions for communicating with the brain daemon.
Used by hook scripts and brain_mcp.py.

Singleton guarantee: ensure_daemon() uses fcntl.flock on a lock file.
First caller acquires lock, starts daemon, releases lock.
All other callers block on the lock, wake up to a running daemon.
No file markers, no polling races.
"""

import fcntl
import json
import os
import socket
import sys
import time
from typing import Any, Dict, Optional

from .daemon_config import (
    _code_fingerprint, _CODE_FINGERPRINT, _IS_WORKTREE, REPO_ROOT,
    get_daemon_addr, get_socket_path, get_pid_path, get_lock_path, get_status_path,
    get_recovery_state_path, is_maintenance_mode,
    LAUNCHD_THROTTLE_INTERVAL_S,
)
from .daemon_launch import (
    kickstart, kill_daemon, manages, port_is_occupied, spawn_detached_daemon,
)


def send_command(cmd: str, args: Optional[Dict[str, Any]] = None,
                 timeout: float = 10.0) -> Dict[str, Any]:
    """Send a command to the running daemon via TCP. Returns response dict."""
    addr = get_daemon_addr()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(addr)
        msg = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        sock.sendall(msg.encode('utf-8'))

        # Read response
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        if data:
            return json.loads(data.decode('utf-8').strip())
        return {"ok": False, "error": "No response"}

    except socket.timeout:
        return {"ok": False, "error": "Timeout"}
    except ConnectionRefusedError:
        return {"ok": False, "error": "Connection refused — daemon may be dead"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        sock.close()


def is_daemon_running() -> bool:
    """Check if the daemon is running."""
    pid_path = get_pid_path()
    if not os.path.exists(pid_path):
        return False

    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        try:
            os.unlink(pid_path)
        except Exception:
            pass
        return False


def _can_connect(timeout: float = 2.0) -> dict:
    """Try to ping the daemon. Returns ping response or empty dict."""
    try:
        return send_command("ping", timeout=timeout)
    except Exception:
        return {}


def is_daemon_responsive(timeout: float = 2.0) -> bool:
    """True only if the daemon answers a ping (liveness).

    Distinguishes a live daemon from a hung "corpse" that holds the port but
    services nothing — which a bare TCP connect or a PID-exists check would
    wrongly report as alive. This is the liveness signal; readiness (e.g. a
    cold-start recall that's slow but alive) is a separate concern and must
    NOT be treated as down."""
    return _can_connect(timeout=timeout).get("ok", False)


# Liveness-grace budgets (WALL-CLOCK seconds) for the daemon (re)start decision.
# A bare 2s ping cannot tell "slow / relaunching" from "dead": a recall runs
# 10-17s, and a launchd relaunch is gated by ThrottleInterval (10s — see
# com.brain.daemon.plist) + an embedder reload (~4-6s). The budget must clear
# that window, or a busy/recovering daemon is mistaken for a corpse and
# (re)started — SIGKILLing it, resetting the throttle, corrupting an in-flight
# hook (the closed-DB boot failure of 2026-06-06). Bounded by WALL-CLOCK, not a
# ping count: a hung corpse whose socket blocks the full ping timeout each probe
# must not blow the budget (a count of 40 × 2s pings would be ~100s, not ~20s).
# Derived from the plist throttle (single-sourced in daemon_config) so the two
# worlds can't drift: the budget must clear ThrottleInterval + an embedder reload.
_EMBEDDER_RELOAD_BUDGET_S = 10.0                                          # cold embedder + Anthropic client warmup
_GRACE_DEADLINE_S = LAUNCHD_THROTTLE_INTERVAL_S + _EMBEDDER_RELOAD_BUDGET_S   # 20.0 — confirm-down / wait out a relaunch
_KICKSTART_DEADLINE_S = _GRACE_DEADLINE_S + 5.0                               # 25.0 — wait for our own kickstart to come back


def _await_responsive(deadline_s: float, ping_timeout: float = 2.0) -> dict:
    """Poll the daemon until it answers a ping or `deadline_s` of WALL-CLOCK time
    elapses; returns the last ping response ({} if never responsive).

    Bounded by real time, not a ping count — a hung corpse whose socket blocks
    the full ping_timeout per probe can't exceed deadline_s + one ping. The
    robust liveness gate the boot (re)start decision uses instead of a bare ping.
    NOT used on the recovery hot path (recover_daemon stays a single fast ping)."""
    deadline = time.monotonic() + max(0.0, deadline_s)
    resp = _can_connect(timeout=ping_timeout)
    while not resp.get("ok") and time.monotonic() < deadline:
        time.sleep(0.5)
        resp = _can_connect(timeout=ping_timeout)
    return resp


def _is_daemon_source(resp: dict) -> bool:
    """Whether THIS checkout is the daemon's launch source — the only checkout
    that may manage its lifecycle.

    NOT an identity statement. The brain (the one shared brain.db) is the same
    for every checkout, so every session is Anchor. This is purely machinery:
    the daemon's CODE is launchd-pinned to its source dir, so only the source
    checkout can converge a (re)start. A linked worktree / 2nd clone is a pure
    client — restarting from there can't change the daemon's code, it only churns.

    Signal: the daemon's reported `source_dir` vs this checkout's REPO_ROOT. When
    the daemon is DOWN (no source_dir to compare) we fall back to the
    linked-worktree heuristic: a linked worktree is never the source, but a
    second full *clone* (its .git is a directory, not a file) is indistinguishable
    from the source here and is treated AS source — acceptable because a kickstart
    from a clone still converges via launchd's pinned source, and the manual kill
    path (_relaunch_daemon) re-checks before ever killing the shared daemon."""
    daemon_src = resp.get("result", {}).get("source_dir", "")
    if daemon_src:
        return os.path.realpath(daemon_src) == os.path.realpath(REPO_ROOT)
    return not _IS_WORKTREE


def _code_changed(resp: dict) -> bool:
    """True iff a restart is warranted: this checkout is the daemon's source AND
    the daemon runs a different code fingerprint. Conservative — an unknown
    fingerprint never restarts, and a non-source checkout is never "changed"
    (it can't converge a restart — the non-convergent churn of 2026-06-06)."""
    if not _is_daemon_source(resp):
        return False
    daemon_fp = resp.get("result", {}).get("code_fingerprint", "")
    # Import-time constant, not a fresh _code_fingerprint(): this process is
    # short-lived (hook/CLI) so its code can't change underneath it, and
    # recomputing re-reads every servers/**/*.py (several MB) on each call.
    return bool(_CODE_FINGERPRINT != "unknown" and daemon_fp and daemon_fp != _CODE_FINGERPRINT)


def ensure_daemon(db_path: str) -> bool:
    """Ensure the daemon is running AND on current code. Returns True if ready.

    launchd owns the daemon lifecycle (com.brain.daemon: KeepAlive, RunAtLoad).
    This function only PINGS and, when a (re)start is needed, routes it through
    launchd via `launchctl kickstart -k` (daemon_launch.kickstart). It never kills +
    Popens its own process alongside launchd.

    Doing both was the Errno-48 storm of 2026-06-04: N sessions booted at once,
    each saw stale code, and each independently killed + respawned while
    launchd's KeepAlive ALSO respawned — several processes raced to bind the
    port. Now every (re)start decision is serialized under the fcntl singleton
    lock and re-checked after acquiring it, so N concurrent callers (re)start at
    most once. Direct Popen survives ONLY as the no-launchd fallback (a fresh
    install where the LaunchAgent isn't bootstrapped) — there's no KeepAlive to
    race there.

    Maintenance mode: if the maintenance lock file exists, skip startup.
    Used during DB operations (VACUUM, schema changes, bulk deletes).
    """
    if is_maintenance_mode():
        sys.stderr.write("[brain-daemon] Maintenance mode active — skipping startup\n")
        return False

    resp = _can_connect()

    # A non-source checkout (linked worktree / 2nd clone) is a PURE CLIENT of the
    # shared daemon. Identity is shared — the one brain.db makes every checkout
    # Anchor — but the daemon's CODE is launchd-pinned to its source, so a
    # (re)start from here can't converge to this checkout's code; it would only
    # churn and SIGKILL in-flight work. So a non-source checkout never restarts:
    # up → done; down → wait out launchd's relaunch (KeepAlive owns recovery).
    if not _is_daemon_source(resp):
        if resp.get("ok"):
            return True
        return _await_responsive(_GRACE_DEADLINE_S).get("ok", False)

    # ── Source checkout: owns lifecycle (kickstart converges; stale code reloads). ──
    # Fast path: running, responsive, and on current code → nothing to do.
    if resp.get("ok") and not _code_changed(resp):
        return True
    # Otherwise (down, or up-but-stale) fall through to the locked (re)start.

    # Serialize the (re)start through the singleton lock.
    lock_path = get_lock_path()
    lock_fd = None
    try:
        lock_fd = open(lock_path, 'w')
        sys.stderr.write("[brain-daemon] Acquiring startup lock...\n")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # Blocks until acquired
        sys.stderr.write("[brain-daemon] Lock acquired.\n")

        # Re-check under the lock — a caller we blocked behind may have already
        # brought up a healthy, current-code daemon while we waited. A bare 2s
        # ping can't tell a slow recall / a launchd relaunch (throttle + reload)
        # from death, so wait it out before forcing a (re)start.
        resp = _can_connect()
        if not resp.get("ok"):
            resp = _await_responsive(_GRACE_DEADLINE_S)
        if resp.get("ok") and not _code_changed(resp):
            sys.stderr.write("[brain-daemon] Daemon healthy (handled by another caller / recovered).\n")
            return True

        # Route the (re)start through launchd. `kickstart -k` kills any running
        # instance (covers healthy-but-stale AND hung-corpse) and respawns it in
        # one launchd-serialized call — no competing-spawn race with KeepAlive.
        if kickstart():
            # Wait past ThrottleInterval (10s) + embedder reload — a tighter
            # window gives up while the daemon is still coming back and would
            # re-kickstart it, resetting the throttle (the self-sustaining storm).
            t0 = time.monotonic()
            resp = _await_responsive(_KICKSTART_DEADLINE_S)
            if resp.get("ok") and not _code_changed(resp):
                sys.stderr.write("[brain-daemon] Daemon ready via launchd (took %.1fs)\n"
                                 % (time.monotonic() - t0))
                return True
            sys.stderr.write("[brain-daemon] Daemon not ready within %.0fs after kickstart\n"
                             % _KICKSTART_DEADLINE_S)
            return False

        # kickstart did not succeed. Decide carefully whether a direct spawn is
        # safe — spawning a rival when one already exists IS the Errno-48 orphan
        # storm (2026-06-05).
        #
        # (a) A daemon is still responsive RIGHT NOW. Re-ping here — the `resp`
        #     from the pre-kickstart re-check is stale: `kickstart -k` SIGKILLs
        #     the incumbent before respawning, so a kickstart that failed/timed
        #     out AFTER the kill leaves the port free. Only defer to an incumbent
        #     that is actually still up (deferring on the stale snapshot would
        #     report "ready" with nothing serving). If it IS still up, kickstart
        #     failing means THIS context couldn't reach launchd, not that the
        #     daemon is unmanaged — defer, never spawn a competitor.
        resp = _can_connect()
        if resp.get("ok"):
            sys.stderr.write(
                "[brain-daemon] launchd kickstart unavailable but a daemon is "
                "already responsive — deferring (NOT spawning a competitor; it "
                "may run stale code until launchd cycles it).\n")
            return True

        # (b) Nothing is serving. Spawn directly ONLY if launchd genuinely is
        #     not managing the daemon (fresh install). If launchd DOES manage it
        #     but kickstart failed transiently, let KeepAlive bring it up rather
        #     than racing a manual spawn. (Role note: _relaunch_daemon KILLS in
        #     this same managed+kickstart-failed state — correct there, where the
        #     target is a live corpse KeepAlive can't respawn past. Here "down"
        #     means the process exited, so KeepAlive owns the respawn. Don't
        #     unify the two arms.)
        if manages():
            sys.stderr.write(
                "[brain-daemon] launchd manages the daemon but kickstart failed "
                "and nothing is serving — leaving it to KeepAlive.\n")
            return False

        # No launchd managing the daemon (fresh install / not bootstrapped).
        # No KeepAlive to race here, so spawn directly — still under the lock.
        sys.stderr.write("[brain-daemon] launchd not managing daemon — spawning directly\n")
        if port_is_occupied():
            sys.stderr.write("[brain-daemon] Port occupied but unresponsive — killing zombie\n")
            kill_daemon()
            time.sleep(1)
        spawn_detached_daemon(db_path)
        for i in range(20):  # 10 seconds max
            time.sleep(0.5)
            if _can_connect().get("ok"):
                sys.stderr.write("[brain-daemon] Daemon ready (took %.1fs)\n" % ((i + 1) * 0.5))
                return True
        sys.stderr.write("[brain-daemon] Daemon failed to start within 10s\n")
        return False

    except Exception as e:
        sys.stderr.write("[brain-daemon] ensure_daemon error: %s\n" % e)
        return False
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def stop_daemon():
    """Gracefully stop the daemon."""
    resp = send_command("shutdown", timeout=5.0)
    if not resp.get("ok"):
        kill_daemon()
        return
    for _ in range(20):
        if not is_daemon_running():
            return
        time.sleep(0.1)
    kill_daemon()


# ─── Hung-daemon recovery ───
#
# A daemon that hangs after host sleep ("corpse") keeps its port bound but
# services nothing, so launchd's crash-respawn never fires and nothing
# inside the frozen process can rescue it. Recovery must come from OUTSIDE
# the frozen process: force it to die and let launchd respawn it. Every recovery caller — the MCP health
# monitor and the recall hook — routes through recover_daemon(); shared
# cooldown + circuit-breaker state (one /tmp file) keeps them from fighting.

_RECOVERY_COOLDOWN_S = 30.0      # don't issue a 2nd restart while one is in flight
_RECOVERY_MAX_ATTEMPTS = 5       # circuit breaker: stop after this many…
_RECOVERY_WINDOW_S = 600.0       # …within this sliding window, then surface loudly


def _read_recovery_state() -> dict:
    try:
        with open(get_recovery_state_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"last_attempt": 0.0, "attempts": 0}


def _write_recovery_state(last_attempt: float, attempts: int):
    try:
        with open(get_recovery_state_path(), "w") as f:
            json.dump({"last_attempt": last_attempt, "attempts": attempts}, f)
    except OSError:
        pass


def _relaunch_daemon(db_path: Optional[str]):
    """Bring a fresh daemon up. launchd is the canonical owner (kickstart -k);
    if launchd isn't managing it (kickstart fails), fall back to kill + Popen
    spawn via ensure_daemon.

    Role note — this kill fires even when launchd MANAGES the daemon, and that
    is deliberate (do not "unify" it with ensure_daemon's defer-to-KeepAlive
    arm): recovery's target is the hung CORPSE — a live process holding the
    port and serving nothing. KeepAlive only respawns on process EXIT, so it
    can never get past a live corpse; when kickstart -k (launchd's own killer)
    is unreachable, the manual SIGKILL is what produces the exit KeepAlive
    needs. ensure_daemon's managed arm defers instead because its "down" means
    the process already exited — there, KeepAlive genuinely owns the respawn."""
    if kickstart():
        return
    # kickstart failed. Re-ping before ANY kill decision: recover_daemon's
    # single 2s ping can't tell slow/busy from dead (a recall runs 10-17s), and
    # a kickstart can fail transiently while a healthy daemon serves. A daemon
    # that answers NOW must be deferred to, never SIGKILLed mid-request —
    # ensure_daemon's ladder has this rung; it was missing here.
    resp = _can_connect()
    if resp.get("ok"):
        sys.stderr.write("[brain-daemon] kickstart unavailable but the daemon answers "
                         "pings again — deferring (recovery not needed).\n")
        return
    # A manual kill+respawn only converges from the daemon's SOURCE checkout:
    # a linked worktree / 2nd clone would SIGKILL the shared daemon and then
    # refuse to spawn (pure client, can't converge its own code), leaving
    # nothing serving. Defer loudly instead.
    if not _is_daemon_source(resp):
        sys.stderr.write("[brain-daemon] kickstart unavailable and this checkout is not the "
                         "daemon's source — deferring kill+respawn to launchd/source "
                         "(NOT killing the shared daemon).\n")
        return
    # No launchd (or kickstart failed) and we ARE the source — own the kill + respawn.
    kill_daemon()
    if not db_path:
        db_dir = os.environ.get("BRAIN_DB_DIR",
                                os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
        db_path = os.path.join(db_dir, "brain.db")
    ensure_daemon(db_path)


def recover_daemon(db_path: Optional[str] = None) -> bool:
    """Recover a hung/unreachable daemon. Returns True iff it's healthy now.

    The single recovery path for every caller. Idempotent and safe to call
    repeatedly — it no-ops when:
      - the daemon already answers a ping (and clears any failure streak),
      - maintenance mode is on (intentional shutdown),
      - a restart is still in flight (cooldown), or
      - the circuit breaker has tripped (too many failed restarts).
    Otherwise it forces a restart and returns False; the next ping confirms.
    """
    if is_maintenance_mode():
        return False

    # Fast single-ping confirm — recover_daemon runs on synchronous hook paths
    # (recall/Stop/Edit, host timeouts 5-21s) and the MCP monitor, so it must NOT
    # pay the multi-second grace ensure_daemon uses; the cooldown + circuit
    # breaker below are what prevent restart storms. A transient blip is absorbed
    # by the cooldown; a genuine corpse is force-restarted (and re-confirmed by
    # the next caller's ping, not by blocking this one).
    if is_daemon_responsive():
        if _read_recovery_state().get("attempts"):
            _write_recovery_state(0.0, 0)  # healthy again — reset the streak
        return True

    now = time.time()
    state = _read_recovery_state()
    last_attempt = state.get("last_attempt", 0.0)
    attempts = state.get("attempts", 0)

    # Sliding window: an old failure streak ages out, so the breaker can't
    # lock recovery out forever after a long-past incident.
    if now - last_attempt > _RECOVERY_WINDOW_S:
        last_attempt, attempts = 0.0, 0

    if now - last_attempt < _RECOVERY_COOLDOWN_S:
        return False  # a restart is already in flight — give it time to boot

    if attempts >= _RECOVERY_MAX_ATTEMPTS:
        sys.stderr.write(
            "[brain-daemon] recovery circuit OPEN — %d restarts in %ds did not "
            "revive the daemon; not restarting again. Investigate manually.\n"
            % (attempts, int(_RECOVERY_WINDOW_S)))
        return False

    attempts += 1
    _write_recovery_state(now, attempts)
    sys.stderr.write("[brain-daemon] daemon unresponsive — forcing restart (attempt %d)\n" % attempts)
    _relaunch_daemon(db_path)
    return False  # not yet verified; the next ping confirms recovery
