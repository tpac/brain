"""
brain — Daemon Client

Client-side functions for communicating with the brain daemon.
Used by hook scripts, brain_mcp.py, and brain_cli.py.

Singleton guarantee: ensure_daemon() uses fcntl.flock on a lock file.
First caller acquires lock, starts daemon, releases lock.
All other callers block on the lock, wake up to a running daemon.
No file markers, no polling races.
"""

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional


def _debugger_friendly_python() -> str:
    """Pick a Python interpreter the daemon can be spawned with.

    macOS SIP blocks debuggers (py-spy, lldb) from attaching to the
    system Python at /Applications/Xcode.app/.../Python. A user-managed
    venv Python is not protected and can be introspected live.

    Priority:
      1. $BRAIN_PYTHON env var (explicit override)
      2. <repo>/venv/bin/python (dev checkout)
      3. <plugin>/venv/bin/python (installed plugin)
      4. Fall back to sys.executable with a stderr warning

    Returning the first hit keeps future `sudo py-spy dump` / `lldb -p`
    calls actually usable when the daemon goes hot.
    """
    override = os.environ.get('BRAIN_PYTHON', '').strip()
    if override and os.path.exists(override):
        return override

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    candidates = [
        os.path.join(repo_root, 'venv', 'bin', 'python'),
        os.path.expanduser(
            '~/.claude/plugins/marketplaces/local-desktop-app-uploads/'
            'brain/venv/bin/python'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Fallback: warn once. The daemon still runs; debugging is just harder.
    if '/Xcode.app/' in sys.executable or sys.executable.startswith('/Applications/'):
        sys.stderr.write(
            '[brain-daemon] WARN: spawning with SIP-protected Python (%s). '
            'Set BRAIN_PYTHON to a user-managed python (e.g. repo venv) '
            'so py-spy/lldb can attach when the daemon spins.\n' % sys.executable)
    return sys.executable

from .daemon_config import (
    _code_fingerprint, _CODE_FINGERPRINT, _IS_WORKTREE, LAUNCHD_LABEL,
    get_daemon_addr, get_socket_path, get_pid_path, get_lock_path, get_status_path,
    get_recovery_state_path, is_maintenance_mode,
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


def _code_changed(resp: dict) -> bool:
    """True if the responding daemon runs different code than this checkout AND
    restarting it would actually pick that code up. Conservative — returns False
    when either fingerprint is unknown, so an indeterminate signal never
    triggers a restart.

    Worktree exception: a linked git worktree NEVER reports code-changed. The
    daemon is a singleton launched from the primary checkout, so kickstarting it
    reboots that checkout — never the worktree's code. Letting a worktree force
    restarts produced a churn loop that can't converge (2026-06-06). Worktree
    edits are picked up by merging to the primary checkout, not by restart."""
    if _IS_WORKTREE:
        return False
    daemon_fp = resp.get("result", {}).get("code_fingerprint", "")
    # Use the import-time constant, not a fresh _code_fingerprint() — this
    # process is short-lived (hook/CLI) so its code can't change underneath it,
    # and recomputing now re-reads every servers/**/*.py on each call (several
    # MB) where ensure_daemon calls this 2-3× per invocation.
    current_fp = _CODE_FINGERPRINT
    return bool(current_fp != "unknown" and daemon_fp and daemon_fp != current_fp)


def ensure_daemon(db_path: str) -> bool:
    """Ensure the daemon is running AND on current code. Returns True if ready.

    launchd owns the daemon lifecycle (com.brain.daemon: KeepAlive, RunAtLoad).
    This function only PINGS and, when a (re)start is needed, routes it through
    launchd via `launchctl kickstart -k` (_launchd_kickstart). It never kills +
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

    # Fast path: running, responsive, and on current code → nothing to do.
    resp = _can_connect()
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
        # brought up a healthy, current-code daemon while we waited.
        resp = _can_connect()
        if resp.get("ok") and not _code_changed(resp):
            sys.stderr.write("[brain-daemon] Daemon already healthy (handled by another caller).\n")
            return True

        # Route the (re)start through launchd. `kickstart -k` kills any running
        # instance (covers healthy-but-stale AND hung-corpse) and respawns it in
        # one launchd-serialized call — no competing-spawn race with KeepAlive.
        if _launchd_kickstart():
            for i in range(30):  # 15s max — a fresh boot reloads the embedder (~4-6s)
                time.sleep(0.5)
                resp = _can_connect()
                if resp.get("ok") and not _code_changed(resp):
                    sys.stderr.write("[brain-daemon] Daemon ready via launchd (took %.1fs)\n" % ((i + 1) * 0.5))
                    return True
            sys.stderr.write("[brain-daemon] Daemon not ready within 15s after kickstart\n")
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
        #     than racing a manual spawn.
        if _launchd_manages_daemon():
            sys.stderr.write(
                "[brain-daemon] launchd manages the daemon but kickstart failed "
                "and nothing is serving — leaving it to KeepAlive.\n")
            return False

        # No launchd managing the daemon (fresh install / not bootstrapped).
        # No KeepAlive to race here, so spawn directly — still under the lock.
        sys.stderr.write("[brain-daemon] launchd not managing daemon — spawning directly\n")
        if _port_is_occupied():
            sys.stderr.write("[brain-daemon] Port occupied but unresponsive — killing zombie\n")
            _kill_daemon()
            time.sleep(1)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(os.path.dirname(db_path), "daemon.log")
        with open(log_path, 'a') as log_fd_file, open(os.devnull, 'r') as devnull:
            daemon_python = _debugger_friendly_python()
            subprocess.Popen(
                [daemon_python, '-c',
                 'import sys, os; sys.path.insert(0, %r); '
                 'os.environ["BRAIN_DB_DIR"] = %r; '
                 'from servers.daemon_server import BrainDaemon; '
                 'd = BrainDaemon(%r); d.start()' % (parent_dir,
                     os.environ.get('BRAIN_DB_DIR', os.path.dirname(db_path)),
                     db_path)],
                stdin=devnull,
                stdout=log_fd_file,
                stderr=log_fd_file,
                start_new_session=True,
                env={
                    **os.environ,
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "ORT_DISABLE_ALL_ACCELERATORS": "1",
                    "ONNX_PROVIDERS": "CPUExecutionProvider",
                    "PYTORCH_MPS_DISABLE": "1",
                    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                },
            )
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


def _port_is_occupied() -> bool:
    """Check if something is holding our daemon port."""
    addr = get_daemon_addr()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(addr)
        s.close()
        return False
    except OSError:
        s.close()
        return True


def _kill_daemon():
    """Kill a running daemon. Escalates SIGTERM → SIGKILL if needed."""
    pid_path = get_pid_path()
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        sys.stderr.write("[brain-daemon] Killing daemon PID={}\n".format(pid))

        os.kill(pid, signal.SIGTERM)

        for _ in range(15):
            time.sleep(0.2)
            try:
                os.kill(pid, 0)
            except OSError:
                break
        else:
            sys.stderr.write("[brain-daemon] SIGTERM failed, SIGKILL PID={}\n".format(pid))
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
            except OSError:
                pass
    except Exception as e:
        sys.stderr.write("[brain-daemon] Kill failed: {}\n".format(e))
    for path in [pid_path, get_lock_path()]:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass


def stop_daemon():
    """Gracefully stop the daemon."""
    resp = send_command("shutdown", timeout=5.0)
    if not resp.get("ok"):
        _kill_daemon()
        return
    for _ in range(20):
        if not is_daemon_running():
            return
        time.sleep(0.1)
    _kill_daemon()


def restart_daemon(db_path: str = None) -> bool:
    """Stop + start daemon."""
    stop_daemon()
    time.sleep(1)
    if not db_path:
        db_dir = os.environ.get("BRAIN_DB_DIR", os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
        db_path = os.path.join(db_dir, "brain.db")
    return ensure_daemon(db_path)


# ─── Hung-daemon recovery ───
#
# A daemon that hangs after host sleep ("corpse") keeps its port bound but
# services nothing, so launchd's crash-respawn never fires and the daemon's
# own in-process suspend detector can't act (no thread is scheduled to send
# the SIGTERM). Recovery must come from OUTSIDE the frozen process: force it
# to die and let launchd respawn it. Every recovery caller — the MCP health
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


def _launchd_kickstart() -> bool:
    """Ask launchd to (re)start the daemon. `kickstart -k` kills any running
    instance and respawns it in one launchd-serialized call. Returns True iff
    launchd accepted it (rc 0); False means launchd isn't managing the daemon
    (fresh install / not bootstrapped) and the caller must spawn it itself.

    The SOLE restart primitive — every (re)start (boot-path code-change in
    ensure_daemon, hung-corpse recovery in _relaunch_daemon) routes through
    launchd so concurrent callers + KeepAlive can't race competing spawns.
    That race was the Errno-48 storm of 2026-06-04."""
    label = "gui/{}/{}".format(os.getuid(), LAUNCHD_LABEL)
    try:
        result = subprocess.run(["launchctl", "kickstart", "-k", label],
                                timeout=10, capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


def _launchd_manages_daemon() -> bool:
    """True iff launchd has the daemon's LaunchAgent loaded in this user's GUI
    domain.

    This is DISTINCT from `_launchd_kickstart()` succeeding. kickstart can
    return nonzero for transient or contextual reasons — a boot context that
    can't address `gui/<uid>`, a timeout — even while launchd IS managing the
    daemon. Conflating "kickstart failed" with "launchd absent" is what let a
    competing direct-spawn orphan be created and squat the port (Errno-48
    storm, 2026-06-05). Gate the no-launchd fallback on THIS, not on kickstart's
    return code."""
    label = "gui/{}/{}".format(os.getuid(), LAUNCHD_LABEL)
    try:
        result = subprocess.run(["launchctl", "print", label],
                                timeout=10, capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


def _relaunch_daemon(db_path: Optional[str]):
    """Bring a fresh daemon up. launchd is the canonical owner (kickstart -k);
    if launchd isn't managing it (kickstart fails), fall back to kill + Popen
    spawn via ensure_daemon."""
    if _launchd_kickstart():
        return
    # No launchd (or kickstart failed) — own the kill + respawn.
    _kill_daemon()
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


# ─── Agent DB Isolation ───

def create_agent_db(agent_id: str, source_db: Optional[str] = None) -> str:
    """Copy production brain.db to /tmp for agent isolation. Returns path."""
    import shutil
    if source_db is None:
        source_db = os.path.join(
            os.environ.get("BRAIN_DB_DIR",
                           os.path.join(os.path.expanduser("~"), "AgentsContext", "brain")),
            "brain.db")
    dest = os.path.join("/tmp", "brain-agent-{}.db".format(agent_id))
    shutil.copy2(source_db, dest)
    return dest


def list_agent_changes(agent_db_path: str, since: str) -> list:
    """List nodes created in agent DB after timestamp."""
    import sqlite3
    db = sqlite3.connect(agent_db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT id, type, title, content, created_at, confidence "
        "FROM nodes WHERE created_at > ? ORDER BY created_at",
        (since,)).fetchall()
    result = [dict(r) for r in rows]
    db.close()
    return result


def cleanup_agent_db(agent_db_path: str):
    """Delete agent's DB copy."""
    try:
        if os.path.exists(agent_db_path):
            os.unlink(agent_db_path)
    except Exception:
        pass
