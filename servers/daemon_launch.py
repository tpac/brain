"""
brain — Daemon Launch Primitives

The ONE home for "how the daemon gets spawned / killed / talked-to-via-launchd".
Both daemon_client (outside callers: hooks, MCP, CLI) and daemon_server (the
restart path) consume these as public names — no private cross-module imports,
no server→client layering inversion.

POLICY lives in the callers (ensure_daemon's decision ladder, _perform_restart's
clean-exit-when-managed); this module owns only the MECHANISMS. launchd is the
sole spawner where present (Errno-48 fix 2026-06-04): spawn_detached_daemon is
legitimate ONLY on a no-launchd platform (Linux / fresh install) — spawning a
rival alongside KeepAlive is the orphan-storm class of bug.
"""

import os
import signal
import socket
import subprocess
import sys
import time

from .daemon_config import (
    DAEMON_CPU_ENV, REPO_ROOT,
    get_daemon_addr, get_daemon_log_path, get_pid_path,
)

# launchd service label (macOS). The daemon runs as this launchd job
# (KeepAlive=true), so external recovery force-restarts a hung daemon with
# `launchctl kickstart -k` and lets launchd own the respawn.
LAUNCHD_LABEL = "com.brain.daemon"


def service_target() -> str:
    """The launchd service target for this user's daemon: gui/{uid}/{label}."""
    return "gui/{}/{}".format(os.getuid(), LAUNCHD_LABEL)


def debugger_friendly_python() -> str:
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

    candidates = [
        os.path.join(REPO_ROOT, 'venv', 'bin', 'python'),
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


def kickstart() -> bool:
    """Ask launchd to (re)start the daemon. `kickstart -k` kills any running
    instance and respawns it in one launchd-serialized call. Returns True iff
    launchd accepted it (rc 0); False means launchd isn't managing the daemon
    (fresh install / not bootstrapped) and the caller must spawn it itself.

    The SOLE restart primitive — every (re)start (boot-path code-change in
    ensure_daemon, hung-corpse recovery in _relaunch_daemon) routes through
    launchd so concurrent callers + KeepAlive can't race competing spawns.
    That race was the Errno-48 storm of 2026-06-04."""
    try:
        result = subprocess.run(["launchctl", "kickstart", "-k", service_target()],
                                timeout=10, capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


def manages() -> bool:
    """True iff launchd has the daemon's LaunchAgent loaded in this user's GUI
    domain.

    A False return AUTHORIZES ensure_daemon()'s direct-spawn fallback — the exact
    path that creates a competing orphan squatting the port/lock (Errno-48 storm
    2026-06-05; orphan-daemon incident 2026-07-03). So False must mean launchd
    DEFINITIVELY does not manage the daemon, never "couldn't tell right now".

    Two DIFFERENT failures must not be conflated:
      • `launchctl` binary MISSING (FileNotFoundError) — the platform has no
        launchd at all (Linux). Deterministic absence → return False so
        ensure_daemon()'s direct spawn can bootstrap. This is the ONLY path that
        starts the daemon off-macOS, so it must stay False (contract pinned by
        test_manages_returns_false_when_launchctl_missing).
      • `launchctl` PRESENT but the call failed — a timeout, a boot context that
        can't address `gui/<uid>`, an unexpected non-zero. INDETERMINATE, not
        evidence of absence: retry, and if still indeterminate assume managed so
        the caller defers to KeepAlive instead of spawning a rival. (The original
        `return result.returncode == 0` / `except Exception: return False`
        conflated this transient failure with absence — the very trap this
        docstring's first version warned about for kickstart, but the code fell
        into it here, orphaning a daemon on 2026-07-03.)

    `launchctl print gui/<uid>/<label>` returns 0 when managed and a clean
    non-zero (113, "Could not find service …") when genuinely absent.
    """
    for _attempt in range(3):
        try:
            result = subprocess.run(["launchctl", "print", service_target()],
                                    timeout=10, capture_output=True, text=True)
        except FileNotFoundError:
            return False  # no launchctl binary = no launchd platform → spawn directly
        except Exception:
            continue  # transient (timeout / other) — retry, never conclude absence
        if result.returncode == 0:
            return True  # definitively managed
        combined = ((result.stdout or "") + (result.stderr or "")).lower()
        if "could not find service" in combined:
            return False  # definitively NOT managed (clean not-found)
        # Unexpected non-zero — indeterminate, retry.
    # No definitive answer after retries → assume managed (defer to KeepAlive
    # rather than spawn a competing orphan).
    return True


def port_is_occupied() -> bool:
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


def kill_daemon():
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
    # Clear the stale PID hint only. The LOCK file is never unlinked — the
    # kernel releases the dead process's flock on its own, and unlinking the
    # path while ANOTHER holder (a daemon, an ensure_daemon mid-ladder) has it
    # open would let a third process lock a fresh inode at the same path: two
    # "singleton" holders, the two-writer corruption class. No code path
    # unlinks a lock file.
    try:
        if os.path.exists(pid_path):
            os.unlink(pid_path)
    except Exception:
        pass


def daemon_argv(db_path: str) -> list:
    """The one startup command every spawn site uses: debugger-friendly
    interpreter + a `-c` bootstrap that pins sys.path to this checkout and
    BRAIN_DB_DIR to the daemon's DB dir."""
    startup = (
        'import sys, os; '
        'sys.path.insert(0, %r); '
        'os.environ["BRAIN_DB_DIR"] = %r; '
        'from servers.daemon_server import BrainDaemon; '
        'd = BrainDaemon(%r); d.start()'
        % (REPO_ROOT,
           os.environ.get('BRAIN_DB_DIR', os.path.dirname(db_path)),
           db_path))
    return [debugger_friendly_python(), '-c', startup]


def spawn_detached_daemon(db_path: str):
    """The ONE detached daemon spawn — hardened: debugger-friendly interpreter,
    stdin from devnull, stdout/stderr appended to the daemon log (devnull if the
    log can't open — the spawn must not fail on a logging problem), its own
    session, and the full CPU-only env.

    Only legitimate on a no-launchd platform (fresh install / Linux) — where
    launchd manages the daemon, spawning is launchd's job (KeepAlive/kickstart);
    a detached rival here is the orphan-storm bug class."""
    try:
        log_fp = open(get_daemon_log_path(db_path), 'a')
    except Exception:
        log_fp = open(os.devnull, 'w')
    try:
        with open(os.devnull, 'r') as devnull:
            subprocess.Popen(
                daemon_argv(db_path),
                stdin=devnull,
                stdout=log_fp,
                stderr=log_fp,
                start_new_session=True,
                env={**os.environ, **DAEMON_CPU_ENV},
            )
    finally:
        log_fp.close()
