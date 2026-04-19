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
    _code_fingerprint, _CODE_FINGERPRINT,
    get_daemon_addr, get_socket_path, get_pid_path, get_lock_path, get_status_path,
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


def ensure_daemon(db_path: str) -> bool:
    """Start the daemon if not running. Returns True if daemon is ready.

    Uses fcntl.flock for singleton guarantee:
    - First caller acquires exclusive lock, starts daemon, releases lock.
    - All concurrent callers block on the lock.
    - When lock releases, they wake up and find daemon already running.
    - No races, no duplicate daemons.

    Maintenance mode: if the maintenance lock file exists, skip startup.
    Used during DB operations (VACUUM, schema changes, bulk deletes).
    """
    from .daemon_config import is_maintenance_mode
    if is_maintenance_mode():
        sys.stderr.write("[brain-daemon] Maintenance mode active — skipping startup\n")
        return False

    # Fast path: already running and responsive?
    resp = _can_connect()
    if resp.get("ok"):
        # Check if code changed — needs graceful restart
        daemon_fp = resp.get("result", {}).get("code_fingerprint", "")
        current_fp = _code_fingerprint()
        if current_fp != "unknown" and daemon_fp and daemon_fp != current_fp:
            sys.stderr.write(
                "[brain-daemon] Code changed ({} → {}) — requesting graceful restart\n"
                .format(daemon_fp[:12], current_fp[:12]))
            restart_resp = send_command("restart", timeout=5.0)
            if restart_resp.get("ok"):
                sys.stderr.write("[brain-daemon] Restart command sent, waiting...\n")
                # Wait for daemon to come back (embedder reload ~4-6s)
                for _ in range(16):
                    time.sleep(0.5)
                    if _can_connect().get("ok"):
                        return True
            sys.stderr.write("[brain-daemon] Graceful restart failed, will kill + respawn\n")
            _kill_daemon()
            time.sleep(1)
        else:
            return True

    # Slow path: need to start daemon. Acquire exclusive lock.
    lock_path = get_lock_path()
    lock_fd = None
    try:
        lock_fd = open(lock_path, 'w')
        sys.stderr.write("[brain-daemon] Acquiring startup lock...\n")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # Blocks until acquired
        sys.stderr.write("[brain-daemon] Lock acquired.\n")

        # Re-check after acquiring lock — another caller may have started it
        resp = _can_connect()
        if resp.get("ok"):
            sys.stderr.write("[brain-daemon] Daemon already started by another caller.\n")
            return True

        # Kill zombie if port occupied but not responding
        if _port_is_occupied():
            sys.stderr.write("[brain-daemon] Port occupied but unresponsive — killing zombie\n")
            _kill_daemon()
            time.sleep(1)

        # Spawn daemon
        sys.stderr.write("[brain-daemon] Spawning daemon...\n")
        import subprocess
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

        # Wait for it to respond
        for i in range(20):  # 10 seconds max
            time.sleep(0.5)
            resp = _can_connect()
            if resp.get("ok"):
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
