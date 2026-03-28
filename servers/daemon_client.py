"""
brain — Daemon Client

Client-side functions for communicating with the brain daemon.
Used by hook scripts, brain_mcp.py, and brain_cli.py.
"""

import json
import os
import signal
import socket
import sys
import time
from typing import Any, Dict, Optional

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
        # Check if process exists
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        # Process doesn't exist or PID file is corrupt
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


def _is_starting() -> bool:
    """Check if another caller recently spawned a daemon (within 15s)."""
    marker = get_pid_path() + ".starting"
    if os.path.exists(marker):
        try:
            age = time.time() - os.path.getmtime(marker)
            return age < 15
        except Exception:
            pass
    return False


def _mark_starting():
    """Mark that we're about to spawn a daemon."""
    marker = get_pid_path() + ".starting"
    try:
        with open(marker, 'w') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass


def _clear_starting():
    """Clear the starting marker."""
    marker = get_pid_path() + ".starting"
    try:
        os.unlink(marker)
    except Exception:
        pass


def _port_is_occupied() -> bool:
    """Check if something is holding our daemon port."""
    import socket as _socket
    addr = get_daemon_addr()
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        s.bind(addr)
        s.close()
        return False  # Port is free
    except OSError:
        s.close()
        return True  # Something is holding it


def ensure_daemon(db_path: str) -> bool:
    """Start the daemon if not running. Returns True if daemon is ready.

    Design: TCP connection is the health check and the mutex.
    1. Try to connect — if alive and responsive, done.
    2. If port occupied but unresponsive — zombie, kill it.
    3. If another caller is already starting one — wait for it.
    4. Otherwise — spawn and wait.
    """
    # Step 1: Already alive and responsive?
    resp = _can_connect()
    if resp.get("ok"):
        daemon_fp = resp.get("result", {}).get("code_fingerprint", "")
        current_fp = _code_fingerprint()
        if current_fp != "unknown" and daemon_fp and daemon_fp != current_fp:
            sys.stderr.write(
                "[brain-daemon] Code changed ({} → {}) — restarting\n"
                .format(daemon_fp[:12], current_fp[:12]))
            _kill_daemon()
            time.sleep(1)
        else:
            return True

    # Step 2: Port occupied but not responding? Zombie — kill it.
    if _port_is_occupied():
        sys.stderr.write("[brain-daemon] Port occupied but unresponsive — killing zombie\n")
        _kill_daemon()
        time.sleep(1)

    # Step 3: Someone else already starting?
    if _is_starting():
        sys.stderr.write("[brain-daemon] Another caller is starting daemon — waiting\n")
        for _ in range(20):
            time.sleep(0.5)
            if _can_connect().get("ok"):
                return True
        # Timed out — stale marker, proceed to spawn
        _clear_starting()

    # Step 4: Spawn daemon
    _mark_starting()
    import subprocess
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(os.path.dirname(db_path), "daemon.log")

    with open(log_path, 'a') as log_fd, open(os.devnull, 'r') as devnull:
        subprocess.Popen(
            [sys.executable, '-c',
             'import sys, os; sys.path.insert(0, %r); '
             'from servers.daemon_server import BrainDaemon; '
             'd = BrainDaemon(%r); d.start()' % (parent_dir, db_path)],
            stdin=devnull,
            stdout=log_fd,
            stderr=log_fd,
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

    # Step 5: Wait for it to respond
    for _ in range(20):  # 10 seconds max
        time.sleep(0.5)
        resp = _can_connect()
        if resp.get("ok"):
            _clear_starting()
            return True

    _clear_starting()
    sys.stderr.write("[brain-daemon] Daemon failed to start within 10s\n")
    return False


def _kill_daemon():
    """Kill a running daemon. Escalates SIGTERM → SIGKILL if needed."""
    pid_path = get_pid_path()
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        sys.stderr.write("[brain-daemon] Killing daemon PID={}\n".format(pid))

        # Try SIGTERM first (graceful)
        os.kill(pid, signal.SIGTERM)

        # Wait up to 3s for process to die
        for _ in range(15):
            time.sleep(0.2)
            try:
                os.kill(pid, 0)  # Check if still alive
            except OSError:
                break  # Dead — good
        else:
            # Still alive after 3s — SIGKILL (force)
            sys.stderr.write("[brain-daemon] SIGTERM failed, sending SIGKILL to PID={}\n".format(pid))
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
            except OSError:
                pass  # Already dead
    except Exception as e:
        sys.stderr.write("[brain-daemon] Kill failed: {}\n".format(e))
    # Clean up files (PID and lock)
    for path in [pid_path, get_lock_path()]:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass


def stop_daemon():
    """Gracefully stop the daemon. Waits up to 2s for clean exit."""
    resp = send_command("shutdown", timeout=5.0)
    if not resp.get("ok"):
        _kill_daemon()
        return
    # Wait for daemon to actually exit (select loop + cleanup)
    for _ in range(20):
        if not is_daemon_running():
            return
        time.sleep(0.1)
    # Didn't exit cleanly — force kill
    _kill_daemon()


def restart_daemon(db_path: str = None) -> bool:
    """Stop + start daemon. Returns True if new daemon is ready."""
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
