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


def ensure_daemon(db_path: str) -> bool:
    """Start the daemon if not running. Returns True if daemon is ready.

    PID file is written before the socket is bound (brain+embedder loading
    takes ~1-2s). We retry pings before declaring zombie.
    """
    # Clean stale lock: lock exists but no live process
    lock_path = get_lock_path()
    pid_path = get_pid_path()
    if os.path.exists(lock_path) and not os.path.exists(pid_path):
        sys.stderr.write("[brain-daemon] Removing stale lock (no PID file)\n")
        try:
            os.unlink(lock_path)
        except Exception:
            pass
    elif os.path.exists(lock_path) and os.path.exists(pid_path):
        try:
            with open(pid_path) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # Check if process is alive
        except (OSError, ValueError):
            sys.stderr.write("[brain-daemon] Removing stale lock (PID {} not alive)\n".format(
                pid if 'pid' in dir() else '?'))
            for p in [lock_path, pid_path]:
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    if is_daemon_running():
        # Daemon process exists — wait for socket to be ready
        for attempt in range(25):  # 5 seconds total
            resp = send_command("ping", timeout=2.0)
            if resp.get("ok"):
                # Check if code has been updated since daemon loaded
                daemon_fp = resp.get("result", {}).get("code_fingerprint", "")
                current_fp = _code_fingerprint()
                if current_fp != "unknown" and daemon_fp != current_fp:
                    sys.stderr.write(
                        "[brain-daemon] Code changed (daemon={}, current={}) — restarting\n"
                        .format(daemon_fp[:12] or "none", current_fp[:12]))
                    _kill_daemon()
                    break  # Fall through to start below
                return True
            time.sleep(0.2)
        else:
            # Still not responding after 5s — truly zombie, kill and restart
            sys.stderr.write(
                "[brain-daemon] Killing zombie daemon (PID alive but unresponsive for 5s)\n")
            _kill_daemon()

    # Spawn daemon as a detached subprocess.
    # subprocess.Popen (not fork) — macOS Accelerate/Metal uses XPC connections
    # that are invalid in forked children, causing SIGABRT. A clean subprocess
    # with CPU-only env vars avoids this entirely.
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

    # Wait for daemon to become ready
    for attempt in range(50):  # 10 seconds total
        resp = send_command("ping", timeout=2.0)
        if resp.get("ok"):
            return True
        time.sleep(0.2)

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
