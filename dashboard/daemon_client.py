"""TCP client for the brain daemon.

Dashboard is a passive observer; this module is the only place that opens a
socket to the daemon. If the daemon is unreachable, callers fall back to
direct read-only SQLite queries via `dashboard.db`.
"""

import json
import os
import socket

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 47200 + (os.getuid() % 100)


def daemon_send(cmd: str, args=None, timeout: float = 10):
    """Send a command to the daemon, return result or None.

    Returns None on any failure (connection refused, timeout, daemon-side
    error). Callers must handle None — the dashboard never crashes because
    the daemon is down.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((DAEMON_HOST, DAEMON_PORT))
        payload = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        s.sendall(payload.encode("utf-8"))
        chunks = []
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            try:
                json.loads(b"".join(chunks))
                break
            except json.JSONDecodeError:
                continue
        s.close()
        resp = json.loads(b"".join(chunks))
        if resp.get("ok"):
            return resp.get("result")
        return None
    except Exception:
        return None


def daemon_alive() -> bool:
    """Quick check if daemon is responding."""
    return daemon_send("ping", timeout=3) is not None
