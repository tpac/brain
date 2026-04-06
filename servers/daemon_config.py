"""
brain — Daemon Configuration

Constants, paths, and fingerprinting for the brain daemon.
Pure config — no classes, no side effects beyond CPU-only env setup.
"""

import os
import sys
import hashlib

# ─── Force CPU-only BEFORE any downstream import ───
# On macOS Apple Silicon, CoreML/Metal XPC connections cause SIGABRT
# in background/daemon processes that lack GPU context.
os.environ["ORT_DISABLE_ALL_ACCELERATORS"] = "1"
os.environ.setdefault("ONNX_PROVIDERS", "CPUExecutionProvider")
os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")

# ─── Constants ───

IDLE_TIMEOUT_SECONDS = 4 * 60 * 60  # 4 hours
AUTOSAVE_INTERVAL_SECONDS = 60  # Save every 60 seconds if dirty
SOCKET_BACKLOG = 5
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max message
THREAD_POOL_SIZE = 1  # Serial: SQLite single-connection deadlocks on concurrent access.
                       # Future: in-memory graph eliminates this. Pool size=1 for now.
DAEMON_HOST = ""  # Empty string = all interfaces (IPv4+IPv6), fixes macOS localhost→::1
DAEMON_PORT = 47200 + (os.getuid() % 100)  # Per-user port to avoid collisions


# ─── Code Fingerprinting ───

def _code_fingerprint() -> str:
    """Return a deterministic fingerprint of the server code files.
    Changes when any .py file in servers/ is modified."""
    try:
        servers_dir = os.path.dirname(os.path.abspath(__file__))
        mtimes = []
        for f in sorted(os.listdir(servers_dir)):
            if f.endswith('.py'):
                mtimes.append("{}:{}".format(f, os.path.getmtime(os.path.join(servers_dir, f))))
        return hashlib.md5("|".join(mtimes).encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


# Captured at import time — represents the code version this process loaded
_CODE_FINGERPRINT = _code_fingerprint()


# ─── Path Helpers ───

def get_daemon_addr():
    """Get (host, port) for TCP daemon connection."""
    return (DAEMON_HOST, DAEMON_PORT)


def get_socket_path() -> str:
    """DEPRECATED: Use get_daemon_addr() for TCP. Kept for cleanup of stale sockets."""
    return os.path.join("/tmp", "brain-daemon-{}.sock".format(os.getuid()))


def get_pid_path() -> str:
    """Get the daemon PID file path."""
    return os.path.join("/tmp", "brain-daemon-{}.pid".format(os.getuid()))


def get_lock_path() -> str:
    """Get the daemon lock file path for startup serialization."""
    return os.path.join("/tmp", "brain-daemon-{}.lock".format(os.getuid()))


def get_status_path() -> str:
    """Get the daemon status file path (read by statusline script)."""
    return os.path.join("/tmp", "brain-status-{}.json".format(os.getuid()))


def get_maintenance_path() -> str:
    """Get the maintenance mode lock file path."""
    return os.path.join("/tmp", "brain-maintenance-{}.lock".format(os.getuid()))


def is_maintenance_mode() -> bool:
    """Check if brain is in maintenance mode (DB operations, no daemon)."""
    return os.path.exists(get_maintenance_path())
