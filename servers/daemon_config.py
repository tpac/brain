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

IDLE_TIMEOUT_SECONDS = 4 * 60 * 60  # 4 hours — shutdown after this
# S2 scheduling moved to Brain.run_maintenance_if_due (see brain.py).
# Brain owns MAINTENANCE_IDLE_THRESHOLD_SECONDS and MAINTENANCE_MIN_INTERVAL_SECONDS
# as class constants; daemon just polls. Old S2_IDLE_THRESHOLD / S2_MIN_INTERVAL
# deleted along with the scheduling logic in daemon_server._serve.
AUTOSAVE_INTERVAL_SECONDS = 60  # Save every 60 seconds if dirty
SOCKET_BACKLOG = 5
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max message
THREAD_POOL_SIZE = max(4, (os.cpu_count() or 4) // 2)
# Scales hook/request concurrency with the host. SQLite WAL allows concurrent
# readers, and daemon_server serializes writers — no deadlock risk. Embedder
# (fastembed + ORT 1.24.4) is thread-safe for concurrent InferenceSession.run().
DAEMON_HOST = ""  # Empty string = all interfaces (IPv4+IPv6), fixes macOS localhost→::1
DAEMON_PORT = 47200 + (os.getuid() % 100)  # Per-user port to avoid collisions

# launchd service label (macOS). The daemon runs as this launchd job
# (KeepAlive=true), so external recovery force-restarts a hung daemon with
# `launchctl kickstart -k` and lets launchd own the respawn.
LAUNCHD_LABEL = "com.brain.daemon"


# ─── Developer mode ───
# Set BRAIN_DEV_MODE=1 in your shell rc to opt out of safety nets that
# are appropriate for end users but actively hostile while developing
# the brain itself (suspend-detector auto-restart, future hung-corpse
# autokill, anything else that pulls the rug out from under you mid-
# investigation). In dev mode the daemon logs loudly when these would
# have fired, but takes no action — you control the lifecycle.
#
# IMPORTANT: this flag must NOT be set in environments where the brain
# is shipped as a plugin to other users. End-user safety nets exist
# because most users won't run py-spy / lldb / brain CLI to diagnose a
# hung daemon — they just see a non-functional plugin. Repackaging
# checklist must include "BRAIN_DEV_MODE unset / unexported".


def is_dev_mode() -> bool:
    """True if BRAIN_DEV_MODE is set to a truthy value."""
    return os.environ.get('BRAIN_DEV_MODE', '').strip().lower() in ('1', 'true', 'yes', 'on')


# ─── Identity binding ───
# Concrete names for the human operator and the agent (Anchor). Stamped
# onto S0 trace_events metadata at write time so each event independently
# records who said what. Source: ~/.config/brain/env (sourced by
# boot-brain.sh, inherited into daemon environment). Empty string when
# unset — DAL skips stamping, no placeholder sentinels.

def get_operator_name() -> str:
    """Canonical name of the current human partner. Empty if unset."""
    return os.environ.get('BRAIN_OPERATOR_NAME', '').strip()


def get_agent_name() -> str:
    """Canonical name of the agent (the brain's self-token). Empty if unset."""
    return os.environ.get('BRAIN_AGENT_NAME', '').strip()


# ─── Code Fingerprinting ───

def _fingerprint_dir(path: str) -> str:
    """Recursive content hash (md5) of every *.py under `path`, keyed by
    RELATIVE path + NUL-separated bytes. Recursive so changes in subpackages
    (scales/, db_backends/, ...) are detected; content-based + path-relative so
    identical code matches across checkouts; pure/path-parameterized for tests."""
    h = hashlib.md5()
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for f in sorted(g for g in files if g.endswith('.py')):
            rel = os.path.relpath(os.path.join(root, f), path).encode()
            with open(os.path.join(root, f), 'rb') as fh:
                h.update(rel + b'\0' + fh.read() + b'\0')
    return h.hexdigest()[:16]


def _code_fingerprint() -> str:
    """Fingerprint of server code by file CONTENT not mtime — mtime hashing made
    every worktree look "stale" with identical code → restart churn (2026-06-05)."""
    try:
        return _fingerprint_dir(os.path.dirname(os.path.abspath(__file__)))
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


def get_recovery_state_path() -> str:
    """Path to the cross-process daemon-recovery state file.

    Holds the cooldown timestamp + circuit-breaker failure count shared by
    every recovery caller (hooks, MCP health monitor) so concurrent callers
    don't double-restart a daemon mid-respawn or loop forever on one that
    can't come back."""
    return os.path.join("/tmp", "brain-recovery-{}.json".format(os.getuid()))
