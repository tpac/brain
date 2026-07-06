"""
brain — Daemon Configuration

Constants, paths, and fingerprinting for the brain daemon.
Pure config — no classes, no side effects beyond CPU-only env setup.
"""

import os
import sys
import hashlib

# ─── Force CPU-only BEFORE any downstream import ───
# On macOS Apple Silicon, CoreML/Metal XPC connections cause SIGABRT in
# background/daemon processes that lack GPU context. DAEMON_CPU_ENV is the SINGLE
# source of the CPU-only invariant: the launchd plist, every spawn site's env=,
# and this import-time application all draw from it, so a daemon started fresh,
# restarted, or launchd-spawned runs with the IDENTICAL accelerator env (a prior
# drift left the restart path without VECLIB_MAXIMUM_THREADS / MPS_FALLBACK).
DAEMON_CPU_ENV = {
    "ORT_DISABLE_ALL_ACCELERATORS": "1",
    "ONNX_PROVIDERS": "CPUExecutionProvider",
    "PYTORCH_MPS_DISABLE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}
# ORT accelerator-disable is the load-bearing SIGABRT guard — force it even over
# an ambient value (matches pre-consolidation behavior); setdefault the rest so
# an explicit plist/operator value wins.
os.environ["ORT_DISABLE_ALL_ACCELERATORS"] = DAEMON_CPU_ENV["ORT_DISABLE_ALL_ACCELERATORS"]
for _cpu_k, _cpu_v in DAEMON_CPU_ENV.items():
    os.environ.setdefault(_cpu_k, _cpu_v)

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

# launchd's plist ThrottleInterval — launchd won't relaunch the daemon faster
# than this after an exit. The recovery deadlines in daemon_client DERIVE from it
# (they must exceed throttle + embedder reload, or a recovering daemon is
# mistaken for a corpse and re-kickstarted — the self-sustaining storm). Keep in
# sync with the plist's <key>ThrottleInterval</key> (Step 7 generates the plist
# from this + a contract test). SIGKILL grace is launchd's SIGTERM→SIGKILL window
# (~macOS default); SHUTDOWN_BACKSTOP_S must stay under it so we exit on our own
# terms.
LAUNCHD_THROTTLE_INTERVAL_S = 10
LAUNCHD_SIGKILL_GRACE_S = 20


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


def _is_worktree_checkout(repo_root: str) -> bool:
    """True if `repo_root` is a *linked* git worktree — its `.git` is a FILE
    (a `gitdir:` pointer) rather than a DIRECTORY (primary checkout) or absent
    (tarball install → False). Pure/path-parameterized for tests. Gates daemon
    staleness restarts; the why lives in `daemon_client._code_changed`."""
    return os.path.isfile(os.path.join(repo_root, ".git"))


# Absolute repo root of THIS checkout (servers/'s parent). Canonical home: the
# daemon ping reports it as `source_dir`; _code_changed compares against it.
# Bootstrap sites (brain_mcp, brain_cli, daemon_server.start) recompute it locally
# because they set sys.path BEFORE they can import this module.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Linked-worktree heuristic — FALLBACK used by _code_changed only when the daemon
# predates `source_dir` reporting; the exact signal is the source_dir comparison.
_IS_WORKTREE = _is_worktree_checkout(REPO_ROOT)


# ─── Path Helpers ───

def brain_tmp_dir() -> str:
    """Root dir for brain's ephemeral per-session/per-run files — recall
    candidates, surface selections, encoder/consolidation prompts, the
    current-stop marker. Honors BRAIN_TMP_DIR, defaults to /tmp.

    Production leaves BRAIN_TMP_DIR unset → /tmp. The seam is wired on BOTH
    sides for the production path: the daemon WRITES via this helper, and the
    out-of-process READERS honor the same env protocol — the PostToolUse hook
    (hooks/scripts/post_tool_trace.py, current-stop) and the dashboard
    (dashboard/queries/recalls.py judge-result, dashboard/server.py
    consolidation-prompt). Those readers are deliberately servers-decoupled, so
    they read os.environ.get('BRAIN_TMP_DIR','/tmp') directly rather than import
    this — keep them in sync if the default ever changes.

    BrainTestBase + IsolatedBrain set it to a per-test/run temp dir so two
    concurrent test processes don't collide on fixed filenames (hardcoded
    session_ids, fixed encoder-prompt counters) and the files are cleaned up
    with the temp dir instead of leaking into /tmp.

    The EVAL path is wired the same way: the longmem harness sets BRAIN_TMP_DIR
    to the per-item brain dir at a single seam (fresh_brain.create_fresh_eval_brain,
    alongside BRAIN_DB_DIR — every brain-building harness funnels through it),
    and the eval/script glob readers honor the same env protocol (inline
    os.environ.get, or this helper where the module already imports servers).
    To enumerate the readers, grep 'BRAIN_TMP_DIR' and 'brain_tmp_dir' across
    eval/ and scripts/ — they are NOT listed here because a hand-maintained
    inventory rots (this docstring already did once). So concurrent eval runs
    are isolated: each item's ephemeral dumps land in its own dir and readers
    never cross-copy another run's files.

    NOT for the daemon DISCOVERY paths (socket/pid/lock/status/maintenance) —
    those are the well-known uid-keyed rendezvous hooks and the statusline must
    find, and stay at /tmp regardless. NOT for user-facing post-mortem artifacts
    (mcp-crash, diagnose) — those want a fixed, findable location too.
    """
    return os.environ.get('BRAIN_TMP_DIR', '/tmp')


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


def get_daemon_log_path(db_path: str) -> str:
    """Path to the daemon log — where the daemon writes and where every spawn
    site redirects the successor's stdout/stderr. Honors BRAIN_DB_DIR, else the
    DB's own directory. Single source so a restart-spawned and a boot-spawned
    daemon can't log to different files under an env override."""
    db_dir = os.environ.get("BRAIN_DB_DIR") or os.path.dirname(db_path)
    return os.path.join(db_dir, "daemon.log")


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
