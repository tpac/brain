"""
brain — Persistent Daemon (Facade)

Backwards-compatible re-export of all public symbols.
The actual implementation lives in:
  - daemon_config.py   — constants, paths, fingerprinting
  - daemon_server.py   — BrainDaemon class (socket server + thread pool)
  - daemon_dispatch.py — table-driven command routing (42 commands)
  - daemon_client.py   — send_command, ensure_daemon, lifecycle

PROTOCOL:
  Client sends: {"cmd": "...", "args": {...}}\\n
  Server sends: {"ok": true, "result": {...}}\\n
"""

# Re-export config
from .daemon_config import (
    IDLE_TIMEOUT_SECONDS,
    AUTOSAVE_INTERVAL_SECONDS,
    SOCKET_BACKLOG,
    MAX_MESSAGE_SIZE,
    THREAD_POOL_SIZE,
    _code_fingerprint,
    _CODE_FINGERPRINT,
    get_socket_path,
    get_pid_path,
    get_lock_path,
    get_status_path,
)

# Re-export server
from .daemon_server import BrainDaemon

# Re-export client
from .daemon_client import (
    send_command,
    is_daemon_running,
    ensure_daemon,
    _kill_daemon,
    stop_daemon,
    create_agent_db,
    list_agent_changes,
    cleanup_agent_db,
)

# Re-export dispatch table (for introspection/testing)
from .daemon_dispatch import COMMAND_TABLE


# ─── CLI Entry Point ───

if __name__ == "__main__":
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(description="brain persistent daemon")
    parser.add_argument("action", choices=["start", "stop", "status", "restart"],
                       help="Daemon action")
    parser.add_argument("--db", help="Path to brain.db")
    args = parser.parse_args()

    if args.action == "start":
        if not args.db:
            print("Error: --db required for start", file=sys.stderr)
            sys.exit(1)
        if is_daemon_running():
            print("Daemon already running")
        else:
            if ensure_daemon(args.db):
                print("Daemon started")
            else:
                print("Failed to start daemon", file=sys.stderr)
                sys.exit(1)

    elif args.action == "stop":
        if is_daemon_running():
            stop_daemon()
            print("Daemon stopped")
        else:
            print("Daemon not running")

    elif args.action == "status":
        if is_daemon_running():
            resp = send_command("ping")
            if resp.get("ok"):
                print("Daemon running (PID {})".format(resp["result"]["pid"]))
            else:
                print("Daemon zombie (PID file exists but not responding)")
        else:
            print("Daemon not running")

    elif args.action == "restart":
        if is_daemon_running():
            stop_daemon()
            time.sleep(1)
        if args.db:
            if ensure_daemon(args.db):
                print("Daemon restarted")
            else:
                print("Failed to restart", file=sys.stderr)
                sys.exit(1)
        else:
            print("Error: --db required for restart", file=sys.stderr)
            sys.exit(1)
