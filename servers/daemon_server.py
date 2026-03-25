"""
brain — Daemon Server

BrainDaemon class: loads Brain, serves commands over TCP localhost.
Thread pool (5 workers) handles concurrent connections.
Reads run without lock, writes serialize via _write_lock.
"""

import sys
import os
import json
import socket
import select
import signal
import time
import threading
import traceback
import atexit
import fcntl
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

from .daemon_config import (
    IDLE_TIMEOUT_SECONDS, AUTOSAVE_INTERVAL_SECONDS,
    SOCKET_BACKLOG, MAX_MESSAGE_SIZE, THREAD_POOL_SIZE,
    DAEMON_HOST, DAEMON_PORT,
    _CODE_FINGERPRINT,
    get_daemon_addr, get_socket_path, get_pid_path, get_lock_path, get_status_path,
)
from .daemon_dispatch import COMMAND_TABLE


class BrainDaemon:
    """Persistent Brain daemon that listens on TCP localhost."""

    def __init__(self, db_path: str, socket_path: Optional[str] = None):
        self.db_path = db_path
        self.socket_path = socket_path or get_socket_path()  # kept for stale cleanup
        self.daemon_addr = get_daemon_addr()
        self.pid_path = get_pid_path()
        self.brain = None
        self.server_socket = None
        self.running = False
        self.last_activity = time.time()
        self.dirty = False
        self.graph_changes = []  # In-memory graph mutation log
        self._write_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

    # Hook dispatch table: hook_name → (module_attr, marks_dirty)
    HOOK_TABLE = {
        "hook_recall": ("hook_recall", True),
        "hook_post_response_track": ("hook_post_response_track", True),
        "hook_idle_maintenance": ("hook_idle_maintenance", True),
        "hook_post_compact_reboot": ("hook_post_compact_reboot", True),
        "hook_pre_edit": ("hook_pre_edit", True),
        "hook_pre_bash_safety": ("hook_pre_bash_safety", False),
        "hook_pre_compact_save": ("hook_pre_compact_save", True),
        "hook_session_end": ("hook_session_end", True),
        "hook_stop_failure_log": ("hook_stop_failure_log", True),
        "hook_config_change_host": ("hook_config_change_host", True),
        "hook_post_bash_host_check": ("hook_post_bash_host_check", True),
        "hook_worktree_context": ("hook_worktree_context", True),
        "hook_worktree_cleanup": ("hook_worktree_cleanup", True),
    }

    def start(self):
        """Start the daemon — load brain, bind socket, serve."""
        # Acquire exclusive lock with retry (handles stale locks from crashes)
        lock_path = get_lock_path()
        self._lock_fd = open(lock_path, 'w')
        for _ in range(50):  # 5s total
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (IOError, OSError):
                time.sleep(0.1)
        else:
            self._lock_fd.close()
            self._log("Another daemon is starting — exiting duplicate (waited 5s)")
            return

        # Write PID file
        with open(self.pid_path, 'w') as f:
            f.write(str(os.getpid()))

        # Clean up stale Unix socket if it exists (migration from old protocol)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Load brain (expensive — done once)
        self._load_brain()

        # Bind TCP socket on localhost
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.daemon_addr)
        self.server_socket.listen(SOCKET_BACKLOG)
        self.server_socket.setblocking(False)

        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGHUP, self._handle_signal)

        atexit.register(self._cleanup)

        self.running = True
        self._log("Daemon started. PID={}, addr={}:{}, workers={}".format(
            os.getpid(), self.daemon_addr[0], self.daemon_addr[1], THREAD_POOL_SIZE))

        # Dashboard is now a separate process (brain_dashboard_standalone.py)
        # launched by .claude/launch.json — no longer embedded in daemon.

        # Start autosave thread
        autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
        autosave_thread.start()

        self._serve()

    def _load_brain(self):
        """Load the Brain instance + embedder."""
        try:
            import torch
            torch.backends.mps.is_available = lambda: False
            torch.backends.mps.is_built = lambda: False
        except ImportError:
            pass

        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent not in sys.path:
            sys.path.insert(0, parent)

        from servers.brain import Brain
        self.brain = Brain(self.db_path)
        self._log("Brain loaded from {}".format(self.db_path))

    def _serve(self):
        """Main event loop — accept connections, dispatch to thread pool."""
        last_idle_check = time.time()
        while self.running:
            # Check idle timeout every ~10 iterations (not every loop)
            now = time.time()
            if now - last_idle_check > 5.0:
                last_idle_check = now
                idle = now - self.last_activity
                if idle > IDLE_TIMEOUT_SECONDS:
                    self._log("Idle timeout ({}s). Shutting down.".format(int(idle)))
                    break

            try:
                # 0.5s select — balances responsiveness to shutdown vs CPU usage
                readable, _, _ = select.select([self.server_socket], [], [], 0.5)
            except (select.error, OSError):
                break

            for sock in readable:
                try:
                    client, _ = sock.accept()
                    client.settimeout(30.0)
                    # Submit to thread pool — non-blocking
                    self._pool.submit(self._handle_client, client)
                except Exception as e:
                    self._log("Accept error: {}".format(e))

        self._shutdown()

    def _handle_client(self, client: socket.socket):
        """Handle a single client connection (runs in thread pool)."""
        try:
            data = b""
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data or len(data) > MAX_MESSAGE_SIZE:
                    break

            if not data:
                return

            try:
                msg = json.loads(data.decode('utf-8').strip())
            except json.JSONDecodeError as e:
                self._send_error(client, "Invalid JSON: {}".format(e))
                return

            cmd = msg.get("cmd", "")
            args = msg.get("args", {})

            self.last_activity = time.time()
            result = self._dispatch(cmd, args)
            self._send_response(client, result)

        except Exception as e:
            try:
                self._send_error(client, "Internal error: {}".format(e))
            except Exception:
                pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _observe_command(self, cmd, args, result=None):
        """Emit command to dashboard observer channel (no-op if nobody's listening)."""
        try:
            from servers.brain_dashboard import emit, has_listeners
            if not has_listeners():
                return
            obs_args = {k: str(v)[:120] for k, v in (args or {}).items()}
            obs_result = {k: str(v)[:300] for k, v in result.items()} if result else None
            emit("command", command=cmd, args=obs_args, result=obs_result)
        except Exception:
            pass

    def _locked_exec(self, fn, cmd, args):
        """Acquire write lock with timeout, execute fn, observe result."""
        if not self._write_lock.acquire(timeout=10.0):
            self._log("Write lock timeout (10s) for: {}".format(cmd))
            return {"ok": False, "error": "Write lock timeout — another operation is holding the lock"}
        try:
            result = fn()
            self._observe_command(cmd, args, result)
            return result
        finally:
            self._write_lock.release()

    def _dispatch(self, cmd: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Route command to handler with appropriate locking."""
        try:
            if cmd == "shutdown":
                self.running = False
                return {"ok": True, "result": {"status": "shutting_down"}}

            # Hook commands — always write-locked
            if cmd.startswith("hook_"):
                return self._locked_exec(lambda: self._dispatch_hook(cmd, args), cmd, args)

            # Table-driven dispatch
            entry = COMMAND_TABLE.get(cmd)
            if entry is None:
                return {"ok": False, "error": "Unknown command: {}".format(cmd)}

            if entry.is_write:
                def _write():
                    result = entry.handler(self.brain, args, self.graph_changes)
                    if entry.marks_dirty:
                        self.dirty = True
                    return result
                return self._locked_exec(_write, cmd, args)
            else:
                result = entry.handler(self.brain, args, self.graph_changes)
                self._observe_command(cmd, args, result)
                return result

        except Exception as e:
            tb = traceback.format_exc()
            self._log("Command '{}' failed: {}".format(cmd, tb))
            try:
                self.brain._log_error("daemon_dispatch", str(e),
                                       "cmd={}, args={}".format(cmd, str(args)[:200]))
            except Exception:
                pass
            return {"ok": False, "error": str(e)}

    def _dispatch_hook(self, cmd: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch hook with telemetry. Must be called under _write_lock."""
        import servers.daemon_hooks as _hooks

        entry = self.HOOK_TABLE.get(cmd)
        if not entry:
            return {"error": "Unknown hook: %s" % cmd}

        func_name, marks_dirty = entry
        hook_func = getattr(_hooks, func_name)

        start_t = time.time()
        result = hook_func(self.brain, args, self.graph_changes)
        latency_ms = (time.time() - start_t) * 1000

        if marks_dirty:
            self.dirty = True

        # Measure injection volume
        injection_chars = 0
        if isinstance(result, dict):
            reason = result.get("json", {}).get("reason", "") if "json" in result else ""
            output = result.get("output", "")
            injection_chars = len(reason) + len(output)

        # Log telemetry (best-effort)
        try:
            self.brain.log_debug(
                event_type=cmd, source="hook_telemetry",
                latency_ms=latency_ms,
                metadata=json.dumps({
                    "injection_chars": injection_chars,
                    "decision": result.get("json", {}).get("decision", "")
                    if isinstance(result, dict) and "json" in result else "",
                }))
        except Exception as e:
            self._log("Telemetry write failed for %s: %s" % (cmd, e))

        self._write_status()
        return {"ok": True, "result": result}

    def _send_response(self, client: socket.socket, data: Dict[str, Any]):
        """Send JSON response to client."""
        try:
            response = json.dumps(data, default=str) + "\n"
            client.sendall(response.encode('utf-8'))
        except Exception as e:
            self._log("Send error: {}".format(e))

    def _send_error(self, client: socket.socket, message: str):
        """Send error response."""
        self._send_response(client, {"ok": False, "error": message})

    def _write_status(self):
        """Write brain status JSON for statusline script."""
        try:
            brain = self.brain
            if not brain:
                return

            node_count = brain.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            locked_count = brain.conn.execute("SELECT COUNT(*) FROM nodes WHERE locked = 1").fetchone()[0]
            edge_count = brain.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            tension_count = brain.conn.execute("SELECT COUNT(*) FROM nodes WHERE type = 'tension'").fetchone()[0]

            from servers import embedder
            emb_ready = embedder.is_ready()
            emb_stats = embedder.get_stats() if emb_ready else {}

            last_encode = brain.conn.execute(
                "SELECT created_at FROM nodes ORDER BY created_at DESC LIMIT 1").fetchone()

            status = {
                "nodes": node_count, "edges": edge_count,
                "locked": locked_count, "tensions": tension_count,
                "embedder_ready": emb_ready,
                "model_name": emb_stats.get("model_name", ""),
                "last_encode_at": last_encode[0] if last_encode else None,
                "pid": os.getpid(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            status_path = get_status_path()
            tmp_path = status_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(status, f)
            os.replace(tmp_path, status_path)
        except Exception:
            pass  # Status file is best-effort

    def _autosave_loop(self):
        """Periodically save brain if dirty + run internal health check."""
        while self.running:
            time.sleep(AUTOSAVE_INTERVAL_SECONDS)
            if self.dirty:
                if self._write_lock.acquire(timeout=5.0):
                    try:
                        self.brain.save()
                        self.dirty = False
                        self._log("Autosaved")
                    except Exception as e:
                        self._log("Autosave error: {}".format(e))
                    finally:
                        self._write_lock.release()
            # Internal health check — verify SQLite alive (skip during shutdown)
            if self.running and self.brain:
                try:
                    self.brain.conn.execute("SELECT 1").fetchone()
                except Exception as e:
                    self._log("HEALTH: SQLite check failed: {}".format(e))
            self._write_status()

    def _handle_signal(self, signum, frame):
        self._log("Received signal {}".format(signum))
        self.running = False
        # Close server socket immediately to unblock select() and reject new connections
        try:
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception:
            pass

    def _shutdown(self):
        """Clean shutdown — save brain, close sockets, release all resources.
        Forces exit after 5s if workers are stuck (e.g. embedder CPU loop)."""
        self._log("Shutting down...")
        try:
            if self.brain:
                self.brain.save()
                self.brain.close()
        except Exception as e:
            self._log("Save error during shutdown: {}".format(e))
        self._cleanup()
        # Give workers 5s to finish, then force exit
        import _thread
        def _force_exit():
            time.sleep(5)
            self._log("Workers stuck — forcing exit")
            os._exit(0)
        _thread.start_new_thread(_force_exit, ())
        try:
            self._pool.shutdown(wait=True, cancel_futures=False)
        except TypeError:
            self._pool.shutdown(wait=True)

    def _cleanup(self):
        """Close server socket, observer channel, remove PID and lock files.
        Idempotent — safe to call multiple times (signal + atexit + explicit)."""
        try:
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception:
            pass
        for path in [self.pid_path, get_status_path()]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
        try:
            if hasattr(self, '_lock_fd') and self._lock_fd:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
        except Exception:
            pass

    def _log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        print("[brain-daemon {}] {}".format(ts, message), file=sys.stderr)
