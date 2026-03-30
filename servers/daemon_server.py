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

    MAX_SUPERVISOR_RESTARTS = 5      # Max restarts before giving up
    SUPERVISOR_RESTART_COOLDOWN = 2   # Seconds between restart attempts
    SOCKET_BIND_RETRIES = 10          # Retries for port binding after crash
    SOCKET_BIND_RETRY_DELAY = 1.0     # Seconds between bind retries

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
        self._embed_lock = threading.Lock()  # Serialize embedder calls (prevent CPU explosion)
        self._pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
        self._restart_count = 0

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
        """Supervisor loop — start daemon and restart on internal crashes.

        Handles: brain errors, socket errors, thread pool crashes.
        Does NOT handle: SIGKILL, OOM (external watchdog needed — MCP plugin).
        Gives up after MAX_SUPERVISOR_RESTARTS consecutive crashes.
        """
        # Acquire exclusive lock (one daemon per user)
        lock_path = get_lock_path()
        self._lock_fd = open(lock_path, 'w')
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            # Lock held — check if the holder is actually alive
            pid_path = get_pid_path()
            stale = False
            if os.path.exists(pid_path):
                try:
                    old_pid = int(open(pid_path).read().strip())
                    os.kill(old_pid, 0)  # signal 0 = check if alive
                    # Process exists — real duplicate
                    self._lock_fd.close()
                    self._log("Another daemon running (PID {}). Exiting duplicate.".format(old_pid))
                    return
                except (ProcessLookupError, ValueError):
                    stale = True
                    self._log("Stale PID file (PID {} dead). Cleaning up.".format(old_pid if 'old_pid' in dir() else '?'))
                except PermissionError:
                    # Process exists but we can't signal it — assume alive
                    self._lock_fd.close()
                    self._log("Another daemon running (PID {}, permission denied). Exiting.".format(old_pid))
                    return

            if stale or not os.path.exists(pid_path):
                # Stale lock — clean up and retry
                try:
                    os.unlink(lock_path)
                    os.unlink(pid_path)
                except OSError:
                    pass
                self._lock_fd = open(lock_path, 'w')
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._log("Acquired lock after cleaning stale files.")
                except (IOError, OSError):
                    self._lock_fd.close()
                    self._log("Cannot acquire lock even after cleanup. Exiting.")
                    return

        # Write PID, register cleanup, install signal handlers (once)
        with open(self.pid_path, 'w') as f:
            f.write(str(os.getpid()))
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGHUP, self._handle_signal)
        atexit.register(self._cleanup)

        # Clean up stale Unix socket (migration from old protocol)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # ── Supervisor loop ──
        while self._restart_count <= self.MAX_SUPERVISOR_RESTARTS:
            try:
                self._run()
                # _run() returns normally on clean shutdown (signal or idle timeout)
                break
            except Exception as e:
                self._restart_count += 1
                self._log_crash(e)

                if self._restart_count > self.MAX_SUPERVISOR_RESTARTS:
                    self._log("FATAL: %d consecutive crashes. Giving up." % self._restart_count)
                    break

                self._log("SUPERVISOR: Restart %d/%d in %ds..." % (
                    self._restart_count, self.MAX_SUPERVISOR_RESTARTS,
                    self.SUPERVISOR_RESTART_COOLDOWN))

                # Clean up before restart
                self._close_socket()
                time.sleep(self.SUPERVISOR_RESTART_COOLDOWN)

        self._shutdown()

    def _run(self):
        """Single daemon lifecycle — load brain, bind socket, serve until stopped.

        Raises on fatal errors so the supervisor can restart.
        Normal shutdown (signal/idle) returns cleanly.
        """
        # Load brain if not loaded (first run or after crash that corrupted it)
        if not self.brain:
            self._load_brain()

        # Bind socket with retry (handles TIME_WAIT after crash)
        self._bind_socket()

        self.running = True
        self._restart_count = 0  # Reset on successful start
        self._log("Daemon started. PID={}, addr={}:{}, workers={}, restarts={}".format(
            os.getpid(), self.daemon_addr[0], self.daemon_addr[1],
            THREAD_POOL_SIZE, self._restart_count))

        # Start autosave thread
        autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
        autosave_thread.start()

        # HTTP MCP server disabled — causing CPU spirals when MCP client retries.
        # The encoding hook endpoint still works via the command hook path.
        # TODO: Fix HTTP MCP protocol compliance before re-enabling.
        # mcp_http_thread = threading.Thread(target=self._start_mcp_http, daemon=True)
        # mcp_http_thread.start()

        self._serve()

    def _bind_socket(self):
        """Bind TCP socket with retry for TIME_WAIT recovery."""
        for attempt in range(self.SOCKET_BIND_RETRIES):
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except (AttributeError, OSError):
                    pass
                self.server_socket.bind(self.daemon_addr)
                self.server_socket.listen(SOCKET_BACKLOG)
                self.server_socket.setblocking(False)
                return  # Success
            except OSError as e:
                self._close_socket()
                if attempt < self.SOCKET_BIND_RETRIES - 1:
                    self._log("BIND: Port %d busy (attempt %d/%d): %s" % (
                        self.daemon_addr[1], attempt + 1, self.SOCKET_BIND_RETRIES, e))
                    time.sleep(self.SOCKET_BIND_RETRY_DELAY)
                else:
                    raise  # Give up — supervisor will handle

    def _close_socket(self):
        """Close the server socket safely."""
        try:
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception:
            pass

    def _log_crash(self, error: Exception):
        """Log crash details to daemon.log and brain error log."""
        tb = traceback.format_exc()
        self._log("CRASH: %s\n%s" % (error, tb))
        # Also log to brain's error table if brain is alive
        try:
            if self.brain:
                self.brain._log_error('daemon_crash', error,
                                       'restart_count=%d' % self._restart_count)
        except Exception:
            pass

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
                if IDLE_TIMEOUT_SECONDS > 0:
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
            # Update activity immediately — even if parsing fails, someone is talking to us
            self.last_activity = time.time()
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

            if not isinstance(msg, dict):
                self._send_error(client, "Message must be a JSON object, got: {}".format(type(msg).__name__))
                return

            cmd = msg.get("cmd")
            if cmd is None:
                # Common mistake: using "command" instead of "cmd"
                alt_cmd = msg.get("command")
                if alt_cmd:
                    self._send_error(client, "Wrong key: use 'cmd' not 'command'. Got: {}".format(alt_cmd))
                else:
                    self._send_error(client, "Missing 'cmd' field. Message keys: {}".format(list(msg.keys())))
                return

            if not isinstance(cmd, str):
                self._send_error(client, "Field 'cmd' must be a string, got: {} ({})".format(type(cmd).__name__, str(cmd)[:100]))
                return

            args = msg.get("args", {})

            self.last_activity = time.time()

            # Watchdog: kill hung requests after 20s
            import threading as _threading
            dispatch_result = [None]
            dispatch_error = [None]

            def _run_dispatch():
                try:
                    dispatch_result[0] = self._dispatch(cmd, args)
                except Exception as _e:
                    dispatch_error[0] = _e

            worker = _threading.Thread(target=_run_dispatch, daemon=True)
            worker.start()
            worker.join(timeout=20.0)

            if worker.is_alive():
                # Request hung — log and return timeout error
                self.brain._log_error('daemon_watchdog',
                    Exception('Request timed out after 20s: %s' % cmd),
                    'cmd=%s' % cmd)
                self._send_error(client, "timeout: %s took >20s" % cmd)
                return
            elif dispatch_error[0]:
                raise dispatch_error[0]
            else:
                result = dispatch_result[0]

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

            if cmd == "restart":
                self._log("Restart requested — scheduling re-exec after response...")
                # No marker needed — daemon_client uses fcntl.flock for singleton.
                # Schedule restart AFTER response is sent to client
                def _do_restart():
                    time.sleep(0.5)  # Let response reach client
                    self._log("Executing restart...")
                    try:
                        if self.brain:
                            self.brain.save()
                    except Exception as e:
                        self._log("Save error during restart: {}".format(e))
                    import shutil
                    servers_dir = os.path.dirname(os.path.abspath(__file__))
                    project_dir = os.path.dirname(servers_dir)
                    cache_dir = os.path.join(servers_dir, '__pycache__')
                    if os.path.isdir(cache_dir):
                        shutil.rmtree(cache_dir, ignore_errors=True)
                    self._cleanup()
                    db_dir = os.environ.get('BRAIN_DB_DIR', os.path.dirname(self.db_path))
                    startup = (
                        "import sys, os; "
                        "sys.path.insert(0, %r); "
                        "os.environ['BRAIN_DB_DIR'] = %r; "
                        "from servers.daemon_server import BrainDaemon; "
                        "d = BrainDaemon(%r); d.start()"
                        % (project_dir, db_dir, self.db_path)
                    )
                    self._log("Re-exec: %s -c ..." % sys.executable)
                    os.execv(sys.executable, [sys.executable, '-c', startup])

                import threading as _t
                _t.Thread(target=_do_restart, daemon=True).start()
                return {"ok": True, "result": {"status": "restarting"}}

            # Hook commands — read hooks run without lock, write hooks serialize
            if cmd.startswith("hook_"):
                # Read-only hooks run without lock — safe to run concurrently
                read_hooks = (
                    "hook_recall",             # cosine scan, candidates file
                    "hook_pre_edit",           # reads rules, returns context
                    "hook_pre_bash_safety",    # reads rules, returns context
                    "hook_post_bash_host_check",  # checks env, returns context
                    "hook_worktree_context",   # returns context
                )
                if cmd in read_hooks:
                    return self._dispatch_hook(cmd, args)
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
                self.brain = None
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
        self._close_socket()
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

    # ── HTTP MCP Server ──
    # Serves MCP JSON-RPC over HTTP on DAEMON_PORT+1.
    # Uses same dispatch as TCP. Same locking. Same brain.
    # Tool schemas imported from brain_mcp.py (single source of truth).

    def _start_mcp_http(self):
        """Start HTTP MCP server on DAEMON_PORT+1 in a daemon thread."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from . import brain_mcp

        daemon_ref = self
        mcp_port = self.daemon_addr[1] + 1

        class MCPHTTPHandler(BaseHTTPRequestHandler):
            """HTTP server handling MCP JSON-RPC + encoding hook.

            Routes:
              POST / or /mcp  → MCP JSON-RPC (tools/list, tools/call, etc.)
              POST /encoding-hook → Stop hook encoding agent (calls Sonnet API)
            """

            def do_POST(self):
                path = self.path.rstrip('/')
                try:
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length)
                except Exception as e:
                    self._respond(400, {"error": str(e)})
                    return

                if path == '/encoding-hook':
                    self._handle_encoding_hook(body)
                else:
                    self._handle_mcp(body)

            def _handle_mcp(self, body):
                """MCP JSON-RPC handler."""
                try:
                    msg = json.loads(body.decode('utf-8'))
                except Exception as e:
                    self._respond(400, {"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}, "id": None})
                    return

                method = msg.get("method", "")
                request_id = msg.get("id")
                params = msg.get("params", {})

                if request_id is None:
                    self._respond(202, "")
                    return

                try:
                    if method == "initialize":
                        resp = brain_mcp.handle_initialize(request_id)
                    elif method == "tools/list":
                        resp = brain_mcp.handle_tools_list(request_id)
                    elif method == "tools/call":
                        resp = self._handle_tools_call(request_id, params)
                    elif method == "ping":
                        resp = brain_mcp.handle_ping(request_id)
                    else:
                        resp = brain_mcp.make_error(request_id, -32601, "Method not found: %s" % method)
                except Exception as e:
                    daemon_ref._log("MCP HTTP error in %s: %s" % (method, e))
                    resp = brain_mcp.make_error(request_id, -32603, "Internal error: %s" % e)

                self._respond(200, resp)

            def _handle_tools_call(self, request_id, params):
                """Route MCP tool call through daemon dispatch — DIRECT, no TCP relay."""
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                daemon_ref.last_activity = time.time()
                result = daemon_ref._dispatch(tool_name, arguments)
                if result.get("ok"):
                    result_text = brain_mcp._format_result(tool_name, result.get("result", {}))
                    return brain_mcp.make_response(request_id, {
                        "content": [{"type": "text", "text": result_text}]
                    })
                else:
                    return brain_mcp.make_response(request_id, {
                        "content": [{"type": "text", "text": "ERROR: %s" % result.get("error", "Unknown")}],
                        "isError": True
                    })

            def _handle_encoding_hook(self, body):
                """Encoding agent — fires on Stop hook via HTTP.

                Increments counter, stores exchange, triggers encoding every 5th stop.
                Encoding runs in a background thread (non-blocking).
                Logic lives in encoding_agent.py — daemon just dispatches.
                """
                daemon_ref.last_activity = time.time()
                brain = daemon_ref.brain
                if not brain:
                    self._respond(200, {"decision": "allow"})
                    return

                try:
                    counter = int(brain.get_config('stop_counter', '0') or '0') + 1
                    brain.set_config('stop_counter', str(counter))

                    # Store exchange
                    try:
                        hook_input = json.loads(body.decode('utf-8'))
                        user_msg = hook_input.get('prompt', '') or hook_input.get('message', '')
                        assistant_msg = (hook_input.get('last_assistant_message', '') or '')[:4000]
                        session_id = brain.get_config('session_id', '')
                        if user_msg or assistant_msg:
                            brain.store_exchange(user_msg, assistant_msg, session_id)
                    except Exception as e:
                        brain._log_error('encoding_hook_store', e, 'store exchange')

                    if counter % 5 != 0:
                        self._respond(200, {"decision": "allow"})
                        return

                    # Fire encoding in background thread
                    from .encoding_agent import run_encoding
                    def _run():
                        try:
                            run_encoding(brain, daemon_ref._dispatch, counter, daemon_ref._log)
                        except Exception as e:
                            brain._log_error('encoding_hook_run', e, 'Encoding agent failed')
                        daemon_ref.dirty = True

                    threading.Thread(target=_run, daemon=True).start()
                    self._respond(200, {"decision": "allow"})

                except Exception as e:
                    brain._log_error('encoding_hook', e, 'Encoding hook')
                    self._respond(200, {"decision": "allow"})

            def _respond(self, code, body):
                self.send_response(code)
                if isinstance(body, str) and not body:
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                else:
                    payload = json.dumps(body, default=str).encode('utf-8')
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def log_message(self, format, *args):
                pass

        def _do_encoding(daemon_ref, brain, counter):
            """Run the encoding agent: gather context, call Sonnet, dispatch tool calls.

            Same pattern as eval/capabilities/base.py but running inside the daemon
            with direct brain access (no TCP relay).
            """
            import anthropic
            from .pipeline_contract import ENCODING_AGENT
            from .brain_voice import BrainVoice

            # Load API key
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            os.environ.setdefault(k.strip(), v.strip())

            try:
                client = anthropic.Anthropic()
            except Exception as e:
                brain._log_error('encoding_agent_api', e, 'Cannot create Anthropic client')
                return

            session_id = brain.get_config('session_id', 'unknown')

            # 1. Gather messages from DB
            messages = []
            try:
                rows = brain.logs_conn.execute(
                    "SELECT role, content, signal_type, timestamp "
                    "FROM message_stream WHERE session_id = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (session_id, ENCODING_AGENT['max_messages'])
                ).fetchall()
                messages = [{"role": r[0], "content": (r[1] or "")[:ENCODING_AGENT['message_content_limit']],
                             "signal": r[2], "timestamp": r[3]}
                            for r in reversed(rows)]
            except Exception as e:
                brain._log_error('encoding_agent_messages', e, 'Failed to fetch messages')

            if not messages:
                daemon_ref._log("Encoding agent: no messages, skipping.")
                return

            # 2. Independent recall from conversation topics
            recall_context = ""
            try:
                user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
                if user_msgs:
                    recall_query = " ".join(msg[:200] for msg in user_msgs[-3:])
                    enc_recall = brain.recall(query=recall_query, limit=ENCODING_AGENT['recall_candidates_limit'])
                    enc_results = enc_recall.get("results", [])
                    if enc_results:
                        lines = []
                        for r in enc_results:
                            c = {"id": r.get("id", ""), "type": r.get("type", ""),
                                 "title": r.get("title", ""), "content": r.get("content", ""),
                                 "confidence": r.get("confidence", 0), "locked": r.get("locked", False),
                                 "revised_at": r.get("revised_at"), "created_at": r.get("created_at"),
                                 "_graph": r.get("_graph", {})}
                            BrainVoice.format_node_deep(c, lines, conn=brain.conn,
                                max_d1=ENCODING_AGENT['max_d1'],
                                max_d2=ENCODING_AGENT['max_d2'],
                                max_d3=ENCODING_AGENT['max_d3'])
                        recall_context = "\n".join(lines)
            except Exception as e:
                brain._log_error('encoding_agent_recall', e, 'Failed independent recall')

            # 3. Build encoding prompt
            msg_text = ""
            for m in messages:
                role = (m.get("role") or "?").upper()
                content = (m.get("content") or "")[:ENCODING_AGENT['message_display_limit']]
                msg_text += "[%s]: %s\n\n" % (role, content)

            # Read encoding instructions
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prompt_path = os.path.join(project_dir, 'hooks', 'prompts', 'encoding-agent.md')
            try:
                with open(prompt_path) as pf:
                    system_prompt = pf.read()
            except Exception:
                system_prompt = "You are the encoding agent. Search before encoding. Revise stale nodes."

            # Append contract field summary
            try:
                from .contract import generate_field_summary
                system_prompt += "\n\n## Available Fields (from contract)\n\n" + generate_field_summary()
            except Exception:
                pass

            previous_state = brain.get_config('encoding_agent_state', '') or 'First run.'

            user_content = "## ENCODING RUN #%d\n\n" % counter
            user_content += "### Previous State\n%s\n\n" % previous_state
            user_content += "### Conversation (last %d exchanges)\n\n%s\n" % (len(messages), msg_text)
            if recall_context:
                user_content += "### Brain Context\n\n%s\n" % recall_context
            else:
                user_content += "### Brain Context\nNo recall data available.\n\n"

            # 4. Build tool schemas (from brain_mcp — single source of truth)
            from . import brain_mcp
            tools = [{"name": t["name"], "description": t["description"],
                      "input_schema": t["inputSchema"]}
                     for t in brain_mcp.TOOLS
                     if t["name"] in {
                         'recall', 'find_node_by_title', 'get_node',
                         'remember', 'revise', 'connect',
                         'record_divergence', 'learn_vocabulary',
                         'remember_lesson', 'remember_mechanism',
                         'remember_mental_model', 'remember_impact',
                         'remember_convention',
                     }]

            # 5. Call Sonnet with tool use
            daemon_ref._log("Encoding agent: calling Sonnet with %d tools, %d chars context..." % (
                len(tools), len(user_content)))
            api_messages = [{"role": "user", "content": user_content}]

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=4096,
                    system=system_prompt, messages=api_messages, tools=tools)

                # Tool use loop (max 8 rounds)
                for round_num in range(8):
                    tool_uses = [b for b in response.content if b.type == "tool_use"]
                    if not tool_uses:
                        break

                    tool_results = []
                    for tu in tool_uses:
                        # Dispatch directly against brain (no TCP)
                        result = daemon_ref._dispatch(tu.name, tu.input)
                        if result.get("ok"):
                            result_text = brain_mcp._format_result(tu.name, result.get("result", {}))
                        else:
                            result_text = "ERROR: %s" % result.get("error", "Unknown")
                        tool_results.append({
                            "type": "tool_result", "tool_use_id": tu.id,
                            "content": result_text
                        })
                        daemon_ref._log("  [%s] %s" % (tu.name,
                            tu.input.get("title", tu.input.get("query", tu.input.get("node_id", "")))[:50]))

                    api_messages.append({"role": "assistant", "content": [
                        {"type": b.type, **({"text": b.text} if b.type == "text" else
                                            {"id": b.id, "name": b.name, "input": b.input})}
                        for b in response.content]})
                    api_messages.append({"role": "user", "content": tool_results})
                    response = client.messages.create(
                        model="claude-sonnet-4-6", max_tokens=4096,
                        system=system_prompt, messages=api_messages, tools=tools)

                # Save final state
                final_text = ""
                for b in response.content:
                    if b.type == "text":
                        final_text += b.text
                if final_text:
                    brain.set_config('encoding_agent_state', final_text[:2000])

                brain.save()
                daemon_ref.dirty = False
                daemon_ref._log("Encoding agent: done. %d rounds." % (round_num + 1))

            except Exception as e:
                brain._log_error('encoding_agent_sonnet', e, 'Sonnet API call failed')
                daemon_ref._log("Encoding agent FAILED: %s" % e)

        # Bind with retry (same pattern as TCP)
        for attempt in range(5):
            try:
                httpd = HTTPServer(('127.0.0.1', mcp_port), MCPHTTPHandler)
                httpd.timeout = 1.0  # 1s poll for shutdown
                self._log("MCP HTTP server listening on 127.0.0.1:%d" % mcp_port)
                break
            except OSError as e:
                if attempt < 4:
                    self._log("MCP HTTP bind failed (attempt %d/5): %s" % (attempt + 1, e))
                    time.sleep(1)
                else:
                    self._log("MCP HTTP server FAILED to start: %s" % e)
                    return

        # Serve until daemon stops
        while self.running:
            try:
                httpd.handle_request()  # 1s timeout from above
            except Exception as e:
                self._log("MCP HTTP error: %s" % e)

        try:
            httpd.server_close()
        except Exception:
            pass
        self._log("MCP HTTP server stopped.")

    def _log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        print("[brain-daemon {}] {}".format(ts, message), file=sys.stderr)
