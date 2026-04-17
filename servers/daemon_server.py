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
    IDLE_TIMEOUT_SECONDS, S2_IDLE_THRESHOLD, S2_MIN_INTERVAL, AUTOSAVE_INTERVAL_SECONDS,
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
        # Clear pycache so launchd restarts always use latest code
        import shutil
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__pycache__')
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

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
        threads = threading.enumerate()
        self._log("Daemon started. PID={}, addr={}:{}, workers={}, restarts={}, threads={}({})".format(
            os.getpid(), self.daemon_addr[0], self.daemon_addr[1],
            THREAD_POOL_SIZE, self._restart_count, len(threads),
            ", ".join(t.name for t in threads if t.name != "MainThread")))

        # Start autosave thread
        autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True, name="autosave")
        autosave_thread.start()

        self._serve()

    def _bind_socket(self):
        """Bind TCP socket with retry for TIME_WAIT recovery."""
        for attempt in range(self.SOCKET_BIND_RETRIES):
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # DO NOT set SO_REUSEPORT — it allows duplicate daemons to bind the same port
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
        self._last_s2_run = 0  # epoch — allows first run after S2_MIN_INTERVAL
        self._s2_running = False
        while self.running:
            # Check idle timeout every ~10 iterations (not every loop)
            now = time.time()
            if now - last_idle_check > 5.0:
                last_idle_check = now
                idle = now - self.last_activity

                # S2 maintenance: run when idle 5min+ AND at least 1h since last run
                if (idle > S2_IDLE_THRESHOLD
                        and now - self._last_s2_run > S2_MIN_INTERVAL
                        and not self._s2_running):
                    self._last_s2_run = now
                    self._s2_running = True
                    self._pool.submit(self._run_idle_maintenance)

                # Shutdown after long idle
                if IDLE_TIMEOUT_SECONDS > 0 and idle > IDLE_TIMEOUT_SECONDS:
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
        t = threading.current_thread()
        t_name = t.name
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

            # Direct dispatch — no watchdog thread.
            # The old pattern spawned a thread per request and joined with 20s timeout.
            # If it timed out, the thread kept running forever → thread leak → CPU spiral.
            # Now: dispatch runs inline in the pool worker. Client has its own timeout (30s).
            # Pool worker finishes and returns to pool. No orphans.
            t.name = "pool-worker:%s" % cmd
            t0 = time.time()
            result = self._dispatch(cmd, args)
            elapsed_ms = int((time.time() - t0) * 1000)

            if elapsed_ms > 2000:
                self._log("[thread:%s] %s took %dms" % (t_name, cmd, elapsed_ms))

            self._send_response(client, result)

        except Exception as e:
            self._log("[thread:%s] EXCEPTION in %s: %s" % (
                t_name, msg.get("cmd", "?") if 'msg' in dir() else "?", e))
            try:
                self._send_error(client, "Internal error: {}".format(e))
            except Exception:
                pass
        finally:
            t.name = t_name
            try:
                client.close()
            except Exception:
                pass

    def _observe_command(self, cmd, args, result=None):
        """Emit command to dashboard observer channel. DEPRECATED — brain_dashboard.py removed."""
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
                    import shutil, subprocess
                    servers_dir = os.path.dirname(os.path.abspath(__file__))
                    project_dir = os.path.dirname(servers_dir)
                    # Clear bytecode cache
                    for cache_dir in [os.path.join(servers_dir, '__pycache__'),
                                      os.path.join(project_dir, 'hooks', 'scripts', '__pycache__')]:
                        if os.path.isdir(cache_dir):
                            shutil.rmtree(cache_dir, ignore_errors=True)
                    db_dir = os.environ.get('BRAIN_DB_DIR', os.path.dirname(self.db_path))
                    startup = (
                        "import sys, os; "
                        "sys.path.insert(0, %r); "
                        "os.environ['BRAIN_DB_DIR'] = %r; "
                        "from servers.daemon_server import BrainDaemon; "
                        "d = BrainDaemon(%r); d.start()"
                        % (project_dir, db_dir, self.db_path)
                    )
                    self._log("Spawning new daemon: %s -c ..." % sys.executable)
                    # Spawn new process BEFORE cleanup — ensures new daemon starts
                    subprocess.Popen([sys.executable, '-c', startup],
                                     start_new_session=True,
                                     stdout=open('/dev/null', 'w'),
                                     stderr=open('/dev/null', 'w'))
                    self._log("New daemon spawned. Shutting down old.")
                    self._cleanup()
                    os._exit(0)

                import threading as _t
                _t.Thread(target=_do_restart, daemon=False).start()
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
        """Periodically save brain if dirty + run internal health check + thread monitor."""
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
            # Thread inventory — detect runaway threads (Python level)
            threads = threading.enumerate()
            alive = [t for t in threads if t.is_alive()]
            non_daemon = [t for t in alive if not t.daemon and t.name != "MainThread"]
            if len(alive) > 15 or non_daemon:
                self._log("THREADS: %d alive (%d non-daemon): %s" % (
                    len(alive), len(non_daemon),
                    ", ".join("%s(%s)" % (t.name, "daemon" if t.daemon else "NON-DAEMON") for t in alive
                             if t.name != "MainThread")))
            # Native thread CPU monitor — catches onnxruntime/tokenizers spin
            try:
                import subprocess
                r = subprocess.run(
                    ['ps', '-M', '-p', str(os.getpid()), '-o', 'pcpu='],
                    capture_output=True, text=True, timeout=3)
                hot = [float(x) for x in r.stdout.strip().split('\n') if x.strip() and float(x.strip()) > 50]
                if hot:
                    self._log("CPU SPIRAL DETECTED: %d native threads at %s%%" % (
                        len(hot), "+".join("%.0f" % h for h in hot)))
            except Exception:
                pass
            self._write_status()

    def _run_idle_maintenance(self):
        """Run S2 + maintenance when daemon is idle. Runs in thread pool.

        Same work as hook_idle_maintenance but triggered internally —
        no dependency on Claude Code idle_prompt notifications.
        """
        self._log("Idle maintenance triggered (%.0fs idle)" % (time.time() - self.last_activity))
        try:
            from .daemon_hooks import hook_idle_maintenance
            graph_changes = []
            hook_idle_maintenance(self.brain, {"session_id": "daemon-idle"}, graph_changes)
            if self.brain:
                self.brain.save()
            self._log("Idle maintenance complete: %s" % "; ".join(graph_changes) if graph_changes else "Idle maintenance complete (no changes)")
        except Exception as e:
            self._log("Idle maintenance error: {}".format(e))
        finally:
            self._s2_running = False

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

    def _log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        print("[brain-daemon {}] {}".format(ts, message), file=sys.stderr)

