"""
brain — Persistent Daemon

Keeps a Brain instance alive across hook invocations via a Unix domain socket.
Instead of each hook spawning python3 → import brain → load embedder (~1-2s),
hooks become thin clients that send commands and receive results in <10ms.

ARCHITECTURE:
  - daemon listens on a Unix socket (default: /tmp/brain-daemon-{uid}.sock)
  - commands are newline-delimited JSON: {"cmd": "recall", "args": {...}}
  - responses are newline-delimited JSON: {"ok": true, "result": {...}}
  - daemon auto-starts on first hook call if not running
  - daemon auto-exits after idle timeout (default: 30 minutes)
  - PID file at /tmp/brain-daemon-{uid}.pid for lifecycle management

LIFECYCLE:
  - boot-brain.sh starts daemon via subprocess.Popen (detached, CPU-only)
  - hooks send commands via Python Unix socket client
  - daemon saves brain periodically and on shutdown
  - session-end.sh sends "shutdown" command

PROTOCOL:
  Client sends: {"cmd": "...", "args": {...}}\n
  Server sends: {"ok": true, "result": {...}}\n

  Commands:
    context_boot {user, project, task}  → boot context
    recall {query, limit}               → recall_with_embeddings
    remember {type, title, content, ...} → remember
    record_message {}                   → record_message + heartbeat check
    heartbeat {}                        → get_encoding_heartbeat
    vocab_check {message}               → vocabulary gap detection
    save {}                             → force save
    ping {}                             → health check
    shutdown {}                         → graceful shutdown
    eval {code}                         → eval arbitrary brain method (escape hatch)
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
from typing import Optional, Dict, Any

# ─── Configuration ───

IDLE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes

def _code_fingerprint() -> str:
    """Return a deterministic fingerprint of the server code files.
    Changes when any .py file in servers/ is modified."""
    import hashlib
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
AUTOSAVE_INTERVAL_SECONDS = 60  # Save every 60 seconds if dirty
SOCKET_BACKLOG = 5
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max message

def get_socket_path() -> str:
    """Get the daemon socket path, unique per user."""
    uid = os.getuid()
    return os.path.join("/tmp", "brain-daemon-{}.sock".format(uid))

def get_pid_path() -> str:
    """Get the daemon PID file path."""
    uid = os.getuid()
    return os.path.join("/tmp", "brain-daemon-{}.pid".format(uid))

def get_lock_path() -> str:
    """Get the daemon lock file path for startup serialization."""
    uid = os.getuid()
    return os.path.join("/tmp", "brain-daemon-{}.lock".format(uid))


class BrainDaemon:
    """Persistent Brain daemon that listens on a Unix socket."""

    def __init__(self, db_path: str, socket_path: Optional[str] = None):
        self.db_path = db_path
        self.socket_path = socket_path or get_socket_path()
        self.pid_path = get_pid_path()
        self.brain = None
        self.server_socket = None
        self.running = False
        self.last_activity = time.time()
        self.dirty = False  # Track if brain has unsaved changes
        self.graph_changes = []  # In-memory graph mutation log, drained by hook_recall
        self._lock = threading.Lock()

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

    def _dispatch_hook(self, cmd, args):
        """Dispatch a hook command with telemetry wrapping.

        Measures latency and injection volume for every hook fire,
        logs to debug_log for per-hook activity tracking.
        """
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

        # Log telemetry (non-blocking, best-effort)
        try:
            self.brain.log_debug(
                event_type=cmd,
                source="hook_telemetry",
                latency_ms=latency_ms,
                metadata=json.dumps({
                    "injection_chars": injection_chars,
                    "decision": result.get("json", {}).get("decision", "") if isinstance(result, dict) and "json" in result else "",
                }),
            )
        except Exception as e:
            self._log("Telemetry write failed for %s: %s" % (cmd, e))

        return result

    def start(self):
        """Start the daemon — load brain, bind socket, serve.

        Uses fcntl.flock() on a lockfile to prevent duplicate daemons.
        If another daemon is starting concurrently, we exit immediately.
        """
        # Acquire exclusive lock — prevents duplicate daemon startup race
        lock_path = get_lock_path()
        self._lock_fd = open(lock_path, 'w')
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            # Another daemon is starting — exit cleanly
            self._lock_fd.close()
            self._log("Another daemon is starting — exiting duplicate")
            return

        # Write PID file (we hold the lock, so this is safe)
        with open(self.pid_path, 'w') as f:
            f.write(str(os.getpid()))

        # Clean up stale socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Load brain (this is the expensive part — done once)
        self._load_brain()

        # Bind socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(SOCKET_BACKLOG)
        self.server_socket.setblocking(False)

        # Set permissions — owner only
        os.chmod(self.socket_path, 0o600)

        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGHUP, self._handle_signal)

        # Cleanup on exit
        atexit.register(self._cleanup)

        self.running = True
        self._log("Daemon started. PID={}, socket={}".format(os.getpid(), self.socket_path))

        # Start autosave thread
        autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
        autosave_thread.start()

        # Main event loop
        self._serve()

    def _load_brain(self):
        """Load the Brain instance + embedder — the expensive operation we do once."""
        # Disable PyTorch MPS (Metal Performance Shaders) before any import.
        # PyTorch auto-dispatches attention ops to MPS on Apple Silicon,
        # but MPS requires GPU context that background daemons don't have,
        # causing SIGABRT in MTLComputePipelineStateCache.
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
        """Main event loop — accept connections, handle commands."""
        while self.running:
            # Check idle timeout
            idle = time.time() - self.last_activity
            if idle > IDLE_TIMEOUT_SECONDS:
                self._log("Idle timeout ({}s). Shutting down.".format(int(idle)))
                break

            # Wait for connections (timeout every 5s to check idle)
            try:
                readable, _, _ = select.select([self.server_socket], [], [], 5.0)
            except (select.error, OSError):
                break

            for sock in readable:
                try:
                    client, _ = sock.accept()
                    client.settimeout(30.0)  # 30s timeout per request
                    self._handle_client(client)
                except Exception as e:
                    self._log("Accept error: {}".format(e))

        self._shutdown()

    def _handle_client(self, client: socket.socket):
        """Handle a single client connection."""
        try:
            # Read until newline
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

            # Parse command
            try:
                msg = json.loads(data.decode('utf-8').strip())
            except json.JSONDecodeError as e:
                self._send_error(client, "Invalid JSON: {}".format(e))
                return

            cmd = msg.get("cmd", "")
            args = msg.get("args", {})

            # Dispatch
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

    def _dispatch(self, cmd: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a command to the appropriate handler."""
        with self._lock:
            try:
                if cmd == "ping":
                    return {"ok": True, "result": {"status": "alive", "pid": os.getpid(), "code_fingerprint": _CODE_FINGERPRINT}}

                elif cmd == "shutdown":
                    self.running = False
                    return {"ok": True, "result": {"status": "shutting_down"}}

                elif cmd == "save":
                    self.brain.save()
                    self.dirty = False
                    return {"ok": True, "result": {"status": "saved"}}

                elif cmd == "context_boot":
                    text = self.brain.format_boot_context(
                        user=args.get("user", "User"),
                        project=args.get("project", "default"),
                        db_dir=args.get("db_dir", "")
                    )
                    return {"ok": True, "result": text}

                elif cmd == "recall":
                    try:
                        result = self.brain.recall_with_embeddings(
                            query=args.get("query", ""),
                            limit=args.get("limit", 8)
                        )
                    except Exception:
                        result = self.brain.recall(
                            query=args.get("query", ""),
                            limit=args.get("limit", 8)
                        )
                    return {"ok": True, "result": result}

                elif cmd == "record_message":
                    self.brain.record_message()
                    nudge = self.brain.get_encoding_heartbeat()
                    self.dirty = True
                    return {"ok": True, "result": {"nudge": nudge}}

                elif cmd == "heartbeat":
                    nudge = self.brain.get_encoding_heartbeat(
                        nudge_threshold=args.get("threshold", 8)
                    )
                    return {"ok": True, "result": {"nudge": nudge}}

                elif cmd == "vocab_check":
                    message = args.get("message", "")
                    # Resolve vocabulary terms
                    import re
                    candidates = set()
                    candidates.update(
                        t.strip().lower() for t in
                        re.findall(r"\bthe\s+([\w][\w\s-]{2,25})\b", message, re.IGNORECASE)
                    )
                    candidates.update(
                        t.strip().lower() for t in
                        re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", message)
                        if len(t) > 4
                    )
                    expansions = []
                    for term in candidates:
                        resolved = self.brain.resolve_vocabulary(term)
                        if resolved:
                            content = resolved.get("content", "")
                            if content:
                                expansions.append(content)
                    return {"ok": True, "result": {"expansions": expansions}}

                elif cmd == "reset_session":
                    self.brain.reset_session_activity()
                    self.dirty = True
                    return {"ok": True, "result": {"status": "reset"}}

                elif cmd == "validate_config":
                    warnings = self.brain.validate_config()
                    return {"ok": True, "result": {"warnings": warnings}}

                elif cmd == "health_check":
                    health = self.brain.health_check(
                        session_id=args.get("session_id", "daemon"),
                        auto_fix=args.get("auto_fix", True)
                    )
                    return {"ok": True, "result": health}

                elif cmd == "consciousness":
                    signals = self.brain.get_consciousness_signals()
                    return {"ok": True, "result": signals}

                elif cmd == "urgent_signals":
                    urgent = self.brain.get_urgent_signals()
                    return {"ok": True, "result": urgent}

                elif cmd == "engineering_context":
                    ctx = self.brain.get_engineering_context(
                        project=args.get("project", "default")
                    )
                    return {"ok": True, "result": ctx}

                elif cmd == "correction_patterns":
                    patterns = self.brain.get_correction_patterns(
                        limit=args.get("limit", 5)
                    )
                    return {"ok": True, "result": patterns}

                elif cmd == "last_synthesis":
                    synthesis = self.brain.get_last_synthesis()
                    return {"ok": True, "result": synthesis}

                elif cmd == "scan_host":
                    result = self.brain.scan_host_environment()
                    return {"ok": True, "result": result}

                elif cmd == "dreams":
                    dreams = self.brain.get_surfaceable_dreams(
                        limit=args.get("limit", 2)
                    )
                    return {"ok": True, "result": dreams}

                elif cmd == "self_reflection":
                    self.brain.auto_generate_self_reflection()
                    self.dirty = True
                    return {"ok": True, "result": {"status": "generated"}}

                elif cmd == "staged":
                    staged = self.brain.list_staged(
                        status=args.get("status", "pending"),
                        limit=args.get("limit", 10)
                    )
                    return {"ok": True, "result": staged}

                elif cmd == "promote_staged":
                    self.brain.auto_promote_staged(
                        revisit_threshold=args.get("threshold", 3)
                    )
                    self.dirty = True
                    return {"ok": True, "result": {"status": "promoted"}}

                elif cmd == "suggest_metrics":
                    metrics = self.brain.get_suggest_metrics(
                        period_days=args.get("period_days", 7)
                    )
                    return {"ok": True, "result": metrics}

                elif cmd == "procedure_trigger":
                    procs = self.brain.procedure_trigger(
                        trigger=args.get("trigger", ""),
                        context=args.get("context", {})
                    )
                    return {"ok": True, "result": procs}

                elif cmd == "get_config":
                    key = args.get("key", "")
                    default = args.get("default", "")
                    val = self.brain.get_config(key, default)
                    return {"ok": True, "result": {"value": val}}

                elif cmd == "set_config":
                    self.brain.set_config(args.get("key", ""), args.get("value", ""))
                    self.dirty = True
                    return {"ok": True, "result": {"status": "set"}}

                elif cmd == "get_debug_status":
                    on = self.brain.get_debug_status()
                    return {"ok": True, "result": {"debug": on}}

                elif cmd == "log_debug":
                    self.brain.log_debug(
                        args.get("event", ""),
                        args.get("source", ""),
                        metadata=args.get("metadata")
                    )
                    return {"ok": True, "result": {"status": "logged"}}

                elif cmd == "pre_edit":
                    data = self.brain.pre_edit(
                        file=args.get("file", ""),
                        tool_name=args.get("tool_name", "Edit")
                    )
                    # Also get change impacts
                    change_impacts = []
                    try:
                        change_impacts = self.brain.get_change_impact(args.get("file", ""))
                    except Exception as e:
                        self._log("get_change_impact error: %s" % e)
                    data["change_impacts"] = change_impacts
                    return {"ok": True, "result": data}

                elif cmd == "consolidate":
                    result = self.brain.consolidate()
                    self.dirty = True
                    return {"ok": True, "result": result}

                elif cmd == "dream":
                    result = self.brain.dream()
                    self.dirty = True
                    return {"ok": True, "result": result}

                elif cmd == "auto_heal":
                    result = self.brain.auto_heal()
                    self.dirty = True
                    return {"ok": True, "result": result}

                elif cmd == "auto_tune":
                    result = self.brain.auto_tune()
                    self.dirty = True
                    return {"ok": True, "result": result}

                elif cmd == "prompt_reflection":
                    result = self.brain.prompt_reflection()
                    return {"ok": True, "result": result}

                elif cmd == "backfill_summaries":
                    result = self.brain.backfill_summaries(
                        batch_size=args.get("batch_size", 50)
                    )
                    self.dirty = True
                    return {"ok": True, "result": result}

                elif cmd == "synthesize_session":
                    result = self.brain.synthesize_session()
                    self.dirty = True
                    return {"ok": True, "result": result}

                elif cmd == "get_active_evolutions":
                    types = args.get("types")
                    result = self.brain.get_active_evolutions(types)
                    return {"ok": True, "result": result}

                elif cmd == "assess_developmental_stage":
                    result = self.brain.assess_developmental_stage()
                    return {"ok": True, "result": result}

                elif cmd == "instinct_check":
                    message = args.get("message", "")
                    nudge = self.brain.get_instinct_check(message)
                    return {"ok": True, "result": {"nudge": nudge}}

                elif cmd == "eval":
                    # Escape hatch: eval arbitrary expression on brain
                    # Only for development/debugging — not for production hooks
                    code = args.get("code", "")
                    if not code:
                        return {"ok": False, "error": "No code provided"}
                    # Limited eval — brain is available as 'brain'
                    local_vars = {"brain": self.brain, "json": json}
                    result = eval(code, {"__builtins__": {}}, local_vars)
                    # Try to make result JSON-serializable
                    try:
                        json.dumps(result)
                    except (TypeError, ValueError):
                        result = str(result)
                    return {"ok": True, "result": result}

                # ── Hook commands (centralized in daemon_hooks.py) ──
                elif cmd.startswith("hook_"):
                    result = self._dispatch_hook(cmd, args)
                    return {"ok": True, "result": result}

                # ── Track graph mutations from standard commands ──
                elif cmd == "remember":
                    result = self.brain.remember(
                        type=args.get("type", "context"),
                        title=args.get("title", ""),
                        content=args.get("content", ""),
                        locked=args.get("locked", False),
                        confidence=args.get("confidence", 0.5),
                        emotion=args.get("emotion"),
                        keywords=args.get("keywords"),
                        project=args.get("project")
                    )
                    self.dirty = True
                    node_id = result.get("id", "?")[:8] if isinstance(result, dict) else "?"
                    self.graph_changes.append(
                        "REMEMBER: [%s] %s (%s...)" % (
                            args.get("type", "?"), args.get("title", "")[:50], node_id))
                    return {"ok": True, "result": result}

                elif cmd == "connect":
                    result = self.brain.connect(
                        source_id=args.get("source_id", ""),
                        target_id=args.get("target_id", ""),
                        relation=args.get("relation", "related_to"),
                        weight=args.get("weight", 0.5)
                    )
                    self.dirty = True
                    self.graph_changes.append(
                        "CONNECT: %s -[%s]-> %s" % (
                            args.get("source_id", "?")[:8],
                            args.get("relation", "related_to"),
                            args.get("target_id", "?")[:8]))
                    return {"ok": True, "result": result}

                else:
                    return {"ok": False, "error": "Unknown command: {}".format(cmd)}

            except Exception as e:
                tb = traceback.format_exc()
                self._log("Command '{}' failed: {}".format(cmd, tb))
                try:
                    self.brain._log_error("daemon_dispatch", str(e), "cmd={}, args={}".format(cmd, str(args)[:200]))
                except Exception:
                    pass
                return {"ok": False, "error": str(e)}

    def _send_response(self, client: socket.socket, data: Dict[str, Any]):
        """Send a JSON response to the client."""
        try:
            response = json.dumps(data, default=str) + "\n"
            client.sendall(response.encode('utf-8'))
        except Exception as e:
            self._log("Send error: {}".format(e))

    def _send_error(self, client: socket.socket, message: str):
        """Send an error response."""
        self._send_response(client, {"ok": False, "error": message})

    def _autosave_loop(self):
        """Periodically save brain if dirty."""
        while self.running:
            time.sleep(AUTOSAVE_INTERVAL_SECONDS)
            if self.dirty:
                with self._lock:
                    try:
                        self.brain.save()
                        self.dirty = False
                        self._log("Autosaved")
                    except Exception as e:
                        self._log("Autosave error: {}".format(e))

    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        self._log("Received signal {}".format(signum))
        self.running = False

    def _shutdown(self):
        """Clean shutdown — save brain, close socket, remove files."""
        self._log("Shutting down...")
        try:
            if self.brain:
                self.brain.save()
                self.brain.close()
        except Exception as e:
            self._log("Save error during shutdown: {}".format(e))

        self._cleanup()

    def _cleanup(self):
        """Remove socket, PID, and lock files."""
        try:
            if self.server_socket:
                self.server_socket.close()
        except Exception:
            pass
        for path in [self.socket_path, self.pid_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
        # Release startup lock
        try:
            if hasattr(self, '_lock_fd') and self._lock_fd:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
        except Exception:
            pass

    def _log(self, message: str):
        """Log to stderr (daemon output)."""
        ts = time.strftime("%H:%M:%S")
        print("[brain-daemon {}] {}".format(ts, message), file=sys.stderr)


# ─── Client Functions (used by hooks) ───

def send_command(cmd: str, args: Optional[Dict[str, Any]] = None,
                 timeout: float = 10.0) -> Dict[str, Any]:
    """Send a command to the running daemon. Returns response dict."""
    socket_path = get_socket_path()

    if not os.path.exists(socket_path):
        return {"ok": False, "error": "Daemon not running"}

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(socket_path)
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
    if is_daemon_running():
        # Daemon process exists — wait for socket to be ready
        for attempt in range(25):  # 5 seconds total
            resp = send_command("ping", timeout=2.0)
            if resp.get("ok"):
                # Check if code has been updated since daemon loaded
                daemon_fp = resp.get("result", {}).get("code_fingerprint", "")
                current_fp = _code_fingerprint()
                if current_fp != "unknown" and daemon_fp != current_fp:
                    sys.stderr.write("[brain-daemon] Code changed (daemon={}, current={}) — restarting\n".format(
                        daemon_fp[:12] or "none", current_fp[:12]))
                    _kill_daemon()
                    break  # Fall through to start below
                return True
            time.sleep(0.2)
        else:
            # Still not responding after 5s — truly zombie, kill and restart
            sys.stderr.write("[brain-daemon] Killing zombie daemon (PID alive but unresponsive for 5s)\n")
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
             'from servers.daemon import BrainDaemon; '
             'd = BrainDaemon(%r); d.start()' % (parent_dir, db_path)],
            stdin=devnull,
            stdout=log_fd,
            stderr=log_fd,
            start_new_session=True,
            env={**os.environ, "VECLIB_MAXIMUM_THREADS": "1", "ONNX_CPU_ONLY": "1"},
        )

    # Wait for daemon to become ready
    for attempt in range(30):  # 6 seconds total
        resp = send_command("ping", timeout=2.0)
        if resp.get("ok"):
            return True
        time.sleep(0.2)

    sys.stderr.write("[brain-daemon] Daemon failed to start within 6s\n")
    return False


def _kill_daemon():
    """Kill a running daemon."""
    pid_path = get_pid_path()
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        sys.stderr.write("[brain-daemon] Killing daemon PID={}\n".format(pid))
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
    except Exception as e:
        sys.stderr.write("[brain-daemon] Kill failed: {}\n".format(e))
    # Clean up files
    for path in [pid_path, get_socket_path()]:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass


def stop_daemon():
    """Gracefully stop the daemon."""
    resp = send_command("shutdown", timeout=5.0)
    if not resp.get("ok"):
        _kill_daemon()


# ─── CLI Entry Point ───

if __name__ == "__main__":
    import argparse

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
