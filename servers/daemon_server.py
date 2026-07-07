"""
brain — Daemon Server

BrainDaemon class: loads Brain, serves commands over TCP localhost.
Thread pool (5 workers) handles concurrent connections.
Reads run without lock; all writers (daemon dispatch, S2 encoder, embed
queue, autosave) serialize via brain.write_lock — the lock lives on the
brain itself, not the daemon, so any caller with a brain reference takes
the same lock.
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
import errno
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

from .daemon_config import (
    IDLE_TIMEOUT_SECONDS, AUTOSAVE_INTERVAL_SECONDS,
    SOCKET_BACKLOG, MAX_MESSAGE_SIZE, THREAD_POOL_SIZE,
    DAEMON_HOST, DAEMON_PORT,
    _CODE_FINGERPRINT,
    get_daemon_addr, get_socket_path, get_pid_path, get_lock_path, get_status_path,
)
from .daemon_launch import manages, spawn_detached_daemon
from .daemon_dispatch import COMMAND_TABLE, check_unknown_keys
from .dispatch_common import caller_session


class DuplicateDaemonError(Exception):
    """The port is already served by a responsive brain daemon, so this process
    is a duplicate. Signals the supervisor to exit cleanly (defer to the
    incumbent) instead of restarting into the same bind collision — which would
    turn a single duplicate into a perpetual Errno-48 crash storm (2026-06-05).
    """


class BrainDaemon:
    """Persistent Brain daemon that listens on TCP localhost."""

    MAX_SUPERVISOR_RESTARTS = 5      # Max restarts before giving up
    SUPERVISOR_RESTART_COOLDOWN = 2   # Seconds between restart attempts
    HEALTHY_UPTIME_RESET_S = 300      # A crash after this much healthy serving is
                                      # a FRESH incident, not part of a rapid crash-
                                      # loop — reset the streak. Without this the
                                      # count either reset on every bind (an endless
                                      # serve-crash loop never hit MAX) or accrued
                                      # across unrelated crashes hours apart.
    SHUTDOWN_BACKSTOP_S = 15          # Force-exit if teardown wedges. > a typical
                                      # in-flight recall (so it finishes on open
                                      # conns), < launchd's ~20s SIGTERM→SIGKILL
                                      # window (so we exit on our own terms).
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
        # The S2/keepalive gating signals (last_user_activity + encode-runs
        # counter) live on brain.activity (ActivityState) — single source of
        # truth, mutated here via record_* and read by run_maintenance_if_due.
        # self.last_activity (any IPC) stays here: it's a lifecycle concern
        # (idle-timeout shutdown), distinct from the user-activity gate.
        self.dirty = False
        self.graph_changes = []  # In-memory graph mutation log
        # Write serialization lives on the brain (brain.write_lock) — see
        # Brain.__init__. Daemon acquires it via _locked_exec / autosave.
        self._pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
        # S1 Scribe single-flight: at most one encode at a time across all
        # sessions. The daemon owns encode concurrency now that the Scribe is
        # poll-driven (brain.scribe_due decides; this lock serializes; the encode
        # thread releases it). threading.Lock (not RLock) → cross-thread release.
        self._encode_lock = threading.Lock()
        # Per-session Scribe retry cooldown. A failed/skipped encode never resets
        # the cadence, so the session stays "due" — without this the ~5s poll
        # would re-fire it every tick. {session_id: last-attempt epoch}; a
        # successful encode pops its entry. _scribe_failures counts re-fires that
        # never advanced (wedged encode) → loud escalation. GIL makes the dict
        # set/pop atomic; the cooldown is advisory, so the poll-thread rebuild vs
        # encode-thread pop race is benign.
        self._scribe_attempts = {}
        self._scribe_failures = {}
        self._restart_count = 0
        self._run_started_at = 0  # wall-clock when the current _run() began serving
                                  # (set after bind) — the supervisor's healthy-uptime
                                  # streak reset reads it.
        # Only the instance that actually binds the port writes the PID file —
        # so a duplicate that defers (DuplicateDaemonError) never claims it, and
        # _cleanup never unlinks the incumbent's PID file out from under it.
        self._wrote_pid = False

    # Hook dispatch table: hook_name → (is_write, marks_dirty)
    #   is_write     — True takes _write_lock; False runs concurrently
    #   marks_dirty  — True sets self.dirty so autosave persists soon
    # Single source of truth for hook routing — no parallel "read_hooks"
    # list to keep in sync. Function name in daemon_hooks always matches
    # the cmd name; getattr(_hooks, cmd) resolves it.
    HOOK_TABLE = {
        "hook_recall":               (False, True),
        "hook_post_response_track":  (True,  True),
        "hook_idle_maintenance":     (True,  True),
        "hook_pre_edit":             (False, True),
        "hook_pre_bash_safety":      (False, False),
        "hook_session_end":          (True,  True),
        "hook_stop_failure_log":     (True,  True),
        "hook_config_change_host":   (True,  True),
        "hook_post_bash_host_check": (False, True),
        "hook_worktree_context":     (False, True),
        "hook_worktree_cleanup":     (True,  True),
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

        # Acquire exclusive lock (one daemon per user). flock IS the singleton
        # primitive: the kernel auto-releases it when the holder dies, so a
        # LOCK_NB failure ALWAYS means a live holder — there is no "stale flock".
        # Do NOT unlink + recreate the lock on failure: that let the incumbent
        # keep its lock on the old (unlinked) inode while a recreator acquired a
        # SECOND lock on a fresh inode — defeating mutual exclusion and feeding
        # the Errno-48 storm (2026-06-05). A lingering lock file is harmless:
        # flock on the reused inode succeeds once the dead holder's lock is gone.
        lock_path = get_lock_path()
        self._lock_fd = open(lock_path, 'w')
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            self._lock_fd.close()
            pid_hint = ""
            try:
                pid_hint = " (PID %s)" % open(get_pid_path()).read().strip()
            except Exception:
                pass
            self._log("Another daemon holds the singleton lock%s. Exiting duplicate." % pid_hint)
            return

        # Register cleanup + signal handlers (once). The PID file is NOT written
        # here — it's claimed only after _bind_socket() succeeds (see _run), so a
        # duplicate that defers before binding never overwrites or (via _cleanup)
        # unlinks the incumbent's PID file.
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
            except DuplicateDaemonError as e:
                # Another responsive daemon owns the port. Restarting would just
                # hit the same bind collision — exit cleanly and defer to it.
                self._log("DEFER: %s — exiting cleanly (no restart)." % e)
                break
            except Exception as e:
                # LOUD first: the traceback goes to daemon.log (and the brain error
                # table if the brain is up) before any branch decision.
                self._log_crash(e)

                # Phase-scoped, not exception-type-scoped. self.brain is set ONLY
                # after Brain() constructs (see _load_brain). So brain is None ⟺
                # the crash happened before/during the load — a brain-level fault.
                # Warm-retrying that re-runs the SAME deterministic failure
                # ×MAX, each paying the full load cost while holding the flock and
                # serving nothing. Exit for a fresh process (a clean reload)
                # instead of crash-looping in place: on macOS launchd KeepAlive
                # respawns it (throttled); off-launchd the next client
                # ensure_daemon does. Either way, exit beats in-place retry.
                if self.brain is None:
                    self._log("FATAL: crash before the brain loaded — exiting for a "
                              "fresh reload (KeepAlive / next ensure_daemon respawns; "
                              "no in-place retry).")
                    break

                # Brain is up: a transient serve/bind/socket fault, where the warm-
                # brain retry is the real latency win (no reload). Reset the streak
                # if the daemon had been serving healthily — a crash after sustained
                # uptime is a fresh incident, not a continuation of a rapid loop.
                if self._run_started_at and (
                        time.time() - self._run_started_at) > self.HEALTHY_UPTIME_RESET_S:
                    self._restart_count = 0
                self._restart_count += 1

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

        Singleton identity is the flock, acquired in start() BEFORE this runs.
        A serving daemon holds its flock for its whole serving life (released in
        _cleanup, after the socket closes), so while WE hold it no other same-uid
        process can be past the flock-acquire — none is binding or serving our
        port. The only residual port owner is a DIFFERENT uid colliding via
        uid%100, and the EADDRINUSE backstop in _bind_socket defers cleanly on
        that (and on the acquire→bind race window). So there is no reachable
        "responsive daemon already owns the port" state here to pre-check for.
        """
        # Load brain if not loaded (first run or after crash that corrupted it)
        if not self.brain:
            self._load_brain()

        # Bind socket with retry (handles TIME_WAIT after crash)
        self._bind_socket()

        # We own the port now — claim the PID file. Done here (not in start())
        # so a duplicate that deferred before binding never wrote it, and
        # _cleanup only unlinks a PID file this process actually wrote.
        with open(self.pid_path, 'w') as f:
            f.write(str(os.getpid()))
        self._wrote_pid = True

        self.running = True
        self._run_started_at = time.time()  # healthy-uptime clock for the supervisor's
                                            # streak reset (see the except handler).
        threads = threading.enumerate()
        self._log("Daemon started. PID={}, addr={}:{}, workers={}, restarts={}, threads={}({})".format(
            os.getpid(), self.daemon_addr[0], self.daemon_addr[1],
            THREAD_POOL_SIZE, self._restart_count, len(threads),
            ", ".join(t.name for t in threads if t.name != "MainThread")))

        # Start autosave thread
        autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True, name="autosave")
        autosave_thread.start()

        # Memory watchdog — opt-in via config (memory_watchdog.enabled). Off
        # by default. When the daemon leaks (it has — see brain memory
        # `b6e32edd` category), flip the config to start RSS sampling so the
        # next leak shows up in daemon.log without instrumenting allocations.
        # Allocation-level profiling is intentionally NOT in the watchdog —
        # tracemalloc inside the daemon turned recall into a 5-min spin
        # (2026-05-08); use a one-shot diagnostic script or `py-spy` instead.
        # See servers/memory_watchdog.py for config keys.
        try:
            from .memory_watchdog import MemoryWatchdog
            self._memory_watchdog = MemoryWatchdog.maybe_start(
                self.brain, log_fn=lambda m: self._log("[mem] " + m))
        except Exception as _wd_e:
            # Watchdog must never block daemon startup.
            self._log("memory_watchdog start failed: %s" % _wd_e)
            self._memory_watchdog = None

        # DB maintenance — WAL checkpoint every 5 min, PRAGMA optimize
        # every 30 min, plus a daily compressed rolling backup, on
        # brain.db and brain_logs.db. Lives in its own backend-agnostic
        # scheduler (servers/db_maintenance.py) that calls into
        # db_backends.current for SQLite-specific work; if we ever migrate
        # off SQLite, only the backend module changes. Always-on (no
        # config gate) — periodic checkpointing caps WAL growth, which
        # directly reduces writer-slot wait time. Failures caught + logged
        # via brain._log_error; loop never dies.
        #
        # Auto-backups land in a `backups/` subdir beside the DBs (keeps
        # the rolling snapshots out of the cluttered top-level dir) under
        # GFS retention (7 daily + 4 weekly + 3 monthly per DB). These are
        # distinct from the named pre-destructive-op `.bak` files a human
        # makes by hand — this scheduler only manages its own snapshots.
        try:
            from .db_maintenance import DBMaintenance
            backup_dir = os.path.join(
                os.path.dirname(self.brain.db_path), 'backups')
            self._db_maintenance = DBMaintenance(
                log_fn=self._log,
                log_error_fn=getattr(self.brain, '_log_error', None))
            self._db_maintenance.register(
                'brain', self.brain.db_path, backup_dir=backup_dir)
            self._db_maintenance.register(
                'brain_logs', self.brain.logs_db_path, backup_dir=backup_dir)
            self._db_maintenance.start()
        except Exception as _dm_e:
            # Scheduler must never block daemon startup.
            self._log("db_maintenance start failed: %s" % _dm_e)
            self._db_maintenance = None

        # Brain warmup — fault embeddings into mmap + build structural degree
        # cache off the user's critical path. The first recall before this
        # was paying ~15-20s and a ~2 GB RSS spike (observed 2026-05-09 with
        # brain.db at 218 MB, 3 reproductions across daemon restarts). Both
        # bills belong to boot, not the user's first prompt.
        #
        # Background thread: boot-blocking would re-introduce latency at a
        # different lifecycle point. The recall hot path's lazy paths are
        # idempotence guards — if the user prompts before warmup finishes,
        # recall finishes the work itself.
        warmup_thread = threading.Thread(
            target=self._run_warmup, daemon=True, name="warmup")
        warmup_thread.start()

        # Connection keepalive — re-warm the Anthropic httpx pool during idle
        # so the first recall after a quiet period doesn't pay a cold-TLS tax
        # (idle inflates surface_haiku ~6s->~10s; see
        # Brain.warm_anthropic_connection). Always-on thread; the work is both
        # config-gated and idle-gated inside the loop, so flipping
        # surface_keepalive.enabled off makes every tick a no-op — no restart.
        keepalive_thread = threading.Thread(
            target=self._keepalive_loop, daemon=True, name="keepalive")
        keepalive_thread.start()

        self._serve()

    def _run_warmup(self):
        """Background warmup. See Brain.warm_up() for what's covered."""
        try:
            timings = self.brain.warm_up()
            self._log("Warmup done: %s" % timings)
        except Exception as e:
            # Warmup failure must never affect the daemon. Falling through
            # to the recall hot path's lazy guards is acceptable degradation.
            self._log("Warmup failed: %s" % e)

    # Connection keepalive ---------------------------------------------------
    # Config (brain config table, hot-read each tick so edits take effect live
    # without a daemon restart):
    #   surface_keepalive.enabled           bool  default True
    #   surface_keepalive.interval_seconds  int   default 300 (5 min)

    @staticmethod
    def _keepalive_due(idle_s: float, since_last_ping_s: float,
                       interval_s: float) -> bool:
        """Whether to re-warm the Anthropic connection this tick.

        Fire only when the daemon has been idle a full interval AND we haven't
        already warmed within the last interval. Active sessions keep the pool
        hot via real recalls, so nothing fires while prompts are flowing; once
        idle, it re-warms once per interval to hold the connection open. Pure
        function — no clock, no I/O — so the gating stays unit-testable.
        """
        return idle_s >= interval_s and since_last_ping_s >= interval_s

    def _keepalive_loop(self):
        """Keep the Anthropic connection warm during idle (see _keepalive_due
        and Brain.warm_anthropic_connection).

        Gated on last_user_activity (the UserPromptSubmit clock) — NOT
        last_activity — so a keepalive tick never resets the clock S2 idle
        maintenance gates on, and so quiet stretches between prompts (Anchor
        working, or the operator away) correctly count as idle and re-warm.

        Scope: this warms `brain.anthropic_client`, the client the S1 surface
        step reuses. The encoder / S2 units / scouts each build their own
        anthropic.Anthropic() per use and are NOT covered here — by design,
        since the recall timeout this addresses is a surface-path problem.
        """
        CHECK_CADENCE_S = 30.0    # how often to re-check idle; cheap, bounded
        last_ping = time.time()   # avoid double-warming right after boot warmup
        while self.running:
            time.sleep(CHECK_CADENCE_S)
            if not self.running or not self.brain:
                continue
            last_ping = self._keepalive_tick(time.time(), last_ping)

    def _keepalive_tick(self, now: float, last_ping: float) -> float:
        """One keepalive decision, factored out of _keepalive_loop so the
        gating and backoff are unit-testable without driving the thread.
        Returns the (possibly advanced) last_ping.

        Config (hot-read each tick): surface_keepalive.enabled (bool, default
        True), surface_keepalive.interval_seconds (int, default 300). A
        non-numeric interval falls back to the default rather than disabling
        the loop.
        """
        try:
            if not self.brain.get_config('surface_keepalive.enabled', True):
                return last_ping
            interval = self.brain.get_config(
                'surface_keepalive.interval_seconds', 300)
            try:
                interval = float(interval)
            except (TypeError, ValueError):
                # Bad config value -> fall back to the default; never let a
                # typo silently disable the keepalive.
                interval = 300.0
            if interval <= 0:
                return last_ping
            if self._keepalive_due(now - self.brain.activity.last_user_activity,
                                   now - last_ping, interval):
                # Advance last_ping BEFORE warming so a raised API error still
                # backs off to one attempt per interval (not one per tick).
                # warm_anthropic_connection raises on API error and returns
                # False if there's no client yet or a warm is already in
                # flight — all three outcomes back off the same way.
                last_ping = now
                self.brain.warm_anthropic_connection()
            return last_ping
        except Exception as e:
            # A keepalive tick must never crash the daemon.
            try:
                self.brain._log_error(
                    'keepalive_tick', e, 'keepalive tick failed')
            except Exception:
                pass
            return last_ping

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
                # If a healthy daemon already owns the port, we're a duplicate —
                # defer (clean exit) rather than retry into a crash storm. Backstop
                # for the race where an incumbent binds between the _run pre-check
                # and here. (A non-responsive holder — TIME_WAIT or a hung corpse —
                # falls through to the normal retry; recovery handles corpses.)
                if e.errno == errno.EADDRINUSE:
                    from .daemon_client import is_daemon_responsive
                    if is_daemon_responsive(timeout=1.0):
                        raise DuplicateDaemonError(
                            "Port %d already served by a responsive daemon." %
                            self.daemon_addr[1])
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

        # Start the embed queue drain worker. remember/revise/remember_batch
        # enqueue dirty node_ids; this worker embeds them in batches every
        # EMBED_DRAIN_INTERVAL seconds. S2 Heal catches gaps on idle.
        try:
            from servers import embed_queue
            embed_queue.start(self.brain)
            self._log("Embed queue worker started (drain every {}s)".format(
                embed_queue.EMBED_DRAIN_INTERVAL))
        except Exception as e:
            self._log("embed_queue start failed: {}".format(e))

        # Cold-start auto-enqueue: any entity lacking entity_dates rows
        # (no real intervals, no sentinel) gets enqueued for processing
        # by the embed_queue worker. Idempotent: re-runs at every boot
        # find only NEW gaps because the worker writes either real
        # intervals or sentinel rows. After first full pass this is a
        # cheap LEFT-JOIN scan returning empty.
        try:
            self._enqueue_temporal_backfill_gaps()
        except Exception as e:
            self._log("temporal backfill enqueue failed: {}".format(e))

    def _enqueue_temporal_backfill_gaps(self):
        """Find entities without entity_dates rows and enqueue them."""
        from servers import embed_queue
        node_ids = self.brain._entity_dates.node_ids_without_dates()
        edge_ids = self.brain._entity_dates.edge_ids_without_dates()
        for nid in node_ids:
            embed_queue.enqueue(nid)
        for eid in edge_ids:
            embed_queue.enqueue_edge(eid)
        if node_ids or edge_ids:
            self._log("Temporal backfill enqueued {} nodes + {} edges".format(
                len(node_ids), len(edge_ids)))

    def _serve(self):
        """Main event loop — accept connections, dispatch to thread pool."""
        last_idle_check = time.time()
        self._s2_running = False
        self._scribe_poll_running = False
        while self.running:
            # Check idle timeout every ~10 iterations (not every loop)
            now = time.time()
            if now - last_idle_check > 5.0:
                last_idle_check = now
                idle = now - self.last_activity

                # Maintenance decision lives in brain.run_maintenance_if_due().
                # Daemon owns the polling cadence and the concurrency lock;
                # brain owns "is it time?" (idle threshold + min interval +
                # persisted last-run timestamp). This separation means
                # restarting the daemon doesn't forget when maintenance
                # last ran — the brain_meta record survives.
                if not self._s2_running:
                    self._s2_running = True
                    try:
                        self._pool.submit(self._run_idle_maintenance)
                    except Exception:
                        # submit can raise on pool shutdown/saturation; clear the
                        # flag so the next tick retries instead of wedging S2.
                        self._s2_running = False

                # S1 Scribe reactor — poll for a session whose encode is due
                # (mid-session ENCODE_EVERY turns, or the idle tail). Distinct
                # from idle maintenance: the Scribe fires DURING active sessions,
                # gated per-session, not on the global idle clock. The poll task
                # returns fast (the encode runs on its own thread); the flag just
                # prevents poll-task pile-up if the pool is momentarily saturated.
                if not self._scribe_poll_running:
                    self._scribe_poll_running = True
                    try:
                        self._pool.submit(self._run_scribe_poll)
                    except Exception:
                        # submit can raise if the pool is shutting down/saturated;
                        # clear the flag so the next tick retries instead of
                        # wedging the reactor permanently.
                        self._scribe_poll_running = False

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

            # last_activity was already set pre-parse (line above) — that
            # covers IDLE_TIMEOUT_SECONDS shutdown. Only thing to update
            # here is the user-prompt clock that gates S2 maintenance.
            # Tool-use hooks, internal IPC, pings are noise from S2's
            # perspective. See run_maintenance_if_due in brain.py.
            if cmd == "hook_recall":
                self.brain.activity.record_user_activity()

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

    def _locked_exec(self, fn, cmd, args):
        """Acquire brain's write lock with 10s timeout, execute fn.

        The 10s timeout is for client-facing dispatch — a hung write
        returns an error rather than blocking the client. Background
        writers (S2 encoder dispatch, embed_queue) acquire the same
        brain.write_lock without timeout via `with brain.write_lock:`.
        """
        if not self.brain.write_lock.acquire(timeout=10.0):
            self._log("Write lock timeout (10s) for: {}".format(cmd))
            return {"ok": False, "error": "Write lock timeout — another operation is holding the lock"}
        try:
            return fn()
        finally:
            self.brain.write_lock.release()

    def _dispatch(self, cmd: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Route command to handler with appropriate locking."""
        try:
            if cmd == "shutdown":
                self.running = False
                return {"ok": True, "result": {"status": "shutting_down"}}

            if cmd == "restart":
                self._log("Restart requested — scheduling re-exec after response...")
                # Restart runs on its own thread AFTER this response reaches the
                # client. The re-exec logic lives in _perform_restart (a method,
                # not a closure, so it is unit-testable — it calls os._exit).
                import threading as _t
                _t.Thread(target=self._perform_restart, daemon=False).start()
                return {"ok": True, "result": {"status": "restarting"}}

            # Hook commands — HOOK_TABLE is the single source of truth
            # for is_write (lock or concurrent) and marks_dirty.
            if cmd.startswith("hook_"):
                entry = self.HOOK_TABLE.get(cmd)
                if entry is None:
                    return {"ok": False, "error": "Unknown hook: %s" % cmd}
                is_write, _marks_dirty = entry
                if is_write:
                    return self._locked_exec(lambda: self._dispatch_hook(cmd, args), cmd, args)
                return self._dispatch_hook(cmd, args)

            # Table-driven dispatch
            entry = COMMAND_TABLE.get(cmd)
            if entry is None:
                return {"ok": False, "error": "Unknown command: {}".format(cmd)}

            check_unknown_keys(cmd, entry, args, self.brain)

            # Caller identity for the touched accumulator — resolved BEFORE the
            # handler runs, because handlers that pass **args to a brain method
            # pop `_caller_session` (see dispatch_common._pop_session_ctx), so it
            # would be gone by the time _accumulate_touched runs. caller_session
            # prefers an explicit session_id filter, else the proxy-stamped
            # `_caller_session` — the write tools send only the latter.
            caller_sess = caller_session(args)
            if entry.is_write:
                def _write():
                    result = entry.handler(self.brain, args, self.graph_changes)
                    if entry.marks_dirty:
                        self.dirty = True
                    return result
                result = self._locked_exec(_write, cmd, args)
            else:
                result = entry.handler(self.brain, args, self.graph_changes)
            # Record what Anchor's own tools touched this turn (Piece 3a). Only
            # Anchor's TCP calls reach _dispatch — the in-process encoder bypasses
            # it (calls COMMAND_TABLE handlers directly via _make_encoder_dispatch)
            # — so this is structurally Anchor-only, no encoding_source check.
            self._accumulate_touched(caller_sess, cmd, result)
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

    def _accumulate_touched(self, sess, cmd, result):
        """Append the node ids Anchor touched this turn to the session's per-turn
        accumulator (flushed as one `anchor_touched` S0 delta in
        post_response_common). Writes contribute their dispatch-authoritative
        `affected` (created/revised/archived); deliberate reads (get_node[s])
        contribute the resolved ids they returned. `sess` is the caller identity
        resolved before the handler ran. Failure-isolated, but LOUD: a real error
        here is logged (the silent version once hid the session-keying bug)."""
        if not isinstance(result, dict):
            return
        aff = result.get('affected')
        is_read = cmd in ('get_node', 'get_nodes')
        # Early-out BEFORE the session lookup, so the recall hot path (and any
        # other non-contributing command) pays nothing.
        if not isinstance(aff, dict) and not is_read:
            return
        if not sess:
            return
        try:
            touched = self.brain.get_or_create_session(sess).touched
            if isinstance(aff, dict):
                for k in ('created', 'revised', 'archived'):
                    ids = aff.get(k)
                    if ids:
                        touched[k].extend(ids)
            elif is_read:
                payload = result.get('result')
                nodes = payload if isinstance(payload, list) else [payload]
                for n in nodes:
                    # get_nodes mixes resolved node dicts with not-found entries
                    # {'id': <raw>, 'error': ...} — skip the latter so a bad ref
                    # doesn't land in `recalled` as a phantom node id.
                    if isinstance(n, dict) and n.get('id') and 'error' not in n:
                        touched['recalled'].append(n['id'])
        except Exception as e:
            try:
                self.brain._log_error('accumulate_touched', e,
                                      'cmd=%s sess=%s' % (cmd, str(sess)[:16]))
            except Exception:
                pass

    def _dispatch_hook(self, cmd: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch hook with telemetry. Caller (_dispatch) handles locking
        based on HOOK_TABLE.is_write — this function trusts that decision."""
        import servers.daemon_hooks as _hooks

        entry = self.HOOK_TABLE.get(cmd)
        if not entry:
            return {"error": "Unknown hook: %s" % cmd}

        _is_write, marks_dirty = entry
        hook_func = getattr(_hooks, cmd)

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

            # Semantics: nodes/edges report total store size (incl. archived);
            # locked/tensions report the active (non-archived) subset via the DAL
            # defaults — the identity-meaningful live counts.
            node_count = brain._nodes.count(archived=True)
            locked_count = brain._nodes.count_locked()
            edge_count = brain._graph.count_total()
            tension_count = brain._nodes.count_by_type('tension')

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
        """Periodically save brain if dirty + flush SessionContexts + health check + thread monitor.

        Recovering a hung daemon is not this loop's job — that's reactive,
        via the ping-based nets (`ensure_daemon` at session start, the MCP
        health monitor during a session), both force-restarting through
        `launchctl kickstart -k`.
        """
        while self.running:
            time.sleep(AUTOSAVE_INTERVAL_SECONDS)
            if self.dirty:
                if self.brain.write_lock.acquire(timeout=5.0):
                    try:
                        self.brain.save()
                        self.dirty = False
                        self._log("Autosaved")
                    except Exception as e:
                        self._log("Autosave error: {}".format(e))
                    finally:
                        self.brain.write_lock.release()
            # Flush cached SessionContexts (fatigue + counters) regardless of
            # self.dirty — ctx mutations don't toggle the main dirty flag, and
            # losing fatigue / counters on crash is acceptable but losing 60s
            # of message_count drift would mis-time the encoding heartbeat.
            try:
                self.brain.save_session_contexts()
            except Exception as e:
                self._log("SessionContext autosave error: {}".format(e))
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

    def _perform_restart(self):
        """Restart with fresh code, letting launchd own the respawn.

        When launchd manages the daemon (macOS), tear down cleanly and exit —
        launchd's KeepAlive (unconditional in com.brain.daemon.plist) respawns a
        fresh, launchd-managed instance. We do NOT spawn a detached rival: that
        orphan (PPID 1, non-launchd) is exactly what wedged KeepAlive into a
        respawn storm and had to be killed by hand (incidents 2026-07-03/04). We
        also do NOT `kickstart -k` ourselves — a daemon restarting *itself* only
        needs to exit; kickstart is for recovering a daemon from OUTSIDE, and
        self-kickstart just reopens the two-writer/self-SIGKILL races.

        Only on a genuine no-launchd platform (Linux / fresh install), where
        nothing would respawn us, do we spawn the successor directly.

        Both branches run the ordered `_shutdown()` teardown (drain → save →
        CLOSE DB → release lock LAST) BEFORE anything else can open brain.db, so
        a respawn/successor can never hold a second writer on it (two writers
        corrupt the indexes — see _teardown_brain).
        """
        time.sleep(0.5)  # let the {status: restarting} response reach the client
        self._log("Executing restart...")

        # Clear bytecode cache so the respawned daemon loads fresh code (the
        # point of a restart). Both the KeepAlive respawn and the no-launchd
        # Popen re-run start-daemon.sh, which imports servers.* from __pycache__.
        import shutil
        servers_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(servers_dir)
        for cache_dir in [os.path.join(servers_dir, '__pycache__'),
                          os.path.join(project_dir, 'hooks', 'scripts', '__pycache__')]:
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)

        if manages():
            # launchd owns the lifecycle: teardown (closes DB, releases lock
            # LAST) then exit; KeepAlive brings up a fresh managed instance.
            self._log("Restart: launchd-managed — clean teardown + exit, KeepAlive respawns.")
            self._shutdown()
            os._exit(0)
        else:
            # No launchd: nothing respawns us, so spawn the successor ourselves
            # via the ONE hardened spawn (daemon_launch). Teardown FIRST (closes
            # DB, releases lock) so the successor can't double-open brain.db,
            # THEN spawn, THEN exit.
            self._log("Restart: no launchd — spawning successor directly.")
            self._shutdown()
            spawn_detached_daemon(self.db_path)
            self._log("New daemon spawned. Shutting down old.")
            os._exit(0)

    def _run_idle_maintenance(self):
        """Poll brain's maintenance decision. Runs in thread pool.

        The polling cadence is the daemon's responsibility; the decision
        (idle threshold, min interval, persisted last-run timestamp) is
        the brain's. If the brain declines (returns None), this is a
        cheap no-op and we try again on the next poll. Previously the
        daemon owned this logic and lost its last-run state on restart —
        brain.run_maintenance_if_due() persists via brain_meta instead.
        """
        try:
            # Gate reads brain.activity (last_user_activity + encode-runs) and
            # consumes the encode runs on fire — single source of truth.
            result = self.brain.run_maintenance_if_due()
            if result is None:
                return  # not due — quiet no-op
            self._log("Maintenance ran (idle %.0fs): %s" % (
                result.get('idle_seconds', 0),
                ", ".join("%s=%s" % (u, str(r)[:60])
                          for u, r in result.get('units', {}).items())))
            # Sweep expired self-channel messages (dead-letter) on the
            # maintenance cadence — bounds self_inflight / self_delivered growth.
            try:
                from .scales.self_channel import signal as _self_signal
                reaped = _self_signal.reap_expired(self.brain)
                if reaped:
                    self._log("self-channel: reaped %d expired message(s)" % reaped)
            except Exception as _re:
                try:
                    self.brain._log_error('self_signal_reap', _re,
                                          'reap_expired in idle maintenance')
                except Exception:
                    pass
            self.brain.save()
        except Exception as e:
            self._log("Maintenance error: {}".format(e))
            try:
                self.brain._log_error('daemon_maintenance_poll', e,
                                      'brain.run_maintenance_if_due raised')
            except Exception:
                pass
        finally:
            self._s2_running = False

    def _run_scribe_poll(self):
        """Poll for a session whose S1 Scribe is due and fire it. Thread pool.

        Single-flight via _encode_lock — one encode at a time across sessions.
        A busy lock makes this a cheap no-op: the skipped session isn't lost, it
        re-qualifies and drains on a later poll (most-overdue first — the
        multi-session "queue"). The daemon owns concurrency; brain.scribe_due()
        owns the decision (who's due) via higher session functions. The encode
        runs on its own thread (run_unit_in_background), so this poll returns
        fast; the encode thread releases _encode_lock when it completes.
        """
        import time as _time
        from servers.scales.s1.encode_contract import (
            SCRIBE_RETRY_COOLDOWN_SECONDS, SCRIBE_MAX_FAILED_RETRIES,
            SCRIBE_CANDIDATE_WINDOW_MIN)
        spawned = False
        acquired = False
        try:
            if not self._encode_lock.acquire(blocking=False):
                return  # an encode is already running
            acquired = True
            now = _time.time()

            # Prune attempts older than the candidate window — those sessions
            # have aged out of consideration anyway (keeps the dicts bounded).
            horizon = SCRIBE_CANDIDATE_WINDOW_MIN * 60
            self._scribe_attempts = {s: t for s, t in self._scribe_attempts.items()
                                     if now - t < horizon}
            # Keep _scribe_failures keyed to live attempts — a session aged out
            # of _scribe_attempts has no live cooldown, so its failure count is
            # stale (otherwise _scribe_failures would leak entries forever).
            self._scribe_failures = {s: f for s, f in self._scribe_failures.items()
                                     if s in self._scribe_attempts}
            # Cooling = attempted within the cooldown → exclude from selection so
            # a failing (still-"due") session can't monopolize the poll.
            cooling = {s for s, t in self._scribe_attempts.items()
                       if now - t < SCRIBE_RETRY_COOLDOWN_SECONDS}

            due = self.brain.scribe_due(now=now, skip_sessions=cooling)
            if not due:
                return
            sid = due['session_id']

            # Re-firing a session we already attempted (past its cooldown but
            # STILL due) means the prior attempt never advanced the cadence — the
            # encode is crashing or skipping. Count + escalate loudly: the
            # starvation alarm can't catch this (turns is frozen at a fixed value,
            # so its `% ENCODE_EVERY` rate-limit never trips).
            if sid in self._scribe_attempts:
                fails = self._scribe_failures.get(sid, 0) + 1
                self._scribe_failures[sid] = fails
                if fails >= SCRIBE_MAX_FAILED_RETRIES:
                    try:
                        self.brain._log_error(
                            'scribe_repeated_failure',
                            RuntimeError('session %s re-fired %d times without '
                                         'advancing the encode cadence — encode '
                                         'is crashing or skipping' % (sid[:8], fails)),
                            'check s1e_* errors; turns_since_last_encode is stuck')
                    except Exception:
                        pass
            self._scribe_attempts[sid] = now

            from servers.scales.s1.scribe import S1Scribe
            from servers.scales.runner import run_unit_in_background

            def _count_encode(write_actions, _sid=sid):
                # on_complete fires only when the encode COMPLETED — a crash
                # skips it (run_unit_in_background's except). So any call here
                # means the cadence advanced (encoding_prompt was written), even
                # for a 0-write encode. Clear the cooldown/failure state
                # unconditionally — only a genuine crash (no on_complete) leaves
                # the session cooling, which is exactly what should bound the
                # retry. (Clearing only on write>0 falsely escalated a healthy
                # 0-write encode on its next legit re-fire.) record_encode_run
                # stays gated on write_actions — the S2 gate cares only about
                # material actually written.
                self._scribe_attempts.pop(_sid, None)
                self._scribe_failures.pop(_sid, None)
                if write_actions > 0:
                    self.brain.activity.record_encode_run()

            scribe = S1Scribe(self.brain, session_id=sid, counter=due['counter'])
            run_unit_in_background(scribe, name='s1e', lock=self._encode_lock,
                                   on_complete=_count_encode)
            spawned = True  # lock ownership transferred to the encode thread
        except Exception as e:
            try:
                self.brain._log_error('scribe_poll', e, 'Scribe reactor poll')
            except Exception:
                pass
        finally:
            # Release the encode lock only if we acquired it but did NOT hand it
            # to an encode thread (nothing due, or an error before spawn). A
            # failed release means the lock state is corrupt and the encoder is
            # jammed — log loud, don't swallow.
            if acquired and not spawned:
                try:
                    self._encode_lock.release()
                except Exception as _re:
                    try:
                        self.brain._log_error(
                            'scribe_lock_release_failed', _re,
                            'encode lock release failed — encoder may be jammed')
                    except Exception:
                        pass
            self._scribe_poll_running = False

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

    def _arm_force_exit_backstop(self):
        """Last-resort guard: if a drain wedges (e.g. an embedder CPU spin), force
        the process to exit so teardown always terminates. SHUTDOWN_BACKSTOP_S is
        sized to let a normal in-flight command finish first, yet stay under
        launchd's SIGTERM→SIGKILL window so we exit on our own terms."""
        import _thread
        def _force_exit():
            time.sleep(self.SHUTDOWN_BACKSTOP_S)
            self._log("Teardown wedged — forcing exit after %ds" % self.SHUTDOWN_BACKSTOP_S)
            os._exit(0)
        _thread.start_new_thread(_force_exit, ())

    def _signal_drain_shutdown(self):
        """Signal the single bg-writer drain worker to stop (it drains BOTH the
        embed and recall-write queues, so one signal covers both). Wakes it out
        of its interval wait so it exits at the next check. Idempotent (the Event
        is the single signal). Loud on failure."""
        try:
            from servers import embed_queue
            embed_queue.request_shutdown()
        except Exception as e:
            self._log_shutdown_error('queue_shutdown', e)

    def _teardown_brain(self):
        """The single ordered teardown: drain everything holding a DB connection,
        save+close, release the singleton lock/PID LAST. Never raises — every step
        is guarded and logged. Order is load-bearing:
          1. signal queues    → wakes the bg-writer so join_worker can return
          2. drain pool       → in-flight commands finish on OPEN connections
          3. join bg-writer   → its in-flight drain settles off conn_bg_writer
          4. save + close     → no live consumer touches the connections now
          5. release lock/PID → LAST: a racer's ensure_daemon blocks on this lock
             before spawning, so it can't open a 2nd writer on brain.db while our
             connections were still open (two writers corrupt the indexes).
        This is the fix for the 'Cannot operate on a closed database' boot failure
        of 2026-06-06, where close() ran before the pool + bg-writer were drained."""
        self._signal_drain_shutdown()
        # Drain the pool. Catch EVERYTHING (not only the cancel_futures TypeError)
        # so a pool-drain error can never skip the save+close below.
        try:
            try:
                self._pool.shutdown(wait=True, cancel_futures=False)
            except TypeError:                       # Python <3.9 has no cancel_futures
                self._pool.shutdown(wait=True)
        except Exception as e:
            self._log_shutdown_error('pool_drain', e)
        try:
            from servers import embed_queue
            embed_queue.join_worker(timeout=3.0)
        except Exception as e:
            self._log_shutdown_error('bg_writer_join', e)
        try:
            if self.brain:
                self.brain.save()
                self.brain.close()
                self.brain = None
        except Exception as e:
            self._log_shutdown_error('brain_save_close', e)
        # Resources LAST — see step 5. _cleanup re-signals the queues (idempotent)
        # and releases the socket, PID file, and the singleton fcntl lock.
        self._cleanup()

    def _shutdown(self):
        """Clean shutdown: arm the force-exit backstop, then run the shared
        drain-then-close teardown (see _teardown_brain for the ordering)."""
        self._log("Shutting down...")
        self._arm_force_exit_backstop()
        self._teardown_brain()

    def _cleanup(self):
        """Close server socket, observer channel, remove PID and lock files.
        Also signals background-writer queues to stop draining so the worker
        thread exits cleanly at its next interval check (daemon thread is
        force-killed on process exit anyway, but signaling lets a partial
        drain finish without crash-rollback).
        Idempotent — safe to call multiple times (signal + atexit + explicit)."""
        self._signal_drain_shutdown()
        self._close_socket()
        # Only the instance that became the serving daemon (wrote the PID after
        # binding) owns these files. A duplicate that deferred before binding
        # must NOT unlink the incumbent's PID/status out from under it.
        if getattr(self, '_wrote_pid', False):
            for path in [self.pid_path, get_status_path()]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception as _ue:
                    # Stale PID/status files cause "Another daemon running"
                    # errors on next boot. Silent failure here hides that
                    # class of bug. Stderr is the right channel (we may
                    # not have a brain handle at this point in shutdown).
                    print('[brain-daemon] failed to remove %s: %s' %
                          (path, _ue), file=sys.stderr)
        try:
            # Idempotent: _cleanup runs more than once (explicit _shutdown +
            # atexit). A closed file object is still truthy, so the bare
            # `and self._lock_fd` guard let the 2nd call re-flock an
            # already-closed fd → "I/O operation on closed file" — dozens of
            # false alarms that looked like a lock leak but weren't. Skip when
            # already closed; null in finally so later calls short-circuit.
            if getattr(self, '_lock_fd', None) and not self._lock_fd.closed:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
        except Exception as _le:
            # A GENUINE release failure (not the harmless double-call above).
            # A leaked lock prevents daemon restart, so surface it where the
            # operator can SEE it (the dashboard reads debug_log, not stderr).
            self._log_shutdown_error('release_lock_fd', _le)
        finally:
            self._lock_fd = None

    def _log_shutdown_error(self, source, err):
        """Surface a shutdown-time error to debug_log (which the dashboard
        reads) AND stderr. brain is already closed during _cleanup, so write
        via a fresh connection — the way log_hook_error logs without a Brain
        handle. Never raises: logging must not crash shutdown."""
        print('[brain-daemon] %s: %s' % (source, err), file=sys.stderr)
        try:
            import sqlite3
            from .clock import iso_now
            logs_db = os.path.join(os.path.dirname(self.db_path), 'brain_logs.db')
            conn = sqlite3.connect(logs_db, timeout=5)
            conn.execute(
                "INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) "
                "VALUES ('daemon', 'error', ?, ?, ?)",
                (source, json.dumps({'error': str(err), 'type': type(err).__name__}), iso_now()))
            conn.commit()
            conn.close()
        except Exception:
            pass  # stderr already carries it

    def _log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        print("[brain-daemon {}] {}".format(ts, message), file=sys.stderr)

