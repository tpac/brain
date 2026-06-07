"""Tests for hung-daemon detection + recovery (the consolidated path).

Recovery lives in servers/daemon_client.py — one primitive, recover_daemon(),
called by both the MCP health monitor and the recall hook. These tests lock:
  - liveness (is_daemon_responsive) tells a live daemon from a hung corpse,
  - recover_daemon's guards: responsive no-op, maintenance, cooldown, breaker,
  - _relaunch_daemon prefers launchd kickstart, falls back to kill + spawn,
  - hook_common delegates instead of carrying its own copy.

Pure unit tests — no Brain, no embedder, no real daemon; launchctl + the
relaunch are mocked.

Context: 2026-05-28 host slept mid-request, woke to a daemon holding its port
but servicing nothing. The old liveness check (bare connect / PID-exists) and
the old "restart" (a no-op deferring to launchd, which can't kill a non-exiting
corpse) both missed it.
"""

import os
import sys
import json
import time
import errno
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import servers.daemon_client as dc


class TestIsDaemonResponsive(unittest.TestCase):
    def test_responsive_when_ping_ok(self):
        with patch.object(dc, "_can_connect", return_value={"ok": True}):
            self.assertTrue(dc.is_daemon_responsive())

    def test_unresponsive_when_no_reply(self):
        # A corpse: connect succeeds but ping never returns ok → empty dict.
        with patch.object(dc, "_can_connect", return_value={}):
            self.assertFalse(dc.is_daemon_responsive())


class TestRecoverDaemon(unittest.TestCase):
    def setUp(self):
        self.state = os.path.join(tempfile.gettempdir(),
                                  "brain-recovery-test-%d.json" % os.getpid())
        self._clear()

    def tearDown(self):
        self._clear()

    def _clear(self):
        if os.path.exists(self.state):
            os.remove(self.state)

    def _seed(self, last_attempt, attempts):
        with open(self.state, "w") as f:
            json.dump({"last_attempt": last_attempt, "attempts": attempts}, f)

    def _read(self):
        with open(self.state) as f:
            return json.load(f)

    def test_responsive_is_noop(self):
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=True), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertTrue(dc.recover_daemon())
            relaunch.assert_not_called()

    def test_responsive_clears_failure_streak(self):
        self._seed(time.time(), 3)
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=True), \
             patch.object(dc, "_relaunch_daemon"):
            dc.recover_daemon()
        self.assertEqual(self._read()["attempts"], 0)

    def test_maintenance_mode_skips(self):
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=True), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_not_called()

    def test_down_and_clean_triggers_restart(self):
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_called_once()
        self.assertEqual(self._read()["attempts"], 1)

    def test_cooldown_blocks_double_restart(self):
        self._seed(time.time(), 1)  # restart issued just now
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_not_called()

    def test_circuit_breaker_opens_at_max(self):
        # Past cooldown, but already at the attempt ceiling within the window.
        self._seed(time.time() - dc._RECOVERY_COOLDOWN_S - 1, dc._RECOVERY_MAX_ATTEMPTS)
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_not_called()

    def test_stale_streak_ages_out_and_restarts(self):
        # Old incident beyond the window → streak resets, recovery resumes.
        self._seed(time.time() - dc._RECOVERY_WINDOW_S - 1, dc._RECOVERY_MAX_ATTEMPTS)
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_called_once()
        self.assertEqual(self._read()["attempts"], 1)


class TestRelaunchDaemon(unittest.TestCase):
    def test_kickstart_success_skips_fallback(self):
        with patch.object(dc.subprocess, "run", return_value=MagicMock(returncode=0)) as run, \
             patch.object(dc, "_kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon") as ensure:
            dc._relaunch_daemon(None)
            run.assert_called_once()
            argv = run.call_args[0][0]
            self.assertEqual(argv[:3], ["launchctl", "kickstart", "-k"])
            self.assertIn("com.brain.daemon", argv[3])
            kill.assert_not_called()
            ensure.assert_not_called()

    def test_kickstart_failure_falls_back_to_kill_and_spawn(self):
        # kickstart fails AND we ARE the daemon's source → own the kill + respawn.
        with patch.object(dc.subprocess, "run", return_value=MagicMock(returncode=1)), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={}), \
             patch.object(dc, "_kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon", return_value=True) as ensure:
            dc._relaunch_daemon("/tmp/x/brain.db")
            kill.assert_called_once()
            ensure.assert_called_once_with("/tmp/x/brain.db")

    def test_kickstart_failure_from_non_source_defers_never_kills(self):
        # A worktree / 2nd clone must NEVER SIGKILL the shared daemon: kickstart
        # failed AND we are not the source → defer to launchd/source, don't kill.
        with patch.object(dc.subprocess, "run", return_value=MagicMock(returncode=1)), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value={}), \
             patch.object(dc, "_kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon") as ensure:
            dc._relaunch_daemon("/tmp/x/brain.db")
            kill.assert_not_called()
            ensure.assert_not_called()


class TestHookDelegation(unittest.TestCase):
    """hook_common must delegate, not carry its own recovery copy."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts'))
        import hook_common
        cls.hc = hook_common

    def test_daemon_available_delegates_to_responsive(self):
        with patch("servers.daemon_client.is_daemon_responsive", return_value=True):
            self.assertTrue(self.hc.daemon_available())
        with patch("servers.daemon_client.is_daemon_responsive", return_value=False):
            self.assertFalse(self.hc.daemon_available())

    def test_unavailable_error_calls_recover_daemon(self):
        with patch("servers.daemon_client.recover_daemon") as recover:
            msg = self.hc.daemon_unavailable_error("recall")
            recover.assert_called_once()
            self.assertIn("CRITICAL", msg)

    def test_hook_common_has_no_local_recovery(self):
        # The duplicate must be gone — recovery lives only in daemon_client.
        self.assertFalse(hasattr(self.hc, "force_restart_daemon"))


class TestEnsureDaemonRoutesThroughLaunchd(unittest.TestCase):
    """ensure_daemon must route every (re)start through launchd (kickstart),
    serialized under the singleton lock — never Popen alongside KeepAlive.

    Regression guard for the Errno-48 storm (2026-06-04): N concurrent boots
    each saw stale code and independently killed + respawned while launchd's
    KeepAlive also respawned, so several processes raced to bind the port.
    """

    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="brain-ensure-test-")
        self._db = os.path.join(self._dir, "brain.db")
        self._lock = os.path.join(self._dir, "daemon.lock")

    def tearDown(self):
        for p in (self._lock, os.path.join(self._dir, "daemon.log"), self._db):
            if os.path.exists(p):
                os.remove(p)
        try:
            os.rmdir(self._dir)
        except OSError:
            pass

    def test_healthy_current_code_is_noop(self):
        # Responsive + same code → return True without touching the lifecycle.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_launchd_kickstart") as ks, \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            popen.assert_not_called()

    def test_stale_code_kickstarts_once_never_popen(self):
        # Responsive but stale → exactly one launchd kickstart, never a Popen.
        # _code_changed: fast-path(stale) → under-lock recheck(stale) → ready.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", side_effect=[True, True, False]), \
             patch.object(dc, "_launchd_kickstart", return_value=True) as ks, \
             patch.object(dc.subprocess, "Popen") as popen, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_called_once()
            popen.assert_not_called()

    def test_concurrent_winner_already_restarted_skips_kickstart(self):
        # The anti-storm guarantee: fast-path sees stale and falls through to
        # the lock, but by the time we hold it another caller already restarted
        # to current code → we re-check and do NOTHING (no second kickstart).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", side_effect=[True, False]), \
             patch.object(dc, "_launchd_kickstart") as ks, \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            popen.assert_not_called()

    def test_no_launchd_falls_back_to_direct_spawn(self):
        # Daemon down, nothing serving, AND launchd genuinely not managing it
        # (kickstart fails AND _launchd_manages_daemon False) → the only case a
        # direct Popen is legitimate (no KeepAlive to race). _can_connect calls:
        # fast-path, under-lock recheck, post-kickstart re-ping, post-spawn ready.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect",
                          side_effect=[{"ok": False}, {"ok": False}, {"ok": False}, {"ok": True}]), \
             patch.object(dc, "_await_responsive", return_value={"ok": False}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_launchd_kickstart", return_value=False) as ks, \
             patch.object(dc, "_launchd_manages_daemon", return_value=False), \
             patch.object(dc, "_port_is_occupied", return_value=False), \
             patch.object(dc, "_debugger_friendly_python", return_value="/usr/bin/python3"), \
             patch.object(dc.subprocess, "Popen") as popen, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_called_once()
            popen.assert_called_once()

    def test_kickstart_killed_incumbent_then_failed_does_not_false_defer(self):
        # `kickstart -k` SIGKILLs the incumbent before respawning. If it then
        # returns nonzero/timed-out, the port is FREE — but the pre-kickstart
        # ping said {ok:True}. The defer branch must RE-PING (now {ok:False}) and
        # NOT defer on the stale snapshot (which would report "ready" with
        # nothing serving). _can_connect: fast(ok), under-lock(ok), re-ping(DOWN),
        # post-spawn(ok).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect",
                          side_effect=[{"ok": True}, {"ok": True}, {"ok": False}, {"ok": True}]), \
             patch.object(dc, "_code_changed", return_value=True), \
             patch.object(dc, "_launchd_kickstart", return_value=False), \
             patch.object(dc, "_launchd_manages_daemon", return_value=False), \
             patch.object(dc, "_port_is_occupied", return_value=False), \
             patch.object(dc, "_debugger_friendly_python", return_value="/usr/bin/python3"), \
             patch.object(dc.subprocess, "Popen") as popen, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
            popen.assert_called_once()   # spawned — did NOT false-defer on stale resp

    def test_responsive_stale_kickstart_unreachable_defers_not_spawn(self):
        # Regression guard for the 2026-06-05 orphan storm. A worktree session
        # runs stale code, so the running daemon looks stale; kickstart can't
        # reach launchd from that context. We MUST defer to the responsive
        # incumbent — never kill it and Popen a competitor (the rival orphaned,
        # squatted the port, and crash-looped the real launchd daemon).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", return_value=True), \
             patch.object(dc, "_launchd_kickstart", return_value=False), \
             patch.object(dc, "_kill_daemon") as kill, \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertTrue(dc.ensure_daemon(self._db))
            kill.assert_not_called()
            popen.assert_not_called()

    def test_down_but_launchd_managed_kickstart_failed_defers_to_keepalive(self):
        # Daemon down, kickstart failed (transient), but launchd DOES manage the
        # service → don't race a manual spawn; let KeepAlive bring it up.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": False}), \
             patch.object(dc, "_await_responsive", return_value={"ok": False}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_launchd_kickstart", return_value=False), \
             patch.object(dc, "_launchd_manages_daemon", return_value=True), \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertFalse(dc.ensure_daemon(self._db))
            popen.assert_not_called()

    def test_maintenance_mode_skips_everything(self):
        with patch.object(dc, "is_maintenance_mode", return_value=True), \
             patch.object(dc, "_launchd_kickstart") as ks, \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertFalse(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            popen.assert_not_called()

    def test_worktree_client_up_is_noop(self):
        # A non-source checkout (worktree) is a PURE CLIENT: daemon up → return
        # True without ever touching kickstart/spawn, whatever the code version.
        # It's still Anchor (shared brain.db) — just not the daemon's lifecycle owner.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_launchd_kickstart") as ks, \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            popen.assert_not_called()

    def test_worktree_client_down_waits_never_kickstarts(self):
        # A worktree never (re)starts the shared daemon — launchd KeepAlive owns
        # recovery (a worktree restart can't converge its code, it only churns).
        # Down → wait out the relaunch grace, return whether it came up; the grace
        # here recovers it → True, with NO kickstart and NO spawn.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value={"ok": False}), \
             patch.object(dc, "_await_responsive", return_value={"ok": True}), \
             patch.object(dc, "_launchd_kickstart") as ks, \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            popen.assert_not_called()

    def test_owner_slow_daemon_recovers_within_grace_not_kickstarted(self):
        # THE core fix. The source checkout sees a non-responsive 2s ping, but the
        # daemon is only slow / mid-relaunch. The grace probe catches it coming
        # back → NO kickstart (the false-down that started the restart storm).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": False}), \
             patch.object(dc, "_await_responsive", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_launchd_kickstart") as ks, \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()   # slow ≠ dead → not restarted
            popen.assert_not_called()


class TestDuplicateDaemonDefers(unittest.TestCase):
    """A duplicate (port already served by a responsive daemon) must raise
    DuplicateDaemonError so the supervisor exits cleanly instead of crash-
    looping on bind. Backstop for the Errno-48 storm (2026-06-05) — closes the
    class regardless of HOW a second daemon was spawned."""

    def _daemon(self):
        from servers.daemon_server import BrainDaemon
        return BrainDaemon("/tmp/nonexistent-brain-test.db")

    def test_run_precheck_defers_when_responsive(self):
        from servers import daemon_server as ds
        d = self._daemon()
        # Responsive incumbent → defer BEFORE loading the brain (no DB writers).
        with patch("servers.daemon_client.is_daemon_responsive", return_value=True), \
             patch.object(d, "_load_brain") as load:
            with self.assertRaises(ds.DuplicateDaemonError):
                d._run()
            load.assert_not_called()

    def test_bind_socket_defers_on_eaddrinuse_when_responsive(self):
        from servers import daemon_server as ds
        d = self._daemon()
        fake_sock = MagicMock()
        fake_sock.bind.side_effect = OSError(errno.EADDRINUSE, "in use")
        with patch.object(ds.socket, "socket", return_value=fake_sock), \
             patch("servers.daemon_client.is_daemon_responsive", return_value=True):
            with self.assertRaises(ds.DuplicateDaemonError):
                d._bind_socket()

    def test_bind_socket_retries_when_holder_unresponsive(self):
        # A non-responsive holder (TIME_WAIT / hung corpse) is NOT a duplicate —
        # normal retry, then raise OSError for recovery to handle.
        from servers import daemon_server as ds
        d = self._daemon()
        d.SOCKET_BIND_RETRIES = 2
        fake_sock = MagicMock()
        fake_sock.bind.side_effect = OSError(errno.EADDRINUSE, "in use")
        with patch.object(ds.socket, "socket", return_value=fake_sock), \
             patch("servers.daemon_client.is_daemon_responsive", return_value=False), \
             patch.object(ds.time, "sleep"):
            with self.assertRaises(OSError):
                d._bind_socket()


class TestShutdownDrainOrder(unittest.TestCase):
    """_shutdown must drain the worker pool AND settle the bg-writer/queues
    BEFORE saving + closing the brain — otherwise an in-flight hook or a mid-drain
    touches a closed connection ('Cannot operate on a closed database', the
    2026-06-06 boot failure, where close() ran before the pool/bg-writer drained).
    Regression guard for the drain ORDER."""

    def test_drains_then_closes_then_releases_resources_last(self):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__ — no real brain/socket
        order = []
        d.brain = MagicMock()
        d.brain.save.side_effect = lambda: order.append("save")
        d.brain.close.side_effect = lambda: order.append("close")
        d._pool = MagicMock()
        d._pool.shutdown.side_effect = lambda **k: order.append("pool")
        d._log = lambda *a, **k: None
        d._log_shutdown_error = lambda *a, **k: None
        d._signal_drain_shutdown = lambda: order.append("signal")   # don't touch the real queues
        d._cleanup = lambda: order.append("release")                # _cleanup = release lock/PID/socket
        with patch("servers.embed_queue.join_worker",
                   side_effect=lambda **k: order.append("join")), \
             patch("_thread.start_new_thread"):                     # don't arm the os._exit backstop
            d._shutdown()
        # signal queues → drain pool → join bg-writer → save+close → release lock/PID LAST.
        # Releasing the singleton lock LAST is what stops a racer opening a 2nd
        # writer on brain.db while our connections were still open.
        self.assertEqual(order, ["signal", "pool", "join", "save", "close", "release"])


class TestEmbedQueueShutdown(unittest.TestCase):
    """request_shutdown() must wake the drain worker out of its interval wait
    immediately (an Event, not a passive flag), so daemon shutdown doesn't block
    for a full EMBED_DRAIN_INTERVAL and brain.close() can't race a mid-drain."""

    def setUp(self):
        import servers.embed_queue as eq
        self.eq = eq
        self._reset()

    def tearDown(self):
        self._reset()

    def _reset(self):
        eq = self.eq
        with eq._lock:
            eq._worker_started = False
        eq._shutdown_event.clear()      # the single bg-writer shutdown signal
        eq._worker_thread = None

    def test_request_shutdown_wakes_worker_promptly(self):
        eq = self.eq
        # Long interval — only the Event (not the timeout) can make it exit fast.
        with patch.object(eq, "EMBED_DRAIN_INTERVAL", 30):
            eq.start(MagicMock())
            self.assertTrue(eq._worker_thread.is_alive())
            eq.request_shutdown()                 # sets the Event → wakes the worker out of the 30s wait now
            eq.join_worker(timeout=2.0)
            self.assertFalse(eq._worker_thread.is_alive(),
                             "worker must exit on the Event, not wait the full 30s interval")


class TestAwaitResponsive(unittest.TestCase):
    """_await_responsive — the boot-path liveness gate. WALL-CLOCK bounded,
    returns on the first ok ping (no sleep), returns the last resp when never up."""

    def test_returns_on_first_ok_without_sleeping(self):
        with patch.object(dc, "_can_connect", return_value={"ok": True}) as ping, \
             patch.object(dc.time, "sleep") as slept:
            resp = dc._await_responsive(20.0)
        self.assertEqual(resp, {"ok": True})
        ping.assert_called_once()       # one ping, no retry when already up
        slept.assert_not_called()       # never sleeps a live daemon

    def test_wall_clock_bound_returns_last_resp_when_never_up(self):
        # deadline_s=0 → no retry past the first ping; returns the last (down) resp.
        with patch.object(dc, "_can_connect", return_value={}), \
             patch.object(dc.time, "sleep") as slept:
            resp = dc._await_responsive(0.0)
        self.assertEqual(resp, {})      # never ok → returns the last resp (down)
        slept.assert_not_called()       # zero deadline never enters the retry loop


if __name__ == "__main__":
    unittest.main()
