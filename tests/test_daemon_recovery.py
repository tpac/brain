"""Tests for hung-daemon detection + recovery (the consolidated path).

Recovery lives in servers/daemon_client.py — one primitive, recover_daemon(),
called by both the MCP health monitor and the recall hook. The launchd + spawn
MECHANISMS (kickstart, manages, kill_daemon, spawn_detached_daemon) live in
servers/daemon_launch.py — daemon_client and daemon_server both consume them as
public names. These tests lock:
  - liveness (is_daemon_responsive) tells a live daemon from a hung corpse,
  - recover_daemon's guards: responsive no-op, maintenance, cooldown, breaker,
  - _relaunch_daemon prefers launchd kickstart, falls back to kill + spawn,
  - the one hardened spawn primitive — no raw Popen outside daemon_launch,
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
import servers.daemon_launch as dl


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
        with patch.object(dl.subprocess, "run", return_value=MagicMock(returncode=0)) as run, \
             patch.object(dc, "kill_daemon") as kill, \
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
        with patch.object(dl.subprocess, "run", return_value=MagicMock(returncode=1)), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={}), \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon", return_value=True) as ensure:
            dc._relaunch_daemon("/tmp/x/brain.db")
            kill.assert_called_once()
            ensure.assert_called_once_with("/tmp/x/brain.db")

    def test_kickstart_failed_but_incumbent_responsive_defers_never_kills(self):
        # Step 3 drift fix. recover_daemon's single 2s ping can call a slow/busy
        # daemon "down"; if kickstart then also fails transiently, the old code
        # went straight to SIGKILL — killing a live daemon mid-request that
        # ensure_daemon's ladder would have deferred to. A daemon that answers
        # the re-ping must be deferred to, never killed.
        with patch.object(dl.subprocess, "run", return_value=MagicMock(returncode=1)), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon") as ensure:
            dc._relaunch_daemon("/tmp/x/brain.db")
            kill.assert_not_called()
            ensure.assert_not_called()

    def test_kickstart_failure_from_non_source_defers_never_kills(self):
        # A worktree / 2nd clone must NEVER SIGKILL the shared daemon: kickstart
        # failed AND we are not the source → defer to launchd/source, don't kill.
        with patch.object(dl.subprocess, "run", return_value=MagicMock(returncode=1)), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value={}), \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon") as ensure:
            dc._relaunch_daemon("/tmp/x/brain.db")
            kill.assert_not_called()
            ensure.assert_not_called()


class TestLaunchdHelpersDegradeWhenAbsent(unittest.TestCase):
    """The launchd helpers must return False — not raise — when the `launchctl`
    binary is absent (i.e. on Linux). The no-launchd direct-spawn fallback
    (test_no_launchd_falls_back_to_direct_spawn) RELIES on this: it mocks both
    helpers to False but never proves they actually degrade that way when
    launchctl is missing. Narrowing the `except Exception` in either helper to a
    subprocess-specific error would let FileNotFoundError propagate, break the
    Linux daemon path, and slip past every other test. This locks the contract.
    """

    def test_kickstart_returns_false_when_launchctl_missing(self):
        with patch.object(dl.subprocess, "run",
                          side_effect=FileNotFoundError("launchctl")):
            self.assertFalse(dl.kickstart())

    def test_manages_returns_false_when_launchctl_missing(self):
        with patch.object(dl.subprocess, "run",
                          side_effect=FileNotFoundError("launchctl")):
            self.assertFalse(dl.manages())


class TestManagesDaemonTransientVsAbsent(unittest.TestCase):
    """manages()==False AUTHORIZES the direct-spawn fallback that
    can orphan a competing daemon (incident 2026-07-03: a transient `launchctl
    print` failure was read as 'launchd absent', spawned an orphan that squatted
    the lock, and launchd's KeepAlive stormed 135k exits). False must mean
    DEFINITIVELY-not-managed; a launchctl that merely failed to answer is
    INDETERMINATE and must defer (True), never spawn a rival.
    """

    def _ok(self, returncode, stderr=""):
        return MagicMock(returncode=returncode, stdout="", stderr=stderr)

    def test_true_when_running(self):
        with patch.object(dl.subprocess, "run", return_value=self._ok(0)):
            self.assertTrue(dl.manages())

    def test_false_on_clean_not_found(self):
        # rc 113 + "Could not find service …" is a definitive absence.
        stderr = 'Could not find service "com.brain.daemon" in domain for user gui: 503'
        with patch.object(dl.subprocess, "run", return_value=self._ok(113, stderr)):
            self.assertFalse(dl.manages())

    def test_true_on_timeout(self):
        # launchctl present but hung → indeterminate → defer, do NOT spawn.
        with patch.object(dl.subprocess, "run",
                          side_effect=dl.subprocess.TimeoutExpired(
                              cmd=["launchctl"], timeout=10)):
            self.assertTrue(dl.manages())

    def test_true_on_unexpected_nonzero(self):
        # non-zero WITHOUT the not-found signature (e.g. gui/<uid> unaddressable)
        # is indeterminate, not absence.
        with patch.object(dl.subprocess, "run",
                          return_value=self._ok(1, "Bootstrap operation not permitted")):
            self.assertTrue(dl.manages())

    def test_retries_transient_then_reads_answer(self):
        # A one-off timeout must not decide the outcome — the retry sees rc 0.
        with patch.object(dl.subprocess, "run",
                          side_effect=[dl.subprocess.TimeoutExpired(cmd=["launchctl"], timeout=10),
                                       self._ok(0)]) as run:
            self.assertTrue(dl.manages())
            self.assertEqual(run.call_count, 2)


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
        # Existing brain.db + dead daemon = a crash: recovery must fire.
        with tempfile.TemporaryDirectory(prefix="brain-hook-deleg-") as tmp:
            db = os.path.join(tmp, "brain.db")
            open(db, "w").close()
            with patch.object(self.hc, "db_dir", tmp), \
                 patch.object(self.hc, "db_path", db), \
                 patch("servers.daemon_client.recover_daemon") as recover:
                msg = self.hc.daemon_unavailable_error("recall")
                recover.assert_called_once_with(db)
                self.assertIn("CRITICAL", msg)

    def test_unconfigured_install_exits_without_recovery(self):
        # No brain.db AND no daemon = Anchor never came up, not a crash.
        # ANCHOR OFFLINE contract (b2aaa1f): log + exit 1, NO recovery attempt —
        # boot creates brains, recovery must not.
        with tempfile.TemporaryDirectory(prefix="brain-hook-deleg-") as tmp:
            with patch.object(self.hc, "db_dir", tmp), \
                 patch.object(self.hc, "db_path", os.path.join(tmp, "brain.db")), \
                 patch("servers.daemon_client.recover_daemon") as recover:
                with self.assertRaises(SystemExit) as cm:
                    self.hc.daemon_unavailable_error("recall")
                self.assertEqual(cm.exception.code, 1)
                recover.assert_not_called()

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
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_stale_code_kickstarts_once_never_popen(self):
        # Responsive but stale → exactly one launchd kickstart, never a spawn.
        # _code_changed: fast-path(stale) → under-lock recheck(stale) → ready.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", side_effect=[True, True, False]), \
             patch.object(dc, "kickstart", return_value=True) as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_called_once()
            spawn.assert_not_called()

    def test_concurrent_winner_already_restarted_skips_kickstart(self):
        # The anti-storm guarantee: fast-path sees stale and falls through to
        # the lock, but by the time we hold it another caller already restarted
        # to current code → we re-check and do NOTHING (no second kickstart).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", side_effect=[True, False]), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_no_launchd_falls_back_to_direct_spawn(self):
        # Daemon down, nothing serving, AND launchd genuinely not managing it
        # (kickstart fails AND manages() False) → the only case a direct spawn
        # is legitimate (no KeepAlive to race). _can_connect calls:
        # fast-path, under-lock recheck, post-kickstart re-ping, post-spawn ready.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect",
                          side_effect=[{"ok": False}, {"ok": False}, {"ok": False}, {"ok": True}]), \
             patch.object(dc, "_await_responsive", return_value={"ok": False}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "kickstart", return_value=False) as ks, \
             patch.object(dc, "manages", return_value=False), \
             patch.object(dc, "port_is_occupied", return_value=False), \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_called_once()
            spawn.assert_called_once_with(self._db)

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
             patch.object(dc, "kickstart", return_value=False), \
             patch.object(dc, "manages", return_value=False), \
             patch.object(dc, "port_is_occupied", return_value=False), \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
            spawn.assert_called_once()   # spawned — did NOT false-defer on stale resp

    def test_responsive_stale_kickstart_unreachable_defers_not_spawn(self):
        # Regression guard for the 2026-06-05 orphan storm. A worktree session
        # runs stale code, so the running daemon looks stale; kickstart can't
        # reach launchd from that context. We MUST defer to the responsive
        # incumbent — never kill it and spawn a competitor (the rival orphaned,
        # squatted the port, and crash-looped the real launchd daemon).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", return_value=True), \
             patch.object(dc, "kickstart", return_value=False), \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            kill.assert_not_called()
            spawn.assert_not_called()

    def test_down_but_launchd_managed_kickstart_failed_defers_to_keepalive(self):
        # Daemon down, kickstart failed (transient), but launchd DOES manage the
        # service → don't race a manual spawn; let KeepAlive bring it up.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": False}), \
             patch.object(dc, "_await_responsive", return_value={"ok": False}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "kickstart", return_value=False), \
             patch.object(dc, "manages", return_value=True), \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertFalse(dc.ensure_daemon(self._db))
            spawn.assert_not_called()

    def test_maintenance_mode_skips_everything(self):
        with patch.object(dc, "is_maintenance_mode", return_value=True), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertFalse(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_worktree_client_up_is_noop(self):
        # A non-source checkout (worktree) is a PURE CLIENT: daemon up → return
        # True without ever touching kickstart/spawn, whatever the code version.
        # It's still Anchor (shared brain.db) — just not the daemon's lifecycle owner.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_worktree_client_down_waits_never_kickstarts(self):
        # A worktree never (re)starts the shared daemon — launchd KeepAlive owns
        # recovery (a worktree restart can't converge its code, it only churns).
        # Down → wait out the relaunch grace, return whether it came up; the grace
        # here recovers it → True, with NO kickstart and NO spawn.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value={"ok": False}), \
             patch.object(dc, "_await_responsive", return_value={"ok": True}), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

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
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()   # slow ≠ dead → not restarted
            spawn.assert_not_called()


class TestDuplicateDaemonDefers(unittest.TestCase):
    """A duplicate (port already served by a responsive daemon) must raise
    DuplicateDaemonError so the supervisor exits cleanly instead of crash-
    looping on bind. Backstop for the Errno-48 storm (2026-06-05) — closes the
    class regardless of HOW a second daemon was spawned.

    Step 6 removed _run's is_daemon_responsive PRE-check: the flock (acquired in
    start() before the supervisor loop) is the singleton primitive, and while we
    hold it no other same-uid process is past the flock-acquire — so none can be
    serving our port. test_lock_holder_blocks_start_before_run pins that; the
    bind-time EADDRINUSE backstop below covers the only residue (a different uid
    colliding via uid%100, and the acquire→bind race)."""

    def _daemon(self):
        from servers.daemon_server import BrainDaemon
        return BrainDaemon("/tmp/nonexistent-brain-test.db")

    def test_lock_holder_blocks_start_before_run(self):
        # The flock — not a port pre-check — is what rejects a duplicate. With an
        # incumbent holding the singleton lock, start() must return BEFORE ever
        # invoking _run() (which would open DB writer connections). This is the
        # invariant the deleted _run pre-check was redundantly asserting.
        import fcntl
        from servers import daemon_server as ds
        with tempfile.TemporaryDirectory(prefix="brain-lock-test-") as tmp:
            lock_path = os.path.join(tmp, "daemon.lock")
            incumbent = open(lock_path, "w")
            fcntl.flock(incumbent, fcntl.LOCK_EX | fcntl.LOCK_NB)  # hold it
            try:
                d = ds.BrainDaemon.__new__(ds.BrainDaemon)
                d._log = lambda *a, **k: None
                with patch("shutil.rmtree"), \
                     patch("servers.daemon_server.get_lock_path", return_value=lock_path), \
                     patch("servers.daemon_server.get_pid_path",
                           return_value=os.path.join(tmp, "daemon.pid")), \
                     patch.object(d, "_run") as run, \
                     patch.object(d, "_shutdown") as shutdown:
                    d.start()
                run.assert_not_called()       # flock rejected us → never reached _run
                shutdown.assert_not_called()  # early return, before the loop/_shutdown
            finally:
                fcntl.flock(incumbent, fcntl.LOCK_UN)
                incumbent.close()

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


class TestSupervisorPhaseScoping(unittest.TestCase):
    """Step 6: the supervisor is PHASE-scoped, not exception-type-scoped.
      • Crash while self.brain is None (before/during the brain load) → a brain-
        level fault. Retrying re-runs the same deterministic failure ×MAX, each
        paying the full load cost while holding the flock and serving nothing —
        so exit to KeepAlive for a fresh process (no in-place retry).
      • Crash with the brain up → a transient serve/socket fault. Warm-retry
        (keep the loaded brain) up to MAX, then give up.
      • The crash streak resets only after HEALTHY_UPTIME_RESET_S of serving, so
        an endless serve-crash loop actually reaches MAX (the old reset-on-every-
        bind let it loop forever) while unrelated crashes hours apart don't
        accumulate.
    Every path stays LOUD — _log_crash writes the traceback before any branch."""

    def _make(self, tmp):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__ — no pool/brain
        d._restart_count = 0
        d._run_started_at = 0
        d.socket_path = os.path.join(tmp, "x.sock")  # absent → start() skips unlink
        d.logs = []
        d._log = lambda m: d.logs.append(m)
        d._log_crash = MagicMock()
        d._shutdown = MagicMock()
        d._close_socket = MagicMock()
        return d

    def _start(self, d, tmp):
        # Run the REAL supervisor loop with only the flock/signal/pycache/sleep
        # machinery stubbed. The flock acquires cleanly (nothing else holds it).
        lock_path = os.path.join(tmp, "daemon.lock")
        with patch("shutil.rmtree"), \
             patch("servers.daemon_server.get_lock_path", return_value=lock_path), \
             patch("servers.daemon_server.signal.signal"), \
             patch("servers.daemon_server.atexit.register"), \
             patch("servers.daemon_server.time.sleep"):
            d.start()

    def test_load_crash_exits_to_keepalive_no_retry(self):
        with tempfile.TemporaryDirectory(prefix="brain-sup-test-") as tmp:
            d = self._make(tmp)
            d.brain = None                       # load-phase fault
            calls = []
            def boom():
                calls.append(1)
                raise RuntimeError("brain load failed")
            d._run = boom
            self._start(d, tmp)
            self.assertEqual(len(calls), 1, "load-phase crash must NOT warm-retry")
            self.assertEqual(d._restart_count, 0, "no restart counted on the exit-to-KeepAlive path")
            self.assertTrue(any("FATAL: crash before the brain loaded" in m for m in d.logs),
                            "must log the FATAL exit reason")
            d._log_crash.assert_called_once()    # LOUD: traceback before the branch
            d._shutdown.assert_called_once()     # clean teardown → flock released → KeepAlive respawns

    def test_serve_crash_warm_retries_then_gives_up(self):
        with tempfile.TemporaryDirectory(prefix="brain-sup-test-") as tmp:
            d = self._make(tmp)
            d.brain = MagicMock()                # brain up → serve-phase fault
            calls = []
            def boom():
                calls.append(1)                  # never sets _run_started_at (crashes "fast")
                raise RuntimeError("serve socket died")
            d._run = boom
            self._start(d, tmp)
            # 1 initial run + MAX warm retries, then FATAL give-up.
            self.assertEqual(len(calls), d.MAX_SUPERVISOR_RESTARTS + 1)
            self.assertTrue(any("Giving up" in m for m in d.logs))

    def test_healthy_uptime_resets_streak(self):
        with tempfile.TemporaryDirectory(prefix="brain-sup-test-") as tmp:
            d = self._make(tmp)
            d.brain = MagicMock()
            seq = []
            def run():
                seq.append(1)
                if len(seq) <= 3:
                    # each crash lands AFTER a full healthy-uptime interval
                    d._run_started_at = time.time() - (d.HEALTHY_UPTIME_RESET_S + 10)
                    raise RuntimeError("crash after healthy uptime")
                return  # 4th run: clean shutdown → loop breaks
            d._run = run
            self._start(d, tmp)
            # 3 healthy-uptime crashes each reset the streak to 0 (→ +1 = 1), so we
            # NEVER approach MAX(5) — they don't accumulate. 4th run returns clean.
            self.assertEqual(len(seq), 4)
            self.assertEqual(d._restart_count, 1)
            self.assertFalse(any("Giving up" in m for m in d.logs),
                             "healthy-uptime crashes must not accumulate toward give-up")


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


class TestPerformRestartLaunchdSoleSpawner(unittest.TestCase):
    """_perform_restart must never spawn a detached rival when launchd manages
    the daemon — that orphan (non-launchd, PPID 1) squats the singleton lock and
    wedges KeepAlive into a respawn storm, needing a manual kill (incidents
    2026-07-03/04). Managed → clean _shutdown() (drain→save→close DB→release lock
    LAST) then os._exit; KeepAlive respawns. Direct spawn ONLY on no-launchd."""

    def _make_daemon(self, order):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__ — no real brain/socket
        d.db_path = "/tmp/nonexistent-brain-test.db"
        d._log = lambda *a, **k: None
        # the ordered teardown: closes DB, releases lock LAST
        d._shutdown = MagicMock(side_effect=lambda: order.append("teardown"))
        return d

    def _run(self, managed):
        order = []
        d = self._make_daemon(order)
        # os._exit truly halts in production; SystemExit models that so control
        # can't fall through, and the if/else already makes the branches exclusive.
        with patch("time.sleep"), \
             patch("os._exit", side_effect=SystemExit) as m_exit, \
             patch("subprocess.Popen") as m_popen, \
             patch("shutil.rmtree"), \
             patch("servers.daemon_server.spawn_detached_daemon",
                   side_effect=lambda _db: order.append("spawn")) as m_spawn, \
             patch("servers.daemon_server.manages", return_value=managed):
            with self.assertRaises(SystemExit):
                d._perform_restart()
        return d, m_exit, m_spawn, m_popen, order

    def test_managed_clean_exit_never_popens(self):
        d, m_exit, m_spawn, m_popen, _ = self._run(managed=True)
        d._shutdown.assert_called_once()   # ordered teardown before exit (closes DB, releases lock LAST)
        m_spawn.assert_not_called()        # THE invariant — no detached rival; KeepAlive respawns
        m_popen.assert_not_called()        # …and no raw Popen sneaking around the primitive
        m_exit.assert_called_once_with(0)

    def test_no_launchd_direct_spawns_after_teardown(self):
        d, m_exit, m_spawn, m_popen, order = self._run(managed=False)
        d._shutdown.assert_called_once()
        m_spawn.assert_called_once_with(d.db_path)   # the ONE hardened spawn (daemon_launch)
        m_popen.assert_not_called()                  # no second, un-hardened Popen path
        # teardown FIRST — DB closed + lock released before the successor can
        # open brain.db (two writers corrupt the indexes).
        self.assertEqual(order, ["teardown", "spawn"])
        m_exit.assert_called_once_with(0)


class TestSpawnDetachedDaemon(unittest.TestCase):
    """spawn_detached_daemon is the ONE spawn, and it is the HARDENED one:
    debugger-friendly interpreter, devnull stdin, log redirect, own session,
    full CPU-only env. Step 2 unified the two drifted spawn sites (ensure_daemon
    fallback vs _perform_restart no-launchd — the latter ran raw sys.executable
    with inherited stdin) into this primitive."""

    def test_spawn_is_hardened(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="brain-spawn-test-") as tmp, \
             patch.object(dl.subprocess, "Popen") as popen, \
             patch.object(dl, "debugger_friendly_python", return_value="/usr/bin/python3"), \
             patch.dict(os.environ, {"BRAIN_DB_DIR": tmp}):
            dl.spawn_detached_daemon(os.path.join(tmp, "brain.db"))
        popen.assert_called_once()
        argv = popen.call_args[0][0]
        kwargs = popen.call_args[1]
        self.assertEqual(argv[0], "/usr/bin/python3")  # debugger-friendly, not raw sys.executable
        self.assertEqual(argv[1], "-c")
        self.assertIn("BrainDaemon", argv[2])
        self.assertTrue(kwargs["start_new_session"])
        self.assertIsNotNone(kwargs["stdin"])           # devnull, never inherited
        from servers.daemon_config import DAEMON_CPU_ENV
        self.assertTrue(set(DAEMON_CPU_ENV).issubset(kwargs["env"]),
                        "spawn must merge the full CPU-only env")

    def test_no_raw_popen_outside_daemon_launch(self):
        # Both spawn callers must route through spawn_detached_daemon — a direct
        # Popen in either module is the un-hardened-drift bug class coming back.
        import inspect
        import servers.daemon_server
        for mod in (dc, servers.daemon_server):
            self.assertNotIn(
                "Popen(", inspect.getsource(mod),
                "%s must spawn via daemon_launch.spawn_detached_daemon" % mod.__name__)


class TestKillDaemonLockDiscipline(unittest.TestCase):
    """Step 5: NO code path unlinks a lock file. kill_daemon relies on the
    kernel releasing the dead PID's flock; unlinking the path while another
    holder (a daemon, an ensure_daemon mid-ladder) has it open would let a
    third process lock a fresh inode at the same path — two "singleton"
    holders, the two-writer corruption class."""

    def test_kill_daemon_clears_pid_but_never_unlinks_lock(self):
        import inspect
        import tempfile
        self.assertNotIn("get_lock_path", inspect.getsource(dl.kill_daemon),
                         "kill_daemon must not touch the lock path at all")
        with tempfile.TemporaryDirectory(prefix="brain-kill-test-") as tmp:
            pid_path = os.path.join(tmp, "daemon.pid")
            with open(pid_path, "w") as f:
                f.write("999999")
            with patch.object(dl, "get_pid_path", return_value=pid_path), \
                 patch.object(dl.os, "kill", side_effect=ProcessLookupError), \
                 patch.object(dl.time, "sleep"):
                dl.kill_daemon()
            self.assertFalse(os.path.exists(pid_path),
                             "stale PID hint should still be cleared")


class TestDaemonConfigSingleSource(unittest.TestCase):
    """Step 1: the launch facts (CPU env, log path, launchd timing) are
    single-sourced in daemon_config so no start path drifts from another."""

    def test_cpu_env_is_the_full_set(self):
        from servers.daemon_config import DAEMON_CPU_ENV
        self.assertEqual(
            set(DAEMON_CPU_ENV),
            {"ORT_DISABLE_ALL_ACCELERATORS", "ONNX_PROVIDERS", "PYTORCH_MPS_DISABLE",
             "VECLIB_MAXIMUM_THREADS", "PYTORCH_ENABLE_MPS_FALLBACK"},
            "DAEMON_CPU_ENV must carry every var the spawn paths used to hand-list")

    def test_log_path_honors_brain_db_dir(self):
        # The divergence fix: ensure_daemon used to ignore BRAIN_DB_DIR, the
        # restart path honored it — now both go through get_daemon_log_path.
        from servers.daemon_config import get_daemon_log_path
        with patch.dict(os.environ, {"BRAIN_DB_DIR": "/tmp/brain-test-logdir"}):
            self.assertEqual(get_daemon_log_path("/other/place/brain.db"),
                             "/tmp/brain-test-logdir/daemon.log")

    def test_recovery_deadlines_derive_from_throttle(self):
        from servers import daemon_client as dc, daemon_config as cfg
        # Deadlines must clear the plist throttle + reload, or a recovering
        # daemon is mistaken for a corpse and re-kickstarted (the storm).
        self.assertGreater(dc._GRACE_DEADLINE_S, cfg.LAUNCHD_THROTTLE_INTERVAL_S)
        self.assertGreater(dc._KICKSTART_DEADLINE_S, dc._GRACE_DEADLINE_S)


class TestDaemonPlistTemplateContract(unittest.TestCase):
    """Step 7: the repo plist template (hooks/scripts/com.brain.daemon.plist,
    installed per-machine by install-daemon-service.sh) DERIVES its CPU env and
    launchd timing from the Step-1 daemon_config constants. Pins the equality so
    the XML can't silently drift from the Python again — the drift Step 1 named,
    where the hand-installed plist carried a stale 4-var CPU set missing
    PYTORCH_ENABLE_MPS_FALLBACK."""

    def _load(self):
        import plistlib
        path = os.path.join(os.path.dirname(__file__), '..',
                            'hooks', 'scripts', 'com.brain.daemon.plist')
        with open(path, 'rb') as f:
            raw = f.read()
        return plistlib.loads(raw), raw.decode()

    def test_plist_env_matches_daemon_cpu_env(self):
        from servers.daemon_config import DAEMON_CPU_ENV
        plist, _ = self._load()
        env = plist["EnvironmentVariables"]
        # The env is exactly DAEMON_CPU_ENV plus the per-machine BRAIN_DB_DIR path.
        self.assertEqual(set(env), set(DAEMON_CPU_ENV) | {"BRAIN_DB_DIR"},
                         "plist env must be DAEMON_CPU_ENV + BRAIN_DB_DIR, nothing more/less")
        for k, v in DAEMON_CPU_ENV.items():
            self.assertEqual(env.get(k), v,
                             "plist %s=%r must match DAEMON_CPU_ENV" % (k, v))

    def test_plist_throttle_matches_constant(self):
        from servers.daemon_config import LAUNCHD_THROTTLE_INTERVAL_S
        plist, _ = self._load()
        self.assertEqual(plist["ThrottleInterval"], LAUNCHD_THROTTLE_INTERVAL_S,
                         "plist ThrottleInterval must equal LAUNCHD_THROTTLE_INTERVAL_S")

    def test_plist_is_an_unresolved_template(self):
        # The repo copy is a TEMPLATE — install-daemon-service.sh sed-substitutes
        # the tokens. If a resolved (machine-specific) copy is ever committed, the
        # placeholders vanish and a fresh install materializes wrong paths.
        plist, raw = self._load()
        self.assertIn("__PLUGIN_DIR__", raw)
        self.assertIn("__BRAIN_DB_DIR__", raw)
        self.assertEqual(plist["ProgramArguments"],
                         ["__PLUGIN_DIR__/hooks/scripts/start-daemon.sh"],
                         "entrypoint must be start-daemon.sh under the plugin-dir token")
        self.assertTrue(plist["KeepAlive"])
        self.assertTrue(plist["RunAtLoad"])


class TestDbDirDivergence(unittest.TestCase):
    """Step 4b (D-13): the daemon ping reports its db_dir; ensure_daemon treats
    a mismatch vs the session's resolved dir like stale code — kickstart. The
    split-brain killer: after the user adopts a new brain location, a daemon
    launched off the old plist-baked path would keep writing the old brain
    forever (the old brain.db still exists per never-auto-move, so the baked
    path stays 'valid')."""

    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="brain-dbdir-test-")
        self._db = os.path.join(self._dir, "brain.db")
        self._lock = os.path.join(self._dir, "daemon.lock")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    def _resp(self, db_dir):
        r = {"ok": True, "result": {}}
        if db_dir is not None:
            r["result"]["db_dir"] = db_dir
        return r

    # ── _db_dir_changed unit ──

    def test_mismatch_detected(self):
        with patch.object(dc, "_is_daemon_source", return_value=True):
            self.assertTrue(dc._db_dir_changed(self._resp("/somewhere/else"), self._db))

    def test_match_is_not_changed(self):
        with patch.object(dc, "_is_daemon_source", return_value=True):
            self.assertFalse(dc._db_dir_changed(self._resp(self._dir), self._db))

    def test_missing_db_dir_never_changed(self):
        # A pre-step-4 daemon doesn't report db_dir → conservative no-restart
        # on this signal (the code-fingerprint check already covers it).
        with patch.object(dc, "_is_daemon_source", return_value=True):
            self.assertFalse(dc._db_dir_changed(self._resp(None), self._db))

    def test_non_source_never_changed(self):
        # Same policy as _code_changed: a non-source checkout can't converge a
        # restart, so it never claims divergence.
        with patch.object(dc, "_is_daemon_source", return_value=False):
            self.assertFalse(dc._db_dir_changed(self._resp("/somewhere/else"), self._db))

    def test_symlinked_same_dir_is_not_changed(self):
        # realpath comparison: a symlink to the same brain is not divergence.
        link = self._dir + "-link"
        os.symlink(self._dir, link)
        try:
            with patch.object(dc, "_is_daemon_source", return_value=True):
                self.assertFalse(dc._db_dir_changed(self._resp(link), self._db))
        finally:
            os.unlink(link)

    # ── ensure_daemon flow ──

    def test_db_dir_mismatch_kickstarts_once(self):
        # Healthy, current code, but writing ANOTHER brain → exactly one
        # launchd kickstart (never a spawn); post-kickstart the daemon reports
        # the session's dir (plist re-materialized at boot + start-daemon.sh
        # re-ran the ladder) → ready.
        stale = self._resp("/somewhere/else")
        fresh = self._resp(self._dir)
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_can_connect", return_value=stale), \
             patch.object(dc, "_await_responsive", return_value=fresh), \
             patch.object(dc, "kickstart", return_value=True) as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_called_once()
            spawn.assert_not_called()

    def test_db_dir_match_is_noop(self):
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_can_connect", return_value=self._resp(self._dir)), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_old_daemon_without_db_dir_is_noop(self):
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_can_connect", return_value=self._resp(None)), \
             patch.object(dc, "kickstart") as ks:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()

    def test_worktree_client_mismatch_never_kickstarts(self):
        # A non-source checkout stays a pure client even when it sees a
        # diverged daemon — the source checkout's boot converges it.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value=self._resp("/somewhere/else")), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()


class TestInstallerPlistDrift(unittest.TestCase):
    """Step 4c (D-13): the installers render the plist template to a temp file
    on EVERY run and diff against the installed copy — drift (template evolved,
    or the brain moved) re-materializes + re-bootstraps; no drift leaves launchd
    untouched. Runs the real shell scripts against a fake HOME + fake launchctl."""

    DAEMON_SCRIPT = os.path.join(os.path.dirname(__file__), '..',
                                 'hooks', 'scripts', 'install-daemon-service.sh')
    DASH_SCRIPT = os.path.join(os.path.dirname(__file__), '..',
                               'hooks', 'scripts', 'ensure-dashboard.sh')

    def setUp(self):
        if sys.platform != "darwin":
            self.skipTest("launchd installer paths are macOS-only")
        self._home = tempfile.mkdtemp(prefix="brain-plist-home-")
        self._dbdir = os.path.join(self._home, "brain-data")
        os.makedirs(self._dbdir)
        open(os.path.join(self._dbdir, "brain.db"), "w").close()
        self._agents = os.path.join(self._home, "Library", "LaunchAgents")
        os.makedirs(self._agents)
        self._bin = os.path.join(self._home, "bin")
        os.makedirs(self._bin)
        self._log = os.path.join(self._home, "launchctl.log")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._home, ignore_errors=True)

    def _fake_launchctl(self, print_rc=0):
        """launchctl stub: records every invocation and models load state like
        the real thing — `print` answers from a state file (seeded by print_rc:
        0 = service managed, 1 = fresh machine), `bootout` unloads, `bootstrap`
        loads. The installers verify unload-before-copy and load-after-
        bootstrap, so both transitions must be observable."""
        path = os.path.join(self._bin, "launchctl")
        state = os.path.join(self._home, "launchctl.loaded")
        if print_rc == 0:
            open(state, "w").close()
        elif os.path.exists(state):
            os.remove(state)
        with open(path, "w") as f:
            f.write('#!/bin/bash\necho "$@" >> "%s"\n'
                    'case "$1" in\n'
                    '  bootstrap) touch "%s";;\n'
                    '  bootout) rm -f "%s";;\n'
                    '  print) [ -f "%s" ] && exit 0; exit 1;;\n'
                    'esac\nexit 0\n'
                    % (self._log, state, state, state))
        os.chmod(path, 0o755)

    def _fake_curl(self, code="200"):
        """curl stub for ensure-dashboard's _up probe."""
        path = os.path.join(self._bin, "curl")
        with open(path, "w") as f:
            f.write('#!/bin/bash\nprintf "%s"\n' % code)
        os.chmod(path, 0o755)

    def _run(self, script):
        import subprocess
        env = dict(os.environ,
                   HOME=self._home,
                   PATH=self._bin + os.pathsep + os.environ.get("PATH", ""),
                   BRAIN_DB_DIR=self._dbdir,
                   XDG_CONFIG_HOME=os.path.join(self._home, ".config"))
        return subprocess.run(["bash", script], env=env,
                              capture_output=True, text=True, timeout=60)

    def _calls(self):
        if not os.path.exists(self._log):
            return []
        return [line.split() for line in open(self._log).read().splitlines()]

    def _verbs(self):
        return [c[0] for c in self._calls()]

    # ── install-daemon-service.sh ──

    def test_daemon_drift_rematerializes_and_rebootstraps(self):
        self._fake_launchctl(print_rc=0)  # managed
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        with open(target, "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(self._dbdir, content, "re-materialized plist must carry the resolved brain dir")
        self.assertNotIn("__BRAIN_DB_DIR__", content)
        self.assertNotIn("__PLUGIN_DIR__", content)
        verbs = self._verbs()
        self.assertIn("bootout", verbs, "drift must re-bootstrap, not kickstart (env is frozen at bootstrap)")
        self.assertIn("bootstrap", verbs)

    def test_daemon_no_drift_leaves_launchd_untouched(self):
        self._fake_launchctl(print_rc=0)
        with open(os.path.join(self._agents, "com.brain.daemon.plist"), "w") as f:
            f.write("seed")
        self._run(self.DAEMON_SCRIPT)  # first run converges the target
        os.remove(self._log)
        r = self._run(self.DAEMON_SCRIPT)  # second run: no drift
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(set(self._verbs()), {"print"},
                         "no drift → only the managed probe, no bootout/bootstrap")

    def test_daemon_fresh_install_bootstraps_without_bootout(self):
        self._fake_launchctl(print_rc=1)  # not managed
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        self.assertTrue(os.path.exists(target))
        self.assertIn(self._dbdir, open(target).read())
        verbs = self._verbs()
        self.assertIn("bootstrap", verbs)
        self.assertNotIn("bootout", verbs)

    # ── ensure-dashboard.sh ──

    def test_dashboard_drift_rematerializes_even_when_up(self):
        # An "up" dashboard whose plist points at a moved brain serves the
        # WRONG data — drift must be fixed before the up fast-path exits.
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        target = os.path.join(self._agents, "com.brain.dashboard.plist")
        with open(target, "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(self._dbdir, content)
        self.assertNotIn("__BRAIN_DB_DIR__", content)
        verbs = self._verbs()
        self.assertIn("bootout", verbs)
        self.assertIn("bootstrap", verbs)

    def test_dashboard_no_drift_up_is_noop(self):
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        target = os.path.join(self._agents, "com.brain.dashboard.plist")
        with open(target, "w") as f:
            f.write("seed")
        self._run(self.DASH_SCRIPT)  # converge
        os.remove(self._log)
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(set(self._verbs()), {"print"},
                         "up + no drift → early exit, launchd untouched")

    # ── identity preservation (review findings: tree-flip / db-dir hijack) ──

    def _fake_owner_tree(self):
        """A second 'checkout' that owns the installed plist — valid launcher
        scripts under a tree that is NOT the one running the installer."""
        tree = os.path.join(self._home, "other-tree")
        scripts = os.path.join(tree, "hooks", "scripts")
        os.makedirs(scripts)
        for name in ("start-daemon.sh", "brain-dashboard"):
            p = os.path.join(scripts, name)
            with open(p, "w") as f:
                f.write("#!/bin/bash\n")
            os.chmod(p, 0o755)
        return tree

    def test_daemon_drift_preserves_installed_plugin_dir(self):
        # Hooks run from the plugin copy while the daemon is launchd-pinned to
        # the repo: re-materialization must keep the INSTALLED tree, not flip
        # service ownership to whichever tree ran the installer.
        self._fake_launchctl(print_rc=0)
        owner = self._fake_owner_tree()
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        rendered = open(template).read().replace("__PLUGIN_DIR__", owner) \
                                        .replace("__BRAIN_DB_DIR__", self._dbdir)
        with open(target, "w") as f:
            f.write(rendered + "\n<!-- old template residue -->\n")  # force drift
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(owner + "/hooks/scripts/start-daemon.sh", content,
                      "installed tree must survive re-materialization")
        repo = os.path.realpath(os.path.join(os.path.dirname(self.DAEMON_SCRIPT), "..", ".."))
        self.assertNotIn(repo + "/hooks/scripts/start-daemon.sh", content,
                         "the caller's tree must not capture the service")

    def test_daemon_foreign_owned_current_plist_is_no_drift(self):
        # Same template shape, different (valid) owner tree → NOT drift.
        # Without this, plugin-copy boots and repo-side runs would ping-pong
        # the plist with a daemon bootout on every flip.
        self._fake_launchctl(print_rc=0)
        owner = self._fake_owner_tree()
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        with open(target, "w") as f:
            f.write(open(template).read().replace("__PLUGIN_DIR__", owner)
                                         .replace("__BRAIN_DB_DIR__", self._dbdir))
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(set(self._verbs()), {"print"},
                         "foreign-owned but current plist must not re-bootstrap")

    def test_daemon_env_override_does_not_repoint_db_dir(self):
        # An ephemeral BRAIN_DB_DIR (eval run, isolated copy) must not hijack
        # the managed daemon onto its brain: installed BRAIN_DB_DIR wins
        # unless the override came from a durable adoption channel.
        self._fake_launchctl(print_rc=0)
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        repo = os.path.realpath(os.path.join(os.path.dirname(self.DAEMON_SCRIPT), "..", ".."))
        with open(target, "w") as f:
            f.write(open(template).read().replace("__PLUGIN_DIR__", repo)
                                         .replace("__BRAIN_DB_DIR__", prod))
        # self._dbdir (the run's BRAIN_DB_DIR) differs from prod → would be a
        # hijack; no knob, no plugin option → render must keep prod.
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(prod, open(target).read(),
                      "ephemeral env override must not re-point the daemon's brain")
        self.assertEqual(set(self._verbs()), {"print"})

    def test_daemon_knob_adoption_repoints_db_dir(self):
        # The durable channel: BRAIN_DB_DIR in ~/.config/brain/env IS a
        # sanctioned adoption — the plist must converge to it.
        self._fake_launchctl(print_rc=0)
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        cfg = os.path.join(self._home, ".config", "brain")
        os.makedirs(cfg)
        with open(os.path.join(cfg, "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % self._dbdir)
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        repo = os.path.realpath(os.path.join(os.path.dirname(self.DAEMON_SCRIPT), "..", ".."))
        with open(target, "w") as f:
            f.write(open(template).read().replace("__PLUGIN_DIR__", repo)
                                         .replace("__BRAIN_DB_DIR__", prod))
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(self._dbdir, content, "knob adoption must re-point the plist")
        self.assertNotIn(prod, content)
        self.assertIn("bootout", self._verbs())

    def test_daemon_bootout_failure_keeps_installed_plist(self):
        # bootout that doesn't unload → the plist FILE must stay stale so the
        # next run re-detects drift ("file current, launchd stale" never heals).
        path = os.path.join(self._bin, "launchctl")
        with open(path, "w") as f:
            f.write('#!/bin/bash\necho "$@" >> "%s"\n'
                    'case "$1" in print) exit 0;; esac\nexit 0\n' % self._log)
        os.chmod(path, 0o755)  # print always managed; bootout never unloads
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        with open(target, "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        r = self._run(self.DAEMON_SCRIPT)
        self.assertNotEqual(r.returncode, 0, "must fail loudly, not bless the divergence")
        self.assertEqual(open(target).read(), "<plist>stale frozen snapshot</plist>\n",
                         "installed plist must be untouched after a failed unload")


class TestResolverEnvHintDemotion(unittest.TestCase):
    """Step 4a refinement (review finding): a BRAIN_DB_DIR env dir WITHOUT
    brain.db is a hint of last resort, not a verdict — a daemon relaunched off
    a stale plist-baked path follows the ladder to the moved brain instead of
    birthing a shadow brain, while a fresh install (dir exists, daemon creates
    brain.db) still lands on the hint."""

    RESOLVER = os.path.join(os.path.dirname(__file__), '..',
                            'hooks', 'scripts', 'resolve-brain-db.sh')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-resolve-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._home, ignore_errors=True)

    def _resolve(self, env_extra):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        env.pop("BRAIN_DB_DIR", None)
        env.pop("CLAUDE_PLUGIN_DATA", None)
        env.update(env_extra)
        out = subprocess.run(
            ["bash", "-c",
             'source "%s" >/dev/null 2>&1; printf %%s "$BRAIN_DB_DIR"' % self.RESOLVER],
            env=env, capture_output=True, text=True, timeout=60)
        return out.stdout.strip()

    def test_moved_brain_follows_resolved_env_over_stale_hint(self):
        stale = os.path.join(self._home, "old-brain")   # dir exists, no brain.db
        os.makedirs(stale)
        new = os.path.join(self._home, "new-brain")
        os.makedirs(new)
        open(os.path.join(new, "brain.db"), "w").close()
        with open(os.path.join(self._xdg, "brain", "resolved.env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % new)
        self.assertEqual(self._resolve({"BRAIN_DB_DIR": stale}), new,
                         "the record of where the brain moved must beat the stale baked hint")

    def test_fresh_install_hint_survives(self):
        hint = os.path.join(self._home, "fresh-brain")  # installer mkdir'd it
        os.makedirs(hint)
        self.assertEqual(self._resolve({"BRAIN_DB_DIR": hint}), hint,
                         "fresh install: nothing else exists → the baked hint wins")

    def test_hint_with_brain_db_is_adopted_outright(self):
        d = os.path.join(self._home, "live-brain")
        os.makedirs(d)
        open(os.path.join(d, "brain.db"), "w").close()
        self.assertEqual(self._resolve({"BRAIN_DB_DIR": d}), d)

    def test_no_persist_guard_skips_resolved_env_write(self):
        d = os.path.join(self._home, "live-brain")
        os.makedirs(d)
        open(os.path.join(d, "brain.db"), "w").close()
        self._resolve({"BRAIN_DB_DIR": d, "BRAIN_RESOLVE_NO_PERSIST": "1"})
        self.assertFalse(
            os.path.exists(os.path.join(self._xdg, "brain", "resolved.env")),
            "a NO_PERSIST consumer (start-daemon.sh) must not write the record")


class TestDaemonPortEnvFirst(unittest.TestCase):
    """Step 2 (D-13 family): BRAIN_DAEMON_PORT is a real contract — the
    daemon's own DAEMON_PORT reads the env override first, exactly like every
    shell/hook client, so setting it can't split clients and daemon across
    two ports."""

    _REPO = os.path.join(os.path.dirname(__file__), '..')

    def setUp(self):
        # Sandbox XDG so the real ~/.config/brain/env can't leak into asserts.
        self._xdg = tempfile.mkdtemp(prefix="brain-port-xdg-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._xdg, ignore_errors=True)

    def _port(self, env_extra):
        import subprocess
        env = {k: v for k, v in os.environ.items() if k != "BRAIN_DAEMON_PORT"}
        env["XDG_CONFIG_HOME"] = self._xdg
        env.update(env_extra)
        out = subprocess.run(
            [sys.executable, "-c",
             "from servers.daemon_config import DAEMON_PORT; print(DAEMON_PORT)"],
            env=env, capture_output=True, text=True, cwd=self._REPO, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout.strip().splitlines()[-1]

    def test_env_override_wins(self):
        self.assertEqual(self._port({"BRAIN_DAEMON_PORT": "47299"}), "47299")

    def test_formula_when_unset(self):
        self.assertEqual(self._port({}), str(47200 + os.getuid() % 100))

    def test_env_file_fallback_covers_bare_env_processes(self):
        # The MCP server is spawned by CC without brain-env.sh — it must read
        # the same knob from the user env file or the daemon (which sources it)
        # binds one port while the MCP health monitor pings another, and
        # recover_daemon kickstart-storms a healthy daemon forever.
        os.makedirs(os.path.join(self._xdg, "brain"))
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("export BRAIN_DAEMON_PORT=47288\n")
        self.assertEqual(self._port({}), "47288")

    def test_malformed_value_warns_and_falls_back(self):
        # A garbage knob must not crash-loop the daemon under KeepAlive.
        self.assertEqual(self._port({"BRAIN_DAEMON_PORT": "auto"}),
                         str(47200 + os.getuid() % 100))


if __name__ == "__main__":
    unittest.main()
