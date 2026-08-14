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
import shutil
import socket
import tempfile
import threading
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
        # Step 6c: `-m servers.daemon_server <db>`, the same command
        # hooks/scripts/brain-daemon execs. The DB path is an argv element,
        # never interpolated into Python source.
        self.assertEqual(argv[1:3], ["-m", "servers.daemon_server"])
        self.assertEqual(argv[3], os.path.join(tmp, "brain.db"))
        self.assertEqual(len(argv), 4)
        self.assertTrue(kwargs["start_new_session"])
        self.assertIsNotNone(kwargs["stdin"])           # devnull, never inherited
        from servers.daemon_config import DAEMON_CPU_ENV, REPO_ROOT
        self.assertTrue(set(DAEMON_CPU_ENV).issubset(kwargs["env"]),
                        "spawn must merge the full CPU-only env")
        # The two pins that moved out of the old `-c` string and into the env,
        # where they apply before exec instead of after interpreter start.
        self.assertEqual(kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], REPO_ROOT,
                         "`-m` must resolve the package whatever the spawner's cwd is")
        self.assertEqual(kwargs["env"]["BRAIN_DB_DIR"], tmp)

    def test_daemon_env_prepends_inherited_pythonpath(self):
        # Replacing an inherited PYTHONPATH would break a caller that put
        # something on it deliberately (eval harnesses, isolated brains).
        with patch.dict(os.environ, {"PYTHONPATH": "/somewhere/else"}):
            env = dl.daemon_env("/tmp/x/brain.db")
        from servers.daemon_config import REPO_ROOT
        self.assertEqual(env["PYTHONPATH"],
                         REPO_ROOT + os.pathsep + "/somewhere/else")

    def test_daemon_env_falls_back_to_db_parent(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAIN_DB_DIR", None)
            env = dl.daemon_env("/tmp/some-brain/brain.db")
        self.assertEqual(env["BRAIN_DB_DIR"], "/tmp/some-brain")

    def test_entry_point_rejects_bad_argv(self):
        # A daemon that resolved its own DB would be a second resolver; the
        # entry point takes exactly one path and refuses anything else.
        import servers.daemon_server as ds
        self.assertEqual(ds.main([]), 2)
        self.assertEqual(ds.main([""]), 2)
        self.assertEqual(ds.main(["a", "b"]), 2)

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
                         ["__PLUGIN_DIR__/hooks/scripts/brain-daemon"],
                         "entrypoint must be brain-daemon under the plugin-dir token")
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
        # the session's dir (plist re-materialized at boot + brain-daemon
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
        for name in ("brain-daemon", "brain-dashboard"):
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
        self.assertIn(owner + "/hooks/scripts/brain-daemon", content,
                      "installed tree must survive re-materialization")
        repo = os.path.realpath(os.path.join(os.path.dirname(self.DAEMON_SCRIPT), "..", ".."))
        self.assertNotIn(repo + "/hooks/scripts/brain-daemon", content,
                         "the caller's tree must not capture the service")

    def test_daemon_legacy_launcher_name_still_preserves_tree(self):
        # An installed plist from BEFORE the start-daemon.sh → brain-daemon
        # rename: extraction is launcher-name-agnostic, validity checks the
        # launcher the CURRENT template ships — so the old tree is preserved
        # and the re-materialized plist execs the new launcher in it.
        self._fake_launchctl(print_rc=0)
        owner = self._fake_owner_tree()  # has brain-daemon, not start-daemon.sh
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        legacy = open(template).read() \
            .replace("__PLUGIN_DIR__/hooks/scripts/brain-daemon",
                     "__PLUGIN_DIR__/hooks/scripts/start-daemon.sh") \
            .replace("__PLUGIN_DIR__", owner) \
            .replace("__BRAIN_DB_DIR__", self._dbdir)
        with open(target, "w") as f:
            f.write(legacy)
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(owner + "/hooks/scripts/brain-daemon", content,
                      "old tree preserved, new launcher name materialized")
        self.assertNotIn("start-daemon.sh", content)
        self.assertIn("bootout", self._verbs())

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

    # The identity guards below existed for the daemon only — the asymmetry
    # step 6b removed by giving both installers one implementation
    # (launchd-install.sh). They are pinned on the DASHBOARD side because that
    # is the copy that had drifted.

    def _install_dashboard_plist(self, db_dir):
        target = os.path.join(self._agents, "com.brain.dashboard.plist")
        template = os.path.join(os.path.dirname(self.DASH_SCRIPT),
                                "com.brain.dashboard.plist")
        repo = os.path.realpath(os.path.join(os.path.dirname(self.DASH_SCRIPT), "..", ".."))
        with open(target, "w") as f:
            f.write(open(template).read().replace("__PLUGIN_DIR__", repo)
                                         .replace("__BRAIN_DB_DIR__", db_dir))
        return target

    def test_dashboard_env_override_does_not_repoint_db_dir(self):
        # An ephemeral BRAIN_DB_DIR must not hijack the singleton dashboard
        # onto a temp brain any more than it may hijack the daemon.
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        target = self._install_dashboard_plist(prod)
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(prod, open(target).read(),
                      "ephemeral env override must not re-point the dashboard's brain")
        self.assertEqual(set(self._verbs()), {"print"})

    def test_dashboard_knob_adoption_repoints_db_dir(self):
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        cfg = os.path.join(self._home, ".config", "brain")
        os.makedirs(cfg, exist_ok=True)
        with open(os.path.join(cfg, "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % self._dbdir)
        target = self._install_dashboard_plist(prod)
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(self._dbdir, content, "knob adoption must re-point the plist")
        self.assertNotIn(prod, content)
        self.assertIn("bootout", self._verbs())

    def test_knob_read_ignores_env_file_stdout(self):
        # The env file is shell grammar, so a user `echo`/banner line in it
        # writes to the reader's stdout. The dashboard's copy of the knob read
        # lacked the stdout discard until step 6b unified it: the polluted
        # value then failed the "is this a durable adoption?" comparison and
        # the installer silently kept the OLD brain despite a valid knob.
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        cfg = os.path.join(self._home, ".config", "brain")
        os.makedirs(cfg, exist_ok=True)
        with open(os.path.join(cfg, "env"), "w") as f:
            f.write("echo 'loading brain env'\nBRAIN_DB_DIR='%s'\n" % self._dbdir)
        target = self._install_dashboard_plist(prod)
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(self._dbdir, open(target).read(),
                      "a chatty env file must not defeat knob adoption")

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
            "a NO_PERSIST consumer (brain-daemon) must not write the record")


class TestResolverMkdirsSafeUnderSetE(unittest.TestCase):
    """The resolver is sourced under the daemon launcher's `set -e`, and its
    mkdirs are best-effort: a read-only parent must degrade to the next rung
    (or to the existing no-brain path), never abort the whole resolver — that
    would take the daemon down with it. Covers the persist-state dir and the
    plugin-option hint; the Cowork-mount and config-hint mkdirs share the same
    `|| true` shape but need /sessions or rung isolation to exercise."""

    RESOLVER = os.path.join(os.path.dirname(__file__), '..',
                            'hooks', 'scripts', 'resolve-brain-db.sh')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-resolve-home-")
        self._xdg = os.path.join(self._home, ".config")
        self._ro = []

    def tearDown(self):
        import shutil
        for d in self._ro:
            os.chmod(d, 0o755)
        shutil.rmtree(self._home, ignore_errors=True)

    def _read_only(self, path):
        os.chmod(path, 0o555)
        self._ro.append(path)

    def _shells(self):
        # No dash here: the resolver's `source` keyword at its brain-env.sh
        # include is a bashism dash lacks, so dash + set -e dies at that line
        # before any mkdir — a separate, pre-existing limitation. The set -e
        # daemon path runs bash; zsh covers the other hook shell.
        for shell in ("bash", "zsh"):
            if shutil.which(shell):
                yield shell

    def _source_under_set_e(self, shell, env_extra):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("BRAIN_DB_DIR", "CLAUDE_PLUGIN_DATA",
                  "CLAUDE_PLUGIN_OPTION_BRAIN_PATH",
                  "CLAUDE_PLUGIN_OPTION_brain_path"):
            env.pop(k, None)
        env.update(env_extra)
        return subprocess.run(
            [shell, "-c",
             'set -e\n. "%s" >/dev/null 2>&1\nprintf %%s "ok:$BRAIN_DB_DIR"'
             % self.RESOLVER],
            env=env, capture_output=True, text=True, timeout=60)

    def test_unwritable_persist_dir_does_not_abort(self):
        # _brain_persist_state: ~/.config exists read-only and ~/.config/brain
        # does not — its mkdir -p fails. The brain itself is healthy; only the
        # record write must be skipped.
        live = os.path.join(self._home, "live-brain")
        os.makedirs(live)
        open(os.path.join(live, "brain.db"), "w").close()
        os.makedirs(self._xdg)
        self._read_only(self._xdg)
        for shell in self._shells():
            with self.subTest(shell=shell):
                r = self._source_under_set_e(shell, {"BRAIN_DB_DIR": live})
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(r.stdout, "ok:%s" % live, r.stderr)

    def test_uncreatable_plugin_option_hint_does_not_abort(self):
        # The plugin-option hint points under a read-only parent — its
        # selection-time mkdir fails; the resolution chain must still run.
        ro_parent = os.path.join(self._home, "ro")
        os.makedirs(ro_parent)
        self._read_only(ro_parent)
        hint = os.path.join(ro_parent, "brain")
        for shell in self._shells():
            with self.subTest(shell=shell):
                r = self._source_under_set_e(
                    shell, {"CLAUDE_PLUGIN_OPTION_BRAIN_PATH": hint,
                            "BRAIN_RESOLVE_NO_PERSIST": "1"})
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(r.stdout.startswith("ok:"), r.stderr)


class TestAdoptionNetAndXdgCreate(unittest.TestCase):
    """5.0a: new brains are born at the XDG service dir (D-13), and the
    create branch refuses to run while an existing brain sits unreachable at
    an old host-owned default (plugin renamed/reinstalled) — it names the
    candidate for boot to surface instead. An explicit choice (the
    ~/.config/brain/env knob) always beats the net."""

    RESOLVER = os.path.join(os.path.dirname(__file__), '..',
                            'hooks', 'scripts', 'resolve-brain-db.sh')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-resolve-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        self._native = os.path.join(self._home, ".local", "share", "brain")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._home, ignore_errors=True)

    def _resolve(self, env_extra=None):
        """Returns (BRAIN_DB_DIR, BRAIN_ADOPTION_CANDIDATE) after sourcing."""
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("BRAIN_DB_DIR", "CLAUDE_PLUGIN_DATA", "XDG_DATA_HOME",
                  "BRAIN_ADOPTION_CANDIDATE"):
            env.pop(k, None)
        env.update(env_extra or {})
        out = subprocess.run(
            ["bash", "-c",
             'source "%s" >/dev/null 2>&1; '
             'printf "%%s\\n%%s" "$BRAIN_DB_DIR" "$BRAIN_ADOPTION_CANDIDATE"'
             % self.RESOLVER],
            env=env, capture_output=True, text=True, timeout=60)
        parts = out.stdout.split("\n")
        return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")

    def _make_brain(self, *segments):
        d = os.path.join(self._home, *segments)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "brain.db"), "w").close()
        return d

    def test_fresh_install_creates_at_xdg(self):
        db_dir, cand = self._resolve()
        self.assertEqual(db_dir, self._native,
                         "nothing anywhere: a new brain is born at the XDG service dir")
        self.assertEqual(cand, "")
        self.assertTrue(os.path.isdir(self._native), "create branch mkdirs the dir")

    def test_xdg_existing_brain_adopted_over_legacy(self):
        native = self._make_brain(".local", "share", "brain")
        self._make_brain("AgentsContext", "brain")
        db_dir, _ = self._resolve()
        self.assertEqual(db_dir, native)

    def test_candidate_at_plugin_data_blocks_create(self):
        cand_dir = self._make_brain(".claude", "plugins", "data",
                                    "old-plugin-name", "brain")
        db_dir, cand = self._resolve()
        self.assertEqual(db_dir, "",
                         "the net must refuse to create while a candidate exists")
        self.assertEqual(cand, cand_dir)
        self.assertFalse(os.path.exists(self._native),
                         "refusal means nothing is created at the XDG dir")

    def test_candidate_via_cpd_sibling_scan(self):
        # rename case with a NON-default plugin-data root: only the live
        # $CLAUDE_PLUGIN_DATA sibling scan can find the old plugin's brain
        cand_dir = self._make_brain("custom-root", "data", "old-name", "brain")
        cpd = os.path.join(self._home, "custom-root", "data", "new-name")
        os.makedirs(cpd, exist_ok=True)
        db_dir, cand = self._resolve({"CLAUDE_PLUGIN_DATA": cpd})
        self.assertEqual(db_dir, "")
        self.assertEqual(cand, cand_dir)

    def test_config_knob_with_brain_adopted(self):
        d = self._make_brain("my-brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % d)
        db_dir, _ = self._resolve()
        self.assertEqual(db_dir, d)

    def test_config_knob_beats_adoption_net(self):
        # the user's explicit fresh-start choice defeats the repeating notice
        self._make_brain(".claude", "plugins", "data", "old-plugin", "brain")
        fresh = os.path.join(self._home, "chosen-fresh")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % fresh)
        db_dir, cand = self._resolve()
        self.assertEqual(db_dir, fresh,
                         "an explicit knob is a choice — the net must not fire")
        self.assertEqual(cand, "")
        self.assertTrue(os.path.isdir(fresh), "knob dir is honored (mkdir'd)")

    def test_config_knob_beats_stale_env_hint(self):
        stale = os.path.join(self._home, "stale-baked")
        os.makedirs(stale)
        knob = os.path.join(self._home, "knob-choice")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % knob)
        db_dir, _ = self._resolve({"BRAIN_DB_DIR": stale})
        self.assertEqual(db_dir, knob,
                         "durable knob beats a stale plist-baked empty dir")

    def test_cpd_existing_brain_still_adopted(self):
        # pre-D-13 install: brain at the live $CLAUDE_PLUGIN_DATA — adopted,
        # no net, no XDG create
        cpd = os.path.join(self._home, ".claude", "plugins", "data", "brain-x")
        d = self._make_brain(".claude", "plugins", "data", "brain-x", "brain")
        db_dir, cand = self._resolve({"CLAUDE_PLUGIN_DATA": cpd})
        self.assertEqual(db_dir, d)
        self.assertEqual(cand, "")

    # -- review-finding regressions (2026-08-12 multi-lens pass) --

    def test_knob_file_stdout_does_not_pollute_value(self):
        # a user `echo` in the sourced config file must not prepend to the
        # captured knob value
        d = self._make_brain("my-brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write('echo "loading brain env"\nBRAIN_DB_DIR=\'%s\'\n' % d)
        db_dir, _ = self._resolve()
        self.assertEqual(db_dir, d)

    def test_quoted_tilde_knob_is_expanded(self):
        # BRAIN_DB_DIR="~/x" (quoted, so the shell won't expand it) — the
        # resolver expands the ~/ prefix like the Python reader does
        d = self._make_brain("tilde-brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write('BRAIN_DB_DIR="~/tilde-brain"\n')
        db_dir, _ = self._resolve()
        self.assertEqual(db_dir, d)

    def test_relative_knob_is_ignored(self):
        # a relative path would resolve against each consumer's cwd — ignore
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR=relative-dir\n")
        db_dir, cand = self._resolve()
        self.assertEqual(db_dir, self._native)
        self.assertEqual(cand, "")

    def test_plugin_data_without_brains_does_not_block_create(self):
        # populated plugins/data with no brain.db anywhere: the scan must
        # come up empty (no glob abort, no false candidate) and create at XDG
        os.makedirs(os.path.join(self._home, ".claude", "plugins", "data",
                                 "some-other-plugin"), exist_ok=True)
        db_dir, cand = self._resolve()
        self.assertEqual(db_dir, self._native)
        self.assertEqual(cand, "")

    def test_resolver_survives_zsh_sourcing(self):
        # zsh nomatch treats a no-match glob in a sourced file as fatal —
        # the net's scan must not be a glob; whole ladder must complete
        import subprocess
        os.makedirs(os.path.join(self._home, ".claude", "plugins", "data",
                                 "some-other-plugin"), exist_ok=True)
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("BRAIN_DB_DIR", "CLAUDE_PLUGIN_DATA", "XDG_DATA_HOME",
                  "BRAIN_ADOPTION_CANDIDATE"):
            env.pop(k, None)
        out = subprocess.run(
            ["zsh", "-c",
             'source "%s" >/dev/null 2>&1; printf %%s "$BRAIN_DB_DIR"'
             % self.RESOLVER],
            env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), self._native)

    # -- the 2026-08-08 rename sandbox matrix, automated --

    def test_rename_with_current_record_rescues_silently(self):
        # old plugin's brain + resolved.env pointing at it → 4b rescue, no net
        old = self._make_brain(".claude", "plugins", "data", "old-name", "brain")
        with open(os.path.join(self._xdg, "brain", "resolved.env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % old)
        cpd = os.path.join(self._home, ".claude", "plugins", "data", "new-name")
        os.makedirs(cpd, exist_ok=True)
        db_dir, cand = self._resolve({"CLAUDE_PLUGIN_DATA": cpd})
        self.assertEqual(db_dir, old)
        self.assertEqual(cand, "")

    def test_rename_with_stale_record_hits_the_net(self):
        # resolved.env points at a deleted dir (the uninstall shape) — the
        # 2026-08-08 silent-fresh-brain case; now the net refuses instead
        old = self._make_brain(".claude", "plugins", "data", "old-name", "brain")
        with open(os.path.join(self._xdg, "brain", "resolved.env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % os.path.join(self._home, "deleted"))
        db_dir, cand = self._resolve()
        self.assertEqual(db_dir, "")
        self.assertEqual(cand, old)


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



class TestApiKeyEnvHelper(unittest.TestCase):
    """Step 6a: api-key-env.sh is the ONE definition of where
    ANTHROPIC_API_KEY comes from — boot-brain.sh and brain-env.sh both source
    it instead of carrying the copy that made a casing fix land on one side
    only (the 2026-07-15 keyless-daemon failure).

    The helper is sourced transitively by brain-daemon, which runs `set -e`,
    and by resolvers that zsh also sources — so the shape is pinned here too.
    """

    HELPER = os.path.join(os.path.dirname(__file__), '..',
                          'hooks', 'scripts', 'api-key-env.sh')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-apikey-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._home, ignore_errors=True)

    def _write_env_file(self, body):
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write(body)

    def _run(self, script, env_extra=None, shell="bash"):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("CLAUDE_PLUGIN_OPTION_API_KEY", None)
        env.pop("CLAUDE_PLUGIN_OPTION_api_key", None)
        env.update(env_extra or {})
        return subprocess.run(
            [shell, "-c", '. "%s"\n%s' % (self.HELPER, script)],
            env=env, capture_output=True, text=True, timeout=30)

    def _key(self, env_extra=None, shell="bash", prelude=""):
        r = self._run(
            '%sbrain_source_user_env\nbrain_api_key_from_plugin_option\n'
            'printf %%s "${ANTHROPIC_API_KEY:-}"' % prelude,
            env_extra, shell)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_env_file_supplies_key(self):
        self._write_env_file("ANTHROPIC_API_KEY=sk-from-file\n")
        self.assertEqual(self._key(), "sk-from-file")

    def test_plugin_option_fills_when_env_file_silent(self):
        self.assertEqual(
            self._key({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-upper"}), "sk-upper")

    def test_plugin_option_lowercase_casing_also_works(self):
        # The plugins-reference does not pin <KEY>'s case; a one-sided fix here
        # is the exact drift this extraction exists to prevent.
        self.assertEqual(
            self._key({"CLAUDE_PLUGIN_OPTION_api_key": "sk-lower"}), "sk-lower")

    def test_env_file_beats_plugin_option(self):
        self._write_env_file("ANTHROPIC_API_KEY=sk-from-file\n")
        self.assertEqual(
            self._key({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-option"}),
            "sk-from-file", "the file is the durable channel — it must win")

    def test_shell_export_beats_plugin_option(self):
        self.assertEqual(
            self._key({"ANTHROPIC_API_KEY": "sk-shell",
                       "CLAUDE_PLUGIN_OPTION_API_KEY": "sk-option"}),
            "sk-shell")

    def test_source_exports_every_var_not_just_the_key(self):
        # brain-env.sh depends on this: identity tokens in the file must reach
        # the daemon, so the source is `set -a`, not a key-only read.
        self._write_env_file("BRAIN_OPERATOR_NAME=tom\n")
        r = self._run('brain_source_user_env\n'
                      'bash -c \'printf %s "$BRAIN_OPERATOR_NAME"\'')
        self.assertEqual(r.stdout, "tom", r.stderr)

    def test_missing_env_file_is_not_an_error(self):
        r = self._run('brain_source_user_env; printf %s "$?"')
        self.assertEqual(r.stdout, "0", r.stderr)

    def test_safe_under_set_e(self):
        # brain-daemon runs `set -e` and sources this transitively: a bare
        # failing test as a standalone command would abort the whole resolver.
        r = self._run('set -e\nbrain_source_user_env\n'
                      'brain_api_key_from_plugin_option\nprintf ok')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "ok")

    def test_safe_under_zsh(self):
        if not shutil.which("zsh"):
            self.skipTest("zsh not available")
        self.assertEqual(
            self._key({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-zsh"}, shell="zsh"),
            "sk-zsh")

    def test_no_trailing_newline_env_file_still_reads(self):
        # Hand-edited files without a trailing newline exist in production
        # (2026-08-12 finding) — `.`-sourcing must still see the last line.
        self._write_env_file("ANTHROPIC_API_KEY=sk-no-newline")
        self.assertEqual(self._key(), "sk-no-newline")

    def test_failing_command_in_env_file_survives_set_e(self):
        # A failing command in the hand-edited file (FOO=$(typo-cmd)) used to
        # crash-loop the daemon: on macOS bash 3.2 errexit stays ACTIVE inside
        # a sourced file even when the `.` call is guarded with `|| true`
        # (verified 2026-08-13). The owner must drop errexit around the source
        # so the good variables on either side of the bad line still load.
        self._write_env_file(
            "BRAIN_BEFORE=loaded-before\n"
            "BRAIN_BROKEN=$(definitely-not-a-command-xyz)\n"
            "BRAIN_AFTER=loaded-after\n")
        for shell in ("bash", "zsh", "dash"):
            if not shutil.which(shell):
                continue
            with self.subTest(shell=shell):
                r = self._run(
                    'set -e\nbrain_source_user_env\n'
                    'printf %s "${BRAIN_BEFORE:-}:${BRAIN_AFTER:-}"',
                    shell=shell)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(r.stdout, "loaded-before:loaded-after",
                                 r.stderr)

    def test_errexit_restored_after_source(self):
        # The errexit drop is scoped to the source itself: a caller that had
        # `set -e` on must get it back, or the daemon launcher's own failure
        # discipline silently degrades after the first env-file read.
        self._write_env_file("BRAIN_OK=1\n")
        r = self._run('set -e\nbrain_source_user_env\nfalse\nprintf leaked')
        self.assertNotEqual(r.returncode, 0,
                            "errexit was not restored after the source")
        self.assertEqual(r.stdout, "")

    def test_source_never_enables_errexit(self):
        # Hooks run WITHOUT set -e; the restore must be conditional on the
        # caller's prior state, not an unconditional `set -e`.
        self._write_env_file("BRAIN_OK=1\n")
        r = self._run('brain_source_user_env\nfalse\nprintf ok')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "ok")


class TestBootKeyMirror(unittest.TestCase):
    """Step 6a: boot-brain.sh mirrors a userConfig-resolved key into the env
    file, and ONLY a userConfig-resolved one.

    CLAUDE_PLUGIN_OPTION_* exists only inside hook executions, so without the
    mirror a user who fills the plugin's key field still runs a keyless daemon
    (launchd spawns it in a separate process tree that resolves the key from
    the env file alone — first laptop install, 2026-07-15). A key that came
    from the shell or the file itself must NOT be written back: the file is
    already the daemon's channel, and rewriting it would persist an ephemeral
    eval/session key into the user's durable config.

    Runs the real boot-brain.sh on its cold-install branch (no .runtime-ready),
    where key resolution has already happened and the provisioning chain is
    stubbed out.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-bootkey-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        # A plugin tree carrying the real boot path + stubs for everything the
        # cold branch would otherwise launch for real (runtime bootstrap...).
        self._tree = os.path.join(self._home, "plugin")
        self._sd = os.path.join(self._tree, "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("boot-brain.sh", "api-key-env.sh", "runtime-state.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name),
                        os.path.join(self._sd, name))
        for name in ("ensure-runtime.sh", "install-daemon-service.sh",
                     "ensure-dashboard.sh"):
            stub = os.path.join(self._sd, name)
            with open(stub, "w") as f:
                f.write("#!/bin/bash\nexit 0\n")
            os.chmod(stub, 0o755)

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    @property
    def _env_file(self):
        return os.path.join(self._xdg, "brain", "env")

    def _boot(self, env_extra):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("ANTHROPIC_API_KEY", "CLAUDE_PLUGIN_OPTION_API_KEY",
                  "CLAUDE_PLUGIN_OPTION_api_key"):
            env.pop(k, None)
        env.update(env_extra)
        return subprocess.run(
            ["bash", os.path.join(self._sd, "boot-brain.sh")],
            input="{}", env=env, capture_output=True, text=True, timeout=60)

    def _file_body(self):
        if not os.path.exists(self._env_file):
            return None
        with open(self._env_file) as f:
            return f.read()

    def test_plugin_option_key_is_mirrored(self):
        r = self._boot({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-ant-mirrored"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ANTHROPIC_API_KEY=sk-ant-mirrored", self._file_body() or "",
                      "a userConfig key must reach the launchd-spawned daemon")
        self.assertEqual(os.stat(self._env_file).st_mode & 0o777, 0o600)

    def test_lowercase_plugin_option_key_is_mirrored(self):
        self._boot({"CLAUDE_PLUGIN_OPTION_api_key": "sk-ant-lower"})
        self.assertIn("ANTHROPIC_API_KEY=sk-ant-lower", self._file_body() or "")

    def test_shell_key_is_not_mirrored(self):
        # An ephemeral shell key (eval run, isolated copy) must not be written
        # into the user's durable config.
        self._boot({"ANTHROPIC_API_KEY": "sk-ant-ephemeral"})
        self.assertIsNone(self._file_body(),
                          "a shell-supplied key must never be persisted")

    def test_existing_key_line_is_never_overwritten(self):
        with open(self._env_file, "w") as f:
            f.write("ANTHROPIC_API_KEY=sk-ant-original\n")
        self._boot({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-ant-newer"})
        self.assertEqual(self._file_body(), "ANTHROPIC_API_KEY=sk-ant-original\n")

    def test_mirror_does_not_glue_onto_a_file_without_a_trailing_newline(self):
        # Hand-edited config files with no trailing newline exist in
        # production. A bare `>>` would produce
        # `BRAIN_AGENT_NAME=xxxANTHROPIC_API_KEY=sk-...`, corrupting both lines.
        with open(self._env_file, "w") as f:
            f.write("BRAIN_AGENT_NAME=anchor")   # no trailing \n
        self._boot({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-ant-appended"})
        body = self._file_body()
        self.assertIn("\nANTHROPIC_API_KEY=sk-ant-appended\n", body)
        self.assertNotIn("anchorANTHROPIC_API_KEY", body)
        # and the file still parses as shell, which is how everything reads it
        import subprocess
        r = subprocess.run(
            ["bash", "-c", '. "%s"; printf "%%s:%%s" "$BRAIN_AGENT_NAME" "$ANTHROPIC_API_KEY"'
             % self._env_file],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(r.stdout, "anchor:sk-ant-appended")

    def test_non_sk_plugin_option_is_not_mirrored(self):
        self._boot({"CLAUDE_PLUGIN_OPTION_API_KEY": "not-a-key"})
        self.assertIsNone(self._file_body())


class TestBootImportsPluginTreeNotCwd(unittest.TestCase):
    """boot-brain.sh's `python3 -c` sites must import the PLUGIN tree's
    servers/ package, never one sitting in the hook's cwd.

    `python3 -c` puts the cwd at sys.path[0], AHEAD of PYTHONPATH — so a user
    project carrying its own top-level `servers/` package shadowed the plugin
    tree at the ensure_daemon and MCP-verify calls: confusing tracebacks and a
    false "MCP SERVER BROKEN" notice every session booted from that project.
    restart-daemon.sh already pins cwd with a `cd "$PLUGIN_ROOT"`; these tests
    hold boot-brain.sh to the same contract.

    Runs the real boot-brain.sh warm path (runtime-ready sentinel + resolved
    brain) in a fake plugin tree whose servers/ package is a minimal stub —
    the invariant under test is import resolution order, not daemon behavior.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-bootcwd-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))

        # Fake plugin tree: real boot chain, stubs for everything it launches.
        self._tree = os.path.join(self._home, "plugin")
        self._sd = os.path.join(self._tree, "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("boot-brain.sh", "api-key-env.sh", "runtime-state.sh",
                     "resolve-brain-db.sh", "brain-env.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name),
                        os.path.join(self._sd, name))
        for name in ("ensure-runtime.sh", "install-daemon-service.sh",
                     "ensure-dashboard.sh"):
            stub = os.path.join(self._sd, name)
            with open(stub, "w") as f:
                f.write("#!/bin/bash\nexit 0\n")
            os.chmod(stub, 0o755)
        with open(os.path.join(self._sd, "boot_brain.py"), "w") as f:
            f.write("")  # boot's final exec — a no-op here

        # Warm path: runtime-ready sentinel + a venv python (brain-env.sh
        # prepends venv/bin to PATH, so link python3 too — the boot script
        # invokes bare `python3`).
        with open(os.path.join(self._tree, ".runtime-ready"), "w") as f:
            f.write("ok\n")
        vbin = os.path.join(self._tree, "venv", "bin")
        os.makedirs(vbin)
        os.symlink(sys.executable, os.path.join(vbin, "python"))
        os.symlink(sys.executable, os.path.join(vbin, "python3"))

        # The plugin tree's servers/ package — what boot MUST import.
        srv = os.path.join(self._tree, "servers")
        os.makedirs(srv)
        open(os.path.join(srv, "__init__.py"), "w").close()
        with open(os.path.join(srv, "daemon_client.py"), "w") as f:
            f.write("def ensure_daemon(db):\n    return True\n")
        with open(os.path.join(srv, "brain_mcp.py"), "w") as f:
            f.write("TOOLS = ['a', 'b']\n")

        # A resolved brain, so boot reaches the daemon + MCP-verify blocks.
        self._db_dir = os.path.join(self._home, "braindb")
        os.makedirs(self._db_dir)
        open(os.path.join(self._db_dir, "brain.db"), "w").close()

        # The user project boot runs from — carrying the decoy package.
        self._project = os.path.join(self._home, "project")
        os.makedirs(os.path.join(self._project, "servers"))
        with open(os.path.join(self._project, "servers", "__init__.py"), "w") as f:
            f.write("raise ImportError('decoy servers package imported "
                    "(cwd leaked into sys.path)')\n")

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _boot(self, cwd):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg,
                   ANTHROPIC_API_KEY="sk-ant-test",
                   BRAIN_DB_DIR=self._db_dir)
        for k in ("PYTHONPATH", "PYTHONHOME", "CLAUDE_PLUGIN_DATA",
                  "CLAUDE_PLUGIN_OPTION_BRAIN_PATH",
                  "CLAUDE_PLUGIN_OPTION_brain_path"):
            env.pop(k, None)
        return subprocess.run(
            ["bash", os.path.join(self._sd, "boot-brain.sh")],
            input="{}", cwd=cwd, env=env,
            capture_output=True, text=True, timeout=60)

    def test_boot_imports_plugin_tree_when_cwd_has_decoy_servers(self):
        r = self._boot(cwd=self._project)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("MCP server OK — 2 tools available", out)
        self.assertIn("Daemon ready", out)
        self.assertNotIn("MCP SERVER BROKEN", out)
        self.assertNotIn("decoy", out)

    def test_boot_succeeds_from_neutral_cwd(self):
        # Control: proves a decoy-test failure means the decoy leaked in,
        # not that the harness itself rotted.
        r = self._boot(cwd=self._home)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("MCP server OK — 2 tools available", out)
        self.assertIn("Daemon ready", out)


class TestBothSpawnPathsExecIdentically(unittest.TestCase):
    """Step 6c: the daemon has ONE boot incantation.

    hooks/scripts/brain-daemon (launchd's entry) and
    daemon_launch.daemon_argv (the no-launchd fallback) used to be a shell
    heredoc and a Python `-c` string that had already drifted apart on env
    pinning — and the heredoc interpolated $DB_PATH into Python source
    unquoted, so a brain path with a space, quote or backslash produced a
    SyntaxError and a launchd respawn loop.

    Runs the real brain-daemon against a stub interpreter that records its
    argv, and compares it to what daemon_argv builds.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-spawnpath-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        self._tree = os.path.join(self._home, "plugin")
        self._sd = os.path.join(self._tree, "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("brain-daemon", "resolve-brain-db.sh", "brain-env.sh",
                     "api-key-env.sh", "launchd-install.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name),
                        os.path.join(self._sd, name))
        stub = os.path.join(self._sd, "ensure-runtime.sh")
        with open(stub, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(stub, 0o755)
        # The venv python brain-env.sh wires in — records argv + PYTHONPATH
        # instead of starting a daemon.
        venv = os.path.join(self._tree, "venv", "bin")
        os.makedirs(venv)
        self._record = os.path.join(self._home, "argv.txt")
        py = os.path.join(venv, "python")
        with open(py, "w") as f:
            f.write('#!/bin/bash\nprintf "%s\\n" "$@" > "{rec}"\n'
                    'printf "PYTHONPATH=%s\\n" "$PYTHONPATH" >> "{rec}"\n'
                    .format(rec=self._record))
        os.chmod(py, 0o755)

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _run_launcher(self, db_dir):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg,
                   BRAIN_DB_DIR=db_dir)
        env.pop("CLAUDE_PLUGIN_DATA", None)
        env.pop("PYTHONPATH", None)
        r = subprocess.run(["bash", os.path.join(self._sd, "brain-daemon")],
                           env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self._record) as f:
            lines = f.read().splitlines()
        argv = [l for l in lines if not l.startswith("PYTHONPATH=")]
        pythonpath = [l for l in lines if l.startswith("PYTHONPATH=")][0][11:]
        return argv, pythonpath

    def _make_brain(self, name):
        d = os.path.join(self._home, name)
        os.makedirs(d)
        open(os.path.join(d, "brain.db"), "w").close()
        return d

    def test_launcher_execs_the_module_entry_point(self):
        db_dir = self._make_brain("brain-data")
        argv, pythonpath = self._run_launcher(db_dir)
        self.assertEqual(argv, ["-m", "servers.daemon_server",
                                os.path.join(db_dir, "brain.db")])
        self.assertIn(os.path.realpath(self._tree),
                      [os.path.realpath(p) for p in pythonpath.split(os.pathsep)],
                      "the plugin tree must be on PYTHONPATH for `-m` to resolve")

    def test_both_paths_build_the_same_command(self):
        db_dir = self._make_brain("brain-data")
        db_path = os.path.join(db_dir, "brain.db")
        shell_argv, _ = self._run_launcher(db_dir)
        python_argv = dl.daemon_argv(db_path)[1:]  # drop the interpreter
        self.assertEqual(shell_argv, python_argv,
                         "launchd and the direct-spawn fallback must exec the "
                         "same daemon command — divergence here is the class "
                         "step 6c removed")

    def test_path_with_spaces_and_quotes_survives(self):
        # The old heredoc pasted this straight into Python source.
        db_dir = self._make_brain("my brain's data")
        argv, _ = self._run_launcher(db_dir)
        self.assertEqual(argv[-1], os.path.join(db_dir, "brain.db"))

class TestSendCommandTransportContract(unittest.TestCase):
    """Step 6d: `transport` is the STABLE classification of a WIRE failure.

    Three consumers branch on it — hook_common.daemon_call_raw (which error
    text Claude sees and which hook_errors row is written), brain_mcp.daemon_send
    and restart-daemon.sh. They must never match on the prose in `error`, which
    is for humans; and the key must be ABSENT when the daemon answered, because
    a daemon-level ok=false reported as a transport failure turns every
    daemon-side error into "cannot reach the daemon".
    """

    def _serve(self, handler):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        host, port = srv.getsockname()

        def run():
            try:
                conn, _ = srv.accept()
                try:
                    conn.recv(4096)
                    handler(conn)
                finally:
                    conn.close()
            except OSError:
                pass
            finally:
                srv.close()

        threading.Thread(target=run, daemon=True).start()
        return host, port

    def _send(self, handler, timeout=3.0):
        host, port = self._serve(handler)
        with patch.object(dc, "get_daemon_addr", return_value=(host, port)):
            return dc.send_command("probe", timeout=timeout)

    def test_success_carries_no_transport(self):
        resp = self._send(lambda c: c.sendall(b'{"ok": true, "result": {"x": 1}}\n'))
        self.assertTrue(resp.get("ok"))
        self.assertIsNone(resp.get("transport"))

    def test_daemon_level_error_carries_no_transport(self):
        # THE invariant: the daemon answered. Marking this a transport failure
        # would make every daemon-side error read as an unreachable daemon.
        resp = self._send(lambda c: c.sendall(b'{"ok": false, "error": "no such node"}\n'))
        self.assertFalse(resp.get("ok"))
        self.assertIsNone(resp.get("transport"),
                          "a daemon that answers is not a wire failure")
        self.assertEqual(resp.get("error"), "no such node")

    def test_empty_reply_is_transport_empty(self):
        resp = self._send(lambda c: None)  # accept, then close
        self.assertEqual(resp.get("transport"), "empty")
        self.assertIn("empty response", resp.get("error", ""))

    def test_garbled_reply_is_transport_protocol(self):
        resp = self._send(lambda c: c.sendall(b"not json at all\n"))
        self.assertEqual(resp.get("transport"), "protocol")

    def test_timeout_is_transport_timeout(self):
        resp = self._send(lambda c: time.sleep(2), timeout=0.3)
        self.assertEqual(resp.get("transport"), "timeout")

    def test_refused_is_transport_refused(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        host, port = srv.getsockname()
        srv.close()  # nothing listening
        with patch.object(dc, "get_daemon_addr", return_value=(host, port)):
            resp = dc.send_command("probe", timeout=1.0)
        self.assertEqual(resp.get("transport"), "refused")

    def test_non_object_reply_is_a_protocol_failure_not_a_raise(self):
        # Valid JSON that isn't an object (a foreign listener on the port).
        # The contract is "returns a response dict" — callers must not each
        # discover separately that it sometimes doesn't.
        resp = self._send(lambda c: c.sendall(b'"OK"\n'))
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp.get("transport"), "protocol")
        self.assertIn("non-object", resp.get("error", ""))


class TestBrainEnvCallsTheApiKeyOwner(unittest.TestCase):
    """Step 6a left brain-env.sh — the file EVERY hook sources — as a one-line
    caller of api-key-env.sh. Deleting that one line is far easier to do by
    accident than deleting the inline block it replaced, and nothing else
    covers this wiring: TestApiKeyEnvHelper tests the owner in isolation and
    TestBootKeyMirror tests boot-brain.sh's separate copy of the call.

    Blast radius if it goes: the 2026-07-15 failure, where a user filled in the
    plugin's key field and the daemon ran keyless anyway."""

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-envwire-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _source(self, env_extra, echo):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("ANTHROPIC_API_KEY", "CLAUDE_PLUGIN_OPTION_API_KEY",
                  "CLAUDE_PLUGIN_OPTION_api_key"):
            env.pop(k, None)
        env.update(env_extra)
        r = subprocess.run(
            ["bash", "-c", '. "%s" >/dev/null 2>&1; printf %%s "%s"'
             % (os.path.join(self.SCRIPTS, "brain-env.sh"), echo)],
            env=env, capture_output=True, text=True, timeout=90)
        return r.stdout

    def test_brain_env_resolves_the_key_from_the_plugin_option(self):
        self.assertEqual(
            self._source({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-wired"},
                         "$ANTHROPIC_API_KEY"),
            "sk-wired",
            "brain-env.sh must call the api-key-env.sh owner, not skip it")

    def test_brain_env_sources_the_user_config_file(self):
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_OPERATOR_NAME=tom\nANTHROPIC_API_KEY=sk-from-file\n")
        out = self._source({}, "$BRAIN_OPERATOR_NAME:$ANTHROPIC_API_KEY")
        self.assertEqual(out, "tom:sk-from-file",
                         "every var in the user config must reach the daemon, "
                         "not just the key")


class TestKnobReadSurvivesDamagedInstall(unittest.TestCase):
    """Step 6 review finding: reading the BRAIN_DB_DIR knob must depend on
    NOTHING but $HOME.

    Before step 6 all three knob readers used the literal
    `${XDG_CONFIG_HOME:-$HOME/.config}/brain/env` — a string that cannot fail.
    Extraction (a) routed that read through `brain_user_env_file`, creating a
    failure mode that did not exist: if `api-key-env.sh` is missing or
    unreadable, the function is undefined. The first repair REFUSED in that
    state, which made a brain the user explicitly named unreachable and told
    them it wasn't configured. The correct repair is a fallback to the literal
    path, so a damaged install still resolves the right brain.

    Second half of the same finding: `.` on a missing file is a SPECIAL BUILTIN
    failure that `|| true` cannot rescue — dash exits 2 and bash under `set -e`
    (which brain-daemon sets) exits 1, both before the resolver can do
    anything. Only a readability guard survives, so that is what both sourcing
    sites use.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-damaged-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        # A plugin tree we can damage without touching the repo.
        self._sd = os.path.join(self._home, "plugin", "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("resolve-brain-db.sh", "brain-env.sh", "api-key-env.sh",
                     "launchd-install.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name), os.path.join(self._sd, name))
        stub = os.path.join(self._sd, "ensure-runtime.sh")
        with open(stub, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(stub, 0o755)
        # A real brain, named only by the knob.
        self._brain = os.path.join(self._home, "knobbrain")
        os.makedirs(self._brain)
        open(os.path.join(self._brain, "brain.db"), "w").close()
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % self._brain)

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _resolve(self, shell="bash", errexit=False):
        import subprocess
        resolver = os.path.join(self._sd, "resolve-brain-db.sh")
        script = '%s. "%s" >/dev/null 2>&1 %s printf %%s "$BRAIN_DB_DIR"' % (
            "set -e; " if errexit else "",
            resolver,
            "|| true;" if errexit else ";")
        r = subprocess.run([shell, "-c", script],
                           cwd="/tmp",
                           env={"HOME": self._home, "PATH": "/usr/bin:/bin",
                                "XDG_CONFIG_HOME": self._xdg},
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip()

    def _shadow_brain_created(self):
        return os.path.exists(os.path.join(self._home, ".local", "share", "brain"))

    def _damage(self, mode="remove"):
        p = os.path.join(self._sd, "api-key-env.sh")
        if mode == "remove":
            os.remove(p)
        else:
            os.chmod(p, 0o000)

    def test_healthy_install_adopts_the_knob_brain(self):
        for shell in ("bash", "zsh", "dash"):
            with self.subTest(shell=shell):
                if not shutil.which(shell):
                    self.skipTest("%s not available" % shell)
                self.assertEqual(self._resolve(shell), self._brain)

    def test_missing_api_key_env_still_adopts_the_knob_brain(self):
        self._damage("remove")
        for shell in ("bash", "zsh", "dash"):
            with self.subTest(shell=shell):
                if not shutil.which(shell):
                    self.skipTest("%s not available" % shell)
                self.assertEqual(
                    self._resolve(shell), self._brain,
                    "a damaged install must not make the user's named brain "
                    "unreachable — fall back to the literal config path")
        self.assertFalse(self._shadow_brain_created(),
                         "and it must certainly not birth a second brain")

    def test_unreadable_api_key_env_still_adopts_the_knob_brain(self):
        self._damage("chmod")
        self.assertEqual(self._resolve("bash"), self._brain)
        self.assertFalse(self._shadow_brain_created())

    def test_damaged_install_does_not_kill_the_shell_under_errexit(self):
        # brain-daemon runs `set -e`; a special-builtin failure there exits
        # before any FATAL can print, and launchd respawns every 10s forever.
        self._damage("remove")
        self.assertEqual(self._resolve("bash", errexit=True), self._brain,
                         "the resolver must survive `set -e` on a damaged install")

    def test_damaged_install_does_not_kill_dash(self):
        if not shutil.which("dash"):
            self.skipTest("dash not available")
        self._damage("remove")
        self.assertEqual(self._resolve("dash"), self._brain,
                         "`.` on a missing file exits dash outright unless guarded")


if __name__ == "__main__":
    unittest.main()
