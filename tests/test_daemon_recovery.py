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
        with patch.object(dc.subprocess, "run", return_value=MagicMock(returncode=1)), \
             patch.object(dc, "_kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon", return_value=True) as ensure:
            dc._relaunch_daemon("/tmp/x/brain.db")
            kill.assert_called_once()
            ensure.assert_called_once_with("/tmp/x/brain.db")


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
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", side_effect=[True, False]), \
             patch.object(dc, "_launchd_kickstart") as ks, \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            popen.assert_not_called()

    def test_no_launchd_falls_back_to_direct_spawn(self):
        # Daemon down AND launchd not managing it (kickstart rc!=0) → the only
        # case a direct Popen is legitimate (no KeepAlive to race).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "_can_connect",
                          side_effect=[{"ok": False}, {"ok": False}, {"ok": True}]), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_launchd_kickstart", return_value=False) as ks, \
             patch.object(dc, "_port_is_occupied", return_value=False), \
             patch.object(dc, "_debugger_friendly_python", return_value="/usr/bin/python3"), \
             patch.object(dc.subprocess, "Popen") as popen, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_called_once()
            popen.assert_called_once()

    def test_maintenance_mode_skips_everything(self):
        with patch.object(dc, "is_maintenance_mode", return_value=True), \
             patch.object(dc, "_launchd_kickstart") as ks, \
             patch.object(dc.subprocess, "Popen") as popen:
            self.assertFalse(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
