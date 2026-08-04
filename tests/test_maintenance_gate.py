"""Tests for brain.run_maintenance_if_due() — the S2 fire decision.

Covers the gating contract:
- min_interval is absolute (never fire more often than that)
- idle threshold is the normal trigger
- 24h force-fire safety valve overrides idle when S2 is stale

Background: S2 was firing every 2.5 days because daemon's `last_activity`
was reset by every IPC call (Claude editing files, internal pings). The
fix split `last_user_activity` (only hook_recall) from `last_activity`
(any IPC), but the brain-side gate still needed a stale-S2 escape valve
for the case where even prompt cadence keeps the idle window closed.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_constants import (
    MAINTENANCE_IDLE_THRESHOLD_SECONDS,
    MAINTENANCE_MIN_INTERVAL_SECONDS,
    MAINTENANCE_MIN_ENCODE_RUNS,
    MAINTENANCE_FORCE_FIRE_SECONDS,
    MAINTENANCE_BOOT_GRACE_SECONDS,
)


class MaintenanceGateTests(BrainTestBase):
    needs_embedder = False

    def _set_last_run(self, ts: float):
        self.brain._maintenance_set_last_run_ts(ts)

    def _call(self, idle_seconds: float, since_last_run: float,
              encode_runs: int = None):
        """Invoke gate with controlled idle + since_last_run via brain.activity.

        encode_runs defaults to MAINTENANCE_MIN_ENCODE_RUNS so the idle/interval
        tests aren't incidentally blocked by the activity gate — they set it
        high enough to pass and exercise the gate they care about. Tests that
        target the activity gate pass an explicit value.

        Patches run_s2 so a real fire returns a sentinel (no actual S2 work).
        """
        now = 1_000_000.0
        # Clear the boot-grace gate (2026-05-08): it suppresses maintenance for
        # the first BOOT_GRACE seconds after _boot_time. Anchor _boot_time to
        # the fake `now` so boot_age is well past the grace window — otherwise
        # the gate short-circuits before the idle/interval logic under test.
        self.brain._boot_time = now - MAINTENANCE_BOOT_GRACE_SECONDS - 60
        self._set_last_run(now - since_last_run)
        self.brain.activity.last_user_activity = now - idle_seconds
        self.brain.activity.encode_runs_since_maintenance = (
            MAINTENANCE_MIN_ENCODE_RUNS if encode_runs is None else encode_runs)
        with patch('servers.scales.s2.coordinator.run_s2',
                   return_value={'fired': True}):
            return self.brain.run_maintenance_if_due(now=now)

    # ── min_interval is absolute ──────────────────────────────────────

    def test_min_interval_blocks_even_when_idle(self):
        result = self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 60,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS - 60,
        )
        self.assertIsNone(result, "min_interval must block even when idle")

    def test_min_interval_blocks_force_fire_window(self):
        # Even at 24h+ stale, if min_interval not satisfied we still skip.
        # Defensive: ensures the safety valve doesn't undermine min_interval.
        result = self._call(
            idle_seconds=10,  # not idle
            since_last_run=10,  # min_interval not satisfied
        )
        self.assertIsNone(result)

    # ── idle threshold is the normal trigger ──────────────────────────

    def test_idle_satisfied_fires(self):
        result = self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 1,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result['units'], {'fired': True})

    def test_not_idle_no_force_fire_blocks(self):
        # Active session, min_interval satisfied, but not stale enough to
        # force-fire. Should skip.
        result = self._call(
            idle_seconds=10,  # user is active
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 60,
        )
        self.assertIsNone(result)

    # ── 24h force-fire safety valve ───────────────────────────────────

    def test_stale_s2_force_fires_despite_active_user(self):
        # The 2.5-day-gap class: user prompts every minute (idle_seconds
        # always small), but S2 is 24h+ stale. Safety valve must override.
        result = self._call(
            idle_seconds=10,  # user actively typing
            since_last_run=MAINTENANCE_FORCE_FIRE_SECONDS + 60,
        )
        self.assertIsNotNone(result,
            "stale S2 must fire even when user is active")

    def test_force_fire_persists_last_run(self):
        before = self.brain._maintenance_last_run_ts()
        self._call(
            idle_seconds=10,
            since_last_run=MAINTENANCE_FORCE_FIRE_SECONDS + 60,
        )
        after = self.brain._maintenance_last_run_ts()
        self.assertGreater(after, before,
            "fire must update s2_last_run_ts")

    # ── cold-boot path (last_activity_ts == 0) ────────────────────────

    def test_zero_last_activity_treated_as_infinite_idle(self):
        # last_user_activity = 0.0 (no prompts yet) must map to "infinitely
        # idle" so the gate fires — once past the boot-grace window. (Boot
        # grace, 2026-05-08, deliberately suppresses the very first post-start
        # poll so the first user recall isn't blocked behind consolidation.)
        now = 1_000_000.0
        self.brain._boot_time = now - MAINTENANCE_BOOT_GRACE_SECONDS - 60
        self._set_last_run(now - MAINTENANCE_MIN_INTERVAL_SECONDS - 60)
        self.brain.activity.last_user_activity = 0.0
        self.brain.activity.encode_runs_since_maintenance = MAINTENANCE_MIN_ENCODE_RUNS
        with patch('servers.scales.s2.coordinator.run_s2',
                   return_value={'fired': True}):
            result = self.brain.run_maintenance_if_due(now=now)
        self.assertIsNotNone(result,
            "past boot-grace, a daemon with no user activity (0.0) is treated "
            "as infinitely idle and fires")


    # ── encode-runs activity gate (≥N Scribe runs since last run) ─────

    def test_insufficient_encode_runs_blocks(self):
        # Idle + interval both satisfied, but only 1 encoder run since last S2.
        result = self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 1,
            encode_runs=MAINTENANCE_MIN_ENCODE_RUNS - 1,
        )
        self.assertIsNone(result,
            "too few encoder runs since last run must block the fire")

    def test_sufficient_encode_runs_fires(self):
        result = self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 1,
            encode_runs=MAINTENANCE_MIN_ENCODE_RUNS,
        )
        self.assertIsNotNone(result, "enough encoder runs + idle + interval fires")

    def test_force_fire_overrides_encode_runs(self):
        # 24h stale with zero encoder runs — the safety valve still fires.
        result = self._call(
            idle_seconds=10,
            since_last_run=MAINTENANCE_FORCE_FIRE_SECONDS + 60,
            encode_runs=0,
        )
        self.assertIsNotNone(result,
            "force-fire must override the encode-runs gate for stale S2")

    def test_fire_consumes_encode_runs(self):
        # When S2 fires, it consumes exactly the encode runs it gated on, so
        # the counter resets toward the next cycle.
        self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 1,
            encode_runs=MAINTENANCE_MIN_ENCODE_RUNS,
        )
        self.assertEqual(self.brain.activity.encode_runs_since_maintenance, 0,
            "a fire must consume the encode runs it gated on")


if __name__ == '__main__':
    unittest.main()


class S2SingleDoorTests(BrainTestBase):
    """brain.run_s2() — the one door, and the single-flight it owns.

    The guard used to live on the daemon (`_s2_running`), an attribute of the
    SERVER, so it could only block a second poll entry. That is how the
    2026-06 parallel-run bug happened (node:daaf63a9): a second caller could
    neither see it nor be seen by it. These tests pin the guard to the brain,
    where every in-process caller shares it.
    """
    needs_embedder = False

    def test_run_s2_returns_units_and_timing(self):
        with patch('servers.scales.s2.coordinator.run_s2',
                   return_value={'healer': {'actions': 3}}):
            out = self.brain.run_s2()
        self.assertEqual(out['units'], {'healer': {'actions': 3}})
        self.assertIn('elapsed_ms', out)
        self.assertNotIn('skipped', out)

    def test_second_concurrent_call_skips_instead_of_overlapping(self):
        """Two cycles must never overlap. Exercised by re-entering the door from
        inside a running cycle — which also pins that S2 cannot re-enter itself
        (a plain Lock, deliberately not an RLock)."""
        inner = {}

        def _reenter(brain_arg):
            inner['result'] = self.brain.run_s2()
            return {'healer': {'actions': 1}}

        with patch('servers.scales.s2.coordinator.run_s2', side_effect=_reenter):
            outer = self.brain.run_s2()

        self.assertEqual(outer['units'], {'healer': {'actions': 1}},
                         "the first caller must complete normally")
        self.assertEqual(inner['result'].get('skipped'), 'already running',
                         "the second caller must skip, not run or block")
        self.assertEqual(inner['result']['units'], {})

    def test_lock_is_released_even_when_a_cycle_raises(self):
        with patch('servers.scales.s2.coordinator.run_s2',
                   side_effect=RuntimeError('unit blew up')):
            with self.assertRaises(RuntimeError):
                self.brain.run_s2()
        self.assertFalse(self.brain.s2_running,
                         "a raising cycle must not wedge S2 forever")

    def test_gate_does_not_stamp_or_consume_while_a_cycle_is_running(self):
        """THE subtlety. run_maintenance_if_due stamps the last-run timestamp
        BEFORE executing (so a concurrent poller skips on min-interval). If it
        stamped and then run_s2() skipped because the lock was held, the cycle
        would be burned without running AND the encode runs it gated on would
        be eaten — starving the next one. So the gate must check first."""
        now = 1_000_000.0
        self.brain._boot_time = now - MAINTENANCE_BOOT_GRACE_SECONDS - 60
        before_ts = now - MAINTENANCE_MIN_INTERVAL_SECONDS - 60
        self.brain._maintenance_set_last_run_ts(before_ts)
        self.brain.activity.last_user_activity = (
            now - MAINTENANCE_IDLE_THRESHOLD_SECONDS - 60)
        self.brain.activity.encode_runs_since_maintenance = (
            MAINTENANCE_MIN_ENCODE_RUNS + 1)
        expected_runs = self.brain.activity.encode_runs_since_maintenance

        # Every gate is open — the ONLY thing that should stop it is the lock.
        self.brain._s2_lock.acquire()
        try:
            result = self.brain.run_maintenance_if_due(now=now)
        finally:
            self.brain._s2_lock.release()

        self.assertIsNone(result, "must not report a run that did not happen")
        self.assertEqual(self.brain.activity.encode_runs_since_maintenance,
                         expected_runs,
                         "encode runs were consumed for a cycle that never ran")
        self.assertAlmostEqual(
            self.brain._maintenance_last_run_ts(), before_ts, places=3,
            msg="last-run timestamp advanced for a cycle that never ran")
