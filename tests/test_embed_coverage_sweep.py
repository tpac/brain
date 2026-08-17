"""Unscoped vector-coverage sweep (embed_queue._coverage_sweep).

The queue repairs only what reached it. A writer that skips the enqueue
hooks, or a crash between insert and enqueue, leaves a node unembedded
with no other route back — and an unembedded node is invisible to LAF
entirely. This sweep is that route.

It previously lived in `hook_idle_maintenance`, which is driven by a
Claude Code Notification/idle_prompt event. That event stopped firing
2026-07-04 and took vector coverage with it, silently, for six weeks.
These tests hold the contract that it now lives beside the queue that
owns embedding, that it cannot be starved, and that BOTH a gap and a
stuck node are reported.

Pure-function tests — stub brain, no DB, no embedder.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import embed_queue


class _StubVecDal:
    def __init__(self, missing=None):
        self.missing = missing or []

    def find_missing(self, vector_type, limit=50, **kw):
        return self.missing[:limit]


class _StubBrain:
    """Records backfill_vectors calls and _log_error calls."""

    def __init__(self, result=None, raises=None, missing=None):
        self.result = result if result is not None else {}
        self.raises = raises
        self.calls = []
        self.errors = []
        self._vec_dal = _StubVecDal(missing)

    def backfill_vectors(self, batch_size=None, node_ids=None):
        self.calls.append({'batch_size': batch_size, 'node_ids': node_ids})
        if self.raises is not None:
            raise self.raises
        return self.result

    def _log_error(self, source, error, context='', ctx=None):
        self.errors.append({'source': source, 'error': error,
                            'context': context})

    def sources(self):
        return [e['source'] for e in self.errors]


def _reset():
    embed_queue._last_sweep_at = 0.0
    embed_queue._queue.clear()
    embed_queue._edge_queue.clear()


class CoverageSweepTest(unittest.TestCase):

    setUp = tearDown = staticmethod(_reset)

    def test_sweep_is_unscoped(self):
        """node_ids must be omitted — a scoped sweep repairs nothing new."""
        brain = _StubBrain()
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 1)
        self.assertIsNone(brain.calls[0]['node_ids'])
        self.assertEqual(brain.calls[0]['batch_size'],
                         embed_queue.COVERAGE_SWEEP_BATCH)

    def test_throttled_within_interval(self):
        brain = _StubBrain()
        embed_queue._coverage_sweep(brain)
        embed_queue._coverage_sweep(brain)
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 1)

    def test_runs_again_after_interval_elapses(self):
        brain = _StubBrain()
        embed_queue._coverage_sweep(brain)
        embed_queue._last_sweep_at -= (embed_queue.COVERAGE_SWEEP_INTERVAL + 1)
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 2)

    def test_defers_to_scoped_work(self):
        """A non-empty queue means the drain is about to cover those ids."""
        brain = _StubBrain()
        embed_queue._queue.add('node-1')
        embed_queue._coverage_sweep(brain)
        self.assertEqual(brain.calls, [])

    def test_defers_to_scoped_edge_work(self):
        brain = _StubBrain()
        embed_queue._edge_queue.add('edge-1')
        embed_queue._coverage_sweep(brain)
        self.assertEqual(brain.calls, [])

    def test_repair_is_reported_as_an_error(self):
        brain = _StubBrain(result={'_primary': 3, '_situation': 2})
        embed_queue._coverage_sweep(brain)
        self.assertIn('embed_coverage_gap', brain.sources())
        self.assertIn('5', brain.errors[0]['context'])

    def test_stuck_node_is_reported_when_nothing_was_repaired(self):
        """The case reporting-on-repair-only would hide forever.

        A node that CANNOT be embedded repairs 0 on every sweep. If the only
        signal were `repaired > 0` it would stay silent indefinitely — the
        same absence-vs-failure blind spot that hid the orphaned backfill.
        """
        brain = _StubBrain(result={}, missing=[{'id': 'abc123'}])
        embed_queue._coverage_sweep(brain)
        self.assertIn('embed_coverage_stuck', brain.sources())
        self.assertIn('abc123', brain.errors[0]['context'])

    def test_backlog_clears_the_throttle(self):
        """Recovery must not be capped at one batch per interval.

        A model change marks the whole corpus missing; at BATCH per INTERVAL
        that is hours of drip-feed with LAF blind to every unrepaired node.
        """
        brain = _StubBrain(result={'_primary': 30}, missing=[{'id': 'more'}])
        embed_queue._coverage_sweep(brain)
        self.assertEqual(embed_queue._last_sweep_at, 0.0,
                         'throttle must clear while a backlog remains')
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 2)

    def test_clean_sweep_keeps_the_throttle(self):
        brain = _StubBrain(result={}, missing=[])
        embed_queue._coverage_sweep(brain)
        self.assertNotEqual(embed_queue._last_sweep_at, 0.0)
        self.assertEqual(brain.errors, [])

    def test_non_int_values_do_not_count_as_repairs(self):
        brain = _StubBrain(result={'error': 'embedder unavailable'})
        embed_queue._coverage_sweep(brain)
        self.assertNotIn('embed_coverage_gap', brain.sources())

    def test_failure_is_logged_and_swallowed(self):
        """The worker thread must never die — see _worker_loop's mandate."""
        brain = _StubBrain(raises=RuntimeError('embedder down'))
        embed_queue._coverage_sweep(brain)          # must not raise
        self.assertIn('embed_coverage_sweep', brain.sources())


class SweepIsNotInTheDrainTest(unittest.TestCase):
    """The sweep belongs to the worker loop, not to _drain_once.

    Two reasons, both load-bearing:
      1. _drain_once's only sweep-reachable branch is its empty-tick
         early-return, so under sustained write load the branch is never
         entered and the safety net silently switches off — exactly when an
         enqueue miss is most likely.
      2. The test harness (brain_test_base, isolated_brain) calls
         _drain_once directly. Sweeping there means every test process runs
         an unscoped backfill against whatever brain it holds — including
         IsolatedBrain's copy of production data.
    """

    setUp = tearDown = staticmethod(_reset)

    def test_drain_does_not_sweep(self):
        brain = _StubBrain()
        embed_queue._drain_once(brain)
        self.assertEqual(brain.calls, [],
                         '_drain_once must not trigger the coverage sweep')

    def test_worker_loop_calls_the_sweep(self):
        """Guards the wiring — the whole failure this fix addresses was a
        repair path nothing called."""
        import inspect
        src = inspect.getsource(embed_queue._worker_loop)
        self.assertIn('_coverage_sweep(brain)', src)


class HookNoLongerOwnsVectorsTest(unittest.TestCase):
    """Directional guardrail: vector coverage must not drift back behind
    the Claude Code notification hook."""

    def test_idle_maintenance_does_not_backfill_vectors(self):
        path = os.path.join(os.path.dirname(__file__), '..',
                            'servers', 'daemon_hooks.py')
        with open(path) as fh:
            src = fh.read()
        self.assertNotIn('brain.backfill_vectors(', src,
                         'vector backfill belongs to embed_queue, not the '
                         'Notification/idle_prompt hook')


class LogsMaintenanceUsesPrivateConnTest(unittest.TestCase):
    """The orphan sweep must never run on brain.conn.

    run_maintenance ends in a bare graph_conn.commit(). On the foreground
    connection that commit lands on whatever brain_batch envelope is open —
    the stray-commit class that killed a savepoint mid-merge once already.
    A private maintenance connection makes it structurally impossible.
    """

    def test_brain_door_uses_a_maintenance_connection(self):
        import inspect
        from servers.brain import Brain
        src = inspect.getsource(Brain.run_logs_maintenance)
        self.assertIn('connect_maintenance', src)
        self.assertNotIn('graph_conn=self.conn', src)

    def test_daemon_does_not_reach_into_logs_dal(self):
        path = os.path.join(os.path.dirname(__file__), '..',
                            'servers', 'daemon_server.py')
        with open(path) as fh:
            src = fh.read()
        self.assertNotIn('_logs_dal.run_maintenance', src,
                         'route through Brain.run_logs_maintenance')


if __name__ == '__main__':
    unittest.main()
