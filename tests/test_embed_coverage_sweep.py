"""Unscoped vector-coverage sweep (embed_queue._coverage_sweep).

The queue repairs only what reached it. A writer that skips the enqueue
hooks, or a crash between insert and enqueue, leaves a node unembedded
with no other route back — and an unembedded node is invisible to LAF
entirely. This sweep is that route.

It previously lived in `hook_idle_maintenance`, which is driven by a
Claude Code Notification/idle_prompt event. That event stopped firing
2026-07-04 and took vector coverage with it, silently, for six weeks.
These tests hold the contract that it now lives beside the queue that
owns embedding, and that a gap is reported rather than quietly healed.

Pure-function tests — stub brain, no DB, no embedder.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import embed_queue


class _StubBrain:
    """Records backfill_vectors calls and _log_error calls."""

    def __init__(self, result=None, raises=None):
        self.result = result if result is not None else {}
        self.raises = raises
        self.calls = []
        self.errors = []

    def backfill_vectors(self, batch_size=None, node_ids=None):
        self.calls.append({'batch_size': batch_size, 'node_ids': node_ids})
        if self.raises is not None:
            raise self.raises
        return self.result

    def _log_error(self, source, error, context='', ctx=None):
        self.errors.append({'source': source, 'error': error,
                            'context': context})


class CoverageSweepTest(unittest.TestCase):

    def setUp(self):
        embed_queue._last_sweep_at = 0.0
        embed_queue._queue.clear()
        embed_queue._edge_queue.clear()

    tearDown = setUp

    def test_sweep_is_unscoped(self):
        """node_ids must be omitted — a scoped sweep repairs nothing new."""
        brain = _StubBrain()
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 1)
        self.assertIsNone(brain.calls[0]['node_ids'])
        self.assertEqual(brain.calls[0]['batch_size'],
                         embed_queue.COVERAGE_SWEEP_BATCH)

    def test_throttled_within_interval(self):
        """Second call inside the window is a no-op — the drain ticks at 5s."""
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

    def test_repair_is_reported_as_an_error(self):
        """A repair means a node reached a drained queue unembedded.

        That is an enqueue-path gap, not a routine self-heal — it must be
        visible, or this regresses to the silence it was built to end.
        """
        brain = _StubBrain(result={'_primary': 3, '_situation': 2})
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.errors), 1)
        self.assertEqual(brain.errors[0]['source'], 'embed_coverage_gap')
        self.assertIn('5', brain.errors[0]['context'])

    def test_no_error_when_nothing_repaired(self):
        """The common case is silence — every empty tick must not log."""
        brain = _StubBrain(result={})
        embed_queue._coverage_sweep(brain)
        self.assertEqual(brain.errors, [])

    def test_non_int_values_do_not_count_as_repairs(self):
        brain = _StubBrain(result={'error': 'embedder unavailable'})
        embed_queue._coverage_sweep(brain)
        self.assertEqual(brain.errors, [])

    def test_failure_is_logged_and_swallowed(self):
        """The worker thread must never die — see _worker_loop's mandate."""
        brain = _StubBrain(raises=RuntimeError('embedder down'))
        embed_queue._coverage_sweep(brain)          # must not raise
        self.assertEqual(len(brain.errors), 1)
        self.assertEqual(brain.errors[0]['source'], 'embed_coverage_sweep')


class EmptyTickWiringTest(unittest.TestCase):
    """The sweep is reachable — the failure mode this whole fix addresses
    was a repair path nothing called."""

    def setUp(self):
        embed_queue._last_sweep_at = 0.0
        embed_queue._queue.clear()
        embed_queue._edge_queue.clear()

    tearDown = setUp

    def test_empty_drain_triggers_sweep(self):
        brain = _StubBrain()
        embed_queue._drain_once(brain)
        self.assertEqual(len(brain.calls), 1,
                         'empty tick must run the coverage sweep')


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


if __name__ == '__main__':
    unittest.main()
