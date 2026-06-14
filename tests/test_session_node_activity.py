"""Per-session node activity — SessionContext storage + persistence.

SessionContext.node_activity is the parallel-session replacement for global
nodes.{activation, recency_score, last_accessed, access_count} for reads
that should be session-scoped (spreading-activation kernel, recency
filtering, live-session Frame composition).

Tests cover the bump semantics + the session_state JSON persistence path.
Behavioral wiring tests (foreground recall bumps activity, spreading kernel
reads from ctx) land in test_surface_transitions.py + spread-activation
tests as those layers ship.
"""

import json
import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.session_context import SessionContext


class TestBumpNodeActivity(unittest.TestCase):
    """SessionContext.bump_node_activity write semantics."""

    def test_first_access_creates_entry(self):
        ctx = SessionContext(session_id='sess-A')
        rec = ctx.bump_node_activity('nid-1', '2026-05-25T10:00:00+00:00')
        self.assertEqual(rec['access_count'], 1)
        self.assertAlmostEqual(rec['activation'], 1.0)
        self.assertAlmostEqual(rec['recency_score'], 1.0)
        self.assertEqual(rec['last_accessed'], '2026-05-25T10:00:00+00:00')

    def test_repeat_access_increments_count_and_caps_activation(self):
        ctx = SessionContext(session_id='sess-A')
        ctx.bump_node_activity('nid-1', '2026-05-25T10:00:00+00:00')
        # Second access — activation already at 1.0, should stay capped.
        rec = ctx.bump_node_activity('nid-1', '2026-05-25T10:05:00+00:00')
        self.assertEqual(rec['access_count'], 2)
        self.assertAlmostEqual(rec['activation'], 1.0)
        self.assertEqual(rec['last_accessed'], '2026-05-25T10:05:00+00:00')

    def test_activation_bumps_from_low_value(self):
        """If an existing record has activation below 1.0 (decay path), the
        +0.1 bump applies; cap at 1.0 only when overflow."""
        ctx = SessionContext(session_id='sess-A')
        ctx.node_activity['nid-1'] = {
            'activation': 0.3, 'recency_score': 0.1,
            'access_count': 2, 'last_accessed': '2026-05-24T00:00:00+00:00',
        }
        rec = ctx.bump_node_activity('nid-1', '2026-05-25T10:00:00+00:00')
        self.assertAlmostEqual(rec['activation'], 0.4, places=4)
        self.assertAlmostEqual(rec['recency_score'], 1.0)
        self.assertEqual(rec['access_count'], 3)

    def test_monotonic_last_accessed(self):
        """Earlier ts shouldn't overwrite a later one (defensive against
        out-of-order drains)."""
        ctx = SessionContext(session_id='sess-A')
        ctx.bump_node_activity('nid-1', '2026-05-25T10:00:00+00:00')
        rec = ctx.bump_node_activity('nid-1', '2026-05-25T09:00:00+00:00')
        # access_count still increments (the access happened), but last_accessed
        # keeps the later ts.
        self.assertEqual(rec['access_count'], 2)
        self.assertEqual(rec['last_accessed'], '2026-05-25T10:00:00+00:00')

    def test_empty_node_id_is_noop(self):
        ctx = SessionContext(session_id='sess-A')
        rec = ctx.bump_node_activity('', '2026-05-25T10:00:00+00:00')
        self.assertEqual(rec, {})
        self.assertEqual(ctx.node_activity, {})

    def test_get_node_activity_returns_record_or_empty(self):
        ctx = SessionContext(session_id='sess-A')
        self.assertEqual(ctx.get_node_activity('nid-1'), {})
        ctx.bump_node_activity('nid-1', '2026-05-25T10:00:00+00:00')
        rec = ctx.get_node_activity('nid-1')
        self.assertEqual(rec['access_count'], 1)


class TestPersistenceRoundTrip(unittest.TestCase):
    """SessionContext.save() + load() preserve node_activity through the
    session_state JSON blob.
    """

    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        # Mirror the session_state DDL from schema.py (logs DB).
        self.conn.execute(
            'CREATE TABLE session_state ('
            '  session_id TEXT, key TEXT, node_id TEXT, value TEXT, '
            '  updated_at TEXT, '
            '  PRIMARY KEY (session_id, key, node_id))')

    def tearDown(self):
        self.conn.close()

    def test_save_and_load_round_trip(self):
        ctx = SessionContext(session_id='sess-A')
        ctx.bump_node_activity('nid-1', '2026-05-25T10:00:00+00:00')
        ctx.bump_node_activity('nid-2', '2026-05-25T10:05:00+00:00')
        ctx.bump_node_activity('nid-1', '2026-05-25T10:10:00+00:00')
        ctx.save(self.conn)

        loaded = SessionContext.load(self.conn, 'sess-A')
        self.assertIsNotNone(loaded)
        self.assertIn('nid-1', loaded.node_activity)
        self.assertIn('nid-2', loaded.node_activity)
        self.assertEqual(loaded.node_activity['nid-1']['access_count'], 2)
        self.assertEqual(loaded.node_activity['nid-2']['access_count'], 1)
        self.assertEqual(loaded.node_activity['nid-1']['last_accessed'],
                         '2026-05-25T10:10:00+00:00')

    def test_load_backward_compat_no_node_activity(self):
        """Older session_state rows have no 'node_activity' key — load must
        default to empty dict, not raise."""
        legacy = json.dumps({
            'stop_counter': 3,
            'fatigue': {'nA': 1},
            'edge_fatigue': {},
            'remember_count': 0, 'message_count': 0, 'edit_check_count': 0,
            # conversational_count + last_encode_at_message are PRE-MIGRATION keys
            # (removed when the Scribe cadence moved to trace-pull). load() must
            # tolerate old rows that still carry them — they're silently ignored.
            'conversational_count': 7, 'last_encode_at_message': 0, 'boot_time': '',
            'segment_id': 0, 'segment_embeddings': [], 'segment_node_ids': [],
            # NO node_activity key
        })
        self.conn.execute(
            'INSERT INTO session_state (session_id, key, node_id, value, updated_at) '
            'VALUES (?, ?, ?, ?, ?)',
            ('sess-old', '_session_context', '', legacy, '2026-05-25T00:00:00+00:00'))
        self.conn.commit()

        loaded = SessionContext.load(self.conn, 'sess-old')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.node_activity, {})
        # Other fields still load correctly.
        self.assertEqual(loaded.stop_counter, 3)
        self.assertEqual(loaded.fatigue, {'nA': 1})
        # Removed cadence keys in the old blob don't resurrect as attributes.
        self.assertFalse(hasattr(loaded, 'conversational_count'))
        self.assertFalse(hasattr(loaded, 'last_encode_at_message'))

    def test_per_session_isolation_across_save(self):
        """Two sessions saved to the same DB → load returns isolated state."""
        a = SessionContext(session_id='sess-A')
        a.bump_node_activity('nid-X', '2026-05-25T10:00:00+00:00')
        a.save(self.conn)

        b = SessionContext(session_id='sess-B')
        b.bump_node_activity('nid-X', '2026-05-25T10:05:00+00:00')
        b.bump_node_activity('nid-Y', '2026-05-25T10:06:00+00:00')
        b.save(self.conn)

        loaded_a = SessionContext.load(self.conn, 'sess-A')
        loaded_b = SessionContext.load(self.conn, 'sess-B')

        self.assertEqual(set(loaded_a.node_activity.keys()), {'nid-X'})
        self.assertEqual(set(loaded_b.node_activity.keys()), {'nid-X', 'nid-Y'})
        # Same node, different per-session timestamps preserved.
        self.assertEqual(loaded_a.node_activity['nid-X']['last_accessed'],
                         '2026-05-25T10:00:00+00:00')
        self.assertEqual(loaded_b.node_activity['nid-X']['last_accessed'],
                         '2026-05-25T10:05:00+00:00')


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestLiveSessionsHelper(BrainTestBase):
    """brain.live_sessions returns the last X session_ids with >=Y messages,
    ordered by recency. Used as the cross-session aggregation seed list."""

    needs_embedder = False

    def _save_ctx(self, sid, message_count):
        ctx = self.brain.get_or_create_session(sid)
        ctx.message_count = message_count
        ctx.save(self.brain.logs_conn)
        return ctx

    def test_filters_by_min_messages(self):
        self._save_ctx('low-msg', 2)
        self._save_ctx('hi-msg', 10)
        result = self.brain.live_sessions(limit=10, min_messages=5)
        self.assertIn('hi-msg', result)
        self.assertNotIn('low-msg', result)

    def test_respects_limit(self):
        for i in range(8):
            self._save_ctx('sess-%d' % i, 10)
        result = self.brain.live_sessions(limit=3, min_messages=5)
        self.assertLessEqual(len(result), 3)

    def test_orders_by_recency(self):
        """updated_at on the session_state row drives order (most recent first)."""
        import time
        self._save_ctx('older', 10)
        time.sleep(0.02)
        self._save_ctx('newer', 10)
        result = self.brain.live_sessions(limit=5, min_messages=5)
        # newer should appear before older
        self.assertLess(result.index('newer'), result.index('older'))


class TestLiveSessionActivity(BrainTestBase):
    """brain.live_session_activity aggregates node_activity across live
    sessions. Tests cache-vs-persisted hybrid + aggregation correctness."""

    needs_embedder = False

    def _seed_session(self, sid, message_count, node_activity_pairs):
        """Create a session with the given activity records. node_activity_pairs
        is a list of (node_id, ts) tuples — each is a single bump call.
        """
        ctx = self.brain.get_or_create_session(sid)
        ctx.message_count = message_count
        for nid, ts in node_activity_pairs:
            ctx.bump_node_activity(nid, ts)
        ctx.save(self.brain.logs_conn)
        return ctx

    def test_empty_when_no_live_sessions(self):
        result = self.brain.live_session_activity()
        self.assertEqual(result, {})

    def test_aggregates_max_last_accessed(self):
        """Same node touched by two sessions → MAX(last_accessed) wins."""
        self._seed_session('s1', 10,
                           [('node-X', '2026-05-20T10:00:00+00:00')])
        self._seed_session('s2', 10,
                           [('node-X', '2026-05-25T10:00:00+00:00')])
        agg = self.brain.live_session_activity()
        self.assertEqual(agg['node-X']['last_accessed'],
                         '2026-05-25T10:00:00+00:00')

    def test_aggregates_sum_access_count_and_max_activation(self):
        """access_count sums; activation takes the MAX across sessions."""
        # First session — accessed twice (bumps to activation 1.0, count 2)
        self._seed_session(
            's1', 10,
            [('node-X', '2026-05-20T10:00:00+00:00'),
             ('node-X', '2026-05-20T10:05:00+00:00')])
        # Second session — accessed once (count 1, activation 1.0)
        self._seed_session(
            's2', 10,
            [('node-X', '2026-05-25T10:00:00+00:00')])
        agg = self.brain.live_session_activity()
        self.assertEqual(agg['node-X']['access_count'], 3)
        self.assertAlmostEqual(agg['node-X']['activation'], 1.0)
        self.assertEqual(agg['node-X']['session_count'], 2)

    def test_node_ids_filter_narrows(self):
        self._seed_session(
            's1', 10,
            [('node-X', '2026-05-25T10:00:00+00:00'),
             ('node-Y', '2026-05-25T10:01:00+00:00')])
        agg = self.brain.live_session_activity(node_ids={'node-X'})
        self.assertIn('node-X', agg)
        self.assertNotIn('node-Y', agg)

    def test_low_message_sessions_excluded(self):
        """Sessions below min_messages don't contribute to aggregation."""
        self._seed_session(
            'lo', 2,  # below min_messages=5
            [('node-X', '2026-05-25T10:00:00+00:00')])
        self._seed_session(
            'hi', 10,
            [('node-Y', '2026-05-25T10:01:00+00:00')])
        agg = self.brain.live_session_activity()
        self.assertNotIn('node-X', agg)
        self.assertIn('node-Y', agg)

    def test_persisted_session_loaded_when_not_in_cache(self):
        """Session whose ctx is not in the in-memory cache still aggregates
        — load() reads from session_state JSON. Covers the daemon-restart
        case Tom asked about ("if im off for a week ...")."""
        self._seed_session(
            'persisted', 10,
            [('node-X', '2026-05-25T10:00:00+00:00')])
        # Evict the cache entry — simulate post-restart / SessionEnd.
        self.brain._session_contexts.pop('persisted', None)
        agg = self.brain.live_session_activity()
        self.assertIn('node-X', agg,
                      'live aggregation must load from session_state '
                      'when the session is not in the in-memory cache')


if __name__ == '__main__':
    unittest.main()


class TestForegroundRecallBumpsActivity(BrainTestBase):
    """Behavioral test: brain.recall() with a session_id populates the
    matching SessionContext's node_activity dict for recalled nodes."""

    needs_embedder = True

    def test_recall_populates_session_node_activity(self):
        """After recall with session_id, ctx.node_activity holds records for
        the recalled nodes — bumped via _mark_accessed in the foreground path."""
        sid = 'sess-recall-test'

        n_a = self.brain.remember(
            type='rule', title='Recall populates activity — node A',
            content='input validation prevents injection attacks',
            keywords='validation security input')
        n_b = self.brain.remember(
            type='rule', title='Recall populates activity — node B',
            content='use parameterized queries to prevent sql injection',
            keywords='sql injection parameterized')

        # Pre-condition: this session's node_activity is empty.
        ctx_before = self.brain.get_or_create_session(sid)
        self.assertEqual(ctx_before.node_activity, {},
                         'Fresh session should have empty node_activity')

        result = self.brain.recall(
            query='how do we prevent injection attacks',
            session_id=sid, limit=10)
        recalled_ids = {r['id'] for r in result.get('results', [])}
        self.assertTrue(recalled_ids,
                        'recall returned no results — test query is wrong')

        ctx_after = self.brain.get_or_create_session(sid)
        # Every recalled node should have an activity record on the session.
        for nid in recalled_ids:
            self.assertIn(nid, ctx_after.node_activity,
                          'Recalled node %s missing from ctx.node_activity' % nid[:8])
            rec = ctx_after.node_activity[nid]
            self.assertGreaterEqual(rec['access_count'], 1)
            self.assertAlmostEqual(rec['recency_score'], 1.0)
            self.assertGreaterEqual(rec['activation'], 0.0)
            self.assertTrue(rec['last_accessed'],
                            'last_accessed should be a non-empty ISO ts')

    def test_two_sessions_recall_independent_activity(self):
        """Parallel sessions' node_activity dicts don't bleed into each other."""
        self.brain.remember(
            type='rule', title='Cross-session activity isolation A',
            content='alpha topic for session A',
            keywords='alpha session-a')
        self.brain.remember(
            type='rule', title='Cross-session activity isolation B',
            content='beta topic for session B',
            keywords='beta session-b')

        self.brain.recall(query='alpha topic', session_id='sess-A', limit=5)
        self.brain.recall(query='beta topic', session_id='sess-B', limit=5)

        ctx_a = self.brain.get_or_create_session('sess-A')
        ctx_b = self.brain.get_or_create_session('sess-B')

        # Each ctx records the nodes its own recall touched. The dicts
        # may overlap when both queries surface the same node (semantic
        # neighbors), so the isolation claim is: the two ctx dicts hold
        # INDEPENDENT records, not that they have disjoint keys.
        self.assertTrue(ctx_a.node_activity, 'A recall should populate ctx A')
        self.assertTrue(ctx_b.node_activity, 'B recall should populate ctx B')

        # Modifying one doesn't affect the other.
        ctx_a.bump_node_activity('synthetic-A-only', '2026-05-25T11:00:00+00:00')
        self.assertIn('synthetic-A-only', ctx_a.node_activity)
        self.assertNotIn('synthetic-A-only', ctx_b.node_activity,
                         'Mutations on ctx A leaked into ctx B — '
                         'parallel-session contamination')


if __name__ == '__main__':
    unittest.main()
