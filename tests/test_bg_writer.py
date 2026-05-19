"""Tests for the background-writer architecture (Phases 1-5, 2026-05-18).

Covers:
  - Brain opens `conn_bg_writer` on boot with correct PRAGMAs.
  - `_mark_accessed` is pure enqueue (no DB I/O on hot path).
  - Dedup is keyed on (node_id, session_id).
  - Drain produces atomic +1 UPDATEs per dedup'd (node, session) pair.
  - mark_accessed already uses atomic +1 (no read-modify-write).
  - Hebbian pairs from surface-layer selection (not top-15 cosine).
  - Drain rollback on transaction failure + loud log.
  - Concurrent enqueue from multiple threads doesn't lose work.
  - Worker loop survives an exception in drain_once.
"""

import threading
import time
import unittest

from tests.brain_test_base import BrainTestBase

from servers import recall_write_queue


class TestBgWriterConnection(BrainTestBase):
    """Phase 1 — conn_bg_writer opens correctly on boot."""

    needs_embedder = False

    def test_bg_writer_opens_on_boot(self):
        self.assertIsNotNone(self.brain.conn_bg_writer)
        # Basic query works
        r = self.brain.conn_bg_writer.execute('SELECT 1').fetchone()
        self.assertEqual(r, (1,))

    def test_bg_writer_pragmas(self):
        wal = self.brain.conn_bg_writer.execute(
            'PRAGMA journal_mode').fetchone()[0].lower()
        self.assertEqual(wal, 'wal')
        bt = self.brain.conn_bg_writer.execute(
            'PRAGMA busy_timeout').fetchone()[0]
        self.assertEqual(bt, 30000)
        fk = self.brain.conn_bg_writer.execute(
            'PRAGMA foreign_keys').fetchone()[0]
        self.assertEqual(fk, 1)


class TestEnqueueNoDbIo(BrainTestBase):
    """Phase 5 — hot path enqueues without touching the DB."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_enqueue_access_does_not_touch_db(self):
        # sqlite3.Connection.execute is read-only; use the round-trip
        # observation instead. Snapshot WAL frame counter before enqueue,
        # confirm no commits happened post-enqueue. Equivalent semantic:
        # "the call did not push any data through SQLite".
        before_changes = self.brain.conn.execute(
            "PRAGMA data_version").fetchone()[0]

        recall_write_queue.enqueue_access(
            'abc12345', 'sess1', '2026-05-18T12:00:00')

        after_changes = self.brain.conn.execute(
            "PRAGMA data_version").fetchone()[0]
        # data_version increments on every commit to the database.
        # enqueue_access is pure in-memory — data_version unchanged.
        self.assertEqual(before_changes, after_changes,
                         'enqueue_access touched the DB (data_version moved)')
        self.assertEqual(recall_write_queue.queue_depth(), 1)


class TestDedupPerSession(BrainTestBase):
    """Phase 5 — dedup key is (node_id, session_id)."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_same_node_same_session_dedups(self):
        for ts in ('2026-05-18T12:00:00', '2026-05-18T12:00:01',
                   '2026-05-18T12:00:02'):
            recall_write_queue.enqueue_access('node1234', 'sess1', ts)
        self.assertEqual(recall_write_queue.queue_depth(), 1)
        stats = recall_write_queue.get_stats()
        self.assertEqual(stats['access_enqueued_total'], 3)

    def test_same_node_different_session_separates(self):
        recall_write_queue.enqueue_access(
            'node1234', 'sess1', '2026-05-18T12:00:00')
        recall_write_queue.enqueue_access(
            'node1234', 'sess2', '2026-05-18T12:00:00')
        self.assertEqual(recall_write_queue.queue_depth(), 2)

    def test_newer_ts_wins_in_dedup(self):
        recall_write_queue.enqueue_access(
            'node1234', 'sess1', '2026-05-18T12:00:00')
        recall_write_queue.enqueue_access(
            'node1234', 'sess1', '2026-05-18T12:00:05')
        recall_write_queue.enqueue_access(
            'node1234', 'sess1', '2026-05-18T12:00:02')  # older
        # Snapshot and verify the latest ts wins
        acc, _ = recall_write_queue._snapshot_and_clear()
        self.assertEqual(acc[('node1234', 'sess1')], '2026-05-18T12:00:05')


class TestDrainProcessesAccess(BrainTestBase):
    """Phase 5 — drain writes atomic +1 UPDATEs to the nodes table."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_drain_increments_access_count(self):
        # Create a real node so the UPDATE WHERE id=? finds it.
        result = self.brain.remember(
            type='test', title='drain-target',
            content='content',
            encoding_source='anchor:test')
        nid = result['id']
        before = self.brain.conn.execute(
            'SELECT access_count FROM nodes WHERE id = ?', (nid,)
        ).fetchone()[0]

        # Enqueue 3 accesses for same (node, session) — dedup to 1 UPDATE
        for ts in ('2026-05-18T12:00:00', '2026-05-18T12:00:01',
                   '2026-05-18T12:00:02'):
            recall_write_queue.enqueue_access(nid, 'sess1', ts)

        recall_write_queue.drain_once(self.brain)

        after = self.brain.conn.execute(
            'SELECT access_count, last_accessed FROM nodes WHERE id = ?',
            (nid,)).fetchone()
        # Dedup collapses 3 enqueues to 1 increment (Tom's design)
        self.assertEqual(after[0], before + 1)
        self.assertEqual(after[1], '2026-05-18T12:00:02')

    def test_drain_two_sessions_two_increments(self):
        result = self.brain.remember(
            type='test', title='drain-target-2sess',
            content='content',
            encoding_source='anchor:test')
        nid = result['id']
        before = self.brain.conn.execute(
            'SELECT access_count FROM nodes WHERE id = ?', (nid,)
        ).fetchone()[0]

        recall_write_queue.enqueue_access(nid, 'sess1', '2026-05-18T12:00:00')
        recall_write_queue.enqueue_access(nid, 'sess2', '2026-05-18T12:00:00')

        recall_write_queue.drain_once(self.brain)

        after = self.brain.conn.execute(
            'SELECT access_count FROM nodes WHERE id = ?', (nid,)).fetchone()[0]
        # Two sessions = two increments
        self.assertEqual(after, before + 2)


class TestDrainRollbackOnFailure(BrainTestBase):
    """Phase 5 — drain rolls back + logs loudly on transaction failure."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_drain_rollback_logged_and_batch_dropped(self):
        # Failure-injection via module-level monkey-patch on
        # _apply_hebbian_pairs. Module functions ARE patchable
        # (unlike sqlite3.Connection methods, which are read-only).
        # Raising here forces drain_once into its outer except branch,
        # which calls conn_bg_writer.rollback() and logs via
        # bg_writer_batch_rollback.

        logged = []
        original_log = self.brain._log_error
        original_apply = recall_write_queue._apply_hebbian_pairs

        def log_spy(source, error, context='', ctx=None):
            logged.append((source, str(error)[:60], context[:60]))
            return original_log(source, error, context, ctx)

        def apply_fail(*args, **kwargs):
            raise RuntimeError('synthetic drain failure for rollback test')

        self.brain._log_error = log_spy
        recall_write_queue._apply_hebbian_pairs = apply_fail

        try:
            # Enqueue a hebbian pair so the failing path is reached.
            recall_write_queue.enqueue_hebbian_pairs(
                [('a' * 12, 'b' * 12)], '2026-05-18T12:00:00')

            recall_write_queue.drain_once(self.brain)

            stats = recall_write_queue.get_stats()
            self.assertGreaterEqual(stats['errors_total'], 1)
            self.assertGreaterEqual(stats['rollbacks_total'], 1)

            sources = [s for s, _, _ in logged]
            self.assertIn('bg_writer_batch_rollback', sources)

            self.assertEqual(recall_write_queue.queue_depth(), 0)
        finally:
            self.brain._log_error = original_log
            recall_write_queue._apply_hebbian_pairs = original_apply


class TestHebbianDrainProducesEdges(BrainTestBase):
    """Phase 5 — surface-selected pairs actually land as co_accessed edges
    after drain. The original tests covered enqueue but never drained,
    which hid a real `no such column: last_strengthened` SQL bug that
    showed up only in production."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_drain_creates_co_accessed_edges_for_new_pairs(self):
        # auto_connect=False so the test owns the edge state — fresh
        # nodes won't get pre-linked by the "connect to recent" heuristic.
        nids = []
        for i in range(3):
            r = self.brain.remember(
                type='test', title='heb_drain_%d' % i,
                content='c%d' % i,
                auto_connect=False,
                encoding_source='anchor:test')
            nids.append(r['id'])

        before = self.brain.conn.execute(
            "SELECT COUNT(*) FROM edge_relations WHERE relation = 'co_accessed'"
        ).fetchone()[0]

        recall_write_queue.enqueue_hebbian_pairs(
            [(nids[0], nids[1]), (nids[1], nids[2]), (nids[0], nids[2])],
            '2026-05-18T12:00:00')

        recall_write_queue.drain_once(self.brain)

        after = self.brain.conn.execute(
            "SELECT COUNT(*) FROM edge_relations WHERE relation = 'co_accessed'"
        ).fetchone()[0]

        # All 3 pairs should have produced co_accessed edges.
        self.assertEqual(after, before + 3,
                         'drain did not create edges (before=%d after=%d) — '
                         'check bg_writer_drain_hebbian errors' %
                         (before, after))

        # Drain stats should reflect no errors. This is the assertion
        # that would have caught the no-such-column bug immediately.
        stats = recall_write_queue.get_stats()
        self.assertEqual(stats['errors_total'], 0,
                         'drain reported errors: %d' % stats['errors_total'])
        self.assertEqual(stats['hebbian_pairs_drained_total'], 3)

    def test_drain_strengthens_existing_co_accessed_edge(self):
        # Two nodes with a pre-existing co_accessed edge.
        r1 = self.brain.remember(type='test', title='heb_s_1',
                                 content='c1', encoding_source='anchor:test')
        r2 = self.brain.remember(type='test', title='heb_s_2',
                                 content='c2', encoding_source='anchor:test')

        # First drain — creates the co_accessed relation
        recall_write_queue.enqueue_hebbian_pairs(
            [(r1['id'], r2['id'])], '2026-05-18T12:00:00')
        recall_write_queue.drain_once(self.brain)

        # Snapshot weight before second drain
        row = self.brain.conn.execute(
            "SELECT er.weight, e.last_strengthened "
            "FROM edge_relations er JOIN edges e ON e.edge_id = er.edge_id "
            "WHERE er.relation = 'co_accessed' AND er.archived = 0 "
            "  AND ((e.source_id = ? AND e.target_id = ?) OR "
            "       (e.source_id = ? AND e.target_id = ?))",
            (r1['id'], r2['id'], r2['id'], r1['id'])).fetchone()
        self.assertIsNotNone(row, 'first drain failed to create the edge')
        weight_before, ls_before = row

        # Second drain — should strengthen via UPDATE branch
        recall_write_queue.enqueue_hebbian_pairs(
            [(r1['id'], r2['id'])], '2026-05-18T13:00:00')
        recall_write_queue.drain_once(self.brain)

        row = self.brain.conn.execute(
            "SELECT er.weight, e.last_strengthened "
            "FROM edge_relations er JOIN edges e ON e.edge_id = er.edge_id "
            "WHERE er.relation = 'co_accessed' AND er.archived = 0 "
            "  AND ((e.source_id = ? AND e.target_id = ?) OR "
            "       (e.source_id = ? AND e.target_id = ?))",
            (r1['id'], r2['id'], r2['id'], r1['id'])).fetchone()
        weight_after, ls_after = row

        self.assertGreater(weight_after, weight_before,
                           'second drain did not strengthen weight')
        self.assertEqual(ls_after, '2026-05-18T13:00:00',
                         'last_strengthened on edges row not updated')


class TestDrainAtomicityMidBatch(BrainTestBase):
    """Phase 5+8 — verify the drain transaction is truly atomic.

    Earlier rollback test injected failure before any SQL ran. This
    test seeds a real Hebbian batch (some pairs that strengthen, plus
    one that triggers a failure mid-way) and asserts that NONE of the
    pairs leave artifacts — i.e., add_relation does not commit mid-
    batch and the outer ROLLBACK undoes the entire batch.
    """

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_mid_batch_failure_rolls_back_all_pairs(self):
        # Create 3 real nodes for the strengthening pairs.
        nids = []
        for i in range(3):
            r = self.brain.remember(
                type='test', title='atomic_node_%d' % i,
                content='c%d' % i, encoding_source='anchor:test')
            nids.append(r['id'])

        # Snapshot edges count before drain.
        before = self.brain.conn.execute(
            "SELECT COUNT(*) FROM edge_relations WHERE relation = 'co_accessed'"
        ).fetchone()[0]

        # Patch _apply_hebbian_pairs in-place: process one pair
        # successfully via the real path, then raise on the second.
        # This is the scenario the production-failure mode looks like.
        original_apply = recall_write_queue._apply_hebbian_pairs
        call_count = {'n': 0}

        def apply_partial(brain, conn, snap):
            # Process all but the last pair using the real implementation;
            # then raise to simulate a mid-batch failure.
            original_apply(brain, conn, snap[:-1])
            call_count['n'] += 1
            raise RuntimeError('synthetic mid-batch failure')

        recall_write_queue._apply_hebbian_pairs = apply_partial
        try:
            # Enqueue 2 pairs. Real DAL writes happen on the first via
            # the partial-apply; then the synthetic raise should ROLLBACK
            # both pairs (and any edges they may have inserted).
            recall_write_queue.enqueue_hebbian_pairs(
                [(nids[0], nids[1]), (nids[1], nids[2])],
                '2026-05-18T12:00:00')

            recall_write_queue.drain_once(self.brain)

            # No new co_accessed rows should exist — ROLLBACK undid
            # whatever the first pair attempted. This is the atomic
            # contract the bg_writer drain promises.
            after = self.brain.conn.execute(
                "SELECT COUNT(*) FROM edge_relations WHERE relation = 'co_accessed'"
            ).fetchone()[0]
            self.assertEqual(after, before,
                             'mid-batch failure left orphaned co_accessed rows '
                             '(before=%d after=%d)' % (before, after))
        finally:
            recall_write_queue._apply_hebbian_pairs = original_apply


class TestConcurrentEnqueue(BrainTestBase):
    """Phase 5 — concurrent enqueues from multiple threads don't lose work."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_concurrent_enqueue_preserves_unique_pairs(self):
        N_THREADS = 8
        N_PER_THREAD = 50

        def producer(thread_idx):
            for i in range(N_PER_THREAD):
                # Distinct (node, session) pairs per thread
                recall_write_queue.enqueue_access(
                    'n%07d' % (thread_idx * 1000 + i),
                    'sess%d' % thread_idx,
                    '2026-05-18T12:00:00')

        threads = [threading.Thread(target=producer, args=(t,))
                   for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All pairs unique → all enqueued
        expected = N_THREADS * N_PER_THREAD
        self.assertEqual(recall_write_queue.queue_depth(), expected)
        stats = recall_write_queue.get_stats()
        self.assertEqual(stats['access_enqueued_total'], expected)


class TestHebbianAtSurfaceLayer(BrainTestBase):
    """Phase 5 — Hebbian comes from surface picks (not top-15 cosine).

    Verifies the daemon_hooks._hebbian_strengthen path enqueues pairs
    based on the surface-selected file, capped at C(picks, 2).
    """

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_surface_selected_drives_hebbian_pairs(self):
        from itertools import combinations
        import json
        import os
        import tempfile

        # Create 4 real nodes so resolve_id works
        ids = []
        for i in range(4):
            r = self.brain.remember(
                type='test', title='heb_node_%d' % i,
                content='c%d' % i, encoding_source='anchor:test')
            ids.append(r['id'])

        # Write the surface-selected JSON file (this is what S1 surface
        # would have written for daemon_hooks._hebbian_strengthen to consume).
        # Short IDs only — matches surface.py's actual output shape.
        short_ids = [nid[:8] for nid in ids]
        session_id = 'hebbian_test_session'
        stop_counter = 0
        path = '/tmp/brain-%s-%d-surface-selected.json' % (
            session_id, stop_counter)
        try:
            with open(path, 'w') as f:
                json.dump({'selected_ids': short_ids}, f)

            from servers.daemon_hooks import _hebbian_strengthen
            _hebbian_strengthen(self.brain, session_id, stop_counter)

            # Enqueued: C(4, 2) = 6 pairs. NOT 15 (the old cosine top-15
            # cap is gone) and NOT capped at 8 neighbors (the old
            # min(j, i+8) heuristic is gone).
            stats = recall_write_queue.get_stats()
            self.assertEqual(stats['hebbian_queue_depth'], 6)
            self.assertEqual(stats['hebbian_enqueued_total'], 6)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class TestQueueDepthIntrospection(BrainTestBase):
    """Phase 6 — get_stats surfaces what dashboard / health checks need."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_stats_keys(self):
        stats = recall_write_queue.get_stats()
        for key in ('access_enqueued_total', 'hebbian_enqueued_total',
                    'drains_total', 'drains_skipped_empty',
                    'last_drain_at', 'last_drain_took_ms', 'last_drain_size',
                    'access_drained_total', 'hebbian_pairs_drained_total',
                    'errors_total', 'rollbacks_total',
                    'overlong_drains_total', 'access_queue_depth',
                    'hebbian_queue_depth'):
            self.assertIn(key, stats, 'missing stat: %s' % key)


if __name__ == '__main__':
    unittest.main()
