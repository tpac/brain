"""Tests for the brain_batch single-transaction contract.

Before the refactor each sub-op in brain_batch committed independently
— N ops = N commits = N WAL writer-slot grabs, no rollback semantic.
This test suite locks in the new contract:

1. A batch fires brain.conn.commit() exactly once for the whole batch
   (per-op commits are no-ops while brain.conn.in_batch is True).
2. A failure that escapes the per-op try/except (e.g., during deferred
   connect_to resolution) rolls back EVERY op in the batch.
3. brain.conn.in_batch is always reset to False after the function returns,
   even on exception.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.daemon_dispatch import _handle_brain_batch


class TestBrainBatchTransaction(BrainTestBase):
    needs_embedder = False  # no embedding writes triggered by these tests

    def _batch(self, operations, **extra):
        """Helper — invoke _handle_brain_batch like the daemon does."""
        args = {"operations": operations}
        args.update(extra)
        return _handle_brain_batch(self.brain, args, [])

    def test_single_commit_per_batch(self):
        """All N ops share one commit, not N commits. We count real
        COMMITs via sqlite3.Connection.set_trace_callback (the conn's
        Python methods are read-only built-ins and can't be patched)."""
        statements = []
        self.brain.conn.set_trace_callback(statements.append)
        try:
            r = self._batch([
                {"op": "remember", "type": "rule", "title": "node A",
                 "content": "first node in batch"},
                {"op": "remember", "type": "rule", "title": "node B",
                 "content": "second node in batch"},
                {"op": "remember", "type": "rule", "title": "node C",
                 "content": "third node in batch"},
            ])
        finally:
            self.brain.conn.set_trace_callback(None)

        self.assertTrue(r['ok'])
        self.assertEqual(r['result']['succeeded'], 3)
        # Count COMMIT statements observed in the trace.
        commits = sum(1 for s in statements if s.strip().upper() == 'COMMIT')
        self.assertEqual(commits, 1,
                         "Expected one COMMIT for the batch, got %d. "
                         "Statements: %s" % (commits, statements[-20:]))
        # And exactly one BEGIN IMMEDIATE opens the transaction.
        begins = sum(1 for s in statements if 'BEGIN IMMEDIATE' in s.upper())
        self.assertEqual(begins, 1,
                         "Expected one BEGIN IMMEDIATE, got %d" % begins)

    def test_batch_mode_resets_after_success(self):
        """conn.in_batch must end False even after a happy-path return."""
        self.assertFalse(self.brain.conn.in_batch)
        self._batch([
            {"op": "remember", "type": "rule", "title": "x", "content": "y"},
        ])
        self.assertFalse(self.brain.conn.in_batch)

    def test_batch_mode_resets_after_outer_exception(self):
        """conn.in_batch must end False even when an exception propagates."""
        self.assertFalse(self.brain.conn.in_batch)

        with patch.object(self.brain, '_apply_connect_to',
                          side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                # Two ops; one has connect_to to force _apply_connect_to to fire.
                self._batch([
                    {"op": "remember", "type": "rule",
                     "title": "src", "content": "source node",
                     "connect_to": [{"title": "tgt", "relation": "tests"}]},
                    {"op": "remember", "type": "rule",
                     "title": "tgt", "content": "target node"},
                ])

        self.assertFalse(self.brain.conn.in_batch,
                         "conn.in_batch leaked True after exception")

    def test_outer_exception_rolls_back_whole_batch(self):
        """If _apply_connect_to raises, NEITHER remember in the batch persists."""
        precount = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE title LIKE 'rollback-test-%'"
        ).fetchone()[0]
        self.assertEqual(precount, 0)

        with patch.object(self.brain, '_apply_connect_to',
                          side_effect=RuntimeError('forced rollback')):
            with self.assertRaises(RuntimeError):
                self._batch([
                    {"op": "remember", "type": "rule",
                     "title": "rollback-test-A",
                     "content": "should be rolled back",
                     "connect_to": [{"title": "rollback-test-B",
                                     "relation": "tests"}]},
                    {"op": "remember", "type": "rule",
                     "title": "rollback-test-B",
                     "content": "also should be rolled back"},
                ])

        postcount = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE title LIKE 'rollback-test-%'"
        ).fetchone()[0]
        self.assertEqual(postcount, 0,
                         "Both nodes should have been rolled back; %d found" % postcount)

    def test_recovers_from_leaked_transaction(self):
        """Guard: if self.conn is already mid-transaction at entry (an upstream
        op left a deferred auto-BEGIN open without committing), brain_batch must
        flush the orphan, recover, and commit its own ops cleanly — NOT throw
        'cannot start a transaction within a transaction'.

        Regression for the S2 multi-pass community crash surfaced by the Frozen
        Corpus eval build. See docs/WRITE-TXN-ISOLATION-ROOTFIX.md.
        """
        # Simulate the leak: open a transaction + a DML, do NOT commit.
        self.brain.conn.execute('BEGIN')
        self.brain.conn.execute("UPDATE nodes SET title = title WHERE 0")
        self.assertTrue(self.brain.conn.in_transaction,
                        "precondition: connection should be mid-transaction")

        r = self._batch([
            {"op": "remember", "type": "rule",
             "title": "leaked-txn-recovery", "content": "should persist"},
        ])

        self.assertTrue(r['ok'], "batch should recover, not crash")
        self.assertEqual(r['result']['succeeded'], 1)
        self.assertFalse(self.brain.conn.in_transaction,
                         "connection should be clean after the batch")
        self.assertFalse(self.brain.conn.in_batch, "conn.in_batch must reset")
        persisted = self.brain.conn.execute(
            "SELECT 1 FROM nodes WHERE title = 'leaked-txn-recovery' AND archived = 0"
        ).fetchone()
        self.assertIsNotNone(persisted, "the batch op should have persisted")

    def test_per_op_failure_still_commits_other_ops(self):
        """Per-op exceptions stay in results — the batch as a whole still commits
        the ops that succeeded. (Best-effort surface preserved.)"""
        r = self._batch([
            {"op": "remember", "type": "rule",
             "title": "happy-A", "content": "first ok"},
            # Invalid op — per-op handler catches, batch continues.
            {"op": "remember"},  # missing required type/title
            {"op": "remember", "type": "rule",
             "title": "happy-B", "content": "second ok"},
        ])

        self.assertTrue(r['ok'])
        # Two of three succeeded.
        succeeded_titles = [
            self.brain.conn.execute(
                "SELECT 1 FROM nodes WHERE title = ? AND archived = 0", (t,)
            ).fetchone()
            for t in ('happy-A', 'happy-B')
        ]
        self.assertTrue(all(succeeded_titles),
                        "Both happy ops should have persisted")

    # ── F3: GraphDAL writers must not self-commit inside a batch ──────────────
    # Regression for docs/WRITE-TXN-ISOLATION-ROOTFIX.md. test_single_commit_per_batch
    # above uses only plain `remember` ops, so it never exercised the connect /
    # disconnect / source-ref paths whose DAL writers (add_relation via
    # connect_typed, remove_relation, add_source_refs/replace_source_refs) used to
    # self-commit mid-batch — silently breaking both the single-COMMIT property and
    # the all-or-nothing rollback guarantee. These tests drive those paths.

    def _commit_count(self, operations, **extra):
        """Run a batch and return (result, number of real COMMITs observed)."""
        statements = []
        self.brain.conn.set_trace_callback(statements.append)
        try:
            r = self._batch(operations, **extra)
        finally:
            self.brain.conn.set_trace_callback(None)
        commits = sum(1 for s in statements if s.strip().upper() == 'COMMIT')
        return r, commits

    def _node_id(self, title):
        row = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE title = ? AND archived = 0", (title,)
        ).fetchone()
        return row[0] if row else None

    def _tests_relation_count(self, a, b, relation='tests'):
        """Count active edge_relations named `relation` between a and b in either
        direction. remember() may auto-bridge similar nodes, so a physical edge
        can pre-exist — we assert on the SPECIFIC relation the connect op writes,
        not bare edge existence."""
        return self.brain.conn.execute(
            "SELECT COUNT(*) FROM edge_relations er JOIN edges e ON er.edge_id = e.edge_id "
            "WHERE er.relation = ? AND er.archived = 0 AND "
            "((e.source_id = ? AND e.target_id = ?) OR (e.source_id = ? AND e.target_id = ?))",
            (relation, a, b, b, a)).fetchone()[0]

    def test_connect_op_single_commit(self):
        """`connect` routes through connect_typed -> add_relation, which must defer
        its commit inside a batch (gated on conn.in_batch). One COMMIT total."""
        self.brain.remember(type='rule', title='f3-conn-A', content='a')
        self.brain.remember(type='rule', title='f3-conn-B', content='b')
        a, b = self._node_id('f3-conn-A'), self._node_id('f3-conn-B')
        r, commits = self._commit_count([
            {"op": "connect", "source_id": a, "target_id": b, "relation": "tests"},
        ])
        self.assertTrue(r['ok'])
        self.assertEqual(commits, 1,
                         "connect op self-committed inside the batch; got %d COMMITs" % commits)

    def test_disconnect_op_single_commit(self):
        """`disconnect` routes through remove_relation, which must defer its commit."""
        self.brain.remember(type='rule', title='f3-disc-A', content='a')
        self.brain.remember(type='rule', title='f3-disc-B', content='b')
        a, b = self._node_id('f3-disc-A'), self._node_id('f3-disc-B')
        self.brain.connect_typed(a, b, 'tests')  # pre-existing edge to remove
        r, commits = self._commit_count([
            {"op": "disconnect", "source_id": a, "target_id": b, "relation": "tests"},
        ])
        self.assertTrue(r['ok'])
        self.assertEqual(commits, 1,
                         "disconnect op self-committed inside the batch; got %d COMMITs" % commits)

    def test_remember_with_source_refs_single_commit(self):
        """A `remember` carrying source_refs routes through add_source_refs, which
        must defer its commit."""
        r, commits = self._commit_count([
            {"op": "remember", "type": "rule", "title": "f3-srcrefs",
             "content": "node with refs", "source_refs": ["aabbccdd", "11223344"]},
        ])
        self.assertTrue(r['ok'])
        self.assertEqual(commits, 1,
                         "source_refs write self-committed inside the batch; got %d COMMITs" % commits)

    def test_connect_rolled_back_on_outer_failure(self):
        """The prime F3 leak: connect_typed -> add_relation used to self-commit, so
        a later op's failure could NOT roll the edge back. Now the edge must vanish
        with the batch's ROLLBACK."""
        self.brain.remember(type='rule', title='f3-rb-A', content='a')
        self.brain.remember(type='rule', title='f3-rb-B', content='b')
        a, b = self._node_id('f3-rb-A'), self._node_id('f3-rb-B')
        self.assertEqual(self._tests_relation_count(a, b), 0,
                         "precondition: no 'tests' relation between A and B yet")

        with patch.object(self.brain, '_apply_connect_to',
                          side_effect=RuntimeError('forced rollback')):
            with self.assertRaises(RuntimeError):
                self._batch([
                    {"op": "connect", "source_id": a, "target_id": b,
                     "relation": "tests"},
                    {"op": "remember", "type": "rule", "title": "f3-rb-C",
                     "content": "triggers connect_to resolution failure",
                     "connect_to": [{"title": "f3-rb-A", "relation": "tests"}]},
                ])

        self.assertEqual(self._tests_relation_count(a, b), 0,
                         "connect op did not roll back with the batch — 'tests' relation persisted")


    # ── More brain_batch coverage — it's load-bearing infrastructure ──────────
    # Phase-1 F3 covered connect/disconnect/source-ref single-commit + rollback.
    # These extend to the paths the F3 fix originally MISSED (co_anchored auto-
    # edges, the untyped connect() helper) plus mixed-op and revise.

    def test_co_anchored_op_single_commit(self):
        """A remember op whose source_refs overlap an existing node fires the
        co_anchored auto-edge (brain_remember.py). That add_relation must defer its
        commit so the batch COMMITs once. Regression: it used to self-commit."""
        self.brain.remember(type='rule', title='co-anchor-A', content='anchor',
                            source_refs=['deadbeef'])
        a = self._node_id('co-anchor-A')
        self.assertIsNotNone(a)
        r, commits = self._commit_count([
            {"op": "remember", "type": "rule", "title": "co-anchor-B",
             "content": "shares the anchor ref", "source_refs": ["deadbeef"]},
        ])
        self.assertTrue(r['ok'])
        self.assertEqual(commits, 1,
                         "co_anchored auto-edge self-committed inside the batch; got %d COMMITs" % commits)
        b = self._node_id('co-anchor-B')
        self.assertGreater(self._tests_relation_count(a, b, 'co_anchored'), 0,
                           "co_anchored edge B->A should exist (proves the branch ran)")

    def test_co_anchored_rolls_back_on_outer_failure(self):
        """The co_anchored auto-edge must roll back with the batch when a later op
        fails — it used to self-commit and survive the rollback."""
        self.brain.remember(type='rule', title='co-rb-A', content='anchor',
                            source_refs=['cafebabe'])
        with patch.object(self.brain, '_apply_connect_to',
                          side_effect=RuntimeError('forced rollback')):
            with self.assertRaises(RuntimeError):
                self._batch([
                    {"op": "remember", "type": "rule", "title": "co-rb-B",
                     "content": "shares anchor", "source_refs": ["cafebabe"]},
                    {"op": "remember", "type": "rule", "title": "co-rb-C",
                     "content": "triggers connect_to failure",
                     "connect_to": [{"title": "co-rb-A", "relation": "tests"}]},
                ])
        self.assertIsNone(self._node_id('co-rb-B'),
                          "co-rb-B (and its co_anchored edge) must roll back with the batch")

    def test_connections_op_in_batch_retired_no_edge_single_commit(self):
        """Evolved contract (2026-06-18): the `connections=` store-time edge param
        was retired (connect_to replaced it). A brain_batch remember op carrying it
        now materializes NO edge, yet the batch still commits exactly once. Pins the
        BATCH dispatch path (brain_batch op -> _handle_remember -> remember()); the
        direct-call retirement contract is covered by
        test_core.py::test_remember_connections_param_retired_is_loud_not_silent.
        Was: asserted the connections edge PERSISTED via the now-removed connect()
        store path (assertGreater(.., 0))."""
        self.brain.remember(type='rule', title='conns-target', content='target')
        tgt = self._node_id('conns-target')
        r, commits = self._commit_count([
            {"op": "remember", "type": "rule", "title": "conns-src",
             "content": "carries the retired connections param",
             "connections": [{"target_id": tgt, "relation": "tests"}]},
        ])
        self.assertTrue(r['ok'])
        # Retired param must not trigger an extra self-commit inside the batch.
        self.assertEqual(commits, 1,
                         "batch must commit exactly once; got %d" % commits)
        src = self._node_id('conns-src')
        # connections= is retired — the edge must NOT be created (was >0).
        self.assertEqual(self._tests_relation_count(src, tgt, 'tests'), 0,
                         "retired connections= must NOT materialize an edge")

    def test_mixed_op_batch_single_commit(self):
        """Core contract: a batch mixing remember + connect + disconnect + archive
        must COMMIT exactly once across all op types."""
        self.brain.remember(type='rule', title='mix-A', content='a')
        self.brain.remember(type='rule', title='mix-B', content='b')
        self.brain.remember(type='rule', title='mix-C', content='to archive')
        a, b, c = self._node_id('mix-A'), self._node_id('mix-B'), self._node_id('mix-C')
        self.brain.connect_typed(a, b, 'old_rel')  # pre-existing edge to disconnect
        r, commits = self._commit_count([
            {"op": "remember", "type": "rule", "title": "mix-D", "content": "new node"},
            {"op": "connect", "source_id": a, "target_id": b, "relation": "new_rel"},
            {"op": "disconnect", "source_id": a, "target_id": b, "relation": "old_rel"},
            {"op": "archive", "node_id": c, "reason": "test"},
        ])
        self.assertTrue(r['ok'])
        self.assertEqual(commits, 1,
                         "mixed-op batch must COMMIT exactly once; got %d" % commits)

    def test_revise_op_single_commit(self):
        """A revise op must COMMIT exactly once inside a batch."""
        self.brain.remember(type='rule', title='rev-X', content='original')
        x = self._node_id('rev-X')
        r, commits = self._commit_count([
            {"op": "revise", "node_id": x, "reason": "test revise",
             "content": "revised content"},
        ])
        self.assertTrue(r['ok'])
        self.assertEqual(commits, 1, "revise op must COMMIT once; got %d" % commits)


if __name__ == '__main__':
    unittest.main()
