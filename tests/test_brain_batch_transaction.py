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
from servers.daemon_dispatch import _handle_brain_batch, dispatch_command


def _error_rows(brain, source):
    """Error rows from debug_log for one `source` — what the dashboard reads."""
    rows = brain.logs_conn.execute(
        "SELECT source, metadata FROM debug_log "
        "WHERE event_type='error' AND source = ? ORDER BY id DESC",
        (source,)).fetchall()
    return [{'source': r[0], 'metadata': r[1]} for r in rows]


class TestBrainBatchTransaction(BrainTestBase):
    needs_embedder = False  # no embedding writes triggered by these tests

    def _batch(self, operations, **extra):
        """Helper — invoke _handle_brain_batch like the daemon does."""
        args = {"operations": operations}
        args.update(extra)
        return _handle_brain_batch(self.brain, args, [])

    def test_failed_sub_op_is_loud_through_the_chokepoint(self):
        """A brain_batch returns ok=True even when a sub-op carries ok=False.
        `dispatch_command` must scan for that and write a `batch_op_failed`
        error row — otherwise a per-op error string never reaches the errors
        table and error scans miss it entirely.

        Routed through `dispatch_command` on purpose, NOT `_handle_brain_batch`:
        the scan lives at the chokepoint, so calling the handler directly would
        pass while the real dispatch path was broken. This mechanism had no test
        when it was moved out of the encoder dispatch — this is that test.
        """
        before = len(_error_rows(self.brain, 'batch_op_failed'))

        r = dispatch_command(self.brain, 'brain_batch', {
            "operations": [
                {"op": "remember", "type": "rule", "title": "good node",
                 "content": "this op succeeds"},
                # `revise` with no node_id — fails the per-op required-field
                # pre-check, landing as ok=False inside an ok=True batch.
                {"op": "revise", "reason": "no node_id supplied"},
            ],
        }, [])

        # The batch as a whole reports success ...
        self.assertTrue(r['ok'])
        per_op = r['result']['results']
        self.assertTrue(any(o.get('ok') is False for o in per_op),
                        "expected one ok=False sub-op, got %s" % per_op)

        # ... and the buried failure is loud anyway.
        rows = _error_rows(self.brain, 'batch_op_failed')
        self.assertEqual(
            len(rows), before + 1,
            "a per-op failure inside an ok=True batch must write exactly one "
            "batch_op_failed row; saw %d new" % (len(rows) - before))
        # The row must name the op that failed, or it can't be acted on.
        self.assertIn('revise', rows[0]['metadata'])

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


class TestBatchArchiveAbsorbManifests(BrainTestBase):
    """Step 7 — the archive/absorb batch ops feed the mutation emitter, and
    the ORPHAN PROPERTY becomes provable: a batch that rolls back leaves zero
    emitter traces, because every mutation kind now flows through the manifest
    the chokepoint emits post-commit.

    Traces are asserted by id-set diff over trace_events, never by count
    (debug-log pruning makes global counts flaky — memory 2026-06-25)."""
    needs_embedder = False

    def _node(self, title, **kw):
        r = self.brain.remember(type='fact', title=title,
                                content='content of %s' % title,
                                encoding_source='anchor', **kw)
        return r['id']

    def _all_trace_ids(self):
        return {r[0] for r in self.brain.logs_conn.execute(
            "SELECT id FROM trace_events").fetchall()}

    def _dispatch_batch(self, operations, **extra):
        args = {"operations": operations}
        args.update(extra)
        return dispatch_command(self.brain, 'brain_batch', args, [])

    def test_archive_op_emits_node_archived_on_the_callers_chain(self):
        nid = self._node('arch-manifest-A')
        other = self._node('arch-manifest-other')
        self.brain._graph.add_relation(nid, other, 'depends_on')
        r = self._dispatch_batch(
            [{"op": "archive", "node_id": nid, "reason": "step7 probe"}],
            chain_id='test-step7-archive', session_id='sess-step7')
        self.assertTrue(r['ok'], r)
        rows = self.brain._trace_dal.get_chain('test-step7-archive')
        archived = [t for t in rows if t['ref_type'] == 'node_archived']
        self.assertEqual(len(archived), 1, rows)
        t = archived[0]
        self.assertEqual(t['ref_id'], nid)
        self.assertEqual(t['session_id'], 'sess-step7')
        meta = t['metadata']
        self.assertEqual(meta['node_id'], nid)
        self.assertEqual(meta['type'], 'fact')
        self.assertIn('arch-manifest-A', meta['title'])
        # 'unknown' mirrors what archive_node stored in _sys_archived_by —
        # _resolve_archived_by's fallback for an unstamped batch op. The trace
        # records the graph's truth, not a prettier guess; encoding_source
        # follows the same actor so one op's rows never split across scales.
        self.assertEqual(meta['archived_by'], 'unknown')
        self.assertEqual(meta['encoding_source'], 'unknown')
        self.assertEqual(meta['reason'], 'step7 probe')
        # The flipped (edge_id, relation) pairs — observed truth from the
        # UPDATE, not the raw edge list.
        self.assertEqual([p[1] for p in meta['edge_relations']],
                         ['depends_on'])
        # The manifest and its collections must NOT ride into the
        # agent-visible per-op result; the scalar counts stay.
        sub = r['result']['results'][0]
        for leaked in ('mutations', 'edge_relations', 'absorbed_into_edge'):
            self.assertNotIn(leaked, sub, sub)
        self.assertEqual(sub['edges_deleted'], 1)

    def test_absorb_op_emits_survivor_revise_and_migrated_edges(self):
        survivor = self._node('abs-manifest-survivor')
        absorbed = self._node('abs-manifest-absorbed')
        neighbor = self._node('abs-manifest-neighbor')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on',
                                       description='external — migrates')
        r = self._dispatch_batch(
            [{"op": "absorb", "survivor_id": survivor, "absorbed_id": absorbed,
              "content": 'merged synthesis body', "reason": "step7 merge"}],
            chain_id='test-step7-absorb', session_id='sess-step7')
        self.assertTrue(r['ok'], r)
        rows = self.brain._trace_dal.get_chain('test-step7-absorb')
        by_type = {}
        for t in rows:
            by_type.setdefault(t['ref_type'], []).append(t)
        # Survivor revise: the content override produced a real delta.
        revised = by_type.get('node_revised', [])
        self.assertEqual(len(revised), 1, rows)
        self.assertEqual(revised[0]['ref_id'], survivor)
        self.assertTrue(any(d.get('field') == 'content'
                            for d in revised[0]['metadata']['deltas']),
                        revised[0]['metadata'])
        # Two edge rows: the migrated depends_on (absorbed->neighbor
        # re-pointed to survivor->neighbor) + the absorbed_into redirect
        # minted by the internal archive.
        edges = by_type.get('edge_relation_revised', [])
        by_rel = {t['metadata']['relation']: t['metadata'] for t in edges}
        self.assertEqual(sorted(by_rel), ['absorbed_into', 'depends_on'], rows)
        em = by_rel['depends_on']
        self.assertEqual((em['source_id'], em['target_id']),
                         (survivor, neighbor))
        self.assertEqual(
            (by_rel['absorbed_into']['source_id'],
             by_rel['absorbed_into']['target_id']),
            (absorbed, survivor))
        # The row mirrors what absorb stamped on the migrated edge —
        # _resolve_archived_by's fallback for a bare op is 'unknown', and the
        # graph row carries exactly that.
        graph_es = self.brain.conn.execute(
            "SELECT er.encoding_source FROM edges e JOIN edge_relations er "
            "ON er.edge_id = e.edge_id WHERE e.source_id=? AND e.target_id=? "
            "AND er.relation='depends_on'", (survivor, neighbor)).fetchone()[0]
        self.assertEqual(em['encoding_source'], graph_es)
        # The absorbed node's archive rides the same chain, with the flipped
        # pair for its OLD (pre-migration) edge — the redirect edge is exempt
        # and must not be claimed.
        archived = by_type.get('node_archived', [])
        self.assertEqual(len(archived), 1, rows)
        am = archived[0]['metadata']
        self.assertEqual(am['node_id'], absorbed)
        self.assertEqual([p[1] for p in am['edge_relations']], ['depends_on'])
        # Nothing internal rides into the agent-visible per-op result.
        sub = r['result']['results'][0]
        for leaked in ('mutations', 'deltas', 'migrated_edges',
                       'absorbed_archive'):
            self.assertNotIn(leaked, sub, sub)

    def test_archive_op_with_survivor_records_lineage_and_traces_it(self):
        """Step 10 — the archive op carries optional survivor_id
        (supersession without a merge). The redirect must land in the graph
        (absorbed_into edge, live + exempt; _sys_archived_survivor_id) AND
        in the manifest (edge row on the caller's chain), with nothing
        leaking into the agent-visible result."""
        old = self._node('lineage-old')
        new = self._node('lineage-new')
        r = self._dispatch_batch(
            [{"op": "archive", "node_id": old, "reason": "superseded probe",
              "survivor_id": new}],
            chain_id='test-step10-lineage', session_id='sess-step10')
        self.assertTrue(r['ok'], r)
        sub = r['result']['results'][0]
        self.assertTrue(sub['ok'], sub)
        for leaked in ('mutations', 'edge_relations', 'absorbed_into_edge'):
            self.assertNotIn(leaked, sub, sub)

        # Graph truth: redirect edge live, pointer stored.
        row = self.brain.conn.execute(
            "SELECT er.archived FROM edges e JOIN edge_relations er "
            "ON er.edge_id = e.edge_id WHERE e.source_id=? AND e.target_id=? "
            "AND er.relation='absorbed_into'", (old, new)).fetchone()
        self.assertIsNotNone(row, 'absorbed_into redirect missing')
        self.assertEqual(row[0], 0, 'redirect must survive the edge sweep')
        ptr = self.brain._meta_kv.get_all_bulk([old])[old].get(
            '_sys_archived_survivor_id')
        self.assertEqual(ptr, new)

        # Trace truth: node_archived + the redirect edge row, same chain.
        rows = self.brain._trace_dal.get_chain('test-step10-lineage')
        kinds = {t['ref_type'] for t in rows}
        self.assertIn('node_archived', kinds, rows)
        edge_rows = [t for t in rows if t['ref_type'] == 'edge_relation_revised']
        self.assertEqual(len(edge_rows), 1, rows)
        em = edge_rows[0]['metadata']
        self.assertEqual((em['source_id'], em['target_id'], em['relation']),
                         (old, new, 'absorbed_into'))

    def test_archive_op_rejects_bogus_or_self_survivor(self):
        """Review 2026-08-07: archive_node stores the lineage pointer BEFORE
        any existence check (trusted-caller contract), so the op boundary
        must validate — a garbage survivor_id returning ok=True would leave
        a dead pointer that resolve_live drops as a permanent orphan."""
        target = self._node('bogus-survivor-target')
        r = self._dispatch_batch(
            [{"op": "archive", "node_id": target, "reason": "probe",
              "survivor_id": "ffffffffdeadbeef"}])
        sub = r['result']['results'][0]
        self.assertFalse(sub['ok'], sub)
        self.assertIn('survivor', sub['error'])
        # Nothing was written: node live, no pointer.
        self.assertEqual(self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?",
            (target,)).fetchone()[0], 0)
        self.assertNotIn('_sys_archived_survivor_id',
                         self.brain._meta_kv.get_all_bulk([target])
                         .get(target, {}))

        r = self._dispatch_batch(
            [{"op": "archive", "node_id": target, "reason": "probe",
              "survivor_id": target}])
        sub = r['result']['results'][0]
        self.assertFalse(sub['ok'], sub)
        self.assertIn('different node', sub['error'])
        self.assertEqual(self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?",
            (target,)).fetchone()[0], 0)

    def test_absorb_rows_follow_op_level_archived_by(self):
        """Review 2026-08-06: an op carrying only archived_by (no
        encoding_source anywhere) stamps the graph with it via
        _resolve_archived_by — the trace rows must carry the SAME value, or
        the trace contradicts the graph and scale routing follows the command
        runner instead of the stamped actor (s2:* rows landing on s0)."""
        survivor = self._node('attr-survivor')
        absorbed = self._node('attr-absorbed')
        neighbor = self._node('attr-neighbor')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on')
        r = self._dispatch_batch(
            [{"op": "absorb", "survivor_id": survivor, "absorbed_id": absorbed,
              "content": 'merged body', "reason": "attr probe",
              "archived_by": "s2:cleanup"}],
            chain_id='test-step7-attr')
        self.assertTrue(r['ok'], r)
        rows = self.brain._trace_dal.get_chain('test-step7-attr')
        self.assertTrue(rows, "no rows on the explicit chain")
        for t in rows:
            self.assertEqual(t['metadata']['encoding_source'], 's2:cleanup', t)
            self.assertEqual(t['scale'], 's2',
                             "row scale must follow the stamped actor")

    def test_rolled_back_batch_emits_zero_traces_of_any_kind(self):
        """THE ORPHAN TEST. A batch that rolls back after every kind of
        mutation ran (create, revise, archive, absorb, migrated edges) must
        leave zero trace rows OF ANY KIND — the chokepoint never sees a
        manifest because the handler re-raises past it, and no inline
        emitter exists anywhere below (the last one, archive_node's, died
        at step 8). Impossible to satisfy for 10 of the 12 legacy sites."""
        target = self._node('orphan-archive-target')
        survivor = self._node('orphan-survivor')
        absorbed = self._node('orphan-absorbed')
        neighbor = self._node('orphan-neighbor')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on')
        before = self._all_trace_ids()

        with patch.object(self.brain, '_apply_connect_to',
                          side_effect=RuntimeError('forced rollback')):
            with self.assertRaises(RuntimeError):
                self._dispatch_batch([
                    {"op": "remember", "type": "rule",
                     "title": "orphan-created", "content": "rolls back",
                     "connect_to": [{"title": "orphan-neighbor",
                                     "relation": "tests"}]},
                    {"op": "archive", "node_id": target, "reason": "rolls back"},
                    {"op": "absorb", "survivor_id": survivor,
                     "absorbed_id": absorbed, "content": "rolls back",
                     "reason": "rolls back"},
                ], chain_id='test-step7-orphan')

        self.assertEqual(self._all_trace_ids() - before, set(),
                         "a rolled-back batch left trace rows — orphaned "
                         "rows lie about the graph")
        # And the graph writes really did roll back.
        self.assertEqual(self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (target,)).fetchone()[0], 0)
        self.assertEqual(self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (absorbed,)).fetchone()[0], 0)

    def test_absorb_unwind_emits_zero_traces_for_the_merge(self):
        """An absorb whose archive leg fails unwinds its savepoint — the rest
        of the batch commits and traces, but the unwound merge must contribute
        nothing: no survivor revise row, no migrated-edge rows."""
        survivor = self._node('unwind-survivor')
        absorbed = self._node('unwind-absorbed')
        neighbor = self._node('unwind-neighbor')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on')

        with patch.object(self.brain, 'archive_node',
                          return_value={'ok': False, 'error': 'forced refusal'}):
            r = self._dispatch_batch([
                {"op": "absorb", "survivor_id": survivor,
                 "absorbed_id": absorbed, "content": "must not persist",
                 "reason": "unwind probe"},
                {"op": "remember", "type": "rule",
                 "title": "unwind-witness", "content": "this op commits"},
            ], chain_id='test-step7-unwind')

        self.assertTrue(r['ok'], r)
        subs = r['result']['results']
        self.assertFalse(subs[0].get('ok'), subs[0])
        self.assertTrue(subs[1].get('ok'), subs[1])

        rows = self.brain._trace_dal.get_chain('test-step7-unwind')
        kinds = sorted(t['ref_type'] for t in rows)
        self.assertEqual(kinds, ['node_created'],
                         "only the witness remember may trace; the unwound "
                         "merge contributed: %s" % kinds)
        # The unwound merge's internals must not leak into the agent result.
        for leaked in ('mutations', 'deltas', 'migrated_edges'):
            self.assertNotIn(leaked, subs[0], subs[0])
        # And the merge really unwound.
        c = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (survivor,)).fetchone()[0]
        self.assertNotEqual(c, 'must not persist')
        self.assertEqual(self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (absorbed,)).fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
