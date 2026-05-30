"""Contract: write-path transaction discipline is STRUCTURAL, not by-convention.

brain_batch (and the bg-writer drain) wrap many sub-ops in one BEGIN IMMEDIATE
/ COMMIT envelope. For that to be atomic, the DAL writers running inside must
NOT self-commit. Pre-2026-05-30 this was enforced by a fragile convention:
every writer took a `commit` kwarg and every batch-context caller had to
remember `commit=not _batch_mode`. We shipped exactly the bug that convention
invites — 3 callers forgot the kwarg; only a code review caught it.

The structural fix (docs/WRITE-TXN-ISOLATION-ROOTFIX.md, Option A): batch state
lives on the CONNECTION (BatchAwareConnection.in_batch); the envelope owner
flips it; writers consult it via commit_unless_batched(). There is no kwarg to
forget. These tests lock that so a future writer can't reintroduce the gap:

1. commit_unless_batched honors conn.in_batch (behavioral).
2. No DAL writer self-commits via a bare `self.conn.commit()` — all route
   through the helper (source contract — fails on a planted regression).
3. The GraphDAL batch-reachable writers expose no `commit` parameter (the old
   forgettable knob is gone — signature contract).
4. The Brain's three SQLite connections are BatchAwareConnection, so in_batch
   is always present in production (wiring contract).
"""

import os
import re
import sys
import glob
import inspect
import sqlite3
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.db_backends.sqlite import BatchAwareConnection, commit_unless_batched
from servers.dal import GraphDAL

_SERVERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'servers')


class TestCommitHelperBehavior(unittest.TestCase):
    """commit_unless_batched is the single gate — it must read conn.in_batch."""

    def _fresh_conn(self):
        conn = sqlite3.connect(':memory:', factory=BatchAwareConnection)
        conn.execute('CREATE TABLE t (x INTEGER)')
        conn.commit()
        return conn

    def test_commits_when_not_batched(self):
        conn = self._fresh_conn()
        conn.execute('INSERT INTO t (x) VALUES (1)')
        self.assertTrue(conn.in_transaction, "DML should open a deferred txn")
        commit_unless_batched(conn)  # in_batch defaults False
        self.assertFalse(conn.in_transaction,
                         "standalone write must commit (in_batch=False)")

    def test_defers_when_batched(self):
        conn = self._fresh_conn()
        conn.in_batch = True
        conn.execute('INSERT INTO t (x) VALUES (1)')
        commit_unless_batched(conn)
        self.assertTrue(conn.in_transaction,
                        "must NOT commit inside a batch (in_batch=True)")
        conn.rollback()  # the envelope owner would commit/rollback; prove it CAN

    def test_safe_on_plain_connection(self):
        """A non-BatchAware connection has no in_batch attr — getattr default
        False → treated as standalone → commits. (Maintenance/test paths.)"""
        plain = sqlite3.connect(':memory:')
        plain.execute('CREATE TABLE t (x INTEGER)')
        plain.execute('INSERT INTO t (x) VALUES (1)')
        commit_unless_batched(plain)  # must not raise
        self.assertFalse(plain.in_transaction,
                         "plain conn should commit (safe default)")


class TestNoSelfCommitInDAL(unittest.TestCase):
    """Source contract: no DAL writer may self-commit via `self.conn.commit()`.
    Every commit routes through commit_unless_batched(self.conn) so batch
    atomicity can't be broken by a writer that bypasses the gate."""

    def test_no_bare_self_conn_commit(self):
        offenders = []
        for path in glob.glob(os.path.join(_SERVERS_DIR, 'dal*.py')):
            with open(path, encoding='utf-8') as f:
                for lineno, line in enumerate(f, 1):
                    if 'self.conn.commit()' in line:
                        offenders.append('%s:%d' % (os.path.basename(path), lineno))
        self.assertEqual(
            offenders, [],
            "DAL writers must commit via commit_unless_batched(self.conn), not a "
            "bare self.conn.commit() — that bypasses the batch gate and breaks "
            "brain_batch atomicity. Offending lines: %s" % offenders)


class TestGraphWritersHaveNoCommitKwarg(unittest.TestCase):
    """Signature contract: the batch-reachable GraphDAL writers expose no
    `commit` parameter. The old forgettable knob is gone — the writer reads
    conn.in_batch itself, so a caller can't pass the wrong value."""

    BATCH_REACHABLE_WRITERS = [
        'add_relation', 'remove_relation', 'delete_node_edges',
        'decay_edges', 'add_source_refs', 'replace_source_refs',
    ]

    def test_writers_have_no_commit_param(self):
        for name in self.BATCH_REACHABLE_WRITERS:
            method = getattr(GraphDAL, name)
            params = inspect.signature(method).parameters
            self.assertNotIn(
                'commit', params,
                "GraphDAL.%s still takes a `commit` kwarg — the structural fix "
                "removes it; the writer must gate on conn.in_batch instead." % name)


class TestBrainConnectionsAreBatchAware(BrainTestBase):
    """Wiring contract: the Brain's SQLite connections carry in_batch so the
    gate is always live in production (not just when a stray getattr defaults)."""
    needs_embedder = False

    def test_all_three_connections_batch_aware(self):
        for attr in ('conn', 'conn_bg_writer', 'logs_conn'):
            conn = getattr(self.brain, attr)
            self.assertIsInstance(
                conn, BatchAwareConnection,
                "brain.%s must be a BatchAwareConnection so conn.in_batch is "
                "always present" % attr)
            self.assertFalse(conn.in_batch,
                             "brain.%s.in_batch must default False" % attr)


if __name__ == '__main__':
    unittest.main()
