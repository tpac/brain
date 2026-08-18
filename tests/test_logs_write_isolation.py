"""Write-boundary guardrails for brain_logs.db (the SQLITE_BUSY_SNAPSHOT fix).

The bug (brain id:371895a8): brain_logs.db writes ran on the same shared
connection that serves concurrent reads. An open read cursor pins a WAL
snapshot on the connection; once ANY other connection commits (a hook
process, the MCP monitor), a write on the snapshot-holding connection is a
read->write upgrade from a stale snapshot — SQLite fails it INSTANTLY with
'database is locked', never invoking the busy handler. busy_timeout offers
zero protection. Every such failure silently dropped a trace event.

The fix: each logs DAL routes writes to a dedicated write connection used
only under the write lock, so no read cursor can ever exist on it at write
time. These tests pin both halves: the mechanism (single shared connection
still fails — if SQLite/Python semantics ever change, we learn) and the fix
(split-connection DAL survives the exact production interleave).

Run: ./dev pytest tests/test_logs_write_isolation.py -v
"""
import os
import sqlite3
import tempfile
import threading
import unittest

from servers.dal_logs import LogsDAL, TraceDAL, SessionStateDAL
from servers.schema import ensure_logs_schema
from tests.brain_test_base import BrainTestBase


def _connect(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute('PRAGMA journal_mode = WAL')
    conn.execute('PRAGMA busy_timeout = 200')
    return conn


class LogsWriteIsolationBase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, 'brain_logs.db')
        self.read_conn = _connect(self.db_path)
        ensure_logs_schema(self.read_conn, db_path=self.db_path)
        self.write_conn = _connect(self.db_path)
        self.external = _connect(self.db_path)  # models a hook process
        self.lock = threading.RLock()

    def tearDown(self):
        for conn in (self.read_conn, self.write_conn, self.external):
            try:
                conn.close()
            except Exception:
                pass
        self.tmpdir.cleanup()

    def _seed_traces(self, dal, n=300):
        dal.append_batch([{
            'chain_id': 'seed', 'scale': 's0', 'event_type': 'O',
            'ref_type': 'user_message', 'ref_id': '', 'summary': 'row %d' % i,
            'metadata': None, 'session_id': 'seed'} for i in range(n)])

    def _poison(self):
        """Reproduce the production interleave: pin a read snapshot with a
        live, partially-fetched cursor, then advance the WAL from an external
        connection. Returns the open cursor (kept alive by the caller)."""
        cur = self.read_conn.execute(
            'SELECT id, summary FROM trace_events')
        cur.fetchone()  # partially fetched -> statement stays open
        self.external.execute(
            "INSERT INTO debug_log (event_type, source, created_at) "
            "VALUES ('e', 'external', '2026-01-01T00:00:00+00:00')")
        self.external.commit()
        return cur


class TestMechanismStillExists(LogsWriteIsolationBase):
    """Pin the failure mode itself on a single shared connection. If this
    ever fails, SQLite/Python cursor semantics changed and the split (plus
    this file) should be re-evaluated."""

    def test_single_connection_write_fails_instantly(self):
        dal = TraceDAL(self.read_conn)  # pre-fix wiring: one shared conn
        self._seed_traces(dal)
        cur = self._poison()
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            dal.append(chain_id='c1', scale='s0', event_type='O',
                       ref_type='user_message', session_id='s')
        self.assertIn('locked', str(ctx.exception))
        cur.close()


class TestSplitConnectionSurvives(LogsWriteIsolationBase):
    """The fix: the exact same interleave, with the DAL wired the way
    Brain.__init__ wires it (read conn + write conn + shared lock)."""

    def _dal(self, cls):
        return cls(self.read_conn, write_conn=self.write_conn,
                   write_lock=self.lock)

    def test_trace_append_survives_poisoned_read_snapshot(self):
        dal = self._dal(TraceDAL)
        self._seed_traces(dal)
        cur = self._poison()
        event_id = dal.append(chain_id='c1', scale='s0', event_type='O',
                              ref_type='user_message', session_id='s')
        self.assertTrue(event_id)
        row = self.external.execute(
            'SELECT chain_id FROM trace_events WHERE id = ?',
            (event_id,)).fetchone()
        self.assertEqual(row[0], 'c1')
        cur.close()

    def test_store_embeddings_survives_poisoned_read_snapshot(self):
        dal = self._dal(TraceDAL)
        self._seed_traces(dal, n=50)
        tid = dal.append(chain_id='c2', scale='s0', event_type='O',
                         ref_type='user_message', session_id='s')
        cur = self._poison()
        n = dal.store_embeddings([(tid, b'\x00\x01', 'text')], model='m')
        self.assertEqual(n, 1)
        cur.close()

    def test_debug_log_and_session_state_survive(self):
        logs = self._dal(LogsDAL)
        state = self._dal(SessionStateDAL)
        trace = self._dal(TraceDAL)
        self._seed_traces(trace, n=50)
        cur = self._poison()
        logs.write_event('error', 'test', {'k': 'v'}, session_id='s')
        state.set('sess', 'key', 'value')
        cur.close()
        self.assertEqual(self.external.execute(
            "SELECT COUNT(*) FROM debug_log WHERE source='test'"
        ).fetchone()[0], 1)
        self.assertEqual(self.external.execute(
            "SELECT value FROM session_state WHERE session_id='sess'"
        ).fetchone()[0], 'value')

    def test_concurrent_readers_writers_external_commits(self):
        """Stress the production topology: reader threads iterating live
        cursors on the read conn, writer threads appending through the DAL,
        an external committer advancing the WAL. Zero drops expected."""
        dal = self._dal(TraceDAL)
        self._seed_traces(dal)
        errors = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                cur = self.read_conn.execute(
                    'SELECT id, summary FROM trace_events')
                cur.fetchone()          # hold a snapshot briefly
                cur.fetchall()

        def external_committer():
            conn = _connect(self.db_path)
            while not stop.is_set():
                conn.execute(
                    "INSERT INTO debug_log (event_type, source, created_at) "
                    "VALUES ('e', 'ext', '2026-01-01T00:00:00+00:00')")
                conn.commit()
            conn.close()

        def writer(k):
            try:
                for i in range(25):
                    dal.append(chain_id='w%d' % k, scale='s0',
                               event_type='O', ref_type='user_message',
                               session_id='s%d' % k)
            except Exception as e:
                errors.append(e)

        side = [threading.Thread(target=reader, daemon=True),
                threading.Thread(target=external_committer, daemon=True)]
        writers = [threading.Thread(target=writer, args=(k,))
                   for k in range(4)]
        for t in side + writers:
            t.start()
        for t in writers:
            t.join(timeout=60)
        stop.set()
        for t in side:
            t.join(timeout=10)

        self.assertEqual(errors, [], 'dropped writes: %r' % errors)
        n = self.external.execute(
            "SELECT COUNT(*) FROM trace_events WHERE chain_id LIKE 'w%'"
        ).fetchone()[0]
        self.assertEqual(n, 100)


class TestBrainWiresTheSplit(BrainTestBase):
    """Pin the Brain.__init__ wiring itself. The DAL tests above construct
    split-connection DALs by hand — a revert of the brain.py kwargs would
    leave them green while production silently fell back to single-connection
    semantics (the whole bug, undetected). This closes that gap."""
    needs_embedder = False

    def test_every_logs_dal_gets_write_conn_and_logs_lock(self):
        b = self.brain
        for name in ('_logs_dal', '_trace_dal', '_interaction_dal',
                     '_session_state'):
            dal = getattr(b, name)
            self.assertIs(dal.wconn, b.logs_conn_w,
                          '%s.wconn is not brain.logs_conn_w' % name)
            self.assertIsNot(dal.wconn, b.logs_conn,
                             '%s writes on the shared read connection' % name)
            self.assertIs(dal._wlock, b.logs_write_lock,
                          '%s does not serialize on logs_write_lock' % name)


if __name__ == '__main__':
    unittest.main()
