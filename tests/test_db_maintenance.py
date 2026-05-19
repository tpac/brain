"""Tests for db_maintenance scheduler + db_backends.sqlite pragma application.

What this locks down:
1. apply_pragmas sets the documented per-connection knobs (not the
   SQLite defaults) on every fresh sqlite3.connect.
2. checkpoint() actually shrinks the WAL file after writes.
3. The scheduler fires registered ops at the configured cadence and
   catches per-op failures without killing the loop.
"""

import os
import sqlite3
import tempfile
import threading
import time
import unittest

from servers.db_backends import sqlite as sqlite_backend
from servers.db_maintenance import DBMaintenance


class TestApplyPragmas(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'test.db')

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        sqlite_backend.apply_pragmas(conn)
        return conn

    def test_pragmas_match_documented_values(self):
        conn = self._open()
        try:
            self.assertEqual(conn.execute('PRAGMA busy_timeout').fetchone()[0], 30000)
            # cache_size in negative form = -kibibytes; -65536 = 64 MB
            self.assertEqual(conn.execute('PRAGMA cache_size').fetchone()[0], -65536)
            self.assertEqual(conn.execute('PRAGMA mmap_size').fetchone()[0], 268435456)
            # temp_store: 0=DEFAULT, 1=FILE, 2=MEMORY
            self.assertEqual(conn.execute('PRAGMA temp_store').fetchone()[0], 2)
            # synchronous: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
            self.assertEqual(conn.execute('PRAGMA synchronous').fetchone()[0], 1)
            self.assertEqual(conn.execute('PRAGMA foreign_keys').fetchone()[0], 1)
            self.assertEqual(
                conn.execute('PRAGMA journal_mode').fetchone()[0].lower(), 'wal')
        finally:
            conn.close()

    def test_pragmas_idempotent(self):
        conn = self._open()
        try:
            sqlite_backend.apply_pragmas(conn)
            sqlite_backend.apply_pragmas(conn)
            # Still correct after double-apply.
            self.assertEqual(conn.execute('PRAGMA synchronous').fetchone()[0], 1)
        finally:
            conn.close()


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_truncates_wal(self):
        """A held reader keeps the WAL from auto-truncating, so we can
        observe checkpoint(TRUNCATE) actually shrinking it. Without the
        held reader, SQLite's auto-checkpoint trigger fires on commit
        once WAL exceeds 1000 pages and the test becomes flaky."""
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'test.db')

        writer = sqlite3.connect(db_path)
        sqlite_backend.apply_pragmas(writer)
        writer.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, payload BLOB)')
        # Reader on a separate connection — its existence prevents the
        # auto-checkpoint from truncating WAL during writes.
        reader = sqlite3.connect(db_path)
        sqlite_backend.apply_pragmas(reader)
        reader.execute('SELECT COUNT(*) FROM t').fetchone()

        try:
            blob = b'x' * 4096
            for _ in range(500):
                writer.execute('INSERT INTO t (payload) VALUES (?)', (blob,))
            writer.commit()

            wal_path = db_path + '-wal'
            wal_before = os.path.getsize(wal_path)
            self.assertGreater(wal_before, 0,
                               "Held reader should leave a non-empty WAL")

            # Release the reader so checkpoint can truncate.
            reader.close()
            reader = None

            result = sqlite_backend.checkpoint(db_path)
            self.assertEqual(result['busy'], 0)
            self.assertEqual(result['wal_size_after'], 0)
            self.assertGreater(result['wal_size_before'], 0)
        finally:
            writer.close()
            if reader is not None:
                reader.close()


class TestOptimize(unittest.TestCase):
    def test_optimize_runs_without_error(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'test.db')
        conn = sqlite3.connect(db_path)
        sqlite_backend.apply_pragmas(conn)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER)')
        conn.commit()
        conn.close()
        result = sqlite_backend.optimize(db_path)
        self.assertTrue(result.get('ok'))


class TestStats(unittest.TestCase):
    def test_stats_returns_documented_fields(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'test.db')
        conn = sqlite3.connect(db_path)
        sqlite_backend.apply_pragmas(conn)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY)')
        conn.commit()
        conn.close()
        s = sqlite_backend.stats(db_path)
        for key in ('db_size_bytes', 'wal_size_bytes', 'page_count',
                    'page_size', 'freelist_pages', 'pages_in_use_pct'):
            self.assertIn(key, s)


class TestSchedulerFiresOps(unittest.TestCase):
    def test_due_op_fires_within_one_tick(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'test.db')
        conn = sqlite3.connect(db_path)
        sqlite_backend.apply_pragmas(conn)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY)')
        conn.commit()
        conn.close()

        fired = {'checkpoint': 0, 'optimize': 0}

        class StubBackend:
            def apply_pragmas(self, conn):
                pass
            def checkpoint(self, p):
                fired['checkpoint'] += 1
                return {}
            def optimize(self, p):
                fired['optimize'] += 1
                return {}
            def stats(self, p):
                return {}

        m = DBMaintenance(
            checkpoint_interval_s=0.05,   # ~immediately due
            optimize_interval_s=0.05,
            tick_interval_s=0.05,
        )
        m._backend = StubBackend()
        m.register('test', db_path)
        m.start()
        try:
            # Two ticks gives the scheduler time to fire both ops.
            time.sleep(0.25)
        finally:
            m.stop()

        self.assertGreaterEqual(fired['checkpoint'], 1)
        self.assertGreaterEqual(fired['optimize'], 1)

    def test_op_exception_does_not_kill_loop(self):
        """A failing op must be caught + logged; loop continues."""
        fired = {'count': 0}

        class FlakeyBackend:
            def apply_pragmas(self, conn):
                pass
            def checkpoint(self, p):
                fired['count'] += 1
                if fired['count'] == 1:
                    raise RuntimeError('first call fails')
                return {}
            def optimize(self, p):
                return {}
            def stats(self, p):
                return {}

        logged = []
        m = DBMaintenance(
            log_fn=lambda msg: logged.append(msg),
            log_error_fn=lambda origin, exc, ctx: logged.append('ERR %s' % origin),
            checkpoint_interval_s=0.05,
            optimize_interval_s=999.0,
            tick_interval_s=0.05,
        )
        m._backend = FlakeyBackend()
        m.register('test', '/dev/null')
        m.start()
        try:
            time.sleep(0.4)
        finally:
            m.stop()

        # First call raised; subsequent calls must have happened.
        self.assertGreaterEqual(fired['count'], 2)
        # The error was logged via log_error_fn (not silently swallowed).
        self.assertTrue(any('ERR db_maintenance_checkpoint' in s for s in logged),
                        "Expected ERR log, got: %s" % logged)


if __name__ == '__main__':
    unittest.main()
