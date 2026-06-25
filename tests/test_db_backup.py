"""Tests for db_backup — GFS retention policy + the snapshot round-trip.

What this locks down:
1. select_retained() implements GFS correctly per tier (daily/weekly/
   monthly) and dedups the union.
2. list_backups() parses only this module's `{base}.{ts}.gz` files and
   ignores hand-made `.bak*` files (retention must never touch them).
3. backup_database() produces a gzip that decompresses to a valid SQLite
   DB with the original rows (the online-backup + gzip path works), and
   prunes to the retained set.
4. seconds_since_last_backup() reads the newest snapshot's age.
"""

import gzip
import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from servers import db_backup
from servers.db_backends import sqlite as sqlite_backend

UTC = timezone.utc


class TestGFSRetention(unittest.TestCase):
    def test_daily_tier_keeps_newest_per_day(self):
        base = datetime(2026, 6, 25, tzinfo=UTC)
        # Three snapshots on the same day; daily=1 keeps only the newest.
        ts = [base.replace(hour=9), base.replace(hour=12), base.replace(hour=15)]
        retained = db_backup.select_retained(ts, keep_daily=1, keep_weekly=0,
                                             keep_monthly=0)
        self.assertEqual(retained, {base.replace(hour=15)})

    def test_weekly_tier_keeps_newest_n_weeks(self):
        # One snapshot per ISO week (7-day spacing), 6 weeks back.
        base = datetime(2026, 6, 25, 12, tzinfo=UTC)
        ts = [base - timedelta(days=7 * i) for i in range(6)]
        retained = db_backup.select_retained(ts, keep_daily=0, keep_weekly=4,
                                             keep_monthly=0)
        self.assertEqual(retained, set(ts[:4]))   # 4 most recent weeks

    def test_monthly_tier_keeps_newest_n_months(self):
        ts = [datetime(2026, m, 25, 12, tzinfo=UTC) for m in (6, 5, 4, 3, 2, 1)]
        retained = db_backup.select_retained(ts, keep_daily=0, keep_weekly=0,
                                             keep_monthly=3)
        self.assertEqual(retained, set(ts[:3]))   # Jun, May, Apr

    def test_union_dedups_and_bounds(self):
        # 120 consecutive daily snapshots. Union of 7+4+3 tiers with heavy
        # overlap (recent days are also recent weeks/months).
        base = datetime(2026, 6, 25, 3, tzinfo=UTC)
        ts = [base - timedelta(days=i) for i in range(120)]
        retained = db_backup.select_retained(ts, keep_daily=7, keep_weekly=4,
                                             keep_monthly=3)
        # Newest is always kept; the 7 most recent days are all kept.
        for i in range(7):
            self.assertIn(base - timedelta(days=i), retained)
        # Union never exceeds the sum of tier sizes.
        self.assertLessEqual(len(retained), 7 + 4 + 3)
        # A mid-range daily that is not any tier's representative is dropped.
        self.assertNotIn(base - timedelta(days=40), retained)


class TestListBackups(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ignores_handmade_bak_files(self):
        db_path = os.path.join(self.tmp, 'brain.db')
        # Two real auto-backups + clutter that must be ignored.
        for name in ('brain.db.20260625T120000Z.gz',
                     'brain.db.20260624T120000Z.gz',
                     'brain.db.bak-20260101-orphan-audit',     # hand-made
                     'brain.db.bak-20260101-orphan-audit.gz',  # no ts segment
                     'brain_logs.db.20260625T120000Z.gz'):     # different DB
            open(os.path.join(self.tmp, name), 'w').close()
        found = db_backup.list_backups(db_path, self.tmp)
        names = sorted(os.path.basename(p) for _, p in found)
        self.assertEqual(names, ['brain.db.20260624T120000Z.gz',
                                 'brain.db.20260625T120000Z.gz'])
        # Newest first ordering.
        self.assertEqual(os.path.basename(found[0][1]),
                         'brain.db.20260625T120000Z.gz')

    def test_seconds_since_last_backup(self):
        db_path = os.path.join(self.tmp, 'brain.db')
        self.assertEqual(
            db_backup.seconds_since_last_backup(db_path, self.tmp), float('inf'))
        open(os.path.join(self.tmp, 'brain.db.20260625T120000Z.gz'), 'w').close()
        now = datetime(2026, 6, 25, 13, tzinfo=UTC)   # 1h after the snapshot
        age = db_backup.seconds_since_last_backup(db_path, self.tmp, now=now)
        self.assertEqual(age, 3600.0)


class TestBackupRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        conn = sqlite3.connect(self.db_path)
        sqlite_backend.apply_pragmas(conn)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)')
        conn.executemany('INSERT INTO t (v) VALUES (?)',
                         [('row-%d' % i,) for i in range(200)])
        conn.commit()
        conn.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_snapshot_decompresses_to_valid_db_with_rows(self):
        backup_dir = os.path.join(self.tmp, 'backups')
        now = datetime(2026, 6, 25, 12, tzinfo=UTC)
        result = db_backup.backup_database(self.db_path, backup_dir, now=now)

        gz_path = os.path.join(backup_dir, result['dest'])
        self.assertTrue(os.path.exists(gz_path))
        self.assertGreater(result['raw_bytes'], 0)
        self.assertGreater(result['gz_bytes'], 0)
        self.assertEqual(result['kept'], 1)
        self.assertEqual(result['deleted'], 0)
        # No stray intermediates left behind.
        self.assertFalse(os.path.exists(gz_path + '.tmp.db'))
        self.assertFalse(os.path.exists(gz_path + '.part'))

        # Decompress and verify it's a real SQLite DB with all the rows.
        restored = os.path.join(self.tmp, 'restored.db')
        with gzip.open(gz_path, 'rb') as f_in, open(restored, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        conn = sqlite3.connect(restored)
        try:
            count = conn.execute('SELECT COUNT(*) FROM t').fetchone()[0]
            self.assertEqual(count, 200)
            self.assertEqual(
                conn.execute('SELECT v FROM t WHERE id=1').fetchone()[0], 'row-0')
        finally:
            conn.close()

    def test_prune_drops_non_retained_on_second_run(self):
        backup_dir = os.path.join(self.tmp, 'backups')
        # Two snapshots on the same day; daily-keep collapses to the newest,
        # so the second run should prune the first.
        db_backup.backup_database(self.db_path, backup_dir,
                                  now=datetime(2026, 6, 25, 9, tzinfo=UTC))
        res2 = db_backup.backup_database(
            self.db_path, backup_dir,
            keep_daily=1, keep_weekly=0, keep_monthly=0,
            now=datetime(2026, 6, 25, 15, tzinfo=UTC))
        self.assertEqual(res2['kept'], 1)
        self.assertEqual(res2['deleted'], 1)
        remaining = sorted(os.listdir(backup_dir))
        self.assertEqual(remaining, ['brain.db.20260625T150000Z.gz'])

    def test_prune_never_deletes_newest_even_with_zero_keep(self):
        """Defensive guard: an all-zero keep config must not wipe the
        just-created snapshot — the newest is always retained."""
        backup_dir = os.path.join(self.tmp, 'backups')
        db_backup.backup_database(self.db_path, backup_dir,
                                  now=datetime(2026, 6, 24, 12, tzinfo=UTC))
        res = db_backup.backup_database(
            self.db_path, backup_dir,
            keep_daily=0, keep_weekly=0, keep_monthly=0,
            now=datetime(2026, 6, 25, 12, tzinfo=UTC))
        # Older one pruned, newest survives — never zero backups.
        self.assertEqual(res['kept'], 1)
        self.assertEqual(os.listdir(backup_dir), ['brain.db.20260625T120000Z.gz'])


if __name__ == '__main__':
    unittest.main()
