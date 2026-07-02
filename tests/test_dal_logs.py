"""Tests for LogsDAL writers in servers/dal_logs.py.

LogsDAL takes a raw sqlite connection, so these run against an in-memory DB —
no Brain, no embedder. Fast + isolated.
"""
import sqlite3
from servers.dal_logs import LogsDAL


def _dal():
    return LogsDAL(sqlite3.connect(':memory:'))


class TestLogHookError:
    """log_hook_error is the in-process route to hook_errors. The MCP health
    monitor (brain_mcp.py) writes DAEMON_DOWN through this instead of raw SQL,
    so the hook_errors write SQL lives in exactly one DAL."""

    def test_writes_row_with_fields(self):
        dal = _dal()
        dal.log_hook_error('DAEMON_DOWN', 'daemon unreachable',
                            context='mcp_health_monitor', level='critical')
        row = dal.conn.execute(
            "SELECT hook_name, level, error, context FROM hook_errors").fetchone()
        assert row == ('DAEMON_DOWN', 'critical', 'daemon unreachable', 'mcp_health_monitor')

    def test_creates_table_if_absent(self):
        # No CREATE beforehand — the method must self-create hook_errors (the
        # daemon-down path can hit a brain_logs.db that predates the table).
        dal = _dal()
        dal.log_hook_error('h', 'e')  # must not raise
        assert dal.conn.execute("SELECT count(*) FROM hook_errors").fetchone()[0] == 1

    def test_created_at_is_iso_t_not_space_separated(self):
        # Guards the time-window contract: hook_errors.created_at is lex-compared
        # by the dashboard's `created_at > ?`, so it MUST be ISO-T (T-separated),
        # never SQLite's space-separated datetime('now') form.
        dal = _dal()
        dal.log_hook_error('h', 'e')
        ts = dal.conn.execute("SELECT created_at FROM hook_errors").fetchone()[0]
        assert 'T' in ts and ' ' not in ts

    def test_prunes_to_most_recent_200(self):
        dal = _dal()
        for i in range(205):
            dal.log_hook_error('h%d' % i, 'e')
        assert dal.conn.execute("SELECT count(*) FROM hook_errors").fetchone()[0] == 200
