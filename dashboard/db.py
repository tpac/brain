"""DB path resolution + read-only SQLite connect helper.

The dashboard reads two databases:
  brain.db       — nodes, edges, embeddings
  brain_logs.db  — traces, hook errors, telemetry

All connections open in read-only mode (`mode=ro`) — the dashboard never
writes to brain. Writers go through the daemon.
"""

import os
import sqlite3
from contextlib import contextmanager

from .log import warn


def _brain_dir() -> str:
    return os.environ.get(
        "BRAIN_DB_DIR",
        os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"),
    )


def brain_db_path() -> str:
    return os.path.join(_brain_dir(), "brain.db")


def logs_db_path() -> str:
    return os.path.join(_brain_dir(), "brain_logs.db")


@contextmanager
def ro_connect(path: str, timeout: float = 3):
    """Open a read-only connection. Yields None if file is missing; the caller
    decides how to degrade (usually: return an empty list)."""
    if not os.path.exists(path):
        yield None
        return
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
    try:
        yield conn
    finally:
        conn.close()


def direct_query(sql: str, args=(), db_path: str = None):
    """Read-only query against brain.db (or any other path).

    Returns [] on missing file or any sqlite error — silent degradation is
    correct here because the dashboard must keep painting even when the
    underlying DB is being recreated.
    """
    path = db_path or brain_db_path()
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3)
        rows = conn.execute(sql, args).fetchall()
        conn.close()
        return rows
    except Exception as e:
        warn('db', 'direct_query against %s failed' % path, exc=e)
        return []
