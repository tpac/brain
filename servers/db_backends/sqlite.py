"""SQLite backend ops + pragma centralization.

Single source of truth for:
- Connection pragmas applied at every `sqlite3.connect` site in the brain
- Maintenance operations (checkpoint, optimize) invoked by the scheduler

If we ever swap SQLite for another store, this is the file that gets
replaced. The brain calls `apply_pragmas(conn)` from this module; the
scheduler calls `checkpoint(db_path)` / `optimize(db_path)`. Neither
caller knows about SQLite-specific syntax.

Pragma reasoning (see CLAUDE.md for the full table):
- busy_timeout=30000: tolerate short-lived WAL writer-slot contention
  before raising "database is locked". Default 0 fails immediately.
- cache_size=-65536: 64 MB per-connection page cache (default is 2 MB).
  Brain.db is ~340 MB; this caches the hot working set.
- mmap_size=256 MB: memory-map the DB file so reads go through page
  cache instead of read() syscalls. Big win for read-heavy recall.
- temp_store=MEMORY: temp tables (ORDER BY without index, complex
  joins) live in RAM, not on disk.
- synchronous=NORMAL: with WAL, NORMAL only fsyncs at checkpoint, not
  at every commit. 5-10x faster commits with no durability loss for a
  daemon that can replay from journal. This pragma matters most for
  reducing time-under-write-lock during commit-heavy workloads.
- journal_mode=WAL: concurrent readers + faster commits. Set per-file
  (not per-connection) so we only need it once per DB; included here
  for completeness and so fresh DBs get it on first connect.
- foreign_keys=ON: enforce FK constraints (off by default in SQLite!).
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, Any


# Per-connection pragmas. Applied at every sqlite3.connect site via
# apply_pragmas(conn). Reset when the connection closes, so every new
# connection needs them.
_CONNECTION_PRAGMAS = (
    'PRAGMA busy_timeout = 30000',
    'PRAGMA cache_size = -65536',          # 64 MB
    'PRAGMA mmap_size = 268435456',        # 256 MB
    'PRAGMA temp_store = MEMORY',
    'PRAGMA synchronous = NORMAL',
    'PRAGMA foreign_keys = ON',
)

# Per-database pragmas. Persisted in the DB file; setting from any
# connection sticks across processes.
_DATABASE_PRAGMAS = (
    'PRAGMA journal_mode = WAL',
)


def apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply the standard pragma set to a SQLite connection.

    Idempotent. Call once per connection, immediately after connect().
    Order matters: journal_mode goes first because cache_size and
    mmap_size behave differently in DELETE vs WAL.
    """
    for stmt in _DATABASE_PRAGMAS:
        conn.execute(stmt)
    for stmt in _CONNECTION_PRAGMAS:
        conn.execute(stmt)


# ─── Maintenance ops ──────────────────────────────────────────────────
# Each op opens a short-lived connection, runs its work, closes. This
# keeps the maintenance thread decoupled from the daemon's primary
# connections — no shared state, no contention with the worker, no
# lock plumbing across the module boundary.


def checkpoint(db_path: str) -> Dict[str, Any]:
    """`PRAGMA wal_checkpoint(TRUNCATE)` — flush WAL into the DB and
    truncate the WAL file. Without periodic checkpointing the WAL
    grows unbounded; growing WAL means longer recovery on every
    connection open, which compounds writer-slot wait time.

    Returns: dict with 'busy', 'log_pages', 'checkpointed_pages',
    plus 'wal_size_before' / 'wal_size_after' for observability.
    """
    wal_path = db_path + '-wal'
    size_before = _file_size_or_zero(wal_path)
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        apply_pragmas(conn)
        row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
    finally:
        conn.close()
    size_after = _file_size_or_zero(wal_path)
    busy, log_pages, ckpt_pages = (row or (None, None, None))
    return {
        'busy': busy,
        'log_pages': log_pages,
        'checkpointed_pages': ckpt_pages,
        'wal_size_before': size_before,
        'wal_size_after': size_after,
    }


def optimize(db_path: str) -> Dict[str, Any]:
    """`PRAGMA optimize` — let SQLite refresh query planner statistics
    for tables whose row distribution has shifted enough to matter.
    Cheap to run periodically (no-op for stable tables); expensive
    queries become quietly faster as statistics catch up to reality.
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        apply_pragmas(conn)
        conn.execute('PRAGMA optimize')
    finally:
        conn.close()
    return {'ok': True}


def stats(db_path: str) -> Dict[str, Any]:
    """Diagnostic snapshot — DB size, WAL size, page count, freelist.
    Cheap to collect; useful for tracking DB growth between ticks.
    """
    db_size = _file_size_or_zero(db_path)
    wal_size = _file_size_or_zero(db_path + '-wal')
    shm_size = _file_size_or_zero(db_path + '-shm')

    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        apply_pragmas(conn)
        page_count = conn.execute('PRAGMA page_count').fetchone()[0]
        page_size = conn.execute('PRAGMA page_size').fetchone()[0]
        freelist = conn.execute('PRAGMA freelist_count').fetchone()[0]
    finally:
        conn.close()
    return {
        'db_size_bytes': db_size,
        'wal_size_bytes': wal_size,
        'shm_size_bytes': shm_size,
        'page_count': page_count,
        'page_size': page_size,
        'freelist_pages': freelist,
        'pages_in_use_pct': round(
            (page_count - freelist) / max(1, page_count) * 100, 1),
    }


# ─── Helpers ──────────────────────────────────────────────────────────

def _file_size_or_zero(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
