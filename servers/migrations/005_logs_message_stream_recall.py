"""
Migration 005: Add recall columns to message_stream.

Stores what the brain surfaced alongside each user message, so the
encoding agent sees pre-attached recall without re-doing searches.

Two columns:
  - recalled_node_ids: JSON array of node IDs surfaced for this turn
  - recalled_raw: JSON array of {id, type, title, content_snippet, score}

Both nullable — existing rows and assistant messages have NULL.
"""

import sqlite3

description = "Add recalled_node_ids and recalled_raw to message_stream for pre-attached recall"


_NEW_COLUMNS = [
    ("recalled_node_ids", "TEXT DEFAULT NULL"),
    ("recalled_raw", "TEXT DEFAULT NULL"),
]


def up(conn: sqlite3.Connection, db_path=None) -> None:
    for col_name, col_def in _NEW_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE message_stream ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    # Don't commit — runner handles transaction via savepoint


def down(conn: sqlite3.Connection) -> None:
    pass  # SQLite < 3.35 can't DROP COLUMN; extra columns are harmless
