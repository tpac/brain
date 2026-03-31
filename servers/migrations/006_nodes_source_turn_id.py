"""
Migration 006: Add source_turn_id to nodes.

Episode linkage — traces any node back to the message_stream row
that produced it. Enables re-extraction and provenance tracking.

Nullable TEXT — existing nodes and manually created nodes have NULL.
"""

import sqlite3

description = "Add source_turn_id to nodes for episode linkage"


def up(conn: sqlite3.Connection, db_path=None) -> None:
    try:
        conn.execute("ALTER TABLE nodes ADD COLUMN source_turn_id TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists


def down(conn: sqlite3.Connection) -> None:
    pass
