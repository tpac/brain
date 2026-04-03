"""
Migration 009: Remove vector_type CHECK constraint from node_enrichments.

The old constraint limited vector_type to ('question','anchor','bridge','keywords').
The new z-indexed multi-vector architecture adds 'title', 'high_meta', 'other_meta'
and needs to support emergent types without schema changes.

SQLite doesn't support ALTER TABLE DROP CONSTRAINT, so we recreate the table.
"""

import sqlite3

description = "Remove vector_type CHECK constraint from node_enrichments"


def up(conn: sqlite3.Connection, db_path=None) -> None:
    # Check if the constraint exists by trying an insert
    try:
        conn.execute(
            "INSERT INTO node_enrichments (id, node_id, vector_type, text, embedding) "
            "VALUES ('_test_check', '_test', 'title', '', NULL)")
        # If it worked, constraint is already gone — clean up and return
        conn.execute("DELETE FROM node_enrichments WHERE id = '_test_check'")
        return
    except sqlite3.IntegrityError:
        pass  # Constraint exists, proceed with migration

    # Recreate table without CHECK constraint
    conn.execute("ALTER TABLE node_enrichments RENAME TO node_enrichments_old")
    conn.execute("""CREATE TABLE node_enrichments (
        id TEXT PRIMARY KEY,
        node_id TEXT NOT NULL,
        vector_type TEXT NOT NULL,
        text TEXT,
        embedding BLOB,
        model TEXT,
        created_at TEXT,
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
    )""")
    conn.execute("""INSERT INTO node_enrichments
        SELECT id, node_id, vector_type, text, embedding, model, created_at
        FROM node_enrichments_old""")
    conn.execute("DROP TABLE node_enrichments_old")

    # Recreate index
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_enrichments_node ON node_enrichments(node_id)")


def down(conn: sqlite3.Connection) -> None:
    pass  # Don't re-add constraint
