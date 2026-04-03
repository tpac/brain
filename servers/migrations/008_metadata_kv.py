"""
Migration 008: Create node_metadata_kv and migrate from fixed-column node_metadata.

Key-value metadata table replaces fixed columns. Adding new metadata fields
now requires only a contract change, not a schema migration.

Existing data migrated: each non-null column in node_metadata becomes a KV row.
Old table kept for backward compat but no longer written to.
"""

import sqlite3

description = "Create node_metadata_kv table and migrate existing metadata"

# Columns to migrate from old table (skip node_id and created_at)
_MIGRATE_COLUMNS = [
    'reasoning', 'alternatives', 'user_raw_quote', 'correction_of',
    'correction_pattern', 'source_context', 'confidence_rationale',
    'last_validated', 'validation_count', 'change_impacts',
]


def up(conn: sqlite3.Connection, db_path=None) -> None:
    # Create KV table
    conn.execute("""CREATE TABLE IF NOT EXISTS node_metadata_kv (
        node_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT,
        PRIMARY KEY (node_id, key),
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
    )""")

    # Migrate existing data from fixed-column table
    try:
        rows = conn.execute("SELECT * FROM node_metadata").fetchall()
        if not rows:
            return

        # Get column names
        cols = [d[0] for d in conn.execute("SELECT * FROM node_metadata LIMIT 1").description]

        migrated = 0
        for row in rows:
            row_dict = dict(zip(cols, row))
            node_id = row_dict.get('node_id')
            if not node_id:
                continue

            for col in _MIGRATE_COLUMNS:
                val = row_dict.get(col)
                if val is not None and str(val).strip():
                    conn.execute(
                        "INSERT OR IGNORE INTO node_metadata_kv (node_id, key, value) VALUES (?, ?, ?)",
                        (node_id, col, str(val)))
                    migrated += 1

            # Preserve created_at as a KV entry too
            created = row_dict.get('created_at')
            if created:
                conn.execute(
                    "INSERT OR IGNORE INTO node_metadata_kv (node_id, key, value) VALUES (?, ?, ?)",
                    (node_id, 'metadata_created_at', str(created)))

    except sqlite3.OperationalError:
        pass  # node_metadata table doesn't exist (fresh install)


def down(conn: sqlite3.Connection) -> None:
    # Don't drop — data loss risk. Just leave both tables.
    pass
