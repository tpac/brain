"""
Migration 007: Migrate vocabulary nodes to concept type.

Vocab nodes (type='vocabulary') become concept nodes (type='concept').
The vocab type never worked well — concept is the broader, richer
grounding layer that replaces it.

Non-destructive: updates type field only. Content unchanged.
"""

import sqlite3

description = "Migrate vocabulary nodes to concept type"


def up(conn: sqlite3.Connection, db_path=None) -> None:
    count = conn.execute(
        "UPDATE nodes SET type='concept' WHERE type='vocabulary'"
    ).rowcount
    if count:
        print("[migration-007] Migrated %d vocabulary nodes to concept type" % count)


def down(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE nodes SET type='vocabulary' WHERE type='concept'")
    conn.commit()
