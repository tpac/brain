"""
Migration 003: One-time data backfills for existing brain databases.

These backfills were previously embedded in schema.py's _backfill_data().
They are safe to run on both fresh and existing databases.
"""

import sqlite3

description = "Backfill confidence, edge_type, and content_summary for existing data"


def up(conn: sqlite3.Connection) -> None:
    """Apply data backfills."""

    # Backfill confidence on nodes that predate the confidence column
    try:
        conn.execute(
            'UPDATE nodes SET confidence = 1.0 '
            'WHERE locked = 1 AND confidence IS NULL'
        )
        conn.execute(
            'UPDATE nodes SET confidence = 0.5 '
            'WHERE locked = 0 AND confidence IS NULL'
        )
    except Exception:
        pass

    # Backfill edge_type from relation column (pre-v6 databases)
    try:
        conn.execute(
            "UPDATE edges SET edge_type = 'corrected_by' "
            "WHERE relation = 'corrected_by'"
        )
        conn.execute(
            "UPDATE edges SET edge_type = 'part_of' "
            "WHERE relation = 'part_of'"
        )
        conn.execute(
            "UPDATE edges SET edge_type = 'exemplifies' "
            "WHERE relation = 'example_of'"
        )
        conn.execute(
            "UPDATE edges SET edge_type = 'related' "
            "WHERE edge_type IS NULL"
        )
    except Exception:
        pass

    # Generate content_summary for nodes that have content but no summary
    try:
        cur = conn.execute(
            "SELECT id, title, content FROM nodes "
            "WHERE content IS NOT NULL AND content != '' "
            "AND content_summary IS NULL"
        )
        for row in cur.fetchall():
            node_id, title, content = row
            summary = content
            period_idx = content.find('. ')
            if 0 < period_idx < 200:
                summary = content[:period_idx + 1]
            elif len(content) > 200:
                summary = content[:197] + '...'
            conn.execute(
                "UPDATE nodes SET content_summary = ? WHERE id = ?",
                (summary, node_id)
            )
    except Exception:
        pass
