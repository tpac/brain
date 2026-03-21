"""
Migration 004: Add precision tracking columns to recall_log.

These columns support the three-signal evaluation model for recall precision:
  1. Claude's response (weak signal — biased by context injection)
  2. User's next message (strong signal — unbiased two-turn evaluation)
  3. Explicit operator feedback (strongest — direct confirmation)

The recall_log previously only recorded what was returned (returned_ids,
returned_count). These new columns capture:
  - What the recalled nodes contained (titles/snippets for audit)
  - Whether embeddings were used (degraded vs full recall)
  - Claude's response snippet (for future LLM-based evaluation)
  - Match method and evaluation metadata (per-node match details)
  - Followup signal from user's next message
  - Explicit operator feedback that overrides computed scores

Each ALTER is wrapped in try/except because:
  - Fresh databases already have these columns from schema.py
  - This migration handles existing databases that need the upgrade
  - Idempotent: safe to run multiple times
"""

import sqlite3

description = "Add precision tracking columns to recall_log (embeddings_used, recalled_titles, etc.)"


# Columns to add: (name, type + default)
_NEW_COLUMNS = [
    ("embeddings_used", "INTEGER DEFAULT 0"),
    ("recalled_titles", "TEXT"),
    ("recalled_snippets", "TEXT"),
    ("assistant_response_snippet", "TEXT"),
    ("match_method", "TEXT"),
    ("evaluation_metadata", "TEXT"),
    ("followup_signal", "TEXT"),
    ("explicit_feedback", "TEXT"),
    ("evaluated_at", "TEXT"),
]


def up(conn: sqlite3.Connection, db_path=None) -> None:
    for col_name, col_def in _NEW_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE recall_log ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError:
            # Column already exists — safe to ignore
            pass
    conn.commit()


def down(conn: sqlite3.Connection) -> None:
    # SQLite does not support DROP COLUMN before 3.35.0.
    # For safety, down() is a no-op. The extra columns are harmless.
    pass
