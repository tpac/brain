"""Tests for the v24 schema migration: situation → node_metadata_kv.

Run: ./dev python3 -m pytest tests/test_situation_migration.py -v
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.schema import ensure_schema, _migrate_situation_to_kv_v24


def _seed_node(conn, node_id, situation_text=None):
    """Create a node, optionally with a legacy enrichment situation row."""
    conn.execute(
        "INSERT INTO nodes (id, type, title, content, created_at) "
        "VALUES (?, 'rule', 't', 'c', '2026-01-01T00:00:00Z')",
        (node_id,))
    if situation_text is not None:
        conn.execute(
            "INSERT INTO node_enrichments "
            "(id, node_id, vector_type, text, embedding, model, created_at) "
            "VALUES (?, ?, '_situation', ?, NULL, 'm', '2026-01-01')",
            (f'{node_id}__situation', node_id, situation_text))


@pytest.fixture
def v24_db(tmp_path):
    """A brain.db at schema v24, no data."""
    db = str(tmp_path / 'brain.db')
    conn = sqlite3.connect(db)
    ensure_schema(conn, db_path=db)
    yield conn
    conn.close()


class TestMigration:
    def test_legacy_situation_promoted_to_kv(self, v24_db):
        """Enrichment-only situation → kv row after migration runs."""
        _seed_node(v24_db, 'n1', situation_text='legacy value')
        v24_db.commit()

        _migrate_situation_to_kv_v24(v24_db)

        row = v24_db.execute(
            "SELECT value FROM node_metadata_kv "
            "WHERE node_id='n1' AND key='situation'").fetchone()
        assert row is not None
        assert row[0] == 'legacy value'

    def test_idempotent_on_rerun(self, v24_db):
        """Running migration twice doesn't duplicate or change kv rows."""
        _seed_node(v24_db, 'n1', situation_text='v')
        v24_db.commit()

        _migrate_situation_to_kv_v24(v24_db)
        _migrate_situation_to_kv_v24(v24_db)

        rows = v24_db.execute(
            "SELECT value FROM node_metadata_kv "
            "WHERE node_id='n1' AND key='situation'").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 'v'

    def test_existing_kv_not_overwritten(self, v24_db):
        """If kv already has a situation value, migration leaves it alone."""
        _seed_node(v24_db, 'n1', situation_text='old enrichment text')
        # kv gets a "newer" value first (simulating post-commit-1 dual-write).
        v24_db.execute(
            "INSERT INTO node_metadata_kv (node_id, key, value) "
            "VALUES ('n1', 'situation', 'newer kv text')")
        v24_db.commit()

        _migrate_situation_to_kv_v24(v24_db)

        row = v24_db.execute(
            "SELECT value FROM node_metadata_kv "
            "WHERE node_id='n1' AND key='situation'").fetchone()
        assert row[0] == 'newer kv text'

    def test_empty_enrichment_text_skipped(self, v24_db):
        """Enrichment rows with empty text aren't inserted as kv rows.

        NOTE: NULL text is schema-prevented (NOT NULL on node_enrichments.text),
        so the `ne.text IS NOT NULL` guard in the migration SQL is defensive-
        only. The reachable edge case is the empty-string text.
        """
        _seed_node(v24_db, 'n1', situation_text='')
        v24_db.commit()

        _migrate_situation_to_kv_v24(v24_db)

        row = v24_db.execute(
            "SELECT value FROM node_metadata_kv "
            "WHERE node_id='n1' AND key='situation'").fetchone()
        assert row is None

    def test_embedding_blob_preserved(self, v24_db):
        """Migration only promotes text — the BLOB on the enrichment row stays."""
        _seed_node(v24_db, 'n1', situation_text='v')
        # Add a fake embedding so we can confirm it's untouched.
        v24_db.execute(
            "UPDATE node_enrichments SET embedding = ? "
            "WHERE node_id='n1' AND vector_type='_situation'",
            (b'\x01' * 32,))
        v24_db.commit()

        _migrate_situation_to_kv_v24(v24_db)

        row = v24_db.execute(
            "SELECT embedding FROM node_enrichments "
            "WHERE node_id='n1' AND vector_type='_situation'").fetchone()
        assert row[0] == b'\x01' * 32  # BLOB intact

    def test_many_nodes_batch(self, v24_db):
        """Migration handles 100+ rows at once without per-row issues."""
        for i in range(150):
            _seed_node(v24_db, f'n{i:03d}', situation_text=f'sit {i}')
        v24_db.commit()

        _migrate_situation_to_kv_v24(v24_db)

        count = v24_db.execute(
            "SELECT COUNT(*) FROM node_metadata_kv WHERE key='situation'"
        ).fetchone()[0]
        assert count == 150


class TestSchemaIntegration:
    def test_fresh_schema_reaches_v24(self, tmp_path):
        """ensure_schema on a fresh DB lands at v24."""
        db = str(tmp_path / 'brain.db')
        conn = sqlite3.connect(db)
        ensure_schema(conn, db_path=db)

        version = conn.execute(
            "SELECT value FROM brain_meta WHERE key='brain_schema_version'"
        ).fetchone()
        assert version[0] == '24'
        conn.close()

    def test_upgrade_from_v23_runs_migration(self, tmp_path):
        """A v23 DB with legacy situation data gets migrated on next boot."""
        db = str(tmp_path / 'brain.db')
        conn = sqlite3.connect(db)
        ensure_schema(conn, db_path=db)
        _seed_node(conn, 'n1', situation_text='pre-upgrade')
        conn.execute(
            "UPDATE brain_meta SET value = '23' "
            "WHERE key = 'brain_schema_version'")
        conn.commit()
        conn.close()

        # Reopen — ensure_schema should see v23, run v24 migration.
        conn = sqlite3.connect(db)
        ensure_schema(conn, db_path=db)

        row = conn.execute(
            "SELECT value FROM node_metadata_kv "
            "WHERE node_id='n1' AND key='situation'").fetchone()
        assert row is not None
        assert row[0] == 'pre-upgrade'
        conn.close()
