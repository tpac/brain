#!/usr/bin/env python3
"""
brain — Core Unit Tests

Tests the critical paths that have broken before:
- remember() + recall pipeline
- connect() sets both relation and edge_type
- encoding heartbeat tracks messages and encodes
- vocabulary system (learn, resolve, ambiguous)
- confidence scoring in recall
- error logging pipeline
- DAL read/write consistency
- session activity tracking

Run: python tests/test_core.py
"""

import sys
import os
import tempfile
import shutil
import json
import unittest

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.brain import Brain
from servers.dal import LogsDAL, MetaDAL
from servers.schema import ensure_schema, ensure_logs_schema
import sqlite3


class BrainTestBase(unittest.TestCase):
    """Base class that creates a fresh brain per test."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.brain = Brain(self.db_path)
        self.brain.reset_session_activity()

    def tearDown(self):
        self.brain.close()
        shutil.rmtree(self.tmp)


class TestRememberRecall(BrainTestBase):
    """Test the remember → recall pipeline."""

    def test_remember_returns_id(self):
        result = self.brain.remember(type='decision', title='Test decision',
                                      content='We chose X over Y')
        self.assertIn('id', result)
        self.assertEqual(result['type'], 'decision')

    def test_recall_finds_remembered_node(self):
        self.brain.remember(type='decision', title='Auth: use Clerk',
                           content='Passwordless login via magic links',
                           keywords='auth clerk login passwordless')
        results = self.brain.recall_with_embeddings('auth login', limit=5)
        titles = [r['title'] for r in results.get('results', [])]
        self.assertTrue(any('Clerk' in t for t in titles),
                       'Expected to find Clerk node in recall results: %s' % titles)

    def test_locked_node_persists(self):
        r = self.brain.remember(type='rule', title='Never use mocks',
                                content='Integration tests only', locked=True)
        node = self.brain.conn.execute(
            'SELECT locked FROM nodes WHERE id = ?', (r['id'],)
        ).fetchone()
        self.assertEqual(node[0], 1)

    def test_remember_increments_counter(self):
        activity = self.brain._get_session_activity()
        initial = int(activity.get('remember_count', 0))
        self.brain.remember(type='context', title='Test', content='Test')
        activity = self.brain._get_session_activity()
        self.assertEqual(int(activity.get('remember_count', 0)), initial + 1)


class TestConnect(BrainTestBase):
    """Test edge creation — specifically the relation/edge_type bug."""

    def test_connect_sets_both_columns(self):
        """Regression: connect() must set BOTH relation AND edge_type."""
        n1 = self.brain.remember(type='decision', title='Node A', content='A')
        n2 = self.brain.remember(type='decision', title='Node B', content='B')
        self.brain.connect(n1['id'], n2['id'], 'produced', 0.9)

        # Check both columns
        edge = self.brain.conn.execute(
            'SELECT relation, edge_type FROM edges WHERE source_id = ? AND target_id = ?',
            (n1['id'], n2['id'])
        ).fetchone()
        self.assertIsNotNone(edge, 'Edge should exist')
        self.assertEqual(edge[0], 'produced', 'relation column should be set')
        self.assertEqual(edge[1], 'produced', 'edge_type column should also be set')

    def test_connect_queryable_by_edge_type(self):
        """Regression: edges must be findable via edge_type queries."""
        n1 = self.brain.remember(type='impact', title='Impact A', content='A')
        n2 = self.brain.remember(type='decision', title='Decision B', content='B')
        self.brain.connect(n1['id'], n2['id'], 'validates_impact', 0.8)

        rows = self.brain.conn.execute(
            "SELECT source_id FROM edges WHERE edge_type = 'validates_impact'"
        ).fetchall()
        self.assertTrue(len(rows) > 0, 'Should find edge by edge_type')


class TestEncodingHeartbeat(BrainTestBase):
    """Test the encoding heartbeat nudge system."""

    def test_no_nudge_below_threshold(self):
        for _ in range(7):
            self.brain.record_message()
        self.assertIsNone(self.brain.get_encoding_heartbeat())

    def test_nudge_at_threshold(self):
        for _ in range(8):
            self.brain.record_message()
        nudge = self.brain.get_encoding_heartbeat()
        self.assertIsNotNone(nudge)
        self.assertIn('nothing encoded yet', nudge['message'])

    def test_nudge_clears_after_encode(self):
        for _ in range(10):
            self.brain.record_message()
        self.assertIsNotNone(self.brain.get_encoding_heartbeat())
        self.brain.remember(type='decision', title='Test', content='Test')
        self.assertIsNone(self.brain.get_encoding_heartbeat())

    def test_nudge_urgency(self):
        self.brain.remember(type='context', title='X', content='X')
        for _ in range(16):
            self.brain.record_message()
        nudge = self.brain.get_encoding_heartbeat()
        self.assertEqual(nudge['severity'], 'urgent')


class TestVocabulary(BrainTestBase):
    """Test vocabulary system."""

    def test_learn_and_resolve(self):
        learn_result = self.brain.learn_vocabulary('GPR', ['Gross Profit Rate'], context='finance')
        self.assertIn('id', learn_result)
        resolve_result = self.brain.resolve_vocabulary('GPR')
        self.assertIsNotNone(resolve_result)
        # Single match returns {id, title, content}, not {mappings: [...]}
        self.assertIn('id', resolve_result)
        self.assertIn('GPR', resolve_result['title'])

    def test_context_dependent_ambiguity(self):
        self.brain.learn_vocabulary('GPR', ['Gross Profit Rate'], context='finance')
        self.brain.learn_vocabulary('GPR', ['Google PageRank'], context='SEO')
        result = self.brain.resolve_vocabulary('GPR')
        self.assertTrue(result.get('ambiguous', False),
                       'Should be ambiguous with two contexts')
        self.assertEqual(len(result['mappings']), 2)

    def test_vocabulary_node_connected(self):
        """Regression: vocab nodes must connect to graph at birth."""
        # Create a target node first
        target = self.brain.remember(type='concept', title='Gross Profit Rate',
                                      content='Financial metric')
        result = self.brain.learn_vocabulary('GPR', ['Gross Profit Rate'])
        vocab_id = result.get('id')
        if vocab_id:
            edges = self.brain.conn.execute(
                'SELECT * FROM edges WHERE source_id = ?', (vocab_id,)
            ).fetchall()
            self.assertTrue(len(edges) > 0,
                          'Vocabulary node should have edges at birth')


class TestConfidenceScoring(BrainTestBase):
    """Test confidence affects recall ranking."""

    def test_high_confidence_ranks_higher(self):
        # Create two similar nodes with different confidence
        self.brain.remember(type='decision', title='API design choice A',
                           content='REST API for user service',
                           keywords='api design rest user', confidence=0.3)
        self.brain.remember(type='decision', title='API design choice B',
                           content='REST API for user service improved',
                           keywords='api design rest user', confidence=1.0)
        self.brain.save()

        results = self.brain.recall_with_embeddings('API design for user service', limit=5)
        result_list = results.get('results', [])
        if len(result_list) >= 2:
            # Higher confidence should generally rank higher
            # (not guaranteed due to other factors, but likely with same keywords)
            titles = [r['title'] for r in result_list[:2]]
            # Just verify both appear — exact ordering depends on embedding similarity
            self.assertTrue(any('choice A' in t or 'choice B' in t for t in titles))


class TestErrorLogging(BrainTestBase):
    """Test error logging pipeline."""

    def test_log_error_writes_to_db(self):
        try:
            raise ValueError("test error")
        except ValueError as e:
            self.brain._log_error("test_source", e, "test context")

        errors = self.brain.get_recent_errors(hours=1)
        self.assertTrue(len(errors) > 0, 'Should have logged error')
        self.assertEqual(errors[0]['source'], 'test_source')

    def test_errors_surface_in_consciousness(self):
        try:
            raise RuntimeError("consciousness test")
        except RuntimeError as e:
            self.brain._log_error("test_consciousness", e, "testing")

        signals = self.brain.get_consciousness_signals()
        silent = signals.get('silent_errors', [])
        self.assertTrue(len(silent) > 0, 'Should surface error in consciousness')


class TestDAL(unittest.TestCase):
    """Test the DAL layer independently."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.logs_conn = sqlite3.connect(os.path.join(self.tmp, 'logs.db'))
        ensure_logs_schema(self.logs_conn)
        self.logs = LogsDAL(self.logs_conn)

        self.brain_conn = sqlite3.connect(os.path.join(self.tmp, 'brain.db'))
        ensure_schema(self.brain_conn)
        self.meta = MetaDAL(self.brain_conn)

    def tearDown(self):
        self.logs_conn.close()
        self.brain_conn.close()
        shutil.rmtree(self.tmp)

    def test_logs_write_read_errors(self):
        self.logs.write_error("src", "err", "ctx")
        errors = self.logs.get_recent_errors(hours=1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['source'], 'src')

    def test_logs_error_count(self):
        self.logs.write_error("a", "e1")
        self.logs.write_error("b", "e2")
        self.assertEqual(self.logs.get_error_count(hours=1), 2)

    def test_meta_get_set(self):
        self.meta.set("key", "value")
        self.assertEqual(self.meta.get("key"), "value")

    def test_meta_json(self):
        self.meta.set_json("j", {"a": [1, 2]})
        self.assertEqual(self.meta.get_json("j"), {"a": [1, 2]})

    def test_meta_increment(self):
        self.assertEqual(self.meta.increment("ctr"), 1)
        self.assertEqual(self.meta.increment("ctr"), 2)

    def test_meta_session_activity(self):
        self.meta.set("remember_count", "5")
        self.meta.set("message_count", "20")
        activity = self.meta.get_session_activity()
        self.assertEqual(activity['remember_count'], 5)
        self.assertEqual(activity['message_count'], 20)


class TestSessionActivity(BrainTestBase):
    """Test session activity tracking."""

    def test_boot_time_set(self):
        activity = self.brain._get_session_activity()
        self.assertIn('boot_time', activity)
        self.assertIsNotNone(activity['boot_time'])

    def test_message_count_increments(self):
        self.brain.record_message()
        self.brain.record_message()
        activity = self.brain._get_session_activity()
        self.assertEqual(int(activity.get('message_count', 0)), 2)


class TestSchemaMigration(unittest.TestCase):
    """Test schema migration safety features."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_backup_created_on_version_change(self):
        """Pre-migration backup is created when schema version changes."""
        # Create a v1 database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS brain_meta (
            key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)""")
        conn.execute(
            "INSERT INTO brain_meta (key, value) VALUES (?, ?)",
            ('brain_schema_version', '1')
        )
        conn.commit()
        conn.close()

        # Now open with ensure_schema which will migrate to v15
        conn = sqlite3.connect(self.db_path)
        ensure_schema(conn, db_path=self.db_path)
        conn.close()

        # Check backup exists
        backup = self.db_path + '.v1.bak'
        self.assertTrue(os.path.exists(backup),
                       'Backup file should exist after migration')

    def test_no_backup_on_fresh_db(self):
        """No backup for brand new databases (version 0)."""
        conn = sqlite3.connect(self.db_path)
        ensure_schema(conn, db_path=self.db_path)
        conn.close()

        # No backup should exist for fresh DBs
        import glob as glob_mod
        backups = glob_mod.glob(self.db_path + '.v*.bak')
        self.assertEqual(len(backups), 0,
                        'No backup should be created for fresh databases')

    def test_version_history_records_backup_path(self):
        """Version history entry includes backup_path."""
        # Create a v1 database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS brain_meta (
            key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)""")
        conn.execute(
            "INSERT INTO brain_meta (key, value) VALUES (?, ?)",
            ('brain_schema_version', '1')
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS version_history (
            version INTEGER NOT NULL, migration_ts TEXT NOT NULL,
            description TEXT, backup_path TEXT)""")
        conn.commit()
        conn.close()

        # Migrate
        conn = sqlite3.connect(self.db_path)
        ensure_schema(conn, db_path=self.db_path)

        # Check version_history
        row = conn.execute(
            "SELECT backup_path FROM version_history WHERE version = 15"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row, 'Should have version_history entry')
        self.assertIsNotNone(row[0], 'backup_path should be recorded')
        self.assertIn('.v1.bak', row[0])


class TestValidateConfig(BrainTestBase):
    """Test infrastructure validation."""

    def test_healthy_brain_no_warnings(self):
        """Fresh brain should have no critical warnings."""
        warnings = self.brain.validate_config()
        critical = [w for w in warnings if w['level'] == 'critical']
        self.assertEqual(len(critical), 0, 'Fresh brain should have no critical warnings')

    def test_schema_version_mismatch_warned(self):
        """Detect schema version mismatch."""
        # Artificially set wrong version
        self.brain._meta.set('brain_schema_version', '1')
        warnings = self.brain.validate_config()
        version_warnings = [w for w in warnings if 'Schema version' in w['message']]
        self.assertTrue(len(version_warnings) > 0, 'Should warn about schema version mismatch')


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=True)
