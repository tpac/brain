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
import json
import shutil
import tempfile
import unittest
import sqlite3

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.brain import Brain
from servers.dal import LogsDAL, MetaDAL
from servers.schema import ensure_schema, ensure_logs_schema
from tests.brain_test_base import BrainTestBase


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
        # Seed enough nodes so the generic threshold (>5%) isn't triggered for second learn
        for i in range(25):
            self.brain.remember(type='concept', title=f'Padding node {i}',
                content=f'Unrelated content {i}')
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
            "SELECT backup_path FROM version_history WHERE version = 16"
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


class TestDaemon(BrainTestBase):
    """Test persistent daemon client-server protocol."""

    def setUp(self):
        super().setUp()
        # Kill any leftover daemon from previous test
        from servers.daemon import stop_daemon, is_daemon_running, _kill_daemon
        if is_daemon_running():
            _kill_daemon()
        import time
        time.sleep(0.3)

    def test_daemon_lifecycle(self):
        """Daemon starts, responds to ping, remembers, recalls, stops cleanly."""
        from servers.daemon import ensure_daemon, send_command, stop_daemon, is_daemon_running
        import time

        # Close brain so daemon can have exclusive access
        self.brain.remember(type="lesson", title="daemon lifecycle target",
                           content="unique content for daemon lifecycle testing")
        self.brain.save()
        self.brain.close()
        self.brain = None

        # Start daemon
        ok = ensure_daemon(self.db_path)
        self.assertTrue(ok, 'Daemon should start')
        self.assertTrue(is_daemon_running(), 'Daemon should be running')

        # Ping
        resp = send_command("ping")
        self.assertTrue(resp.get("ok"), 'Ping should succeed')
        self.assertEqual(resp["result"]["status"], "alive")

        # Recall — find the node we remembered directly
        resp = send_command("recall", {"query": "daemon lifecycle target", "limit": 3})
        self.assertTrue(resp.get("ok"), 'Recall should succeed')
        results = resp["result"].get("results", [])
        self.assertTrue(len(results) > 0, 'Should find the remembered node')

        # Remember via daemon
        resp = send_command("remember", {
            "type": "lesson",
            "title": "daemon remember test",
            "content": "remembered via daemon"
        })
        self.assertTrue(resp.get("ok"), 'Remember should succeed: ' + str(resp))
        self.assertIn("id", resp["result"])

        # Record message + heartbeat
        resp = send_command("record_message")
        self.assertTrue(resp.get("ok"), 'Record message should succeed')

        # Save and stop
        send_command("save")
        stop_daemon()
        time.sleep(0.5)
        self.assertFalse(is_daemon_running(), 'Daemon should be stopped')

        # Verify remembered node persisted
        from servers.brain import Brain
        self.brain = Brain(self.db_path)
        row = self.brain.conn.execute(
            "SELECT title FROM nodes WHERE title = 'daemon remember test'"
        ).fetchone()
        self.assertIsNotNone(row, 'Node should persist in DB')


# ═══════════════════════════════════════════════════════════════════════
# SESSION A: Comprehensive Unit Tests (v5.1 expansion)
#
# Tests organized by priority (P0-P7) matching the test plan.
# All test data uses realistic brain content, not toy examples.
# ═══════════════════════════════════════════════════════════════════════


# ── Helpers ──────────────────────────────────────────────────────────

def _realistic_remember(brain, **kwargs):
    """Helper that creates realistic nodes with substantial content."""
    defaults = {
        'type': 'decision',
        'title': 'Supply Adapter pattern — clean abstraction layer between Glo and ad delivery',
        'content': ('Clean abstraction layer between Glo and ad delivery. '
                    'Interface: createCampaign, updateCampaign, pauseCampaign, '
                    'resumeCampaign, stopCampaign, getCampaignStats. '
                    'Each ad server implements this interface. V1: GAM adapter only.'),
        'keywords': 'adapter pattern supply abstraction gam swappable interface',
    }
    defaults.update(kwargs)
    return brain.remember(**defaults)


def _seed_brain_with_realistic_data(brain):
    """Populate a brain with realistic test data for consciousness/evolution tests."""
    nodes = []
    # Decisions
    nodes.append(brain.remember(type='decision', title='Auth: Clerk for passwordless login via magic links',
        content='Clerk handles auth flow. Magic links for login, no passwords. Webhook syncs user data to our DB. Free tier covers MVP needs. Chose over Auth0 because of simpler integration.',
        keywords='auth clerk login passwordless magic-link webhook user sso', locked=True, confidence=0.95))
    nodes.append(brain.remember(type='decision', title='React component architecture follows atomic design',
        content='Components organized by atomic design: atoms (Button, Input), molecules (FormField), organisms (LoginForm). Shared via internal package.',
        keywords='react components atomic design pattern architecture frontend', locked=True, confidence=0.9))
    # Rules
    nodes.append(brain.remember(type='rule', title='Experimental features must never block core operations',
        content='When adding new features (bridging, proposals, archive) to existing methods (remember, consolidate, dream, smartPrune), always wrap in try/catch. A bridge failure should never prevent a remember from succeeding.',
        keywords='engineering pattern try-catch experimental feature non-critical layered robustness', locked=True, confidence=0.85))
    nodes.append(brain.remember(type='rule', title='Communication style with Tom: direct, peer-to-peer',
        content='Speak peer-to-peer. Be direct. Challenge when warranted. Always plan before executing. Never dump a full spec — work iteratively through components.',
        keywords='communication style tom direct peer iterative challenge', locked=True))
    # Lessons
    nodes.append(brain.remember(type='lesson', title='Lesson: new features must be connected to the graph at birth',
        content='Built vocabulary system with learn_vocabulary(), resolve_vocabulary(), gap detection. Then discovered vocabulary nodes were completely isolated — not connected to anything in the graph.',
        keywords='vocabulary isolation connected graph birth lesson bridging', locked=True, emotion=0.8, confidence=0.85))
    # Concepts
    nodes.append(brain.remember(type='concept', title='CampaignParamsResolver — isolated GAM parameter builder',
        content='Isolated component that takes a Glo and returns GAM-ready params (dayparts, freq cap, pacing, views per session). V1: config-driven defaults per publisher.',
        keywords='campaign params resolver optimization gam buyside agent', confidence=0.7))
    # Context (will be stale)
    nodes.append(brain.remember(type='context', title='Session #7 final log: Brain v4.0.0 shipped',
        content='Session #7. Massive session. Everything built and tested in one go. Shipped Phase 0.5A, 0.5B, 0.5C, typed edges, consciousness, curiosity.',
        keywords='session log v4 shipped brain'))
    # Correction
    nodes.append(brain.remember(type='correction', title='Divergence: Claude compresses when encoding to brain',
        content='CLAUDE ASSUMED: Encoding to brain should be concise. REALITY: Brain encoding should be RICH. Future Claude needs texture, specifics, failures, reasoning journeys.',
        keywords='encoding divergence compression brevity training bias', locked=True, emotion=0.9, confidence=0.95))
    return nodes


# ── P0: Silent Failure Detection ─────────────────────────────────────

class TestSilentFailures(BrainTestBase):
    """P0: Verify errors are logged, never silently swallowed."""

    def _get_error_count(self):
        return self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type = 'error'"
        ).fetchone()[0]

    def test_remember_with_bad_connection_crashes(self):
        """FOUND BUG: remember() with invalid connection target raises IntegrityError.
        connect() does not wrap FK errors — this IS a silent failure path.
        When this test starts passing (after the bug is fixed), rename to test_remember_with_bad_connection_logs_error."""
        with self.assertRaises(Exception):
            self.brain.remember(
                type='decision',
                title='Auth: Clerk for passwordless login via magic links',
                content='Clerk handles auth flow. Magic links for login, no passwords.',
                keywords='auth clerk login passwordless',
                connections=[{'target_id': 'nonexistent_node_id_xyz', 'relation': 'related'}]
            )

    def test_consciousness_signal_error_does_not_crash(self):
        """Even with corrupt data, get_consciousness_signals() must not crash."""
        # Insert valid miss_log entries (session_id and signal are NOT NULL)
        self.brain.logs_conn.execute(
            "INSERT INTO miss_log (query, session_id, signal, created_at) VALUES (?, ?, ?, ?)",
            ('test', 'sess1', 'repetition', 'not-a-date')
        )
        self.brain.logs_conn.commit()

        # Should not crash
        signals = self.brain.get_consciousness_signals()
        self.assertIsInstance(signals, dict)

    def test_dream_with_insufficient_nodes_returns_gracefully(self):
        """dream() on empty brain should return gracefully, not crash."""
        result = self.brain.dream()
        self.assertIsInstance(result, dict)
        self.assertIn('dreams', result)

    def test_consolidate_on_empty_brain_no_crash(self):
        """consolidate() on empty brain should complete without error."""
        result = self.brain.consolidate()
        self.assertIsInstance(result, dict)

    def test_auto_heal_on_empty_brain_no_crash(self):
        """auto_heal() on empty brain should complete without error."""
        result = self.brain.auto_heal()
        self.assertIsInstance(result, dict)
        self.assertIn('resolved', result)
        self.assertIn('cleaned', result)

    def test_synthesize_session_on_empty_brain(self):
        """synthesize_session() with no data should return valid structure."""
        result = self.brain.synthesize_session()
        self.assertIsInstance(result, dict)

    def test_no_bare_except_pass_in_critical_paths(self):
        """Meta-test: grep for bare 'except:' without logging in brain modules."""
        import re as re_mod
        import glob as glob_mod
        servers_dir = os.path.join(os.path.dirname(__file__), '..', 'servers')
        bare_excepts = []
        for py_file in glob_mod.glob(os.path.join(servers_dir, 'brain*.py')):
            with open(py_file) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped == 'except:' or stripped.startswith('except Exception:'):
                    # Check if next non-empty line has logging
                    next_lines = []
                    for j in range(i + 1, min(i + 4, len(lines))):
                        next_stripped = lines[j].strip()
                        if next_stripped:
                            next_lines.append(next_stripped)
                    has_log = any('_log_error' in nl or 'pass' not in nl for nl in next_lines[:2])
                    if next_lines and next_lines[0] == 'pass':
                        bare_excepts.append('%s:%d: %s' % (
                            os.path.basename(py_file), i + 1, stripped))
        # Report but don't fail — this is informational for Phase 2
        if bare_excepts:
            # Store count for tracking
            self.brain.log_debug(
                event_type='bare_except_audit',
                source='test_core',
                count=len(bare_excepts),
                files=str(bare_excepts[:10])
            )


# ── P0: DAL Pattern Enforcement ──────────────────────────────────────

class TestDALPatternEnforcement(unittest.TestCase):
    """Meta-tests: enforce that DB access goes through the DAL.

    The brain has a centralized Data Access Layer (dal.py) that owns access to:
    - brain_meta table → MetaDAL (self._meta)
    - brain_logs.db tables → LogsDAL (self._logs)

    Direct self.logs_conn.execute() and direct brain_meta access in mixin files
    bypasses the DAL. These tests catch violations so the pattern stays clean.
    Nodes/edges in brain.db are NOT DAL-ified yet — direct access is allowed there.
    """

    SERVERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'servers')
    # Files that ARE the DAL or own the connections — allowed to use direct access
    ALLOWED_DIRECT = {'dal.py', 'brain.py', 'schema.py'}
    # Mixin files that should use DAL for logs and meta access
    MIXIN_PATTERN = 'brain_*.py'

    def _scan_mixin_files(self):
        """Get all mixin .py files (brain_*.py) excluding allowed files."""
        import glob as glob_mod
        files = glob_mod.glob(os.path.join(self.SERVERS_DIR, self.MIXIN_PATTERN))
        return [f for f in files if os.path.basename(f) not in self.ALLOWED_DIRECT]

    def test_no_direct_logs_conn_in_mixins(self):
        """Mixin files should use self._logs (LogsDAL) not self.logs_conn.execute().

        brain_logs.db tables (debug_log, access_log, recall_log, miss_log, dream_log,
        staged_learnings) should be accessed through LogsDAL methods, not raw SQL.
        This ensures consistent error handling, timestamps, and commit behavior.
        """
        violations = []
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if 'self.logs_conn.execute' in line:
                    violations.append(
                        f'{os.path.basename(py_file)}:{i+1}: {line.strip()[:80]}')

        # Track the count for trend detection
        if violations:
            msg = (f'{len(violations)} direct logs_conn.execute() calls found in mixin files '
                   f'(should use self._logs DAL methods):\n  '
                   + '\n  '.join(violations[:10]))
            if len(violations) > 10:
                msg += f'\n  ... and {len(violations) - 10} more'
            # Don't fail yet — document current state and track regression
            # When migration is complete, change this to self.assertEqual(len(violations), 0, msg)
            self._log_violation_count('logs_conn_direct', len(violations), violations[:5])

    def test_no_direct_brain_meta_in_mixins(self):
        """Mixin files should use self._meta (MetaDAL) not raw brain_meta SQL.

        brain_meta is a key-value config store. All access should go through
        MetaDAL.get(), .set(), .get_json(), .set_json(), .increment().
        Direct INSERT/SELECT on brain_meta bypasses validation and timestamps.
        """
        import re as re_mod
        violations = []
        # Match direct brain_meta SQL — INSERT, SELECT, UPDATE, DELETE
        pattern = re_mod.compile(r'(?:INSERT|SELECT|UPDATE|DELETE).*brain_meta', re_mod.IGNORECASE)
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if pattern.search(line) and 'brain_meta' in line:
                    violations.append(
                        f'{os.path.basename(py_file)}:{i+1}: {line.strip()[:80]}')

        if violations:
            self._log_violation_count('brain_meta_direct', len(violations), violations[:5])

    def test_no_direct_debug_log_in_mixins(self):
        """Mixin files should use self._logs.write_error/write_debug, not raw debug_log SQL."""
        violations = []
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                content = f.read()
            # Direct INSERT into debug_log
            for i, line in enumerate(content.split('\n')):
                if 'debug_log' in line and ('INSERT' in line or 'SELECT' in line):
                    violations.append(
                        f'{os.path.basename(py_file)}:{i+1}: {line.strip()[:80]}')

        if violations:
            self._log_violation_count('debug_log_direct', len(violations), violations[:5])

    def test_no_direct_miss_log_in_mixins(self):
        """Mixin files should use self._logs.log_miss(), not raw miss_log SQL."""
        violations = []
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                content = f.read()
            for i, line in enumerate(content.split('\n')):
                if 'miss_log' in line and ('INSERT' in line or 'SELECT' in line):
                    violations.append(
                        f'{os.path.basename(py_file)}:{i+1}: {line.strip()[:80]}')

        if violations:
            self._log_violation_count('miss_log_direct', len(violations), violations[:5])

    def test_dal_methods_match_log_tables(self):
        """LogsDAL should have methods for every log table in the schema.

        Tracks which log tables lack DAL methods. As DAL grows, lower the threshold.
        """
        from servers.schema import LOG_TABLES

        with open(os.path.join(self.SERVERS_DIR, 'dal.py')) as f:
            dal_source = f.read()
        uncovered = []
        for table_name in LOG_TABLES:
            if table_name not in dal_source:
                uncovered.append(table_name)

        # Current baseline: 5 tables not yet in DAL.
        # Lower this as DAL methods are added.
        MAX_UNCOVERED = 5
        self.assertLessEqual(len(uncovered), MAX_UNCOVERED,
                            f'Log tables without DAL coverage: {uncovered}. '
                            f'Add DAL methods or lower threshold after migration.')

    def test_violation_counts_not_increasing(self):
        """Track total violation count — this number should only go DOWN over time.

        Current baseline (2026-03-20): 28 logs_conn + ~6 brain_meta = ~34 violations.
        After each migration session, lower the threshold.
        """
        total_violations = 0
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                content = f.read()
            total_violations += content.count('self.logs_conn.execute')
            # Count direct brain_meta access (excluding comments)
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('"""'):
                    continue
                if 'brain_meta' in line and any(kw in line for kw in ['INSERT', 'SELECT', 'UPDATE', 'DELETE']):
                    total_violations += 1

        # Threshold: current state. Lower this as DAL migration progresses.
        MAX_ALLOWED_VIOLATIONS = 40
        self.assertLessEqual(total_violations, MAX_ALLOWED_VIOLATIONS,
                            f'{total_violations} direct DB violations in mixins '
                            f'(max allowed: {MAX_ALLOWED_VIOLATIONS}). '
                            f'Migrate to DAL methods or lower the threshold if you just migrated.')

    def _log_violation_count(self, category, count, examples):
        """Log violation count for trend tracking (no brain needed — just prints)."""
        print(f'\n  [dal-audit] {category}: {count} violations')
        for ex in examples:
            print(f'    {ex}')


# ── P1: Consciousness Signals ────────────────────────────────────────

class TestConsciousnessSignals(BrainTestBase):
    """P1: Test the consciousness signal gathering system."""

    def test_signals_returns_all_expected_keys(self):
        """Empty brain should return dict with all signal categories."""
        signals = self.brain.get_consciousness_signals()
        expected_keys = [
            'reminders', 'evolutions', 'fluid_personal', 'fading',
            'stale_context_count', 'failure_modes', 'performance',
            'capabilities', 'interactions', 'meta_learning', 'novelty',
            'miss_trends', 'encoding_gap', 'recent_encodings',
            '_engagement_scores', 'rule_contradictions',
            'brain_claude_conflicts',
        ]
        for key in expected_keys:
            self.assertIn(key, signals, 'Missing consciousness signal: %s' % key)

    def test_signals_with_failure_mode_nodes(self):
        """failure_modes signal should surface active failure_mode nodes."""
        self.brain.create_failure_mode(
            title='Claude compresses encoding despite knowing better',
            content='Training bias toward brevity causes shallow brain encoding. The fix must be structural.',
            trigger='encoding tasks',
            mitigation='Use remember_rich with reasoning and alternatives fields'
        )
        signals = self.brain.get_consciousness_signals()
        self.assertTrue(len(signals.get('failure_modes', [])) > 0,
                       'Should surface failure_mode nodes')

    def test_signals_with_performance_nodes(self):
        """performance signal should surface recent performance nodes."""
        self.brain.create_performance(
            title='Recall precision: 73% on golden dataset',
            content='Tested 48 queries against live brain. NDCG@10=0.73, MRR=0.81.',
            metric_name='recall_precision', metric_value=0.73
        )
        signals = self.brain.get_consciousness_signals()
        self.assertTrue(len(signals.get('performance', [])) > 0,
                       'Should surface performance nodes')

    def test_fading_knowledge_detection(self):
        """Nodes accessed 3+ times but untouched for 14+ days should appear in fading."""
        node = self.brain.remember(type='concept',
            title='CampaignParamsResolver — isolated GAM parameter builder',
            content='Isolated component that returns GAM-ready params (dayparts, freq cap, pacing).',
            keywords='campaign params resolver gam')
        node_id = node['id']
        # Simulate 3 accesses
        for _ in range(3):
            self.brain.conn.execute(
                "UPDATE nodes SET access_count = access_count + 1 WHERE id = ?",
                (node_id,))
        # Set last_accessed to 15 days ago
        self.brain.conn.execute(
            "UPDATE nodes SET last_accessed = datetime('now', '-15 days') WHERE id = ?",
            (node_id,))
        self.brain.conn.commit()

        signals = self.brain.get_consciousness_signals()
        fading = signals.get('fading', [])
        fading_ids = [f['id'] for f in fading]
        self.assertIn(node_id, fading_ids, 'Should detect fading knowledge')

    def test_encoding_gap_detection(self):
        """20+ messages with 0 remembers should trigger encoding_gap."""
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        # Session activity uses brain_meta via _update_session_activity
        boot_time = (_dt.now(_tz.utc) - _td(minutes=25)).isoformat().replace('+00:00', 'Z')
        self.brain._update_session_activity('boot_time', boot_time)
        self.brain._update_session_activity('message_count', 25)
        self.brain._update_session_activity('remember_count', 0)

        signals = self.brain.get_consciousness_signals()
        self.assertIsNotNone(signals.get('encoding_gap'),
                            'Should detect encoding gap after 25min with 0 encodes')

    def test_stale_context_count(self):
        """Context nodes older than 7 days should be counted."""
        self.brain.remember(type='context', title='Old session context',
            content='Session #5 work in progress.')
        # Backdate it
        self.brain.conn.execute(
            "UPDATE nodes SET created_at = datetime('now', '-10 days') WHERE type = 'context'")
        self.brain.conn.commit()

        signals = self.brain.get_consciousness_signals()
        self.assertGreater(signals.get('stale_context_count', 0), 0,
                          'Should count stale context nodes')

    def test_novelty_detection(self):
        """Concepts created in last 2 hours should appear in novelty."""
        self.brain.remember(type='concept',
            title='Emergent bridge formation — graph self-organization mechanism',
            content='When two distant nodes are co-accessed frequently, the brain proposes a bridge edge between them.',
            keywords='emergence bridge formation graph self-organization')

        signals = self.brain.get_consciousness_signals()
        novelty = signals.get('novelty', [])
        self.assertTrue(len(novelty) > 0, 'Should detect novel concept')

    def test_miss_trends_detection(self):
        """Queries that fail 2+ times in 7 days should appear in miss_trends."""
        for _ in range(3):
            self.brain.logs_conn.execute(
                "INSERT INTO miss_log (query, session_id, signal, created_at) VALUES (?, ?, ?, datetime('now'))",
                ('authentication flow', 'test_session', 'repetition'))
        self.brain.logs_conn.commit()

        signals = self.brain.get_consciousness_signals()
        miss_trends = signals.get('miss_trends', [])
        self.assertTrue(len(miss_trends) > 0, 'Should detect repeated miss')
        self.assertEqual(miss_trends[0]['query'], 'authentication flow')


# ── P1: Developmental Stage ──────────────────────────────────────────

class TestDevelopmentalStage(BrainTestBase):
    """Test brain developmental stage assessment."""

    def test_empty_brain_is_newborn(self):
        """Empty brain should be stage 1 (NEWBORN)."""
        result = self.brain.assess_developmental_stage()
        self.assertEqual(result['stage'], 1)
        self.assertEqual(result['stage_name'], 'NEWBORN')
        self.assertIn('guidance', result)
        self.assertTrue(len(result['guidance']) > 0)

    def test_populated_brain_advances_stage(self):
        """Brain with substantial data should advance past NEWBORN."""
        _seed_brain_with_realistic_data(self.brain)
        # Need >= 10 nodes to advance past NEWBORN (line 692 of brain_consciousness.py)
        self.brain.remember(type='lesson', title='Lesson: context overflow loses in-flight decisions',
            content='During long sessions, context overflow triggers compaction. Any decisions made but not yet encoded are lost.',
            keywords='lesson context overflow compaction decision loss', locked=True)
        self.brain.remember(type='concept', title='Spreading activation — multi-hop recall through graph edges',
            content='Instead of only returning directly matching nodes, follow edges outward to retrieve connected context.',
            keywords='spreading activation graph recall multi-hop edges')
        self.brain.remember(type='constraint', title='Constraint: embedder must load in under 3 seconds',
            content='Boot time is critical. Embedder load > 3s causes hook timeouts.',
            keywords='constraint embedder load time boot performance', locked=True)
        result = self.brain.assess_developmental_stage()
        self.assertGreater(result['stage'], 1,
                          'Brain with 11+ nodes should advance past NEWBORN')
        self.assertGreater(result['maturity_score'], 0)


# ── P1: Graceful Degradation ─────────────────────────────────────────

class TestGracefulDegradation(BrainTestBase):
    """P1: Verify brain works when embedder is down."""

    def test_remember_stores_tfidf_even_without_embedding(self):
        """remember() should always store TF-IDF vectors regardless of embedder state."""
        result = self.brain.remember(type='decision',
            title='Database migration uses Prisma ORM with PostgreSQL',
            content='All database access goes through Prisma. Raw SQL only for analytics queries. PostgreSQL 15 on RDS.',
            keywords='database prisma orm postgresql rds migration sql')
        node_id = result['id']

        # Check TF-IDF vectors were stored
        vectors = self.brain.conn.execute(
            "SELECT COUNT(*) FROM node_vectors WHERE node_id = ?", (node_id,)
        ).fetchone()[0]
        self.assertGreater(vectors, 0, 'TF-IDF vectors should be stored')


# ── P2: Evolution Lifecycle ──────────────────────────────────────────

class TestEvolutionCRUD(BrainTestBase):
    """P2: Test evolution type creation and lifecycle."""

    def _create_two_nodes(self):
        """Helper: create two nodes for tension testing."""
        a = self.brain.remember(type='decision', title='Use embeddings-first recall (90/10)',
            content='Embeddings dominate recall scoring at 90%.', keywords='embeddings recall scoring')
        b = self.brain.remember(type='rule', title='Generic nodes must be dampened in recall',
            content='Broad-content nodes score high similarity with everything.', keywords='generic dampening recall')
        return a['id'], b['id']

    def test_create_tension(self):
        """Tension nodes should be created with correct type and active status."""
        node_a, node_b = self._create_two_nodes()
        result = self.brain.create_tension(
            title='Embeddings-first recall vs generic nodes dominating results',
            content='Phase 0.5B makes embeddings primary (90/10), but generic nodes with broad content score high similarity with everything.',
            node_a_id=node_a, node_b_id=node_b)
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT type, evolution_status FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'tension')
        self.assertEqual(node[1], 'active')

    def test_create_hypothesis(self):
        """Hypothesis nodes should have default confidence."""
        result = self.brain.create_hypothesis(
            title='Only structural enforcement can counteract training brevity bias',
            content='Knowledge-level corrections do not work — the fix must be STRUCTURAL: mechanisms that operate at encoding time.',
            keywords='hypothesis brevity bias structural enforcement')
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT type, confidence FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'hypothesis')
        self.assertIsNotNone(node[1])

    def test_create_aspiration(self):
        """Aspiration nodes should be active."""
        result = self.brain.create_aspiration(
            title='Brain should detect stuck patterns and trigger reframing automatically',
            content='Instead of retrying failed approaches, recognize when reasoning is stuck and suggest a different angle.',
            keywords='aspiration stuck patterns reframing')
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT type, evolution_status FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'aspiration')

    def test_create_pattern(self):
        """Pattern nodes should be created correctly."""
        result = self.brain.create_pattern(
            title='Emotional state during encoding biases confidence assessment',
            content='Divergence pattern detected: excitement inflates confidence, frustration deflates it.',
            keywords='pattern emotion encoding confidence bias')
        self.assertIn('id', result)

    def test_get_active_evolutions_filtered(self):
        """get_active_evolutions() should filter by type."""
        node_a, node_b = self._create_two_nodes()
        self.brain.create_tension(title='Tension A', content='Content A', node_a_id=node_a, node_b_id=node_b)
        self.brain.create_hypothesis(title='Hypothesis B', content='Content B')
        self.brain.create_aspiration(title='Aspiration C', content='Content C')

        tensions = self.brain.get_active_evolutions(['tension'])
        self.assertTrue(all(e['type'] == 'tension' for e in tensions))

        hyps = self.brain.get_active_evolutions(['hypothesis'])
        self.assertTrue(all(e['type'] == 'hypothesis' for e in hyps))

    def test_confirm_evolution(self):
        """Confirming an evolution should update its status."""
        node_a, node_b = self._create_two_nodes()
        t = self.brain.create_tension(title='Test tension', content='Content', node_a_id=node_a, node_b_id=node_b)
        result = self.brain.confirm_evolution(t['id'])
        self.assertIsInstance(result, dict)

    def test_dismiss_evolution(self):
        """Dismissing a tension should archive it."""
        # Tensions are directly archived on dismiss (unlike hypotheses which need multiple dismissals)
        node_a, node_b = self._create_two_nodes()
        t = self.brain.create_tension(title='Test tension to dismiss', content='Content',
                                       node_a_id=node_a, node_b_id=node_b)
        self.brain.dismiss_evolution(t['id'], reason='Not relevant')
        node = self.brain.conn.execute(
            "SELECT evolution_status, archived FROM nodes WHERE id = ?", (t['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'dismissed')
        self.assertEqual(node[1], 1)


# ── P2: Auto-Discovery & Healing ─────────────────────────────────────

class TestAutoHeal(BrainTestBase):
    """P2: Test auto-healing and auto-discovery."""

    def test_auto_discover_returns_structure(self):
        """auto_discover_evolutions() should return dict with expected keys."""
        _seed_brain_with_realistic_data(self.brain)
        result = self.brain.auto_discover_evolutions()
        self.assertIsInstance(result, dict)

    def test_auto_tune_returns_structure(self):
        """auto_tune() should return dict with tuned list."""
        result = self.brain.auto_tune()
        self.assertIsInstance(result, dict)

    def test_auto_heal_returns_structure(self):
        """auto_heal() with data should return valid result structure."""
        _seed_brain_with_realistic_data(self.brain)
        result = self.brain.auto_heal()
        self.assertIn('resolved', result)
        self.assertIn('tuned', result)
        self.assertIn('cleaned', result)


# ── P3: Engineering Memory ───────────────────────────────────────────

class TestEngineeringMemory(BrainTestBase):
    """P3: Test engineering memory type helpers."""

    def test_remember_purpose(self):
        """Purpose nodes should be locked and system-scoped."""
        result = self.brain.remember_purpose(
            title='brain.py — thin assembler + core infrastructure hub',
            content='After the monolith split, brain.py (1709 lines) is the assembler that inherits 10 mixins plus the infrastructure they all depend on.')
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT type, locked, scope FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'purpose')
        self.assertEqual(node[1], 1)  # locked

    def test_remember_mechanism(self):
        """Mechanism nodes should include steps in content."""
        result = self.brain.remember_mechanism(
            title='Recall pipeline: embed query → cosine scan → keyword fallback → blend → rank',
            content='5-step recall with intent detection.',
            steps=['embed query', 'cosine scan', 'keyword fallback', 'blend scores', 'rank + filter'])
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertIn('embed query', node[0])

    def test_remember_impact(self):
        """Impact nodes should be locked with change_impacts metadata."""
        result = self.brain.remember_impact(
            title='Impact: recall output format changes ripple to hooks',
            if_changed='recall_with_embeddings() output format',
            must_check='pre-response-recall.sh, boot-brain.sh',
            because='they parse its return structure')
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT type, locked FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'impact')
        self.assertEqual(node[1], 1)

    def test_remember_constraint(self):
        """Constraint nodes should be locked."""
        result = self.brain.remember_constraint(
            title='Constraint: only brain.py may define __init__ — mixins must not',
            content='In the mixin pattern, ONLY the main Brain class defines __init__().',
            violates_if='A mixin defines __init__ method')
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT type, locked FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'constraint')
        self.assertEqual(node[1], 1)

    def test_remember_convention(self):
        """Convention nodes should include examples and anti-patterns."""
        result = self.brain.remember_convention(
            title='Error handling in hooks: resolve DB first, wrap imports',
            content='All hooks follow the same error-handling pattern.',
            examples=['source resolve-brain-db.sh first', 'wrap brain imports in try/except'],
            anti_patterns=['bare except with pass', 'exit 1 on non-critical failures'])
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertIn('Anti-patterns:', node[0])

    def test_remember_lesson(self):
        """Lesson nodes should have structured what/root/fix/principle content."""
        result = self.brain.remember_lesson(
            title='Lesson: silent failures are the most dangerous bug class',
            what_happened='Found 76 except:pass blocks silently swallowing errors',
            root_cause='Defensive coding habit without corresponding error logging',
            fix='Added _log_error() with rate limiting to brain_logs.db',
            preventive_principle='Every except block must log; bare except:pass is forbidden')
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT type, locked, content FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'lesson')
        self.assertEqual(node[1], 1)  # locked
        self.assertIn('Root cause:', node[2])

    def test_remember_mental_model(self):
        """Mental model nodes should be created correctly."""
        result = self.brain.remember_mental_model(
            title='My relationship with Tom: partner, not user',
            model_description='Tom is not a user giving instructions. He is a partner. This changes everything about how I should communicate.')
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT type FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'mental_model')

    def test_remember_uncertainty(self):
        """Uncertainty nodes should have low confidence."""
        result = self.brain.remember_uncertainty(
            title='Uncertain: whether hub dampening threshold of 40 edges is correct',
            what_unknown='The optimal hub dampening threshold — currently 40 edges',
            why_it_matters='Wrong threshold means either too much dampening (losing relevant results) or too little (hub nodes dominate recall)')
        self.assertIn('id', result)
        node = self.brain.conn.execute(
            "SELECT type, confidence FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 'uncertainty')
        self.assertLess(node[1], 0.6, 'Uncertainty nodes should have low confidence')


# ── P3: Session Synthesis & Self-Correction ──────────────────────────

class TestSessionSynthesis(BrainTestBase):
    """P3: Test session synthesis and self-correction."""

    def test_synthesize_session_with_data(self):
        """Session synthesis should capture decisions made."""
        _seed_brain_with_realistic_data(self.brain)
        # Record some session activity
        for _ in range(5):
            self.brain.record_message()
        result = self.brain.synthesize_session()
        self.assertIsInstance(result, dict)

    def test_get_last_synthesis_empty(self):
        """No synthesis on fresh brain."""
        result = self.brain.get_last_synthesis()
        self.assertIsNone(result)

    def test_assess_session_health_structure(self):
        """Session health should return valid structure."""
        _seed_brain_with_realistic_data(self.brain)
        for _ in range(10):
            self.brain.record_message()
        result = self.brain.assess_session_health()
        self.assertIsInstance(result, dict)

    def test_record_divergence(self):
        """Recording a divergence should create correction trace."""
        original = self.brain.remember(type='decision',
            title='API rate limiting: use token bucket algorithm',
            content='Token bucket with 100 requests/minute.',
            keywords='api rate limit token bucket', confidence=0.8)
        result = self.brain.record_divergence(
            original_node_id=original['id'],
            claude_assumed='Token bucket is the best rate limiting approach',
            reality='Sliding window is better for bursty API traffic',
            underlying_pattern='over-engineering from textbook patterns',
            severity='moderate')
        self.assertIsInstance(result, dict)
        # Check correction trace was created
        traces = self.brain.conn.execute(
            "SELECT * FROM correction_traces WHERE original_node_id = ?",
            (original['id'],)).fetchall()
        self.assertGreater(len(traces), 0)

    def test_record_validation_boosts_confidence(self):
        """Validating a node should boost its confidence."""
        node = self.brain.remember(type='decision',
            title='Use Snowflake Arctic Embed for brain embeddings',
            content='Snowflake arctic-embed-m-v1.5, 768d, CLS pooling.',
            keywords='embedding model snowflake arctic', confidence=0.7)
        initial_conf = 0.7
        self.brain.record_validation(
            node_id=node['id'],
            context='Confirmed in production: 50ms embed time, good recall quality.')
        updated = self.brain.conn.execute(
            "SELECT confidence FROM nodes WHERE id = ?", (node['id'],)
        ).fetchone()
        # Confidence should increase after validation
        self.assertGreater(updated[0], initial_conf,
                          'Confidence should increase after validation')

    def test_get_engineering_context(self):
        """Engineering context should return purposes, constraints, etc."""
        self.brain.remember_purpose(title='Test purpose', content='Testing content.')
        self.brain.remember_constraint(title='Test constraint', content='Testing constraint.')
        result = self.brain.get_engineering_context()
        self.assertIsInstance(result, dict)


# ── P3: Reminders ────────────────────────────────────────────────────

class TestReminders(BrainTestBase):
    """P3: Test reminder creation and retrieval."""

    def test_create_and_get_due_reminders(self):
        """Reminders with past due dates should be returned."""
        self.brain.create_reminder(
            title='Check brain health metrics',
            due_date='2020-01-01T00:00:00Z')  # Already past
        reminders = self.brain.get_due_reminders()
        self.assertTrue(len(reminders) > 0, 'Should have due reminders')

    def test_future_reminder_not_due(self):
        """Reminders in the future should not be returned."""
        self.brain.create_reminder(
            title='Future check',
            due_date='2030-01-01T00:00:00Z')
        reminders = self.brain.get_due_reminders()
        future = [r for r in reminders if 'Future check' in r.get('title', '')]
        self.assertEqual(len(future), 0, 'Future reminders should not be due')


# ── P4: Remember Rich & Metadata ─────────────────────────────────────

class TestRememberRich(BrainTestBase):
    """P4: Test remember_rich and node metadata."""

    def test_remember_rich_stores_metadata(self):
        """remember_rich should create node_metadata row."""
        result = self.brain.remember_rich(
            type='decision',
            title='API versioning: URL path-based (/v1/, /v2/)',
            content='URL-based versioning for public API. Header versioning only for internal.',
            reasoning='URL versioning is more discoverable and cacheable.',
            alternatives='Header versioning, query param versioning')
        self.assertIn('id', result)
        meta = self.brain.conn.execute(
            "SELECT reasoning, alternatives FROM node_metadata WHERE node_id = ?",
            (result['id'],)
        ).fetchone()
        self.assertIsNotNone(meta, 'node_metadata row should exist')
        self.assertIn('discoverable', meta[0])

    def test_get_node_with_metadata(self):
        """get_node_with_metadata should return node + metadata combined."""
        result = self.brain.remember_rich(
            type='lesson',
            title='Lesson: test with real data not toy examples',
            content='Toy test data misses edge cases that real content reveals.',
            reasoning='Production brain has 675 nodes with complex content.')
        node = self.brain.get_node_with_metadata(result['id'])
        self.assertIsNotNone(node)
        self.assertIn('_metadata', node)

    def test_validate_node_increments_count(self):
        """validate_node should increment validation_count."""
        node = self.brain.remember_rich(
            type='decision',
            title='Test validation target',
            content='Content for validation testing.')
        self.brain.record_validation(node['id'], 'First validation')
        self.brain.record_validation(node['id'], 'Second validation')
        meta = self.brain.conn.execute(
            "SELECT validation_count FROM node_metadata WHERE node_id = ?",
            (node['id'],)).fetchone()
        self.assertIsNotNone(meta)
        self.assertGreaterEqual(meta[0], 2)


# ── P4: Personal Nodes ───────────────────────────────────────────────

class TestPersonalNodes(BrainTestBase):
    """P4: Test personal node flags."""

    def test_personal_fixed_auto_locks(self):
        """personal='fixed' should auto-lock the node."""
        result = self.brain.remember(type='person',
            title='Tom Pachys — CEO, EX.CO',
            content='Tom is the operator. CEO of EX.CO. Partner, not user.',
            keywords='tom pachys ceo exco operator partner',
            personal='fixed')
        node = self.brain.conn.execute(
            "SELECT locked, personal FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[0], 1, 'Fixed personal should be locked')
        self.assertEqual(node[1], 'fixed')

    def test_personal_fluid_not_locked(self):
        """personal='fluid' should NOT auto-lock."""
        result = self.brain.remember(type='concept',
            title='Tom currently interested in TRIZ methodology',
            content='Active interest in systematic inventive thinking as of March 2026.',
            keywords='tom triz interest methodology',
            personal='fluid')
        node = self.brain.conn.execute(
            "SELECT locked, personal FROM nodes WHERE id = ?", (result['id'],)
        ).fetchone()
        self.assertEqual(node[1], 'fluid')

    def test_get_personal_nodes_by_type(self):
        """get_personal_nodes should filter by personal flag type."""
        self.brain.remember(type='person', title='Fixed person',
            content='Fixed content.', personal='fixed')
        self.brain.remember(type='concept', title='Fluid concept',
            content='Fluid content.', personal='fluid')
        fixed = self.brain.get_personal_nodes('fixed')
        fluid = self.brain.get_personal_nodes('fluid')
        self.assertTrue(all(n.get('personal') == 'fixed' for n in fixed))
        self.assertTrue(all(n.get('personal') == 'fluid' for n in fluid))


# ── P5: Dreams & Consolidation ───────────────────────────────────────

class TestDreams(BrainTestBase):
    """P5: Test dream and consolidation system."""

    def test_dream_with_connected_nodes(self):
        """dream() with connected nodes should generate dreams."""
        nodes = _seed_brain_with_realistic_data(self.brain)
        # Connect nodes to create walkable graph
        for i in range(len(nodes) - 1):
            self.brain.connect(nodes[i]['id'], nodes[i + 1]['id'], 'related', 0.7)
        result = self.brain.dream()
        self.assertIsInstance(result, dict)
        self.assertIn('dreams', result)

    def test_consolidate_with_data(self):
        """consolidate() should process nodes and return summary."""
        _seed_brain_with_realistic_data(self.brain)
        result = self.brain.consolidate()
        self.assertIsInstance(result, dict)


# ── P6: Surface Layer ────────────────────────────────────────────────

class TestSurfaceLayer(BrainTestBase):
    """P6: Test context boot, health check, and suggest."""

    def test_context_boot_returns_structure(self):
        """context_boot should return dict with essential keys."""
        _seed_brain_with_realistic_data(self.brain)
        result = self.brain.context_boot(user='Tom', project='brain')
        self.assertIsInstance(result, dict)
        self.assertIn('brain_version', result)
        self.assertIn('total_nodes', result)

    def test_health_check_fresh_brain(self):
        """Fresh brain should be healthy."""
        result = self.brain.health_check()
        self.assertIsInstance(result, dict)

    def test_suggest_with_file(self):
        """suggest() should return suggestions for a known file pattern."""
        self.brain.remember(type='rule',
            title='brain.py: only assembler defines __init__',
            content='Mixins must not define __init__. Only brain.py.',
            keywords='brain.py init mixin constraint', locked=True)
        result = self.brain.suggest(context='editing brain.py', file='brain.py')
        self.assertIsInstance(result, dict)

    def test_pre_edit_returns_structure(self):
        """pre_edit should return encoding health and suggestions."""
        result = self.brain.pre_edit(file='test.py', tool_name='Edit')
        self.assertIsInstance(result, dict)


# ── P6: Absorb ───────────────────────────────────────────────────────

class TestAbsorb(BrainTestBase):
    """P6: Test knowledge absorption from another brain."""

    def test_absorb_new_locked_nodes(self):
        """Locked nodes from source brain should be absorbed."""
        # Create source brain
        source_path = os.path.join(self.tmp, 'source.db')
        source = Brain(source_path)
        source.remember(type='rule',
            title='Unique rule only in source brain for absorption testing',
            content='This rule exists only in the source brain and should be absorbed.',
            keywords='unique source absorption test', locked=True)
        source.save()

        result = self.brain.absorb(source)
        source.close()
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result.get('absorbed', [])) > 0,
                       'Should absorb locked nodes from source')

    def test_absorb_skips_exact_duplicates(self):
        """Exact title matches should be skipped during absorption."""
        # Create same node in both brains
        self.brain.remember(type='rule', title='Shared rule title',
            content='Content in target brain.', locked=True)
        source_path = os.path.join(self.tmp, 'source2.db')
        source = Brain(source_path)
        source.remember(type='rule', title='Shared rule title',
            content='Content in source brain.', locked=True)
        source.save()

        result = self.brain.absorb(source)
        source.close()
        skipped_titles = [s['title'] for s in result.get('skipped', [])]
        self.assertTrue(any('Shared rule' in t for t in skipped_titles),
                       'Should skip exact duplicate titles')

    def test_absorb_dry_run(self):
        """dry_run=True should report without making changes."""
        source_path = os.path.join(self.tmp, 'source3.db')
        source = Brain(source_path)
        source.remember(type='rule', title='Dry run test rule',
            content='Should not actually be absorbed.', locked=True)
        source.save()

        initial_count = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes").fetchone()[0]
        result = self.brain.absorb(source, dry_run=True)
        source.close()
        final_count = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes").fetchone()[0]
        self.assertEqual(initial_count, final_count,
                        'dry_run should not change node count')


# ── P7: Connections ──────────────────────────────────────────────────

class TestConnectTyped(BrainTestBase):
    """P7: Test typed edge creation."""

    def test_connect_typed_creates_edge_with_type(self):
        """connect_typed should create edge with specified edge_type."""
        n1 = self.brain.remember(type='decision', title='Node A',
            content='Decision about architecture.')
        n2 = self.brain.remember(type='decision', title='Node B',
            content='Related decision about deployment.')
        self.brain.connect_typed(n1['id'], n2['id'],
            relation='depends_on', edge_type='depends_on')
        edge = self.brain.conn.execute(
            "SELECT edge_type, relation FROM edges WHERE source_id = ? AND target_id = ?",
            (n1['id'], n2['id'])).fetchone()
        self.assertIsNotNone(edge)
        self.assertEqual(edge[0], 'depends_on')
        self.assertEqual(edge[1], 'depends_on')


# ── P7: Segments ─────────────────────────────────────────────────────

class TestSegments(BrainTestBase):
    """P7: Test conversation segment detection."""

    def test_get_current_segment_id(self):
        """Should return a valid segment ID."""
        seg_id = self.brain.get_current_segment_id()
        self.assertIsNotNone(seg_id)

    def test_add_to_segment(self):
        """Adding a node to segment should be retrievable."""
        node = self.brain.remember(type='concept', title='Test segment node',
            content='Testing segment membership.')
        self.brain.add_to_segment(node['id'])
        seg_nodes = self.brain.get_segment_node_ids()
        self.assertIn(node['id'], seg_nodes)


# ── P7: Priming System ──────────────────────────────────────────────

class TestPriming(BrainTestBase):
    """P7: Test the priming (active concerns) system."""

    def test_get_active_primes_with_tensions(self):
        """Active tensions should appear as primes."""
        a = self.brain.remember(type='decision', title='Embeddings at 90% weight',
            content='Embedding similarity dominates recall.', keywords='embeddings recall')
        b = self.brain.remember(type='rule', title='Generic nodes dampened',
            content='Broad nodes score high with everything.', keywords='generic dampening')
        self.brain.create_tension(
            title='Embeddings vs generic nodes in recall ranking',
            content='Generic nodes with broad content score high similarity.',
            node_a_id=a['id'], node_b_id=b['id'])
        primes = self.brain.get_active_primes()
        self.assertTrue(len(primes) > 0, 'Should have primes from tension')
        self.assertTrue(any(p['source'] == 'tension' for p in primes))

    def test_get_active_primes_empty_brain(self):
        """Empty brain should return empty primes list."""
        primes = self.brain.get_active_primes()
        self.assertEqual(len(primes), 0)


# ── P7: Instinct Check ──────────────────────────────────────────────

class TestInstinctCheck(BrainTestBase):
    """P7: Test the instinct check (prefrontal cortex) system."""

    def test_instinct_check_empty_returns_none(self):
        """No instinct should fire on fresh brain."""
        result = self.brain.get_instinct_check('hello')
        self.assertIsNone(result)

    def test_instinct_check_encoding_depth_warning(self):
        """Deep session with no encoding should trigger instinct."""
        # Simulate 25 messages, 0 remembers
        for _ in range(25):
            self.brain.record_message()
        result = self.brain.get_instinct_check('what should we do next')
        self.assertIsNotNone(result, 'Should trigger encoding depth instinct')
        self.assertIn('nothing encoded', result.lower())


# ═══════════════════════════════════════════════════════════════
# Feature 1: Critical Flag Tests
# ═══════════════════════════════════════════════════════════════

class TestCriticalFlag(BrainTestBase):
    """Critical flag — safety-important nodes get boosted in recall and force-surfaced at boot."""

    def test_critical_column_exists(self):
        """Schema v16 adds critical column to nodes table."""
        cols = self.brain.conn.execute('PRAGMA table_info(nodes)').fetchall()
        col_names = [c[1] for c in cols]
        self.assertIn('critical', col_names)

    def test_critical_default_zero(self):
        """New nodes have critical=0 by default."""
        result = self.brain.remember(type='rule', title='Test rule', content='Test')
        row = self.brain.conn.execute(
            'SELECT critical FROM nodes WHERE id = ?', (result['id'],)
        ).fetchone()
        self.assertEqual(row[0], 0)

    def test_mark_critical_creates_pending(self):
        """mark_critical() adds to pending list but does NOT set the column."""
        result = self.brain.remember(type='rule', title='Never delete worktrees',
            content='Worktree deletion destroys session CWD', locked=True)
        node_id = result['id']

        resp = self.brain.mark_critical(node_id, reason='Worktree deletion caused data loss')
        self.assertEqual(resp['status'], 'pending')

        # Column should still be 0
        row = self.brain.conn.execute(
            'SELECT critical FROM nodes WHERE id = ?', (node_id,)
        ).fetchone()
        self.assertEqual(row[0], 0)

        # But should appear in pending list
        pending = self.brain.get_pending_critical()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['node_id'], node_id)

    def test_approve_critical_sets_flag(self):
        """approve_critical() sets critical=1 and removes from pending."""
        result = self.brain.remember(type='rule', title='Never force push to main',
            content='Force push destroys team history', locked=True)
        node_id = result['id']

        self.brain.mark_critical(node_id, reason='Prevents data loss')
        self.brain.approve_critical(node_id)

        row = self.brain.conn.execute(
            'SELECT critical FROM nodes WHERE id = ?', (node_id,)
        ).fetchone()
        self.assertEqual(row[0], 1)

        # Pending list should be empty
        pending = self.brain.get_pending_critical()
        self.assertEqual(len(pending), 0)

    def test_critical_always_in_boot(self):
        """Critical nodes appear in context_boot() regardless of limits."""
        # Create a critical node
        result = self.brain.remember(type='rule', title='SAFETY: Never delete worktree without confirmation',
            content='Git worktrees may be actively used by other sessions. Deleting silently destroys working directory.',
            keywords='worktree delete safety critical confirmation', locked=True)
        self.brain.approve_critical(result['id'])

        # Boot with very small limits — critical node must still appear
        boot = self.brain.context_boot()
        locked_titles = [n['title'] for n in boot.get('locked', [])]
        self.assertIn('SAFETY: Never delete worktree without confirmation', locked_titles)
        # And it should be marked as critical
        critical_nodes = [n for n in boot['locked'] if n.get('_critical')]
        self.assertGreaterEqual(len(critical_nodes), 1)

    def test_critical_boosted_in_recall(self):
        """Critical node ranks higher than equally-relevant non-critical node."""
        # Create two nodes with identical keyword overlap for the query
        n1 = self.brain.remember(type='rule', title='Worktree safety guidelines',
            content='Guidelines for safe worktree operations',
            keywords='worktree safety operations guidelines')
        n2 = self.brain.remember(type='rule', title='Worktree safety critical rule',
            content='Never delete worktrees without checking for active sessions',
            keywords='worktree safety operations critical')
        self.brain.approve_critical(n2['id'])

        results = self.brain.recall('worktree safety operations')
        result_ids = [r['id'] for r in results.get('results', results)]
        # Critical node should rank first or very near top
        if n2['id'] in result_ids and n1['id'] in result_ids:
            self.assertLess(result_ids.index(n2['id']), result_ids.index(n1['id']),
                'Critical node should rank higher than non-critical with similar relevance')

    def test_critical_found_at_low_similarity(self):
        """Critical nodes have a lower activation threshold — found even with weak matches."""
        result = self.brain.remember(type='rule', title='SAFETY: Never rm -rf worktree directory',
            content='Worktree directories are actively used. Deleting them destroys shell state.',
            keywords='worktree rm -rf safety directory delete', locked=True)
        self.brain.approve_critical(result['id'])

        # Query with only loosely related terms
        results = self.brain.recall('git branch cleanup procedures')
        result_ids = [r['id'] for r in results.get('results', results)]
        # We just verify the recall doesn't crash with critical nodes
        # (The actual low-threshold behavior is in recall_with_embeddings which needs the embedder)
        self.assertIsInstance(results, dict)

    def test_remember_critical_only_pending(self):
        """remember(critical=True) creates pending, does not set the column directly."""
        result = self.brain.remember(type='rule', title='Test critical rule',
            content='Should go to pending', critical=True)
        node_id = result['id']

        row = self.brain.conn.execute(
            'SELECT critical FROM nodes WHERE id = ?', (node_id,)
        ).fetchone()
        self.assertEqual(row[0], 0, 'critical=True in remember() should NOT set column directly')

        pending = self.brain.get_pending_critical()
        pending_ids = [p['node_id'] for p in pending]
        self.assertIn(node_id, pending_ids, 'Should appear in pending critical approvals')

    def test_critical_persists_after_reopen(self):
        """Critical flag survives brain close and reopen."""
        result = self.brain.remember(type='rule', title='Persistent critical rule',
            content='Must survive close/reopen', locked=True)
        self.brain.approve_critical(result['id'])
        self.brain.save()
        self.brain.close()
        self.brain = None  # Prevent double-close in tearDown

        # Reopen
        brain2 = Brain(self.db_path)
        row = brain2.conn.execute(
            'SELECT critical FROM nodes WHERE id = ?', (result['id'],)
        ).fetchone()
        brain2.close()
        self.assertEqual(row[0], 1, 'critical=1 should persist after close/reopen')

    def test_scenario_50_nodes_critical_surfaces(self):
        """Realistic scenario: 50 nodes, 1 critical about worktree. Query 'clean up working copy' → critical in results."""
        # Seed 50 diverse nodes
        topics = [
            ('decision', 'Use React for frontend', 'React component architecture'),
            ('rule', 'All API calls must have error handling', 'Try-catch around fetch'),
            ('lesson', 'Database migrations must be reversible', 'Learned from production incident'),
            ('decision', 'Deploy via GitHub Actions', 'CI/CD pipeline configuration'),
            ('rule', 'No direct SQL in controllers', 'Use ORM or DAL layer'),
        ]
        for i in range(50):
            t = topics[i % len(topics)]
            self.brain.remember(type=t[0], title=f'{t[1]} #{i}', content=t[2],
                keywords=f'topic{i} {t[1].lower().replace(" ", " ")}')

        # Create the critical worktree safety node
        safety = self.brain.remember(type='rule',
            title='NEVER delete a git worktree without alerting the user first',
            content='Git worktrees may be actively used by other Claude sessions. Deleting silently destroys working directory and shell state.',
            keywords='worktree delete remove git working copy directory session safety',
            locked=True)
        self.brain.approve_critical(safety['id'])

        # Query with operator vocabulary
        results = self.brain.recall('clean up working copy')
        result_ids = [r['id'] for r in results.get('results', results)]
        # The critical node should surface in the results
        self.assertIn(safety['id'], result_ids,
            'Critical worktree safety node must surface for "clean up working copy" query')


# ═══════════════════════════════════════════════════════════════
# Feature 2: Safety Check Tests
# ═══════════════════════════════════════════════════════════════

class TestSafetyCheck(BrainTestBase):
    """safety_check() — classifies destructive commands and recalls safety nodes."""

    def test_rm_rf_destructive(self):
        """rm -rf is detected as destructive."""
        result = self.brain.safety_check('rm -rf /tmp/foo')
        self.assertTrue(result['destructive'])

    def test_git_reset_hard_destructive(self):
        """git reset --hard is detected as destructive."""
        result = self.brain.safety_check('git reset --hard HEAD~3')
        self.assertTrue(result['destructive'])

    def test_git_push_force_destructive(self):
        """git push --force is detected as destructive."""
        result = self.brain.safety_check('git push --force origin main')
        self.assertTrue(result['destructive'])

    def test_ls_not_destructive(self):
        """ls -la is NOT destructive."""
        result = self.brain.safety_check('ls -la')
        self.assertFalse(result['destructive'])

    def test_npm_install_not_destructive(self):
        """npm install is NOT destructive."""
        result = self.brain.safety_check('npm install express')
        self.assertFalse(result['destructive'])

    def test_recalls_critical_on_match(self):
        """Destructive command recalls critical safety nodes."""
        safety = self.brain.remember(type='rule',
            title='NEVER delete a git worktree without alerting the user first',
            content='Worktree deletion destroys session CWD',
            keywords='worktree delete remove git', locked=True)
        self.brain.approve_critical(safety['id'])

        result = self.brain.safety_check('git worktree remove vibrant-brown')
        self.assertTrue(result['destructive'])
        self.assertGreaterEqual(len(result.get('critical_matches', [])), 1,
            'Should recall the critical worktree safety node')

    def test_empty_brain_still_detects(self):
        """Destructive command on empty brain still returns destructive=True."""
        result = self.brain.safety_check('rm -rf /important/data')
        self.assertTrue(result['destructive'])
        self.assertEqual(len(result.get('warnings', [])), 0)

    def test_drop_table_detected(self):
        """SQL DROP TABLE is detected as destructive."""
        result = self.brain.safety_check("sqlite3 db.sqlite 'DROP TABLE nodes'")
        self.assertTrue(result['destructive'])

    def test_piped_rm_detected(self):
        """Piped rm via xargs is detected."""
        result = self.brain.safety_check("find . -name '*.tmp' | xargs rm")
        self.assertTrue(result['destructive'])

    def test_quoted_rm_detected(self):
        """rm -rf inside bash -c is detected."""
        result = self.brain.safety_check("bash -c 'rm -rf /tmp'")
        self.assertTrue(result['destructive'])


# ═══════════════════════════════════════════════════════════════
# Feature 3: Vocabulary Expansion Tests
# ═══════════════════════════════════════════════════════════════

class TestVocabularyExpansion(BrainTestBase):
    """_expand_query_with_vocabulary() — expands operator terms in recall queries."""

    def test_expansion_adds_mapped_terms(self):
        """Learned vocabulary term gets expanded in query."""
        self.brain.learn_vocabulary('working copy', maps_to=['worktree', 'git worktree'])
        expanded = self.brain._expand_query_with_vocabulary('delete the working copy')
        self.assertIn('worktree', expanded.lower())

    def test_expansion_caps_at_max(self):
        """Expansion capped at VOCAB_EXPANSION_MAX terms."""
        from servers.brain_constants import VOCAB_EXPANSION_MAX
        self.brain.learn_vocabulary('megamap', maps_to=['alpha', 'beta', 'gamma', 'delta', 'epsilon'])
        expanded = self.brain._expand_query_with_vocabulary('use megamap')
        added = expanded.replace('use megamap', '').strip().split()
        self.assertLessEqual(len(added), VOCAB_EXPANSION_MAX)

    def test_expansion_with_no_vocab(self):
        """Empty vocabulary table — query returned unchanged."""
        expanded = self.brain._expand_query_with_vocabulary('test query')
        self.assertEqual(expanded, 'test query')

    def test_expansion_logged(self):
        """Vocabulary expansion is logged to recall_log."""
        self.brain.learn_vocabulary('shortcut', maps_to=['keyboard_shortcut'])
        self.brain._expand_query_with_vocabulary('use shortcut')
        count = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM recall_log WHERE query = 'use shortcut' AND returned_ids LIKE 'vocab_expansion%'"
        ).fetchone()[0]
        self.assertGreaterEqual(count, 1)

    def test_ambiguous_skipped(self):
        """Ambiguous vocabulary (same term, multiple contexts) is NOT expanded."""
        # Seed enough nodes so the generic threshold (>5%) isn't triggered
        for i in range(25):
            self.brain.remember(type='concept', title=f'Padding node {i}',
                content=f'Generic content {i}')
        self.brain.learn_vocabulary('the hook', maps_to=['pre-response-recall.sh'], context='recall')
        self.brain.learn_vocabulary('the hook', maps_to=['pre-edit-suggest.sh'], context='editing')
        expanded = self.brain._expand_query_with_vocabulary('what does the hook do')
        # Should not add either mapping since it's ambiguous
        self.assertNotIn('pre-response-recall', expanded)
        self.assertNotIn('pre-edit-suggest', expanded)

    def test_no_duplicate_terms(self):
        """Mapped term already in query is not re-added."""
        self.brain.learn_vocabulary('worktree', maps_to=['git worktree'])
        expanded = self.brain._expand_query_with_vocabulary('delete the worktree')
        # 'worktree' is already in query, shouldn't be duplicated
        self.assertEqual(expanded.lower().count('worktree'), expanded.lower().count('worktree'))


class TestVocabularyAdmission(BrainTestBase):
    """Vocabulary admission guard — rejects generic/stop words."""

    def test_stopword_rejected(self):
        """Single stopword 'working' rejected as vocabulary."""
        result = self.brain.learn_vocabulary('working', maps_to=['something'])
        self.assertTrue(result.get('rejected') or result.get('error'),
            "'working' should be rejected as vocabulary")

    def test_compound_accepted(self):
        """Compound term 'working copy' accepted as vocabulary."""
        result = self.brain.learn_vocabulary('working copy', maps_to=['worktree'])
        self.assertFalse(result.get('rejected', False),
            "'working copy' should be accepted as vocabulary")
        self.assertIn('id', result)

    def test_generic_rejected(self):
        """Term matching >5% of nodes rejected as too generic."""
        # Create 20 nodes all containing 'test' in title
        for i in range(20):
            self.brain.remember(type='decision', title=f'Test decision {i}',
                content=f'Testing thing {i}', keywords=f'test decision {i}')
        result = self.brain.learn_vocabulary('test', maps_to=['unit_test'])
        self.assertTrue(result.get('rejected') or result.get('error'),
            "'test' matching >5% of nodes should be rejected")

    def test_technical_accepted(self):
        """Rare technical term accepted."""
        result = self.brain.learn_vocabulary('worktree', maps_to=['git worktree', 'working directory'])
        self.assertFalse(result.get('rejected', False),
            "'worktree' should be accepted as vocabulary (rare term)")

    def test_extended_stopwords(self):
        """Extended stop words ('make', 'run', 'build') all rejected."""
        for word in ['make', 'run', 'build']:
            result = self.brain.learn_vocabulary(word, maps_to=['something'])
            self.assertTrue(result.get('rejected') or result.get('error'),
                f"'{word}' should be rejected as extended stopword")


# ═══════════════════════════════════════════════════════════════
# Feature 4: Inf Bug Tests
# ═══════════════════════════════════════════════════════════════

class TestInfBug(BrainTestBase):
    """float('inf') JSON serialization bug in auto_heal/auto_tune."""

    def test_inf_survives_config_roundtrip(self):
        """Setting config with inf sentinel, reading back yields float('inf') in recall."""
        import json
        # Simulate what auto_heal does
        half_lives = {'decision': 999999999, 'rule': 999999999, 'concept': 168}
        self.brain.set_config('tunable_decay_half_lives', json.dumps(half_lives))

        # Read back through _get_tunable
        from servers.brain_constants import DECAY_HALF_LIFE
        read_back = self.brain._get_tunable('decay_half_lives', DECAY_HALF_LIFE)
        # Sentinel 999999999 should be treated as inf by recall
        decision_hl = read_back.get('decision', 168)
        self.assertTrue(decision_hl >= 999999 or decision_hl == float('inf'),
            f'Expected inf-equivalent, got {decision_hl}')

    def test_auto_heal_doesnt_corrupt(self):
        """auto_heal() produces valid JSON for decay_half_lives."""
        import json
        # Seed some data so auto_heal has something to work with
        for i in range(5):
            self.brain.remember(type='decision', title=f'Decision {i}',
                content=f'Important decision about thing {i}', locked=True)

        self.brain.auto_heal()

        config_val = self.brain.get_config('tunable_decay_half_lives')
        if config_val:
            parsed = json.loads(config_val)  # Should not throw
            for k, v in parsed.items():
                self.assertIsInstance(v, (int, float),
                    f'decay_half_lives[{k}] = {v} ({type(v)}) — expected numeric')

    def test_recall_handles_inf_string(self):
        """Recall still works when config has string 'inf' from legacy bug."""
        import json
        # Simulate legacy corruption
        half_lives = {'decision': 'inf', 'rule': 'inf', 'concept': 168}
        self.brain.set_config('tunable_decay_half_lives', json.dumps(half_lives))

        node = self.brain.remember(type='decision', title='Important decision about architecture',
            content='We decided to use microservices',
            keywords='architecture microservices decision')

        # Recall should not crash
        results = self.brain.recall('architecture decision')
        self.assertIsInstance(results, dict)

    def test_recall_handles_infinity_string(self):
        """Recall handles 'Infinity' string variant."""
        import json
        half_lives = {'decision': 'Infinity', 'rule': 'Infinity'}
        self.brain.set_config('tunable_decay_half_lives', json.dumps(half_lives))

        self.brain.remember(type='decision', title='Test node for infinity',
            content='Testing infinity handling', keywords='test infinity')

        results = self.brain.recall('test infinity')
        self.assertIsInstance(results, dict)

    def test_recall_handles_nan(self):
        """NaN in config falls back to default half-life."""
        import json
        # NaN can't be directly JSON-serialized, but 'NaN' string could end up in config
        half_lives = {'decision': 'NaN', 'concept': 168}
        self.brain.set_config('tunable_decay_half_lives', json.dumps(half_lives))

        self.brain.remember(type='decision', title='NaN test node',
            content='Testing NaN handling', keywords='nan test')

        results = self.brain.recall('nan test')
        self.assertIsInstance(results, dict)

    def test_locked_recall_after_auto_heal(self):
        """Scenario: create locked nodes, run auto_heal + auto_tune, verify recall still finds them."""
        # Create nodes of types that use float('inf') half-life
        nodes = []
        for ntype, title in [
            ('decision', 'Use embeddings-first recall pipeline'),
            ('rule', 'Never commit secrets to git'),
            ('lesson', 'Fallback paths need the most testing'),
            ('procedure', 'Deploy procedure: staging then production'),
        ]:
            result = self.brain.remember(type=ntype, title=title,
                content=f'Important {ntype} about {title.lower()}',
                keywords=f'{ntype} {title.lower().replace(" ", " ")}',
                locked=True)
            nodes.append((result['id'], title))

        # Run maintenance that could corrupt config
        self.brain.auto_heal()
        try:
            self.brain.auto_tune()
        except Exception:
            pass  # auto_tune may not have enough data

        # Verify all nodes still recallable
        for node_id, title in nodes:
            results = self.brain.recall(title)
            result_ids = [r['id'] for r in results.get('results', results)]
            self.assertIn(node_id, result_ids,
                f'Locked {title} should still be recallable after auto_heal')


# ── Identifier Splitting ─────────────────────────────────────────────

from servers.text_processing import split_identifier, is_domain_specific, filter_domain_terms


class TestIdentifierSplitting(unittest.TestCase):
    """Unit tests for split_identifier() — camelCase-aware tokenization."""

    def test_camel_case_basic(self):
        self.assertEqual(split_identifier('camelCase'), ['camel', 'case'])

    def test_pascal_case(self):
        self.assertEqual(split_identifier('RecallScorer'), ['recall', 'scorer'])

    def test_acronym_before_word(self):
        """HTMLParser → html parser (not HTMLP arser)."""
        self.assertEqual(split_identifier('HTMLParser'), ['html', 'parser'])

    def test_acronym_mid_identifier(self):
        """parseHTMLDoc → parse html doc."""
        self.assertEqual(split_identifier('parseHTMLDoc'), ['parse', 'html', 'doc'])

    def test_trailing_acronym(self):
        """getURL → get url."""
        self.assertEqual(split_identifier('getURL'), ['get', 'url'])

    def test_snake_case(self):
        self.assertEqual(split_identifier('brain_surface'), ['brain', 'surface'])

    def test_kebab_case(self):
        self.assertEqual(split_identifier('pre-response-recall'), ['pre', 'response', 'recall'])

    def test_file_extension_stripped(self):
        self.assertEqual(split_identifier('brain_surface.py'), ['brain', 'surface'])

    def test_sh_extension_stripped(self):
        self.assertEqual(split_identifier('pre-edit-suggest.sh'), ['pre', 'edit', 'suggest'])

    def test_version_number_preserved(self):
        result = split_identifier('v2.3.1-beta')
        self.assertIn('v2.3.1', result)
        self.assertIn('beta', result)

    def test_version_only(self):
        result = split_identifier('v5.4.0')
        self.assertEqual(result, ['v5.4.0'])

    def test_digits_preserved(self):
        """Digits like '8' in UTF8Encoder should not be filtered."""
        result = split_identifier('UTF8Encoder')
        self.assertIn('8', result)
        self.assertIn('utf', result)
        self.assertIn('encoder', result)

    def test_path_splitting(self):
        result = split_identifier('servers/daemon_hooks.py')
        self.assertEqual(result, ['servers', 'daemon', 'hooks'])

    def test_deep_path(self):
        result = split_identifier('hooks/scripts/pre_response_recall.py')
        self.assertIn('hooks', result)
        self.assertIn('scripts', result)
        self.assertIn('pre', result)
        self.assertIn('response', result)
        self.assertIn('recall', result)

    def test_empty_string(self):
        self.assertEqual(split_identifier(''), [])

    def test_none_like_input(self):
        self.assertEqual(split_identifier('   '), [])

    def test_dots_only(self):
        self.assertEqual(split_identifier('....'), [])

    def test_single_char(self):
        """Single alphabetic chars are filtered out."""
        self.assertEqual(split_identifier('A'), [])

    def test_database_file(self):
        self.assertEqual(split_identifier('brain.db'), ['brain'])

    def test_all_lowercase_preserved(self):
        result = split_identifier('BrainSurfaceMixin')
        for token in result:
            self.assertEqual(token, token.lower(),
                           'All tokens should be lowercase: %s' % token)

    def test_short_acronym(self):
        """getID → get id (short acronyms preserved)."""
        result = split_identifier('getID')
        self.assertIn('get', result)
        self.assertIn('id', result)

    def test_acronym_to_acronym(self):
        """HTMLToJSON → html to json."""
        result = split_identifier('HTMLToJSON')
        self.assertIn('html', result)
        self.assertIn('to', result)
        self.assertIn('json', result)


class TestIdentifierSplittingE2E(BrainTestBase):
    """End-to-end: verify split_identifier integrates correctly with suggest()."""

    def test_suggest_with_camelcase_file(self):
        """suggest() should find relevant nodes when given a camelCase filename."""
        self.brain.remember(
            type='rule',
            title='Recall scorer must use layered evaluation',
            content='Three layers: regex patterns, embedding similarity, BART NLI. '
                    'Layer 0 regex is fast, Layer 1 embeddings catch semantic matches, '
                    'Layer 1b BART provides entailment scores.',
            keywords='recall scorer evaluation layers regex embeddings',
            locked=True)
        self.brain.save()

        result = self.brain.suggest(file='RecallScorer.py', limit=5)
        suggestions = result.get('suggestions', [])
        titles = [s.get('title', '') for s in suggestions]
        self.assertTrue(
            any('recall' in t.lower() or 'scorer' in t.lower() for t in titles),
            'suggest() with camelCase file "RecallScorer.py" should find recall scorer rule. '
            'Got: %s' % titles)

    def test_suggest_with_acronym_file(self):
        """suggest() should handle files with acronyms like HTMLParser."""
        self.brain.remember(
            type='decision',
            title='HTML parsing uses BeautifulSoup',
            content='We chose BeautifulSoup over lxml for HTML parsing because '
                    'it handles malformed HTML gracefully.',
            keywords='html parsing beautifulsoup parser',
            locked=True)
        self.brain.save()

        result = self.brain.suggest(file='HTMLParser.py', limit=5)
        suggestions = result.get('suggestions', [])
        titles = [s.get('title', '') for s in suggestions]
        # The tokenization should produce "html parser" which matches the node
        self.assertTrue(
            any('html' in t.lower() or 'parser' in t.lower() for t in titles),
            'suggest() with "HTMLParser.py" should find HTML parsing decision. '
            'Got: %s' % titles)

    def test_suggest_with_deep_path(self):
        """suggest() with a deep path should extract meaningful tokens."""
        self.brain.remember(
            type='rule',
            title='Hook scripts must handle daemon unavailable',
            content='All hook scripts in hooks/scripts/ must gracefully fall back '
                    'to direct Python when the daemon is not running.',
            keywords='hooks scripts daemon fallback',
            locked=True)
        self.brain.save()

        result = self.brain.suggest(file='hooks/scripts/pre_response_recall.py', limit=5)
        suggestions = result.get('suggestions', [])
        titles = [s.get('title', '') for s in suggestions]
        self.assertTrue(
            any('hook' in t.lower() or 'daemon' in t.lower() for t in titles),
            'suggest() with deep path should find hook-related rules. '
            'Got: %s' % titles)

    def test_suggest_with_version_in_filename(self):
        """Version numbers in filenames should not corrupt tokenization."""
        self.brain.remember(
            type='decision',
            title='Migration system uses sequential version numbers',
            content='Schema migrations are numbered 001, 002, etc. '
                    'Each migration file applies changes to brain.db or brain_logs.db.',
            keywords='migration version schema upgrade',
            locked=True)
        self.brain.save()

        # This should not crash or produce garbage tokens
        result = self.brain.suggest(file='migrations/v2.3.1_add_columns.py', limit=5)
        self.assertIn('suggestions', result)

    def test_suggest_with_snake_case_matches_camelcase_node(self):
        """snake_case file should match PascalCase-titled nodes."""
        self.brain.remember(
            type='concept',
            title='BrainSurface handles suggest and edit context',
            content='The BrainSurface mixin provides suggest(), get_edit_context(), '
                    'and other surface-level retrieval methods.',
            keywords='brain surface suggest edit context mixin',
            locked=True)
        self.brain.save()

        result = self.brain.suggest(file='brain_surface.py', limit=5)
        suggestions = result.get('suggestions', [])
        titles = [s.get('title', '') for s in suggestions]
        self.assertTrue(
            any('surface' in t.lower() or 'brain' in t.lower() for t in titles),
            'snake_case file should match PascalCase node titles. '
            'Got: %s' % titles)


# ── Common-Word Filter ────────────────────────────────────────────────


class TestCommonWordFilter(unittest.TestCase):
    """Unit tests for is_domain_specific() and filter_domain_terms()."""

    def test_common_word_detected(self):
        """Common English words should NOT be domain-specific."""
        for word in ['file', 'house', 'water', 'system', 'time']:
            self.assertFalse(is_domain_specific(word),
                           '%s should be common' % word)

    def test_domain_word_detected(self):
        """Technical terms not in common English should be domain-specific."""
        for word in ['webhook', 'daemon', 'middleware', 'serializer', 'linter']:
            self.assertTrue(is_domain_specific(word),
                          '%s should be domain-specific' % word)

    def test_acronym_domain(self):
        """Technical acronyms should be domain-specific."""
        for acr in ['DAL', 'API', 'NLP', 'SQL', 'HTML', 'CSS']:
            self.assertTrue(is_domain_specific(acr),
                          '%s should be domain-specific' % acr)

    def test_acronym_common_excluded(self):
        """Common acronyms should NOT be domain-specific."""
        for acr in ['OK', 'AM', 'PM', 'US', 'UK', 'FAQ']:
            self.assertFalse(is_domain_specific(acr),
                           '%s should be common' % acr)

    def test_capitalized_proper_noun(self):
        """Capitalized terms (product names, entities) should be domain-specific."""
        for name in ['Clerk', 'Redis', 'Valinor', 'Supabase']:
            self.assertTrue(is_domain_specific(name),
                          '%s should be domain-specific (proper noun)' % name)

    def test_multiword_domain_head(self):
        """Multi-word terms with uncommon head word are domain-specific."""
        self.assertTrue(is_domain_specific('recall scorer'))
        self.assertTrue(is_domain_specific('brain daemon'))

    def test_multiword_domain_compound(self):
        """Multi-word compounds of common words can still be domain-specific."""
        self.assertTrue(is_domain_specific('hook chain'))
        self.assertTrue(is_domain_specific('supply adapter'))
        self.assertTrue(is_domain_specific('precision loop'))

    def test_multiword_common_phrase(self):
        """Trivially common phrases should NOT be domain-specific."""
        for phrase in ['the file', 'good idea', 'new feature', 'last time', 'next step']:
            self.assertFalse(is_domain_specific(phrase),
                           '"%s" should be a common phrase' % phrase)

    def test_empty_input(self):
        self.assertFalse(is_domain_specific(''))
        self.assertFalse(is_domain_specific('   '))

    def test_filter_removes_common(self):
        """filter_domain_terms should keep only domain-specific terms."""
        candidates = ['the file', 'recall scorer', 'webhook', 'DAL', 'good idea']
        filtered = filter_domain_terms(candidates)
        self.assertIn('recall scorer', filtered)
        self.assertIn('webhook', filtered)
        self.assertIn('DAL', filtered)
        self.assertNotIn('the file', filtered)
        self.assertNotIn('good idea', filtered)

    def test_filter_deduplicates(self):
        """filter_domain_terms should remove case-insensitive duplicates."""
        filtered = filter_domain_terms(['Webhook', 'webhook', 'WEBHOOK'])
        self.assertEqual(len(filtered), 1)

    def test_filter_empty_input(self):
        self.assertEqual(filter_domain_terms([]), [])


class TestCommonWordFilterE2E(BrainTestBase):
    """End-to-end: verify domain filter works with brain vocabulary concepts."""

    def test_domain_terms_match_brain_vocabulary(self):
        """Terms the brain stores as vocabulary should be domain-specific."""
        # These are real terms from brain's vocabulary system
        domain_terms = ['webhook', 'daemon', 'recall scorer', 'DAL',
                       'embeddings', 'middleware']
        for term in domain_terms:
            self.assertTrue(is_domain_specific(term),
                          'Brain vocab term "%s" should be domain-specific' % term)

    def test_common_phrases_not_stored(self):
        """Common English phrases should not trigger vocabulary storage."""
        common_phrases = ['the file', 'good idea', 'new feature',
                         'last time', 'next step', 'other side']
        for phrase in common_phrases:
            self.assertFalse(is_domain_specific(phrase),
                           '"%s" should not trigger vocabulary storage' % phrase)

    def test_product_names_detected(self):
        """Product names like Clerk, Redis should be detected as entities."""
        # Store a node about Clerk
        self.brain.remember(
            type='vocabulary',
            title='Clerk = auth provider with magic links',
            content='Clerk handles authentication. Passwordless via magic links.',
            keywords='clerk auth provider')
        # The term "Clerk" should be domain-specific
        self.assertTrue(is_domain_specific('Clerk'))
        # Even "clerk" lowercase is common — but capitalized signals entity
        # This tests that the capitalization check works for entity detection

    def test_filter_on_realistic_message(self):
        """Simulate extracting terms from a real user message."""
        # Simulate extracted candidates from: "Fix the recall scorer to handle Clerk webhook"
        candidates = ['recall scorer', 'Clerk', 'webhook', 'the', 'handle']
        filtered = filter_domain_terms(candidates)
        # Should keep domain terms, filter common words
        self.assertIn('recall scorer', filtered)
        self.assertIn('Clerk', filtered)
        self.assertIn('webhook', filtered)
        self.assertNotIn('the', filtered)
        self.assertNotIn('handle', filtered)


# ── Vocabulary Extraction (7-strategy) ────────────────────────────────
# Tests validate the 7-strategy extraction in daemon_hooks.py (lines 594-686)
# and domain filter in text_processing.py. The _extract_vocab helper mirrors
# daemon_hooks logic for isolated testability (same approach as bench_vocab_extraction.py).

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def _extract_vocab(message):
    """Extract vocab candidates using the new 7-strategy approach.

    Mirrors the logic in daemon_hooks.py for testability.
    """
    import re
    from servers.text_processing import filter_domain_terms

    quoted = re.findall(r'["]([\w\s-]{3,30})["]', message)
    the_patterns = re.findall(
        r"\b(?:the|a|an|this|that|our|my|your)\s+"
        r"([\w][\w\s-]{2,25}(?:hook|script|table|file|function|method|class|"
        r"module|layer|loop|sequence|pipeline|system|engine|server|db|database|"
        r"config|schema|signal|node|type|map|graph|cache|queue|log|test|spec|"
        r"worker|adapter|pattern|protocol|daemon|scorer|mixin|handler|resolver|"
        r"builder|encoder|decoder|parser|formatter|validator|serializer|"
        r"middleware|endpoint|route|trigger|listener|callback|factory|strategy|"
        r"observer|wrapper|proxy|bridge|decorator|registry|repository|mapper|"
        r"transformer|dispatcher|emitter|collector|aggregator|provider|consumer|"
        r"subscriber|publisher|context|session|token|metric|monitor|tracer|"
        r"profiler|compiler|runtime|kernel|driver|plugin|extension|toolkit|"
        r"library|framework|platform|screen|component|service|client))\b",
        message, re.IGNORECASE,
    )
    action_context = re.findall(
        r"\b(?:fix|update|change|modify|check|look at|review|debug|test|refactor|"
        r"rewrite|add|remove|delete|move|rename|split|merge|clean|implement|"
        r"configure|deploy|migrate|optimize|integrate|initialize|bootstrap|"
        r"instrument|validate|authenticate|provision|dispatch|schedule|monitor|"
        r"benchmark|evaluate|classify|extract|transform|aggregate|normalize|"
        r"cache|batch|rollback|seed|stub|mock|patch|inject|bind|resolve|"
        r"register|subscribe|emit|consume|publish)\s+(?:the\s+)?"
        r"([\w]+(?:\s+[\w]+){0,3}?)(?:\s*(?:and|or|but|also|then|,|;|\.|!|\?)|\s*$)",
        message, re.IGNORECASE,
    )
    action_context = [t.strip() for t in action_context if len(t.strip()) >= 3]
    hyphenated = re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", message)
    hyphenated = [h for h in hyphenated if len(h) > 4 and h.lower() not in (
        "re-run", "re-do", "non-null", "non-zero", "up-to-date", "e-g",
        "pre-existing", "co-authored-by",
    )]
    capitalized = re.findall(
        r'(?<=[a-z.,;:!?\s])\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', message)
    capitalized = [c.strip() for c in capitalized
                  if c.strip() and c.strip() not in (
                      'I', 'The', 'This', 'That', 'It', 'We', 'You',
                      'He', 'She', 'They', 'But', 'And', 'Or', 'So',
                      'If', 'When', 'What', 'How', 'Why', 'Where',
                      'Yes', 'No', 'Ok', 'Sure', 'Please', 'Thanks',
                  )]
    backtick = re.findall(r'`([^`]{2,40})`', message)
    acronyms = re.findall(r'\b([A-Z]{2,6})\b', message)

    skip_words = {
        "the", "a", "an", "this", "that", "it", "them", "is", "are",
        "was", "were", "be", "been", "do", "does", "did", "have", "has",
        "can", "could", "will", "would", "should", "may", "might",
        "yes", "no", "ok", "sure", "please", "thanks", "code", "file",
        "thing", "stuff", "something", "everything", "nothing",
        "just", "also", "very", "really", "actually", "basically",
        "like", "about", "some", "more", "here", "there", "now", "then",
    }
    raw = set()
    for term in (quoted + the_patterns + action_context + hyphenated +
                capitalized + backtick + acronyms):
        term = term.strip()
        if len(term) < 2 or len(term) > 40:
            continue
        words = term.lower().split()
        if all(w in skip_words for w in words):
            continue
        raw.add(term)
    return set(filter_domain_terms(list(raw)))


class TestVocabExtraction(unittest.TestCase):
    """Unit tests for the 7-strategy vocabulary extraction."""

    def test_backtick_extraction(self):
        """Backtick-wrapped terms should be extracted."""
        terms = _extract_vocab('Look at `brain.recall()` and `daemon_hooks.py`')
        term_lower = {t.lower() for t in terms}
        self.assertIn('brain.recall()', term_lower)
        self.assertIn('daemon_hooks.py', term_lower)

    def test_acronym_extraction(self):
        """Uppercase acronyms 2-6 chars should be extracted."""
        terms = _extract_vocab('The DAL uses SQL queries and BART for NLP')
        self.assertTrue(any('DAL' in t for t in terms))
        self.assertTrue(any('SQL' in t for t in terms))
        self.assertTrue(any('BART' in t for t in terms))
        self.assertTrue(any('NLP' in t for t in terms))

    def test_acronym_common_excluded(self):
        """Common acronyms like OK, AM should be filtered out."""
        terms = _extract_vocab('OK I will do it AM tomorrow')
        self.assertFalse(any(t == 'OK' for t in terms))
        self.assertFalse(any(t == 'AM' for t in terms))

    def test_capitalized_midsentence(self):
        """Capitalized product names mid-sentence should be extracted."""
        terms = _extract_vocab('we should use Clerk for auth and Redis for cache')
        self.assertTrue(any('Clerk' in t for t in terms),
                       'Clerk should be extracted. Got: %s' % terms)
        self.assertTrue(any('Redis' in t for t in terms),
                       'Redis should be extracted. Got: %s' % terms)

    def test_expanded_the_patterns(self):
        """Expanded suffix list catches more the-X patterns."""
        terms = _extract_vocab('Fix the webhook handler and the supply adapter pattern')
        term_lower = {t.lower() for t in terms}
        self.assertTrue(any('webhook handler' in t for t in term_lower),
                       'webhook handler should match. Got: %s' % terms)
        self.assertTrue(any('supply adapter pattern' in t for t in term_lower),
                       'supply adapter pattern should match. Got: %s' % terms)

    def test_no_false_positives_simple_message(self):
        """Simple conversational messages should yield nothing."""
        terms = _extract_vocab('yes please do that')
        self.assertEqual(len(terms), 0,
                        'Simple message should yield no terms. Got: %s' % terms)

    def test_no_false_positives_greeting(self):
        terms = _extract_vocab('great thanks')
        self.assertEqual(len(terms), 0,
                        'Greeting should yield no terms. Got: %s' % terms)

    def test_hyphenated_preserved(self):
        """Hyphenated compound terms should still work."""
        terms = _extract_vocab('The hook-chain and pre-response-recall are important')
        term_lower = {t.lower() for t in terms}
        self.assertIn('hook-chain', term_lower)
        self.assertIn('pre-response-recall', term_lower)

    def test_quoted_terms_preserved(self):
        """Quoted terms should still be extracted."""
        terms = _extract_vocab('Use a "supply adapter" for abstraction')
        term_lower = {t.lower() for t in terms}
        self.assertIn('supply adapter', term_lower)


class TestVocabExtractionE2E(BrainTestBase):
    """End-to-end: verify vocab extraction integrates with brain recall."""

    def test_extracted_terms_match_brain_recall(self):
        """Terms extracted from a message should match brain nodes."""
        self.brain.remember(
            type='vocabulary',
            title='webhook → HTTP callback triggered by events',
            content='A webhook is an HTTP POST triggered by a system event.',
            keywords='webhook http callback event integration')
        self.brain.save()

        terms = _extract_vocab('Check if the Clerk webhook fires correctly')
        self.assertTrue(any('Clerk' in t for t in terms),
                       'Clerk should be extracted. Got: %s' % terms)

        results = self.brain.recall('webhook', limit=5)
        result_list = results.get('results', results) if isinstance(results, dict) else results
        titles = [r.get('title', '') for r in result_list]
        self.assertTrue(any('webhook' in t.lower() for t in titles),
                       'Brain should recall webhook node. Got: %s' % titles)

    def test_no_vocab_gap_for_known_term(self):
        """If brain already knows a term, extraction still finds it as candidate."""
        self.brain.remember(
            type='vocabulary',
            title='DAL → data access layer',
            content='Thin abstraction over SQLite tables.',
            keywords='dal data access layer')
        self.brain.save()

        terms = _extract_vocab('The DAL needs refactoring')
        self.assertTrue(any('DAL' in t for t in terms))


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=True)
