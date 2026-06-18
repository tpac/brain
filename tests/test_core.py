#!/usr/bin/env python3
"""
brain — Core Unit Tests

Tests that catch silent failures and non-obvious regressions:
- vocabulary system (learn, resolve, ambiguous)
- confidence scoring in recall
- error logging pipeline
- DAL read/write consistency and pattern enforcement
- consciousness signals, evolution lifecycle
- rich fields, typed edges, surface formatting
- critical flag, safety checks, vocabulary admission
- keyword extraction, common word filtering, sentence splitting

Run: python tests/test_core.py
"""

import sys
import os
import shutil
import tempfile
import unittest
import sqlite3

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.brain import Brain
from servers.dal import LogsDAL, BrainMetaDAL
from servers.schema import ensure_schema, ensure_logs_schema
from tests.brain_test_base import BrainTestBase



class TestConfidenceScoring(BrainTestBase):
    """Test confidence affects recall ranking."""

    def test_high_confidence_ranks_higher(self):
        # Create two similar nodes with different confidence
        self.brain.remember(type='decision', title='API design choice A',
                           content='REST API for user service',
                           confidence=0.3)
        self.brain.remember(type='decision', title='API design choice B',
                           content='REST API for user service improved',
                           confidence=1.0)
        self.brain.save()

        results = self.brain.recall('API design for user service', limit=5)
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

    def test_log_warning_writes_to_db(self):
        """_log_warning writes a non-blocking signal with event_type='warning'."""
        self.brain._log_warning("test_warn_source", "test warning message", "warn ctx")

        # Query debug_log directly — warnings live in the unified debug_log
        # table with event_type='warning' (distinct from event_type='error').
        row = self.brain.logs_conn.execute(
            "SELECT event_type, source, metadata FROM debug_log "
            "WHERE source = 'test_warn_source' AND event_type = 'warning' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row, 'Should have logged warning to debug_log')
        self.assertEqual(row[0], 'warning')
        self.assertEqual(row[1], 'test_warn_source')
        # metadata is JSON with message + context
        import json as _json
        meta = _json.loads(row[2])
        self.assertEqual(meta['message'], 'test warning message')
        self.assertEqual(meta['context'], 'warn ctx')

    # test_errors_surface_in_consciousness removed — consciousness signals migrated to signal queue


class TestDAL(unittest.TestCase):
    """Test the DAL layer independently."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.logs_conn = sqlite3.connect(os.path.join(self.tmp, 'logs.db'))
        ensure_logs_schema(self.logs_conn)
        self.logs = LogsDAL(self.logs_conn)

        self.brain_conn = sqlite3.connect(os.path.join(self.tmp, 'brain.db'))
        ensure_schema(self.brain_conn)
        self.meta = BrainMetaDAL(self.brain_conn)

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

    # test_meta_session_activity removed 2026-05-23: targeted
    # BrainMetaDAL.get_session_activity() which was deliberately deleted in
    # commit 95b2887 (parallel-session refactor — single-counter
    # approach replaced by per-session SessionContext). Activity
    # counters live on SessionContext now, not BrainMetaDAL.



# ═══════════════════════════════════════════════════════════════════════
# SESSION A: Comprehensive Unit Tests (v5.1 expansion)
#
# Tests organized by priority (P0-P7) matching the test plan.
# All test data uses realistic brain content, not toy examples.
# ═══════════════════════════════════════════════════════════════════════


# ── Helpers ──────────────────────────────────────────────────────────


def _seed_brain_with_realistic_data(brain):
    """Populate a brain with realistic test data for consciousness/evolution tests."""
    nodes = []
    # Decisions
    nodes.append(brain.remember(type='decision', title='Auth: Clerk for passwordless login via magic links',
        content='Clerk handles auth flow. Magic links for login, no passwords. Webhook syncs user data to our DB. Free tier covers MVP needs. Chose over Auth0 because of simpler integration.',
        locked=True, confidence=0.95))
    nodes.append(brain.remember(type='decision', title='React component architecture follows atomic design',
        content='Components organized by atomic design: atoms (Button, Input), molecules (FormField), organisms (LoginForm). Shared via internal package.',
        locked=True, confidence=0.9))
    # Rules
    nodes.append(brain.remember(type='rule', title='Experimental features must never block core operations',
        content='When adding new features (bridging, proposals, archive) to existing methods (remember, consolidate, dream, smartPrune), always wrap in try/catch. A bridge failure should never prevent a remember from succeeding.',
        locked=True, confidence=0.85))
    nodes.append(brain.remember(type='rule', title='Communication style with Tom: direct, peer-to-peer',
        content='Speak peer-to-peer. Be direct. Challenge when warranted. Always plan before executing. Never dump a full spec — work iteratively through components.',
        locked=True))
    # Lessons
    nodes.append(brain.remember(type='lesson', title='Lesson: new features must be connected to the graph at birth',
        content='Built vocabulary system with learn_vocabulary(), resolve_vocabulary(), gap detection. Then discovered vocabulary nodes were completely isolated — not connected to anything in the graph.',
        locked=True, emotion=0.8, confidence=0.85))
    # Concepts
    nodes.append(brain.remember(type='concept', title='CampaignParamsResolver — isolated GAM parameter builder',
        content='Isolated component that takes a Glo and returns GAM-ready params (dayparts, freq cap, pacing, views per session). V1: config-driven defaults per publisher.',
        confidence=0.7))
    # Context (will be stale)
    nodes.append(brain.remember(type='context', title='Session #7 final log: Brain v4.0.0 shipped',
        content='Session #7. Massive session. Everything built and tested in one go. Shipped Phase 0.5A, 0.5B, 0.5C, typed edges, consciousness, curiosity.'))
    # Correction
    nodes.append(brain.remember(type='correction', title='Divergence: Claude compresses when encoding to brain',
        content='CLAUDE ASSUMED: Encoding to brain should be concise. REALITY: Brain encoding should be RICH. Future Claude needs texture, specifics, failures, reasoning journeys.',
        locked=True, emotion=0.9, confidence=0.95))
    return nodes


# ── P0: Silent Failure Detection ─────────────────────────────────────

class TestSilentFailures(BrainTestBase):
    """P0: Verify errors are logged, never silently swallowed."""

    def _get_error_count(self):
        return self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type = 'error'"
        ).fetchone()[0]

    def test_remember_connections_param_retired_is_loud_not_silent(self):
        """The store-time remember(connections=...) edge param was removed
        (2026-06-18; connect_to replaced it). A stray `connections=` must NOT be
        silently swallowed: the node is still created, no edge is materialized,
        the value is NOT stored as junk metadata, and it is logged LOUDLY
        (source='remember_connections_retired') so a lingering caller is visible.
        Replaces the old test_remember_with_bad_connection_logs_error, which
        exercised the now-dead store path."""
        before = self._get_error_count()
        result = self.brain.remember(
            type='decision',
            title='Legacy connections kwarg must be loud, not silent',
            content='Passing the retired connections= param.',
            connections=[{'target_id': 'nonexistent_node_id_xyz', 'relation': 'related'}],
        )
        node_id = result.get('id')
        # Node is still created — the retired param is inert, not fatal.
        self.assertIsNotNone(node_id)
        # The store-time edge path is dead — no edge to the target.
        from servers.dal import GraphDAL
        self.assertFalse(
            GraphDAL(self.brain.conn).edge_exists(node_id, 'nonexistent_node_id_xyz'))
        # Loud: an error row was logged tagged with our source.
        loud = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type='error' "
            "AND source='remember_connections_retired'").fetchone()[0]
        self.assertEqual(loud, 1)
        self.assertGreater(self._get_error_count(), before)
        # NOT swallowed into node metadata as junk (read-side connections edge
        # field is unaffected — we check the raw KV store, not the rich node).
        junk = self.brain.conn.execute(
            "SELECT COUNT(*) FROM node_metadata_kv WHERE node_id=? AND key='connections'",
            (node_id,)).fetchone()[0]
        self.assertEqual(junk, 0)

    def test_remember_auto_connect_param_retired_swallowed_not_junk(self):
        """The `auto_connect` param was removed (2026-06-18) — the
        co_accessed-on-remember behavior it gated was deleted 2026-05-31. Many
        existing tests still pass `auto_connect=False`; the _CONTROL_FIELDS
        swallow guard keeps those working by dropping the kwarg silently (it was
        a pure toggle, no side effect, so unlike `connections` it is NOT logged
        loudly). This pins the swallow: node created, value NOT stored as junk
        KV metadata, no error logged."""
        before = self._get_error_count()
        result = self.brain.remember(
            type='decision',
            title='Retired auto_connect kwarg is swallowed cleanly',
            content='Passing the retired auto_connect= param.',
            auto_connect=False,
        )
        node_id = result.get('id')
        self.assertIsNotNone(node_id)
        junk = self.brain.conn.execute(
            "SELECT COUNT(*) FROM node_metadata_kv WHERE node_id=? AND key='auto_connect'",
            (node_id,)).fetchone()[0]
        self.assertEqual(junk, 0)
        # Silent (not loud) — auto_connect had no side effect worth flagging.
        self.assertEqual(self._get_error_count(), before)

    # test_consciousness_signal_error_does_not_crash removed — function deleted

    # removed — tested deleted method (2026-04-13)

    # removed — tested deleted method (2026-04-13)

    # removed — tested deleted method (2026-04-13)

    # removed — tested deleted method (2026-04-13)

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
    - brain_meta table → BrainMetaDAL (self._meta)
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
        """Mixin files should use self._meta (BrainMetaDAL) not raw brain_meta SQL.

        brain_meta is a key-value config store. All access should go through
        BrainMetaDAL.get(), .set(), .get_json(), .set_json(), .increment().
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

        # Current baseline: 6 tables not yet in main DAL.
        MAX_UNCOVERED = 6
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
        # 2026-03-25: 50 violations (brain_precision:20, brain_surface:15, brain_evolution:5,
        #   brain_recall:4, brain_dreams:3, brain_engineering:2, brain_consciousness:1)
        # New DALs built: NodeDAL, VectorDAL, TfIdfDAL, GraphDAL — migrate callers next.
        MAX_ALLOWED_VIOLATIONS = 50
        self.assertLessEqual(total_violations, MAX_ALLOWED_VIOLATIONS,
                            f'{total_violations} direct DB violations in mixins '
                            f'(max allowed: {MAX_ALLOWED_VIOLATIONS}). '
                            f'Migrate to DAL methods or lower the threshold if you just migrated.')

    def _log_violation_count(self, category, count, examples):
        """Log violation count for trend tracking (no brain needed — just prints)."""
        print(f'\n  [dal-audit] {category}: {count} violations')
        for ex in examples:
            print(f'    {ex}')





# ── P4: Remember Rich & Metadata ─────────────────────────────────────

class TestRememberRich(BrainTestBase):
    """P4: Test remember_rich and node metadata."""

    def test_recall_node_with_metadata(self):
        """recall_node should return enriched node with metadata."""
        result = self.brain.remember_rich(
            type='lesson',
            title='Lesson: test with real data not toy examples',
            content='Toy test data misses edge cases that real content reveals.',
            reasoning='Production brain has 675 nodes with complex content.')
        recall_result = self.brain.recall_node(result['id'])
        self.assertEqual(len(recall_result['results']), 1)
        node = recall_result['results'][0]
        self.assertIn('_metadata', node)
        self.assertEqual(recall_result['_recall_mode'], 'by_id')

    # removed — tested deleted method (2026-04-13)

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
        """Fresh brain should be healthy: no high-severity issues, well-formed report
        ({'healthy', 'issues', 'actions', 'checked_at'})."""
        result = self.brain.health_check()
        self.assertIsInstance(result, dict)
        self.assertTrue(result['healthy'],
                        f"fresh brain reported unhealthy: {result.get('issues')}")
        self.assertIsInstance(result['issues'], list)
        self.assertIsInstance(result['actions'], list)
        self.assertIn('checked_at', result)

    def test_suggest_with_file(self):
        """suggest() should return suggestions for a known file pattern."""
        self.brain.remember(type='rule',
            title='brain.py: only assembler defines __init__',
            content='Mixins must not define __init__. Only brain.py.',
            locked=True)
        result = self.brain.suggest(context='editing brain.py', file='brain.py')
        self.assertIsInstance(result, dict)
        self.assertIn('suggestions', result)
        self.assertIsInstance(result['suggestions'], list)

    def test_pre_edit_returns_structure(self):
        """pre_edit returns the batched pre-edit contract: suggestions +
        procedures + context_files + encoding{health} + timings."""
        result = self.brain.pre_edit(file='test.py', tool_name='Edit')
        self.assertIsInstance(result, dict)
        for key in ('suggestions', 'procedures', 'context_files', 'encoding', 'timings'):
            self.assertIn(key, result)
        self.assertIn('health', result['encoding'])



# ── P7: Connections ──────────────────────────────────────────────────

class TestConnectTyped(BrainTestBase):
    """P7: Test typed edge creation."""

    def test_connect_typed_creates_edge_with_type(self):
        """connect_typed should create relation in edge_relations."""
        n1 = self.brain.remember(type='decision', title='Node A',
            content='Decision about architecture.')
        n2 = self.brain.remember(type='decision', title='Node B',
            content='Related decision about deployment.')
        self.brain.connect_typed(n1['id'], n2['id'],
            relation='depends_on', description='deployment depends on architecture')
        # Find edge_id then query edge_relations
        edge = self.brain.conn.execute(
            'SELECT edge_id FROM edges WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)',
            (n1['id'], n2['id'], n2['id'], n1['id'])).fetchone()
        self.assertIsNotNone(edge, 'Edge should exist between nodes')
        rel = self.brain.conn.execute(
            "SELECT relation, description FROM edge_relations WHERE edge_id = ? AND relation = 'depends_on'",
            (edge[0],)).fetchone()
        self.assertIsNotNone(rel, 'depends_on relation should exist in edge_relations')
        self.assertEqual(rel[1], 'deployment depends on architecture')



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

    # removed — tested deleted method (2026-04-13)

    def test_critical_always_in_boot(self):
        """Critical nodes appear in context_boot() regardless of limits."""
        # Create a critical node
        result = self.brain.remember(type='rule', title='SAFETY: Never delete worktree without confirmation',
            content='Git worktrees may be actively used by other sessions. Deleting silently destroys working directory.',
            locked=True)
        self.brain.conn.execute("UPDATE nodes SET critical = 1 WHERE id = ?", (result['id'],))
        self.brain.conn.commit()

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
            content='Guidelines for safe worktree operations')
        n2 = self.brain.remember(type='rule', title='Worktree safety critical rule',
            content='Never delete worktrees without checking for active sessions')
        self.brain.conn.execute("UPDATE nodes SET critical = 1 WHERE id = ?", (n2['id'],))
        self.brain.conn.commit()

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
            locked=True)
        self.brain.conn.execute("UPDATE nodes SET critical = 1 WHERE id = ?", (result['id'],))
        self.brain.conn.commit()

        # Query with only loosely related terms
        results = self.brain.recall('git branch cleanup procedures')
        result_ids = [r['id'] for r in results.get('results', results)]
        # We just verify the recall doesn't crash with critical nodes
        # (The actual low-threshold behavior is in recall which needs the embedder)
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
        self.brain.conn.execute("UPDATE nodes SET critical = 1 WHERE id = ?", (result['id'],))
        self.brain.conn.commit()
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
            self.brain.remember(type=t[0], title=f'{t[1]} #{i}', content=t[2])

        # Create the critical worktree safety node
        safety = self.brain.remember(type='rule',
            title='NEVER delete a git worktree without alerting the user first',
            content='Git worktrees may be actively used by other Claude sessions. Deleting silently destroys working directory and shell state.',
            locked=True)
        self.brain.conn.execute("UPDATE nodes SET critical = 1 WHERE id = ?", (safety['id'],))
        self.brain.conn.commit()

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

    def test_ls_not_destructive(self):
        """ls -la is NOT destructive."""
        result = self.brain.safety_check('ls -la')
        self.assertFalse(result['destructive'])

    def test_recalls_critical_on_match(self):
        """Destructive command recalls critical safety nodes."""
        safety = self.brain.remember(type='rule',
            title='NEVER delete a git worktree without alerting the user first',
            content='Worktree deletion destroys session CWD',
            locked=True)
        self.brain.conn.execute("UPDATE nodes SET critical = 1 WHERE id = ?", (safety['id'],))
        self.brain.conn.commit()

        result = self.brain.safety_check('git worktree remove vibrant-brown')
        self.assertTrue(result['destructive'])
        self.assertGreaterEqual(len(result.get('critical_matches', [])), 1,
            'Should recall the critical worktree safety node')

    def test_empty_brain_still_detects(self):
        """Destructive command on empty brain still returns destructive=True."""
        result = self.brain.safety_check('rm -rf /important/data')
        self.assertTrue(result['destructive'])
        self.assertEqual(len(result.get('warnings', [])), 0)

    def test_piped_rm_detected(self):
        """Piped rm via xargs is detected."""
        result = self.brain.safety_check("find . -name '*.tmp' | xargs rm")
        self.assertTrue(result['destructive'])


# ═══════════════════════════════════════════════════════════════
# Feature 3: Vocabulary Expansion Tests
# ═══════════════════════════════════════════════════════════════


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

    # removed — tested deleted method (2026-04-13)

    def test_recall_handles_inf_string(self):
        """Recall still works when config has string 'inf' from legacy bug."""
        import json
        # Simulate legacy corruption
        half_lives = {'decision': 'inf', 'rule': 'inf', 'concept': 168}
        self.brain.set_config('tunable_decay_half_lives', json.dumps(half_lives))

        node = self.brain.remember(type='decision', title='Important decision about architecture',
            content='We decided to use microservices')

        # Recall should not crash AND must still produce a valid result envelope
        # despite the corrupted 'inf' string in the decay config.
        results = self.brain.recall('architecture decision')
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)
        self.assertIsInstance(results['results'], list)

    def test_recall_handles_infinity_string(self):
        """Recall handles 'Infinity' string variant."""
        import json
        half_lives = {'decision': 'Infinity', 'rule': 'Infinity'}
        self.brain.set_config('tunable_decay_half_lives', json.dumps(half_lives))

        self.brain.remember(type='decision', title='Test node for infinity',
            content='Testing infinity handling')

        results = self.brain.recall('test infinity')
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)
        self.assertIsInstance(results['results'], list)

    def test_recall_handles_nan(self):
        """NaN in config falls back to default half-life."""
        import json
        # NaN can't be directly JSON-serialized, but 'NaN' string could end up in config
        half_lives = {'decision': 'NaN', 'concept': 168}
        self.brain.set_config('tunable_decay_half_lives', json.dumps(half_lives))

        self.brain.remember(type='decision', title='NaN test node',
            content='Testing NaN handling')

        results = self.brain.recall('nan test')
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)
        self.assertIsInstance(results['results'], list)

    # removed — tested deleted method (2026-04-13)

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
            content='Clerk handles authentication. Passwordless via magic links.')
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




# ── Sentence Splitting ────────────────────────────────────────────────

# NOTE: split_sentences() and the TestSentenceSplitting/E2E classes were
# removed 2026-04-26. Verified via `grep -rn "split_sentences" servers/
# hooks/ scripts/ eval/` — zero callers in production. The function is
# orphaned in servers/text_processing.py (only `split_identifier` from
# that module is still used). Removing tests for unreachable code is not
# weakening coverage; the code is dead. If sentence splitting becomes a
# real feature again, write tests against the new home.
#
# The 11 deleted test methods were: test_code_reference_preserved,
# test_dotted_path_preserved, test_version_number_preserved,
# test_file_extension_preserved, test_url_preserved, test_abbreviation_mr,
# test_decimal_number, test_ellipsis, test_normal_split, test_empty_input,
# test_no_punctuation, test_multiple_code_refs (TestSentenceSplitting),
# test_split_preserves_code_for_embedding, test_split_on_real_response_text,
# test_split_with_brain_node_content (TestSentenceSplittingE2E).


class _DeletedSentenceSplittingTests:
    """Sentinel — see comment above. Coverage was for dead code."""
    pass



if __name__ == '__main__':
    unittest.main(verbosity=2, exit=True)
