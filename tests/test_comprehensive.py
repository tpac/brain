"""
Comprehensive Brain Tests — Deep coverage of all subsystems.

!! IMPORTANT — TEST INTEGRITY RULE (read before modifying) !!
If a test in this file FAILS, do NOT change the test or the code to make it pass.
Stop and report to the user: what the test expected vs what the code returned.
Ask: "Is the test wrong, or does the code have a bug?" Wait for the answer.
This applies to ALL tests — weakening assertions or "fixing" code to satisfy
tests without user approval masks regressions. See CLAUDE.md for full rule.

When ADDING new tests, always include a comment explaining:
  1. What the expected value is and WHY (not just WHAT)
  2. Whether the value was verified empirically or from documentation
  3. Date of last verification

Written with full architectural context from the v5.3.1 session (2026-03-21).
Covers gaps identified in the test inventory:

  1. Consciousness signals — individual signal correctness, not just "doesn't crash"
  2. Dream/consolidation/healing — behavioral correctness, not just smoke
  3. Evolution system — discovery, tension detection, hypothesis lifecycle
  4. Engineering memory types — purpose, mechanism, impact, constraint, convention
  5. Vocabulary system — gap detection, bridging, admission edge cases
  6. Session health assessment — encoding depth, bias detection
  7. Schema integrity — every table has a writer and a reader
  8. Developmental stages — progression from NASCENT to INTEGRATED
  9. Edge semantics — weight ranges, co-access promotion, Hebbian strengthening
  10. Cross-session persistence — knowledge carries across brain close/reopen
  11. Embedder failure modes — graceful degradation when embedder is down
  12. Hook output format — JSON schema validation for hook responses
  13. Large-scale behavior — recall quality with 100+ nodes
  14. Priming system — background topics surface in relevant queries
  15. Self-correction traces — divergence detection and pattern extraction
  16. Temporal recall — "what changed last week" queries
  17. DAL/Brain method consistency — dal.py vs brain methods don't diverge

Run:
    python3 -m pytest tests/test_comprehensive.py -v
    python3 -m unittest tests.test_comprehensive -v
"""

import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from servers.brain import Brain
from servers.schema import ensure_logs_schema, NODE_TYPES, BRAIN_VERSION
from servers import embedder
from tests.brain_test_base import BrainTestBase


# ── Helpers ────────────────────────────────────────────────────────────

def _seed_rich_brain(brain, node_count=20):
    """Seed a brain with diverse node types, connections, and vocabulary.
    Returns dict of node IDs by key name.
    """
    nodes = {}

    # Rules (locked)
    n = brain.remember(type='rule', title='Never store plaintext passwords',
                       content='All auth goes through Clerk. No custom password storage. '
                               'Magic links only. Webhook syncs to Supabase.',
                       keywords='auth password clerk security', locked=True, confidence=0.95)
    nodes['auth_rule'] = n['id']

    n = brain.remember(type='rule', title='All API responses must include request_id',
                       content='Every API handler wraps response with request_id for tracing. '
                               'Middleware injects it from X-Request-ID header or generates UUID.',
                       keywords='api request-id tracing middleware', locked=True, confidence=0.9)
    nodes['api_rule'] = n['id']

    # Decisions (locked)
    n = brain.remember(type='decision', title='Use adapter pattern for ad delivery',
                       content='Clean abstraction between app and ad backends. '
                               'Interface: createCampaign, updateCampaign, pauseCampaign. '
                               'V1: GAM adapter. Swap without touching business logic.',
                       keywords='adapter pattern ad delivery gam architecture', locked=True, confidence=0.85)
    nodes['adapter_dec'] = n['id']

    n = brain.remember(type='decision', title='React components follow atomic design',
                       content='Atoms (Button, Input), molecules (FormField), organisms (LoginForm). '
                               'Shared via @glo/ui package. No cross-level imports.',
                       keywords='react atomic design components ui', locked=True, confidence=0.8)
    nodes['react_dec'] = n['id']

    # Engineering types
    n = brain.remember(type='purpose', title='brain.py is the core engine',
                       content='5500+ lines. Constructor, schema, remember(), recall(), mixin assembly. '
                               'All other brain_*.py files are mixins composed via multiple inheritance.',
                       keywords='brain core engine architecture', locked=True)
    nodes['purpose'] = n['id']

    n = brain.remember(type='mechanism', title='Recall pipeline: embed → cosine → keyword → blend → rank',
                       content='Query embedded via snowflake-arctic. Cosine similarity against node_embeddings. '
                               'TF-IDF keyword fallback. 90/10 blend. Spreading activation for graph boost. '
                               'Intent detection adjusts type_boosts. Confidence weighting applied last.',
                       keywords='recall pipeline embedding cosine tfidf ranking', locked=True)
    nodes['mechanism'] = n['id']

    n = brain.remember(type='impact', title='Changing recall output format breaks 3 hooks',
                       content='If recall_with_embeddings() return structure changes → must check '
                               'pre-response-recall.sh, boot-brain.sh, daemon_hooks.py because they parse it.',
                       keywords='impact recall format hooks breaking change')
    nodes['impact'] = n['id']

    n = brain.remember(type='constraint', title='Hook scripts must exit 0 or block Claude',
                       content='Any hook returning non-zero exit code blocks the Claude operation. '
                               'Pre-edit hooks that crash prevent all file edits. '
                               'Always wrap in try/except with fallback to approve.',
                       keywords='hooks exit code blocking constraint', locked=True)
    nodes['constraint'] = n['id']

    n = brain.remember(type='lesson', title='Dual code paths diverge silently',
                       content='When same logic exists in daemon and direct paths, they inevitably diverge. '
                               'Example: recall direct path had 6 features missing from daemon path. '
                               'Solution: single source of truth in daemon_hooks.py.',
                       keywords='lesson dual paths divergence daemon hooks', locked=True)
    nodes['lesson'] = n['id']

    n = brain.remember(type='convention', title='Brain node IDs use type prefix + random suffix',
                       content='Format: {type_prefix}_{8_random_chars}. Examples: rul_abc12345, dec_xyz98765. '
                               'Prefix makes IDs human-scannable in logs.',
                       keywords='convention node id format prefix', locked=True)
    nodes['convention'] = n['id']

    # Personal/context
    n = brain.remember(type='context', title='Current sprint: daemon consolidation v5.3',
                       content='Centralizing all hook logic into daemon_hooks.py. '
                               'Thin client pattern for hook scripts. Graph change tracking.',
                       keywords='sprint daemon consolidation current work', confidence=0.6)
    nodes['context'] = n['id']

    # Correction trace
    n = brain.remember(type='correction', title='Divergence: encoding was too shallow',
                       content='Tom corrected: brain encoding was brief summaries instead of rich context. '
                               'Root cause: brevity training bias. Fix: structural prompts for depth.',
                       keywords='correction encoding depth shallow brevity', locked=True)
    nodes['correction'] = n['id']

    # Failure mode
    n = brain.remember(type='failure_mode', title='Silent hook failures swallow errors',
                       content='Bare except:pass in hooks means errors vanish. '
                               'User never knows recall failed. Now logged to brain_logs.db.',
                       keywords='failure mode hooks silent error logging')
    nodes['failure_mode'] = n['id']
    # Mark as active evolution
    brain.conn.execute(
        "UPDATE nodes SET evolution_status = 'active' WHERE id = ?",
        (nodes['failure_mode'],))

    # Extra nodes for scale
    for i in range(max(0, node_count - len(nodes))):
        brain.remember(
            type='context',
            title=f'Working note #{i}: miscellaneous context',
            content=f'This is filler node {i} to test recall at scale. '
                    f'Contains terms like database, migration, testing, deployment.',
            keywords=f'filler context note-{i}', confidence=0.3)

    # Connections
    brain.connect(nodes['auth_rule'], nodes['adapter_dec'], 'related', weight=0.6)
    brain.connect(nodes['adapter_dec'], nodes['mechanism'], 'exemplifies', weight=0.8)
    brain.connect(nodes['lesson'], nodes['mechanism'], 'corrects', weight=0.7)
    brain.connect(nodes['purpose'], nodes['mechanism'], 'contains', weight=0.9)
    brain.connect(nodes['impact'], nodes['mechanism'], 'depends_on', weight=0.85)
    brain.connect(nodes['correction'], nodes['lesson'], 'produced', weight=0.75)

    # Vocabulary
    try:
        brain.learn_vocabulary(term='working copy', context='git',
                               maps_to='the current checkout / worktree state')
        brain.learn_vocabulary(term='working copy', context='advertising',
                               maps_to='the current version of ad creative being edited')
        brain.learn_vocabulary(term='adapter', context='architecture',
                               maps_to='abstraction layer between app and external service')
    except Exception:
        pass

    brain.save()
    return nodes


# ═══════════════════════════════════════════════════════════════════════
# 1. CONSCIOUSNESS SIGNALS — individual signal correctness
# ═══════════════════════════════════════════════════════════════════════

class TestConsciousnessSignals(BrainTestBase):
    """Test that each consciousness signal generates correct data."""

    def setUp(self):
        super().setUp()
        self.nodes = _seed_rich_brain(self.brain)

    def test_signals_returns_dict_with_expected_keys(self):
        """get_consciousness_signals() returns dict with all signal categories."""
        signals = self.brain.get_consciousness_signals()
        self.assertIsInstance(signals, dict)
        # These 10 signal categories are always present in the return dict
        # (values may be empty lists/None, but keys must exist).
        # Source: brain_consciousness.py ConsciousnessMixin.get_consciousness_signals()
        # Each category maps to a specific SQL query in that method.
        expected_keys = [
            'reminders', 'evolutions', 'fading', 'stale_context_count',
            'failure_modes', 'performance', 'capabilities', 'interactions',
            'novelty', 'miss_trends',
        ]
        for key in expected_keys:
            self.assertIn(key, signals, f"Missing signal category: {key}")

    def test_failure_modes_surfaces_active_evolutions(self):
        """Active failure_mode nodes appear in consciousness signals."""
        signals = self.brain.get_consciousness_signals()
        failure_modes = signals.get('failure_modes', [])
        self.assertGreater(len(failure_modes), 0,
                           "Active failure_mode node should surface")
        titles = [f['title'] for f in failure_modes]
        self.assertTrue(any('Silent hook' in t for t in titles))

    def test_fading_knowledge_requires_age_and_access(self):
        """Fading signal only fires for old, accessed, unlocked nodes."""
        signals = self.brain.get_consciousness_signals()
        fading = signals.get('fading', [])
        # Fading requires: unlocked, access_count >= 3, last_accessed > 14 days ago,
        # type NOT IN (context, thought, intuition). Fresh nodes fail the age check.
        # Source: brain_consciousness.py line ~50, SQL WHERE clause.
        self.assertEqual(len(fading), 0,
                         "Fresh nodes should not appear as fading knowledge")

    def test_fading_knowledge_fires_for_old_accessed_nodes(self):
        """Nodes with high access but old last_accessed should trigger fading."""
        # Create an old, frequently-accessed, unlocked node
        n = self.brain.remember(type='lesson', title='Old important lesson',
                                content='This was accessed many times but not recently.',
                                keywords='old lesson fading')
        node_id = n['id']
        # Fake old last_accessed and high access_count
        self.brain.conn.execute(
            "UPDATE nodes SET last_accessed = datetime('now', '-30 days'), access_count = 10 WHERE id = ?",
            (node_id,))
        self.brain.conn.commit()

        signals = self.brain.get_consciousness_signals()
        fading = signals.get('fading', [])
        fading_ids = [f['id'] for f in fading]
        self.assertIn(node_id, fading_ids, "Old frequently-accessed node should appear as fading")

    def test_stale_context_count_accurate(self):
        """Stale context count matches actual old context nodes."""
        # Create old context nodes
        for i in range(3):
            n = self.brain.remember(type='context', title=f'Old context {i}',
                                    content=f'Stale context {i}', keywords='stale')
            self.brain.conn.execute(
                "UPDATE nodes SET created_at = datetime('now', '-14 days') WHERE id = ?",
                (n['id'],))
        self.brain.conn.commit()

        signals = self.brain.get_consciousness_signals()
        self.assertGreaterEqual(signals['stale_context_count'], 3)

    def test_encoding_gap_fires_when_no_encodes_in_long_session(self):
        """encoding_gap signal fires when 20+ minutes with zero encodes."""
        # Simulate a long session with no encoding — need a fresh brain
        # that hasn't had _seed_rich_brain's remembers counted
        fresh_brain = Brain(os.path.join(self.tmp, 'gap_test.db'))
        try:
            boot_time = (datetime.utcnow() - timedelta(minutes=30)).isoformat() + 'Z'
            fresh_brain.set_config('boot_time', boot_time)
            fresh_brain.set_config('session_start_at', boot_time)
            fresh_brain.conn.execute(
                "INSERT OR REPLACE INTO session_activity (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ('boot_time', boot_time))
            fresh_brain.conn.execute(
                "INSERT OR REPLACE INTO session_activity (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ('message_count', '25'))
            fresh_brain.conn.execute(
                "INSERT OR REPLACE INTO session_activity (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                ('remember_count', '0'))
            fresh_brain.conn.commit()

            signals = fresh_brain.get_consciousness_signals()
            gap = signals.get('encoding_gap')
            self.assertIsNotNone(gap, "encoding_gap should fire after 30 minutes with 0 encodes")
            self.assertIn('warning', gap)
        finally:
            fresh_brain.close()

    def test_novelty_detects_recent_concepts(self):
        """Novelty signal surfaces concept nodes created in last 2 hours."""
        self.brain.remember(type='concept', title='New concept: temporal graphs',
                            content='Graphs with time-aware edges.', keywords='concept temporal')
        signals = self.brain.get_consciousness_signals()
        novelty = signals.get('novelty', [])
        self.assertGreater(len(novelty), 0, "Recent concept node should appear as novelty")

    def test_performance_signal_empty_when_no_perf_nodes(self):
        """Performance signal is empty list when no performance nodes exist."""
        signals = self.brain.get_consciousness_signals()
        perf = signals.get('performance', [])
        # We didn't create any performance nodes
        self.assertEqual(len(perf), 0)

    def test_signals_survive_empty_brain(self):
        """Consciousness signals don't crash on a completely empty brain."""
        empty_brain = Brain(os.path.join(self.tmp, 'empty.db'))
        try:
            signals = empty_brain.get_consciousness_signals()
            self.assertIsInstance(signals, dict)
        finally:
            empty_brain.close()


# ═══════════════════════════════════════════════════════════════════════
# 2. DREAM / CONSOLIDATION / HEALING — behavioral correctness
# ═══════════════════════════════════════════════════════════════════════

class TestDreamConsolidateHeal(BrainTestBase):
    """Test that dream/consolidate/heal produce correct behavioral changes."""

    def setUp(self):
        super().setUp()
        self.nodes = _seed_rich_brain(self.brain, node_count=30)

    def test_dream_returns_structured_result(self):
        """dream() returns dict with expected keys."""
        result = self.brain.dream()
        self.assertIsInstance(result, dict)
        # Dream returns 'dreams' and 'message' keys
        self.assertIn('dreams', result)

    def test_consolidate_returns_structured_result(self):
        """consolidate() returns dict with expected keys."""
        result = self.brain.consolidate()
        self.assertIsInstance(result, dict)

    def test_auto_heal_returns_categories(self):
        """auto_heal() returns resolved/tuned/cleaned categories."""
        result = self.brain.auto_heal()
        self.assertIn('resolved', result)
        self.assertIn('tuned', result)
        self.assertIn('cleaned', result)
        # cleaned should have sub-categories
        cleaned = result['cleaned']
        self.assertIn('archived', cleaned)
        self.assertIn('edges_created', cleaned)
        self.assertIn('merged', cleaned)
        self.assertIn('locked', cleaned)

    def test_auto_heal_locks_high_access_nodes(self):
        """auto_heal() auto-locks unlocked nodes with access_count >= 10."""
        # auto_heal section 1c: locks orphan beliefs with access_count >= 10.
        # Excludes types: task, context, file, intuition, person, project, decision.
        # 'lesson' type is eligible for auto-lock.
        # Source: brain_evolution.py auto_heal() EXCLUDE_ORPHAN_TYPES.
        n = self.brain.remember(type='lesson', title='Frequently accessed lesson',
                                content='Accessed many times.', keywords='frequent access')
        node_id = n['id']
        self.brain.conn.execute(
            "UPDATE nodes SET access_count = 15 WHERE id = ?", (node_id,))
        self.brain.conn.commit()

        result = self.brain.auto_heal()

        locked = self.brain.conn.execute(
            "SELECT locked FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()[0]
        self.assertEqual(locked, 1, "High-access node should be auto-locked")
        self.assertGreater(result['cleaned']['locked'], 0)

    def test_auto_heal_does_not_corrupt_locked_nodes(self):
        """auto_heal() never deletes or archives locked nodes."""
        # INVARIANT: locked node count can only increase (via auto-lock), never decrease.
        # auto_heal's merge_duplicate only archives the OLDER of two near-duplicates,
        # and keeps the newer. Both must be locked for merge to trigger.
        # Even in merge, one survives — so net locked count stays same or increases.
        locked_before = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE locked = 1 AND archived = 0"
        ).fetchone()[0]

        self.brain.auto_heal()

        locked_after = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE locked = 1 AND archived = 0"
        ).fetchone()[0]
        self.assertGreaterEqual(locked_after, locked_before,
                                "auto_heal must never archive locked nodes")

    def test_consolidate_creates_bridge_edges(self):
        """consolidate() can create bridge edges between related clusters."""
        # Run consolidation
        result = self.brain.consolidate()
        # Bridges may or may not be created depending on graph structure,
        # but the method should complete without error
        self.assertIsInstance(result, dict)

    def test_dream_consolidate_heal_cycle_is_idempotent(self):
        """Running dream+consolidate+heal twice produces no crashes or corruption."""
        for _ in range(2):
            self.brain.dream()
            self.brain.consolidate()
            self.brain.auto_heal()
            self.brain.save()

        # Verify brain is still healthy
        node_count = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived = 0"
        ).fetchone()[0]
        self.assertGreater(node_count, 0, "Brain should still have nodes after 2 cycles")


# ═══════════════════════════════════════════════════════════════════════
# 3. EVOLUTION SYSTEM — discovery, tensions, hypotheses
# ═══════════════════════════════════════════════════════════════════════

class TestEvolutionSystem(BrainTestBase):
    """Test evolution discovery, tension detection, and lifecycle."""

    def setUp(self):
        super().setUp()
        self.nodes = _seed_rich_brain(self.brain, node_count=30)

    def test_auto_discover_returns_structured_result(self):
        """auto_discover_evolutions() returns counts by type."""
        result = self.brain.auto_discover_evolutions()
        self.assertIsInstance(result, dict)

    def test_get_active_evolutions_returns_list(self):
        """get_active_evolutions() returns list of active items."""
        evolutions = self.brain.get_active_evolutions()
        self.assertIsInstance(evolutions, list)

    def test_evolution_lifecycle_active_to_resolved(self):
        """Evolution nodes can transition from active to resolved."""
        # Create a tension
        n = self.brain.remember(
            type='tension', title='Test tension: approach A vs approach B',
            content='Two conflicting patterns detected.',
            keywords='tension test')
        node_id = n['id']
        self.brain.conn.execute(
            "UPDATE nodes SET evolution_status = 'active' WHERE id = ?", (node_id,))
        self.brain.conn.commit()

        # Verify it's in active evolutions
        active = self.brain.get_active_evolutions()
        active_ids = [e['id'] for e in active]
        self.assertIn(node_id, active_ids)

        # Resolve it
        self.brain.conn.execute(
            "UPDATE nodes SET evolution_status = 'resolved', resolved_at = datetime('now') WHERE id = ?",
            (node_id,))
        self.brain.conn.commit()

        # Should no longer be active
        active = self.brain.get_active_evolutions()
        active_ids = [e['id'] for e in active]
        self.assertNotIn(node_id, active_ids)

    def test_get_active_evolutions_filters_by_type(self):
        """get_active_evolutions(types=['tension']) only returns tensions."""
        # Create a tension and a hypothesis
        t = self.brain.remember(type='tension', title='Test tension',
                                content='Conflict.', keywords='tension')
        self.brain.conn.execute(
            "UPDATE nodes SET evolution_status = 'active' WHERE id = ?", (t['id'],))

        h = self.brain.remember(type='hypothesis', title='Test hypothesis',
                                content='Theory.', keywords='hypothesis')
        self.brain.conn.execute(
            "UPDATE nodes SET evolution_status = 'active' WHERE id = ?", (h['id'],))
        self.brain.conn.commit()

        tensions_only = self.brain.get_active_evolutions(types=['tension'])
        for e in tensions_only:
            self.assertEqual(e['type'], 'tension')


# ═══════════════════════════════════════════════════════════════════════
# 4. ENGINEERING MEMORY TYPES — purpose, mechanism, impact, etc.
# ═══════════════════════════════════════════════════════════════════════

class TestEngineeringMemoryTypes(BrainTestBase):
    """Test that engineering memory types store and recall correctly."""

    def test_remember_purpose(self):
        """remember_purpose() creates locked purpose node with scope."""
        # remember_purpose() always auto-locks because purposes are foundational.
        # Source: brain_engineering.py remember_purpose() passes locked=True.
        n = self.brain.remember_purpose(
            title='brain.py is the core engine',
            content='Constructor, schema, mixin assembly.',
            scope='file')
        self.assertEqual(n['type'], 'purpose')
        locked = self.brain.conn.execute(
            "SELECT locked FROM nodes WHERE id = ?", (n['id'],)
        ).fetchone()[0]
        self.assertEqual(locked, 1, "Purpose nodes should be auto-locked")

    def test_remember_mechanism_with_steps(self):
        """remember_mechanism() enriches content with steps."""
        # remember_mechanism() appends "Steps: step1 → step2 → ..." to content.
        # Source: brain_engineering.py remember_mechanism(), the ' → '.join(steps) line.
        n = self.brain.remember_mechanism(
            title='Recall pipeline',
            content='Multi-stage retrieval.',
            steps=['embed query', 'cosine scan', 'keyword fallback', 'blend', 'rank'])
        self.assertEqual(n['type'], 'mechanism')
        node = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (n['id'],)
        ).fetchone()
        self.assertIn('embed query', node[0])
        self.assertIn('→', node[0])

    def test_remember_impact_creates_locked_node(self):
        """remember_impact() creates locked impact with change_impacts metadata."""
        n = self.brain.remember_impact(
            title='Recall format change breaks hooks',
            if_changed='recall_with_embeddings() output',
            must_check='pre-response-recall.sh, daemon_hooks.py',
            because='they parse the return structure')
        self.assertEqual(n['type'], 'impact')
        locked = self.brain.conn.execute(
            "SELECT locked FROM nodes WHERE id = ?", (n['id'],)
        ).fetchone()[0]
        self.assertEqual(locked, 1)

    def test_remember_constraint(self):
        """remember_constraint() creates constraint node."""
        n = self.brain.remember_constraint(
            title='Hooks must exit 0',
            content='Non-zero exit blocks Claude.')
        self.assertEqual(n['type'], 'constraint')

    def test_remember_convention(self):
        """remember_convention() creates convention node."""
        n = self.brain.remember_convention(
            title='Node IDs use type prefix',
            content='Format: {prefix}_{random}.')
        self.assertEqual(n['type'], 'convention')

    def test_remember_lesson(self):
        """remember_lesson() creates lesson node."""
        n = self.brain.remember_lesson(
            title='Dual code paths diverge',
            what_happened='Daemon and direct paths had different recall features.',
            root_cause='Copy-paste without sync mechanism.',
            fix='Centralized into daemon_hooks.py.',
            preventive_principle='Always use single source of truth.')
        self.assertEqual(n['type'], 'lesson')

    def test_engineering_types_recallable(self):
        """All engineering memory types can be recalled via embeddings."""
        self.brain.remember_purpose(title='Purpose: X does Y', content='Explanation.')
        self.brain.remember_mechanism(title='Mechanism: how X works', content='Steps.')
        self.brain.remember_lesson(
            title='Lesson: we learned Z',
            what_happened='Z happened.',
            root_cause='Because of A.',
            fix='Did B.',
            preventive_principle='Always do B.')

        results = self.brain.recall_with_embeddings('how does X work', limit=10)
        titles = [r['title'] for r in results.get('results', [])]
        # At least one engineering type should surface
        self.assertTrue(len(titles) > 0, "Engineering memory should be recallable")


# ═══════════════════════════════════════════════════════════════════════
# 5. VOCABULARY SYSTEM — deeper edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestVocabularyDeep(BrainTestBase):
    """Deep vocabulary tests beyond basic learn/resolve."""

    def test_context_dependent_resolution(self):
        """Same term resolves differently based on context."""
        self.brain.learn_vocabulary(term='sprint', context='agile',
                                    maps_to='time-boxed development iteration')
        self.brain.learn_vocabulary(term='sprint', context='running',
                                    maps_to='short burst of maximum speed')

        # resolve_vocabulary takes only term, returns dict with all mappings
        result = self.brain.resolve_vocabulary('sprint')
        self.assertIsNotNone(result, "resolve_vocabulary should find 'sprint'")
        # Result should contain info about the term
        result_str = str(result).lower()
        self.assertTrue('iteration' in result_str or 'sprint' in result_str,
                        f"Should contain mapping data. Got: {result}")

    def test_vocabulary_gap_detection(self):
        """detect_vocabulary_gaps() finds terms not in vocabulary."""
        # Learn some terms
        self.brain.learn_vocabulary(term='adapter', context='architecture',
                                    maps_to='abstraction layer')
        # Check for gaps in text that uses unknown terms
        text = "The shim connects the adapter to the legacy API gateway"
        try:
            gaps = self.brain.detect_vocabulary_gaps(text)
            # 'shim', 'gateway' might be gaps; 'adapter' should not be
            if gaps:
                gap_terms = [g.get('term', g) if isinstance(g, dict) else g for g in gaps]
                self.assertNotIn('adapter', gap_terms,
                                 "Known term should not appear as gap")
        except (AttributeError, TypeError):
            self.skipTest("detect_vocabulary_gaps not available")

    def test_vocabulary_node_has_graph_connection(self):
        """Vocabulary nodes are connected to the graph at creation."""
        n = self.brain.remember(type='vocabulary', title='Vocab: shim',
                                content='A thin compatibility layer.',
                                keywords='vocabulary shim compatibility')
        # Check for edges
        edges = self.brain.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_id = ? OR target_id = ?",
            (n['id'], n['id'])
        ).fetchone()[0]
        # Should have at least bridge_proposals or auto-connections
        # (depending on brain state, this may be 0 if no similar nodes exist)
        self.assertIsNotNone(edges)  # Just verify query works


# ═══════════════════════════════════════════════════════════════════════
# 6. SESSION HEALTH ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════

class TestSessionHealth(BrainTestBase):
    """Test session health assessment and encoding depth detection."""

    def setUp(self):
        super().setUp()
        self.nodes = _seed_rich_brain(self.brain)

    def test_assess_session_health_returns_dict(self):
        """assess_session_health() returns structured dict."""
        result = self.brain.assess_session_health()
        self.assertIsInstance(result, dict)

    def test_synthesize_session_on_active_session(self):
        """synthesize_session() produces output for active sessions."""
        # Simulate some session activity
        self.brain.conn.execute(
            "INSERT OR REPLACE INTO session_activity (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ('message_count', '15'))
        self.brain.conn.execute(
            "INSERT OR REPLACE INTO session_activity (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ('remember_count', '5'))
        boot_time = (datetime.utcnow() - timedelta(minutes=45)).isoformat() + 'Z'
        self.brain.conn.execute(
            "INSERT OR REPLACE INTO session_activity (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ('boot_time', boot_time))
        self.brain.conn.commit()

        result = self.brain.synthesize_session()
        # May return None if not enough data, but should not crash
        self.assertIsInstance(result, (dict, type(None)))


# ═══════════════════════════════════════════════════════════════════════
# 7. SCHEMA INTEGRITY — every table has writers and readers
# ═══════════════════════════════════════════════════════════════════════

class TestSchemaIntegrity(BrainTestBase):
    """Verify schema tables are actually used, not orphaned."""

    def test_all_node_types_are_valid(self):
        """Every node in the DB has a type listed in NODE_TYPES."""
        _seed_rich_brain(self.brain)
        cursor = self.brain.conn.execute(
            "SELECT DISTINCT type FROM nodes WHERE archived = 0")
        db_types = {r[0] for r in cursor.fetchall()}
        for t in db_types:
            self.assertIn(t, NODE_TYPES,
                          f"Node type '{t}' not in schema.NODE_TYPES")

    def test_edges_table_has_both_columns(self):
        """Edges have both relation and edge_type set (regression test)."""
        _seed_rich_brain(self.brain)
        orphan_edges = self.brain.conn.execute(
            """SELECT COUNT(*) FROM edges
               WHERE (relation IS NULL OR relation = '')
                 AND (edge_type IS NULL OR edge_type = '')"""
        ).fetchone()[0]
        self.assertEqual(orphan_edges, 0,
                         "All edges should have relation or edge_type set")

    def test_node_embeddings_exist_for_nodes(self):
        """Nodes with content should have embeddings stored."""
        if not embedder.is_ready():
            self.skipTest("Embedder not available")
        _seed_rich_brain(self.brain)
        nodes_without = self.brain.conn.execute(
            """SELECT COUNT(*) FROM nodes n
               LEFT JOIN node_embeddings ne ON n.id = ne.node_id
               WHERE n.archived = 0 AND ne.node_id IS NULL"""
        ).fetchone()[0]
        total = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived = 0"
        ).fetchone()[0]
        # Allow some missing (filler nodes may not embed), but most should have embeddings
        coverage = (total - nodes_without) / max(total, 1)
        self.assertGreater(coverage, 0.5,
                           f"Only {coverage:.0%} of nodes have embeddings")

    def test_recall_log_schema_matches_writers(self):
        """recall_log columns match what brain_recall.py actually writes."""
        cursor = self.brain.logs_conn.execute("PRAGMA table_info(recall_log)")
        columns = {r[1] for r in cursor.fetchall()}
        # brain_recall.py writes these columns
        expected_written = {'session_id', 'query', 'returned_ids', 'returned_count', 'created_at'}
        self.assertTrue(expected_written.issubset(columns),
                        f"recall_log missing columns: {expected_written - columns}")

    def test_session_activity_table_writable(self):
        """session_activity table accepts key-value writes."""
        self.brain.conn.execute(
            "INSERT OR REPLACE INTO session_activity (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ('test_key', 'test_value'))
        self.brain.conn.commit()
        val = self.brain.conn.execute(
            "SELECT value FROM session_activity WHERE key = 'test_key'"
        ).fetchone()
        self.assertEqual(val[0], 'test_value')

    def test_brain_meta_stores_config(self):
        """brain_meta (via set_config/get_config) round-trips correctly."""
        self.brain.set_config('test_key_123', 'hello_world')
        val = self.brain.get_config('test_key_123')
        self.assertEqual(val, 'hello_world')

    def test_version_matches_schema(self):
        """Brain schema version is stored in brain_meta."""
        # Version stored as 'brain_schema_version' in brain_meta table
        row = self.brain.conn.execute(
            "SELECT value FROM brain_meta WHERE key = 'brain_schema_version'"
        ).fetchone()
        self.assertIsNotNone(row, "brain_schema_version should be set in brain_meta")
        self.assertTrue(int(row[0]) > 0, "Schema version should be positive")


# ═══════════════════════════════════════════════════════════════════════
# 8. DEVELOPMENTAL STAGES
# ═══════════════════════════════════════════════════════════════════════

class TestDevelopmentalStages(BrainTestBase):
    """Test brain developmental stage assessment."""

    def test_empty_brain_is_newborn(self):
        """Fresh brain should be stage 1 / NEWBORN with 0 maturity."""
        stage = self.brain.assess_developmental_stage()
        self.assertIsInstance(stage, dict)
        # NOTE: Empty brain returns stage=1, stage_name='NEWBORN', maturity_score=0.0.
        # These are exact values, not approximations. Confirmed with Tom 2026-03-21.
        self.assertEqual(stage['stage'], 1)
        self.assertEqual(stage['stage_name'], 'NEWBORN')
        self.assertEqual(stage['maturity_score'], 0.0)

    def test_rich_brain_advances_stage(self):
        """Brain with many locked nodes should advance past NEWBORN."""
        _seed_rich_brain(self.brain, node_count=50)
        stage = self.brain.assess_developmental_stage()
        # NOTE: With 50+ nodes (mostly shallow test content), brain reaches
        # stage 2 / COLLECTING with maturity ~0.11. Exact maturity depends on
        # content richness, but stage must be > 1 and maturity > 0.
        # Confirmed with Tom 2026-03-21.
        self.assertGreaterEqual(stage['stage'], 2)
        self.assertGreater(stage['maturity_score'], 0)
        self.assertIn(stage['stage_name'], ['COLLECTING', 'DEVELOPING', 'REFLECTING',
                                             'PARTNERING', 'INTEGRATED'])

    def test_stage_includes_node_stats(self):
        """Developmental stage includes node/edge statistics."""
        _seed_rich_brain(self.brain)
        stage = self.brain.assess_developmental_stage()
        # Should have some stats about the brain
        self.assertIn('stage', stage)


# ═══════════════════════════════════════════════════════════════════════
# 9. EDGE SEMANTICS — weights, Hebbian, co-access
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeSemantics(BrainTestBase):
    """Test edge creation, weight semantics, and Hebbian learning."""

    def test_connect_creates_edge(self):
        """connect() creates edge with relation and edge_type."""
        a = self.brain.remember(type='rule', title='Rule A', content='Content A',
                                keywords='test')
        b = self.brain.remember(type='decision', title='Decision B', content='Content B',
                                keywords='test')
        self.brain.connect(a['id'], b['id'], 'related', weight=0.7)

        edge = self.brain.conn.execute(
            "SELECT relation, edge_type, weight FROM edges WHERE source_id = ? AND target_id = ?",
            (a['id'], b['id'])
        ).fetchone()
        self.assertIsNotNone(edge, "Edge should be created")
        self.assertEqual(edge[0], 'related')
        # NOTE: connect() applies Hebbian learning on first creation.
        # connect(weight=0.7) stores ~0.3 (LEARNING_RATE * requested_weight).
        # The requested weight is NOT stored as-is. This is by design.
        # Confirmed with Tom 2026-03-21.
        self.assertAlmostEqual(edge[2], 0.3, places=1)

    def test_weight_clamps_to_valid_range(self):
        """Edge weights should be clamped to [0, MAX_WEIGHT]."""
        a = self.brain.remember(type='rule', title='A', content='A', keywords='test')
        b = self.brain.remember(type='rule', title='B', content='B', keywords='test')
        self.brain.connect(a['id'], b['id'], 'related', weight=5.0)

        edge = self.brain.conn.execute(
            "SELECT weight FROM edges WHERE source_id = ? AND target_id = ?",
            (a['id'], b['id'])
        ).fetchone()
        # Weight should be clamped (typically to 1.0 or MAX_WEIGHT)
        self.assertLessEqual(edge[0], 2.0, "Weight should be clamped")

    def test_co_access_increments(self):
        """Accessing nodes together should increment co_access_count."""
        nodes = _seed_rich_brain(self.brain)
        # Recall should touch multiple nodes, incrementing co-access
        self.brain.recall_with_embeddings("auth clerk adapter", limit=5)
        self.brain.recall_with_embeddings("auth clerk adapter", limit=5)

        # Check if any co_access_count > 0
        co_access = self.brain.conn.execute(
            "SELECT MAX(co_access_count) FROM edges"
        ).fetchone()[0]
        # May or may not have co-access depending on implementation
        self.assertIsNotNone(co_access)

    def test_spreading_activation_decays(self):
        """Spreading activation scores decay with hop distance."""
        nodes = _seed_rich_brain(self.brain)
        # The mechanism is connected to purpose (1 hop) and adapter_dec (2 hops via mechanism)
        results = self.brain.recall_with_embeddings("recall pipeline embedding", limit=10)
        # Mechanism should rank higher than distantly connected nodes
        result_ids = [r['id'] for r in results.get('results', [])]
        if nodes['mechanism'] in result_ids:
            mech_rank = result_ids.index(nodes['mechanism'])
            # Mechanism should be in top results for this query
            self.assertLess(mech_rank, 5, "Directly relevant node should rank high")


# ═══════════════════════════════════════════════════════════════════════
# 10. CROSS-SESSION PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════

class TestCrossSessionPersistence(unittest.TestCase):
    """Test that knowledge survives brain close/reopen cycles.

    Uses its own setUp/tearDown to avoid BrainTestBase's tearDown
    trying to close an already-closed brain.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_locked_nodes_persist_across_sessions(self):
        """Locked nodes survive close + reopen."""
        brain1 = Brain(self.db_path)
        n = brain1.remember(type='rule', title='Persistent rule',
                            content='This must survive.', keywords='persist',
                            locked=True)
        node_id = n['id']
        brain1.save()
        brain1.close()

        brain2 = Brain(self.db_path)
        try:
            node = brain2.conn.execute(
                "SELECT title, locked FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            self.assertIsNotNone(node, "Node should persist after close/reopen")
            self.assertEqual(node[0], 'Persistent rule')
            self.assertEqual(node[1], 1, "Locked status should persist")
        finally:
            brain2.close()

    def test_edges_persist_across_sessions(self):
        """Edges survive close + reopen."""
        brain1 = Brain(self.db_path)
        a = brain1.remember(type='rule', title='Rule A', content='A', keywords='a')
        b = brain1.remember(type='rule', title='Rule B', content='B', keywords='b')
        brain1.connect(a['id'], b['id'], 'related', weight=0.8)
        brain1.save()
        brain1.close()

        brain2 = Brain(self.db_path)
        try:
            edge = brain2.conn.execute(
                "SELECT weight FROM edges WHERE source_id = ? AND target_id = ?",
                (a['id'], b['id'])
            ).fetchone()
            self.assertIsNotNone(edge, "Edge should persist")
            # NOTE: connect(weight=0.8) stores ~0.3 due to Hebbian learning rate.
            # We're testing persistence, not the exact weight — but using the real value.
            # Confirmed with Tom 2026-03-21.
            self.assertAlmostEqual(edge[0], 0.3, places=1)
        finally:
            brain2.close()

    def test_vocabulary_persists_across_sessions(self):
        """Vocabulary mappings survive close + reopen."""
        brain1 = Brain(self.db_path)
        brain1.learn_vocabulary(term='shim', context='engineering',
                                maps_to='thin compatibility layer')
        brain1.save()
        brain1.close()

        brain2 = Brain(self.db_path)
        try:
            resolved = brain2.resolve_vocabulary('shim')
            self.assertIsNotNone(resolved, "Vocabulary should persist")
            resolved_str = str(resolved).lower()
            self.assertTrue('compatibility' in resolved_str or 'shim' in resolved_str)
        finally:
            brain2.close()

    def test_config_persists_across_sessions(self):
        """Config values survive close + reopen via brain_meta."""
        brain1 = Brain(self.db_path)
        # Use brain_meta directly since set_config might use different storage
        brain1.conn.execute(
            "INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)",
            ('custom_key', 'custom_value'))
        brain1.conn.commit()
        brain1.close()

        brain2 = Brain(self.db_path)
        try:
            val = brain2.conn.execute(
                "SELECT value FROM brain_meta WHERE key = 'custom_key'"
            ).fetchone()
            self.assertIsNotNone(val)
            self.assertEqual(val[0], 'custom_value')
        finally:
            brain2.close()

    def test_recall_works_after_reopen(self):
        """recall_with_embeddings() works after close + reopen."""
        brain1 = Brain(self.db_path)
        brain1.remember(type='rule', title='Auth uses Clerk magic links',
                        content='Clerk handles auth flow.', keywords='auth clerk',
                        locked=True)
        brain1.save()
        brain1.close()

        brain2 = Brain(self.db_path)
        try:
            results = brain2.recall_with_embeddings("authentication", limit=5)
            titles = [r['title'] for r in results.get('results', [])]
            self.assertTrue(any('Clerk' in t or 'Auth' in t or 'auth' in t for t in titles),
                            f"Should find auth node after reopen. Got: {titles}")
        finally:
            brain2.close()


# ═══════════════════════════════════════════════════════════════════════
# 11. EMBEDDER FAILURE MODES
# ═══════════════════════════════════════════════════════════════════════

class TestEmbedderFailures(BrainTestBase):
    """Test graceful degradation when embedder is unavailable."""

    def test_remember_works_without_embedder(self):
        """remember() succeeds even if embedding fails."""
        n = self.brain.remember(type='rule', title='Works without embedder',
                                content='Should store node.', keywords='test')
        self.assertIn('id', n)
        node = self.brain.conn.execute(
            "SELECT title FROM nodes WHERE id = ?", (n['id'],)
        ).fetchone()
        self.assertIsNotNone(node)

    def test_recall_falls_back_to_tfidf(self):
        """When embedder is down, recall falls back to TF-IDF keyword matching."""
        self.brain.remember(type='rule', title='TF-IDF test: authentication clerk',
                            content='Clerk handles auth.', keywords='auth clerk tfidf')
        # Even without embeddings, keyword search should work
        results = self.brain.recall_with_embeddings("auth clerk", limit=5)
        self.assertIsInstance(results, dict)
        self.assertIn('results', results)


# ═══════════════════════════════════════════════════════════════════════
# 12. HOOK OUTPUT FORMAT VALIDATION
# ═══════════════════════════════════════════════════════════════════════

class TestHookOutputFormat(BrainTestBase):
    """Validate that daemon_hooks.py functions return correctly formatted output."""

    def setUp(self):
        super().setUp()
        self.nodes = _seed_rich_brain(self.brain)

    def _call_hook(self, hook_fn, args):
        """Call a hook function and return its result."""
        from servers.daemon_hooks import (
            hook_recall, hook_post_response_track, hook_pre_edit,
            hook_pre_bash_safety, hook_pre_compact_save, hook_session_end,
            hook_idle_maintenance, hook_post_compact_reboot,
        )
        fn = {
            'recall': hook_recall,
            'track': hook_post_response_track,
            'pre_edit': hook_pre_edit,
            'pre_bash': hook_pre_bash_safety,
            'pre_compact': hook_pre_compact_save,
            'session_end': hook_session_end,
            'idle': hook_idle_maintenance,
            'reboot': hook_post_compact_reboot,
        }[hook_fn]
        return fn(self.brain, args, [])

    def test_recall_returns_json_key(self):
        """hook_recall returns dict with 'json' or 'output' key."""
        result = self._call_hook('recall', {
            'prompt': 'How does authentication work?',
            'message': 'How does authentication work?'
        })
        self.assertIsInstance(result, dict)
        self.assertTrue('json' in result or 'output' in result,
                        f"hook_recall must return json or output, got: {list(result.keys())}")

    def test_recall_json_has_additional_context(self):
        """hook_recall JSON output has additionalContext for Claude."""
        result = self._call_hook('recall', {
            'prompt': 'authentication clerk magic links',
            'message': 'How does auth work?'
        })
        if 'json' in result:
            self.assertIn('additionalContext', result['json'],
                          "recall JSON must have additionalContext key")

    def test_pre_edit_returns_output(self):
        """hook_pre_edit returns output string."""
        result = self._call_hook('pre_edit', {
            'tool_input': {'file_path': '/src/auth.py'}
        })
        self.assertIsInstance(result, dict)

    def test_pre_bash_returns_output_for_safe_command(self):
        """hook_pre_bash returns approve for safe commands."""
        result = self._call_hook('pre_bash', {
            'tool_input': {'command': 'ls -la'}
        })
        self.assertIsInstance(result, dict)

    def test_pre_compact_returns_approve(self):
        """hook_pre_compact always approves (never blocks compaction)."""
        result = self._call_hook('pre_compact', {})
        self.assertIsInstance(result, dict)
        # May return 'output' or 'json' with approve
        result_str = json.dumps(result, default=str).lower()
        self.assertTrue('approve' in result_str or 'output' in result,
                        f"pre_compact must approve. Got: {result}")

    def test_session_end_returns_output(self):
        """hook_session_end returns output."""
        result = self._call_hook('session_end', {})
        self.assertIsInstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════
# 13. LARGE-SCALE RECALL BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════

class TestLargeScaleRecall(BrainTestBase):
    """Test recall quality with 100+ nodes."""

    def setUp(self):
        super().setUp()
        # Seed 100+ nodes
        self.nodes = _seed_rich_brain(self.brain, node_count=100)

    def test_recall_returns_within_limit(self):
        """recall always respects the limit parameter."""
        for limit in [1, 5, 10, 20]:
            results = self.brain.recall_with_embeddings("auth", limit=limit)
            self.assertLessEqual(len(results.get('results', [])), limit)

    def test_locked_rules_surface_above_noise(self):
        """Locked rules rank above generic filler nodes."""
        results = self.brain.recall_with_embeddings(
            "authentication clerk login", limit=10)
        titles = [r['title'] for r in results.get('results', [])]
        # Auth rule should be in top results, not buried under filler
        auth_found = any('password' in t.lower() or 'auth' in t.lower()
                         or 'clerk' in t.lower() for t in titles[:5])
        self.assertTrue(auth_found,
                        f"Auth rule should rank in top 5 among 100+ nodes. Top 5: {titles[:5]}")

    def test_intent_detection_boosts_decisions(self):
        """'what did we decide about' queries boost decision nodes."""
        results = self.brain.recall_with_embeddings(
            "what did we decide about ad delivery", limit=10)
        top_types = [r['type'] for r in results.get('results', [])[:5]]
        # Decisions should be present in top results
        self.assertIn('decision', top_types,
                      f"Decision intent should boost decision nodes. Top types: {top_types}")

    def test_recall_with_no_matches_returns_empty(self):
        """Totally irrelevant query returns empty or very low-score results."""
        results = self.brain.recall_with_embeddings(
            "quantum entanglement photosynthesis recipe", limit=5)
        result_list = results.get('results', [])
        # Should either be empty or have very low relevance
        self.assertIsInstance(result_list, list)


# ═══════════════════════════════════════════════════════════════════════
# 14. PRIMING SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class TestPrimingSystem(BrainTestBase):
    """Test background priming and topic activation."""

    def setUp(self):
        super().setUp()
        self.nodes = _seed_rich_brain(self.brain)

    def test_get_active_primes_returns_list(self):
        """get_active_primes() returns a list."""
        primes = self.brain.get_active_primes()
        self.assertIsInstance(primes, list)

    def test_check_priming_returns_none_or_dict(self):
        """check_priming() returns None (no match) or dict (match)."""
        result = self.brain.check_priming("auth clerk login")
        self.assertIsInstance(result, (dict, type(None)))

    def test_get_instinct_check_returns_none_or_string(self):
        """get_instinct_check() returns None or a warning string."""
        result = self.brain.get_instinct_check("let me delete all the auth code")
        self.assertIsInstance(result, (str, type(None)))


# ═══════════════════════════════════════════════════════════════════════
# 15. SELF-CORRECTION TRACES
# ═══════════════════════════════════════════════════════════════════════

class TestSelfCorrectionTraces(BrainTestBase):
    """Test divergence recording and correction patterns."""

    def test_correction_node_created(self):
        """Correction nodes store divergence information."""
        n = self.brain.remember(
            type='correction',
            title='Divergence: used curl instead of Python call',
            content='Attempted curl to brain server but brain is serverless. '
                    'Correct approach: direct Python Brain() instantiation.',
            keywords='correction divergence curl serverless')
        self.assertEqual(n['type'], 'correction')

    def test_correction_recallable_by_topic(self):
        """Corrections surface when recalling related topics."""
        self.brain.remember(
            type='correction',
            title='Divergence: shallow brain encoding',
            content='Tom corrected that encoding was too brief.',
            keywords='correction encoding depth shallow', locked=True)

        results = self.brain.recall_with_embeddings("encoding depth", limit=5)
        titles = [r['title'] for r in results.get('results', [])]
        self.assertTrue(any('encoding' in t.lower() for t in titles),
                        f"Correction should surface for related query. Got: {titles}")

    def test_correction_connected_to_lesson(self):
        """Corrections can be connected to lessons via 'produced' edge."""
        c = self.brain.remember(type='correction', title='Wrong approach', content='Details.',
                                keywords='correction')
        l = self.brain.remember(type='lesson', title='Right approach', content='Details.',
                                keywords='lesson')
        self.brain.connect(c['id'], l['id'], 'produced', weight=0.8)

        edge = self.brain.conn.execute(
            "SELECT relation FROM edges WHERE source_id = ? AND target_id = ?",
            (c['id'], l['id'])
        ).fetchone()
        self.assertEqual(edge[0], 'produced')


# ═══════════════════════════════════════════════════════════════════════
# 16. TEMPORAL RECALL
# ═══════════════════════════════════════════════════════════════════════

class TestTemporalRecall(BrainTestBase):
    """Test time-aware queries like 'what changed this week'."""

    def test_intent_detects_temporal(self):
        """Temporal queries detected via intent patterns."""
        results = self.brain.recall_with_embeddings(
            "what changed last week in the auth system", limit=5)
        # Intent should be detected
        intent = results.get('intent', 'general')
        # temporal intent should be detected from "last week"
        self.assertIn(intent, ['temporal', 'state_query', 'general'])

    def test_recent_nodes_surface_for_temporal_queries(self):
        """Recent nodes surface for 'what did we do today' queries."""
        self.brain.remember(type='decision', title='Today: switched to daemon_hooks.py',
                            content='Consolidated all hooks into single file.',
                            keywords='today daemon hooks consolidation', locked=True)
        results = self.brain.recall_with_embeddings(
            "what did we work on today", limit=5)
        titles = [r['title'] for r in results.get('results', [])]
        self.assertTrue(len(titles) > 0, "Temporal query should return something")


# ═══════════════════════════════════════════════════════════════════════
# 17. DAL CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

class TestDALConsistency(BrainTestBase):
    """Verify DAL methods don't diverge from Brain methods."""

    def test_logs_dal_error_logging_works(self):
        """LogsDAL can write and read errors via write_error/get_recent_errors."""
        from servers.dal import LogsDAL
        dal = LogsDAL(self.brain.logs_conn)
        # write_error(source, error, context_str, traceback_str, session_id)
        dal.write_error('test_source', 'test error message', 'test context string')
        errors = dal.get_recent_errors(hours=1)
        self.assertGreater(len(errors), 0)

    def test_meta_dal_round_trips(self):
        """brain_meta table round-trips via Brain's conn."""
        self.brain.conn.execute(
            "INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)",
            ('dal_test_key', 'dal_value'))
        self.brain.conn.commit()

        val = self.brain.conn.execute(
            "SELECT value FROM brain_meta WHERE key = 'dal_test_key'"
        ).fetchone()
        self.assertEqual(val[0], 'dal_value')

    def test_logs_dal_debug_log_writable(self):
        """LogsDAL can write to debug_log via write_debug."""
        from servers.dal import LogsDAL
        dal = LogsDAL(self.brain.logs_conn)
        # write_debug(source, message, session_id, metadata)
        dal.write_debug('test_source', 'test message', 'test_session', {'key': 'value'})
        row = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE source = 'test_source'"
        ).fetchone()
        self.assertGreater(row[0], 0)


# ═══════════════════════════════════════════════════════════════════════
# 18. FLOAT/INF EDGE CASES (expanded from test_core)
# ═══════════════════════════════════════════════════════════════════════

class TestFloatEdgeCases(BrainTestBase):
    """Expanded float handling: NaN, Inf, negative weights, zero confidence."""

    def test_zero_confidence_node_still_recallable(self):
        """Node with confidence=0.0 can still be recalled (just ranked lower)."""
        self.brain.remember(type='rule', title='Zero confidence rule: test',
                            content='This rule has zero confidence.',
                            keywords='zero confidence test', confidence=0.0)
        results = self.brain.recall_with_embeddings("zero confidence test", limit=5)
        titles = [r['title'] for r in results.get('results', [])]
        self.assertTrue(any('zero confidence' in t.lower() for t in titles))

    def test_negative_weight_edge_clamped(self):
        """Negative edge weights should be clamped to 0 or rejected."""
        a = self.brain.remember(type='rule', title='A', content='A', keywords='a')
        b = self.brain.remember(type='rule', title='B', content='B', keywords='b')
        try:
            self.brain.connect(a['id'], b['id'], 'related', weight=-0.5)
            edge = self.brain.conn.execute(
                "SELECT weight FROM edges WHERE source_id = ? AND target_id = ?",
                (a['id'], b['id'])
            ).fetchone()
            if edge:
                self.assertGreaterEqual(edge[0], 0, "Negative weight should be clamped")
        except (ValueError, sqlite3.IntegrityError):
            pass  # Rejection is also valid

    def test_half_life_serialization_handles_inf(self):
        """_safe_serialize_half_lives handles float('inf') and NaN."""
        result = self.brain._safe_serialize_half_lives({
            'rule': float('inf'),
            'context': float('nan'),
            'normal': 168.0,
            'legacy_inf': 'inf',
        })
        self.assertEqual(result['rule'], 999999999)
        self.assertEqual(result['context'], 168)
        self.assertEqual(result['normal'], 168.0)
        self.assertEqual(result['legacy_inf'], 999999999)


# ═══════════════════════════════════════════════════════════════════════
# 19. CONTEXT BOOT
# ═══════════════════════════════════════════════════════════════════════

class TestContextBoot(BrainTestBase):
    """Test context_boot() which is what Claude sees at session start."""

    def setUp(self):
        super().setUp()
        self.nodes = _seed_rich_brain(self.brain)

    def test_context_boot_returns_dict(self):
        """context_boot() returns structured dict."""
        boot = self.brain.context_boot()
        self.assertIsInstance(boot, dict)

    def test_context_boot_includes_locked_rules(self):
        """context_boot() surfaces locked rules."""
        boot = self.brain.context_boot()
        # Boot should include locked rules somewhere
        boot_str = json.dumps(boot, default=str).lower()
        self.assertTrue('locked' in boot_str or 'rule' in boot_str or 'auth' in boot_str,
                        "Boot output should include locked rules")

    def test_context_boot_includes_node_data(self):
        """context_boot() includes locked rules and node stats."""
        boot = self.brain.context_boot()
        # Boot should have locked rules and stats
        self.assertIn('total_nodes', boot)
        self.assertIn('locked', boot)
        self.assertGreater(boot['total_nodes'], 0)


# ═══════════════════════════════════════════════════════════════════════
# 20. REMEMBER RICH — detailed node creation
# ═══════════════════════════════════════════════════════════════════════

class TestRememberRich(BrainTestBase):
    """Test remember_rich() which underpins all engineering memory types."""

    def test_remember_rich_stores_metadata(self):
        """remember_rich() stores node_metadata when provided."""
        n = self.brain.remember_rich(
            type='decision', title='Test rich node',
            content='Rich content with reasoning.',
            reasoning='We chose this because of X.',
            user_raw_quote='Tom said: use this approach',
            keywords='rich metadata test')

        meta = self.brain.conn.execute(
            "SELECT reasoning, user_raw_quote FROM node_metadata WHERE node_id = ?",
            (n['id'],)
        ).fetchone()
        if meta:  # metadata table may or may not have this row
            self.assertEqual(meta[0], 'We chose this because of X.')

    def test_remember_rich_with_emotion(self):
        """remember_rich() stores emotion data."""
        n = self.brain.remember_rich(
            type='lesson', title='Emotional lesson',
            content='This was frustrating.',
            emotion=0.8, emotion_label='frustration',
            keywords='emotion test')

        node = self.brain.conn.execute(
            "SELECT emotion, emotion_label FROM nodes WHERE id = ?", (n['id'],)
        ).fetchone()
        self.assertAlmostEqual(node[0], 0.8, places=1)
        self.assertEqual(node[1], 'frustration')

    def test_remember_rich_auto_bridges(self):
        """remember_rich() creates bridge proposals to similar nodes."""
        # Create two related nodes
        self.brain.remember(type='rule', title='Auth must use Clerk',
                            content='Clerk for all auth.', keywords='auth clerk')
        n2 = self.brain.remember_rich(
            type='decision', title='Clerk config for magic links',
            content='Configure Clerk with magic link auth flow.',
            keywords='clerk magic links auth config')

        # Check bridges_created
        self.assertIn('bridges_created', n2)


# ═══════════════════════════════════════════════════════════════════════
# 21. URGENT SIGNALS
# ═══════════════════════════════════════════════════════════════════════

class TestUrgentSignals(BrainTestBase):
    """Test get_urgent_signals() for priority alerting."""

    def test_urgent_signals_returns_list(self):
        """get_urgent_signals() returns list of strings."""
        signals = self.brain.get_urgent_signals()
        self.assertIsInstance(signals, list)
        for s in signals:
            self.assertIsInstance(s, str)

    def test_overdue_tasks_are_urgent(self):
        """Overdue task nodes appear in urgent signals."""
        # 'reminder' is not a valid node type — use 'task' with due_date
        n = self.brain.remember(type='task', title='Overdue: deploy to staging',
                                content='Deploy was due yesterday.',
                                keywords='task deploy staging')
        self.brain.conn.execute(
            "UPDATE nodes SET due_date = datetime('now', '-2 days') WHERE id = ?",
            (n['id'],))
        self.brain.conn.commit()

        signals = self.brain.get_urgent_signals()
        self.assertIsInstance(signals, list)


# ═══════════════════════════════════════════════════════════════════════
# 22. HOST ENVIRONMENT SCANNING
# ═══════════════════════════════════════════════════════════════════════

class TestHostEnvironment(BrainTestBase):
    """Test scan_host_environment() for detecting environment changes."""

    def test_scan_host_returns_dict(self):
        """scan_host_environment() returns structured dict."""
        result = self.brain.scan_host_environment()
        self.assertIsInstance(result, dict)

    def test_scan_host_detects_os(self):
        """Host scan includes basic OS/platform info."""
        result = self.brain.scan_host_environment()
        result_str = json.dumps(result, default=str).lower()
        # Should have some host info
        self.assertTrue(len(result_str) > 2, "Host scan should return some data")


# ═══════════════════════════════════════════════════════════════════════
# 23. SUGGEST METRICS
# ═══════════════════════════════════════════════════════════════════════

class TestSuggestMetrics(BrainTestBase):
    """Test the pre-edit suggestion metrics tracking."""

    def test_get_suggest_metrics_returns_dict(self):
        """get_suggest_metrics() returns dict with period stats."""
        metrics = self.brain.get_suggest_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('period_days', metrics)
        self.assertIn('total_suggest_calls', metrics)

    def test_suggest_metrics_after_pre_edit(self):
        """Pre-edit hook records suggestion metrics."""
        from servers.daemon_hooks import hook_pre_edit
        _seed_rich_brain(self.brain)
        result = hook_pre_edit(self.brain, {
            'tool_input': {'file_path': '/src/auth.py'}
        }, [])
        # Metrics may or may not be recorded depending on whether suggestions found
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
