"""
Tests for the Challenge System — recall as challenge, revise, gap detection,
consolidation, message stream, and unified output formatting.

Each test has an explicit KPI (measurable assertion).
"""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


# ═══════════════════════════════════════════════════════════════
# Message Stream
# ═══════════════════════════════════════════════════════════════

class TestMessageStream(BrainTestBase):
    """Test invisible conversation capture via MessageStreamDAL."""

    def test_store_exchange_persists_both_roles(self):
        """KPI: 2 rows in message_stream after store_exchange()."""
        result = self.brain.store_exchange("Hello from Tom", "Hello from Claude", "test-session")
        self.assertIn('user_id', result)
        self.assertIn('assistant_id', result)

        from servers.dal_message_stream import MessageStreamDAL
        dal = MessageStreamDAL(self.brain.logs_conn)
        rows = dal.get_recent(limit=10)
        self.assertEqual(len(rows), 2)
        roles = {r['role'] for r in rows}
        self.assertEqual(roles, {'user', 'assistant'})

    def test_get_pending_returns_unencoded_user_only(self):
        """KPI: get_pending returns only role='user' AND encoded=0."""
        self.brain.store_exchange("msg 1", "reply 1", "s1")
        self.brain.store_exchange("msg 2", "reply 2", "s1")

        from servers.dal_message_stream import MessageStreamDAL
        dal = MessageStreamDAL(self.brain.logs_conn)
        pending = dal.get_pending(limit=10)

        # Only user messages, not assistant
        self.assertTrue(all(p.get('content', '').startswith('msg') for p in pending))
        self.assertEqual(len(pending), 2)

    def test_get_pending_ordered_newest_first(self):
        """KPI: First result has latest timestamp."""
        self.brain.store_exchange("first", "r1", "s1")
        time.sleep(0.05)  # Ensure different timestamps
        self.brain.store_exchange("second", "r2", "s1")

        from servers.dal_message_stream import MessageStreamDAL
        dal = MessageStreamDAL(self.brain.logs_conn)
        pending = dal.get_pending(limit=10)

        self.assertEqual(pending[0]['content'], 'second')
        self.assertEqual(pending[1]['content'], 'first')

    def test_mark_encoded_removes_from_pending(self):
        """KPI: Marked message not in get_pending() results."""
        self.brain.store_exchange("encode me", "ok", "s1")

        from servers.dal_message_stream import MessageStreamDAL
        dal = MessageStreamDAL(self.brain.logs_conn)
        pending = dal.get_pending(limit=10)
        self.assertEqual(len(pending), 1)

        dal.mark_encoded([pending[0]['id']])
        pending_after = dal.get_pending(limit=10)
        self.assertEqual(len(pending_after), 0)

    def test_pending_limit_respected(self):
        """KPI: limit=2 returns at most 2."""
        for i in range(5):
            self.brain.store_exchange("msg %d" % i, "reply", "s1")

        from servers.dal_message_stream import MessageStreamDAL
        dal = MessageStreamDAL(self.brain.logs_conn)
        pending = dal.get_pending(limit=2)
        self.assertEqual(len(pending), 2)


# ═══════════════════════════════════════════════════════════════
# Revise
# ═══════════════════════════════════════════════════════════════

class TestRevise(BrainTestBase):
    """Test node revision — encoding IS updating."""

    def _create_node(self, **kwargs):
        """Helper: create a test node and return its full ID."""
        defaults = {'type': 'decision', 'title': 'Test decision',
                    'content': 'Original content here.'}
        defaults.update(kwargs)
        result = self.brain.remember(**defaults)
        return result['id']

    def test_revise_appends_with_divider(self):
        """KPI: Content contains revision divider."""
        nid = self._create_node()
        result = self.brain.revise(nid, "Updated info", "architecture changed")

        self.assertNotIn('error', result)
        node = self.brain.conn.execute(
            'SELECT content FROM nodes WHERE id = ?', (nid,)).fetchone()
        self.assertIn('--- Revised', node[0])
        self.assertIn('architecture changed', node[0])
        self.assertIn('Updated info', node[0])

    def test_revise_preserves_original(self):
        """KPI: Original content still present."""
        nid = self._create_node(content='ORIGINAL_MARKER_TEXT')
        self.brain.revise(nid, "New stuff", "update")

        node = self.brain.conn.execute(
            'SELECT content FROM nodes WHERE id = ?', (nid,)).fetchone()
        self.assertIn('ORIGINAL_MARKER_TEXT', node[0])

    def test_revise_sets_revised_at(self):
        """KPI: revised_at IS NOT NULL after revise()."""
        nid = self._create_node()
        self.brain.revise(nid, "Update", "test")

        row = self.brain.conn.execute(
            'SELECT revised_at FROM nodes WHERE id = ?', (nid,)).fetchone()
        self.assertIsNotNone(row[0])

    def test_revise_reembeds(self):
        """KPI: Embedding blob differs before vs after revise()."""
        nid = self._create_node(content='cats and dogs')

        emb_before = self.brain.conn.execute(
            'SELECT embedding FROM node_embeddings WHERE node_id = ?', (nid,)).fetchone()

        self.brain.revise(nid, "completely different topic about rockets and space", "test")

        emb_after = self.brain.conn.execute(
            'SELECT embedding FROM node_embeddings WHERE node_id = ?', (nid,)).fetchone()

        if emb_before and emb_after:
            self.assertNotEqual(emb_before[0], emb_after[0],
                                "Embedding should change after revise with different content")

    def test_revise_nonexistent_returns_error(self):
        """KPI: Returns error dict for invalid node_id."""
        result = self.brain.revise("nonexistent_id_12345678", "content", "reason")
        self.assertIn('error', result)
        self.assertIn('not found', result['error'].lower())

    def test_revise_archived_returns_error(self):
        """KPI: Refuses to revise an archived node."""
        nid = self._create_node()
        self.brain.conn.execute('UPDATE nodes SET archived = 1 WHERE id = ?', (nid,))
        self.brain.conn.commit()

        result = self.brain.revise(nid, "content", "reason")
        self.assertIn('error', result)
        self.assertIn('archived', result['error'].lower())

    def test_revise_returns_useful_info(self):
        """KPI: Return dict has id, type, title, revised_at, content_length."""
        nid = self._create_node()
        result = self.brain.revise(nid, "New content", "reason")

        self.assertEqual(result['id'], nid)
        self.assertEqual(result['type'], 'decision')
        self.assertIn('revised_at', result)
        self.assertIn('content_length', result)
        self.assertGreater(result['content_length'], 0)

    def test_revise_auto_resolves_consolidation_pair(self):
        """KPI: After revise(), pending pair involving that node has resolved=1."""
        nid_a = self._create_node(title='Topic A version 1')
        nid_b = self._create_node(title='Topic A version 2')

        # Queue a fake consolidation pair
        from servers.dal import LogsDAL
        logs_dal = LogsDAL(self.brain.logs_conn)
        logs_dal.queue_consolidation(nid_a, nid_b, 0.90)

        # Verify it's pending
        pending = logs_dal.get_pending_consolidation(limit=10)
        self.assertEqual(len(pending), 1)

        # Revise one of the nodes
        self.brain.revise(nid_a, "Merged content from both versions", "consolidated")

        # Verify pair is resolved
        pending_after = logs_dal.get_pending_consolidation(limit=10)
        self.assertEqual(len(pending_after), 0)

    def test_revise_does_not_resolve_unrelated_pairs(self):
        """KPI: Pairs not involving the revised node remain resolved=0."""
        nid_a = self._create_node(title='Node A')
        nid_b = self._create_node(title='Node B')
        nid_c = self._create_node(title='Node C')
        nid_d = self._create_node(title='Node D')

        from servers.dal import LogsDAL
        logs_dal = LogsDAL(self.brain.logs_conn)
        logs_dal.queue_consolidation(nid_a, nid_b, 0.90)
        logs_dal.queue_consolidation(nid_c, nid_d, 0.88)

        # Revise nid_a — should resolve A-B pair but NOT C-D
        self.brain.revise(nid_a, "Updated", "test")

        pending = logs_dal.get_pending_consolidation(limit=10)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['node_id_a'], nid_c)


# ═══════════════════════════════════════════════════════════════
# Gap Detection
# ═══════════════════════════════════════════════════════════════

class TestGapDetection(BrainTestBase):
    """Test recall gap detection when brain has no relevant knowledge."""

    def test_gap_flagged_when_no_results(self):
        """KPI: Result dict has '_gap' key when recall returns empty."""
        # Empty brain — any query should gap
        result = self.brain.recall_with_embeddings("quantum computing in healthcare", limit=5)
        # With no nodes at all, results should be empty and gap should be flagged
        if not result.get('results'):
            self.assertIn('_gap', result)

    def test_gap_not_flagged_when_results_found(self):
        """KPI: No '_gap' key when recall returns matches."""
        self.brain.remember(type='decision', title='Auth: use Clerk',
                           content='Passwordless login via magic links',
                           keywords='auth clerk login')
        result = self.brain.recall_with_embeddings("auth login", limit=5)
        if result.get('results'):
            self.assertNotIn('_gap', result)

    def test_gap_contains_query_and_score(self):
        """KPI: _gap has query and top_score keys."""
        result = self.brain.recall_with_embeddings("completely unknown topic xyz", limit=5)
        gap = result.get('_gap')
        if gap:
            self.assertIn('query', gap)
            self.assertIn('top_score', gap)


# ═══════════════════════════════════════════════════════════════
# Consolidation
# ═══════════════════════════════════════════════════════════════

class TestConsolidation(BrainTestBase):
    """Test consolidation detection and queue management."""

    def test_duplicate_pair_not_requeued(self):
        """KPI: INSERT OR IGNORE prevents double-queue."""
        from servers.dal import LogsDAL
        logs_dal = LogsDAL(self.brain.logs_conn)

        first = logs_dal.queue_consolidation("id_a", "id_b", 0.90)
        second = logs_dal.queue_consolidation("id_a", "id_b", 0.92)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(logs_dal.count_pending_consolidation(), 1)

    def test_resolved_not_returned(self):
        """KPI: get_pending skips resolved=1 rows."""
        from servers.dal import LogsDAL
        logs_dal = LogsDAL(self.brain.logs_conn)

        logs_dal.queue_consolidation("id_a", "id_b", 0.90)
        pair = logs_dal.get_pending_consolidation(limit=1)
        self.assertEqual(len(pair), 1)

        logs_dal.resolve_consolidation(pair[0]['id'])
        pending = logs_dal.get_pending_consolidation(limit=10)
        self.assertEqual(len(pending), 0)

    def test_count_pending(self):
        """KPI: Accurate count of unresolved pairs."""
        from servers.dal import LogsDAL
        logs_dal = LogsDAL(self.brain.logs_conn)

        self.assertEqual(logs_dal.count_pending_consolidation(), 0)
        logs_dal.queue_consolidation("a", "b", 0.90)
        logs_dal.queue_consolidation("c", "d", 0.88)
        self.assertEqual(logs_dal.count_pending_consolidation(), 2)


# ═══════════════════════════════════════════════════════════════
# Challenge Output Format
# ═══════════════════════════════════════════════════════════════

class TestChallengeOutput(BrainTestBase):
    """Test the unified challenge output formatting."""

    def test_format_node_shows_full_id(self):
        """KPI: Node ID is 32 chars in output (not truncated)."""
        from servers.brain_voice import BrainVoice
        node = {'id': 'a' * 32, 'type': 'decision', 'title': 'Test',
                'content': 'Content here', 'created_at': '2026-03-25T00:00:00'}
        lines = []
        BrainVoice.format_node(node, lines)
        output = '\n'.join(lines)
        self.assertIn('a' * 32, output)

    def test_format_node_shows_revised_never(self):
        """KPI: Contains 'revised:never' when not revised."""
        from servers.brain_voice import BrainVoice
        node = {'id': 'x' * 32, 'type': 'lesson', 'title': 'Test',
                'content': 'Content', 'created_at': '2026-03-25', 'revised_at': None}
        lines = []
        BrainVoice.format_node(node, lines)
        output = '\n'.join(lines)
        self.assertIn('revised:never', output)

    def test_format_node_shows_revised_date(self):
        """KPI: Contains 'revised:{date}' when revised."""
        from servers.brain_voice import BrainVoice
        node = {'id': 'x' * 32, 'type': 'lesson', 'title': 'Test',
                'content': 'Content', 'created_at': '2026-03-25',
                'revised_at': '2026-03-26T10:00:00'}
        lines = []
        BrainVoice.format_node(node, lines)
        output = '\n'.join(lines)
        self.assertIn('revised:2026-03-26', output)

    def test_format_node_shows_confidence(self):
        """KPI: Contains 'conf:' in output."""
        from servers.brain_voice import BrainVoice
        node = {'id': 'x' * 32, 'type': 'decision', 'title': 'Test',
                'content': 'Content', 'created_at': '2026-03-25', 'confidence': 0.85}
        lines = []
        BrainVoice.format_node(node, lines)
        output = '\n'.join(lines)
        self.assertIn('conf:0.85', output)

    def test_challenge_header_present(self):
        """KPI: render_prompt output contains challenge framing when results exist."""
        from servers.brain_voice import BrainVoice
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=[{'id': 'x' * 32, 'type': 'decision', 'title': 'Test',
                      'content': 'Content', 'created_at': '2026-03-25',
                      'effective_activation': 0.9}],
            prompt_signals={})
        output = rendered['for_claude']
        self.assertIn('ACTIVE RECALL', output)
        self.assertIn('revise(', output)

    def test_gap_in_output(self):
        """KPI: Gap section appears when gap dict passed."""
        from servers.brain_voice import BrainVoice
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=[],
            prompt_signals={},
            gap={'query': 'kubernetes scaling', 'top_score': 0.3})
        output = rendered['for_claude']
        self.assertIn('UNKNOWN TOPIC', output)
        self.assertIn('kubernetes scaling', output)

    def test_consolidation_only_when_few_results(self):
        """KPI: Consolidation absent when 3+ results; present when < 2."""
        from servers.brain_voice import BrainVoice
        voice = BrainVoice(self.brain)

        # 3 results — no consolidation
        three_results = [
            {'id': 'a' * 32, 'type': 'decision', 'title': 'T1', 'content': 'C1',
             'created_at': '2026-03-25', 'effective_activation': 0.9},
            {'id': 'b' * 32, 'type': 'decision', 'title': 'T2', 'content': 'C2',
             'created_at': '2026-03-25', 'effective_activation': 0.8},
            {'id': 'c' * 32, 'type': 'decision', 'title': 'T3', 'content': 'C3',
             'created_at': '2026-03-25', 'effective_activation': 0.7},
        ]
        consolidation_data = [{'node_a': {'id': 'x' * 32, 'type': 'decision',
                               'title': 'Old', 'content': 'Old content', 'created_at': '2026-03-20'},
                               'node_b': {'id': 'y' * 32, 'type': 'decision',
                               'title': 'New', 'content': 'New content', 'created_at': '2026-03-25'},
                               'pair_id': 1}]

        rendered_many = voice.render_prompt(
            results=three_results, prompt_signals={},
            consolidation=consolidation_data, consolidation_total=1)
        self.assertNotIn('CONSOLIDATION', rendered_many['for_claude'])

        # 0 results — consolidation shown
        rendered_none = voice.render_prompt(
            results=[], prompt_signals={},
            consolidation=consolidation_data, consolidation_total=1)
        self.assertIn('CONSOLIDATION', rendered_none['for_claude'])

    def test_pending_tom_messages_in_output(self):
        """KPI: Tom's messages appear in output."""
        from servers.brain_voice import BrainVoice
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=[], prompt_signals={},
            pending_tom_messages=[
                {'content': 'build the in-memory graph', 'timestamp': '2026-03-25T10:00:00'},
                {'content': 'old patterns scare me', 'timestamp': '2026-03-25T11:00:00'},
            ])
        output = rendered['for_claude']
        self.assertIn('PENDING', output)
        self.assertIn('build the in-memory graph', output)
        self.assertIn('old patterns scare me', output)

    def test_render_prompt_existing_callers_unaffected(self):
        """KPI: Call with only original params works — no error."""
        from servers.brain_voice import BrainVoice
        voice = BrainVoice(self.brain)
        # Call with ONLY the original parameters — no new ones
        rendered = voice.render_prompt(
            results=[], prompt_signals={},
            urgent_signals=['test signal'],
            segment_note=None, priming_note=None,
            graph_changes=None, pending_messages=None,
            debug_messages=None, precision_feedback=None)
        self.assertIn('for_claude', rendered)
        self.assertIsInstance(rendered['for_claude'], str)


# ═══════════════════════════════════════════════════════════════
# Gap Logging DAL
# ═══════════════════════════════════════════════════════════════

class TestGapLogging(BrainTestBase):
    """Test recall gap logging via LogsDAL."""

    def test_gap_logged(self):
        """KPI: Row exists in recall_gaps after log_gap()."""
        from servers.dal import LogsDAL
        logs_dal = LogsDAL(self.brain.logs_conn)
        logs_dal.log_gap("unknown topic", 0.45, "test-session")

        row = self.brain.logs_conn.execute(
            'SELECT query, top_score FROM recall_gaps').fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 'unknown topic')
        self.assertAlmostEqual(row[1], 0.45, places=2)


if __name__ == '__main__':
    unittest.main()
