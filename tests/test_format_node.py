"""Tests for contract.format_node() — the standard node renderer for LLM consumers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.contract import format_node, NODE_FORMAT_DEFAULTS


class TestFormatNode(BrainTestBase):
    needs_embedder = False

    def _make_node(self, **kwargs):
        """Create a node and return its full ID."""
        defaults = dict(type='rule', title='Test node', content='Test content')
        defaults.update(kwargs)
        result = self.brain.remember(**defaults)
        return result['id']

    def _add_edge(self, source_id, target_id, relation='related_to',
                  weight=0.8, description=''):
        """Insert an edge directly (bypasses Hebbian logic)."""
        self.brain.conn.execute(
            "INSERT OR REPLACE INTO edges "
            "(source_id, target_id, weight, relation, edge_type, description, "
            "co_access_count, stability, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, 1.0, datetime('now'))",
            (source_id, target_id, weight, relation, relation, description))
        self.brain.conn.commit()

    # ── Basic rendering ──

    def test_full_node(self):
        """Full node with core fields renders header + content + keywords."""
        nid = self._make_node(
            type='decision', title='Use Postgres', content='We chose Postgres for reliability.',
            keywords='db postgres sql', confidence=0.9, locked=True)
        out = format_node(nid, self.brain.conn)
        self.assertIsNotNone(out)
        self.assertIn('[decision]', out)
        self.assertIn('"Use Postgres"', out)
        self.assertIn('conf:0.9', out)
        self.assertIn('locked', out)
        self.assertIn('We chose Postgres', out)
        self.assertIn('Keywords: db postgres sql', out)

    def test_nonexistent_node_returns_none(self):
        """format_node returns None for an ID that doesn't exist."""
        out = format_node('nonexistent-id-12345678', self.brain.conn)
        self.assertIsNone(out)

    # ── Header format ──

    def test_header_includes_id_prefix(self):
        """Header shows first 8 chars of node ID."""
        nid = self._make_node(type='lesson', title='Header test')
        out = format_node(nid, self.brain.conn)
        self.assertIn('id:%s' % nid[:8], out)

    def test_header_unlocked_no_locked_flag(self):
        """Unlocked nodes don't show ', locked' in header."""
        nid = self._make_node(locked=False)
        out = format_node(nid, self.brain.conn)
        self.assertNotIn('locked', out)

    def test_header_encoding_source(self):
        """encoding_source appears in header when set."""
        nid = self._make_node(encoding_source='encoder:sonnet')
        out = format_node(nid, self.brain.conn)
        self.assertIn('src:encoder:sonnet', out)

    # ── Content truncation ──

    def test_content_limit_truncates(self):
        """content_limit config truncates long content."""
        long_content = 'A' * 500
        nid = self._make_node(content=long_content)
        out = format_node(nid, self.brain.conn, config={'content_limit': 50})
        # Content line should have exactly 50 A's, not 500
        self.assertIn('A' * 50, out)
        self.assertNotIn('A' * 51, out)

    def test_no_content_limit_shows_full(self):
        """Default (content_limit=None) shows full content."""
        long_content = 'B' * 500
        nid = self._make_node(content=long_content)
        out = format_node(nid, self.brain.conn)
        self.assertIn('B' * 500, out)

    # ── Metadata ──

    def test_metadata_situation(self):
        """Situation text from node_embeddings is rendered."""
        nid = self._make_node()
        # Insert situation directly — embedder is off in tests, so we need
        # a dummy embedding blob to satisfy the NOT NULL constraint
        self.brain.conn.execute(
            "INSERT OR REPLACE INTO node_embeddings "
            "(node_id, embedding, situation_text, model, created_at) "
            "VALUES (?, X'00', ?, 'test', datetime('now'))",
            (nid, 'When choosing a database for OLTP workloads'))
        self.brain.conn.commit()
        out = format_node(nid, self.brain.conn)
        self.assertIn('Situation: When choosing a database', out)

    def test_metadata_reasoning(self):
        """Reasoning from metadata_kv is rendered."""
        nid = self._make_node(reasoning='Postgres has better JSON support than MySQL')
        out = format_node(nid, self.brain.conn)
        self.assertIn('Reasoning:', out)
        self.assertIn('Postgres has better JSON support', out)

    def test_metadata_correction_of(self):
        """correction_of links the node to the node it supersedes via a correction warning."""
        original_id = self._make_node(title='Use MySQL')
        correction_id = self._make_node(
            title='Use Postgres instead', correction_of=original_id)
        out = format_node(correction_id, self.brain.conn)
        # render_rich_node shows "⚠ Updated by:" — the correction_of edge is stored as
        # corrected_by (new→old), so correction_enrich sees direction="corrected_by"
        self.assertIn('⚠ Updated by:', out)
        self.assertIn('Use MySQL', out)

    # ── Edges ──

    def test_edges_shown(self):
        """Edges appear with target title and relation."""
        nid = self._make_node(title='Source node')
        target_id = self._make_node(type='mechanism', title='Target node')
        self._add_edge(nid, target_id, relation='depends_on', weight=0.9,
                       description='runtime dependency')
        out = format_node(nid, self.brain.conn)
        self.assertIn('Edges:', out)
        self.assertIn('"Target node"', out)
        self.assertIn('depends_on', out)
        self.assertIn('runtime dependency', out)

    def test_edge_limit_config(self):
        """edge_limit config caps the number of edges shown."""
        nid = self._make_node(title='Hub node')
        for i in range(6):
            tid = self._make_node(title='Spoke %d' % i)
            self._add_edge(nid, tid, weight=0.5 + i * 0.01)
        # Default limit is 5 — should show 5 of 6
        out = format_node(nid, self.brain.conn)
        arrow_count = out.count('    \u2192 ')
        self.assertEqual(arrow_count, 5)
        # With limit 2
        out2 = format_node(nid, self.brain.conn, config={'edge_limit': 2})
        self.assertEqual(out2.count('    \u2192 '), 2)

    def test_edge_filter_excludes_co_accessed(self):
        """Default edge_filter excludes co_accessed and emergent_bridge."""
        nid = self._make_node(title='Main')
        t1 = self._make_node(title='Related')
        t2 = self._make_node(title='Co-accessed')
        self._add_edge(nid, t1, relation='related_to')
        self._add_edge(nid, t2, relation='co_accessed')
        out = format_node(nid, self.brain.conn)
        self.assertIn('"Related"', out)
        self.assertNotIn('"Co-accessed"', out)


if __name__ == '__main__':
    unittest.main()
