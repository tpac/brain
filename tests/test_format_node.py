"""Tests for render_rich_node() — the standard node renderer for LLM consumers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.contract import render_rich_node, NODE_FORMAT_DEFAULTS


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
        """Insert an edge using the new multi-relation model."""
        from servers.dal_graph import GraphDAL
        dal = GraphDAL(self.brain.conn)
        dal.add_relation(source_id, target_id, relation, description, weight)

    def _render(self, node_id, config=None):
        """Fetch node + render: the two-step pattern replacing format_node()."""
        node = self.brain.get_node(node_id)
        if not node:
            return None
        return render_rich_node(node, config)

    # ── Basic rendering ──

    def test_full_node(self):
        """Full node with core fields renders header + content + keywords."""
        nid = self._make_node(
            type='decision', title='Use Postgres', content='We chose Postgres for reliability.',
            keywords='db postgres sql', confidence=0.9, locked=True)
        out = self._render(nid)
        self.assertIsNotNone(out)
        self.assertIn('[decision]', out)
        self.assertIn('"Use Postgres"', out)
        # Confidence display rooted out 2026-05-31: show_confidence defaults
        # off (the field is dormant — read by no ranking path). Default render
        # must NOT show it. Same strictness, opposite contract.
        self.assertNotIn('conf:', out)
        self.assertIn('locked', out)
        self.assertIn('We chose Postgres', out)
        # Keywords column dropped in schema v28; render block removed.
        # Asserting absence (same strictness, opposite contract).
        self.assertNotIn('Keywords:', out)

    def test_confidence_renders_when_enabled(self):
        """Confidence is hidden by default but still renders when a caller
        explicitly opts in via show_confidence=True (the dormant field's
        opt-in path stays covered after the 2026-05-31 default flip)."""
        nid = self._make_node(
            type='decision', title='Use Postgres', content='We chose Postgres.',
            confidence=0.9)
        node = self.brain.get_node(nid)
        out = render_rich_node(node, {'show_confidence': True})
        self.assertIn('conf:0.9', out)

    def test_nonexistent_node_returns_none(self):
        """Render returns None for an ID that doesn't exist."""
        out = self._render('nonexistent-id-12345678')
        self.assertIsNone(out)

    # ── Header format ──

    def test_header_includes_id_prefix(self):
        """Header shows first 8 chars of node ID."""
        nid = self._make_node(type='lesson', title='Header test')
        out = self._render(nid)
        self.assertIn('id:%s' % nid[:8], out)

    def test_header_unlocked_no_locked_flag(self):
        """Unlocked nodes don't show ', locked' in header."""
        nid = self._make_node(locked=False)
        out = self._render(nid)
        self.assertNotIn('locked', out)

    def test_header_encoding_source(self):
        """encoding_source appears in header when set."""
        nid = self._make_node(encoding_source='encoder:sonnet')
        out = self._render(nid)
        self.assertIn('src:encoder:sonnet', out)

    # ── Content truncation ──

    def test_content_limit_truncates(self):
        """content_limit config truncates long content."""
        long_content = 'A' * 500
        nid = self._make_node(content=long_content)
        out = self._render(nid, config={'content_limit': 50})
        # _truncate caps at <= limit chars INCLUDING the ellipsis
        # (s[:limit-1] + '…'), so a 500-char body renders as 49 A's + '…'.
        self.assertIn('A' * 49 + '…', out)
        self.assertNotIn('A' * 50, out)

    def test_no_content_limit_shows_full(self):
        """Default (content_limit=None) shows full content."""
        long_content = 'B' * 500
        nid = self._make_node(content=long_content)
        out = self._render(nid)
        self.assertIn('B' * 500, out)

    # ── Metadata ──

    def test_metadata_situation(self):
        """Situation from node_metadata_kv (canonical store) is rendered.

        v24+: situation lives in node_metadata_kv, not the removed
        node_embeddings table. Stronger than the old test — verifies
        both the canonical storage location AND the rendered output.
        """
        sit = 'When choosing a database for OLTP workloads'
        nid = self._make_node(situation=sit)
        # Verify storage location — situation must be in kv, not elsewhere
        kv_row = self.brain.conn.execute(
            "SELECT value FROM node_metadata_kv WHERE node_id=? AND key='situation'",
            (nid,)).fetchone()
        self.assertIsNotNone(kv_row, 'situation should be stored in node_metadata_kv')
        self.assertEqual(kv_row[0], sit, 'kv value must match the written situation exactly')
        # Verify render
        out = self._render(nid)
        self.assertIn('Situation: When choosing a database', out)
        self.assertIn(sit, out)  # full exact-match, stronger than substring

    def test_metadata_reasoning(self):
        """Reasoning from metadata_kv is rendered."""
        nid = self._make_node(reasoning='Postgres has better JSON support than MySQL')
        out = self._render(nid)
        self.assertIn('Reasoning:', out)
        self.assertIn('Postgres has better JSON support', out)

    def test_correction_edge_annotations(self):
        """A `corrects` edge surfaces correction context on both endpoints.

        correction_enrich walks correction_improvement-aspect edges
        (corrects, supersedes, reframes, ...). The renderer annotates
        the corrector's view with 'Corrects:' and the corrected node's
        view with 'Updated by:'.
        """
        original_id = self._make_node(title='Use MySQL')
        correction_id = self._make_node(title='Use Postgres instead')
        # Edge: correction_id corrects original_id
        self.brain.connect_typed(
            source_id=correction_id, target_id=original_id,
            relation='corrects', weight=0.5,
            description='Postgres reliability beats MySQL for this workload',
            encoding_source='test:correction_edge_annotations')

        # Corrector's view: 'Corrects:' the original
        out_corrector = self._render(correction_id)
        self.assertIn('Corrects:', out_corrector)
        self.assertIn('Use MySQL', out_corrector)
        # Corrected node's view: 'Updated by:' the correction
        out_original = self._render(original_id)
        self.assertIn('Updated by:', out_original)
        self.assertIn('Use Postgres', out_original)

    # ── Edges ──

    def test_edges_shown(self):
        """Edges appear with target title and relation."""
        nid = self._make_node(title='Source node')
        target_id = self._make_node(type='mechanism', title='Target node')
        self._add_edge(nid, target_id, relation='depends_on', weight=0.9,
                       description='runtime dependency')
        out = self._render(nid)
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
        # Edge lines rendered as '    [type id:XX date] ...' (4-space indent + bracket)
        out = self._render(nid)
        edge_lines = [l for l in out.split('\n') if l.startswith('    [')]
        self.assertEqual(len(edge_lines), 5)
        # With limit 2
        out2 = self._render(nid, config={'edge_limit': 2})
        edge_lines2 = [l for l in out2.split('\n') if l.startswith('    [')]
        self.assertEqual(len(edge_lines2), 2)

    def test_differential_project_mark_on_mismatch(self):
        """cfg['scope']: foreign project renders the ⚠ mark, the generic
        'Project:' KV line is suppressed."""
        nid = self._make_node(title='Foreign fact', project='exco')
        out = self._render(nid, {'scope': {'project': 'brain'}})
        self.assertIn('⚠ From another project: exco', out)
        self.assertNotIn('Project: exco', out)

    def test_differential_project_silent_on_match(self):
        """Same-project node renders NO project line at all in differential
        mode — a same-project line on a one-project corpus is noise."""
        nid = self._make_node(title='Home fact', project='brain')
        out = self._render(nid, {'scope': {'project': 'brain'}})
        self.assertNotIn('From another project', out)
        self.assertNotIn('Project:', out)

    def test_differential_project_neutral_on_unscoped_node(self):
        """A node with no project provenance is never marked foreign —
        unknown is neutral, matching the scope lane semantics."""
        nid = self._make_node(title='Unscoped fact')
        out = self._render(nid, {'scope': {'project': 'brain'}})
        self.assertNotIn('From another project', out)

    def test_differential_counterpart_mark_on_mismatch(self):
        """The counterpart dimension marks through the SAME central
        scope_marks path — adding a dimension re-threads nothing."""
        nid = self._make_node(title='Other-speaker fact', counterpart='Dana')
        out = self._render(nid, {'scope': {'project': 'brain',
                                           'counterpart': 'Tom'}})
        self.assertIn('⚠ Learned with another counterpart: Dana', out)
        self.assertNotIn('Counterpart: Dana', out)

    def test_differential_counterpart_silent_on_match(self):
        nid = self._make_node(title='Same-speaker fact', counterpart='Tom')
        out = self._render(nid, {'scope': {'counterpart': 'Tom'}})
        self.assertNotIn('another counterpart', out)
        self.assertNotIn('Counterpart:', out)

    def test_legacy_render_keeps_generic_project_line(self):
        """Callers that don't declare a scope keep the pre-existing generic
        KV render — no information loss for unwired consumers."""
        nid = self._make_node(title='Legacy view', project='exco',
                              counterpart='Tom')
        out = self._render(nid)
        self.assertIn('Project: exco', out)
        self.assertNotIn('From another project', out)
        # counterpart is differential-ONLY: its value is the install default
        # (identical on every node), so the generic KV line is pure noise
        # and stays suppressed even for undeclared callers.
        self.assertNotIn('Counterpart:', out)

if __name__ == '__main__':
    unittest.main()
