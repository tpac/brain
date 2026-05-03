"""Decode Pipeline Transition Tests — verify wiring between stages.

Each test checks that the output of one pipeline stage has the shape
the next stage expects. These catch silent breakage when a format
changes in one place but the consumer doesn't update.

Stages tested:
  remember → recall (embedding roundtrip)
  recall → build_judge_prompt (candidate fields)
  judge output → format_judge_output (voice surface)
  correction_enrich (correction chain lookup)
  judge-selected IDs → _hebbian_strengthen (co_accessed edges)
  recall → build_judge_prompt end-to-end (candidate format compatibility)
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestDecodeTransitions(BrainTestBase):
    """Tests for wiring between decode pipeline stages."""

    needs_embedder = True

    def test_remember_recall_embedding_roundtrip(self):
        """remember → recall: stored node must be retrievable by semantic query."""
        node = self.brain.remember(
            type='decision',
            title='Use PostgreSQL for the analytics database',
            content='Chose PostgreSQL over MongoDB for analytics because we need '
                    'complex joins and ACID transactions for financial data.',
            keywords='postgres analytics database',
        )
        node_id = node['id']

        result = self.brain.recall(query='which database for analytics', limit=10)
        results = result.get('results', [])
        found_ids = [r['id'] for r in results]

        self.assertIn(node_id, found_ids,
                      'Stored node not found in recall results — embedding storage or '
                      'retrieval is broken')

    def test_recall_result_shape_for_surface(self):
        """recall → surface: recall results must have all fields build_surface_prompt expects.

        build_surface_prompt calls format_candidate_for_surface which reads:
        id, type, title, content, confidence, locked, score (via effective_activation),
        keywords. If recall drops any of these, the surface gets malformed candidates.
        """
        self.brain.remember(
            type='rule',
            title='Always validate input before processing',
            content='Input validation prevents injection attacks and data corruption.',
            keywords='validation security input',
        )
        result = self.brain.recall(query='input validation', limit=5)
        results = result.get('results', [])
        self.assertTrue(len(results) > 0, 'No recall results — test data not stored')

        # Fields that format_candidate_for_surface reads from each candidate
        required_fields = {'id', 'type', 'title', 'content', 'confidence', 'locked'}
        node = results[0]
        missing = required_fields - set(node.keys())
        self.assertEqual(missing, set(),
                         'Recall result missing fields needed by surface: %s' % missing)

        # score comes from effective_activation — surface reads it as 'score'
        # The hook_recall code in daemon_hooks enriches candidates before passing
        # to build_surface_prompt. But the raw recall result must have the base fields.
        self.assertIn('effective_activation', node,
                      'Missing effective_activation — surface needs this as candidate score')

    def test_surface_output_format_surface_output_compatibility(self):
        """surface JSON → format_surface_output: fake surface response must produce valid output.

        format_surface_output expects selected=[{"id": "...", "why": "..."}] and
        matches them against candidates by id[:8]. If the format diverges,
        Claude gets empty context despite the surface selecting nodes.
        """
        from servers.scales.s1.surface_contract import format_surface_output

        # Create a node so we have a real ID
        node = self.brain.remember(
            type='mechanism',
            title='Rate limiting uses token bucket algorithm',
            content='Each API key gets 100 tokens per minute. Tokens replenish at a '
                    'fixed rate. Burst allowed up to bucket size.',
            keywords='rate limiting api throttle',
        )
        node_id = node['id']
        short_id = node_id[:8]

        # Simulate candidates (what recall returns after enrichment)
        candidates = [{
            'id': node_id,
            'type': 'mechanism',
            'title': 'Rate limiting uses token bucket algorithm',
            'content': 'Each API key gets 100 tokens per minute.',
            'confidence': 0.9,
            'locked': False,
            'score': 0.85,
            'created_at': '2026-04-01T12:00:00Z',
        }]

        # Simulate surface output (what Haiku returns)
        selected = [{'id': short_id, 'why': 'directly answers rate limit question'}]

        output = format_surface_output(selected, candidates)

        self.assertIn('Brain recalled', output,
                      'format_surface_output did not produce header')
        self.assertIn('Rate limiting', output,
                      'format_surface_output did not include node title')
        self.assertIn('token bucket', output,
                      'format_surface_output did not include node content')
        self.assertIn(short_id, output,
                      'format_surface_output did not include node ID')

    def test_surface_output_unmatched_id_skipped(self):
        """surface JSON → format_surface_output: selected ID not in candidates must not crash.

        If the surface hallucinates an ID that doesn't match any candidate,
        format_surface_output should skip it gracefully.
        """
        from servers.scales.s1.surface_contract import format_surface_output

        candidates = [{
            'id': 'real1234-full-id',
            'type': 'rule',
            'title': 'Real node',
            'content': 'Real content',
            'confidence': 1.0,
        }]
        # Surface selects a non-existent ID
        selected = [
            {'id': 'nonexist', 'why': 'hallucinated'},
            {'id': 'real1234', 'why': 'actual match'},
        ]

        output = format_surface_output(selected, candidates)

        # Should include the real node but not crash on the fake one
        self.assertIn('Real node', output)
        # Should show 2 in header (surface selected 2) but only render 1
        # Actually format_surface_output uses len(selected) for header count
        self.assertIn('Brain recalled', output)

    def test_correction_enrich_finds_correction_chains(self):
        """correction_enrich: node B corrects node A must be discoverable.

        The encoding agent sets correction_of when creating corrections.
        correction_enrich must find these relationships so the voice surface
        can warn Claude about superseded knowledge.
        """
        from servers.scales.s1.surface_contract import correction_enrich

        node_a = self.brain.remember(
            type='decision',
            title='Use REST API for all endpoints',
            content='REST for everything, no GraphQL.',
        )
        node_b = self.brain.remember(
            type='decision',
            title='Use GraphQL for complex queries',
            content='Switched to GraphQL for queries needing joins across 3+ tables.',
            correction_of=node_a['id'],
        )

        # correction_enrich takes a set of node IDs and the db connection
        # It should find that node_a has a correction (node_b corrects it)
        corrections = correction_enrich({node_a['id']}, self.brain.conn)

        self.assertIn(node_a['id'], corrections,
                      'correction_enrich did not find correction for node A')

        # Verify the correction info has the expected shape
        chain = corrections[node_a['id']]
        self.assertTrue(len(chain) > 0, 'No correction entries found')

        # Check all entries have the required fields format_judge_output reads
        for entry in chain:
            self.assertIn('id', entry, 'Correction entry missing "id"')
            self.assertIn('title', entry, 'Correction entry missing "title"')
            self.assertIn('direction', entry, 'Correction entry missing "direction"')
            self.assertIn(entry['direction'], ('corrects', 'corrected_by'),
                          'Unknown correction direction: %s' % entry['direction'])

        # node_b's ID must appear somewhere in the correction chain
        correction_ids = {e['id'] for e in chain}
        self.assertIn(node_b['id'][:8], correction_ids,
                      'Correcting node B not found in correction chain for node A')

    def test_hebbian_surface_selected_to_co_accessed_edges(self):
        """surface-selected IDs → _hebbian_strengthen: creates co_accessed edges.

        The Stop hook reads surface-selected IDs from a tmp file and calls
        _hebbian_strengthen. Only selected nodes get edges — the third
        (unselected) node must not get edges.
        """
        from servers.daemon_hooks import _hebbian_strengthen

        # Create 3 nodes
        n1 = self.brain.remember(type='rule', title='Hebbian test node alpha',
                                 content='First test node')
        n2 = self.brain.remember(type='rule', title='Hebbian test node beta',
                                 content='Second test node')
        n3 = self.brain.remember(type='rule', title='Hebbian test node gamma',
                                 content='Third test node (not selected)')

        session_id = 'test-hebbian-session'

        # Write surface-selected file (what _hebbian_strengthen reads)
        surface_path = '/tmp/brain-%s-surface-selected.json' % session_id
        try:
            with open(surface_path, 'w') as f:
                json.dump({'selected_ids': [n1['id'][:8], n2['id'][:8]]}, f)

            # Count co_accessed edges before Hebbian strengthening
            # (remember() auto-connects recent nodes, so some may already exist)
            pre_edges_n1_n2 = self.brain.conn.execute(
                """SELECT COUNT(*) FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE ((e.source_id = ? AND e.target_id = ?) OR (e.source_id = ? AND e.target_id = ?))
                AND er.relation = 'co_accessed'""",
                (n1['id'], n2['id'], n2['id'], n1['id'])
            ).fetchone()[0]
            pre_edges_n3 = self.brain.conn.execute(
                """SELECT COUNT(*) FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE (e.source_id = ? OR e.target_id = ?)
                AND er.relation = 'co_accessed'""",
                (n3['id'], n3['id'])
            ).fetchone()[0]

            # _hebbian_strengthen now takes stop_counter for trace correlation.
            _hebbian_strengthen(self.brain, session_id, stop_counter=1)

            # _hebbian_strengthen creates co_accessed edges between surface-selected nodes
            # Check that a co_accessed edge exists between n1 and n2
            edge = self.brain.conn.execute(
                """SELECT er.relation FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE ((e.source_id = ? AND e.target_id = ?) OR (e.source_id = ? AND e.target_id = ?))
                AND er.relation = 'co_accessed'""",
                (n1['id'], n2['id'], n2['id'], n1['id'])
            ).fetchone()
            self.assertIsNotNone(edge,
                                 'No co_accessed edge between surface-selected nodes')
            self.assertEqual(edge[0], 'co_accessed',
                             'Edge relation should be co_accessed, got: %s' % edge[0])

            # Verify n3 has no new co_accessed edges (since it was not selected)
            post_edges_n3 = self.brain.conn.execute(
                """SELECT COUNT(*) FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE (e.source_id = ? OR e.target_id = ?)
                AND er.relation = 'co_accessed'""",
                (n3['id'], n3['id'])
            ).fetchone()[0]
            self.assertEqual(post_edges_n3, pre_edges_n3,
                             'Unselected node got new co_accessed edges')
        finally:
            if os.path.exists(surface_path):
                os.unlink(surface_path)

    def test_recall_candidates_feed_into_surface_prompt(self):
        """recall → build_surface_prompt: recall output must produce a valid surface prompt.

        This is the full transition: recall results (enriched as candidates)
        passed through build_surface_prompt. The prompt must contain all candidate
        IDs and not raise errors. Tests that the recall output shape is compatible
        with the surface input shape.
        """
        from servers.scales.s1.surface_contract import build_surface_prompt

        # Store several nodes to get real recall results
        self.brain.remember(
            type='decision', title='Deploy to AWS us-east-1',
            content='Primary region is us-east-1 for latency to east coast users.',
            keywords='aws deploy region')
        self.brain.remember(
            type='mechanism', title='Blue-green deployment strategy',
            content='Zero-downtime deploys using blue-green with ALB target group switching.',
            keywords='deploy blue green zero downtime')

        result = self.brain.recall(query='how do we deploy', limit=10)
        results = result.get('results', [])
        self.assertTrue(len(results) > 0, 'No recall results for judge prompt test')

        # Enrich candidates the way hook_recall does: add 'score' from effective_activation
        candidates = []
        for r in results:
            c = dict(r)
            c['score'] = c.get('effective_activation', 0)
            candidates.append(c)

        # build_surface_prompt should not raise and should produce a string.
        # 2026-05-03: session_context parameter removed (per-session leak fix +
        # Frame replaces it). Frame becomes the prior; here we pass empty Frame
        # since the test is about candidate-shape compatibility, not Frame content.
        prompt, max_tokens = build_surface_prompt(
            candidates,
            user_message='how do we deploy to production?',
            recent_messages=[
                {'role': 'user', 'content': 'tell me about our deploy process'},
                {'role': 'assistant', 'content': 'Let me check the brain for deployment info.'},
            ],
        )

        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 100,
                        'Surface prompt suspiciously short: %d chars' % len(prompt))
        self.assertIsInstance(max_tokens, int)

        # Every candidate ID (first 8 chars) should appear in the prompt
        for c in candidates:
            short_id = str(c['id'])[:8]
            self.assertIn(short_id, prompt,
                          'Candidate %s not found in surface prompt — '
                          'format_candidate_for_surface dropped it' % short_id)

    def test_corrections_render_via_rich_node_metadata(self):
        """Corrections appear in surface output via the candidate's
        `_corrections` field (populated by correction_enrich) and rendered
        by render_rich_node — not via a `corrections=` kwarg on
        format_surface_output (that wiring was removed when corrections
        moved into the candidate object).

        Was: test_format_surface_output_corrections_wiring, which called
        `format_surface_output(selected, candidates, corrections=...)`.
        That kwarg no longer exists. The new path is direct: enrichment
        attaches `_corrections` to the candidate, and the renderer
        consumes it. Coverage of correction_enrich itself lives in
        tests/test_s1_data_assembly.py.
        """
        from servers.contract import render_rich_node

        node = {
            'id': 'abc12345',
            'type': 'decision',
            'title': 'Use MySQL for everything',
            'content': 'MySQL handles all our data needs.',
            'confidence': 0.8,
            '_corrections': [{
                'id': 'newer123',
                'title': 'Switch to PostgreSQL for analytics',
                'direction': 'corrected_by',
            }],
        }
        output = render_rich_node(node)
        # The renderer should surface the correction relationship.
        self.assertIn('PostgreSQL', output,
                      'Correction title should be visible in rendered output')


if __name__ == '__main__':
    unittest.main()
