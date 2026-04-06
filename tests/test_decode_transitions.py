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

    def test_recall_result_shape_for_judge(self):
        """recall → judge: recall results must have all fields build_judge_prompt expects.

        build_judge_prompt calls format_candidate_for_judge which reads:
        id, type, title, content, confidence, locked, score (via effective_activation),
        keywords. If recall drops any of these, the judge gets malformed candidates.
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

        # Fields that format_candidate_for_judge reads from each candidate
        required_fields = {'id', 'type', 'title', 'content', 'confidence', 'locked'}
        node = results[0]
        missing = required_fields - set(node.keys())
        self.assertEqual(missing, set(),
                         'Recall result missing fields needed by judge: %s' % missing)

        # score comes from effective_activation — judge reads it as 'score'
        # The hook_recall code in daemon_hooks enriches candidates before passing
        # to build_judge_prompt. But the raw recall result must have the base fields.
        self.assertIn('effective_activation', node,
                      'Missing effective_activation — judge needs this as candidate score')

    def test_judge_output_format_judge_output_compatibility(self):
        """judge JSON → format_judge_output: fake judge response must produce valid output.

        format_judge_output expects selected=[{"id": "...", "why": "..."}] and
        matches them against candidates by id[:8]. If the format diverges,
        Claude gets empty context despite the judge selecting nodes.
        """
        from servers.scales.s1.recall_contract import format_judge_output

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

        # Simulate judge output (what Haiku returns)
        selected = [{'id': short_id, 'why': 'directly answers rate limit question'}]

        output = format_judge_output(selected, candidates)

        self.assertIn('Brain recalled', output,
                      'format_judge_output did not produce header')
        self.assertIn('Rate limiting', output,
                      'format_judge_output did not include node title')
        self.assertIn('token bucket', output,
                      'format_judge_output did not include node content')
        self.assertIn(short_id, output,
                      'format_judge_output did not include node ID')

    def test_judge_output_unmatched_id_skipped(self):
        """judge JSON → format_judge_output: selected ID not in candidates must not crash.

        If the judge hallucinates an ID that doesn't match any candidate,
        format_judge_output should skip it gracefully.
        """
        from servers.scales.s1.recall_contract import format_judge_output

        candidates = [{
            'id': 'real1234-full-id',
            'type': 'rule',
            'title': 'Real node',
            'content': 'Real content',
            'confidence': 1.0,
        }]
        # Judge selects a non-existent ID
        selected = [
            {'id': 'nonexist', 'why': 'hallucinated'},
            {'id': 'real1234', 'why': 'actual match'},
        ]

        output = format_judge_output(selected, candidates)

        # Should include the real node but not crash on the fake one
        self.assertIn('Real node', output)
        # Should show 2 in header (judge selected 2) but only render 1
        # Actually format_judge_output uses len(selected) for header count
        self.assertIn('Brain recalled', output)

    def test_correction_enrich_finds_correction_chains(self):
        """correction_enrich: node B corrects node A must be discoverable.

        The encoding agent sets correction_of when creating corrections.
        correction_enrich must find these relationships so the voice surface
        can warn Claude about superseded knowledge.
        """
        from servers.scales.s1.recall_contract import correction_enrich

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

    def test_hebbian_judge_selected_to_co_accessed_edges(self):
        """judge-selected IDs → _hebbian_strengthen: creates co_accessed edges.

        The Stop hook reads judge-selected IDs from a tmp file and calls
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

        # Write judge-selected file (what run_judge writes)
        judge_path = '/tmp/brain-%s-judge-selected.json' % session_id
        try:
            with open(judge_path, 'w') as f:
                json.dump({'selected_ids': [n1['id'][:8], n2['id'][:8]]}, f)

            # Count co_accessed edges before Hebbian strengthening
            # (remember() auto-connects recent nodes, so some may already exist)
            pre_edges_n1_n2 = self.brain.conn.execute(
                "SELECT COUNT(*) FROM edges "
                "WHERE source_id = ? AND target_id = ? AND relation = 'co_accessed'",
                (n1['id'], n2['id'])
            ).fetchone()[0]
            pre_edges_n3 = self.brain.conn.execute(
                "SELECT COUNT(*) FROM edges "
                "WHERE (source_id = ? OR target_id = ?) "
                "AND relation = 'co_accessed' AND description = 'judge-selected'",
                (n3['id'], n3['id'])
            ).fetchone()[0]

            _hebbian_strengthen(self.brain, session_id)

            # _hebbian_strengthen uses connect_typed with description='judge-selected'
            # Check that a judge-selected co_accessed edge exists between n1 and n2
            edge = self.brain.conn.execute(
                "SELECT relation, description FROM edges "
                "WHERE source_id = ? AND target_id = ? "
                "AND description = 'judge-selected'",
                (n1['id'], n2['id'])
            ).fetchone()
            self.assertIsNotNone(edge,
                                 'No judge-selected co_accessed edge between selected nodes')
            self.assertEqual(edge[0], 'co_accessed',
                             'Edge relation should be co_accessed, got: %s' % edge[0])

            # Verify n3 has no judge-selected co_accessed edges
            n3_judge_edges = self.brain.conn.execute(
                "SELECT COUNT(*) FROM edges "
                "WHERE (source_id = ? OR target_id = ?) "
                "AND relation = 'co_accessed' AND description = 'judge-selected'",
                (n3['id'], n3['id'])
            ).fetchone()[0]
            self.assertEqual(n3_judge_edges, 0,
                             'Unselected node got judge-selected co_accessed edges')
        finally:
            if os.path.exists(judge_path):
                os.unlink(judge_path)

    def test_recall_candidates_feed_into_judge_prompt(self):
        """recall → build_judge_prompt: recall output must produce a valid judge prompt.

        This is the full transition: recall results (enriched as candidates)
        passed through build_judge_prompt. The prompt must contain all candidate
        IDs and not raise errors. Tests that the recall output shape is compatible
        with the judge input shape.
        """
        from servers.scales.s1.recall_contract import build_judge_prompt

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

        # build_judge_prompt should not raise and should produce a string
        prompt, max_tokens = build_judge_prompt(
            candidates,
            user_message='how do we deploy to production?',
            session_context='Working on deployment pipeline',
            recent_messages=[
                {'role': 'user', 'content': 'tell me about our deploy process'},
                {'role': 'assistant', 'content': 'Let me check the brain for deployment info.'},
            ],
        )

        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 100,
                        'Judge prompt suspiciously short: %d chars' % len(prompt))
        self.assertIsInstance(max_tokens, int)

        # Every candidate ID (first 8 chars) should appear in the prompt
        for c in candidates:
            short_id = str(c['id'])[:8]
            self.assertIn(short_id, prompt,
                          'Candidate %s not found in judge prompt — '
                          'format_candidate_for_judge dropped it' % short_id)

    def test_format_judge_output_corrections_wiring(self):
        """format_judge_output + corrections: correction data must appear in output.

        When correction_enrich returns data for a selected node,
        format_judge_output must include the correction warning in the
        additionalContext string. Tests the corrections kwarg wiring.
        """
        from servers.scales.s1.recall_contract import format_judge_output

        node_id = 'abc12345-full-uuid-here'
        short_id = node_id[:8]

        candidates = [{
            'id': node_id,
            'type': 'decision',
            'title': 'Use MySQL for everything',
            'content': 'MySQL handles all our data needs.',
            'confidence': 0.8,
        }]
        selected = [{'id': short_id, 'why': 'answers database question'}]

        # Simulate correction_enrich output
        corrections = {
            node_id: [{
                'id': 'newer123',
                'title': 'Switch to PostgreSQL for analytics',
                'direction': 'corrected_by',
            }]
        }

        output = format_judge_output(selected, candidates, corrections=corrections)

        self.assertIn('Updated by', output,
                      'Correction warning not in format_judge_output — '
                      'corrections kwarg is not wired through')
        self.assertIn('PostgreSQL', output,
                      'Correction title not shown in output')


if __name__ == '__main__':
    unittest.main()
