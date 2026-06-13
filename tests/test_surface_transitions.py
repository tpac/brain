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

    def test_correction_enrich_finds_correction_chains(self):
        """brain.correction_enrich: node B corrects node A must be discoverable.

        The encoder writes correction-aspect edges (corrects, supersedes,
        reframes, ...). brain.correction_enrich walks those edges so the
        surface and downstream consumers can warn Claude about superseded
        knowledge.
        """
        node_a = self.brain.remember(
            type='decision',
            title='Use REST API for all endpoints',
            content='REST for everything, no GraphQL.',
        )
        node_b = self.brain.remember(
            type='decision',
            title='Use GraphQL for complex queries',
            content='Switched to GraphQL for queries needing joins across 3+ tables.',
        )
        # Edge: node_b corrects node_a
        self.brain.connect_typed(
            source_id=node_b['id'], target_id=node_a['id'],
            relation='corrects', weight=0.5,
            description='GraphQL handles the cross-table queries REST struggled with',
            encoding_source='test:correction_chains')

        # brain.correction_enrich finds node_a's correction (node_b corrects it)
        corrections = self.brain.correction_enrich({node_a['id']})

        # Result keyed by both full and short id — accept either
        chain = corrections.get(node_a['id']) or corrections.get(node_a['id'][:8])
        self.assertTrue(chain, 'correction_enrich did not find correction for node A')

        # Verify the correction info has the expected shape
        self.assertTrue(len(chain) > 0, 'No correction entries found')

        # Check all entries carry the required fields downstream consumers read
        for entry in chain:
            self.assertIn('id', entry, 'Correction entry missing "id"')
            self.assertIn('title', entry, 'Correction entry missing "title"')
            self.assertIn('direction', entry, 'Correction entry missing "direction"')
            self.assertIn(entry['direction'], ('corrects', 'corrected_by'),
                          'Unknown correction direction: %s' % entry['direction'])
            # Heavy payload now also carries relation + edge_description
            self.assertIn('relation', entry, 'Correction entry missing "relation"')
            self.assertIn('edge_description', entry,
                          'Correction entry missing "edge_description"')

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

        # Write surface-selected file (what _hebbian_strengthen reads).
        # Path via the contract helper — this test once hardcoded the old
        # counter-less format and silently tested nothing (file_missing).
        from servers.scales.s1.surface_contract import surface_selected_path
        surface_path = surface_selected_path(session_id, 1)
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

            # Phase 5 (2026-05-18) made Hebbian async: _hebbian_strengthen
            # only enqueues; the bg worker's drain does the edge SQL. Drain
            # synchronously so the assertions below see the edges.
            from servers import recall_write_queue
            recall_write_queue.drain_once(self.brain)

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

    def test_recently_surfaced_is_session_scoped(self):
        """_get_recently_surfaced must not leak surfaces from parallel sessions.

        Each session's Haiku gets an exclusion list of nodes already shown to
        Anchor. That list must come from THIS session's surface_selected traces
        only — otherwise Session B sees Session A's picks marked 'already seen'
        and Haiku skips re-selecting them even when relevant to B.
        """
        from servers.scales.s1.surface import _get_recently_surfaced

        n_a = self.brain.remember(type='rule', title='Session A surfaced this',
                                  content='Only A should see this in exclusion list')
        n_b = self.brain.remember(type='rule', title='Session B surfaced this',
                                  content='Only B should see this in exclusion list')

        self.brain._trace_dal.append(
            chain_id='s1r-sessA-1', scale='s1', event_type='K',
            ref_type='surface_selected', ref_id=json.dumps([n_a['id']]),
            session_id='sess-A')
        self.brain._trace_dal.append(
            chain_id='s1r-sessB-1', scale='s1', event_type='K',
            ref_type='surface_selected', ref_id=json.dumps([n_b['id']]),
            session_id='sess-B')

        only_a = _get_recently_surfaced(self.brain, 'sess-A')
        only_b = _get_recently_surfaced(self.brain, 'sess-B')

        a_ids = {entry['id'] for entry in only_a}
        b_ids = {entry['id'] for entry in only_b}

        self.assertIn(n_a['id'], a_ids, "Session A's surface missing from A's exclusion list")
        self.assertNotIn(n_b['id'], a_ids,
                         "Session B's surface LEAKED into Session A's exclusion list")
        self.assertIn(n_b['id'], b_ids, "Session B's surface missing from B's exclusion list")
        self.assertNotIn(n_a['id'], b_ids,
                         "Session A's surface LEAKED into Session B's exclusion list")


class TestSelectionLivenessGate(BrainTestBase):
    """2026-06-12 — archived nodes must be dropped from Haiku's resolved
    selection before they become spread seeds. Production incident: node
    90664c51 was absorbed by S2 consolidation mid-session; Haiku kept
    re-selecting its id from session history (conversation text +
    recently-surfaced block). Each acceptance seeded a vector-less node
    (spread_seed_no_vectors) and re-wrote the dead id into the
    surface_selected trace — a self-perpetuating loop."""

    needs_embedder = False

    def test_archived_selection_dropped_live_kept(self):
        from servers.scales.s1.surface import _drop_archived_selected

        live = self.brain.remember(type='test', title='gate_live',
                                   content='c', auto_connect=False,
                                   encoding_source='anchor:test')
        dead = self.brain.remember(type='test', title='gate_dead',
                                   content='c', auto_connect=False,
                                   encoding_source='anchor:test')
        arch = self.brain.archive_node(dead['id'], archived_by='anchor:test',
                                       reason='liveness gate test')
        self.assertTrue(arch.get('ok'))

        selected_why = {live['id']: 'relevant', dead['id']: 'stale'}
        selected_mode = {live['id']: 'arc', dead['id']: 'arc'}
        selected_short_ids = {live['id'][:8], dead['id'][:8]}

        dropped = _drop_archived_selected(
            self.brain, selected_why, selected_mode, selected_short_ids)

        self.assertEqual(dropped, [dead['id']])
        self.assertNotIn(dead['id'], selected_why)
        self.assertNotIn(dead['id'], selected_mode)
        self.assertNotIn(dead['id'][:8], selected_short_ids)
        self.assertIn(live['id'], selected_why)
        self.assertIn(live['id'][:8], selected_short_ids)

    def test_all_live_selection_untouched(self):
        from servers.scales.s1.surface import _drop_archived_selected

        a = self.brain.remember(type='test', title='gate_a', content='c',
                                auto_connect=False,
                                encoding_source='anchor:test')
        selected_why = {a['id']: 'relevant'}
        selected_mode = {a['id']: 'arc'}
        selected_short_ids = {a['id'][:8]}

        dropped = _drop_archived_selected(
            self.brain, selected_why, selected_mode, selected_short_ids)

        self.assertEqual(dropped, [])
        self.assertIn(a['id'], selected_why)

    def test_surface_selected_file_write(self):
        # Single write site: the file lands post-gate with whatever the
        # caller passes — filtered ids only, by construction of run_surface.
        from servers.scales.s1.surface import _write_surface_selected_file
        from servers.scales.s1.surface_contract import surface_selected_path

        path = surface_selected_path('test-liveness-file', 7)
        try:
            _write_surface_selected_file(
                self.brain, 'test-liveness-file', 7, {'aaaa1111', 'bbbb2222'})
            with open(path) as f:
                on_disk = set(json.load(f)["selected_ids"])
            self.assertEqual(on_disk, {'aaaa1111', 'bbbb2222'})
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_run_surface_drops_archived_end_to_end(self):
        """Wiring test: an archived node Haiku selects must not reach the
        surfaced-ids file when it goes through the REAL run_surface path.

        The other gate tests call _drop_archived_selected directly — they'd
        still pass if the gate call were deleted from run_surface. This one
        drives run_surface with a canned Haiku selection (live + archived)
        and asserts the archived id never lands in the file Hebbian reads,
        so a wiring regression (gate removed / called in the wrong place /
        file written before the gate) fails loudly. query_vec=None skips
        spread expansion, so no embedder is needed."""
        from servers.scales.s1 import surface as surface_mod
        from servers.scales.s1.surface_contract import surface_selected_path

        live = self.brain.remember(type='test', title='wire_live', content='c',
                                   auto_connect=False,
                                   encoding_source='anchor:test')
        dead = self.brain.remember(type='test', title='wire_dead', content='c',
                                   auto_connect=False,
                                   encoding_source='anchor:test')
        self.assertTrue(self.brain.archive_node(
            dead['id'], archived_by='anchor:test',
            reason='wiring test').get('ok'))

        session_id = 'test-run-surface-wiring'
        ctx = self.brain.get_or_create_session(session_id)
        candidates_data = [
            {'id': live['id'], 'title': 'wire_live', 'type': 'test', 'score': 0.9},
            {'id': dead['id'], 'title': 'wire_dead', 'type': 'test', 'score': 0.9},
        ]

        # Canned Haiku selection: both the live node and the archived one.
        def _fake_call_surface(brain, cands, user_message, recent_messages,
                               sid, result, frame=''):
            return ({'selected': [
                {'id': live['id'][:8], 'why': 'relevant'},
                {'id': dead['id'][:8], 'why': 'stale'},
            ]}, 'prompt', 100, None)

        orig = surface_mod._call_surface
        surface_mod._call_surface = _fake_call_surface
        path = surface_selected_path(session_id, ctx.stop_counter)
        try:
            surface_mod.run_surface(
                self.brain, ctx, candidates_data, 'user msg', [], {},
                'enriched query', [], 'test-recall-ref', session_id, None,
                query_vec=None)
            with open(path) as f:
                on_disk = set(json.load(f)['selected_ids'])
            self.assertIn(live['id'][:8], on_disk,
                          'live node missing from surfaced-ids file')
            self.assertNotIn(
                dead['id'][:8], on_disk,
                'archived node reached the surfaced-ids file — '
                'liveness gate is not wired into run_surface')
        finally:
            surface_mod._call_surface = orig
            if os.path.exists(path):
                os.remove(path)
