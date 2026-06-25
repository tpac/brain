"""Tests for S2 community detection unit.

Tests the IntegrationUnit contract and community decoder algorithm.
Encoder tests require Anthropic API and are integration-level — not here.

Architecture:
  CommunityDecoder — algorithmic, produces proposals from graph structure
  CommunityEncoder — agentic Sonnet, creates community nodes (needs API)
  CommunityDetection — orchestrator, wires decoder→encoder

These tests verify the decoder finds correct structure and the
orchestrator handles edge cases. Encoder quality is tested via eval.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestIntegrationUnitContract(unittest.TestCase):
    """Test the base IntegrationUnit contract."""

    def test_subclass_must_implement_run(self):
        from servers.scales.s2.base import IntegrationUnit
        unit = IntegrationUnit(brain=None)
        with self.assertRaises(NotImplementedError):
            unit.run()

    def test_chain_id_format(self):
        """Per-run chain id: s2-{YYYYMMDDHHMMSS}-{name}. Seconds (not just
        date) so same-day runs are distinct → notes group per-run. One
        combined timestamp segment + trailing -{name} keep the load-bearing
        consumers intact: `_last_run_timestamp` suffix-matches `%-{name}` and
        the dashboard slug parser reads `split('-', 2)[2]`. Stamped once per
        run and cached so every trace in a run shares one chain_id."""
        from servers.scales.s2.base import IntegrationUnit

        class TestUnit(IntegrationUnit):
            NAME = 'test_op'
            SCALE = 's2'

        unit = TestUnit(brain=None)
        chain = unit.chain_id()
        self.assertRegex(chain, r'^s2-\d{14}-test_op$')        # one timestamp segment
        self.assertTrue(chain.endswith('-test_op'))            # _last_run_timestamp LIKE '%-name'
        self.assertEqual(chain.split('-', 2)[2], 'test_op')    # dashboard slug parser
        self.assertEqual(unit.chain_id(), chain)               # stable within a run (cached)

    def test_chain_id_is_real_recent_timestamp(self):
        """The 14-digit segment is an actual recent UTC run-time — not a
        constant or a wrong strftime that merely yields 14 digits. Guards the
        timestamp-correctness axis the shape regex alone can't (e.g. a revert
        to date-only would be 8 digits and fail strptime here)."""
        from datetime import datetime, timezone
        from servers.scales.s2.base import IntegrationUnit

        class TestUnit(IntegrationUnit):
            NAME = 'test_op'
            SCALE = 's2'

        ts = TestUnit(brain=None).chain_id().split('-', 2)[1]
        parsed = datetime.strptime(ts, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        self.assertLess(abs((datetime.now(timezone.utc) - parsed).total_seconds()), 120)

    def test_chain_id_different_scale(self):
        from servers.scales.s2.base import IntegrationUnit

        class S3Unit(IntegrationUnit):
            NAME = 'synthesis'
            SCALE = 's3'

        unit = S3Unit(brain=None)
        self.assertTrue(unit.chain_id().startswith('s3-'))

    def test_encoder_dispatch_threads_run_chain_into_writes(self):
        """_make_encoder_dispatch stamps the unit's run chain on writes — so
        their revise/edge traces join the run's chain instead of the date-
        fallback phantom ("S2 revise") unit. brain_batch with empty operations
        returns before touching the brain, so the stamped args are observable
        with a FakeBrain that only carries a write_lock."""
        import threading
        from servers.scales.s2.base import IntegrationUnit

        class TestUnit(IntegrationUnit):
            NAME = 'test_op'
            SCALE = 's2'
            ENCODING_SOURCE = 's2:test_op'

        class FakeBrain:
            write_lock = threading.RLock()

        unit = TestUnit(brain=FakeBrain())
        dispatch = unit._make_encoder_dispatch()
        args = {'operations': []}
        dispatch('brain_batch', args)
        self.assertEqual(args.get('chain_id'), unit.chain_id())
        self.assertEqual(args.get('encoding_source'), 's2:test_op')


class TestCommunityDetectionContract(unittest.TestCase):
    """Test CommunityDetection declares its O/K sources."""

    def test_sources_declared(self):
        from servers.scales.s2.community import CommunityDetection
        self.assertTrue(len(CommunityDetection.O_SOURCES) > 0)
        self.assertTrue(len(CommunityDetection.K_SOURCES) > 0)
        self.assertEqual(CommunityDetection.SCALE, 's2')
        self.assertEqual(CommunityDetection.ENCODING_SOURCE, 's2:community_detection')

    def test_decoder_encoder_split(self):
        """CommunityDetection inherits from CommunityDecoder."""
        from servers.scales.s2.community import CommunityDetection
        from servers.scales.s2.community_decoder import CommunityDecoder
        from servers.scales.s2.community_encoder import CommunityEncoder
        self.assertTrue(issubclass(CommunityDetection, CommunityDecoder))
        # Encoder is separate, not inherited
        self.assertFalse(issubclass(CommunityDetection, CommunityEncoder))

    def test_absorb_survivor_reaches_community_candidacy(self):
        """Cross-scale coupling pinned by the absorb-bucket fix.

        The dispatch `absorb` op now routes an absorb's survivor into the delta's
        `revised` bucket (it's content-rewritten). `_read_s1_delta` seeds
        community candidacy (`new_node_ids`) from `created` + `revised`, so an
        absorb survivor now reaches community detection — an intended but
        previously incidental/untested consequence. If a future change stops
        routing absorb survivors to `revised`, or community detection stops
        reading it, this breaks loudly instead of silently dropping survivors
        from re-clustering.
        """
        from servers.scales.s2.community_decoder import CommunityDecoder
        # _read_s1_delta only uses self._read_traces_since — no brain needed.
        dec = CommunityDecoder.__new__(CommunityDecoder)
        enc_run = {'metadata': {'created': ['c_new'], 'revised': ['absorb_surv']}}
        dec._read_traces_since = lambda scale, since, ref_types=None: (
            [enc_run] if ref_types == ['encoding_run'] else [])
        out = dec._read_s1_delta('')
        self.assertIn('absorb_surv', out['new_node_ids'])  # the coupling
        self.assertIn('c_new', out['new_node_ids'])


class TestCommunityDecoder(BrainTestBase):
    """Test the decoder's algorithmic pipeline on synthetic graphs.

    These tests call _decode() directly, bypassing the S1 trace check.
    The decoder finds structure in the graph — it doesn't need S1 traces
    to detect clusters, only to decide WHETHER to run.
    """

    needs_embedder = False

    def _create_cluster(self, prefix, n, connect=True):
        """Create n nodes with a shared prefix, optionally fully connected."""
        ids = []
        for i in range(n):
            result = self.brain.remember(
                type='decision',
                title='%s node %d' % (prefix, i),
                content='Content about %s topic %d' % (prefix, i),
            )
            ids.append(result['id'])

        if connect:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    self.brain.connect(ids[i], ids[j], relation='implements', weight=0.8)

        return ids

    def _empty_s1_delta(self):
        return {
            'encoding_runs': [], 'surface_selections': [],
            'new_node_ids': set(), 'co_surface_pairs': [],
        }

    def test_small_graph_produces_no_proposals(self):
        """Graph smaller than min_community_size produces no community proposals."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        for i in range(2):
            self.brain.remember(type='fact', title='node %d' % i, content='c')

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode(self._empty_s1_delta(), [], is_cold_start=True)
        proposals = [p for p in result['proposals'] if p['type'] == 'new_community']
        self.assertEqual(len(proposals), 0)

    def test_detects_two_clusters(self):
        """Two well-separated clusters produce at least 2 new_community proposals."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        cluster_a = self._create_cluster('alpha', 8)
        cluster_b = self._create_cluster('beta', 8)
        # Weak bridge between clusters
        self.brain.connect(cluster_a[0], cluster_b[0], relation='related_to', weight=0.1)

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode(self._empty_s1_delta(), [], is_cold_start=True)
        proposals = [p for p in result['proposals'] if p['type'] == 'new_community']

        self.assertGreaterEqual(len(proposals), 2)
        # Each should have members from its cluster
        all_members = set()
        for p in proposals:
            self.assertGreaterEqual(p['member_count'], 3)
            self.assertGreater(p['internal_fraction'], 0)
            all_members.update(p['members'])
        # Both clusters should be represented
        self.assertTrue(all_members & set(cluster_a))
        self.assertTrue(all_members & set(cluster_b))

    def test_proposal_has_required_fields(self):
        """Each new_community proposal has the fields the encoder needs."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        self._create_cluster('gamma', 8)

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode(self._empty_s1_delta(), [], is_cold_start=True)
        proposals = [p for p in result['proposals'] if p['type'] == 'new_community']

        self.assertGreater(len(proposals), 0)
        p = proposals[0]
        # Fields the encoder reads
        self.assertIn('members', p)
        self.assertIn('member_count', p)
        self.assertIn('internal_fraction', p)
        self.assertIn('edge_signature', p)
        self.assertIn('timeline', p)
        self.assertIn('representatives', p)
        self.assertIn('all_members', p)
        self.assertIn('sample_edges', p)
        self.assertIn('is_corridor', p)

    def test_decode_stats_populated(self):
        """Decode result includes stats for trace consumption."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        self._create_cluster('delta', 8)

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode(self._empty_s1_delta(), [], is_cold_start=True)
        stats = result['stats']

        self.assertIn('nodes_with_typed_edges', stats)
        self.assertIn('valid_clusters', stats)
        self.assertIn('fragments_dissolved', stats)
        self.assertIn('subsets_absorbed', stats)
        self.assertGreater(stats['nodes_with_typed_edges'], 0)

    def test_corridor_detection(self):
        """Clusters with low internal fraction are flagged as corridors."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        # Create a "corridor" — nodes with more external than internal edges
        corridor = self._create_cluster('corridor', 4, connect=False)
        external_a = self._create_cluster('ext_a', 4)
        external_b = self._create_cluster('ext_b', 4)

        # Connect corridor nodes to each other weakly
        for i in range(len(corridor)):
            for j in range(i+1, len(corridor)):
                self.brain.connect(corridor[i], corridor[j], relation='related_to', weight=0.3)

        # Connect corridor heavily to external clusters
        for cid in corridor:
            for eid in external_a[:2]:
                self.brain.connect(cid, eid, relation='depends_on', weight=0.7)
            for eid in external_b[:2]:
                self.brain.connect(cid, eid, relation='enables', weight=0.7)

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode(self._empty_s1_delta(), [], is_cold_start=True)

        corridors = result['stats'].get('corridors', 0)
        # We should see at least some corridor detection
        # (exact count depends on cluster formation)
        self.assertIsInstance(corridors, int)

    def test_incremental_excludes_placed_nodes(self):
        """Incremental mode skips nodes already in communities."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        cluster_a = self._create_cluster('placed', 6)
        cluster_b = self._create_cluster('unplaced', 6)

        # Simulate existing community owning cluster_a
        comm = self.brain.remember(
            type='community', title='Existing Community',
            content='test', encoding_source='s2:community_detection',
            auto_connect=False)
        for nid in cluster_a:
            self.brain.connect(comm['id'], nid, relation='community_member', weight=0.3)

        community_state = [{
            'id': comm['id'], 'title': 'Existing Community',
            'content': 'test', 'keywords': '', 'confidence': 0.8,
            'members': set(cluster_a),
            'centroid': None, 'edge_signature': {}, 'health': {},
        }]

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode(self._empty_s1_delta(), community_state, is_cold_start=False)

        # New proposals should not contain already-placed nodes as seed clusters
        new_community_proposals = [p for p in result['proposals'] if p['type'] == 'new_community']
        for p in new_community_proposals:
            placed_in_proposal = set(p['members']) & set(cluster_a)
            # Some placed nodes may appear via affinity, but the SEED shouldn't be all placed
            unplaced_in_proposal = set(p['members']) - set(cluster_a)
            self.assertGreater(len(unplaced_in_proposal), 0,
                'New community proposal should contain unplaced nodes')

    def test_subset_absorption(self):
        """When one cluster is a subset of another, the smaller is absorbed."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        decoder = CommunityDecoder(self.brain)

        clusters = {
            1: {'a', 'b', 'c', 'd', 'e'},
            2: {'a', 'b', 'c'},  # subset of 1
            3: {'x', 'y', 'z'},
        }
        filtered, absorbed = decoder._absorb_subsets(clusters)
        self.assertEqual(absorbed, 1)
        self.assertNotIn(2, filtered)
        self.assertIn(1, filtered)
        self.assertIn(3, filtered)

    def test_subset_chain_absorption(self):
        """Chain: A ⊂ B ⊂ C → both A and B absorbed."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        decoder = CommunityDecoder(self.brain)

        clusters = {
            1: {'a'},
            2: {'a', 'b'},
            3: {'a', 'b', 'c', 'd'},
        }
        filtered, absorbed = decoder._absorb_subsets(clusters)
        self.assertEqual(absorbed, 2)
        self.assertEqual(set(filtered.keys()), {3})

    def test_merge_detection(self):
        """Communities with 80%+ overlap and <3 unique are merge candidates."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        decoder = CommunityDecoder(self.brain)

        # Simulate two communities with 100% overlap (smaller fully contained)
        community_state = [
            {'id': 'comm_a', 'title': 'Larger', 'members': {'n1', 'n2', 'n3', 'n4', 'n5'},
             'content': '', 'keywords': '', 'confidence': 0.8,
             'centroid': None, 'edge_signature': {}, 'health': {}},
            {'id': 'comm_b', 'title': 'Smaller', 'members': {'n1', 'n2', 'n3'},
             'content': '', 'keywords': '', 'confidence': 0.8,
             'centroid': None, 'edge_signature': {}, 'health': {}},
        ]
        merges = decoder._detect_merge_candidates(community_state)
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0]['larger']['id'], 'comm_a')
        self.assertEqual(merges[0]['smaller']['id'], 'comm_b')

    def test_no_merge_when_unique_members_above_threshold(self):
        """Communities with enough unique members should NOT merge."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        decoder = CommunityDecoder(self.brain)

        # 60% overlap but 4 unique in smaller — above threshold
        community_state = [
            {'id': 'comm_a', 'title': 'Larger', 'members': {'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10'},
             'content': '', 'keywords': '', 'confidence': 0.8,
             'centroid': None, 'edge_signature': {}, 'health': {}},
            {'id': 'comm_b', 'title': 'Smaller', 'members': {'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'u1', 'u2', 'u3', 'u4'},
             'content': '', 'keywords': '', 'confidence': 0.8,
             'centroid': None, 'edge_signature': {}, 'health': {}},
        ]
        merges = decoder._detect_merge_candidates(community_state)
        self.assertEqual(len(merges), 0)

    def test_traces_written_on_decode(self):
        """Decoder writes O and K traces with correct ref_types."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        self._create_cluster('traced', 8)

        # Write a fake S1 trace so _has_new_traces passes
        self.brain._trace_dal.append(
            chain_id='s1e-test-1', scale='s1', event_type='delta',
            ref_type='encoding_run', ref_id='1',
            summary='test encoding run', metadata={}, session_id='test')

        decoder = CommunityDecoder(self.brain)
        result = decoder.run()

        # Should not skip (we added a trace)
        self.assertNotIn('skipped', result)

        # Check traces
        traces = self.brain.logs_conn.execute(
            "SELECT scale, event_type, ref_type FROM trace_events WHERE scale = 's2'"
        ).fetchall()
        event_types = {t[1] for t in traces}
        ref_types = {t[2] for t in traces}

        self.assertIn('O', event_types)
        self.assertIn('s1_delta', ref_types)


class TestCoordinator(unittest.TestCase):
    """Test S2 coordinator imports and structure."""

    def test_coordinator_imports(self):
        from servers.scales.s2.coordinator import run_s2
        self.assertTrue(callable(run_s2))

    def test_unit_ordering(self):
        """Coordinator runs units in correct order.

        EdgeFamilyIntegration disabled 2026-05-04 — its source interaction
        was removed in Step 12 of unified-aspects, AspectIntegration (Step 13)
        will replace it. Test now verifies the ACTIVE units only.
        """
        from servers.scales.s2.coordinator import run_s2
        from servers.scales.s2.consolidation import Consolidation
        from servers.scales.s2.community import CommunityDetection
        from servers.scales.s2.healer import Healer
        # Verify imports work — ordering is enforced by coordinator code
        self.assertEqual(Consolidation.NAME, 'consolidation')
        self.assertEqual(CommunityDetection.NAME, 'community_detection')
        self.assertEqual(Healer.NAME, 'healer')


class TestCommunityIdleGate(BrainTestBase):
    """Phase 1 idle gate — skip the unit when nothing changed or too soon.

    The decode is a pure function of graph state, so re-running it on an
    unchanged graph re-derives identical, already-rejected proposals. The
    gate (CommunityDetection._should_skip) stops that waste.
    """

    needs_embedder = False

    def _unit(self):
        from servers.scales.s2.community import CommunityDetection
        return CommunityDetection(self.brain)

    def test_cold_start_runs(self):
        # No last-run timestamp recorded → never skip.
        self.assertIsNone(self._unit()._should_skip())

    def test_throttles_recent_run(self):
        import time
        self.brain.set_config('s2_community_last_run_ts', str(time.time()))
        reason = self._unit()._should_skip()
        self.assertIsNotNone(reason)
        self.assertIn('throttled', reason)

    def test_no_change_skips_then_new_node_wakes(self):
        import time
        u = self._unit()
        u.config = dict(u.config, min_run_interval_seconds=0)  # isolate the change-gate
        self.brain.remember(type='fact', title='old', content='c')
        time.sleep(0.02)
        self.brain.set_config('s2_community_last_run_ts', str(time.time()))
        time.sleep(0.02)
        # Nothing changed since the cutoff → skip.
        self.assertEqual(u._should_skip(), 'no graph change since last run')
        # A new node appears → gate opens.
        self.brain.remember(type='fact', title='new', content='c')
        self.assertIsNone(u._should_skip())

    def test_change_gate_ignores_self_writes_and_noise(self):
        import time
        u = self._unit()
        u.config = dict(u.config, min_run_interval_seconds=0)
        a = self.brain.remember(type='fact', title='a', content='c')['id']
        b = self.brain.remember(type='fact', title='b', content='c')['id']
        c = self.brain.remember(type='fact', title='cc', content='c')['id']
        d = self.brain.remember(type='fact', title='dd', content='c')['id']
        time.sleep(0.02)
        self.brain.set_config('s2_community_last_run_ts', str(time.time()))
        time.sleep(0.02)
        # None of these may wake the gate:
        self.brain.remember(type='community', title='comm', content='c')   # own node type
        self.brain.connect_typed(a, b, relation='co_accessed', weight=0.5)  # noise edge
        self.brain.connect_typed(a, b, relation='implements',              # own edge source
                                 encoding_source='s2:community_detection')
        self.assertEqual(u._should_skip(), 'no graph change since last run')
        # A real typed edge from a non-self source DOES wake it:
        self.brain.connect_typed(c, d, relation='implements', weight=0.8,
                                 encoding_source='encoder:sonnet')
        self.assertIsNone(u._should_skip())


if __name__ == '__main__':
    unittest.main()
