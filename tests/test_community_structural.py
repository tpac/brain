"""Community structural-field derivation — servers/scales/s2/community_structural.

The five structural fields (size, members, internal_fraction, is_corridor,
dominant_type) are pure arithmetic over community_member edges, computed here
instead of hand-written by the encoder LLM. Two things must hold:
  1. correctness — derived values match the edges by hand-computation;
  2. PARITY — internal_fraction/is_corridor equal what the decoder computes
     fresh (the stamp must never disagree with the decoder's own number).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.brain_test_base import BrainTestBase
from servers.scales.s2.community_structural import (
    structural_metrics, compute_community_structural)
from servers.scales.s2.community_decoder import CommunityDecoder
from servers.scales.s2.community_encoder import CommunityEncoder


class TestCommunityStructural(BrainTestBase):
    needs_embedder = False

    def _node(self, title, ntype='finding'):
        return self.brain.remember(type=ntype, title=title, content='c',
                                   encoding_source='test')['id']

    def _community(self, title='C'):
        return self.brain.remember(
            type='community', title=title, content='c',
            encoding_source='s2:community_detection')['id']

    def _connect(self, a, b, rel, w=0.5):
        self.brain.connect(a, b, relation=rel, weight=w)

    # ── correctness ──

    def test_size_members_and_dominant_type(self):
        cid = self._community()
        a, b, c = self._node('a'), self._node('b'), self._node('c')
        d = self._node('d', 'decision')
        for m in (a, b, c, d):
            self._connect(cid, m, 'community_member', 0.3)

        res = compute_community_structural(self.brain, [cid])[cid]

        self.assertEqual(res['community_size'], 4)
        # 3 finding vs 1 decision → finding is dominant.
        self.assertEqual(res['community_dominant_type'], 'finding')

    def test_clique_is_fully_internal_not_corridor(self):
        cid = self._community()
        ms = [self._node('m%d' % i) for i in range(4)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        for i in range(4):
            for j in range(i + 1, 4):
                self._connect(ms[i], ms[j], 'extends')

        res = compute_community_structural(self.brain, [cid])[cid]

        self.assertEqual(res['community_internal_fraction'], 1.0)
        self.assertFalse(res['community_is_corridor'])

    def test_corridor_low_internal_fraction(self):
        cid = self._community()
        ms = [self._node('c%d' % i) for i in range(5)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        self._connect(ms[0], ms[1], 'extends')          # 1 internal
        for i in range(5):                                # 5 external
            self._connect(ms[0], self._node('x%d' % i), 'extends')

        res = compute_community_structural(self.brain, [cid])[cid]

        # int_frac = 1 / (1 + 5) ≈ 0.167 < 0.20, size 5 > 3 → corridor.
        self.assertEqual(res['community_internal_fraction'], round(1 / 6, 3))
        self.assertTrue(res['community_is_corridor'])

    def test_non_cohesion_relations_excluded(self):
        cid = self._community()
        a, b, c = self._node('a'), self._node('b'), self._node('c')
        for m in (a, b, c):
            self._connect(cid, m, 'community_member', 0.3)
        # 'related' (pure-generic, non_cohesion_relations) must NOT count
        # as internal cohesion.
        self._connect(a, b, 'related')

        res = compute_community_structural(self.brain, [cid])[cid]

        self.assertEqual(res['community_internal_fraction'], 0.0)
        self.assertFalse(res['community_is_corridor'])

    def test_empty_community_zero_size(self):
        cid = self._community()  # declared, no member edges
        res = compute_community_structural(self.brain, [cid])[cid]
        self.assertEqual(res['community_size'], 0)
        self.assertIsNone(res['community_dominant_type'])

    def test_multiple_communities_one_call(self):
        c1, c2 = self._community('C1'), self._community('C2')
        a, b = self._node('a'), self._node('b')
        x, y, z = self._node('x'), self._node('y'), self._node('z')
        for m in (a, b):
            self._connect(c1, m, 'community_member', 0.3)
        for m in (x, y, z):
            self._connect(c2, m, 'community_member', 0.3)

        res = compute_community_structural(self.brain, [c1, c2])

        self.assertEqual(res[c1]['community_size'], 2)
        self.assertEqual(res[c2]['community_size'], 3)

    # ── parity: stamp must equal the decoder's fresh computation ──

    def test_parity_with_decoder_adjacency(self):
        cid = self._community()
        ms = [self._node('p%d' % i) for i in range(5)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        self._connect(ms[0], ms[1], 'extends')
        self._connect(ms[1], ms[2], 'extends')
        for i in range(3):
            self._connect(ms[0], self._node('q%d' % i), 'extends')

        helper = compute_community_structural(self.brain, [cid])[cid]

        # Build the decoder's OWN whole-graph adjacency and compute fresh.
        dec = CommunityDecoder(self.brain)
        rel_to_fam = {rel: name
                      for name, asp in self.brain.aspects.all().items()
                      for rel in asp.edge_relations}
        edges_by_node, _ = dec._build_typed_adjacency(
            rel_to_fam, {'generic_relation', 'noise'})
        dec_metrics = structural_metrics(set(ms), edges_by_node)

        self.assertEqual(helper['community_internal_fraction'],
                         round(dec_metrics['internal_fraction'], 3))
        self.assertEqual(helper['community_is_corridor'],
                         dec_metrics['is_corridor'])


class TestStructuralStamp(BrainTestBase):
    """The second algorithmic Δ — CommunityEncoder._stamp_structural_fields."""
    needs_embedder = False

    def _node(self, title, ntype='finding'):
        return self.brain.remember(type=ntype, title=title, content='c',
                                   encoding_source='test')['id']

    def _community(self, title='C'):
        return self.brain.remember(
            type='community', title=title, content='c',
            encoding_source='s2:community_detection')['id']

    def _connect(self, a, b, rel, w=0.5):
        self.brain.connect(a, b, relation=rel, weight=w)

    def _stamped(self, cid):
        keys = ['community_size', 'community_internal_fraction',
                'community_is_corridor', 'community_dominant_type']
        return self.brain._meta_kv.get_fields_bulk([cid], keys).get(cid, {})

    def test_stamps_newly_created_community(self):
        # Live now, absent from pre_community_ids → treated as created.
        cid = self._community()
        ms = [self._node('m%d' % i) for i in range(4)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        for i in range(4):
            for j in range(i + 1, 4):
                self._connect(ms[i], ms[j], 'extends')

        enc = CommunityEncoder(self.brain, None, {})
        n = enc._stamp_structural_fields(
            encoder_proposals=[], pre_community_ids=set(), reconciled_ids=[])

        self.assertEqual(n, 1)
        meta = self._stamped(cid)
        self.assertEqual(meta['community_size'], '4')
        self.assertEqual(meta['community_internal_fraction'], '1.0')
        self.assertEqual(meta['community_is_corridor'], 'false')
        self.assertEqual(meta['community_dominant_type'], 'finding')

    def test_stamps_add_to_existing_target(self):
        cid = self._community()
        a, b = self._node('a'), self._node('b')
        for m in (a, b):
            self._connect(cid, m, 'community_member', 0.3)

        enc = CommunityEncoder(self.brain, None, {})
        proposals = [{'type': 'add_to_existing', 'communities': [{'id': cid}]}]
        # pre_community_ids includes cid → NOT new; it's touched via the proposal.
        n = enc._stamp_structural_fields(
            proposals, pre_community_ids={cid}, reconciled_ids=[])

        self.assertEqual(n, 1)
        self.assertEqual(self._stamped(cid)['community_size'], '2')

    def test_untouched_existing_community_not_stamped(self):
        cid = self._community()
        m = self._node('m')
        self._connect(cid, m, 'community_member', 0.3)

        enc = CommunityEncoder(self.brain, None, {})
        # Pre-existing, not in proposals, not new, not reconciled → skipped.
        n = enc._stamp_structural_fields(
            encoder_proposals=[], pre_community_ids={cid}, reconciled_ids=[])

        self.assertEqual(n, 0)
        self.assertEqual(self._stamped(cid), {})  # nothing written

    def test_backfill_all_communities_chunked(self):
        # THREE communities with drifted stored sizes, backfilled with chunk=1
        # (three separate chunks). Each chunk must stamp ONLY its own community:
        # the count is exactly 3 (a broken chunking that re-expands to all live
        # communities per chunk would report 9).
        c1, c2, c3 = (self._community('C1'), self._community('C2'),
                      self._community('C3'))
        for m in (self._node('a'), self._node('b'), self._node('c')):
            self._connect(c1, m, 'community_member', 0.3)
        for m in (self._node('x'), self._node('y')):
            self._connect(c2, m, 'community_member', 0.3)
        self._connect(c3, self._node('z'), 'community_member', 0.3)
        self.brain._meta_kv.set(c1, 'community_size', '99')   # wrong
        self.brain._meta_kv.set(c2, 'community_size', '0')    # wrong

        enc = CommunityEncoder(self.brain, None, {})
        n = enc.backfill_all_communities(chunk=1)

        self.assertEqual(n, 3)  # exactly one per chunk, not 3×3
        self.assertEqual(self._stamped(c1)['community_size'], '3')
        self.assertEqual(self._stamped(c2)['community_size'], '2')
        self.assertEqual(self._stamped(c3)['community_size'], '1')

    def test_corrects_a_drifted_stored_value(self):
        # The drift bug: a stale wrong size sitting in metadata. The stamp must
        # overwrite it with the true edge count.
        cid = self._community()
        ms = [self._node('d%d' % i) for i in range(3)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        self.brain._meta_kv.set(cid, 'community_size', '9')  # Haiku's wrong guess

        enc = CommunityEncoder(self.brain, None, {})
        enc._stamp_structural_fields(
            [{'type': 'health_update', 'community_id': cid}],
            pre_community_ids={cid}, reconciled_ids=[])

        self.assertEqual(self._stamped(cid)['community_size'], '3')  # corrected


if __name__ == '__main__':
    unittest.main()
