"""Supersession handling in S2 consolidation — heal + render contract.

Two halves of the survivor-ladder age-bias fix (journal audit 2026-07-25):

1. Healing pass (decoder): a live handoff with a live handoff successor
   (`supersedes` edge) is archived before clustering — so consolidation can
   never again absorb a session opener into its stale predecessor. Keyed on
   the edge AND both endpoint types, fail-safe for untyped openers.
   Superseded KNOWLEDGE (decision/fact/...) deliberately stays live — its
   staleness is annotated at read time by correction_enrich, and its
   reasoning keeps recall value.

2. Render contract (encoder): intra-cluster edges appear in the cluster
   text WITH direction (actor → relation → target). They used to be dropped
   by the external-edges-only filter and collapsed to a directionless
   CORRECTION_EDGE flag — which is exactly why the encoder merged
   supersession pairs backwards (it couldn't see which node supersedes
   which). A future change that re-filters intra-member edges out of
   _format_clusters must fail here.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class SupersessionBase(BrainTestBase):
    needs_embedder = False

    def _node(self, title, type='handoff', locked=False, **kw):
        r = self.brain.remember(type=type, title=title, content='content of %s' % title,
                                locked=locked, encoding_source='anchor', **kw)
        return r['id']

    def _archived(self, nid):
        return self.brain.conn.execute(
            'SELECT archived FROM nodes WHERE id = ?', (nid,)).fetchone()[0]

    def _decoder(self):
        from servers.scales.s2.consolidation_decoder import ConsolidationDecoder
        return ConsolidationDecoder(self.brain)


class TestHealSupersededHandoffs(SupersessionBase):
    """Decoder healing pass: predecessor openers are retired pre-clustering."""

    def test_superseded_handoff_archived_successor_live(self):
        old = self._node('opener 07-21')
        new = self._node('opener 07-23')
        self.brain.connect(new, old, relation='supersedes')

        dec = self._decoder()
        healed = dec._heal_graph()

        self.assertEqual(self._archived(old), 1)
        self.assertEqual(self._archived(new), 0)
        self.assertTrue(any(h['id'] == old and 'superseded' in h['reason']
                            for h in healed))
        # Step 10: the heal routes through dispatch — survivor lineage must
        # survive the routing (the batch archive op carries survivor_id) and
        # the archive must trace on the decoder's run chain.
        ptr = self.brain._meta_kv.get_all_bulk([old])[old].get(
            '_sys_archived_survivor_id')
        self.assertEqual(ptr, new, 'lineage lost in dispatch routing')
        rows = [t for t in self.brain._trace_dal.get_chain(dec.chain_id())
                if t['ref_type'] == 'node_archived' and t['ref_id'] == old]
        self.assertEqual(len(rows), 1,
                         'superseded-handoff heal must emit node_archived')

    def test_chain_archives_every_predecessor(self):
        # a ← b ← c (c newest): both a and b retire, head stays.
        a = self._node('opener 1')
        b = self._node('opener 2')
        c = self._node('opener 3')
        self.brain.connect(b, a, relation='supersedes')
        self.brain.connect(c, b, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(a), 1)
        self.assertEqual(self._archived(b), 1)
        self.assertEqual(self._archived(c), 0)

    def test_inverse_verb_superseded_by_also_retires(self):
        # Same semantics, inverse verb: old --superseded_by--> new.
        # The heal handles both canonical verbs — but ONLY these two,
        # never the (LLM-grown) correction_improvement aspect list.
        old = self._node('opener written inverse')
        new = self._node('opener successor')
        self.brain.connect(old, new, relation='superseded_by')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 1)
        self.assertEqual(self._archived(new), 0)

    def test_inverted_stored_orientation_still_retires_older(self):
        # Finding 1 of the a4f934c review: add_relation reuses the pair's
        # physical edge row in either orientation (Hebbian co-access creates
        # them in recall order), so `supersedes` can be STORED inverted —
        # old as source, new as target. The heal must not trust orientation:
        # created_at decides, and the OLDER node retires regardless.
        old = self._node('opener corrupt-direction old')
        new = self._node('opener corrupt-direction new')
        self.brain.connect(old, new, relation='supersedes')  # inverted on purpose

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 1)
        self.assertEqual(self._archived(new), 0)

    def test_self_loop_never_archives(self):
        # Finding 2: connect(x, x, 'supersedes') succeeds at the DAL; the
        # heal must not archive a node as superseded by itself.
        x = self._node('opener self-loop')
        self.brain.connect(x, x, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(x), 0)

    def test_equal_created_at_skips(self):
        # No date ground truth → never guess an archive.
        a = self._node('opener twin a')
        b = self._node('opener twin b')
        ts = self.brain.conn.execute(
            'SELECT created_at FROM nodes WHERE id = ?', (a,)).fetchone()[0]
        self.brain.conn.execute(
            'UPDATE nodes SET created_at = ? WHERE id = ?', (ts, b))
        self.brain.conn.commit()
        self.brain.connect(b, a, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(a), 0)
        self.assertEqual(self._archived(b), 0)

    def test_parallel_openers_untouched(self):
        # Two live threads, deliberately NO supersedes edge — both stay live.
        t1 = self._node('opener thread A')
        t2 = self._node('opener thread B')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(t1), 0)
        self.assertEqual(self._archived(t2), 0)

    def test_superseded_knowledge_stays_live(self):
        # supersedes between non-handoff types is knowledge lifecycle —
        # handled by correction_enrich at read time, never auto-archived.
        old = self._node('old decision', type='decision')
        new = self._node('new decision', type='decision')
        self.brain.connect(new, old, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 0)
        self.assertEqual(self._archived(new), 0)

    def test_locked_predecessor_skipped_quietly(self):
        old = self._node('locked opener', locked=True)
        new = self._node('successor opener')
        self.brain.connect(new, old, relation='supersedes')

        healed = self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 0)
        # Excluded in the query, not bounced off archive_node's guard —
        # so it never appears in the healed list (and never error-logs).
        self.assertFalse(any(h['id'] == old for h in healed))

    def test_mixed_type_supersession_untouched(self):
        # Edge exists but only one endpoint is a handoff → not opener
        # lifecycle; leave it alone.
        old = self._node('old opener')
        new = self._node('new decision', type='decision')
        self.brain.connect(new, old, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 0)


class TestIntraClusterEdgeRenderContract(SupersessionBase):
    """Encoder contract: intra-member edges render WITH direction.

    Pins the fix for the survivor-direction blindness. Do not weaken:
    if intra-cluster edges disappear from the render, the encoder is
    back to guessing survivor direction from age-correlated signals.
    """

    def _cluster(self, a, b, edge_details):
        return {
            'nodes': [a, b],
            'size': 2,
            'pre_class': 'needs_judgment',
            'content_cosine_max': 0.95,
            'title_cosine_max': 0.97,
            'node_details': {
                a: {'title': 'opener old', 'type': 'handoff'},
                b: {'title': 'opener new', 'type': 'handoff'},
            },
            'co_recall_count': 0,
            'judge_preference': {},
            'catalog_blind': {},
            'shared_edge_count': 0,
            'same_community': False,
            'has_correction_edge': True,
            'edge_details': edge_details,
            'communities': {},
        }

    def _encoder(self):
        from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
        return ConsolidationEncoder(self.brain)

    def test_intra_cluster_edge_rendered_with_direction(self):
        old = self._node('opener old')
        new = self._node('opener new')
        # Production shape (get_neighbors_bulk): an intra-cluster edge is
        # assigned to its SOURCE member only, direction='outgoing'. The
        # target member has NO mirror entry.
        edge_details = {
            new: {old: [{'relation': 'supersedes', 'description': 'newer opener',
                         'title': 'opener old', 'type': 'handoff',
                         'direction': 'outgoing'}]},
            old: {},
        }
        text = self._encoder()._format_clusters([self._cluster(old, new, edge_details)])

        expected = '%s → supersedes → %s' % (new[:8], old[:8])
        self.assertIn(expected, text,
                      'intra-cluster supersedes edge must render with direction '
                      '(actor first) — without it the encoder merges chains backwards')
        # Dedup: the mirrored incoming entry must NOT produce a reversed line.
        reversed_line = '%s → supersedes → %s' % (old[:8], new[:8])
        self.assertNotIn(reversed_line, text)
        self.assertEqual(text.count('supersedes →'), 1)

    def test_external_edges_still_external_only(self):
        # The intra block must not leak external edges, and vice versa.
        old = self._node('opener old')
        new = self._node('opener new')
        outsider = self._node('elsewhere', type='fact')
        edge_details = {
            new: {
                old: [{'relation': 'supersedes', 'description': '',
                       'title': 'opener old', 'type': 'handoff',
                       'direction': 'outgoing'}],
                outsider: [{'relation': 'extends', 'description': '',
                            'title': 'elsewhere', 'type': 'fact',
                            'direction': 'outgoing'}],
            },
            old: {},
        }
        text = self._encoder()._format_clusters([self._cluster(old, new, edge_details)])

        self.assertIn('%s → supersedes → %s' % (new[:8], old[:8]), text)
        # The external neighbor renders in the per-node External block,
        # not in the intra-cluster block.
        intra_section = text.split('Intra-cluster edges')[1].split('Nodes:')[0]
        self.assertNotIn(outsider[:8], intra_section)


if __name__ == '__main__':
    unittest.main()
