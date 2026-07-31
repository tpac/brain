"""connect_to catalog-title resolution — the deterministic ladder's Pass 2
(`_match_catalog_title`, decision 2026-07-30: vectors deleted from the write
path; token-exact or bounded near-title only).

Contract under test:
  • distance 0 (normalized token equality) resolves at any length
  • distance 1..NEAR_TITLE_MAX_OPS resolves ONLY behind three gates:
    ≥ NEAR_TITLE_MIN_TOKENS distinct query tokens, unique best candidate,
    runner-up ≥ NEAR_TITLE_MARGIN ops further out
  • ambiguity/photo-finish REFUSES (loud) rather than picks
  • acceptance at distance ≥ 1 logs connect_to_near_title (tolerance is
    visible, never silent)
  • the whole pass works with no embedder — needs_embedder=False here IS the
    determinism proof (the old fuzzy pass required vectors)

The class exercises the real write path via remember(connect_to=...), not the
helper in isolation, so the ladder order (id → sibling → catalog) stays pinned.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_remember import _title_tokens, _token_edit_distance


class TestTitleTokens(unittest.TestCase):
    def test_punctuation_variance_vanishes(self):
        # Hyphen/em-dash/percent/colon variance normalizes away; numbers stay.
        self.assertEqual(_title_tokens('cap-of-4 change: 19% — done'),
                         ['cap', 'of', '4', 'change', '19', 'done'])
        self.assertEqual(_title_tokens('Cap of 4 change 19 done'),
                         ['cap', 'of', '4', 'change', '19', 'done'])

    def test_edit_distance_order_sensitive(self):
        a = ['recall', 'surface', 'design']
        b = ['surface', 'recall', 'design']
        self.assertEqual(_token_edit_distance(a, b, 4), 2)

    def test_edit_distance_caps_out(self):
        self.assertEqual(
            _token_edit_distance(['a'] * 10, ['b'] * 10, 2), 3)


class TestConnectToCatalogTitleMatch(BrainTestBase):
    needs_embedder = False   # determinism proof: no vectors anywhere

    def _mk(self, title):
        return self.brain.remember(type='fact', title=title, content='c',
                                   encoding_source='anchor')['id']

    def _connect(self, title_ref):
        src = self._mk('source node for edge %s' % title_ref[:12])
        r = self.brain._apply_connect_to(
            src, [{'title': title_ref, 'relation': 'extends',
                   'why': 'test edge with enough length to pass'}],
            encoding_source='anchor')
        return r

    def test_exact_normalized_title_resolves_any_length(self):
        nid = self._mk('Daemon restart protocol')   # 3 tokens < MIN_TOKENS
        r = self._connect('daemon RESTART — protocol')  # dist 0 after normalize
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)

    def test_one_op_paraphrase_resolves_with_loud_log(self):
        # The Graph→Enrichment replica: one substituted token among many.
        nid = self._mk('Enrichment lane ceiling conversion: 19% of 52-gold '
                       'ceiling at current scoring')
        r = self._connect('Graph lane ceiling conversion: 19% of 52-gold '
                          'ceiling at current scoring')
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)
        warned = self.brain._logs_dal.conn.execute(
            "SELECT COUNT(*) FROM debug_log "
            "WHERE source='connect_to_near_title'").fetchone()[0]
        self.assertGreaterEqual(warned, 1)

    def test_two_op_paraphrase_resolves(self):
        nid = self._mk('Enrichment lane ceiling conversion nineteen of '
                       'fifty-two gold ceiling at current scoring')
        r = self._connect('Graph lane ceiling conversion nineteen of '
                          'fifty-two gold FLOOR at current scoring')
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)

    def test_three_op_paraphrase_drops(self):
        self._mk('Enrichment lane ceiling conversion nineteen of '
                 'fifty-two gold ceiling at current scoring')
        r = self._connect('Graph lane FLOOR conversion nineteen of '
                          'fifty-two gold BASEMENT at current scoring')
        self.assertEqual(r['created'], [])
        self.assertEqual(len(r['failed']), 1)

    def test_short_query_near_miss_drops(self):
        # 4 distinct tokens < NEAR_TITLE_MIN_TOKENS: near-title gate closed.
        self._mk('daemon restart protocol steps')
        r = self._connect('daemon reboot protocol steps')   # 1 op, too short
        self.assertEqual(r['created'], [])

    def test_ambiguous_tie_refuses(self):
        self._mk('recall pipeline stage one design for surface selection')
        self._mk('recall pipeline stage two design for surface selection')
        r = self._connect('recall pipeline stage six design for surface '
                          'selection')   # 1 op from BOTH
        self.assertEqual(r['created'], [])
        self.assertIn('ambiguous', r['failed'][0]['reason'])

    def test_photo_finish_inside_margin_refuses(self):
        # best=1, runner=2 → margin 1 < NEAR_TITLE_MARGIN → refuse.
        self._mk('recall pipeline stage one design for surface selection')
        self._mk('recall pipeline stage one design for surface projection')
        r = self._connect('recall pipeline stage six design for surface '
                          'selection')
        self.assertEqual(r['created'], [])
        self.assertIn('photo-finish', r['failed'][0]['reason'])

    def test_exact_match_ignores_near_twin(self):
        # A verbatim copy is unambiguous intent even when a 1-op twin exists.
        nid = self._mk('recall pipeline stage one design for surface selection')
        self._mk('recall pipeline stage two design for surface selection')
        r = self._connect('recall pipeline stage one design for surface '
                          'selection')
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)

    def test_duplicate_exact_titles_refuse(self):
        self._mk('duplicated exact title for ambiguity test case')
        self._mk('duplicated exact title for ambiguity test case')
        r = self._connect('duplicated exact title for ambiguity test case')
        self.assertEqual(r['created'], [])
        self.assertIn('ambiguous', r['failed'][0]['reason'])

    def test_id_pass_still_wins(self):
        nid = self._mk('a node referenced by id not title')
        r = self._connect(nid)
        self.assertEqual(len(r['created']), 1)
        self.assertEqual(r['created'][0]['target_id'], nid)

    def test_archived_nodes_invisible(self):
        nid = self._mk('archived target title that must never resolve here')
        self.brain.archive_node(nid, archived_by='test')
        r = self._connect('archived target title that must never resolve here')
        self.assertEqual(r['created'], [])


if __name__ == '__main__':
    unittest.main()
