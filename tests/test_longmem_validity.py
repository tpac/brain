"""Contract tests for the eval measurement-validity layer.

Two contracts that must not drift:
  1. suspect_reasons is the ONE vocabulary for per-rep suspect marking —
     table-tested so a renamed label breaks here, not silently in a report.
  2. The analyzer's gold-scan basis agrees with the classifier's own
     scannability gates (term extractor fallback + shared phrase constant) —
     the two modules once disagreed on golds like "220" ("encoder gap" in
     one, found in the other).
No brain needed — pure functions.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.longmem.validity import suspect_reasons, RECALL_FAILURE_MODES
from eval.longmem.analyzer import (_gold_scan_basis, _gold_scan_terms,
                                   _find_gold_bearing_nodes)
from eval.longmem.classifier import _extract_key_terms, PHRASE_SCAN_MIN_CHARS


class TestSuspectReasons(unittest.TestCase):
    def test_clean_rep_is_valid(self):
        self.assertEqual(suspect_reasons(), [])

    def test_each_signal_maps_to_its_label(self):
        cases = [
            (dict(recall_mode='embed_failed'),
             ['recall_degraded:embed_failed']),
            (dict(recall_mode='embedder_unavailable'),
             ['recall_degraded:embedder_unavailable']),
            (dict(recall_mode='laf_v1'), []),  # healthy mode, not suspect
            (dict(new_errors=[{'type': 'x_failed'}]), ['brain_error:x_failed']),
            (dict(new_errors=[{'source': 's1e'}]), ['brain_error:s1e']),
            (dict(answerer_error='529'), ['answerer_error']),
            (dict(judge_parse_failed=True), ['judge_parse_failed']),
            (dict(harness_error='boom'), ['harness_error']),
        ]
        for kwargs, want in cases:
            self.assertEqual(suspect_reasons(**kwargs), want, kwargs)

    def test_signals_compose(self):
        got = suspect_reasons(recall_mode='embed_failed', harness_error='x',
                              answerer_error='y', judge_parse_failed=True,
                              new_errors=[{'type': 't'}])
        self.assertEqual(len(got), 5)

    def test_failure_modes_are_the_owners(self):
        # The tuple must come from servers.brain_recall — a local copy is
        # exactly the drift this module exists to prevent.
        from servers.brain_recall import RECALL_FAILURE_MODES as OWNER
        self.assertIs(RECALL_FAILURE_MODES, OWNER)


class TestGoldScanParity(unittest.TestCase):
    """The analyzer must be able to scan every gold the classifier can scan
    (and refuse the ones it refuses)."""

    def _classifier_scannable(self, gold: str) -> bool:
        terms = _extract_key_terms(gold)
        if len(terms) == 1 and len(terms[0]) < 2:
            terms = []
        phrase = len(gold) >= PHRASE_SCAN_MIN_CHARS
        return bool(terms or phrase)

    def test_parity_on_representative_golds(self):
        golds = ['220', '22', '1.5', 'AI', 'USD', '6 PM', '3', 'no',
                 'a walking tour of Kyoto temples', '3200 yen', 'x']
        for g in golds:
            analyzer_scannable = _gold_scan_basis(g) != 'unscannable'
            self.assertEqual(analyzer_scannable, self._classifier_scannable(g),
                             'scannability drift on gold %r' % g)

    def test_digit_gold_scans_by_terms(self):
        # "220" was the headline false "encoder gap": zero >=4-char terms.
        self.assertEqual(_gold_scan_basis('220'), 'terms')
        self.assertEqual(_gold_scan_terms('220'), ['220'])
        nodes = [{'id': 'n1' * 4, 'title': 'pace', 'content': 'reads 220 pages',
                  'kv': {}}]
        self.assertTrue(_find_gold_bearing_nodes(nodes, '220'))

    def test_short_phrase_never_substring_matches(self):
        # "AI" must not hit 'explain'; unscannable, not a verdict.
        self.assertEqual(_gold_scan_basis(''), 'unscannable')
        nodes = [{'id': 'n1' * 4, 'title': 'explain the plan', 'content': '',
                  'kv': {}}]
        self.assertEqual(_find_gold_bearing_nodes(nodes, ''), [])

    def test_voice_alias_fields_searched(self):
        nodes = [{'id': 'n1' * 4, 'title': 't', 'content': '',
                  'kv': {'anchor_raw_quote': 'the fare was 3200 yen'}}]
        self.assertTrue(_find_gold_bearing_nodes(nodes, '3200 yen fare'))


if __name__ == '__main__':
    unittest.main()
