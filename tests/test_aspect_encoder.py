"""Tests for the S2 Aspect Encoder's classification validation.

Focus: the noise-exclusivity boundary (2026-06-08). `noise` means "no
semantic claim"; if the encoder dual-routes a string to noise AND a real
aspect, the guard strips noise and keeps the meaning. That preserves the
invariant noise ∩ {any other aspect} = ∅, which is what lets downstream
exclusion filters trust "not in noise" == "is real knowledge".

These exercise _validate_classifications directly with a stub brain — no
LLM, no daemon. Validation derives its closed list + accepts-categories from
brain.aspects (the per-aspect `routable`/`accepts` facts, Step 4), so the
stub carries a real registry built from the repo seed.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.aspect_store import SEED_ASPECTS_JSON_PATH
from servers.aspects import AspectRegistry
from servers.scales.s2.aspect_encoder import AspectEncoder

with open(SEED_ASPECTS_JSON_PATH) as _f:
    _SEED_REGISTRY = AspectRegistry.from_dict(None, json.load(_f))


class _StubBrain:
    aspects = _SEED_REGISTRY

    def __init__(self):
        self.errors = []
        self.warnings = []

    def _log_error(self, name, exc, msg):
        self.errors.append((str(exc), msg))

    def _log_warning(self, name, message, context=''):
        self.warnings.append((message, context))


def _encoder():
    # Bypass __init__: _validate_classifications only touches self.brain and
    # the class-level self.NAME — no dispatch_fn / config needed.
    enc = AspectEncoder.__new__(AspectEncoder)
    enc.brain = _StubBrain()
    return enc


def _props(*values):
    return [{'category': 'edge_relations', 'value': v, 'count': 3, 'examples': []}
            for v in values]


class TestMenuDerivation(unittest.TestCase):
    """The ASPECT MENU derives from the per-aspect facts (routable / accepts /
    prompt_visible) instead of the deleted ASPECT_ACCEPTS + `order` literals.
    Pins that the derivation reproduces the hand-curated menu this replaced —
    verified byte-identical against the pre-Step-4 prompt when it shipped."""

    def setUp(self):
        self.enc = _encoder()
        self.prompt = self.enc._format_prompt(
            _SEED_REGISTRY.all(),
            [{'category': 'edge_relations', 'value': 'x', 'count': 1,
              'examples': []}])

    def test_menu_order_matches_the_replaced_literal(self):
        expected = [
            'identity_bearing', 'episodic_anchor', 'active_thread',
            'lesson_insight', 'wisdom',
            'correction_improvement',
            'extension_refinement', 'explanation_causation', 'dependency_flow',
            'contradiction_conflict', 'validation_evidence',
            'hierarchical_structure', 'temporal_sequence',
            'generic_relation', 'noise',
        ]
        shown = [line[3:-3].strip() for line in self.prompt.splitlines()
                 if line.startswith('── ') and line.endswith(' ──')
                 and not line.startswith('── #')]
        self.assertEqual(shown, expected)

    def test_menu_header_counts_routable_aspects(self):
        self.assertIn('ASPECT MENU — 15 aspects', self.prompt)

    def test_non_routable_aspect_never_offered(self):
        self.assertNotIn('survivor_lineage', self.prompt)


class TestNoiseDualRouting(unittest.TestCase):
    """noise + semantic is a LEGAL classification, kept as returned.
    This boundary used to strip noise on the reasoning that a
    semantic claim refutes it — which deleted the half that does the
    filtering, silently un-noising machinery the classifier had flagged.
    Noise vetoes at read time instead; here we only record and preserve.
    """

    def test_both_kept_when_dual_routed_with_semantic(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'foo_rel',
                'aspects': ['noise', 'correction_improvement']}]
        accepted, rejected = enc._validate_classifications(cls, _props('foo_rel'))
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['aspects'],
                         ['noise', 'correction_improvement'])

    def test_both_kept_when_noise_listed_secondary(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'baz_rel',
                'aspects': ['correction_improvement', 'noise']}]
        accepted, _ = enc._validate_classifications(cls, _props('baz_rel'))
        self.assertEqual(accepted[0]['aspects'],
                         ['correction_improvement', 'noise'])

    def test_pure_noise_preserved(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'bar_rel', 'aspects': ['noise']}]
        accepted, _ = enc._validate_classifications(cls, _props('bar_rel'))
        self.assertEqual(accepted[0]['aspects'], ['noise'])

    def test_two_semantic_aspects_untouched(self):
        # The guard targets noise only — legitimate semantic multi-membership
        # (e.g. dependency_flow + temporal_sequence) must pass through intact.
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'qux_rel',
                'aspects': ['dependency_flow', 'temporal_sequence']}]
        accepted, _ = enc._validate_classifications(cls, _props('qux_rel'))
        self.assertEqual(accepted[0]['aspects'], ['dependency_flow', 'temporal_sequence'])

    def test_dual_route_recorded_as_warning_not_error(self):
        """Visibility without a false alarm: dual routing is legal now, so it
        must not land in the errors table — but it stays recorded, since it is
        the only signal telling us whether the classifier uses it at all."""
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'foo_rel',
                'aspects': ['noise', 'correction_improvement']}]
        enc._validate_classifications(cls, _props('foo_rel'))
        self.assertTrue(any('noise vetoes on read' in m
                            for m, _ctx in enc.brain.warnings),
                        enc.brain.warnings)
        self.assertEqual(enc.brain.errors, [])


if __name__ == '__main__':
    unittest.main()
