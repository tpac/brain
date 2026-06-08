"""Tests for the S2 Aspect Encoder's classification validation.

Focus: the noise-exclusivity boundary (2026-06-08). `noise` means "no
semantic claim"; if the encoder dual-routes a string to noise AND a real
aspect, the guard strips noise and keeps the meaning. That preserves the
invariant noise ∩ {any other aspect} = ∅, which is what lets downstream
exclusion filters trust "not in noise" == "is real knowledge".

These exercise _validate_classifications directly with a stub brain — no
LLM, no daemon, no real aspect data needed (validation reads ASPECT_ACCEPTS,
a module constant, not the seed file).
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.scales.s2.aspect_encoder import AspectEncoder


class _StubBrain:
    def __init__(self):
        self.errors = []

    def _log_error(self, name, exc, msg):
        self.errors.append((str(exc), msg))


def _encoder():
    # Bypass __init__: _validate_classifications only touches self.brain and
    # the class-level self.NAME — no dispatch_fn / config needed.
    enc = AspectEncoder.__new__(AspectEncoder)
    enc.brain = _StubBrain()
    return enc


def _props(*values):
    return [{'category': 'edge_relations', 'value': v, 'count': 3, 'examples': []}
            for v in values]


class TestNoiseExclusivityGuard(unittest.TestCase):

    def test_noise_stripped_when_dual_routed_with_semantic(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'foo_rel',
                'aspects': ['noise', 'correction_improvement']}]
        accepted, rejected = enc._validate_classifications(cls, _props('foo_rel'), {})
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['aspects'], ['correction_improvement'])

    def test_noise_stripped_when_listed_secondary(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'baz_rel',
                'aspects': ['correction_improvement', 'noise']}]
        accepted, _ = enc._validate_classifications(cls, _props('baz_rel'), {})
        self.assertEqual(accepted[0]['aspects'], ['correction_improvement'])

    def test_pure_noise_preserved(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'bar_rel', 'aspects': ['noise']}]
        accepted, _ = enc._validate_classifications(cls, _props('bar_rel'), {})
        self.assertEqual(accepted[0]['aspects'], ['noise'])

    def test_two_semantic_aspects_untouched(self):
        # The guard targets noise only — legitimate semantic multi-membership
        # (e.g. dependency_flow + temporal_sequence) must pass through intact.
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'qux_rel',
                'aspects': ['dependency_flow', 'temporal_sequence']}]
        accepted, _ = enc._validate_classifications(cls, _props('qux_rel'), {})
        self.assertEqual(accepted[0]['aspects'], ['dependency_flow', 'temporal_sequence'])

    def test_guard_logs_loud(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'foo_rel',
                'aspects': ['noise', 'correction_improvement']}]
        enc._validate_classifications(cls, _props('foo_rel'), {})
        self.assertTrue(
            any('stripped noise' in e[0] for e in enc.brain.errors),
            "noise-strip should log loudly to the brain errors table")


if __name__ == '__main__':
    unittest.main()
