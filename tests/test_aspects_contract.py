"""Contract test — locks the equivalence between REQUIRED_ASPECTS and aspects_v1.json.

The contract: every name in REQUIRED_ASPECTS must appear as a key in
aspects_v1.json, and vice versa. Adding a name to one without the other is a
failure — the test catches drift before it reaches a fresh brain.

Also verifies the seed JSON parses cleanly via AspectRegistry.from_dict — so
the in-memory shape stays in sync with the seed shape.

Step 4 of 14 in the unified-aspects work.
"""

import json
import os
import unittest

from servers.aspects import REQUIRED_ASPECTS, AspectRegistry, Aspect


SEED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'servers', 'scales', 's2', 'aspects_v1.json'
)


def _load_seed():
    with open(SEED_PATH) as f:
        return json.load(f)


class TestSeedExists(unittest.TestCase):
    def test_seed_file_exists(self):
        self.assertTrue(os.path.isfile(SEED_PATH),
                        "aspects_v1.json must exist at %s" % SEED_PATH)

    def test_seed_parses_as_json(self):
        # raises if not valid JSON
        data = _load_seed()
        self.assertIsInstance(data, dict)


class TestContractEquivalence(unittest.TestCase):
    """REQUIRED_ASPECTS keys ≡ aspects_v1.json keys."""

    def setUp(self):
        self.seed = _load_seed()
        self.required = set(REQUIRED_ASPECTS)
        self.seeded = set(self.seed.keys())

    def test_no_required_aspect_missing_from_seed(self):
        missing = self.required - self.seeded
        self.assertEqual(
            missing, set(),
            "REQUIRED_ASPECTS contains names absent from aspects_v1.json: %s" % missing
        )

    def test_no_extra_aspect_in_seed(self):
        # Seed should be EXACTLY the required set in v1. Emergent aspects
        # are created by AspectIntegration after boot, not seeded.
        extras = self.seeded - self.required
        self.assertEqual(
            extras, set(),
            "aspects_v1.json contains names not in REQUIRED_ASPECTS: %s" % extras
        )
        # Consolidated from the former test_set_equivalence: with no missing and
        # no extra names, the sets are equal — assert it directly here.
        self.assertEqual(self.required, self.seeded)


class TestSeedShape(unittest.TestCase):
    """Each seed entry has the structural keys AspectRegistry.from_dict expects."""

    def setUp(self):
        self.seed = _load_seed()

    def test_each_entry_has_required_fields(self):
        required_fields = {'node_types', 'edge_relations', 'meaning',
                           'dimension', 'locked', 'metadata'}
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                missing = required_fields - set(spec.keys())
                self.assertEqual(
                    missing, set(),
                    "%s missing required fields: %s" % (name, missing)
                )

    def test_node_types_and_edge_relations_are_lists(self):
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                self.assertIsInstance(spec['node_types'], list)
                self.assertIsInstance(spec['edge_relations'], list)

    def test_no_aspect_is_empty(self):
        # Required aspects must have at least one member somewhere — otherwise
        # they're a husk. Emergent aspects (when they appear later) may be
        # empty transiently but required ones are seeded with members.
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                has_members = bool(spec['node_types']) or bool(spec['edge_relations'])
                self.assertTrue(
                    has_members,
                    "Required aspect '%s' has no node_types AND no edge_relations — empty husk" % name
                )

    def test_all_seeded_aspects_are_locked(self):
        # Required aspects ship locked=True so the maintenance unit can't
        # remove or rename them
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                self.assertTrue(
                    spec['locked'],
                    "Required aspect '%s' must be locked=True in seed" % name
                )

    def test_meaning_is_substantive(self):
        # Meaning is embedded for recall — empty/short strings hurt retrieval.
        # Require at least 30 chars (sentence-ish).
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                self.assertGreaterEqual(
                    len(spec['meaning']), 30,
                    "%s.meaning is too short (<30 chars) — won't embed usefully" % name
                )

    def test_dimension_is_known(self):
        # v1 only has 'semantic'. Future dimensions emerge via AspectIntegration.
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                self.assertEqual(
                    spec['dimension'], 'semantic',
                    "v1 seed only supports dimension='semantic'; got '%s' for %s" % (
                        spec['dimension'], name)
                )


class TestSeedLoadsViaRegistry(unittest.TestCase):
    """The seed parses cleanly through AspectRegistry.from_dict."""

    def setUp(self):
        self.seed = _load_seed()
        self.registry = AspectRegistry.from_dict(brain=None, data=self.seed)

    def test_all_aspects_present(self):
        self.assertEqual(set(self.registry.all().keys()), set(self.seed.keys()))

    def test_required_attributes_resolve(self):
        # Every name in REQUIRED_ASPECTS resolves via attribute access
        for name in REQUIRED_ASPECTS:
            with self.subTest(aspect=name):
                aspect = getattr(self.registry, name)
                self.assertIsInstance(aspect, Aspect)
                self.assertEqual(aspect.name, name)

    def test_required_member_lists_non_empty(self):
        # Round-trip: registry's view of each required aspect should match
        # the seed's view (members, meaning).
        for name in REQUIRED_ASPECTS:
            with self.subTest(aspect=name):
                aspect = self.registry.by_name(name)
                seed_entry = self.seed[name]
                self.assertEqual(
                    list(aspect.node_types), seed_entry['node_types'],
                    "%s.node_types mismatch" % name
                )
                self.assertEqual(
                    list(aspect.edge_relations), seed_entry['edge_relations'],
                    "%s.edge_relations mismatch" % name
                )
                self.assertEqual(aspect.meaning, seed_entry['meaning'])
                self.assertEqual(aspect.dimension, seed_entry['dimension'])
                self.assertEqual(aspect.locked, seed_entry['locked'])

    def test_correction_improvement_has_both_sides(self):
        # The natural unification case — correction_improvement carries BOTH
        # node_types (from old correction_supersession) and edge_relations
        # (from old correction_improvement).
        a = self.registry.correction_improvement
        self.assertGreater(len(a.node_types), 0,
                           "correction_improvement should have node_types after unification")
        self.assertGreater(len(a.edge_relations), 0,
                           "correction_improvement should have edge_relations after unification")
        # specific markers
        self.assertIn('correction', a.node_types)
        self.assertIn('corrects', a.edge_relations)

    def test_healer_display_labels_present(self):
        # All 8 healer-facing aspects should carry display_label metadata
        healer_aspects = (
            'correction_improvement', 'extension_refinement', 'explanation_causation',
            'dependency_flow', 'contradiction_conflict', 'validation_evidence',
            'hierarchical_structure', 'temporal_sequence',
        )
        for name in healer_aspects:
            with self.subTest(aspect=name):
                aspect = self.registry.by_name(name)
                self.assertIn('display_label', aspect.metadata,
                              "%s should have metadata.display_label for healer" % name)
                self.assertGreater(len(aspect.metadata['display_label']), 0)


class TestMultiMembershipShape(unittest.TestCase):
    """Multi-membership is allowed (a string can appear in 2+ aspects).
    Reverse lookup remains deterministic — the FIRST aspect to list a
    string in JSON iteration order wins for `by_node_type`/`by_edge_relation`.

    Was previously TestNoMemberOverlap, which enforced single-membership.
    Single-membership was dropped 2026-05-08 — recall expressivity (one
    edge serving multiple aspect-filtered queries) beats reverse-lookup
    determinism. The deterministic-lookup property is now achieved by
    JSON-iteration-order convention rather than uniqueness.
    """

    def setUp(self):
        self.seed = _load_seed()

    def test_member_strings_are_well_formed(self):
        # Lists of strings, no nested structure, no empty strings
        for name, spec in self.seed.items():
            for t in spec['node_types']:
                self.assertIsInstance(t, str, '%s node_type entry not a string' % name)
                self.assertTrue(t, '%s has empty node_type entry' % name)
            for r in spec['edge_relations']:
                self.assertIsInstance(r, str, '%s edge_relation entry not a string' % name)
                self.assertTrue(r, '%s has empty edge_relation entry' % name)

    def test_reverse_lookup_resolves_deterministically(self):
        # If a string is in N aspects, by_X should still resolve to ONE.
        # The contract: first aspect to claim it (JSON iteration order) wins.
        from servers.aspects import AspectRegistry
        registry = AspectRegistry.from_dict(brain=None, data=self.seed)
        for name, spec in self.seed.items():
            for t in spec['node_types']:
                resolved = registry.by_node_type(t)
                self.assertIsNotNone(resolved, 'by_node_type(%s) returned None' % t)
            for r in spec['edge_relations']:
                resolved = registry.by_edge_relation(r)
                self.assertIsNotNone(resolved, 'by_edge_relation(%s) returned None' % r)


if __name__ == '__main__':
    unittest.main()
