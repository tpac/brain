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
    """Seed WITHOUT the '_'-prefixed documentation keys (_schema) — the
    aspect entries every curation test iterates. Use _load_seed_raw for the
    file as-is."""
    return {k: v for k, v in _load_seed_raw().items() if not k.startswith('_')}


def _load_seed_raw():
    with open(SEED_PATH) as f:
        return json.load(f)


class TestSchemaDocEntry(unittest.TestCase):
    """The `_schema` entry is the file's own documentation (JSON has no
    comments; '_'-prefixed keys are the reserved convention, skipped by the
    registry loader, the validator, the write door, and the dashboard).
    It must document every key an aspect entry carries — a new field without
    an explainer fails here."""

    def test_schema_documents_every_entry_key(self):
        raw = _load_seed_raw()
        self.assertIn('_schema', raw)
        doc = raw['_schema']
        entry_keys = set()
        for name, spec in raw.items():
            if not name.startswith('_'):
                entry_keys |= set(spec.keys())
        undocumented = entry_keys - set(doc.keys())
        self.assertEqual(
            undocumented, set(),
            'aspect entry keys missing a _schema explainer: %s' % undocumented)

    def test_schema_is_skipped_by_the_registry(self):
        from servers.aspects import AspectRegistry
        registry = AspectRegistry.from_dict(None, _load_seed_raw())
        self.assertNotIn('_schema', registry.all())

    def test_schema_passes_the_write_gate(self):
        from servers.aspect_store import validate_taxonomy
        self.assertEqual(validate_taxonomy(_load_seed_raw()), [])


# Verbs that REPLACE prior knowledge. Every one must live in
# correction_improvement, because `brain.correction_enrich()` walks that aspect
# alone — a replacement verb missing from it means superseded nodes surface with
# no pointer to their successor. Shared with
# tests/test_surface_transitions.py::test_correction_enrich_walks_every_replacement_verb
# so the membership contract and the end-to-end behaviour cover the same list.
REPLACEMENT_VERBS = ('supersedes', 'superseded_by', 'absorbed_into',
                     'consolidated_into')


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
                           'dimension', 'locked', 'metadata',
                           'accepts', 'routable', 'prompt_visible',
                           'structural_lineage'}
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                missing = required_fields - set(spec.keys())
                self.assertEqual(
                    missing, set(),
                    "%s missing required fields: %s" % (name, missing)
                )

    def test_accepts_is_nonempty_and_valid(self):
        # This assertion is what makes a JSON-only aspect addition safe: an
        # entry cannot ship without declaring which categories it takes.
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                accepts = spec['accepts']
                self.assertTrue(accepts, "%s.accepts is empty" % name)
                self.assertTrue(
                    set(accepts) <= {'node_types', 'edge_relations'},
                    "%s.accepts has unknown categories: %s" % (name, accepts))

    def test_fact_flags_are_booleans(self):
        for name, spec in self.seed.items():
            for flag in ('routable', 'prompt_visible', 'structural_lineage'):
                with self.subTest(aspect=name, flag=flag):
                    self.assertIsInstance(spec[flag], bool)

    def test_non_routable_set_is_the_system_owned_pair(self):
        # survivor_lineage is system-written (absorbed_into), settlement is
        # the closed consolidation-suppression list — neither is ever offered
        # to the classifier. Everything else routes. A new non-routable
        # aspect is a deliberate edit — update this pin consciously.
        non_routable = {name for name, spec in self.seed.items()
                        if not spec['routable']}
        self.assertEqual(non_routable, {'survivor_lineage', 'settlement'})

    def test_prompt_invisible_set_is_pinned(self):
        # The catch-alls + the system aspects stay out of encoder vocabulary
        # blocks (the deleted EDGE_ASPECT_PROMPT_SKIP contract, now declared).
        # settlement's verbs already appear via their semantic home aspects —
        # offering them again as a "settlement" block would double-list them.
        invisible = {name for name, spec in self.seed.items()
                     if not spec['prompt_visible']}
        self.assertEqual(
            invisible,
            {'generic_relation', 'noise', 'survivor_lineage', 'settlement'})

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

    def test_replacement_verbs_are_correction_class(self):
        """`supersedes`/`superseded_by` must be in correction_improvement.

        `brain.correction_enrich()` walks correction_improvement ALONE, so a
        replacement verb missing from it means every superseded node surfaces
        with no pointer to its successor — the exact bug the seed shipped until
        2026-07-24 (332 supersedes edges, none reaching recall). They stay
        multi-homed in hierarchical_structure too; this pins the correction
        half, which is the one recall depends on.
        """
        rels = self.registry.correction_improvement.edge_relations
        for verb in REPLACEMENT_VERBS:
            with self.subTest(verb=verb):
                self.assertIn(
                    verb, rels,
                    "%s is a replacement verb — correction_enrich() must see it"
                    % verb)
        # Still structurally homed as well — multi-membership, not a move.
        self.assertIn('supersedes',
                      self.registry.hierarchical_structure.edge_relations)

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


class TestWisdomCuration(unittest.TestCase):
    """The wisdom aspect stays curated: generative types in, operational and
    tactical record-keeping out. (Moved from test_frame.py 2026-08-05 when
    the Frame stopped consuming the aspect — the curation contract outlives
    that consumer: encoder routing and scope policy still read it.)"""

    def setUp(self):
        self.aspects = AspectRegistry.from_dict(brain=None, data=_load_seed())

    def test_wisdom_includes_generative_types(self):
        members = self.aspects.wisdom.node_types
        for t in ('insight', 'lesson', 'principle', 'vision'):
            self.assertIn(t, members)

    def test_wisdom_excludes_operational_and_tactical(self):
        members = self.aspects.wisdom.node_types
        for t in ('rule', 'operator', 'decision', 'fact', 'bug', 'mechanism'):
            self.assertNotIn(t, members)


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
        # Pin FIRST-claimant on real multi-homed members — this is what caught
        # the from_dict/_load disagreement (last-claimant reported
        # hierarchical_structure / survivor_lineage for these instead).
        self.assertEqual(registry.by_edge_relation('supersedes').name,
                         'correction_improvement')
        self.assertEqual(registry.by_edge_relation('absorbed_into').name,
                         'correction_improvement')
        # And per-string agreement between the two construction paths, whole
        # seed: from_dict must resolve every string exactly like a file-loaded
        # registry would (one _adopt body — this pins that it stays one).
        for t, primary in registry._reverse_node.items():
            self.assertEqual(primary, next(
                n for n, s in self.seed.items() if t in s['node_types']),
                'first-claimant violated for node type %s' % t)
        for r, primary in registry._reverse_edge.items():
            self.assertEqual(primary, next(
                n for n, s in self.seed.items() if r in s['edge_relations']),
                'first-claimant violated for edge relation %s' % r)


class TestLineageDerivation(unittest.TestCase):
    """Spread activation's lineage ride-along derives from the per-aspect
    `structural_lineage` fact in aspects_v1.json (Step 4) — the deleted
    LINEAGE_FAMILIES literal is the drift class this replaces (the 2026-06-08
    bug: 5 of 8 hardcoded family names were dead, so dependency_flow +
    multi-membership lineage silently stopped riding along). Derivation means
    the names cannot go stale; these tests pin the seed's deliberate flag
    choices and that the derived union is live.
    """

    def setUp(self):
        self.seed = _load_seed()
        from servers.aspects import AspectRegistry
        self.registry = AspectRegistry.from_dict(None, self.seed)

    def test_seed_flags_exactly_the_four_lineage_aspects(self):
        flagged = {name for name, spec in self.seed.items()
                   if spec.get('structural_lineage')}
        self.assertEqual(
            flagged,
            {'correction_improvement', 'extension_refinement',
             'hierarchical_structure', 'dependency_flow'},
            "seed structural_lineage flags changed — deliberate edit only "
            "(this set drives spread-activation ride-along)")

    def test_lineage_relations_derived_and_nonempty(self):
        self.assertTrue(
            self.registry.lineage_relations,
            "derived lineage_relations is empty — ride-along silently dead")

    def test_lineage_aspects_carry_edge_relations(self):
        # A flagged aspect with no edge_relations contributes nothing to the
        # ride-along union — almost certainly a wrong flag.
        for name, spec in self.seed.items():
            if spec.get('structural_lineage'):
                self.assertTrue(
                    spec.get('edge_relations'),
                    "lineage aspect %r has no edge_relations" % name)

    def test_union_covers_every_flagged_aspect(self):
        for name, spec in self.seed.items():
            if spec.get('structural_lineage'):
                for r in spec['edge_relations']:
                    self.assertIn(r, self.registry.lineage_relations)


class TestStructuralValidity(unittest.TestCase):
    """The seed satisfies the FULL structural invariant set — one call.

    validate_taxonomy is the single home of the structural rules (entries are
    objects; member lists are unique non-empty strings; all required aspects
    present; noise ∩ semantic = ∅). The registry's write door refuses on it
    and the boot load reports on it, so pinning the seed against the same
    function means the shipped baseline can never fail the gate a write is
    held to. Replaces the hand-rolled lists-shape / well-formed-members /
    noise-exclusivity tests, which each re-implemented one slice of it.

    Seed-only CURATION standards (locked, meaning length, display labels,
    no-extra-aspects) deliberately stay as separate tests above — the working
    copy legitimately grows emergent unlocked aspects, so those must never
    enter the write gate.
    """

    def setUp(self):
        self.seed = _load_seed()

    def test_seed_passes_the_write_gate(self):
        from servers.aspect_store import validate_taxonomy
        self.assertEqual(validate_taxonomy(self.seed), [])

    def test_validator_catches_noise_overlap(self):
        # The 2026-07-24 class of defect: a noise member that also carries a
        # semantic claim. The validator must name both sides.
        from servers.aspect_store import validate_taxonomy
        import copy
        broken = copy.deepcopy(self.seed)
        broken['noise']['edge_relations'].append('corrects')
        violations = validate_taxonomy(broken)
        self.assertTrue(any('noise-exclusivity' in v and 'corrects' in v
                            for v in violations), violations)

    def test_validator_catches_malformed_and_duplicate_members(self):
        from servers.aspect_store import validate_taxonomy
        import copy
        broken = copy.deepcopy(self.seed)
        broken['correction_improvement']['edge_relations'].append('')
        broken['correction_improvement']['edge_relations'].append('corrects')
        violations = validate_taxonomy(broken)
        self.assertTrue(any('malformed member' in v for v in violations), violations)
        self.assertTrue(any("duplicate member 'corrects'" in v for v in violations),
                        violations)

    def test_validator_catches_missing_required_aspect(self):
        from servers.aspect_store import validate_taxonomy
        broken = {k: v for k, v in self.seed.items() if k != 'survivor_lineage'}
        violations = validate_taxonomy(broken)
        self.assertTrue(any('survivor_lineage' in v for v in violations), violations)

    def test_validator_catches_malformed_fact_fields(self):
        # Invariant 5: accepts must be a non-empty unique subset of the two
        # categories; the three flags must be booleans — when present.
        # Absence is legitimate (emergent aspects, pre-heal working copies).
        from servers.aspect_store import validate_taxonomy
        import copy
        broken = copy.deepcopy(self.seed)
        broken['noise']['accepts'] = []
        broken['wisdom']['accepts'] = ['node_types', 'everything']
        broken['lesson_insight']['routable'] = 'yes'
        violations = validate_taxonomy(broken)
        self.assertTrue(any("'noise'.accepts" in v for v in violations), violations)
        self.assertTrue(any("'wisdom'.accepts" in v for v in violations), violations)
        self.assertTrue(any("'lesson_insight'.routable" in v for v in violations),
                        violations)

    def test_validator_allows_absent_fact_fields(self):
        # A pre-Step-4 working copy (no fact keys anywhere) must still pass
        # the structural gate — presence is seed curation, not structure.
        from servers.aspect_store import validate_taxonomy
        import copy
        legacy = copy.deepcopy(self.seed)
        for spec in legacy.values():
            for k in ('accepts', 'routable', 'prompt_visible',
                      'structural_lineage'):
                spec.pop(k, None)
        self.assertEqual(validate_taxonomy(legacy), [])


if __name__ == '__main__':
    unittest.main()
