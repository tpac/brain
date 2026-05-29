"""Step 6: AspectRegistry wired to Brain.__init__.

Tests cover the eager-validation + auto-heal path. Every fresh Brain ends up
with the 14 required aspect-nodes; the registry exposes them via attribute
access. Reverse lookups and surface helpers all work end-to-end.
"""

from tests.brain_test_base import BrainTestBase
from servers.aspects import REQUIRED_ASPECTS, Aspect


class TestAspectRegistryWired(BrainTestBase):
    """brain.aspects exists after Brain.__init__ + auto-heals required."""

    needs_embedder = False

    def test_all_14_required_present_after_auto_heal(self):
        # Fresh brain → empty type='aspect' → auto-heal seeds 14
        all_aspects = self.brain.aspects.all()
        for name in REQUIRED_ASPECTS:
            with self.subTest(aspect=name):
                self.assertIn(name, all_aspects,
                              '%s should be auto-healed on fresh brain' % name)

    def test_required_aspects_accessible_by_attribute(self):
        ib = self.brain.aspects.identity_bearing
        self.assertIsInstance(ib, Aspect)
        self.assertEqual(ib.name, 'identity_bearing')
        self.assertIn('principle', ib.node_types)

    def test_correction_improvement_has_both_slots(self):
        # The unification case
        ci = self.brain.aspects.correction_improvement
        self.assertIn('correction', ci.node_types)
        self.assertIn('corrects', ci.edge_relations)

    def test_required_aspects_locked(self):
        # Auto-heal seeds with locked=True via anchor:seed_aspects
        for name in REQUIRED_ASPECTS:
            with self.subTest(aspect=name):
                aspect = self.brain.aspects.by_name(name)
                self.assertTrue(aspect.locked,
                                '%s should be locked' % name)

    def test_reverse_lookups_work(self):
        # by_node_type
        principle_aspect = self.brain.aspects.by_node_type('principle')
        self.assertIsNotNone(principle_aspect)
        self.assertEqual(principle_aspect.name, 'identity_bearing')

        # by_edge_relation
        corrects_aspect = self.brain.aspects.by_edge_relation('corrects')
        self.assertIsNotNone(corrects_aspect)
        self.assertEqual(corrects_aspect.name, 'correction_improvement')

        # Missing → None
        self.assertIsNone(self.brain.aspects.by_node_type('not_a_type'))
        self.assertIsNone(self.brain.aspects.by_edge_relation('not_a_relation'))

    def test_unions_work(self):
        types = self.brain.aspects.types_in(['identity_bearing', 'episodic_anchor'])
        self.assertIn('principle', types)
        self.assertIn('moment', types)

        relations = self.brain.aspects.relations_in(['noise', 'generic_relation'])
        self.assertIn('co_accessed', relations)
        self.assertIn('related_to', relations)

    def test_surface_meaning_maps_populated(self):
        # relation_meaning_map for edge enrichment
        rmap = self.brain.aspects.relation_meaning_map()
        self.assertIn('corrects', rmap)
        self.assertGreater(len(rmap['corrects']), 30)

        # type_meaning_map for type enrichment
        tmap = self.brain.aspects.type_meaning_map()
        self.assertIn('principle', tmap)

    def test_dimensions_only_semantic_in_v1(self):
        self.assertEqual(self.brain.aspects.dimensions(), {'semantic'})

    def test_required_emergent_partition(self):
        # required() + emergent() partition all() — disjoint, complete
        all_keys = set(self.brain.aspects.all().keys())
        required_keys = set(self.brain.aspects.required().keys())
        emergent_keys = set(self.brain.aspects.emergent().keys())
        self.assertEqual(required_keys | emergent_keys, all_keys)
        self.assertEqual(required_keys & emergent_keys, set())

    def test_all_with_counts_shape(self):
        result = self.brain.aspects.all_with_counts()
        self.assertEqual(len(result), 14)
        for entry in result:
            self.assertIn('name', entry)
            self.assertIn('node_types_count', entry)
            self.assertIn('edge_relations_count', entry)
            self.assertIn('dimension', entry)
            self.assertIn('locked', entry)


class TestAspectRegistryIdempotentAcrossBrains(BrainTestBase):
    """A second Brain on the same DB sees existing aspects, doesn't re-seed."""

    needs_embedder = False

    def test_second_brain_loads_existing_aspects(self):
        # First brain auto-heals
        first_count = len(self.brain.aspects.all())
        self.assertEqual(first_count, 14)

        # Second brain on same DB should LOAD, not re-create
        from servers.brain import Brain
        second_brain = Brain(self.db_path, skip_embedder=True)
        second_count = len(second_brain.aspects.all())
        self.assertEqual(second_count, 14,
                         'second brain should see existing aspects, not double-seed')


class TestAspectRegistryHealerDisplayLabel(BrainTestBase):
    """Healer-facing aspects carry display_label in metadata."""

    needs_embedder = False

    def test_display_label_round_trips_through_load(self):
        ci = self.brain.aspects.correction_improvement
        self.assertEqual(ci.metadata.get('display_label'), 'corrects/improves')

        er = self.brain.aspects.extension_refinement
        self.assertEqual(er.metadata.get('display_label'), 'extends/refines')
