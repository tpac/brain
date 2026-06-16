"""Tests for the Aspect value object + AspectRegistry skeleton.

Covers Step 2 scope: data shapes, from_dict construction, attribute access,
reverse lookups, cross-aspect unions, discovery/enumeration, surface helpers.

The brain-loading + validation paths (_load + _validate) are stubbed in
Step 2 — exercised by Step 6 tests when wired to Brain.
"""

import unittest

from servers.aspects import (
    Aspect, AspectRegistry, AspectContractError, REQUIRED_ASPECTS,
)


# Sample data — mirrors aspects_v1.json shape (Step 3 will write the real file)
SAMPLE_DATA = {
    'identity_bearing': {
        'node_types': ['principle', 'identity', 'rule'],
        'edge_relations': [],
        'meaning': 'Nodes that anchor identity',
        'dimension': 'semantic',
        'locked': True,
        'metadata': {},
    },
    'correction_improvement': {
        'node_types': ['correction', 'bug_lesson'],
        'edge_relations': ['corrects', 'supersedes'],
        'meaning': 'Knowledge that updates prior knowledge',
        'dimension': 'semantic',
        'locked': True,
        'metadata': {'display_label': 'corrects/improves'},
    },
    'noise': {
        'node_types': [],
        'edge_relations': ['co_accessed', 'emergent_bridge'],
        'meaning': 'Structural edges, not semantic claims',
        'dimension': 'semantic',
        'locked': True,
        'metadata': {},
    },
    'emergent_xyz': {
        'node_types': ['hypothesis_v2'],
        'edge_relations': [],
        'meaning': 'An emergent aspect for testing',
        'dimension': 'semantic',
        'locked': False,
        'metadata': {},
    },
}


class TestAspect(unittest.TestCase):
    """The Aspect value object — frozen dataclass."""

    def test_construction_and_defaults(self):
        a = Aspect(name='x')
        self.assertEqual(a.name, 'x')
        self.assertEqual(a.node_types, ())
        self.assertEqual(a.edge_relations, ())
        self.assertEqual(a.meaning, '')
        self.assertEqual(a.dimension, 'semantic')
        self.assertFalse(a.locked)
        self.assertEqual(a.metadata, {})

    def test_frozen_blocks_field_reassignment(self):
        a = Aspect(name='x')
        with self.assertRaises(Exception):
            a.name = 'y'

    def test_contains(self):
        a = Aspect(name='x', node_types=('p', 'q'), edge_relations=('r',))
        self.assertIn('p', a)
        self.assertIn('r', a)
        self.assertNotIn('z', a)

    def test_shape_helpers(self):
        node_only = Aspect(name='x', node_types=('p',))
        edge_only = Aspect(name='y', edge_relations=('r',))
        both = Aspect(name='z', node_types=('p',), edge_relations=('r',))
        empty = Aspect(name='e')

        self.assertTrue(node_only.is_node_only())
        self.assertFalse(node_only.is_edge_only())
        self.assertTrue(edge_only.is_edge_only())
        self.assertFalse(edge_only.is_node_only())
        self.assertFalse(both.is_node_only())
        self.assertFalse(both.is_edge_only())
        self.assertTrue(empty.is_empty())
        self.assertFalse(node_only.is_empty())

    def test_member_count(self):
        a = Aspect(name='x', node_types=('p', 'q'), edge_relations=('r',))
        self.assertEqual(a.member_count, 3)


class TestAspectRegistryFromDict(unittest.TestCase):
    """AspectRegistry constructed via from_dict (the test/seed path)."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_contains_all_seeded(self):
        self.assertEqual(set(self.registry.all().keys()), set(SAMPLE_DATA.keys()))

    def test_attribute_access_required(self):
        a = self.registry.identity_bearing
        self.assertEqual(a.name, 'identity_bearing')
        self.assertIn('principle', a.node_types)

    def test_attribute_access_emergent_works_too(self):
        # __getattr__ doesn't distinguish required vs emergent — both resolve
        # if present. by_name() is the safer surface for emergent.
        a = self.registry.emergent_xyz
        self.assertEqual(a.name, 'emergent_xyz')

    def test_attribute_access_missing_raises(self):
        with self.assertRaises(AspectContractError):
            _ = self.registry.no_such_aspect

    def test_attribute_access_dunder_skipped(self):
        # Dunder names should raise AttributeError (not AspectContractError),
        # so Python internals (pickling, repr, etc.) work normally.
        with self.assertRaises(AttributeError):
            _ = self.registry.__some_dunder__

    def test_by_name_returns_none_for_missing(self):
        self.assertIsNone(self.registry.by_name('no_such_aspect'))
        # And returns the aspect for present names
        self.assertEqual(self.registry.by_name('identity_bearing').name,
                         'identity_bearing')


class TestAspectRegistryReverseLookups(unittest.TestCase):
    """by_node_type + by_edge_relation."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_by_node_type(self):
        a = self.registry.by_node_type('principle')
        self.assertEqual(a.name, 'identity_bearing')
        a2 = self.registry.by_node_type('correction')
        self.assertEqual(a2.name, 'correction_improvement')

    def test_by_node_type_missing(self):
        self.assertIsNone(self.registry.by_node_type('not_a_type'))

    def test_by_edge_relation(self):
        a = self.registry.by_edge_relation('corrects')
        self.assertEqual(a.name, 'correction_improvement')
        a2 = self.registry.by_edge_relation('co_accessed')
        self.assertEqual(a2.name, 'noise')

    def test_by_edge_relation_missing(self):
        self.assertIsNone(self.registry.by_edge_relation('not_a_relation'))


class TestAspectRegistryUnions(unittest.TestCase):
    """types_in + relations_in cross-aspect unions."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_types_in_single(self):
        result = self.registry.types_in(['identity_bearing'])
        self.assertEqual(set(result), {'principle', 'identity', 'rule'})

    def test_types_in_union_dedupes(self):
        # Add an aspect that overlaps to test dedup
        data_with_overlap = dict(SAMPLE_DATA)
        data_with_overlap['overlap_aspect'] = {
            'node_types': ['principle', 'new_type'],
            'edge_relations': [],
            'meaning': 'overlap test', 'dimension': 'semantic',
            'locked': False, 'metadata': {},
        }
        reg = AspectRegistry.from_dict(brain=None, data=data_with_overlap)
        result = reg.types_in(['identity_bearing', 'overlap_aspect'])
        # principle appears once despite being in both
        self.assertEqual(result.count('principle'), 1)
        self.assertIn('new_type', result)

    def test_relations_in(self):
        result = self.registry.relations_in(['correction_improvement', 'noise'])
        self.assertEqual(set(result),
                         {'corrects', 'supersedes', 'co_accessed', 'emergent_bridge'})

    def test_unions_skip_unknown_names(self):
        # Unknown aspect names are silently skipped, not errors
        result = self.registry.types_in(['identity_bearing', 'no_such_aspect'])
        self.assertEqual(set(result), {'principle', 'identity', 'rule'})


class TestAspectRegistryDiscovery(unittest.TestCase):
    """all / required / emergent / by_dimension / dimensions."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_all(self):
        result = self.registry.all()
        self.assertEqual(set(result.keys()), set(SAMPLE_DATA.keys()))

    def test_all_returns_fresh_dict(self):
        # Mutating the returned dict shouldn't affect the registry
        result = self.registry.all()
        result['injected'] = 'should_not_persist'
        result2 = self.registry.all()
        self.assertNotIn('injected', result2)

    def test_required_filters_by_REQUIRED_ASPECTS(self):
        # SAMPLE_DATA has 3 required (identity_bearing, correction_improvement, noise)
        # and 1 emergent (emergent_xyz). required() only returns the 3.
        result = self.registry.required()
        self.assertIn('identity_bearing', result)
        self.assertIn('correction_improvement', result)
        self.assertIn('noise', result)
        self.assertNotIn('emergent_xyz', result)

    def test_emergent_inverse(self):
        result = self.registry.emergent()
        self.assertIn('emergent_xyz', result)
        self.assertNotIn('identity_bearing', result)

    def test_required_emergent_partition(self):
        # required() + emergent() partition all() — disjoint, complete
        all_keys = set(self.registry.all().keys())
        required_keys = set(self.registry.required().keys())
        emergent_keys = set(self.registry.emergent().keys())
        self.assertEqual(required_keys | emergent_keys, all_keys)
        self.assertEqual(required_keys & emergent_keys, set())

    def test_by_dimension(self):
        result = self.registry.by_dimension('semantic')
        self.assertEqual(set(result.keys()), set(SAMPLE_DATA.keys()))
        # Adding a non-semantic should split
        data = dict(SAMPLE_DATA)
        data['temporal_recent'] = {
            'node_types': [], 'edge_relations': [],
            'meaning': 't', 'dimension': 'temporal',
            'locked': False, 'metadata': {},
        }
        reg = AspectRegistry.from_dict(brain=None, data=data)
        self.assertIn('temporal_recent', reg.by_dimension('temporal'))
        self.assertNotIn('temporal_recent', reg.by_dimension('semantic'))

    def test_dimensions(self):
        # SAMPLE_DATA only has semantic
        self.assertEqual(self.registry.dimensions(), {'semantic'})


class TestAspectRegistryListWithCounts(unittest.TestCase):
    """all_with_counts shape — feeds the list_aspects MCP tool."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_shape_includes_required_keys(self):
        result = self.registry.all_with_counts()
        self.assertEqual(len(result), len(SAMPLE_DATA))
        for entry in result:
            self.assertIn('name', entry)
            self.assertIn('meaning', entry)
            self.assertIn('node_types_count', entry)
            self.assertIn('edge_relations_count', entry)
            self.assertIn('node_types_preview', entry)
            self.assertIn('edge_relations_preview', entry)
            self.assertIn('dimension', entry)
            self.assertIn('locked', entry)

    def test_counts_match_member_lengths(self):
        result = {e['name']: e for e in self.registry.all_with_counts()}
        ib = result['identity_bearing']
        self.assertEqual(ib['node_types_count'], 3)
        self.assertEqual(ib['edge_relations_count'], 0)
        ci = result['correction_improvement']
        self.assertEqual(ci['node_types_count'], 2)
        self.assertEqual(ci['edge_relations_count'], 2)


class TestAspectRegistrySurfaceHelpers(unittest.TestCase):
    """relation_meaning_map + type_meaning_map for surface edge enrichment."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_relation_meaning_map(self):
        m = self.registry.relation_meaning_map()
        self.assertEqual(m['corrects'], 'Knowledge that updates prior knowledge')
        self.assertEqual(m['co_accessed'], 'Structural edges, not semantic claims')
        self.assertNotIn('principle', m)  # principle is a type, not a relation

    def test_type_meaning_map(self):
        m = self.registry.type_meaning_map()
        self.assertEqual(m['principle'], 'Nodes that anchor identity')
        self.assertEqual(m['correction'], 'Knowledge that updates prior knowledge')
        self.assertNotIn('corrects', m)  # corrects is a relation


class TestRequiredAspectsConstant(unittest.TestCase):
    """REQUIRED_ASPECTS is the contract surface."""

    def test_count(self):
        # 5 node-facing + 11 edge-facing = 16
        self.assertEqual(len(REQUIRED_ASPECTS), 16)

    def test_no_duplicates(self):
        self.assertEqual(len(set(REQUIRED_ASPECTS)), len(REQUIRED_ASPECTS))

    def test_immutable(self):
        # tuple, not list — can't be mutated
        self.assertIsInstance(REQUIRED_ASPECTS, tuple)


if __name__ == '__main__':
    unittest.main()
