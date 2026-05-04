"""Tests for aspect_migration — seed_required_aspects + migrate_emergent_from_legacy.

Step 5b coverage. Uses BrainTestBase (fresh brain per test, no production data).
"""

import json
import unittest

from tests.brain_test_base import BrainTestBase
from servers.aspect_migration import (
    seed_required_aspects, migrate_emergent_from_legacy, migrate_to_aspects,
    NAME_RENAMES, _load_seed,
)
from servers.aspects import REQUIRED_ASPECTS
from servers.dal_metadata import MetadataDAL, decode_value


class TestSeedRequiredAspects(BrainTestBase):
    """seed_required_aspects creates 14 aspect-nodes from JSON, idempotent."""

    needs_embedder = False  # We don't test embedding, just node creation

    def setUp(self):
        super().setUp()
        # Clear auto-seeded legacy interactions so the test brain starts
        # without s2_node_families / s2_edge_families. Tests that need
        # legacy fixtures register them explicitly.
        self.brain.logs_conn.execute(
            "DELETE FROM interactions WHERE name IN ('s2_node_families', 's2_edge_families')"
        )
        self.brain.logs_conn.commit()

    def _aspect_titles_in_db(self):
        res = self.brain.filter_nodes(field='type', include=['aspect'],
                                      rich=True, limit=500)
        return {n['title'] for n in res.get('nodes', []) if n.get('title')}

    def test_seeds_all_14_required_on_fresh_brain(self):
        result = seed_required_aspects(self.brain)
        self.assertEqual(set(result['created']), set(REQUIRED_ASPECTS))
        self.assertEqual(result['skipped'], [])
        self.assertEqual(result['errors'], [])
        # Verify by querying the brain
        titles = self._aspect_titles_in_db()
        for name in REQUIRED_ASPECTS:
            self.assertIn(name, titles, '%s should be in brain after seed' % name)

    def test_idempotent_second_run_skips_existing(self):
        # First run creates
        seed_required_aspects(self.brain)
        # Second run should skip everything
        result = seed_required_aspects(self.brain)
        self.assertEqual(result['created'], [])
        self.assertEqual(set(result['skipped']), set(REQUIRED_ASPECTS))
        self.assertEqual(result['errors'], [])

    def test_required_aspects_are_locked(self):
        seed_required_aspects(self.brain)
        # rich=True returns the locked field (filter_nodes skinny shape doesn't)
        res = self.brain.filter_nodes(field='type', include=['aspect'],
                                      rich=True, limit=500)
        for node in res['nodes']:
            with self.subTest(aspect=node['title']):
                self.assertTrue(
                    node.get('locked'),
                    '%s should be locked=True after seed' % node['title']
                )

    def test_required_aspects_have_member_lists_in_metadata(self):
        seed_required_aspects(self.brain)
        res = self.brain.filter_nodes(field='type', include=['aspect'],
                                      rich=True, limit=500)
        node_id = next(n['id'] for n in res['nodes']
                       if n['title'] == 'correction_improvement')

        meta_dal = MetadataDAL(self.brain.conn)
        node_types = decode_value(meta_dal.get_field(node_id, 'node_types'))
        edge_relations = decode_value(meta_dal.get_field(node_id, 'edge_relations'))

        self.assertIsInstance(node_types, list)
        self.assertIsInstance(edge_relations, list)
        self.assertIn('correction', node_types)
        self.assertIn('corrects', edge_relations)

    def test_healer_aspects_have_display_label(self):
        seed_required_aspects(self.brain)
        res = self.brain.filter_nodes(field='type', include=['aspect'],
                                      rich=True, limit=500)
        meta_dal = MetadataDAL(self.brain.conn)

        node_id = next(n['id'] for n in res['nodes']
                       if n['title'] == 'correction_improvement')
        display_label = meta_dal.get_field(node_id, 'display_label')
        self.assertEqual(display_label, 'corrects/improves')

    def test_dimension_metadata_set(self):
        seed_required_aspects(self.brain)
        res = self.brain.filter_nodes(field='type', include=['aspect'],
                                      rich=True, limit=500)
        meta_dal = MetadataDAL(self.brain.conn)

        node_id = next(n['id'] for n in res['nodes']
                       if n['title'] == 'identity_bearing')
        dimension = meta_dal.get_field(node_id, 'dimension')
        self.assertEqual(dimension, 'semantic')


class TestMigrateEmergentFromLegacy(BrainTestBase):
    """migrate_emergent_from_legacy reads old interactions, creates emergent."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        # Clear auto-seeded legacy interactions for a controllable baseline
        self.brain.logs_conn.execute(
            "DELETE FROM interactions WHERE name IN ('s2_node_families', 's2_edge_families')"
        )
        self.brain.logs_conn.commit()

    def _seed_legacy_interactions(self):
        """Set up legacy s2_node_families + s2_edge_families to migrate from."""
        # Mix of required (in REQUIRED_ASPECTS) and emergent names
        node_families = {
            'identity_bearing': {  # required — should NOT become emergent
                'members': ['principle', 'rule'],
                'meaning': 'identity stuff'
            },
            'architecture_design': {  # emergent — should be migrated
                'members': ['architecture', 'mechanism', 'pattern'],
                'meaning': 'how things are built'
            },
            'correction_supersession': {  # legacy name → renamed to required
                'members': ['correction', 'bug_lesson'],
                'meaning': 'corrects'
            },
        }
        edge_families = {
            'correction_improvement': {  # required — should NOT become emergent
                'members': ['corrects', 'supersedes'],
                'meaning': 'corrections via edges'
            },
            'similarity_complement': {  # emergent — should be migrated
                'members': ['parallels', 'mirrors', 'instance_of'],
                'meaning': 'similar / parallel relationships'
            },
        }
        self.brain._interaction_dal.register(
            's2_node_families', template='legacy node families',
            parameters=json.dumps(node_families),
            created_by='test:legacy_seed')
        self.brain._interaction_dal.register(
            's2_edge_families', template='legacy edge families',
            parameters=json.dumps(edge_families),
            created_by='test:legacy_seed')

    def test_emergent_aspects_created(self):
        self._seed_legacy_interactions()
        result = migrate_emergent_from_legacy(self.brain)

        # architecture_design + similarity_complement should be created
        # correction_supersession is renamed to correction_improvement (required)
        # so it doesn't go through emergent — handled by seed_required.
        self.assertIn('architecture_design', result['created'])
        self.assertIn('similarity_complement', result['created'])
        # Required names skip the emergent path
        self.assertNotIn('identity_bearing', result['created'])
        self.assertNotIn('correction_improvement', result['created'])

    def test_emergent_aspects_unlocked(self):
        self._seed_legacy_interactions()
        migrate_emergent_from_legacy(self.brain)

        res = self.brain.filter_nodes(field='type', include=['aspect'],
                                      rich=True, limit=500)
        emergent_node = next((n for n in res['nodes']
                              if n['title'] == 'architecture_design'), None)
        self.assertIsNotNone(emergent_node, 'architecture_design should exist')
        # Constitution forces locked=False for non-anchor encoding sources
        self.assertFalse(emergent_node.get('locked'),
                         'emergent aspects should be unlocked')

    def test_emergent_member_lists_preserved(self):
        self._seed_legacy_interactions()
        migrate_emergent_from_legacy(self.brain)

        res = self.brain.filter_nodes(field='type', include=['aspect'],
                                      rich=True, limit=500)
        node_id = next(n['id'] for n in res['nodes']
                       if n['title'] == 'architecture_design')
        meta_dal = MetadataDAL(self.brain.conn)
        node_types = decode_value(meta_dal.get_field(node_id, 'node_types'))
        self.assertEqual(set(node_types), {'architecture', 'mechanism', 'pattern'})

    def test_idempotent_on_emergent(self):
        self._seed_legacy_interactions()
        migrate_emergent_from_legacy(self.brain)
        result = migrate_emergent_from_legacy(self.brain)
        # Second run skips
        self.assertEqual(result['created'], [])
        self.assertIn('architecture_design', result['skipped'])
        self.assertIn('similarity_complement', result['skipped'])

    def test_empty_legacy_means_nothing_emergent(self):
        # No legacy interactions → no emergent
        result = migrate_emergent_from_legacy(self.brain)
        self.assertEqual(result['created'], [])

    def test_name_rename_correction_supersession(self):
        # Legacy 'correction_supersession' → canonical 'correction_improvement'
        self.assertEqual(NAME_RENAMES['correction_supersession'],
                         'correction_improvement')


class TestMigrateOrchestrator(BrainTestBase):
    """migrate_to_aspects = seed_required + migrate_emergent."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        # Clear auto-seeded legacy interactions
        self.brain.logs_conn.execute(
            "DELETE FROM interactions WHERE name IN ('s2_node_families', 's2_edge_families')"
        )
        self.brain.logs_conn.commit()

    def test_full_orchestrator(self):
        # Seed legacy interactions
        node_families = {
            'architecture_design': {
                'members': ['architecture', 'pattern'],
                'meaning': 'arch stuff'
            },
        }
        edge_families = {
            'similarity_complement': {
                'members': ['mirrors', 'parallels'],
                'meaning': 'sim'
            },
        }
        self.brain._interaction_dal.register(
            's2_node_families', template='legacy',
            parameters=json.dumps(node_families), created_by='test')
        self.brain._interaction_dal.register(
            's2_edge_families', template='legacy',
            parameters=json.dumps(edge_families), created_by='test')

        result = migrate_to_aspects(self.brain)

        # Required: 14 created
        self.assertEqual(set(result['required']['created']), set(REQUIRED_ASPECTS))
        # Emergent: 2
        self.assertEqual(set(result['emergent']['created']),
                         {'architecture_design', 'similarity_complement'})
        # Total aspect_nodes: 14 + 2 = 16
        self.assertEqual(result['aspect_node_count'], 16)

    def test_orchestrator_idempotent(self):
        migrate_to_aspects(self.brain)
        result = migrate_to_aspects(self.brain)
        self.assertEqual(result['required']['created'], [])
        self.assertEqual(result['emergent']['created'], [])
        # All 14 required, no emergent (no legacy seeded)
        self.assertEqual(result['aspect_node_count'], 14)


class TestSeedJsonStillValid(unittest.TestCase):
    """The seed JSON still loads via _load_seed without errors."""

    def test_load_seed_returns_14(self):
        seed = _load_seed()
        self.assertEqual(len(seed), 14)
        self.assertEqual(set(seed.keys()), set(REQUIRED_ASPECTS))


if __name__ == '__main__':
    unittest.main()
