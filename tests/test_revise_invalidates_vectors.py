"""When `revise()` updates a source field, the corresponding embedding
vector(s) must be deleted so embed_queue's backfill creates fresh ones.

Without this, `VectorDAL.find_missing()` returns "row exists, not missing"
and the vector keeps encoding outdated text indefinitely.

Tests use a fake `model='test-stale-model'` to seed initial vectors
without invoking the real embedder. The invalidation path doesn't care
about the embedding contents — it cares that the row gets deleted.
"""

import unittest
from tests.brain_test_base import BrainTestBase
from servers.pipeline_contract import vectors_affected_by


class TestVectorsAffectedBy(unittest.TestCase):
    """The contract helper itself — single source of truth for which fields
    invalidate which vectors."""

    def test_title_invalidates_title_and_primary(self):
        # title feeds the `title` slot AND the title+content blend (`_primary`)
        result = vectors_affected_by('title')
        self.assertIn('title', result)
        self.assertIn('_primary', result)

    def test_content_invalidates_primary_and_field_content(self):
        # content feeds `_primary` (title+content blend) AND field-cohort `content`
        result = vectors_affected_by('content')
        self.assertIn('_primary', result)
        self.assertIn('content', result)

    def test_situation_invalidates_three_vectors(self):
        # situation feeds: legacy `_situation` (kv-derived), legacy `high_meta`
        # blend, and field-cohort `situation`
        result = vectors_affected_by('situation')
        self.assertIn('_situation', result)
        self.assertIn('high_meta', result)
        self.assertIn('situation', result)

    def test_user_raw_quote_invalidates_high_meta_and_field(self):
        result = vectors_affected_by('user_raw_quote')
        self.assertIn('high_meta', result)
        self.assertIn('user_raw_quote', result)

    def test_anchor_raw_quote_invalidates_high_meta_and_field(self):
        result = vectors_affected_by('anchor_raw_quote')
        self.assertIn('high_meta', result)
        self.assertIn('anchor_raw_quote', result)

    def test_reasoning_invalidates_other_meta_and_field(self):
        result = vectors_affected_by('reasoning')
        self.assertIn('other_meta', result)
        self.assertIn('reasoning', result)

    def test_correction_pattern_invalidates_other_meta_only(self):
        # correction_pattern feeds `other_meta` blend; no field-cohort vector
        result = vectors_affected_by('correction_pattern')
        self.assertEqual(result, {'other_meta'})

    def test_question_invalidates_question_only(self):
        result = vectors_affected_by('question')
        self.assertEqual(result, {'question'})

    def test_unknown_field_invalidates_nothing(self):
        # An emergent metadata field with no embedding dependency
        result = vectors_affected_by('totally_unknown_field')
        self.assertEqual(result, set())


class TestReviseInvalidatesVectors(BrainTestBase):
    """End-to-end: revise() deletes the affected rows from node_enrichments."""

    needs_embedder = False

    def _seed_vectors(self, node_id, vector_types, model='test-stale-model'):
        """Seed fake vector rows so we can assert deletion. Bypasses the real
        embedder — we only care that `revise()` deletes these rows."""
        for vt in vector_types:
            self.brain.conn.execute(
                'INSERT OR REPLACE INTO node_enrichments '
                '(node_id, vector_type, text, embedding, model, created_at) '
                "VALUES (?, ?, 'old text', x'00', ?, datetime('now'))",
                (node_id, vt, model))
        self.brain.conn.commit()

    def _vector_types_for_node(self, node_id):
        rows = self.brain.conn.execute(
            'SELECT vector_type FROM node_enrichments WHERE node_id = ?',
            (node_id,)).fetchall()
        return {r[0] for r in rows}

    def test_revise_situation_invalidates_situation_vectors(self):
        node = self.brain.remember(
            type='fact', title='Test situation invalidation',
            content='Initial content.', situation='When this old situation applies')
        nid = node['id']
        # Seed all situation-affected vectors as if backfill had populated them
        self._seed_vectors(nid, ['_situation', 'high_meta', 'situation', 'title', '_primary'])
        before = self._vector_types_for_node(nid)
        self.assertIn('_situation', before)
        self.assertIn('high_meta', before)
        self.assertIn('situation', before)

        self.brain.revise(node_id=nid, situation='Updated situation text', reason='test')

        after = self._vector_types_for_node(nid)
        self.assertNotIn('_situation', after, 'situation revise must drop _situation')
        self.assertNotIn('high_meta', after, 'situation revise must drop high_meta blend')
        self.assertNotIn('situation', after, 'situation revise must drop field-cohort situation')
        # Untouched vectors remain
        self.assertIn('title', after, 'title vector must NOT be invalidated by situation revise')
        self.assertIn('_primary', after, '_primary must NOT be invalidated by situation revise')

    def test_revise_content_invalidates_primary_and_content(self):
        node = self.brain.remember(
            type='fact', title='Test content invalidation', content='Old content.')
        nid = node['id']
        self._seed_vectors(nid, ['_primary', 'content', 'title', 'high_meta'])

        self.brain.revise(node_id=nid, content='Replacement content.', reason='test')

        after = self._vector_types_for_node(nid)
        self.assertNotIn('_primary', after)
        self.assertNotIn('content', after)
        self.assertIn('title', after, 'title alone is unaffected by content change')
        self.assertIn('high_meta', after, 'high_meta unrelated to content change')

    def test_revise_title_invalidates_title_and_primary(self):
        node = self.brain.remember(
            type='fact', title='Old title', content='Stable content.')
        nid = node['id']
        self._seed_vectors(nid, ['title', '_primary', 'high_meta', 'content'])

        self.brain.revise(node_id=nid, title='Replacement title', reason='test')

        after = self._vector_types_for_node(nid)
        self.assertNotIn('title', after)
        self.assertNotIn('_primary', after)
        self.assertIn('high_meta', after, 'high_meta unrelated to title')
        self.assertIn('content', after,
                      'field-cohort content unaffected by title change')

    def test_revise_user_raw_quote_invalidates_high_meta_and_field(self):
        node = self.brain.remember(
            type='quote', title='Test', content='c',
            user_raw_quote='old quote')
        nid = node['id']
        self._seed_vectors(nid, ['high_meta', 'user_raw_quote', 'situation', '_primary'])

        self.brain.revise(node_id=nid, user_raw_quote='new quote', reason='test')

        after = self._vector_types_for_node(nid)
        self.assertNotIn('high_meta', after)
        self.assertNotIn('user_raw_quote', after)
        self.assertIn('situation', after, 'field-cohort situation unaffected by quote change')
        self.assertIn('_primary', after)

    def test_revise_unknown_field_invalidates_nothing(self):
        # Emergent fields with no embedding dependency must not delete vectors.
        node = self.brain.remember(type='fact', title='Test', content='c')
        nid = node['id']
        self._seed_vectors(nid, ['title', '_primary', 'high_meta', 'other_meta'])
        before = self._vector_types_for_node(nid)

        self.brain.revise(node_id=nid, my_emergent_field='value', reason='test')

        after = self._vector_types_for_node(nid)
        self.assertEqual(before, after,
                         'unknown field revise must not delete any vectors')


if __name__ == '__main__':
    unittest.main()
