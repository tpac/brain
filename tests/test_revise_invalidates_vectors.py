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

    def test_situation_invalidates_two_vectors(self):
        # situation feeds the dedicated `_situation` vector (kv-derived) and the
        # `high_meta` blend. There is NO field-cohort `situation` vector —
        # situation is served by `_situation` via FIELD_VECTOR_FALLBACK (one
        # vector, two consumers), so only these two go stale on a situation edit.
        result = vectors_affected_by('situation')
        self.assertEqual(result, {'_situation', 'high_meta'})

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
        # Seed the situation-affected vectors as if backfill had populated them.
        # There is no field-cohort `situation` vector — situation is served by
        # `_situation` via FIELD_VECTOR_FALLBACK — so only `_situation` and the
        # `high_meta` blend go stale on a situation edit.
        self._seed_vectors(nid, ['_situation', 'high_meta', 'title', '_primary'])
        before = self._vector_types_for_node(nid)
        self.assertIn('_situation', before)
        self.assertIn('high_meta', before)

        self.brain.revise(node_id=nid, situation='Updated situation text', reason='test')

        after = self._vector_types_for_node(nid)
        self.assertNotIn('_situation', after, 'situation revise must drop _situation')
        self.assertNotIn('high_meta', after, 'situation revise must drop high_meta blend')
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
        self._seed_vectors(nid, ['high_meta', 'user_raw_quote', '_situation', '_primary'])

        self.brain.revise(node_id=nid, user_raw_quote='new quote', reason='test')

        after = self._vector_types_for_node(nid)
        self.assertNotIn('high_meta', after)
        self.assertNotIn('user_raw_quote', after)
        self.assertIn('_situation', after, 'dedicated _situation vector unaffected by quote change')
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

    def test_revise_does_not_orphan_surviving_vectors_from_cache(self):
        """The 2026-07-17 healer-invisibility bug: revise deleted only the
        affected DB rows but dropped the WHOLE node from the in-memory vector
        cache — the surviving types (notably _primary) stayed in the DB, the
        backfill (DB-truth) saw nothing missing, and the node vanished from
        every cache-served recall scan until process restart. The cache drop
        must mirror the SQL delete exactly."""
        node = self.brain.remember(type='fact', title='Cache orphan probe',
                                   content='Body for cache orphan test.')
        nid = node['id']
        self._seed_vectors(nid, ['_primary', 'title', 'question'])
        # mirror the seeded rows into the cache the way the drain would
        if hasattr(self.brain._vec_dal, '_cache'):
            self.brain._vec_dal._cache.add_batch(
                (nid, vt, b'\x00', 'old text', 'test-stale-model')
                for vt in ('_primary', 'title', 'question'))

        # question-field revise → invalidates ONLY the question vector
        self.brain.revise(node_id=nid, question='What is this?', reason='test')

        after_db = self._vector_types_for_node(nid)
        self.assertIn('_primary', after_db)
        self.assertNotIn('question', after_db)
        if hasattr(self.brain._vec_dal, '_cache'):
            cached = self.brain._vec_dal._cache
            self.assertIsNotNone(
                cached.get(nid, '_primary'),
                'revise must NOT orphan the surviving _primary out of the '
                'cache — the node goes recall-invisible until restart')
            self.assertIsNone(cached.get(nid, 'question'),
                              'the invalidated type must leave the cache')


if __name__ == '__main__':
    unittest.main()
