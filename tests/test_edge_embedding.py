"""Edge embedding stability + write-path coverage (schema v26).

Three properties this suite locks:

1. **Stability across partner-title revisions.** The composed edge text
   is INTRINSIC — `[relation] description family: meaning`, no partner
   title. Revising a partner node's title must NOT invalidate the edge
   embedding. Without this, every node revision would cascade-stale all
   incoming/outgoing edge embeddings, defeating the whole point of
   storing them.

2. **Write-path populates the column.** A `connect_typed` call on a
   relation that gets read by spread/select_edges (i.e. NOT in
   DEFAULT_EXCLUDED_RELATIONS) must populate `edge_relations.embedding`
   AND `embedding_model` synchronously, so downstream reads find a
   stored vector.

3. **Excluded relations skip embedding.** `co_accessed` and
   `emergent_bridge` are excluded from spread + select_edges by default,
   so embedding them is wasted work. The early-out in
   `_maybe_embed_edge_relation` must keep them at NULL.

These guard against drift in the edge-storage symmetry. If they fail,
spread will silently re-pay fastembed cost or use stale geometry.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.brain_test_base import BrainTestBase  # noqa: E402


class TestEdgeEmbedding(BrainTestBase):
    needs_embedder = True

    def _create_pair(self, title_a='Source node', title_b='Target node'):
        a = self.brain.remember(type='fact', title=title_a, content='A')
        b = self.brain.remember(type='fact', title=title_b, content='B')
        return a['id'], b['id']

    def _read_embedding(self, source_id, target_id, relation):
        """Return (blob, model) for the (source, target, relation) edge.
        Edges are single-direction in v22+, so look up via the edge_id
        we get from a forward query first.
        """
        edge_row = self.brain.conn.execute(
            'SELECT edge_id FROM edges '
            'WHERE (source_id = ? AND target_id = ?) '
            'OR (source_id = ? AND target_id = ?)',
            (source_id, target_id, target_id, source_id)).fetchone()
        if not edge_row:
            return None, None
        row = self.brain.conn.execute(
            'SELECT embedding, embedding_model FROM edge_relations '
            'WHERE edge_id = ? AND relation = ?',
            (edge_row[0], relation)).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    def test_write_path_populates_embedding(self):
        """connect_typed on a non-excluded relation must store an
        embedding + model name synchronously. If this fails, spread
        falls through to fastembed every time the edge is touched."""
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends',
                                 description='A extends B')
        blob, model = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob, 'edge embedding should be stored at write')
        self.assertGreater(len(blob), 0)
        self.assertTrue(model, 'embedding_model should be populated')

    def test_excluded_relations_skip_embedding(self):
        """co_accessed and emergent_bridge are filtered out by both
        spread and select_edges (DEFAULT_EXCLUDED_RELATIONS), so
        embedding them is pure waste. _maybe_embed_edge_relation must
        early-out for these."""
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='co_accessed',
                                 description='surface co-access')
        blob, model = self._read_embedding(a, b, 'co_accessed')
        self.assertIsNone(blob,
                          'co_accessed edges should not be embedded — '
                          'they are excluded from spread/select_edges')

    def test_partner_title_revision_does_not_stale_embedding(self):
        """Revising the partner node's title must not change the edge's
        stored embedding — that's the whole point of dropping
        partner_title from the composed text. If this assertion fails,
        every node title revise would silently invalidate every incoming
        and outgoing edge embedding."""
        a, b = self._create_pair(title_a='Original A', title_b='Original B')
        self.brain.connect_typed(a, b, relation='extends',
                                 description='A extends B')
        blob_before, _ = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_before)

        # Revise B's title. Edge embedding must NOT change.
        self.brain.revise(b, title='Renamed B')
        blob_after, _ = self._read_embedding(a, b, 'extends')
        self.assertEqual(blob_before, blob_after,
                         'edge embedding must be stable across partner '
                         'title revisions — embedding is intrinsic to '
                         '(relation, description, family meaning) only')

    def test_description_change_updates_embedding(self):
        """Conversely, when the description changes, the embedding must
        update — the edge text DOES depend on description."""
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends',
                                 description='first description')
        blob_v1, _ = self._read_embedding(a, b, 'extends')

        self.brain.connect_typed(a, b, relation='extends',
                                 description='entirely different rationale')
        blob_v2, _ = self._read_embedding(a, b, 'extends')

        self.assertNotEqual(blob_v1, blob_v2,
                            'description change should produce a new '
                            'edge embedding (new input → new vector)')

    def test_archive_clears_embedding(self):
        """Archived edges must NULL their embedding (parallel to node
        archive's DELETE FROM node_enrichments). Archived rows are
        never read by spread/select_edges, so the blob is dead weight.
        Revive via add_relation Branch 3 re-embeds, so this is safe."""
        from servers.dal import GraphDAL
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends',
                                 description='A extends B')
        blob_before, model_before = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_before, 'embedding should be populated at write')
        self.assertIsNotNone(model_before)

        # Archive via the explicit per-relation path.
        gdal = GraphDAL(self.brain.conn)
        gdal.remove_relation(a, b, 'extends', archived_by='test')

        blob_after, model_after = self._read_embedding(a, b, 'extends')
        self.assertIsNone(blob_after,
                          'archived edge_relations.embedding must be NULL — '
                          'archived rows are unread; blob is wasted storage')
        self.assertIsNone(model_after,
                          'archived edge_relations.embedding_model must be '
                          'NULL too (paired with embedding)')

        # The row itself should still exist (soft-archive, not hard-delete)
        # so decay/audit/revive paths still find it. Filter by relation
        # because the same edge_id can carry multiple relations (the
        # daemon's Hebbian path adds 'co_accessed' on the same edge).
        edge_row = self.brain.conn.execute(
            'SELECT er.archived FROM edges e '
            'JOIN edge_relations er ON er.edge_id = e.edge_id '
            'WHERE er.relation = ? '
            'AND ((e.source_id = ? AND e.target_id = ?) '
            '     OR (e.source_id = ? AND e.target_id = ?))',
            ('extends', a, b, b, a)).fetchone()
        self.assertIsNotNone(edge_row,
                             'archived edge_relations row should still exist '
                             '(soft-archive, not hard-delete)')
        self.assertEqual(edge_row[0], 1, 'row should be archived=1')

    def test_revive_re_embeds_after_archive(self):
        """When an archived edge is revived via add_relation, the
        embedding must be repopulated. Without this, archived → revived
        edges stay NULL and fall through to the on-demand fastembed path
        forever — a slow leak back into the bug we just fixed."""
        from servers.dal import GraphDAL
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends',
                                 description='original description')
        blob_v1, _ = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_v1)

        # Archive
        gdal = GraphDAL(self.brain.conn)
        gdal.remove_relation(a, b, 'extends', archived_by='test')
        self.assertIsNone(self._read_embedding(a, b, 'extends')[0])

        # Revive (add_relation Branch 3 — fresh state, created=True)
        self.brain.connect_typed(a, b, relation='extends',
                                 description='revived description')
        blob_v2, model_v2 = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_v2,
                             'revived edge must get a fresh embedding')
        self.assertIsNotNone(model_v2)
        # New description → new vector
        self.assertNotEqual(blob_v1, blob_v2,
                            'revived edge with new description should '
                            'produce a new vector')


if __name__ == '__main__':
    unittest.main()
