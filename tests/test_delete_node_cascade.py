"""Brain.delete_node_cascade — hard delete leaves NO orphan child rows.

Phase 5: replaces NodeDAL.purge, which deleted node_enrichments /
node_metadata_kv / edges / nodes but LEAKED node_vectors and node_source_refs.
This test proves the cascade clears every child table — asserting the two
leaked tables had rows before and are empty after — and leaves neighbors intact.

Run: BRAIN_ALLOW_ANY_PYTHON=1 python3 -m pytest tests/test_delete_node_cascade.py -v
"""
import unittest

from tests.brain_test_base import BrainTestBase


class TestDeleteNodeCascade(BrainTestBase):
    needs_embedder = True  # so node_enrichments (primary vector) is populated

    def _count(self, table, node_id, col='node_id'):
        return self.brain.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (node_id,)).fetchone()[0]

    def test_cascade_clears_every_child_table(self):
        b = self.brain
        victim = b.remember(type='fact', title='cascade victim',
                            content='delete me fully, leave nothing behind')
        neighbor = b.remember(type='fact', title='cascade neighbor',
                             content='this one stays')
        nid = victim['id']
        b._meta_kv.set(nid, 'testkey', 'testval')
        b._source_refs.add_source_refs(nid, ['aabbccdd', 'bbccddee'])
        b._graph.add_relation(nid, neighbor['id'], 'related')
        b.save()

        # Preconditions — rows exist in the two tables purge used to LEAK.
        self.assertGreater(self._count('node_vectors', nid), 0,
                           'precondition: tf-idf vectors present')
        self.assertEqual(self._count('node_source_refs', nid), 2,
                         'precondition: source_refs present')
        edge_id = b.conn.execute(
            "SELECT edge_id FROM edges WHERE source_id = ? OR target_id = ?",
            (nid, nid)).fetchone()[0]
        self.assertGreater(self._count('edge_relations', edge_id, col='edge_id'), 0,
                           'precondition: edge_relations present')

        b.delete_node_cascade(nid)

        # The edge_relations rows are hard-deleted too (not just soft-archived).
        self.assertEqual(self._count('edge_relations', edge_id, col='edge_id'), 0,
                         'edge_relations leaked after cascade')

        for table in ('node_enrichments', 'node_metadata_kv',
                      'node_vectors', 'node_source_refs'):
            self.assertEqual(self._count(table, nid), 0,
                             f'{table} leaked rows after cascade')
        self.assertEqual(self._count('nodes', nid, col='id'), 0, 'node row gone')
        edges = b.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_id = ? OR target_id = ?",
            (nid, nid)).fetchone()[0]
        self.assertEqual(edges, 0, 'edges gone')

        # The neighbor is untouched.
        self.assertEqual(self._count('nodes', neighbor['id'], col='id'), 1,
                         'cascade must not touch other nodes')

    def test_empty_node_id_is_noop(self):
        self.brain.delete_node_cascade('')  # must not raise

    def test_cascade_clears_fts5(self):
        """Pre-consolidation blind spot (2026-07-17 plan §5.1): cascade never
        cleaned nodes_fts — no triggers exist, nothing else ever deletes the
        row, and this suite didn't pin it. _deindex_node closes the gap."""
        has_fts5 = self.brain.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes_fts'"
        ).fetchone() is not None
        if not has_fts5:
            self.skipTest('fixture DB has no FTS5 virtual table')
        node = self.brain.remember(type='fact', title='FTS cascade probe',
                                   content='Body for the FTS cascade pin.')
        nid = node['id']
        n_before = self.brain.conn.execute(
            'SELECT COUNT(*) FROM nodes_fts WHERE node_id = ?', (nid,)).fetchone()[0]
        self.assertEqual(n_before, 1, 'remember must index into FTS5')
        self.brain.delete_node_cascade(nid)
        n_after = self.brain.conn.execute(
            'SELECT COUNT(*) FROM nodes_fts WHERE node_id = ?', (nid,)).fetchone()[0]
        self.assertEqual(n_after, 0, 'cascade must clear nodes_fts')

    def test_archive_is_soft_deindex_keeps_tfidf(self):
        """archive_node is SOFT delete: drop the expensive embeddings (+cache)
        and the FTS row, but KEEP the tfidf/node_vectors rows — deleting them
        inflates doc_freq (TfIdfDAL.delete_for_node doesn't decrement) and
        strips include_archived lexical reachability. Contrast the hard
        delete_node_cascade, which drops everything (tests above)."""
        node = self.brain.remember(type='fact', title='Archive soft-deindex probe',
                                   content='Body for the archive soft deindex pin.')
        nid = node['id']
        tfidf_before = self.brain.conn.execute(
            'SELECT COUNT(*) FROM node_vectors WHERE node_id = ?', (nid,)).fetchone()[0]
        self.assertGreater(tfidf_before, 0, 'remember must write tfidf rows')

        self.brain.archive_node(nid, reason='test', archived_by='test')

        enr = self.brain.conn.execute(
            'SELECT COUNT(*) FROM node_enrichments WHERE node_id = ?', (nid,)).fetchone()[0]
        self.assertEqual(enr, 0, 'archive drops embeddings')
        tfidf_after = self.brain.conn.execute(
            'SELECT COUNT(*) FROM node_vectors WHERE node_id = ?', (nid,)).fetchone()[0]
        self.assertEqual(tfidf_after, tfidf_before,
                         'archive is soft — tfidf rows KEPT (doc_freq stays consistent)')
        if hasattr(self.brain._vec_dal, '_cache'):
            self.assertEqual(self.brain._vec_dal.get_for_node(nid), [],
                             'archive drops the cache view of the vectors')


if __name__ == '__main__':
    unittest.main()
