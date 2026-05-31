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
        b._graph.add_source_refs(nid, ['aabbccdd', 'bbccddee'])
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


if __name__ == '__main__':
    unittest.main()
