"""LogsDAL.run_maintenance graph-orphan cleanup — the two-sided delete contract.

2026-06-12 audit: the cleanup deleted `edges` rows whose endpoints vanished
but left their `edge_relations` rows behind — 17,982 stranded relation rows
(84% of active) accumulated, invisible to every JOIN-based read. The contract
now:

1. Deleting an orphaned edge also deletes ITS relation rows (scoped).
2. Pre-existing orphan relations (no parent edge at all) are NOT touched —
   they are the recovery corpus for the trace-based restoration effort.
3. Healthy edges and their relations survive untouched.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestMaintenanceOrphanCleanup(BrainTestBase):
    needs_embedder = False

    def _node(self, title):
        return self.brain.remember(type='fact', title=title, content='c',
                                   encoding_source='anchor')['id']

    def test_orphan_edge_delete_takes_relations_along(self):
        a, b, c = self._node('A'), self._node('B'), self._node('C')
        # Healthy edge A—B and soon-to-be-orphaned edge A—C
        self.brain._graph.add_relation(a, b, 'extends', description='healthy')
        self.brain._graph.add_relation(a, c, 'extends', description='doomed')
        doomed_edge = self.brain._graph.get_edge_id(a, c)
        healthy_edge = self.brain._graph.get_edge_id(a, b)
        # Pre-existing orphan relation (no parent edge at all) — must SURVIVE
        # cleanup; it's recovery corpus, not garbage.
        self.brain.conn.execute(
            "INSERT INTO edge_relations (edge_id, relation, description, "
            "weight, encoding_source, created_at) "
            "VALUES ('edg_deadbeef', 'extends', 'pre-existing orphan', "
            "0.5, 'test', '2026-01-01T00:00:00+00:00')")
        # Hard-delete node C (simulates the condition the cleanup targets)
        self.brain.conn.execute("DELETE FROM nodes WHERE id = ?", (c,))
        self.brain.conn.commit()

        stats = self.brain._logs_dal.run_maintenance(graph_conn=self.brain.conn)

        def edge_exists(eid):
            return self.brain.conn.execute(
                "SELECT 1 FROM edges WHERE edge_id = ?", (eid,)).fetchone()

        def relations_for(eid):
            return self.brain.conn.execute(
                "SELECT COUNT(*) FROM edge_relations WHERE edge_id = ?",
                (eid,)).fetchone()[0]

        # Doomed edge AND its relations are gone — no new orphans
        self.assertIsNone(edge_exists(doomed_edge))
        self.assertEqual(relations_for(doomed_edge), 0)
        self.assertGreaterEqual(stats.get('orphaned_edge_relations', 0), 1)
        # Healthy edge untouched
        self.assertIsNotNone(edge_exists(healthy_edge))
        self.assertEqual(relations_for(healthy_edge), 1)
        # Pre-existing orphan preserved for recovery
        self.assertEqual(relations_for('edg_deadbeef'), 1)
