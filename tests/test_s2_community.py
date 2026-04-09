"""Tests for S2 community detection unit.

Tests the IntegrationUnit contract, community detection algorithm,
naming heuristic, diff logic, and write results.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestIntegrationUnitContract(unittest.TestCase):
    """Test the base IntegrationUnit contract."""

    def test_subclass_must_implement_run(self):
        from servers.scales.s2.base import IntegrationUnit
        unit = IntegrationUnit(brain=None)
        with self.assertRaises(NotImplementedError):
            unit.run()

    def test_chain_id_format(self):
        from servers.scales.s2.base import IntegrationUnit
        from datetime import date

        class TestUnit(IntegrationUnit):
            NAME = 'test_op'
            SCALE = 's2'

        unit = TestUnit(brain=None)
        chain = unit.chain_id()
        today = date.today().strftime('%Y%m%d')
        self.assertEqual(chain, 's2-%s-test_op' % today)

    def test_chain_id_different_scale(self):
        from servers.scales.s2.base import IntegrationUnit

        class S3Unit(IntegrationUnit):
            NAME = 'synthesis'
            SCALE = 's3'

        unit = S3Unit(brain=None)
        self.assertTrue(unit.chain_id().startswith('s3-'))


class TestCommunityDetectionContract(unittest.TestCase):
    """Test CommunityDetection declares its O/K sources."""

    def test_sources_declared(self):
        from servers.scales.s2.community import CommunityDetection
        self.assertTrue(len(CommunityDetection.O_SOURCES) > 0)
        self.assertTrue(len(CommunityDetection.K_SOURCES) > 0)
        self.assertEqual(CommunityDetection.SCALE, 's2')
        self.assertEqual(CommunityDetection.ENCODING_SOURCE, 's2:community_detection')


class TestCommunityDetection(BrainTestBase):
    """Test community detection on synthetic graphs."""

    needs_embedder = False

    def _create_cluster(self, prefix, n, connect=True):
        """Create n nodes with a shared prefix, optionally fully connected."""
        ids = []
        for i in range(n):
            result = self.brain.remember(
                type='decision',
                title='%s node %d' % (prefix, i),
                content='Content about %s topic %d' % (prefix, i),
                keywords='%s testing' % prefix,
            )
            ids.append(result['id'])

        if connect:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    self.brain.connect(ids[i], ids[j], relation='related', weight=0.8)

        return ids

    def test_skip_when_graph_too_small(self):
        """Should skip when graph has fewer nodes than min_graph_nodes."""
        from servers.scales.s2.community import CommunityDetection
        # Create fewer than 20 nodes
        for i in range(5):
            self.brain.remember(type='fact', title='node %d' % i, content='c')

        unit = CommunityDetection(self.brain)
        result = unit.run()
        self.assertEqual(result['actions'], 0)
        self.assertIn('skipped', result)

    def test_detects_two_clusters(self):
        """Two well-separated clusters should be detected as communities."""
        from servers.scales.s2.community import CommunityDetection

        # Create two clusters of 12 nodes each, well connected internally
        cluster_a = self._create_cluster('alpha', 12)
        cluster_b = self._create_cluster('beta', 12)

        # Add one weak cross-cluster edge (bridge)
        self.brain.connect(cluster_a[0], cluster_b[0], relation='related', weight=0.1)

        unit = CommunityDetection(self.brain)
        result = unit.run()

        self.assertGreaterEqual(result.get('communities', 0), 2)
        self.assertGreater(result.get('actions', 0), 0)

        # Check node_communities table populated
        rows = self.brain.conn.execute('SELECT COUNT(*) FROM node_communities').fetchone()
        self.assertGreater(rows[0], 0)

    def test_community_nodes_created(self):
        """Community nodes should be created as regular nodes in the graph."""
        from servers.scales.s2.community import CommunityDetection

        cluster_a = self._create_cluster('alpha', 12)
        cluster_b = self._create_cluster('beta', 12)
        self.brain.connect(cluster_a[0], cluster_b[0], relation='related', weight=0.1)

        unit = CommunityDetection(self.brain)
        unit.run()

        # Find community nodes
        community_nodes = self.brain.conn.execute(
            "SELECT id, title, type, encoding_source FROM nodes WHERE type = 'community'"
        ).fetchall()

        self.assertGreaterEqual(len(community_nodes), 2)
        for nid, title, ntype, esource in community_nodes:
            self.assertEqual(ntype, 'community')
            self.assertEqual(esource, 's2:community_detection')
            self.assertTrue(len(title) > 0)

    def test_bidirectional_edges_created(self):
        """Edges between community nodes and members should be bidirectional."""
        from servers.scales.s2.community import CommunityDetection

        cluster_a = self._create_cluster('alpha', 12)
        cluster_b = self._create_cluster('beta', 12)
        self.brain.connect(cluster_a[0], cluster_b[0], relation='related', weight=0.1)

        unit = CommunityDetection(self.brain)
        unit.run()

        # Find a community node
        community_node = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE type = 'community' LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(community_node)
        cid = community_node[0]

        # Check community_member edges exist
        members = self.brain.conn.execute("""
            SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE (e.source_id = ? OR e.target_id = ?) AND er.relation = 'community_member'
        """, (cid, cid, cid)).fetchall()
        self.assertGreater(len(members), 0)

    def test_idempotent_second_run(self):
        """Running twice with same graph should not create duplicate community nodes."""
        from servers.scales.s2.community import CommunityDetection
        from servers.scales.s2.community_contract import COMMUNITY_DETECTION

        cluster_a = self._create_cluster('alpha', 12)
        cluster_b = self._create_cluster('beta', 12)
        self.brain.connect(cluster_a[0], cluster_b[0], relation='related', weight=0.1)

        config = dict(COMMUNITY_DETECTION)
        config['stability_threshold_pct'] = 0  # disable stability gate for test

        unit = CommunityDetection(self.brain, config=config)
        result1 = unit.run()
        count1 = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE type = 'community' AND archived = 0"
        ).fetchone()[0]

        # Run again
        result2 = unit.run()
        count2 = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE type = 'community' AND archived = 0"
        ).fetchone()[0]

        self.assertEqual(count1, count2)

    def test_traces_written(self):
        """S2 trace events should be written for the run."""
        from servers.scales.s2.community import CommunityDetection

        cluster_a = self._create_cluster('alpha', 12)
        cluster_b = self._create_cluster('beta', 12)
        self.brain.connect(cluster_a[0], cluster_b[0], relation='related', weight=0.1)

        unit = CommunityDetection(self.brain)
        unit.run()

        # Check traces were written
        traces = self.brain.logs_conn.execute(
            "SELECT scale, event_type, ref_type FROM trace_events WHERE scale = 's2'"
        ).fetchall()

        scales = {t[0] for t in traces}
        event_types = {t[1] for t in traces}
        ref_types = {t[2] for t in traces}

        self.assertIn('s2', scales)
        self.assertIn('O', event_types)
        self.assertIn('K', event_types)
        self.assertIn('delta', event_types)
        self.assertIn('graph_structure', ref_types)


class TestNamingHeuristic(BrainTestBase):
    """Test community naming from member keywords."""

    needs_embedder = False

    def test_name_from_keywords(self):
        from servers.scales.s2.community import CommunityDetection

        ids = []
        for i in range(5):
            result = self.brain.remember(
                type='decision',
                title='Recall improvement %d' % i,
                content='Content about recall pipeline',
                keywords='recall pipeline quality embedding',
            )
            ids.append(result['id'])

        unit = CommunityDetection(self.brain)
        title, content, keywords, situation, confidence = unit._name_community(ids)

        self.assertTrue(len(title) > 0)
        self.assertIn('members', content.lower())
        self.assertTrue(len(keywords) > 0)


if __name__ == '__main__':
    unittest.main()
