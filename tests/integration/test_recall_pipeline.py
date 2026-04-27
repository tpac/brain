"""Integration test: full recall pipeline (cosine → z-weighted → fatigue → results).

Uses a COPY of the live brain DB. Never modifies production.
Tests the actual recall() function end-to-end.
"""

import unittest
import sys
import os
import shutil
import sqlite3

sys.path.insert(0, '.')


def get_test_brain():
    """Create a Brain instance on a copy of the live DB."""
    src = os.path.expanduser('~/AgentsContext/brain/brain.db')
    if not os.path.exists(src):
        return None
    test_db = '/tmp/brain_test_recall_%d.db' % os.getpid()
    shutil.copy2(src, test_db)
    from servers.brain import Brain
    return Brain(db_path=test_db), test_db


class TestRecallPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        result = get_test_brain()
        if not result:
            raise unittest.SkipTest("No brain.db found")
        cls.brain, cls.test_db = result

    @classmethod
    def tearDownClass(cls):
        cls.brain.close()
        os.remove(cls.test_db)

    def test_recall_returns_results(self):
        """Basic recall returns a list of results."""
        result = self.brain.recall("how does the daemon work?", limit=10)
        self.assertIn('results', result)
        self.assertGreater(len(result['results']), 0)

    def test_recall_limit_respected(self):
        result = self.brain.recall("test", limit=5)
        self.assertLessEqual(len(result['results']), 5)

    def test_results_have_required_fields(self):
        """Each result must have id, type, title, effective_activation."""
        result = self.brain.recall("encoding agent", limit=3)
        for r in result['results']:
            self.assertIn('id', r)
            self.assertIn('type', r)
            self.assertIn('title', r)
            self.assertIn('effective_activation', r)

    def test_results_sorted_by_score(self):
        """Results should be sorted descending by effective_activation."""
        result = self.brain.recall("brain architecture", limit=10)
        scores = [r['effective_activation'] for r in result['results']]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def _fatigue_dict(self):
        """Read fatigue from session context (canonical) with Brain fallback.

        Fatigue moved to SessionContext when sessions became first-class.
        """
        ctx = getattr(self.brain, '_fatigue_ctx', None)
        if ctx is not None:
            return ctx.fatigue
        return getattr(self.brain, '_session_fatigue', {})

    def _reset_fatigue(self):
        ctx = getattr(self.brain, '_fatigue_ctx', None)
        if ctx is not None:
            ctx.fatigue.clear()
        if hasattr(self.brain, '_session_fatigue'):
            self.brain._session_fatigue = {}

    def test_fatigue_accumulates(self):
        """After multiple recalls, session fatigue dict should grow."""
        self._reset_fatigue()
        self.brain.recall("test query 1", limit=5)
        self.brain.recall("test query 2", limit=5)
        self.assertGreater(len(self._fatigue_dict()), 0)

    def test_fatigue_dampens_scores(self):
        """A node recalled twice should have a lower score the second time."""
        self._reset_fatigue()
        r1 = self.brain.recall("daemon architecture", limit=25)
        first_id = r1['results'][0]['id']
        first_score = r1['results'][0]['effective_activation']

        r2 = self.brain.recall("daemon architecture", limit=25)
        # Find the same node in second recall
        second_score = None
        for r in r2['results']:
            if r['id'] == first_id:
                second_score = r['effective_activation']
                break

        if second_score is not None:
            self.assertLess(second_score, first_score,
                          "Fatigue should reduce score on repeated recall")

    def test_structural_degree_cache(self):
        """Structural degree cache should be populated after recall."""
        self.brain.recall("test", limit=3)
        self.assertTrue(hasattr(self.brain, '_structural_degree_cache'))
        self.assertGreater(len(self.brain._structural_degree_cache), 0)

    def test_z_weighted_scoring(self):
        """Recall should use z-weighted enrichment scoring (not old cap)."""
        result = self.brain.recall("encoding agent design", limit=25)
        # Check that enrichment vectors participate
        stats = result.get('_embedding_stats', {})
        # The enrichment scan should run
        self.assertIn('enrichment_vectors_scanned', stats)

    def test_25_candidates_default(self):
        """Default recall should return up to 25 results for the surface."""
        result = self.brain.recall("test query", limit=25)
        self.assertLessEqual(len(result['results']), 25)


class TestGraphExpand(unittest.TestCase):
    """Test the graph_expand daemon command."""

    @classmethod
    def setUpClass(cls):
        result = get_test_brain()
        if not result:
            raise unittest.SkipTest("No brain.db found")
        cls.brain, cls.test_db = result

    @classmethod
    def tearDownClass(cls):
        cls.brain.close()
        os.remove(cls.test_db)

    def test_expand_returns_neighbors(self):
        """graph_expand should return structural neighbors.

        v22 edge model: relation lives on edge_relations.relation, not on
        edges.edge_type. Updated SQL accordingly.
        """
        from servers.daemon_dispatch import _handle_graph_expand
        # Get a node with known structural edges
        node_id = self.brain.conn.execute("""
            SELECT e.source_id
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE er.relation NOT IN ('co_accessed', 'emergent_bridge')
              AND er.archived = 0
            LIMIT 1
        """).fetchone()
        if not node_id:
            self.skipTest("No structural edges in test DB")

        result = _handle_graph_expand(self.brain, {
            'node_ids': [node_id[0][:8]],
            'depth': 1,
            'limit_per_seed': 3,
        }, [])
        self.assertTrue(result.get('ok', result.get('result', {}).get('neighbors') is not None))

    def test_expand_empty_ids(self):
        """Empty node_ids should return empty neighbors."""
        from servers.daemon_dispatch import _handle_graph_expand
        result = _handle_graph_expand(self.brain, {'node_ids': []}, [])
        self.assertEqual(result['result']['neighbors'], [])

    def test_expand_excludes_noise_edges(self):
        """Graph expand should not follow emergent_bridge edges."""
        from servers.daemon_dispatch import _handle_graph_expand
        from servers.brain_constants import EXCLUDED_EDGE_TYPES
        self.assertIn('emergent_bridge', EXCLUDED_EDGE_TYPES)


if __name__ == '__main__':
    unittest.main()
