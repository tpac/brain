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

    def test_fatigue_accumulates_within_a_session(self):
        """Recall with a session increments per-node fatigue on that session.

        Fatigue is per-session: it lives on SessionContext and only accumulates
        when recall is given one (production always is — the hook passes ctx=,
        the MCP path passes session_id=). A recall with no session deliberately
        drops the increment, so this test must supply one. The tests that used
        to live here read `brain._fatigue_ctx` / `brain._session_fatigue` —
        attributes that do not exist — so they could never observe accumulation
        and sat deselected in pytest.ini for months.

        NOT asserted here, deliberately — two things measured 2026-08-13 under
        laf_v1 that nobody has explained yet, and pinning a guess about either
        would be worse than leaving them visible:
        (1) The count does not pass 1. A second recall of the same query in the
            same session leaves the top node's fatigue at 1, though
            `_mark_accessed` increments unconditionally when ctx is not None and
            `get_or_create_session` returns a cached instance by reference.
        (2) Fatigue does not move the surfaced score. Two such recalls return
            `effective_activation` identical to 16 decimal places, though the
            path reads as intact: LAF field score → `sim *= (1 - fatigue)` →
            embedding_scores → blended → effective_activation.
        Both need a debug run, not more reading. Until then this test pins only
        what is proven: a recall with a session records fatigue.
        """
        sid = 'fatigue-contract-test'
        ctx = self.brain.get_or_create_session(sid)
        ctx.fatigue.clear()

        r1 = self.brain.recall("daemon architecture", limit=25, session_id=sid)
        self.assertTrue(r1['results'], "need at least one result to fatigue")
        first_id = r1['results'][0]['id']

        self.assertGreater(len(ctx.fatigue), 0,
                           "recall with a session must increment fatigue")
        self.assertGreater(ctx.fatigue.get(first_id, 0), 0,
                           "the top result must carry a fatigue count")


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
        """Graph expand excludes the traversal set (was a drifted 1-member
        literal that leaked co_accessed edges) — community_member deliberately
        NOT excluded here: Anchor's manual graph walk keeps community links."""
        exclusions = self.brain.aspects.traversal_exclusions
        self.assertIn('emergent_bridge', exclusions)
        self.assertIn('co_accessed', exclusions)
        self.assertNotIn('community_member', exclusions)


if __name__ == '__main__':
    unittest.main()
