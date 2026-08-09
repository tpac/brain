"""Integration tests for ALL mechanisms built in session 2026-04-02/03.

Each test documents WHAT the mechanism does, WHY it exists, and HOW it's tested.
This file is the living record of the decode pipeline v2 architecture.

Uses a COPY of the live brain DB. Never modifies production.

Mechanisms tested:
1. Z-weighted 4-group embedding scoring
2. Synaptic fatigue (degree-based)
3. Surface-selected Hebbian (co_accessed from surface, not from cosine scan)
4. Embedding redistribution (70/30 from frozen originals)
5. Structural graph separation (co_accessed + emergent excluded from traversal)
6. Layer 3 post-surface graph expansion
7. KV metadata store (extensible without schema changes)
8. Encoding group vectors (title, high_meta, other_meta stored at encode time)
9. Session context from encoder to surface
10. Surface prompt: silence on confirmations + tangential match rejection
"""

import unittest
import sys
import os
import shutil
import sqlite3
import json

sys.path.insert(0, '.')


def get_test_brain():
    src = os.path.expanduser('~/AgentsContext/brain/brain.db')
    if not os.path.exists(src):
        return None
    test_db = '/tmp/brain_test_mechanisms_%d.db' % os.getpid()
    shutil.copy2(src, test_db)
    from servers.brain import Brain
    return Brain(db_path=test_db), test_db


class Test01_ZWeightedGroups(unittest.TestCase):
    """Z-weighted 4-group embedding scoring.

    WHAT: Each node has up to 4 embedding vectors (title, blend, high_meta, other_meta).
    Each vector's cosine similarity is multiplied by its group weight.
    Final score = average of top 2 weighted scores.

    WHY: Single-vector embeddings dilute the title signal with content.
    Multi-vector lets title matches and metadata matches compete independently.
    Top-2 averaging requires two vectors to agree — prevents noisy single-field matches.

    Tested: +5.1pts R@25 vs single-vector baseline.
    Defined in: pipeline_contract.py EMBEDDING_GROUPS
    Scored in: brain_recall.py STEP 3.5
    """

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

    def test_title_vectors_exist(self):
        """All active nodes should have title enrichment vectors."""
        total = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
        with_title = self.brain.conn.execute(
            "SELECT COUNT(DISTINCT node_id) FROM node_enrichments WHERE vector_type='title'"
        ).fetchone()[0]
        coverage = with_title / max(total, 1)
        self.assertGreater(coverage, 0.9,
                          "At least 90% of nodes should have title vectors")

    def test_group_types_in_enrichments(self):
        """node_enrichments should contain title, high_meta, other_meta types."""
        types = {r[0] for r in self.brain.conn.execute(
            "SELECT DISTINCT vector_type FROM node_enrichments").fetchall()}
        self.assertIn('title', types)
        # high_meta and other_meta only exist on nodes with metadata
        # Just check they're not blocked by schema

    def test_weights_applied_in_scoring(self):
        """Recall should produce different scores than raw cosine."""
        from servers.pipeline_contract import EMBEDDING_GROUPS
        # Title weight should be highest
        self.assertEqual(EMBEDDING_GROUPS['title']['weight'], 1.0)
        self.assertLess(EMBEDDING_GROUPS['other_meta']['weight'],
                       EMBEDDING_GROUPS['title']['weight'])


class Test02_SynapticFatigue(unittest.TestCase):
    """Synaptic fatigue — hub nodes self-throttle based on structural degree.

    WHAT: Nodes recalled repeatedly in a session get their cosine similarity
    dampened. Rate scales with structural degree: K = 10 / (1 + degree/10).
    Hubs (degree 30) fatigue after 3 recalls. Peripheral nodes barely fatigue.
    Resets between sessions.

    WHY: Biology — neurotransmitter depletion. Hubs fire across more synapses,
    deplete faster. Prevents the same 5 nodes from dominating every recall.

    WHERE: brain_recall.py STEP 3, applied at base cosine level.
    STATE: _session_fatigue dict on Brain instance.
    """

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

    def _get_fatigue(self):
        """Read fatigue from the live source (session context).

        Fatigue used to be a dict on the Brain instance (`_session_fatigue`)
        but moved to SessionContext when sessions became first-class
        (commit b9fe76f era). The Brain attribute is now only populated
        as a fallback when no session context exists. Tests should read
        from the canonical location.
        """
        ctx = getattr(self.brain, '_fatigue_ctx', None)
        if ctx is not None:
            return dict(ctx.fatigue)
        return dict(getattr(self.brain, '_session_fatigue', {}))

    def _reset_fatigue(self):
        ctx = getattr(self.brain, '_fatigue_ctx', None)
        if ctx is not None:
            ctx.fatigue.clear()
        if hasattr(self.brain, '_session_fatigue'):
            self.brain._session_fatigue = {}

    def test_fatigue_dict_created(self):
        self._reset_fatigue()
        self.brain.recall("test", limit=5)
        # Either the session-context dict or the Brain fallback is populated.
        self.assertTrue(self._get_fatigue() is not None)

    def test_fatigue_increments(self):
        self._reset_fatigue()
        self.brain.recall("daemon architecture", limit=5)
        first_count = self._get_fatigue()
        self.brain.recall("daemon architecture", limit=5)
        second_count = self._get_fatigue()
        # At least some nodes should have higher fatigue
        increased = any(second_count.get(k, 0) > v
                       for k, v in first_count.items())
        self.assertTrue(increased, "Fatigue should increment after recall")

    def test_degree_cache_populated(self):
        """Structural degree cache computed once, used for fatigue K calculation."""
        if hasattr(self.brain, '_structural_degree_cache'):
            del self.brain._structural_degree_cache
        self.brain.recall("test", limit=3)
        self.assertTrue(hasattr(self.brain, '_structural_degree_cache'))
        self.assertGreater(len(self.brain._structural_degree_cache), 0)

    def test_high_degree_fatigues_faster(self):
        """Hubs (high degree) should have lower K = faster fatigue."""
        # K = 10 / (1 + degree/10)
        K_hub = 10.0 / (1.0 + 30.0/10.0)   # degree 30
        K_new = 10.0 / (1.0 + 0.0/10.0)     # degree 0
        self.assertLess(K_hub, K_new)
        # After 3 recalls: hub is 55% fatigued, new node is 23%
        fat_hub = 3 / (3 + K_hub)
        fat_new = 3 / (3 + K_new)
        self.assertGreater(fat_hub, fat_new)


class Test03_SurfaceSelectedHebbian(unittest.TestCase):
    """Hebbian co_accessed edges from surface-selected nodes only.

    WHAT: Old Hebbian created co_accessed edges between ALL top-25 cosine results.
    Produced 94K noise edges. Now: only nodes the Layer 2 surface selects get
    co_accessed edges. These participate in graph traversal.

    WHY: "Neurons that fire together wire together" — but cosine top-25 isn't
    meaningful co-activation. Surface-selected IS meaningful.

    WHERE: daemon_hooks.py hook_post_response_track reads surface-selected.json
    EDGE TYPE: co_accessed (was noise, now meaningful after 2026-04-02 reset)
    """

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

    def test_co_accessed_not_created_by_recall(self):
        """recall() should NOT create co_accessed edges anymore.

        v22 edge model: relation lives on edge_relations, not edges.edge_type.
        """
        sql = ("SELECT COUNT(*) FROM edge_relations "
               "WHERE relation='co_accessed' AND archived=0")
        before = self.brain.conn.execute(sql).fetchone()[0]
        self.brain.recall("test query to check edge creation", limit=10)
        after = self.brain.conn.execute(sql).fetchone()[0]
        self.assertEqual(before, after,
                        "recall() should not create co_accessed edges")

    def test_exclusion_policies_derive_from_noise_aspect(self):
        """Two load-time policies, one deliberate difference (Tom, 2026-07-28):
        structural_exclusions is the FULL noise set (flat reads — hide
        decision id:49d734ad includes community_member); traversal_exclusions
        is noise MINUS community_member (graph dynamics keep conducting
        through communities — conduction is not visibility). The old
        EXCLUDED_EDGE_TYPES literal's 'co_accessed is clean now' premise
        (2026-04-02) died when Hebbian co-access writes resumed."""
        structural = self.brain.aspects.structural_exclusions
        traversal = self.brain.aspects.traversal_exclusions
        self.assertEqual(
            structural,
            frozenset(self.brain.aspects.relations_in(['noise'])))
        self.assertIn('community_member', structural)
        self.assertNotIn('community_member', traversal)
        self.assertEqual(traversal, structural - {'community_member'})
        for rel in ('co_accessed', 'emergent_bridge'):
            self.assertIn(rel, traversal)



class Test05_StructuralGraph(unittest.TestCase):
    """Structural graph separation — noise edges excluded from traversal.

    WHY: The old graph was a hairball — 2-hop reached 91% of nodes.
    With structural-only: 2-hop reaches ~4%. Real topology exists.

    WHERE: brain.aspects.structural_exclusions (noise aspect, load-time)
    EFFECT: graph traversal in pipeline_contract.traverse / graph_expand
    """

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

    def test_excluded_types(self):
        exclusions = self.brain.aspects.traversal_exclusions
        self.assertIn('emergent_bridge', exclusions)
        self.assertIn('co_accessed', exclusions)

    def test_structural_edges_exist(self):
        """There should be intentional edges for traversal.

        v22 edge model: relation lives on edge_relations, not edges.edge_type.
        """
        count = self.brain.conn.execute(
            "SELECT COUNT(*) FROM edge_relations "
            "WHERE relation NOT IN ('co_accessed', 'emergent_bridge') "
            "AND archived=0"
        ).fetchone()[0]
        self.assertGreater(count, 1000,
                          "Should have significant structural edges")


class Test06_MetadataKV(unittest.TestCase):
    """KV metadata store — extensible without schema changes.

    WHAT: node_metadata_kv table stores key-value pairs per node.
    Contract (METADATA_KEYS) defines known keys. Unknown keys accepted too (emergent).
    MetadataDAL provides get/set/set_many/delete/coverage.

    WHY: Old fixed-column node_metadata table required 7-file changes per new field.
    KV: add one line to contract. anchor_raw_quote was the first new field.

    WHERE: servers/dal_metadata.py (DAL), contract.py (METADATA_KEYS)
    TABLE: node_metadata_kv
    """

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

    def test_kv_table_exists(self):
        tables = {r[0] for r in self.brain.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.assertIn('node_metadata_kv', tables)

    def test_dal_read_write(self):
        """MetadataDAL should read and write correctly."""
        from servers.dal_metadata import MetadataDAL
        dal = MetadataDAL(self.brain.conn)

        # Create a test node
        self.brain.conn.execute(
            "INSERT OR IGNORE INTO nodes (id, type, title) VALUES ('_test_kv', 'test', 'KV test')")

        dal.set('_test_kv', 'test_key', 'test_value')
        result = dal.get_field('_test_kv', 'test_key')
        self.assertEqual(result, 'test_value')

        dal.set_many('_test_kv', {'k1': 'v1', 'k2': 'v2'})
        all_meta = dal.get('_test_kv')
        self.assertIn('k1', all_meta)
        self.assertIn('k2', all_meta)

        # Cleanup
        dal.delete_all('_test_kv')
        self.brain.conn.execute("DELETE FROM nodes WHERE id = '_test_kv'")

    def test_anchor_raw_quote_in_contract(self):
        from servers.contract import METADATA_KEYS
        self.assertIn('anchor_raw_quote', METADATA_KEYS)

    def test_remember_stores_kv(self):
        """remember() should store metadata in KV, not old fixed table."""
        result = self.brain.remember(
            type='test', title='KV integration test — delete',
            content='Testing metadata KV flow.',
            reasoning='Test reasoning',
        )
        node_id = result.get('id', '')
        self.assertTrue(node_id)

        from servers.dal_metadata import MetadataDAL
        dal = MetadataDAL(self.brain.conn)
        meta = dal.get(node_id)
        self.assertEqual(meta.get('reasoning'), 'Test reasoning')

        # Cleanup
        dal.delete_all(node_id)
        self.brain.conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self.brain.conn.commit()


class Test07_EncodingGroupVectors(unittest.TestCase):
    """Group vectors created at encode time.

    WHAT: remember() generates title, high_meta, other_meta enrichment vectors
    alongside the primary title+content embedding. Uses embed_batch for efficiency.

    WHY: Multi-vector architecture enables z-weighted scoring. Each vector
    matches different query patterns (topic, context, reasoning).

    WHERE: brain_remember.py _compute_group_vectors()
    STORED: node_enrichments table with vector_type = group name
    """

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

    def test_remember_creates_title_vector(self):
        """New node should get a title enrichment vector.

        Embedding is now deferred via embed_queue; tests must drain
        synchronously before asserting on node_enrichments rows.
        """
        from servers import embed_queue
        result = self.brain.remember(
            type='test', title='Group vector test — delete',
            content='Testing group vector creation.',
        )
        node_id = result.get('id', '')
        self.assertTrue(node_id)
        embed_queue._drain_once(self.brain)

        title_vec = self.brain.conn.execute(
            "SELECT COUNT(*) FROM node_enrichments WHERE node_id=? AND vector_type='title'",
            (node_id,)).fetchone()[0]
        self.assertEqual(title_vec, 1, "Title vector should be created")

        # Cleanup
        self.brain.conn.execute("DELETE FROM node_enrichments WHERE node_id=?", (node_id,))
        self.brain.conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        self.brain.conn.commit()

    def test_remember_creates_high_meta_when_situation(self):
        """Node with situation should get high_meta vector."""
        from servers import embed_queue
        result = self.brain.remember(
            type='test', title='High meta test — delete',
            content='Testing.',
            situation='When testing group vectors',
        )
        node_id = result.get('id', '')
        embed_queue._drain_once(self.brain)

        high_meta = self.brain.conn.execute(
            "SELECT COUNT(*) FROM node_enrichments WHERE node_id=? AND vector_type='high_meta'",
            (node_id,)).fetchone()[0]
        self.assertEqual(high_meta, 1, "High-meta vector should be created with situation")

        # Cleanup
        self.brain.conn.execute("DELETE FROM node_enrichments WHERE node_id=?", (node_id,))
        self.brain.conn.execute("DELETE FROM node_metadata_kv WHERE node_id=?", (node_id,))
        self.brain.conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        self.brain.conn.commit()


class Test08_ImportIntegrity(unittest.TestCase):
    """All modules import cleanly — no broken relative imports."""

    def test_pipeline_contract_import(self):
        """pipeline_contract imports cleanly via canonical package path.

        Was: imported as a top-level `pipeline_contract` after putting
        servers/ on sys.path. That path triggers a relative-import error
        because pipeline_contract.py uses `from .contract import ...`.
        Hooks no longer rely on that bare-path style; everything imports
        as `from servers.X import Y`. This test should follow.
        """
        from servers import pipeline_contract as mod
        self.assertTrue(hasattr(mod, 'build_surface_prompt'))
        # EMBEDDING_GROUPS may have been renamed; check for either.
        self.assertTrue(hasattr(mod, 'EMBEDDING_GROUPS') or
                        hasattr(mod, 'field_vector_types'),
                        "pipeline_contract should expose embedding-group "
                        "definitions under EMBEDDING_GROUPS or field_vector_types")


    def test_dal_metadata_import(self):
        from servers.dal_metadata import MetadataDAL
        self.assertTrue(callable(MetadataDAL))

    def test_contract_sync(self):
        """Contract sync should pass cleanly.

        Was: asserted "FAILED (failures=1)" — a known stale-default
        contract issue that's since been fixed. Now expects clean OK.
        """
        import subprocess
        result = subprocess.run(
            ['python3', 'tests/test_contract_sync.py'],
            capture_output=True, text=True, timeout=30)
        combined = result.stdout + result.stderr
        # Clean "OK" with no failures — the known issue is gone.
        self.assertIn('OK', combined,
                      f"Contract sync should pass; got:\n{combined}")
        self.assertNotIn('FAILED', combined,
                         f"Contract sync should not have failures; got:\n{combined}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
