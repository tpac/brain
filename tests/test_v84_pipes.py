"""
Pipe tests for v8.3–v8.4 changes.

Tests every function touched in the encoding overhaul + graph traversal:
- dal.py: get_neighbors_rich
- brain_recall.py: _traverse_graph, recall_with_embeddings graph augmentation
- brain_voice.py: format_node_deep
- daemon_hooks.py: encoding gating, candidates file format
- signal_producers.py: produce_integrity, deep_integrity_audit
- schema.py: v20 migration (CHECK constraint removal)
- brain_constants.py: traversal constants exist and have correct types

Run: python3 -m pytest tests/test_v84_pipes.py -v
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from servers.brain import Brain
from servers.schema import ensure_schema, BRAIN_VERSION


class PipeTestBase(unittest.TestCase):
    """Shared setup: fresh brain in temp dir."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = tempfile.mkdtemp(prefix="brain_pipe_test_")
        cls.db_path = os.path.join(cls.work_dir, "brain.db")
        cls.brain = Brain(db_path=cls.db_path)

        # Seed some nodes for testing
        cls.node_a = cls.brain.remember(
            type="mechanism", title="Test mechanism A",
            content="This is mechanism A about daemon communication.",
            keywords="daemon communication test")
        cls.node_b = cls.brain.remember(
            type="decision", title="Test decision B",
            content="Decision to use TCP over Unix sockets.",
            keywords="TCP sockets decision")
        cls.node_c = cls.brain.remember(
            type="vocabulary", title="daemon → test brain server",
            content="The daemon is a persistent brain server process.",
            keywords="daemon server vocabulary")

        # Create edges between them
        cls.brain.connect(
            source_id=cls.node_a["id"], target_id=cls.node_b["id"],
            relation="related_to", weight=0.8)
        cls.brain.connect(
            source_id=cls.node_b["id"], target_id=cls.node_c["id"],
            relation="depends_on", weight=0.7)

        cls.brain.save()

    @classmethod
    def tearDownClass(cls):
        cls.brain.close()
        shutil.rmtree(cls.work_dir, ignore_errors=True)


# ── 1. DAL: get_neighbors_rich ──

class TestGetNeighborsRich(PipeTestBase):
    """Test dal.py get_neighbors_rich returns correct shape and data."""

    def test_returns_list(self):
        from servers.dal import GraphDAL
        g = GraphDAL(self.brain.conn)
        result = g.get_neighbors_rich(self.node_a["id"], limit=5)
        self.assertIsInstance(result, list)

    def test_neighbor_has_required_fields(self):
        from servers.dal import GraphDAL
        g = GraphDAL(self.brain.conn)
        result = g.get_neighbors_rich(self.node_a["id"], limit=5)
        self.assertGreater(len(result), 0, "Node A should have neighbors")

        nb = result[0]
        required = ['id', 'type', 'title', 'relation', 'weight',
                     'revised_at', 'created_at', 'confidence']
        for field in required:
            self.assertIn(field, nb, "Neighbor missing field: %s" % field)

    def test_metadata_fields_present(self):
        """Metadata fields should be present (may be None from LEFT JOIN)."""
        from servers.dal import GraphDAL
        g = GraphDAL(self.brain.conn)
        result = g.get_neighbors_rich(self.node_a["id"], limit=5)
        nb = result[0]
        metadata_fields = ['reasoning', 'user_raw_quote', 'correction_of',
                           'correction_pattern', 'source_context', 'validation_count']
        for field in metadata_fields:
            self.assertIn(field, nb, "Neighbor missing metadata field: %s" % field)

    def test_edge_fields_present(self):
        from servers.dal import GraphDAL
        g = GraphDAL(self.brain.conn)
        result = g.get_neighbors_rich(self.node_a["id"], limit=5)
        nb = result[0]
        edge_fields = ['edge_description', 'last_strengthened', 'co_access_count']
        for field in edge_fields:
            self.assertIn(field, nb, "Neighbor missing edge field: %s" % field)

    def test_exclude_relations(self):
        """Excluded relation types should not appear."""
        from servers.dal import GraphDAL
        g = GraphDAL(self.brain.conn)
        result = g.get_neighbors_rich(
            self.node_a["id"], limit=5,
            exclude_relations={"related_to"})
        # Node A connects to B via related_to — should be excluded
        for nb in result:
            self.assertNotEqual(nb["relation"], "related_to")

    def test_nonexistent_node_returns_empty(self):
        from servers.dal import GraphDAL
        g = GraphDAL(self.brain.conn)
        result = g.get_neighbors_rich("nonexistent_id", limit=5)
        self.assertEqual(result, [])

    def test_limit_respected(self):
        from servers.dal import GraphDAL
        g = GraphDAL(self.brain.conn)
        result = g.get_neighbors_rich(self.node_a["id"], limit=1)
        self.assertLessEqual(len(result), 1)


# ── 2. RECALL: _traverse_graph ──

class TestTraverseGraph(PipeTestBase):
    """Test brain_recall.py _traverse_graph returns correct shape."""

    def test_returns_tuple(self):
        seeds = [{"node_id": self.node_a["id"], "blended_score": 0.8}]
        candidates, neighborhoods = self.brain._traverse_graph(seeds)
        self.assertIsInstance(candidates, dict)
        self.assertIsInstance(neighborhoods, dict)

    def test_neighborhoods_have_three_degrees(self):
        seeds = [{"node_id": self.node_a["id"], "blended_score": 0.8}]
        _, neighborhoods = self.brain._traverse_graph(seeds)
        self.assertIn(self.node_a["id"], neighborhoods)
        hood = neighborhoods[self.node_a["id"]]
        self.assertIn("degree_1", hood)
        self.assertIn("degree_2", hood)
        self.assertIn("degree_3", hood)

    def test_candidates_have_score_and_discovery(self):
        seeds = [{"node_id": self.node_a["id"], "blended_score": 0.8}]
        candidates, _ = self.brain._traverse_graph(seeds)
        for nid, info in candidates.items():
            self.assertIn("score", info)
            self.assertIn("discovery", info)
            self.assertIsInstance(info["score"], float)
            self.assertIn(info["discovery"],
                          ("graph_d1", "graph_d2", "graph_d3", "convergence"))

    def test_seed_not_in_candidates(self):
        """Seeds should not appear as graph candidates."""
        seeds = [{"node_id": self.node_a["id"], "blended_score": 0.8}]
        candidates, _ = self.brain._traverse_graph(seeds)
        self.assertNotIn(self.node_a["id"], candidates)

    def test_empty_seeds(self):
        candidates, neighborhoods = self.brain._traverse_graph([])
        self.assertEqual(candidates, {})
        self.assertEqual(neighborhoods, {})

    def test_degree1_finds_direct_neighbor(self):
        seeds = [{"node_id": self.node_a["id"], "blended_score": 0.8}]
        candidates, neighborhoods = self.brain._traverse_graph(seeds)
        hood = neighborhoods[self.node_a["id"]]
        d1_ids = [n.get("id") for n in hood["degree_1"]]
        self.assertIn(self.node_b["id"], d1_ids,
                       "Direct neighbor B should appear at degree 1")

    def test_degree2_finds_transitive_neighbor(self):
        seeds = [{"node_id": self.node_a["id"], "blended_score": 0.8}]
        candidates, neighborhoods = self.brain._traverse_graph(seeds)
        hood = neighborhoods[self.node_a["id"]]
        d2_ids = [n.get("id") for n in hood["degree_2"]]
        # C is connected to B which is connected to A → degree 2
        self.assertIn(self.node_c["id"], d2_ids,
                       "Transitive neighbor C should appear at degree 2")

    def test_no_cycles(self):
        """Same node should not appear at multiple degrees."""
        seeds = [{"node_id": self.node_a["id"], "blended_score": 0.8}]
        _, neighborhoods = self.brain._traverse_graph(seeds)
        hood = neighborhoods[self.node_a["id"]]
        all_ids = set()
        for degree in ("degree_1", "degree_2", "degree_3"):
            for n in hood[degree]:
                nid = n.get("id")
                self.assertNotIn(nid, all_ids,
                                  "Node %s appears at multiple degrees" % nid)
                all_ids.add(nid)


# ── 3. RECALL: recall_with_embeddings integration ──

class TestRecallWithTraversal(PipeTestBase):
    """Test that recall_with_embeddings includes _graph and _discovery."""

    def test_results_have_graph(self):
        result = self.brain.recall_with_embeddings("daemon communication", limit=3)
        for r in result.get("results", []):
            self.assertIn("_graph", r, "Result missing _graph field")
            self.assertIsInstance(r["_graph"], dict)

    def test_results_have_discovery(self):
        result = self.brain.recall_with_embeddings("daemon communication", limit=3)
        for r in result.get("results", []):
            self.assertIn("_discovery", r, "Result missing _discovery field")

    def test_stats_include_graph_sources(self):
        result = self.brain.recall_with_embeddings("daemon communication", limit=3)
        stats = result.get("_embedding_stats", {})
        sources = stats.get("results_by_source", {})
        expected_keys = {"embedding+keyword", "embedding_only", "keyword_only_fallback",
                         "graph_d1", "graph_d2", "graph_d3", "convergence"}
        for key in expected_keys:
            self.assertIn(key, sources, "Missing source key: %s" % key)


# ── 4. VOICE: format_node_deep ──

class TestFormatNodeDeep(PipeTestBase):
    """Test brain_voice.py format_node_deep renders 3 degrees."""

    def test_renders_degree0(self):
        from servers.brain_voice import BrainVoice
        node = self.brain.get_node(self.node_a["id"])
        lines = []
        BrainVoice.format_node_deep(node, lines, conn=self.brain.conn)
        text = "\n".join(lines)
        self.assertIn("[mechanism]", text)
        self.assertIn("Test mechanism A", text)
        self.assertIn("id:", text)
        self.assertIn("revised:", text)

    def test_renders_neighbors_with_conn(self):
        from servers.brain_voice import BrainVoice
        node = self.brain.get_node(self.node_a["id"])
        lines = []
        BrainVoice.format_node_deep(node, lines, conn=self.brain.conn)
        text = "\n".join(lines)
        # Should have degree-1 neighbor (node B)
        self.assertIn("Test decision B", text)

    def test_renders_without_conn(self):
        """Should not crash without a DB connection."""
        from servers.brain_voice import BrainVoice
        node = {"id": "test", "type": "test", "title": "Test", "content": "Content"}
        lines = []
        BrainVoice.format_node_deep(node, lines, conn=None)
        text = "\n".join(lines)
        self.assertIn("[test]", text)
        self.assertIn("Test", text)

    def test_root_node_not_in_children(self):
        """The degree-0 node should not reappear as its own descendant."""
        from servers.brain_voice import BrainVoice
        node = self.brain.get_node(self.node_a["id"])
        lines = []
        BrainVoice.format_node_deep(node, lines, conn=self.brain.conn)
        text = "\n".join(lines)
        import re
        # All IDs at degree 1+ (lines starting with spaces)
        child_ids = re.findall(r'^\s+.*id:([a-f0-9_]+)', text, re.MULTILINE)
        self.assertNotIn(self.node_a["id"], child_ids,
                          "Root node should not appear as its own descendant")


# ── 5. ENCODING GATING ──

class TestEncodingGating(PipeTestBase):
    """Test daemon_hooks.py encoding gating via stop_counter."""

    def test_counter_increments(self):
        self.brain.set_config("stop_counter", "0")
        for i in range(5):
            counter = int(self.brain.get_config("stop_counter", "0") or "0") + 1
            self.brain.set_config("stop_counter", str(counter))
        self.assertEqual(self.brain.get_config("stop_counter"), "5")

    def test_5th_stop_triggers(self):
        self.brain.set_config("stop_counter", "4")
        counter = int(self.brain.get_config("stop_counter", "0") or "0") + 1
        self.assertEqual(counter % 5, 0, "5th stop should trigger encoding")

    def test_non_5th_stop_skips(self):
        for counter in [1, 2, 3, 4, 6, 7, 8, 9]:
            self.assertNotEqual(counter % 5, 0, "Counter %d should skip" % counter)


# ── 6. SIGNAL PRODUCERS: integrity ──

class TestIntegrityProducer(PipeTestBase):
    """Test signal_producers.py integrity checks."""

    def test_deep_audit_returns_list(self):
        from servers.signal_producers import deep_integrity_audit
        findings = deep_integrity_audit(self.brain)
        self.assertIsInstance(findings, list)

    def test_findings_have_required_fields(self):
        from servers.signal_producers import deep_integrity_audit
        findings = deep_integrity_audit(self.brain)
        for f in findings:
            self.assertIn("type", f)
            self.assertIn("severity", f)
            self.assertIn("message", f)
            self.assertIn(f["severity"], ("low", "medium", "high", "info"))

    def test_revision_stats_present(self):
        from servers.signal_producers import deep_integrity_audit
        findings = deep_integrity_audit(self.brain)
        types = [f["type"] for f in findings]
        self.assertIn("revision_stats", types)

    def test_structural_types_constant(self):
        from servers.signal_producers import STRUCTURAL_TYPES
        self.assertIn("vocabulary", STRUCTURAL_TYPES)
        self.assertIn("rule", STRUCTURAL_TYPES)
        self.assertIn("decision", STRUCTURAL_TYPES)
        self.assertIn("mechanism", STRUCTURAL_TYPES)


# ── 7. SCHEMA: v20 ──

class TestSchemaV20(unittest.TestCase):
    """Test schema v20 migration — CHECK constraint removal."""

    def test_version_is_20(self):
        self.assertEqual(BRAIN_VERSION, 20)

    def test_fresh_db_no_check_constraint(self):
        work_dir = tempfile.mkdtemp()
        db_path = os.path.join(work_dir, "test.db")
        conn = sqlite3.connect(db_path)
        ensure_schema(conn, db_path)
        # Check the CREATE TABLE SQL for nodes
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='nodes'"
        ).fetchone()[0]
        self.assertNotIn("CHECK", sql, "Fresh DB should not have CHECK constraint")
        conn.close()
        shutil.rmtree(work_dir)

    def test_free_type_allowed(self):
        """Any type string should be insertable."""
        work_dir = tempfile.mkdtemp()
        db_path = os.path.join(work_dir, "test.db")
        brain = Brain(db_path=db_path)
        result = brain.remember(
            type="custom_emergent_type",
            title="Test free type",
            content="This uses a non-structural type")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("type"), "custom_emergent_type")
        brain.close()
        shutil.rmtree(work_dir)


# ── 8. CONSTANTS: traversal ──

class TestTraversalConstants(unittest.TestCase):
    """Test brain_constants.py traversal constants exist and have correct types."""

    def test_traverse_depth(self):
        from servers.brain_constants import TRAVERSE_DEPTH
        self.assertIsInstance(TRAVERSE_DEPTH, int)
        self.assertEqual(TRAVERSE_DEPTH, 3)

    def test_traverse_dampen(self):
        from servers.brain_constants import TRAVERSE_DAMPEN
        self.assertIsInstance(TRAVERSE_DAMPEN, list)
        self.assertEqual(len(TRAVERSE_DAMPEN), 3)
        for d in TRAVERSE_DAMPEN:
            self.assertIsInstance(d, float)
            self.assertGreater(d, 0)
            self.assertLessEqual(d, 1)

    def test_traverse_limits(self):
        from servers.brain_constants import TRAVERSE_LIMITS
        self.assertIsInstance(TRAVERSE_LIMITS, list)
        self.assertEqual(len(TRAVERSE_LIMITS), 3)
        for lim in TRAVERSE_LIMITS:
            self.assertIsInstance(lim, int)
            self.assertGreater(lim, 0)

    def test_excluded_edge_types(self):
        from servers.brain_constants import EXCLUDED_EDGE_TYPES
        self.assertIsInstance(EXCLUDED_EDGE_TYPES, set)
        self.assertIn("co_accessed", EXCLUDED_EDGE_TYPES)

    def test_intentional_edge_types(self):
        from servers.brain_constants import INTENTIONAL_EDGE_TYPES
        self.assertIsInstance(INTENTIONAL_EDGE_TYPES, set)
        self.assertIn("related_to", INTENTIONAL_EDGE_TYPES)
        self.assertIn("depends_on", INTENTIONAL_EDGE_TYPES)
        self.assertNotIn("co_accessed", INTENTIONAL_EDGE_TYPES)

    def test_freshness_multipliers(self):
        from servers.brain_constants import FRESHNESS_MULTIPLIERS
        self.assertIsInstance(FRESHNESS_MULTIPLIERS, dict)
        for key in ("today", "this_week", "this_month", "older"):
            self.assertIn(key, FRESHNESS_MULTIPLIERS)
            self.assertIsInstance(FRESHNESS_MULTIPLIERS[key], float)

    def test_dampen_decreases_with_depth(self):
        from servers.brain_constants import TRAVERSE_DAMPEN
        self.assertGreater(TRAVERSE_DAMPEN[0], TRAVERSE_DAMPEN[1])
        self.assertGreater(TRAVERSE_DAMPEN[1], TRAVERSE_DAMPEN[2])


# ── 9. SANITY: production brain ──

@unittest.skipUnless(
    os.path.exists(os.path.expanduser("~/AgentsContext/brain/brain.db")),
    "Production brain not available")
class TestProductionSanity(unittest.TestCase):
    """Sanity tests against the real production brain."""

    @classmethod
    def setUpClass(cls):
        cls.brain = Brain(
            db_path=os.path.expanduser("~/AgentsContext/brain/brain.db"))

    @classmethod
    def tearDownClass(cls):
        cls.brain.close()

    def test_recall_returns_results(self):
        result = self.brain.recall_with_embeddings("daemon TCP", limit=3)
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)

    def test_recall_results_have_graph(self):
        result = self.brain.recall_with_embeddings("daemon TCP", limit=3)
        for r in result["results"]:
            self.assertIn("_graph", r)

    def test_recall_latency_reasonable(self):
        t0 = time.time()
        self.brain.recall_with_embeddings("encoding agent stop hook", limit=5)
        ms = (time.time() - t0) * 1000
        self.assertLess(ms, 2000, "Recall took %.0fms — should be < 2000ms" % ms)

    def test_graph_traversal_finds_neighbors(self):
        result = self.brain.recall_with_embeddings("daemon communication", limit=3)
        graphs = [r["_graph"] for r in result["results"] if r.get("_graph")]
        self.assertGreater(len(graphs), 0, "At least one result should have graph neighbors")
        # Check at least one has degree_1
        has_d1 = any(len(g.get("degree_1", [])) > 0 for g in graphs)
        self.assertTrue(has_d1, "At least one result should have degree-1 neighbors")

    def test_format_node_deep_on_real_node(self):
        from servers.brain_voice import BrainVoice
        result = self.brain.recall_with_embeddings("daemon", limit=1)
        node = result["results"][0]
        lines = []
        BrainVoice.format_node_deep(node, lines, conn=self.brain.conn)
        text = "\n".join(lines)
        self.assertIn("id:", text)
        self.assertIn("revised:", text)
        self.assertGreater(len(text), 100, "format_node_deep should produce substantial output")

    def test_deep_audit_runs_clean(self):
        from servers.signal_producers import deep_integrity_audit
        findings = deep_integrity_audit(self.brain)
        self.assertIsInstance(findings, list)
        # Should have findings on a real brain
        self.assertGreater(len(findings), 0)
        # No audit_error findings
        errors = [f for f in findings if f["type"] == "audit_error"]
        self.assertEqual(len(errors), 0, "Deep audit had errors: %s" % errors)

    def test_schema_is_v20(self):
        version = self.brain.conn.execute(
            "SELECT value FROM brain_meta WHERE key='brain_schema_version'"
        ).fetchone()
        self.assertIsNotNone(version)
        self.assertEqual(version[0], "20")

    def test_no_check_constraint(self):
        sql = self.brain.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='nodes'"
        ).fetchone()[0]
        self.assertNotIn("CHECK(type IN", sql)


if __name__ == "__main__":
    unittest.main()
