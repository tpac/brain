"""
MCP Round-Trip Tests — verifies every MCP tool works end-to-end.

Each tool exposed via brain_mcp.py is tested through the daemon dispatch
layer against a real Brain instance. This catches:
  - MCP tool → daemon command mapping failures
  - Parameter name mismatches between MCP schema and handler
  - Brain method errors not surfaced properly
  - Missing dispatch handlers for new tools

Run: python3 -m pytest tests/test_mcp_roundtrip.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_mcp import TOOLS
from servers.daemon_dispatch import COMMAND_TABLE


class TestMCPRoundTrip(BrainTestBase):
    """Test every MCP tool through daemon dispatch against a real brain."""

    def _dispatch(self, cmd, args):
        """Simulate daemon dispatch — call handler, unwrap result envelope."""
        entry = COMMAND_TABLE.get(cmd)
        self.assertIsNotNone(entry, "No dispatch handler for: %s" % cmd)
        raw = entry.handler(self.brain, args, [])
        # Handlers return {"ok": True, "result": {...}} — unwrap
        if isinstance(raw, dict) and "result" in raw:
            return raw["result"]
        return raw

    # ── Core memory operations ──

    def test_recall(self):
        """recall returns results list."""
        result = self._dispatch("recall", {"query": "test", "limit": 3})
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)

    def test_remember(self):
        """remember stores a node and returns its ID."""
        result = self._dispatch("remember", {
            "type": "lesson", "title": "Test lesson",
            "content": "This is a test lesson for round-trip verification."
        })
        self.assertIn("id", result)
        self.assertTrue(len(result["id"]) > 0)

    def test_remember_then_recall(self):
        """remember → recall: stored node should be findable."""
        r1 = self._dispatch("remember", {
            "type": "decision", "title": "Use Arctic v1.5 for embeddings",
            "content": "Chose Arctic v1.5 because it balances quality and speed."
        })
        node_id = r1["id"]

        r2 = self._dispatch("recall", {"query": "Arctic embedding model choice", "limit": 5})
        found_ids = [r["id"] for r in r2["results"]]
        self.assertIn(node_id, found_ids,
                      "Freshly stored node not found by recall — embedding may have failed")

    def test_connect(self):
        """connect creates an edge between two nodes."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "Node A", "content": "First node"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "Node B", "content": "Second node"})
        self._dispatch("connect", {
            "source_id": n1["id"], "target_id": n2["id"],
            "relation": "related_to", "weight": 0.8
        })
        # Verify edge exists in DB
        edge = self.brain.conn.execute(
            "SELECT relation, weight FROM edges WHERE source_id = ? AND target_id = ?",
            (n1["id"], n2["id"])).fetchone()
        self.assertIsNotNone(edge, "Edge should exist after connect")
        self.assertEqual(edge[0], "related_to")

    def test_enrich(self):
        """enrich stores enrichment vectors for a node."""
        n = self._dispatch("remember", {"type": "lesson", "title": "Enrich test", "content": "Test content"})
        result = self._dispatch("enrich", {
            "node_id": n["id"],
            "question": "What was the enrich test about?",
            "keywords": "enrich, test, verification"
        })
        self.assertIn("enrichments_stored", result)

    # ── Specialized encoding (promoted from eval) ──

    def test_remember_lesson(self):
        """remember_lesson stores structured lesson with all fields."""
        result = self._dispatch("remember_lesson", {
            "title": "Lock timeout prevents deadlocks",
            "what_happened": "Write lock had no timeout, causing hangs",
            "root_cause": "threading.Lock.acquire() blocks forever by default",
            "fix": "Added timeout=10.0 to acquire calls",
            "preventive_principle": "All lock acquisitions need timeouts"
        })
        self.assertIn("id", result)
        # Verify it's locked (lessons are auto-locked)
        node = self.brain.conn.execute(
            "SELECT locked FROM nodes WHERE id = ?", (result["id"],)).fetchone()
        self.assertTrue(node[0], "Lessons should be auto-locked")

    def test_remember_impact(self):
        """remember_impact stores dependency tracking."""
        result = self._dispatch("remember_impact", {
            "title": "daemon_config change requires daemon restart",
            "if_changed": "DAEMON_HOST in daemon_config.py",
            "must_check": "All daemon clients, MCP server, dashboard",
            "because": "Host is read at import time, cached in module scope"
        })
        self.assertIn("id", result)

    def test_remember_mechanism(self):
        """remember_mechanism stores how something works."""
        result = self._dispatch("remember_mechanism", {
            "title": "MCP retry flow",
            "content": "MCP server retries 3 times with backoff on daemon connection failure.",
            "steps": ["Send command", "On failure: wait 0.5s", "Restart daemon", "Retry"],
        })
        self.assertIn("id", result)

    def test_remember_convention(self):
        """remember_convention stores coding patterns."""
        result = self._dispatch("remember_convention", {
            "title": "Lock acquire always with timeout",
            "content": "Never call lock.acquire() without timeout parameter.",
            "pattern": "lock.acquire(timeout=10.0)",
            "anti_pattern": "lock.acquire()  # blocks forever"
        })
        self.assertIn("id", result)

    def test_remember_uncertainty(self):
        """remember_uncertainty stores honest not-knowing."""
        result = self._dispatch("remember_uncertainty", {
            "title": "Is 10s lock timeout optimal?",
            "what_unknown": "Whether 10 seconds is the right timeout for write lock",
            "why_it_matters": "Too short = false failures, too long = perceived hangs"
        })
        self.assertIn("id", result)
        # Verify low confidence (uncertainties auto-set 0.3)
        node = self.brain.conn.execute(
            "SELECT confidence FROM nodes WHERE id = ?", (result["id"],)).fetchone()
        self.assertLessEqual(node[0], 0.5, "Uncertainties should have low confidence")

    def test_remember_mental_model(self):
        """remember_mental_model stores understanding of systems."""
        result = self._dispatch("remember_mental_model", {
            "title": "Daemon has 3 responsibilities",
            "model_description": "The daemon: 1) keeps Brain loaded in memory, 2) serializes writes, 3) serves the dashboard. These should be separated.",
            "applies_to": "servers/daemon_server.py",
            "confidence": 0.8
        })
        self.assertIn("id", result)

    def test_record_divergence(self):
        """record_divergence tracks corrections."""
        result = self._dispatch("record_divergence", {
            "claude_assumed": "DAEMON_HOST=127.0.0.1 works for all localhost connections",
            "reality": "macOS resolves localhost to ::1 (IPv6), so 127.0.0.1 binding misses it",
            "underlying_pattern": "IPv4-only assumptions break on dual-stack systems",
            "severity": "medium"
        })
        self.assertIn("id", result)

    def test_learn_vocabulary(self):
        """learn_vocabulary maps operator terms."""
        result = self._dispatch("learn_vocabulary", {
            "term": "the anchor",
            "maps_to": "Claude's persistent identity across sessions via brain + SKILL.md",
            "context": "Tom named the continuity mechanism 'Anchor' — it's who Claude chooses to be"
        })
        self.assertIn("id", result)

    # ── Introspection ──

    def test_consciousness(self):
        """consciousness returns signal categories."""
        result = self._dispatch("consciousness", {})
        # Should return a dict with signal categories
        self.assertIsInstance(result, dict)

    def test_engineering_context(self):
        """engineering_context returns project-scoped memory."""
        result = self._dispatch("engineering_context", {"project": "brain"})
        self.assertIsInstance(result, dict)

    # ── Compound operations ──

    def test_find_node_by_title(self):
        """find_node_by_title locates nodes by fuzzy title matching with context."""
        self._dispatch("remember", {
            "type": "decision", "title": "Use Arctic v1.5 for embeddings",
            "content": "Chose Arctic v1.5 because it balances quality and speed."
        })
        result = self._dispatch("find_node_by_title", {
            "title_query": "Arctic embedding model", "threshold": 0.5
        })
        self.assertIsNotNone(result, "Should find a node matching 'Arctic embedding model'")
        self.assertIn("id", result)
        self.assertIn("content_snippet", result, "Should include content snippet for verification")
        self.assertIn("keywords", result, "Should include keywords for verification")
        self.assertGreater(result["similarity"], 0.5)

    def test_find_node_by_title_no_match(self):
        """find_node_by_title returns None when nothing matches."""
        result = self._dispatch("find_node_by_title", {
            "title_query": "completely unrelated xyzzy topic", "threshold": 0.9
        })
        self.assertIsNone(result)

    def test_encode_cluster(self):
        """encode_cluster stores multiple nodes with connections in one call."""
        # Seed an existing node to connect to
        self._dispatch("remember", {
            "type": "decision", "title": "Brain uses SQLite with WAL mode",
            "content": "WAL allows concurrent readers with one writer."
        })

        result = self._dispatch("encode_cluster", {
            "nodes": [
                {"type": "lesson", "title": "Lock timeout prevents deadlocks",
                 "content": "Write lock had no timeout, could hang forever.",
                 "enrichment": {"question": "What prevents daemon deadlocks?", "keywords": "lock, timeout, deadlock"}},
                {"type": "mechanism", "title": "Autosave loop with health check",
                 "content": "Every 60s: save if dirty, verify SQLite alive."},
            ],
            "connect_to": ["Brain uses SQLite"],
            "auto_connect": True
        })
        self.assertEqual(result["nodes_created"], 2)
        self.assertGreater(result["connections_created"], 0, "Should create inter-cluster + connect_to edges")
        self.assertEqual(len(result["node_ids"]), 2)

    def test_encode_cluster_detects_duplicates(self):
        """encode_cluster warns about near-duplicate nodes."""
        self._dispatch("remember", {
            "type": "lesson", "title": "Lock timeout prevents deadlocks",
            "content": "Original lesson about lock timeouts."
        })
        result = self._dispatch("encode_cluster", {
            "nodes": [
                {"type": "lesson", "title": "Lock timeout prevents deadlock issues",
                 "content": "Very similar to existing node."}
            ]
        })
        self.assertEqual(result["nodes_created"], 1)
        # Should detect the near-duplicate
        self.assertGreater(len(result["duplicates"]), 0, "Should detect near-duplicate title")

    # ── Safety: find_node_by_title edge cases ──

    def test_find_node_does_not_match_self(self):
        """find_node_by_title shouldn't cause self-referential connections in encode_cluster."""
        r = self._dispatch("remember", {
            "type": "lesson", "title": "Unique title for self-test",
            "content": "Content for self-reference test."
        })
        # Searching for exact same title should find the node
        match = self._dispatch("find_node_by_title", {
            "title_query": "Unique title for self-test", "threshold": 0.5
        })
        self.assertIsNotNone(match)
        self.assertEqual(match["id"], r["id"])

    def test_find_node_high_threshold_rejects_weak_matches(self):
        """High threshold prevents false matches that would pollute connections."""
        self._dispatch("remember", {
            "type": "lesson", "title": "Daemon race condition fix",
            "content": "PID written before socket bound."
        })
        # A loosely related query should NOT match at high threshold
        result = self._dispatch("find_node_by_title", {
            "title_query": "HTTP server configuration", "threshold": 0.85
        })
        self.assertIsNone(result, "Weak semantic match should be rejected at 0.85 threshold")

    def test_find_node_returns_best_not_first(self):
        """find_node_by_title returns highest similarity, not first found."""
        self._dispatch("remember", {"type": "concept", "title": "Database indexing strategies", "content": "B-trees vs hash indexes"})
        self._dispatch("remember", {"type": "lesson", "title": "SQLite WAL mode for concurrent access", "content": "WAL allows readers and writers"})
        result = self._dispatch("find_node_by_title", {
            "title_query": "SQLite WAL concurrent", "threshold": 0.5
        })
        self.assertIsNotNone(result)
        self.assertIn("WAL", result["title"], "Should match the WAL node, not the generic indexing node")

    # ── Safety: encode_cluster edge cases ──

    def test_encode_cluster_empty_nodes(self):
        """encode_cluster with empty nodes list creates nothing."""
        result = self._dispatch("encode_cluster", {"nodes": []})
        self.assertEqual(result["nodes_created"], 0)
        self.assertEqual(result["connections_created"], 0)

    def test_encode_cluster_single_node(self):
        """encode_cluster with one node doesn't create self-connections."""
        result = self._dispatch("encode_cluster", {
            "nodes": [{"type": "lesson", "title": "Solo node test", "content": "Just one."}],
            "auto_connect": False
        })
        self.assertEqual(result["nodes_created"], 1)
        # Single node + no auto_connect = 0 inter-cluster connections
        self.assertEqual(result["connections_created"], 0)

    def test_encode_cluster_connect_to_nonexistent(self):
        """encode_cluster with connect_to that doesn't match reports in missing."""
        # Seed a node so there's something in the brain, but not a match for our query
        self._dispatch("remember", {
            "type": "decision", "title": "Use PostgreSQL for production database",
            "content": "PostgreSQL chosen for ACID compliance."
        })
        result = self._dispatch("encode_cluster", {
            "nodes": [{"type": "concept", "title": "Unrelated topic about gardening", "content": "How to grow tomatoes."}],
            "connect_to": ["Quantum physics string theory multiverse"],
            "auto_connect": False
        })
        self.assertEqual(result["nodes_created"], 1)
        # Quantum physics shouldn't match PostgreSQL at threshold 0.75
        self.assertTrue(
            any("no match" in m for m in result["missing"]) or len(result.get("connected_to", [])) == 0,
            "Should report no match for unrelated connect_to query")

    def test_encode_cluster_missing_enrichment_reported(self):
        """encode_cluster reports nodes without enrichments in missing."""
        result = self._dispatch("encode_cluster", {
            "nodes": [
                {"type": "lesson", "title": "Has enrichment", "content": "Content.",
                 "enrichment": {"question": "What is this?"}},
                {"type": "lesson", "title": "No enrichment", "content": "Content."},
            ],
            "auto_connect": False
        })
        self.assertEqual(result["nodes_created"], 2)
        self.assertTrue(any("No enrichment" in m for m in result["missing"]),
                        "Should report node without enrichment")

    def test_encode_cluster_inter_cluster_connections(self):
        """encode_cluster connects all nodes in the cluster to each other."""
        result = self._dispatch("encode_cluster", {
            "nodes": [
                {"type": "concept", "title": "Node Alpha", "content": "First."},
                {"type": "concept", "title": "Node Beta", "content": "Second."},
                {"type": "concept", "title": "Node Gamma", "content": "Third."},
            ],
            "auto_connect": False
        })
        self.assertEqual(result["nodes_created"], 3)
        # 3 nodes = 3 inter-cluster pairs: A-B, A-C, B-C
        self.assertGreaterEqual(result["connections_created"], 3)
        # Verify edges exist in DB
        ids = result["node_ids"]
        for i, src in enumerate(ids):
            for dst in ids[i+1:]:
                edge = self.brain.conn.execute(
                    "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ?",
                    (src, dst)).fetchone()
                self.assertIsNotNone(edge, "Missing edge between cluster nodes %s → %s" % (src[:8], dst[:8]))

    # ── Escape hatch ──

    def test_eval(self):
        """eval executes arbitrary Python on brain object."""
        result = self._dispatch("eval", {"code": "brain.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]"})
        # eval returns the raw Python value (unwrapped from ok/result envelope)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    # ── Coverage check ──

    def test_all_mcp_tools_have_roundtrip_tests(self):
        """Every MCP tool should have a corresponding test above."""
        mcp_tool_names = {t["name"] for t in TOOLS}
        test_methods = {m for m in dir(self) if m.startswith("test_") and m != "test_all_mcp_tools_have_roundtrip_tests"}

        # Map test method names to tool names they cover
        tested = set()
        for method in test_methods:
            # test_remember → "remember", test_remember_lesson → "remember_lesson"
            tool_name = method.replace("test_", "").replace("_then_recall", "")
            if tool_name in mcp_tool_names:
                tested.add(tool_name)

        untested = mcp_tool_names - tested
        self.assertEqual(untested, set(),
                         "MCP tools without round-trip tests: %s" % untested)


if __name__ == "__main__":
    unittest.main()
