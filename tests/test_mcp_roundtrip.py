"""
MCP Round-Trip Tests — verifies every MCP tool works end-to-end.

Each tool exposed via brain_mcp.py is tested through the daemon dispatch
layer against a real Brain instance. This catches:
  - MCP tool -> daemon command mapping failures
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
        """remember -> recall: stored node should be findable."""
        r1 = self._dispatch("remember", {
            "type": "decision", "title": "Use Arctic v1.5 for embeddings",
            "content": "Chose Arctic v1.5 because it balances quality and speed."
        })
        node_id = r1["id"]

        r2 = self._dispatch("recall", {"query": "Arctic embedding model choice", "limit": 5})
        found_ids = [r["id"] for r in r2["results"]]
        self.assertIn(node_id, found_ids,
                      "Freshly stored node not found by recall — embedding may have failed")

    def test_remember_batch(self):
        """remember_batch stores multiple nodes in one call."""
        result = self._dispatch("remember_batch", {
            "nodes": [
                {"type": "concept", "title": "Batch node A", "content": "First batch node"},
                {"type": "concept", "title": "Batch node B", "content": "Second batch node"},
            ],
        })
        self.assertIn("nodes_created", result)
        self.assertEqual(result["nodes_created"], 2)

    def test_revise(self):
        """revise updates an existing node."""
        n = self._dispatch("remember", {
            "type": "lesson", "title": "Revise me", "content": "Original content"
        })
        result = self._dispatch("revise", {
            "node_id": n["id"], "content": "Revised content", "reason": "test"
        })
        self.assertIn("id", result)
        self.assertTrue(result.get("embedding_updated") or result.get("revised_at"))
        # Verify content changed
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (n["id"],)).fetchone()
        self.assertEqual(row[0], "Revised content")

    def test_revise_batch(self):
        """revise_batch updates multiple nodes in one call."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "Rev batch A", "content": "A"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "Rev batch B", "content": "B"})
        result = self._dispatch("revise_batch", {
            "revisions": [
                {"node_id": n1["id"], "content": "A revised", "reason": "test"},
                {"node_id": n2["id"], "content": "B revised", "reason": "test"},
            ]
        })
        self.assertIn("revised", result)
        self.assertEqual(result["revised"], 2)

    def test_connect(self):
        """connect creates an edge between two nodes."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "Node A", "content": "First node"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "Node B", "content": "Second node"})
        self._dispatch("connect", {
            "source_id": n1["id"], "target_id": n2["id"],
            "relation": "related_to", "weight": 0.8
        })
        edge = self.brain.conn.execute(
            "SELECT edge_id FROM edges WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
            (n1["id"], n2["id"], n2["id"], n1["id"])).fetchone()
        self.assertIsNotNone(edge, "Edge should exist")
        rel = self.brain.conn.execute(
            "SELECT relation, weight FROM edge_relations WHERE edge_id = ? AND relation = 'related_to'",
            (edge[0],)).fetchone()
        self.assertIsNotNone(rel, "Edge relation should exist after connect")
        self.assertEqual(rel[0], "related_to")

    def test_connect_batch(self):
        """connect_batch creates multiple edges in one call."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "CB A", "content": "A"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "CB B", "content": "B"})
        n3 = self._dispatch("remember", {"type": "concept", "title": "CB C", "content": "C"})
        result = self._dispatch("connect_batch", {
            "connections": [
                {"source_id": n1["id"], "target_id": n2["id"], "relation": "related_to"},
                {"source_id": n2["id"], "target_id": n3["id"], "relation": "related_to"},
            ]
        })
        self.assertIn("edges_created", result)
        self.assertEqual(result["edges_created"], 2)

    def test_brain_batch(self):
        """brain_batch runs mixed operations in one call."""
        result = self._dispatch("brain_batch", {
            "operations": [
                {"op": "remember", "type": "concept", "title": "Batch op node", "content": "Created via brain_batch"},
            ]
        })
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1)

    def test_enrich(self):
        """enrich stores enrichment vectors for a node."""
        n = self._dispatch("remember", {"type": "lesson", "title": "Enrich test", "content": "Test content"})
        result = self._dispatch("enrich", {
            "node_id": n["id"],
            "question": "What was the enrich test about?",
            "keywords": "enrich, test, verification"
        })
        self.assertIn("enrichments_stored", result)

    def test_recall_batch(self):
        """recall_batch runs multiple queries in one call."""
        result = self._dispatch("recall_batch", {
            "queries": ["test query one", "test query two"],
            "limit": 3
        })
        # Returns a list of per-query results
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    # ── Lookup operations ──

    def test_find_node_by_title(self):
        """find_node_by_title locates nodes by fuzzy title matching."""
        self._dispatch("remember", {
            "type": "decision", "title": "Use Arctic v1.5 for embeddings",
            "content": "Chose Arctic v1.5 because it balances quality and speed."
        })
        result = self._dispatch("find_node_by_title", {
            "title_query": "Arctic embedding model", "threshold": 0.5
        })
        self.assertIsNotNone(result)
        self.assertIn("id", result)
        self.assertGreater(result["similarity"], 0.5)

    def test_find_node_by_title_no_match(self):
        """find_node_by_title returns None when nothing matches."""
        result = self._dispatch("find_node_by_title", {
            "title_query": "completely unrelated xyzzy topic", "threshold": 0.9
        })
        self.assertIsNone(result)

    def test_get_node(self):
        """get_node returns full node data by ID."""
        n = self._dispatch("remember", {"type": "concept", "title": "Get me", "content": "Full content here"})
        result = self._dispatch("get_node", {"node_id": n["id"]})
        self.assertIn("id", result)
        self.assertEqual(result["title"], "Get me")

    def test_get_nodes(self):
        """get_nodes returns multiple nodes by ID."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "Multi A", "content": "A"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "Multi B", "content": "B"})
        result = self._dispatch("get_nodes", {"node_ids": [n1["id"], n2["id"]]})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_filter_nodes(self):
        """filter_nodes returns nodes matching structural criteria."""
        result = self._dispatch("filter_nodes", {"field": "type"})
        # Without include/exclude, returns distinct values
        self.assertIsInstance(result, dict)

    # ── Introspection ──

    # NOTE: test_engineering_context removed 2026-04-26.
    # The engineering_context tool was removed 2026-04-13 (was a stub
    # returning {}). Three production-side comments confirm the deletion:
    # daemon_dispatch.py:989, brain_mcp.py:504, brain_reminders.py:22.
    # Roundtrip coverage of a removed tool is meaningless.

    def test_eval(self):
        """eval executes arbitrary Python on brain object."""
        result = self._dispatch("eval", {"code": "brain.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]"})
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    # ── Trace/log queries ──

    def test_query_logs(self):
        """query_logs returns log entries."""
        result = self._dispatch("query_logs", {"hours": 1})
        self.assertIsInstance(result, dict)

    def test_query_traces(self):
        """query_traces returns trace events."""
        result = self._dispatch("query_traces", {"hours": 1, "limit": 5})
        self.assertIsInstance(result, dict)

    def test_get_trace(self):
        """get_trace returns the full row for a known trace_id, or error for missing."""
        # Append a trace and read it back
        tid = self.brain._trace_dal.append(
            chain_id='roundtrip-get-trace', scale='s0', event_type='K',
            ref_type='user_message', summary='roundtrip get_trace probe')
        result = self._dispatch("get_trace", {"trace_id": tid})
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], tid)
        self.assertEqual(result['summary'], 'roundtrip get_trace probe')

    def test_get_traces(self):
        """get_traces returns a list; missing ids silently skipped."""
        a = self.brain._trace_dal.append(
            chain_id='roundtrip-get-traces', scale='s0', event_type='K',
            ref_type='user_message', summary='get_traces probe A')
        b = self.brain._trace_dal.append(
            chain_id='roundtrip-get-traces', scale='s0', event_type='delta',
            ref_type='assistant_message', summary='get_traces probe B')
        result = self._dispatch("get_traces", {"trace_ids": [a, b, 99999999]})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        ids = {r['id'] for r in result}
        self.assertEqual(ids, {a, b})

    def test_query_outcomes(self):
        """query_outcomes returns outcome events."""
        result = self._dispatch("query_outcomes", {"hours": 1})
        self.assertIsInstance(result, list)

    def test_count_traces(self):
        """count_traces returns grouped counts."""
        result = self._dispatch("count_traces", {"field": "event_type"})
        self.assertIsInstance(result, dict)

    # ── Interactions ──

    def test_list_interactions(self):
        """list_interactions returns registered interactions."""
        result = self._dispatch("list_interactions", {})
        self.assertIsInstance(result, list)

    def test_get_interaction(self):
        """get_interaction returns a specific interaction."""
        result = self._dispatch("get_interaction", {"name": "surface"})
        # May return None if interactions not seeded in isolated brain
        self.assertTrue(result is None or isinstance(result, dict))

    def test_register_interaction(self):
        """register_interaction adds a new interaction version."""
        result = self._dispatch("register_interaction", {
            "name": "test_roundtrip_interaction",
            "template": "Hello, {name}!",
            "parameters": '{"max_messages": 5}',
            "created_by": "roundtrip_test",
        })
        self.assertIsInstance(result, dict)
        # Should return version info (registered_version since 2026-05-10)
        self.assertTrue(
            'registered_version' in result or 'version' in result or 'name' in result,
            f"Expected version info in result, got: {result}")

    def test_set_interaction_active(self):
        """set_interaction_active flips the active version pointer."""
        # First register two versions so we have something to flip between
        self._dispatch("register_interaction", {
            "name": "test_active_flip",
            "template": "v1 content",
            "parameters": '{}',
            "created_by": "roundtrip_test",
        })
        self._dispatch("register_interaction", {
            "name": "test_active_flip",
            "template": "v2 content",
            "parameters": '{}',
            "created_by": "roundtrip_test",
        })
        # Now flip active to v2
        result = self._dispatch("set_interaction_active", {
            "name": "test_active_flip",
            "version": 2,
            "set_by": "roundtrip_test",
        })
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("version"), 2,
                         f"Expected version 2, got: {result}")

    def test_clear_errors(self):
        """clear_errors empties the rate-limit cache for the brain errors table."""
        result = self._dispatch("clear_errors", {})
        self.assertIsInstance(result, dict)

    # ── Coverage check ──

    def test_all_mcp_tools_have_roundtrip_tests(self):
        """Every MCP tool should have a corresponding test above."""
        mcp_tool_names = {t["name"] for t in TOOLS}
        test_methods = {m for m in dir(self) if m.startswith("test_") and m != "test_all_mcp_tools_have_roundtrip_tests"}

        tested = set()
        for method in test_methods:
            tool_name = method.replace("test_", "").replace("_then_recall", "").replace("_no_match", "")
            if tool_name in mcp_tool_names:
                tested.add(tool_name)

        # Commands that can't be round-trip tested (they kill/restart the daemon)
        untestable = {'restart', 'shutdown'}
        untested = mcp_tool_names - tested - untestable
        self.assertEqual(untested, set(),
                         "MCP tools without round-trip tests: %s" % untested)


if __name__ == "__main__":
    unittest.main()
