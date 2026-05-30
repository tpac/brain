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
from servers.brain_mcp import TOOLS, CRITICAL_TOOLS
from servers.daemon_dispatch import COMMAND_TABLE


class TestMCPRoundTrip(BrainTestBase):
    """Test every MCP tool through daemon dispatch against a real brain."""

    def _dispatch(self, cmd, args):
        """Simulate daemon dispatch — call handler, ENFORCE + unwrap the
        {"ok": True, "result": ...} envelope.

        Every table handler MUST return that envelope; the daemon sends the
        return verbatim, so a raw dict reaches the MCP client as a falsy `ok`
        with no `error` — surfacing as "Unknown daemon error" on a call that
        actually succeeded (the dispatch_self bug, c4f6386). The previous
        lenient `if "result" in raw: ... else return raw` passed un-enveloped
        returns straight through, which is exactly why this suite — though it
        exercises self_presence/send/etc. — never caught that bug. Enforcing
        the envelope here makes the violation fail for EVERY tool, in CI."""
        entry = COMMAND_TABLE.get(cmd)
        self.assertIsNotNone(entry, "No dispatch handler for: %s" % cmd)
        raw = entry.handler(self.brain, args, [])
        self.assertIsInstance(raw, dict, "%s handler returned non-dict: %r" % (cmd, raw))
        self.assertIs(raw.get("ok"), True,
                      "%s handler must return the {'ok': True, 'result': ...} "
                      "envelope (raw return reads as 'Unknown daemon error'), got: %r"
                      % (cmd, raw))
        self.assertIn("result", raw, "%s handler envelope missing 'result'" % cmd)
        return raw["result"]

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
        """filter_nodes returns {'nodes': [...], 'total_count': N} for a
        structural-field query (field='type', no include/exclude → all
        non-archived nodes). NOTE: the old comment here claimed 'returns
        distinct values' — the DAL never did that; it always returns the
        nodes+count shape. Asserting the real contract catches shape drift."""
        result = self._dispatch("filter_nodes", {"field": "type"})
        self.assertIsInstance(result, dict)
        self.assertIn("nodes", result)
        self.assertIsInstance(result["nodes"], list)
        self.assertIn("total_count", result)
        self.assertIsInstance(result["total_count"], int)

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
        """query_logs returns {'entries': [...], 'counts': {...}} (DAL contract)."""
        result = self._dispatch("query_logs", {"hours": 1})
        self.assertIsInstance(result, dict)
        self.assertIn("entries", result)
        self.assertIsInstance(result["entries"], list)
        self.assertIn("counts", result)
        self.assertIsInstance(result["counts"], dict)

    def test_query_traces(self):
        """query_traces default mode returns {'events': [...]} (flat recent)."""
        result = self._dispatch("query_traces", {"hours": 1, "limit": 5})
        self.assertIsInstance(result, dict)
        self.assertIn("events", result)
        self.assertIsInstance(result["events"], list)

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
        """get_traces returns a list; missing ids silently skipped.
        v29: ids are 8-char hex strings (reviewer F2 — int input rejected
        loudly). The phantom id 'deadbeef' won't match any real row."""
        a = self.brain._trace_dal.append(
            chain_id='roundtrip-get-traces', scale='s0', event_type='K',
            ref_type='user_message', summary='get_traces probe A')
        b = self.brain._trace_dal.append(
            chain_id='roundtrip-get-traces', scale='s0', event_type='delta',
            ref_type='assistant_message', summary='get_traces probe B')
        result = self._dispatch("get_traces", {"trace_ids": [a, b, 'deadbeef']})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        ids = {r['id'] for r in result}
        self.assertEqual(ids, {a, b})

    def test_query_outcomes(self):
        """query_outcomes returns the outcome events for a chain. Write one,
        then read it back — stronger than the old isinstance-list smoke check."""
        self.brain._trace_dal.append(
            chain_id="roundtrip-oc", scale="s1", event_type="O",
            ref_type="recall", summary="seed for outcome")
        self.brain._trace_dal.append_outcome(
            chain_id="roundtrip-oc", scale="s1", ref_type="correction",
            ref_id="node-xyz", summary="operator corrected this")
        result = self._dispatch("query_outcomes", {"chain_id": "roundtrip-oc"})
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)
        self.assertTrue(
            any(o.get("ref_type") == "correction" for o in result),
            "appended correction outcome not returned by query_outcomes")

    def test_count_traces(self):
        """count_traces returns {event_type: int}. Append two known events and
        verify they're counted — stronger than the old isinstance-dict check."""
        for _ in range(2):
            self.brain._trace_dal.append(
                chain_id="roundtrip-count", scale="s1", event_type="O",
                ref_type="recall", summary="count probe")
        result = self._dispatch("count_traces", {"field": "event_type"})
        self.assertIsInstance(result, dict)
        self.assertTrue(all(isinstance(v, int) for v in result.values()),
                        "count values must be ints")
        self.assertGreaterEqual(result.get("O", 0), 2,
                                "two appended 'O' events not reflected in counts")

    # ── Interactions ──

    def test_list_interactions(self):
        """list_interactions returns registered interactions. Register a probe
        and confirm it appears by name — stronger than isinstance-list."""
        self._dispatch("register_interaction", {
            "name": "roundtrip_list_probe", "template": "probe",
            "parameters": "{}", "created_by": "roundtrip_test"})
        result = self._dispatch("list_interactions", {})
        self.assertIsInstance(result, list)
        names = {i.get("name") for i in result}
        self.assertIn("roundtrip_list_probe", names)

    def test_get_interaction(self):
        """get_interaction returns the stored interaction row. Register a probe
        with a known template, then read it back. The old check
        (`result is None or isinstance(result, dict)`) passed even on the
        handler's {'ok': False, 'error': ...} envelope — it asserted nothing
        about correctness. get_active returns id/name/version/template/parameters."""
        self._dispatch("register_interaction", {
            "name": "roundtrip_get_probe", "template": "PROBE TEMPLATE BODY",
            "parameters": "{}", "created_by": "roundtrip_test"})
        result = self._dispatch("get_interaction", {"name": "roundtrip_get_probe"})
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("name"), "roundtrip_get_probe")
        self.assertEqual(result.get("template"), "PROBE TEMPLATE BODY")

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
        """clear_errors deletes hook_errors and reports the rowcount. Insert a
        hook error, clear, and confirm it's actually gone — the old check only
        asserted the return was a dict."""
        self.brain.logs_conn.execute(
            "INSERT INTO hook_errors (created_at, hook_name, level, error) "
            "VALUES (?, ?, ?, ?)",
            ("2026-05-29T00:00:00+00:00", "test_hook", "error", "probe error"))
        self.brain.logs_conn.commit()
        result = self._dispatch("clear_errors", {})
        self.assertIsInstance(result, dict)
        self.assertIn("hook_errors", result)
        self.assertGreaterEqual(result["hook_errors"], 1)
        remaining = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM hook_errors").fetchone()[0]
        self.assertEqual(remaining, 0, "hook_errors not actually cleared")

    # ── Coverage check ──

    # ── Self channel (presence — pull) ──

    def test_self_presence(self):
        """self_presence returns a roster of live streams + a rendered line."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='roundtrip-stream-A')
        ctx.message_count = 3
        ctx.save(self.brain.logs_conn)
        self.brain.set_config('session_context_roundtrip-stream-A', 'roundtrip focus A')
        result = self._dispatch("self_presence", {"session_id": "roundtrip-self", "limit": 5})
        self.assertIn("streams", result)
        self.assertIn("line", result)
        self.assertIsInstance(result["streams"], list)
        ids = {s["session_id"] for s in result["streams"]}
        self.assertIn("roundtrip-stream-A", ids)

    def test_self_peek(self):
        """self_peek returns one stream's full current focus."""
        self.brain.set_config('session_context_roundtrip-peek',
                              'peek focus line one\nline two')
        result = self._dispatch("self_peek", {"stream_id": "roundtrip-peek"})
        self.assertTrue(result["found"])
        self.assertEqual(result["focus"], "peek focus line one\nline two")

    def test_self_send(self):
        """self_send places a directed message in the courier."""
        result = self._dispatch("self_send", {
            "to": "roundtrip-recipient", "body": "tap on the shoulder",
            "from_session": "roundtrip-sender"})
        self.assertIn("id", result)
        self.assertEqual(result["address"], "self:roundtrip-recipient")

    def test_self_inbox(self):
        """self_inbox drains messages addressed to the caller, consume-once."""
        self._dispatch("self_send", {
            "to": "roundtrip-inbox-user", "body": "you have mail",
            "from_session": "roundtrip-sender"})
        result = self._dispatch("self_inbox", {"session_id": "roundtrip-inbox-user"})
        self.assertIn("you have mail", [m["body"] for m in result["messages"]])
        again = self._dispatch("self_inbox", {"session_id": "roundtrip-inbox-user"})
        self.assertNotIn("you have mail", [m["body"] for m in again["messages"]])

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

    def test_roundtrip_tests_assert_on_content(self):
        """Gate against smoke-test theater: every MCP tool's round-trip test
        must assert on actual keys/values, not merely that *something* came back.

        A test whose only assertion is `assertIsInstance(result, dict/list)` on
        the bare top-level result passes no matter what the handler returns — it
        can't catch a handler that drifts to the wrong shape. This AST check
        requires each tool to have at least one test method with a substantive
        assertion (assertEqual / assertIn / assertGreater / ... or an
        assertIsInstance on a SUBSCRIPT like result['key'], which checks a
        sub-field rather than the envelope).

        Pairs with test_all_mcp_tools_have_roundtrip_tests: that gate ensures
        coverage *exists*; this one ensures the coverage *bites*. Without it,
        the easy way to satisfy the coverage gate is an isinstance smoke test
        (bucket B of the 2026-05 redundancy hunt was exactly that backlog)."""
        import ast
        import inspect
        import textwrap

        SUBSTANTIVE = {
            'assertEqual', 'assertNotEqual', 'assertIn', 'assertNotIn',
            'assertGreater', 'assertGreaterEqual', 'assertLess', 'assertLessEqual',
            'assertTrue', 'assertFalse', 'assertIsNone', 'assertIsNotNone',
            'assertAlmostEqual', 'assertListEqual', 'assertDictEqual', 'assertRegex',
        }

        def _is_substantive(method_name):
            src = textwrap.dedent(inspect.getsource(getattr(type(self), method_name)))
            for node in ast.walk(ast.parse(src)):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)):
                    continue
                attr = node.func.attr
                if attr in SUBSTANTIVE:
                    return True
                # assertIsInstance(result['key'], ...) inspects a sub-field's
                # shape, not just the top-level envelope — that counts.
                if (attr == 'assertIsInstance' and node.args
                        and isinstance(node.args[0], ast.Subscript)):
                    return True
            return False

        mcp_tool_names = {t["name"] for t in TOOLS}
        untestable = {'restart', 'shutdown'}

        # Map each tool to the test methods exercising it (same name-stripping
        # as the coverage gate above), skipping the two meta-gates themselves.
        tool_methods = {}
        for method in dir(self):
            if not method.startswith("test_"):
                continue
            if method in ('test_all_mcp_tools_have_roundtrip_tests',
                          'test_roundtrip_tests_assert_on_content'):
                continue
            tool_name = method.replace("test_", "").replace("_then_recall", "").replace("_no_match", "")
            if tool_name in mcp_tool_names:
                tool_methods.setdefault(tool_name, []).append(method)

        theater = []
        for tool, methods in sorted(tool_methods.items()):
            if tool in untestable:
                continue
            if not any(_is_substantive(m) for m in methods):
                theater.append("%s (only: %s)" % (tool, ", ".join(sorted(methods))))

        self.assertEqual(theater, [],
                         "MCP tools whose round-trip tests assert only type, not "
                         "content (smoke-test theater — strengthen to assert "
                         "keys/values): %s" % theater)


class TestCriticalToolsAlwaysLoad(unittest.TestCase):
    """Tool-search deferral contract.

    Claude Code defers MCP tools behind ToolSearch when they'd exceed ~10% of
    the context window (the default). CRITICAL_TOOLS opt out via
    anthropic/alwaysLoad in their _meta so the brain's hot-path tools load
    eagerly for EVERY installer — the flag ships in tools/list, not in any
    user's settings.json. These gates lock that contract: the right tools are
    eager, the long tail still defers. Pure module inspection — no brain.
    """

    def test_critical_names_are_real_tools(self):
        """Every CRITICAL_TOOLS name must match an actual tool. A typo would
        silently defer a tool we meant eager. _stamp_always_load raises at
        import on mismatch; this locks the same invariant as a test."""
        names = {t["name"] for t in TOOLS}
        self.assertTrue(
            CRITICAL_TOOLS <= names,
            "CRITICAL_TOOLS contains non-existent tool(s): %s"
            % sorted(CRITICAL_TOOLS - names))

    def test_critical_tools_carry_always_load(self):
        """Each critical tool emits _meta['anthropic/alwaysLoad'] == True."""
        for t in TOOLS:
            if t["name"] in CRITICAL_TOOLS:
                self.assertEqual(
                    t.get("_meta", {}).get("anthropic/alwaysLoad"), True,
                    "%s should carry anthropic/alwaysLoad" % t["name"])

    def test_non_critical_tools_do_not_carry_always_load(self):
        """Non-critical tools must NOT be eager — otherwise the whole point
        (lean context, defer the long tail) collapses to load-everything."""
        for t in TOOLS:
            if t["name"] not in CRITICAL_TOOLS:
                self.assertNotIn(
                    "anthropic/alwaysLoad", t.get("_meta", {}),
                    "%s unexpectedly marked alwaysLoad" % t["name"])


class TestMissingEnvelopeIsLoud(unittest.TestCase):
    """brain_mcp must NAME a missing-envelope response, not bury it in the
    generic "Unknown daemon error". Locks the diagnostic added after the
    dispatch_self envelope bug (c4f6386): a handler returning a raw dict should
    produce an error that points at the envelope and shows the offending keys,
    so the next slip is a one-line diagnosis instead of a multi-turn hunt."""

    def test_raw_dict_response_yields_descriptive_error(self):
        from unittest import mock
        from servers import brain_mcp
        # daemon_send returns a raw payload dict (no ok/result/error) — exactly
        # what an un-enveloped handler produces over the wire.
        with mock.patch.object(brain_mcp, "daemon_send",
                               return_value={"streams": [], "line": ""}):
            resp = brain_mcp.handle_tools_call(
                "req-1", {"name": "self_presence", "arguments": {}})
        payload = resp["result"]
        self.assertTrue(payload.get("isError"))
        text = payload["content"][0]["text"]
        self.assertIn("envelope", text)
        self.assertNotIn("Unknown daemon error", text)
        # the offending keys are surfaced for fast diagnosis
        self.assertIn("streams", text)
        self.assertIn("line", text)


if __name__ == "__main__":
    unittest.main()
