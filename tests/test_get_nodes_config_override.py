"""get_nodes_config override + result-shape handling on _format_result (2026-06-23).

Pins the fix for the S2 community encoder's 217K-token blowup.

Root cause (two layers):
  1. SHAPE: the dispatch handler `_handle_get_nodes` (dispatch_read.py) returns
     a LIST of rich nodes (+ {"id","error"} entries) — the shape every encoder
     and Anchor's MCP get_nodes actually receive. `_format_result` originally
     gated its render branch on `isinstance(result, dict)`, so on the real
     (list) path the branch never fired and output fell through to
     `json.dumps` — the raw `_corrections` firehose, at every batch size.
  2. CONFIG: even when rendered, a small (<=3) batch took a raw-JSON escape
     hatch. run_llm_loop now threads a caller-declared `get_nodes_config` so
     S2 encoders render bounded at every batch size.

Fix: _format_result handles BOTH list and dict shapes; with get_nodes_config
set it renders via render_rich_node at every batch size. The community encoder
passes S2CE_NODE_FORMAT (content 800, edges 5, corrections 'balanced').

Pure-function tests — _format_result + render_rich_node take a result value,
no brain/embedder.

Run: ./dev python3 -m pytest tests/test_get_nodes_config_override.py -v
"""
import json
import unittest

from servers.brain_mcp import _format_result
from servers.scales.s2.community_contract import S2CE_NODE_FORMAT


def _hub_node(node_id="hub00001"):
    """A rich node shaped like a hub member: long content + a heavy correction
    carrying full K/V. Sentinels mark the firehose-only tails."""
    return {
        "id": node_id,
        "title": "Hub member",
        "type": "decision",
        "content": "A" * 900 + " CONTENT_TAIL_SENTINEL",   # tail past 800
        "situation": "When the encoder inspects a hub member",
        "confidence": 0.9,
        "created_at": "2026-06-20T00:00:00+00:00",
        # Real edge shapes — exercise both render branches (multi-relation via
        # the `relations` list; single-relation via flat `relation`) and the
        # edge_limit bound. 7 edges > S2CE_NODE_FORMAT's edge_limit=5, so the
        # last two (OVERFLOW sentinels) must be dropped.
        "connections": [
            {"id": "nbr00001", "type": "decision", "title": "Neighbor one",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "weight": 0.6, "relations": [
                 {"relation": "depends_on", "description": "needs neighbor one", "weight": 0.6},
                 {"relation": "extends", "description": "also extends it", "weight": 0.4}]},
            {"id": "nbr00002", "type": "finding", "title": "Neighbor two",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "incoming",
             "relation": "implements", "description": "implements the thing", "weight": 0.5},
            {"id": "nbr00003", "type": "rule", "title": "Neighbor three",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d3"},
            {"id": "nbr00004", "type": "rule", "title": "Neighbor four",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d4"},
            {"id": "nbr00005", "type": "rule", "title": "Neighbor five",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d5"},
            {"id": "nbr00006", "type": "rule", "title": "OVERFLOW_EDGE_SENTINEL six",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d6"},
            {"id": "nbr00007", "type": "rule", "title": "OVERFLOW_EDGE_SENTINEL seven",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d7"},
        ],
        "_corrections": [{
            "id": "corr0001",
            "title": "A correction",
            "type": "correction",
            "direction": "incoming",
            "relation": "corrects",
            "edge_description": "supersedes the earlier assumption",
            "content": "C" * 300 + " CORR_TAIL_SENTINEL",
            "reasoning": "REASONING_SENTINEL why the correction landed",
            "anchor_raw_quote": "ANCHOR_QUOTE_SENTINEL",
            "user_raw_quote": "USER_QUOTE_SENTINEL",
        }],
    }


def _dispatch_list(n=1, with_error=False):
    """The shape _handle_get_nodes actually returns: a LIST of rich nodes,
    with {"id","error"} entries appended for unresolved ids."""
    out = [_hub_node("hub%05d" % i) for i in range(n)]
    if with_error:
        out.append({"id": "missing1", "error": "not found"})
    return out


def _is_raw_json(s):
    """Raw json.dumps output is parseable JSON; render_rich_node output starts
    with a `[type] "title"` header and is NOT valid JSON. (Can't discriminate
    on a leading '[' — rendered nodes lead with their bracketed type tag.)"""
    try:
        json.loads(s)
        return True
    except Exception:
        return False


class TestGetNodesConfigOverride(unittest.TestCase):

    # ── production (list) shape ──────────────────────────────────────

    def test_default_small_batch_list_still_raw_json(self):
        """<=3 nodes, no config → raw JSON (Anchor drill-down). Production
        shape is a LIST, so the dump is a JSON array."""
        result = _dispatch_list(1)
        out = _format_result("get_nodes", result)          # config=None
        self.assertTrue(_is_raw_json(out))                   # raw list dump
        parsed = json.loads(out)
        self.assertEqual(parsed[0]["id"], "hub00000")
        # Firehose: full content + heavy correction K/V present.
        self.assertIn("CONTENT_TAIL_SENTINEL", out)
        self.assertIn("ANCHOR_QUOTE_SENTINEL", out)

    def test_default_large_batch_list_renders(self):
        """REGRESSION GUARD for the dead branch: a >10-node LIST with no config
        must render via the batch-size heuristic — NOT fall through to raw JSON.
        This is exactly what was broken (isinstance(result, dict) gate)."""
        result = _dispatch_list(12)
        out = _format_result("get_nodes", result)          # config=None
        self.assertFalse(_is_raw_json(out))                  # rendered, not raw
        self.assertEqual(out.count("Hub member"), 12)

    def test_config_override_bounds_small_batch_list(self):
        """With S2CE_NODE_FORMAT, a <=3-node LIST renders bounded — no raw JSON,
        content truncated to 800, corrections 'balanced' (no heavy K/V)."""
        result = _dispatch_list(1)
        out = _format_result("get_nodes", result,
                             get_nodes_config=S2CE_NODE_FORMAT)

        self.assertFalse(_is_raw_json(out))                  # rendered
        self.assertIn("Hub member", out)                     # title rendered
        self.assertNotIn("CONTENT_TAIL_SENTINEL", out)       # content capped at 800
        # 'balanced' corrections: title + id + edge "why" + short excerpt,
        # but NOT heavy K/V (reasoning / raw quotes) or full correction content.
        self.assertIn("A correction", out)
        self.assertIn("supersedes the earlier assumption", out)
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)
        self.assertNotIn("USER_QUOTE_SENTINEL", out)
        self.assertNotIn("REASONING_SENTINEL", out)
        self.assertNotIn("CORR_TAIL_SENTINEL", out)          # 300-char tail > 150 cap
        # Edge path exercised (the heaviest field) and bounded by edge_limit=5:
        self.assertIn("depends_on", out)                     # multi-relation edge rendered
        self.assertIn("implements", out)                     # single-relation edge rendered
        self.assertNotIn("OVERFLOW_EDGE_SENTINEL", out)      # edges 6-7 dropped at edge_limit=5
        # Bounded render is far smaller than the firehose.
        self.assertLess(len(out), len(_format_result("get_nodes", result)))

    def test_config_override_large_batch_list(self):
        """Override is batch-size agnostic on the list shape too."""
        result = _dispatch_list(12)
        out = _format_result("get_nodes", result,
                             get_nodes_config=S2CE_NODE_FORMAT)
        self.assertFalse(_is_raw_json(out))
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)       # balanced, not heavy
        self.assertEqual(out.count("Hub member"), 12)

    def test_error_entries_filtered_not_rendered(self):
        """{"id","error"} entries from unresolved ids must not be rendered as
        nodes (they lack node fields) — render the real nodes, skip errors."""
        result = _dispatch_list(2, with_error=True)
        out = _format_result("get_nodes", result,
                             get_nodes_config=S2CE_NODE_FORMAT)
        self.assertEqual(out.count("Hub member"), 2)         # both real nodes
        self.assertNotIn("missing1", out)                    # error entry skipped

    # ── dict shape (brain.get_node(ids) direct) still works ──────────

    def test_dict_shape_also_renders(self):
        """Defensive: the {node_id: rich_node} dict shape (brain.get_node)
        renders the same way under a config override."""
        result = {"hub00001": _hub_node("hub00001")}
        out = _format_result("get_nodes", result,
                             get_nodes_config=S2CE_NODE_FORMAT)
        self.assertFalse(_is_raw_json(out))                  # rendered, not raw
        self.assertIn("Hub member", out)
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)


if __name__ == "__main__":
    unittest.main()
