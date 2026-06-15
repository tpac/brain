"""Dashboard consolidation-runs: trace-authoritative enrichment.

Pins the deconstruct logic that replaced the ±60min encoding_source window.
A consolidation run's product is read from the run's OWN delta record:
  • survivors / archived originals — from the ok-gated `revised`/`archived`
    buckets when present, else deconstructed from the recorded op input;
  • links — EVERY connect decision (KEEP/SKIP similar_to, SUPERSESSION
    supersedes, CONTRADICTION corrects, partition depends_on), carrying its
    relation — fetched by id, NOT reconstructed by time window.
"""
import json
import sqlite3

from dashboard.queries.s2_runs import (
    _deconstruct_consolidation_ops,
    _enrich_consolidation,
)


def _delta_meta(operations, **top):
    """Delta metadata carrying one brain_batch action with `operations`."""
    m = {"action_details": [{"tool": "brain_batch",
                             "input": {"operations": operations}}]}
    m.update(top)
    return m


class TestDeconstructConsolidationOps:
    def test_absorb_splits_into_synth_and_archived(self):
        synth, archived, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "absorb", "survivor_id": "surv1", "absorbed_id": "orig1"},
        ]))
        assert synth == ["surv1"]      # the enriched survivor
        assert archived == ["orig1"]   # the folded-in original
        assert links == []

    def test_connect_similar_to_captured(self):
        _, _, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "connect", "source_id": "a", "target_id": "b",
             "relation": "similar_to", "description": "two locked dups"},
        ]))
        assert links == [{"source_id": "a", "target_id": "b",
                          "relation": "similar_to", "description": "two locked dups"}]

    def test_connect_supersedes_captured(self):
        # SUPERSESSION — was silently dropped before the fix (only similar_to
        # matched). It is a standalone connect(supersedes), no absorb.
        _, _, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "connect", "source_id": "new", "target_id": "old",
             "relation": "supersedes", "description": "Q2 caps supersede static"},
        ]))
        assert links == [{"source_id": "new", "target_id": "old",
                          "relation": "supersedes",
                          "description": "Q2 caps supersede static"}]

    def test_connect_corrects_captured(self):
        # CONTRADICTION — the correction substrate; also dropped before the fix.
        _, _, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "connect", "source_id": "right", "target_id": "wrong",
             "relation": "corrects", "description": "CDN, not DB"},
        ]))
        assert links[0]["relation"] == "corrects"

    def test_connect_depends_on_captured(self):
        _, _, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "connect", "source_id": "x", "target_id": "y",
             "relation": "depends_on"},
        ]))
        assert links and links[0]["relation"] == "depends_on"

    def test_hebbian_relations_excluded(self):
        # System relations are never encoder-emitted; excluded defensively.
        _, _, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "connect", "source_id": "a", "target_id": "b",
             "relation": "co_accessed"},
        ]))
        assert links == []

    def test_revise_is_synth_archive_is_archived(self):
        synth, archived, _ = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "revise", "node_id": "r1", "reason": "x"},
            {"op": "archive", "node_id": "a1"},
        ]))
        assert synth == ["r1"]
        assert archived == ["a1"]

    def test_mixed_batch(self):
        synth, archived, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "absorb", "survivor_id": "s1", "absorbed_id": "o1"},
            {"op": "absorb", "survivor_id": "s2", "absorbed_id": "o2"},
            {"op": "connect", "source_id": "k1", "target_id": "k2",
             "relation": "similar_to"},
            {"op": "connect", "source_id": "n", "target_id": "o",
             "relation": "supersedes"},
        ]))
        assert synth == ["s1", "s2"]
        assert archived == ["o1", "o2"]
        assert [l["relation"] for l in links] == ["similar_to", "supersedes"]

    def test_ok_gated_buckets_win_over_op_input(self):
        # Post-fix deltas carry ok-gated top-level buckets; those are the truth
        # (a failed absorb is excluded from them), so they win over the
        # request-side op input.
        synth, archived, links = _deconstruct_consolidation_ops(_delta_meta(
            [{"op": "absorb", "survivor_id": "OP_S", "absorbed_id": "OP_A"},
             {"op": "connect", "source_id": "x", "target_id": "y",
              "relation": "similar_to"}],
            revised=["BUCKET_S"], archived=["BUCKET_A"]))
        assert synth == ["BUCKET_S"]      # bucket wins over op survivor
        assert archived == ["BUCKET_A"]   # bucket wins over op absorbed
        assert links[0]["source_id"] == "x"  # links always from ops (not bucketed as pairs)

    def test_op_input_fallback_when_no_buckets(self):
        # Pre-fix historical deltas have empty buckets → deconstruct op input.
        synth, archived, _ = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "absorb", "survivor_id": "histS", "absorbed_id": "histA"},
        ]))
        assert synth == ["histS"]
        assert archived == ["histA"]

    def test_empty_meta(self):
        assert _deconstruct_consolidation_ops({}) == ([], [], [])


class TestEnrichConsolidationById:
    def _conn(self):
        c = sqlite3.connect(":memory:")
        c.execute(
            "CREATE TABLE nodes (id TEXT PRIMARY KEY, type TEXT, title TEXT, "
            "content TEXT, confidence REAL, encoding_source TEXT, "
            "created_at TEXT, archived INTEGER DEFAULT 0)")
        # legacy-fallback path joins these; present so it never crashes.
        c.execute("CREATE TABLE edges (edge_id INTEGER, source_id TEXT, "
                  "target_id TEXT, weight REAL, created_at TEXT)")
        c.execute("CREATE TABLE edge_relations (edge_id INTEGER, relation TEXT, "
                  "description TEXT, archived INTEGER DEFAULT 0)")
        return c

    def _node(self, c, nid, archived=0, etype='principle', title=None):
        c.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?)",
                  (nid, etype, title or ('Node ' + nid), 'content-' + nid,
                   0.9, 's2:consolidation', '2026-06-14T10:00:00+00:00', archived))

    def test_fetches_recorded_nodes_by_id_no_window(self):
        c = self._conn()
        self._node(c, 'surv1', archived=0, title='Survivor title')
        self._node(c, 'orig1', archived=1, etype='fact', title='Folded original')
        c.commit()
        meta = _delta_meta([{"op": "absorb", "survivor_id": "surv1",
                             "absorbed_id": "orig1"}])
        delta_row = ("s2-20260614-consolidation", "1 action",
                     json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, {})
        assert [n["id"] for n in out["synthesized"]] == ["surv1"]
        assert out["synthesized"][0]["title"] == "Survivor title"
        assert [n["id"] for n in out["archived"]] == ["orig1"]
        assert "kept" not in out and "evolved" not in out  # unified into links

    def test_supersedes_link_carries_relation(self):
        c = self._conn()
        self._node(c, 'new1', title='New state')
        self._node(c, 'old1', title='Prior state')
        c.commit()
        meta = _delta_meta([{"op": "connect", "source_id": "new1",
                             "target_id": "old1", "relation": "supersedes",
                             "description": "new supersedes old"}])
        delta_row = ("chain", "s", json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, {})
        assert out["links"] == [{"source": "New state", "target": "Prior state",
                                 "relation": "supersedes",
                                 "description": "new supersedes old"}]
        # supersedes-only run must NOT fall into the window (synthesized empty,
        # link present)
        assert out["synthesized"] == [] and out["archived"] == []

    def test_archived_survivor_shown_as_archived_not_live(self):
        # A survivor archived by a later run (DB archived=1) must not render as
        # a live SYNTHESIZED node — the DB flag is the liveness tiebreaker.
        c = self._conn()
        self._node(c, 'survX', archived=1)   # later archived
        self._node(c, 'origX', archived=1)
        c.commit()
        meta = _delta_meta([{"op": "absorb", "survivor_id": "survX",
                             "absorbed_id": "origX"}])
        delta_row = ("chain", "s", json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, {})
        assert out["synthesized"] == []
        assert {n["id"] for n in out["archived"]} == {"survX", "origX"}

    def test_chain_merge_dedup(self):
        # A node that is a survivor in op1 and absorbed in op2 (chain merge):
        # shown once, in archived (it didn't survive), never duplicated.
        c = self._conn()
        self._node(c, 'A', archived=1)   # survived op1, absorbed in op2 → archived
        self._node(c, 'B', archived=1)
        self._node(c, 'C', archived=0)
        c.commit()
        meta = _delta_meta([
            {"op": "absorb", "survivor_id": "A", "absorbed_id": "B"},
            {"op": "absorb", "survivor_id": "C", "absorbed_id": "A"},
        ])
        delta_row = ("chain", "s", json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, {})
        assert [n["id"] for n in out["synthesized"]] == ["C"]
        assert {n["id"] for n in out["archived"]} == {"A", "B"}

    def test_missing_link_endpoint_falls_back_to_short_id(self):
        c = self._conn()
        self._node(c, 'present', title='Present node')
        c.commit()
        meta = _delta_meta([{"op": "connect", "source_id": "present",
                             "target_id": "deadbeef", "relation": "similar_to"}])
        delta_row = ("chain", "s", json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, {})
        assert out["links"][0]["source"] == "Present node"
        assert out["links"][0]["target"] == "deadbeef"  # short-id fallback

    def test_legacy_window_fallback_when_no_ops(self):
        c = self._conn()
        # Pre-absorb synth node created in the window; delta carries no ops.
        c.execute("INSERT INTO nodes VALUES ('legacy1','fact','Legacy synth',"
                  "'c',0.7,'s2:consolidation','2026-04-20T10:05:00+00:00',0)")
        c.commit()
        delta_row = ("chain", "s", json.dumps({"action_details": []}),
                     "2026-04-20T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, {})
        assert [n["id"] for n in out["synthesized"]] == ["legacy1"]
        assert "links" in out  # legacy path produces the unified shape
