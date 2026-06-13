"""Tests for the `absorbed_into` survivor-redirect edge (Phase 1 of
TRACE-NODE-RESOLUTION).

On a merge-archive, archive_node writes a first-class `absorbed_into` edge
(source = absorbed/dead, target = survivor/live) in the correction_improvement
aspect. The edge must:
  - be written only on a MERGE (survivor passed via extra), not a plain archive
  - land and stay live (archived=0) despite archive_node soft-archiving the
    node's other edges
  - survive a chained absorb (A→B→C) so edge-level chain traversal holds
  - coexist with the `_sys_archived_survivor_id` audit breadcrumb (backfill
    source + resolve_live's current read path)

The edge's source endpoint is archived, so get_connections_bulk (which filters
archived endpoints) won't surface it — we read the raw edge tables, which is
also how a future edge-based resolve_live must read it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestAbsorbedIntoEdge(BrainTestBase):
    needs_embedder = False

    def _node(self, title, content='content', **kw):
        r = self.brain.remember(type='fact', title=title, content=content,
                                encoding_source='anchor', **kw)
        return r['id']

    def _absorbed_into(self, source):
        """Raw rows for absorbed_into edges out of `source` (archived endpoint
        is invisible to get_connections_bulk, so query the tables directly)."""
        return self.brain.conn.execute(
            "SELECT e.source_id, e.target_id, er.relation, er.archived, "
            "       er.encoding_source "
            "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE er.relation = 'absorbed_into' AND e.source_id = ?",
            (source,)).fetchall()

    # ── aspect membership (the SHIPPED SEED) ──

    def test_absorbed_into_in_seed_correction_improvement_aspect(self):
        """Phase 1 changes the repo SEED baseline. The runtime registry reads a
        per-operator working copy ($BRAIN_DB_DIR/aspects_v1.json) seeded from
        this file only on first boot — so updating an EXISTING brain's live
        registry is a separate production step (flagged to the supervising
        stream), not something a fresh-brain test would observe. Assert the
        thing this slice actually edits: the shipped seed."""
        import json
        from servers.scales.s2.aspect_contract import SEED_ASPECTS_JSON_PATH
        with open(SEED_ASPECTS_JSON_PATH) as f:
            seed = json.load(f)
        self.assertIn('absorbed_into',
                      seed['correction_improvement']['edge_relations'])

    # ── writer: merge writes the edge ──

    def test_absorb_writes_live_absorbed_into_edge(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)

        edges = self._absorbed_into(absorbed)
        self.assertEqual(len(edges), 1, edges)
        src, tgt, rel, archived, enc = edges[0]
        self.assertEqual(src, absorbed)      # source = absorbed/dead
        self.assertEqual(tgt, survivor)      # target = survivor/live
        self.assertEqual(archived, 0)        # LIVE despite node being archived
        self.assertTrue(enc)                 # attributable

    def test_sys_breadcrumb_still_written_alongside_edge(self):
        """_sys_archived_survivor_id is kept as the audit/backfill source even
        though the edge is now the first-class link."""
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        self.brain.absorb(survivor, absorbed)
        prov = self.brain.conn.execute(
            "SELECT value FROM node_metadata_kv WHERE node_id = ? "
            "AND key = '_sys_archived_survivor_id'", (absorbed,)).fetchone()
        self.assertIsNotNone(prov)
        self.assertEqual(prov[0], survivor)
        self.assertEqual(len(self._absorbed_into(absorbed)), 1)

    # ── writer: plain archive does NOT write the edge ──

    def test_plain_archive_writes_no_edge(self):
        n = self._node('plain')
        r = self.brain.archive_node(n, archived_by='anchor', reason='just archive')
        self.assertTrue(r['ok'], r)
        self.assertEqual(self._absorbed_into(n), [])

    # ── chain: edge survives a later absorb of the survivor ──

    def test_chain_absorbed_into_edges_survive(self):
        """A→B→C: absorbing B into C must NOT re-archive the A→B absorbed_into
        edge (the step-3 exemption). Both hops stay live and traversable."""
        a = self._node('a')
        b = self._node('b')
        c = self._node('c')
        self.assertTrue(self.brain.absorb(b, a)['ok'])   # a absorbed into b
        self.assertTrue(self.brain.absorb(c, b)['ok'])   # b absorbed into c

        a_edges = self._absorbed_into(a)
        b_edges = self._absorbed_into(b)
        self.assertEqual(len(a_edges), 1, a_edges)
        self.assertEqual(a_edges[0][1], b)               # a -> b
        self.assertEqual(a_edges[0][3], 0)               # STILL live after b archived
        self.assertEqual(len(b_edges), 1, b_edges)
        self.assertEqual(b_edges[0][1], c)               # b -> c
        self.assertEqual(b_edges[0][3], 0)               # live

    def test_chain_resolves_end_to_end_via_resolve_live(self):
        """The metadata read-path (resolve_live, unchanged) still resolves the
        chain end-to-end — edge writing doesn't regress it."""
        a = self._node('a')
        b = self._node('b')
        c = self._node('c')
        self.brain.absorb(b, a)
        self.brain.absorb(c, b)
        out = self.brain._nodes.resolve_live([a])
        self.assertEqual(out['live'], [c])
        self.assertEqual(out['redirected'], {a: c})
        self.assertEqual(out['orphans'], [])

    # ── plain archive of other edges still archives them ──

    def test_other_edges_still_archived_on_merge(self):
        """The exemption is surgical: only absorbed_into survives the archive.
        An external edge on the absorbed node is still soft-archived."""
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        neighbor = self._node('neighbor')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on')
        self.brain.absorb(survivor, absorbed)
        # the depends_on edge_relation touching absorbed is archived
        row = self.brain.conn.execute(
            "SELECT er.archived FROM edges e "
            "JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE er.relation = 'depends_on' AND e.source_id = ?",
            (absorbed,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)


if __name__ == '__main__':
    unittest.main()
