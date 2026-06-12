"""Per-node `connect_to` in `remember_batch` and `brain_batch`.

Locks the design contract for the sibling-aware connect_to feature:

- Per-node `connect_to` lives on each node spec (remember_batch) or each
  `remember` op (brain_batch).
- Resolution order: same-batch siblings (case-insensitive exact match) first,
  then catalog via find_node_by_title(threshold=0.75).
- NEW wins: when both a sibling AND a catalog node share the title, the
  sibling created in this batch resolves. If you really mean the catalog
  node, use `revise` on its id, not duplicate-title `remember`.
- Sequencing-agnostic: declaration order of remember ops within a batch
  doesn't matter — sibling resolution happens after all nodes exist.
- Failure-resilient AND loud: an unresolved / invalid / self-referential
  connect_to entry is LOUDLY logged to `debug_log` (event_type='error',
  visible on the dashboard) and skipped; it does not fail the batch or
  other entries. Sources used:
    * `connect_to_unresolved` — title matched neither sibling nor catalog
    * `connect_to_self`       — would create a self-edge
    * `connect_to_invalid`    — entry shape malformed (no title field, wrong type)
    * `connect_to_failed`     — connect_typed raised (FK violation, etc.)
- Top-level `connect_to` (batch-level argument on remember_batch) keeps its
  existing semantics — applies to all created nodes, excludes siblings.
"""

import unittest
from tests.brain_test_base import BrainTestBase
from servers.dal import GraphDAL
from servers.daemon_dispatch import COMMAND_TABLE


def _edges_from(brain, source_id):
    """Return list of {target_id, relation, description, weight} edges
    sourced FROM source_id (active relations only)."""
    rows = brain.conn.execute(
        '''SELECT e.edge_id, e.target_id, er.relation, er.description, er.weight
           FROM edges e
           JOIN edge_relations er ON er.edge_id = e.edge_id
           WHERE e.source_id = ? AND er.archived = 0''',
        (source_id,)).fetchall()
    return [{'target_id': r[1], 'relation': r[2],
             'description': r[3] or '', 'weight': r[4]} for r in rows]


def _dispatch(brain, cmd, args):
    """Call a dispatch command via COMMAND_TABLE — same path MCP and hooks use."""
    entry = COMMAND_TABLE.get(cmd)
    if not entry:
        raise AssertionError("unknown command: %s" % cmd)
    return entry.handler(brain, args, [])


def _recent_errors(brain, source_prefix=''):
    """Return error rows from debug_log — same source the dashboard reads.
    Filter by `source` prefix (e.g. 'connect_to_')."""
    rows = brain.logs_conn.execute(
        "SELECT source, metadata FROM debug_log "
        "WHERE event_type='error' AND source LIKE ? "
        "ORDER BY id DESC", (source_prefix + '%',)
    ).fetchall()
    return [{'source': r[0], 'metadata': r[1]} for r in rows]


# ────────────────────────────────────────────────────────────────────────
# remember_batch — per-node connect_to
# ────────────────────────────────────────────────────────────────────────


class TestRememberBatchIntraBatch(BrainTestBase):
    needs_embedder = True

    def test_basic_sibling_link(self):
        """B's per-node connect_to to A creates a typed edge B→A."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'TCP migration',
             'content': 'Replaced unix sockets with TCP for the daemon.'},
            {'type': 'fact', 'title': 'Stale socket files',
             'content': 'Unix sockets leave files behind after process death.',
             'connect_to': [{'title': 'TCP migration', 'relation': 'grounds',
                             'why': 'this fact is the technical reason TCP was chosen'}]},
        ])
        b_id = result['results'][1]['id']
        edges = [e for e in _edges_from(self.brain, b_id) if e['relation'] == 'grounds']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], result['results'][0]['id'])
        self.assertIn('technical reason', edges[0]['description'])

    def test_ordering_agnostic_forward(self):
        """A defined before B that connects to A — works."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'Anchor node A',
             'content': 'First node.'},
            {'type': 'fact', 'title': 'Second node B',
             'content': 'Second.',
             'connect_to': [{'title': 'Anchor node A', 'relation': 'extends', 'why': 'B builds on A'}]},
        ])
        b_id = result['results'][1]['id']
        a_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, b_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], a_id)

    def test_ordering_agnostic_backward(self):
        """B with connect_to 'A' declared BEFORE A — must still link.

        This is the key sequencing-agnostic guarantee: encoder doesn't have
        to think about order. Sibling resolution happens AFTER all nodes
        in the batch are created.
        """
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'First-declared node B',
             'content': 'B comes first in the array.',
             'connect_to': [{'title': 'Late-declared node A', 'relation': 'extends',
                             'why': 'B references A even though A appears later in the batch'}]},
            {'type': 'principle', 'title': 'Late-declared node A',
             'content': 'A comes second.'},
        ])
        b_id = result['results'][0]['id']
        a_id = result['results'][1]['id']
        edges = [e for e in _edges_from(self.brain, b_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1, "backward-declared sibling should resolve")
        self.assertEqual(edges[0]['target_id'], a_id)

    def test_new_wins_over_catalog(self):
        """Catalog has node X; batch creates new X with same title.
        Another sibling's connect_to 'X' resolves to the NEW sibling, not catalog.
        """
        catalog = self.brain.remember(type='principle', title='Shared title',
                                      content='Pre-existing catalog node.')
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'Shared title',
             'content': 'New batch sibling with same title.'},
            {'type': 'fact', 'title': 'Linker node',
             'content': 'Links to Shared title.',
             'connect_to': [{'title': 'Shared title', 'relation': 'extends',
                             'why': 'should resolve to sibling, not catalog'}]},
        ])
        sibling_id = result['results'][0]['id']
        linker_id = result['results'][1]['id']
        edges = [e for e in _edges_from(self.brain, linker_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], sibling_id,
                         "NEW wins: should link to sibling, not catalog")
        self.assertNotEqual(edges[0]['target_id'], catalog['id'])

    def test_catalog_fallback_when_no_sibling(self):
        """connect_to title only matches catalog (no sibling) — resolves to catalog."""
        catalog = self.brain.remember(type='principle', title='Catalog-only target',
                                      content='Only in catalog, no sibling with this title.')
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'Linker node',
             'content': 'Links to a catalog-only target.',
             'connect_to': [{'title': 'Catalog-only target', 'relation': 'grounds',
                             'why': 'fallback path to catalog'}]},
        ])
        linker_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, linker_id) if e['relation'] == 'grounds']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], catalog['id'])

    def test_unmatched_title_skipped_not_failed(self):
        """connect_to title nobody matches — entry is skipped, batch still succeeds."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'Lonely node',
             'content': 'Has a connect_to nobody can match.',
             'connect_to': [{'title': 'zzz_nonexistent_title_no_match_possible',
                             'relation': 'extends', 'why': 'should be dropped'}]},
        ])
        self.assertEqual(result['nodes_created'], 1)
        node_id = result['results'][0]['id']
        edges = _edges_from(self.brain, node_id)
        # Only auto-context co_accessed edges may exist; no 'extends' from connect_to
        self.assertEqual([e for e in edges if e['relation'] == 'extends'], [])
        # Loud-by-default: dashboard sees the unresolved entry
        errors = _recent_errors(self.brain, 'connect_to_unresolved')
        self.assertGreaterEqual(len(errors), 1, "unresolved title must log loudly")

    def test_self_reference_skipped(self):
        """Node's connect_to to its own title — skipped (no self-edge)."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'Self-targeting node',
             'content': 'Has connect_to to itself.',
             'connect_to': [{'title': 'Self-targeting node', 'relation': 'extends',
                             'why': 'should not create self-edge'}]},
        ])
        node_id = result['results'][0]['id']
        edges = _edges_from(self.brain, node_id)
        # No edge to self
        self.assertEqual([e for e in edges if e['target_id'] == node_id], [])
        # Loud-by-default: dashboard sees the self-reference attempt
        errors = _recent_errors(self.brain, 'connect_to_self')
        self.assertGreaterEqual(len(errors), 1, "self-reference must log loudly")

    def test_invalid_entry_shape_skipped(self):
        """connect_to entry without title — skipped, batch still succeeds, valid entries still process."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'Valid sibling', 'content': 'First.'},
            {'type': 'fact', 'title': 'Mixed-validity linker',
             'content': 'Has one valid + one invalid connect_to.',
             'connect_to': [
                 {'why': 'no title field — invalid'},  # malformed
                 {'title': 'Valid sibling', 'relation': 'extends', 'why': 'this one works'},
             ]},
        ])
        self.assertEqual(result['nodes_created'], 2)
        linker_id = result['results'][1]['id']
        valid_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, linker_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], valid_id)
        # Loud-by-default: dashboard sees the malformed entry
        errors = _recent_errors(self.brain, 'connect_to_invalid')
        self.assertGreaterEqual(len(errors), 1, "malformed entry must log loudly")

    def test_non_string_title_returns_failed_not_raises(self):
        """A non-string `title` (e.g. an int) must NOT crash _apply_connect_to.

        Regression: `entry.get('title', '')` cleared the falsy guard for a
        non-empty non-string (int like 12345678, or a list), then the
        downstream .strip()/.lower()/regex calls raised AttributeError. That
        exception escaped _apply_connect_to — violating its "never raises"
        contract — and rolled back the entire brain_batch. The entry must
        instead surface as a {'created': [], 'failed': [...]} result.
        """
        src = self.brain.remember(type='fact', title='Non-string-title source',
                                  content='Source for the non-string title test.')
        # 12345678 is also 8 digits — would match the hex-id regex AS A STRING,
        # so the type guard must fire before the regex/.strip() ever runs.
        result = self.brain._apply_connect_to(src['id'], [
            {'title': 12345678, 'relation': 'extends', 'why': 'int title — invalid'},
        ])
        self.assertEqual(result['created'], [])
        self.assertEqual(len(result['failed']), 1)
        self.assertIn('must be a string', result['failed'][0]['reason'])
        # A list title is also non-string — same path, no crash.
        result2 = self.brain._apply_connect_to(src['id'], [
            {'title': ['not', 'a', 'string'], 'relation': 'extends', 'why': 'list title'},
        ])
        self.assertEqual(result2['created'], [])
        self.assertEqual(len(result2['failed']), 1)
        # Loud-by-default: dashboard sees the malformed entry
        errors = _recent_errors(self.brain, 'connect_to_invalid')
        self.assertGreaterEqual(len(errors), 1, "non-string title must log loudly")

    def test_relations_array_format(self):
        """Per-node connect_to with `relations: [...]` produces multiple typed edges."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'Multi-rel target', 'content': 'Anchor.'},
            {'type': 'fact', 'title': 'Multi-rel source', 'content': 'Has 2 relations to the anchor.',
             'connect_to': [{'title': 'Multi-rel target',
                             'relations': [
                                 {'relation': 'grounds', 'why': 'first reason'},
                                 {'relation': 'extends', 'why': 'second reason'},
                             ]}]},
        ])
        src_id = result['results'][1]['id']
        tgt_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, src_id) if e['target_id'] == tgt_id]
        rels = sorted(e['relation'] for e in edges)
        self.assertEqual(rels, ['extends', 'grounds'])

    def test_top_level_connect_to_unchanged_excludes_siblings(self):
        """Top-level connect_to keeps existing behavior: applies to all created nodes,
        but siblings being created in this batch are excluded — only catalog targets."""
        catalog = self.brain.remember(type='principle', title='External anchor',
                                      content='Catalog target for batch top-level.')
        # Sibling shares a title with the top-level connect_to target
        result = self.brain.remember_batch(
            nodes=[
                {'type': 'principle', 'title': 'External anchor', 'content': 'Sibling with same title.'},
                {'type': 'fact', 'title': 'Other sibling', 'content': 'Plain.'},
            ],
            connect_to=[{'title': 'External anchor', 'relation': 'related',
                         'why': 'top-level anchors all batch members'}],
        )
        # Top-level resolves to catalog, NOT the sibling (back-compat).
        # Both siblings should have an edge to the CATALOG anchor.
        sibling_id = result['results'][0]['id']
        other_id = result['results'][1]['id']
        for nid in (sibling_id, other_id):
            edges = [e for e in _edges_from(self.brain, nid) if e['relation'] == 'related']
            tgts = [e['target_id'] for e in edges]
            self.assertIn(catalog['id'], tgts,
                          "top-level connect_to must resolve to catalog target")
            self.assertNotIn(sibling_id, [t for t in tgts if t != sibling_id],
                             "top-level must NOT link to sibling with same title")


# ────────────────────────────────────────────────────────────────────────
# brain_batch — per-op connect_to inside `remember` ops
# ────────────────────────────────────────────────────────────────────────


class TestBrainBatchIntraBatch(BrainTestBase):
    needs_embedder = True

    def test_basic_sibling_link(self):
        """remember ops in brain_batch with per-op connect_to link siblings."""
        r = _dispatch(self.brain, 'brain_batch', {
            'operations': [
                {'op': 'remember', 'type': 'principle', 'title': 'BB target',
                 'content': 'Anchor.'},
                {'op': 'remember', 'type': 'fact', 'title': 'BB source',
                 'content': 'Links to BB target.',
                 'connect_to': [{'title': 'BB target', 'relation': 'grounds',
                                 'why': 'sibling link via brain_batch'}]},
            ]
        })
        self.assertTrue(r['ok'], r)
        results = r['result']['results']
        tgt_id = results[0]['result']['id']
        src_id = results[1]['result']['id']
        edges = [e for e in _edges_from(self.brain, src_id) if e['relation'] == 'grounds']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], tgt_id)

    def test_ordering_agnostic_backward_in_brain_batch(self):
        """Source remember declared BEFORE target remember — must still link.

        Mirrors the remember_batch backward test for the brain_batch surface.
        """
        r = _dispatch(self.brain, 'brain_batch', {
            'operations': [
                {'op': 'remember', 'type': 'fact', 'title': 'BB early source',
                 'content': 'Declared first.',
                 'connect_to': [{'title': 'BB late target', 'relation': 'extends',
                                 'why': 'forward reference to a sibling not yet declared'}]},
                {'op': 'remember', 'type': 'principle', 'title': 'BB late target',
                 'content': 'Declared after the source.'},
            ]
        })
        self.assertTrue(r['ok'], r)
        results = r['result']['results']
        src_id = results[0]['result']['id']
        tgt_id = results[1]['result']['id']
        edges = [e for e in _edges_from(self.brain, src_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1, "backward-declared sibling should resolve in brain_batch")
        self.assertEqual(edges[0]['target_id'], tgt_id)

    def test_mixed_ops_dont_break_connect_to(self):
        """remember + connect + revise mixed — per-op connect_to still resolves.

        Revise/connect ops interleaved with remembers must not interfere with
        the deferred sibling resolution.
        """
        # Pre-existing node to revise + connect to
        existing = self.brain.remember(type='fact', title='Existing node X',
                                       content='Pre-existing for mixed-op test.')
        r = _dispatch(self.brain, 'brain_batch', {
            'operations': [
                {'op': 'remember', 'type': 'principle', 'title': 'Mixed sibling',
                 'content': 'Created in mixed batch.'},
                {'op': 'revise', 'node_id': existing['id'],
                 'reason': 'mixed-op test', 'content': 'Revised content.'},
                {'op': 'remember', 'type': 'fact', 'title': 'Mixed source',
                 'content': 'Links to a sibling AND to existing.',
                 'connect_to': [
                     {'title': 'Mixed sibling', 'relation': 'extends',
                      'why': 'sibling link mid-mixed-batch'},
                     {'title': 'Existing node X', 'relation': 'grounds',
                      'why': 'catalog fallback in mixed-batch'},
                 ]},
            ]
        })
        self.assertTrue(r['ok'], r)
        results = r['result']['results']
        sibling_id = results[0]['result']['id']
        src_id = results[2]['result']['id']
        edges = _edges_from(self.brain, src_id)
        rel_to_target = {e['target_id']: e['relation'] for e in edges
                         if e['relation'] in ('extends', 'grounds')}
        self.assertEqual(rel_to_target.get(sibling_id), 'extends')
        self.assertEqual(rel_to_target.get(existing['id']), 'grounds')


if __name__ == '__main__':
    unittest.main()
