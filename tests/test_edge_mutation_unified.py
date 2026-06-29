"""Unified edge mutation contract tests (Stage 1B).

Mirrors tests/test_revise_unified.py for the edge side of the contract.
Covers GraphDAL.add_relation upsert, the dispatch handlers' trace emission,
and the wrapper layer (connect / connect_typed). (The Hebbian co-access bump
lives in recall_write_queue._apply_hebbian_pairs — covered by test_bg_writer.)

  Class A — Connect upsert behavior
  Class B — (removed) strengthen_relation unit tests — see test_bg_writer
  Class C — Auto-strengthen dropped (regression guard)
  Class D — Edge trace events emitted via dispatch
  Class E — Field preservation through wrappers
  Class F — Edge cases + regressions

The unified contract:
  - add_relation is field-preserving upsert with sentinel pattern
  - No auto-strengthen — Hebbian bumps live in recall_write_queue
  - Archived row → revive with passed values + defaults (semantic fresh row)
  - Each mutation emits 1 (delta, edge_relation_revised) trace event
  - connect_typed wrapper passes None=preserve through to add_relation
"""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.brain_test_base import BrainTestBase


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_node(brain, **kwargs):
    """Create a node with sensible defaults; returns the id."""
    defaults = {
        'type': 'concept',
        'title': 'Test node %d' % int(time.time() * 1000000),
        'content': 'Initial content',
    }
    defaults.update(kwargs)
    result = brain.remember(**defaults)
    return result['id']


def _get_edge_relation_row(brain, source_id, target_id, relation,
                            include_archived=True):
    """Read a single (edge, relation) row's raw state. Returns dict or None."""
    from servers.dal import GraphDAL
    edge_id = GraphDAL(brain.conn).get_edge_id(source_id, target_id)
    if not edge_id:
        return None
    where = "WHERE edge_id = ? AND relation = ?"
    params = [edge_id, relation]
    if not include_archived:
        where += " AND archived = 0"
    row = brain.conn.execute(
        "SELECT description, weight, encoding_source, archived, created_at "
        "FROM edge_relations " + where, params
    ).fetchone()
    if not row:
        return None
    return {
        'description': row[0],
        'weight': row[1],
        'encoding_source': row[2],
        'archived': row[3],
        'created_at': row[4],
        'edge_id': edge_id,
    }


def _query_edge_revise_traces(brain, edge_id, relation):
    """Query trace_events for edge_relation_revised on a specific (edge, relation)."""
    rows = brain._trace_dal.conn.execute(
        "SELECT chain_id, scale, event_type, ref_type, ref_id, summary, metadata "
        "FROM trace_events "
        "WHERE ref_type = 'edge_relation_revised' AND ref_id = ?",
        ('%s:%s' % (edge_id, relation),)
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            'chain_id': r[0], 'scale': r[1], 'event_type': r[2],
            'ref_type': r[3], 'ref_id': r[4], 'summary': r[5],
            'metadata': json.loads(r[6]) if r[6] else None,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════
# Class A — Connect upsert behavior
# ═══════════════════════════════════════════════════════════════════════

class TestConnectUpsertBehavior(BrainTestBase):
    needs_embedder = False

    def test_no_row_creates_with_passed_values(self):
        """First connect creates row with all passed field values."""
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        result = GraphDAL(self.brain.conn).add_relation(
            a, b, 'extends', description='because X', weight=0.7,
            encoding_source='test:create')

        self.assertTrue(result['created'])
        self.assertFalse(result['updated'])
        self.assertFalse(result['revived_from_archive'])

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['description'], 'because X')
        self.assertEqual(row['weight'], 0.7)
        self.assertEqual(row['encoding_source'], 'test:create')
        self.assertEqual(row['archived'], 0)

    def test_no_row_uses_defaults_for_omitted_fields(self):
        """Fresh INSERT uses sensible defaults for unspecified fields."""
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        # Only relation passed; defaults: description='', weight=0.5, encoding_source=''
        GraphDAL(self.brain.conn).add_relation(a, b, 'extends')

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['description'], '')
        self.assertEqual(row['weight'], 0.5)
        self.assertEqual(row['encoding_source'], '')

    def test_active_row_field_preserving_update(self):
        """Active row + new weight → weight updates, description preserved."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        gdal.add_relation(a, b, 'extends', description='original desc', weight=0.5)
        # Update only weight
        result = gdal.add_relation(a, b, 'extends', weight=0.9)

        self.assertFalse(result['created'])
        self.assertTrue(result['updated'])

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['weight'], 0.9)
        self.assertEqual(row['description'], 'original desc')  # preserved

    def test_active_row_no_op_when_no_fields_change(self):
        """Active row + no specified fields differ → true no-op."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        gdal.add_relation(a, b, 'extends', description='X', weight=0.5)
        # Same values → no-op
        result = gdal.add_relation(a, b, 'extends', description='X', weight=0.5)

        self.assertFalse(result['created'])
        self.assertFalse(result['updated'])
        self.assertEqual(result['deltas'], [])

    def test_active_row_partial_update_only_specified(self):
        """Multi-field update touches only specified fields."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        gdal.add_relation(a, b, 'extends',
                          description='orig', weight=0.5,
                          encoding_source='orig_source')
        # Update description + weight, preserve encoding_source
        gdal.add_relation(a, b, 'extends',
                          description='new', weight=0.9)

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['description'], 'new')
        self.assertEqual(row['weight'], 0.9)
        self.assertEqual(row['encoding_source'], 'orig_source')

    def test_archived_row_revived_with_passed_values(self):
        """Archived row + connect → revives with new values + defaults."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        gdal.add_relation(a, b, 'extends', description='old', weight=0.5)
        gdal.remove_relation(a, b, 'extends', archived_by='test')

        # Confirm archived state
        row_archived = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row_archived['archived'], 1)

        # Revive via connect with new values
        result = gdal.add_relation(a, b, 'extends',
                                   description='revived', weight=0.7)
        self.assertTrue(result['created'])
        self.assertTrue(result['revived_from_archive'])

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['archived'], 0)
        self.assertEqual(row['description'], 'revived')
        self.assertEqual(row['weight'], 0.7)

    def test_revived_row_unspecified_fields_use_defaults(self):
        """Revive on archived row resets unspecified fields to defaults (fresh row)."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        gdal.add_relation(a, b, 'extends',
                          description='old desc', weight=0.7,
                          encoding_source='old_source')
        gdal.remove_relation(a, b, 'extends', archived_by='test')

        # Revive without specifying any fields → defaults
        gdal.add_relation(a, b, 'extends')

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['archived'], 0)
        # All fields reset to defaults (semantic fresh row)
        self.assertEqual(row['description'], '')
        self.assertEqual(row['weight'], 0.5)
        self.assertEqual(row['encoding_source'], '')

    def test_multiple_relations_on_same_edge_independent(self):
        """Different relations on same (source, target) pair are independent rows."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        gdal.add_relation(a, b, 'extends', description='ext desc')
        gdal.add_relation(a, b, 'validates', description='val desc')

        ext = _get_edge_relation_row(self.brain, a, b, 'extends')
        val = _get_edge_relation_row(self.brain, a, b, 'validates')
        self.assertEqual(ext['description'], 'ext desc')
        self.assertEqual(val['description'], 'val desc')
        # Both share the same physical edge_id
        self.assertEqual(ext['edge_id'], val['edge_id'])

    def test_create_deltas_have_old_none(self):
        """First create: all deltas have old=None (semantic 'created from nothing')."""
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        result = GraphDAL(self.brain.conn).add_relation(
            a, b, 'extends', description='X', weight=0.7)
        for delta in result['deltas']:
            self.assertIsNone(delta['old'])
            self.assertIn('new', delta)


# ═══════════════════════════════════════════════════════════════════════
# Class B — strengthen_relation (Hebbian)
# ═══════════════════════════════════════════════════════════════════════

# TestStrengthenRelation removed in the DAL Phase-A cleanup: GraphDAL.
# strengthen_relation was deleted (the Hebbian co-access bump is inlined in
# recall_write_queue._apply_hebbian_pairs). The bump/cap/no-op behavior is
# covered there by tests/test_bg_writer.py::TestHebbianDrainProducesEdges
# (test_drain_strengthens_existing_co_accessed_edge).


# ═══════════════════════════════════════════════════════════════════════
# Class C — Auto-strengthen dropped (regression guard)
# ═══════════════════════════════════════════════════════════════════════

class TestAutoStrengthenDropped(BrainTestBase):
    needs_embedder = False

    def test_repeated_connect_does_not_strengthen(self):
        """Stage 1B: connect with same weight is idempotent (no auto-bump)."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        gdal.add_relation(a, b, 'extends', weight=0.5)
        gdal.add_relation(a, b, 'extends', weight=0.5)
        gdal.add_relation(a, b, 'extends', weight=0.5)

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['weight'], 0.5,
                         "weight should not bump from repeated identical connects")

    def test_explicit_weight_replaces_not_bumps(self):
        """connect with weight=0.9 REPLACES the previous weight, doesn't add to it."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        gdal.add_relation(a, b, 'extends', weight=0.5)
        gdal.add_relation(a, b, 'extends', weight=0.9)

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['weight'], 0.9,
                         "explicit new weight should replace, not add to existing")


# ═══════════════════════════════════════════════════════════════════════
# Class D — Edge trace events emitted via dispatch
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeTraceEvents(BrainTestBase):
    needs_embedder = False

    def test_connect_create_emits_trace_with_create_deltas(self):
        """Fresh create via dispatch emits trace with old=None deltas."""
        from servers.daemon_dispatch import _handle_connect
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'why', 'weight': 0.7, 'reason': 'test create',
        }, [])

        edge_id = GraphDAL(self.brain.conn).get_edge_id(a, b)
        traces = _query_edge_revise_traces(self.brain, edge_id, 'extends')
        self.assertEqual(len(traces), 1)
        meta = traces[0]['metadata']
        self.assertEqual(meta['relation'], 'extends')
        # Create deltas all have old=None
        self.assertTrue(all(d['old'] is None for d in meta['deltas']))

    def test_connect_update_emits_trace_with_field_deltas(self):
        """Update via dispatch emits trace with field-level deltas."""
        from servers.daemon_dispatch import _handle_connect
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        # Initial create
        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'orig', 'weight': 0.5, 'reason': 'init',
        }, [])
        # Update description only
        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'updated', 'reason': 'desc shift',
        }, [])

        edge_id = GraphDAL(self.brain.conn).get_edge_id(a, b)
        traces = _query_edge_revise_traces(self.brain, edge_id, 'extends')
        self.assertEqual(len(traces), 2)
        # Second trace: description delta with populated old
        update_deltas = traces[1]['metadata']['deltas']
        desc_deltas = [d for d in update_deltas if d['field'] == 'description']
        self.assertEqual(len(desc_deltas), 1)
        self.assertEqual(desc_deltas[0]['old'], 'orig')
        self.assertEqual(desc_deltas[0]['new'], 'updated')

    def test_connect_no_op_emits_no_trace(self):
        """connect with no field changes → no trace emitted."""
        from servers.daemon_dispatch import _handle_connect
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'X', 'weight': 0.5, 'reason': 'init',
        }, [])
        # Same values — no-op
        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'X', 'weight': 0.5, 'reason': 'noop',
        }, [])

        edge_id = GraphDAL(self.brain.conn).get_edge_id(a, b)
        traces = _query_edge_revise_traces(self.brain, edge_id, 'extends')
        self.assertEqual(len(traces), 1,
                         "expected 1 trace (create only), got %d" % len(traces))

    def test_connect_revive_emits_trace_with_create_deltas(self):
        """Connect on archived row → trace with create-style deltas (old=None)."""
        from servers.daemon_dispatch import _handle_connect
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        gdal.add_relation(a, b, 'extends', description='orig', weight=0.5)
        gdal.remove_relation(a, b, 'extends', archived_by='test')

        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'revived', 'weight': 0.7, 'reason': 'revive',
        }, [])

        edge_id = gdal.get_edge_id(a, b)
        traces = _query_edge_revise_traces(self.brain, edge_id, 'extends')
        # Most recent trace (the revive) should have create-style deltas
        revive_trace = traces[-1]
        self.assertTrue(all(d['old'] is None
                            for d in revive_trace['metadata']['deltas']))

    def test_disconnect_emits_archive_trace(self):
        """disconnect via brain_batch emits trace with archived flag flip."""
        from servers.daemon_dispatch import _handle_brain_batch, _handle_connect
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        # Setup: create the edge first
        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'X', 'weight': 0.5, 'reason': 'init',
        }, [])

        # Then disconnect via brain_batch
        _handle_brain_batch(self.brain, {
            'operations': [
                {'op': 'disconnect', 'source_id': a, 'target_id': b,
                 'relation': 'extends', 'reason': 'cleanup'},
            ],
        }, [])

        edge_id = GraphDAL(self.brain.conn).get_edge_id(a, b)
        traces = _query_edge_revise_traces(self.brain, edge_id, 'extends')
        # Should have create trace + disconnect trace
        self.assertEqual(len(traces), 2)
        disc_trace = traces[-1]
        deltas = disc_trace['metadata']['deltas']
        archived_deltas = [d for d in deltas if d['field'] == 'archived']
        self.assertEqual(len(archived_deltas), 1)
        self.assertEqual(archived_deltas[0]['old'], 0)
        self.assertEqual(archived_deltas[0]['new'], 1)

    def test_chain_id_override_respected(self):
        """Caller-provided chain_id is used verbatim."""
        from servers.daemon_dispatch import _handle_connect
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'X', 'reason': 'r',
            'chain_id': 's2-20260504-aspect_integration',
        }, [])

        edge_id = GraphDAL(self.brain.conn).get_edge_id(a, b)
        traces = _query_edge_revise_traces(self.brain, edge_id, 'extends')
        self.assertEqual(traces[0]['chain_id'], 's2-20260504-aspect_integration')

    def test_scale_inferred_from_encoding_source(self):
        """encoding_source='s2:foo' → trace.scale='s2'."""
        from servers.daemon_dispatch import _handle_connect
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'X', 'reason': 'r',
            'encoding_source': 's2:healer',
        }, [])

        edge_id = GraphDAL(self.brain.conn).get_edge_id(a, b)
        traces = _query_edge_revise_traces(self.brain, edge_id, 'extends')
        self.assertEqual(traces[0]['scale'], 's2')


# ═══════════════════════════════════════════════════════════════════════
# Class E — Field preservation through wrappers
# ═══════════════════════════════════════════════════════════════════════

class TestWrapperFieldPreservation(BrainTestBase):
    needs_embedder = False

    def test_connect_typed_None_default_preserves(self):
        """connect_typed(description=None) preserves existing on update."""
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        self.brain.connect_typed(a, b, 'extends',
                                 description='original', weight=0.5)
        # Second call without description (defaults to None) → preserve
        self.brain.connect_typed(a, b, 'extends', weight=0.9)

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['description'], 'original')
        self.assertEqual(row['weight'], 0.9)

    def test_connect_typed_explicit_empty_clears(self):
        """connect_typed(description='') explicitly clears the description."""
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        self.brain.connect_typed(a, b, 'extends',
                                 description='original', weight=0.5)
        self.brain.connect_typed(a, b, 'extends', description='')

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['description'], '',
                         "explicit '' should clear, not preserve")

    def test_connect_typed_explicit_value_replaces(self):
        """connect_typed(description='new') replaces existing."""
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        self.brain.connect_typed(a, b, 'extends', description='orig')
        self.brain.connect_typed(a, b, 'extends', description='new')

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['description'], 'new')

    def test_dispatch_handler_preserves_when_no_description_arg(self):
        """_handle_connect with no description arg → preserves existing."""
        from servers.daemon_dispatch import _handle_connect
        a = _make_node(self.brain)
        b = _make_node(self.brain)

        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'original', 'weight': 0.5, 'reason': 'init',
        }, [])
        # Same call without description → preserves
        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'weight': 0.9, 'reason': 'weight bump',
        }, [])

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['description'], 'original')
        self.assertEqual(row['weight'], 0.9)


# ═══════════════════════════════════════════════════════════════════════
# Class F — Edge cases + regressions
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases(BrainTestBase):
    needs_embedder = False

    def test_invalid_source_node_raises(self):
        """add_relation raises ValueError when source node doesn't exist."""
        from servers.dal import GraphDAL
        b = _make_node(self.brain)
        with self.assertRaises(ValueError) as cm:
            GraphDAL(self.brain.conn).add_relation(
                'nonexistent_source_xxx', b, 'extends')
        self.assertIn('source', str(cm.exception).lower())

    def test_invalid_target_node_raises(self):
        """add_relation raises ValueError when target node doesn't exist."""
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        with self.assertRaises(ValueError) as cm:
            GraphDAL(self.brain.conn).add_relation(
                a, 'nonexistent_target_xxx', 'extends')
        self.assertIn('target', str(cm.exception).lower())

    def test_unicode_in_description_survives(self):
        """Unicode in description round-trips correctly through upsert."""
        from servers.dal import GraphDAL
        a = _make_node(self.brain)
        b = _make_node(self.brain)
        unicode_desc = 'Anchor — 持続 — émergent — 🧠'
        GraphDAL(self.brain.conn).add_relation(
            a, b, 'extends', description=unicode_desc)

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['description'], unicode_desc)

    def test_disconnect_nonexistent_no_error(self):
        """remove_relation on missing (source, target, relation) → no-op no error."""
        from servers.dal import GraphDAL
        # No edge exists yet — should silently no-op
        GraphDAL(self.brain.conn).remove_relation(
            'nonexistent_a', 'nonexistent_b', 'extends', archived_by='test')
        # If we got here, no exception was raised. Pass.

    def test_long_description_replace(self):
        """Large description (10KB) replaces cleanly through upsert."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)
        big = 'x' * 10000

        gdal.add_relation(a, b, 'extends', description='small')
        gdal.add_relation(a, b, 'extends', description=big)

        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(len(row['description']), 10000)
        self.assertEqual(row['description'], big)


# ═══════════════════════════════════════════════════════════════════════
# Class G — Creator attribution (encoding_source), creation-only
#
# encoding_source is the CREATOR mark — set once at birth, never rewritten on an
# edit or re-connect. It's a denormalized cache of the creation event the trace
# log recorded, so a later overwrite would make it drift from that event. Two
# birth points, both defaulting an absent source to 'anchor' (the convention's
# "direct via MCP"; the encoder pre-stamps its own source upstream):
#   - nodes: remember() INSERT defaults `encoding_source or 'anchor'`
#   - edges: the connect handlers default `encoding_source or 'anchor'`, applied
#     on add_relation's CREATE branch only — an active-row update preserves it.
# connect_to edges inherit their just-created node's source via _apply_connect_to
# (always fresh, so create-only is automatic).
#
# Regression guards for: edges landing '' on create (the reported bug), the
# re-connect clobber, and the revise_batch node-relabel — the two clobbers the
# earlier proxy-stamp approach introduced.
# ═══════════════════════════════════════════════════════════════════════

class TestCreatorAttribution(BrainTestBase):
    needs_embedder = False

    # ── connect_to edges inherit the (just-created) node's creator ──

    def test_remember_connect_to_defaults_to_anchor(self):
        """remember(connect_to=) with no source → edge tagged 'anchor'.
        The reported bug: this edge used to land encoding_source ''."""
        target = _make_node(self.brain, title='Target alpha')
        res = self.brain.remember(
            type='concept', title='Source alpha', content='c',
            connect_to=[{'title': target, 'relation': 'extends', 'why': 'because'}])
        row = _get_edge_relation_row(self.brain, res['id'], target, 'extends')
        self.assertIsNotNone(row, "connect_to edge should exist")
        self.assertEqual(row['encoding_source'], 'anchor')

    def test_remember_connect_to_not_empty_regression(self):
        """Regression: a connect_to edge must never carry encoding_source ''."""
        target = _make_node(self.brain, title='Target gamma')
        res = self.brain.remember(
            type='concept', title='Source gamma', content='c',
            connect_to=[{'title': target, 'relation': 'extends', 'why': 'w'}])
        row = _get_edge_relation_row(self.brain, res['id'], target, 'extends')
        self.assertNotEqual(row['encoding_source'], '')

    def test_remember_connect_to_inherits_explicit_source(self):
        """connect_to edges inherit the node's explicit source — the encoder's
        edges become 'encoder:sonnet', not 'anchor' and not ''."""
        target = _make_node(self.brain, title='Target beta')
        res = self.brain.remember(
            type='concept', title='Source beta', content='c',
            encoding_source='encoder:sonnet',
            connect_to=[{'title': target, 'relation': 'grounds', 'why': 'w'}])
        row = _get_edge_relation_row(self.brain, res['id'], target, 'grounds')
        self.assertEqual(row['encoding_source'], 'encoder:sonnet')

    def test_remember_batch_per_node_connect_to_anchor(self):
        target = _make_node(self.brain, title='Target delta')
        res = self.brain.remember_batch(nodes=[
            {'type': 'concept', 'title': 'Batch src', 'content': 'c',
             'connect_to': [{'title': target, 'relation': 'extends', 'why': 'w'}]},
        ])
        src = res['results'][0]['id']
        row = _get_edge_relation_row(self.brain, src, target, 'extends')
        self.assertEqual(row['encoding_source'], 'anchor')

    def test_remember_batch_per_node_connect_to_inherits_source(self):
        target = _make_node(self.brain, title='Target delta2')
        res = self.brain.remember_batch(nodes=[
            {'type': 'concept', 'title': 'Batch src2', 'content': 'c',
             'encoding_source': 's2:consolidation',
             'connect_to': [{'title': target, 'relation': 'extends', 'why': 'w'}]},
        ])
        src = res['results'][0]['id']
        row = _get_edge_relation_row(self.brain, src, target, 'extends')
        self.assertEqual(row['encoding_source'], 's2:consolidation')

    def test_remember_batch_top_level_connect_to_anchor(self):
        """The batch-wide connect_to mechanism (LIKE-resolved title) also tags."""
        tgt = _make_node(self.brain, title='ZZZUniqueTopTarget')
        res = self.brain.remember_batch(
            nodes=[{'type': 'concept', 'title': 'Top src', 'content': 'c'}],
            connect_to=['ZZZUniqueTopTarget'])
        src = res['results'][0]['id']
        row = _get_edge_relation_row(self.brain, src, tgt, 'related')
        self.assertIsNotNone(row, "top-level connect_to edge should exist")
        self.assertEqual(row['encoding_source'], 'anchor')

    def test_brain_batch_remember_connect_to_defaults_to_anchor(self):
        """brain_batch remember+connect_to with NO top-level source → edge
        'anchor' (the deferred connect_to falls back to 'anchor' — no proxy)."""
        from servers.daemon_dispatch import _handle_brain_batch
        target = _make_node(self.brain, title='Target epsilon')
        r = _handle_brain_batch(self.brain, {
            'operations': [
                {'op': 'remember', 'type': 'concept', 'title': 'BB src',
                 'content': 'c',
                 'connect_to': [{'title': target, 'relation': 'extends',
                                 'why': 'w'}]},
            ],
        }, [])
        src = r['result']['results'][0]['result']['id']
        row = _get_edge_relation_row(self.brain, src, target, 'extends')
        self.assertEqual(row['encoding_source'], 'anchor')

    # ── standalone connect/connect_batch: 'anchor' on create ──

    def test_handle_connect_defaults_to_anchor_on_create(self):
        """Direct connect with no source → fresh edge tagged 'anchor'
        (via _handle_connect's `or 'anchor'`, not a proxy stamp)."""
        from servers.daemon_dispatch import _handle_connect
        a = _make_node(self.brain)
        b = _make_node(self.brain)
        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'w', 'reason': 'r',
        }, [])
        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['encoding_source'], 'anchor')

    def test_connect_batch_defaults_to_anchor(self):
        from servers.daemon_dispatch import _handle_connect_batch
        a = _make_node(self.brain)
        b = _make_node(self.brain)
        _handle_connect_batch(self.brain, {
            'connections': [{'source_id': a, 'target_id': b,
                             'relation': 'extends', 'description': 'w'}],
        }, [])
        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['encoding_source'], 'anchor')

    def test_brain_batch_connect_op_defaults_to_anchor(self):
        """brain_batch connect op with NO top-level source → 'anchor'."""
        from servers.daemon_dispatch import _handle_brain_batch
        a = _make_node(self.brain)
        b = _make_node(self.brain)
        _handle_brain_batch(self.brain, {
            'operations': [
                {'op': 'connect', 'source_id': a, 'target_id': b,
                 'relation': 'extends', 'description': 'w'},
            ],
        }, [])
        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['encoding_source'], 'anchor')

    # ── creation-only: an active-row update never relabels the creator ──

    def test_handle_connect_preserves_source_on_reconnect(self):
        """Re-connecting an existing edge updates description but must NOT
        relabel its creator — the clobber the proxy-stamp approach introduced."""
        from servers.daemon_dispatch import _handle_connect
        a = _make_node(self.brain)
        b = _make_node(self.brain)
        # Born from the encoder
        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'orig', 'reason': 'init',
            'encoding_source': 'encoder:sonnet',
        }, [])
        # Anchor re-connects with no source — must preserve 'encoder:sonnet'
        _handle_connect(self.brain, {
            'source_id': a, 'target_id': b, 'relation': 'extends',
            'description': 'updated', 'reason': 'reconnect',
        }, [])
        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['encoding_source'], 'encoder:sonnet',
                         "re-connect must preserve the original creator")
        self.assertEqual(row['description'], 'updated')

    def test_add_relation_creator_immutable_on_update(self):
        """add_relation: an active-row update preserves encoding_source even
        when an explicit, DIFFERENT source is passed (creator set once)."""
        from servers.dal import GraphDAL
        gdal = GraphDAL(self.brain.conn)
        a = _make_node(self.brain)
        b = _make_node(self.brain)
        gdal.add_relation(a, b, 'extends', description='orig',
                          encoding_source='encoder:sonnet')
        # Pass a different source on update — must be ignored (preserved)
        gdal.add_relation(a, b, 'extends', description='new',
                          encoding_source='anchor')
        row = _get_edge_relation_row(self.brain, a, b, 'extends')
        self.assertEqual(row['encoding_source'], 'encoder:sonnet',
                         "active-row update must not relabel the creator")
        self.assertEqual(row['description'], 'new')

    # ── revise_batch must not relabel a node's creator (leak guard) ──

    def test_revise_batch_does_not_clobber_node_source(self):
        """A bulk-edit must not rewrite a node's creator. The dispatch layer
        injects encoding_source for trace attribution, but revise_batch must
        never write it onto the node — the HIGH-severity leak the review found."""
        from servers.daemon_dispatch import _handle_revise_batch
        r = self.brain.remember(type='concept', title='Encoder node',
                                content='c', encoding_source='encoder:sonnet')
        nid = r['id']
        # Anchor revise_batch — dispatch injects top-level 'anchor' into each spec
        _handle_revise_batch(self.brain, {
            'encoding_source': 'anchor',
            'revisions': [{'node_id': nid, 'reason': 'typo fix',
                           'title': 'Encoder node (fixed)'}],
        }, [])
        row = self.brain.conn.execute(
            "SELECT encoding_source, title FROM nodes WHERE id = ?",
            (nid,)).fetchone()
        self.assertEqual(row[0], 'encoder:sonnet',
                         "revise_batch must not relabel the node's creator")
        self.assertEqual(row[1], 'Encoder node (fixed)')


if __name__ == '__main__':
    unittest.main()
