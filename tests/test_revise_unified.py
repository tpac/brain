"""Unified revise contract tests (Stage 1A).

Covers the post-refactor behavior of brain.revise() and brain.revise_batch():

  Class A — Per-field replace + preservation
  Class B — Immutable handling (skip + warn, never fail call)
  Class C — Locked-archive guard (warning + trace event)
  Class D — Deltas (computed before any write, returned in result)
  Class E — Trace events emitted via dispatch
  Class F — revise_batch threading
  Class G — Edge cases + regressions

The unified contract:
  - Immutable {id, created_at, locked} → skipped, surfaces in `warnings`
  - All other fields → REPLACE (specified) / PRESERVE (unspecified)
  - History → trace events (event_type='delta', ref_type='node_revised')
  - No more _sys_revision_history KV blob
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

def _kv_value(brain, node_id, key):
    """Read a single value from node_metadata_kv. Returns None if absent."""
    row = brain.conn.execute(
        "SELECT value FROM node_metadata_kv WHERE node_id = ? AND key = ?",
        (node_id, key)).fetchone()
    return row[0] if row else None


def _kv_keys(brain, node_id):
    """All metadata keys present for a node."""
    rows = brain.conn.execute(
        "SELECT key FROM node_metadata_kv WHERE node_id = ?",
        (node_id,)).fetchall()
    return {r[0] for r in rows}


def _query_revise_traces(brain, node_id):
    """Query trace_events for node_revised events on a specific node."""
    rows = brain._trace_dal.conn.execute(
        "SELECT chain_id, scale, event_type, ref_type, ref_id, summary, metadata "
        "FROM trace_events "
        "WHERE ref_type = 'node_revised' AND ref_id = ?", (node_id,)
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            'chain_id': r[0], 'scale': r[1], 'event_type': r[2],
            'ref_type': r[3], 'ref_id': r[4], 'summary': r[5],
            'metadata': json.loads(r[6]) if r[6] else None,
        })
    return out


def _make_node(brain, **kwargs):
    """Create a node with sensible defaults; returns the id."""
    defaults = {
        'type': 'concept',
        'title': 'Test node %d' % int(time.time() * 1000),
        'content': 'Initial content',
    }
    defaults.update(kwargs)
    result = brain.remember(**defaults)
    return result['id']


# ═══════════════════════════════════════════════════════════════════════
# Class A — Per-field replace + preservation
# ═══════════════════════════════════════════════════════════════════════

class TestPerFieldReplace(BrainTestBase):
    needs_embedder = False

    def test_top_level_field_replace(self):
        """revise(confidence=0.9) updates confidence; title/content preserved."""
        nid = _make_node(self.brain, title='T', content='C', confidence=0.5)
        result = self.brain.revise(node_id=nid, confidence=0.9, reason='bump')

        self.assertNotIn('error', result)
        row = self.brain.conn.execute(
            "SELECT title, content, confidence FROM nodes WHERE id = ?",
            (nid,)).fetchone()
        self.assertEqual(row[0], 'T')
        self.assertEqual(row[1], 'C')
        self.assertEqual(row[2], 0.9)

    def test_kv_field_replace(self):
        """revise(situation='X') updates KV row; reasoning preserved."""
        nid = _make_node(self.brain, situation='When debugging boot',
                         reasoning='Original reasoning')
        result = self.brain.revise(node_id=nid, situation='When debugging recall',
                                   reason='shift focus')

        self.assertNotIn('error', result)
        self.assertEqual(_kv_value(self.brain, nid, 'situation'),
                         'When debugging recall')
        self.assertEqual(_kv_value(self.brain, nid, 'reasoning'),
                         'Original reasoning')

    def test_multi_field_call_top_and_kv(self):
        """Single revise can update both a top-level field and a KV field."""
        nid = _make_node(self.brain, confidence=0.5,
                         situation='Original situation',
                         reasoning='Original reasoning')
        self.brain.revise(node_id=nid, confidence=0.9,
                          situation='New situation', reason='multi-update')

        row = self.brain.conn.execute(
            "SELECT confidence FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 0.9)
        self.assertEqual(_kv_value(self.brain, nid, 'situation'), 'New situation')
        self.assertEqual(_kv_value(self.brain, nid, 'reasoning'),
                         'Original reasoning')  # untouched

    def test_content_replace_is_exact(self):
        """Content REPLACES (no append). Readback must match exactly."""
        nid = _make_node(self.brain, content='Original')
        self.brain.revise(node_id=nid, content='Replaced', reason='r')

        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 'Replaced')
        self.assertNotIn('Original', row[0])

    def test_content_via_updates_dict_replaces_and_verifies(self):
        """content passed inside updates={...} replaces AND is verified.

        Pre-Stage 1A bug: verification only fired when content was passed as
        the named arg, not via updates. Caught during code review of B-B.1.
        """
        nid = _make_node(self.brain, content='Original')
        result = self.brain.revise(node_id=nid, reason='r',
                                   updates={'content': 'NewContent'})
        self.assertNotIn('error', result)
        # Verification should have fired and confirmed the write
        self.assertTrue(result.get('verified', False),
                        "verification did not fire for content-via-updates: %r"
                        % result.get('verification_failures'))
        # Readback confirms replace
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 'NewContent')
        # Delta should capture the change
        content_deltas = [d for d in result.get('deltas', [])
                          if d['field'] == 'content']
        self.assertEqual(len(content_deltas), 1)
        self.assertEqual(content_deltas[0]['old'], 'Original')
        self.assertEqual(content_deltas[0]['new'], 'NewContent')

    def test_unspecified_kv_fields_preserved(self):
        """Revising one KV field does not touch other KV fields."""
        nid = _make_node(self.brain,
                         situation='S', reasoning='R',
                         user_raw_quote='U', anchor_raw_quote='A')
        self.brain.revise(node_id=nid, situation='S2', reason='r')

        # Other KV keys still present and unchanged
        self.assertEqual(_kv_value(self.brain, nid, 'reasoning'), 'R')
        self.assertEqual(_kv_value(self.brain, nid, 'user_raw_quote'), 'U')
        self.assertEqual(_kv_value(self.brain, nid, 'anchor_raw_quote'), 'A')


# ═══════════════════════════════════════════════════════════════════════
# Class B — Immutable handling (skip + warn, never fail the call)
# ═══════════════════════════════════════════════════════════════════════

class TestImmutableHandling(BrainTestBase):
    needs_embedder = False

    def test_immutable_id_skipped_with_warning(self):
        """revise(id=...) is skipped; no error returned; warning surfaces."""
        nid = _make_node(self.brain)
        result = self.brain.revise(node_id=nid, reason='r',
                                   updates={'id': 'other'})
        self.assertNotIn('error', result)
        warnings = result.get('warnings', [])
        self.assertTrue(any('id' in w for w in warnings),
                        "warnings should mention skipped 'id', got: %r" % warnings)
        # node id unchanged
        row = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], nid)

    def test_immutable_created_at_skipped(self):
        """revise(created_at=...) is skipped with warning."""
        nid = _make_node(self.brain)
        original_created = self.brain.conn.execute(
            "SELECT created_at FROM nodes WHERE id = ?", (nid,)).fetchone()[0]

        result = self.brain.revise(node_id=nid, reason='r',
                                   updates={'created_at': '1999-01-01'})
        warnings = result.get('warnings', [])
        self.assertTrue(any('created_at' in w for w in warnings))
        new_created = self.brain.conn.execute(
            "SELECT created_at FROM nodes WHERE id = ?", (nid,)).fetchone()[0]
        self.assertEqual(new_created, original_created)

    def test_immutable_locked_skipped(self):
        """revise(locked=True) is skipped — locking is a separate path."""
        nid = _make_node(self.brain)
        result = self.brain.revise(node_id=nid, reason='r',
                                   updates={'locked': True})
        warnings = result.get('warnings', [])
        self.assertTrue(any('locked' in w for w in warnings))

    def test_legacy_keywords_routes_to_kv_not_crash(self):
        """revise(keywords=...) must NOT crash on the v28-dropped column. keywords
        is no longer a nodes column, so it falls through the generic extra-fields
        path to node_metadata_kv (a legacy/unknown field is stored as KV, not
        special-cased), and a real field passed in the same call still applies."""
        nid = _make_node(self.brain, confidence=0.5)
        result = self.brain.revise(node_id=nid, reason='r',
                                   updates={'keywords': 'dead kw', 'confidence': 0.9})
        self.assertNotIn('error', result)
        # co-passed real field applied
        row = self.brain.conn.execute(
            "SELECT confidence FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 0.9)
        # legacy field routed to KV (not lost, not crashed)
        self.assertEqual(_kv_value(self.brain, nid, 'keywords'), 'dead kw')

    def test_multiple_immutables_all_skipped_others_apply(self):
        """Multi-field call with mixed immutable/valid: skips immutable, applies others."""
        nid = _make_node(self.brain, confidence=0.5)
        result = self.brain.revise(
            node_id=nid, reason='r',
            updates={'id': 'x', 'locked': True, 'confidence': 0.9})

        self.assertNotIn('error', result)
        warnings = result.get('warnings', [])
        # Both immutables flagged
        self.assertTrue(any('id' in w for w in warnings))
        self.assertTrue(any('locked' in w for w in warnings))
        # Valid field applied
        row = self.brain.conn.execute(
            "SELECT confidence FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 0.9)

    def test_immutable_not_written_to_kv(self):
        """Immutable field passed to revise must NOT leak into node_metadata_kv."""
        nid = _make_node(self.brain)
        self.brain.revise(node_id=nid, reason='r',
                          updates={'id': 'other', 'situation': 'OK'})
        kv_keys = _kv_keys(self.brain, nid)
        self.assertNotIn('id', kv_keys, "immutable 'id' leaked into KV")
        self.assertIn('situation', kv_keys)


# ═══════════════════════════════════════════════════════════════════════
# Class C — Locked-archive guard (warning + trace event)
# ═══════════════════════════════════════════════════════════════════════

class TestLockedArchiveGuard(BrainTestBase):
    needs_embedder = False

    def test_locked_node_archive_blocked(self):
        """revise(archived=True) on locked node is blocked, warning surfaces."""
        nid = _make_node(self.brain)
        # Manually lock the node (locked is immutable via revise; set via SQL)
        self.brain.conn.execute(
            "UPDATE nodes SET locked = 1 WHERE id = ?", (nid,))
        self.brain.conn.commit()

        result = self.brain.revise(node_id=nid, reason='attempt archive',
                                   updates={'archived': True})

        self.assertNotIn('error', result)
        # Archive did NOT land
        archived = self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (nid,)).fetchone()[0]
        self.assertEqual(archived, 0)
        # Warning surfaced
        warnings = result.get('warnings', [])
        self.assertTrue(
            any('archive blocked' in w.lower() for w in warnings),
            "expected archive-blocked warning, got: %r" % warnings)

    def test_locked_archive_other_fields_still_apply(self):
        """When archive blocked on locked node, other fields still update."""
        nid = _make_node(self.brain, confidence=0.5)
        self.brain.conn.execute(
            "UPDATE nodes SET locked = 1 WHERE id = ?", (nid,))
        self.brain.conn.commit()

        self.brain.revise(node_id=nid, reason='r',
                          updates={'archived': True, 'confidence': 0.9})

        row = self.brain.conn.execute(
            "SELECT archived, confidence FROM nodes WHERE id = ?",
            (nid,)).fetchone()
        self.assertEqual(row[0], 0)        # archive blocked
        self.assertEqual(row[1], 0.9)      # confidence applied

    def test_locked_archive_emits_trace_event_with_warning(self):
        """Even with no deltas, archive-blocked must emit a trace event with warnings."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain)
        self.brain.conn.execute(
            "UPDATE nodes SET locked = 1 WHERE id = ?", (nid,))
        self.brain.conn.commit()

        # Dispatch path emits the trace
        graph_changes = []
        _handle_revise(self.brain, {
            'node_id': nid, 'reason': 'attempt archive',
            'archived': True,
            'encoding_source': 'test:locked_archive',
        }, graph_changes)

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(len(traces), 1,
                         "expected 1 trace, got %d" % len(traces))
        meta = traces[0]['metadata']
        self.assertEqual(meta['deltas'], [])
        self.assertTrue(len(meta['warnings']) >= 1,
                        "expected warnings in trace, got: %r" % meta)
        self.assertTrue(any('archive' in w.lower() for w in meta['warnings']))


# ═══════════════════════════════════════════════════════════════════════
# Class D — Deltas
# ═══════════════════════════════════════════════════════════════════════

class TestDeltas(BrainTestBase):
    needs_embedder = False

    def test_deltas_for_top_level_field(self):
        """Result dict contains delta for each top-level field changed."""
        nid = _make_node(self.brain, confidence=0.5)
        result = self.brain.revise(node_id=nid, confidence=0.9, reason='r')

        deltas = result.get('deltas', [])
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]['field'], 'confidence')
        self.assertEqual(deltas[0]['old'], 0.5)
        self.assertEqual(deltas[0]['new'], 0.9)

    def test_deltas_for_kv_field(self):
        """Result dict contains delta for each KV field changed."""
        nid = _make_node(self.brain, situation='Original')
        result = self.brain.revise(node_id=nid, situation='New', reason='r')

        deltas = result.get('deltas', [])
        situation_deltas = [d for d in deltas if d['field'] == 'situation']
        self.assertEqual(len(situation_deltas), 1)
        self.assertEqual(situation_deltas[0]['old'], 'Original')
        self.assertEqual(situation_deltas[0]['new'], 'New')

    def test_delta_for_content(self):
        """Content delta uses the resolved new content value."""
        nid = _make_node(self.brain, content='Original content')
        result = self.brain.revise(node_id=nid, content='New content', reason='r')

        deltas = result.get('deltas', [])
        content_deltas = [d for d in deltas if d['field'] == 'content']
        self.assertEqual(len(content_deltas), 1)
        self.assertEqual(content_deltas[0]['old'], 'Original content')
        self.assertEqual(content_deltas[0]['new'], 'New content')

    def test_no_delta_when_value_unchanged(self):
        """Setting a field to its current value produces no delta."""
        nid = _make_node(self.brain, confidence=0.5)
        result = self.brain.revise(node_id=nid, confidence=0.5, reason='r')
        deltas = result.get('deltas', [])
        self.assertEqual(deltas, [],
                         "expected no deltas, got: %r" % deltas)

    def test_multiple_deltas_in_single_call(self):
        """Multi-field revise produces multiple deltas in one result."""
        nid = _make_node(self.brain, confidence=0.5,
                         situation='S', reasoning='R')
        result = self.brain.revise(node_id=nid, reason='multi',
                                   updates={'confidence': 0.9,
                                            'situation': 'S2', 'reasoning': 'R2'})
        deltas = result.get('deltas', [])
        self.assertEqual(len(deltas), 3)
        fields = {d['field'] for d in deltas}
        self.assertEqual(fields, {'confidence', 'situation', 'reasoning'})

    def test_no_sys_revision_history_written(self):
        """After revise, _sys_revision_history must NOT appear in KV — Stage 1A."""
        nid = _make_node(self.brain, content='Original')
        self.brain.revise(node_id=nid, content='Changed', reason='r')
        kv_keys = _kv_keys(self.brain, nid)
        self.assertNotIn('_sys_revision_history', kv_keys,
                         "Stage 1A regression: _sys_revision_history was written")

    def test_delta_for_kv_field_not_previously_set(self):
        """Setting a KV field that didn't exist yields a delta with old=None."""
        nid = _make_node(self.brain)  # no situation set
        result = self.brain.revise(node_id=nid, situation='New', reason='r')
        deltas = [d for d in result.get('deltas', []) if d['field'] == 'situation']
        self.assertEqual(len(deltas), 1)
        self.assertIsNone(deltas[0]['old'])
        self.assertEqual(deltas[0]['new'], 'New')


# ═══════════════════════════════════════════════════════════════════════
# Class E — Trace events emitted via dispatch
# ═══════════════════════════════════════════════════════════════════════

class TestTraceEvents(BrainTestBase):
    needs_embedder = False

    def test_single_revise_emits_one_trace(self):
        """Dispatch revise → exactly one node_revised trace event."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain, situation='Original')
        graph_changes = []
        _handle_revise(self.brain, {
            'node_id': nid, 'reason': 'r', 'situation': 'New',
        }, graph_changes)

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]['ref_type'], 'node_revised')
        self.assertEqual(traces[0]['event_type'], 'delta')

    def test_revise_batch_emits_one_trace_per_row(self):
        """Dispatch revise_batch with 3 nodes → 3 trace events."""
        from servers.daemon_dispatch import _handle_revise_batch

        n1 = _make_node(self.brain, situation='S1')
        n2 = _make_node(self.brain, situation='S2')
        n3 = _make_node(self.brain, situation='S3')

        graph_changes = []
        _handle_revise_batch(self.brain, {
            'revisions': [
                {'node_id': n1, 'reason': 'r1', 'situation': 'NS1'},
                {'node_id': n2, 'reason': 'r2', 'situation': 'NS2'},
                {'node_id': n3, 'reason': 'r3', 'situation': 'NS3'},
            ]
        }, graph_changes)

        for nid in (n1, n2, n3):
            traces = _query_revise_traces(self.brain, nid)
            self.assertEqual(len(traces), 1,
                             "node %s had %d traces, expected 1" % (
                                 nid[:8], len(traces)))

    def test_trace_metadata_shape(self):
        """Trace metadata matches REVISE_METADATA_SHAPE."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain, situation='Original')
        _handle_revise(self.brain, {
            'node_id': nid, 'reason': 'shape test', 'situation': 'New',
            'encoding_source': 'test:shape',
        }, [])

        traces = _query_revise_traces(self.brain, nid)
        meta = traces[0]['metadata']
        self.assertIn('node_id', meta)
        self.assertIn('reason', meta)
        self.assertIn('encoding_source', meta)
        self.assertIn('deltas', meta)
        self.assertIn('warnings', meta)
        self.assertEqual(meta['reason'], 'shape test')
        self.assertEqual(meta['encoding_source'], 'test:shape')
        self.assertEqual(len(meta['deltas']), 1)
        self.assertEqual(meta['deltas'][0]['field'], 'situation')

    def test_chain_id_override_respected(self):
        """Caller-provided chain_id is used verbatim."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain, situation='X')
        _handle_revise(self.brain, {
            'node_id': nid, 'reason': 'r', 'situation': 'Y',
            'chain_id': 's2-20260504-aspect_integration',
        }, [])

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(traces[0]['chain_id'], 's2-20260504-aspect_integration')

    def test_chain_id_default_is_date_based(self):
        """No chain_id arg → date-based fallback chain (`{scale}-{YYYYMMDD}-revise`)."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain, situation='X')
        _handle_revise(self.brain, {
            'node_id': nid, 'reason': 'r', 'situation': 'Y',
        }, [])

        traces = _query_revise_traces(self.brain, nid)
        chain = traces[0]['chain_id']
        # Format: s0-YYYYMMDD-revise (no encoding_source → s0)
        self.assertTrue(chain.startswith('s0-'),
                        "expected chain to start with 's0-', got: %s" % chain)
        self.assertTrue(chain.endswith('-revise'),
                        "expected chain to end with '-revise', got: %s" % chain)

    def test_scale_inferred_from_encoding_source(self):
        """encoding_source='s2:foo' → trace.scale='s2'."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain, situation='X')
        _handle_revise(self.brain, {
            'node_id': nid, 'reason': 'r', 'situation': 'Y',
            'encoding_source': 's2:healer',
        }, [])

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(traces[0]['scale'], 's2')

    def test_no_trace_when_no_changes_no_warnings(self):
        """revise(field=<same value>) with no warnings → no trace emitted."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain, confidence=0.5)
        _handle_revise(self.brain, {
            'node_id': nid, 'reason': 'noop', 'confidence': 0.5,
        }, [])

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(traces, [],
                         "expected no traces for no-op revise, got: %r" % traces)


# ═══════════════════════════════════════════════════════════════════════
# Class F — revise_batch threading
# ═══════════════════════════════════════════════════════════════════════

class TestReviseBatchThreading(BrainTestBase):
    needs_embedder = False

    def test_per_row_results_contain_deltas_and_warnings(self):
        """revise_batch result['results'][i] has deltas and warnings keys."""
        nid = _make_node(self.brain, situation='Original')
        result = self.brain.revise_batch([
            {'node_id': nid, 'reason': 'r', 'situation': 'New'},
        ])
        rows = result['results']
        self.assertEqual(len(rows), 1)
        self.assertIn('deltas', rows[0])
        self.assertIn('warnings', rows[0])
        self.assertEqual(len(rows[0]['deltas']), 1)
        self.assertEqual(rows[0]['deltas'][0]['field'], 'situation')

    def test_mixed_success_and_error_rows(self):
        """Bad node_id row gets error; valid row gets deltas. Both in results."""
        nid = _make_node(self.brain, situation='Original')
        result = self.brain.revise_batch([
            {'node_id': nid, 'reason': 'r', 'situation': 'New'},
            {'node_id': 'nonexistent_id_zzz',
             'reason': 'r', 'situation': 'X'},
        ])
        self.assertEqual(result['revised'], 1)
        rows = result['results']
        self.assertEqual(rows[0]['status'], 'revised')
        self.assertIn('deltas', rows[0])
        self.assertEqual(rows[1]['status'], 'error')

    def test_warnings_per_row(self):
        """Per-row warnings carry the immutable-skip messages."""
        n1 = _make_node(self.brain, situation='S1')
        result = self.brain.revise_batch([
            {'node_id': n1, 'reason': 'r', 'situation': 'X', 'locked': True},
        ])
        rows = result['results']
        self.assertEqual(rows[0]['status'], 'revised')
        self.assertTrue(any('locked' in w for w in rows[0].get('warnings', [])))


# ═══════════════════════════════════════════════════════════════════════
# Class G — Edge cases + regressions
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases(BrainTestBase):
    needs_embedder = False

    def test_empty_updates_returns_error(self):
        """revise with no fields to update returns error."""
        nid = _make_node(self.brain)
        result = self.brain.revise(node_id=nid, reason='r')
        self.assertIn('error', result)
        self.assertIn('No updates', result['error'])

    def test_nonexistent_node_returns_error(self):
        """revise on unknown node_id returns 'Node not found' error."""
        result = self.brain.revise(node_id='nonexistent_xxx',
                                   reason='r', updates={'situation': 'X'})
        self.assertIn('error', result)
        self.assertIn('not found', result['error'].lower())

    def test_archived_node_returns_error(self):
        """revise on archived node returns 'Cannot revise archived node' error."""
        nid = _make_node(self.brain)
        self.brain.conn.execute(
            "UPDATE nodes SET archived = 1 WHERE id = ?", (nid,))
        self.brain.conn.commit()
        result = self.brain.revise(node_id=nid, reason='r',
                                   updates={'situation': 'X'})
        self.assertIn('error', result)
        self.assertIn('archived', result['error'].lower())

    def test_long_content_replace(self):
        """Large content (10KB) replaces cleanly."""
        nid = _make_node(self.brain, content='small')
        big = 'x' * 10000
        self.brain.revise(node_id=nid, content=big, reason='r')
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], big)
        self.assertEqual(len(row[0]), 10000)

    def test_unicode_in_fields(self):
        """Unicode characters in title/situation/content survive revise."""
        nid = _make_node(self.brain, title='Original')
        unicode_str = 'Anchor — 持続 — émergent — 🧠'
        self.brain.revise(node_id=nid, title=unicode_str,
                          situation=unicode_str, reason='unicode')
        row = self.brain.conn.execute(
            "SELECT title FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], unicode_str)
        self.assertEqual(_kv_value(self.brain, nid, 'situation'), unicode_str)

    def test_reason_field_is_not_a_node_field(self):
        """`reason` is required for audit but does not become node metadata."""
        nid = _make_node(self.brain)
        self.brain.revise(node_id=nid, situation='X', reason='audit reason')
        # `reason` should never appear in node_metadata_kv
        kv_keys = _kv_keys(self.brain, nid)
        self.assertNotIn('reason', kv_keys)

    def test_fields_updated_excludes_skipped(self):
        """fields_updated lists only fields actually written."""
        nid = _make_node(self.brain, confidence=0.5)
        result = self.brain.revise(
            node_id=nid, reason='r',
            updates={'id': 'x', 'locked': True, 'confidence': 0.9})
        # fields_updated should NOT include skipped immutables
        self.assertNotIn('id', result.get('fields_updated', []))
        self.assertNotIn('locked', result.get('fields_updated', []))
        self.assertIn('confidence', result.get('fields_updated', []))

    def test_emergent_field_writes_to_kv(self):
        """Unknown field name (emergent) writes to KV without error."""
        nid = _make_node(self.brain)
        self.brain.revise(node_id=nid, reason='r',
                          updates={'my_emergent_field': 'value'})
        self.assertEqual(_kv_value(self.brain, nid, 'my_emergent_field'), 'value')

    def test_dispatch_keys_not_treated_as_fields(self):
        """encoding_source / chain_id / session_id passed to dispatch are
        recognized as dispatch-level args, not node fields."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain)
        _handle_revise(self.brain, {
            'node_id': nid, 'reason': 'r',
            'situation': 'X',
            'encoding_source': 's2:test',
            'chain_id': 's2-test-chain',
            'session_id': 'sess-xyz',
        }, [])

        kv_keys = _kv_keys(self.brain, nid)
        # situation should land
        self.assertIn('situation', kv_keys)
        # but the dispatch keys should NOT leak into KV
        self.assertNotIn('encoding_source', kv_keys)
        self.assertNotIn('chain_id', kv_keys)
        self.assertNotIn('session_id', kv_keys)

    def test_legacy_sys_revision_history_does_not_block_revise(self):
        """A node carrying legacy _sys_revision_history (pre-Stage 1A) can
        still be revised. The legacy data is NOT touched by the new code."""
        nid = _make_node(self.brain)
        legacy_history = json.dumps([
            {'timestamp': '2024-01-01T00:00:00Z', 'reason': 'old',
             'old_content': 'pre-stage1a content'},
        ])
        self.brain.conn.execute(
            "INSERT INTO node_metadata_kv (node_id, key, value) "
            "VALUES (?, '_sys_revision_history', ?)", (nid, legacy_history))
        self.brain.conn.commit()

        # Revise should succeed
        result = self.brain.revise(node_id=nid, situation='New', reason='r')
        self.assertNotIn('error', result)

        # Legacy history blob still exists (the migration script removes it,
        # not the revise() path). This documents the expected post-B.1 state.
        legacy_after = _kv_value(self.brain, nid, '_sys_revision_history')
        self.assertIsNotNone(legacy_after,
                             "legacy _sys_revision_history should remain "
                             "until migration script runs")


class TestReviseEdge(BrainTestBase):
    """revise_edge: in-place edge-relation revise (rename + desc/weight)."""
    needs_embedder = False

    def _relations(self, src, tgt):
        eid = self.brain._graph.get_edge_id(src, tgt)
        return {r['relation']: r for r in self.brain._graph.get_relations(eid)}

    def test_rename_relation_in_place_preserves_desc_and_weight(self):
        """new_relation renames in place — description + weight carry over, the
        old relation is gone from the active set (not a delete+recreate)."""
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(a, b, relation='related',
                                 description='both about X', weight=0.7)
        res = self.brain.revise_edge(a, b, relation='related', new_relation='complements')
        self.assertTrue(res['ok'], res)
        rels = self._relations(a, b)
        self.assertIn('complements', rels)
        self.assertNotIn('related', rels)               # renamed, not duplicated
        self.assertEqual(rels['complements']['description'], 'both about X')  # preserved
        self.assertEqual(rels['complements']['weight'], 0.7)                  # preserved

    def test_update_description_without_rename(self):
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(a, b, relation='grounds', description='old', weight=0.5)
        res = self.brain.revise_edge(a, b, relation='grounds', description='new why')
        self.assertTrue(res['ok'], res)
        self.assertEqual(self._relations(a, b)['grounds']['description'], 'new why')

    def test_loud_on_missing_edge(self):
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        res = self.brain.revise_edge(a, b, relation='related', new_relation='x')
        self.assertFalse(res['ok'])
        self.assertIn('no edge', res['error'])

    def test_loud_on_missing_relation(self):
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(a, b, relation='grounds', description='d', weight=0.5)
        res = self.brain.revise_edge(a, b, relation='related', new_relation='x')
        self.assertFalse(res['ok'])
        self.assertIn('no active relation', res['error'])

    def test_rename_collision_is_loud(self):
        """Renaming to a relation the edge already has is rejected, not merged."""
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(a, b, relation='related', description='d1', weight=0.5)
        self.brain.connect_typed(a, b, relation='grounds', description='d2', weight=0.6)
        res = self.brain.revise_edge(a, b, relation='related', new_relation='grounds')
        self.assertFalse(res['ok'])
        self.assertIn('collide', res['error'])


if __name__ == '__main__':
    unittest.main()
