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
                         their_raw_quote='U', my_raw_quote='A')
        self.brain.revise(node_id=nid, situation='S2', reason='r')

        # Other KV keys still present and unchanged
        self.assertEqual(_kv_value(self.brain, nid, 'reasoning'), 'R')
        self.assertEqual(_kv_value(self.brain, nid, 'their_raw_quote'), 'U')
        self.assertEqual(_kv_value(self.brain, nid, 'my_raw_quote'), 'A')


# ═══════════════════════════════════════════════════════════════════════
# Class B — Immutable handling (skip + warn, never fail the call)
# ═══════════════════════════════════════════════════════════════════════

class TestRevisedAtClaimGate(BrainTestBase):
    """revised_at bumps ONLY on claim changes (content/title).

    The claim is what the consolidation clustering embeddings are built
    from; metadata enrichment must not re-enter a node into the change set
    (B1 2026-08-11: healer question stamps were resetting suppression on
    ~24 claim-unchanged nodes/day). updated_at always bumps.
    """

    needs_embedder = False

    def _stamps(self, nid):
        return self.brain.conn.execute(
            "SELECT revised_at, updated_at FROM nodes WHERE id = ?",
            (nid,)).fetchone()

    def test_metadata_only_revise_does_not_bump_revised_at(self):
        nid = _make_node(self.brain, title='T', content='C')
        before_revised, _ = self._stamps(nid)
        time.sleep(0.01)
        result = self.brain.revise(node_id=nid, question='what changed?',
                                   confidence=0.8, reason='enrichment stamp')
        self.assertNotIn('error', result)
        after_revised, after_updated = self._stamps(nid)
        self.assertEqual(after_revised, before_revised)
        self.assertIsNone(result['revised_at'])
        self.assertGreater(after_updated, before_revised or '')

    def test_content_change_bumps_revised_at(self):
        nid = _make_node(self.brain, title='T', content='C')
        before_revised, _ = self._stamps(nid)
        time.sleep(0.01)
        result = self.brain.revise(node_id=nid, content='C2', reason='claim')
        after_revised, _ = self._stamps(nid)
        self.assertNotEqual(after_revised, before_revised)
        self.assertEqual(result['revised_at'], after_revised)

    def test_title_change_bumps_revised_at(self):
        nid = _make_node(self.brain, title='T', content='C')
        before_revised, _ = self._stamps(nid)
        time.sleep(0.01)
        self.brain.revise(node_id=nid, title='T2', reason='claim')
        after_revised, _ = self._stamps(nid)
        self.assertNotEqual(after_revised, before_revised)

    def test_identical_content_does_not_bump_revised_at(self):
        # Passing content equal to the stored value is not a claim change.
        nid = _make_node(self.brain, title='T', content='C')
        before_revised, _ = self._stamps(nid)
        time.sleep(0.01)
        result = self.brain.revise(node_id=nid, content='C', reason='no-op')
        after_revised, _ = self._stamps(nid)
        self.assertEqual(after_revised, before_revised)
        self.assertIsNone(result['revised_at'])


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
        from servers.daemon_dispatch import dispatch_command

        nid = _make_node(self.brain)
        self.brain.conn.execute(
            "UPDATE nodes SET locked = 1 WHERE id = ?", (nid,))
        self.brain.conn.commit()

        # Emission happens at the dispatch chokepoint (mutation emitter),
        # so the test must enter through dispatch_command, not the handler.
        graph_changes = []
        dispatch_command(self.brain, 'revise', {
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
#
# Step 4 of the mutation-emitter migration: handlers return a `mutations`
# manifest and dispatch_command emits it — the handler alone writes NO trace.
# These tests therefore enter through dispatch_command (the one real door),
# pinning the same invariants the legacy inline emit satisfied.
# ═══════════════════════════════════════════════════════════════════════

class TestTraceEvents(BrainTestBase):
    needs_embedder = False

    def _dispatch(self, cmd, args):
        from servers.daemon_dispatch import dispatch_command
        return dispatch_command(self.brain, cmd, args, [])

    def test_single_revise_emits_one_trace(self):
        """Dispatch revise → exactly one node_revised trace event."""
        nid = _make_node(self.brain, situation='Original')
        self._dispatch('revise', {
            'node_id': nid, 'reason': 'r', 'situation': 'New',
        })

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]['ref_type'], 'node_revised')
        self.assertEqual(traces[0]['event_type'], 'delta')
        # The human summary line must match the legacy emit format exactly.
        self.assertEqual(traces[0]['summary'], 'revised 1 field(s): situation')

    def test_revise_result_carries_no_manifest(self):
        """`mutations` is dispatch plumbing — it must never reach the caller's
        tool result (the 6M-char class)."""
        nid = _make_node(self.brain, situation='Original')
        result = self._dispatch('revise', {
            'node_id': nid, 'reason': 'r', 'situation': 'New',
        })
        self.assertNotIn('mutations', result)

    def test_revise_batch_emits_one_trace_per_row(self):
        """Dispatch revise_batch with 3 nodes → 3 trace events."""
        n1 = _make_node(self.brain, situation='S1')
        n2 = _make_node(self.brain, situation='S2')
        n3 = _make_node(self.brain, situation='S3')

        self._dispatch('revise_batch', {
            'revisions': [
                {'node_id': n1, 'reason': 'r1', 'situation': 'NS1'},
                {'node_id': n2, 'reason': 'r2', 'situation': 'NS2'},
                {'node_id': n3, 'reason': 'r3', 'situation': 'NS3'},
            ]
        })

        for nid in (n1, n2, n3):
            traces = _query_revise_traces(self.brain, nid)
            self.assertEqual(len(traces), 1,
                             "node %s had %d traces, expected 1" % (
                                 nid[:8], len(traces)))

    def test_trace_metadata_shape(self):
        """Trace metadata matches REVISE_METADATA_SHAPE."""
        nid = _make_node(self.brain, situation='Original')
        self._dispatch('revise', {
            'node_id': nid, 'reason': 'shape test', 'situation': 'New',
            'encoding_source': 'test:shape',
        })

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
        nid = _make_node(self.brain, situation='X')
        self._dispatch('revise', {
            'node_id': nid, 'reason': 'r', 'situation': 'Y',
            'chain_id': 's2-20260504-aspect_integration',
        })

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(traces[0]['chain_id'], 's2-20260504-aspect_integration')

    def test_chain_id_default_is_date_based(self):
        """No chain_id arg → date-based fallback chain (`{scale}-{YYYYMMDD}-revise`)."""
        nid = _make_node(self.brain, situation='X')
        self._dispatch('revise', {
            'node_id': nid, 'reason': 'r', 'situation': 'Y',
        })

        traces = _query_revise_traces(self.brain, nid)
        chain = traces[0]['chain_id']
        # Format: s0-YYYYMMDD-revise (no encoding_source → s0)
        self.assertTrue(chain.startswith('s0-'),
                        "expected chain to start with 's0-', got: %s" % chain)
        self.assertTrue(chain.endswith('-revise'),
                        "expected chain to end with '-revise', got: %s" % chain)

    def test_scale_inferred_from_encoding_source(self):
        """encoding_source='s2:foo' → trace.scale='s2'."""
        nid = _make_node(self.brain, situation='X')
        self._dispatch('revise', {
            'node_id': nid, 'reason': 'r', 'situation': 'Y',
            'encoding_source': 's2:healer',
        })

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(traces[0]['scale'], 's2')

    def test_no_trace_when_no_changes_no_warnings(self):
        """revise(field=<same value>) with no warnings → no trace emitted."""
        nid = _make_node(self.brain, confidence=0.5)
        self._dispatch('revise', {
            'node_id': nid, 'reason': 'noop', 'confidence': 0.5,
        })

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(traces, [],
                         "expected no traces for no-op revise, got: %r" % traces)

    def test_rolled_back_batch_emits_zero_traces(self):
        """A brain_batch whose transaction rolls back must produce ZERO
        node_revised traces — the property the legacy inline emits could not
        guarantee (they fired mid-envelope). The manifest only reaches the
        emitter if the handler returns, and the handler re-raises on
        rollback, so this is structural."""
        from servers.daemon_dispatch import dispatch_command

        nid = _make_node(self.brain, situation='Original')

        real_commit = self.brain.conn.commit
        calls = {'n': 0}

        def failing_commit():
            # The batch's own final commit is the one that must fail; per-op
            # commits are no-ops while in_batch is set.
            calls['n'] += 1
            raise RuntimeError('injected commit failure')

        self.brain.conn.commit = failing_commit
        try:
            with self.assertRaises(RuntimeError):
                dispatch_command(self.brain, 'brain_batch', {
                    'operations': [{'op': 'revise', 'node_id': nid,
                                    'reason': 'r', 'situation': 'New'}],
                }, [])
        finally:
            self.brain.conn.commit = real_commit
            try:
                self.brain.conn.rollback()
            except Exception:
                pass

        self.assertTrue(calls['n'] >= 1, 'injected commit never reached')
        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(traces, [],
                         'rolled-back batch must leave zero mutation traces, '
                         'got: %r' % traces)

    def test_batch_revise_trace_lands_post_commit(self):
        """revise inside brain_batch → one node_revised trace, emitted at the
        chokepoint after the batch commit, and the manifest never appears in
        the per-op results the agent sees."""
        from servers.daemon_dispatch import dispatch_command

        nid = _make_node(self.brain, situation='Original')
        result = dispatch_command(self.brain, 'brain_batch', {
            'operations': [{'op': 'revise', 'node_id': nid,
                            'reason': 'r', 'situation': 'New'}],
        }, [])

        self.assertTrue(result.get('ok'))
        self.assertNotIn('mutations', result)
        for op_row in result['result']['results']:
            self.assertNotIn('mutations', op_row)

        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(len(traces), 1)


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

    def test_rename_preserves_encoding_source_when_omitted(self):
        """A rename WITHOUT encoding_source preserves the row's provenance — it
        must not clobber it to 'anchor'. description + weight also survive."""
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(a, b, relation='related', description='d',
                                 weight=0.7, encoding_source='encoder:sonnet')
        res = self.brain.revise_edge(a, b, relation='related', new_relation='complements')
        self.assertTrue(res['ok'], res)
        row = self._relations(a, b)['complements']
        self.assertEqual(row['encoding_source'], 'encoder:sonnet')  # preserved, not 'anchor'
        self.assertEqual(row['description'], 'd')                   # preserved
        self.assertEqual(row['weight'], 0.7)                        # preserved

    def test_explicit_encoding_source_overrides_on_rename(self):
        """When the caller DOES pass encoding_source, the rename records it."""
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(a, b, relation='related', description='d',
                                 weight=0.5, encoding_source='encoder:sonnet')
        res = self.brain.revise_edge(a, b, relation='related', new_relation='complements',
                                     encoding_source='s2:reclassify')
        self.assertTrue(res['ok'], res)
        self.assertEqual(self._relations(a, b)['complements']['encoding_source'],
                         's2:reclassify')

    def test_rename_to_archived_relation_is_loud(self):
        """Collision check spans ARCHIVED rows — renaming onto an archived
        relation name is rejected, not a PK-violating crash (PK is
        (edge_id, relation), shared by active + archived rows)."""
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(a, b, relation='foo', description='d1', weight=0.5)
        self.brain._graph.remove_relation(a, b, 'foo', archived_by='test')  # soft-archive
        self.brain.connect_typed(a, b, relation='related', description='d2', weight=0.6)
        res = self.brain.revise_edge(a, b, relation='related', new_relation='foo')
        self.assertFalse(res['ok'])
        self.assertIn('collide', res['error'])


# ═══════════════════════════════════════════════════════════════════════
# Class H — reason vs reasoning disambiguation (2026-06-12)
# ═══════════════════════════════════════════════════════════════════════
# `reason` = audit note for a revision (trace event, never stored on node).
# `reasoning` = PROMOTED node field (why this was encoded, node_metadata_kv).
# Near-identical names; agents confuse them. The contract: revise without
# `reason` fails LOUD with a disambiguating error (naming `reasoning` when
# present); `reasoning` is never aliased/rerouted — it stays a field update.
# remember with a stray `reason` keeps drop semantics but surfaces a warning.

class TestReasonReasoningDisambiguation(BrainTestBase):
    needs_embedder = False

    def test_revise_missing_reason_with_reasoning_disambiguates(self):
        """The observed live failure: revise op with `reasoning` but no
        `reason` must error AND explain the field-vs-audit distinction."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain)
        r = _handle_revise(self.brain, {
            'node_id': nid,
            'reasoning': 'meant as audit note',
            'content': 'new content',
        }, [])
        self.assertFalse(r['ok'])
        self.assertIn('reason is required', r['error'])
        self.assertIn('reasoning', r['error'])
        self.assertIn('FIELD', r['error'])
        # The node must be untouched — no partial write on validation error
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 'Initial content')
        self.assertIsNone(_kv_value(self.brain, nid, 'reasoning'))

    def test_revise_missing_reason_without_reasoning_plain_error(self):
        """No `reasoning` present → plain (but self-explanatory) error,
        no mention of the reasoning field."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain)
        r = _handle_revise(self.brain, {'node_id': nid, 'content': 'x'}, [])
        self.assertFalse(r['ok'])
        self.assertIn('reason is required', r['error'])
        self.assertIn('audit note', r['error'])
        self.assertNotIn('You passed', r['error'])

    def test_revise_with_both_updates_reasoning_field(self):
        """Regression guard against aliasing: `reasoning` alongside `reason`
        is a normal field update — stored on the node, NOT rerouted."""
        from servers.daemon_dispatch import _handle_revise

        nid = _make_node(self.brain)
        r = _handle_revise(self.brain, {
            'node_id': nid,
            'reason': 'audit note',
            'reasoning': 'updated rationale',
        }, [])
        self.assertTrue(r['ok'], r)
        self.assertEqual(_kv_value(self.brain, nid, 'reasoning'),
                         'updated rationale')
        # `reason` itself must never land in node metadata
        self.assertNotIn('reason', _kv_keys(self.brain, nid))

    def test_revise_batch_spec_missing_reason_disambiguates(self):
        """Per-spec validation carries the same disambiguation, indexed."""
        from servers.daemon_dispatch import _handle_revise_batch

        nid = _make_node(self.brain)
        r = _handle_revise_batch(self.brain, {'revisions': [
            {'node_id': nid, 'reasoning': 'meant as audit', 'content': 'x'},
        ]}, [])
        self.assertFalse(r['ok'])
        self.assertIn('revisions[0]', r['error'])
        self.assertIn('reason is required', r['error'])
        self.assertIn('reasoning', r['error'])

    def test_brain_batch_revise_op_inherits_disambiguation(self):
        """brain_batch revise ops route through _handle_revise — the per-op
        error must carry the same disambiguation."""
        from servers.daemon_dispatch import _handle_brain_batch

        nid = _make_node(self.brain)
        r = _handle_brain_batch(self.brain, {'operations': [
            {'op': 'revise', 'node_id': nid,
             'reasoning': 'meant as audit', 'content': 'x'},
        ]}, [])
        op_result = r['result']['results'][0]
        self.assertFalse(op_result['ok'])
        self.assertIn('reason is required', op_result['error'])
        self.assertIn('reasoning', op_result['error'])

    def test_remember_stray_reason_warns_and_drops(self):
        """Mirror confusion: remember with `reason` (no `reasoning`) keeps the
        drop semantics but surfaces a warning naming `reasoning`."""
        from servers.daemon_dispatch import _handle_remember

        r = _handle_remember(self.brain, {
            'type': 'concept', 'title': 'Stray reason node',
            'content': 'c', 'reason': 'meant as reasoning',
        }, [])
        self.assertTrue(r['ok'], r)
        result = r['result']
        warnings = result.get('warnings', [])
        self.assertTrue(any('reasoning' in w for w in warnings),
                        'expected reason-drop warning, got: %r' % warnings)
        nid = result['id']
        self.assertNotIn('reason', _kv_keys(self.brain, nid))   # still dropped
        self.assertIsNone(_kv_value(self.brain, nid, 'reasoning'))  # not aliased

    def test_remember_with_reasoning_no_warning(self):
        """The legitimate path stays quiet: `reasoning` stores, no warning."""
        from servers.daemon_dispatch import _handle_remember

        r = _handle_remember(self.brain, {
            'type': 'concept', 'title': 'Proper reasoning node',
            'content': 'c', 'reasoning': 'the rationale',
        }, [])
        self.assertTrue(r['ok'], r)
        result = r['result']
        self.assertEqual(result.get('warnings'), None)
        self.assertEqual(_kv_value(self.brain, result['id'], 'reasoning'),
                         'the rationale')

    def test_remember_batch_stray_reason_warns_per_spec(self):
        """remember_batch surfaces an indexed warning per offending spec."""
        from servers.daemon_dispatch import _handle_remember_batch

        r = _handle_remember_batch(self.brain, {'nodes': [
            {'type': 'concept', 'title': 'Clean node b0', 'content': 'c',
             'reasoning': 'fine'},
            {'type': 'concept', 'title': 'Stray node b1', 'content': 'c',
             'reason': 'meant as reasoning'},
        ]}, [])
        self.assertTrue(r['ok'], r)
        warnings = r['result'].get('warnings', [])
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn('nodes[1]', warnings[0])
        self.assertIn('reasoning', warnings[0])


OLD_LINE = 'Committed as 69ba06b, awaiting review before merge.'
BODY = ('The guard sits in build-plugin.sh.\n'
        + OLD_LINE + '\n'
        'Three paths: new version records, unchanged no-op, changed aborts.')


class TestContentEdits(BrainTestBase):
    """Patch-mode content: content_edits compiles to a content replace inside
    revise() against the stored content — exact, unique, in-order matches;
    loud errors; mutually exclusive with content."""
    needs_embedder = False

    def test_patch_applies_and_preserves_rest(self):
        # Also covers the patch-alone shape: content_edits with no other
        # field update passes the empty-updates check.
        nid = _make_node(self.brain, content=BODY,
                         situation='When picking up the ratchet')
        result = self.brain.revise(
            node_id=nid, reason='branch deleted',
            content_edits=[{'old': OLD_LINE,
                            'new': 'NEVER MERGED — branch deleted 2026-08-17.'}])

        self.assertNotIn('error', result)
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertIn('NEVER MERGED — branch deleted 2026-08-17.', row[0])
        self.assertNotIn(OLD_LINE, row[0])
        # Untouched lines survive verbatim; other fields preserved
        self.assertIn('The guard sits in build-plugin.sh.', row[0])
        self.assertIn('Three paths: new version records', row[0])
        self.assertEqual(_kv_value(self.brain, nid, 'situation'),
                         'When picking up the ratchet')
        # Deltas carry the content change like any ordinary revise
        self.assertTrue(any(d.get('field') == 'content'
                            for d in result.get('deltas', [])))

    def test_patch_via_updates_dict(self):
        nid = _make_node(self.brain, content=BODY)
        result = self.brain.revise(
            node_id=nid, reason='r',
            updates={'content_edits': [{'old': 'awaiting review',
                                        'new': 'dead'}]})
        self.assertNotIn('error', result)
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertIn('Committed as 69ba06b, dead before merge.', row[0])

    def test_patch_edits_apply_in_order(self):
        nid = _make_node(self.brain, content='status: open. next: review.')
        result = self.brain.revise(
            node_id=nid, reason='r',
            content_edits=[
                {'old': 'status: open.', 'new': 'status: closed.'},
                # Matches text produced by the previous edit — order is real.
                {'old': 'closed. next: review.', 'new': 'closed.'},
            ])
        self.assertNotIn('error', result)
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 'status: closed.')

    def test_no_match_errors_and_node_unchanged(self):
        nid = _make_node(self.brain, content=BODY)
        result = self.brain.revise(
            node_id=nid, reason='r',
            content_edits=[{'old': 'text that is not there', 'new': 'x'}])
        self.assertIn('not found', result['error'])
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], BODY)

    def test_ambiguous_match_errors(self):
        nid = _make_node(self.brain, content='alpha beta alpha')
        result = self.brain.revise(
            node_id=nid, reason='r',
            content_edits=[{'old': 'alpha', 'new': 'gamma'}])
        self.assertIn('matches 2 places', result['error'])

    def test_mutually_exclusive_with_content(self):
        nid = _make_node(self.brain, content=BODY)
        result = self.brain.revise(
            node_id=nid, reason='r', content='full rewrite',
            content_edits=[{'old': OLD_LINE, 'new': 'x'}])
        self.assertIn('mutually exclusive', result['error'])
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], BODY)

    def test_empty_content_also_mutually_exclusive(self):
        """content='' + content_edits is two competing intents — the falsy
        named-arg path must not slip past the guard."""
        nid = _make_node(self.brain, content=BODY)
        result = self.brain.revise(node_id=nid, reason='r', content='',
                                   content_edits=[{'old': '69ba06b',
                                                   'new': 'x'}])
        self.assertIn('mutually exclusive', result['error'])
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], BODY)

    def test_remember_swallows_content_edits_loudly(self):
        """content_edits on a remember is misrouted patching — never stored
        as junk KV, and logged to the errors table."""
        result = self.brain.remember(
            type='concept', title='patch misroute canary', content='C',
            content_edits=[{'old': 'a', 'new': 'b'}])
        nid = result['id']
        self.assertIsNone(_kv_value(self.brain, nid, 'content_edits'))
        err = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type = 'error' "
            "AND source = ?",
            ('remember_content_edits_misrouted',)).fetchone()
        self.assertGreaterEqual(err[0], 1)

    def test_absorb_rejects_content_edits(self):
        """A patch cannot fold the absorbed node in — absorb refuses loudly
        instead of silently patching the survivor."""
        from servers.daemon_dispatch import _handle_brain_batch
        a = _make_node(self.brain, content='survivor body')
        b = _make_node(self.brain, content='absorbed body')
        r = _handle_brain_batch(self.brain, {'operations': [
            {'op': 'absorb', 'survivor_id': a, 'absorbed_id': b,
             'content_edits': [{'old': 'survivor', 'new': 'patched'}]}]}, [])
        op = r['result']['results'][0]
        self.assertFalse(op['ok'])
        self.assertIn('not supported on absorb', op['error'])
        row = self.brain.conn.execute(
            "SELECT content, archived FROM nodes WHERE id = ?",
            (b,)).fetchone()
        self.assertEqual(row[1], 0)  # absorbed node untouched

    def test_revise_batch_rows_carry_ok(self):
        """Per-row ok rides beside status so log_failed_batch_ops sees
        failures on the encoder's primary revise surface."""
        nid = _make_node(self.brain, content=BODY)
        r = self.brain.revise_batch(revisions=[
            {'node_id': nid, 'reason': 'r',
             'content_edits': [{'old': '69ba06b', 'new': 'x'}]},
            {'node_id': nid, 'reason': 'r',
             'content_edits': [{'old': 'no such text', 'new': 'y'}]}])
        self.assertIs(r['results'][0]['ok'], True)
        self.assertIs(r['results'][1]['ok'], False)
        # index/op make the batch_op_failed log line attributable
        self.assertEqual(r['results'][1]['index'], 1)
        self.assertEqual(r['results'][1]['op'], 'revise')

    def test_stringified_content_edits_unwrapped(self):
        """A caller whose schema predates the field emits the array as a
        JSON string — recovered losslessly, logged loudly (same tolerance
        brain_batch already gives stringified operations)."""
        import json as _json
        nid = _make_node(self.brain, content=BODY)
        result = self.brain.revise(
            node_id=nid, reason='r',
            content_edits=_json.dumps([{'old': '69ba06b', 'new': 'e7b36a9'}]))
        self.assertNotIn('error', result)
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertIn('e7b36a9', row[0])
        logged = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type = 'error' "
            "AND source = ?", ('revise_content_edits_stringified',)).fetchone()
        self.assertGreaterEqual(logged[0], 1)

    def test_bad_shapes_error(self):
        nid = _make_node(self.brain, content=BODY)
        for bad in ([], 'patch', [{'old': '', 'new': 'x'}],
                    [{'old': 'a'}], [{'old': 'a', 'new': 'a'}]):
            result = self.brain.revise(node_id=nid, reason='r',
                                       content_edits=bad)
            self.assertIn('error', result, 'shape %r must fail' % (bad,))
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], BODY)

    def test_brain_batch_revise_op_carries_content_edits(self):
        """The batch path: a revise op with content_edits patches; a failing
        op stays loud in its per-op result while others proceed."""
        from servers.daemon_dispatch import _handle_brain_batch
        nid = _make_node(self.brain, content=BODY)
        r = _handle_brain_batch(self.brain, {'operations': [
            {'op': 'revise', 'node_id': nid, 'reason': 'r',
             'content_edits': [{'old': '69ba06b', 'new': 'e7b36a9'}]},
            {'op': 'revise', 'node_id': nid, 'reason': 'r',
             'content_edits': [{'old': 'nonexistent claim', 'new': 'x'}]},
        ]}, [])
        results = r['result']['results']
        self.assertTrue(results[0]['ok'], results[0])
        self.assertFalse(results[1]['ok'])
        self.assertIn('not found', results[1]['error'])
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertIn('e7b36a9', row[0])

    def test_revise_batch_carries_content_edits(self):
        nid = _make_node(self.brain, content=BODY)
        r = self.brain.revise_batch(revisions=[
            {'node_id': nid, 'reason': 'r',
             'content_edits': [{'old': 'awaiting review', 'new': 'dead'}]}])
        self.assertEqual(r['revised'], 1)
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertIn('dead before merge', row[0])

    def test_schema_exposes_content_edits(self):
        """MCP surfaces: the op spec, the revise tool, and revise_batch items
        all carry content_edits (contract → schema derivation)."""
        from servers.contract import BATCH_OP_SPECS
        self.assertIn('content_edits', BATCH_OP_SPECS['revise']['properties'])
        from servers import brain_mcp
        revise_tool = next(t for t in brain_mcp.TOOLS if t['name'] == 'revise')
        self.assertIn('content_edits',
                      revise_tool['inputSchema']['properties'])
        batch_tool = next(t for t in brain_mcp.TOOLS
                          if t['name'] == 'revise_batch')
        items = batch_tool['inputSchema']['properties']['revisions']['items']
        self.assertIn('content_edits', items['properties'])


# ═══════════════════════════════════════════════════════════════════════
# Class H — value or swap on every text field (contract.REVISE_RULE)
# ═══════════════════════════════════════════════════════════════════════

def _title(brain, nid):
    return brain.conn.execute(
        "SELECT title FROM nodes WHERE id = ?", (nid,)).fetchone()[0]


def _content(brain, nid):
    return brain.conn.execute(
        "SELECT content FROM nodes WHERE id = ?", (nid,)).fetchone()[0]


class TestValueOrSwap(BrainTestBase):
    """A field takes its new value or {old, new} swaps — title, situation,
    open KV keys, content; exactly-once or loud; nothing written on error;
    bare_only fields refuse swaps."""
    needs_embedder = False

    def test_title_swap(self):
        nid = _make_node(self.brain, title='brain/9.6.0 — manifests still say 9.6.0',
                         content='Body stays.')
        r = self.brain.revise(node_id=nid, reason='bumped',
                              title={'old': 'brain/9.6.0', 'new': 'brain/9.7.2'})
        self.assertNotIn('error', r, r)
        self.assertEqual(_title(self.brain, nid),
                         'brain/9.7.2 — manifests still say 9.6.0')
        self.assertEqual(_content(self.brain, nid), 'Body stays.')
        self.assertIn('title', r['fields_updated'])
        d = [x for x in r['deltas'] if x['field'] == 'title'][0]
        self.assertEqual(d['old'], 'brain/9.6.0 — manifests still say 9.6.0')
        self.assertEqual(d['new'], 'brain/9.7.2 — manifests still say 9.6.0')

    def test_situation_swap_hits_kv(self):
        nid = _make_node(self.brain, situation='When picking up Phase 5 — version is 9.6.0')
        r = self.brain.revise(node_id=nid, reason='r',
                              situation={'old': '9.6.0', 'new': '9.7.2'})
        self.assertNotIn('error', r, r)
        self.assertEqual(_kv_value(self.brain, nid, 'situation'),
                         'When picking up Phase 5 — version is 9.7.2')

    def test_open_kv_key_swap(self):
        nid = _make_node(self.brain, note='alpha then beta')
        r = self.brain.revise(node_id=nid, reason='r',
                              note=[{'old': 'alpha', 'new': 'gamma'}])
        self.assertNotIn('error', r, r)
        self.assertEqual(_kv_value(self.brain, nid, 'note'), 'gamma then beta')

    def test_content_swap_list_applies_in_order(self):
        nid = _make_node(self.brain, content='status: open. next: review.')
        r = self.brain.revise(node_id=nid, reason='r', content=[
            {'old': 'status: open.', 'new': 'status: closed.'},
            {'old': 'closed. next: review.', 'new': 'closed.'}])
        self.assertNotIn('error', r, r)
        self.assertEqual(_content(self.brain, nid), 'status: closed.')

    def test_bare_value_still_replaces_whole_field(self):
        nid = _make_node(self.brain, title='old title')
        r = self.brain.revise(node_id=nid, reason='r', title='new title')
        self.assertNotIn('error', r, r)
        self.assertEqual(_title(self.brain, nid), 'new title')

    def test_no_match_is_loud_and_writes_nothing(self):
        nid = _make_node(self.brain, title='T one', situation='S one')
        r = self.brain.revise(node_id=nid, reason='r',
                              title={'old': 'one', 'new': 'two'},
                              situation={'old': 'absent', 'new': 'x'})
        self.assertIn('not found', r['error'])
        self.assertIn('situation', r['error'])
        self.assertEqual(_title(self.brain, nid), 'T one')   # all-or-nothing
        self.assertEqual(_kv_value(self.brain, nid, 'situation'), 'S one')

    def test_ambiguous_match_is_loud(self):
        nid = _make_node(self.brain, content='a b a')
        r = self.brain.revise(node_id=nid, reason='r',
                              content={'old': 'a', 'new': 'c'})
        self.assertIn('matches 2 places', r['error'])
        self.assertEqual(_content(self.brain, nid), 'a b a')

    def test_swap_on_bare_only_field_refused(self):
        nid = _make_node(self.brain)
        r = self.brain.revise(node_id=nid, reason='r',
                              type={'old': 'concept', 'new': 'decision'})
        self.assertIn('bare value', r['error'])
        r = self.brain.revise(node_id=nid, reason='r',
                              confidence={'old': '1', 'new': '0.5'})
        self.assertIn('bare value', r['error'])

    def test_swap_on_field_without_stored_value_refused(self):
        nid = _make_node(self.brain)  # no question
        r = self.brain.revise(node_id=nid, reason='r',
                              question={'old': 'x', 'new': 'y'})
        self.assertIn('no stored value', r['error'])

    def test_content_edits_alias_and_content_swaps_conflict(self):
        nid = _make_node(self.brain, content='a b')
        r = self.brain.revise(node_id=nid, reason='r',
                              content=[{'old': 'a', 'new': 'c'}],
                              content_edits=[{'old': 'b', 'new': 'd'}])
        self.assertIn('mutually exclusive', r['error'])
        self.assertEqual(_content(self.brain, nid), 'a b')

    def test_malformed_swap_on_open_kv_key_refused(self):
        """A dict that is TRYING to be a swap (extra or missing key) on an open
        KV key is refused, not JSON-written over the stored text."""
        nid = _make_node(self.brain, note='alpha beta')
        r = self.brain.revise(node_id=nid, reason='r',
                              note={'old': 'alpha', 'new': 'x', 'extra': 1})
        self.assertIn('error', r)
        self.assertEqual(_kv_value(self.brain, nid, 'note'), 'alpha beta')
        r = self.brain.revise(node_id=nid, reason='r',
                              note=[{'old': 'alpha', 'new': 'x'}, {'old': 'beta'}])
        self.assertIn('error', r)
        self.assertEqual(_kv_value(self.brain, nid, 'note'), 'alpha beta')

    def test_open_kv_list_that_is_not_a_swap_still_stores(self):
        nid = _make_node(self.brain)
        r = self.brain.revise(node_id=nid, reason='r', tags=['a', 'b'])
        self.assertNotIn('error', r, r)
        self.assertIn('tags', _kv_keys(self.brain, nid))


class TestSwapDispatch(BrainTestBase):
    """The dispatch validator knows the swap shape: a swap on a text field
    passes to the brain, a swap on a bare_only field or a malformed swap is
    refused before any write."""
    needs_embedder = False

    def _dispatch(self, cmd, args):
        from servers.daemon_dispatch import dispatch_command
        return dispatch_command(self.brain, cmd, args, [])

    def test_title_swap_through_dispatch(self):
        nid = _make_node(self.brain, title='v 9.6.0')
        r = self._dispatch('revise', {'node_id': nid, 'reason': 'r',
                                      'title': {'old': '9.6.0', 'new': '9.7.2'}})
        self.assertTrue(r.get('ok'), r)
        self.assertEqual(_title(self.brain, nid), 'v 9.7.2')
        traces = _query_revise_traces(self.brain, nid)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]['summary'], 'revised 1 field(s): title')

    def test_swap_on_type_refused_at_dispatch(self):
        nid = _make_node(self.brain)
        r = self._dispatch('revise', {'node_id': nid, 'reason': 'r',
                                      'type': {'old': 'concept', 'new': 'x'}})
        self.assertFalse(r.get('ok'))
        self.assertIn('bare value', r['error'])

    def test_malformed_swap_refused_at_dispatch(self):
        nid = _make_node(self.brain, title='t')
        r = self._dispatch('revise', {'node_id': nid, 'reason': 'r',
                                      'title': {'old': '', 'new': 'x'}})
        self.assertFalse(r.get('ok'))
        self.assertIn('non-empty', r['error'])
        r = self._dispatch('revise', {'node_id': nid, 'reason': 'r',
                                      'title': [{'old': 't', 'new': 't'}]})
        self.assertFalse(r.get('ok'))
        self.assertIn('identical', r['error'])

    def test_swap_on_remember_is_a_type_error(self):
        """Swaps patch a stored value — they exist only on revise. On remember
        a swap dict on a text field is refused as before, never stored."""
        r = self._dispatch('remember', {'type': 'concept', 'title': 'T',
                                        'content': 'C',
                                        'situation': {'old': 'a', 'new': 'b'}})
        self.assertFalse(r.get('ok'), r)
        self.assertIn('must be string', r['error'])


# ═══════════════════════════════════════════════════════════════════════
# Class I — connect_to on revise (edges ride inside the node's own op)
# ═══════════════════════════════════════════════════════════════════════

WHY = 'both manifests still say 9.6.0 while the release plan asks for 0.9.0'


class TestConnectToOnRevise(BrainTestBase):
    needs_embedder = False

    def _rels(self, a, b):
        eid = self.brain._graph.get_edge_id(a, b)
        return {r['relation']: r for r in self.brain._graph.get_relations(eid)}

    def _pair(self):
        a = _make_node(self.brain, title='A', content='A body')
        b = _make_node(self.brain, title='B', content='B body')
        self.brain.connect_typed(a, b, relation='gaps_in', description=WHY, weight=0.6)
        return a, b

    def test_update_why_bare(self):
        a, b = self._pair()
        r = self.brain.revise(node_id=a, reason='manifests moved',
                              connect_to=[{'target': b, 'relation': 'gaps_in',
                                           'why': 'manifests moved to 9.7.2 and still miss 0.9.0'}])
        self.assertNotIn('error', r, r)
        self.assertEqual(len(r['connect_to_result']['revised']), 1)
        self.assertEqual(self._rels(a, b)['gaps_in']['description'],
                         'manifests moved to 9.7.2 and still miss 0.9.0')

    def test_update_why_swap(self):
        a, b = self._pair()
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': b, 'relation': 'gaps_in',
                                           'why': {'old': '9.6.0', 'new': '9.7.2'}}])
        self.assertNotIn('error', r, r)
        self.assertEqual(r['connect_to_result']['failed'], [])
        self.assertIn('9.7.2', self._rels(a, b)['gaps_in']['description'])
        self.assertNotIn('9.6.0', self._rels(a, b)['gaps_in']['description'])

    def test_rename_relation_swap_preserves_description(self):
        a, b = self._pair()
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': b,
                                           'relation': {'old': 'gaps_in', 'new': 'short_of'}}])
        self.assertNotIn('error', r, r)
        rels = self._rels(a, b)
        self.assertIn('short_of', rels)
        self.assertNotIn('gaps_in', rels)
        self.assertEqual(rels['short_of']['description'], WHY)

    def test_relation_optional_when_pair_has_one(self):
        a, b = self._pair()
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': b, 'why': {'old': '9.6.0', 'new': '9.7.2'}}])
        self.assertEqual(r['connect_to_result']['failed'], [], r)
        self.assertIn('9.7.2', self._rels(a, b)['gaps_in']['description'])

    def test_relation_required_when_pair_has_several(self):
        a, b = self._pair()
        self.brain.connect_typed(a, b, relation='grounds', description=WHY, weight=0.5)
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': b, 'why': 'x' * 40}])
        f = r['connect_to_result']['failed']
        self.assertEqual(len(f), 1)
        self.assertIn('carries 2 relations', f[0]['reason'])

    def test_create_when_absent_is_outgoing(self):
        a = _make_node(self.brain, title='A')
        c = _make_node(self.brain, title='C')
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': c, 'relation': 'grounds', 'why': WHY}])
        self.assertEqual(r['connect_to_result']['failed'], [], r)
        self.assertEqual(len(r['connect_to_result']['created']), 1)
        eid = self.brain._graph.get_edge_id(a, c)
        self.assertEqual(self.brain._graph.get_edge_endpoints(eid), (a, c))
        self.assertEqual(r['warnings'], [])

    def test_create_needs_bare_why(self):
        a = _make_node(self.brain, title='A')
        c = _make_node(self.brain, title='C')
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': c, 'relation': 'grounds', 'why': 'short'}])
        self.assertIn('30+', r['connect_to_result']['failed'][0]['reason'])
        self.assertIsNone(self.brain._graph.get_edge_id(a, c))

    def test_sibling_title_rejected(self):
        a, b = self._pair()
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': 'B', 'relation': 'gaps_in', 'why': WHY}])
        self.assertIn('sibling titles', r['connect_to_result']['failed'][0]['reason'])

    def test_incoming_edge_is_found_and_revised(self):
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(b, a, relation='grounds', description=WHY, weight=0.6)
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': b, 'relation': 'grounds',
                                           'why': {'old': '9.6.0', 'new': '9.7.2'}}])
        self.assertEqual(r['connect_to_result']['failed'], [], r)
        self.assertIn('9.7.2', self._rels(a, b)['grounds']['description'])

    def test_new_relation_on_incoming_edge_rides_it_and_warns(self):
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(b, a, relation='grounds', description=WHY, weight=0.6)
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': b, 'relation': 'supersedes', 'why': WHY}])
        self.assertEqual(r['connect_to_result']['failed'], [], r)
        eid = self.brain._graph.get_edge_id(a, b)
        self.assertEqual(self.brain._graph.get_edge_endpoints(eid), (b, a))  # one edge, stored direction
        self.assertIn('supersedes', self._rels(a, b))
        ct = r['connect_to_result']
        self.assertTrue(any('passive verb' in w for w in ct['warnings']), ct)
        self.assertEqual(r['warnings'], [])  # edge warning, not a node warning

    def test_title_alias_accepted(self):
        a, b = self._pair()
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'title': b, 'relation': 'gaps_in',
                                           'why': 'x' * 40}])
        self.assertEqual(r['connect_to_result']['failed'], [], r)

    def test_connect_to_never_lands_as_node_field(self):
        a, b = self._pair()
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': b, 'relation': 'gaps_in', 'why': 'x' * 40}])
        self.assertNotIn('error', r, r)
        self.assertNotIn('connect_to', r['fields_updated'])
        self.assertFalse(any(d['field'] == 'connect_to' for d in r['deltas']))
        self.assertNotIn('connect_to', _kv_keys(self.brain, a))
        self.assertTrue(r['verified'])

    def test_field_swap_and_edge_in_one_op(self):
        a, b = self._pair()
        self.brain.revise(node_id=a, reason='seed', title='brain/9.6.0',
                          situation='When picking up — version is 9.6.0')
        r = self.brain.revise(node_id=a, reason='manifests moved 9.6.0 → 9.7.2',
                              title={'old': '9.6.0', 'new': '9.7.2'},
                              situation={'old': '9.6.0', 'new': '9.7.2'},
                              connect_to=[{'target': b, 'relation': 'gaps_in',
                                           'why': {'old': '9.6.0', 'new': '9.7.2'}}])
        self.assertNotIn('error', r, r)
        self.assertEqual(_title(self.brain, a), 'brain/9.7.2')
        self.assertEqual(_kv_value(self.brain, a, 'situation'),
                         'When picking up — version is 9.7.2')
        self.assertIn('9.7.2', self._rels(a, b)['gaps_in']['description'])

    def test_revise_batch_carries_connect_to(self):
        a, b = self._pair()
        r = self.brain.revise_batch(revisions=[
            {'node_id': a, 'reason': 'r',
             'connect_to': [{'target': b, 'relation': 'gaps_in', 'why': 'y' * 40}]}])
        self.assertEqual(r['revised'], 1, r)
        self.assertEqual(self._rels(a, b)['gaps_in']['description'], 'y' * 40)

    def _edge_traces(self, eid, relation):
        # Edge ref ids are composite `edge_id:relation` (mutation_emitter).
        rows = self.brain._trace_dal.conn.execute(
            "SELECT ref_type, metadata FROM trace_events WHERE ref_id = ?",
            ('%s:%s' % (eid, relation),)).fetchall()
        return [(r[0], json.loads(r[1]) if r[1] else {}) for r in rows]

    def test_dispatch_emits_edge_trace_not_node_delta(self):
        from servers.daemon_dispatch import dispatch_command
        a, b = self._pair()
        eid = self.brain._graph.get_edge_id(a, b)
        r = dispatch_command(self.brain, 'revise', {
            'node_id': a, 'reason': 'r',
            'connect_to': [{'target': b, 'relation': 'gaps_in',
                            'why': {'old': '9.6.0', 'new': '9.7.2'}}]}, [])
        self.assertTrue(r.get('ok'), r)
        self.assertNotIn('mutations', r)
        self.assertEqual(_query_revise_traces(self.brain, a), [])  # no node field changed
        traces = self._edge_traces(eid, 'gaps_in')
        self.assertEqual([t[0] for t in traces], ['edge_relation_revised'])

    def test_direction_warning_stays_on_the_edge(self):
        """A connect_to-only revise that warns about direction must not mint a
        node_revised trace; the warning rides the edge trace, and the manifest
        records the edge's STORED endpoints, not the caller's order."""
        from servers.daemon_dispatch import dispatch_command
        a = _make_node(self.brain, title='A')
        b = _make_node(self.brain, title='B')
        self.brain.connect_typed(b, a, relation='grounds', description=WHY, weight=0.6)
        r = dispatch_command(self.brain, 'revise', {
            'node_id': a, 'reason': 'r',
            'connect_to': [{'target': b, 'relation': 'supersedes', 'why': WHY}]}, [])
        self.assertTrue(r.get('ok'), r)
        self.assertEqual(r['result']['warnings'], [])           # not a node warning
        self.assertTrue(r['result']['connect_to_result']['warnings'])
        self.assertEqual(_query_revise_traces(self.brain, a), [])  # no phantom node trace
        eid = self.brain._graph.get_edge_id(a, b)
        traces = self._edge_traces(eid, 'supersedes')
        self.assertEqual(len(traces), 1, traces)
        md = traces[0][1]
        self.assertEqual((md.get('source_id'), md.get('target_id')), (b, a))
        self.assertTrue(any('passive verb' in w for w in md.get('warnings', [])), md)

    def test_revise_batch_surfaces_edge_failures_and_emits_edge_traces(self):
        from servers.daemon_dispatch import dispatch_command
        a, b = self._pair()
        c = _make_node(self.brain, title='C')
        eid = self.brain._graph.get_edge_id(a, b)
        r = dispatch_command(self.brain, 'revise_batch', {'revisions': [
            {'node_id': a, 'reason': 'r',
             'connect_to': [{'target': b, 'relation': 'gaps_in', 'why': 'z' * 40},
                            {'target': c, 'relation': 'grounds', 'why': 'short'}]}]}, [])
        self.assertTrue(r.get('ok'), r)
        row = r['result']['results'][0]
        ct = row['connect_to_result']
        self.assertEqual(len(ct['revised']), 1)
        self.assertEqual(len(ct['failed']), 1)
        self.assertIn('30+', ct['failed'][0]['reason'])
        self.assertEqual([t[0] for t in self._edge_traces(eid, 'gaps_in')],
                         ['edge_relation_revised'])

    def test_description_alias_and_short_bare_why(self):
        a, b = self._pair()
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': b, 'relation': 'gaps_in',
                                           'description': 'd' * 40}])
        self.assertEqual(r['connect_to_result']['failed'], [], r)
        self.assertEqual(self._rels(a, b)['gaps_in']['description'], 'd' * 40)
        r = self.brain.revise(node_id=a, reason='r',
                              connect_to=[{'target': b, 'relation': 'gaps_in', 'why': ''}])
        self.assertIn('30+', r['connect_to_result']['failed'][0]['reason'])
        self.assertEqual(self._rels(a, b)['gaps_in']['description'], 'd' * 40)


if __name__ == '__main__':
    unittest.main()
