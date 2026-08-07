"""The mutation-trace emitter: contract, mapping, and the one-writer pin.

Three groups, each catching a different failure class:

TestEmitterContract   — the ref_types the emitter can produce are all REGISTERED,
                        at every scale it can produce them at. This is a RUNTIME
                        check on purpose: test_trace_contract_sync's extractor
                        greps for literal quoted kwargs, so a table-driven emitter
                        with a variable ref_type yields ZERO triples there and
                        would pass vacuously.
TestBuildEvents       — the mapping itself, including per-row scale routing (the
                        defect a single command-level encoding_source would cause).
TestEmitBehaviour     — the timing gate and the never-raise guarantee.
TestOneWriterPin      — nobody adds a twelfth hand-rolled emit while the
                        migration is in flight.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import mutation_emitter as me
from servers import trace_contract as tc
from tests.brain_test_base import BrainTestBase

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeBrain:
    """Enough brain for build_events, which is pure apart from brain_today."""
    pass


def _node_row(**kw):
    """A lifecycle row (created / archived / deleted) — carries type+title."""
    row = {'node_id': 'n0000001', 'type': 'rule', 'title': 't', 'reason': 'r'}
    row.update(kw)
    return row


def _revise_row(**kw):
    """A revise row — deltas+warnings, and deliberately NO type/title: each slot
    has its own key set, matching its builder."""
    row = {'node_id': 'n0000001', 'reason': 'r',
           'deltas': [{'field': 'content', 'old': 'a', 'new': 'b'}], 'warnings': []}
    row.update(kw)
    return row


def _edge_row(**kw):
    row = {'edge_id': 'e0000001', 'source_id': 'a', 'target_id': 'b',
           'relation': 'extends', 'reason': 'r',
           'deltas': [{'field': 'weight', 'old': None, 'new': 0.5}], 'warnings': []}
    row.update(kw)
    return row


class TestEmitterContract(unittest.TestCase):

    def test_every_producible_ref_type_is_registered_at_every_scale(self):
        """The emitter derives scale per row, so any of its ref_types can land at
        s0, s1 OR s2. validate_trace_event raises on an unregistered triple and
        the emitter's loud-wrap would swallow it into an error row — meaning a
        missing registration is silent trace loss, not a crash. Pin all of it."""
        for _path, ref_type, _b, _r, _w, _s in me.MANIFEST_TRACE_MAP:
            for scale in ('s0', 's1', 's2'):
                ok, err = tc.validate_trace_event(scale, 'delta', ref_type)
                self.assertTrue(ok, "emitter can produce (%s, delta, %s) but the "
                                    "contract rejects it: %s" % (scale, ref_type, err))

    def test_map_covers_exactly_the_declared_emitter_ref_types(self):
        """EMITTER_REF_TYPES is what consumers exclude (S2 idle gate, dashboard
        run cards). If the map grows a kind that isn't in it, that kind starts
        re-arming the S2 gate and rendering as a phantom run card."""
        self.assertEqual(
            sorted({rt for _p, rt, _b, _r, _w, _s in me.MANIFEST_TRACE_MAP}),
            sorted(tc.EMITTER_REF_TYPES))

    def test_every_emitter_ref_type_has_an_enforced_metadata_shape(self):
        """All five, not just the new three. The revise pair's shapes were
        DECLARED but unregistered for months — validation was dead for the two
        highest-volume mutation events in the system."""
        for rt in tc.EMITTER_REF_TYPES:
            self.assertIn(rt, tc.METADATA_REQUIRED_BY_REF_TYPE,
                          "%s can be written by the emitter but its metadata "
                          "shape is not enforced" % rt)

    def test_each_builder_satisfies_its_shape_with_minimal_args(self):
        """Enforcement is only safe if the builders can't violate it. Minimal
        args is the worst case: every optional field falls to its default, and
        the shape requires ALL keys."""
        minimal = {
            'node_created': dict(node_id='n1'),
            'node_archived': dict(node_id='n1'),
            'node_deleted': dict(node_id='n1'),
            'node_revised': dict(node_id='n1', reason='r'),
            'edge_relation_revised': dict(edge_id='e1', relation='extends', reason='r'),
        }
        for _p, ref_type, builder, _r, _w, _s in me.MANIFEST_TRACE_MAP:
            md = builder(**minimal[ref_type])
            ok, err = tc.validate_trace_metadata('delta', ref_type, md)
            self.assertTrue(ok, "%s builder violates its own enforced shape: %s"
                                % (ref_type, err))

    def test_adding_a_kind_needs_no_branch(self):
        """The unification is DATA, not code: every entry must be fully described
        by its table row. If a builder stops accepting the row as kwargs, someone
        has started special-casing and the one-table property is gone."""
        import inspect
        for path, ref_type, builder, ref_id_of, emit_when, summary_of in me.MANIFEST_TRACE_MAP:
            self.assertTrue(callable(builder) and callable(ref_id_of)
                            and callable(emit_when), ref_type)
            params = inspect.signature(builder).parameters
            self.assertTrue(all(p.kind == p.KEYWORD_ONLY for p in params.values()),
                            "%s's builder must be keyword-only so a manifest row "
                            "can be splatted into it" % ref_type)
            # _builder_kwargs injects encoding_source into EVERY builder call —
            # scale and chain derive from it, so it is not optional. A builder
            # without it TypeErrors at emit time, which would make the
            # "one table row plus one builder" claim above false.
            self.assertIn('encoding_source', params,
                          "%s's builder must accept encoding_source — the emitter "
                          "sets it on every row (it drives scale AND chain)" % ref_type)


class TestBuildEvents(unittest.TestCase):

    def setUp(self):
        self.brain = _FakeBrain()

    def test_maps_each_slot_to_its_ref_type_and_ref_id(self):
        events = me.build_events(self.brain, {
            'nodes': {
                'created': [_node_row(node_id='c1')],
                'archived': [_node_row(node_id='a1')],
                'deleted': [_node_row(node_id='d1')],
                'revised': [_revise_row(node_id='r1')],
            },
            'edges': [_edge_row()],
        })
        got = {(e['ref_type'], e['ref_id']) for e in events}
        self.assertEqual(got, {
            ('node_created', 'c1'), ('node_archived', 'a1'),
            ('node_deleted', 'd1'), ('node_revised', 'r1'),
            ('edge_relation_revised', 'e0000001:extends'),
        })

    def test_per_row_encoding_source_routes_scale_and_chain(self):
        """THE defect a single command-level value would cause: a brain_batch
        carrying an s2 archive and an anchor revise must NOT collapse onto one
        scale/chain — scale is derived from each row's own encoding_source."""
        events = me.build_events(self.brain, {
            'nodes': {
                'archived': [_node_row(node_id='a1',
                                       encoding_source='s2:consolidation')],
                'created': [_node_row(node_id='c1', encoding_source='encoder:sonnet')],
                'deleted': [_node_row(node_id='d1', encoding_source='anchor')],
            },
        })
        by_id = {e['ref_id']: e for e in events}
        self.assertEqual(by_id['a1']['scale'], 's2')
        self.assertEqual(by_id['c1']['scale'], 's1')
        self.assertEqual(by_id['d1']['scale'], 's0')
        # and each row's metadata keeps its own attribution
        self.assertEqual(by_id['a1']['metadata']['encoding_source'], 's2:consolidation')
        self.assertEqual(by_id['c1']['metadata']['encoding_source'], 'encoder:sonnet')

    def test_caller_chain_wins_over_the_date_fallback(self):
        events = me.build_events(self.brain,
                                 {'nodes': {'created': [_node_row()]}},
                                 chain_id='s1e-abc12345-7')
        self.assertEqual(events[0]['chain_id'], 's1e-abc12345-7')

    def test_new_types_fall_back_to_mutation_chain_not_revise(self):
        """A creation or hard delete on a `-revise` chain renders in the dashboard
        as 'Refined N memories' — a lie. New types get `-mutation`; the
        pre-existing pair keeps `-revise` for reader bit-compatibility."""
        created = me.build_events(self.brain, {'nodes': {'created': [_node_row()]}})
        revised = me.build_events(self.brain, {'nodes': {'revised': [_revise_row()]}})
        self.assertTrue(created[0]['chain_id'].endswith('-mutation'),
                        created[0]['chain_id'])
        self.assertTrue(revised[0]['chain_id'].endswith('-revise'),
                        revised[0]['chain_id'])

    def test_unchanged_rows_stay_silent(self):
        """An idempotent re-connect (no deltas, no warnings) emitted nothing
        before the emitter and must emit nothing after it."""
        self.assertEqual(me.build_events(self.brain, {
            'edges': [_edge_row(deltas=[], warnings=[])],
            'nodes': {'revised': [_revise_row(deltas=[], warnings=[])]},
        }), [])

    def test_unknown_row_keys_are_dropped_and_recorded_not_fatal(self):
        """A key the builder doesn't accept must NOT cost the command its traces.
        Splatting the row raised TypeError, and the loud-wrap turned that into
        zero traces for the whole command — far too much blast radius for a
        spurious field. Drop it, record it, still emit."""
        drops = []
        events = me.build_events(
            self.brain,
            {'nodes': {'revised': [_revise_row(title='revise rows have no title')]}},
            drops=drops)
        self.assertEqual(len(events), 1, "the trace must still be written")
        self.assertNotIn('title', events[0]['metadata'])
        self.assertTrue(any('title' in d for d in drops),
                        "the drift must be recorded so it can be logged: %s" % drops)

    def test_creations_and_deletions_always_emit(self):
        """They carry no deltas — gating them on `_changed` would silence them."""
        self.assertEqual(len(me.build_events(
            self.brain, {'nodes': {'created': [_node_row()],
                                   'deleted': [_node_row()]}})), 2)

    def test_absent_and_partial_manifest_slots_are_tolerated(self):
        self.assertEqual(me.build_events(self.brain, {}), [])
        self.assertEqual(me.build_events(self.brain, {'nodes': {}}), [])
        self.assertEqual(me.build_events(self.brain, {'edges': []}), [])

    def test_unusable_ref_id_raises(self):
        """A trace whose subject can't be identified is unresolvable forever —
        better to lose the batch loudly than to write an orphan row."""
        with self.assertRaises(ValueError):
            me.build_events(self.brain, {'nodes': {'created': [_node_row(node_id='')]}})
        with self.assertRaises(ValueError):
            me.build_events(self.brain, {'edges': [_edge_row(relation='')]})


class TestEmitBehaviour(BrainTestBase):
    needs_embedder = False

    def _errors(self, source):
        return self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type='error' AND source=?",
            (source,)).fetchone()[0]

    def test_open_transaction_skips_and_is_loud(self):
        """The timing gate. Post-commit is CONDITIONAL — handlers can return with
        brain.conn mid-transaction (MetadataKVDAL.set_many doesn't commit;
        archive_dangling_edges has no commit at all). Emitting then would orphan
        the trace if the write rolled back, so we skip and shout. This is the
        emitter's most valuable behaviour: it detects the leak class."""
        before = self._errors('mutation_trace_txn_open')
        self.brain.conn.execute(
            "CREATE TABLE IF NOT EXISTS _emitter_probe (x int)")
        self.brain.conn.execute("INSERT INTO _emitter_probe VALUES (1)")
        self.assertTrue(self.brain.conn.in_transaction, "probe setup failed")
        try:
            me.emit_mutation_traces(self.brain, 'remember',
                                    {'nodes': {'created': [_node_row(node_id='zz1')]}})
            self.assertEqual(self._errors('mutation_trace_txn_open'), before + 1)
            self.assertEqual(
                self.brain._trace_dal.get_chain(
                    's0-%s-mutation' % me.brain_today(self.brain).strftime('%Y%m%d')),
                [], "a trace was written while a transaction was open")
        finally:
            self.brain.conn.rollback()

    def test_emits_one_batch_and_never_raises_on_a_bad_manifest(self):
        before = self._errors('mutation_trace_emit')
        # `nodes.created` holding a string instead of a row dict — a programmer
        # error that must not propagate into the caller's write result.
        me.emit_mutation_traces(self.brain, 'remember',
                                {'nodes': {'created': ['not-a-dict']}})
        self.assertEqual(self._errors('mutation_trace_emit'), before + 1)

    def test_unknown_keys_are_logged_loud_while_the_row_still_lands(self):
        before = self._errors('mutation_trace_unknown_keys')
        me.emit_mutation_traces(
            self.brain, 'revise',
            {'nodes': {'revised': [_revise_row(node_id='uk000001',
                                               title='not a revise field')]}},
            chain_id='test-emitter-unknown')
        self.assertEqual(self._errors('mutation_trace_unknown_keys'), before + 1,
                         "manifest/contract drift must be loud")
        self.assertEqual(len(self.brain._trace_dal.get_chain('test-emitter-unknown')), 1,
                         "...but the trace still goes out — loud, never blocking")

    def test_enforcing_the_revise_pair_is_silent_on_real_production_writes(self):
        """3e's safety claim, tested on the REAL path rather than by inspection.

        Registering node_revised / edge_relation_revised makes
        validate_trace_metadata start checking live traffic. It warns to stderr
        and never blocks, so the rest of the suite would stay green even if every
        write were spraying warnings — nothing would fail. This is the only test
        that would notice.
        """
        import io
        from contextlib import redirect_stderr
        from servers.daemon_dispatch import dispatch_command

        r = dispatch_command(self.brain, 'remember', {
            'type': 'rule', 'title': 'emitter 3e probe',
            'content': 'a node to revise so the revise trace has a real payload',
        }, [])
        node_id = (r.get('result') or {}).get('id') or (r.get('result') or {}).get('node_id')
        self.assertTrue(node_id, "probe setup failed: %s" % r)

        buf = io.StringIO()
        with redirect_stderr(buf):
            # A revise (node_revised) and a connect (edge_relation_revised) —
            # both enforced ref_types, both on their real production paths.
            dispatch_command(self.brain, 'revise', {
                'node_id': node_id, 'reason': '3e probe',
                'content': 'revised content so there is a real delta',
            }, [])
            dispatch_command(self.brain, 'connect', {
                'source_id': node_id, 'target_id': node_id,
                'relation': 'extends', 'reason': '3e probe',
            }, [])
        err = buf.getvalue()

        # NOT VACUOUS: "no warning" also describes "nothing was written", so
        # prove the enforced path actually ran. Both rows must exist, or this
        # test would pass while validation was never reached.
        wrote = self.brain.logs_conn.execute(
            "SELECT ref_type, COUNT(*) FROM trace_events "
            "WHERE ref_type IN ('node_revised','edge_relation_revised') "
            "GROUP BY ref_type").fetchall()
        self.assertEqual(
            sorted(rt for rt, _n in wrote),
            ['edge_relation_revised', 'node_revised'],
            "the probe wrote no enforced rows, so this test proves nothing: %s" % wrote)

        self.assertNotIn('trace metadata invalid', err,
                         "enforcing the revise pair fires on live writes — 3e is "
                         "NOT warning-silent:\n%s" % err)

    def test_happy_path_writes_the_rows(self):
        me.emit_mutation_traces(
            self.brain, 'remember',
            {'nodes': {'created': [_node_row(node_id='hp000001')]}},
            chain_id='test-emitter-happy')
        rows = self.brain._trace_dal.get_chain('test-emitter-happy')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['ref_type'], 'node_created')
        self.assertEqual(rows[0]['ref_id'], 'hp000001')


class TestIntegrityArchiveTraces(BrainTestBase):
    """health_check(auto_fix=True) archives stale context nodes at every
    session boot WITHOUT crossing the dispatch chokepoint. When archive_node's
    inline trace died (step 8), this path had to gain a direct emit or every
    boot's hook:integrity archives would be permanently invisible
    (review 2026-08-06)."""
    needs_embedder = False

    def test_auto_fix_archives_emit_node_archived_rows(self):
        from servers.clock import brain_today
        ids = []
        for i in range(11):
            r = self.brain.remember(type='context',
                                    title='stale-ctx-%d' % i,
                                    content='stale context body %d' % i,
                                    encoding_source='anchor')
            ids.append(r['id'])
        ph = ','.join('?' * len(ids))
        self.brain.conn.execute(
            "UPDATE nodes SET created_at = '2026-01-01T00:00:00+00:00' "
            "WHERE id IN (%s)" % ph, ids)
        self.brain.conn.commit()

        self.brain.health_check(session_id='hc-probe', auto_fix=True)

        # -integrity, not -mutation: the junk purge's maint chain lives at
        # s2, and one chain_id must never span two scales.
        chain = 'maint-%s-integrity' % brain_today(self.brain).strftime('%Y%m%d')
        rows = [t for t in self.brain._trace_dal.get_chain(chain)
                if t['ref_type'] == 'node_archived'
                and t['ref_id'] in set(ids)]
        self.assertEqual(len(rows), len(ids),
                         "every hook:integrity archive must leave a row")
        for t in rows:
            self.assertEqual(t['scale'], 's0')
            self.assertEqual(t['metadata']['archived_by'], 'hook:integrity')
            self.assertTrue(t['metadata']['title'].startswith('stale-ctx-'))


class TestOneWriterPin(unittest.TestCase):
    """The emitter must become the ONLY mutation-trace writer.

    ALLOWLIST is the migration in flight: dispatch_write.py still holds the
    hand-rolled emits, and brain_remember.archive_node its inline trace. Plan
    steps 4-7 convert them one handler per commit — DELETE the entry as each
    site dies, so this pin tightens automatically and a NEW hand-rolled emit
    fails immediately.
    """
    ALLOWLIST = {
        'servers/mutation_emitter.py',  # the sanctioned writer — and nothing else
    }

    def test_no_new_hand_rolled_mutation_emitters(self):
        writes = re.compile(r"_trace_dal\.(append|append_batch)\b")
        mutation_rt = re.compile(r"['\"](node_created|node_archived|node_deleted|"
                                 r"node_revised|edge_relation_revised)['\"]")
        offenders = []
        for root, _dirs, files in os.walk(os.path.join(REPO, 'servers')):
            for name in files:
                if not name.endswith('.py'):
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, REPO)
                if rel in self.ALLOWLIST:
                    continue
                with open(path, encoding='utf-8') as fh:
                    body = fh.read()
                if writes.search(body) and mutation_rt.search(body):
                    offenders.append(rel)
        self.assertEqual(offenders, [],
                         "these write mutation trace ref_types directly — route "
                         "them through mutation_emitter instead: %s" % offenders)

    def test_allowlist_has_no_stale_entries(self):
        """A cleared site left on the allowlist silently re-opens the hole."""
        writes = re.compile(r"_trace_dal\.(append|append_batch)\b")
        for rel in sorted(self.ALLOWLIST - {'servers/mutation_emitter.py'}):
            with open(os.path.join(REPO, rel), encoding='utf-8') as fh:
                self.assertTrue(writes.search(fh.read()),
                                "%s no longer writes traces — remove it from "
                                "ALLOWLIST so the pin tightens" % rel)


if __name__ == '__main__':
    unittest.main()
