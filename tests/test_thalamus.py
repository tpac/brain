"""Thalamus Phase 1 — the door, the pull, the verbs, the sweep.

Covers the core contracts:
- file(): routing (queue vs live delegation), defaults by kind, dedup upsert,
  the LOUD synchronous budget rejection, when/for_whom resolution errors.
- pull(): the delivery predicate (due/window/audience/target/ledger),
  asks-at-boot-only, push-once-per-session idempotence, once vs all.
- resolve(): answer / defer (re-arm) / dismiss, exactly-one-action guard.
- withdraw(): source-scoped producer retraction.
- expire_due(): window sweep; an unanswered ask dies LOUDLY (dead-letter).
- render: pre-filled resolve call, loud truncation.
- dispatch handlers return the {"ok", "result"} envelope (the dispatch_self
  lesson — a raw payload surfaces as "Unknown daemon error").
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.channels.thalamus import thalamus, thalamus_contract as tc
from servers.dispatch_thalamus import (
    _handle_remind, _handle_thalamus_list, _handle_thalamus_resolve)
from servers.clock import iso_now, iso_cutoff

S1 = 'aaaaaaaa-1111-2222-3333-444444444444'
S2 = 'bbbbbbbb-1111-2222-3333-444444444444'


class ThalamusBase(BrainTestBase):
    needs_embedder = False

    def _file(self, body='check the thing', **kw):
        kw.setdefault('source', 'test')
        return thalamus.file(self.brain, kw.pop('source'), body, **kw)

    def _row(self, item_id):
        return self.brain.logs_conn.execute(
            'SELECT state, audience, needs_answer, deliver_at, expires_at,'
            ' target_session FROM thalamus_items WHERE id = ?',
            (item_id,)).fetchone()

    def _age_out(self, item_id):
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_items SET expires_at = ? WHERE id = ?',
                (iso_cutoff(hours=1), item_id))
            self.brain.logs_conn_w.commit()


class TestFile(ThalamusBase):

    def test_notice_defaults(self):
        r = self._file()
        self.assertTrue(r['filed'])
        self.assertEqual(r['route'], 'queue')
        state, audience, needs_answer, deliver_at, expires_at, _ = self._row(r['id'])
        self.assertEqual(state, tc.STATE_OPEN)
        self.assertEqual(audience, tc.AUDIENCE_FIRST)
        self.assertEqual(needs_answer, 0)
        self.assertIsNone(deliver_at)
        self.assertGreater(expires_at, iso_now())

    def test_ask_defaults_to_all_audience(self):
        r = self._file('should X be configurable?', needs_answer=True)
        _, audience, needs_answer, _, expires_at, _ = self._row(r['id'])
        self.assertEqual(audience, tc.AUDIENCE_EVERY)
        self.assertEqual(needs_answer, 1)
        # Ask window is the long one.
        self.assertGreater(expires_at, iso_cutoff(days=-(tc.ASK_EXPIRES_DAYS - 1)))

    def test_when_shorthand_sets_future_deliver_at(self):
        r = self._file(when='2h')
        _, _, _, deliver_at, _, _ = self._row(r['id'])
        self.assertIsNotNone(deliver_at)
        self.assertGreater(deliver_at, iso_now())

    def test_bad_when_rejects_loudly(self):
        r = self._file(when='next full moon')
        self.assertFalse(r['filed'])
        self.assertIn('when', r['error'])

    def test_bad_for_whom_rejects_loudly(self):
        r = self._file(for_whom='everyone-ish')
        self.assertFalse(r['filed'])
        self.assertIn('for_whom', r['error'])

    def test_directed_requires_full_uuid_and_stamps_target(self):
        r = self._file(for_whom=S1)
        _, audience, _, _, _, target = self._row(r['id'])
        self.assertEqual(target, S1)
        self.assertEqual(audience, tc.AUDIENCE_FIRST)
        # An 8-char short is a display convention, not a key.
        r2 = self._file(for_whom=S1[:8])
        self.assertFalse(r2['filed'])

    def test_empty_body_rejects(self):
        self.assertFalse(self._file('  ')['filed'])

    def test_directed_ask_rejects_as_undeliverable(self):
        # Asks render at boot only, and a nameable session has already booted —
        # a directed ask would wait out its window and dead-letter, guaranteed.
        r = self._file('judge this?', needs_answer=True, for_whom=S1)
        self.assertFalse(r['filed'])
        self.assertIn('self_send', r['error'])

    def test_expires_before_when_rejects(self):
        r = self._file(when='3d', expires='1d')
        self.assertFalse(r['filed'])
        self.assertIn('before it ever becomes due', r['error'])

    def test_dedup_key_updates_not_duplicates(self):
        r1 = self._file('v1 of the concern', dedup_key='concern-x')
        r2 = self._file('v2 of the concern', dedup_key='concern-x')
        self.assertTrue(r2.get('updated'))
        self.assertEqual(r1['id'], r2['id'])
        body = self.brain.logs_conn.execute(
            'SELECT body FROM thalamus_items WHERE id = ?',
            (r1['id'],)).fetchone()[0]
        self.assertEqual(body, 'v2 of the concern')
        n = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM thalamus_items WHERE dedup_key='concern-x'"
        ).fetchone()[0]
        self.assertEqual(n, 1)

    def test_dedup_refile_escalates_a_note_into_an_ask(self):
        """The change-gate covers every producer-controlled delivery
        attribute, so escalating a standing note into an ask actually lands.
        Comparing only (body, refs, deliver_at) left needs_answer=0 — still a
        note, still Stop-delivered, never dead-lettering — while the door had
        already granted it the ask's longer window."""
        r1 = self._file('the recurring thing', dedup_key='esc')
        self.assertEqual(self._row(r1['id'])[2], 0)
        r2 = self._file('the recurring thing', dedup_key='esc',
                        needs_answer=True)
        self.assertEqual(r2['id'], r1['id'])
        self.assertTrue(r2['rearmed'])
        _, audience, needs_answer, _, _, _ = self._row(r2['id'])
        self.assertEqual(needs_answer, 1)
        # ...including the audience the new kind implies — an ask left on the
        # notice audience would deliver to exactly one session, ever.
        self.assertEqual(audience, tc.AUDIENCE_EVERY)

    def test_dedup_refile_retargets_and_rearms(self):
        """for_whom is producer-controlled too: re-filing the same text to a
        different recipient set must move the row, not silently keep the old
        one."""
        r1 = self._file('heads up', dedup_key='aim')
        self.assertEqual(self._row(r1['id'])[1], tc.AUDIENCE_FIRST)
        r2 = self._file('heads up', dedup_key='aim', for_whom=S1)
        self.assertTrue(r2['rearmed'])
        _, audience, _, _, _, target = self._row(r2['id'])
        self.assertEqual(target, S1)
        self.assertEqual(audience, tc.AUDIENCE_FIRST)

    def test_dated_item_keeps_full_window(self):
        """Expiry anchors at deliver_at, not now — an ask due in 3 weeks must
        not expire after 2 (it would be undeliverable and fire a FALSE loud
        dead-letter)."""
        r = self._file('late ask', needs_answer=True, when='3w')
        _, _, _, deliver_at, expires_at, _ = self._row(r['id'])
        self.assertGreater(expires_at, deliver_at)
        r2 = self._file('late notice', for_whom='all', when='1w')
        _, _, _, deliver_at2, expires_at2, _ = self._row(r2['id'])
        self.assertGreater(expires_at2, deliver_at2)

    def test_dedup_with_second_matching_row_still_files(self):
        """Regression for the write-conn snapshot hazard: a second open row
        sharing (source, dedup_key) must not break file() — the dedup lookup
        is LIMIT 1 and its cursor unbound."""
        r1 = self._file('first', dedup_key='dup')
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'INSERT INTO thalamus_items (id, source, body, refs, audience,'
                ' target_session, needs_answer, dedup_key, deliver_at,'
                ' expires_at, state, answer, created_at)'
                " VALUES ('th_dupdup', 'test', 'second', '', 'first_session',"
                " '', 0,"
                " 'dup', NULL, ?, 'open', '', ?)",
                (iso_cutoff(days=-7), iso_now()))
            self.brain.logs_conn_w.commit()
        r2 = self._file('updated', dedup_key='dup')
        self.assertTrue(r2['filed'])
        self.assertTrue(r2.get('updated'))

    def test_live_rejects_queue_shaped_params(self):
        r = self._file('fyi', for_whom='live', session_id=S1, when='2h')
        self.assertFalse(r['filed'])
        self.assertIn('live', r['error'])
        r = self._file('fyi', for_whom='live', session_id=S1,
                       needs_answer=True)
        self.assertFalse(r['filed'])

    def test_budget_ignores_expired_items(self):
        for i in range(tc.MAX_OPEN_PER_SOURCE):
            self._file('item %d' % i)
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                "UPDATE thalamus_items SET expires_at = ? WHERE state='open'",
                (iso_cutoff(hours=1),))
            self.brain.logs_conn_w.commit()
        # Expired-but-unswept items must not wedge the producer at its cap.
        self.assertTrue(self._file('fits without a sweep')['filed'])

    def test_budget_rejects_at_cap_with_guidance(self):
        for i in range(tc.MAX_OPEN_PER_SOURCE):
            self.assertTrue(self._file('item %d' % i)['filed'])
        r = self._file('one too many')
        self.assertFalse(r['filed'])
        self.assertIn('budget', r['error'])
        self.assertIn('withdraw', r['error'])
        # Closing one frees the slot — the cap counts OPEN items only.
        first = self.brain.logs_conn.execute(
            "SELECT id FROM thalamus_items WHERE state='open' LIMIT 1"
        ).fetchone()[0]
        thalamus.resolve(self.brain, first, dismiss=True)
        self.assertTrue(self._file('fits again')['filed'])

    def test_live_now_requires_filing_session(self):
        r = self._file(for_whom='live')
        self.assertFalse(r['filed'])
        self.assertIn('session', r['error'])

    def test_live_now_delegates_to_courier_and_goes_terminal(self):
        r = self._file('main moved — rebase', for_whom='live', session_id=S1)
        self.assertTrue(r['filed'])
        self.assertEqual(r['route'], 'live')
        self.assertEqual(self._row(r['id'])[0], tc.STATE_SENT)
        # The courier holds the actual broadcast.
        courier = self.brain.logs_conn.execute(
            'SELECT address, from_session FROM self_inflight WHERE id = ?',
            (r['courier_id'],)).fetchone()
        self.assertEqual(courier[0], 'self:broadcast')
        self.assertEqual(courier[1], S1)
        # Terminal rows never enter the pull path.
        block, n = thalamus.pull(self.brain, S2, via='stop')
        self.assertEqual(n, 0)


class TestDoorContract(ThalamusBase):
    """The door's producer-facing contract: ONE result envelope across the
    module, and `source` as vocabulary rather than free text."""

    def test_one_envelope_across_the_module(self):
        """file() answers in resolve()/withdraw()'s shape — a Phase 2 producer
        learns one result contract, not two."""
        filed = self._file('x')
        self.assertIs(filed['ok'], True)
        self.assertIn('id', filed)
        self.assertIs(thalamus.resolve(self.brain, filed['id'],
                                       dismiss=True)['ok'], True)
        # A rejection carries the same key, inverted.
        self.assertIs(self._file('  ')['ok'], False)

    def test_filed_alias_never_disagrees_with_ok(self):
        """'filed' survives one release for anything still reading it."""
        for r in (self._file('x'), self._file('  '),
                  self._file('y', when='next full moon')):
            self.assertEqual(r['filed'], r['ok'])

    def test_free_text_source_rejects_loudly(self):
        """source is the budget key AND the withdraw-ownership key — a typo'd
        one gets a fresh budget and orphans its own items."""
        for bad in ('', 'Anchor', 'my source', 's2:', ':unit', 'a:b:c'):
            r = self._file('x', source=bad)
            self.assertIs(r['ok'], False, 'source %r should not file' % bad)
            self.assertIn('source', r['error'])

    def test_category_process_grammar_files(self):
        for good in ('anchor', 'encoder:sonnet', 's2:consolidation', 'hook'):
            self.assertIs(self._file('x', source=good)['ok'], True,
                          'source %r should file' % good)


class TestPull(ThalamusBase):

    def test_notice_delivers_once_across_sessions(self):
        r = self._file('heads up')
        block, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 1)
        self.assertIn(r['id'], block)
        # audience first_session: a second session gets nothing.
        _, n2 = thalamus.pull(self.brain, S2, via='stop')
        self.assertEqual(n2, 0)

    def test_push_once_per_session_is_ledger_idempotent(self):
        self._file('standing note', for_whom='all')
        _, n1 = thalamus.pull(self.brain, S1, via='boot')
        _, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual((n1, n2), (1, 0))
        # A DIFFERENT session still gets its copy.
        _, n3 = thalamus.pull(self.brain, S2, via='boot')
        self.assertEqual(n3, 1)

    def test_ask_boot_only(self):
        r = self._file('decide X?', needs_answer=True)
        _, n_stop = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n_stop, 0)
        block, n_boot = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n_boot, 1)
        self.assertIn('thalamus_resolve("%s"' % r['id'], block)

    def test_future_deliver_at_not_due_yet(self):
        self._file('later', when='1w')
        _, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 0)

    def test_directed_reaches_only_its_target(self):
        self._file('for one stream', for_whom=S1)
        _, n_other = thalamus.pull(self.brain, S2, via='boot')
        self.assertEqual(n_other, 0)
        _, n_target = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n_target, 1)

    def test_ledger_records_via_and_session(self):
        r = self._file('note')
        thalamus.pull(self.brain, S1, via='stop')
        row = self.brain.logs_conn.execute(
            'SELECT session_id, via FROM thalamus_deliveries WHERE item_id = ?',
            (r['id'],)).fetchone()
        self.assertEqual((row[0], row[1]), (S1, 'stop'))

    def test_empty_session_is_noop(self):
        self._file('note')
        self.assertEqual(thalamus.pull(self.brain, '', via='boot'), ('', 0))

    def test_overflow_names_the_true_count(self):
        for i in range(tc.PULL_MAX_ITEMS + 4):
            self._file('item %d' % i, source='src-%d' % i)  # distinct budgets
        block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, tc.PULL_MAX_ITEMS)
        self.assertIn('(+4 more due', block)  # the real number, not "+1"


class TestResolveWithdraw(ThalamusBase):

    def test_answer_closes_and_stores_payload(self):
        r = self._file('decide X?', needs_answer=True)
        out = thalamus.resolve(self.brain, r['id'], answer='derive from aspects')
        self.assertTrue(out['ok'])
        state, answer = self.brain.logs_conn.execute(
            'SELECT state, answer FROM thalamus_items WHERE id = ?',
            (r['id'],)).fetchone()
        self.assertEqual(state, tc.STATE_ANSWERED)
        self.assertEqual(answer, 'derive from aspects')

    def test_exactly_one_action(self):
        r = self._file('x')
        out = thalamus.resolve(self.brain, r['id'],
                               answer='a', dismiss=True)
        self.assertFalse(out['ok'])
        out = thalamus.resolve(self.brain, r['id'])
        self.assertFalse(out['ok'])

    def test_empty_answer_rejected(self):
        r = self._file('decide?', needs_answer=True)
        out = thalamus.resolve(self.brain, r['id'], answer='   ')
        self.assertFalse(out['ok'])
        self.assertIn('empty', out['error'])
        # Item stays open and answerable.
        self.assertEqual(self._row(r['id'])[0], tc.STATE_OPEN)

    def test_defer_rearms_delivery(self):
        r = self._file('note')
        thalamus.pull(self.brain, S1, via='stop')  # delivered
        out = thalamus.resolve(self.brain, r['id'], defer_until='1h')
        self.assertTrue(out['ok'])
        # Re-armed (new epoch), but not due yet → still nothing.
        _, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 0)
        deliver_at = self._row(r['id'])[3]
        self.assertGreater(deliver_at, iso_now())
        # The ledger is APPEND-ONLY: defer preserves the delivery history —
        # "delivered, then deferred" must stay distinguishable from "never
        # delivered" (Phase 3 retry gates on unacked).
        ledger = self.brain.logs_conn.execute(
            'SELECT COUNT(*) FROM thalamus_deliveries WHERE item_id = ?',
            (r['id'],)).fetchone()[0]
        self.assertEqual(ledger, 1)
        epoch = self.brain.logs_conn.execute(
            'SELECT armed_epoch FROM thalamus_items WHERE id = ?',
            (r['id'],)).fetchone()[0]
        self.assertEqual(epoch, 1)

    def test_rearm_redelivers_to_same_session_and_keeps_history(self):
        r = self._file('note')
        _, n1 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n1, 1)
        thalamus.resolve(self.brain, r['id'], defer_until='now')
        _, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n2, 1)  # same session, new epoch → delivers again
        rows = self.brain.logs_conn.execute(
            'SELECT armed_epoch FROM thalamus_deliveries WHERE item_id = ?'
            ' ORDER BY armed_epoch', (r['id'],)).fetchall()
        self.assertEqual([e for (e,) in rows], [0, 1])  # both generations kept
        out = thalamus.list_items(self.brain)
        item = next(i for i in out['items'] if i['id'] == r['id'])
        self.assertEqual(item['deliveries'], 2)        # all-time
        self.assertEqual(item['deliveries_epoch'], 1)  # current generation

    def test_dedup_refile_rearms_delivery(self):
        """A re-file under the same (source, dedup_key) is a re-arm: a
        once-item already delivered must deliver again with the updated
        content — not stay suppressed by the old generation's ledger row."""
        self._file('v1', dedup_key='concern-x')
        _, n1 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n1, 1)
        r2 = self._file('v2 updated concern', dedup_key='concern-x')
        self.assertTrue(r2.get('updated'))
        block, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n2, 1)
        self.assertIn('v2 updated concern', block)

    def test_identical_refile_is_idempotent(self):
        """An UNCHANGED re-file (a cyclic producer re-asserting its standing
        item) must not re-arm — bumping on a no-op would re-deliver the same
        text every producer cycle, unbounded. It refreshes the window only."""
        self._file('standing concern', dedup_key='concern-x')
        _, n1 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n1, 1)
        r2 = self._file('standing concern', dedup_key='concern-x')
        self.assertTrue(r2['updated'])
        self.assertFalse(r2['rearmed'])
        _, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n2, 0)  # no re-delivery
        # Window refreshed: expires_at moved forward with the re-file.
        self.assertGreater(self._row(r2['id'])[4], iso_now())

    def test_deferred_ask_keeps_ask_window(self):
        """extend_window composes through window_for — a deferred ask keeps
        its kind's full span (14d) past the new due date, not a reminder's
        7-day grace."""
        from datetime import datetime, timedelta
        r = self._file('decide X?', needs_answer=True)
        out = thalamus.resolve(self.brain, r['id'], defer_until='30d')
        self.assertTrue(out['ok'])
        due = datetime.fromisoformat(out['deliver_at'])
        expires = datetime.fromisoformat(out['expires_at'])
        self.assertGreaterEqual(expires - due,
                                timedelta(days=tc.ASK_EXPIRES_DAYS))

    def test_resolve_unknown_or_closed(self):
        self.assertFalse(thalamus.resolve(self.brain, 'th_nope',
                                          dismiss=True)['ok'])
        r = self._file('x')
        thalamus.resolve(self.brain, r['id'], dismiss=True)
        again = thalamus.resolve(self.brain, r['id'], dismiss=True)
        self.assertFalse(again['ok'])
        self.assertIn('already', again['error'])

    def test_withdraw_is_source_scoped(self):
        r = self._file('mine', source='s2:consolidation')
        stolen = thalamus.withdraw(self.brain, 'someone-else', item_id=r['id'])
        self.assertFalse(stolen['ok'])
        own = thalamus.withdraw(self.brain, 's2:consolidation', item_id=r['id'])
        self.assertTrue(own['ok'])
        self.assertEqual(self._row(r['id'])[0], tc.STATE_WITHDRAWN)

    def test_withdraw_by_dedup_key(self):
        self._file('mine', source='s2:x', dedup_key='k1')
        out = thalamus.withdraw(self.brain, 's2:x', dedup_key='k1')
        self.assertTrue(out['ok'])


class TestExpiry(ThalamusBase):

    def test_notice_expires_naturally(self):
        r = self._file('old news')
        self._age_out(r['id'])
        self.assertEqual(thalamus.expire_due(self.brain), 1)
        self.assertEqual(self._row(r['id'])[0], tc.STATE_EXPIRED)

    def test_expired_ask_is_loud_dead_letter(self):
        r = self._file('never answered?', needs_answer=True)
        self._age_out(r['id'])
        before = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE source='thalamus_ask_expired'"
        ).fetchone()[0]
        self.assertEqual(thalamus.expire_due(self.brain), 1)
        after = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE source='thalamus_ask_expired'"
        ).fetchone()[0]
        self.assertEqual(after, before + 1)

    def test_expired_never_delivers(self):
        r = self._file('too late')
        self._age_out(r['id'])
        _, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 0)


class TestVocabulary(ThalamusBase):
    """Contract-first: moments and audiences are closed vocabulary — unknown
    values fail LOUDLY instead of silently behaving as a default."""

    def test_unknown_via_is_loud(self):
        self._file('x')
        with self.assertRaises(ValueError):
            thalamus.pull(self.brain, S1, via='bot')  # the ledgered-typo case

    def test_internal_audience_drift_is_loud(self):
        """An audience outside tc.AUDIENCES matches neither pull-predicate
        branch (open forever, silent death at expiry) — the door rejects."""
        from unittest import mock
        with mock.patch.object(tc, 'resolve_for_whom',
                               return_value=('queue', 'weekly', '')):
            r = self._file('x')
        self.assertFalse(r['filed'])
        self.assertIn('audience', r['error'])

    def test_resolve_for_whom_outputs_stay_inside_audiences(self):
        """The contract half of the audience guard (gate 4): every reachable
        resolve_for_whom audience is in the closed set, and the pull
        predicate binds BOTH members — resolver, vocabulary, and predicate
        cannot drift apart without this failing at merge time. (The runtime
        tripwire in file() covers the future-new-branch case a fixed
        enumeration can't see.)"""
        cases = [(None, False), (None, True), ('', False), ('', True),
                 ('all', False), ('all', True),
                 ('123e4567-e89b-12d3-a456-426614174000', False)]
        for fw, na in cases:
            route, audience, _ = tc.resolve_for_whom(fw, na)
            if route == 'queue':
                self.assertIn(
                    audience, tc.AUDIENCES,
                    'resolve_for_whom(%r, %r) -> %r escaped the closed set'
                    % (fw, na, audience))
        _, params = thalamus._due_filter(
            S1, 'boot', '2027-01-01T00:00:00+00:00')
        for member in tc.AUDIENCES:
            self.assertIn(member, params,
                          'pull predicate does not bind audience %r' % member)

    def test_ddl_audience_default_is_inside_the_closed_set(self):
        """The CREATED table's audience default must be a member of the closed
        set. A default outside it matches neither pull-predicate branch, so any
        insert omitting the column writes a row that never delivers and dies
        silently at expiry — below the reach of file()'s runtime tripwire. The
        v3 rename updated existing rows but left the column default at 'once'.
        """
        default = None
        for col in self.brain.logs_conn.execute(
                'PRAGMA table_info(thalamus_items)').fetchall():
            if col[1] == 'audience':
                default = (col[4] or '').strip("'")
        self.assertIn(
            default, tc.AUDIENCES,
            'thalamus_items.audience DDL default %r is not in %s — an insert '
            'omitting the column would be undeliverable' % (default,
                                                            tc.AUDIENCES))

    def test_kind_is_one_derivation_for_verb_and_span(self):
        self.assertEqual(tc.kind_of({'needs_answer': 1}), tc.KIND_ASK)
        self.assertEqual(tc.kind_of({'deliver_at': '2027-01-01T00:00:00+00:00'}),
                         tc.KIND_REMINDER)
        self.assertEqual(tc.kind_of({}), tc.KIND_NOTICE)
        # Both partitions read the same derivation.
        self.assertEqual(set(tc.KIND_VERB), set(tc.KIND_EXPIRES_DAYS))
        # window_for anchors a dated item's span at its due date.
        due = '2027-01-01T00:00:00+00:00'
        self.assertGreater(tc.window_for(False, due), due)

    def test_kind_verbs_render(self):
        self._file('decide?', needs_answer=True)
        self._file('fyi', source='s3')
        block, _ = thalamus.pull(self.brain, S1, via='boot')
        self.assertIn('asks', block)
        self.assertIn('notes', block)


class TestMaintenanceSweep(ThalamusBase):

    def test_sweep_fires_without_s2_conditions(self):
        """The channel sweeps run AHEAD of the S2 fire gate inside
        run_maintenance_if_due — a brain whose S2 cycle declines (keyless,
        boot grace, idle gates) still expires its items and dead-letters
        unanswered asks."""
        r = self._file('never answered?', needs_answer=True)
        self._age_out(r['id'])
        result = self.brain.run_maintenance_if_due()
        # S2 itself must have declined (test brain: boot grace / no LLM) —
        # the sweep firing anyway is exactly the point.
        self.assertIsNone(result)
        self.assertEqual(self._row(r['id'])[0], tc.STATE_EXPIRED)

    def test_sweep_is_hourly_throttled(self):
        self.brain.run_maintenance_if_due()
        r = self._file('expires between polls')
        self._age_out(r['id'])
        self.brain.run_maintenance_if_due()  # within the hour — no sweep
        self.assertEqual(self._row(r['id'])[0], tc.STATE_OPEN)
        self.brain._thalamus_sweep_checked = 0  # an hour passes
        self.brain.run_maintenance_if_due()
        self.assertEqual(self._row(r['id'])[0], tc.STATE_EXPIRED)


class TestRefs(ThalamusBase):
    """pull() owns ref resolution — ONE veil-aware batch; the contract only
    formats what it is handed (ref_lines)."""

    def test_refs_resolve_in_one_batched_call(self):
        from unittest import mock
        from tests.test_scopes import _mint
        a = _mint(self.brain, 'sess-a', '', 'first ref title')
        b = _mint(self.brain, 'sess-a', '', 'second ref title')
        self._file('note one', refs=[a, b])
        self._file('note two', refs=[a], source='other')
        with mock.patch.object(self.brain, 'filter_nodes',
                               wraps=self.brain.filter_nodes) as fn:
            block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 2)
        self.assertEqual(fn.call_count, 1)  # batched: one call for the block
        self.assertIn('first ref title', block)
        self.assertIn('second ref title', block)

    def test_walled_ref_renders_bare_not_title(self):
        """A globally-filed item can ref a walled node — its title must not
        print into another session's pull (default-deny outward veil)."""
        from tests.test_scopes import _mint, _set_scopes, ISOLATE_CLIENT_X
        walled = _mint(self.brain, 'sess-client', 'client-x',
                       'client secret plan')
        _set_scopes(self.brain, ISOLATE_CLIENT_X)
        self._file('global note', refs=[walled])
        block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 1)
        self.assertNotIn('client secret plan', block)
        self.assertIn(walled[:8], block)  # the ref itself renders bare

    def test_bad_ref_renders_bare(self):
        self._file('note', refs=['deadbeef'])
        block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 1)
        self.assertIn('deadbeef', block)


class TestRender(ThalamusBase):

    def test_long_body_truncates_loudly(self):
        r = self._file('x' * (tc.BODY_MAX + 500))
        block, _ = thalamus.pull(self.brain, S1, via='boot')
        self.assertIn('chars', block)          # the loud marker, not a bare cut
        self.assertIn('thalamus_list', block)  # points at the full text
        self.assertLess(len(block), tc.BLOCK_MAX + 500)

    def test_cap_dropped_items_not_ledgered(self):
        """Only items the block actually shows are ledgered — a cap-dropped
        item was never delivered and stays armed for the next moment."""
        for i in range(4):
            self._file('x' * 1400, source='src-%d' % i)  # distinct budgets
        _, n1 = thalamus.pull(self.brain, S1, via='boot')
        self.assertLess(n1, 4)  # the block cap dropped at least one
        ledgered = self.brain.logs_conn.execute(
            'SELECT COUNT(*) FROM thalamus_deliveries').fetchone()[0]
        self.assertEqual(ledgered, n1)
        _, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n1 + n2, 4)  # the dropped items deliver next moment

    def test_block_head_counts_shown_items(self):
        """Head, tail, ledger, and pull's count all say `kept` — the head
        must not claim fetched items the cap dropped."""
        for i in range(4):
            self._file('x' * 1400, source='src-%d' % i)
        block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertLess(n, 4)
        self.assertIn('— %d item(s)' % n, block)
        self.assertIn('(+%d more due' % (4 - n), block)

    def test_list_items_shows_delivery_counts(self):
        r = self._file('note')
        thalamus.pull(self.brain, S1, via='boot')
        out = thalamus.list_items(self.brain)
        item = next(i for i in out['items'] if i['id'] == r['id'])
        self.assertEqual(item['deliveries'], 1)


class TestDispatchEnvelope(ThalamusBase):
    """Every handler returns {"ok", ...} — the dispatch_self lesson."""

    def test_remind_enveloped(self):
        r = _handle_remind(self.brain, {'what': 'check X', 'when': '1d'}, [])
        self.assertIs(r.get('ok'), True)
        self.assertIn('result', r)
        self.assertTrue(r['result']['id'].startswith('th_'))

    def test_remind_budget_rejection_is_error_envelope(self):
        for i in range(tc.MAX_OPEN_PER_SOURCE):
            _handle_remind(self.brain, {'what': 'item %d' % i}, [])
        r = _handle_remind(self.brain, {'what': 'over'}, [])
        self.assertIs(r.get('ok'), False)
        self.assertIn('budget', r.get('error', ''))

    def test_list_and_resolve_enveloped(self):
        filed = _handle_remind(self.brain, {'what': 'x'}, [])
        r = _handle_thalamus_list(self.brain, {}, [])
        self.assertIs(r.get('ok'), True)
        self.assertEqual(r['result']['count'], 1)
        rid = filed['result']['id']
        r = _handle_thalamus_resolve(self.brain, {'id': rid, 'dismiss': True}, [])
        self.assertIs(r.get('ok'), True)
        r = _handle_thalamus_resolve(self.brain, {'id': rid, 'dismiss': True}, [])
        self.assertIs(r.get('ok'), False)


if __name__ == '__main__':
    unittest.main()
