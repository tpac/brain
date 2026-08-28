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
from servers.scales.thalamus import thalamus, thalamus_contract as tc
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


class TestFile(ThalamusBase):

    def test_notice_defaults(self):
        r = self._file()
        self.assertTrue(r['filed'])
        self.assertEqual(r['route'], 'queue')
        state, audience, needs_answer, deliver_at, expires_at, _ = self._row(r['id'])
        self.assertEqual(state, tc.STATE_OPEN)
        self.assertEqual(audience, tc.AUDIENCE_ONCE)
        self.assertEqual(needs_answer, 0)
        self.assertIsNone(deliver_at)
        self.assertGreater(expires_at, iso_now())

    def test_ask_defaults_to_all_audience(self):
        r = self._file('should X be configurable?', needs_answer=True)
        _, audience, needs_answer, _, expires_at, _ = self._row(r['id'])
        self.assertEqual(audience, tc.AUDIENCE_ALL)
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
        self.assertEqual(audience, tc.AUDIENCE_ONCE)
        # An 8-char short is a display convention, not a key.
        r2 = self._file(for_whom=S1[:8])
        self.assertFalse(r2['filed'])

    def test_empty_body_rejects(self):
        self.assertFalse(self._file('  ')['filed'])

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
                " VALUES ('th_dupdup', 'test', 'second', '', 'once', '', 0,"
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


class TestPull(ThalamusBase):

    def test_notice_delivers_once_across_sessions(self):
        r = self._file('heads up')
        block, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 1)
        self.assertIn(r['id'], block)
        # audience 'once': a second session gets nothing.
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
        # Ledger cleared, but not due yet → still nothing.
        _, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 0)
        deliver_at = self._row(r['id'])[3]
        self.assertGreater(deliver_at, iso_now())
        ledger = self.brain.logs_conn.execute(
            'SELECT COUNT(*) FROM thalamus_deliveries WHERE item_id = ?',
            (r['id'],)).fetchone()[0]
        self.assertEqual(ledger, 0)

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

    def _age_out(self, item_id):
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_items SET expires_at = ? WHERE id = ?',
                (iso_cutoff(hours=1), item_id))
            self.brain.logs_conn_w.commit()

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


class TestRender(ThalamusBase):

    def test_long_body_truncates_loudly(self):
        r = self._file('x' * (tc.BODY_MAX + 500))
        block, _ = thalamus.pull(self.brain, S1, via='boot')
        self.assertIn('chars', block)          # the loud marker, not a bare cut
        self.assertIn('thalamus_list', block)  # points at the full text
        self.assertLess(len(block), tc.BLOCK_MAX + 500)

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
