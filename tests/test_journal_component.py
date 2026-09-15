"""Component tests for JournalBinding (servers/scales/journal.py) — the one
object that attaches the journal to any agent request.

Pins the Phase 2 contract: decorate order (arc → review → closure-if-loop),
harvest as write+strip (the single-shot envelope rule: journal sections are
stripped BEFORE the caller's JSON extraction, so a `]`/`}` inside a fence
can't corrupt extract_json's rfind-based scan), and failure isolation on the
continuity read.
"""
import sys
import os
import unittest
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase  # noqa: E402
from servers.scales.journal import JournalBinding  # noqa: E402
from servers.scales.runner import extract_json  # noqa: E402


FINAL_TEXT = (
    (
        {'operations': [{'op': 'note', 'subject': 'abc', 'text': 'score capped at 0.9]', 'label': 'friction'}]}
    )
)


class TestHarvest(BrainTestBase):
    needs_embedder = False

    def _binding(self, **kw):
        kw.setdefault('scale', 's2')
        kw.setdefault('unit', 'consolidation')
        return JournalBinding(self.brain, **kw)

    def test_harvest_writes_notes_and_strips(self):
        b = self._binding()
        remainder = b.apply(FINAL_TEXT, 's2-20260728000000-consolidation')

        notes = self.brain.journal_notes(scale='s2', unit='consolidation')
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]['tag'], 'friction')
        self.assertEqual(notes[0]['subject'], 'abc')
        self.assertIsInstance(remainder, dict)


    def test_arc_bound_harvest_writes_session_arc(self):
        b = JournalBinding(self.brain, scale='s1', session_id='sess-arc', arc=True)
        b.harvest_arc('## Arc\n```\nsurvivor ladder shipped\n```')
        self.assertIn('survivor ladder shipped', self.brain.session_context_for('sess-arc'))

    def test_continuity_failure_isolated(self):
        """A broken notes read degrades to no continuity, never raises."""
        b = self._binding()
        orig = self.brain.journal_view
        self.brain.journal_view = None       # any call → TypeError
        try:
            import json
            view = json.loads(b.continuity().split('\n', 1)[1])
            self.assertEqual(view['items'], [])
            self.assertEqual(view['coverage'], {'stale': True, 'selection_failed': True})
        finally:
            self.brain.journal_view = orig

    def test_continuity_round_trip(self):
        b = self._binding(unit='community_detection')
        b.apply((
            {'operations': [{'op': 'note', 'subject': 'xyz', 'text': 'unsure about placement', 'label': 'doubt'}]}
        ),
                  's2-20260728000002-community_detection')
        prefix = b.continuity()
        self.assertIn('xyz', prefix)
        self.assertIn('JOURNAL — existing state', prefix)


S1 = 'aaaaaaaa-1111-2222-3333-444444444444'
S2 = 'bbbbbbbb-1111-2222-3333-444444444444'
CHAIN = 's1e-aaaaaaaa-3'
ADDRESSED_TEXT = (
    (
        {'operations': [{'op': 'note', 'subject': '1ca943af', 'text': 'turn-row projection built in two places', 'label': 'friction'}, {'op': 'tell', 'subject': 'segment 6.a', 'text': 'you are proceeding on "I wonder if", not a yes'}, {'op': 'ask', 'subject': '7e6decd2', 'text': 'milestone says merge pending; it merged — revise, or leave?'}]}
    )
)


class TestAddressedRouting(BrainTestBase):
    """Step 13(b): a binding with a source files its tell/ask lines as
    Thalamus items (the non-LLM entrance to the one door) — never as
    journal_note rows; a session makes them directed, none makes them
    broadcast. Without a source the lines stay plain notes and warn. The
    filing rides the run chain (13d)."""
    needs_embedder = False

    def _scribe(self, source='encoder:sonnet', session_id=S1):
        return JournalBinding(self.brain, scale='s1', session_id=session_id,
                              source=source)

    def _items(self, **kw):
        from servers.channels.thalamus import thalamus
        kw.setdefault('source', 'encoder:sonnet')
        kw.setdefault('target_session', S1)
        return thalamus.list_items(self.brain, **kw)['items']

    def _warnings(self, source):
        return self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type='warning' "
            "AND source = ?", (source,)).fetchone()[0]

    def test_tell_and_ask_become_items_not_notes(self):
        self._scribe().apply(ADDRESSED_TEXT, CHAIN)
        notes = self.brain.journal_notes(scale='s1', session_id=S1)
        self.assertEqual([n['tag'] for n in notes], ['friction'])
        items = {i['body']: i for i in self._items()}
        self.assertEqual(len(items), 2)
        tell = next(i for b, i in items.items() if b.startswith('you are'))
        ask = next(i for b, i in items.items() if b.startswith('milestone'))
        # The subject is always the dedup key ("one line per subject" holds
        # literally); only a node-id subject is also a ref.
        self.assertEqual((tell['needs_answer'], tell['dedup_key'], tell['refs']),
                         (False, 'segment 6.a', []))
        self.assertEqual((ask['needs_answer'], ask['dedup_key'], ask['refs']),
                         (True, '7e6decd2', ['7e6decd2']))
        self.assertTrue(all(i['target_session'] == S1 for i in items.values()))
        filed = [e for e in self.brain.query_traces(chain_id=CHAIN)['chain']
                 if e['ref_type'] == 'thalamus_filed']
        self.assertEqual(sorted(e['ref_id'] for e in filed),
                         sorted(i['id'] for i in items.values()))

    def test_directed_ask_reaches_the_session_at_stop(self):
        from servers.channels.thalamus import thalamus
        self._scribe().apply(ADDRESSED_TEXT, CHAIN)
        block, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 2)
        self.assertIn('encoder:sonnet asks', block)
        self.assertIn('encoder:sonnet notes', block)

    def test_no_source_keeps_them_as_notes_and_warns(self):
        # No source (a bare eval binding) → plain residue, one loud warning.
        JournalBinding(self.brain, scale='s1', session_id=S1).apply(
            ADDRESSED_TEXT, CHAIN)
        notes = self.brain.journal_notes(scale='s1', session_id=S1)
        self.assertEqual(sorted(n['tag'] for n in notes),
                         ['ask', 'friction', 'tell'])
        self.assertEqual(self._items(), [])
        b = JournalBinding(self.brain, scale='s1', session_id=S1)
        self.assertEqual(self._warnings(b._log_key('addressed_unbound')), 1)

    def test_s2_unit_files_broadcast_delivered_at_boot(self):
        """An S2 binding has a source and no session: its ask is broadcast,
        due at every session's boot (not Stop); its tell reaches the first
        session to boot. Same instruction, same door, other audience."""
        from servers.channels.thalamus import thalamus
        JournalBinding(self.brain, scale='s2', unit='consolidation',
                       source='s2:consolidation').apply(
            ADDRESSED_TEXT, 's2-20260905120000-consolidation')
        items = self._items(source='s2:consolidation', target_session='')
        self.assertEqual(len(items), 2)
        self.assertTrue(all(i['target_session'] == '' for i in items))
        _, n_stop = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n_stop, 1)          # the tell (a notice) only
        _, n_boot = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n_boot, 1)          # the ask
        _, n_other = thalamus.pull(self.brain, S2, via='boot')
        self.assertEqual(n_other, 1)         # every session sees the ask once
        filed = [e for e in self.brain.query_traces(
            chain_id='s2-20260905120000-consolidation')['chain']
                 if e['ref_type'] == 'thalamus_filed']
        self.assertEqual(len(filed), 2)
        self.assertTrue(all(e['scale'] == 's2' for e in filed))

    def test_rejection_is_loud_and_kept_as_residue(self):
        from servers.channels.thalamus import thalamus
        from servers.channels.thalamus import thalamus_contract as tc
        for i in range(tc.MAX_OPEN_PER_SOURCE):
            thalamus.file(self.brain, 'encoder:sonnet', 'fill %d' % i,
                          for_whom=S1)
        b = self._scribe()
        b.apply(ADDRESSED_TEXT, CHAIN)
        self.assertEqual(len(self._items()), tc.MAX_OPEN_PER_SOURCE)
        self.assertEqual(self._warnings(b._log_key('addressed_rejected')), 2)
        notes = self.brain.journal_notes(scale='s1', session_id=S1)
        kept = [n for n in notes if n['undelivered']]
        self.assertEqual(sorted(n['tag'] for n in kept), ['ask', 'tell'])
        self.assertTrue(all('budget' in n['undelivered'] for n in kept))
        # The note is the line the encoder wrote — the reason is a field.
        self.assertIn('you are proceeding on "I wonder if", not a yes',
                      [n['note'] for n in kept])

    def test_explicit_withdraw_closes_the_named_messages(self):
        """Explicit subjects use the producer's normalized message keys."""
        self._scribe().apply(ADDRESSED_TEXT, CHAIN)
        self.assertEqual(len(self._items()), 2)
        self._scribe().apply(
            (
                {'operations': [{'op': 'withdraw', 'subject': '7e6decd2', 'reason': 'the operator said revise'}, {'op': 'withdraw', 'subject': 'Segment 6.A', 'reason': 'confirmed'}]}
            ), 's1e-aaaaaaaa-4')
        self.assertEqual(self._items(), [])
        states = {i['dedup_key']: i['state']
                  for i in self._items(include_closed=True)}
        self.assertEqual(states, {'7e6decd2': 'withdrawn',
                                  'segment 6.a': 'withdrawn'})

    def test_routing_failure_is_isolated_and_the_arc_survives(self):
        """A Thalamus read hiccup during routing must neither escape harvest
        (S2's batch loop relies on it) nor cost the session arc."""
        from unittest import mock
        from servers.channels.thalamus import thalamus
        b = JournalBinding(self.brain, scale='s1', session_id=S1,
                           source='encoder:sonnet', arc=True)
        text = ADDRESSED_TEXT
        with mock.patch.object(thalamus, 'file',
                               side_effect=RuntimeError('database is locked')):
            remainder = b.apply(text, CHAIN)
        self.assertIsInstance(remainder, dict)
        b.harvest_arc('## Arc\n```\nrouting survived\n```')
        self.assertIn('routing survived',
                      self.brain.session_context_for(S1) or '')
        # Every filing failed loudly and was kept as residue.
        b2 = self._scribe()
        self.assertEqual(self._warnings(b2._log_key('addressed_rejected')), 2)
        kept = [n for n in self.brain.journal_notes(scale='s1', session_id=S1)
                if n['undelivered']]
        self.assertEqual(len(kept), 2)

    def test_same_subject_in_two_sessions_is_two_items(self):
        """The Scribe's source is one string for every session, so the
        identity a re-file matches on carries the target: session B's ask
        under the same subject is B's item, not a retarget of A's."""
        self._scribe().apply(ADDRESSED_TEXT, CHAIN)
        JournalBinding(self.brain, scale='s1', session_id=S2,
                       source='encoder:sonnet').apply(
            ADDRESSED_TEXT, 's1e-bbbbbbbb-3')
        a = {i['dedup_key']: i for i in self._items(target_session=S1)}
        bb = {i['dedup_key']: i for i in self._items(target_session=S2)}
        self.assertEqual(set(a), set(bb))
        self.assertNotEqual(a['7e6decd2']['id'], bb['7e6decd2']['id'])
        # B's resolve closes B's item only.
        JournalBinding(self.brain, scale='s1', session_id=S2,
                       source='encoder:sonnet').apply(
            {'operations': [{'op': 'withdraw', 'subject': '7e6decd2', 'reason': 'done'}]},
            's1e-bbbbbbbb-4')
        self.assertIn('7e6decd2', {i['dedup_key'] for i in
                                   self._items(target_session=S1)})
        self.assertNotIn('7e6decd2', {i['dedup_key'] for i in
                                      self._items(target_session=S2)})

    def test_dark_ship_no_addressed_lines_no_items(self):
        """Without the prompt paragraph no encoder writes the verbs — a
        review block with none files nothing and traces nothing."""
        self._scribe().apply(
            (
                {'operations': [{'op': 'note', 'subject': '1ca943af', 'text': 'drift risk', 'label': 'friction'}]}
            ),
            CHAIN)
        self.assertEqual(self._items(), [])
        self.assertEqual(
            [e for e in self.brain.query_traces(chain_id=CHAIN)['chain']
             if e['ref_type'] == 'thalamus_filed'], [])


class TestProducerView(BrainTestBase):
    """Step 13(e): continuity() on a source-bound binding renders the
    encoder's own tell/ask items with their SETTLED outcome — answered
    (text) / dismissed / expired — plus every open one as a bare 'open'.
    Never delivery counts, moments or dates; withdrawn items don't render;
    settled items fall out after PRODUCER_VIEW_SETTLED_DAYS."""
    needs_embedder = False

    SRC = 'encoder:sonnet'

    def _rows(self, view):
        return re.findall(r'^- .*?(?=^- |\Z)', view, re.M | re.S)

    def _file(self, body, **kw):
        from servers.channels.thalamus import thalamus
        kw.setdefault('for_whom', S1)
        kw.setdefault('session_id', S1)
        r = thalamus.file(self.brain, self.SRC, body, **kw)
        self.assertTrue(r['ok'], r)
        return r['id']

    def _view(self, **kw):
        kw.setdefault('scale', 's1')
        kw.setdefault('session_id', S1)
        kw.setdefault('source', self.SRC)
        return JournalBinding(self.brain, **kw).continuity()

    def _backdate(self, item_id, days):
        """Age an item's clocks by `days` (test fixture — the door never
        writes the past)."""
        from servers.clock import iso_cutoff
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_items SET created_at = ?, updated_at = ? '
                'WHERE id = ?', (iso_cutoff(days=days), iso_cutoff(days=days),
                                 item_id))
            self.brain.logs_conn_w.commit()

    def test_fates_render_minimal(self):
        from servers.channels.thalamus import thalamus
        from servers.clock import iso_cutoff
        answered = self._file('milestone says pending; it merged — revise?',
                              needs_answer=True, dedup_key='7e6decd2')
        thalamus.resolve(self.brain, answered, answer='revise it')
        dismissed = self._file('configurable, or fixed?', needs_answer=True,
                               dedup_key='suppression-class')
        thalamus.resolve(self.brain, dismissed, dismiss=True)
        expired = self._file('is this a false cluster?', needs_answer=True,
                             dedup_key='40c1a6b4')
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_items SET expires_at = ? WHERE id = ?',
                (iso_cutoff(hours=1), expired))
            self.brain.logs_conn_w.commit()
        thalamus.expire_due(self.brain)
        self._file('you are proceeding on "I wonder if", not a yes',
                   dedup_key='segment 6.a')
        withdrawn = self._file('never mind', dedup_key='nm')
        thalamus.withdraw(self.brain, self.SRC, dedup_key='nm',
                          target_session=S1)
        view = self._view()
        self.assertIn('YOUR MESSAGES', view)
        for line in (
            '- ask [7e6decd2]\n  Message: milestone says pending; it merged — revise?\n  Status: answered: revise it',
            '- ask [suppression-class]\n  Message: configurable, or fixed?\n  Status: dismissed',
            '- ask [40c1a6b4]\n  Message: is this a false cluster?\n  Status: expired, unanswered',
            '- tell [segment 6.a]\n  Message: you are proceeding on "I wonder if", not a yes\n  Status: open',
        ):
            self.assertIn(line, view)
        self.assertNotIn('never mind', view)
        # Nothing about delivery: no counts, no moments, no dates.
        for word in ('delivered', 'boot', 'stop', '×', '2026-'):
            self.assertNotIn(word, view)

    def test_settled_window_and_open_pin(self):
        from servers.channels.thalamus import thalamus
        from servers.channels.thalamus import thalamus_contract as tc
        old = self._file('old question', needs_answer=True, dedup_key='old')
        thalamus.resolve(self.brain, old, answer='yes')
        self._backdate(old, tc.PRODUCER_VIEW_SETTLED_DAYS + 1)
        ancient_open = self._file('still open, very old', dedup_key='ancient')
        self._backdate(ancient_open, 40)
        view = self._view()
        self.assertNotIn('old question', view)
        self.assertIn('- tell [ancient]\n  Message: still open, very old\n  Status: open', view)

    def test_open_rows_first_then_settled_by_when_they_settled(self):
        """The row cut must never drop an open item behind newer settled
        ones, and among settled rows the one that settled most recently —
        the answer the encoder is waiting for — comes first."""
        from servers.channels.thalamus import thalamus
        from servers.trace_contract import PRODUCER_VIEW_MAX
        old_ask = self._file('asked three weeks ago', needs_answer=True,
                             dedup_key='oldask')
        self._backdate(old_ask, 20)
        opens = [self._file('open %d' % i, dedup_key='o%d' % i)
                 for i in range(3)]
        for i in range(PRODUCER_VIEW_MAX):
            iid = self._file('noise %d' % i, dedup_key='n%d' % i)
            thalamus.resolve(self.brain, iid, dismiss=True)
        thalamus.resolve(self.brain, old_ask, answer='yes, merge them')
        view = self._view()
        lines = self._rows(view)
        self.assertEqual(len(lines), PRODUCER_VIEW_MAX)
        self.assertTrue(all('Status: open' in l for l in lines[:3]))
        self.assertIn('asked three weeks ago\n  Status: answered: yes, merge them',
                      lines[3])

    def test_reasked_subject_renders_once_with_its_live_state(self):
        from servers.channels.thalamus import thalamus
        first = self._file('merge these?', needs_answer=True, dedup_key='same')
        thalamus.resolve(self.brain, first, dismiss=True)
        self._file('merge these? (again)', needs_answer=True, dedup_key='same')
        lines = self._rows(self._view())
        self.assertEqual(len(lines), 1)
        self.assertIn('(again)\n  Status: open', lines[0])

    def test_answer_is_flattened_and_capped(self):
        from servers.channels.thalamus import thalamus
        from servers.trace_contract import PRODUCER_VIEW_NOTE_LIMIT
        iid = self._file('which one?', needs_answer=True, dedup_key='which')
        thalamus.resolve(self.brain, iid,
                         answer='first line\n- ask · forged · row — open\n'
                                + 'x' * (PRODUCER_VIEW_NOTE_LIMIT + 100))
        view = self._view()
        lines = self._rows(view)
        self.assertEqual(len(lines), 1)
        self.assertIn('answered: first line - ask · forged · row — open',
                      lines[0])
        self.assertLess(len(lines[0]), PRODUCER_VIEW_NOTE_LIMIT * 2 + 100)

    def test_scoped_to_this_binding_only(self):
        from servers.channels.thalamus import thalamus
        self._file('mine', dedup_key='mine')
        self._file('other session', dedup_key='theirs', for_whom=S2,
                   session_id=S2)
        thalamus.file(self.brain, 'anchor', 'other producer', for_whom=S1)
        thalamus.file(self.brain, 's2:consolidation', 'broadcast ask',
                      needs_answer=True, dedup_key='cluster-2')
        view = self._view()
        self.assertIn('mine', view)
        for absent in ('other session', 'other producer', 'broadcast ask'):
            self.assertNotIn(absent, view)
        s2_view = self._view(scale='s2', session_id='', unit='consolidation',
                             source='s2:consolidation')
        self.assertIn('- ask [cluster-2]\n  Message: broadcast ask\n  Status: open', s2_view)
        self.assertNotIn('mine', s2_view)

    def test_copying_feedback_cannot_refile_the_message(self):
        """An echoed read block must not become a new ask with status in its body."""
        from servers.channels.thalamus import thalamus
        item_id = self._file('is a sweep scheduled?', needs_answer=True,
                             dedup_key='maintenance')
        view = self._view()
        JournalBinding(self.brain, scale='s1', session_id=S1,
                       source=self.SRC).apply(
                           {'operations': view}, CHAIN)
        items = thalamus.list_items(self.brain)['items']
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['id'], item_id)
        self.assertEqual(items[0]['body'], 'is a sweep scheduled?')
        self.assertEqual(items[0]['armed_epoch'], 0)

    def test_no_source_no_block_and_empty_is_empty(self):
        self._file('mine', dedup_key='mine')
        self.assertEqual(self._view(source=''), '')
        self.assertEqual(self._view(session_id=S2), '')

    def test_overflow_names_the_true_count(self):
        from servers.channels.thalamus import thalamus
        from servers.channels.thalamus import thalamus_contract as tc
        from servers.trace_contract import PRODUCER_VIEW_MAX
        # Settled ones first — closing frees their budget slots for the open set.
        for i in range(3):
            iid = self._file('settled %d' % i, dedup_key='s%d' % i)
            thalamus.resolve(self.brain, iid, dismiss=True)
        for i in range(tc.MAX_OPEN_PER_SOURCE):
            self._file('open %d' % i, dedup_key='o%d' % i)
        view = self._view()
        total = tc.MAX_OPEN_PER_SOURCE + 3
        self.assertEqual(view.count('\n- '), PRODUCER_VIEW_MAX)
        self.assertIn('(+%d older, not shown)' % (total - PRODUCER_VIEW_MAX),
                      view)

    def test_view_follows_the_notes(self):
        self._file('mine', dedup_key='mine')
        b = JournalBinding(self.brain, scale='s1', session_id=S1,
                           source=self.SRC)
        b.apply((
            {'operations': [{'op': 'note', 'subject': 'abc12345', 'text': 'drift', 'label': 'friction'}]}
        ), CHAIN)
        view = b.continuity()
        self.assertLess(view.index('JOURNAL — existing state'),
                        view.index('YOUR MESSAGES'))
