"""Stable journal state, partial edits and invocation-based persistence."""
import json
from unittest.mock import patch

from tests.brain_test_base import BrainTestBase
from servers.scales.journal import JournalBinding
from servers.trace_contract import parse_journal_operations, render_journal_view
from servers.channels.thalamus import thalamus


def review(*operations):
    return {'operations': list(operations)}


class JournalTestBase(BrainTestBase):
    needs_embedder = False

    def binding(self, unit='healer', session='', source=''):
        return JournalBinding(self.brain, scale='s1' if session else 's2',
                              unit='' if session else unit, session_id=session, source=source)

    def chain(self, binding, number):
        return ('s1e-%s-%s' % (binding.session_id, number) if binding.session_id
                else 's2-%s-%s' % (number, binding.unit))

    def prepare(self, binding, number=1):
        binding.continuity(chain_id=self.chain(binding, number))
        return list(binding._reference_context['references'].values())

    def create(self, binding, **fields):
        self.prepare(binding, 0)
        binding.apply(review(dict(op='note', subject='deployment', text='Pending',
                                    **fields)), self.chain(binding, 0))
        return self.prepare(binding, 1)[0]


class TestJournalItems(JournalTestBase):
    def test_partial_edit_renames_stable_identity_and_preserves_history(self):
        b = self.binding(); first = self.create(b, persist=True)
        b.apply(review(dict(op='edit', id=first['id'], subject='new subject')),
                  self.chain(b, 1))
        changed = self.prepare(b, 1)[0]
        self.assertEqual(changed['id'], first['id'])
        self.assertEqual(changed['subject'], 'new subject')
        self.assertEqual(changed['text'], 'Pending')
        self.assertTrue(changed['persist'])
        self.assertEqual(changed['runsPersisted'], first['runsPersisted'])
        self.assertEqual(len(self.brain.journal_notes(subject='deployment', scale='s2')), 1)
        self.assertEqual(len(self.brain.journal_notes(subject='new subject', scale='s2')), 1)
        self.assertEqual(len(self.prepare(b, 2)), 1)

    def test_silent_invocations_count_once_and_do_not_write_each_item(self):
        b = self.binding(); first = self.create(b, persist=True)
        for number in range(2, 6):
            for batch in range(3):
                self.prepare(b, number)
                b.apply(review(), self.chain(b, number))
        last = self.prepare(b, 5)[0]
        self.assertEqual(last['runsPersisted'], first['runsPersisted'] + 4)
        self.assertTrue(last['reviewDue'])
        rows = self.brain.journal_notes(subject='deployment', scale='s2')
        self.assertEqual(len(rows), 1)

    def test_persistence_false_does_not_resolve_or_withdraw_live_message(self):
        b = self.binding(source='s2:healer'); first = self.create(b, persist=True)
        b.apply(review(dict(op='ask', subject='deployment', text='Can you check?'),
                         dict(op='edit', id=first['id'], persist=False)), self.chain(b, 1))
        after = self.prepare(b, 2)[0]
        self.assertFalse(after['persist']); self.assertEqual(after['text'], 'Pending')
        self.assertEqual(len(thalamus.producer_items(self.brain, b.source, '')), 1)
        b.apply(review(dict(op='withdraw', subject='deployment', reason='No longer needed')),
                  self.chain(b, 2))
        self.assertEqual(thalamus.producer_items(self.brain, b.source, ''), [])

    def test_legacy_checkpoint_rename_cannot_resurrect_original(self):
        b = self.binding()
        self.brain._trace_dal.append(chain_id=self.chain(b, 'old'), scale='s2',
            event_type='delta', ref_type='journal_note', ref_id='old',
            metadata={'tag': 'open', 'note': 'pending', 'undelivered': ''})
        first = self.prepare(b)[0]
        self.assertEqual(first['runsPersisted'], 1)
        b.apply(review(dict(op='edit', id=first['id'], subject='renamed', text='updated')),
                  self.chain(b, 1))
        for number in (1, 2, 3):
            items = self.prepare(b, number)
            self.assertEqual([(n['subject'], n['text']) for n in items], [('renamed', 'updated')])
        self.assertEqual(len(self.brain.journal_notes(subject='old', scale='s2')), 1)

    def test_persistent_item_survives_200_unrelated_writes(self):
        b = self.binding(); first = self.create(b, persist=True)
        for number in range(2, 7):
            self.prepare(b, number)
            b.apply(review(*(dict(op='note', subject='noise-%d-%d' % (number, i), text='observation')
                               for i in range(50))), self.chain(b, number))
        items = self.prepare(b, 7)
        self.assertTrue(any(n['id'] == first['id'] for n in items))
        self.assertGreater(next(n['runsPersisted'] for n in items if n['id'] == first['id']), 5)

    def test_omitted_items_count_but_cannot_be_edited(self):
        b = self.binding(); first = self.create(b, persist=True)
        with patch('servers.trace_contract.JOURNAL_VIEW_MAX_CHARS', 100):
            self.assertEqual(self.prepare(b, 2), [])
        b.apply(review(dict(op='edit', id=first['id'], text='unseen edit')), self.chain(b, 2))
        item = next(n for n in self.prepare(b, 3) if n['id'] == first['id'])
        self.assertEqual(item['text'], 'Pending')
        self.assertEqual(item['runsPersisted'], first['runsPersisted'] + 2)

    def test_stale_version_and_cross_scope_receipts_are_rejected(self):
        a = self.binding(session='a'); b = self.binding(session='b')
        original = self.create(a, persist=True); self.create(b, persist=True)
        receipt = a._reference_context
        other = self.binding(session='a'); self.prepare(other, 1)
        other.apply(review(dict(op='edit', id=original['id'], text='newer')),
                      self.chain(other, 1))
        a.apply(review(dict(op='edit', id=original['id'], text='stale')), self.chain(a, 1))
        b._reference_context = receipt
        b.apply(review(dict(op='edit', id=original['id'], text='foreign')), self.chain(b, 1))
        item = next(n for n in self.prepare(a, 2) if n['id'] == original['id'])
        self.assertEqual(item['text'], 'newer')
        self.assertFalse(any(n['text'] == 'foreign' for n in self.prepare(b, 2)))

    def test_noop_edit_does_not_create_a_version(self):
        b = self.binding(); original = self.create(b)
        b.apply(review(dict(op='edit', id=original['id'], text='Pending')), self.chain(b, 1))
        after = self.prepare(b, 1)[0]
        self.assertEqual(after['version'], original['version'])
        self.assertFalse(after['persist'])

    def test_repeated_edits_age_out_older_nonpersistent_items(self):
        b = self.binding()
        with patch('servers.trace_contract.JOURNAL_CONTINUITY_RUNS', {'healer': 3}):
            old = self.create(b)
            b.apply(review(dict(op='note', subject='active', text='version 1')),
                      self.chain(b, 1))
            for number in range(2, 6):
                items = self.prepare(b, number)
                active = next(n for n in items if n['subject'] == 'active')
                b.apply(review(dict(op='edit', id=active['id'], text='version %d' % number)),
                          self.chain(b, number))
            final = self.prepare(b, 6)
        self.assertEqual([(n['subject'], n['text']) for n in final], [('active', 'version 5')])
        self.assertNotIn(old['id'], [n['id'] for n in final])
        self.assertEqual(len(self.brain.journal_notes(scale='s2', unit='healer')), 6)

    def test_readonly_inspection_does_not_advance_run_clock(self):
        b = self.binding(); self.create(b, persist=True)
        before = self.brain._trace_dal.journal_item_state(scale='s2', unit='healer')['run_count']
        for _ in range(3):
            self.binding().continuity()
        after = self.brain._trace_dal.journal_item_state(scale='s2', unit='healer')['run_count']
        self.assertEqual(before, after)

    def test_parser_rejects_readonly_fields_and_wrong_types_per_operation(self):
        good, bad = parse_journal_operations([
            dict(op='note', subject='x', text='keep', persist=False),
            dict(op='edit', id='journal_a1b2c3d4', runsPersisted=8),
            dict(op='edit', id='journal_a1b2c3d4', persist='false'),
            dict(op='note', subject='x', text='wrong', reviewDue=True)])
        self.assertEqual(len(good), 1); self.assertEqual(len(bad), 3)

    def test_json_read_cap_preserves_whole_items(self):
        b = self.binding(); self.create(b, persist=True)
        state = b._view
        state = dict(state, notes=[dict(state['notes'][0], id='journal_%08x' % i,
                                       text='x' * 600) for i in range(30)])
        text, stats, refs = render_journal_view(state)
        parsed = json.loads(text.split('\n', 1)[1])
        self.assertLessEqual(len(text), 8000)
        self.assertGreater(parsed['omitted'], 0)
        self.assertEqual(len(parsed['items']), len(refs))
        self.assertTrue(all(len(n['text']) == 600 for n in parsed['items']))
