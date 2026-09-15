"""Only shown, current journal versions authorize edits; failures stay isolated."""
from unittest.mock import patch
from tests.test_journal_items import JournalTestBase, review


class TestJournalReferenceFailures(JournalTestBase):
    # Reuse only helpers, not an alternate implementation of the journal.
    def test_write_failure_does_not_publish_edit_or_withdraw(self):
        b = self.binding(source='s2:healer'); first = self.create(b, persist=True)
        b.apply(review(dict(op='ask', subject='deployment', text='Check deployment?')), self.chain(b, 1))
        self.prepare(b, 1)
        with patch.object(self.brain._trace_dal, 'append_batch', side_effect=RuntimeError('disk offline')):
            b.apply(review(dict(op='edit', id=first['id'], persist=False)), self.chain(b, 1))
        after = next(n for n in self.prepare(b, 2) if n['id'] == first['id'])
        self.assertTrue(after['persist'])
        from servers.channels.thalamus import thalamus
        self.assertEqual(len(thalamus.producer_items(self.brain, b.source, '')), 1)

    def test_toggle_multiple_times_counts_invocation_once(self):
        b = self.binding(); first = self.create(b, persist=True)
        b.apply(review(dict(op='edit', id=first['id'], persist=False),
                         dict(op='edit', id=first['id'], persist=True)), self.chain(b, 1))
        after = self.prepare(b, 1)[0]
        self.assertEqual(after['runsPersisted'], first['runsPersisted'])
        b.apply(review(dict(op='edit', id=first['id'], persist=False)), self.chain(b, 1))
        self.prepare(b, 1)
        b.apply(review(dict(op='edit', id=first['id'], persist=True)), self.chain(b, 1))
        self.assertEqual(self.prepare(b, 1)[0]['runsPersisted'], first['runsPersisted'])

    def test_reference_receipt_is_consumed(self):
        b = self.binding(); first = self.create(b, persist=True)
        b.apply(review(), self.chain(b, 1))
        b.apply(review(dict(op='edit', id=first['id'], text='second response')), self.chain(b, 1))
        self.assertEqual(next(n['text'] for n in self.prepare(b, 2) if n['id'] == first['id']), 'Pending')

    def test_initial_read_failure_keeps_selection_frozen_until_next_run(self):
        b = self.binding()
        with patch.object(self.brain, 'journal_view', side_effect=RuntimeError('read unavailable')):
            self.assertEqual(self.prepare(b, 1), [])
        b.apply(review(dict(op='note', subject='new observation', text='recorded during failure')),
                  self.chain(b, 1))
        self.assertEqual(self.prepare(b, 1), [])
        self.assertEqual([n['subject'] for n in self.prepare(b, 2)], ['new observation'])

    def test_explicit_withdraw_subject_is_authoritative(self):
        from servers.channels.thalamus import thalamus
        b = self.binding(source='s2:healer')
        self.prepare(b)
        b.apply(review(dict(op='ask', subject='first', text='First question?'),
                         dict(op='ask', subject='second', text='Second question?')),
                  self.chain(b, 1))
        b.apply(review(dict(op='withdraw', subject='first', reason='second')),
                  self.chain(b, 1))
        self.assertEqual([n['dedup_key'] for n in thalamus.producer_items(
            self.brain, b.source, '')], ['second'])

    def test_private_read_failure_preserves_live_operations(self):
        from servers.channels.thalamus import thalamus
        b = self.binding(source='s2:healer')
        first = self.create(b, persist=True)
        with patch.object(self.brain._trace_dal, 'journal_item_state',
                          side_effect=RuntimeError('journal read unavailable')):
            b.apply(review(dict(op='edit', id=first['id'], text='cannot validate'),
                             dict(op='ask', subject='independent', text='Please check?')),
                      self.chain(b, 1))
        self.assertEqual([n['dedup_key'] for n in thalamus.producer_items(
            self.brain, b.source, '')], ['independent'])
        self.assertEqual(self.prepare(b, 2)[0]['text'], 'Pending')

    def test_failed_filing_feedback_does_not_prevent_withdrawal(self):
        from servers.channels.thalamus import thalamus
        b = self.binding(source='s2:healer')
        self.prepare(b)
        b.apply(review(dict(op='ask', subject='existing', text='Please check?')), self.chain(b, 1))
        with patch.object(thalamus, 'file', return_value={'ok': False, 'error': 'rejected'}), \
                patch.object(self.brain, 'write_journal_note_rows', side_effect=RuntimeError('disk offline')):
            b.apply(review(dict(op='ask', subject='new', text='New question?'),
                             dict(op='withdraw', subject='existing', reason='No longer needed')),
                      self.chain(b, 1))
        self.assertEqual(thalamus.producer_items(self.brain, b.source, ''), [])

    def test_concurrent_edit_rejection_preserves_independent_addition(self):
        from servers.dal_logs import JournalWriteConflict
        b = self.binding()
        first = self.create(b, persist=True)
        append = self.brain._trace_dal.append_batch

        def conflict(events, **kwargs):
            if kwargs.get('journal_guard'):
                raise JournalWriteConflict('intervening write')
            return append(events, **kwargs)

        with patch.object(self.brain._trace_dal, 'append_batch', side_effect=conflict):
            b.apply(review(dict(op='edit', id=first['id'], text='stale edit'),
                             dict(op='note', subject='independent', text='keep this')),
                      self.chain(b, 1))
        items = self.prepare(b, 2)
        self.assertEqual(next(n['text'] for n in items if n['id'] == first['id']), 'Pending')
        self.assertTrue(any(n['subject'] == 'independent' for n in items))
