"""Stored legacy journal adoption, current JSON display, and boot items."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class JournalLifecycleBase(BrainTestBase):
    needs_embedder = False

    UNIT = 'consolidation'

    def _write_run(self, n, *lines):
        """Write one run's notes under a distinct chain_id."""
        # Historical traces are migration input, not a supported output protocol.
        from servers.trace_contract import build_journal_note_metadata
        for line in lines:
            tag, subject, text = line.split('·', 2)
            self.brain._trace_dal.append(
                chain_id='s2-2026072800%04d-%s' % (n, self.UNIT), scale='s2',
                event_type='delta', ref_type='journal_note', ref_id=subject.strip(),
                metadata=build_journal_note_metadata(note=text, tag=tag))

    def _continuity(self, k=None):
        from servers.trace_contract import JOURNAL_CONTINUITY_RUNS
        with mock.patch.dict(JOURNAL_CONTINUITY_RUNS, {self.UNIT: k or 3}):
            notes = self.brain.journal_view(scale='s2', unit=self.UNIT,
                invocation='s2-adoption-' + self.UNIT)['notes']
        return [dict(n, tag=n['label'], note=n['text']) for n in notes]

    def _subjects(self, notes):
        return [n['subject'] for n in notes]


class TestResolveVerb(JournalLifecycleBase):

    def test_resolved_drops_older_same_subject(self):
        self._write_run(1, 'doubt · pin-42 · might be wrong')
        self._write_run(2, 'resolved · pin-42 · verified correct')

        notes = self._continuity(k=5)
        tags = [(n['tag'], n['subject']) for n in notes]
        self.assertIn(('resolved', 'pin-42'), tags)
        self.assertNotIn(('doubt', 'pin-42'), tags)

    def test_matching_is_normalized(self):
        self._write_run(1, 'doubt · Pin-42 · might be wrong')
        self._write_run(2, 'resolved ·   pin-42  · verified')

        notes = self._continuity(k=5)
        self.assertNotIn('Pin-42', self._subjects(notes))

    def test_other_subjects_untouched(self):
        self._write_run(1, 'doubt · pin-42 · might be wrong')
        self._write_run(2, 'resolved · other-thing · done')

        notes = self._continuity(k=5)
        self.assertIn('pin-42', self._subjects(notes))

    def test_resolve_does_not_reach_newer_notes(self):
        self._write_run(1, 'resolved · pin-42 · done')
        self._write_run(2, 'doubt · pin-42 · new doubt, after the resolve')

        notes = self._continuity(k=5)
        tags = [(n['tag'], n['subject']) for n in notes]
        self.assertIn(('doubt', 'pin-42'), tags)

    def test_hotspot_view_unfiltered(self):
        self._write_run(1, 'doubt · pin-42 · might be wrong')
        self._write_run(2, 'resolved · pin-42 · verified correct')

        history = self.brain.journal_notes(subject='pin-42', scale='s2',
                                           unit=self.UNIT)
        self.assertEqual(len(history), 2)


class TestOpenPins(JournalLifecycleBase):

    def test_open_survives_beyond_k_window(self):
        self._write_run(1, 'open · repo-question · still undecided')
        for i in range(2, 6):
            self._write_run(i, 'friction · run-%d · unrelated note' % i)

        notes = self._continuity(k=2)   # window holds runs 5,4 only
        opens = [n for n in notes if n['subject'] == 'repo-question']
        self.assertEqual(len(opens), 1)
        self.assertTrue(opens[0].get('persist'))

    def test_resolved_unpins(self):
        self._write_run(1, 'open · repo-question · still undecided')
        self._write_run(2, 'resolved · repo-question · promoted to abc12345')
        for i in range(3, 7):
            self._write_run(i, 'friction · run-%d · unrelated' % i)

        notes = self._continuity(k=2)
        self.assertNotIn('repo-question', self._subjects(notes))

    def test_repeated_legacy_mentions_do_not_invent_elapsed_invocations(self):
        # Adoption collapses repeated mentions and starts a truthful new run clock.
        for i in (1, 2, 3):
            self._write_run(i, 'open · repo-question · still undecided')
        self._write_run(4, 'friction · other · noise')

        notes = self._continuity(k=2)
        opens = [n for n in notes if n['subject'] == 'repo-question']
        self.assertEqual(len(opens), 1)          # deduped to newest
        self.assertEqual(opens[0]['runsPersisted'], 1)

    def test_still_open_alias_pins(self):
        self._write_run(1, 'still-open · legacy-item · from the wild corpus')
        for i in range(2, 6):
            self._write_run(i, 'friction · run-%d · unrelated' % i)

        notes = self._continuity(k=2)
        self.assertIn('legacy-item', self._subjects(notes))

    def test_pin_cap_bounds_carryover(self):
        # Exact-count both sides: the cap is hit (not zero pins vacuously),
        # and never exceeded. All open runs are pushed outside the k=2 window
        # by the trailing friction runs, so every pin is carry-over.
        from servers.trace_contract import JOURNAL_PERSIST_PIN_CAP
        for i in range(JOURNAL_PERSIST_PIN_CAP + 4):
            self._write_run(i + 1, 'open · item-%02d · lingering' % i)
        for j in range(30, 33):
            self._write_run(j, 'friction · run-%d · unrelated' % j)

        notes = self._continuity(k=2)
        pinned = [n for n in notes if n.get('persist')]
        self.assertEqual(len(pinned), JOURNAL_PERSIST_PIN_CAP)

    def test_reopen_after_resolve_starts_fresh_epoch(self):
        # A resolve closes the epoch: a re-opened subject counts ×1 with a
        # fresh first_seen — runs retired by the resolution don't bleed into
        # the new count (review finding 3).
        self._write_run(1, 'open · repo-question · first epoch')
        self._write_run(2, 'open · repo-question · first epoch again')
        self._write_run(3, 'resolved · repo-question · settled for now')
        self._write_run(4, 'open · repo-question · re-opened, new grounds')
        self._write_run(5, 'friction · other · noise')

        notes = self._continuity(k=2)
        opens = [n for n in notes if n['subject'] == 'repo-question'
                 and n.get('persist')]
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0]['runsPersisted'], 1)
        self.assertEqual(opens[0]['note'], 're-opened, new grounds')


class TestRenderLifecycle(JournalLifecycleBase):

    def test_render_shows_runtime_count_and_review_due(self):
        import json
        from servers.trace_contract import render_journal_view
        for count, due in ((2, False), (5, True)):
            with self.subTest(count=count):
                note = dict(id='journal_a1b2c3d4', subject='repo-question',
                            text='undecided', persist=True,
                            runsPersisted=count, reviewDue=due)
                text, stats, references = render_journal_view({'notes': [note]})
                shown = json.loads(text.split('\n', 1)[1])['items'][0]
                self.assertEqual(shown, note)
                self.assertEqual(stats['rendered_rows'], 1)
                self.assertNotIn('hand it up', text)

    def test_tool_schema_owns_all_operation_shapes(self):
        from servers.trace_contract import journal_tool_schema
        schema = journal_tool_schema()
        variants = schema['input_schema']['properties']['operations']['items']['oneOf']
        self.assertEqual({v['properties']['op']['enum'][0] for v in variants},
                         {'note', 'edit', 'tell', 'ask', 'withdraw'})
        self.assertIn('persistence', schema['description'])



class TestBootStandingItems(BrainTestBase):
    needs_embedder = False

    def test_default_injects_no_type(self):
        """The boot default ships EMPTY: a long-lived journal item reaches a
        human through the Thalamus ask (budgeted, expiring, answerable), not
        through a node type printed at every boot. The mechanism stays for
        operators who name their own types."""
        from servers.scales.s1.frame import (render_standing_items,
                                             BOOT_INJECT_TYPES_DEFAULT)
        self.assertEqual(BOOT_INJECT_TYPES_DEFAULT, '')
        self.brain.remember(type='some-standing-type',
                            title='repo-question — open 3 sessions',
                            content='a standing item of an unconfigured type',
                            encoding_source='encoder:sonnet')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('BRAIN_BOOT_INJECT_TYPES', None)
            self.assertEqual(render_standing_items(self.brain), '')

    def test_env_var_extends_types(self):
        from servers.scales.s1.frame import render_standing_items
        self.brain.remember(type='my-custom-boot-type', title='custom item',
                            content='c', encoding_source='anchor')
        with mock.patch.dict(os.environ,
                             {'BRAIN_BOOT_INJECT_TYPES':
                              'some-standing-type, my-custom-boot-type'}):
            text = render_standing_items(self.brain)
        self.assertIn('custom item', text)

    def test_empty_when_nothing_qualifies(self):
        from servers.scales.s1.frame import render_standing_items
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('BRAIN_BOOT_INJECT_TYPES', None)
            self.assertEqual(render_standing_items(self.brain), '')

    def test_archived_items_leave_the_boot(self):
        from servers.scales.s1.frame import render_standing_items
        r = self.brain.remember(type='my-custom-boot-type', title='handled item',
                                content='c', encoding_source='anchor')
        self.brain.archive_node(r['id'], archived_by='anchor', reason='handled')
        with mock.patch.dict(os.environ,
                             {'BRAIN_BOOT_INJECT_TYPES': 'my-custom-boot-type'}):
            self.assertNotIn('handled item', render_standing_items(self.brain))


if __name__ == '__main__':
    unittest.main()
