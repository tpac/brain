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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase  # noqa: E402
from servers.scales.journal import JournalBinding  # noqa: E402
from servers.scales.runner import extract_json  # noqa: E402


FINAL_TEXT = (
    'Batch judged. Payload:\n[{"node": "abc", "score": 0.9}]\n\n'
    '## Review\n```\nfriction · abc · score capped at 0.9]\n```\n'
)


class TestHarvest(BrainTestBase):
    needs_embedder = False

    def _binding(self, **kw):
        kw.setdefault('scale', 's2')
        kw.setdefault('unit', 'consolidation')
        return JournalBinding(self.brain, **kw)

    def test_harvest_writes_notes_and_strips(self):
        b = self._binding()
        remainder = b.harvest(FINAL_TEXT, 's2-20260728000000-consolidation')

        notes = self.brain.journal_notes(scale='s2', unit='consolidation')
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]['tag'], 'friction')
        self.assertEqual(notes[0]['subject'], 'abc')
        self.assertNotIn('## Review', remainder)

    def test_stripped_remainder_survives_extract_json(self):
        """The envelope rule: the journal fence corrupts extract_json on the
        raw text (its markdown-fence split discards the pre-fence payload, and
        a `]` inside the fence poisons the rfind scan) — after harvest the
        payload parses cleanly."""
        corrupted = extract_json(FINAL_TEXT)
        self.assertIsNone(corrupted)          # the raw text IS the hazard

        b = self._binding()
        remainder = b.harvest(FINAL_TEXT, 's2-20260728000001-consolidation')
        self.assertEqual(extract_json(remainder),
                         [{"node": "abc", "score": 0.9}])

    def test_arc_bound_harvest_writes_session_arc(self):
        text = ('done.\n\n## Arc\n```\nsurvivor ladder shipped\n```\n\n'
                '## Review\n```\n```\n')
        b = JournalBinding(self.brain, scale='s1', session_id='sess-arc',
                           arc=True)
        remainder = b.harvest(text, 's1e-sessarc-5')
        self.assertIn('survivor ladder shipped',
                      self.brain.get_config('session_context_sess-arc', ''))
        self.assertNotIn('## Arc', remainder)
        self.assertNotIn('## Review', remainder)

    def test_continuity_failure_isolated(self):
        """A broken notes read degrades to no continuity, never raises."""
        b = self._binding()
        orig = self.brain.journal_notes
        self.brain.journal_notes = None       # any call → TypeError
        try:
            self.assertEqual(b.continuity(), '')
        finally:
            self.brain.journal_notes = orig

    def test_continuity_round_trip(self):
        b = self._binding(unit='community_detection')
        b.harvest('## Review\n```\ndoubt · xyz · unsure about placement\n```',
                  's2-20260728000002-community_detection')
        prefix = b.continuity()
        self.assertIn('xyz', prefix)
        self.assertIn('RECENT REVIEW NOTES', prefix)


if __name__ == '__main__':
    unittest.main()
