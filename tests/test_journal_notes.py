"""brain.journal_notes() — the encoder-journal read door.

Reads journal_note traces through the public trace API (query_traces), never
raw SQL. Covers the two modes: subject pull (every note about one ref_id) and
continuity (notes from the last K note-bearing runs of an encoder, grouped by
chain_id). Part of the encoder-journal redesign Phase 2.
"""
from tests.brain_test_base import BrainTestBase
from servers.trace_contract import build_journal_note_metadata


class TestJournalNotesRead(BrainTestBase):
    needs_embedder = False

    def _note(self, chain_id, scale, ref_id, note, tag='', session_id=''):
        """Append one journal_note trace via the DAL (the write path #6 will
        wrap this; here we exercise the READ)."""
        self.brain._trace_dal.append(
            chain_id=chain_id, scale=scale, event_type='delta',
            ref_type='journal_note', ref_id=ref_id, summary=note[:80],
            metadata=build_journal_note_metadata(note=note, tag=tag),
            session_id=session_id)

    def test_subject_pull_returns_all_notes_about_subject(self):
        self._note('s2-20260101000001-consolidation', 's2', 'nodeA', 'first', tag='doubt')
        self._note('s2-20260101000002-consolidation', 's2', 'nodeA', 'second', tag='friction')
        self._note('s2-20260101000003-consolidation', 's2', 'nodeB', 'other')
        out = self.brain.journal_notes(subject='nodeA')
        assert {n['note'] for n in out} == {'first', 'second'}
        assert all(n['subject'] == 'nodeA' for n in out)
        assert {n['tag'] for n in out} == {'doubt', 'friction'}

    def test_continuity_keeps_only_last_k_runs(self):
        for i in range(1, 5):  # 4 runs, one note each, appended oldest→newest
            self._note('s2-2026010100000%d-consolidation' % i, 's2',
                       'node%d' % i, 'run %d' % i)
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=3)
        assert {n['note'] for n in out} == {'run 2', 'run 3', 'run 4'}  # oldest dropped

    def test_continuity_groups_multi_note_run_as_one(self):
        # An older run with TWO notes + a newer run with one; k=1 → only newest run.
        self._note('s2-20260101000001-consolidation', 's2', 'a', 'old1')
        self._note('s2-20260101000001-consolidation', 's2', 'b', 'old2')
        self._note('s2-20260101000002-consolidation', 's2', 'c', 'new1')
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=1)
        assert {n['note'] for n in out} == {'new1'}

    def test_unit_scoping_excludes_other_units_same_scale(self):
        self._note('s2-20260101000001-consolidation', 's2', 'a', 'consol-note')
        self._note('s2-20260101000001-community_detection', 's2', 'b', 'community-note')
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {n['note'] for n in out} == {'consol-note'}

    def test_continuity_k_defaults_from_contract(self):
        # No explicit k → JOURNAL_CONTINUITY_RUNS['consolidation'] == 3.
        for i in range(1, 6):  # 5 runs
            self._note('s2-2026010100000%d-consolidation' % i, 's2',
                       'n%d' % i, 'r%d' % i)
        out = self.brain.journal_notes(scale='s2', unit='consolidation')
        assert len({n['chain_id'] for n in out}) == 3  # default K

    def test_s1_continuity_scoped_by_session(self):
        self._note('s1e-aaaa-5', 's1', 'nodeX', 'sess-a note', session_id='sess-a')
        self._note('s1e-bbbb-5', 's1', 'nodeY', 'sess-b note', session_id='sess-b')
        out = self.brain.journal_notes(scale='s1', session_id='sess-a', k=5)
        assert {n['note'] for n in out} == {'sess-a note'}

    def test_empty_when_no_notes(self):
        assert self.brain.journal_notes(scale='s2', unit='consolidation') == []

    def test_underscore_unit_matches_literally(self):
        # The unit filter is a LIKE under the hood; a '_' in the unit name must
        # match literally, not as a single-char wildcard.
        self._note('s2-20260101000001-community_detection', 's2', 'a', 'real')
        self._note('s2-20260101000001-communityXdetection', 's2', 'b', 'decoy')
        out = self.brain.journal_notes(scale='s2', unit='community_detection', k=5)
        assert {n['note'] for n in out} == {'real'}

    def test_unit_filter_pushed_down_so_limit_bounds_per_unit(self):
        # One (older) community note + many (newer) consolidation notes. A small
        # limit must STILL surface the community note: the unit filter runs in
        # SQL, so LIMIT bounds the filtered set — not the global delta stream
        # (the old Python post-filter would have lost it behind the limit).
        self._note('s2-20260101000001-community_detection', 's2', 'c', 'community')
        for i in range(2, 8):
            self._note('s2-2026010100000%d-consolidation' % i, 's2',
                       'n%d' % i, 'consol %d' % i)
        out = self.brain.journal_notes(scale='s2', unit='community_detection', limit=2)
        assert {n['note'] for n in out} == {'community'}
