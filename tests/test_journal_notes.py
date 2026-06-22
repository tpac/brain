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


class TestJournalNotesWrite(BrainTestBase):
    """brain.write_journal_notes() — the write door. Extracts the encoder's
    ## Review fenced block, parses it, writes journal_note rows. Mirror of the
    read door; round-trips through brain.journal_notes."""
    needs_embedder = False

    def test_write_then_read_roundtrip(self):
        final = (
            "Encoded 3 nodes, merged 1.\n\n"
            "## Review\n"
            "```\n"
            "friction · temporal-scout · misread a number again — 3rd run\n"
            "doubt · nodeA · merged but unsure the claims match\n"
            "```\n"
        )
        n = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000005-consolidation', scale='s2')
        assert n == 2
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {x['subject'] for x in out} == {'temporal-scout', 'nodeA'}
        assert {x['tag'] for x in out} == {'friction', 'doubt'}

    def test_no_op_without_review_section(self):
        n = self.brain.write_journal_notes(
            final_text="Encoded 3 nodes. No review here.",
            chain_id='s2-20260101000006-consolidation', scale='s2')
        assert n == 0

    def test_prose_outside_fence_not_parsed(self):
        # Review #4: prose before the fence with a stray '·' must NOT become a
        # malformed note — only the fenced block is parsed.
        final = (
            "## Review\n"
            "Some reflection · with a stray middot in prose, not a note.\n"
            "```\n"
            "surprise · recall-ranking · IDF boost helped, unexpectedly\n"
            "```\n"
        )
        n = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000007-consolidation', scale='s2')
        assert n == 1
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {x['note'] for x in out} == {'IDF boost helped, unexpectedly'}

    def test_malformed_line_in_fence_isolated(self):
        # One good note + one delimiter-less line: the good note is written, the
        # malformed one skipped (logged loud), the batch survives.
        final = (
            "## Review\n"
            "```\n"
            "doubt · nodeX · a real note\n"
            "this line has no delimiter and is malformed\n"
            "```\n"
        )
        n = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000008-consolidation', scale='s2')
        assert n == 1
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {x['note'] for x in out} == {'a real note'}


class TestJournalNotesRecallGuard(BrainTestBase):
    """journal_note must never leak into Anchor's recall. Structural guard:
    EAGER_TRACE_SCALES=('s0',) means s1/s2 traces are never embedded, so they
    can't surface in recall() (node + s0-trace vectors) or recall_episodes()
    (s0-scoped). Locked here so a future scope change can't silently break it."""
    needs_embedder = False

    def test_journal_note_scales_never_embedded(self):
        from servers.embed_queue import EAGER_TRACE_SCALES
        # journal_note rides on s1/s2; the embed worker only ever pulls s0.
        assert EAGER_TRACE_SCALES == ('s0',)
        assert 's1' not in EAGER_TRACE_SCALES and 's2' not in EAGER_TRACE_SCALES

    def test_journal_note_absent_from_recall_episodes(self):
        # A real s2 journal_note with a unique marker must not surface in
        # recall_episodes (which scans s0 only).
        self.brain._trace_dal.append(
            chain_id='s2-20260101000009-consolidation', scale='s2',
            event_type='delta', ref_type='journal_note', ref_id='nodeZ',
            summary='zebra-unique-marker',
            metadata=build_journal_note_metadata(note='zebra-unique-marker note'))
        res = self.brain.recall_episodes(contains='zebra-unique-marker')
        assert 'zebra-unique-marker' not in str(res)
