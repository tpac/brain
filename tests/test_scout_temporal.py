"""Tests for servers/scales/s1/scouts/temporal.py — algorithmic temporal scout.

Exercises each extraction pass in isolation plus end-to-end behavior:
- Phrase-shape filter (digits, months, weekday+modifier, time-only rejection)
- Trailing-noise trim
- Modifier+weekday local resolution ("last Tuesday")
- Modifier+unit local resolution ("next month")
- Word-number relatives ("three weeks ago")
- Vague quantifiers ("a few weeks ago")
- Fuzzy anchors ("recently", "a while back")
- Catalog lookup for existing time_anchor
- Specificity ranking + max_candidates cap
- Envelope shape matches scout contract
- Missing interaction / invalid base date → stub with logged errors
"""

import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.scales.s1.scouts import temporal as t


class TestFilterHelpers(unittest.TestCase):

    def test_looks_like_date_accepts_digit_relatives(self):
        self.assertTrue(t._looks_like_date('2 weeks ago'))
        self.assertTrue(t._looks_like_date('5 days ago'))
        self.assertTrue(t._looks_like_date('March 15'))

    def test_looks_like_date_accepts_keywords(self):
        self.assertTrue(t._looks_like_date('today'))
        self.assertTrue(t._looks_like_date('yesterday'))
        self.assertTrue(t._looks_like_date('tomorrow'))

    def test_looks_like_date_accepts_weekday_with_modifier(self):
        self.assertTrue(t._looks_like_date('last Tuesday'))
        self.assertTrue(t._looks_like_date('next Friday'))

    def test_looks_like_date_rejects_bare_weekday(self):
        self.assertFalse(t._looks_like_date('Tuesday'))
        self.assertFalse(t._looks_like_date('friday'))

    def test_looks_like_date_rejects_time_only(self):
        self.assertFalse(t._looks_like_date('7 PM'))
        self.assertFalse(t._looks_like_date('6 AM'))
        self.assertFalse(t._looks_like_date('3 hours'))

    def test_looks_like_date_rejects_noise(self):
        self.assertFalse(t._looks_like_date('to'))
        self.assertFalse(t._looks_like_date('we'))
        self.assertFalse(t._looks_like_date(''))
        self.assertFalse(t._looks_like_date('   '))

    def test_looks_like_date_rejects_bare_numbers(self):
        # R1: a bare integer is not a date. Line numbers, percentages, PIDs,
        # versions, arxiv ids, section numbers, token/char counts all carry
        # digits but no date shape. Must reject — the old `has_digit_word`
        # gate accepted every one of these.
        self.assertFalse(t._looks_like_date('47'))
        self.assertFalse(t._looks_like_date('73'))
        self.assertFalse(t._looks_like_date('61167'))
        self.assertFalse(t._looks_like_date('v22'))
        self.assertFalse(t._looks_like_date('2305.12345'))
        self.assertFalse(t._looks_like_date('3.2'))
        self.assertFalse(t._looks_like_date('4096'))

    def test_looks_like_date_accepts_date_shapes(self):
        # Date-shaped digit tokens still pass: 4-digit year, ISO/slash
        # numeric date, ordinal day, and digit relatives.
        self.assertTrue(t._looks_like_date('2026'))
        self.assertTrue(t._looks_like_date('2026-06-28'))
        self.assertTrue(t._looks_like_date('5/28/2026'))
        self.assertTrue(t._looks_like_date('March 5th'))
        self.assertTrue(t._looks_like_date('2 weeks ago'))

    def test_trim_trailing_noise(self):
        self.assertEqual(t._trim_trailing_noise('2 weeks ago at the'),
                         '2 weeks ago')
        self.assertEqual(t._trim_trailing_noise('last Tuesday on'),
                         'last Tuesday')
        self.assertEqual(t._trim_trailing_noise('March 15'),
                         'March 15')


class TestResolvers(unittest.TestCase):

    def setUp(self):
        # Thursday, 2026-04-23. Used as the base for all tests.
        self.base = dt.datetime(2026, 4, 23)

    def test_last_tuesday_is_two_days_back(self):
        r = t._resolve_modifier_weekday('last', 'Tuesday', self.base)
        self.assertEqual(r.date(), dt.date(2026, 4, 21))

    def test_last_thursday_is_seven_days_back(self):
        """Same weekday as base → 'last' means prior week, never today."""
        r = t._resolve_modifier_weekday('last', 'Thursday', self.base)
        self.assertEqual(r.date(), dt.date(2026, 4, 16))

    def test_next_tuesday_is_five_days_forward(self):
        r = t._resolve_modifier_weekday('next', 'Tuesday', self.base)
        self.assertEqual(r.date(), dt.date(2026, 4, 28))

    def test_next_thursday_is_seven_days_forward(self):
        """Same weekday as base → 'next' means next week, never today."""
        r = t._resolve_modifier_weekday('next', 'Thursday', self.base)
        self.assertEqual(r.date(), dt.date(2026, 4, 30))

    def test_this_tuesday_is_two_days_back(self):
        r = t._resolve_modifier_weekday('this', 'Tuesday', self.base)
        self.assertEqual(r.date(), dt.date(2026, 4, 21))

    def test_this_thursday_is_today(self):
        """'this Thursday' when today IS Thursday → today."""
        r = t._resolve_modifier_weekday('this', 'Thursday', self.base)
        self.assertEqual(r.date(), dt.date(2026, 4, 23))

    def test_last_month_is_one_month_back(self):
        r = t._resolve_modifier_unit('last', 'month', self.base)
        self.assertEqual((self.base - r).days, 30)

    def test_next_year_is_one_year_forward(self):
        r = t._resolve_modifier_unit('next', 'year', self.base)
        self.assertEqual((r - self.base).days, 365)

    def test_this_week_is_base(self):
        r = t._resolve_modifier_unit('this', 'week', self.base)
        self.assertEqual(r, self.base)


class TestExtractionPasses(unittest.TestCase):

    def setUp(self):
        self.base = dt.datetime(2026, 4, 23)

    def _extract(self, text):
        return t._extract_candidates_from_text(text, self.base)

    def test_dateparser_primary_catches_digit_relative(self):
        results = self._extract('I met her 2 weeks ago.')
        found = [r['phrase_clean'] for r in results]
        self.assertTrue(any('2 weeks ago' in p for p in found))

    def test_dateparser_primary_catches_today(self):
        results = self._extract('Today I booked the flight.')
        isos = [r['iso_date'] for r in results]
        self.assertIn('2026-04-23', isos)

    def test_modifier_weekday_supplement_catches_last_tuesday(self):
        """dateparser.parse returns None for 'last Tuesday' with
        RELATIVE_BASE — our supplement must still catch it."""
        results = self._extract('Last Tuesday was a big day.')
        isos = [r['iso_date'] for r in results]
        self.assertIn('2026-04-21', isos)

    def test_modifier_unit_supplement_catches_next_month(self):
        results = self._extract('Next month I start the new role.')
        # Should resolve to base + 30 days = 2026-05-23
        isos = [r['iso_date'] for r in results]
        self.assertTrue(any(iso.startswith('2026-05') for iso in isos))

    def test_word_relative_catches_three_months_ago(self):
        results = self._extract('I submitted it three months ago.')
        isos = [r['iso_date'] for r in results]
        # Roughly 2026-01-23
        self.assertTrue(any(iso.startswith('2026-01') for iso in isos))

    def test_word_relative_catches_five_days_ago(self):
        results = self._extract('I did it five days ago.')
        isos = [r['iso_date'] for r in results]
        self.assertIn('2026-04-18', isos)

    def test_vague_quantifier_catches_a_few_weeks_ago(self):
        results = self._extract('It happened a few weeks ago.')
        relevant = [r for r in results
                    if r.get('precision') == 'approximate']
        self.assertTrue(len(relevant) >= 1,
                        f'expected approximate candidate, got {results}')

    def test_vague_quantifier_catches_several_months_ago(self):
        results = self._extract('Several months ago I moved.')
        relevant = [r for r in results
                    if r.get('precision') == 'approximate']
        self.assertTrue(len(relevant) >= 1)

    def test_fuzzy_anchor_catches_recently(self):
        results = self._extract('I moved recently, about a week back.')
        phrases = [r['phrase_clean'].lower() for r in results]
        self.assertTrue(any('recently' in p for p in phrases))

    def test_time_only_phrases_ignored(self):
        results = self._extract('Let\'s meet at 7 PM instead of 6 PM.')
        # No date phrases — time-only should be filtered
        self.assertEqual(results, [])

    def test_dedup_same_phrase_same_date(self):
        """"Today" appearing twice in text should produce ONE candidate."""
        results = self._extract('Today I worked. Today was a great day.')
        todays = [r for r in results
                  if r['phrase_clean'].lower() == 'today']
        self.assertEqual(len(todays), 1)


class TestScoutOutput(BrainTestBase):
    needs_embedder = False

    def _run(self, turns, catalog=None, current_date='2026-04-23'):
        return t.run_temporal_scout(
            brain=self.brain, turns=turns,
            catalog_nodes=catalog or [],
            current_date=current_date)

    def test_envelope_shape_matches_contract(self):
        turns = [{'turn_id': 't1', 'role': 'user',
                  'text': 'I met her 2 weeks ago.'}]
        out = self._run(turns)
        self.assertEqual(out['scout'], 'temporal')
        self.assertTrue(out['category_statement'])
        self.assertIsInstance(out['candidates'], list)
        self.assertIn('turns', out['scanned'])
        self.assertIn('date_phrases_found', out['scanned'])

    def test_candidate_has_required_fields(self):
        turns = [{'turn_id': 't1', 'role': 'user',
                  'text': 'I met her 2 weeks ago at the airport.'}]
        out = self._run(turns)
        self.assertTrue(len(out['candidates']) >= 1)
        c = out['candidates'][0]
        for field in ('handle', 'evidence_quote', 'evidence_turns',
                      'why_candidate', 'source_phrase', 'resolution',
                      'event_description'):
            self.assertIn(field, c, f'missing {field}')

    def test_existing_anchor_detected_in_catalog(self):
        """When a catalog has a time_anchor for the resolved date, scout
        emits existing_anchor_id so S1S can reuse instead of duplicating."""
        # Seed a time_anchor manually into the brain + pass in catalog
        catalog = [{'id': 'ta_20260423', 'type': 'time_anchor',
                    'title': '2026-04-23'}]
        turns = [{'turn_id': 't1', 'role': 'user',
                  'text': 'Today I booked the flight.'}]
        out = self._run(turns, catalog=catalog)
        today_cand = [c for c in out['candidates']
                      if c['handle'] == '2026-04-23']
        self.assertTrue(len(today_cand) >= 1)
        self.assertEqual(today_cand[0]['existing_anchor_id'], 'ta_20260423')

    def test_no_catalog_match_gives_null_anchor_id(self):
        turns = [{'turn_id': 't1', 'role': 'user',
                  'text': 'I met her 2 weeks ago.'}]
        out = self._run(turns)
        for c in out['candidates']:
            self.assertIsNone(c['existing_anchor_id'])

    def test_max_candidates_cap_enforced(self):
        """Generate more than max_candidates — cap is honored, highest
        specificity wins."""
        # Build a turn text mentioning many dates
        text = ('March 15. April 20. May 1. June 7. July 18. August 22. '
                'September 3. October 9. November 25. December 30.')
        turns = [{'turn_id': 't1', 'role': 'user', 'text': text}]
        out = self._run(turns)
        # interaction parameters set max_candidates=8
        self.assertLessEqual(len(out['candidates']), 8)

    def test_explicit_dates_rank_above_approximate(self):
        text = ('I did X on March 15 and Y a few weeks ago.')
        turns = [{'turn_id': 't1', 'role': 'user', 'text': text}]
        out = self._run(turns)
        # Scout should rank explicit above approximate — first candidate is
        # the explicit one.
        self.assertTrue(len(out['candidates']) >= 2)
        # March 15 → 2026-03-15
        self.assertEqual(out['candidates'][0]['handle'], '2026-03-15')

    def test_multi_turn_evidence_turns_attributed_correctly(self):
        turns = [
            {'turn_id': 't1', 'role': 'user', 'text': 'Today I booked it.'},
            {'turn_id': 't5', 'role': 'user', 'text': 'Last Tuesday we met.'},
        ]
        out = self._run(turns)
        by_handle = {c['handle']: c for c in out['candidates']}
        self.assertIn('2026-04-23', by_handle)
        self.assertEqual(by_handle['2026-04-23']['evidence_turns'], ['t1'])
        self.assertIn('2026-04-21', by_handle)
        self.assertEqual(by_handle['2026-04-21']['evidence_turns'], ['t5'])

    def test_assistant_only_date_tags_source_role_assistant(self):
        """Regression for gpt4_85da3956 (2026-05-13): when only the assistant
        mentions a date phrase ('three weeks ago' paraphrasing a user's
        proximal phrase), source_role MUST be 'assistant' so downstream
        S1E can choose conversation_now over assistant-attributed dates."""
        turns = [
            {'turn_id': 't0', 'role': 'user',
             'text': 'I just got back from an amazing day at Universal.'},
            {'turn_id': 't1', 'role': 'assistant',
             'text': "I didn't know you went to Universal three weeks ago."},
        ]
        out = self._run(turns)
        # Find the 'three weeks ago' candidate (3 weeks before base = 2026-04-02)
        cand = next((c for c in out['candidates']
                     if c['source_phrase'].lower().startswith('three weeks')),
                    None)
        self.assertIsNotNone(cand, 'three-weeks-ago candidate missing')
        self.assertEqual(cand['source_role'], 'assistant',
                         'date came from assistant turn — must be tagged')
        self.assertEqual(cand['evidence_roles'], ['assistant'])

    def test_user_attribution_wins_when_both_roles_mention_same_date(self):
        """When the same date appears in both user and assistant turns,
        source_role promotes to 'user' (operator attribution wins) even
        if the assistant-turn wording happens to be more specific. The
        encoder reads source_role as the authority for resolution."""
        turns = [
            {'turn_id': 't0', 'role': 'assistant',
             'text': "OK so on April 21 you did the thing."},
            {'turn_id': 't1', 'role': 'user',
             'text': "Yeah, last Tuesday I did it."},
        ]
        out = self._run(turns)
        # Both phrases resolve to 2026-04-21 against base 2026-04-23.
        cand = next((c for c in out['candidates']
                     if c['handle'] == '2026-04-21'), None)
        self.assertIsNotNone(cand, 'April 21 candidate missing')
        self.assertEqual(cand['source_role'], 'user',
                         'user attribution must win over assistant')
        # evidence_roles should show both
        self.assertIn('user', cand['evidence_roles'])
        self.assertIn('assistant', cand['evidence_roles'])

    def test_role_missing_in_turn_yields_blank_source_role(self):
        """Defensive: turns without a role key still produce candidates;
        source_role is just empty for those (probably a caller bug, but
        the scout doesn't crash and doesn't lie about attribution)."""
        turns = [{'turn_id': 't1', 'text': 'I met her 2 weeks ago.'}]
        out = self._run(turns)
        self.assertTrue(len(out['candidates']) >= 1)
        c = out['candidates'][0]
        self.assertEqual(c.get('source_role', '_missing'), '')
        self.assertEqual(c.get('evidence_roles', '_missing'), [])

    def test_empty_turns_yields_empty_candidates(self):
        out = self._run([])
        self.assertEqual(out['candidates'], [])
        self.assertEqual(out['scanned']['turns'], 0)

    def test_no_dates_yields_empty_candidates(self):
        turns = [{'turn_id': 't1', 'role': 'user',
                  'text': 'Hello how are you? This is a test.'}]
        out = self._run(turns)
        self.assertEqual(out['candidates'], [])


class TestFailurePaths(BrainTestBase):
    needs_embedder = False

    def test_invalid_current_date_returns_stub(self):
        turns = [{'turn_id': 't1', 'role': 'user',
                  'text': 'today I did a thing'}]
        out = t.run_temporal_scout(
            brain=self.brain, turns=turns, catalog_nodes=[],
            current_date='not-a-date')
        self.assertEqual(out['candidates'], [])
        errors = out['_errors']
        self.assertTrue(any(e['type'] == 'bad_current_date' for e in errors))

    def test_missing_interaction_returns_stub(self):
        """Patch target is the `_load_interaction` name imported into
        temporal.py (not scout_base's copy) — otherwise temporal.py's
        local reference still resolves the real implementation."""
        from unittest.mock import patch
        turns = [{'turn_id': 't1', 'role': 'user',
                  'text': 'today I booked it'}]
        with patch('servers.scales.s1.scouts.temporal._load_interaction',
                   return_value=None):
            out = t.run_temporal_scout(
                brain=self.brain, turns=turns, catalog_nodes=[],
                current_date='2026-04-23')
        self.assertEqual(out['candidates'], [])
        self.assertTrue(any(e['type'] == 'missing_interaction'
                            for e in out['_errors']))


class TestBareNumberFalsePositives(unittest.TestCase):
    """R1 end-to-end: digit-bearing non-dates must emit NO anchor, real
    dates and relatives must still emit. Two-layer fix — dropping
    dateparser's 'timestamp' parser (which reads bare digit runs as Unix
    epochs) plus the _looks_like_date date-shape gate."""

    BASE = dt.datetime(2026, 6, 28)

    NEGATIVES = [
        'see line 47 for the bug',
        'coverage is 73%',
        'we shipped v22 to prod',
        'the daemon PID 61167 restarted',
        'read arxiv 2305.12345 on this',
        'refactored section 3.2 of the doc',
        'the cache is 4096 tokens wide',
    ]

    POSITIVES = [
        'we discussed this 2026-06-28',
        'the meeting is March 5',
        'shipped on March 5 2026',
        'the deadline is 5/28/2026',
        'that was 5 days ago',
        'three weeks ago we decided',
        'last Tuesday we shipped',
        'I saw it yesterday',
    ]

    def test_negatives_emit_no_anchor(self):
        for text in self.NEGATIVES:
            cands = t._extract_candidates_from_text(text, self.BASE)
            self.assertEqual(
                cands, [],
                'expected no temporal anchor for %r, got %r'
                % (text, [c['phrase_clean'] for c in cands]))

    def test_positives_emit_anchor(self):
        for text in self.POSITIVES:
            cands = t._extract_candidates_from_text(text, self.BASE)
            self.assertTrue(
                cands, 'expected a temporal anchor for %r, got none' % text)


if __name__ == '__main__':
    unittest.main()
