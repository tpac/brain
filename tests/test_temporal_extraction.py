"""Tests for `servers.temporal_extraction` — interval extraction primitives.

Covers:
  - Pattern matching for each of the 5 active patterns
  - Year-required behavior (rejects bare years, month-only)
  - Overlap consumption (higher-precision wins)
  - Dedup within source / across sources
  - Node-level + edge-level extraction wiring
  - write_entity_dates: sentinel rows + real rows + idempotency
  - Regression cases observed on the live brain (year-2000 false positive)

No DB needed for pattern tests; in-memory sqlite for write_entity_dates.
"""

from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime, timezone

from servers.temporal_extraction import (
    extract_intervals_from_text,
    extract_node_intervals,
    extract_edge_intervals,
    write_entity_dates,
    _SENTINEL_SOURCE,
    MAX_INTERVALS_PER_ENTITY,
)


def _fmt(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')


def _intervals(text: str) -> list:
    """Return list of (start_date, end_date, raw) for readability."""
    return [(_fmt(s), _fmt(e), r) for s, e, r in extract_intervals_from_text(text)]


class TestPatternMatching(unittest.TestCase):
    """Each pattern produces the expected interval."""

    def test_quarter(self):
        self.assertEqual(_intervals("Q1 2024 was the launch"),
                         [('2024-01-01', '2024-03-31', 'Q1 2024')])

    def test_quarter_all_four(self):
        for q, (s, e) in [(1, ('01-01', '03-31')), (2, ('04-01', '06-30')),
                          (3, ('07-01', '09-30')), (4, ('10-01', '12-31'))]:
            with self.subTest(q=q):
                result = _intervals(f"Q{q} 2024 launch")
                self.assertEqual(result, [(f'2024-{s}', f'2024-{e}', f'Q{q} 2024')])

    def test_iso_day(self):
        self.assertEqual(_intervals("on 2023-05-22 we shipped"),
                         [('2023-05-22', '2023-05-22', '2023-05-22')])

    def test_iso_month(self):
        self.assertEqual(_intervals("event_time = 2023-05"),
                         [('2023-05-01', '2023-05-31', '2023-05')])

    def test_month_day_year(self):
        self.assertEqual(_intervals("Tom finished on May 27, 2023."),
                         [('2023-05-27', '2023-05-27', 'May 27, 2023')])

    def test_month_year(self):
        self.assertEqual(_intervals("Read in May 2023, 440 pages"),
                         [('2023-05-01', '2023-05-31', 'May 2023')])

    def test_month_range(self):
        self.assertEqual(_intervals("Active during January through April 2023"),
                         [('2023-01-01', '2023-04-30',
                           'January through April 2023')])

    def test_month_range_with_between(self):
        self.assertEqual(_intervals("between Feb and April 2023"),
                         [('2023-02-01', '2023-04-30',
                           'between Feb and April 2023')])

    def test_month_range_with_from_to(self):
        self.assertEqual(_intervals("from January to March 2023"),
                         [('2023-01-01', '2023-03-31',
                           'from January to March 2023')])


class TestYearRequired(unittest.TestCase):
    """Year context is mandatory; bare numbers and dateless month names
    are skipped to avoid current-year fallback errors."""

    def test_month_without_year_skipped(self):
        self.assertEqual(_intervals("read The Power in December"), [])

    def test_day_without_year_skipped(self):
        # Just a month and day — no year — must be skipped
        self.assertEqual(_intervals("on May 27 we shipped"), [])

    def test_bare_year_skipped_v1(self):
        # Standalone year matching is intentionally disabled in v1 to avoid
        # false positives on technical numbers like "2000 chars" / "4096 tokens".
        self.assertEqual(_intervals("In 2024 we shipped"), [])

    def test_pure_number_skipped(self):
        for s in ["2000 chars of Sonnet reasoning",
                  "4096 tokens limit",
                  "truncation at 2048ms",
                  "limit was 2000 chars"]:
            with self.subTest(input=s):
                self.assertEqual(_intervals(s), [])


class TestOverlapConsumption(unittest.TestCase):
    """Higher-precision matches consume the year/month tokens so lower-
    precision patterns don't double-count."""

    def test_day_consumes_month_and_year(self):
        # "May 27, 2023" should emit ONE day interval, not also a month + year
        result = _intervals("Tom finished on May 27, 2023.")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], '2023-05-27')
        self.assertEqual(result[0][1], '2023-05-27')

    def test_iso_day_does_not_double_count(self):
        result = _intervals("ISO date: 2023-05-22")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], '2023-05-22')

    def test_quarter_does_not_emit_extra_year(self):
        result = _intervals("Q3 2024 was the launch")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], '2024-07-01')


class TestMultipleIntervalsAndDedup(unittest.TestCase):
    """Multiple dates per text emit separately; duplicates collapse."""

    def test_two_months_separate(self):
        result = _intervals("Started January 2023, finished March 2023.")
        self.assertEqual(len(result), 2)
        self.assertEqual({r[0] for r in result},
                         {'2023-01-01', '2023-03-01'})

    def test_quarter_and_month_same_text(self):
        result = _intervals("Q1 2024 was the launch quarter, then May 2024 rollout")
        self.assertEqual(len(result), 2)
        starts = {r[0] for r in result}
        self.assertEqual(starts, {'2024-01-01', '2024-05-01'})

    def test_duplicate_same_interval_dedup(self):
        # Same interval written twice → emitted once
        result = _intervals("May 2023 was great. May 2023 was great.")
        self.assertEqual(len(result), 1)


class TestOverlapGuard(unittest.TestCase):
    """The consumed-span guard prevents the lower-precision patterns
    (MonthName Year, MonthName Day Year) from double-counting their
    matches when a higher-precision pattern has already claimed the
    overlapping span. These tests lock the guard's contract."""

    def test_day_pattern_does_not_emit_extra_month(self):
        # "May 22, 2023" — day pattern claims the span. Month pattern
        # would otherwise re-match "May" + "2023" via separate substrings.
        result = _intervals("May 22, 2023")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], '2023-05-22')

    def test_two_distinct_mentions_both_emit(self):
        # "Tom finished in May 2023 on May 27, 2023" — there ARE two
        # distinct mentions (a month and a day). Both should emit.
        result = _intervals("Tom finished in May 2023 on May 27, 2023")
        self.assertEqual(len(result), 2)
        starts = {r[0] for r in result}
        self.assertEqual(starts, {'2023-05-01', '2023-05-27'})

    def test_mixed_precision_in_one_text(self):
        # Two separate mentions, different precision — both emit.
        result = _intervals(
            "Started January 2023, finished March 22, 2023")
        self.assertEqual(len(result), 2)
        starts = sorted(r[0] for r in result)
        self.assertEqual(starts, ['2023-01-01', '2023-03-22'])


class TestKnownLimitations(unittest.TestCase):
    """Cases the v1 implementation does NOT handle — locked-in so we
    notice when behavior changes."""

    def test_cross_year_range_emits_two_separate_intervals(self):
        # "December 2022 to March 2023" — month1 > month2 in the range
        # regex (Dec=12, Mar=3), so range pattern skips. Falls through
        # to MonthName-Year, which matches "December 2022" and "March 2023"
        # individually. NOT a single range interval.
        result = _intervals("December 2022 to March 2023 launch window")
        self.assertEqual(len(result), 2)
        starts = sorted(r[0] for r in result)
        self.assertEqual(starts, ['2022-12-01', '2023-03-01'])

    def test_reversed_range_emits_single_month(self):
        # "March to January 2023" — month1 > month2, range skips; only
        # "January 2023" matches MonthName-Year ("March" has no year context).
        result = _intervals("March to January 2023")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], '2023-01-01')

    def test_compact_dash_form_matches_range(self):
        # "Jan-March 2023" — compact dash form, no whitespace around the
        # dash. After audit fix S3, the range pattern allows optional
        # whitespace around dash separators (while still requiring
        # whitespace around word separators like 'to', 'and', 'through').
        result = _intervals("Jan-March 2023 launch window")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], '2023-01-01')
        self.assertEqual(result[0][1], '2023-03-31')


class TestExtractNodeIntervals(unittest.TestCase):
    """Node-level extraction: title + content + KV fields."""

    def test_title_only(self):
        result = extract_node_intervals(
            title="Shipped in May 2023",
            content=None,
            kv_pairs=[])
        self.assertEqual(len(result), 1)
        _, _, source, _ = result[0]
        self.assertEqual(source, 'node.title')

    def test_content_only(self):
        result = extract_node_intervals(
            title=None,
            content="Tom finished on May 27, 2023",
            kv_pairs=[])
        self.assertEqual(len(result), 1)
        _, _, source, _ = result[0]
        self.assertEqual(source, 'node.content')

    def test_explicit_temporal_kv(self):
        result = extract_node_intervals(
            title=None, content=None,
            kv_pairs=[('event_time', '2023-05')])
        sources = [r[2] for r in result]
        self.assertIn('node.kv:event_time', sources)

    def test_canonical_event_time_kv_inputs(self):
        # Real encoder output for event_time uses YYYY-MM-DD form
        # (verified on the live brain — see audit S5). Lock canonical
        # input handling.
        for value, expected_start, expected_end in [
            ('2023-05-22', '2023-05-22', '2023-05-22'),
            ('2023-05', '2023-05-01', '2023-05-31'),
            ('Q1 2024', '2024-01-01', '2024-03-31'),
        ]:
            with self.subTest(value=value):
                result = extract_node_intervals(
                    None, None, [('event_time', value)])
                self.assertEqual(len(result), 1)
                s, e, src, _ = result[0]
                self.assertEqual(src, 'node.kv:event_time')
                self.assertEqual(_fmt(s), expected_start)
                self.assertEqual(_fmt(e), expected_end)

    def test_unparseable_kv_returns_empty(self):
        # Malformed / vague KV values gracefully return no intervals
        # (no false positives from the extractor's regex patterns).
        for value in ['', 'unknown', 'sometime in 2023', 'around May', 'TBD']:
            with self.subTest(value=value):
                result = extract_node_intervals(
                    None, None, [('event_time', value)])
                self.assertEqual(result, [])

    def test_scanned_text_kv(self):
        result = extract_node_intervals(
            title=None, content=None,
            kv_pairs=[('situation', 'mentioned in May 2023 conversation')])
        sources = [r[2] for r in result]
        self.assertIn('node.kv_scan:situation', sources)

    def test_same_interval_two_sources_keeps_both(self):
        # Title and content both mention May 2023 — emit both as separate
        # source rows (downstream can dedupe by entity if needed)
        result = extract_node_intervals(
            title='Shipped May 2023',
            content='Released in May 2023',
            kv_pairs=[])
        self.assertEqual(len(result), 2)
        sources = {r[2] for r in result}
        self.assertEqual(sources, {'node.title', 'node.content'})

    def test_no_dates_returns_empty(self):
        result = extract_node_intervals(
            title="Some abstract principle",
            content="Just text with no temporal anchors",
            kv_pairs=[('keywords', 'principle decision')])
        self.assertEqual(result, [])


class TestExtractEdgeIntervals(unittest.TestCase):
    """Edge-level extraction: description + relation."""

    def test_description_with_date(self):
        result = extract_edge_intervals(
            description="validates the May 2024 audit findings",
            relation="validates")
        self.assertEqual(len(result), 1)
        _, _, source, _ = result[0]
        self.assertEqual(source, 'edge.description')

    def test_relation_without_date(self):
        result = extract_edge_intervals(
            description="K factor split when v11 launched",
            relation="replaces")
        # No explicit dates — v11 is a named anchor we don't resolve yet
        self.assertEqual(result, [])

    def test_both_none(self):
        result = extract_edge_intervals(description=None, relation=None)
        self.assertEqual(result, [])


class TestWriteEntityDates(unittest.TestCase):
    """SQL contract: sentinel rows for empty extractions, real rows for
    non-empty, idempotent DELETE+INSERT semantics."""

    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.execute('''
            CREATE TABLE entity_dates (
                entity_kind TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                start_ts INTEGER NOT NULL,
                end_ts INTEGER NOT NULL,
                extraction_source TEXT NOT NULL,
                raw_text TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (entity_kind, entity_id, start_ts, end_ts, extraction_source)
            )
        ''')

    def _count(self) -> int:
        return self.conn.execute('SELECT COUNT(*) FROM entity_dates').fetchone()[0]

    def _sentinel_count(self) -> int:
        return self.conn.execute(
            'SELECT COUNT(*) FROM entity_dates WHERE start_ts = 0'
        ).fetchone()[0]

    def test_empty_intervals_writes_sentinel(self):
        written = write_entity_dates(self.conn, 'node', 'abc', [])
        self.assertEqual(written, 0)
        self.assertEqual(self._count(), 1)
        self.assertEqual(self._sentinel_count(), 1)
        row = self.conn.execute(
            'SELECT extraction_source FROM entity_dates'
        ).fetchone()
        self.assertEqual(row[0], _SENTINEL_SOURCE)

    def test_real_intervals_written(self):
        written = write_entity_dates(self.conn, 'node', 'abc', [
            (1683935999, 1685577599, 'node.title', 'May 2023'),
            (1685577600, 1688169599, 'node.content', 'June 2023'),
        ])
        self.assertEqual(written, 2)
        self.assertEqual(self._count(), 2)
        self.assertEqual(self._sentinel_count(), 0)

    def test_idempotent_re_run_replaces(self):
        write_entity_dates(self.conn, 'node', 'abc', [
            (1, 2, 'node.title', 'first'),
        ])
        self.assertEqual(self._count(), 1)
        write_entity_dates(self.conn, 'node', 'abc', [
            (10, 20, 'node.title', 'second'),
        ])
        # Second call replaces the first; only one row remains for this entity
        self.assertEqual(self._count(), 1)
        row = self.conn.execute(
            'SELECT start_ts, raw_text FROM entity_dates'
        ).fetchone()
        self.assertEqual(row, (10, 'second'))

    def test_sentinel_replaced_by_real_on_re_extract(self):
        # Sentinel written first
        write_entity_dates(self.conn, 'node', 'abc', [])
        self.assertEqual(self._sentinel_count(), 1)
        # Later real extraction (e.g., node was revised with dated content)
        write_entity_dates(self.conn, 'node', 'abc', [
            (1, 2, 'node.title', 'May 2023'),
        ])
        self.assertEqual(self._sentinel_count(), 0)
        self.assertEqual(self._count(), 1)

    def test_two_entities_independent(self):
        write_entity_dates(self.conn, 'node', 'abc', [
            (1, 2, 'node.title', 'a'),
        ])
        write_entity_dates(self.conn, 'edge', 'xyz', [
            (3, 4, 'edge.description', 'b'),
        ])
        self.assertEqual(self._count(), 2)

    def test_intervals_capped_at_max_per_entity(self):
        # Pathological input — 50 intervals for one entity. Cap kicks in.
        many = [(i, i + 1, 'node.content', f'date-{i}')
                for i in range(50)]
        written = write_entity_dates(self.conn, 'node', 'abc', many)
        self.assertEqual(written, MAX_INTERVALS_PER_ENTITY)
        self.assertEqual(self._count(), MAX_INTERVALS_PER_ENTITY)
        # The kept rows are the first N (emission order = source priority).
        kept_starts = [r[0] for r in self.conn.execute(
            'SELECT start_ts FROM entity_dates ORDER BY start_ts'
        ).fetchall()]
        self.assertEqual(kept_starts, list(range(MAX_INTERVALS_PER_ENTITY)))

    def test_sentinel_is_identifiable_by_extraction_source(self):
        # Sentinel rows are filtered by recall_by_time via
        # extraction_source = _SENTINEL_SOURCE — NOT by start_ts > 0.
        # Lock this contract so a future change to the marker breaks
        # this test instead of silently leaking sentinels.
        write_entity_dates(self.conn, 'node', 'abc', [])
        row = self.conn.execute(
            'SELECT extraction_source FROM entity_dates'
        ).fetchone()
        self.assertEqual(row[0], _SENTINEL_SOURCE)
        # Real intervals NEVER use the sentinel source.
        real_with_sentinel_source = self.conn.execute(
            'SELECT COUNT(*) FROM entity_dates '
            'WHERE start_ts > 0 AND extraction_source = ?',
            (_SENTINEL_SOURCE,)
        ).fetchone()[0]
        self.assertEqual(real_with_sentinel_source, 0)


class TestRegressionCases(unittest.TestCase):
    """Cases observed on the live brain that caused bugs in prior versions."""

    def test_no_year_2000_false_positive_from_technical_text(self):
        # Before the year-only pattern was disabled, "2000 chars" was tagged
        # as a year 2000 interval. 44 false positives on the live brain.
        for s in ["2000 chars of Sonnet reasoning",
                  "Encoder prompt: 2000 chars rejected — needs contract",
                  "Truncation at 2000ms"]:
            with self.subTest(input=s):
                self.assertEqual(_intervals(s), [])

    def test_real_brain_titles_extract_correctly(self):
        # These shapes were observed extracting correctly on the live brain
        cases = [
            ("Tom read 'The Nightingale' by Kristin Hannah — May 2023, 440 pages",
             '2023-05-01'),
            ("Glo project history — research, pivot, build phases (March 2026)",
             '2026-03-01'),
            ("Meta-learning: tmemory renamed to brain on March 17 2026",
             '2026-03-17'),
        ]
        for title, expected_start in cases:
            with self.subTest(title=title):
                ints = _intervals(title)
                self.assertTrue(ints,
                                f"Expected interval for: {title!r}")
                self.assertEqual(ints[0][0], expected_start)


if __name__ == '__main__':
    unittest.main()
