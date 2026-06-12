"""Unit tests for servers/scales/s1/fetch_tools.py.

Each tool is exercised against an IsolatedBrain (copy of production data).
Tests verify:
  - Tool returns the expected candidate shape (id/title/type/score/source_tool).
  - Tool handles edge cases (empty query, unknown aspect, no FTS5 hits).
  - Natural-language parsing produces reasonable date windows.
"""
import os
from datetime import datetime, timedelta, timezone

import pytest

from tests.isolated_brain import IsolatedBrain
from servers.scales.s1.fetch_tools import (
    TOOL_DEFINITIONS,
    _parse_window,
    _parse_date_expr,
    recall_topical,
    recall_recent,
    recall_by_time,
    recall_verbatim,
    recall_by_aspect,
    expand_node,
    execute_tool,
    format_tool_result_for_haiku,
)


# ─── Natural-language parsers (pure, no brain needed) ────────────────────

class TestWindowParser:
    def test_last_n_hours(self):
        now = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
        start, end = _parse_window('last 10 hours', now=now)
        assert end == now
        assert start == now - timedelta(hours=10)

    def test_last_nh_shorthand(self):
        now = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
        start, end = _parse_window('24h', now=now)
        assert start == now - timedelta(hours=24)

    def test_yesterday(self):
        now = datetime(2026, 5, 10, 14, 30, tzinfo=timezone.utc)
        start, end = _parse_window('yesterday', now=now)
        assert start == datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc)

    def test_today(self):
        now = datetime(2026, 5, 10, 14, 30, tzinfo=timezone.utc)
        start, end = _parse_window('today', now=now)
        assert start == datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc)
        assert end == now

    def test_last_session_heuristic(self):
        now = datetime(2026, 5, 10, 14, 30, tzinfo=timezone.utc)
        start, end = _parse_window('since last session', now=now)
        # Heuristic: last 24h
        assert (now - start).total_seconds() == 24 * 3600

    def test_unknown_falls_back_to_24h(self):
        now = datetime(2026, 5, 10, 14, 30, tzinfo=timezone.utc)
        start, end = _parse_window('blah random', now=now)
        # 24h is the conservative fallback
        assert (now - start).total_seconds() == 24 * 3600


class TestDateParser:
    def test_since_iso(self):
        since, until = _parse_date_expr('since 2026-05-01')
        assert since is not None
        assert since.year == 2026
        assert since.month == 5
        assert since.day == 1

    def test_before_iso(self):
        since, until = _parse_date_expr('before 2026-04-30')
        assert until is not None
        assert until.year == 2026
        assert until.month == 4
        assert until.day == 30

    def test_on_specific_day(self):
        since, until = _parse_date_expr('on 2026-05-09')
        assert since.day == 9
        assert until.day == 10  # day after

    def test_dict_form(self):
        since, until = _parse_date_expr({'since': '2026-05-01', 'until': '2026-05-10'})
        assert since.day == 1
        assert until.day == 10


# ─── Tool definitions are well-formed ────────────────────────────────────

class TestToolDefinitions:
    def test_four_tools_defined(self):
        # recall_by_aspect removed from Haiku's tool set 2026-06-08
        # (query-blind, redundant with Frame). expand_node removed 2026-06-12
        # (production audit: 13/13 zero-result calls over 4 days — 100% no-op
        # that still triggered the second Haiku round). Both functions still
        # exist and are tested below; they're just no longer offered to Haiku.
        names = {t['name'] for t in TOOL_DEFINITIONS}
        assert names == {
            'recall_topical', 'recall_recent', 'recall_by_time',
            'recall_verbatim',
        }

    def test_each_tool_has_input_schema(self):
        for tool in TOOL_DEFINITIONS:
            assert 'input_schema' in tool
            assert tool['input_schema']['type'] == 'object'
            assert 'properties' in tool['input_schema']

    def test_descriptions_substantive(self):
        for tool in TOOL_DEFINITIONS:
            assert len(tool['description']) > 80, \
                f"{tool['name']} description too short"


# ─── Tool execution against IsolatedBrain ────────────────────────────────

@pytest.fixture(scope='module')
def env():
    with IsolatedBrain() as env:
        yield env


class TestRecallTopical:
    def test_returns_candidates(self, env):
        results = recall_topical(env.brain, query='partnership target function', k=5)
        assert isinstance(results, list)
        # Live brain should have nodes; may return fewer than 5
        for r in results:
            assert 'id' in r
            assert 'title' in r
            assert 'score' in r
            assert r['source_tool'] == 'recall_topical'

    def test_empty_query_handled(self, env):
        results = recall_topical(env.brain, query='', k=5)
        assert isinstance(results, list)
        # Empty query must not crash AND must not yield malformed candidates:
        # anything returned still carries the tool's source tag.
        for r in results:
            assert r['source_tool'] == 'recall_topical'


class TestRecallRecent:
    def test_last_10h_returns_recent(self, env):
        results = recall_recent(env.brain, window='last 10 hours', k=10)
        assert isinstance(results, list)
        for r in results:
            assert r['source_tool'] == 'recall_recent'

    def test_unknown_window_falls_back(self, env):
        # 'blah' falls back to last 24h (the parse itself is locked by
        # TestWindowParser.test_unknown_falls_back_to_24h). Here we assert the
        # tool still produces well-tagged candidates off that fallback window.
        results = recall_recent(env.brain, window='blah random', k=5)
        assert isinstance(results, list)
        for r in results:
            assert r['source_tool'] == 'recall_recent'


class TestRecallByTime:
    def test_open_ended_start_returns_results(self, env):
        # recall_by_date(when='since 2026-04-01') → recall_by_time with an
        # open-ended upper bound. Needs a query for tiers 1/2; time-only is tier 3.
        results = recall_by_time(env.brain, start_when='2026-04-01',
                                 query='partnership', limit=10)
        assert isinstance(results, list)
        for r in results:
            assert r['source_tool'] == 'recall_by_time'

    def test_bounded_range(self, env):
        # recall_by_date(when={'since':..., 'until':...}) → start_when/end_when.
        results = recall_by_time(env.brain, start_when='2026-04-01',
                                 end_when='2026-05-10', query='partnership', limit=5)
        assert isinstance(results, list)
        for r in results:
            assert r['source_tool'] == 'recall_by_time'


class TestRecallVerbatim:
    def test_known_phrase_returns_hits(self, env):
        # Search for a phrase likely to exist in the brain
        results = recall_verbatim(env.brain, phrase='partnership', k=5)
        assert isinstance(results, list)
        for r in results:
            assert r['source_tool'] == 'recall_verbatim'

    def test_nonsense_phrase_returns_empty(self, env):
        # recall_verbatim is exact-match FTS5 — a made-up token has no hits, so
        # the result is strictly empty (tightened from the old "== [] OR all(...)").
        results = recall_verbatim(env.brain, phrase='zxqwerty_unlikely_phrase_98765', k=5)
        assert results == []


class TestRecallByAspect:
    def test_identity_bearing_returns_nodes(self, env):
        results = recall_by_aspect(env.brain, aspect='identity_bearing', k=10)
        assert isinstance(results, list)
        for r in results:
            assert r['source_tool'] == 'recall_by_aspect'

    def test_unknown_aspect_returns_empty(self, env):
        results = recall_by_aspect(env.brain, aspect='not_a_real_aspect', k=5)
        assert results == []

    def test_edge_only_aspect_returns_empty(self, env):
        # 'temporal_sequence' is edge-only (no node types)
        results = recall_by_aspect(env.brain, aspect='temporal_sequence', k=5)
        assert results == []


class TestExpandNode:
    def test_unknown_ref_returns_empty(self, env):
        results = expand_node(env.brain, node_ref='zzzzzzzz_not_a_node', hops=1)
        assert results == []


# ─── Dispatch + format helpers ───────────────────────────────────────────

class TestExecuteDispatch:
    def test_known_tool(self, env):
        result = execute_tool(env.brain, 'recall_topical',
                              {'query': 'brain', 'k': 3})
        assert 'results' in result
        assert 'latency_ms' in result
        assert isinstance(result['results'], list)

    def test_unknown_tool_returns_error(self, env):
        result = execute_tool(env.brain, 'nonexistent_tool', {})
        assert result['results'] == []
        assert 'error' in result
        assert 'unknown_tool' in result['error']

    def test_session_id_injected_for_recall_recent(self, env):
        # Should not raise even without session_id parameter
        result = execute_tool(env.brain, 'recall_recent',
                              {'window': 'last 10 hours', 'k': 3},
                              session_id='test-session')
        assert 'results' in result


class TestFormatToolResult:
    def test_format_with_results(self):
        result = {'results': [
            {'id': 'abc12345', 'title': 'Test node', 'type': 'lesson', 'score': 0.85},
            {'id': 'def67890', 'title': 'Another', 'type': 'principle', 'score': 0.72},
        ]}
        s = format_tool_result_for_haiku(result)
        assert '2 results' in s
        assert 'abc12345' in s
        assert 'Test node' in s

    def test_format_empty(self):
        s = format_tool_result_for_haiku({'results': []})
        assert 'No results' in s

    def test_format_error(self):
        s = format_tool_result_for_haiku({'results': [], 'error': 'boom'})
        assert 'ERROR' in s
        assert 'boom' in s
