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

from tests.brain_test_base import BrainTestBase
from tests.isolated_brain import IsolatedBrain
from servers.scales.s1.fetch_tools import (
    TOOL_DEFINITIONS,
    _parse_window,
    _parse_date_expr,
    recall_topical,
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
    def test_two_tools_defined(self):
        # Haiku's surface tool set, trimmed over time (functions kept callable +
        # tested below; just no longer offered to Haiku):
        #  - recall_by_aspect removed 2026-06-08 (query-blind, Frame-redundant)
        #  - expand_node removed 2026-06-12 (13/13 zero-result over 4 days)
        #  - recall_verbatim removed 2026-07-02 (name promised phrase-match, but
        #    the mechanism is a bag-of-words OR query; thin/redundant edge over
        #    recall_topical; episodic signal belongs to the LAF episodic lane)
        names = {t['name'] for t in TOOL_DEFINITIONS}
        assert names == {'recall_topical', 'recall_by_time'}

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


def _event_ids(b, event_type, source=None) -> set:
    """Set of debug_log row ids for an event_type (optionally a specific source).

    The discussed-anchor tests assert on what a single recall_by_time call
    WROTE — but the global COUNT(*) delta is not a safe proxy for that. The
    `env` brain copies the ~410MB production logs db (>50MB), so the first
    _log_error/_log_warning of the call triggers `Brain._check_logs_db_size`,
    which DELETEs every debug_log row older than 7 days mid-call. A global
    count then drifts by however many old rows happened to age past that
    cutoff since the baseline snapshot — nondeterministic, and observed to
    flip the sign of the delta day to day (proven flaky 2026-06-25).

    `debug_log.id` is INTEGER PRIMARY KEY AUTOINCREMENT, so ids are monotonic
    and never reused. Set-differencing this across a call yields EXACTLY the
    rows that call inserted, immune to the prune (a DELETE removes old, low
    ids; it can never produce a new one). event_type='error' is the loud,
    dashboard-surfaced severity; 'warning' is the quiet §6 signal — diffing
    these (not the deleted 'fetch_by_time_archived_leak' string, which now has
    zero writers) is what keeps the 'quiet' check non-tautological."""
    if source is not None:
        rows = b.logs_conn.execute(
            "SELECT id FROM debug_log WHERE event_type=? AND source=?",
            (event_type, source)).fetchall()
    else:
        rows = b.logs_conn.execute(
            "SELECT id FROM debug_log WHERE event_type=?",
            (event_type,)).fetchall()
    return {r[0] for r in rows}


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

    def test_score_maps_effective_activation(self, env):
        """Score-contract pin: the tool's score MUST be recall_score(result)
        (= effective_activation), the same field daemon_hooks gives the
        cosine pool. When these fork, the admission floor in surface.py
        silently drops 100% of fetches (the 3-week death of 2026-07)."""
        real_recall = env.brain.recall
        try:
            env.brain.recall = lambda **kw: {'results': [
                {'id': 'aaaa1111', 'title': 't1', 'effective_activation': 0.42},
                {'id': 'bbbb2222', 'title': 't2', 'effective_activation': 0.17},
            ]}
            out = recall_topical(env.brain, query='anything', k=5)
        finally:
            env.brain.recall = real_recall
        assert [c['score'] for c in out] == [0.42, 0.17]

    def test_zero_raw_results_logs_warning(self, env):
        """recall_topical returning 0 raw results means the tool broke, not
        'no matches' — cosine top-k always fills k. Must warn loudly."""
        before = _event_ids(env.brain, 'warning', source='fetch_topical_zero_raw')
        real_recall = env.brain.recall
        try:
            env.brain.recall = lambda **kw: {'results': []}
            out = recall_topical(env.brain, query='anything', k=5)
        finally:
            env.brain.recall = real_recall
        assert out == []
        after = _event_ids(env.brain, 'warning', source='fetch_topical_zero_raw')
        assert len(after - before) == 1


class TestScoreContractSync:
    """Both score producers must route through surface_contract.recall_score.

    Source-level pin: a refactor that reverts either side to a raw field
    read re-opens the silent-drift class this function exists to close."""

    def _source(self, relpath):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, relpath)) as f:
            return f.read()

    def test_daemon_hooks_uses_recall_score(self):
        src = self._source('servers/daemon_hooks.py')
        assert 'node_data["score"] = recall_score(r)' in src

    def test_fetch_tools_uses_recall_score(self):
        src = self._source('servers/scales/s1/fetch_tools.py')
        assert 'recall_score(r)' in src
        # The old drift form must not come back.
        assert "r.get('score', 0.0)" not in src

    def test_encode_stub_ranking_uses_recall_score(self):
        # third producer (associated-stub turnmax) — same pin, same drift class
        src = self._source('servers/scales/s1/encode.py')
        assert 'recall_score(h)' in src
        assert "h.get('effective_activation')" not in src


class TestRecallByTimeDiscussed:
    # recall_recent was DELETED 2026-06-12 — its use case ('the thing we
    # talked about 3 weeks ago') moved to recall_by_time(time_anchor=
    # 'discussed'), which reads surface-selection traces (s1r K events).
    def test_discussed_anchor_finds_traced_node(self, env):
        import json as _json
        b = env.brain
        rows = b.conn.execute("SELECT id FROM nodes LIMIT 1").fetchall()
        assert rows, "fixture brain has no nodes"
        node_id = rows[0][0]
        # Far-past window so ONLY this synthetic trace matches — a rolling
        # window ('yesterday') competes with real production traces in the
        # copied logs db and the limit cut can drop the test node.
        b.logs_conn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, ref_id, summary, session_id, created_at) "
            "VALUES ('te5td15c', 's1r-test-1', 's1', 'K', 'surface_selected', ?, 'test', 'test-sess', "
            "'2009-01-15T12:00:00+00:00')",
            (_json.dumps([node_id]),))
        b.logs_conn.commit()
        results = recall_by_time(b, start_when='January 2009',
                                 end_when='February 2009', time_anchor='discussed')
        ids = {r.get('id') for r in results}
        assert node_id in ids, "discussed anchor should surface the traced node"

    def test_discussed_anchor_empty_window_no_crash(self, env):
        results = recall_by_time(env.brain, start_when='January 2001',
                                 end_when='February 2001', time_anchor='discussed')
        assert isinstance(results, list)

    def test_discussed_anchor_redirects_archived_to_survivor(self, env):
        # TRACE-NODE-RESOLUTION site #1: a trace can point at a node S2
        # absorbed since. The anchor must resolve forward to the live
        # survivor (the thing we discussed survives as its descendant) — the
        # survivor surfaces, the dead id does not — and do it QUIETLY (the
        # old loud `fetch_by_time_archived_leak` stopgap is gone; §6).
        import json as _json
        b = env.brain
        live = b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 LIMIT 1").fetchone()[0]
        src = b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 AND id != ? "
            "LIMIT 1", (live,)).fetchone()[0]
        # Absorb `src` into `live`: archive it + stamp the survivor pointer.
        b.conn.execute("UPDATE nodes SET archived=1 WHERE id=?", (src,))
        b.conn.execute(
            "INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) "
            "VALUES (?, '_sys_archived_survivor_id', ?)", (src, live))
        b.conn.commit()
        # Isolated far-past window so only this synthetic trace matches.
        b.logs_conn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, ref_id, summary, session_id, created_at) "
            "VALUES ('te5tredir', 's1r-test-3', 's1', 'K', 'surface_selected', ?, 'test', 'test-sess', "
            "'2007-05-15T12:00:00+00:00')",
            (_json.dumps([src]),))
        b.logs_conn.commit()
        # `live in ids` is the positive control (get_node is a raw point
        # lookup that does NOT itself resolve survivors, so `live` can only
        # appear if resolve_live actually redirected). Quietness = no loud
        # error row added by this call (event_type='error' is what the
        # dashboard surfaces); a routine redirect logs nothing.
        err_before = _event_ids(b, 'error')
        results = recall_by_time(b, start_when='May 2007',
                                 end_when='June 2007', time_anchor='discussed')
        ids = {r.get('id') for r in results}
        assert src not in ids, "absorbed source must not surface"
        assert live in ids, "survivor must surface in place of the absorbed node"
        new_errs = _event_ids(b, 'error') - err_before
        assert not new_errs, \
            "routine redirect must not write a loud error row (new ids=%s)" % sorted(new_errs)

    def test_discussed_anchor_drops_orphan_quietly(self, env):
        # An archived node with NO survivor pointer is a true orphan: dropped,
        # never surfaced, with no loud error — but COUNTED via a low-severity
        # warning (§6). `ctrl` is a LIVE sibling in the same trace: the
        # positive control. It must still surface, so the test fails if
        # resolve_live returns nothing (rather than passing vacuously on an
        # empty result, e.g. a swallowed crash).
        import json as _json
        b = env.brain
        src = b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 LIMIT 1").fetchone()[0]
        ctrl = b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 AND id != ? "
            "LIMIT 1", (src,)).fetchone()[0]
        b.conn.execute("UPDATE nodes SET archived=1 WHERE id=?", (src,))
        b.conn.execute(
            "DELETE FROM node_metadata_kv WHERE node_id=? "
            "AND key='_sys_archived_survivor_id'", (src,))
        b.conn.commit()
        b.logs_conn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, ref_id, summary, session_id, created_at) "
            "VALUES ('te5torph', 's1r-test-4', 's1', 'K', 'surface_selected', ?, 'test', 'test-sess', "
            "'2006-07-15T12:00:00+00:00')",
            (_json.dumps([src, ctrl]),))
        b.logs_conn.commit()
        err_before = _event_ids(b, 'error')
        warn_before = _event_ids(b, 'warning', 'fetch_by_time_orphans_dropped')
        results = recall_by_time(b, start_when='July 2006',
                                 end_when='August 2006', time_anchor='discussed')
        ids = {r.get('id') for r in results}
        assert src not in ids, "true orphan must be dropped"
        assert ctrl in ids, "live sibling must still surface (positive control)"
        new_errs = _event_ids(b, 'error') - err_before
        assert not new_errs, \
            "orphan drop must not write a loud error row (new ids=%s)" % sorted(new_errs)
        new_warns = _event_ids(b, 'warning', 'fetch_by_time_orphans_dropped') - warn_before
        assert new_warns, \
            "orphan drop must be COUNTED via a low-severity warning (§6)"


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


class TestExpandNodeEndToEnd(BrainTestBase):
    """expand_node against a small fresh brain with a known seed→neighbor
    constellation. Regression for the traverse() shape bug: traverse returns
    {'neighbors': [...], 'corrections': {...}, 'metadata': {...}} but
    expand_node iterated the dict itself (string keys → AttributeError,
    swallowed by the except) — so the tool returned 0 candidates on every
    production call (finding id:18e4a651)."""
    needs_embedder = False

    def _node(self, title):
        r = self.brain.remember(type='fact', title=title, content='content of %s' % title,
                                encoding_source='anchor')
        return r['id']

    def setUp(self):
        super().setUp()
        self.seed = self._node('seed node about daemon restarts')
        self.nb1 = self._node('neighbor one — restart ladder')
        self.nb2 = self._node('neighbor two — launchd kickstart')
        self.brain.connect(self.seed, self.nb1, relation='extends', weight=0.8)
        self.brain.connect(self.seed, self.nb2, relation='grounds', weight=0.6)

    def test_expand_by_id_returns_neighbor_candidates(self):
        results = expand_node(self.brain, node_ref=self.seed, hops=1)
        ids = {r['id'] for r in results}
        self.assertEqual(ids, {self.nb1, self.nb2})
        for r in results:
            self.assertEqual(r['source_tool'], 'expand_node')
            self.assertEqual(r['discovery'], 'expand_node')
            self.assertTrue(r.get('title'))

    def test_expand_by_title_returns_neighbor_candidates(self):
        results = expand_node(self.brain, node_ref='seed node about daemon restarts',
                              hops=1)
        ids = {r['id'] for r in results}
        self.assertEqual(ids, {self.nb1, self.nb2})

    def test_expand_via_execute_tool_dispatch(self):
        # Full path: dispatch → tool → boundary enrichment. No error key,
        # and no fetch_expand_node row logged (the old failure mode logged
        # one per call while returning []).
        out = execute_tool(self.brain, 'expand_node',
                           {'node_ref': self.seed, 'hops': 1})
        self.assertNotIn('error', out)
        ids = {r['id'] for r in out['results']}
        self.assertEqual(ids, {self.nb1, self.nb2})
        self.assertFalse(_event_ids(self.brain, 'error', source='fetch_expand_node'))


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

    def test_by_time_dispatch_without_session(self, env):
        # recall_recent (and its session_id special-case) deleted 2026-06-12.
        result = execute_tool(env.brain, 'recall_by_time',
                              {'start_when': 'yesterday', 'time_anchor': 'discussed'},
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
