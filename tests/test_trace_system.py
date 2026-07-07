"""Unit tests for the fractal trace system.

Tests trace contract validation, TraceDAL methods, and data integrity.
All tests use IsolatedBrain — never touch production DB.

Run: python3 -m pytest tests/test_trace_system.py -v
"""
import json
import pytest
from tests.isolated_brain import IsolatedBrain


def _caller_meta(md):
    """Strip substrate identity stamps before comparing caller-provided
    metadata. TraceDAL injects agent_identity/human_identity onto every event
    when BRAIN_OPERATOR_NAME/BRAIN_AGENT_NAME are set (see set_identity); that
    behavior is covered by test_identity_stamping.py. These round-trip tests
    only assert the caller's own keys survive — and must stay env-independent."""
    return {k: v for k, v in (md or {}).items()
            if k not in ('agent_identity', 'human_identity')}


# ═══════════════════════════════════════════════════════
# C1: Contract validation
# ═══════════════════════════════════════════════════════

class TestTraceContract:
    """Verify trace_contract.py validates correctly."""

    def setup_method(self):
        from servers.trace_contract import validate_trace_event, SCALES, EVENT_TYPES, REF_TYPES
        self.validate = validate_trace_event
        self.SCALES = SCALES
        self.EVENT_TYPES = EVENT_TYPES
        self.REF_TYPES = REF_TYPES

    def test_known_good_s0(self):
        assert self.validate('s0', 'K', 'user_message') == (True, '')
        assert self.validate('s0', 'delta', 'assistant_message') == (True, '')
        assert self.validate('s0', 'delta', 'tool_result') == (True, '')

    def test_known_good_s1(self):
        assert self.validate('s1', 'O', 'recall') == (True, '')
        assert self.validate('s1', 'O', 'encoding_prompt') == (True, '')
        assert self.validate('s1', 'K', 'surface_selected') == (True, '')
        assert self.validate('s1', 'K', 'node_catalog') == (True, '')
        assert self.validate('s1', 'delta', 'additionalContext') == (True, '')
        assert self.validate('s1', 'delta', 'encoding_run') == (True, '')

    def test_rejects_bad_scale(self):
        ok, _ = self.validate('raw', 'K', '')
        assert not ok
        ok, _ = self.validate('turn', 'O', '')
        assert not ok
        ok, _ = self.validate('exchange', 'delta', '')
        assert not ok

    def test_rejects_bad_event_type(self):
        ok, _ = self.validate('s0', 'tool_call', '')
        assert not ok
        ok, _ = self.validate('s0', 'message', '')
        assert not ok
        ok, _ = self.validate('s1', 'recall', '')
        assert not ok
        # 'outcome' event_type was removed in the DAL Phase-A cleanup (the
        # dormant outcome arm) — it must now be rejected.
        ok, _ = self.validate('s1', 'outcome', 'correction')
        assert not ok

    def test_rejects_bad_ref_type(self):
        ok, _ = self.validate('s0', 'delta', 'Edit')
        assert not ok
        ok, _ = self.validate('s0', 'delta', 'Bash')
        assert not ok
        ok, _ = self.validate('s1', 'O', 'query')
        assert not ok

    def test_allows_empty_ref_type(self):
        """Empty ref_type is allowed — some events don't need it."""
        assert self.validate('s0', 'K', '') == (True, '')
        assert self.validate('s1', 'delta', '') == (True, '')

    def test_all_scales_defined(self):
        for s in ('s0', 's1', 's2', 's3', 's4'):
            assert s in self.SCALES, "Scale %s missing from SCALES" % s

    def test_all_event_types_defined(self):
        for et in ('O', 'K', 'delta'):
            assert et in self.EVENT_TYPES, "Event type %s missing" % et
        # 'outcome' removed in the DAL Phase-A cleanup (dormant outcome arm).
        assert 'outcome' not in self.EVENT_TYPES

    def test_s2_ref_types(self):
        """S2 (Graph) ref_types are valid."""
        assert self.validate('s2', 'O', 'graph_structure') == (True, '')
        assert self.validate('s2', 'O', 'graph_stats') == (True, '')
        # `dedup_scan` was renamed to `healer_scan` when the dedup unit
        # became the S2 healer (2026-04-19 era).
        assert self.validate('s2', 'O', 'healer_scan') == (True, '')
        assert self.validate('s2', 'O', 'correction_chains') == (True, '')
        assert self.validate('s2', 'K', 'community_partition') == (True, '')
        assert self.validate('s2', 'K', 'community_diff') == (True, '')
        assert self.validate('s2', 'K', 'stale_nodes') == (True, '')
        assert self.validate('s2', 'delta', 'community_created') == (True, '')
        assert self.validate('s2', 'delta', 'community_updated') == (True, '')
        assert self.validate('s2', 'delta', 'community_removed') == (True, '')
        assert self.validate('s2', 'delta', 'community_assignments') == (True, '')
        # `merge` was renamed/split when consolidation became its own unit:
        # `consolidated` (new node from merge), `evolved` (lineage edge),
        # `kept_distinct` (similar_to with no merge). Asserting consolidated
        # as the canonical post-merge ref_type.
        assert self.validate('s2', 'delta', 'consolidated') == (True, '')
        assert self.validate('s2', 'delta', 'confidence_adjust') == (True, '')

    def test_s3_ref_types(self):
        """S3 (Reasoning) ref_types are valid."""
        assert self.validate('s3', 'O', 'cluster_patterns') == (True, '')
        assert self.validate('s3', 'O', 'correction_trajectories') == (True, '')
        assert self.validate('s3', 'O', 'confidence_landscapes') == (True, '')
        assert self.validate('s3', 'K', 'cross_cluster') == (True, '')
        assert self.validate('s3', 'K', 'learning_curves') == (True, '')
        assert self.validate('s3', 'delta', 'abstract_insight') == (True, '')
        assert self.validate('s3', 'delta', 'resolved_question') == (True, '')
        assert self.validate('s3', 'delta', 'meta_optimization') == (True, '')

    def test_s4_ref_types(self):
        """S4 (Growth) ref_types are valid."""
        assert self.validate('s4', 'O', 'uncertainty_nodes') == (True, '')
        assert self.validate('s4', 'O', 'external_research') == (True, '')
        assert self.validate('s4', 'K', 'stale_decisions') == (True, '')
        assert self.validate('s4', 'delta', 'research_finding') == (True, '')
        assert self.validate('s4', 'delta', 'cross_project') == (True, '')


# ═══════════════════════════════════════════════════════
# C1: TraceDAL methods
# ═══════════════════════════════════════════════════════

class TestTraceDAL:
    """Verify TraceDAL read/write operations."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            yield

    def test_append_validates_contract(self):
        """append() rejects contract violations."""
        with pytest.raises(ValueError, match="Unknown scale"):
            self.dal.append(chain_id='test', scale='raw', event_type='K')

        with pytest.raises(ValueError, match="Unknown event_type"):
            self.dal.append(chain_id='test', scale='s0', event_type='tool_call')

    def test_append_metadata_guard_fires_loud_but_writes(self, capsys):
        """CR6 real-path: a malformed delta payload on a schema'd ref_type is
        caught at the DAL chokepoint (where inline S1/S2 writes actually go) —
        logged loud to stderr, NEVER blocked. The command-boundary check missed
        every in-process delta write; this is the guard that actually fires."""
        tid = self.dal.append(
            chain_id='cr6-dal-1', scale='s1', event_type='delta',
            ref_type='encoding_run', metadata={'created': []})  # missing keys
        assert tid                                   # non-blocking
        assert len(self.dal.get_chain('cr6-dal-1')) == 1   # written anyway
        err = capsys.readouterr().err
        assert 'trace metadata invalid' in err and 'encoding_run' in err  # loud

    def test_append_valid_delta_no_warning(self, capsys):
        from servers.trace_contract import build_delta_metadata
        self.dal.append(chain_id='cr6-dal-2', scale='s1', event_type='delta',
                        ref_type='encoding_run', metadata=build_delta_metadata())
        assert 'trace metadata invalid' not in capsys.readouterr().err

    def test_append_bare_marker_no_warning(self, capsys):
        # A delta ref_type doubling as a bare early-out marker (metadata=None)
        # must not warn — only a present payload is shape-checked.
        self.dal.append(chain_id='cr6-dal-3', scale='s1', event_type='delta',
                        ref_type='encoding_run', metadata=None, summary='no-op')
        assert 'trace metadata invalid' not in capsys.readouterr().err

    def test_append_writes_and_reads(self):
        """Write an event, read it back via get_chain."""
        self.dal.append(
            chain_id='test-chain-1', scale='s0', event_type='K',
            ref_type='user_message', ref_id='msg-123',
            summary='hello world', metadata={'content': 'full message'},
            session_id='test-session')

        chain = self.dal.get_chain('test-chain-1')
        assert len(chain) == 1
        event = chain[0]
        assert event['scale'] == 's0'
        assert event['event_type'] == 'K'
        assert event['ref_type'] == 'user_message'
        assert event['ref_id'] == 'msg-123'
        assert event['summary'] == 'hello world'
        assert _caller_meta(event['metadata']) == {'content': 'full message'}

    def test_metadata_roundtrip(self):
        """JSON metadata survives write → read cycle."""
        meta = {
            'candidates': [
                {'id': 'abc', 'title': 'Test node', 'score': 0.85},
                {'id': 'def', 'title': 'Another node', 'score': 0.72},
            ],
            'query': 'test query',
            'source': 'hook',
        }
        self.dal.append(
            chain_id='meta-test', scale='s1', event_type='O',
            ref_type='recall', metadata=meta, session_id='test')

        chain = self.dal.get_chain('meta-test')
        assert _caller_meta(chain[0]['metadata']) == meta
        assert chain[0]['metadata']['candidates'][0]['score'] == 0.85

    def test_chain_ordering(self):
        """Events in a chain are returned chronologically."""
        import time
        self.dal.append(chain_id='order-test', scale='s1', event_type='O',
                        ref_type='recall', summary='first')
        time.sleep(0.01)
        self.dal.append(chain_id='order-test', scale='s1', event_type='K',
                        ref_type='surface_selected', summary='second')
        time.sleep(0.01)
        self.dal.append(chain_id='order-test', scale='s1', event_type='delta',
                        ref_type='additionalContext', summary='third')

        chain = self.dal.get_chain('order-test')
        assert len(chain) == 3
        assert chain[0]['summary'] == 'first'
        assert chain[1]['summary'] == 'second'
        assert chain[2]['summary'] == 'third'

    def test_get_recent_filters_scale(self):
        """get_recent with scale filter only returns matching scale."""
        self.dal.append(chain_id='s0-test', scale='s0', event_type='K',
                        ref_type='user_message', summary='s0 event')
        self.dal.append(chain_id='s1-test', scale='s1', event_type='O',
                        ref_type='recall', summary='s1 event')

        s0_events = self.dal.get_recent(scale='s0', hours=1)
        s1_events = self.dal.get_recent(scale='s1', hours=1)

        s0_summaries = [e['summary'] for e in s0_events]
        s1_summaries = [e['summary'] for e in s1_events]

        assert 's0 event' in s0_summaries
        assert 's1 event' not in s0_summaries
        assert 's1 event' in s1_summaries

    def test_get_recent_session_id_authoritative_ignores_hours(self):
        """When session_id is set, get_recent must honor it regardless of the
        `hours` window. Regression: the old query_traces handler checked
        session_id but never passed it through — historical sessions older
        than the 24h default silently fell back to current-session events.
        """
        # Append events under two different session_ids
        self.dal.append(chain_id='session-A', scale='s0', event_type='K',
                        ref_type='user_message', summary='session-A event',
                        session_id='session-aaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
        self.dal.append(chain_id='session-B', scale='s0', event_type='K',
                        ref_type='user_message', summary='session-B event',
                        session_id='session-bbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')

        a_events = self.dal.get_recent(
            session_id='session-aaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
            hours=1, limit=10)
        a_summaries = [e['summary'] for e in a_events]
        assert 'session-A event' in a_summaries
        assert 'session-B event' not in a_summaries

        # Authoritative semantics: hours=0 (effectively no cutoff window)
        # must STILL return session events. session_id wins over hours.
        a_events_zero = self.dal.get_recent(
            session_id='session-aaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
            hours=0, limit=10)
        assert any(e['summary'] == 'session-A event' for e in a_events_zero), \
            "session_id must override hours=0 (was the silent-empty bug)"

    def test_get_recent_unknown_session_returns_empty(self):
        """When session_id has zero matches, get_recent returns []. The DAL
        also logs a loud warning to stderr (asserted via captured stderr in
        integration tests; unit test just verifies the return shape)."""
        out = self.dal.get_recent(
            session_id='nonexistent-session-id-zzzzzzzzzzzzzzzz',
            hours=1, limit=10)
        assert out == [], "unknown session_id must return [], not fall back"

    def test_get_recent_session_ids_plural_cross_session(self):
        """session_ids (plural list) returns events from any matching session,
        ignores hours, ordered by time. Useful for cross-session audits."""
        self.dal.append(chain_id='multi-A', scale='s0', event_type='K',
                        ref_type='user_message', summary='multi-A event',
                        session_id='multi-aaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
        self.dal.append(chain_id='multi-B', scale='s0', event_type='K',
                        ref_type='user_message', summary='multi-B event',
                        session_id='multi-bbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')
        self.dal.append(chain_id='multi-C', scale='s0', event_type='K',
                        ref_type='user_message', summary='multi-C event',
                        session_id='multi-ccc-cccc-cccc-cccc-cccccccccccc')

        out = self.dal.get_recent(
            session_ids=[
                'multi-aaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
                'multi-bbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
            ],
            hours=0, limit=10)
        summaries = {e['summary'] for e in out}
        assert 'multi-A event' in summaries
        assert 'multi-B event' in summaries
        assert 'multi-C event' not in summaries, \
            "C was NOT in session_ids, must be excluded"

    def test_get_recent_session_id_plural_and_singular_rejected(self):
        """Passing both session_id and session_ids must raise — ambiguous intent."""
        import pytest
        with pytest.raises(ValueError, match="not both"):
            self.dal.get_recent(
                session_id='single-id',
                session_ids=['multi-1', 'multi-2'])

    def test_get_recent_filters_event_type(self):
        """get_recent with event_type filter only returns matching type."""
        self.dal.append(chain_id='type-test', scale='s0', event_type='K',
                        ref_type='user_message', summary='K event')
        self.dal.append(chain_id='type-test', scale='s0', event_type='delta',
                        ref_type='assistant_message', summary='delta event')

        k_events = self.dal.get_recent(event_type='K', hours=1)
        assert any(e['summary'] == 'K event' for e in k_events)
        assert not any(e['summary'] == 'delta event' for e in k_events)

    def test_get_recent_respects_limit(self):
        """get_recent with limit returns at most that many events."""
        for i in range(10):
            self.dal.append(chain_id='limit-test-%d' % i, scale='s0',
                            event_type='K', ref_type='user_message',
                            summary='event %d' % i)

        events = self.dal.get_recent(scale='s0', hours=1, limit=5)
        assert len(events) <= 5

    def test_chains_scoped_to_session(self):
        """Chains are queryable scoped to a session, deduped by chain_id.

        (get_chains_for_session was removed in the DAL Phase-A cleanup; the live
        session-scoped chain query is get_chains(session_id=...).)
        """
        self.dal.append(chain_id='chain-a', scale='s0', event_type='K',
                        session_id='sess-1')
        self.dal.append(chain_id='chain-a', scale='s0', event_type='delta',
                        session_id='sess-1')
        self.dal.append(chain_id='chain-b', scale='s0', event_type='K',
                        session_id='sess-1')
        self.dal.append(chain_id='chain-c', scale='s0', event_type='K',
                        session_id='sess-2')

        chains = {c['chain_id'] for c in self.dal.get_chains(session_id='sess-1')}
        assert 'chain-a' in chains
        assert 'chain-b' in chains
        assert 'chain-c' not in chains


# ═══════════════════════════════════════════════════════
# A1: New query methods
# ═══════════════════════════════════════════════════════

class TestGetChains:
    """Verify get_chains returns grouped chains with events."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.dal = env.brain._trace_dal
            yield

    def _write_chain(self, chain_id, scale, session_id='sess-1', events=None):
        """Helper: write a complete O/K/delta chain."""
        events = events or [('O', 'recall'), ('K', 'surface_selected'), ('delta', 'additionalContext')]
        for et, rt in events:
            self.dal.append(chain_id=chain_id, scale=scale, event_type=et,
                            ref_type=rt, summary='%s %s' % (et, rt),
                            session_id=session_id)

    def test_returns_grouped(self):
        """Returns chains with nested events, not flat list."""
        self._write_chain('s1r-abc-1', 's1')
        result = self.dal.get_chains(session_id='sess-1', scale='s1')
        assert len(result) == 1
        assert result[0]['chain_id'] == 's1r-abc-1'
        assert len(result[0]['events']) == 3

    def test_filters_by_session(self):
        """Only returns chains from requested session."""
        self._write_chain('s1r-abc-1', 's1', session_id='sess-1')
        self._write_chain('s1r-def-1', 's1', session_id='sess-2')
        result = self.dal.get_chains(session_id='sess-1', scale='s1')
        chain_ids = [c['chain_id'] for c in result]
        assert 's1r-abc-1' in chain_ids
        assert 's1r-def-1' not in chain_ids

    def test_filters_by_scale(self):
        """Only returns chains from requested scale."""
        self._write_chain('s0-abc-1', 's0', events=[('K', 'user_message'), ('delta', 'assistant_message')])
        self._write_chain('s1r-abc-1', 's1')
        result = self.dal.get_chains(session_id='sess-1', scale='s1')
        assert all(c['chain_id'].startswith('s1') for c in result)

    def test_respects_limit(self):
        """Limit caps number of chains returned."""
        import time
        for i in range(10):
            self._write_chain('s1r-abc-%d' % i, 's1')
            time.sleep(0.01)
        result = self.dal.get_chains(session_id='sess-1', scale='s1', limit=3)
        assert len(result) <= 3

    def test_events_include_metadata(self):
        """Events within chains include parsed metadata."""
        self.dal.append(chain_id='meta-chain', scale='s1', event_type='O',
                        ref_type='recall', metadata={'query': 'test', 'count': 25},
                        session_id='sess-1')
        result = self.dal.get_chains(session_id='sess-1', scale='s1')
        assert _caller_meta(result[0]['events'][0]['metadata']) == {'query': 'test', 'count': 25}


class TestGetByRefType:
    """Verify get_by_ref_type filters correctly."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.dal = env.brain._trace_dal
            yield

    def test_filters_by_ref_type(self):
        """Only returns events with matching ref_type (seed-robust via tag —
        IsolatedBrain carries real encoding_run/recall traces)."""
        tag = 'reffilter_%s' % id(self)
        self.dal.append(chain_id='c1', scale='s1', event_type='delta',
                        ref_type='encoding_run', summary=tag + ' encoded')
        self.dal.append(chain_id='c2', scale='s1', event_type='O',
                        ref_type='recall', summary=tag + ' recalled')

        result = self.dal.get_by_ref_type('encoding_run')
        tagged = [r for r in result if tag in (r['summary'] or '')]
        assert len(tagged) == 1
        assert tagged[0]['summary'] == tag + ' encoded'

    def test_filters_by_scale(self):
        """Scale filter narrows results further."""
        tag = 'refscale_%s' % id(self)
        self.dal.append(chain_id='c1', scale='s0', event_type='delta',
                        ref_type='tool_result', summary=tag + ' s0 tool')
        self.dal.append(chain_id='c2', scale='s1', event_type='delta',
                        ref_type='additionalContext', summary=tag + ' s1 context')

        result = self.dal.get_by_ref_type('tool_result', scale='s0')
        tagged = [r for r in result if tag in r['summary']]
        assert len(tagged) == 1
        assert 's0 tool' in tagged[0]['summary']

    def test_respects_limit(self):
        for i in range(10):
            self.dal.append(chain_id='c%d' % i, scale='s1', event_type='delta',
                            ref_type='encoding_run', summary='item %d' % i)
        result = self.dal.get_by_ref_type('encoding_run', limit=3)
        assert len(result) <= 3

    def test_filters_by_session_id(self):
        """session_id scopes results to a single session — parallel sessions
        don't see each other's events. Required by surface dedup."""
        self.dal.append(chain_id='sA-1', scale='s1', event_type='K',
                        ref_type='surface_selected', ref_id='["nA"]',
                        session_id='sess-A')
        self.dal.append(chain_id='sB-1', scale='s1', event_type='K',
                        ref_type='surface_selected', ref_id='["nB"]',
                        session_id='sess-B')

        only_a = self.dal.get_by_ref_type(
            'surface_selected', scale='s1', hours=None, session_id='sess-A')
        only_b = self.dal.get_by_ref_type(
            'surface_selected', scale='s1', hours=None, session_id='sess-B')
        unscoped = self.dal.get_by_ref_type(
            'surface_selected', scale='s1', hours=None)

        # Per-session filter: each session sees only its own row.
        assert {r['ref_id'] for r in only_a} == {'["nA"]'}
        assert {r['ref_id'] for r in only_b} == {'["nB"]'}
        # Unscoped sees both (IsolatedBrain may also have seed traces — assert
        # both are present, not that the set equals exactly two).
        unscoped_ids = {r['ref_id'] for r in unscoped}
        assert {'["nA"]', '["nB"]'} <= unscoped_ids


class TestCountBy:
    """Verify count_by aggregation."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.dal = env.brain._trace_dal
            # Clear trace_events for exact count tests
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def test_count_by_event_type(self):
        """Counts events grouped by event_type."""
        self.dal.append(chain_id='c1', scale='s1', event_type='O', ref_type='recall')
        self.dal.append(chain_id='c1', scale='s1', event_type='K', ref_type='surface_selected')
        self.dal.append(chain_id='c1', scale='s1', event_type='delta', ref_type='additionalContext')
        self.dal.append(chain_id='c2', scale='s1', event_type='O', ref_type='recall')

        result = self.dal.count_by('event_type', scale='s1')
        assert result.get('O', 0) == 2
        assert result.get('K', 0) == 1
        assert result.get('delta', 0) == 1

    def test_count_by_ref_type(self):
        """Counts events grouped by ref_type (delta vs baseline — seed-robust,
        since IsolatedBrain may carry real encoding_run/additionalContext)."""
        before = self.dal.count_by('ref_type', scale='s1')
        self.dal.append(chain_id='c1', scale='s1', event_type='delta',
                        ref_type='encoding_run')
        self.dal.append(chain_id='c2', scale='s1', event_type='delta',
                        ref_type='encoding_run')
        self.dal.append(chain_id='c3', scale='s1', event_type='delta',
                        ref_type='additionalContext')

        after = self.dal.count_by('ref_type', scale='s1')
        assert after.get('encoding_run', 0) - before.get('encoding_run', 0) == 2
        assert after.get('additionalContext', 0) - before.get('additionalContext', 0) == 1

    def test_count_by_filters_scale(self):
        """Scale filter only counts matching events."""
        self.dal.append(chain_id='c1', scale='s0', event_type='K', ref_type='user_message')
        self.dal.append(chain_id='c2', scale='s1', event_type='O', ref_type='recall')

        result = self.dal.count_by('event_type', scale='s0')
        assert result.get('K', 0) == 1
        assert result.get('O', 0) == 0


# ═══════════════════════════════════════════════════════
# C1: get_session_turns
# ═══════════════════════════════════════════════════════

class TestGetSessionTurns:
    """Verify get_session_turns reconstructs conversation correctly."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            yield

    def _write_turn(self, session_id, stop, user_msg, assistant_msg,
                    surface_output='', recall_chain=''):
        """Helper: write one S0 turn + optional S1 recall delta."""
        chain = 's0-%s-%s' % (session_id[:8], stop)
        meta_k = {'content': user_msg}
        if recall_chain:
            meta_k['recall_chain'] = recall_chain
        self.dal.append(chain_id=chain, scale='s0', event_type='K',
                        ref_type='user_message', summary=user_msg[:200],
                        metadata=meta_k, session_id=session_id)
        self.dal.append(chain_id=chain, scale='s0', event_type='delta',
                        ref_type='assistant_message', summary=assistant_msg[:200],
                        metadata={'content': assistant_msg}, session_id=session_id)
        if surface_output and recall_chain:
            self.dal.append(chain_id=recall_chain, scale='s1', event_type='delta',
                            ref_type='additionalContext', summary='surfaced',
                            metadata={'content': surface_output}, session_id=session_id)

    def test_shape(self):
        """Returns list of dicts with expected keys."""
        self._write_turn('sess-1', '1', 'hello', 'hi there')
        turns = self.dal.get_session_turns('sess-1')
        assert len(turns) == 2
        assert turns[0]['role'] == 'user'
        assert turns[1]['role'] == 'assistant'
        for key in ('trace_id', 'content', 'timestamp', 'judge_output'):
            assert key in turns[0], "Missing key: %s" % key

    def test_chronological(self):
        """Turns are in chronological order."""
        import time
        self._write_turn('sess-2', '1', 'first message', 'first reply')
        time.sleep(0.01)
        self._write_turn('sess-2', '2', 'second message', 'second reply')
        turns = self.dal.get_session_turns('sess-2')
        user_msgs = [t['content'] for t in turns if t['role'] == 'user']
        assert user_msgs == ['first message', 'second message']

    def test_cross_reference_surface_output(self):
        """Surface output from S1 delta is cross-referenced via recall_chain."""
        recall_chain = 's1r-sess3333-5'
        self._write_turn('sess3333aabbccdd', '5', 'what is X?', 'X is Y',
                         surface_output='Brain recalled: node about X',
                         recall_chain=recall_chain)
        turns = self.dal.get_session_turns('sess3333aabbccdd')
        user_turn = [t for t in turns if t['role'] == 'user'][0]
        assert user_turn['judge_output'] == 'Brain recalled: node about X'

    def test_reads_content_from_metadata(self):
        """Content comes from metadata (full), not summary (truncated)."""
        long_msg = 'A' * 500
        self._write_turn('sess-4', '1', long_msg, 'reply')
        turns = self.dal.get_session_turns('sess-4')
        user_turn = [t for t in turns if t['role'] == 'user'][0]
        assert len(user_turn['content']) == 500

    def test_limit(self):
        """Respects limit parameter."""
        import time
        for i in range(10):
            self._write_turn('sess-5', str(i), 'msg %d' % i, 'reply %d' % i)
            time.sleep(0.01)
        turns = self.dal.get_session_turns('sess-5', limit=4)
        assert len(turns) <= 4

    def test_limit_returns_most_recent(self):
        """The SQL LIMIT must select the NEWEST turns, chronological order."""
        import time
        for i in range(6):
            self._write_turn('sess-6', str(i), 'msg %d' % i, 'reply %d' % i)
            time.sleep(0.01)
        turns = self.dal.get_session_turns('sess-6', limit=4)
        assert [t['content'] for t in turns] == [
            'msg 4', 'reply 4', 'msg 5', 'reply 5']

    def test_interrupted_turn_same_chain_keeps_all_user_messages(self):
        """An interrupted turn never fires Stop, so stop_counter doesn't
        advance and the NEXT prompt's user_message lands in the SAME s0
        chain. Every user message must survive — the old chain grouping
        kept one user slot per chain and silently overwrote the earlier
        prompt for every consumer (surface window, Scribe, historic
        lookups)."""
        import time
        sid = 'sess-interrupt-aabb'
        chain = 's0-%s-1' % sid[:8]
        self.dal.append(chain_id=chain, scale='s0', event_type='K',
                        ref_type='user_message', summary='first prompt',
                        metadata={'content': 'first prompt (interrupted)'},
                        session_id=sid)
        time.sleep(0.01)
        self.dal.append(chain_id=chain, scale='s0', event_type='K',
                        ref_type='user_message', summary='second prompt',
                        metadata={'content': 'second prompt'},
                        session_id=sid)
        time.sleep(0.01)
        self.dal.append(chain_id=chain, scale='s0', event_type='delta',
                        ref_type='assistant_message', summary='reply',
                        metadata={'content': 'reply'}, session_id=sid)
        turns = self.dal.get_session_turns(sid)
        assert [t['role'] for t in turns] == ['user', 'user', 'assistant']
        assert [t['content'] for t in turns] == [
            'first prompt (interrupted)', 'second prompt', 'reply']

    def test_with_judge_output_false_skips_cross_reference(self):
        """with_judge_output=False leaves judge_output empty on user turns
        (hot-path callers only read role/content)."""
        recall_chain = 's1r-sessjjjj-1'
        self._write_turn('sessjjjjaabbccdd', '1', 'what is X?', 'X is Y',
                         surface_output='Brain recalled: node about X',
                         recall_chain=recall_chain)
        turns = self.dal.get_session_turns('sessjjjjaabbccdd',
                                           with_judge_output=False)
        user_turn = [t for t in turns if t['role'] == 'user'][0]
        assert user_turn['judge_output'] == ''
