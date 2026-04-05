"""Unit tests for the fractal trace system.

Tests trace contract validation, TraceDAL methods, and data integrity.
All tests use IsolatedBrain — never touch production DB.

Run: python3 -m pytest tests/test_trace_system.py -v
"""
import json
import pytest
from tests.isolated_brain import IsolatedBrain


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
        assert self.validate('s1', 'K', 'judge_selected') == (True, '')
        assert self.validate('s1', 'K', 'node_catalog') == (True, '')
        assert self.validate('s1', 'delta', 'additionalContext') == (True, '')
        assert self.validate('s1', 'delta', 'encoding_run') == (True, '')
        assert self.validate('s1', 'outcome', 'correction') == (True, '')
        assert self.validate('s1', 'outcome', 'recall_hit') == (True, '')

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
        for et in ('O', 'K', 'delta', 'outcome'):
            assert et in self.EVENT_TYPES, "Event type %s missing" % et


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
        assert event['metadata'] == {'content': 'full message'}

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
        assert chain[0]['metadata'] == meta
        assert chain[0]['metadata']['candidates'][0]['score'] == 0.85

    def test_chain_ordering(self):
        """Events in a chain are returned chronologically."""
        import time
        self.dal.append(chain_id='order-test', scale='s1', event_type='O',
                        ref_type='recall', summary='first')
        time.sleep(0.01)
        self.dal.append(chain_id='order-test', scale='s1', event_type='K',
                        ref_type='judge_selected', summary='second')
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

    def test_get_chains_for_session(self):
        """get_chains_for_session returns unique chain IDs."""
        self.dal.append(chain_id='chain-a', scale='s0', event_type='K',
                        session_id='sess-1')
        self.dal.append(chain_id='chain-a', scale='s0', event_type='delta',
                        session_id='sess-1')
        self.dal.append(chain_id='chain-b', scale='s0', event_type='K',
                        session_id='sess-1')
        self.dal.append(chain_id='chain-c', scale='s0', event_type='K',
                        session_id='sess-2')

        chains = self.dal.get_chains_for_session('sess-1')
        assert 'chain-a' in chains
        assert 'chain-b' in chains
        assert 'chain-c' not in chains

    def test_append_outcome(self):
        """append_outcome writes an outcome event to existing chain."""
        self.dal.append(chain_id='outcome-test', scale='s1', event_type='O',
                        ref_type='recall', summary='original')
        self.dal.append_outcome(
            chain_id='outcome-test', scale='s1',
            ref_type='correction', ref_id='node-123',
            summary='Tom corrected this')

        chain = self.dal.get_chain('outcome-test')
        assert len(chain) == 2
        assert chain[1]['event_type'] == 'outcome'
        assert chain[1]['ref_type'] == 'correction'


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
                    judge_output='', recall_chain=''):
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
        if judge_output and recall_chain:
            self.dal.append(chain_id=recall_chain, scale='s1', event_type='delta',
                            ref_type='additionalContext', summary='surfaced',
                            metadata={'content': judge_output}, session_id=session_id)

    def test_shape(self):
        """Returns list of dicts with expected keys."""
        self._write_turn('sess-1', '1', 'hello', 'hi there')
        turns = self.dal.get_session_turns('sess-1')
        assert len(turns) == 2
        assert turns[0]['role'] == 'user'
        assert turns[1]['role'] == 'assistant'
        for key in ('content', 'timestamp', 'signal', 'judge_output', 'recalled_raw'):
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

    def test_cross_reference_judge_output(self):
        """Judge output from S1 delta is cross-referenced via recall_chain."""
        recall_chain = 's1r-sess3333-5'
        self._write_turn('sess3333aabbccdd', '5', 'what is X?', 'X is Y',
                         judge_output='Brain recalled: node about X',
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
