"""Conversation ownership comes from creation evidence, never time proximity.

Fresh isolated brains reproduce the two observed wrong-session shapes and pin
the boundary with general MCP trace/episode search. No model calls are needed.
"""
from unittest.mock import patch

from tests.brain_test_base import BrainTestBase
from servers.trace_contract import build_delta_metadata, build_node_created_metadata


class TestConversationProvenance(BrainTestBase):
    needs_embedder = False
    center = '2026-09-10T20:16:36.781948+00:00'

    def _trace(self, session, timestamp, ref_type='user_message', *,
               ref_id='', metadata=None, content='', scale='s0'):
        with patch('servers.dal_logs.iso_now', return_value=timestamp):
            return self.brain._trace_dal.append(
                chain_id='s0-%s-1' % session, scale=scale,
                event_type='K' if ref_type == 'user_message' else 'delta',
                ref_type=ref_type, ref_id=ref_id, session_id=session,
                summary=content, metadata=metadata)

    def _creation(self, node, session, timestamp=None):
        return self._trace(
            session, timestamp or self.center, 'node_created', ref_id=node,
            metadata=build_node_created_metadata(node_id=node, encoding_source='anchor'))

    def _run(self, session, timestamp=None, **metadata):
        delta = build_delta_metadata()
        delta.update(metadata)
        return self._trace(session, timestamp or self.center, 'encoding_run',
                           metadata=delta, scale='s1')

    def _ids(self, **kwargs):
        return [turn['trace_id'] for turn in self.brain.get_conversation_around(
            before=0, after=0, **kwargs)]

    def test_observed_authored_memories_resolve_to_their_recorded_sessions(self):
        s1e = '01a08172-63d2-7913-be97-3a99ac8d0830'
        host = '01a08cf2-be83-78b2-a59a-4c97a173859c'
        cases = (
            ('e1642b9a', s1e, host, '2026-09-11T17:50:12.995636+00:00'),
            ('7ed4d8ae', host, s1e, self.center),
        )
        for node, origin, other, timestamp in cases:
            with self.subTest(node=node):
                expected = self._trace(origin, timestamp, content='origin conversation')
                self._trace(other, timestamp, content='unrelated concurrent conversation')
                self._run(other, timestamp, created=['different-node'])
                self._creation(node, origin, timestamp)
                assert self._ids(node_id=node) == [expected]

    def test_encoded_origin_uses_exact_created_membership_and_run_center(self):
        self._creation('encoded-node', 'origin', '2026-09-10T20:00:00+00:00')
        self._trace('origin', '2026-09-10T20:00:00+00:00', content='earlier turn')
        expected = self._trace('origin', self.center, content='run context')
        self._trace('other', self.center, content='wrong conversation')
        self._run('other', created=['other-node'], revised=['encoded-node'],
                  journal_entry='Mentioning encoded-node does not mean creating it')
        self._run('origin', created=['encoded-node'])
        assert self._ids(node_id='encoded-node') == [expected]

    def test_encoding_run_alone_is_sufficient_creation_evidence(self):
        expected = self._trace('origin', self.center, content='encoded here')
        self._run('origin', created=['encoded-node'])
        assert self._ids(node_id='encoded-node') == [expected]

    def test_mentions_revisions_prefixes_and_non_array_created_are_not_origins(self):
        self._trace('other', self.center, content='tempting nearby conversation')
        for metadata in (
            {'created': ['node1234-longer']},
            {'created': 'node1234'},
            {'created': {'node1234': True}},
            {'created': [{'id': 'node1234'}]},
            {'created': None},
            {'revised': ['node1234']},
            {'journal_entry': 'node1234'},
        ):
            with self.subTest(metadata=metadata):
                self._run('other', **metadata)
                assert self._ids(node_id='node1234') == []

    def test_creation_ref_id_must_match_exactly(self):
        self._trace('other', self.center, content='wrong conversation')
        self._creation('node1234-longer', 'other')
        assert self._ids(node_id='node1234') == []

    def test_historic_malformed_metadata_cannot_establish_creation(self):
        trace = self._run('other')
        for raw in ('broken JSON', 'null', '["node1234"]', '"node1234"'):
            with self.subTest(raw=raw):
                # Raw legacy cells bypass the writer deliberately, in this
                # test's fresh database only. They must not break the reader.
                self.brain.logs_conn_w.execute(
                    'UPDATE trace_events SET metadata=? WHERE id=?', (raw, trace))
                self.brain.logs_conn_w.commit()
                assert self.brain._trace_dal.get_node_creation_traces('node1234') == []

    def test_conflicting_recorded_sessions_are_reported_without_guessing(self):
        self._trace('one', self.center, content='one')
        self._trace('two', self.center, content='two')
        self._creation('conflict', 'one')
        self._run('two', created=['conflict'])
        with patch.object(self.brain, '_log_error') as error:
            assert self._ids(node_id='conflict') == []
        assert error.call_count == 1
        assert 'conflicting sessions' in str(error.call_args.args[1])

    def test_sessionless_creation_cannot_select_a_nearby_session(self):
        self._trace('other', self.center, content='wrong conversation')
        self._creation('unscoped', '')
        assert self._ids(node_id='unscoped') == []

    def test_empty_explicit_or_resolved_session_stays_empty(self):
        self._trace('other', self.center, content='wrong conversation')
        self._creation('authored', 'empty')
        assert self._ids(session_id='empty', timestamp=self.center) == []
        assert self._ids(node_id='authored') == []

    def test_timestamp_alone_cannot_select_a_conversation(self):
        self._trace('other', self.center, content='wrong conversation')
        assert self._ids(timestamp=self.center) == []
        assert self.brain.get_conversation('') == []

    def test_explicit_session_and_timestamp_remain_authoritative(self):
        expected = self._trace('explicit', self.center, content='selected context')
        self._creation('authored', 'another')
        assert self._ids(node_id='authored', session_id='explicit',
                         timestamp=self.center) == [expected]

    def test_explicit_timestamp_positions_window_in_recorded_session(self):
        timestamp = '2026-09-10T20:00:00+00:00'
        expected = self._trace('origin', timestamp, content='requested earlier point')
        self._trace('origin', self.center, content='creation point')
        self._creation('authored', 'origin')
        assert self._ids(node_id='authored', timestamp=timestamp) == [expected]

    def test_recent_and_centered_readers_preserve_correspondents_and_order(self):
        first = self._trace('origin', '2026-09-10T20:00:00+00:00', content='question')
        second = self._trace('origin', self.center, 'assistant_message', content='answer')
        self._trace('other', self.center, content='not part of this conversation')
        recent = self.brain.get_conversation('origin', with_judge_output=False)
        centered = self.brain.get_conversation_around(
            session_id='origin', timestamp=self.center)
        assert [row['trace_id'] for row in centered] == [first, second]
        assert [row['ref_type'] for row in centered] == ['user_message', 'assistant_message']
        assert centered == [{k: v for k, v in row.items() if k != 'judge_output'}
                            for row in recent]

    def test_failed_origin_read_is_reported_without_fallback(self):
        self._trace('other', self.center, content='wrong conversation')
        with patch.object(self.brain._trace_dal, 'get_node_creation_traces',
                          side_effect=RuntimeError('origin read failed')):
            with patch.object(self.brain, '_log_error') as error:
                assert self._ids(node_id='authored') == []
        assert error.call_count == 1
        assert 'origin read failed' in str(error.call_args.args[1])

    def test_failed_conversation_read_uses_shared_error_reporting(self):
        with patch.object(self.brain._trace_dal, 'get_session_turns',
                          side_effect=RuntimeError('conversation read failed')):
            with patch.object(self.brain, '_log_error') as error:
                assert self._ids(session_id='origin', timestamp=self.center) == []
        assert error.call_args.args[0] == 'get_conversation'

    def test_general_trace_and_episode_search_remain_cross_session_and_time_filtered(self):
        from servers.clock import iso_now, iso_cutoff
        old = self._trace('old', iso_cutoff(days=30), content='search-boundary old')
        expected = {
            self._trace(session, iso_now(), content='search-boundary recent')
            for session in ('one', 'two')
        }
        traces = self.brain.query_traces(ref_type='user_message', hours=24)['events']
        episodes = self.brain.recall_episodes(
            contains='search-boundary', younger_than='1d')['episodes']
        assert {row['id'] for row in traces} == expected
        assert {row['id'] for row in episodes} == expected
        all_traces = self.brain.query_traces(ref_type='user_message', hours=None)['events']
        all_episodes = self.brain.recall_episodes(
            contains='search-boundary', younger_than='60d')['episodes']
        assert {row['id'] for row in all_traces} == expected | {old}
        assert {row['id'] for row in all_episodes} == expected | {old}
