"""Tests for S1 encode data assembly functions.

Tests the pure data assembly layer that prepares input for the S1 encoder LLM:
- _gather_messages: reads S0 traces into the format expected by the encoder
- build_node_catalog: extracts and formats nodes from surface outputs
- _save_journal: appends encoding run entries to session-scoped journal
- _save_session_context: extracts SESSION_CONTEXT from encoder output

No LLM calls — these are all deterministic data transformations.

Run: python3 -m pytest tests/test_s1_data_assembly.py -v
"""
import json
import pytest
from datetime import datetime, timezone
from tests.isolated_brain import IsolatedBrain


def _insert_hex_node(conn, node_id, node_type='rule', title='Test node', content='Test content.'):
    """Insert a synthetic node with a hex ID (matches build_node_catalog's regex)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO nodes (id, type, title, content, confidence, locked, archived, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 0.9, 0, 0, ?, ?)",
        (node_id, node_type, title, content, now, now))
    conn.commit()


class TestGatherMessages:
    """Tests for _gather_messages(brain, session_id) — trace-based message retrieval."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            # Clean slate for traces
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def test_empty_session_returns_empty(self):
        """No traces for session -> empty list."""
        from servers.scales.s1.encode import _gather_messages
        result = _gather_messages(self.brain, 'nonexistent-session-xyz')
        assert result == []

    def test_returns_turns_from_s0_traces(self):
        """S0 user+assistant traces -> list of turn dicts with correct shape."""
        from servers.scales.s1.encode import _gather_messages
        session_id = 'test-gather-001'

        # Write S0 traces: a user message and an assistant response
        chain = 's0-testgath-1'
        self.dal.append(
            chain_id=chain, scale='s0', event_type='K',
            ref_type='user_message', ref_id='',
            summary='How does recall work?',
            metadata={'content': 'How does recall work?', 'recall_chain': 's1r-testgath-1'},
            session_id=session_id)
        self.dal.append(
            chain_id=chain, scale='s0', event_type='delta',
            ref_type='assistant_message', ref_id='',
            summary='Recall uses embeddings...',
            metadata={'content': 'Recall uses embeddings to find similar nodes.'},
            session_id=session_id)
        self.brain.logs_conn.commit()

        result = _gather_messages(self.brain, session_id)
        assert len(result) == 2

        user_turn = result[0]
        assert user_turn['role'] == 'user'
        assert 'recall' in user_turn['content'].lower()
        assert 'id' in user_turn  # _gather_messages adds id field

        asst_turn = result[1]
        assert asst_turn['role'] == 'assistant'
        assert 'embeddings' in asst_turn['content'].lower()

    def test_respects_max_messages_limit(self):
        """When more turns exist than max_messages, only the most recent are returned."""
        from servers.scales.s1.encode import _gather_messages
        from servers.scales.s1.encode_contract import ENCODING_AGENT

        session_id = 'test-gather-limit'
        max_msgs = ENCODING_AGENT['max_messages']

        # Write more turns than the limit (each turn = user + assistant = 2 messages)
        num_turns = (max_msgs // 2) + 5
        for i in range(num_turns):
            chain = 's0-testlim-%d' % i
            self.dal.append(
                chain_id=chain, scale='s0', event_type='K',
                ref_type='user_message', ref_id='',
                summary='Question %d' % i,
                metadata={'content': 'Question number %d about the brain' % i},
                session_id=session_id)
            self.dal.append(
                chain_id=chain, scale='s0', event_type='delta',
                ref_type='assistant_message', ref_id='',
                summary='Answer %d' % i,
                metadata={'content': 'Answer to question %d' % i},
                session_id=session_id)
        self.brain.logs_conn.commit()

        result = _gather_messages(self.brain, session_id)
        assert len(result) <= max_msgs, (
            "Should respect max_messages=%d, got %d" % (max_msgs, len(result)))

    def test_content_truncated_to_limit(self):
        """Long message content is truncated to message_content_limit."""
        from servers.scales.s1.encode import _gather_messages
        from servers.scales.s1.encode_contract import ENCODING_AGENT

        session_id = 'test-gather-trunc'
        content_limit = ENCODING_AGENT['message_content_limit']
        long_content = 'x' * (content_limit + 500)

        chain = 's0-testtrn-1'
        self.dal.append(
            chain_id=chain, scale='s0', event_type='K',
            ref_type='user_message', ref_id='',
            summary='Long message',
            metadata={'content': long_content},
            session_id=session_id)
        self.brain.logs_conn.commit()

        result = _gather_messages(self.brain, session_id)
        assert len(result) >= 1
        assert len(result[0]['content']) <= content_limit


class TestBuildNodeCatalog:
    """Tests for build_node_catalog(surface_outputs, brain) — catalog from surface selections."""

    NODE_A = 'aa11bb22'
    NODE_B = 'cc33dd44'
    NODE_C = 'ee55ff66'

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.conn = env.brain.conn
            # Insert synthetic nodes with hex IDs (matches the regex in build_node_catalog)
            _insert_hex_node(self.conn, self.NODE_A, 'rule', 'Test catalog rule',
                             'This rule governs recall behavior.')
            _insert_hex_node(self.conn, self.NODE_B, 'lesson', 'Test catalog lesson',
                             'Learned from encoding experiments.')
            yield

    def test_empty_surface_outputs(self):
        """No surface outputs -> empty catalog."""
        from servers.scales.s1.encode_contract import build_node_catalog
        text, ids = build_node_catalog([], self.brain)
        assert text == ''
        assert ids == set()

    def test_none_and_no_selection_skipped(self):
        """None and '(no selection)' entries are ignored."""
        from servers.scales.s1.encode_contract import build_node_catalog
        text, ids = build_node_catalog([None, '(no selection)', None], self.brain)
        assert text == ''
        assert ids == set()

    def test_extracts_node_ids_from_surface_output(self):
        """Extracts id:XXXXXXXX patterns from surface output strings."""
        from servers.scales.s1.encode_contract import build_node_catalog

        surface_output = '[rule] "Test catalog rule" (id:%s, conf:0.9)' % self.NODE_A
        text, ids = build_node_catalog([surface_output], self.brain)

        assert self.NODE_A in ids
        assert 'Node Catalog' in text

    def test_deduplicates_across_outputs(self):
        """Same node ID in multiple surface outputs appears only once in catalog."""
        from servers.scales.s1.encode_contract import build_node_catalog

        so1 = '[rule] "Title A" (id:%s, conf:0.9)' % self.NODE_A
        so2 = '[lesson] "Title B" (id:%s, conf:0.8)' % self.NODE_A

        text, ids = build_node_catalog([so1, so2], self.brain)
        assert len(ids) == 1
        assert self.NODE_A in ids

    def test_multiple_distinct_ids(self):
        """Multiple different node IDs from different surface outputs are all extracted."""
        from servers.scales.s1.encode_contract import build_node_catalog

        so1 = '[rule] "First" (id:%s)' % self.NODE_A
        so2 = '[lesson] "Second" (id:%s)' % self.NODE_B

        text, ids = build_node_catalog([so1, so2], self.brain)
        assert self.NODE_A in ids
        assert self.NODE_B in ids
        assert len(ids) == 2

    def test_uses_format_node_for_rich_display(self):
        """Catalog entries use format_node, which includes type/title/content/edges."""
        from servers.scales.s1.encode_contract import build_node_catalog

        surface_output = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A
        text, ids = build_node_catalog([surface_output], self.brain)

        assert self.NODE_A in ids
        # format_node includes the node type in brackets and title
        assert '[rule]' in text
        assert 'Test catalog rule' in text

    def test_includes_correction_annotations(self):
        """Nodes with corrected_by edges get Updated by annotation in catalog."""
        from servers.scales.s1.encode_contract import build_node_catalog

        # Create a correcting node and a `corrects` edge.
        # correction_enrich walks correction_improvement-aspect edges
        # (corrects, supersedes, reframes, ...), so we set up via a typed
        # edge rather than the legacy correction_of metadata field.
        _insert_hex_node(self.conn, self.NODE_C, 'rule', 'Updated version of rule',
                         'New improved content.')
        # Edge: NODE_C corrects NODE_A
        self.brain.connect_typed(
            source_id=self.NODE_C, target_id=self.NODE_A,
            relation='corrects', weight=0.5,
            description='Test setup: NODE_C supersedes NODE_A',
            encoding_source='test:correction_annotation')

        judge_output = '[rule] "Test" (id:%s)' % self.NODE_A
        text, ids = build_node_catalog([judge_output], self.brain)

        assert self.NODE_A in ids
        assert 'Updated by' in text

    def test_nonexistent_id_excluded_from_formatted(self):
        """IDs extracted from surface output but not in DB are not in formatted_ids."""
        from servers.scales.s1.encode_contract import build_node_catalog

        fake_id = 'deadbeef'
        surface_output = '[rule] "Ghost" (id:%s)' % fake_id
        text, ids = build_node_catalog([surface_output], self.brain)

        # The ID is extracted by regex but format_node returns None -> not in formatted_ids
        assert fake_id not in ids

    def test_regex_handles_typed_prefix_ids(self):
        """Typed prefix IDs (con_xxxx) must be extracted by build_node_catalog."""
        from servers.scales.s1.encode_contract import build_node_catalog
        import re

        # Find a typed-prefix node
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE archived = 0 LIMIT 1").fetchone()
        if not row or re.match(r'^[a-f0-9]{8}$', row[0]):
            pytest.skip("No typed-prefix IDs in isolated brain")

        node_id = row[0][:8]
        surface_output = '[rule] "Test" (id:%s, conf:0.9)' % node_id
        text, ids = build_node_catalog([surface_output], self.brain)

        assert node_id in ids, (
            "Regex should match typed-prefix ID %s" % node_id)

    # ── Piece 3: the widened catalog (surfaced ∪ encoded ∪ authored ∪ recalled) ──

    def test_extra_ids_fold_in_tagged(self):
        """authored/recalled/encoded ids fold into the catalog, each tagged."""
        from servers.scales.s1.encode_contract import build_node_catalog
        _insert_hex_node(self.conn, self.NODE_C, 'note', 'Recalled node', 'looked up')
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A  # surfaced
        text, ids = build_node_catalog(
            [judge], self.brain,
            extra_ids={'authored': {self.NODE_B}, 'recalled': {self.NODE_C},
                       'encoded': set()})
        assert {self.NODE_A, self.NODE_B, self.NODE_C} <= ids   # union
        assert '[anchor-authored]' in text and '[anchor-recalled]' in text

    def test_priority_authored_beats_surfaced(self):
        """A node both surfaced AND authored gets the higher-signal authored tag."""
        from servers.scales.s1.encode_contract import build_node_catalog
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A
        text, ids = build_node_catalog(
            [judge], self.brain, extra_ids={'authored': {self.NODE_A}})
        assert self.NODE_A in ids
        assert '[anchor-authored]' in text          # tagged, not left untagged

    def test_flag_off_surfaced_only_unchanged(self):
        """No extra_ids → no tags, legacy header (the A/B control arm)."""
        from servers.scales.s1.encode_contract import build_node_catalog
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A
        text, _ = build_node_catalog([judge], self.brain)
        assert '[anchor-' not in text and '[encoded]' not in text
        assert 'surfaced this session' in text      # legacy header preserved

    def test_community_node_skipped_even_when_in_extra(self):
        """A community node folded via extra_ids is still skipped (S2CE owns it)."""
        from servers.scales.s1.encode_contract import build_node_catalog
        _insert_hex_node(self.conn, self.NODE_C, 'community', 'A community', 'narrative')
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A
        text, ids = build_node_catalog(
            [judge], self.brain, extra_ids={'encoded': {self.NODE_C}})
        assert self.NODE_A in ids
        assert self.NODE_C not in ids               # community filtered out

    def test_noise_relations_filtered_on_lived_arm_only(self):
        """Lived arm (extra_ids not None): noise-aspect relations (community_member,
        co_accessed, ...) drop from rendered connections; semantic relations
        survive. Control arm (extra_ids=None): unfiltered — byte-behavior of the
        long-standing path preserved."""
        from servers.scales.s1.encode_contract import build_node_catalog
        from servers.dal_graph import GraphDAL
        # sanity: the aspect registry classifies community_member as noise
        noise = set(self.brain.aspects.relations_in(['noise']))
        assert 'community_member' in noise
        gdal = GraphDAL(self.conn)
        gdal.add_relation(self.NODE_A, self.NODE_B, 'community_member',
                          description='structural placement edge')
        gdal.add_relation(self.NODE_A, self.NODE_B, 'extends',
                          description='rule extends the lesson semantically')
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A

        lived_text, _ = build_node_catalog([judge], self.brain, extra_ids={})
        assert 'community_member' not in lived_text     # noise dropped
        assert 'extends' in lived_text                   # semantic survives

        control_text, _ = build_node_catalog([judge], self.brain)
        assert 'community_member' in control_text        # control unfiltered


class TestSaveJournal:
    """Tests for _save_journal(brain, dispatch_fn, session_id, counter, final_text)."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dispatch = env.dispatch
            yield

    def _mock_dispatch(self):
        """Returns (dispatch_fn, calls_list) where calls_list captures all calls."""
        calls = []
        def mock_dispatch(cmd, args):
            calls.append((cmd, args))
            return {'ok': True}
        return mock_dispatch, calls

    def test_first_journal_entry(self):
        """First run creates journal with Run 1 header."""
        from servers.scales.s1.encode import _save_journal

        dispatch_fn, calls = self._mock_dispatch()
        session_id = 'test-journal-001'
        # Ensure no pre-existing journal
        self.brain._meta.set('encoding_journal_%s' % session_id, '')
        self.brain.conn.commit()

        _save_journal(self.brain, dispatch_fn, session_id, 5, 'Encoded 3 nodes.')

        # Find set_config call for journal
        journal_calls = [(cmd, args) for cmd, args in calls if cmd == 'set_config'
                         and args.get('key', '').startswith('encoding_journal_')]
        assert len(journal_calls) >= 1
        journal_value = journal_calls[0][1]['value']
        assert '--- Run 1 (stop #5) ---' in journal_value
        assert 'Encoded 3 nodes.' in journal_value

    def test_appends_to_existing_journal(self):
        """Second run appends with Run 2 header."""
        from servers.scales.s1.encode import _save_journal

        session_id = 'test-journal-002'
        existing = '--- Run 1 (stop #5) ---\nFirst encoding run.'
        self.brain._meta.set('encoding_journal_%s' % session_id, existing)
        self.brain.conn.commit()

        dispatch_fn, calls = self._mock_dispatch()
        _save_journal(self.brain, dispatch_fn, session_id, 10, 'Second run output.')

        journal_calls = [(cmd, args) for cmd, args in calls if cmd == 'set_config'
                         and args.get('key', '').startswith('encoding_journal_')]
        assert len(journal_calls) >= 1
        journal_value = journal_calls[0][1]['value']
        assert '--- Run 1' in journal_value
        assert '--- Run 2 (stop #10) ---' in journal_value
        assert 'Second run output.' in journal_value

    def test_truncates_at_entry_boundaries(self):
        """When journal exceeds max_chars, truncation happens at '--- Run ' boundaries."""
        from servers.scales.s1.encode import _save_journal
        from servers.scales.s1.encode_contract import ENCODING_AGENT

        session_id = 'test-journal-trunc'
        max_chars = ENCODING_AGENT.get('journal_max_chars', 8000)

        # Build a journal that will exceed the limit when new entry is added
        entries = []
        for i in range(1, 60):
            entries.append('--- Run %d (stop #%d) ---\n%s' % (i, i * 5, 'x' * 150))
        existing = '\n'.join(entries)
        self.brain._meta.set('encoding_journal_%s' % session_id, existing)
        self.brain.conn.commit()

        dispatch_fn, calls = self._mock_dispatch()
        _save_journal(self.brain, dispatch_fn, session_id, 300, 'Final run.')

        journal_calls = [(cmd, args) for cmd, args in calls if cmd == 'set_config'
                         and args.get('key', '').startswith('encoding_journal_')]
        assert len(journal_calls) >= 1
        journal_value = journal_calls[0][1]['value']

        # Should be within limit
        assert len(journal_value) <= max_chars
        # Should start at an entry boundary (not mid-entry)
        assert journal_value.startswith('--- Run ')

    def test_labels_runs_correctly(self):
        """Run sequence numbers increment based on existing journal entries."""
        from servers.scales.s1.encode import _save_journal

        session_id = 'test-journal-seq'
        existing = '--- Run 1 (stop #5) ---\nFirst.\n--- Run 2 (stop #10) ---\nSecond.'
        self.brain._meta.set('encoding_journal_%s' % session_id, existing)
        self.brain.conn.commit()

        dispatch_fn, calls = self._mock_dispatch()
        _save_journal(self.brain, dispatch_fn, session_id, 15, 'Third run.')

        journal_calls = [(cmd, args) for cmd, args in calls if cmd == 'set_config'
                         and args.get('key', '').startswith('encoding_journal_')]
        journal_value = journal_calls[0][1]['value']
        assert '--- Run 3 (stop #15) ---' in journal_value


class TestSaveSessionContext:
    """Tests for _save_session_context(brain, dispatch_fn, session_id, final_text).

    2026-05-02 (Frame Phase 2.5): session context is per-session — written to
    `session_context_{session_id}`, not the global `session_context` key — so
    parallel sessions don't clobber each other. Tests seed + assert that key.
    """

    SID = 'test-sess'

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dispatch = env.dispatch
            yield

    def _mock_dispatch(self):
        calls = []
        def mock_dispatch(cmd, args):
            calls.append((cmd, args))
            return {'ok': True}
        return mock_dispatch, calls

    def test_extracts_session_context_line(self):
        """SESSION_CONTEXT: line in encoder output gets extracted and saved."""
        from servers.scales.s1.encode import _save_session_context

        key = 'session_context_' + self.SID
        # Clear existing per-session context
        self.brain._meta.set(key, '')
        self.brain.conn.commit()

        dispatch_fn, calls = self._mock_dispatch()
        final_text = 'Some encoding output.\nSESSION_CONTEXT: Tom exploring recall mechanisms.\nDone.'
        _save_session_context(self.brain, dispatch_fn, self.SID, final_text)

        ctx_calls = [(cmd, args) for cmd, args in calls
                     if cmd == 'set_config' and args.get('key') == key]
        assert len(ctx_calls) == 1
        assert 'Tom exploring recall mechanisms.' in ctx_calls[0][1]['value']

    def test_no_context_line_does_nothing(self):
        """When no SESSION_CONTEXT: line exists, nothing is dispatched."""
        from servers.scales.s1.encode import _save_session_context

        dispatch_fn, calls = self._mock_dispatch()
        _save_session_context(self.brain, dispatch_fn, self.SID, 'Just some text without context.')

        ctx_calls = [(cmd, args) for cmd, args in calls
                     if cmd == 'set_config' and args.get('key') == 'session_context_' + self.SID]
        assert len(ctx_calls) == 0

    def test_appends_to_existing_context(self):
        """New SESSION_CONTEXT appends to existing, newline-separated."""
        from servers.scales.s1.encode import _save_session_context

        key = 'session_context_' + self.SID
        self.brain._meta.set(key, 'Previous context about testing.')
        self.brain.conn.commit()

        dispatch_fn, calls = self._mock_dispatch()
        final_text = 'SESSION_CONTEXT: Now working on encoding pipeline.'
        _save_session_context(self.brain, dispatch_fn, self.SID, final_text)

        ctx_calls = [(cmd, args) for cmd, args in calls
                     if cmd == 'set_config' and args.get('key') == key]
        assert len(ctx_calls) == 1
        value = ctx_calls[0][1]['value']
        assert 'Previous context about testing.' in value
        assert 'Now working on encoding pipeline.' in value
        assert '\n' in value  # Newline-separated, not pipe

    def test_truncates_at_line_boundaries(self):
        """When combined context exceeds limit, truncation happens at line boundaries."""
        from servers.scales.s1.encode import _save_session_context
        from servers.scales.s1.encode_contract import ENCODING_AGENT

        limit = ENCODING_AGENT.get('session_context_limit', 800)

        # Fill existing context close to the limit
        lines = ['Context line %d about something important' % i for i in range(30)]
        existing = '\n'.join(lines)
        key = 'session_context_' + self.SID
        self.brain._meta.set(key, existing)
        self.brain.conn.commit()

        dispatch_fn, calls = self._mock_dispatch()
        final_text = 'SESSION_CONTEXT: Final context entry that pushes over the limit.'
        _save_session_context(self.brain, dispatch_fn, self.SID, final_text)

        ctx_calls = [(cmd, args) for cmd, args in calls
                     if cmd == 'set_config' and args.get('key') == key]
        if ctx_calls:
            value = ctx_calls[0][1]['value']
            assert len(value) <= limit
            # Should not start mid-line (first char should not be a newline)
            assert '\n' not in value[:1]

    def test_case_insensitive_prefix(self):
        """SESSION_CONTEXT: matching is case-insensitive on the prefix."""
        from servers.scales.s1.encode import _save_session_context

        key = 'session_context_' + self.SID
        self.brain._meta.set(key, '')
        self.brain.conn.commit()

        dispatch_fn, calls = self._mock_dispatch()
        # The code uses .upper().startswith('SESSION_CONTEXT:')
        final_text = 'session_context: lowercase prefix works too.'
        _save_session_context(self.brain, dispatch_fn, self.SID, final_text)

        ctx_calls = [(cmd, args) for cmd, args in calls
                     if cmd == 'set_config' and args.get('key') == key]
        assert len(ctx_calls) == 1
        assert 'lowercase prefix works too.' in ctx_calls[0][1]['value']


class _StubRecallBrain:
    """Minimal brain for _associated_stub_ids: canned episodes + recorded
    recalls. No trace DAL — the test passes empty streams so _turn_links
    joins nothing (every turn reads unencoded, the probe's convention)."""

    def __init__(self, episodes, results_by_query):
        self._episodes = episodes
        self._results = results_by_query
        self.recall_calls = []

    def recall_episodes(self, **kw):
        return {'episodes': list(self._episodes)}

    def recall(self, query, **kw):
        self.recall_calls.append((query, kw))
        return {'results': list(self._results.get(query, []))}


def _ep(eid, ref_type, content, created_at):
    return {'id': eid, 'ref_type': ref_type, 'chain_id': 's0-testsess-1',
            'created_at': created_at, 'summary': content[:200],
            'metadata': {'content': content}}


_EMPTY_STREAMS = {'surface': [], 'encode': [], 'touched': []}
# Replay-style date prefix: conversation_now resolves from it, so the fake
# brain is never asked for wall-clock/tz machinery.
_DATED_MSGS = [
    {'role': 'user', 'content': '[Current date: 2026-08-21] alpha question'},
    {'role': 'user', 'content': 'beta question'},
]


class TestAssociatedStubIds:
    """_associated_stub_ids — the subconscious retrieval (turnmax over
    per-message production recall, catalog excluded)."""

    def _episodes(self):
        return [
            _ep('u1', 'user_message', 'alpha question', '2026-08-21T10:00:00+00:00'),
            _ep('a1', 'assistant_message', 'alpha answer', '2026-08-21T10:01:00+00:00'),
            _ep('u2', 'user_message', 'beta question', '2026-08-21T10:02:00+00:00'),
        ]

    def test_turnmax_exclusion_and_ordering(self):
        from servers.scales.s1.encode import _associated_stub_ids
        brain = _StubRecallBrain(self._episodes(), {
            'alpha question': [
                {'id': 'aaaa1111', 'type': 'finding', 'effective_activation': 0.9},
                {'id': 'cata0001', 'type': 'rule', 'effective_activation': 0.95},
                {'id': 'comm0001', 'type': 'community', 'effective_activation': 0.99},
            ],
            # same node again, lower score — turnmax keeps 0.9, never sums
            'alpha answer': [
                {'id': 'aaaa1111', 'type': 'finding', 'effective_activation': 0.5},
            ],
            'beta question': [
                {'id': 'bbbb2222', 'type': 'lesson', 'effective_activation': 0.7},
            ],
        })
        out = _associated_stub_ids(brain, _DATED_MSGS, 'testsess',
                                   {'cata0001'}, streams=_EMPTY_STREAMS)
        # catalog id and community filtered; rank order by best single match
        assert out == ['aaaa1111', 'bbbb2222']
        # both voices of both turns seeded
        assert [q for q, _kw in brain.recall_calls] == [
            'alpha question', 'alpha answer', 'beta question']
        # pure read at conversation time
        for _q, kw in brain.recall_calls:
            assert kw['mark_accessed'] is False
            assert kw['as_of'].startswith('2026-08-21')
            assert kw['session_id'] == 'testsess'

    def test_k_cap(self):
        from servers.scales.s1.encode import _associated_stub_ids
        from servers.scales.s1.encoder_view import ASSOCIATED_STUBS_K
        hits = [{'id': 'node%04d' % i, 'type': 'fact',
                 'effective_activation': 1.0 - i * 0.01} for i in range(9)]
        brain = _StubRecallBrain(self._episodes(), {'alpha question': hits})
        out = _associated_stub_ids(brain, _DATED_MSGS, 'testsess',
                                   {'unrelated'}, streams=_EMPTY_STREAMS)
        assert len(out) == ASSOCIATED_STUBS_K
        assert out == ['node%04d' % i for i in range(ASSOCIATED_STUBS_K)]

    def test_covered_turns_not_seeded(self, monkeypatch):
        from servers.scales.s1 import encode
        brain = _StubRecallBrain(self._episodes(), {
            'beta question': [
                {'id': 'bbbb2222', 'type': 'lesson', 'effective_activation': 0.7}],
        })
        # turn 1 (u1) already encoded by a prior run; turn 2 (u2) is not
        monkeypatch.setattr(encode, '_turn_links', lambda *a, **kw: (
            {'u1': {'surfaced': [], 'encoded': [], 'encoded_by': 'run-1'},
             'u2': {'surfaced': [], 'encoded': [], 'encoded_by': None}}, {}))
        out = encode._associated_stub_ids(brain, _DATED_MSGS, 'testsess',
                                          {'unrelated'}, streams=_EMPTY_STREAMS)
        assert [q for q, _kw in brain.recall_calls] == ['beta question']
        assert out == ['bbbb2222']

    def test_no_seeds_returns_empty_without_recall(self):
        from servers.scales.s1.encode import _associated_stub_ids
        brain = _StubRecallBrain([], {})
        out = _associated_stub_ids(brain, _DATED_MSGS, 'testsess',
                                   {'x'}, streams=_EMPTY_STREAMS)
        assert out == []
        assert brain.recall_calls == []


class TestAssociatedStubsRender:
    """build_node_catalog's associated_ids arm + the _build_catalog gate."""

    NODE_A = 'aa11bb22'
    NODE_B = 'cc33dd44'
    NODE_C = 'ee55ff66'

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.conn = env.brain.conn
            _insert_hex_node(self.conn, self.NODE_A, 'rule', 'Surfaced rule',
                             'This rule was surfaced.')
            _insert_hex_node(self.conn, self.NODE_B, 'lesson', 'Associated lesson',
                             'Recall ranked this near the window.')
            _insert_hex_node(self.conn, self.NODE_C, 'finding', 'Associated finding',
                             'Second subconscious entry.')
            yield

    def _judge(self):
        return '[rule] "Surfaced rule" (id:%s, conf:0.9)' % self.NODE_A

    def test_associated_render_last_tagged_and_in_ids(self):
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG, build_node_catalog
        text, ids = build_node_catalog(
            [self._judge()], self.brain, extra_ids={},
            associated_ids=[self.NODE_B, self.NODE_C])
        assert {self.NODE_A, self.NODE_B, self.NODE_C} <= ids
        assert ASSOCIATED_TAG in text
        # stubs are the catalog's LAST entries, rank order preserved
        assert text.index('Surfaced rule') < text.index('Associated lesson')
        assert text.index('Associated lesson') < text.index('Associated finding')
        assert text.splitlines()[0] == 'Node Catalog (3 nodes)'
        # tag sits on the stub entries only
        for line in text.splitlines():
            if line.startswith(ASSOCIATED_TAG):
                assert 'Associated' in line
            if 'Surfaced rule' in line:
                assert ASSOCIATED_TAG not in line

    def test_associated_none_and_empty_byte_identical(self):
        from servers.scales.s1.encode_contract import build_node_catalog
        base_text, base_ids = build_node_catalog(
            [self._judge()], self.brain, extra_ids={})
        for assoc in (None, []):
            text, ids = build_node_catalog(
                [self._judge()], self.brain, extra_ids={}, associated_ids=assoc)
            assert text == base_text
            assert ids == base_ids

    def test_associated_dupe_of_catalog_dropped(self):
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG, build_node_catalog
        text, ids = build_node_catalog(
            [self._judge()], self.brain, extra_ids={},
            associated_ids=[self.NODE_A])
        # already cataloged: renders once, untagged, not double-counted —
        # and with no stub actually rendering, the surfaced-only header stands
        assert text.count('Surfaced rule') == 1
        assert ASSOCIATED_TAG not in text
        assert text.splitlines()[0] == 'Node Catalog (1 nodes surfaced this session)'

    def test_build_catalog_gate_flag_off_no_retrieval(self, monkeypatch):
        from servers.scales.s1 import encode
        monkeypatch.delenv('BRAIN_S1E_ASSOCIATED_STUBS', raising=False)
        monkeypatch.setattr(encode, '_associated_stub_ids',
                            lambda *a, **kw: pytest.fail('retrieval ran with flag off'))
        messages = [{'role': 'user', 'content': 'hi', 'judge_output': self._judge()}]
        text, ids, _streams = encode._build_catalog(
            self.brain, messages, 'assoc-gate-off', lived=True)
        assert self.NODE_A in ids

    def test_build_catalog_gate_flag_on_renders_stubs(self, monkeypatch):
        from servers.scales.s1 import encode
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG
        monkeypatch.setenv('BRAIN_S1E_ASSOCIATED_STUBS', '1')
        seen = {}

        def fake_retrieval(brain, messages, session_id, exclude_ids, streams=None):
            seen['exclude'] = set(exclude_ids)
            return [self.NODE_B]

        monkeypatch.setattr(encode, '_associated_stub_ids', fake_retrieval)
        messages = [{'role': 'user', 'content': 'hi', 'judge_output': self._judge()}]
        text, ids, _streams = encode._build_catalog(
            self.brain, messages, 'assoc-gate-on', lived=True)
        # exclusion set is exactly what the catalog will show
        assert self.NODE_A in seen['exclude']
        assert ASSOCIATED_TAG in text
        assert self.NODE_B in ids

    def test_build_catalog_retrieval_failure_degrades_loud(self, monkeypatch):
        from servers.scales.s1 import encode
        monkeypatch.setenv('BRAIN_S1E_ASSOCIATED_STUBS', '1')

        def boom(*a, **kw):
            raise RuntimeError('recall exploded')

        monkeypatch.setattr(encode, '_associated_stub_ids', boom)
        messages = [{'role': 'user', 'content': 'hi', 'judge_output': self._judge()}]
        text, ids, _streams = encode._build_catalog(
            self.brain, messages, 'assoc-gate-err', lived=True)
        # catalog still renders, stubs absent
        assert self.NODE_A in ids
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG
        assert ASSOCIATED_TAG not in text
