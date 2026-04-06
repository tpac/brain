"""S1 full cycle test — recall → judge → encode → trace.

End-to-end test of Scale 1 integration: the brain recalls, judges,
encodes, and traces the full cycle. Then verifies everything is
queryable through TraceDAL.

Run: python3 -m pytest tests/test_s1_cycle.py -v
"""
import json
import pytest
from tests.isolated_brain import IsolatedBrain


class TestS1FullCycle:
    """End-to-end: recall → judge simulate → encode simulate → trace verify."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            self.dispatch = env.dispatch
            # Clear traces
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def test_full_cycle(self):
        """Complete S1 cycle: recall → judge → encode → all traces exist."""
        session_id = 'cycle-test-session1'
        stop = '50'
        query = 'how does the encoding agent work?'

        # Step 1: Recall
        result = self.brain.recall(query=query, limit=25, session_id=session_id)
        candidates = result.get('results', [])
        assert len(candidates) > 0, "Recall should return candidates from isolated brain"

        # Step 2: Simulate judge selection (pick top 3)
        selected = candidates[:3]
        selected_ids = [c.get('id', '')[:8] for c in selected]

        # Step 3: Write S1 recall traces (O, K, delta)
        s1r_chain = 's1r-%s-%s' % (session_id[:8], stop)
        cand_detail = ['%s|%s|%.2f|%s' % (
            c.get('id', '')[:8], c.get('title', '')[:60],
            c.get('effective_activation', 0), c.get('type', ''))
            for c in candidates[:25]]
        sel_detail = ['%s|%s' % (c.get('id', '')[:8], c.get('title', ''))
                      for c in selected]

        self.dal.append(
            chain_id=s1r_chain, scale='s1', event_type='O',
            ref_type='recall', ref_id=stop,
            summary='%d candidates for: %s' % (len(candidates), query[:100]),
            metadata={'source': 'hook', 'query': query, 'candidates': cand_detail},
            session_id=session_id)
        self.dal.append(
            chain_id=s1r_chain, scale='s1', event_type='K',
            ref_type='judge_selected',
            ref_id=json.dumps(selected_ids),
            summary='%d selected' % len(selected),
            metadata={'selected': sel_detail, 'expanded': []},
            session_id=session_id)

        additional_context = 'Brain recalled %d memories:\n%s' % (
            len(selected), '\n'.join('[%s] "%s"' % (c.get('type', ''), c.get('title', ''))
                                     for c in selected))
        self.dal.append(
            chain_id=s1r_chain, scale='s1', event_type='delta',
            ref_type='additionalContext',
            summary='%d nodes surfaced' % len(selected),
            metadata={'content': additional_context},
            session_id=session_id)

        # Step 4: Write S0 exchange referencing the S1 chain
        s0_chain = 's0-%s-%s' % (session_id[:8], stop)
        self.dal.append(
            chain_id=s0_chain, scale='s0', event_type='K',
            ref_type='user_message',
            summary=query[:200],
            metadata={'content': query, 'recall_chain': s1r_chain},
            session_id=session_id)
        self.dal.append(
            chain_id=s0_chain, scale='s0', event_type='delta',
            ref_type='assistant_message',
            summary='The encoding agent runs every 5th stop...',
            metadata={'content': 'The encoding agent runs every 5th stop and uses Sonnet to encode.'},
            session_id=session_id)

        # Step 5: Simulate encoding (creates a node via dispatch)
        enc_result = self.dispatch('remember', {
            'type': 'test',
            'title': 'S1 cycle test node',
            'content': 'Created during S1 full cycle test',
            'encoding_source': 'encoder:sonnet',
        })
        assert enc_result.get('ok') or enc_result.get('id'), "Remember should succeed"

        # Step 6: Write S1 encode traces
        s1e_chain = 's1e-%s-%s' % (session_id[:8], stop)
        self.dal.append(
            chain_id=s1e_chain, scale='s1', event_type='O',
            ref_type='encoding_prompt',
            ref_id='/tmp/brain-encoding-prompt-%s.json' % stop,
            summary='5 turns, 20000 chars context',
            session_id=session_id)
        self.dal.append(
            chain_id=s1e_chain, scale='s1', event_type='K',
            ref_type='node_catalog',
            ref_id=','.join(selected_ids),
            summary='%d unique nodes in catalog' % len(selected_ids),
            session_id=session_id)
        self.dal.append(
            chain_id=s1e_chain, scale='s1', event_type='delta',
            ref_type='encoding_run', ref_id=stop,
            summary='1 actions:\nremember: S1 cycle test node\n---\nEncoded test node.',
            session_id=session_id)

        # ── VERIFY ──

        # All 9 trace events exist (3 S1r + 2 S0 + 3 S1e + 1 from recall_log via brain.recall)
        s1r_events = self.dal.get_chain(s1r_chain)
        assert len(s1r_events) == 3, "S1 recall chain should have O, K, delta"

        s0_events = self.dal.get_chain(s0_chain)
        assert len(s0_events) == 2, "S0 chain should have K, delta"

        s1e_events = self.dal.get_chain(s1e_chain)
        assert len(s1e_events) == 3, "S1 encode chain should have O, K, delta"

        # get_session_turns cross-references correctly
        turns = self.dal.get_session_turns(session_id)
        user_turns = [t for t in turns if t['role'] == 'user']
        assert len(user_turns) >= 1
        assert user_turns[0]['judge_output'] == additional_context

        # get_chains returns grouped S1 chains
        s1_chains = self.dal.get_chains(session_id=session_id, scale='s1')
        assert len(s1_chains) >= 2  # recall + encode chains

        # The encoded node exists
        node = self.brain.find_node_by_title('S1 cycle test node')
        assert node is not None or len(node) > 0


class TestS1CycleWithOutcome:
    """S1 cycle with an outcome event added retrospectively."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with IsolatedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def test_outcome_appended_to_chain(self):
        """After S1 recall, an outcome event records what happened next."""
        session_id = 'outcome-test-session'
        s1r_chain = 's1r-outcome--30'

        # Write recall chain
        self.dal.append(chain_id=s1r_chain, scale='s1', event_type='O',
                        ref_type='recall', summary='query',
                        metadata={'query': 'test'}, session_id=session_id)
        self.dal.append(chain_id=s1r_chain, scale='s1', event_type='K',
                        ref_type='judge_selected', summary='2 selected',
                        session_id=session_id)
        self.dal.append(chain_id=s1r_chain, scale='s1', event_type='delta',
                        ref_type='additionalContext', summary='surfaced',
                        metadata={'content': 'Brain recalled 2 memories'},
                        session_id=session_id)

        # Later: Tom corrected what was recalled
        self.dal.append_outcome(
            chain_id=s1r_chain, scale='s1',
            ref_type='correction', ref_id='node-xyz',
            summary='Tom corrected: the recalled info was outdated',
            session_id=session_id)

        # Verify chain has 4 events
        chain = self.dal.get_chain(s1r_chain)
        assert len(chain) == 4
        assert chain[3]['event_type'] == 'outcome'
        assert chain[3]['ref_type'] == 'correction'

        # Verify get_outcomes finds it
        outcomes = self.dal.get_outcomes(chain_id=s1r_chain)
        assert len(outcomes) == 1
        assert 'outdated' in outcomes[0]['summary']

        # Verify get_outcomes by scale finds it
        all_outcomes = self.dal.get_outcomes(scale='s1')
        assert any(o['chain_id'] == s1r_chain for o in all_outcomes)



# TestS1MigrationParity REMOVED 2026-04-06 — message_stream deleted, migration complete
