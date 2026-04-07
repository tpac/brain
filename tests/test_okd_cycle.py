#!/usr/bin/env python3
"""
brain — O/K/Delta Cycle Tests

Tests the fractal property: Delta from one cycle becomes O of the next.
Each test verifies a CROSS-SCALE or CYCLE transition that would fail silently
if the wiring between scales broke.

Run: python -m pytest tests/test_okd_cycle.py -v
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.isolated_brain import IsolatedBrain


class TestS0ToS1EGather(unittest.TestCase):
    """S0 trace write -> S1E _gather_messages reads.

    The S0 hooks write user_message and assistant_message traces.
    The S1 encoder reads them via get_session_turns().
    If the ref_types or metadata shape change, this breaks silently.
    """

    def test_s0_traces_feed_gather_messages(self):
        with IsolatedBrain() as env:
            session_id = 'test-okd-s0-s1e'
            chain_id = 's0-test-1'
            trace = env.brain._trace_dal

            # S0 writes: user_message as K, assistant_message as delta
            trace.append(
                chain_id=chain_id, scale='s0', event_type='K',
                ref_type='user_message', summary='What is Clerk auth?',
                metadata={'content': 'What is Clerk auth?'},
                session_id=session_id)
            trace.append(
                chain_id=chain_id, scale='s0', event_type='delta',
                ref_type='assistant_message',
                summary='Clerk is a user management platform.',
                metadata={'content': 'Clerk is a user management platform.'},
                session_id=session_id)

            # S1E reads via get_session_turns
            from servers.scales.s1.encode import _gather_messages
            messages = _gather_messages(env.brain, session_id)

            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]['role'], 'user')
            self.assertIn('Clerk', messages[0]['content'])
            self.assertEqual(messages[1]['role'], 'assistant')
            self.assertIn('Clerk', messages[1]['content'])


class TestJudgeOutputToNodeCatalog(unittest.TestCase):
    """S1R judge output in S0 traces -> build_node_catalog reads node IDs.

    The judge writes additionalContext containing "id:XXXXXXXX" references.
    The encoder extracts those IDs and builds a rich catalog.
    If the regex or ID format changes, the catalog silently goes empty.
    """

    def test_judge_output_feeds_node_catalog(self):
        with IsolatedBrain() as env:
            # Seed a node so catalog can look it up
            result = env.brain.remember(
                type='decision', title='Use Clerk for auth',
                content='Decided to use Clerk for authentication in the web app.')
            node_id = result['id']
            short_id = node_id[:8]

            # Simulate judge_output string as it appears in trace metadata
            judge_output = 'Brain recalled 1 memory:\n\nid:%s "Use Clerk for auth"\nClerk for authentication...' % short_id

            from servers.scales.s1.encode_contract import build_node_catalog
            catalog_text, catalog_ids = build_node_catalog(
                [judge_output], env.brain.conn)

            self.assertIn(short_id, catalog_ids,
                          "Node ID from judge output must appear in catalog")
            self.assertIn('Clerk', catalog_text,
                          "Catalog must contain node content")


class TestDeltaBecomesNextO(unittest.TestCase):
    """Full cycle: encoder Delta (new node) -> next recall O finds it.

    This is THE core property. A node created by encoding (Delta)
    must be findable by recall (next cycle's O->K selection).
    If embedding, storage, or recall scoring breaks, this fails.
    """

    def test_encoded_node_surfaces_in_recall(self):
        with IsolatedBrain() as env:
            # Delta: encoder creates a node
            result = env.brain.remember(
                type='lesson',
                title='PostgreSQL connection pooling prevents timeout errors',
                content='When running PostgreSQL with many concurrent connections, '
                        'use PgBouncer for connection pooling. Without it, connections '
                        'time out under load. Learned after production outage.',
                keywords='postgresql pgbouncer pooling timeout production')
            node_id = result['id']

            # Next cycle O->K: recall with a matching query
            recall_result = env.brain.recall(
                query='database connection timeout issues', limit=10)
            results = recall_result.get('results', [])

            found_ids = [r['id'][:8] for r in results]
            self.assertIn(node_id[:8], found_ids,
                          "Node created as Delta must surface in next cycle's recall (O->K)")


class TestS1EJournalCarryForward(unittest.TestCase):
    """S1E journal: Delta from one encoding run -> O for the next run.

    The encoder writes a journal entry after each run. The next run
    reads the accumulated journal as part of its observation.
    If the config key or truncation logic breaks, the encoder loses context.
    """

    def test_journal_accumulates_across_runs(self):
        with IsolatedBrain() as env:
            session_id = 'test-okd-journal'
            journal_key = 'encoding_journal_%s' % session_id

            # Simulate _save_journal from first encoding run
            from servers.scales.s1.encode import _save_journal

            # Build a mock dispatch_fn that writes to brain config directly
            def mock_dispatch(cmd, args):
                if cmd == 'set_config':
                    env.brain.set_config(args['key'], args['value'])
                    return {"ok": True}
                return {"ok": False}

            _save_journal(env.brain, mock_dispatch, session_id, counter=5,
                          final_text='Encoded 3 nodes about auth patterns.')

            # Verify journal was written
            journal_after_run1 = env.brain.get_config(journal_key, '')
            self.assertIn('Run 1', journal_after_run1)
            self.assertIn('auth patterns', journal_after_run1)

            # Simulate second run — journal should accumulate
            _save_journal(env.brain, mock_dispatch, session_id, counter=10,
                          final_text='Revised 1 node about database pooling.')

            journal_after_run2 = env.brain.get_config(journal_key, '')
            self.assertIn('Run 1', journal_after_run2)
            self.assertIn('Run 2', journal_after_run2)
            self.assertIn('auth patterns', journal_after_run2)
            self.assertIn('database pooling', journal_after_run2)


class TestCrossScaleTraceIntegrity(unittest.TestCase):
    """S0 and S1 traces for the same session coexist without interference.

    Both scales write to trace_events with the same session_id.
    If scale filtering breaks, S1 events could leak into S0 queries
    or vice versa. This verifies isolation.
    """

    def test_cross_scale_traces_coexist(self):
        with IsolatedBrain() as env:
            session_id = 'test-okd-cross-scale'
            trace = env.brain._trace_dal

            # S0 traces
            trace.append(
                chain_id='s0-test-1', scale='s0', event_type='K',
                ref_type='user_message', summary='user said hello',
                session_id=session_id)
            trace.append(
                chain_id='s0-test-1', scale='s0', event_type='delta',
                ref_type='assistant_message', summary='assistant replied',
                session_id=session_id)

            # S1 recall traces
            trace.append(
                chain_id='s1r-test-1', scale='s1', event_type='O',
                ref_type='recall', summary='25 candidates',
                session_id=session_id)
            trace.append(
                chain_id='s1r-test-1', scale='s1', event_type='K',
                ref_type='surface_selected', summary='5 nodes selected',
                session_id=session_id)

            # Query by session — both scales present
            all_chains = trace.get_chains_for_session(session_id)
            self.assertIn('s0-test-1', all_chains)
            self.assertIn('s1r-test-1', all_chains)

            # Query by scale — no leakage
            s0_events = trace.get_recent(scale='s0', hours=1)
            s0_ref_types = {e['ref_type'] for e in s0_events
                           if e.get('chain_id', '').startswith('s0-test')}
            self.assertNotIn('recall', s0_ref_types,
                             "S1 recall events must not appear in S0 query")
            self.assertNotIn('surface_selected', s0_ref_types,
                             "S1 surface events must not appear in S0 query")

            s1_events = trace.get_recent(scale='s1', hours=1)
            s1_ref_types = {e['ref_type'] for e in s1_events
                           if e.get('chain_id', '').startswith('s1r-test')}
            self.assertNotIn('user_message', s1_ref_types,
                             "S0 user events must not appear in S1 query")

    def test_surface_selected_trace_ref_type(self):
        """Verify that 'surface_selected' is the correct ref_type for S1R K events."""
        from servers.trace_contract import validate_trace_event
        ok, msg = validate_trace_event('s1', 'K', 'surface_selected')
        self.assertTrue(ok, msg)


class TestNodeCatalogRegexMatchesRealIDs(unittest.TestCase):
    """The build_node_catalog regex must match actual node IDs from remember().

    Node IDs are UUIDs (hex). The regex in build_node_catalog extracts
    id:XXXXXXXX patterns. If the ID format changes (e.g., typed prefixes
    like 'dec-XXXX') but the regex stays hex-only, the catalog silently
    goes empty. This test catches that mismatch.
    """

    def test_real_node_id_matches_catalog_regex(self):
        with IsolatedBrain() as env:
            import re

            # Create a real node to get a real ID
            result = env.brain.remember(
                type='mechanism',
                title='Synaptic fatigue dampens hub nodes',
                content='Nodes recalled repeatedly get cosine dampened. '
                        'Rate scales with structural degree.')
            node_id = result['id']
            short_id = node_id[:8]

            # Verify the regex in build_node_catalog can extract this ID
            # This is the exact regex from encode_contract.py
            judge_output = 'id:%s "Synaptic fatigue dampens hub nodes"' % short_id
            matches = re.findall(r'id:([a-f0-9]{8})', judge_output)
            self.assertIn(short_id, matches,
                          "Real node ID '%s' must match catalog regex r'id:([a-f0-9]{8})'" % short_id)

            # End-to-end: build_node_catalog with this ID
            from servers.scales.s1.encode_contract import build_node_catalog
            catalog_text, catalog_ids = build_node_catalog(
                [judge_output], env.brain.conn)
            self.assertIn(short_id, catalog_ids,
                          "Real node ID must survive the full catalog pipeline")


if __name__ == '__main__':
    unittest.main()
