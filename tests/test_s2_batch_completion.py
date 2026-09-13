"""Failed batches remain retryable through suppression, scan gates and scheduling."""
import os
from contextlib import contextmanager
from unittest.mock import patch

from tests.brain_test_base import BrainTestBase
from servers.scales.s2.base import IntegrationUnit
from servers.scales.s2.community import CommunityDetection
from servers.scales.s2.community_decoder import CommunityDecoder
from servers.scales.s2.community_contract import COMMUNITY_DETECTION
from servers.scales.s2.consolidation import Consolidation
from servers.scales.s2.consolidation_decoder import ConsolidationDecoder
from servers.scales.s2.consolidation_contract import CONSOLIDATION
from servers.scales.s2.rejection_table import filter_rejected


class TestBatchCompletion(BrainTestBase):
    needs_embedder = False

    def node(self, title, **kw):
        return self.brain.remember(
            title=title, type=kw.pop('type', 'finding'),
            content=kw.pop('content', title), auto_connect=False, **kw)['id']

    @contextmanager
    def model(self, respond):
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.runner.make_client', return_value=object()), \
                patch('servers.scales.runner.run_llm_loop', side_effect=respond), \
                patch.object(self.brain, 'recall', return_value={'results': []}):
            yield

    def rejection(self):
        # These words in the model's reasoning are not execution failures.
        return {'rounds': 1, 'actions': 0, 'write_actions': 0,
                'final_text': 'ERROR handling is relevant, but the old test FAILED. '
                              'Reject this proposal.\n## Review\n```\n```'}

    def test_partial_run_config_preserves_the_resolved_batch_budget(self):
        from tests.interaction_override import interaction_override
        from servers.scales.s2.community_encoder import CommunityEncoder
        with interaction_override(self.brain, 's2_community', parameters={
                'max_batch_context_chars': 12345}):
            encoder = CommunityEncoder(self.brain, config={'max_proposals_per_call': 1})
        self.assertEqual(encoder.config['max_batch_context_chars'], 12345)
        self.assertEqual(encoder.config['max_proposals_per_call'], 1)

    def test_batch_failure_cannot_be_erased_by_later_success(self):
        unit = IntegrationUnit(self.brain)
        total = {'rounds': 0, 'actions': 0, 'write_actions': 0,
                 'action_details': [], 'read_calls': [], 'final_text': ''}
        self.assertFalse(unit._fold_batch_result(
            total, {'rounds': 2, 'error': 'provider unavailable'}, 1, 'test'))
        self.assertTrue(unit._fold_batch_result(
            total, {'rounds': 1, 'write_actions': 1, 'actions': 1}, 2, 'test'))
        self.assertIn('batch 1: provider unavailable', total['error'])
        self.assertEqual(total['write_actions'], 1)
        self.assertEqual(total['rounds'], 3)

    def test_zero_round_batch_is_an_error(self):
        unit = IntegrationUnit(self.brain)
        total = {'rounds': 0, 'actions': 0, 'write_actions': 0,
                 'action_details': [], 'read_calls': [], 'final_text': ''}
        self.assertFalse(unit._fold_batch_result(total, {'rounds': 0}, 1, 'test'))
        self.assertIn('no completed rounds', total['error'])

    def community_scan(self):
        targets = [self.node('Community %d' % i, type='community',
                             content='A concrete community account. ' * 1100)
                   for i in range(2)]
        pending = [self.node('Pending %d' % i) for i in range(2)]
        proposals = [{'type': 'health_update', 'community_id': cid,
                      'community_title': 'Community %d' % i,
                      'signal': 'corridor_maturing',
                      'old_fraction': 0.1, 'new_fraction': 0.4}
                     for i, cid in enumerate(targets)]
        return {'proposals': proposals,
                'community_state': [{'id': cid, 'members': []} for cid in targets],
                'pending_probes': [{'type': 'unplaceable', 'node_id': nid,
                                    'neighborhood': ''} for nid in pending]}

    def test_split_community_failure_retries_without_new_graph_changes(self):
        # Real default context budget forces two calls. Exercise failures in
        # either order, then a completed retry through the same idle gate.
        for failed_batch in (0, 1):
            with self.subTest(failed_batch=failed_batch):
                decoded = self.community_scan()
                unit = CommunityDetection(self.brain, config={
                    **COMMUNITY_DETECTION, 'min_run_interval_seconds': 0})
                self.brain.set_config(unit.LAST_RUN_KEY, '1')
                calls = []

                def respond(**kw):
                    calls.append(kw['user_content'])
                    if len(calls) - 1 == failed_batch:
                        raise RuntimeError('provider unavailable')
                    return self.rejection()

                with patch.object(CommunityDecoder, 'run', return_value=decoded), \
                        self.model(respond):
                    result = unit.run()
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(len(c) <= 48000 for c in calls))
                self.assertIn('provider unavailable', result['error'])
                self.assertEqual(self.brain.get_config(unit.LAST_RUN_KEY), '1')
                for proposals in (decoded['proposals'], decoded['pending_probes']):
                    self.assertEqual(filter_rejected(self.brain, proposals)[1], 0)
                self.assertIsNone(unit._should_skip())

                # No new node or edge is needed to make the retry eligible.
                with patch.object(CommunityDecoder, 'run', return_value=decoded), \
                        self.model(lambda **kw: self.rejection()):
                    result = unit.run()
                self.assertFalse(result.get('error'))
                self.assertNotEqual(self.brain.get_config(unit.LAST_RUN_KEY), '1')
                for proposals in (decoded['proposals'], decoded['pending_probes']):
                    self.assertEqual(filter_rejected(self.brain, proposals)[1], 2)

    def test_community_partial_write_survives_later_batch_failure(self):
        decoded = self.community_scan()
        cid = decoded['community_state'][0]['id']
        nid = decoded['pending_probes'][0]['node_id']
        decoded['proposals'][0] = {'type': 'add_to_existing', 'node_id': nid,
                                   'communities': [{'id': cid}]}
        operations = [{'op': 'connect', 'source_id': cid, 'target_id': nid,
                       'relation': 'community_member'}]
        calls = []

        def respond(**kw):
            calls.append(kw['user_content'])
            if len(calls) == 2:
                raise RuntimeError('second batch unavailable')
            response = kw['dispatch_fn']('brain_batch', {'operations': operations})
            self.assertTrue(response['ok'])
            return {'rounds': 2, 'actions': 1, 'write_actions': 1,
                    'action_details': [{'tool': 'brain_batch',
                                        'input': {'operations': operations}}]}

        unit = CommunityDetection(self.brain)
        with patch.object(CommunityDecoder, 'run', return_value=decoded), \
                self.model(respond):
            result = unit.run()
        self.assertIn('second batch unavailable', result['error'])
        self.assertEqual(result['actions'], 1)
        self.assertIn(cid, [c['id'] for c in self.brain._graph.get_communities_for([nid])[nid]])
        self.assertIsNone(self.brain.get_config(unit.LAST_RUN_KEY))

    def test_community_setup_exception_does_not_advance_cutoff(self):
        unit = CommunityDetection(self.brain)
        with patch.object(CommunityDecoder, 'run', side_effect=RuntimeError('read failed')):
            with self.assertRaisesRegex(RuntimeError, 'read failed'):
                unit.run()
        self.assertIsNone(self.brain.get_config(unit.LAST_RUN_KEY))

    def test_consolidation_partial_failure_preserves_fingerprints_and_cutoff(self):
        ids = [self.node('Member %d' % i) for i in range(4)]
        clusters = [{'nodes': ids[i:i + 2], 'size': 2,
                     'node_details': self.brain.get_node(ids[i:i + 2]),
                     'content_cosine_max': 0.95, 'title_cosine_max': 0.9,
                     'pre_class': 'needs_judgment'} for i in (0, 2)]
        decoded = {'clusters': clusters, 'stats': {},
                   '_stamp': {'ts': 12345.0, 'threshold': '0.89'}}
        unit = Consolidation(self.brain, config={
            **CONSOLIDATION, 'max_proposals_per_call': 1})
        self.brain.set_config(unit.LAST_RUN_TS_KEY, '1')
        self.brain.set_config(unit.LAST_THRESHOLD_KEY, '0.8')
        calls = []

        def respond(**kw):
            calls.append(kw['user_content'])
            if len(calls) == 2:
                raise RuntimeError('provider unavailable')
            return self.rejection()

        with patch.object(ConsolidationDecoder, 'run', return_value=decoded), \
                self.model(respond):
            result = unit.run()
        self.assertEqual(len(calls), 2)
        self.assertIn('provider unavailable', result['error'])
        self.assertEqual(self.brain.get_config(unit.LAST_RUN_TS_KEY), '1')
        self.assertEqual(self.brain.get_config(unit.LAST_THRESHOLD_KEY), '0.8')
        self.assertEqual(self.brain.conn.execute('SELECT COUNT(*) FROM s2_rejections').fetchone()[0], 0)
        with patch.object(ConsolidationDecoder, 'run', return_value=decoded), \
                self.model(lambda **kw: self.rejection()):
            result = unit.run()
        self.assertFalse(result.get('error'))
        self.assertEqual(result['skipped_recorded'], 2)
        self.assertEqual(self.brain.get_config(unit.LAST_RUN_TS_KEY), '12345.0')

    def test_coordinator_counts_returned_and_raised_errors_and_recovers(self):
        from servers.scales.s2.aspect_integration import AspectIntegration
        from servers.scales.s2.healer import Healer
        key = 's2_community_detection_consecutive_failures'
        partial = {'error': 'batch 2 unavailable', 'actions': 3}
        with patch.object(AspectIntegration, 'run', return_value={'skipped': 'empty'}), \
                patch.object(Healer, 'run', return_value={'skipped': 'empty'}), \
                patch.object(Consolidation, 'run', return_value={'skipped': 'empty'}), \
                patch.object(CommunityDetection, 'run', side_effect=[
                    partial, RuntimeError('raised failure'), partial,
                    {'skipped': 'throttled'}, {'actions': 0}]), \
                patch.object(self.brain, '_log_error') as log:
            for expected in ('1', '2', '3', '3', '0'):
                result = self.brain.run_s2()['units']['community_detection']
                self.assertEqual(self.brain.get_config(key), expected)
                if result.get('error') == partial['error']:
                    self.assertEqual(result['actions'], 3)
        self.assertTrue(any(c.args[0] == 's2_community_detection_persistent_failure'
                            for c in log.call_args_list))
