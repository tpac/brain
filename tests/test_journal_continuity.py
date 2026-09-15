"""Real encoder assembly and harvest paths share the JSON journal contract."""
import os
import re
from unittest.mock import patch
from tests.brain_test_base import BrainTestBase
from tests.test_journal_items import review
from servers.scales.s1.scribe import S1Scribe
from servers.scales.s2.community_encoder import CommunityEncoder
from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
from servers.scales.s2.healer_encoder import HealerEncoder
from servers.scales.s2.aspect_encoder import AspectEncoder


def handle(text):
    return re.search(r'journal_[0-9a-f]{8}', text)[0]


class TestJournalEncoderPaths(BrainTestBase):
    needs_embedder = False

    def seed(self, binding, *lines):
        chain = 's1e-prior-0' if binding.scale == 's1' else 's2-prior-' + binding.unit
        operations = []
        for line in lines:
            label, subject, text = line.split(' · ', 2)
            operations.append(dict(op='note', subject=subject, text=text, persist=True))
        binding.apply(review(*operations), chain)

    def run_scribe(self, unit, loop):
        from servers.scales.s1 import encode
        with patch.object(encode, 'load_env'), \
             patch('servers.scales.runner.make_client', return_value=object()), \
             patch.object(encode, 'run_llm_loop', side_effect=loop), \
             patch.object(encode, '_lived_sequence_enabled', return_value=True), \
             patch('servers.scales.s1.encoder_view.view_policy_enabled', return_value=False), \
             patch.object(encode, '_gather_messages', return_value=[{'role': 'user', 'content': 'verified'}]), \
             patch.object(encode, '_build_catalog', return_value=('', set(), {})), \
             patch.object(encode, '_render_lived_sequence_timeline', return_value='verified'), \
             patch.object(encode, '_write_pre_traces'):
            return unit.run()

    def test_actual_batch_paths_refresh_references(self):
        from servers.scales.s2.healer_contract import HEALER
        from servers.scales.s2.consolidation_contract import CONSOLIDATION
        units = [HealerEncoder(self.brain, config={**HEALER, 'max_nodes_per_call': 1}),
                 ConsolidationEncoder(self.brain, config={**CONSOLIDATION, 'max_proposals_per_call': 1}),
                 CommunityEncoder(self.brain, config={'max_proposals_per_call': 1})]
        for unit in units:
            with self.subTest(unit=unit.NAME):
                self.seed(unit.journal, 'open · target · pending')
                seen = []

                def response(user):
                    seen.append(user)
                    return review(dict(op='edit', id=handle(user), persist=False, text='verified')) if len(seen) == 1 else review()

                def once(client, model, max_tokens, system, user, *, tools):
                    return [dict(name='submit_healings', input={'healings': []}),
                            dict(name='journal', input=response(user))], {}

                def loop(**kw):
                    kw['dispatch_fn']('journal', response(kw['user_content']))
                    return dict(rounds=1, actions=0, write_actions=0, final_text='DONE')

                with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                     patch('servers.scales.s2.base.make_client', return_value=object()), \
                     patch('servers.scales.s2.base.run_llm_once', side_effect=once), \
                     patch('servers.scales.runner.make_client', return_value=object()), \
                     patch('servers.scales.runner.run_llm_loop', side_effect=loop):
                    if unit.NAME == 'healer':
                        unit.run([{'node_id': 'aaaaaaaa'}, {'node_id': 'bbbbbbbb'}])
                    elif unit.NAME == 'consolidation':
                        with patch.object(unit, '_format_clusters', return_value='clusters'):
                            unit._encode([{'nodes': []}, {'nodes': []}])
                    else:
                        with patch.object(unit, '_build_batch_context',
                                          side_effect=lambda batch, state, continuity, limit: (continuity, {})), \
                             patch('servers.scales.s2.community_decoder.CommunityDecoder._read_community_state', return_value=[]):
                            unit._encode([{}, {}], [])
                self.assertEqual(len(seen), 2)
                self.assertEqual(handle(seen[0]), handle(seen[1]))
                self.assertIn('verified', seen[1])
                self.assertNotIn('pending', seen[1])
                self.assertEqual(unit.journal._view['run_number'], 1)

    def test_scribe_run_assembles_escaped_reference_and_harvests_with_arc(self):
        unit = S1Scribe(self.brain, 'session-x', 1)
        self.seed(unit.journal, 'open · target · check <loaded> & current code')
        seen = {}

        def loop(**kw):
            seen.update(kw)
            kw['dispatch_fn']('journal', review(dict(op='edit', id=handle(kw['user_content']),
                                                    persist=False, text='confirmed')))
            return dict(rounds=1, actions=0, write_actions=0,
                        final_text='## Arc\n```\nVerified the deployment.\n```\nDONE')

        result = self.run_scribe(unit, loop)
        self.assertNotIn('error', result)
        self.assertIn('&lt;loaded&gt; &amp; current code', seen['user_content'])
        self.assertIn('never `journal_<id>`', seen['system_prompt'])
        self.assertIn('never belong in targets, fetch', seen['user_content'])
        self.assertIn('Verified the deployment', self.brain.session_context_for('session-x'))
        self.assertIn('confirmed', unit.journal.continuity(chain_id='s1e-session--2'))

    def test_scribe_failure_remains_visible_after_journal_adoption(self):
        unit = S1Scribe(self.brain, 'session-x', 1)
        unit.journal.continuity(chain_id='s1e-session--1')
        result = self.run_scribe(unit, RuntimeError('provider unavailable'))
        self.assertIn('error', result)
        view = unit.journal.continuity(chain_id='s1e-session--2')
        self.assertIn('encoding-run-failure', view)
        self.assertIn('provider unavailable', view)
        events = self.brain.query_traces(ref_type='encoding_run_failed',
                                        session_id='session-x', hours=None)['events']
        self.assertEqual(len(events), 1)
