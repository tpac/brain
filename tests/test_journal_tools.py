"""Native journal tools: one-shot execution, cached requests and terminal loop calls."""
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from tests.test_journal_items import JournalTestBase, review
from tests.test_runner_per_round_stats import FakeClient, FakeStream, FakeMessage, FakeBlock
from servers.scales.runner import run_llm_once, run_llm_loop
from servers.scales.s2.healer_encoder import HealerEncoder
from servers.scales.s2.aspect_encoder import AspectEncoder
from servers.trace_contract import journal_tool_schema


def call(name, arguments, identity='call_1'):
    return FakeBlock('tool_use', name=name, input=arguments, id=identity)


class TestNativeJournalTools(JournalTestBase):
    def client(self, *blocks):
        create = Mock(return_value=FakeMessage(list(blocks), stop_reason='tool_use'))
        return NS(messages=NS(create=create))

    def test_one_response_keeps_all_calls_and_caches_the_entire_prompt(self):
        client = self.client(FakeBlock('text', text='Ignored prose'),
                             call('submit_healings', {'healings': []}),
                             call('journal', review(dict(op='note', subject='x', text='observation')), 'call_2'))
        calls, usage = run_llm_once(client, 'model', 100, 'SYSTEM', 'BATCH',
                                    tools=[journal_tool_schema()])
        self.assertEqual([c['name'] for c in calls], ['submit_healings', 'journal'])
        self.assertEqual(client.messages.create.call_count, 1)
        request = client.messages.create.call_args.kwargs
        self.assertEqual(request['system'][-1]['cache_control'], {'type': 'ephemeral', 'ttl': '5m'})
        self.assertEqual(request['messages'][-1]['content'][-1]['cache_control'],
                         {'type': 'ephemeral', 'ttl': '5m'})
        self.assertEqual(request['messages'][-1]['content'][-1]['text'], 'BATCH')
        self.assertEqual(usage['input_tokens'], 100)

    def test_healer_and_aspect_execute_journal_without_another_model_call(self):
        for unit_type, field in ((HealerEncoder, 'healings'), (AspectEncoder, 'classifications')):
            with self.subTest(unit=unit_type.__name__):
                unit = unit_type(self.brain)
                unit.journal.continuity(chain_id=unit.chain_id())
                client = self.client(
                    call('journal', review(dict(op='note', subject='native', text='keep ] } as text'))),
                    call(unit.RESULT_TOOL['name'], {field: []}, 'call_2'))
                with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                     patch.object(unit, '_llm_client', return_value=client):
                    payload, _ = unit._call_llm('s2_healer' if field == 'healings' else 's2_aspects', 'batch', journal=True)
                self.assertEqual(payload, [])
                self.assertEqual(client.messages.create.call_count, 1)
                self.assertIn('keep ] } as text', unit.journal.continuity(chain_id='s2-next-' + unit.NAME))

    def test_journal_failure_cannot_replace_task_result_or_trigger_a_retry(self):
        unit = HealerEncoder(self.brain)
        expected = [{'node_id': 'aaaaaaaa', 'question': 'What happened?'}]
        client = self.client(call('journal', review()),
                             call('submit_healings', {'healings': expected}, 'call_2'))
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
             patch.object(unit, '_llm_client', return_value=client), \
             patch.object(unit.journal, 'apply', side_effect=RuntimeError('storage failure')):
            payload, _ = unit._call_llm('s2_healer', 'batch', journal=True)
        self.assertEqual(payload, expected)
        self.assertEqual(client.messages.create.call_count, 1)

    def test_duplicate_or_missing_result_calls_are_rejected(self):
        unit = HealerEncoder(self.brain)
        for blocks in ((call('journal', review()),),
                       (call('submit_healings', {'healings': []}),
                        call('submit_healings', {'healings': []}, 'call_2'))):
            client = self.client(*blocks)
            with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                 patch.object(unit, '_llm_client', return_value=client):
                payload, _ = unit._call_llm('s2_healer', 'batch', journal=True)
            self.assertIsNone(payload)
            self.assertEqual(client.messages.create.call_count, 1)

    def test_healer_result_still_cannot_write_outside_its_batch(self):
        unit = HealerEncoder(self.brain)
        client = self.client(call('submit_healings', {'healings': [
            {'node_id': 'bbbbbbbb', 'question': 'Not in this batch'}]}))
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
             patch.object(unit, '_llm_client', return_value=client), \
             patch.object(unit, '_store_fields') as store:
            result = unit.run([{'node_id': 'aaaaaaaa', 'needs_question': True}])
        self.assertEqual(result['skipped'], 1)
        store.assert_not_called()

    def test_journal_only_loop_response_ends_without_counting_as_graph_work(self):
        binding = self.binding()
        self.prepare(binding, 1)
        response = FakeMessage([FakeBlock('text', text='DONE'),
            call('journal', review(dict(op='note', subject='test', text='no graph changes')))], 'tool_use')
        client = FakeClient([FakeStream(response)])
        dispatch = Mock()
        result = run_llm_loop(client, 'model', 1000, 3, 'system', 'user',
                              **binding.bind_tools([], dispatch, self.chain(binding, 1)))
        self.assertEqual(result['rounds'], 1)
        self.assertEqual(result['actions'], 0)
        self.assertEqual(result['write_actions'], 0)
        self.assertEqual(result['read_calls'], [])
        dispatch.assert_not_called()
        self.assertIn('no graph changes', binding.continuity(chain_id=self.chain(binding, 2)))

    def test_mixed_task_and_journal_calls_still_return_task_feedback(self):
        binding = self.binding()
        self.prepare(binding, 1)
        response = FakeMessage([call('revise', {'node_id': 'aaaaaaaa'}),
                                call('journal', review(), 'call_2')], 'tool_use')
        client = FakeClient([FakeStream(response), FakeStream(FakeMessage([FakeBlock('text', text='DONE')]))])
        dispatch = Mock(return_value={'ok': True, 'result': {}})
        result = run_llm_loop(client, 'model', 1000, 3, 'system', 'user',
                              **binding.bind_tools([], dispatch, self.chain(binding, 1)))
        self.assertEqual(result['rounds'], 2)
        self.assertEqual(result['actions'], 1)
        self.assertEqual(result['write_actions'], 1)
        dispatch.assert_called_once_with('revise', {'node_id': 'aaaaaaaa'})

    def test_final_journal_executes_at_round_limit_without_another_request(self):
        binding = self.binding()
        self.prepare(binding, 1)
        client = FakeClient([
            FakeStream(FakeMessage([call('revise', {'node_id': 'aaaaaaaa'})], 'tool_use')),
            FakeStream(FakeMessage([call('journal', review(dict(
                op='note', subject='limit', text='last response is retained')))], 'tool_use'))])
        dispatch = Mock(return_value={'ok': True, 'result': {}})
        result = run_llm_loop(client, 'model', 1000, 1, 'system', 'user',
                              **binding.bind_tools([], dispatch, self.chain(binding, 1)))
        self.assertEqual(result['rounds'], 2)
        self.assertEqual(result['actions'], 1)
        self.assertIn('last response is retained', binding.continuity(chain_id=self.chain(binding, 2)))

    def test_failed_task_does_not_report_journal_as_a_completed_graph_action(self):
        from servers.scales.runner import RunLoopError
        binding = self.binding()
        self.prepare(binding, 1)
        client = FakeClient([FakeStream(FakeMessage([
            call('journal', review(dict(op='note', subject='failure', text='keep'))),
            call('revise', {'node_id': 'aaaaaaaa'}, 'call_2')], 'tool_use'))])
        with self.assertRaises(RunLoopError) as caught:
            run_llm_loop(client, 'model', 1000, 2, 'system', 'user',
                         **binding.bind_tools([], Mock(side_effect=RuntimeError('failed')), self.chain(binding, 1)))
        self.assertEqual(caught.exception.partial_actions, [])
        self.assertIn('keep', binding.continuity(chain_id=self.chain(binding, 2)))
