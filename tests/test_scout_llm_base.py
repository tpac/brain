"""Tests for servers/scales/s1/scouts/base.py — LLM scout runner.

Mocks the Anthropic client so tests don't hit the network. Exercises:
- Happy path: valid output → wrapped + validated
- No DB row → resolver's code default runs (and sentinel overrides reach
  messages.create)
- API call failure → stub + logged error, latency captured
- Non-JSON output → stub + logged error
- Bad JSON shape → stub returned with errors
- Scout + category_statement injected from params (never LLM)
- Usage tokens captured
- Temporal rejected by name
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.scales.s1.scouts import base as scout_base
from servers.scales.s1.scouts import contract as sc


def _mock_anthropic_response(text, input_tokens=100, output_tokens=50,
                             cache_read=0, cache_create=0):
    """Build a fake anthropic SDK message response."""
    resp = MagicMock()
    block = MagicMock()
    block.text = text
    resp.content = [block]
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.usage.cache_read_input_tokens = cache_read
    resp.usage.cache_creation_input_tokens = cache_create
    return resp


def _mock_client(response_text):
    client = MagicMock()
    # run_llm_scout binds per-request timeout/retries via with_options before
    # calling create — make it a pass-through so the configured create is hit.
    client.with_options.return_value = client
    client.messages.create.return_value = _mock_anthropic_response(response_text)
    return client


def _shared_prefix():
    return sc.build_shared_prefix(
        session_context="design discussion",
        current_date="2026-04-23",
        catalog_rendered="Node abc123: 'Example'",
        surfaced_by_turn_rendered="t1: [abc123]",
        conversation_rendered="t1: user said hi\nassistant said hi back",
    )


class TestHappyPath(BrainTestBase):
    needs_embedder = False

    def test_valid_quote_output_wraps_and_validates(self):
        llm_output = json.dumps({
            "candidates": [{
                "handle": "tokens are for thinking",
                "speaker": "operator",
                "evidence_quote": "delegate grunt — tokens are for thinking",
                "evidence_turns": ["t3"],
                "why_candidate": "Echoed across lessons",
                "grounds_candidates": ["Delegation principle"],
                "echo_count": 2,
            }],
            "scanned": {"turns": 10, "phrases_considered": 3, "passed_threshold": 1},
        })
        client = _mock_client(llm_output)

        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)

        # Envelope fields injected by runner
        self.assertEqual(out['scout'], 'quote')
        self.assertTrue(out['category_statement'])  # from interaction.parameters
        self.assertEqual(len(out['candidates']), 1)
        self.assertEqual(out['candidates'][0]['handle'], 'tokens are for thinking')
        self.assertEqual(out[scout_base.SCOUT_ERROR_KEY], [])

    def test_runner_injects_scout_name_even_if_llm_says_otherwise(self):
        """LLM can't poison the scout field — runner overrides."""
        llm_output = json.dumps({
            "scout": "unknown_scout",  # wrong! LLM claims different scout
            "candidates": [],
            "scanned": {"turns": 5},
        })
        client = _mock_client(llm_output)
        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)
        self.assertEqual(out['scout'], 'quote')  # runner wins
        self.assertEqual(out[scout_base.SCOUT_ERROR_KEY], [])

    def test_category_statement_from_parameters_not_llm(self):
        """Even if the LLM invents a category_statement, runner replaces
        with the one from interaction.parameters — deterministic."""
        llm_output = json.dumps({
            "category_statement": "LLM made up this category",
            "candidates": [],
            "scanned": {"turns": 5},
        })
        client = _mock_client(llm_output)
        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)
        # Category is from the seed interaction, not the LLM's text
        self.assertIn("quote atom", out['category_statement'].lower())
        self.assertNotIn("made up", out['category_statement'].lower())

    def test_usage_and_latency_captured(self):
        llm_output = json.dumps({"candidates": [], "scanned": {"turns": 1}})
        resp = _mock_anthropic_response(
            llm_output, input_tokens=500, output_tokens=20,
            cache_read=300, cache_create=100)
        client = MagicMock()
        client.with_options.return_value = client
        client.messages.create.return_value = resp
        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)
        # Stub shape = runner.read_usage's USAGE_FIELDS short names (single
        # source for SDK field names — 2026-08-07 review, finding 8)
        usage = out[scout_base.SCOUT_TOKEN_USAGE_KEY]
        self.assertEqual(usage['input_tokens'], 500)
        self.assertEqual(usage['output_tokens'], 20)
        self.assertEqual(usage['cache_read_tokens'], 300)
        self.assertEqual(usage['cache_creation_tokens'], 100)
        self.assertGreaterEqual(out[scout_base.SCOUT_LATENCY_KEY], 0)


class TestSystemPromptAssembly(BrainTestBase):
    needs_embedder = False

    def test_per_scout_template_moved_to_system_1h_cache(self):
        """System prompt = SCOUT_SYSTEM_PROMPT + per-scout template with
        1h TTL cache_control. User content = shared prefix unchanged."""
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return _mock_anthropic_response(
                json.dumps({"candidates": [], "scanned": {"turns": 0}}))

        client = MagicMock()
        client.with_options.return_value = client
        client.messages.create.side_effect = capture

        scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)

        system = captured['system']
        self.assertIsInstance(system, list)
        self.assertEqual(len(system), 1)
        self.assertEqual(system[0]['cache_control'], {'type': 'ephemeral', 'ttl': '1h'})
        # System carries both the shared framing AND the scout-specific template
        self.assertIn('scout', system[0]['text'].lower())
        self.assertIn('quote', system[0]['text'].lower())  # scout-specific text

    def test_user_content_is_shared_prefix_unchanged(self):
        """Shared prefix passes through as user content; no per-scout
        task appended after the cache break."""
        captured = {}

        def capture(**kwargs):
            captured.update(kwargs)
            return _mock_anthropic_response(
                json.dumps({"candidates": [], "scanned": {"turns": 0}}))

        client = MagicMock()
        client.with_options.return_value = client
        client.messages.create.side_effect = capture

        prefix = _shared_prefix()
        scout_base.run_llm_scout(
            'quote', self.brain, prefix,
            anthropic_client=client)

        messages = captured['messages']
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['role'], 'user')
        # Content blocks match the shared prefix exactly (no extra task block)
        self.assertEqual(len(messages[0]['content']), len(prefix))

    def test_per_request_timeout_and_no_retries_are_bound(self):
        """Regression guard: the scout's timeout_seconds must be bound onto
        THIS request (with_options) + retries disabled — otherwise the shared
        client inherits the SDK ~600s default and a stalled scout becomes a
        ghost thread. timeout_seconds was dead config until this was wired."""
        client = MagicMock()
        client.with_options.return_value = client
        client.messages.create.return_value = _mock_anthropic_response(
            json.dumps({"candidates": [], "scanned": {"turns": 0}}))

        # Sentinel through the K-store: an override of ONE key must reach
        # the request options (resolver overlays it onto the code default).
        # Empty the store (a no-op on a fresh brain — nothing seeds rows
        # anymore) so the fresh registration is deterministically v1.
        self.brain.logs_conn.execute('DELETE FROM interactions')
        self.brain.logs_conn.commit()
        self.brain._interaction_dal.register(
            's1_scout_quote', template='',
            parameters=json.dumps({'timeout_seconds': 33}))
        self.brain._interaction_dal.set_active(
            's1_scout_quote', 1, set_by='test')
        scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)

        client.with_options.assert_called_once_with(timeout=33.0, max_retries=0)


class TestFailurePaths(BrainTestBase):
    needs_embedder = False

    def test_no_db_row_runs_on_code_default(self):
        """Delete every interaction row: the resolver falls back to the code
        default, so the scout RUNS (missing-row and empty-template stub
        paths died with the override migration — a resolved template+config
        is total by construction)."""
        from servers.scales.s1.scouts.contract import (
            SCOUT_QUOTE_INTERACTION_DEFAULT)
        self.brain.logs_conn.execute('DELETE FROM interactions')
        self.brain.logs_conn.commit()
        client = _mock_client(
            json.dumps({"candidates": [], "scanned": {"turns": 0}}))
        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)
        self.assertEqual(out[scout_base.SCOUT_ERROR_KEY], [])
        call_kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs['model'],
                         SCOUT_QUOTE_INTERACTION_DEFAULT['model'])
        self.assertEqual(call_kwargs['max_tokens'],
                         SCOUT_QUOTE_INTERACTION_DEFAULT['max_tokens'])

    def test_sentinel_model_override_reaches_the_call(self):
        """A one-key model override in the K-store must be the model that
        reaches messages.create (the a6dfcfe3 failure shape: an override
        landing on the fallback side of a get() chain fails silently)."""
        self.brain.logs_conn.execute('DELETE FROM interactions')
        self.brain.logs_conn.commit()
        self.brain._interaction_dal.register(
            's1_scout_quote', template='',
            parameters=json.dumps({'model': 'sentinel-scout-model'}))
        self.brain._interaction_dal.set_active(
            's1_scout_quote', 1, set_by='test')
        client = _mock_client(
            json.dumps({"candidates": [], "scanned": {"turns": 0}}))
        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)
        self.assertEqual(out[scout_base.SCOUT_ERROR_KEY], [])
        self.assertEqual(
            client.messages.create.call_args.kwargs['model'],
            'sentinel-scout-model')

    def test_api_error_captures_stub_and_latency(self):
        client = MagicMock()
        client.with_options.return_value = client
        client.messages.create.side_effect = RuntimeError('network down')
        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)
        self.assertEqual(out['candidates'], [])
        self.assertTrue(any(e['type'] == 'api_error'
                            for e in out[scout_base.SCOUT_ERROR_KEY]))
        self.assertIn('network down',
                      out[scout_base.SCOUT_ERROR_KEY][0]['msg'])
        # Latency still captured (the runner timed the failing call)
        self.assertGreaterEqual(out[scout_base.SCOUT_LATENCY_KEY], 0)

    def test_non_json_output_returns_stub_and_logs(self):
        client = _mock_client("not JSON at all, just prose")
        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)
        self.assertEqual(out['candidates'], [])
        self.assertTrue(any(e['type'] == 'json_parse'
                            for e in out[scout_base.SCOUT_ERROR_KEY]))

    def test_bad_json_shape_fails_validation(self):
        """JSON parses but envelope is unusable — returns stub with errors."""
        # scout will be injected as 'quote', but candidates is malformed
        client = _mock_client(json.dumps({"candidates": "not a list"}))
        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)
        self.assertEqual(out['candidates'], [])
        self.assertTrue(any(e['type'] == 'schema_invalid'
                            for e in out[scout_base.SCOUT_ERROR_KEY]))

    def test_soft_truncation_emits_warning_not_error(self):
        llm_output = json.dumps({
            "candidates": [{
                "handle": "x" * 500,  # over the 120-char cap
                "speaker": "operator",
                "evidence_quote": "q",
                "evidence_turns": ["t1"],
                "why_candidate": "w",
            }],
            "scanned": {"turns": 1},
        })
        client = _mock_client(llm_output)
        out = scout_base.run_llm_scout(
            'quote', self.brain, _shared_prefix(),
            anthropic_client=client)
        self.assertEqual(out[scout_base.SCOUT_ERROR_KEY], [])
        self.assertTrue(any('truncated' in w
                            for w in out[scout_base.SCOUT_WARNING_KEY]))
        # Candidate kept, just trimmed
        self.assertEqual(len(out['candidates'][0]['handle']),
                         sc.FIELD_LIMITS['handle'])


class TestProgrammerErrors(unittest.TestCase):

    def test_temporal_rejected_by_llm_runner(self):
        """Temporal is algorithmic; calling run_llm_scout on 'temporal' is
        a programmer mistake and must raise."""
        with self.assertRaises(ValueError):
            scout_base.run_llm_scout('temporal', MagicMock(), [])

    def test_unknown_scout_rejected(self):
        with self.assertRaises(ValueError):
            scout_base.run_llm_scout('unknown', MagicMock(), [])


class TestJSONExtraction(unittest.TestCase):

    def test_plain_json(self):
        result = scout_base._extract_json('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})

    def test_code_fence_json(self):
        result = scout_base._extract_json(
            'Here is the result:\n```json\n{"key": "value"}\n```\n')
        self.assertEqual(result, {"key": "value"})

    def test_code_fence_no_lang(self):
        result = scout_base._extract_json(
            'Result:\n```\n{"key": "value"}\n```')
        self.assertEqual(result, {"key": "value"})

    def test_leading_prose_only(self):
        """If the model prefixes prose but the JSON object is findable."""
        result = scout_base._extract_json(
            'Here you go: {"key": "value"} thanks!')
        self.assertEqual(result, {"key": "value"})

    def test_invalid_returns_none(self):
        self.assertIsNone(scout_base._extract_json('nothing parseable'))
        self.assertIsNone(scout_base._extract_json(''))
        self.assertIsNone(scout_base._extract_json(None))


if __name__ == '__main__':
    unittest.main()
