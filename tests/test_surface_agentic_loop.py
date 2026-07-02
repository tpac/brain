"""Unit tests for the v5_agentic surface loop (_call_surface_agentic).

Covers the 2026-07-02 loop changes:
  - Prompt-cache breakpoint on the round-1 user content, tools on EVERY
    round (byte-identical prefix so round 2 cache-hits).
  - Forced-finalize fallback: Haiku tool-uses on the final round → one
    extra tools-stripped call, loudly logged.
  - Admission-floor tripwire + result_ids/dropped_ids trace attribution.

All tests run against fakes — no brain, no network.
"""
import json
from types import SimpleNamespace

import pytest

from servers.scales.s1 import surface as surface_mod
from servers.scales.s1 import fetch_tools as fetch_tools_mod


SELECTION_JSON = '{"selected":[{"id":"aaaa1111","why":"w","mode":"arc"}]}'


class FakeUsage:
    def __init__(self, input_tokens=10000, output_tokens=100,
                 cache_read=0, cache_creation=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation


def text_response(text=SELECTION_JSON, **usage_kw):
    return SimpleNamespace(
        stop_reason='end_turn',
        content=[SimpleNamespace(type='text', text=text)],
        usage=FakeUsage(**usage_kw))


def tool_use_response(tool='recall_topical', tool_input=None, **usage_kw):
    return SimpleNamespace(
        stop_reason='tool_use',
        content=[SimpleNamespace(
            type='tool_use', id='tu_1', name=tool,
            input=tool_input or {'query': 'x'})],
        usage=FakeUsage(**usage_kw))


class FakeClient:
    """Scripted client — pops responses in order, records every kwargs."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeBrain:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def _log_error(self, source, exc, context=''):
        self.errors.append((source, str(exc), context))

    def _log_warning(self, source, message, context=''):
        self.warnings.append((source, message, context))


def run_loop(client, brain, candidates, max_rounds=2):
    return surface_mod._call_surface_agentic(
        client, brain, candidates, 'SYSTEM', 'USER CONTENT', 600,
        'sess-test', 'claude-haiku-4-5', max_rounds=max_rounds)


class TestCachePrefix:
    def test_first_message_carries_cache_control(self):
        client = FakeClient([text_response()])
        raw, trace, tel = run_loop(client, FakeBrain(), [])
        first_msg = client.calls[0]['messages'][0]
        block = first_msg['content'][0]
        assert block['cache_control'] == {'type': 'ephemeral'}
        assert block['text'] == 'USER CONTENT'
        assert raw == SELECTION_JSON

    def test_tools_present_on_every_round(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        client = FakeClient([tool_use_response(), text_response()])
        run_loop(client, FakeBrain(), [])
        assert len(client.calls) == 2
        for call in client.calls:
            assert call['tools'] == fetch_tools_mod.TOOL_DEFINITIONS

    def test_cache_miss_on_round2_warns(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        brain = FakeBrain()
        client = FakeClient([
            tool_use_response(input_tokens=20000),
            # round 2: big prompt, zero cache read → should warn
            text_response(input_tokens=22000, cache_read=0),
        ])
        run_loop(client, brain, [])
        assert any(w[0] == 'surface_cache_miss' for w in brain.warnings)

    def test_cache_hit_on_round2_quiet(self, monkeypatch):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': [], 'latency_ms': 1})
        brain = FakeBrain()
        client = FakeClient([
            tool_use_response(input_tokens=20000, cache_creation=19000),
            text_response(input_tokens=2000, cache_read=19000),
        ])
        raw, trace, tel = run_loop(client, brain, [])
        assert not any(w[0] == 'surface_cache_miss' for w in brain.warnings)
        assert tel['cache_read_tokens'] == 19000


class TestForcedFinalize:
    def test_final_round_tool_use_forces_selection(self):
        brain = FakeBrain()
        # max_rounds=1 → round 0 is final; Haiku tool-uses → forced call.
        client = FakeClient([tool_use_response(), text_response()])
        raw, trace, tel = run_loop(client, brain, [], max_rounds=1)
        assert len(client.calls) == 2
        assert 'tools' in client.calls[0]
        assert 'tools' not in client.calls[1]      # forced call strips tools
        assert raw == SELECTION_JSON
        assert trace[-1].get('forced_finalize') == 1
        assert tel['rounds'] == 2
        assert any(w[0] == 'surface_forced_finalize' for w in brain.warnings)

    def test_empty_tool_use_round_retries_without_history_append(self):
        # stop_reason='tool_use' with ZERO tool_use blocks (the May-2026
        # Haiku mode) must not append empty messages (API 400 next round).
        # The loop retries with untouched history, then finalizes normally.
        brain = FakeBrain()
        empty_tool_use = SimpleNamespace(
            stop_reason='tool_use', content=[], usage=FakeUsage())
        client = FakeClient([empty_tool_use, text_response()])
        raw, trace, tel = run_loop(client, brain, [])
        assert raw == SELECTION_JSON
        assert len(client.calls) == 2
        assert len(client.calls[1]['messages']) == 1  # history untouched
        assert any(w[0] == 'surface_empty_tool_use' for w in brain.warnings)

    def test_forced_call_reuses_history_without_orphan_tool_use(self):
        # The tool_use assistant message must NOT be appended before the
        # forced call — tool_use without tool_result is an API 400.
        client = FakeClient([tool_use_response(), text_response()])
        run_loop(client, FakeBrain(), [], max_rounds=1)
        forced_messages = client.calls[1]['messages']
        assert len(forced_messages) == 1
        assert forced_messages[0]['role'] == 'user'


class TestFloorAndAttribution:
    def _run_with_fetched(self, monkeypatch, pool, fetched):
        monkeypatch.setattr(
            fetch_tools_mod, 'execute_tool',
            lambda *a, **kw: {'results': list(fetched), 'latency_ms': 1})
        # Fakes carry only id/score — skip the rich candidate renderer.
        monkeypatch.setattr(
            fetch_tools_mod, 'format_tool_result_for_haiku',
            lambda result: '%d results' % len(result.get('results') or []))
        brain = FakeBrain()
        client = FakeClient([tool_use_response(), text_response()])
        raw, trace, tel = run_loop(client, brain, pool)
        calls = [c for r in trace for c in r.get('tool_calls', [])]
        return brain, pool, calls

    def test_floor_drops_all_trips_warning_and_records_ids(self, monkeypatch):
        pool = [{'id': 'p%d' % i, 'score': s}
                for i, s in enumerate([0.5, 0.6, 0.7])]
        fetched = [{'id': 'f1f1f1f1', 'score': 0.1},
                   {'id': 'f2f2f2f2', 'score': 0.2}]
        brain, pool_after, calls = self._run_with_fetched(
            monkeypatch, pool, fetched)
        assert calls[0]['result_count'] == 0
        assert calls[0]['result_ids'] == []
        assert calls[0]['dropped_below_floor'] == 2
        assert calls[0]['dropped_ids'] == ['f1f1f1f1', 'f2f2f2f2']
        assert any(w[0] == 'surface_floor_dropped_all'
                   for w in brain.warnings)
        # Dropped candidates never join the selection pool.
        assert len(pool_after) == 3

    def test_admitted_results_join_pool_with_ids(self, monkeypatch):
        pool = [{'id': 'p%d' % i, 'score': s}
                for i, s in enumerate([0.5, 0.6, 0.7])]
        fetched = [{'id': 'f1f1f1f1', 'score': 0.9},
                   {'id': 'f2f2f2f2', 'score': 0.1}]
        brain, pool_after, calls = self._run_with_fetched(
            monkeypatch, pool, fetched)
        assert calls[0]['result_count'] == 1
        assert calls[0]['result_ids'] == ['f1f1f1f1']
        assert calls[0]['dropped_ids'] == ['f2f2f2f2']
        # Partial drop is normal — no tripwire.
        assert not any(w[0] == 'surface_floor_dropped_all'
                       for w in brain.warnings)
        assert len(pool_after) == 4
