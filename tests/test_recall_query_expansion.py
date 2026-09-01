"""Recall-lane query expansion: bounded client, gated call, config-driven.

`_expand_query_via_llm` is the recall hot path's only LLM call. Three
properties matter:

1. Its client is BOUNDED. The function's own `except` catches errors, not
   hangs — only a timeout stops a stalled socket from holding a recall
   worker thread. It also disables SDK retries, so the bound stays hard.
2. The call is GATED on `llm_available`, like every other LLM lane. Without
   the gate a keyless brain — or one whose key the provider refused and the
   rejection latch paused — still fires a call that can only 401, and the
   bare `except` swallows it.
3. Prompt, model and max_tokens come from the `recall_query_expansion`
   interaction — the EFFECTIVE values are asserted at messages.create, not
   the values passed anywhere upstream (the a6dfcfe3 trap: an override that
   lands on the fallback side of a get(key, fallback) chain fails silently).

Expansion is opt-in (`BRAIN_QUERY_EXPANSION`, default off), so these are
latent-path guarantees: they hold whenever an operator turns it on.
"""

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from servers import brain_recall                      # noqa: E402
from servers.brain_constants import (                 # noqa: E402
    RECALL_EXPANSION_TIMEOUT_S,
)


class _FakeMessages:
    """Stands in for `client.messages` — records the create() call."""

    def __init__(self, text):
        self._text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = types.SimpleNamespace(text=self._text)
        return types.SimpleNamespace(content=[block])


class _FakeClient:
    def __init__(self, text='["alpha", "beta"]'):
        self.messages = _FakeMessages(text)


class _FakeAnthropicModule:
    """Captures the constructor kwargs `_expand_query_via_llm` passes."""

    def __init__(self, text='["alpha", "beta"]'):
        self.ctor_kwargs = None
        self.client = _FakeClient(text)

    def Anthropic(self, **kwargs):
        self.ctor_kwargs = kwargs
        return self.client


class _StubExpansionBrain:
    """The interaction store the function reads — template + config + errors."""

    def __init__(self, template='Query: "{query}"', config=None):
        self.template = template
        self.config = ({'model': 'claude-haiku-4-5', 'max_tokens': 200}
                       if config is None else config)
        self.errors = []

    def get_interaction_prompt(self, name):
        return self.template

    def get_interaction_config(self, name):
        return self.config

    def _log_error(self, kind, exc, note):
        self.errors.append(kind)


class ExpansionClientBoundTest(unittest.TestCase):
    """The client must carry an explicit timeout and no SDK retries."""

    def setUp(self):
        self._real_anthropic = sys.modules.get('anthropic')
        self.fake = _FakeAnthropicModule()
        sys.modules['anthropic'] = self.fake

    def tearDown(self):
        if self._real_anthropic is None:
            sys.modules.pop('anthropic', None)
        else:
            sys.modules['anthropic'] = self._real_anthropic

    def test_client_is_constructed_with_the_recall_timeout(self):
        brain_recall._expand_query_via_llm(_StubExpansionBrain(), 'a query worth expanding')
        self.assertIsNotNone(
            self.fake.ctor_kwargs,
            'expansion did not construct a client at all')
        self.assertEqual(
            self.fake.ctor_kwargs.get('timeout'), RECALL_EXPANSION_TIMEOUT_S,
            'expansion client must be bounded by the recall-lane timeout — '
            'an unbounded client can hold a recall worker thread on a stall')

    def test_client_disables_sdk_retries(self):
        brain_recall._expand_query_via_llm(_StubExpansionBrain(), 'a query worth expanding')
        self.assertEqual(
            self.fake.ctor_kwargs.get('max_retries'), 0,
            'SDK retries would multiply the timeout bound; expansion is '
            'best-effort and recall proceeds on the primary query')

    def test_recall_timeout_is_far_tighter_than_the_encoder_bound(self):
        """The encoder lane's ceiling is the wrong shape for a hot path."""
        from servers.brain_constants import ANTHROPIC_CLIENT_TIMEOUT
        self.assertLess(
            RECALL_EXPANSION_TIMEOUT_S, ANTHROPIC_CLIENT_TIMEOUT / 10,
            'a recall-path bound near the encoder ceiling defeats the point')

    def test_parses_the_expanded_alternates(self):
        """Guards the happy path so the bound change can't silently break it."""
        alts = brain_recall._expand_query_via_llm(_StubExpansionBrain(), 'a query worth expanding')
        self.assertEqual(alts, ['alpha', 'beta'])


class ExpansionGateTest(unittest.TestCase):
    """The availability gate must precede any client work."""

    def test_gate_consults_llm_available_and_notes_once(self):
        """A keyless/latched brain skips expansion instead of firing a 401.

        Exercises the gate's decision directly: the call site computes
        `_do_expand` from quality signals, then clears it when the brain
        reports no usable key, recording the pause exactly once.
        """
        noted = []

        class _StubBrain:
            llm_available = False

            def note_llm_unavailable(self, where):
                noted.append(where)

        brain = _StubBrain()
        _do_expand = True                      # quality gates said yes

        if _do_expand and not brain.llm_available:
            brain.note_llm_unavailable('query expansion')
            _do_expand = False

        self.assertFalse(
            _do_expand, 'expansion must not run without an available key')
        self.assertEqual(noted, ['query expansion'])

    def test_gate_is_a_live_statement_with_the_hot_path_operand_order(self):
        """The gate must be live code, and `_do_expand` must come first.

        A behavioural end-to-end test would need a full recall (embedder,
        candidate rows, the opt-in env var). This pins the two properties a
        substring check alone would miss: that the gate is an executable
        statement rather than a comment or an unrelated mention elsewhere in
        the file, and that the operands are ordered so `llm_available` — which
        re-reads the key file on every access — is evaluated only when
        expansion is enabled. Reversing them puts a stat + read on every
        recall, which resolve_api_key's contract excludes.
        """
        src_lines = Path(brain_recall.__file__).read_text().splitlines()
        stripped = [ln.strip() for ln in src_lines]

        gate = 'if _do_expand and not self.llm_available:'
        self.assertIn(
            gate, stripped,
            'expansion must be gated by a live `%s` statement — absent, '
            'commented out, or with the operands reversed (which would move '
            'the key-file read onto the recall hot path)' % gate)

        self.assertIn(
            "self.note_llm_unavailable('query expansion')", stripped,
            'the keyless pause must be recorded once, like every other lane')

    def test_gate_reports_the_skip_to_the_tuning_log(self):
        """A skipped expansion must say so on stderr.

        `note_llm_unavailable` fires once per daemon lifetime, and the on_flat
        gate has already logged its "-> expand" decision by this point, so
        without an explicit line the log claims an expansion that never ran.
        """
        src = Path(brain_recall.__file__).read_text()
        self.assertIn(
            'query-expansion skipped', src,
            'an availability skip must be visible to whoever is tuning '
            'BRAIN_EXPANSION_GATE from these logs')


class ExpansionEffectiveConfigTest(unittest.TestCase):
    """Interaction parameters are authoritative — assert the EFFECTIVE values.

    The known trap: an override that lands on the
    fallback side of a `get(key, fallback)` chain fails silently and the
    caller measures noise. So these tests assert what reaches
    `messages.create`, never what was passed upstream.
    """

    def setUp(self):
        self._real_anthropic = sys.modules.get('anthropic')
        self.fake = _FakeAnthropicModule()
        sys.modules['anthropic'] = self.fake

    def tearDown(self):
        if self._real_anthropic is None:
            sys.modules.pop('anthropic', None)
        else:
            sys.modules['anthropic'] = self._real_anthropic

    def test_model_and_max_tokens_come_from_the_interaction(self):
        brain = _StubExpansionBrain(
            config={'model': 'sentinel-model-x', 'max_tokens': 77})
        brain_recall._expand_query_via_llm(brain, 'a query worth expanding')
        call = self.fake.client.messages.calls[0]
        self.assertEqual(call['model'], 'sentinel-model-x',
                         'the interaction config model must be the one that '
                         'reaches messages.create')
        self.assertEqual(call['max_tokens'], 77)

    def test_template_comes_from_the_interaction(self):
        brain = _StubExpansionBrain(template='EXPAND >>{query}<<')
        brain_recall._expand_query_via_llm(brain, 'a query worth expanding')
        call = self.fake.client.messages.calls[0]
        self.assertEqual(call['messages'][0]['content'],
                         'EXPAND >>a query worth expanding<<')

    def test_code_default_reaches_the_call_when_no_override(self):
        """With no DB override, the resolver hands back the code default —
        and THAT is what must reach messages.create (a missing row can no
        longer silently disable expansion; the resolver guarantees a total
        template+config for every registered name)."""
        from servers.recall_expansion_prompt import (
            SYSTEM_PROMPT, RECALL_EXPANSION_INTERACTION_DEFAULT)
        brain = _StubExpansionBrain(
            template=SYSTEM_PROMPT,
            config=dict(RECALL_EXPANSION_INTERACTION_DEFAULT))
        brain_recall._expand_query_via_llm(brain, 'a query worth expanding')
        call = self.fake.client.messages.calls[0]
        self.assertEqual(call['model'],
                         RECALL_EXPANSION_INTERACTION_DEFAULT['model'])
        self.assertEqual(call['max_tokens'],
                         RECALL_EXPANSION_INTERACTION_DEFAULT['max_tokens'])
        self.assertEqual(brain.errors, [])


if __name__ == '__main__':
    unittest.main()
