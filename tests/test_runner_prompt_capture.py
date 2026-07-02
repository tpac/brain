"""Full-prompt capture in run_llm_loop (eval/observability).

When BRAIN_PROMPT_CAPTURE_DIR is set AND the caller passes a capture_label,
run_llm_loop dumps the LITERAL per-round payload (system + messages) to a file
so the actual prompt is recoverable without an unfaithful rebuild. OFF by
default (no env, or no label) → zero files, zero production behavior change.

Contract locked here:
  - full system TEXT is captured (not a length, the bug in the legacy dump)
  - one file per round; the tool-result continuation (r1) carries the round-0
    assistant + tool_result blocks
  - files never overwrite each other (label + round + pid + monotonic seq)
  - no label, or no env → no capture
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales import runner
from servers.scales.runner import run_llm_loop


class _Usage:
    def __init__(self):
        self.input_tokens = 10
        self.output_tokens = 5
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _Block:
    def __init__(self, type, text=None, id=None, name=None, input=None):
        self.type, self.text, self.id, self.name, self.input = type, text, id, name, input


class _Msg:
    def __init__(self, content, stop_reason='end_turn'):
        self.content, self.stop_reason, self.usage = content, stop_reason, _Usage()


class _Stream:
    def __init__(self, final):
        self.final = final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        yield {'type': 'e0'}

    def get_final_message(self):
        return self.final


class _Messages:
    def __init__(self, scripted):
        self._s = list(scripted)

    def stream(self, **kw):
        return self._s.pop(0)


class _Client:
    def __init__(self, scripted):
        self.messages = _Messages(scripted)


def _two_round_script():
    # r0: one tool_use → forces a second round; r1: end_turn.
    r0 = _Msg([_Block('tool_use', id='t1', name='remember',
                      input={'title': 'A', 'content': 'c'})], stop_reason='tool_use')
    r1 = _Msg([_Block('text', text='done')], stop_reason='end_turn')
    return [_Stream(r0), _Stream(r1)]


def _run(client, capture_label):
    return run_llm_loop(
        client=client, model='claude-test', max_tokens=256, max_rounds=3,
        system_prompt='FULL SYSTEM PROMPT «with unicode»', user_content='BODY',
        user_preamble='PREAMBLE', tools=[],
        dispatch_fn=lambda name, args: {'ok': True, 'result': {'id': 'n1'},
                                        'affected': {'created': ['n1']}},
        log_fn=lambda m: None, capture_label=capture_label)


class TestPromptCapture(unittest.TestCase):

    def setUp(self):
        self._prev = os.environ.get('BRAIN_PROMPT_CAPTURE_DIR')
        self.tmp = tempfile.mkdtemp(prefix='promptcap-')

    def tearDown(self):
        if self._prev is None:
            os.environ.pop('BRAIN_PROMPT_CAPTURE_DIR', None)
        else:
            os.environ['BRAIN_PROMPT_CAPTURE_DIR'] = self._prev

    def _files(self):
        return sorted(f for f in os.listdir(self.tmp) if f.endswith('.json'))

    def test_captures_full_prompt_per_round(self):
        os.environ['BRAIN_PROMPT_CAPTURE_DIR'] = self.tmp
        _run(_Client(_two_round_script()), capture_label='new__sessX__stop5')
        files = self._files()
        self.assertEqual(len(files), 2, "expected r0 + r1 dumps, got %s" % files)

        payloads = [json.load(open(os.path.join(self.tmp, f))) for f in files]
        by_round = {p['round']: p for p in payloads}
        self.assertEqual(set(by_round), {0, 1})

        # Full system TEXT (not a length) — the legacy-dump bug this fixes.
        for p in payloads:
            self.assertEqual(p['system'], 'FULL SYSTEM PROMPT «with unicode»')
            self.assertEqual(p['model'], 'claude-test')
            self.assertEqual(p['label'], 'new__sessX__stop5')

        # r0 payload = the initial user turn (preamble + body blocks).
        r0_texts = [b['text'] for m in by_round[0]['messages']
                    for b in m['content'] if b.get('type') == 'text']
        self.assertIn('PREAMBLE', r0_texts)
        self.assertIn('BODY', r0_texts)

        # r1 payload carries the round-0 assistant tool_use + the tool_result
        # continuation — i.e. the ACTUAL later-round prompt, not a rebuild.
        roles = [m['role'] for m in by_round[1]['messages']]
        self.assertEqual(roles, ['user', 'assistant', 'user'])

    def test_no_overwrite_distinct_filenames(self):
        os.environ['BRAIN_PROMPT_CAPTURE_DIR'] = self.tmp
        _run(_Client(_two_round_script()), capture_label='new__sessX__stop5')
        _run(_Client(_two_round_script()), capture_label='new__sessX__stop5')  # same label!
        # 2 rounds × 2 runs = 4 files, all distinct despite the identical label.
        self.assertEqual(len(self._files()), 4)

    def test_no_label_no_capture(self):
        os.environ['BRAIN_PROMPT_CAPTURE_DIR'] = self.tmp
        _run(_Client(_two_round_script()), capture_label=None)
        self.assertEqual(self._files(), [])

    def test_no_env_no_capture(self):
        os.environ.pop('BRAIN_PROMPT_CAPTURE_DIR', None)
        _run(_Client(_two_round_script()), capture_label='new__sessX__stop5')
        self.assertEqual(self._files(), [])


if __name__ == '__main__':
    unittest.main()
