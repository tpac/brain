"""Phase 2b delivery — PreToolUse + Stop hook integration.

The self-channel envelope bug (c4f6386) escaped because inner functions were
unit-tested but the dispatch/handler wiring was not. This layer closes that for
DELIVERY: it drives the real daemon handlers — hook_pre_edit (PreToolUse) and
hook_post_response_track (Stop) — and asserts the self-message lands on each
hook's channel, plus the cross-hook consume-once property (a message is
delivered by exactly one hook, never twice). on_prompt is deliberately not a
delivery point (weak channel; would win the consume-once race).
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.daemon_hooks import hook_pre_edit, hook_post_response_track
from servers.scales.self_channel import signal, self_contract


def _seed(brain, to_stream, body):
    """Queue a directed self-message from a different stream (so drain_inbox's
    'not your own' filter delivers it)."""
    signal.send(brain, from_session='other-stream',
                address=self_contract.address_for_stream(to_stream), body=body)


def _edit(brain, session_id, filename='auth.py'):
    return hook_pre_edit(brain, {'filename': filename, 'tool_name': 'Edit',
                                 'session_id': session_id}, [])


def _stop(brain, session_id):
    return hook_post_response_track(brain, {
        'hook_event_name': 'Stop', 'session_id': session_id,
        'prompt': 'a user message long enough to store',
        'last_assistant_message': 'done'}, [])


def _pretool_ac(out):
    """The additionalContext a PreToolUse hook surfaces to the model — the
    channel that actually reaches Claude on allow (NOT `reason`)."""
    return out.get('json', {}).get('hookSpecificOutput', {}).get('additionalContext', '')


class TestPreToolDelivery(BrainTestBase):
    needs_embedder = False

    def test_pretool_delivers_into_additional_context(self):
        _seed(self.brain, 'S', 'heads up: I am in auth.py')
        out = _edit(self.brain, 'S')
        self.assertIn('heads up: I am in auth.py', _pretool_ac(out))

    def test_pretool_consume_once(self):
        _seed(self.brain, 'S', 'one tap')
        self.assertIn('one tap', _pretool_ac(_edit(self.brain, 'S', 'x.py')))
        self.assertNotIn('one tap', _pretool_ac(_edit(self.brain, 'S', 'x.py')))

    def test_pretool_no_message_is_clean_approve(self):
        """Regression: nothing pending → valid approve, no spurious self-block."""
        out = _edit(self.brain, 'S', 'y.py')
        self.assertEqual(out.get('json', {}).get('decision'), 'approve')
        self.assertNotIn('🧵', _pretool_ac(out))


class TestStopDelivery(BrainTestBase):
    needs_embedder = False

    def test_stop_blocks_to_deliver(self):
        _seed(self.brain, 'S', 'before you finish: note X')
        out = _stop(self.brain, 'S')
        self.assertEqual(out.get('decision'), 'block')
        self.assertIn('before you finish: note X', out.get('reason', ''))

    def test_stop_consume_once(self):
        _seed(self.brain, 'S', 'one tap')
        self.assertEqual(_stop(self.brain, 'S').get('decision'), 'block')
        # already consumed → next stop is allowed (no block, no loop)
        self.assertNotEqual(_stop(self.brain, 'S').get('decision'), 'block')

    def test_stop_no_message_allows_stop(self):
        """Regression: nothing pending → no block, normal output."""
        self.assertNotEqual(_stop(self.brain, 'FRESH').get('decision'), 'block')


class TestCrossHookConsumeOnce(BrainTestBase):
    needs_embedder = False

    def test_pretool_wins_then_stop_finds_nothing(self):
        _seed(self.brain, 'S', 'single tap')
        pre = _edit(self.brain, 'S', 'a.py')
        self.assertIn('single tap', _pretool_ac(pre))
        # PreToolUse already consumed it → Stop must NOT re-deliver
        self.assertNotEqual(_stop(self.brain, 'S').get('decision'), 'block')


if __name__ == '__main__':
    unittest.main()
