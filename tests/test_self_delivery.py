"""Self-message delivery — the Stop hook is the SOLE path (Stop-only, 2026-06-04).

The self-channel envelope bug (c4f6386) escaped because inner functions were
unit-tested but the dispatch/handler wiring was not. This layer closes that for
DELIVERY: it drives the real daemon handlers — hook_pre_edit (PreToolUse) and
hook_post_response_track (Stop) — and asserts that self-messages land on the
prominent Stop block and that PreToolUse stays silent. PreToolUse delivery into
additionalContext was removed (the model missed it, and consuming there starved
the reliable Stop block); UserPromptSubmit was never a delivery point. So
PreToolUse must NOT surface or consume the tap — it leaves it for Stop.
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.daemon_hooks import hook_pre_edit, hook_post_response_track
from servers.channels.self_channel import signal, self_contract


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


class TestPreToolDoesNotDeliver(BrainTestBase):
    """Stop-only (2026-06-04): PreToolUse must NOT surface a self-message and
    must NOT consume it — the tap survives untouched so the Stop block delivers."""
    needs_embedder = False

    def test_pretool_does_not_surface_self_message(self):
        _seed(self.brain, 'S', 'heads up: I am in auth.py')
        out = _edit(self.brain, 'S')
        self.assertNotIn('heads up: I am in auth.py', _pretool_ac(out))
        self.assertEqual(out.get('json', {}).get('decision'), 'approve')

    def test_pretool_does_not_consume_tap(self):
        # The tap must survive PreToolUse so Stop can still deliver it.
        _seed(self.brain, 'S', 'one tap')
        _edit(self.brain, 'S', 'x.py')
        self.assertEqual(_stop(self.brain, 'S').get('decision'), 'block')

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

    def test_stop_trace_attributes_to_recipient_session(self):
        """Regression (2026-06-05): the self_message delivery trace must carry
        the recipient session_id. It was written empty while its three sibling
        S0 appends passed it, so a session-scoped (dashboard) query couldn't see
        the delivery. All four S0 turn-traces now bind session_id via _s0_trace —
        a session-scoped query must return the delivery."""
        _seed(self.brain, 'S', 'attributed tap')
        self.assertEqual(_stop(self.brain, 'S').get('decision'), 'block')
        events = self.brain.query_traces(session_id='S').get('events', [])
        self_msgs = [e for e in events if e.get('ref_type') == 'self_message']
        self.assertTrue(
            self_msgs,
            "self_message delivery trace not attributed to recipient session 'S'")


class TestStopDeliveryChain(BrainTestBase):
    needs_embedder = False

    def test_delivery_trace_shares_the_turns_chain(self):
        """The delivery blocked THIS turn's stop, so its trace must land on
        this turn's s0 chain — not the next one. The hook holds the counter
        at N through delivery (post_response_common increment=False) and
        advances it after."""
        _seed(self.brain, 'S', 'chained tap')
        self.assertEqual(_stop(self.brain, 'S').get('decision'), 'block')
        events = self.brain.query_traces(session_id='S').get('events', [])
        chains = {e.get('ref_type'): e.get('chain_id') for e in events}
        self.assertIn('self_message', chains)
        # This turn's other trace (heartbeat: _stop without a prior recall
        # is classified non-conversational) must share the chain.
        self.assertEqual(chains['self_message'], chains.get('heartbeat'))


class TestCrossHookConsumeOnce(BrainTestBase):
    needs_embedder = False

    def test_pretool_leaves_tap_then_stop_delivers_once(self):
        # Stop-only: PreToolUse stays silent and does not consume, so the Stop
        # block delivers — exactly once (the next stop finds nothing, no loop).
        _seed(self.brain, 'S', 'single tap')
        pre = _edit(self.brain, 'S', 'a.py')
        self.assertNotIn('single tap', _pretool_ac(pre))     # PreToolUse silent
        first = _stop(self.brain, 'S')
        self.assertEqual(first.get('decision'), 'block')
        self.assertIn('single tap', first.get('reason', ''))
        self.assertNotEqual(_stop(self.brain, 'S').get('decision'), 'block')


if __name__ == '__main__':
    unittest.main()
