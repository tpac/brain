"""Dispatch-layer contract for the self-channel handlers.

The Phase 1/2a tests (test_self_presence, test_self_signal) exercise the inner
presence.*/signal.* functions, which return raw payloads and pass. But the
daemon table-dispatch (daemon_server._dispatch) sends a handler's return
VERBATIM to the MCP client, so each handler MUST wrap its payload in the
{"ok": True, "result": ...} envelope. A raw return reaches brain_mcp as a
falsy `ok` with no `error` and surfaces as "Unknown daemon error" — a
successful call misread as a failure, silently (no exception, nothing logged).

This is the coverage that was missing when the self-channel first ran live:
nothing had ever invoked _handle_self_* — only the inner functions.
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.dispatch_self import (
    _handle_self_presence, _handle_self_peek, _handle_self_send, _handle_self_inbox)


def _assert_envelope(case, result):
    case.assertIsInstance(result, dict)
    case.assertIs(result.get("ok"), True,
                  "handler must return the {'ok': True, ...} envelope, got: %r" % (result,))
    case.assertIn("result", result, "enveloped return must carry a 'result' payload")


class TestSelfDispatchEnvelope(BrainTestBase):
    needs_embedder = False

    def test_presence_handler_enveloped(self):
        r = _handle_self_presence(self.brain, {'session_id': 'A', 'limit': 3}, [])
        _assert_envelope(self, r)
        self.assertIn('streams', r['result'])
        self.assertIn('line', r['result'])

    def test_peek_handler_enveloped(self):
        r = _handle_self_peek(self.brain, {'stream_id': 'A'}, [])
        _assert_envelope(self, r)
        self.assertIn('found', r['result'])

    def test_send_handler_enveloped(self):
        r = _handle_self_send(self.brain, {'to': 'B', 'body': 'hi B', 'from_session': 'A'}, [])
        _assert_envelope(self, r)
        self.assertIn('id', r['result'])

    def test_inbox_handler_enveloped(self):
        r = _handle_self_inbox(self.brain, {'session_id': 'B'}, [])
        _assert_envelope(self, r)
        self.assertIn('messages', r['result'])

    def test_send_then_inbox_roundtrip_through_dispatch(self):
        """End-to-end through the enveloped dispatch layer (not the inner fns)."""
        sent = _handle_self_send(
            self.brain, {'to': 'B', 'body': 'ping B', 'from_session': 'A'}, [])
        _assert_envelope(self, sent)
        drained = _handle_self_inbox(self.brain, {'session_id': 'B'}, [])
        _assert_envelope(self, drained)
        bodies = [m['body'] for m in drained['result']['messages']]
        self.assertIn('ping B', bodies)


if __name__ == '__main__':
    unittest.main()
