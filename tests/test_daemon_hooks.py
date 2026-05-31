"""Tests for daemon_hooks.py — hook logic layer.

Tests cover:
- hook_recall() output format (surface-formatted additionalContext)
- Early return behavior (no results = approve)
- Surface integration (mock — no API key in test env)
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.daemon_hooks import hook_recall, post_response_common


# Realistic output matching format_surface_output_activation() in surface_contract.py
_MOCK_SURFACE_OUTPUT = (
    'Brain recalled 1 memories:\n\n'
    '[rule] "Test rule for recall" (id:abcd1234, conf:1.0)\n'
    'Content: Important test content\n'
)


def _mock_run_surface(brain, ctx, candidates_data, user_message, **kwargs):
    """Mock surface that returns formatted output for any non-empty candidates."""
    if not candidates_data:
        return None
    return _MOCK_SURFACE_OUTPUT


class TestHookRecallOutput(BrainTestBase):
    """Verify hook_recall() output format."""

    def _call_recall(self, message="test query"):
        """Helper to call hook_recall with standard args."""
        args = {"prompt": message, "message": message}
        return hook_recall(self.brain, args, [])

    def _seed_data(self):
        """Add test data so recall has results (avoids early-return approve)."""
        self.brain.remember(type="rule", title="Test rule for recall", content="Important test content")
        self.brain.remember(type="lesson", title="Test lesson", content="We learned something")

    def test_hook_recall_early_return_when_empty(self):
        """No results/signals -> returns approve (no-op)."""
        result = self._call_recall("xyzzy gibberish")
        self.assertEqual(result["json"], {"decision": "approve"})

    @patch('servers.daemon_hooks._run_surface', side_effect=_mock_run_surface)
    def test_hook_recall_returns_additional_context(self, mock_surface):
        """When results exist and surface selects, returns {'json': {'additionalContext': str}}."""
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertIn("json", result)
        self.assertIn("additionalContext", result["json"])

    def test_hook_recall_no_system_message(self):
        """systemMessage key is never present in output (dead channel removed)."""
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertNotIn("systemMessage", result.get("json", {}))

    @patch('servers.daemon_hooks._run_surface', side_effect=_mock_run_surface)
    def test_hook_recall_has_brain_recalled_header(self, mock_surface):
        """additionalContext contains 'Brain recalled' header from surface output."""
        self._seed_data()
        result = self._call_recall("test rule")
        ctx = result["json"]["additionalContext"]
        self.assertIn("Brain recalled", ctx)
        self.assertIn("memories:", ctx)

    @patch('servers.daemon_hooks._run_surface', side_effect=_mock_run_surface)
    def test_hook_recall_contains_node_content(self, mock_surface):
        """additionalContext includes node type, title, and content from surface formatting."""
        self._seed_data()
        result = self._call_recall("test rule")
        ctx = result["json"]["additionalContext"]
        self.assertIn("[rule]", ctx)
        self.assertIn("Test rule for recall", ctx)

    @patch('servers.daemon_hooks._run_surface', side_effect=_mock_run_surface)
    def test_hook_recall_judge_failure_returns_approve(self, mock_judge):
        """When judge raises an exception, hook_recall returns approve."""
        mock_judge.side_effect = RuntimeError("API key missing")
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertEqual(result["json"].get("decision"), "approve")

    @patch('servers.daemon_hooks._run_surface', return_value=None)
    def test_hook_recall_judge_returns_none_means_approve(self, mock_judge):
        """When judge returns None (no selection), hook_recall returns approve."""
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertEqual(result["json"].get("decision"), "approve")


class TestTurnClassification(BrainTestBase):
    """post_response_common classifies each stop: conversational (a real prompt
    ran recall this stop → last_recall_stop == stop_counter) vs heartbeat (a
    /watch wakeup re-arm, recall skipped client-side). Heartbeats must NOT tick
    stop_counter (which gates the Scribe) or enter the conversation stream.
    See trace_contract S0 TURN CLASSIFICATION."""

    needs_embedder = False

    def _s0_refs(self, session_id):
        rows = self.brain.logs_conn.execute(
            "SELECT ref_type FROM trace_events WHERE scale='s0' AND session_id=? "
            "ORDER BY created_at", (session_id,)).fetchall()
        return [r[0] for r in rows]

    def test_conversational_turn_increments_and_writes_user_message(self):
        sid = 'test-turn-conv'
        ctx = self.brain.get_or_create_session(sid)
        before = ctx.stop_counter
        ctx.last_recall_stop = ctx.stop_counter   # simulate hook_recall having run this stop
        post_response_common(self.brain, sid, "a real operator prompt", "a response")
        self.assertEqual(ctx.stop_counter, before + 1)
        self.assertTrue(ctx.last_turn_conversational)
        refs = self._s0_refs(sid)
        self.assertIn('user_message', refs)
        self.assertIn('assistant_message', refs)
        self.assertNotIn('heartbeat', refs)

    def test_heartbeat_turn_neither_increments_nor_writes_user_message(self):
        sid = 'test-turn-heartbeat'
        ctx = self.brain.get_or_create_session(sid)
        before = ctx.stop_counter                 # last_recall_stop stays -1 != stop_counter
        post_response_common(self.brain, sid, "/watch skill body", "(watching — inbox empty)")
        self.assertEqual(ctx.stop_counter, before)   # NOT incremented → Scribe never dragged along
        self.assertFalse(ctx.last_turn_conversational)
        refs = self._s0_refs(sid)
        self.assertIn('heartbeat', refs)
        self.assertNotIn('user_message', refs)
        self.assertNotIn('assistant_message', refs)

    def test_heartbeats_between_real_turns_dont_inflate_counter(self):
        sid = 'test-turn-mixed'
        ctx = self.brain.get_or_create_session(sid)
        ctx.last_recall_stop = ctx.stop_counter
        post_response_common(self.brain, sid, "real one", "r")
        after_real = ctx.stop_counter
        post_response_common(self.brain, sid, "/watch", "(watching)")   # heartbeat: no recall mark
        self.assertEqual(ctx.stop_counter, after_real)                  # heartbeat doesn't advance
        ctx.last_recall_stop = ctx.stop_counter                         # next real turn
        post_response_common(self.brain, sid, "real two", "r")
        self.assertEqual(ctx.stop_counter, after_real + 1)              # advances by exactly one


if __name__ == '__main__':
    unittest.main()
