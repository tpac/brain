"""Tests for daemon_hooks.py — hook logic layer.

Tests cover:
- hook_recall() output format (merged channels, no systemMessage)
- Early return behavior (no results = approve)
- Reminder surfacing through operator channel
- Debug mode routing through operator channel
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.daemon_hooks import hook_recall


class TestHookRecallOutput(BrainTestBase):
    """Verify hook_recall() output uses merged channel format."""

    def _call_recall(self, message="test query"):
        """Helper to call hook_recall with standard args."""
        args = {"prompt": message, "message": message}
        return hook_recall(self.brain, args, [])

    def _seed_data(self):
        """Add test data so recall has results (avoids early-return approve)."""
        self.brain.remember(type="rule", title="Test rule for recall", content="Important test content")
        self.brain.remember(type="lesson", title="Test lesson", content="We learned something")

    def test_hook_recall_early_return_when_empty(self):
        """No results/signals → returns approve (no-op)."""
        result = self._call_recall("xyzzy gibberish")
        self.assertEqual(result["json"], {"decision": "approve"})

    def test_hook_recall_returns_additional_context(self):
        """When results exist, returns {'json': {'additionalContext': str}}."""
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertIn("json", result)
        self.assertIn("additionalContext", result["json"])

    def test_hook_recall_no_system_message(self):
        """systemMessage key is never present in output (dead channel removed)."""
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertNotIn("systemMessage", result.get("json", {}))

    def test_hook_recall_has_brain_tags(self):
        """additionalContext contains [BRAIN]...[/BRAIN] wrapping."""
        self._seed_data()
        result = self._call_recall("test rule")
        ctx = result["json"]["additionalContext"]
        self.assertIn("[BRAIN]", ctx)
        self.assertIn("[/BRAIN]", ctx)

    def test_hook_recall_with_due_reminder(self):
        """Create reminder with past due_date, verify it appears in operator channel as high priority.

        Reminders trigger urgent_signals, which prevents early return even without recall results.
        """
        self.brain.create_reminder("Ship the feature", "2020-01-01T00:00:00")
        result = self._call_recall("what should I do")
        ctx = result["json"]["additionalContext"]
        self.assertIn("[BRAIN-To-", ctx)
        self.assertIn("Ship the feature", ctx)
        self.assertIn("@priority: high", ctx)

    def test_hook_recall_debug_mode_output(self):
        """Enable debug, verify full injection appears in operator channel."""
        self.brain.set_config("debug_enabled", "1")
        self._seed_data()
        result = self._call_recall("test rule")
        ctx = result["json"]["additionalContext"]
        self.assertIn("[BRAIN-To-", ctx)
        self.assertIn("[DEBUG]", ctx)
        self.assertIn("Brain→Claude injection", ctx)

    def test_hook_recall_debug_mode_disabled(self):
        """Disable debug, verify no debug section in output."""
        self.brain.set_config("debug_enabled", "0")
        self._seed_data()
        result = self._call_recall("test rule")
        ctx = result["json"]["additionalContext"]
        self.assertNotIn("[DEBUG]", ctx)


if __name__ == '__main__':
    unittest.main()
