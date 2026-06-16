"""Tests for BrainVoice — DECIDE + FORMAT layer for consciousness output.

Tests cover:
- Phase 1: fl(), trunc(), format_recall_results, format_encoding_warning, format_suggestions
- Phase 2: render_boot() matches format_boot_context() wrapper
- Phase 4: Operator channel (for_operator) in all render methods
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_voice import BrainVoice, EVOLUTION_TYPES, ENGINEERING_TYPES, CODE_COGNITION_TYPES


class TestBrainVoiceFormatters(unittest.TestCase):
    """Phase 1: Test formatting primitives moved from brain_surface.py and daemon_hooks.py."""

    def test_fl_empty_items_returns_empty(self):
        result = BrainVoice.fl([], "HEADER:")
        self.assertEqual(result, [])

    def test_fl_formats_list_with_header(self):
        items = [{"title": "Item A"}, {"title": "Item B"}]
        result = BrainVoice.fl(items, "SECTION:")
        self.assertIn("SECTION:", result)
        self.assertTrue(any("Item A" in line for line in result))
        self.assertTrue(any("Item B" in line for line in result))

    def test_fl_truncates_at_max_n(self):
        items = [{"title": "Item %d" % i} for i in range(10)]
        result = BrainVoice.fl(items, "HEADER:", max_n=3)
        self.assertIn("HEADER:", result)
        self.assertTrue(any("... and 7 more" in line for line in result))

    def test_fl_custom_formatter(self):
        items = [{"name": "foo"}, {"name": "bar"}]
        result = BrainVoice.fl(items, "HEADER:", fmt=lambda x: "custom: %s" % x["name"])
        self.assertTrue(any("custom: foo" in line for line in result))

    def test_fl_suffix_overrides_more_count(self):
        items = [{"title": "Item %d" % i} for i in range(10)]
        result = BrainVoice.fl(items, "HEADER:", max_n=3, suffix="See all 10.")
        self.assertTrue(any("See all 10." in line for line in result))
        self.assertFalse(any("... and" in line for line in result))

    def test_fl_list_format_result(self):
        """fmt returning a list extends the output directly."""
        items = [{"a": 1}]
        result = BrainVoice.fl(items, "H:", fmt=lambda x: ["  line1", "  line2"])
        self.assertIn("  line1", result)
        self.assertIn("  line2", result)

    def test_trunc_short_string(self):
        self.assertEqual(BrainVoice.trunc("hello", 80), "hello")

    def test_trunc_long_string(self):
        result = BrainVoice.trunc("a" * 100, 50)
        self.assertEqual(len(result), 53)  # 50 + "..."
        self.assertTrue(result.endswith("..."))

    def test_trunc_none(self):
        self.assertEqual(BrainVoice.trunc(None), "")

    def test_format_recall_results_includes_all_types(self):
        """Evolution separation removed — all nodes use unified format."""
        results = [
            {"type": "tension", "title": "T1", "content": "c1", "effective_activation": 0.5},
            {"type": "rule", "title": "R1", "content": "c2", "effective_activation": 0.8},
        ]
        lines = []
        BrainVoice.format_recall_results(results, lines)
        text = "\n".join(lines)
        self.assertIn("T1", text)
        self.assertIn("[rule]", text)

    def test_format_recall_results_empty(self):
        lines = []
        BrainVoice.format_recall_results([], lines)
        self.assertEqual(lines, [])

    def test_format_encoding_warning_none_health(self):
        result = BrainVoice.format_encoding_warning({"health": "NONE", "session_minutes": 10})
        self.assertIn("ENCODING ALERT", result)

    def test_format_encoding_warning_stale_high_edits(self):
        result = BrainVoice.format_encoding_warning({
            "health": "STALE",
            "edits_since_last_remember": 20,
            "minutes_since_last_remember": 15,
        })
        self.assertIn("ENCODING WARNING", result)
        self.assertIn("20 edits", result)

    def test_format_encoding_warning_ok(self):
        result = BrainVoice.format_encoding_warning({"health": "OK"})
        self.assertEqual(result, "")

    def test_format_suggestions_basic(self):
        suggestions = [
            {"type": "purpose", "title": "P1", "content": "desc", "locked": True, "id": "1"},
        ]
        result = BrainVoice.format_suggestions("foo.py", suggestions, [], [], [], "")
        self.assertIn("[BRAIN] AUTO-SUGGEST for foo.py:", result)
        self.assertIn("ENGINEERING MEMORY", result)
        self.assertIn("P1", result)
        self.assertIn("[/BRAIN]", result)

    def test_format_suggestions_with_impacts(self):
        impacts = [{"title": "Breaking change", "content": "This will break X"}]
        result = BrainVoice.format_suggestions("bar.py", [], [], [], impacts, "")
        self.assertIn("CHANGE IMPACT WARNING", result)
        self.assertIn("Breaking change", result)


class TestBrainVoiceRenderBoot(BrainTestBase):
    """Phase 2: render_boot() produces valid output and matches wrapper."""

    def test_render_boot_returns_dict(self):
        voice = BrainVoice(self.brain)
        result = voice.render_boot(user="Tom", project="test")
        self.assertIn('for_claude', result)
        self.assertIn('for_operator', result)

    def test_render_boot_contains_brain_tags(self):
        voice = BrainVoice(self.brain)
        result = voice.render_boot()
        text = result['for_claude']
        # Contract evolved (operator-directed): the SKILL.md stance is injected
        # FIRST, so boot opens with the stance and the [BRAIN] block follows.
        self.assertTrue(text.startswith(voice._load_stance()),
                        "boot must open with the SKILL.md stance, not [BRAIN]")
        self.assertIn("[BRAIN]", text)
        self.assertIn("[/BRAIN]", text)

    def test_wrapper_delegates_to_render_boot(self):
        """format_boot_context() produces structurally valid boot output."""
        wrapper = self.brain.format_boot_context(user="Tom", project="test", db_dir="/test")
        # Core structural elements
        self.assertIn("[BRAIN]", wrapper)
        self.assertIn("[/BRAIN]", wrapper)
        self.assertIn("Anchor", wrapper)
        # Identity line — new boot opens with "Anchor. The brain is yours"
        self.assertIn("The brain is yours", wrapper)
        # Stats line includes memory count
        self.assertIn("memories", wrapper)

    def test_render_boot_structure(self):
        """Boot v2 output has brain tags and identity line."""
        voice = BrainVoice(self.brain)
        result = voice.render_boot()
        text = result['for_claude']
        self.assertIn("[BRAIN]", text)
        self.assertIn("[/BRAIN]", text)
        # New boot opens with identity line instead of "Welcome back, Anchor"
        self.assertIn("Anchor. The brain is yours", text)
        # Stats embedded in the identity line
        self.assertIn("memories", text)

    def test_render_boot_operator_has_summary(self):
        """Phase 4: Operator channel includes boot summary with priority tags."""
        # ADJUSTED: _operator_boot_summary now uses @priority-tagged format
        # instead of emoji one-liners. This is intentional — operator channel
        # content is now structured for Claude relay. (2026-03-22)
        voice = BrainVoice(self.brain)
        result = voice.render_boot()
        self.assertIsNotNone(result['for_operator'])
        self.assertIn("@priority:", result['for_operator'])
        self.assertIn("nodes", result['for_operator'])

    def test_render_boot_v2_stance_first_outside_brain_tags(self):
        """The SKILL.md stance is injected FIRST, before/outside [BRAIN]."""
        voice = BrainVoice(self.brain)
        stance = voice._load_stance()
        self.assertTrue(stance, "stance should load from skills/brain/SKILL.md")

        result = voice.render_boot_v2(
            user="Tom", project="test", db_dir="/test",
            session_id="stance-order-check")
        text = result['for_claude']

        self.assertIn("[BRAIN]", text)
        # Boot opens with the stance, not the [BRAIN] envelope
        self.assertTrue(text.startswith(stance),
                        "boot must open with the identity stance")
        self.assertLess(text.find(stance), text.find("[BRAIN]"),
                        "stance must sit outside/before the [BRAIN] block")



class TestBrainVoiceOperatorChannel(BrainTestBase):
    """Phase 4: Operator channel (for_operator) behavior."""

    def test_format_for_operator_empty_when_nothing(self):
        result = BrainVoice.format_for_operator([])
        self.assertIsNone(result)

    def test_format_for_operator_with_items(self):
        result = BrainVoice.format_for_operator(["🧠 3 nodes recalled"])
        self.assertIsNotNone(result)
        self.assertIn("3 nodes recalled", result)


class TestBrainVoiceOperatorChannelV2(BrainTestBase):
    """Phase 5: Operator channel via Claude relay — wrap_for_hook + render_operator_prompt."""

    # ── wrap_for_hook ──

    def test_wrap_for_hook_claude_only(self):
        """No for_operator → output has [BRAIN], no [BRAIN-To-*]."""
        voice = BrainVoice(self.brain)
        result = voice.wrap_for_hook("[BRAIN]\ntest content\n[/BRAIN]")
        self.assertIn("[BRAIN]", result)
        self.assertNotIn("[BRAIN-To-", result)

    def test_wrap_for_hook_both_channels(self):
        """Operator channel killed (2026-03-28) — wrap_for_hook returns Claude content only.
        Signals handle operator alerts via signal queue now."""
        voice = BrainVoice(self.brain)
        result = voice.wrap_for_hook("[BRAIN]\nclaude stuff\n[/BRAIN]", "@priority: high\nHello Tom")
        self.assertIn("[BRAIN]", result)
        self.assertIn("claude stuff", result)
        # Operator content is ignored — no [BRAIN-To-*] tags
        self.assertNotIn("[BRAIN-To-", result)
        self.assertNotIn("Hello Tom", result)

    def test_wrap_for_hook_empty_operator(self):
        """Empty string for_operator → treated as no operator content."""
        voice = BrainVoice(self.brain)
        result = voice.wrap_for_hook("[BRAIN]\nstuff\n[/BRAIN]", "")
        self.assertNotIn("[BRAIN-To-", result)
        result2 = voice.wrap_for_hook("[BRAIN]\nstuff\n[/BRAIN]", "   ")
        self.assertNotIn("[BRAIN-To-", result2)

    def test_wrap_for_hook_host_name(self):
        """Operator channel killed (2026-03-28) — host_name no longer affects output."""
        self.brain.set_config("host_name", "Tom")
        voice = BrainVoice(self.brain)
        result = voice.wrap_for_hook("[BRAIN]\ntest\n[/BRAIN]", "hello")
        # Operator content ignored — returns Claude content only
        self.assertIn("[BRAIN]", result)
        self.assertNotIn("[BRAIN-To-Tom]", result)

    def test_wrap_for_hook_default_host(self):
        """Operator channel killed (2026-03-28) — no operator tags emitted."""
        voice = BrainVoice(self.brain)
        result = voice.wrap_for_hook("[BRAIN]\ntest\n[/BRAIN]", "hello")
        self.assertNotIn("[BRAIN-To-Operator]", result)
        self.assertIn("[BRAIN]", result)

    # render_operator_prompt: DELETED — migrated to signal queue + assembler (2026-03-27)

    def test_operator_boot_summary_stats(self):
        """Boot summary includes node/edge/locked counts with priority tags."""
        voice = BrainVoice(self.brain)
        result = voice._operator_boot_summary(
            node_count=100, edge_count=200, locked_count=50,
        )
        self.assertIsNotNone(result)
        self.assertIn("@priority:", result)
        self.assertIn("100", result)
        self.assertIn("200", result)
        self.assertIn("50", result)


class TestBrainVoiceConstants(unittest.TestCase):
    """Verify constants are properly shared."""

    def test_evolution_types(self):
        self.assertIn("tension", EVOLUTION_TYPES)
        self.assertIn("hypothesis", EVOLUTION_TYPES)
        self.assertIn("aspiration", EVOLUTION_TYPES)

    def test_engineering_types(self):
        self.assertIn("purpose", ENGINEERING_TYPES)
        self.assertIn("mechanism", ENGINEERING_TYPES)
        self.assertIn("lesson", ENGINEERING_TYPES)

    def test_code_cognition_types(self):
        self.assertIn("fn_reasoning", CODE_COGNITION_TYPES)
        self.assertIn("bug_lesson", CODE_COGNITION_TYPES)


if __name__ == '__main__':
    unittest.main()
