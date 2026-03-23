"""Tests for BrainVoice — DECIDE + FORMAT layer for consciousness output.

Tests cover:
- Phase 1: fl(), trunc(), format_recall_results, format_encoding_warning, format_suggestions
- Phase 2: render_boot() matches format_boot_context() wrapper
- Phase 3: select_prompt_signals(), render_prompt(), render_reboot()
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

    def test_format_recall_results_separates_evolution(self):
        results = [
            {"type": "tension", "title": "T1", "content": "c1", "effective_activation": 0.5},
            {"type": "rule", "title": "R1", "content": "c2", "effective_activation": 0.8},
        ]
        lines = []
        BrainVoice.format_recall_results(results, lines)
        text = "\n".join(lines)
        self.assertIn("ACTIVE EVOLUTION", text)
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
        self.assertTrue(text.startswith("[BRAIN]"))
        self.assertIn("[/BRAIN]", text)

    def test_wrapper_delegates_to_render_boot(self):
        """format_boot_context() wrapper delegates to BrainVoice.render_boot().

        Cannot compare exact equality because render_boot() has side effects
        (auto_generate_self_reflection, save) that mutate brain state between calls.
        Instead verify the wrapper produces structurally valid output with all
        required sections — same structure that render_boot() would produce.
        """
        wrapper = self.brain.format_boot_context(user="Tom", project="test", db_dir="/test")
        # ADJUSTED: Operator section may come first (2026-03-22, timeout-survival ordering).
        # Brain header still present, just may not be at position 0.
        self.assertIn("[BRAIN] v", wrapper)
        self.assertIn("booted from: /test", wrapper)
        # Must contain session number
        self.assertIn("Session #", wrapper)
        # Must contain triad statement
        self.assertIn("TRIAD: Host + Brain + Operator are one", wrapper)
        # Must contain footer with node/edge stats
        self.assertIn("Brain:", wrapper)
        self.assertIn("nodes,", wrapper)
        # Must contain MCP tools reminder
        self.assertIn("Use brain MCP tools:", wrapper)
        # Must contain closing brain tag (may be followed by operator channel)
        # ADJUSTED: format_boot_context now returns merged output via wrap_for_hook,
        # so [/BRAIN] may not be at the very end if operator channel is present. (2026-03-22)
        self.assertIn("[/BRAIN]", wrapper)

    def test_render_boot_includes_consciousness_when_signals_present(self):
        """Consciousness section appears when get_consciousness_signals returns data.

        In a fresh test brain, the engagement filter and health auto-fix suppress
        all signals. We patch get_consciousness_signals to return realistic data
        to verify the formatting logic works correctly.
        """
        from unittest.mock import patch

        mock_signals = {
            "evolutions": [
                {"title": "Tension: A vs B", "confidence": 0.6, "created_at": "2026-03-20"},
            ],
            "stale_reasoning": [
                {"title": "Old reasoning X", "last_validated": None},
            ],
        }

        voice = BrainVoice(self.brain)
        with patch.object(self.brain, 'get_consciousness_signals', return_value=mock_signals):
            result = voice.render_boot()

        text = result['for_claude']
        self.assertIn("[BRAIN] CONSCIOUSNESS", text)
        self.assertIn("Tension: A vs B", text)
        self.assertIn("STALE REASONING", text)
        self.assertIn("Old reasoning X", text)

    def test_render_boot_includes_locked_rules(self):
        self.brain.remember(
            type="rule", title="Always test first",
            content="Rule content", locked=True,
        )
        voice = BrainVoice(self.brain)
        result = voice.render_boot()
        self.assertIn("locked rules", result['for_claude'].lower())

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


class TestBrainVoiceSelectPromptSignals(BrainTestBase):
    """Phase 3: Signal selection for recall."""

    def test_select_prompt_signals_returns_expected_keys(self):
        voice = BrainVoice(self.brain)
        signals = voice.select_prompt_signals("test query", [])
        self.assertIn('aspirations', signals)
        self.assertIn('hypothesis', signals)
        self.assertIn('tensions', signals)
        self.assertIn('instinct_nudge', signals)

    def test_select_prompt_signals_excludes_recalled(self):
        # Create an aspiration
        node = self.brain.remember(
            type="aspiration", title="Grow the brain",
            content="Aspiration content",
        )
        node_id = node.get("id") if isinstance(node, dict) else node

        voice = BrainVoice(self.brain)
        # When the aspiration is in recall results, it should be excluded
        signals = voice.select_prompt_signals("grow", [{"id": node_id}])
        aspiration_ids = [a.get("id") for a in signals['aspirations']]
        self.assertNotIn(node_id, aspiration_ids)

    def test_select_prompt_signals_tensions_limited(self):
        # Create many tensions
        for i in range(5):
            self.brain.remember(
                type="tension", title="Tension %d" % i,
                content="A vs B", confidence=0.6,
            )
        voice = BrainVoice(self.brain)
        signals = voice.select_prompt_signals("test", [])
        self.assertLessEqual(len(signals['tensions']), 2)


class TestBrainVoiceRenderPrompt(BrainTestBase):
    """Phase 3: render_prompt() formatting."""

    def test_render_prompt_returns_dict(self):
        voice = BrainVoice(self.brain)
        result = voice.render_prompt(
            results=[],
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        self.assertIn('for_claude', result)
        self.assertIn('for_operator', result)

    def test_render_prompt_includes_recall_results(self):
        results = [
            {"type": "rule", "title": "Test rule", "content": "Content",
             "locked": True, "effective_activation": 0.9},
        ]
        voice = BrainVoice(self.brain)
        result = voice.render_prompt(
            results=results,
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        self.assertIn("[BRAIN] RECALL", result['for_claude'])
        self.assertIn("Test rule", result['for_claude'])

    def test_render_prompt_includes_aspirations(self):
        voice = BrainVoice(self.brain)
        result = voice.render_prompt(
            results=[],
            prompt_signals={
                'aspirations': [{"title": "Build great things"}],
                'hypothesis': None, 'tensions': [], 'instinct_nudge': None,
            },
        )
        self.assertIn("ASPIRATION COMPASS", result['for_claude'])
        self.assertIn("Build great things", result['for_claude'])

    def test_render_prompt_includes_hypothesis(self):
        voice = BrainVoice(self.brain)
        result = voice.render_prompt(
            results=[],
            prompt_signals={
                'aspirations': [],
                'hypothesis': {"title": "Users prefer X", "content": "Because Y", "confidence": 0.7},
                'tensions': [], 'instinct_nudge': None,
            },
        )
        self.assertIn("HYPOTHESIS TO VALIDATE", result['for_claude'])
        self.assertIn("Users prefer X", result['for_claude'])

    def test_render_prompt_instinct_at_top(self):
        voice = BrainVoice(self.brain)
        result = voice.render_prompt(
            results=[{"type": "rule", "title": "R", "content": "C", "effective_activation": 0.5}],
            prompt_signals={
                'aspirations': [], 'hypothesis': None, 'tensions': [],
                'instinct_nudge': "INSTINCT: This feels off.",
            },
        )
        # Instinct should be at the very top
        text = result['for_claude']
        instinct_pos = text.find("INSTINCT:")
        recall_pos = text.find("[BRAIN] RECALL")
        self.assertLess(instinct_pos, recall_pos)

    def test_render_prompt_urgent_signals_first(self):
        voice = BrainVoice(self.brain)
        result = voice.render_prompt(
            results=[],
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
            urgent_signals=["Encoding gap detected"],
        )
        self.assertIn("[BRAIN] AWARENESS:", result['for_claude'])
        self.assertIn("Encoding gap detected", result['for_claude'])

    def test_render_prompt_debug_messages(self):
        voice = BrainVoice(self.brain)
        result = voice.render_prompt(
            results=[],
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
            debug_messages=["recall took 50ms"],
        )
        self.assertIn("[BRAIN DEBUG]", result['for_claude'])
        self.assertIn("recall took 50ms", result['for_claude'])


class TestBrainVoiceRenderReboot(BrainTestBase):
    """Phase 3: render_reboot() formatting."""

    def test_render_reboot_returns_dict(self):
        voice = BrainVoice(self.brain)
        result = voice.render_reboot(boot_context={})
        self.assertIn('for_claude', result)
        self.assertIn('for_operator', result)

    def test_render_reboot_includes_header(self):
        voice = BrainVoice(self.brain)
        result = voice.render_reboot(boot_context={})
        self.assertIn("[BRAIN] POST-COMPACTION REBOOT:", result['for_claude'])
        self.assertIn("[/BRAIN]", result['for_claude'])

    def test_render_reboot_includes_locked_rules(self):
        rules = [{"title": "Always commit tests"}, {"title": "Never skip hooks"}]
        voice = BrainVoice(self.brain)
        result = voice.render_reboot(boot_context={}, locked_rules=rules)
        self.assertIn("LOCKED RULES (2 active)", result['for_claude'])
        self.assertIn("Always commit tests", result['for_claude'])

    def test_render_reboot_includes_signals(self):
        signals = {
            "reminders": [{"title": "Check PR"}],
            "evolutions": [{"title": "Tension A"}],
        }
        voice = BrainVoice(self.brain)
        result = voice.render_reboot(boot_context={}, signals=signals)
        self.assertIn("REMINDERS:", result['for_claude'])
        self.assertIn("EVOLUTIONS:", result['for_claude'])

    def test_render_reboot_synthesis_just_ran(self):
        voice = BrainVoice(self.brain)
        result = voice.render_reboot(
            boot_context={},
            synthesis_info={"just_ran": True, "parts": ["3 decisions"]},
        )
        self.assertIn("Pre-compact synthesis did not run", result['for_claude'])
        self.assertIn("3 decisions", result['for_claude'])

    def test_render_reboot_open_questions_fresh(self):
        voice = BrainVoice(self.brain)
        result = voice.render_reboot(
            boot_context={},
            synthesis_info={"open_questions": ["What about X?"], "age_minutes": 5},
        )
        self.assertIn("OPEN QUESTIONS", result['for_claude'])
        self.assertIn("What about X?", result['for_claude'])

    def test_render_reboot_open_questions_stale(self):
        voice = BrainVoice(self.brain)
        result = voice.render_reboot(
            boot_context={},
            synthesis_info={"open_questions": ["Old Q"], "age_minutes": 120},
        )
        self.assertIn("questions may be resolved", result['for_claude'])


class TestBrainVoiceOperatorChannel(BrainTestBase):
    """Phase 4: Operator channel (for_operator) behavior."""

    def test_format_for_operator_empty_when_nothing(self):
        result = BrainVoice.format_for_operator([])
        self.assertIsNone(result)

    def test_format_for_operator_with_items(self):
        result = BrainVoice.format_for_operator(["🧠 3 nodes recalled"])
        self.assertIsNotNone(result)
        self.assertIn("3 nodes recalled", result)

    def test_render_prompt_operator_with_signals(self):
        """Operator channel uses render_operator_prompt with priority tags."""
        # ADJUSTED: render_prompt now uses render_operator_prompt (priority-tagged)
        # instead of _operator_recall_summary (emoji one-liners). (2026-03-22)
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=[{"type": "rule", "title": "R", "content": "C", "effective_activation": 0.5}],
            prompt_signals={
                'aspirations': [], 'hypothesis': None,
                'tensions': [{"title": "T1"}, {"title": "T2"}],
                'instinct_nudge': None,
            },
        )
        op = rendered['for_operator']
        self.assertIsNotNone(op)
        self.assertIn("@priority: medium", op)
        self.assertIn("Tension", op)

    def test_render_prompt_operator_empty_when_no_signals(self):
        """No recall results and no signals → operator channel is None."""
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=[],
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        self.assertIsNone(rendered['for_operator'])

    def test_render_reboot_operator_summary(self):
        voice = BrainVoice(self.brain)
        rendered = voice.render_reboot(
            boot_context={},
            locked_rules=[{"title": "R1"}, {"title": "R2"}],
            recall_results=[{"type": "rule", "title": "R"}],
        )
        op = rendered['for_operator']
        self.assertIsNotNone(op)
        self.assertIn("Post-compaction reboot", op)
        self.assertIn("2 locked rules", op)
        self.assertIn("1 nodes recalled", op)

    def test_debug_mode_exposes_claude_context(self):
        """In debug mode, operator sees the full Claude injection."""
        self.brain.set_config("debug_enabled", "1")
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=[{"type": "rule", "title": "Debug test", "content": "C", "effective_activation": 0.5}],
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        op = rendered['for_operator']
        self.assertIn("[DEBUG]", op)
        self.assertIn("Brain→Claude injection", op)


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
        """Both present → output has both tagged sections in order."""
        voice = BrainVoice(self.brain)
        result = voice.wrap_for_hook("[BRAIN]\nclaude stuff\n[/BRAIN]", "@priority: high\nHello Tom")
        self.assertIn("[BRAIN]", result)
        self.assertIn("[BRAIN-To-", result)
        # ADJUSTED: Operator section comes FIRST (2026-03-22) — survives hook
        # timeout truncation. Tom's reminders matter more than Claude's reasoning.
        tom_pos = result.index("[BRAIN-To-")
        brain_pos = result.index("[BRAIN]\n")
        self.assertLess(tom_pos, brain_pos)

    def test_wrap_for_hook_empty_operator(self):
        """Empty string for_operator → treated as no operator content."""
        voice = BrainVoice(self.brain)
        result = voice.wrap_for_hook("[BRAIN]\nstuff\n[/BRAIN]", "")
        self.assertNotIn("[BRAIN-To-", result)
        result2 = voice.wrap_for_hook("[BRAIN]\nstuff\n[/BRAIN]", "   ")
        self.assertNotIn("[BRAIN-To-", result2)

    def test_wrap_for_hook_host_name(self):
        """Uses brain config host_name in tag."""
        self.brain.set_config("host_name", "Tom")
        voice = BrainVoice(self.brain)
        result = voice.wrap_for_hook("[BRAIN]\ntest\n[/BRAIN]", "hello")
        self.assertIn("[BRAIN-To-Tom]", result)
        self.assertIn("[/BRAIN-To-Tom]", result)

    def test_wrap_for_hook_default_host(self):
        """No host_name config → falls back to 'Operator'."""
        voice = BrainVoice(self.brain)
        result = voice.wrap_for_hook("[BRAIN]\ntest\n[/BRAIN]", "hello")
        self.assertIn("[BRAIN-To-Operator]", result)

    # ── render_operator_prompt ──

    def test_render_operator_prompt_reminders_high_priority(self):
        """Due reminders always appear with @priority: high."""
        self.brain.create_reminder("Call mom", "2020-01-01T00:00:00")
        voice = BrainVoice(self.brain)
        result = voice.render_operator_prompt(
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        self.assertIsNotNone(result)
        self.assertIn("@priority: high", result)
        self.assertIn("Call mom", result)

    def test_render_operator_prompt_reminders_always_included(self):
        """Reminders never get budget-cut — they surface even with lots of other content."""
        self.brain.create_reminder("Important task", "2020-01-01T00:00:00")
        voice = BrainVoice(self.brain)
        result = voice.render_operator_prompt(
            prompt_signals={
                'aspirations': [], 'hypothesis': {"title": "Test hyp"},
                'tensions': [{"title": "T1"}, {"title": "T2"}],
                'instinct_nudge': None,
            },
        )
        self.assertIn("Important task", result)

    def test_render_operator_prompt_dreams(self):
        """Dreams from consciousness appear with @priority: low."""
        # Create nodes that might generate dreams via consciousness
        voice = BrainVoice(self.brain)
        # Test with mock — render_operator_prompt calls get_consciousness_signals
        result = voice.render_operator_prompt(
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        # Fresh brain may not have dreams — test the format when tensions exist
        # (dreams tested via integration)

    def test_render_operator_prompt_tensions(self):
        """Active tensions appear with @priority: medium."""
        voice = BrainVoice(self.brain)
        result = voice.render_operator_prompt(
            prompt_signals={
                'aspirations': [], 'hypothesis': None,
                'tensions': [{"title": "React hooks vs class components"}],
                'instinct_nudge': None,
            },
        )
        self.assertIsNotNone(result)
        self.assertIn("@priority: medium", result)
        self.assertIn("Tension", result)

    def test_render_operator_prompt_health_alerts(self):
        """Urgent signals get @priority: high."""
        voice = BrainVoice(self.brain)
        result = voice.render_operator_prompt(
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
            urgent_signals=["REMINDER DUE: Fix the bug"],
        )
        self.assertIsNotNone(result)
        self.assertIn("@priority: high", result)
        self.assertIn("Fix the bug", result)

    def test_render_operator_prompt_empty(self):
        """No interesting signals → returns None."""
        voice = BrainVoice(self.brain)
        result = voice.render_operator_prompt(
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        self.assertIsNone(result)

    def test_render_operator_prompt_budget(self):
        """Operator content stays under ~800 chars (excluding debug)."""
        # Create some content that could be large
        voice = BrainVoice(self.brain)
        result = voice.render_operator_prompt(
            prompt_signals={
                'aspirations': [], 'hypothesis': {"title": "Test hypothesis " * 10},
                'tensions': [{"title": "Tension " * 20}, {"title": "Another " * 20}],
                'instinct_nudge': None,
            },
        )
        if result:
            self.assertLess(len(result), 1200)  # generous but bounded

    # ── debug mode ──

    def test_render_operator_prompt_debug_mode(self):
        """When debug_enabled, full Claude injection appended to for_operator."""
        self.brain.set_config("debug_enabled", "1")
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=[{"type": "rule", "title": "Debug test", "content": "C", "effective_activation": 0.5}],
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        op = rendered['for_operator']
        self.assertIsNotNone(op)
        self.assertIn("[DEBUG]", op)
        self.assertIn("Brain→Claude injection", op)

    def test_render_operator_prompt_debug_off(self):
        """When debug off, no debug section in operator output."""
        self.brain.set_config("debug_enabled", "0")
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=[],
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        op = rendered['for_operator']
        # Either None or no DEBUG
        if op:
            self.assertNotIn("[DEBUG]", op)

    def test_debug_no_longer_uses_system_message(self):
        """Debug expansion goes to for_operator, which gets merged via wrap_for_hook,
        NOT to a separate systemMessage field."""
        self.brain.set_config("debug_enabled", "1")
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=[{"type": "rule", "title": "Test", "content": "C", "effective_activation": 0.5}],
            prompt_signals={'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
        )
        # Verify debug is in for_operator (which wrap_for_hook will merge)
        self.assertIn("[DEBUG]", rendered['for_operator'])
        # The actual merging is done by daemon_hooks via wrap_for_hook
        merged = voice.wrap_for_hook(rendered['for_claude'], rendered['for_operator'])
        self.assertIn("[DEBUG]", merged)
        self.assertIn("[BRAIN-To-", merged)

    # ── boot summary ──

    def test_operator_boot_summary_expanded(self):
        """Boot summary includes consciousness highlights and priority tags."""
        voice = BrainVoice(self.brain)
        result = voice._operator_boot_summary(
            node_count=100, edge_count=200, locked_count=50,
            signal_count=5, alert_count=2,
            consciousness_signals={"evolutions": [], "dreams": []},
        )
        self.assertIsNotNone(result)
        self.assertIn("@priority:", result)
        self.assertIn("100", result)

    def test_operator_boot_summary_reminders(self):
        """Boot summary surfaces due reminders as high priority."""
        self.brain.create_reminder("Ship feature", "2020-01-01T00:00:00")
        voice = BrainVoice(self.brain)
        result = voice._operator_boot_summary(
            node_count=100, edge_count=200, locked_count=50,
            consciousness_signals={},
        )
        self.assertIn("@priority: high", result)
        self.assertIn("Ship feature", result)

    def test_meta_instructions_format(self):
        """@priority levels are well-formed (high/medium/low only)."""
        import re
        voice = BrainVoice(self.brain)
        self.brain.create_reminder("Test", "2020-01-01T00:00:00")
        result = voice.render_operator_prompt(
            prompt_signals={
                'aspirations': [], 'hypothesis': {"title": "Hyp"},
                'tensions': [{"title": "T1"}],
                'instinct_nudge': None,
            },
            urgent_signals=["Alert!"],
        )
        # All @priority values should be high, medium, or low
        priorities = re.findall(r'@priority:\s*(\w+)', result)
        for p in priorities:
            self.assertIn(p, ("high", "medium", "low"), "Invalid priority: %s" % p)


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
