"""Tests for the Frame Constructor — Anchor's structured awareness object.

Phase 2 of the Frame architecture. Tests cover:
- build_frame returns a markdown string with all five sections
- Frame routes by brain.aspects (post Step 11 of unified-aspects)
- ctx.get_frame(brain) is the session-scoped entry point
- build_surface_prompt accepts and renders the frame param
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.scales.s1.frame import build_frame


class TestFrameConstruction(BrainTestBase):
    """Frame builds the expected five sections."""
    needs_embedder = False

    def test_build_frame_returns_markdown_with_all_sections(self):
        # Add a few nodes so sections have content
        self.brain.remember(
            type='principle', title='Test principle for operator',
            content='Some principle content', locked=True)
        self.brain.remember(
            type='moment', title='A moment',
            content='Moment content', locked=True)
        self.brain.remember(
            type='open', title='Open question',
            content='Still investigating')

        frame = build_frame(self.brain, 'test_session')

        self.assertIsInstance(frame, str)
        self.assertIn('## Operator', frame)
        self.assertIn('## Partnership', frame)
        self.assertIn('## Active threads', frame)
        self.assertIn('## Current focus', frame)
        self.assertIn('## Recent moves', frame)

    def test_build_frame_handles_empty_brain(self):
        # No nodes — should still render all sections without crashing
        frame = build_frame(self.brain, 'test_session')

        self.assertIn('## Operator', frame)
        self.assertIn('## Partnership', frame)
        self.assertIn('## Active threads', frame)
        # Empty sections render gracefully, not as errors
        self.assertNotIn('Traceback', frame)
        self.assertNotIn('Error', frame)


class TestAspectRoutingForFrame(BrainTestBase):
    """Frame routes via brain.aspects — replaces the old s2_node_families lookup.

    Step 11 of unified-aspects swapped frame.py from a families-dict shape
    + hardcoded _FALLBACK_FAMILIES + _resolve_families helper to direct
    AspectRegistry attribute access. These tests verify the data Frame
    needs is reachable through the new path.
    """
    needs_embedder = False

    def test_aspects_identity_bearing_includes_principle(self):
        # Frame's Operator section depends on this routing
        members = self.brain.aspects.identity_bearing.node_types
        self.assertIn('principle', members)
        self.assertIn('rule', members)

    def test_aspects_episodic_anchor_includes_moment(self):
        # Partnership.permanent layer depends on this
        members = self.brain.aspects.episodic_anchor.node_types
        self.assertIn('moment', members)

    def test_aspects_active_thread_includes_open(self):
        # Active threads section depends on this
        members = self.brain.aspects.active_thread.node_types
        self.assertIn('open', members)
        self.assertIn('tension', members)
        self.assertIn('hypothesis', members)

    def test_aspects_warm_union_dedups(self):
        # Partnership.warm layer unions episodic_anchor + lesson_insight
        members = self.brain.aspects.types_in(
            ['episodic_anchor', 'lesson_insight'])
        self.assertIn('moment', members)
        self.assertIn('insight', members)
        # No duplicates
        self.assertEqual(len(members), len(set(members)))


class TestSessionContextGetFrame(BrainTestBase):
    """ctx.get_frame(brain) is the public session-scoped entry point."""
    needs_embedder = False

    def test_ctx_get_frame_returns_same_as_build_frame(self):
        ctx = self.brain.get_or_create_session('test_session_ctx')
        from_ctx = ctx.get_frame(self.brain)
        from_direct = build_frame(self.brain, 'test_session_ctx')
        # Both should produce identical output
        self.assertEqual(from_ctx, from_direct)

    def test_ctx_get_frame_includes_session_id(self):
        # Frame should reference the session via current_focus / recent_moves
        ctx = self.brain.get_or_create_session('test_session_id_check')
        frame = ctx.get_frame(self.brain)
        # Recent moves section should render (empty journal → "(fresh session)")
        self.assertIn('## Recent moves', frame)


# TestNodeFamiliesSeed: REMOVED 2026-05-04 (Step 12 of unified-aspects).
# Tested that s2_node_families interaction was seeded on fresh brain.
# Replaced by aspect-nodes seeded via AspectRegistry auto-heal — coverage
# moved to test_aspect_registry_wired.py (TestAspectRegistryWired suite).


class TestSessionContextPerSession(BrainTestBase):
    """session_context must be per-session — no parallel-session leak."""
    needs_embedder = False

    def test_session_context_for_returns_per_session(self):
        # Two distinct sessions write different contexts via direct config
        self.brain.set_config('session_context_session_a', 'arc for session A')
        self.brain.set_config('session_context_session_b', 'arc for session B')

        self.assertEqual(
            self.brain.session_context_for('session_a'),
            'arc for session A')
        self.assertEqual(
            self.brain.session_context_for('session_b'),
            'arc for session B')

    def test_session_context_for_unknown_returns_empty(self):
        self.assertEqual(
            self.brain.session_context_for('never_written'), '')

    def test_session_context_for_no_session_id_returns_empty(self):
        # Defensive: don't read a global key when session_id is missing
        self.assertEqual(self.brain.session_context_for(''), '')

    def test_global_session_context_property_removed(self):
        # The leaky global property must not exist — replaced by session_context_for
        self.assertFalse(hasattr(self.brain, 'session_context'),
                        "brain.session_context property should be removed (leaked across parallel sessions)")


class TestSurfacePromptAcceptsFrame(BrainTestBase):
    """build_surface_prompt accepts frame and renders it as 'Partnership context:'."""
    needs_embedder = False

    def test_frame_renders_as_partnership_context(self):
        from servers.scales.s1.surface_contract import build_surface_prompt

        candidates = [{'id': 'abcd1234', 'title': 'Test',
                       'type': 'fact', 'content': 'x',
                       'confidence': 0.5, 'score': 0.8}]
        prompt, _ = build_surface_prompt(
            candidates, "test query",
            frame="THE FRAME PRIOR CONTENT")
        self.assertIn('THE FRAME PRIOR CONTENT', prompt)
        self.assertIn('Partnership context', prompt)

    def test_empty_frame_renders_explicit_degraded_marker(self):
        from servers.scales.s1.surface_contract import build_surface_prompt

        candidates = [{'id': 'abcd1234', 'title': 'Test',
                       'type': 'fact', 'content': 'x',
                       'confidence': 0.5, 'score': 0.8}]
        prompt, _ = build_surface_prompt(
            candidates, "test query",
            frame="")
        # Explicit degraded marker — no silent fallback to old layout
        self.assertIn('no partnership context', prompt.lower())
        # Make sure no Phase 1 layout artifacts appear
        self.assertNotIn('Session arc', prompt)
        self.assertNotIn("Encoder's recent journal", prompt)


if __name__ == '__main__':
    unittest.main()
