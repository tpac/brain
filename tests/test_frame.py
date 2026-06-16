"""Tests for the Frame Constructor — Anchor's structured awareness object.

Covers (post 2026-06-15 collapse to three sections):
- build_frame returns markdown with What I've learned / Current focus / Recent moves
- the removed Operator + Partnership sections no longer appear
- Frame's wisdom section routes via brain.aspects.wisdom
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
    """Frame builds the expected three sections; removed sections are gone."""
    needs_embedder = False

    def test_build_frame_returns_markdown_with_all_sections(self):
        # A wisdom-type node so the wisdom section has content
        self.brain.remember(
            type='insight', title='Test insight for wisdom',
            content='Some insight content')

        frame = build_frame(self.brain, 'test_session')

        self.assertIsInstance(frame, str)
        self.assertIn("## What I've learned", frame)
        self.assertIn('## Current focus', frame)
        self.assertIn('## Recent moves', frame)
        # Removed sections must NOT appear
        self.assertNotIn('## Operator', frame)
        self.assertNotIn('## Partnership', frame)
        self.assertNotIn('## Active threads', frame)

    def test_build_frame_handles_empty_brain(self):
        # No nodes — should still render the kept sections without crashing
        frame = build_frame(self.brain, 'test_session')

        self.assertIn('## Current focus', frame)
        self.assertIn('## Recent moves', frame)
        # Empty sections render gracefully, not as errors
        self.assertNotIn('Traceback', frame)
        self.assertNotIn('Error', frame)


class TestAspectRoutingForFrame(BrainTestBase):
    """Frame's wisdom section routes via brain.aspects.wisdom."""
    needs_embedder = False

    def test_aspects_wisdom_includes_generative_types(self):
        members = self.brain.aspects.wisdom.node_types
        for t in ('insight', 'lesson', 'principle', 'vision'):
            self.assertIn(t, members)

    def test_aspects_wisdom_excludes_operational_and_tactical(self):
        # The whole point of the curated aspect: no dev-rule / dev-record pollution
        members = self.brain.aspects.wisdom.node_types
        for t in ('rule', 'operator', 'decision', 'fact', 'bug', 'mechanism'):
            self.assertNotIn(t, members)


class TestSessionContextGetFrame(BrainTestBase):
    """ctx.get_frame(brain) is the public session-scoped entry point."""
    needs_embedder = False

    def test_ctx_get_frame_returns_same_as_build_frame(self):
        # Fresh brain: no wisdom nodes → deterministic "(nothing yet)", no
        # random draw, so the two calls match.
        ctx = self.brain.get_or_create_session('test_session_ctx')
        from_ctx = ctx.get_frame(self.brain)
        from_direct = build_frame(self.brain, 'test_session_ctx')
        self.assertEqual(from_ctx, from_direct)

    def test_ctx_get_frame_includes_session_id(self):
        # Frame should reference the session via current_focus / recent_moves
        ctx = self.brain.get_or_create_session('test_session_id_check')
        frame = ctx.get_frame(self.brain)
        self.assertIn('## Recent moves', frame)


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
