"""Tests for the Frame Constructor — Anchor's structured awareness object.

Phase 2 of the Frame architecture. Tests cover:
- build_frame returns a markdown string with all five sections
- _family_members reads from s2_node_families interaction
- _family_members falls back to hardcoded defaults when interaction is empty
- ctx.get_frame(brain) is the session-scoped entry point
- s2_node_families seeds correctly on a fresh brain
- build_surface_prompt accepts and renders the frame param
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.scales.s1.frame import (
    build_frame, _family_members, _members_in_families,
    _FALLBACK_FAMILIES,
    OPERATOR_FAMILY, PERMANENT_FAMILY, WARM_FAMILIES, ACTIVE_FAMILY,
)


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


class TestFamilyResolution(BrainTestBase):
    """Family lookup reads interactions table with hardcoded fallback."""
    needs_embedder = False

    def test_family_members_reads_seeded_families(self):
        # On a fresh brain, seed_interactions should have populated s2_node_families
        members = _family_members(self.brain, OPERATOR_FAMILY)
        self.assertIsInstance(members, list)
        self.assertGreater(len(members), 0)
        # The seeded identity_bearing family should include 'principle'
        self.assertIn('principle', members)

    def test_family_members_falls_back_when_family_missing(self):
        # Ask for a family that doesn't exist in seed
        members = _family_members(self.brain, 'nonexistent_family_xyz')
        # Should return the fallback (empty for unknown family)
        self.assertEqual(members, [])

    def test_family_members_uses_fallback_when_interaction_empty(self):
        # Wipe the interaction so the interaction lookup returns empty config
        self.brain._interaction_dal.register(
            's2_node_families', template='', parameters='{}',
            created_by='test')
        # Now _family_members should fall back to _FALLBACK_FAMILIES
        members = _family_members(self.brain, OPERATOR_FAMILY)
        # Fallback for identity_bearing exists — verify it's used
        self.assertEqual(set(members), set(_FALLBACK_FAMILIES['identity_bearing']))

    def test_members_in_families_unions_multiple(self):
        members = _members_in_families(self.brain, WARM_FAMILIES)
        # Union should include moment (from episodic_anchor) AND insight (from lesson_insight)
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


class TestNodeFamiliesSeed(BrainTestBase):
    """The s2_node_families interaction is seeded on fresh brain."""
    needs_embedder = False

    def test_s2_node_families_present_after_init(self):
        config = self.brain.get_interaction_config('s2_node_families')
        self.assertIsInstance(config, dict)
        self.assertGreater(len(config), 0,
                          's2_node_families should be seeded on Brain.__init__')

    def test_s2_node_families_has_expected_v1_families(self):
        config = self.brain.get_interaction_config('s2_node_families')
        # v1 seed must cover the families Frame depends on
        self.assertIn('identity_bearing', config)
        self.assertIn('episodic_anchor', config)
        self.assertIn('active_thread', config)
        self.assertIn('lesson_insight', config)

    def test_s2_node_families_shape_is_v2_nested(self):
        config = self.brain.get_interaction_config('s2_node_families')
        family = config['identity_bearing']
        # v2 nested shape: {members: [...], meaning: "..."}
        self.assertIsInstance(family, dict)
        self.assertIn('members', family)
        self.assertIn('meaning', family)
        self.assertIsInstance(family['members'], list)
        self.assertIsInstance(family['meaning'], str)


class TestSurfacePromptAcceptsFrame(BrainTestBase):
    """build_surface_prompt accepts frame and renders it as 'Partnership context:'."""
    needs_embedder = False

    def test_frame_replaces_session_context_when_provided(self):
        from servers.scales.s1.surface_contract import build_surface_prompt

        candidates = [{'id': 'abcd1234', 'title': 'Test',
                       'type': 'fact', 'content': 'x',
                       'confidence': 0.5, 'score': 0.8}]
        prompt_with_frame, _ = build_surface_prompt(
            candidates, "test query",
            session_context="THIS SHOULD NOT APPEAR",
            encoding_journal="THIS EITHER",
            frame="THE FRAME PRIOR")
        self.assertIn('THE FRAME PRIOR', prompt_with_frame)
        self.assertNotIn('THIS SHOULD NOT APPEAR', prompt_with_frame)
        self.assertNotIn('THIS EITHER', prompt_with_frame)

    def test_frame_falls_back_to_session_context_when_empty(self):
        from servers.scales.s1.surface_contract import build_surface_prompt

        candidates = [{'id': 'abcd1234', 'title': 'Test',
                       'type': 'fact', 'content': 'x',
                       'confidence': 0.5, 'score': 0.8}]
        prompt, _ = build_surface_prompt(
            candidates, "test query",
            session_context="THE OLD SESSION ARC",
            encoding_journal="THE OLD JOURNAL",
            frame="")  # empty frame triggers Phase 1 fallback layout
        self.assertIn('THE OLD SESSION ARC', prompt)
        self.assertIn('THE OLD JOURNAL', prompt)


if __name__ == '__main__':
    unittest.main()
