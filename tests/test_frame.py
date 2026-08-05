"""Tests for the Frame Constructor — Anchor's structured awareness object.

Covers (post 2026-08-05 lean Frame: deterministic session-state only):
- build_frame returns markdown with Session / Current focus / Recent moves
- removed queried-node sections (wisdom / Operator / Partnership) never appear
- the Session header carries project / clock / worktree from session env
- ctx.get_frame(brain) is the session-scoped entry point
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
        # A wisdom-type node exists, but the wisdom section is disabled
        # (2026-07-16 ruling: no queried nodes in the Frame — deterministic
        # session state only, pending the identity-prior redesign)
        self.brain.remember(
            type='insight', title='Test insight for wisdom',
            content='Some insight content')

        frame = build_frame(self.brain, 'test_session')

        self.assertIsInstance(frame, str)
        self.assertIn('## Session', frame)
        self.assertIn('## Current focus', frame)
        self.assertIn('## Recent moves', frame)
        # Removed / disabled sections must NOT appear
        self.assertNotIn("## What I've learned", frame)
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


class TestSessionHeader(BrainTestBase):
    """The Session header carries the deterministic situational anchor."""
    needs_embedder = False

    def test_header_shows_project_and_worktree_from_session_env(self):
        ctx = self.brain.get_or_create_session('hdr-sess')
        ctx.set_env(cwd='/x/y', branch='main', worktree='wt-1', project='brain')
        ctx.save(self.brain._session_state)

        frame = build_frame(self.brain, 'hdr-sess')
        self.assertIn('- Project: brain', frame)
        self.assertIn('- Worktree: wt-1', frame)

    def test_header_unscoped_project_is_explicit(self):
        # '' project renders as (unscoped) — itself a signal (no project
        # pressure applies), never a silently missing line.
        frame = build_frame(self.brain, 'hdr-fresh')
        self.assertIn('- Project: (unscoped)', frame)
        self.assertNotIn('- Worktree:', frame)   # no worktree → no line

    def test_header_clock_uses_passed_at(self):
        # Replays pass their injected conversation-time; the header must
        # honor it (bare wall-clock would corrupt time-anchored replays).
        import datetime as dt
        at = dt.datetime(2026, 1, 2, 3, 4, tzinfo=dt.timezone.utc)
        frame = build_frame(self.brain, 'hdr-at', at=at)
        self.assertIn('- Now: 2026-01-02 03:04 UTC (Friday)', frame)


class TestSessionContextGetFrame(BrainTestBase):
    """ctx.get_frame(brain) is the public session-scoped entry point."""
    needs_embedder = False

    def test_ctx_get_frame_returns_same_as_build_frame(self):
        # Deterministic sections match; the clock line can tick between the
        # two calls, so compare with the Now line stripped.
        ctx = self.brain.get_or_create_session('test_session_ctx')
        strip = lambda f: '\n'.join(
            l for l in f.splitlines() if not l.startswith('- Now:'))
        from_ctx = strip(ctx.get_frame(self.brain))
        from_direct = strip(build_frame(self.brain, 'test_session_ctx'))
        self.assertEqual(from_ctx, from_direct)
