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
from servers.daemon_hooks import (
    hook_recall, post_response_common,
    hook_worktree_context, hook_worktree_cleanup,
)


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

    def test_hook_recall_writes_user_message_trace(self):
        """hook_recall writes the user_message S0 trace at prompt-arrival. The
        write moved here from post_response_common (it precedes recall, so it
        holds even on the empty/early-return path) so presence/peek can surface a
        stream's current prompt mid-turn instead of only after the turn completes."""
        sid = 'test-recall-usermsg'
        hook_recall(self.brain, {"prompt": "operator prompt here",
                                 "session_id": sid}, [])
        rows = self.brain.logs_conn.execute(
            "SELECT summary FROM trace_events WHERE scale='s0' "
            "AND session_id=? AND ref_type='user_message'", (sid,)).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertIn("operator prompt here", rows[0][0])

    def test_register_only_registers_turn_but_skips_recall_and_surface(self):
        """Short answers ('yes') routed with register_only=True must REGISTER the
        turn — user_message S0 trace + conversational classification
        (last_recall_stop == stop_counter) — while skipping the expensive recall
        and Haiku surface. The pre-fix client dropped these turns entirely (no
        trace, misfiled as a heartbeat); the regression here is that registration
        happens without paying for recall/surface."""
        sid = 'test-register-only'
        with patch('servers.daemon_hooks._run_surface') as mock_surface, \
                patch.object(self.brain, 'recall') as mock_recall:
            result = hook_recall(self.brain, {"prompt": "yes", "session_id": sid,
                                              "register_only": True}, [])
        # registered: user_message trace written...
        rows = self.brain.logs_conn.execute(
            "SELECT summary FROM trace_events WHERE scale='s0' "
            "AND session_id=? AND ref_type='user_message'", (sid,)).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertIn("yes", rows[0][0])
        # ...and classified conversational (so Stop writes assistant_message, not heartbeat)
        ctx = self.brain.get_or_create_session(sid)
        self.assertEqual(ctx.last_recall_stop, ctx.stop_counter)
        # ...but recall + Haiku surface were NOT run (the efficiency win)
        mock_recall.assert_not_called()
        mock_surface.assert_not_called()
        self.assertEqual(result["json"].get("decision"), "approve")

    def test_register_only_turn_is_conversational_through_stop(self):
        """Downstream proof: a register-only turn classifies conversational at
        Stop — post_response_common writes the assistant_message half (NOT a
        heartbeat), so the Scribe sees both halves of a short exchange.

        The assert_not_called pair pins this to the register-only branch: turn
        classification (last_recall_stop) is set unconditionally upstream, so the
        Stop assertions alone would pass even if the branch were deleted. Without
        the branch, recall/surface WOULD run on 'yes' — so these two assertions
        are what make the test fail if register-only regresses."""
        sid = 'test-register-only-stop'
        with patch('servers.daemon_hooks._run_surface') as mock_surface, \
                patch.object(self.brain, 'recall') as mock_recall:
            hook_recall(self.brain, {"prompt": "yes", "session_id": sid,
                                     "register_only": True}, [])
        # branch-dependent: register-only must skip the expensive K stages
        mock_recall.assert_not_called()
        mock_surface.assert_not_called()
        post_response_common(self.brain, sid, "yes", "ok, shipping it")
        refs = [r[0] for r in self.brain.logs_conn.execute(
            "SELECT ref_type FROM trace_events WHERE scale='s0' "
            "AND session_id=?", (sid,)).fetchall()]
        self.assertIn('user_message', refs)        # the short answer registered
        self.assertIn('assistant_message', refs)   # reply registered (conversational)
        self.assertNotIn('heartbeat', refs)        # NOT misfiled as a wakeup

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
    /watch wakeup re-arm, recall skipped client-side). It advances stop_counter
    (the per-stop SEQUENCE → unique chain IDs), sets last_turn_conversational
    (the Stop gate's heartbeat-skip), and writes the right s0 trace type. The
    Scribe's CADENCE is no longer a counter here — it's derived live from the
    user_message traces (see test_turns_since_last_encode_trace_pull); a heartbeat
    writes a `heartbeat` trace, not a user_message, so it can't drag the cadence.
    See trace_contract S0 TURN CLASSIFICATION."""

    needs_embedder = False

    def _s0_refs(self, session_id):
        rows = self.brain.logs_conn.execute(
            "SELECT ref_type FROM trace_events WHERE scale='s0' AND session_id=? "
            "ORDER BY created_at", (session_id,)).fetchall()
        return [r[0] for r in rows]

    def test_conversational_turn_advances_sequence_and_writes_assistant_message(self):
        # post_response_common writes the ASSISTANT half at Stop; the user_message
        # half is now written upstream by hook_recall at prompt-arrival (see
        # TestHookRecallOutput.test_hook_recall_writes_user_message_trace). This
        # test exercises post_response_common in isolation, so only the assistant
        # trace appears here — both share the same chain_id (stop_counter is
        # unchanged between hook_recall and this Stop), so the pair stays grouped.
        sid = 'test-turn-conv'
        ctx = self.brain.get_or_create_session(sid)
        seq_before = ctx.stop_counter
        ctx.last_recall_stop = ctx.stop_counter   # simulate hook_recall having run this stop
        post_response_common(self.brain, sid, "a real operator prompt", "a response")
        self.assertEqual(ctx.stop_counter, seq_before + 1)           # sequence advances
        self.assertTrue(ctx.last_turn_conversational)                # gate will treat as a turn
        refs = self._s0_refs(sid)
        self.assertIn('assistant_message', refs)
        self.assertNotIn('heartbeat', refs)

    def test_heartbeat_advances_sequence_but_not_cadence_no_user_message(self):
        sid = 'test-turn-heartbeat'
        ctx = self.brain.get_or_create_session(sid)
        seq_before = ctx.stop_counter
        post_response_common(self.brain, sid, "/watch skill body", "(watching — inbox empty)")
        self.assertEqual(ctx.stop_counter, seq_before + 1)           # sequence advances → unique chain IDs
        self.assertFalse(ctx.last_turn_conversational)               # gate skips it
        refs = self._s0_refs(sid)
        self.assertIn('heartbeat', refs)
        self.assertNotIn('user_message', refs)                       # no user_message → cadence untouched
        self.assertNotIn('assistant_message', refs)

    def test_mixed_sequence_classifies_each_turn(self):
        # real → heartbeat → real: stop_counter advances on ALL (unique chain IDs),
        # but only real turns enter the conversation stream (assistant_message);
        # the heartbeat writes a `heartbeat` trace and never a user/assistant one,
        # so it can't drag the trace-derived cadence.
        sid = 'test-turn-mixed'
        ctx = self.brain.get_or_create_session(sid)
        ctx.last_recall_stop = ctx.stop_counter
        post_response_common(self.brain, sid, "real one", "r")
        seq_after_real = ctx.stop_counter
        post_response_common(self.brain, sid, "/watch", "(watching)")   # heartbeat: no recall mark
        self.assertFalse(ctx.last_turn_conversational)
        self.assertEqual(ctx.stop_counter, seq_after_real + 1)          # sequence advanced
        ctx.last_recall_stop = ctx.stop_counter                         # next real turn
        post_response_common(self.brain, sid, "real two", "r")
        self.assertTrue(ctx.last_turn_conversational)
        refs = self._s0_refs(sid)
        self.assertEqual(refs.count('assistant_message'), 2)           # two real turns
        self.assertEqual(refs.count('heartbeat'), 1)                   # one heartbeat, off the cadence


class TestEncodingGate(BrainTestBase):
    """The Stop-hook encoder gate (hook_post_response_track). Proves the
    behavioral guarantee the cross-session fix is about: each parallel session
    fires its Scribe every ENCODE_EVERY of ITS OWN conversational turns, the
    global encoder lock serializes runs WITHOUT losing a starved session's
    backlog (LEVEL trigger re-fires once the lock frees), and a sub-threshold
    session never fires. The gate's decision is surfaced in the return
    `output` string, so we assert on that rather than mocking internals.
    """

    needs_embedder = False

    def setUp(self):
        super().setUp()
        from servers.daemon_hooks import _encoding_lock
        self._lock = _encoding_lock
        # Defensive: a prior test that left the global lock held would make
        # every gate here read "skipped". Start from a known-free lock.
        if self._lock.locked():
            self._lock.release()

    def tearDown(self):
        if self._lock.locked():
            self._lock.release()
        super().tearDown()

    _row = [0]

    def _seed_turns(self, sid, n):
        """Insert n conversational (user_message) s0 traces for `sid`."""
        c = self.brain.logs_conn
        for _i in range(n):
            self._row[0] += 1
            c.execute(
                "INSERT INTO trace_events (id, chain_id, scale, event_type, "
                "ref_type, ref_id, summary, metadata, session_id, "
                "interaction_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ('g%07d' % self._row[0], 's0-%s-%d' % (sid[:6], self._row[0]),
                 's0', 'K', 'user_message', '', 'hi', None, sid, None,
                 '2026-06-13T10:%02d:%02d+00:00' % (self._row[0] // 60, self._row[0] % 60)))
        c.commit()

    def _fire_stop(self, sid):
        """Drive one conversational Stop through the gate; return its status."""
        from servers.daemon_hooks import hook_post_response_track
        ctx = self.brain.get_or_create_session(sid)
        ctx.last_recall_stop = ctx.stop_counter   # mark this stop conversational
        out = hook_post_response_track(self.brain, {
            'session_id': sid, 'hook_event_name': 'Stop',
            'prompt': 'p', 'last_assistant_message': 'r'}, [])
        return out.get('output', '')

    def _patched_rib(self):
        """Patch run_in_background to NOT spawn a Sonnet thread, but to honor
        the real lock contract: ownership transfers to the 'thread', which
        releases on completion. We simulate instant completion by releasing
        immediately — so the next turn sees a free lock, exactly as prod does
        once an encode finishes."""
        from unittest.mock import patch
        self.spawned = []

        def fake(name, brain_db_path, session_id, counter, lock,
                 run_fn, on_complete=None):
            self.spawned.append({'session_id': session_id, 'counter': counter})
            lock.release()   # mimic the background thread's finally
        return patch('servers.scales.runner.run_in_background', side_effect=fake)

    def test_fires_at_threshold_and_skips_below(self):
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        with self._patched_rib():
            # Session A: exactly ENCODE_EVERY turns → fires.
            self._seed_turns('gate-A', ENCODE_EVERY)
            self.assertIn('encoding started', self._fire_stop('gate-A'))
            self.assertEqual([s['session_id'] for s in self.spawned], ['gate-A'])
            # Session B: below threshold → does NOT fire, reports progress.
            self._seed_turns('gate-B', ENCODE_EVERY - 3)
            statusB = self._fire_stop('gate-B')
            self.assertIn('encoding %d/%d' % (ENCODE_EVERY - 3, ENCODE_EVERY), statusB)
            self.assertNotIn('gate-B', [s['session_id'] for s in self.spawned])

    def test_each_parallel_session_fires_on_its_own_count(self):
        # Two streams, interleaved, BOTH at threshold independently. Each fires
        # its own encode — neither rides the other's count.
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        with self._patched_rib():
            self._seed_turns('par-A', ENCODE_EVERY)
            self._seed_turns('par-B', ENCODE_EVERY)
            self.assertIn('encoding started', self._fire_stop('par-A'))
            self.assertIn('encoding started', self._fire_stop('par-B'))
            fired = sorted(s['session_id'] for s in self.spawned)
            self.assertEqual(fired, ['par-A', 'par-B'])

    def test_lock_contention_skips_then_refires_when_free(self):
        # The crux of cross-session safety with a single global encoder lock: a
        # starved session is NOT lost. While A's encode holds the lock, B (at
        # threshold) skips — but the LEVEL trigger means B re-fires on its next
        # turn once the lock is free. "Every 5+ turns" survives contention.
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        self._seed_turns('busy-B', ENCODE_EVERY)
        # Simulate session A mid-encode: the global lock is held.
        self._lock.acquire()
        statusBusy = self._fire_stop('busy-B')
        self.assertIn('skipped (previous still running)', statusBusy)
        self.assertFalse(hasattr(self, 'spawned') and
                         any(s['session_id'] == 'busy-B' for s in self.spawned))
        # A's encode finishes → lock frees. B's backlog (still ≥ ENCODE_EVERY,
        # never encoded) must fire on its very next turn.
        self._lock.release()
        with self._patched_rib():
            self.assertIn('encoding started', self._fire_stop('busy-B'))
            self.assertEqual([s['session_id'] for s in self.spawned], ['busy-B'])


class TestWorktreeHooks(BrainTestBase):
    """WorktreeCreate/Remove write the per-session worktree identity and NEVER
    emit context on stdout. Claude Code consumes a WorktreeCreate hook's stdout
    as the new worktree path, so the `[BRAIN] GIT CONTEXT` block that used to
    print here got chdir'd into → the `ENOENT chdir '<repo>' -> '[/BRAIN]'`."""

    needs_embedder = False

    def test_worktree_create_stamps_session_and_emits_no_output(self):
        sid = 'wt-create-sess'
        self.brain.reset_session_activity(session_id=sid, cwd='/tmp')  # boot first
        result = hook_worktree_context(
            self.brain,
            {"session_id": sid, "name": "emb-bench", "cwd": "/tmp"}, [])
        # NO stdout leak — empty output, no [BRAIN] marker for CC to chdir into.
        self.assertEqual(result["output"], "")
        # Recorded on the SESSION OBJECT, not the old global config key.
        self.assertEqual(self.brain.session_env_for(sid)["worktree"], "emb-bench")
        self.assertEqual(self.brain.get_config("current_worktree", "SENTINEL"), "SENTINEL")
        # Persisted immediately (not just cached): a fresh load from logs_conn sees it.
        from servers.session_context import SessionContext
        reloaded = SessionContext.load(self.brain.logs_conn, sid)
        self.assertEqual(reloaded.worktree, "emb-bench")

    def test_worktree_remove_clears_session_worktree(self):
        sid = 'wt-remove-sess'
        self.brain.reset_session_activity(session_id=sid, cwd='/tmp')
        hook_worktree_context(self.brain, {"session_id": sid, "name": "emb-bench", "cwd": "/tmp"}, [])
        self.assertEqual(self.brain.session_env_for(sid)["worktree"], "emb-bench")
        result = hook_worktree_cleanup(self.brain, {"session_id": sid}, [])
        self.assertEqual(result["output"], "")
        self.assertEqual(self.brain.session_env_for(sid)["worktree"], "")

    def test_worktree_create_without_session_id_is_safe(self):
        # No session_id → must NOT touch any session: never fall back to the
        # singleton (which would mis-attribute the worktree to another stream) and
        # never create a phantom row. Asserts the invariant, not just output==''.
        from unittest.mock import patch
        with patch.object(self.brain, 'get_or_create_session',
                          wraps=self.brain.get_or_create_session) as m:
            result = hook_worktree_context(self.brain, {"name": "x", "cwd": "/tmp"}, [])
        self.assertEqual(result["output"], "")
        m.assert_not_called()


if __name__ == '__main__':
    unittest.main()
