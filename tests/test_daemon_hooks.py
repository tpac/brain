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
    (the per-stop SEQUENCE → unique chain IDs), records last_turn_conversational
    (the classification), and writes the right s0 trace type. The
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
        self.assertTrue(ctx.last_turn_conversational)                # classified conversational
        refs = self._s0_refs(sid)
        self.assertIn('assistant_message', refs)
        self.assertNotIn('heartbeat', refs)

    def test_heartbeat_advances_sequence_but_not_cadence_no_user_message(self):
        sid = 'test-turn-heartbeat'
        ctx = self.brain.get_or_create_session(sid)
        seq_before = ctx.stop_counter
        post_response_common(self.brain, sid, "/watch skill body", "(watching — inbox empty)")
        self.assertEqual(ctx.stop_counter, seq_before + 1)           # sequence advances → unique chain IDs
        self.assertFalse(ctx.last_turn_conversational)               # classified heartbeat
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


class TestScribeReactor(unittest.TestCase):
    """The poll-driven S1 Scribe trigger — brain.scribe_due() (the decision) and
    BrainDaemon._run_scribe_poll() (single-flight). Replaces the old Stop-hook
    encoder gate: the trigger moved to the daemon poll in the S1 convergence.

    Preserves the guarantees the gate proved — each session fires on ITS OWN
    conversational count, sub-threshold never fires, one encode runs at a time
    across sessions (a busy session re-qualifies and drains on a later poll) —
    and adds the idle-tail clause. scribe_due reads only HIGHER session
    functions (present_streams + turns_since_last_encode + get_conversation),
    so we stub those: no DB, no SQL.
    """

    def _due(self, streams, turns_map, now=1_000_000.0, skip=None, boot_time=0.0,
             last_role='assistant'):
        """Brain.scribe_due bound to a fake brain whose higher session
        functions are stubbed (present_streams + turns_since_last_encode +
        get_conversation — the latter two are BrainTracesMixin methods, so
        the fake provides them directly). boot_time defaults ancient (past
        the boot-grace); pass ~now to test it. last_role is the newest
        conversational row's role (default 'assistant' = complete exchange,
        so the 5+ clause's wait-for-answer gate passes); None = empty
        conversation."""
        import types
        from servers.brain import Brain

        class FakeBrain:
            _boot_time = 0.0   # ancient → past the boot-grace window

            def present_streams(self, window_min, limit):
                return streams

            def turns_since_last_encode(self, sid):
                return turns_map.get(sid, 0)

            def get_conversation(self, sid, limit=20):
                return [{'role': last_role}] if last_role else []

            def get_or_create_session(self, sid):
                return types.SimpleNamespace(stop_counter=42)

            def _log_error(self, *a, **k):
                pass

        fb = FakeBrain()
        fb._boot_time = boot_time
        return Brain.scribe_due(fb, now=now, skip_sessions=skip)

    def _poll(self, scribe_due_fn, attempts=None, failures=None):
        """Drive BrainDaemon._run_scribe_poll against a fake daemon (scribe_due
        stubbed, run_unit_in_background patched to release the lock = instant
        completion). Returns a dict with the spawned units, the captured
        on_complete, logged error sources, and the fake daemon for inspection."""
        import threading
        import types
        from unittest.mock import patch
        from servers.daemon_server import BrainDaemon
        cap = {'spawned': [], 'logged': [], 'on_complete': None, 'encode_runs': 0}

        def fake_run(unit, name, lock, on_complete=None):
            cap['spawned'].append(unit)
            cap['on_complete'] = on_complete
            lock.release()   # mimic the encode thread's finally

        def _record_encode():
            cap['encode_runs'] += 1

        fake = types.SimpleNamespace(
            _encode_lock=threading.Lock(),
            _scribe_poll_running=True,
            _scribe_attempts=dict(attempts or {}),
            _scribe_failures=dict(failures or {}),
            brain=types.SimpleNamespace(
                scribe_due=scribe_due_fn,
                _log_error=lambda *a, **k: cap['logged'].append(a[0] if a else None),
                activity=types.SimpleNamespace(record_encode_run=_record_encode)),
        )
        with patch('servers.scales.runner.run_unit_in_background',
                   side_effect=fake_run):
            BrainDaemon._run_scribe_poll(fake)
        cap['fake'] = fake
        return cap

    def test_scribe_due_skips_cooldown_sessions(self):
        # par-B is more overdue but cooling down → par-A is picked instead (a
        # failing/cooling session can't monopolize the poll).
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'par-A', 'updated_at': self._iso(now, 10)},
             {'session_id': 'par-B', 'updated_at': self._iso(now, 10)}],
            {'par-A': ENCODE_EVERY, 'par-B': ENCODE_EVERY + 4}, now=now,
            skip={'par-B'})
        self.assertEqual(due['session_id'], 'par-A')

    def test_poll_records_attempt_and_first_poll_has_empty_cooldown(self):
        seen = {}

        def sd(now=None, skip_sessions=None):
            seen['skip'] = set(skip_sessions or ())
            return {'session_id': 'sX', 'counter': 1}
        cap = self._poll(sd)
        self.assertEqual(seen['skip'], set())               # nothing cooling yet
        self.assertIn('sX', cap['fake']._scribe_attempts)   # attempt recorded
        self.assertEqual([u.session_id for u in cap['spawned']], ['sX'])

    def test_poll_cools_down_a_recently_attempted_session(self):
        import time
        # sX attempted just now → must be in the skip set scribe_due receives.
        seen = {}

        def sd(now=None, skip_sessions=None):
            seen['skip'] = set(skip_sessions or ())
            return None
        self._poll(sd, attempts={'sX': time.time()})
        self.assertIn('sX', seen['skip'])

    def test_poll_escalates_repeated_failure(self):
        import time
        from servers.scales.s1.encode_contract import (
            SCRIBE_RETRY_COOLDOWN_SECONDS, SCRIBE_MAX_FAILED_RETRIES)
        # sX attempted past its cooldown and STILL due (cadence never advanced)
        # → a re-fire that didn't progress → failures climbs; at threshold, loud.
        old = time.time() - SCRIBE_RETRY_COOLDOWN_SECONDS - 10

        def sd(now=None, skip_sessions=None):
            return {'session_id': 'sX', 'counter': 1}   # not cooling → re-fired
        cap = self._poll(sd, attempts={'sX': old},
                         failures={'sX': SCRIBE_MAX_FAILED_RETRIES - 1})
        self.assertEqual(cap['fake']._scribe_failures['sX'],
                         SCRIBE_MAX_FAILED_RETRIES)
        self.assertIn('scribe_repeated_failure', cap['logged'])

    def test_poll_clears_cooldown_on_successful_encode(self):
        # on_complete(write_actions>0) clears the session's cooldown + failure
        # state (and counts the encode toward the S2 gate).
        def sd(now=None, skip_sessions=None):
            return {'session_id': 'sX', 'counter': 1}
        cap = self._poll(sd, failures={'sX': 1})
        self.assertIn('sX', cap['fake']._scribe_attempts)   # recorded on fire
        cap['on_complete'](3)                                # encode wrote material
        self.assertNotIn('sX', cap['fake']._scribe_attempts)
        self.assertNotIn('sX', cap['fake']._scribe_failures)
        self.assertEqual(cap['encode_runs'], 1)

    def test_poll_clears_cooldown_on_zero_write_completion(self):
        # A COMPLETED encode that wrote nothing still advanced the cadence
        # (encoding_prompt was written) — so it must clear the cooldown too, or
        # its next legit re-fire would false-trip scribe_repeated_failure. It
        # must NOT count toward the S2 gate (no material written).
        def sd(now=None, skip_sessions=None):
            return {'session_id': 'sX', 'counter': 1}
        cap = self._poll(sd, failures={'sX': 2})
        cap['on_complete'](0)                                # completed, 0 writes
        self.assertNotIn('sX', cap['fake']._scribe_attempts)
        self.assertNotIn('sX', cap['fake']._scribe_failures)
        self.assertEqual(cap['encode_runs'], 0)              # no S2-gate credit

    def test_poll_prunes_orphaned_failures(self):
        # _scribe_failures is kept keyed to live _scribe_attempts — an entry for
        # a session no longer being attempted is dropped (no unbounded leak).
        def sd(now=None, skip_sessions=None):
            return None                                      # nothing due this poll
        cap = self._poll(sd, failures={'ghost': 5})          # ghost not in attempts
        self.assertNotIn('ghost', cap['fake']._scribe_failures)

    @staticmethod
    def _iso(now, ago_s):
        import datetime
        return datetime.datetime.fromtimestamp(
            now - ago_s, datetime.timezone.utc).isoformat()

    def test_fires_at_threshold(self):
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'gate-A', 'updated_at': self._iso(now, 10)}],
            {'gate-A': ENCODE_EVERY}, now=now)
        self.assertEqual(due, {'session_id': 'gate-A', 'counter': 42})

    def test_skips_below_threshold(self):
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        # Sub-threshold AND recently active → neither clause fires.
        due = self._due(
            [{'session_id': 'gate-B', 'updated_at': self._iso(now, 10)}],
            {'gate-B': ENCODE_EVERY - 2}, now=now)
        self.assertIsNone(due)

    def test_each_session_on_its_own_count_most_overdue_first(self):
        # Two sessions both at/over threshold, independently — the MOST-overdue
        # (most turns since encode) is picked first; the backlog then drains
        # one-per-poll (single-flight), so neither rides the other's count.
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'par-A', 'updated_at': self._iso(now, 10)},
             {'session_id': 'par-B', 'updated_at': self._iso(now, 10)}],
            {'par-A': ENCODE_EVERY, 'par-B': ENCODE_EVERY + 4}, now=now)
        self.assertEqual(due['session_id'], 'par-B')

    def test_idle_tail_fires(self):
        # A session that went quiet below threshold still gets its tail encoded
        # after SCRIBE_TAIL_IDLE_SECONDS — if it has > SCRIBE_TAIL_MIN_TURNS.
        from servers.scales.s1.encode_contract import (
            SCRIBE_TAIL_IDLE_SECONDS, SCRIBE_TAIL_MIN_TURNS)
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'tail',
              'updated_at': self._iso(now, SCRIBE_TAIL_IDLE_SECONDS + 60)}],
            {'tail': SCRIBE_TAIL_MIN_TURNS + 1}, now=now)
        self.assertEqual(due['session_id'], 'tail')

    def test_idle_tail_guard_below_min_turns(self):
        # The tail skips trivial leftovers (<= SCRIBE_TAIL_MIN_TURNS) — not worth
        # a Sonnet call.
        from servers.scales.s1.encode_contract import (
            SCRIBE_TAIL_IDLE_SECONDS, SCRIBE_TAIL_MIN_TURNS)
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'trivial',
              'updated_at': self._iso(now, SCRIBE_TAIL_IDLE_SECONDS + 60)}],
            {'trivial': SCRIBE_TAIL_MIN_TURNS}, now=now)
        self.assertIsNone(due)

    def test_idle_tail_not_yet_idle_enough(self):
        # Sub-threshold but only briefly idle → the tail hasn't matured yet.
        from servers.scales.s1.encode_contract import SCRIBE_TAIL_MIN_TURNS
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'recent', 'updated_at': self._iso(now, 120)}],
            {'recent': SCRIBE_TAIL_MIN_TURNS + 1}, now=now)
        self.assertIsNone(due)

    def test_5plus_not_swept_when_session_gone_quiet(self):
        # The fix for "picks up tons of old conversations": a session with 5+
        # unencoded turns but idle past the active window is NOT 5+-encoded — it
        # waits for the 1h tail. The same session within the window DOES fire,
        # proving the bound is what gates it, not the turn count.
        from servers.scales.s1.encode_contract import (
            ENCODE_EVERY, SCRIBE_ACTIVE_WINDOW_SECONDS)
        now = 1_000_000.0
        stale = self._iso(now, SCRIBE_ACTIVE_WINDOW_SECONDS + 60)  # quiet, under 1h
        self.assertIsNone(
            self._due([{'session_id': 'quiet', 'updated_at': stale}],
                      {'quiet': ENCODE_EVERY + 3}, now=now),
            '5+ must not sweep a session that has gone quiet')
        fresh = self._iso(now, 10)
        self.assertEqual(
            self._due([{'session_id': 'active', 'updated_at': fresh}],
                      {'active': ENCODE_EVERY + 3}, now=now)['session_id'],
            'active', 'an actively-conversing 5+ session still fires')

    def test_5plus_waits_for_the_answer_to_the_threshold_prompt(self):
        # The turn count crosses ENCODE_EVERY on the USER prompt, so without a
        # completeness gate the 5+ clause always fires mid-turn and the encode
        # window ends on an unanswered question (the 2026-07-11 incident: the
        # Scribe journaled "Turn 5 has no <me> response" because the answer's
        # Stop trace landed ~30s after the snapshot). The 5+ clause must wait
        # until the newest conversational row is the assistant's answer.
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        fresh = [{'session_id': 'midturn', 'updated_at': self._iso(now, 10)}]
        turns = {'midturn': ENCODE_EVERY + 1}
        self.assertIsNone(
            self._due(fresh, turns, now=now, last_role='user'),
            '5+ must not fire while the newest turn is unanswered')
        self.assertEqual(
            self._due(fresh, turns, now=now, last_role='assistant')
            ['session_id'], 'midturn',
            'the same session fires once the answer trace lands')

    def test_idle_tail_encodes_a_dangling_question(self):
        # The tail is EXEMPT from the completeness gate: a question still
        # unanswered once the session went quiet (interrupt / disconnect —
        # Stop never fires, so no assistant trace ever lands) is genuinely
        # dangling and belongs in the encode as-is; the journal records the
        # gap honestly.
        from servers.scales.s1.encode_contract import (
            SCRIBE_TAIL_IDLE_SECONDS, SCRIBE_TAIL_MIN_TURNS)
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'dangling',
              'updated_at': self._iso(now, SCRIBE_TAIL_IDLE_SECONDS + 60)}],
            {'dangling': SCRIBE_TAIL_MIN_TURNS + 1}, now=now, last_role='user')
        self.assertEqual(due['session_id'], 'dangling')

    def test_boot_grace_suppresses_the_poll(self):
        # Just after a (re)start, the poll is a no-op for the settle window — no
        # backlog flush the instant the daemon comes up.
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'active', 'updated_at': self._iso(now, 10)}],
            {'active': ENCODE_EVERY + 3}, now=now, boot_time=now)   # just booted
        self.assertIsNone(due)

    def test_poll_single_flight_skips_when_encode_running(self):
        # Single-flight: while an encode holds _encode_lock, the poll is a no-op
        # — it never even asks scribe_due. The skipped session re-qualifies and
        # drains on a later poll (level trigger). This is the old gate's
        # lock-contention guarantee, now at the daemon poll.
        import threading
        import types
        from servers.daemon_server import BrainDaemon
        decided = []
        lock = threading.Lock()
        lock.acquire()   # an encode is "running"
        fake = types.SimpleNamespace(
            _encode_lock=lock, _scribe_poll_running=True,
            brain=types.SimpleNamespace(scribe_due=lambda: decided.append(1)))
        BrainDaemon._run_scribe_poll(fake)
        self.assertEqual(decided, [], 'a busy lock must skip the decision')
        self.assertFalse(fake._scribe_poll_running, 'poll flag must clear')
        self.assertTrue(lock.locked(), 'the running encode keeps the lock')


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


class TestAnchorTouched(BrainTestBase):
    """The per-turn 'Anchor touched' feed (Piece 3a), driven end-to-end through
    REAL _dispatch — never by hand-appending to ctx.touched (that shortcut stays
    green even when the feature is fully dead, which is exactly how the original
    feature-killer — keying on args['session_id'] instead of the proxy-stamped
    _caller_session — slipped through). Each test exercises the production path:
    dispatch keys the touch onto the caller's ctx; the Stop flush writes it."""

    def _daemon(self):
        from servers.daemon_server import BrainDaemon
        d = BrainDaemon('/tmp/unused-anchor-touched.db')  # no start() → no sockets
        d.brain = self.brain
        return d

    def _touched_rows(self, sid):
        return self.brain.logs_conn.execute(
            "SELECT metadata FROM trace_events WHERE scale='s0' "
            "AND session_id=? AND ref_type='anchor_touched'", (sid,)).fetchall()

    def test_write_keys_by_caller_session_then_flushes_and_resets(self):
        # remember sends NO session_id — only _caller_session. Full path: dispatch
        # keys the new id onto the caller's ctx (the bug: it used to key on the
        # absent session_id → recorded nothing), then the Stop flush writes one
        # anchor_touched delta on the turn chain and resets the accumulator.
        import json
        sid = 'sid-e2e'
        d = self._daemon()
        res = d._dispatch('remember', {'type': 'note', 'title': 'e2e',
                                       'content': 'b', '_caller_session': sid})
        new_id = (res.get('affected') or {}).get('created', [None])[0]
        self.assertTrue(new_id)
        ctx = self.brain.get_or_create_session(sid)
        self.assertIn(new_id, ctx.touched['created'])        # keyed correctly

        post_response_common(self.brain, sid, "made a node", "done")
        rows = self._touched_rows(sid)
        self.assertEqual(len(rows), 1)
        self.assertIn(new_id, json.loads(rows[0][0])['created'])
        self.assertTrue(all(not v for v in ctx.touched.values()))  # reset

    def test_get_node_records_recalled(self):
        nid = self.brain.remember(type='note', title='look me up', content='x')['id']
        d = self._daemon()
        d._dispatch('get_node', {'node_id': nid, '_caller_session': 'sid-read'})
        self.assertIn(nid, self.brain.get_or_create_session('sid-read').touched['recalled'])

    def test_get_nodes_skips_not_found_error_entries(self):
        nid = self.brain.remember(type='note', title='real one', content='x')['id']
        d = self._daemon()
        d._dispatch('get_nodes', {'node_ids': [nid, 'bogusid0'],
                                  '_caller_session': 'sid-batch'})
        touched = self.brain.get_or_create_session('sid-batch').touched
        self.assertIn(nid, touched['recalled'])
        self.assertNotIn('bogusid0', touched['recalled'])   # error entry skipped

    def test_recall_hot_path_does_not_create_session(self):
        # A non-contributing read must early-out BEFORE get_or_create_session, so
        # it adds no session row and no touched state on the recall hot path.
        from unittest.mock import patch
        d = self._daemon()
        with patch.object(self.brain, 'get_or_create_session',
                          wraps=self.brain.get_or_create_session) as m:
            d._dispatch('query_traces', {'ref_type': 'recall',
                                         '_caller_session': 'sid-hot'})
        m.assert_not_called()

    def test_no_delta_when_nothing_touched(self):
        # A turn with no Anchor tool activity writes no anchor_touched delta.
        sid = 'sid-empty'
        post_response_common(self.brain, sid, "hi", "hello")
        self.assertEqual(len(self._touched_rows(sid)), 0)


if __name__ == '__main__':
    unittest.main()
