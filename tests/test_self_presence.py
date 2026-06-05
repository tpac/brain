"""Phase 1 self-channel presence — the PULL primitives.

Covers brain.present_streams (wall-clock roster), presence.build_presence
(roster + rendered line), and presence.peek (drill into one stream). Pure
reads; no embedder needed.
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.clock import iso_cutoff
from servers.scales.self_channel import presence, self_contract, signal


class TestSelfPresence(BrainTestBase):
    needs_embedder = False

    def _save_stream(self, sid, focus='', updated_at=None):
        """Seed a real-turn S0 trace (user_message) for `sid` — the presence
        liveness + focus source (present_streams reads traces, not
        session_state, so autosave can't fake liveness). `focus` becomes the
        trace summary; `updated_at` backdates created_at to place the turn
        earlier in the window."""
        self.brain._trace_dal.append(
            chain_id='s0-%s-0' % sid[:8], scale='s0', event_type='K',
            ref_type='user_message', summary=focus, session_id=sid)
        if updated_at is not None:
            self.brain.logs_conn.execute(
                "UPDATE trace_events SET created_at = ? "
                "WHERE session_id = ? AND ref_type = 'user_message'",
                (updated_at, sid))
            self.brain.logs_conn.commit()

    def test_present_streams_excludes_self_and_stale(self):
        self._save_stream('streamAAAA', focus='dashboard fix')
        self._save_stream('streamBBBB', focus='self-channel design')
        # Outside the wall-clock window — awake days ago, not now.
        self._save_stream('streamOLD0', focus='ancient',
                          updated_at=iso_cutoff(days=3))

        rows = self.brain.present_streams(
            exclude_session='streamAAAA', window_min=30, limit=5)
        ids = {r['session_id'] for r in rows}

        self.assertIn('streamBBBB', ids)
        self.assertNotIn('streamAAAA', ids, "caller must be excluded from its own roster")
        self.assertNotIn('streamOLD0', ids, "stale stream outside window must drop")

    def test_build_presence_renders_line_and_one_line_focus(self):
        self._save_stream('streamAAAA', focus='dashboard fix\nSECOND LINE IGNORED')
        self._save_stream('streamBBBB', focus='self-channel presence build')

        # limit=10 so both of mine surface (the test brain has its own
        # session_state row too — present_streams correctly returns all live
        # streams except the caller, so we assert behavior, not an exact count).
        out = presence.build_presence(self.brain, my_session_id='streamZZZZ', limit=10)

        ids = {s['session_id'] for s in out['streams']}
        self.assertIn('streamAAAA', ids)
        self.assertIn('streamBBBB', ids)
        focuses = {s['focus'] for s in out['streams']}
        self.assertIn('dashboard fix', focuses)  # first line only
        self.assertNotIn('dashboard fix\nSECOND LINE IGNORED', focuses)
        # the rendered line reflects exactly the streams returned
        self.assertIn('streams of thought live: %d' % len(out['streams']), out['line'])

    def test_peek_returns_full_focus(self):
        # peek drills into the full session ARC (session_context_for), a
        # separate source from the roster's trace-derived one-line focus.
        self.brain.set_config('session_context_streamAAAA', 'line one\nline two')

        hit = presence.peek(self.brain, 'streamAAAA')
        self.assertTrue(hit['found'])
        self.assertEqual(hit['focus'], 'line one\nline two')  # full arc, not one line

        miss = presence.peek(self.brain, 'no-such-stream')
        self.assertFalse(miss['found'])
        self.assertEqual(miss['focus'], '')

    def test_cap_is_respected(self):
        for i in range(5):
            self._save_stream('stream%05d' % i, focus='focus %d' % i)
        out = presence.build_presence(self.brain, my_session_id='other', limit=2)
        self.assertEqual(len(out['streams']), 2)

    def test_default_cap_from_contract(self):
        for i in range(self_contract.PRESENCE_MAX_STREAMS + 3):
            self._save_stream('strm%05d' % i, focus='f%d' % i)
        out = presence.build_presence(self.brain, my_session_id='other')  # no limit → contract cap
        self.assertEqual(len(out['streams']), self_contract.PRESENCE_MAX_STREAMS)

    def test_liveness_states_and_lost_surfacing(self):
        """active/dormant/lost by recency; lost surfaced separately (named in the
        line), kept out of the live roster, not silently dropped at the edge."""
        self._save_stream('streamACTV', focus='working now')                                  # fresh → active
        self._save_stream('streamDORM', focus='quiet', updated_at=iso_cutoff(minutes=15))      # → dormant
        self._save_stream('streamLOST', focus='vanished', updated_at=iso_cutoff(minutes=45))   # → lost (≤60 grace)
        out = presence.build_presence(self.brain, my_session_id='other', limit=10)
        states = {s['session_id']: s['state'] for s in out['streams']}
        self.assertEqual(states.get('streamACTV'), 'active')
        self.assertEqual(states.get('streamDORM'), 'dormant')
        lost_ids = {s['session_id'] for s in out['lost']}
        self.assertIn('streamLOST', lost_ids)              # surfaced...
        self.assertNotIn('streamLOST', states)             # ...but not in the live roster
        self.assertIn('lost', out['line'])                 # and named in the line

    def test_resolve_to_canonical_prefix_and_graceful(self):
        """to= resolution: broadcast + full UUID are canonical; the 8-char short
        (an id-prefix) matches the live roster; unknown is loud (no address)."""
        sid = 'aaaaaaaa-1111-2222-3333-444444444444'
        self._save_stream(sid, focus='dal cleanup')
        self.assertEqual(signal.resolve_to(self.brain, 'broadcast')[0], self_contract.ADDR_BROADCAST)
        addr, err = signal.resolve_to(self.brain, sid)            # full id → canonical
        self.assertIsNone(err)
        self.assertEqual(addr, self_contract.address_for_stream(sid))
        addr, err = signal.resolve_to(self.brain, 'aaaaaaaa')     # 8-char short (prefix) → that stream
        self.assertIsNone(err)
        self.assertEqual(addr, self_contract.address_for_stream(sid))
        addr, err = signal.resolve_to(self.brain, 'nope')         # unknown → loud
        self.assertIsNone(addr)
        self.assertIn('no live stream', err)

    def test_resolve_to_ambiguous_is_loud(self):
        self._save_stream('bbbbbbbb-0000-0000-0000-000000000001', focus='x')
        self._save_stream('bbbbbbbb-0000-0000-0000-000000000002', focus='y')
        addr, err = signal.resolve_to(self.brain, 'bbbbbbbb')     # id-prefix → 2 matches
        self.assertIsNone(addr)
        self.assertIn('matches', err)

    def test_presence_shows_short_id(self):
        """No self-labeling — the roster shows the 8-char short id (+ focus + state)."""
        sid = 'cccccccc-0000-0000-0000-000000000003'
        self._save_stream(sid, focus='docs')
        out = presence.build_presence(self.brain, my_session_id='other', limit=10)
        self.assertIn('cccccccc', {s['short'] for s in out['streams']})
        self.assertIn('cccccccc', out['line'])


class TestPresenceCountsWatchers(BrainTestBase):
    """B2 (2026-06-04): a /watch listener emits only `heartbeat` turns, yet it
    is the MOST reachable stream — it can be triggered to act. It must count as
    present, and its focus must not be polluted by the `<task-notification>`
    wake envelope."""
    needs_embedder = False

    def _turn(self, sid, ref_type, summary, updated_at=None, event_type='K'):
        self.brain._trace_dal.append(
            chain_id='s0-%s-0' % sid[:8], scale='s0', event_type=event_type,
            ref_type=ref_type, summary=summary, session_id=sid)
        if updated_at is not None:
            self.brain.logs_conn.execute(
                "UPDATE trace_events SET created_at = ? "
                "WHERE session_id = ? AND ref_type = ? AND summary = ?",
                (updated_at, sid, ref_type, summary))
            self.brain.logs_conn.commit()

    def test_heartbeat_only_watcher_is_present(self):
        # A pure listener whose only recent turn is a watch tick (heartbeat).
        self._turn('watcherXX', 'heartbeat', 'Quiet and listening.')
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        ids = {r['session_id'] for r in rows}
        self.assertIn('watcherXX', ids,
                      "a heartbeat-only watcher is reachable and must show present")

    def test_real_prompt_wins_over_task_notification_in_focus(self):
        # Recent turns: a real prompt (earlier) then a watch ignition (now).
        # Focus must surface the real prompt, not the <task-notification>.
        self._turn('mixedXXXX', 'user_message', 'fix the recall bug',
                   updated_at=iso_cutoff(minutes=5))
        self._turn('mixedXXXX', 'user_message', '<task-notification>\n<event>twin msg')
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        focus = {r['session_id']: r['focus'] for r in rows}.get('mixedXXXX')
        self.assertEqual(focus, 'fix the recall bug',
                         "the watch-ignition envelope must not become the focus")

    def test_pure_watcher_focus_is_empty_not_notification(self):
        # No real prompt ever — only a heartbeat + a task-notification ignition.
        # Present (heartbeat), but focus clean-empty rather than the envelope.
        self._turn('pureWtch0', 'heartbeat', 'listening')
        self._turn('pureWtch0', 'user_message', '<task-notification>\n<event>x')
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        row = {r['session_id']: r for r in rows}.get('pureWtch0')
        self.assertIsNotNone(row, "heartbeat keeps the watcher present")
        self.assertEqual(row['focus'], '',
                         "a task-notification must not leak into a watcher's focus")

    def test_assistant_message_can_be_focus(self):
        # Tom (2026-06-05): focus is the latest CONVERSATIONAL turn — user OR
        # assistant — so a watcher's own last reply can be the focus. Here the
        # assistant turn is newer than the last user prompt, so it wins.
        self._turn('asstFocus', 'user_message', 'old user prompt',
                   updated_at=iso_cutoff(minutes=5))
        self._turn('asstFocus', 'assistant_message', 'shipped the TTL fix',
                   event_type='delta')   # assistant_message is an (s0, delta) trace
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        focus = {r['session_id']: r['focus'] for r in rows}.get('asstFocus')
        self.assertEqual(focus, 'shipped the TTL fix',
                         "latest conversational turn wins, even when it's the assistant")


if __name__ == '__main__':
    unittest.main()
