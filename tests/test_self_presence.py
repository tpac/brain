"""Phase 1 self-channel presence — the PULL primitives.

Covers brain.present_streams (wall-clock roster), presence.build_presence
(roster + rendered line), and presence.peek (drill into one stream). Pure
reads; no embedder needed.
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.clock import iso_cutoff
from servers.session_context import SessionContext
from servers.scales.self_channel import presence, self_contract, signal


class TestSelfPresence(BrainTestBase):
    needs_embedder = False

    def _save_stream(self, sid, focus='', message_count=3, updated_at=None):
        """Persist a session_state row for `sid`, optionally backdated."""
        ctx = SessionContext(session_id=sid)
        ctx.message_count = message_count
        ctx.save(self.brain.logs_conn)
        if updated_at is not None:
            self.brain.logs_conn.execute(
                "UPDATE session_state SET updated_at = ? "
                "WHERE session_id = ? AND key = '_session_context'",
                (updated_at, sid))
            self.brain.logs_conn.commit()
        if focus:
            self.brain.set_config('session_context_' + sid, focus)

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
        self._save_stream('streamAAAA', focus='line one\nline two')

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


if __name__ == '__main__':
    unittest.main()
