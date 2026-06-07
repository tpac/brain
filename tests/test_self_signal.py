"""Phase 2a self-channel signal — directed/broadcast courier (send + drain + reap)."""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.clock import iso_cutoff
from servers.scales.self_channel import signal, self_contract


class TestSelfSignal(BrainTestBase):
    needs_embedder = False

    def test_send_and_directed_consume_once(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body='hi B')
        first = signal.drain_inbox(self.brain, 'B')
        self.assertEqual([m['body'] for m in first], ['hi B'])
        self.assertEqual(signal.drain_inbox(self.brain, 'B'), [])  # consumed once

    def test_directed_not_delivered_to_others(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body='for B only')
        self.assertEqual(signal.drain_inbox(self.brain, 'C'), [])  # not for C
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'B')],
                         ['for B only'])

    def test_broadcast_fans_out(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='everyone')
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'B')],
                         ['everyone'])
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'C')],
                         ['everyone'])  # each live stream gets it once
        self.assertEqual(signal.drain_inbox(self.brain, 'B'), [])  # B already consumed

    def test_rendered_field_and_attribution(self):
        signal.send(self.brain, from_session='abcd1234ef',
                    address=self_contract.address_for_stream('B'), body='stop editing X')
        msg = signal.drain_inbox(self.brain, 'B')[0]
        self.assertIn('stop editing X', msg['rendered'])
        self.assertEqual(msg['from'], 'abcd1234')  # 8-char short

    def test_empty_body_rejected(self):
        with self.assertRaises(ValueError):
            signal.send(self.brain, from_session='A',
                        address=self_contract.address_for_stream('B'), body='   ')

    def test_expired_not_delivered_then_reaped(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body='stale')
        past = iso_cutoff(hours=1)   # expires_at in the past → already expired
        with self.brain.write_lock:
            self.brain.logs_conn.execute(
                "UPDATE self_inflight SET expires_at = ? WHERE address = ?",
                (past, self_contract.address_for_stream('B')))
            self.brain.logs_conn.commit()
        self.assertEqual(signal.drain_inbox(self.brain, 'B'), [])   # expired → not delivered
        self.assertEqual(signal.reap_expired(self.brain), 1)         # dead-letter swept

    def test_sender_does_not_hear_own_broadcast(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='heads up all')
        self.assertEqual(signal.drain_inbox(self.brain, 'A'), [])   # sender excluded from own broadcast
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'B')],
                         ['heads up all'])

    def test_reap_cleans_delivered_orphans(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='ephemeral')
        signal.drain_inbox(self.brain, 'B')  # B consumes → a self_delivered row now exists
        past = iso_cutoff(hours=1)
        with self.brain.write_lock:
            self.brain.logs_conn.execute("UPDATE self_inflight SET expires_at = ?", (past,))
            self.brain.logs_conn.commit()
        signal.reap_expired(self.brain)
        inflight = self.brain.logs_conn.execute("SELECT COUNT(*) FROM self_inflight").fetchone()[0]
        delivered = self.brain.logs_conn.execute("SELECT COUNT(*) FROM self_delivered").fetchone()[0]
        self.assertEqual(inflight, 0)
        self.assertEqual(delivered, 0, "orphan delivery rows must be swept with the message")

    def test_refs_persisted(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'),
                    body='see node', refs=['node-abc', 'file.py'])
        stored = self.brain.logs_conn.execute("SELECT refs FROM self_inflight").fetchone()[0]
        self.assertIn('node-abc', stored)

    # ── Truncation contract: SEND stores in full, no silent slice ──

    def test_send_stores_full_body_no_silent_cap(self):
        """The body is stored verbatim — no SIGNAL_BODY_MAX slice. Truncation,
        if any, happens only at delivery render, loudly."""
        long_body = "z" * 1500
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body=long_body)
        stored = self.brain.logs_conn.execute("SELECT body FROM self_inflight").fetchone()[0]
        self.assertEqual(len(stored), 1500)

    def test_send_stores_all_refs_no_silent_cap(self):
        """All refs persist — no REFS_MAX slice silently dropping the tether."""
        refs = ['n%d' % i for i in range(30)]
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body='many refs', refs=refs)
        stored = self.brain.logs_conn.execute("SELECT refs FROM self_inflight").fetchone()[0]
        self.assertIn('n29', stored)   # the 30th ref survived (would be gone at a cap of 12)

    def test_outbox_shows_delivery_status(self):
        """A sender can see who drained its message, and whether a directed target
        is still pending — silence read correctly (delivered-vs-never-delivered)."""
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body='ping B')
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='hey all')
        # Before anyone drains: directed message is pending, nobody delivered.
        ob = signal.outbox(self.brain, from_session='A')['messages']
        self.assertEqual(len(ob), 2)
        directed = [m for m in ob if m.get('target')][0]
        self.assertTrue(directed['pending'])
        self.assertEqual(directed['delivered_to'], [])
        # B drains → directed no longer pending; delivered_to names B.
        signal.drain_inbox(self.brain, to_session='B')
        directed2 = [m for m in signal.outbox(self.brain, from_session='A')['messages']
                     if m.get('target')][0]
        self.assertFalse(directed2['pending'])
        self.assertTrue(any(d['to'] == 'B' for d in directed2['delivered_to']))

    def test_outbox_empty_for_silent_stream(self):
        """A stream that never sent anything has an empty outbox (no crash)."""
        self.assertEqual(signal.outbox(self.brain, from_session='Z')['messages'], [])

    def test_drain_attributes_by_short_id(self):
        """No self-labeling — a delivered message is attributed by the sender's
        8-char short id, in both 'from' and the render."""
        signal.send(self.brain, from_session='AAAAAAAA',
                    address=self_contract.address_for_stream('streamB'), body='hi B')
        drained = signal.drain_inbox(self.brain, to_session='streamB')
        self.assertEqual(drained[0]['from'], 'AAAAAAAA')
        self.assertIn('other stream (id:AAAAAAAA) says:', drained[0]['rendered'])

    def test_first_contact_flag_set_once_per_sender(self):
        """drain flags the FIRST message from a sender as first_contact and
        carries the full from_session; later ones from the same sender are lean."""
        addr = self_contract.address_for_stream('rcpt1')
        signal.send(self.brain, from_session='senderXX', address=addr, body='one')
        d1 = signal.drain_inbox(self.brain, to_session='rcpt1')
        self.assertTrue(d1[0]['first_contact'])
        self.assertEqual(d1[0]['from_full'], 'senderXX')
        signal.send(self.brain, from_session='senderXX', address=addr, body='two')
        d2 = signal.drain_inbox(self.brain, to_session='rcpt1')
        self.assertFalse(d2[0]['first_contact'])

    def test_first_contact_intro_rendered_then_dropped(self):
        """drain_and_render attaches the sender's PEEK on the first message from a
        stream — intro carries the reply hint AND what they're working on — and
        nothing on the next. (Regression: the peek enrichment must actually wire.)"""
        self.brain.set_config('session_context_senderYY', 'refactoring the courier')
        addr = self_contract.address_for_stream('rcpt2')
        signal.send(self.brain, from_session='senderYY', address=addr, body='hello')
        block1, _ = signal.drain_and_render(self.brain, 'rcpt2')
        self.assertIn('first contact', block1)
        self.assertIn('self_send to="senderYY"', block1)
        self.assertIn('working on: refactoring the courier', block1)  # peek wired in
        signal.send(self.brain, from_session='senderYY', address=addr, body='again')
        block2, _ = signal.drain_and_render(self.brain, 'rcpt2')
        self.assertNotIn('first contact', block2)

    def test_resolve_short_matches_recent_sender(self):
        """A stream that messaged me but isn't in the live roster is still
        reply-able by its short id — its from_session is in the courier."""
        signal.send(self.brain, from_session='dorm12345678',
                    address=self_contract.address_for_stream('me'), body='hi')
        addr, err = signal.resolve_to(self.brain, 'dorm1234')
        self.assertIsNone(err)
        self.assertEqual(addr, self_contract.address_for_stream('dorm12345678'))

    # ── Phase 2b: render + delivery-into-Observation ──

    def test_render_received_block_composes(self):
        msgs = [
            {"body": "first tap", "from": "aaaa1111", "rendered": '⚡ aaaa1111 says:\n   "first tap"'},
            {"body": "second tap", "from": "bbbb2222", "rendered": '⚡ bbbb2222 says:\n   "second tap"'},
        ]
        block = self_contract.render_received_block(msgs)
        self.assertIn("first tap", block)
        self.assertIn("second tap", block)
        self.assertIn("stream", block.lower())  # framed with a header

    def test_render_received_block_empty(self):
        self.assertEqual(self_contract.render_received_block([]), "")

    def test_render_received_block_overflow_is_loud(self):
        """Over-budget input is bounded AND names the dropped count — no silent cut."""
        msgs = [{"body": "x" * 300, "from": "s%02d" % i,
                 "rendered": ('⚡ s%02d says:\n   "' % i) + "x" * 300 + '"'} for i in range(20)]
        block = self_contract.render_received_block(msgs, cap=800)
        self.assertLess(len(block), 800 + 200)   # bounded near the cap
        self.assertIn("more waiting", block)      # overflow is announced

    def test_render_caps_long_body_loudly(self):
        """A single over-long body is cut at DELIVERED_BODY_MAX with a LOUD inline
        marker — never silently — and the block stays bounded."""
        long_body = "y" * (self_contract.DELIVERED_BODY_MAX + 500)
        block = self_contract.render_received_block([{"body": long_body, "from": "s1"}])
        self.assertIn("full message in the dashboard", block)                 # loud marker
        self.assertNotIn("y" * (self_contract.DELIVERED_BODY_MAX + 1), block)  # body was cut
        self.assertLess(len(block), self_contract.DELIVERED_BODY_MAX + 300)    # bounded

    def test_drain_and_render_and_consumes(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body='ping B')
        block, n = signal.drain_and_render(self.brain, 'B')
        self.assertEqual(n, 1)
        self.assertIn('ping B', block)
        # consume-once: the second drain finds nothing
        block2, n2 = signal.drain_and_render(self.brain, 'B')
        self.assertEqual(n2, 0)
        self.assertEqual(block2, "")

    def test_drain_and_render_empty_inbox(self):
        block, n = signal.drain_and_render(self.brain, 'NOBODY-HOME')
        self.assertEqual(n, 0)
        self.assertEqual(block, "")


class TestPeekInbox(BrainTestBase):
    """Read-only peek — the /watch-live arrival detector. Must NOT consume."""
    needs_embedder = False

    def test_peek_returns_pending_without_consuming(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body='hi B')
        # Peek twice — same message both times, never consumed.
        self.assertEqual([m['body'] for m in signal.peek_inbox(self.brain, 'B')], ['hi B'])
        self.assertEqual([m['body'] for m in signal.peek_inbox(self.brain, 'B')], ['hi B'])
        # The real drain still delivers it — peek didn't mark it delivered.
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'B')], ['hi B'])
        # And after the drain consumes it, peek sees nothing.
        self.assertEqual(signal.peek_inbox(self.brain, 'B'), [])

    def test_peek_excludes_already_drained(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body='once')
        signal.drain_inbox(self.brain, 'B')                 # consumed
        self.assertEqual(signal.peek_inbox(self.brain, 'B'), [])  # peek agrees it's gone

    def test_peek_excludes_own_broadcast(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='to all')
        self.assertEqual(signal.peek_inbox(self.brain, 'A'), [])   # sender doesn't see own
        self.assertEqual([m['body'] for m in signal.peek_inbox(self.brain, 'B')], ['to all'])

    def test_peek_excludes_expired(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_stream('B'), body='stale')
        past = iso_cutoff(hours=1)
        with self.brain.write_lock:
            self.brain.logs_conn.execute(
                "UPDATE self_inflight SET expires_at = ? WHERE address = ?",
                (past, self_contract.address_for_stream('B')))
            self.brain.logs_conn.commit()
        self.assertEqual(signal.peek_inbox(self.brain, 'B'), [])

    def test_peek_empty_for_silent_stream(self):
        self.assertEqual(signal.peek_inbox(self.brain, 'NOBODY'), [])

    def test_peek_attributes_by_short_id(self):
        signal.send(self.brain, from_session='AAAAAAAA',
                    address=self_contract.address_for_stream('B'), body='hi')
        self.assertEqual(signal.peek_inbox(self.brain, 'B')[0]['from'], 'AAAAAAAA')


class TestTTLByCategory(BrainTestBase):
    """Per-message TTL resolved by address: broadcast ephemeral, directed waits
    a day; config-tunable per category."""
    needs_embedder = False

    def test_broadcast_gets_shorter_ttl_than_directed(self):
        b = signal.send(self.brain, from_session='X',
                        address=self_contract.ADDR_BROADCAST, body='who is live')
        d = signal.send(self.brain, from_session='X',
                        address=self_contract.address_for_stream('S'), body='for S')
        self.assertIn('expires_at', b)
        self.assertLess(b['expires_at'], d['expires_at'])  # broadcast expires sooner

    def test_unexpired_directed_still_delivers(self):
        signal.send(self.brain, from_session='X',
                    address=self_contract.address_for_stream('S'), body='fresh')
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'S')],
                         ['fresh'])

    def test_config_override_can_kill_broadcast_immediately(self):
        # 0h broadcast TTL → expires_at == send-time → any later drain excludes it.
        self.brain.set_config('self_channel.broadcast_ttl_hours', 0)
        signal.send(self.brain, from_session='X',
                    address=self_contract.ADDR_BROADCAST, body='instant-dead')
        self.assertEqual(signal.drain_inbox(self.brain, 'anyone'), [])

    def test_config_override_broadcast_does_not_touch_directed(self):
        # Shrinking broadcast TTL must not affect the directed default.
        self.brain.set_config('self_channel.broadcast_ttl_hours', 0)
        signal.send(self.brain, from_session='X',
                    address=self_contract.address_for_stream('S'), body='still here')
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'S')],
                         ['still here'])

    def test_nonnumeric_ttl_config_falls_back_not_crash(self):
        # A typo'd / non-numeric TTL config must NOT crash send() (get_config
        # returns the raw string when its numeric auto-parse fails) — _resolve_ttl_hours
        # falls back to the documented default and the message is still delivered.
        self.brain.set_config('self_channel.broadcast_ttl_hours', 'not-a-number')
        b = signal.send(self.brain, from_session='X',
                        address=self_contract.ADDR_BROADCAST, body='resilient')
        self.assertTrue(b['expires_at'])   # stamped via the default TTL, no crash
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'anyone')],
                         ['resilient'])


class TestLegacyNullExpires(BrainTestBase):
    """A pre-expires_at courier row (the column added to the one existing DB →
    expires_at NULL) is treated as dead: never delivered, and reaped on the next
    sweep. No backfill — the brain was never released, so there's no in-flight
    state worth preserving across the column add, and send() stamps expires_at
    on every new message."""
    needs_embedder = False

    def _insert_legacy(self, mid='m1', to='B', body='legacy'):
        from servers.clock import iso_now
        with self.brain.write_lock:
            self.brain.logs_conn.execute(
                "INSERT INTO self_inflight "
                "(id, from_session, address, body, refs, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (mid, 'A', self_contract.address_for_stream(to), body, '', iso_now()))
            self.brain.logs_conn.commit()

    def test_null_expires_not_delivered(self):
        self._insert_legacy()
        self.assertEqual(signal.drain_inbox(self.brain, 'B'), [])
        self.assertEqual(signal.peek_inbox(self.brain, 'B'), [])

    def test_null_expires_reaped_as_dead(self):
        self._insert_legacy()
        self.assertEqual(signal.reap_expired(self.brain), 1)
        remaining = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM self_inflight").fetchone()[0]
        self.assertEqual(remaining, 0)


if __name__ == '__main__':
    unittest.main()
