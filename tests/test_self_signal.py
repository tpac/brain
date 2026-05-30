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
        old = iso_cutoff(hours=self_contract.DEFAULT_SIGNAL_TTL_HOURS + 1)
        with self.brain.write_lock:
            self.brain.logs_conn.execute(
                "UPDATE self_inflight SET created_at = ? WHERE address = ?",
                (old, self_contract.address_for_stream('B')))
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
        old = iso_cutoff(hours=self_contract.DEFAULT_SIGNAL_TTL_HOURS + 1)
        with self.brain.write_lock:
            self.brain.logs_conn.execute("UPDATE self_inflight SET created_at = ?", (old,))
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


if __name__ == '__main__':
    unittest.main()
