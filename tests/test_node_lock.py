"""set_node_lock — the one door for lock flips on existing nodes.

Contract under test:
  - Two-phase confirm: phase 1 executes nothing, returns a one-shot token +
    summary; phase 2 with the token flips the flag.
  - Guards: unknown node, locking an archived node, no-op (already in state),
    invalid/expired/mismatched token, token is one-shot.
  - revise() immutability is UNCHANGED — locked stays skipped there.
  - Dispatch roundtrip emits a node_revised trace with the locked delta.
  - Guardrail: the shared S1/S2 encoder dispatch closure refuses the command.
"""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.brain_test_base import BrainTestBase


def _locked_in_db(brain, node_id):
    row = brain.conn.execute(
        'SELECT locked FROM nodes WHERE id = ?', (node_id,)).fetchone()
    return bool(row[0])


def _make_node(brain, **kwargs):
    defaults = {'type': 'concept',
                'title': 'Lock test node %d' % int(time.time() * 1000),
                'content': 'content'}
    defaults.update(kwargs)
    return brain.remember(**defaults)['id']


class TestSetNodeLock(BrainTestBase):
    needs_embedder = False

    def _confirm_flip(self, node_id, locked, reason='test'):
        """Run both phases; returns the phase-2 result."""
        p1 = self.brain.set_node_lock(node_id, locked, reason=reason)
        self.assertTrue(p1.get('confirmation_required'))
        return self.brain.set_node_lock(node_id, locked, reason=reason,
                                        confirm_token=p1['confirm_token'])

    def test_phase1_returns_token_and_flips_nothing(self):
        nid = _make_node(self.brain)
        res = self.brain.set_node_lock(nid, True, reason='canonical principle')
        self.assertTrue(res['ok'])
        self.assertTrue(res['confirmation_required'])
        self.assertTrue(res['confirm_token'].startswith('lock-'))
        self.assertIn('needs an explicit yes', res['summary'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_phase2_flips_and_reports_delta(self):
        nid = _make_node(self.brain)
        res = self._confirm_flip(nid, True)
        self.assertTrue(res['ok'])
        self.assertTrue(res['changed'])
        self.assertTrue(_locked_in_db(self.brain, nid))
        self.assertEqual(res['deltas'],
                         [{'field': 'locked', 'old': False, 'new': True}])
        self.assertIn('confirm_latency_s', res)

    def test_unlock_roundtrip(self):
        nid = _make_node(self.brain, locked=True, encoding_source='anchor')
        self.assertTrue(_locked_in_db(self.brain, nid))
        res = self._confirm_flip(nid, False)
        self.assertTrue(res['changed'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_noop_needs_no_confirmation(self):
        nid = _make_node(self.brain)
        res = self.brain.set_node_lock(nid, False, reason='already unlocked')
        self.assertTrue(res['ok'])
        self.assertFalse(res.get('confirmation_required', False))
        self.assertFalse(res['changed'])
        self.assertIn('already unlocked', res['note'])

    def test_unknown_node(self):
        res = self.brain.set_node_lock('ffffffff', True, reason='x')
        self.assertFalse(res['ok'])
        self.assertIn('not found', res['error'].lower())

    def test_cannot_lock_archived_node(self):
        nid = _make_node(self.brain)
        self.brain.archive_node(nid, archived_by='anchor', reason='test')
        res = self.brain.set_node_lock(nid, True, reason='x')
        self.assertFalse(res['ok'])
        self.assertIn('archived', res['error'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_token_is_one_shot(self):
        nid = _make_node(self.brain)
        p1 = self.brain.set_node_lock(nid, True, reason='x')
        tok = p1['confirm_token']
        first = self.brain.set_node_lock(nid, True, reason='x', confirm_token=tok)
        self.assertTrue(first['changed'])
        # unlock so the state allows a second attempt, then replay the token
        self._confirm_flip(nid, False)
        replay = self.brain.set_node_lock(nid, True, reason='x', confirm_token=tok)
        self.assertFalse(replay['ok'])
        self.assertIn('invalid or expired', replay['error'])

    def test_token_bound_to_node_and_direction(self):
        nid_a = _make_node(self.brain)
        nid_b = _make_node(self.brain)
        tok = self.brain.set_node_lock(nid_a, True, reason='x')['confirm_token']
        res = self.brain.set_node_lock(nid_b, True, reason='x', confirm_token=tok)
        self.assertFalse(res['ok'])
        self.assertFalse(_locked_in_db(self.brain, nid_b))

    def test_expired_token_rejected(self):
        nid = _make_node(self.brain)
        tok = self.brain.set_node_lock(nid, True, reason='x')['confirm_token']
        self.brain._pending_lock_confirms[tok]['requested_at'] -= (
            self.brain.LOCK_CONFIRM_TTL_S + 1)
        res = self.brain.set_node_lock(nid, True, reason='x', confirm_token=tok)
        self.assertFalse(res['ok'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_revise_still_refuses_locked_field(self):
        """The new door must not relax revise() immutability."""
        nid = _make_node(self.brain)
        res = self.brain.revise(node_id=nid, reason='attempt lock via revise',
                                updates={'locked': True})
        self.assertFalse(_locked_in_db(self.brain, nid))
        self.assertTrue(any('immutable' in w for w in res.get('warnings', [])))


class TestSetNodeLockDispatch(BrainTestBase):
    """The dispatch door: arg contract + the node_revised trace on a flip."""
    needs_embedder = False

    def _dispatch(self, args):
        from servers.daemon_dispatch import dispatch_command
        return dispatch_command(self.brain, 'set_node_lock', args, [])

    def test_requires_node_id_locked_and_reason(self):
        nid = _make_node(self.brain)
        self.assertFalse(self._dispatch({'locked': True, 'reason': 'x'})['ok'])
        self.assertFalse(self._dispatch({'node_id': nid, 'reason': 'x'})['ok'])
        self.assertFalse(self._dispatch({'node_id': nid, 'locked': True})['ok'])

    def test_flip_emits_node_revised_trace(self):
        nid = _make_node(self.brain)
        p1 = self._dispatch({'node_id': nid, 'locked': True, 'reason': 'canon'})
        self.assertTrue(p1['ok'])
        tok = p1['result']['confirm_token']
        p2 = self._dispatch({'node_id': nid, 'locked': True, 'reason': 'canon',
                             'confirm_token': tok})
        self.assertTrue(p2['ok'])
        self.assertTrue(_locked_in_db(self.brain, nid))
        rows = self.brain._trace_dal.conn.execute(
            "SELECT metadata FROM trace_events "
            "WHERE ref_type = 'node_revised' AND ref_id = ?", (nid,)).fetchall()
        self.assertEqual(len(rows), 1)
        meta = json.loads(rows[0][0])
        self.assertEqual(meta['deltas'],
                         [{'field': 'locked', 'old': False, 'new': True}])
        self.assertIn('canon', meta['reason'])

    def test_phase1_emits_no_trace(self):
        nid = _make_node(self.brain)
        self._dispatch({'node_id': nid, 'locked': True, 'reason': 'canon'})
        rows = self.brain._trace_dal.conn.execute(
            "SELECT 1 FROM trace_events "
            "WHERE ref_type = 'node_revised' AND ref_id = ?", (nid,)).fetchall()
        self.assertEqual(rows, [])


class TestEncoderClosureRefusesLock(unittest.TestCase):
    """Guardrail: the shared S1/S2 encoder dispatch refuses set_node_lock."""

    def test_encoder_dispatch_refuses(self):
        import threading
        from servers.scales.s2.base import IntegrationUnit

        errors = []

        class FakeBrain:
            write_lock = threading.RLock()

            def _log_error(self, kind, exc, ctx=''):
                errors.append(kind)

        class TestUnit(IntegrationUnit):
            NAME = 'test_lock_refusal'
            SCALE = 's2'
            ENCODING_SOURCE = 's2:test_lock_refusal'

        dispatch = TestUnit(brain=FakeBrain())._make_encoder_dispatch()
        res = dispatch('set_node_lock',
                       {'node_id': 'deadbeef', 'locked': True, 'reason': 'x'})
        self.assertFalse(res['ok'])
        self.assertIn('operator-channel only', res['error'])
        self.assertEqual(errors, ['s2_test_lock_refusal_lock_refused'])


if __name__ == '__main__':
    unittest.main()
