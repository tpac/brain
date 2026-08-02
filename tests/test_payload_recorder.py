"""Tests for the payload recorder (docs/TRACE-MODES-DESIGN.md, rollout step 1).

brain.record_payload / brain.read_payload — the ONE capture writer/reader.
Locks: db_dir-derived root (IsolatedBrain by construction), per-kind K-store
gating with modes as config versions, append-only chain dirs (O_EXCL +
attempt ordinals), pointer safety, pruned-read semantics, unknown-kind
loudness, midnight chain-dir reuse.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestPayloadRecorder(BrainTestBase):
    needs_embedder = False

    def _abs(self, pointer):
        return os.path.join(self.tmp, pointer)

    # ── write + read ────────────────────────────────────────────

    def test_roundtrip_prompt(self):
        ptr = self.brain.record_payload('s1e-abc-3', 'prompt', '# The prompt')
        self.assertIsNotNone(ptr)
        self.assertFalse(os.path.isabs(ptr))
        self.assertTrue(ptr.startswith('payloads' + os.sep))
        self.assertTrue(ptr.endswith('000-prompt.md'))
        self.assertIn('s1e-abc-3', ptr)
        # File lives inside THIS brain's db_dir (IsolatedBrain by construction)
        self.assertTrue(os.path.exists(self._abs(ptr)))
        self.assertEqual(self.brain.read_payload(ptr), '# The prompt')

    def test_dict_content_serialized_as_json(self):
        ptr = self.brain.record_payload(
            's1e-abc-4', 'failed_run', {'messages': [{'role': 'user'}]},
            seq=2)
        self.assertTrue(ptr.endswith('002-failed_run.json'))
        self.assertIn('"messages"', self.brain.read_payload(ptr))

    def test_empty_content_records_nothing(self):
        for empty in (None, '', {}, []):
            self.assertIsNone(
                self.brain.record_payload('s1e-abc-5', 'prompt', empty))

    # ── gating ──────────────────────────────────────────────────

    def test_normal_config_gates_round_payload_off(self):
        self.assertIsNone(
            self.brain.record_payload('s1e-abc-6', 'round_payload', 'x'))

    def test_debug_version_flip_opens_round_payload(self):
        """Modes are config versions: v2 (debug) is seeded dormant;
        set_interaction_active is the whole switch — no restart, no env."""
        self.brain.set_interaction_active('trace_recording', 2)
        ptr = self.brain.record_payload('s1e-abc-7', 'round_payload',
                                        {'round': 0})
        self.assertIsNotNone(ptr)
        # Flip back: gated again.
        self.brain.set_interaction_active('trace_recording', 1)
        self.assertIsNone(
            self.brain.record_payload('s1e-abc-7', 'round_payload',
                                      {'round': 1}, seq=1))

    def test_unknown_kind_returns_none_and_logs_loud_once(self):
        self.assertIsNone(
            self.brain.record_payload('s1e-abc-8', 'promt-typo', 'x'))
        self.assertIsNone(
            self.brain.record_payload('s1e-abc-8', 'promt-typo', 'x'))
        errs = [e for e in self.brain.get_recent_errors(hours=1, limit=50)
                if e.get('source') == 'record_payload_unknown_kind']
        self.assertEqual(len(errs), 1)

    # ── append-only ─────────────────────────────────────────────

    def test_collision_gets_attempt_ordinal_never_overwrites(self):
        p1 = self.brain.record_payload('s1e-abc-9', 'prompt', 'attempt one')
        p2 = self.brain.record_payload('s1e-abc-9', 'prompt', 'attempt two')
        self.assertNotEqual(p1, p2)
        self.assertTrue(p2.endswith('000-prompt.2.md'))
        self.assertEqual(self.brain.read_payload(p1), 'attempt one')
        self.assertEqual(self.brain.read_payload(p2), 'attempt two')

    # ── pointer safety + pruned reads ───────────────────────────

    def test_read_rejects_absolute_and_traversal(self):
        secret = os.path.join(self.tmp, 'secret.txt')
        with open(secret, 'w') as f:
            f.write('nope')
        self.assertIsNone(self.brain.read_payload(secret))
        self.assertIsNone(self.brain.read_payload('../secret.txt'))
        self.assertIsNone(
            self.brain.read_payload('payloads/../secret.txt'))
        self.assertIsNone(self.brain.read_payload(''))
        self.assertIsNone(self.brain.read_payload(None))

    def test_deleted_file_reads_as_pruned_none(self):
        ptr = self.brain.record_payload('s1e-abc-10', 'prompt', 'gone soon')
        os.remove(self._abs(ptr))
        self.assertIsNone(self.brain.read_payload(ptr))

    # ── midnight reuse ──────────────────────────────────────────

    def test_existing_chain_dir_under_yesterday_is_reused(self):
        from datetime import datetime, timedelta
        from servers.clock import iso_now
        yesterday = (datetime.strptime(iso_now()[:10], '%Y-%m-%d')
                     - timedelta(days=1)).strftime('%Y-%m-%d')
        pre = os.path.join(self.tmp, 'payloads', yesterday, 's1e-abc-11')
        os.makedirs(pre)
        ptr = self.brain.record_payload('s1e-abc-11', 'prompt', 'late run')
        self.assertIn(yesterday, ptr)
        self.assertEqual(self.brain.read_payload(ptr), 'late run')

    # ── retention ───────────────────────────────────────────────

    def test_prune_removes_expired_date_dirs_only(self):
        old = os.path.join(self.tmp, 'payloads', '2026-01-01', 's1e-old-1')
        os.makedirs(old)
        with open(os.path.join(old, '000-prompt.md'), 'w') as f:
            f.write('ancient')
        keep_ptr = self.brain.record_payload('s1e-abc-13', 'prompt', 'fresh')
        removed = self.brain.prune_payloads_if_due()
        self.assertEqual(removed, 1)
        self.assertFalse(os.path.exists(old))
        self.assertEqual(self.brain.read_payload(keep_ptr), 'fresh')
        # Daily stamp: a second call the same day is a no-op.
        os.makedirs(os.path.join(self.tmp, 'payloads', '2026-01-02'))
        self.assertEqual(self.brain.prune_payloads_if_due(), 0)

    # ── recorder never raises ───────────────────────────────────

    def test_failure_is_loud_not_raised(self):
        # Make the payload root unwritable-as-a-dir by occupying its path
        # with a FILE — os.makedirs inside record_payload must fail.
        with open(os.path.join(self.tmp, 'payloads'), 'w') as f:
            f.write('blocker')
        ptr = self.brain.record_payload('s1e-abc-12', 'prompt', 'x')
        self.assertIsNone(ptr)
        errs = [e for e in self.brain.get_recent_errors(hours=1, limit=50)
                if e.get('source') == 'record_payload']
        self.assertEqual(len(errs), 1)


if __name__ == '__main__':
    unittest.main()
