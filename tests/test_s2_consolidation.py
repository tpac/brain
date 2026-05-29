"""Tests for the S2 consolidation decoder's scan gate + stamp timing.

Consolidation's expensive step is the embedding scan. The gate (2026-05-29)
turns a full O(graph) scan every idle cycle into: one cold-start that covers
the existing backlog, then incremental scans of only changed nodes, skipping
entirely when nothing changed. A similarity-threshold change forces a fresh
cold-start.

Stamp timing: the decoder captures the baseline (`_stamp`) but does NOT write
it. The orchestrator advances the last-run cutoff only AFTER a run fully
completes — never on a skip or a mid-run encoder failure — so failed work is
retried, not skipped past.

The incremental scan's no-miss property is guaranteed by construction — it
compares every changed node against ALL nodes (changed @ all.T). The
fingerprint suppression itself is covered by test_consolidation_fingerprint.py.
"""

import time
import unittest
import sys
import os
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestConsolidationScanGate(BrainTestBase):
    """Decoder-level: mode selection, change detection, skip. No encoder."""

    needs_embedder = False

    def _decoder(self):
        from servers.scales.s2.consolidation_decoder import ConsolidationDecoder
        return ConsolidationDecoder(self.brain)

    def _mark_scanned_now(self, d):
        """Simulate 'we just scanned' — stamp last-run ts + current threshold."""
        time.sleep(0.02)
        self.brain.set_config(d.LAST_RUN_TS_KEY, str(time.time()))
        self.brain.set_config(d.LAST_THRESHOLD_KEY, str(d.config['similarity_threshold']))
        time.sleep(0.02)

    def test_cold_start_scans_and_defers_stamp(self):
        # No recorded timestamp → cold start → it scans (not skipped) and hands
        # a baseline up for the orchestrator to stamp. The decoder itself must
        # NOT write the cutoff. (No embeddings present → empty scan; we assert
        # control flow, not pair discovery.)
        d = self._decoder()
        self.brain.remember(type='fact', title='a', content='c')
        result = d.run()
        self.assertIsNone(result.get('skipped'))
        self.assertIsNotNone(result.get('_stamp'))
        self.assertIsNone(self.brain.get_config(d.LAST_RUN_TS_KEY))  # decoder didn't stamp

    def test_skips_when_nothing_changed(self):
        d = self._decoder()
        self.brain.remember(type='fact', title='a', content='c')
        self._mark_scanned_now(d)
        result = d.run()
        self.assertEqual(result.get('skipped'), 'no graph change')
        self.assertIsNone(result.get('_stamp'))  # nothing to stamp on a skip

    def test_new_node_triggers_scan(self):
        d = self._decoder()
        self.brain.remember(type='fact', title='a', content='c')
        self._mark_scanned_now(d)
        self.brain.remember(type='fact', title='b', content='c')  # after cutoff
        result = d.run()
        self.assertNotEqual(result.get('skipped'), 'no graph change')

    def test_threshold_change_forces_cold_start(self):
        d = self._decoder()
        self.brain.remember(type='fact', title='a', content='c')
        time.sleep(0.02)
        self.brain.set_config(d.LAST_RUN_TS_KEY, str(time.time()))
        self.brain.set_config(d.LAST_THRESHOLD_KEY, '0.99')  # differs from config default
        time.sleep(0.02)
        # No new nodes, but the threshold changed → must re-scan, not skip.
        result = d.run()
        self.assertNotEqual(result.get('skipped'), 'no graph change')

    def test_changed_ids_use_timestamps_not_just_s1e_traces(self):
        # MCP-created nodes carry no encoding_run trace; timestamp-based
        # detection must still catch them (the bug in the old trace-based impl).
        d = self._decoder()
        self.brain.remember(type='fact', title='old', content='c')
        time.sleep(0.02)
        cutoff = datetime.now(timezone.utc).isoformat()
        time.sleep(0.02)
        new_id = self.brain.remember(type='fact', title='new', content='c')['id']
        changed = d._get_changed_node_ids(cutoff)
        self.assertIn(new_id, changed)

    def test_changed_ids_exclude_community_nodes(self):
        d = self._decoder()
        time.sleep(0.02)
        cutoff = datetime.now(timezone.utc).isoformat()
        time.sleep(0.02)
        comm_id = self.brain.remember(type='community', title='comm', content='c')['id']
        fact_id = self.brain.remember(type='fact', title='f', content='c')['id']
        changed = d._get_changed_node_ids(cutoff)
        self.assertIn(fact_id, changed)
        self.assertNotIn(comm_id, changed)  # consolidation scans non-community only


class TestConsolidationStampTiming(BrainTestBase):
    """Orchestrator-level: the cutoff advances ONLY after a run completes."""

    needs_embedder = False

    def _unit(self):
        from servers.scales.s2.consolidation import Consolidation
        return Consolidation(self.brain)

    def test_completed_empty_scan_advances_cutoff(self):
        # Empty brain (no embeddings) → decode cold-start → no candidates →
        # the run completed, so the orchestrator stamps. No encoder involved.
        u = self._unit()
        result = u.run()
        self.assertIsNone(result.get('skipped'))
        self.assertIsNotNone(self.brain.get_config(u.LAST_RUN_TS_KEY))

    def test_skip_does_not_advance_cutoff(self):
        u = self._unit()
        self.brain.remember(type='fact', title='a', content='c')
        time.sleep(0.02)
        ts = str(time.time())
        self.brain.set_config(u.LAST_RUN_TS_KEY, ts)
        self.brain.set_config(u.LAST_THRESHOLD_KEY, str(u.config['similarity_threshold']))
        time.sleep(0.02)
        result = u.run()
        self.assertEqual(result.get('skipped'), 'no graph change')
        self.assertEqual(self.brain.get_config(u.LAST_RUN_TS_KEY), ts)  # unchanged

    def test_encoder_failure_does_not_advance_cutoff(self):
        # The crux of the fix: if the encoder fails mid-run, the cutoff must NOT
        # advance — the work is retried next cycle, not skipped past.
        from servers.scales.s2 import consolidation as consol_mod
        u = self._unit()
        fake_decode = {
            'clusters': [{'nodes': ['n1', 'n2'],
                          'node_details': {'n1': {'updated_at': ''},
                                           'n2': {'updated_at': ''}},
                          'pre_class': 'needs_judgment'}],
            'stats': {},
            '_stamp': {'ts': 12345.0, 'threshold': '0.89'},
        }
        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=fake_decode), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               return_value=None):
            result = u.run()
        self.assertEqual(result.get('error'), 'encoding failed')
        self.assertIsNone(self.brain.get_config(u.LAST_RUN_TS_KEY))  # NOT stamped


if __name__ == '__main__':
    unittest.main()
