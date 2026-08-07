"""query_traces saturated-limit flag — the loud-by-default truncation guard.

A bounded trace query that hits its limit covers only the most recent slice
of the requested window while looking complete. The 2026-08-06 cost tally
aggregated 'a week' of deltas that were actually 2 days (limit=5000 saturated
by S0 tool_result volume) — plausible numbers, wrong window. query_traces now
fetches limit+1 and attaches a 'truncated' payload when the extra row proves
the window holds more than the result carries.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


def _events(n, session_id='trunctest-session'):
    return [{
        'chain_id': f's0-trunc-{i}',
        'scale': 's0',
        'event_type': 'delta',
        'ref_type': 'tool_result',
        'summary': f'event {i}',
        'metadata': {'i': i},
        'session_id': session_id,
    } for i in range(n)]


class TestQueryTracesTruncation(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.brain._trace_dal.append_batch(_events(10))

    def test_saturated_limit_carries_truncated_payload(self):
        res = self.brain.query_traces(event_type='delta', hours=24, limit=4)
        self.assertEqual(len(res['events']), 4)
        self.assertIn('truncated', res)
        t = res['truncated']
        self.assertEqual(t['limit'], 4)
        # coverage_start is the oldest row actually returned
        self.assertEqual(t['coverage_start'],
                         min(e['created_at'] for e in res['events']))
        self.assertIn('limit=4', t['note'])

    def test_unsaturated_limit_has_no_truncated_key(self):
        res = self.brain.query_traces(event_type='delta', hours=24, limit=50)
        self.assertEqual(len(res['events']), 10)
        self.assertNotIn('truncated', res)

    def test_exact_fit_is_not_truncated(self):
        """limit == row count: the +1 probe proves exhaustion, no false alarm."""
        res = self.brain.query_traces(event_type='delta', hours=24, limit=10)
        self.assertEqual(len(res['events']), 10)
        self.assertNotIn('truncated', res)

    def test_ref_type_branch_flags_too(self):
        res = self.brain.query_traces(ref_type='tool_result', hours=24, limit=3)
        self.assertEqual(len(res['events']), 3)
        self.assertIn('truncated', res)


if __name__ == '__main__':
    unittest.main()
