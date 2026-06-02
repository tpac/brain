"""Invalid brain_batch op → retry next cycle, never a silent SKIP-suppression.

When an S2 encoder emits a concept-verb (`absorb` / `reject` / `keep` / `skip`)
as a brain_batch op NAME instead of translating it into one of the 5 real ops,
dispatch drops the op and logs `brain_batch_invalid_op`. The encoder *tried* to
act and was thwarted — so the proposal/cluster must be RETRIED next cycle, never
stamped as a clean SKIP rejection. Stamping would both abandon the intended
action (a merge / a drift-threshold raise) AND suppress the proposal until a
member node's `updated_at` changes — silent capability loss.

These tests lock the detection helper + the matcher/detector composition that
`community_encoder` and `consolidation` use to exclude failed attempts.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.contract import VALID_BATCH_OPS
from servers.scales.s2.rejection_table import (
    node_ids_touched_by_invalid_ops,
    get_proposed_ids,
    match_proposals_to_actions,
)


def _batch(*ops):
    """Shape an action_details entry the way runner.py records a brain_batch call."""
    return [{'tool': 'brain_batch', 'input': {'operations': list(ops)}}]


class TestDetector:
    def test_invalid_op_node_id_collected(self):
        # `consolidate` is a concept-verb, NOT a valid brain_batch op — dispatch
        # drops it. Its node id is collected so the cluster is retried, not
        # stamped. (`absorb` used to live here; it is now a VALID op — see
        # test_valid_absorb_op_not_collected.)
        ad = _batch({'op': 'consolidate', 'node_id': 'aaaa1111', 'reason': 'merge'})
        assert node_ids_touched_by_invalid_ops(ad) == {'aaaa1111'}

    def test_valid_absorb_op_not_collected(self):
        # absorb shipped into VALID_BATCH_OPS — a successful absorb is a real
        # merge, never a thwarted attempt, so the detector must NOT flag it.
        ad = _batch({'op': 'absorb', 'survivor_id': 'aaaa1111',
                     'absorbed_id': 'bbbb2222', 'reason': 'merge'})
        assert node_ids_touched_by_invalid_ops(ad) == set()

    def test_valid_ops_return_empty(self):
        ad = _batch(
            {'op': 'revise', 'node_id': 'aaaa1111'},
            {'op': 'archive', 'node_id': 'bbbb2222'},
        )
        assert node_ids_touched_by_invalid_ops(ad) == set()

    def test_mixed_collects_only_invalid(self):
        ad = _batch(
            {'op': 'revise', 'node_id': 'good0001'},
            {'op': 'reject', 'node_id': 'bad00001', 'reason': 'drift'},
        )
        assert node_ids_touched_by_invalid_ops(ad) == {'bad00001'}

    def test_collects_edge_and_member_ids(self):
        ad = _batch(
            {'op': 'keep', 'source_id': 's1', 'target_id': 't1'},
            {'op': 'skip', 'members': ['m1', 'm2']},
        )
        assert node_ids_touched_by_invalid_ops(ad) == {'s1', 't1', 'm1', 'm2'}

    def test_ignores_non_brain_batch_tools(self):
        ad = [{'tool': 'get_nodes',
               'input': {'operations': [{'op': 'reject', 'node_id': 'x'}]}}]
        assert node_ids_touched_by_invalid_ops(ad) == set()

    def test_tolerates_malformed_entries(self):
        ad = [
            'garbage',
            {'tool': 'brain_batch', 'input': {'operations': ['not-a-dict', {}]}},
        ]
        assert node_ids_touched_by_invalid_ops(ad) == set()

    def test_empty_and_none(self):
        assert node_ids_touched_by_invalid_ops([]) == set()
        assert node_ids_touched_by_invalid_ops(None) == set()


class TestDriftRejectIsRetriedNotSuppressed:
    """The exact case from the error log: `op: reject` on a drift proposal."""

    def _drift(self):
        return {'type': 'drift', 'node_id': 'ff18ae35',
                'foreign': [{'id': 'bc3780cd'}]}

    def test_matcher_alone_marks_invalid_reject_as_skipped(self):
        # The matcher only counts a drift-reject when op==revise carries
        # _sys_drift_threshold — so an `op: reject` lands in skipped. Without
        # the fix this is what gets a suppression fingerprint.
        p = self._drift()
        ad = _batch({'op': 'reject', 'node_id': 'ff18ae35',
                     'reason': 'DRIFT rejected'})
        acted, skipped = match_proposals_to_actions([p], ad)
        assert acted == []
        assert skipped == [p]

    def test_invalid_reject_node_is_pulled_out_of_suppression(self):
        # The fix: detector flags ff18ae35, so the encoder's exclusion filter
        # moves the proposal to retry instead of record_rejections.
        p = self._drift()
        ad = _batch({'op': 'reject', 'node_id': 'ff18ae35',
                     'reason': 'DRIFT rejected'})
        _, skipped = match_proposals_to_actions([p], ad)
        touched = node_ids_touched_by_invalid_ops(ad)
        retry = [q for q in skipped if set(get_proposed_ids(q)) & touched]
        to_suppress = [q for q in skipped
                       if not (set(get_proposed_ids(q)) & touched)]
        assert retry == [p]          # retried next cycle
        assert to_suppress == []     # nothing stamped → no silent abandonment

    def test_valid_drift_reject_is_acted_on_no_retry(self):
        # Positive control: the CORRECT translation (revise + _sys_drift_threshold)
        # is recognized as acted-on and triggers no invalid-op retry.
        p = self._drift()
        ad = _batch({'op': 'revise', 'node_id': 'ff18ae35',
                     '_sys_drift_threshold': '0.6'})
        acted, skipped = match_proposals_to_actions([p], ad)
        assert acted == [p]
        assert skipped == []
        assert node_ids_touched_by_invalid_ops(ad) == set()


class TestConsolidationAbsorbIsRealMerge:
    def test_valid_absorb_cluster_not_pulled_into_retry(self):
        # A consolidation cluster merged via the VALID `absorb` op is a real
        # merge — the absorbed node gets archived. It must NOT land in the
        # invalid-op retry path (that is only for concept-verbs dispatch drops).
        # The orchestrator detects the successful merge via the archived member,
        # not via this helper — see test_s2_consolidation.py.
        ad = _batch({'op': 'absorb', 'survivor_id': '7463a2aa',
                     'absorbed_id': 'c3571f66', 'content': 'merged'})
        members = ['7463a2aa', 'c3571f66']
        assert set(members) & node_ids_touched_by_invalid_ops(ad) == set()


class TestContractSync:
    def test_valid_batch_ops_is_the_closed_six(self):
        # brain_mcp's enum is built from this set and the dispatcher's if/elif
        # must match it. Lock the membership so a new op can't silently diverge.
        # absorb (lossless merge) joined the closed set — see S2-ABSORB-OP-DESIGN.md.
        assert set(VALID_BATCH_OPS) == {
            'remember', 'revise', 'connect', 'disconnect', 'archive', 'absorb'}
