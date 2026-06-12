"""BATCH_OP_SPECS contract — the brain_batch discriminated-op single source.

Three sites derive from contract.BATCH_OP_SPECS and must not drift:
  1. VALID_BATCH_OPS (membership checks in dispatch + S2 rejection table)
  2. brain_mcp's brain_batch inputSchema (oneOf branch per op — generation-
     time signal; probe-validated 2026-06-12, eval/mcp_variants/probe_v2_*)
  3. dispatch_write._handle_brain_batch's per-op required pre-check
     (dispatch-time backstop)

These tests pin each derivation plus the revise reason/reasoning
disambiguation surviving through the batch path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.contract import BATCH_OP_SPECS, VALID_BATCH_OPS
from tests.brain_test_base import BrainTestBase


def test_valid_ops_derive_from_specs():
    assert VALID_BATCH_OPS == frozenset(BATCH_OP_SPECS)


def test_mcp_schema_derives_from_specs():
    """One oneOf branch per op, in dict order, with const discriminator and
    required = ['op'] + spec's required list."""
    from servers import brain_mcp
    tool = next(t for t in brain_mcp.TOOLS if t['name'] == 'brain_batch')
    items = tool['inputSchema']['properties']['operations']['items']
    branches = items['oneOf']
    assert [b['properties']['op']['const'] for b in branches] == \
        list(BATCH_OP_SPECS)
    for branch, (op, spec) in zip(branches, BATCH_OP_SPECS.items()):
        assert branch['required'] == ['op'] + spec['required'], op
        # Every contract property fragment is present in the branch
        for prop in spec['properties']:
            assert prop in branch['properties'], '%s.%s' % (op, prop)
        # additionalProperties stays open — remember/revise/absorb accept
        # open-ended node fields by design
        assert 'additionalProperties' not in branch, op


def test_specs_required_fields_are_declared_properties():
    """Every required field must have a property fragment (the schema can't
    require a field it doesn't describe)."""
    for op, spec in BATCH_OP_SPECS.items():
        for field in spec['required']:
            assert field in spec['properties'], '%s.%s' % (op, field)


class TestDispatchEnforcesRequired(BrainTestBase):
    needs_embedder = False

    # Minimal valid op payloads (ids are fake — the pre-check fires before
    # any handler touches the DB, so broken variants never write).
    VALID = {
        'remember': {'type': 'concept', 'title': 'T', 'content': 'C'},
        'revise': {'node_id': 'aaaabbbb', 'reason': 'r'},
        'connect': {'source_id': 'aaaabbbb', 'target_id': 'ccccdddd'},
        'disconnect': {'source_id': 'aaaabbbb', 'target_id': 'ccccdddd',
                       'relation': 'extends'},
        'archive': {'node_id': 'aaaabbbb'},
        'absorb': {'survivor_id': 'aaaabbbb', 'absorbed_id': 'ccccdddd'},
    }

    def test_each_missing_required_field_errors(self):
        from servers.daemon_dispatch import _handle_brain_batch

        for op, spec in BATCH_OP_SPECS.items():
            for field in spec['required']:
                payload = dict(self.VALID[op])
                payload.pop(field)
                r = _handle_brain_batch(
                    self.brain, {'operations': [{'op': op, **payload}]}, [])
                op_result = r['result']['results'][0]
                self.assertFalse(
                    op_result['ok'],
                    '%s without %s must fail: %r' % (op, field, op_result))
                self.assertIn(
                    field, op_result['error'],
                    '%s-missing error must name the field: %r'
                    % (field, op_result['error']))

    def test_revise_missing_reason_keeps_disambiguation(self):
        """The pre-check must not flatten the rich reason/reasoning error
        shipped 2026-06-12 (tests/test_revise_unified.py Class H)."""
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(self.brain, {'operations': [
            {'op': 'revise', 'node_id': 'aaaabbbb',
             'reasoning': 'meant as audit', 'content': 'x'},
        ]}, [])
        err = r['result']['results'][0]['error']
        self.assertIn('reason is required', err)
        self.assertIn('reasoning', err)
        self.assertIn('FIELD', err)

    def test_revise_missing_reason_and_node_id_keeps_disambiguation(self):
        """`reason` among MULTIPLE missing fields still gets the rich error,
        with the other missing fields appended (review 2026-06-12 #2)."""
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(self.brain, {'operations': [
            {'op': 'revise', 'reasoning': 'meant as audit', 'content': 'x'},
        ]}, [])
        err = r['result']['results'][0]['error']
        self.assertIn('reason is required', err)
        self.assertIn('reasoning', err)
        self.assertIn('Also missing: node_id', err)

    def test_unknown_op_still_hits_invalid_op_guard(self):
        """Ops outside BATCH_OP_SPECS bypass the pre-check and land in the
        loud invalid-op branch (S2 rejection table depends on that error)."""
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(
            self.brain, {'operations': [{'op': 'consolidate'}]}, [])
        err = r['result']['results'][0]['error']
        self.assertIn('Unknown op', err)
