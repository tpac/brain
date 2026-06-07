"""Trace-chain lane (episodic dual-store rescue) — flag-gate contract.

Locks the two guarantees that matter for sacred recall code:
  - FLAG OFF (default): the lane is inert — no 'trace_chain' discovery ever appears. This is the
    "don't break live recall" guarantee; it must hold regardless of DB contents.
  - FLAG ON: the lane runs without error and never returns more than `limit` candidates (the
    reserved tail is additive within the pool, not an overflow).

Data-agnostic by design — asserts behavior contracts, not specific rescues (those are eval-gated in
eval/oracle_audit/trace_chain_wired_check.py against real data). Uses IsolatedBrain (copy of live DBs),
never the live daemon. Design: docs/RECALL-DUAL-STORE-DESIGN.md §3.3 form 1.
"""
import os
import sys

sys.path.insert(0, '/Users/tpac/brain')
from tests.isolated_brain import IsolatedBrain  # noqa: E402

_Q = "what did we do on the last session we work on ex.co?"


def _recall(brain, limit=25):
    if hasattr(brain, '_recall_cache'):
        try:
            brain._recall_cache.clear()
        except Exception:
            pass
    out = brain.recall(query=_Q, limit=limit)
    return out.get('results', []) if isinstance(out, dict) else (out or [])


def test_flag_off_is_inert():
    """Default (flag unset): no trace_chain discovery — live recall behavior unchanged."""
    prev = os.environ.pop('BRAIN_TRACE_CHAIN', None)
    try:
        with IsolatedBrain() as env:
            res = _recall(env.brain)
            assert all(r.get('_discovery') != 'trace_chain' for r in res), \
                "trace_chain candidates leaked with the flag OFF"
    finally:
        if prev is not None:
            os.environ['BRAIN_TRACE_CHAIN'] = prev


def test_flag_on_runs_and_respects_limit():
    """Flag on: lane runs without error and the pool never exceeds `limit`."""
    prev = os.environ.get('BRAIN_TRACE_CHAIN')
    os.environ['BRAIN_TRACE_CHAIN'] = '1'
    try:
        with IsolatedBrain() as env:
            res = _recall(env.brain, limit=25)
            assert len(res) <= 25, "reserved tail overflowed the pool limit"
    finally:
        if prev is None:
            os.environ.pop('BRAIN_TRACE_CHAIN', None)
        else:
            os.environ['BRAIN_TRACE_CHAIN'] = prev
