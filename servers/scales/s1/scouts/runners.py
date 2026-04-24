"""Uniform scout runner registry for the muster.

Each scout has slightly different input needs:
- LLM scouts (quote, facts, synthesis) run on a pre-built shared_prefix
  via run_llm_scout (servers/scales/s1/scouts/base.py).
- Temporal runs algorithmically on raw turns + catalog_nodes via
  run_temporal_scout (servers/scales/s1/scouts/temporal.py).

The muster doesn't want to branch per scout. This module wraps each scout
in a callable with the signature  `(brain, muster_ctx) -> envelope` and
exposes SCOUT_RUNNERS as the single dispatch point. Muster iterates the
dict in parallel.

The `muster_ctx` dict carries every input any scout might need. Scouts
pick what they read; extra keys are ignored:
    turns               : list of {turn_id, role, text} dicts
    catalog_nodes       : list of catalog node dicts (for algo lookups)
    surfaced_by_turn    : dict {turn_id: [node_ids]} (reserved, scouts can use)
    session_context     : str
    current_date        : str 'YYYY-MM-DD'
    shared_prefix       : pre-built cache-friendly content blocks
    anthropic_client    : shared Anthropic client (one per cycle)
    log_fn              : optional line-logger

No per-scout wrapper files — adding a 10-line indirection for each LLM
scout is ceremony without benefit. If we later need per-scout post-
processing (e.g. entity normalization for facts), extract a dedicated
module at that point.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from . import contract as sc
from .base import run_llm_scout
from .temporal import run_temporal_scout


def _quote_runner(brain, ctx: Dict[str, Any]) -> Dict[str, Any]:
    return run_llm_scout(
        'quote', brain,
        shared_prefix=ctx['shared_prefix'],
        anthropic_client=ctx.get('anthropic_client'),
        log_fn=ctx.get('log_fn'),
    )


def _facts_runner(brain, ctx: Dict[str, Any]) -> Dict[str, Any]:
    return run_llm_scout(
        'facts', brain,
        shared_prefix=ctx['shared_prefix'],
        anthropic_client=ctx.get('anthropic_client'),
        log_fn=ctx.get('log_fn'),
    )


def _synthesis_runner(brain, ctx: Dict[str, Any]) -> Dict[str, Any]:
    return run_llm_scout(
        'synthesis', brain,
        shared_prefix=ctx['shared_prefix'],
        anthropic_client=ctx.get('anthropic_client'),
        log_fn=ctx.get('log_fn'),
    )


def _temporal_runner(brain, ctx: Dict[str, Any]) -> Dict[str, Any]:
    return run_temporal_scout(
        brain=brain,
        turns=ctx['turns'],
        catalog_nodes=ctx.get('catalog_nodes'),
        surfaced_node_ids_by_turn=ctx.get('surfaced_by_turn'),
        current_date=ctx['current_date'],
        log_fn=ctx.get('log_fn'),
    )


# Single dispatch point. Order doesn't matter — muster runs in parallel.
# The dict's keys MUST match SCOUT_NAMES in contract.py; validated below.
SCOUT_RUNNERS: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = {
    'quote':     _quote_runner,
    'temporal':  _temporal_runner,
    'facts':     _facts_runner,
    'synthesis': _synthesis_runner,
}


def _validate_registry():
    """Guard against drift between contract.SCOUT_NAMES and this registry."""
    missing = set(sc.SCOUT_NAMES) - set(SCOUT_RUNNERS)
    extra = set(SCOUT_RUNNERS) - set(sc.SCOUT_NAMES)
    if missing or extra:
        raise RuntimeError(
            f"SCOUT_RUNNERS drift from contract.SCOUT_NAMES: "
            f"missing={missing}, extra={extra}")


_validate_registry()


__all__ = ['SCOUT_RUNNERS']
