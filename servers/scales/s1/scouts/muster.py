"""Scout muster — parallel orchestrator for S1Scribe's scouts.

Called from run_encoding() once per encoding cycle, BETWEEN the existing
`_gather_messages` + `_build_user_content` step and the `run_llm_loop` call.
Produces a SCOUT_REPORTS block that gets appended to the user_content
flowing into S1Scribe.

Responsibilities:
  1. Build the muster context from S1E's already-gathered inputs
     (messages, session, catalog text, catalog node dicts, current date).
  2. Assemble the shared_prefix once (cache-friendly, byte-identical
     across LLM scouts in this cycle).
  3. Create ONE Anthropic client for the cycle, shared across LLM scouts
     so cache warm-up after the first scout is honored by the rest.
  4. Fan out all 4 scouts in parallel via ThreadPoolExecutor. Each scout
     returns a valid envelope (or a stub with _errors on failure) —
     never raises.
  5. Collect outputs, format the combined report for S1S, emit trace
     events, return.

Emits trace events on the caller's S1E chain:
  s1 O  ref_type=scout_input     (shared prefix + per-scout tasks)
  s1 K  ref_type=scout_findings  (candidates per scout)

See scouts/contract.py for envelope shape, scouts/runners.py for the
uniform dispatch registry.
"""
from __future__ import annotations

import concurrent.futures as _cf
import datetime as _dt
import json as _json
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from . import contract as sc
from .runners import SCOUT_RUNNERS


# Per-scout wall-clock deadline. When muster provides a shared Anthropic
# client (the typical path), this is the EFFECTIVE upper bound on any
# scout — the client's internal timeout is the SDK default (~600s) which
# is much longer. A scout's own `timeout_seconds` in its interaction
# parameters only applies if the scout builds its own client instead.
#
# 90s covers: algo scout (<1s), Haiku scouts (~2-5s each), Sonnet scout
# for synthesis (~6-15s typical, up to ~30s on cold cache). Generous
# buffer for tail latency before we stub the scout as timed-out.
MUSTER_PER_SCOUT_TIMEOUT_S = 90.0


def build_muster_context(
    *,
    brain,
    messages: List[Dict[str, Any]],
    session_id: str,
    counter: int,
    catalog_rendered: str,
    catalog_node_ids: set,
    session_context: str = '',
    current_date: Optional[str] = None,
    log_fn=None,
) -> Dict[str, Any]:
    """Assemble the ctx dict passed to every scout runner.

    Args match what run_encoding() already has in scope — no extra work
    required at the call site beyond passing these in.
    """
    if current_date is None:
        current_date = _dt.date.today().isoformat()

    # Turn-typed view — scout runners consume turns with (turn_id, role, text)
    turns = [{
        'turn_id': m.get('id') or f'turn-{i}',
        'role': m.get('role', ''),
        'text': m.get('content', '') or '',
    } for i, m in enumerate(messages)]

    # Surfaced-by-turn (scouts that need it look up per-turn surfacer output).
    # User messages carry a "judge_output" field with surfaced IDs embedded
    # as "id:XXXXXXXX"; extract those for a clean per-turn index.
    import re
    surfaced_by_turn: Dict[str, List[str]] = {}
    for m in messages:
        if m.get('role') != 'user':
            continue
        turn_id = m.get('id') or ''
        jo = m.get('judge_output') or ''
        if jo and jo != '(no selection)':
            ids = re.findall(r'id:([a-z0-9_]{6,8})', jo)
            if ids:
                surfaced_by_turn[turn_id] = ids

    surfaced_rendered = _render_surfaced_by_turn(surfaced_by_turn, turns)
    conversation_rendered = _render_conversation(turns, surfaced_by_turn)

    shared_prefix = sc.build_shared_prefix(
        session_context=session_context,
        current_date=current_date,
        catalog_rendered=catalog_rendered,
        surfaced_by_turn_rendered=surfaced_rendered,
        conversation_rendered=conversation_rendered,
    )

    # Fetch catalog node dicts for algorithmic scouts (temporal, later
    # entity/others). If the brain doesn't have a batch getter, fall back
    # to per-id lookup.
    catalog_nodes = _fetch_catalog_nodes(brain, catalog_node_ids)

    # Shared Anthropic client so cache warm-up carries across LLM scouts
    # within this cycle.
    import anthropic
    try:
        client = anthropic.Anthropic()
    except Exception:
        client = None

    return {
        'brain': brain,
        'session_id': session_id,
        'counter': counter,
        'turns': turns,
        'catalog_nodes': catalog_nodes,
        'surfaced_by_turn': surfaced_by_turn,
        'session_context': session_context,
        'current_date': current_date,
        'shared_prefix': shared_prefix,
        'anthropic_client': client,
        'log_fn': log_fn,
    }


def run_muster(
    ctx: Dict[str, Any],
    timeout_s: float = MUSTER_PER_SCOUT_TIMEOUT_S,
) -> Tuple[str, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Run all scouts in parallel and return the combined S1S report.

    Returns:
        (formatted_report, scout_outputs_by_name, metrics)

    formatted_report: string block to append into S1Scribe's user_content
    scout_outputs_by_name: dict {scout_name: envelope_dict}, always has all
        four scouts (any that timed out get a stub envelope with _errors)
    metrics: summary dict — per-scout latency, candidate counts, errors
    """
    brain = ctx['brain']
    log = ctx.get('log_fn') or (lambda _: None)

    t0 = _time.time()
    outputs: Dict[str, Dict[str, Any]] = {}

    # One thread per scout. Per-scout timeout enforced via a wall-clock
    # deadline: each fut.result(timeout=remaining) bounds the wait by the
    # remaining budget, so a single blocking scout never costs more than
    # `timeout_s` for the whole cycle.
    #
    # Note: ThreadPoolExecutor CAN'T interrupt running threads — cancelled
    # or timed-out scouts continue running in the background until their
    # own internal timeouts fire. This is acceptable: the blocking time is
    # bounded by ANTHROPIC_CLIENT_TIMEOUT for LLM scouts, and the temporal
    # scout doesn't block. Ghost threads are released when the pool exits.
    deadline = _time.time() + timeout_s

    # Important: exit the `with` block WITHOUT waiting for pending threads.
    # Default ThreadPoolExecutor.__exit__ blocks on shutdown — if a scout
    # is still running, we'd hang there. Python 3.9+ supports
    # shutdown(wait=False, cancel_futures=True) which we call explicitly.
    pool = _cf.ThreadPoolExecutor(max_workers=len(SCOUT_RUNNERS))
    try:
        futures = {
            name: pool.submit(_safe_run, name, runner, brain, ctx)
            for name, runner in SCOUT_RUNNERS.items()
        }
        for name, fut in futures.items():
            remaining = max(0.0, deadline - _time.time())
            try:
                outputs[name] = fut.result(timeout=remaining)
            except _cf.TimeoutError:
                log(f'[muster] {name} TIMED OUT after ~{timeout_s}s')
                outputs[name] = _timeout_stub(name, timeout_s)
                _log_error(brain, name, 'muster_timeout',
                           f'exceeded {timeout_s}s')
                # Best-effort cancel — won't interrupt a running thread but
                # prevents future-scheduled tasks from starting.
                fut.cancel()
            except Exception as e:
                log(f'[muster] {name} raised: {type(e).__name__}: {e}')
                outputs[name] = _exception_stub(name, e)
                _log_error(brain, name, 'muster_exception',
                           f'{type(e).__name__}: {e}')
    finally:
        # Don't block on shutdown — ghost threads finish on their own,
        # muster returns promptly.
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # Python < 3.9 fallback
            pool.shutdown(wait=False)

    # Any scout that didn't return at all (shouldn't happen) — pad with a
    # stub so formatter always sees four scouts.
    for name in SCOUT_RUNNERS:
        if name not in outputs:
            outputs[name] = _exception_stub(name,
                                            RuntimeError('no result'))

    elapsed_ms = int((_time.time() - t0) * 1000)
    formatted = sc.format_scout_report_for_s1s(outputs)
    metrics = _metrics(outputs, elapsed_ms)

    summary = ', '.join(
        f'{n}={len(outputs[n].get("candidates") or [])}'
        for n in sc.SCOUT_NAMES
    )
    log(f'[muster] done in {elapsed_ms}ms — candidates: {summary}')

    _emit_traces(ctx, outputs, elapsed_ms)

    return formatted, outputs, metrics


# ─── Internals ────────────────────────────────────────────────────────────


def _safe_run(name: str, runner, brain, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke a scout runner, catching any leaked exception. Scout runners
    already swallow their own errors and return envelope stubs, but this
    layer is belt-and-suspenders so muster's parallel dispatch can rely on
    every future resolving to a valid dict."""
    try:
        return runner(brain, ctx)
    except Exception as e:
        return _exception_stub(name, e)


def _exception_stub(scout_name: str, exc: Exception) -> Dict[str, Any]:
    return {
        'scout': scout_name,
        'category_statement': '',
        'candidates': [],
        'scanned': {'turns': 0, 'considered': 0, 'passed_threshold': 0},
        '_usage': {},
        '_latency_ms': 0,
        '_errors': [{'type': 'muster_exception',
                     'msg': f'{type(exc).__name__}: {exc}'}],
        '_warnings': [],
    }


def _timeout_stub(scout_name: str, timeout_s: float) -> Dict[str, Any]:
    return {
        'scout': scout_name,
        'category_statement': '',
        'candidates': [],
        'scanned': {'turns': 0, 'considered': 0, 'passed_threshold': 0},
        '_usage': {},
        '_latency_ms': int(timeout_s * 1000),
        '_errors': [{'type': 'muster_timeout',
                     'msg': f'exceeded {timeout_s}s'}],
        '_warnings': [],
    }


def _log_error(brain, scout_name: str, error_type: str, message: str):
    try:
        brain._log_error(
            f's1_scout_{scout_name}_{error_type}',
            ValueError(message), f'scout={scout_name}')
    except Exception:
        pass


def _render_surfaced_by_turn(
    surfaced_by_turn: Dict[str, List[str]],
    turns: List[Dict[str, Any]],
) -> str:
    """Render "{turn_id}: [id1, id2]" per turn that surfaced something."""
    lines = []
    for t in turns:
        tid = t['turn_id']
        ids = surfaced_by_turn.get(tid)
        if ids:
            lines.append(f"{tid}: [{', '.join(ids)}]")
    return '\n'.join(lines)


def _render_conversation(
    turns: List[Dict[str, Any]],
    surfaced_by_turn: Dict[str, List[str]],
) -> str:
    """Render last-N turns with role + content + surfaced IDs per turn."""
    lines = []
    for t in turns:
        tid = t['turn_id']
        role = t['role']
        text = t['text'] or ''
        ids = surfaced_by_turn.get(tid, [])
        lines.append(f"[{tid}] {role}: {text}")
        if ids:
            lines.append(f"  surfaced: {', '.join(ids)}")
    return '\n'.join(lines)


def _fetch_catalog_nodes(brain, node_ids: set) -> List[Dict[str, Any]]:
    """Return node dicts (id, type, title, content, ...) for scouts that
    need algorithmic lookups. Minimally includes id, type, title — enough
    for temporal's existing_anchor_id check. Graceful on error.
    """
    if not node_ids:
        return []
    try:
        # Most brains expose get_nodes (batch) or get_node (per id).
        if hasattr(brain, 'get_nodes'):
            return list(brain.get_nodes(list(node_ids)) or [])
        if hasattr(brain, 'get_node'):
            out = []
            for nid in node_ids:
                try:
                    n = brain.get_node(nid)
                    if n:
                        out.append(n)
                except Exception:
                    continue
            return out
    except Exception:
        return []
    return []


def _metrics(outputs: Dict[str, Dict[str, Any]], elapsed_ms: int) -> Dict[str, Any]:
    per_scout = {}
    total_errors = 0
    total_candidates = 0
    for name, out in outputs.items():
        cands = out.get('candidates') or []
        errs = out.get('_errors') or []
        per_scout[name] = {
            'candidates': len(cands),
            'latency_ms': out.get('_latency_ms', 0),
            'errors': len(errs),
            'warnings': len(out.get('_warnings') or []),
        }
        total_errors += len(errs)
        total_candidates += len(cands)
    return {
        'elapsed_ms': elapsed_ms,
        'per_scout': per_scout,
        'total_candidates': total_candidates,
        'total_errors': total_errors,
    }


def _emit_traces(ctx: Dict[str, Any],
                 outputs: Dict[str, Dict[str, Any]],
                 elapsed_ms: int):
    """Emit O and K trace events per scout on the S1E chain.

    Graceful on failure — tracing is observability, not correctness.
    Re-uses the caller's s1e-{session}-{counter} chain so dashboard groups
    scout events under the encoding run they informed.
    """
    brain = ctx['brain']
    session_id = ctx.get('session_id') or ''
    counter = ctx.get('counter') or 0
    if not session_id:
        return

    chain = f's1e-{session_id[:8]}-{counter}'

    try:
        dal = brain._trace_dal
    except Exception:
        return

    events = []
    for name in sc.SCOUT_NAMES:
        out = outputs.get(name) or {}
        cands = out.get('candidates') or []
        # O — scout saw this much input (scanned counts)
        events.append({
            'chain_id': chain, 'scale': 's1', 'event_type': 'O',
            'ref_type': 'scout_input',
            'summary': f'{name}: scanned {out.get("scanned", {}).get("turns", 0)} turns',
            'metadata': _json.dumps({
                'scout': name,
                'scanned': out.get('scanned', {}),
                'latency_ms': out.get('_latency_ms', 0),
            })[:4000],
            'session_id': session_id,
        })
        # K — what the scout selected
        events.append({
            'chain_id': chain, 'scale': 's1', 'event_type': 'K',
            'ref_type': 'scout_findings',
            'summary': f'{name}: {len(cands)} candidates',
            'metadata': _json.dumps({
                'scout': name,
                'category_statement': out.get('category_statement', ''),
                'candidate_handles': [c.get('handle') for c in cands][:20],
                'errors': out.get('_errors', []),
                'warnings': out.get('_warnings', []),
            })[:4000],
            'session_id': session_id,
        })

    try:
        dal.append_batch(events)
    except Exception:
        pass


__all__ = [
    'build_muster_context',
    'run_muster',
    'MUSTER_PER_SCOUT_TIMEOUT_S',
]
