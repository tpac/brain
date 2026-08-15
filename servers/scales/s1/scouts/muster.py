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
from .base import SCOUT_TOKEN_USAGE_KEY
from .runners import SCOUT_RUNNERS


# Muster wall-clock BACKSTOP (shared, not per-scout). Each fut.result()
# waits against one t0+timeout deadline, so the whole muster — not each
# scout — is bounded by this; a single stalling scout can consume the
# full budget. The PRIMARY per-scout bound now lives in base.run_llm_scout,
# which binds each LLM scout's `timeout_seconds` (default 25s) +
# max_retries=0 onto the request via with_options — so a stalled LLM scout
# aborts at ~25s and its ghost thread dies there, not at the SDK ~600s
# ceiling. This 90s only fires in the unusual case where that per-request
# timeout doesn't (e.g. an algo scout blocked on something other than the
# Anthropic SDK). The name says "per-scout" for historical reasons.
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
        # Fall back to brain_now() (operator wall-clock) when caller didn't
        # provide a conversation date. encode.py should normally pass one
        # via conversation_now(messages). Loud fallback so a missing-date
        # plumbing regression doesn't silently corrupt scout candidates.
        # See servers/clock.py + brain memory 6d5b789e for context.
        from servers.clock import brain_today
        current_date = brain_today().isoformat()

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
    # entity/others) — one batched get_node call.
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
    exclude_scouts: Tuple[str, ...] = (),
) -> Tuple[str, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Run all scouts in parallel and return the combined S1S report.

    Args:
        exclude_scouts: scout names to NOT run this cycle. Excluded scouts get
            the standard 'disabled' stub via the SCOUT_NAMES padding, so every
            downstream consumer stays shape-safe. The lived arm passes
            ('quote',) — episodes recall preserves verbatim substrate, so the
            dedicated quote scout is retired there (Tom, 2026-07-02); the
            control arm runs the full set (production-faithful baseline).

    Returns:
        (formatted_report, scout_outputs_by_name, metrics)

    formatted_report: string block to append into S1Scribe's user_content
    scout_outputs_by_name: dict {scout_name: envelope_dict}, always has all
        four scouts (any that timed out get a stub envelope with _errors)
    metrics: summary dict — per-scout latency, candidate counts, errors
    """
    brain = ctx['brain']
    log = ctx.get('log_fn') or (lambda _: None)
    runners = {n: r for n, r in SCOUT_RUNNERS.items() if n not in exclude_scouts}

    t0 = _time.time()
    outputs: Dict[str, Dict[str, Any]] = {}

    # One thread per scout. Per-scout timeout enforced via a wall-clock
    # deadline: each fut.result(timeout=remaining) bounds the wait by the
    # remaining budget, so a single blocking scout never costs more than
    # `timeout_s` for the whole cycle.
    #
    # Note: ThreadPoolExecutor CAN'T interrupt running threads — cancelled
    # or timed-out scouts continue running in the background until their
    # own internal timeouts fire. This is acceptable: an LLM scout's ghost
    # thread is bounded by its OWN per-request timeout — scouts/base binds
    # `timeout_seconds` (and max_retries=0) via with_options on every call,
    # which is what actually caps this path; the shared client below carries
    # no constructor timeout. The temporal scout doesn't block at all. Ghost
    # threads are released when the pool exits.
    deadline = _time.time() + timeout_s

    # Important: exit the `with` block WITHOUT waiting for pending threads.
    # Default ThreadPoolExecutor.__exit__ blocks on shutdown — if a scout
    # is still running, we'd hang there. Python 3.9+ supports
    # shutdown(wait=False, cancel_futures=True) which we call explicitly.
    pool = _cf.ThreadPoolExecutor(max_workers=max(1, len(runners)))
    try:
        futures = {
            name: pool.submit(_safe_run, name, runner, brain, ctx)
            for name, runner in runners.items()
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

    # Pad missing scouts so downstream iterations over SCOUT_NAMES are safe.
    # We pad for SCOUT_NAMES (not SCOUT_RUNNERS) so disabled scouts still
    # get a stub — the formatter and trace emitter iterate SCOUT_NAMES and
    # need outputs[name] to exist. Disabled scouts get a marker stub so
    # consumers can distinguish "didn't run" from "ran and found nothing".
    for name in sc.SCOUT_NAMES:
        if name not in outputs:
            reason = ('disabled' if (name not in SCOUT_RUNNERS
                                     or name in exclude_scouts) else 'no result')
            outputs[name] = _exception_stub(name, RuntimeError(reason))

    elapsed_ms = int((_time.time() - t0) * 1000)
    formatted = sc.format_scout_report_for_s1s(outputs)
    metrics = _metrics(outputs, elapsed_ms)

    # Safe summary — use .get since disabled scouts may or may not be padded
    # depending on invariants upstream.
    summary = ', '.join(
        f'{n}={len(outputs.get(n, {}).get("candidates") or [])}'
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
        # brain.get_node(list) batches (5 queries total instead of N×4) and
        # returns {resolved_full_id: node} — short input ids come back under
        # their resolved keys and missing ids are omitted, so consume values,
        # never index by the input id.
        return list((brain.get_node(list(node_ids)) or {}).values())
    except Exception as e:
        try:
            brain._log_error('scout_muster', e,
                             'catalog batch fetch failed — scouts see an '
                             'empty catalog this round')
        except Exception:
            pass
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
        # O — scout saw this much input (scanned counts).
        # Pass metadata as a dict; TraceDAL.append_batch serializes it.
        # Previously this was pre-serialized here, producing a double-
        # encoded string that decoded back to str instead of dict.
        events.append({
            'chain_id': chain, 'scale': 's1', 'event_type': 'O',
            'ref_type': 'scout_input',
            'summary': f'{name}: scanned {out.get("scanned", {}).get("turns", 0)} turns',
            'metadata': {
                'scout': name,
                'scanned': out.get('scanned', {}),
                'latency_ms': out.get('_latency_ms', 0),
            },
            'session_id': session_id,
        })
        # K — what the scout selected. candidate_handles is kept for
        # backward compatibility with analyzers grepping for ISO dates;
        # candidates_detail is the richer per-candidate dump (source_phrase,
        # source_role, evidence_turns, precision, ...) that lets post-hoc
        # diagnosis answer "which turn / role attributed this date" without
        # re-running scouts. Diagnosed regression 2026-05-13: lack of
        # source_role made the gpt4_85da3956 root-cause investigation take
        # an hour instead of seconds.
        # Cost telemetry — the LLM scouts capture per-call usage into the
        # '_usage' stub via runner.read_usage (base.run_llm_scout step 6),
        # so it already carries the short USAGE_FIELDS names the encoder
        # deltas and Surface K trace use — copied verbatim, no mapping. One
        # trace query tallies every agent's API spend. Only present when the
        # scout actually made an LLM call (algo scouts, disabled stubs, and
        # pre-call failures carry no usage).
        usage_fields = dict(out.get(SCOUT_TOKEN_USAGE_KEY) or {})
        events.append({
            'chain_id': chain, 'scale': 's1', 'event_type': 'K',
            'ref_type': 'scout_findings',
            'summary': f'{name}: {len(cands)} candidates',
            'metadata': {
                'scout': name,
                **usage_fields,
                'category_statement': out.get('category_statement', ''),
                'candidate_handles': [c.get('handle') for c in cands][:20],
                'candidates_detail': [
                    {k: v for k, v in c.items()
                     if k in ('handle', 'evidence_turns', 'evidence_roles',
                              'source_phrase', 'source_role', 'precision',
                              'resolution', 'event_description',
                              'why_candidate', 'entity', 'feature', 'value',
                              'speaker', 'turn_evidence')
                     and v not in (None, '', [], {})}
                    for c in cands[:20]
                ],
                'errors': out.get('_errors', []),
                'warnings': out.get('_warnings', []),
            },
            'session_id': session_id,
        })

    try:
        dal.append_batch(events)
    except Exception as trace_exc:
        # Loud-by-default: tracing is observability, not correctness, but a
        # silent drop here hid the metadata-double-encode bug for days. Log
        # to brain_errors so future breakage surfaces in consciousness.
        try:
            brain._log_error('s1_muster_trace_emit', trace_exc,
                             'failed to emit scout_input/scout_findings events')
        except Exception:
            pass  # last-resort — if the error logger itself breaks, proceed
        # Also print so the running process shows the failure in stdout
        print(f'[muster] trace emit failed: {type(trace_exc).__name__}: {trace_exc}',
              flush=True)


__all__ = [
    'build_muster_context',
    'run_muster',
    'MUSTER_PER_SCOUT_TIMEOUT_S',
]
