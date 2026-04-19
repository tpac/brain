"""Replay primitive — ingest LongMemEval haystack through the full S0/S1/S2 loop.

Per turn:
  user turn  → hook_recall (S1R: recall + surface, writes candidates file + surface traces)
  user+asst  → write S0 traces (K=user, delta=assistant), increment stop counter
  every 5th  → run S1E in foreground (NOT background — deterministic for eval)
  every N    → run S2 coordinator (configurable via S2_EVERY_N_ENCODINGS)

This diverges from production in ONE way: S1E runs foreground instead of background
thread. Everything else is the same code path as the live daemon.

Query phase (separate session_id so cold-query semantics):
  query_brain(brain, question, question_date) → additionalContext from surface

The answerer then takes (question, additionalContext) and produces a hypothesis.
"""
import os
import time
from typing import List, Dict, Any, Optional, Tuple


S2_EVERY_N_ENCODINGS = 3   # S2 fires every 3 encodings during ingestion
S2_FINAL_PASSES = 4        # S2 at end of ingestion: run N times to exhaust backlog
                           # S2 consolidation caps at ~10 proposals per run — multiple passes
                           # ensure it sees everything


def _make_local_dispatch(brain):
    """Build an in-process dispatch function — same shape as TCP dispatch, no network."""
    from servers.daemon_dispatch import COMMAND_TABLE

    def dispatch(cmd, args=None):
        entry = COMMAND_TABLE.get(cmd)
        if not entry:
            return {"ok": False, "error": "unknown command: %s" % cmd}
        return entry.handler(brain, args or {}, [])

    return dispatch


def _run_s1e_foreground(brain, dispatch_fn, counter, session_id) -> Dict[str, Any]:
    """Run S1 encoding directly (not in background thread)."""
    from servers.scales.s1.encode import run_encoding
    t0 = time.time()
    result = run_encoding(brain, dispatch_fn, counter, session_id)
    elapsed_ms = int((time.time() - t0) * 1000)
    result["_elapsed_ms"] = elapsed_ms
    return result


def _run_s2_foreground(brain) -> Dict[str, Any]:
    """Run S2 coordinator — consolidation, community, healer. All units decide internally whether to fire."""
    from servers.scales.s2.coordinator import run_s2
    t0 = time.time()
    result = run_s2(brain)
    elapsed_ms = int((time.time() - t0) * 1000)
    if isinstance(result, dict):
        result["_elapsed_ms"] = elapsed_ms
    return result or {"_elapsed_ms": elapsed_ms}


def replay_item(brain, session_id: str, haystack_sessions: List[List[Dict[str, str]]],
                log_prefix: str = "[replay]") -> Dict[str, Any]:
    """Replay a LongMemEval item's haystack through the brain.

    Args:
        brain: Brain instance
        session_id: unique session id for this ingestion run
        haystack_sessions: list of sessions, each a list of {role, content} turns
        log_prefix: prefix for log lines

    Returns:
        {"turns": N, "user_turns": N, "s1e_runs": N, "s2_runs": N,
         "s1r_ms_total": N, "s1e_ms_total": N, "s2_ms_total": N}
    """
    from servers.daemon_hooks import hook_recall

    dispatch_fn = _make_local_dispatch(brain)
    ctx = brain.get_or_create_session(session_id)

    stats = {"turns": 0, "user_turns": 0, "s1e_runs": 0, "s2_runs": 0,
             "s1r_ms_total": 0, "s1e_ms_total": 0, "s2_ms_total": 0}

    encodings_since_s2 = 0
    total_sessions = len(haystack_sessions)

    for sess_idx, session in enumerate(haystack_sessions):
        print(f"{log_prefix} session {sess_idx+1}/{total_sessions} ({len(session)} turns)", flush=True)

        # Walk turns as (user, assistant) pairs. Haystacks are strictly alternating.
        i = 0
        while i < len(session):
            turn = session[i]
            if turn.get("role") != "user":
                i += 1
                continue

            user_msg = turn.get("content", "")
            assistant_msg = ""
            if i + 1 < len(session) and session[i + 1].get("role") == "assistant":
                assistant_msg = session[i + 1].get("content", "")

            stats["turns"] += 1 if not assistant_msg else 2
            stats["user_turns"] += 1

            # S1R: recall + surface (same code path as live daemon)
            t0 = time.time()
            try:
                hook_recall(brain, {"prompt": user_msg, "session_id": session_id}, [])
            except Exception as e:
                print(f"{log_prefix}   WARN s1r failed turn {stats['user_turns']}: {e}", flush=True)
            stats["s1r_ms_total"] += int((time.time() - t0) * 1000)

            # S0 traces: write K=user_message and delta=assistant_message (same as Stop hook)
            try:
                brain._trace_dal.append(
                    chain_id=ctx.s0_chain(), scale='s0', event_type='K',
                    ref_type='user_message',
                    summary=user_msg[:200],
                    metadata={'content': user_msg[:4000]} if user_msg else None,
                    session_id=session_id)
                if assistant_msg:
                    brain._trace_dal.append(
                        chain_id=ctx.s0_chain(), scale='s0', event_type='delta',
                        ref_type='assistant_message',
                        summary=assistant_msg[:200],
                        metadata={'content': assistant_msg[:4000]},
                        session_id=session_id)
            except Exception as e:
                print(f"{log_prefix}   WARN s0 trace failed: {e}", flush=True)

            # Increment stop counter (same as Stop hook)
            ctx.increment_stop()
            ctx.save(brain.logs_conn)

            # S1E gate: every 5th turn, foreground encoding
            if ctx.stop_counter % 5 == 0 and ctx.stop_counter > 0:
                print(f"{log_prefix}   s1e firing at stop={ctx.stop_counter}", flush=True)
                t0 = time.time()
                try:
                    enc_result = _run_s1e_foreground(brain, dispatch_fn, ctx.stop_counter, session_id)
                    stats["s1e_runs"] += 1
                    stats["s1e_ms_total"] += enc_result.get("_elapsed_ms", 0)
                    encodings_since_s2 += 1
                    print(f"{log_prefix}   s1e done in {enc_result.get('_elapsed_ms', 0)}ms", flush=True)
                except Exception as e:
                    print(f"{log_prefix}   WARN s1e failed: {e}", flush=True)

                # S2 gate: every N encodings
                if encodings_since_s2 >= S2_EVERY_N_ENCODINGS:
                    print(f"{log_prefix}   s2 firing", flush=True)
                    t0s = time.time()
                    try:
                        _run_s2_foreground(brain)
                        stats["s2_runs"] += 1
                        stats["s2_ms_total"] += int((time.time() - t0s) * 1000)
                        encodings_since_s2 = 0
                        print(f"{log_prefix}   s2 done in {int((time.time()-t0s)*1000)}ms", flush=True)
                    except Exception as e:
                        print(f"{log_prefix}   WARN s2 failed: {e}", flush=True)

            i += 2 if assistant_msg else 1

    # Trailing flush: force S1E if there are unencoded turns since the last 5-boundary.
    # In production S1E only fires every 5 stops — trailing turns wait for next session.
    # For eval we MUST encode everything before query, or the question finds nothing.
    turns_since_last_encoding = ctx.stop_counter % 5
    if turns_since_last_encoding > 0:
        print(f"{log_prefix}   s1e trailing flush (stop={ctx.stop_counter}, {turns_since_last_encoding} unencoded)",
              flush=True)
        t0 = time.time()
        try:
            enc_result = _run_s1e_foreground(brain, dispatch_fn, ctx.stop_counter, session_id)
            stats["s1e_runs"] += 1
            stats["s1e_ms_total"] += enc_result.get("_elapsed_ms", 0)
            encodings_since_s2 += 1
        except Exception as e:
            print(f"{log_prefix}   WARN s1e trailing failed: {e}", flush=True)

    # Backfill embeddings — production runs this during idle; eval must trigger explicitly.
    # Without this, node_enrichments rows exist with NULL embeddings and recall returns 0.
    print(f"{log_prefix}   backfill_vectors (compute deferred embeddings)", flush=True)
    t0b = time.time()
    try:
        bf_result = brain.backfill_vectors(batch_size=50)
        print(f"{log_prefix}   backfill done in {int((time.time()-t0b)*1000)}ms: {bf_result}",
              flush=True)
    except Exception as e:
        print(f"{log_prefix}   WARN backfill failed: {e}", flush=True)

    # Final S2 flush: run multiple passes. S2 caps at ~10 proposals per run, so one pass
    # may not exhaust the backlog for large ingestions. Loop until no-op or cap hit.
    print(f"{log_prefix}   s2 final flush ({encodings_since_s2} pending encodings, {S2_FINAL_PASSES} passes)",
          flush=True)
    for pass_idx in range(S2_FINAL_PASSES):
        t0s = time.time()
        try:
            _run_s2_foreground(brain)
            stats["s2_runs"] += 1
            elapsed = int((time.time() - t0s) * 1000)
            stats["s2_ms_total"] += elapsed
            print(f"{log_prefix}   s2 pass {pass_idx+1}/{S2_FINAL_PASSES} done in {elapsed}ms", flush=True)
        except Exception as e:
            print(f"{log_prefix}   WARN s2 pass {pass_idx+1} failed: {e}", flush=True)
            break

    print(f"{log_prefix} done: {stats}", flush=True)
    return stats


def query_brain(brain, question: str, question_date: Optional[str] = None,
                log_prefix: str = "[query]") -> Dict[str, Any]:
    """Query the ingested brain with the eval question. Cold query — no prior conversation.

    Uses a fresh session_id separate from ingestion so recent_messages doesn't
    pollute the query with stale ingestion turns.

    Args:
        brain: ingested Brain instance
        question: the eval question
        question_date: optional ISO-ish date string from LongMemEval (for temporal context)

    Returns:
        {"additional_context": str | None, "s1r_ms": int, "query_session_id": str}
    """
    from servers.daemon_hooks import hook_recall
    import uuid

    # Fresh session for cold query — recent_messages will be empty
    query_session_id = "query-" + uuid.uuid4().hex[:12]
    _ = brain.get_or_create_session(query_session_id)

    # Prepend question_date as context if provided (temporal axis cares about this)
    if question_date:
        prompt = f"[Current date: {question_date}]\n\n{question}"
    else:
        prompt = question

    print(f"{log_prefix} question: {question[:100]}{'...' if len(question) > 100 else ''}", flush=True)
    t0 = time.time()
    try:
        result = hook_recall(brain, {"prompt": prompt, "session_id": query_session_id}, [])
    except Exception as e:
        print(f"{log_prefix} WARN hook_recall failed: {e}", flush=True)
        result = {}
    elapsed_ms = int((time.time() - t0) * 1000)

    # hook_recall returns {"json": {"additionalContext": ...}} when surface selected nodes,
    # or {"json": {"decision": "approve"}} when nothing was selected.
    additional_context = None
    if isinstance(result, dict):
        inner = result.get("json", {}) if isinstance(result.get("json"), dict) else {}
        additional_context = inner.get("additionalContext")

    print(f"{log_prefix} s1r {elapsed_ms}ms, context {len(additional_context or '')} chars", flush=True)
    return {
        "additional_context": additional_context,
        "s1r_ms": elapsed_ms,
        "query_session_id": query_session_id,
    }
