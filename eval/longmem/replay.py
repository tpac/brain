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


S2_EVERY_N_ENCODINGS = 2   # S2 fires every 2 encodings during ingestion
S2_FINAL_MAX_PASSES = 10   # Safety cap — final flush loops until every unit reports
                           # no-op (skipped / 0 actions) or this cap is hit


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


def replay_item(brain, session_id: str, haystack_sessions: List[List[Dict[str, str]]],
                haystack_dates: Optional[List[str]] = None,
                log_prefix: str = "[replay]",
                dumper=None,
                s2_every_n: Optional[int] = None,
                final_flush: bool = True,
                s2_carry: int = 0) -> Dict[str, Any]:
    """Replay a LongMemEval item's haystack through the brain.

    Args:
        brain: Brain instance
        session_id: unique session id for this ingestion run
        haystack_sessions: list of sessions, each a list of {role, content} turns
        haystack_dates: optional per-session ISO dates, used to prepend
            [Current date: YYYY-MM-DD] to user messages so the encoder can
            resolve relative time expressions
        log_prefix: prefix for log lines
        dumper: optional EvalArtifactsDumper. When supplied, snapshots
            nodes/edges before and after the final S2 flush so callers can
            diff exactly what S2 (consolidation + community + healer)
            changed in the graph. Files land at:
              pre_final_s2 / post_final_s2  (nodes{suffix}.jsonl, edges{suffix}.jsonl)
            Snapshot failures are non-fatal — the eval keeps going.

        final_flush: run finalize_item (S2 loop-until-quiet + backfill) at the
            end — the per-item default. The pooled builder passes False on its
            per-session calls and runs finalize_item ONCE after all sessions.
        s2_carry: seed for the S2 cadence counter — thread the previous call's
            stats['encodings_since_s2'] through so pooled cadence spans calls.

    Returns:
        {"turns": N, "user_turns": N, "s1e_runs": N, "s2_runs": N,
         "s1r_ms_total": N, "s1e_ms_total": N, "s2_ms_total": N,
         "encodings_since_s2": N}
    """
    from servers.daemon_hooks import hook_recall, post_response_common

    dispatch_fn = _make_local_dispatch(brain)
    ctx = brain.get_or_create_session(session_id)

    def _drain_embeddings(where: str) -> None:
        """Production-faithfulness fix (2026-07-17, smoke cfd549): the daemon's
        embed worker drains node + trace embeddings continuously; eval has no
        worker, so before this fix EVERY frozen corpus was built decode-blind —
        ingest-time recall saw a vector-less brain (empty results → surface
        never fired → zero s1r traces → encoder got no catalog → duplicate
        twins) and trace_embeddings stayed EMPTY (the moment stack's entire
        substrate). Same code paths the worker runs, called inline after each
        encode; local embedder only, no API cost."""
        try:
            brain.backfill_vectors(batch_size=50)
            from servers.embed_queue import _drain_trace_embeddings_once
            _drain_trace_embeddings_once(brain)
        except Exception as e:
            print(f"{log_prefix}   WARN embed drain failed at {where}: {e}",
                  flush=True)

    stats = {"turns": 0, "user_turns": 0, "s1e_runs": 0, "s2_runs": 0,
             "s1r_ms_total": 0, "s1e_ms_total": 0, "s2_ms_total": 0,
             # Every run_s2() return captured here so the corpus build can
             # record what S2 actually did (merges, communities, fills, errors)
             # — S2 is a subject under test, not just a fidelity knob.
             "s2_deltas": []}

    # S2 cadence is configurable so the corpus build can pin it into the
    # corpus_hash; defaults to the module constant when unset. s2_carry seeds
    # the counter so a pooled build's cadence spans per-session calls.
    s2_gate = s2_every_n if s2_every_n is not None else S2_EVERY_N_ENCODINGS
    encodings_since_s2 = int(s2_carry)
    total_sessions = len(haystack_sessions)

    for sess_idx, session in enumerate(haystack_sessions):
        print(f"{log_prefix} session {sess_idx+1}/{total_sessions} ({len(session)} turns)", flush=True)

        # Per-session date — prepended to user messages so the encoder can
        # resolve relative time references ("last Tuesday", "3 weeks ago")
        # against a concrete reference instead of today's actual date.
        session_date = None
        if haystack_dates and sess_idx < len(haystack_dates):
            session_date = haystack_dates[sess_idx]

        # Walk turns as (user, assistant) pairs. Haystacks are strictly alternating.
        i = 0
        while i < len(session):
            turn = session[i]
            if turn.get("role") != "user":
                i += 1
                continue

            user_msg = turn.get("content", "")
            if session_date:
                user_msg = f"[Current date: {session_date}]\n\n{user_msg}"
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

            # Post-response path — S0 traces, Hebbian strengthening, heartbeat,
            # stop counter increment. Calls the same code prod's Stop hook uses,
            # so eval and prod share one write path (no divergence).
            ctx = post_response_common(brain, session_id, user_msg, assistant_msg)

            # Per-turn embed drain: production's worker embeds new traces
            # within seconds, so the NEXT turn's recall (and the moment
            # stack's vector join) sees this turn — eval must match.
            _drain_embeddings("turn")

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
                # New nodes become recallable NOW, not at the final backfill —
                # the decode side the encoder's catalog depends on.
                _drain_embeddings("s1e")

                # S2 gate: every N encodings
                if encodings_since_s2 >= s2_gate:
                    print(f"{log_prefix}   s2 firing", flush=True)
                    t0s = time.time()
                    try:
                        s2_res = brain.run_s2()
                        stats["s2_runs"] += 1
                        stats["s2_ms_total"] += int((time.time() - t0s) * 1000)
                        stats["s2_deltas"].append(s2_res)
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
        _drain_embeddings("s1e_trailing")

    stats["encodings_since_s2"] = encodings_since_s2
    if not final_flush:
        # Pooled build (§20.18): the caller drives many per-session
        # replay_item calls over ONE brain and runs finalize_item() once at
        # the end — flushing loop-until-quiet after every session would be
        # unfaithful to the per-item config AND slow. The S2 cadence carries
        # across calls via stats['encodings_since_s2'] → next call's s2_carry.
        print(f"{log_prefix} done (no final flush): {stats}", flush=True)
        return stats
    finalize_item(brain, stats, encodings_since_s2, log_prefix=log_prefix,
                  dumper=dumper)
    print(f"{log_prefix} done: {stats}", flush=True)
    return stats


def finalize_item(brain, stats, encodings_since_s2, log_prefix="[replay]",
                  dumper=None):
    """The end-of-ingest flush: pre/post S2 snapshots, S2 loop-until-quiet,
    embedding backfill. ONE definition — replay_item's per-item tail and the
    pooled builder's once-at-the-end call are the same code."""
    # A caller-built stats dict (the pooled builder) must carry every counter
    # this flush increments — a missing key killed the whole final S2 flush
    # as a swallowed per-pass WARN (smoke build cfd549, 2026-07-17).
    for k in ("s2_runs", "s2_ms_total", "s2_deltas"):
        stats.setdefault(k, [] if k == "s2_deltas" else 0)
    # Snapshot the graph BEFORE the final S2 flush. Combined with the
    # post_final_s2 snapshot below, this gives a clean diff of exactly what
    # S2 (consolidation + community + healer) wrote during the flush —
    # without it, the post-everything dump conflates ingest-time S2 with
    # final-flush S2.
    if dumper:
        try:
            dumper.dump_nodes(brain, prefix='pre_final_s2')
            dumper.dump_edges(brain, prefix='pre_final_s2')
        except Exception as e:
            print(f"{log_prefix}   pre_final_s2 snapshot failed (non-fatal): {e}",
                  flush=True)

    # Final S2 flush: loop until every unit reports no-op (skipped or 0 actions).
    # S2 consolidation caps at ~10 proposals per run — multiple passes may be needed
    # to exhaust the backlog for large ingestions. Cap at S2_FINAL_MAX_PASSES for safety.
    # Healer writes happen in this phase and enqueue to embed_queue — the backfill
    # below drains them before query fires.
    print(f"{log_prefix}   s2 final flush ({encodings_since_s2} pending encodings, "
          f"loop until quiet, cap={S2_FINAL_MAX_PASSES})", flush=True)
    for pass_idx in range(S2_FINAL_MAX_PASSES):
        t0s = time.time()
        try:
            s2_result = brain.run_s2()
            stats["s2_runs"] += 1
            elapsed = int((time.time() - t0s) * 1000)
            stats["s2_ms_total"] += elapsed
            stats["s2_deltas"].append(s2_result)

            did_work = False
            if isinstance(s2_result, dict):
                # brain.run_s2() nests per-unit results under 'units', so unit
                # names can no longer collide with bookkeeping keys — the old
                # flat shape needed an explicit '_elapsed_ms' skip here.
                for unit_name, unit_result in s2_result.get("units", {}).items():
                    if not isinstance(unit_result, dict):
                        continue
                    if unit_result.get("actions", 0) > 0:
                        did_work = True
                        break

            print(f"{log_prefix}   s2 pass {pass_idx+1}/{S2_FINAL_MAX_PASSES} "
                  f"done in {elapsed}ms (did_work={did_work})", flush=True)

            if not did_work:
                print(f"{log_prefix}   s2 quiet at pass {pass_idx+1} — flush complete", flush=True)
                break
        except Exception as e:
            print(f"{log_prefix}   WARN s2 pass {pass_idx+1} failed: {e}", flush=True)
            break

    # Snapshot AFTER S2 final flush but BEFORE backfill — captures the
    # node/edge graph in the exact state S2 produced, isolated from the
    # embedding backfill that follows.
    if dumper:
        try:
            dumper.dump_nodes(brain, prefix='post_final_s2')
            dumper.dump_edges(brain, prefix='post_final_s2')
        except Exception as e:
            print(f"{log_prefix}   post_final_s2 snapshot failed (non-fatal): {e}",
                  flush=True)

    # Backfill AFTER all S1E + S2 writes (including healer) — production drains
    # via embed_queue worker, but eval runs inline so we must trigger explicitly.
    # Order matters: placed after S2 flush so healer's enqueued writes are included.
    print(f"{log_prefix}   backfill_vectors (drain all deferred embeddings)", flush=True)
    t0b = time.time()
    try:
        bf_result = brain.backfill_vectors(batch_size=50)
        print(f"{log_prefix}   backfill done in {int((time.time()-t0b)*1000)}ms: {bf_result}",
              flush=True)
    except Exception as e:
        print(f"{log_prefix}   WARN backfill failed: {e}", flush=True)
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
