"""Centralized hook logic — all hook brain interactions live here.

Previously each hook .py file had a _run_daemon() and _run_direct() path,
duplicating logic and diverging over time. Now hooks are thin clients that
send a single command to the daemon, and this module contains all the logic.

Each function signature: hook_*(brain, args, graph_changes) -> dict
  - brain: Brain instance (already loaded by daemon)
  - args: dict from the hook client
  - graph_changes: list[str] — in-memory mutation log, drained by hook_recall

Returns: {"output": str} for text hooks, or {"output": str, "json": dict}
for hooks that need structured JSON output (recall, pre-edit, pre-bash, pre-compact).
"""

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime

# Encoding agent: at most 1 running at a time. Non-blocking acquire — skip if busy.
_encoding_lock = threading.Lock()

# ── Constants (canonical definitions in brain_voice.py) ──

from servers.brain_voice import BrainVoice
# Hoisted — if this import ever breaks, the daemon fails at boot instead of
# silently degrading recall 16s in.
from servers.embedder import embed_query

# Backwards-compatible function aliases — delegate to BrainVoice static methods
# format_recall_results used only by MCP tool output, not by hook path
_format_encoding_warning = BrainVoice.format_encoding_warning
_format_suggestions = BrainVoice.format_suggestions


DESTRUCTIVE_REGEXES = [
    r"rm\s+(-[rf]+\s+|.*--force)",
    r"git\s+worktree\s+remove",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[fd]",
    r"git\s+checkout\s+--\s",
    r"git\s+push\s+.*--force",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
    r"TRUNCATE",
    r"\brmdir\b",
    r"xargs\s+rm",
]

ENV_CHANGE_PATTERNS = [
    r"\bpip\b.*\binstall\b", r"\bpip\b.*\buninstall\b",
    r"\bbrew\b.*\binstall\b", r"\bbrew\b.*\buninstall\b",
    r"\bapt\b.*\binstall\b", r"\bnpm\b.*\binstall\b",
    r"\bcargo\b.*\binstall\b", r"\bgem\b.*\binstall\b",
    r"\bpyenv\b", r"\bnvm\b.*\buse\b", r"\bnvm\b.*\binstall\b",
    r"\bconda\b.*\binstall\b", r"\bconda\b.*\bactivate\b",
]


# ── Helpers ──


from .scales.s1.surface import run_surface as _run_surface


class PhaseTimer:
    """Lightweight per-call phase timer — emits one log line at the end.

    Usage:
        pt = PhaseTimer(enabled=brain.get_config('phase_timing.enabled', True))
        ... work ...
        pt.mark('recall')
        ... work ...
        pt.mark('candidates')
        ...
        pt.log(brain, 'hook_recall')      # emits a single debug line

    Each mark records elapsed-since-last-mark in milliseconds. Order is
    preserved. Total wall time is also reported.

    Why it exists: hook_recall has ~14 phases (recall, segment boundary,
    candidates, enrichment, edge selection, traces, signal producers,
    Frame build, surface call, etc.). When the end-to-end takes 12 s we
    need to know which phase ate the budget. Single log line keeps the
    daemon log readable without a per-phase line storm.

    Cost when enabled: ~1µs per mark, ~50µs for the final log line. 14
    phases ≈ 14µs of CPU and one log line per recall. Default ON because
    the diagnostic value swamps the overhead — you should know where
    recall's seconds went, always.

    Disable via `brain.get_config('phase_timing.enabled', False)` or via
    `phase_timing.enabled = False` in the config table. When disabled,
    `mark()` and `log()` are no-ops and the timer holds no state — the
    cost goes to zero.
    """

    __slots__ = ('_enabled', '_t0', '_last', '_phases')

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        if not enabled:
            self._t0 = 0.0
            self._last = 0.0
            self._phases = []
            return
        now = time.monotonic()
        self._t0 = now
        self._last = now
        self._phases: list[tuple[str, int]] = []

    def mark(self, label: str) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        self._phases.append((label, int((now - self._last) * 1000)))
        self._last = now

    def log(self, brain, hook_name: str, **extras) -> None:
        """Emit one debug line. extras land as key=value pairs at the end."""
        if not self._enabled:
            return
        total_ms = int((time.monotonic() - self._t0) * 1000)
        body = ' '.join('%s:%dms' % (label, ms) for label, ms in self._phases)
        if extras:
            body += ' ' + ' '.join('%s=%s' % (k, v) for k, v in extras.items())
        msg = '[%s] total:%dms %s' % (hook_name, total_ms, body)
        try:
            brain.log_debug('hook_phase_timing', msg)
        except Exception:
            # Logging must never affect the hook path. If brain.log_debug
            # is unavailable (test harness, unusual init) fall back to print.
            print(msg, flush=True)







# ══════════════════════════════════════════════════════════════════════════════
# HOOK FUNCTIONS — one per hook event
# ══════════════════════════════════════════════════════════════════════════════


def hook_recall(brain, args, graph_changes):
    """Pre-response recall — surfaces brain context before Claude responds.

    Fires on UserPromptSubmit. Returns JSON with additionalContext.

    Flow:
    1. Session setup + store user message
    2. Recall candidates from brain
    3. Segment boundary detection
    4. Build candidates file for encoding agent
    5. Priming check, gap logging, signal producers
    6. S1 Surface (Haiku) → select → expand → format → trace
    7. Return additionalContext or approve

    Phase timing: every recall emits a single 'hook_phase_timing' debug
    line splitting the total wall time across phases. See PhaseTimer.
    Use it to find which phase ate the budget when latencies spike;
    don't waste time guessing where seconds went.

    Gated by `phase_timing.enabled` in brain config (default True).
    Flip to False to silence the per-recall log line entirely; mark/log
    become no-ops and the cost goes to zero.
    """
    pt = PhaseTimer(
        enabled=bool(brain.get_config('phase_timing.enabled', True)))
    user_message = args.get("prompt", "") or args.get("message", "")
    ctx = brain.get_or_create_session(args.get('session_id', ''))
    session_id = ctx.session_id
    _current_stop = str(ctx.stop_counter)

    # Write current stop counter to tmp file — PostToolUse reads this (cross-process)
    try:
        with open('/tmp/brain-%s-current-stop.txt' % session_id, 'w') as _f:
            _f.write(_current_stop)
    except Exception as e:
        brain._log_error('write_current_stop', e, 'hook_recall')

    # ── Explicit feedback detection ──
    # If the user says "useful", "not useful", "garbage", etc., process it as
    # feedback on the most recent ask_operator recall. This is ground truth.
    try:
        _msg_lower = user_message.lower().strip()
        _feedback_map = {
            'useful': 'useful', 'helpful': 'useful', 'yes useful': 'useful',
            'that was useful': 'useful', 'good recall': 'useful',
            'not useful': 'not_useful', 'not helpful': 'not_useful',
            'garbage': 'not_useful', 'irrelevant': 'not_useful',
            'partially useful': 'partially_useful', 'somewhat useful': 'partially_useful',
            'partly': 'partially_useful',
        }
        _matched_feedback = None
        for phrase, signal in _feedback_map.items():
            if _msg_lower.startswith(phrase) or _msg_lower == phrase:
                _matched_feedback = signal
                break
        if _matched_feedback:
            # DEPRECATED 2026-04-03: Precision evaluation removed.
            # Judge + encoder coupling replaces regex/embedding evaluation.
            # Feedback detection kept for future use but doesn't write anywhere.
            brain.log_debug("feedback_detected", "Operator feedback: %s (not evaluated — precision deprecated)" % _matched_feedback)
    except Exception as e:
        brain._log_error('explicit_feedback', e, 'hook_recall')

    # DEPRECATED 2026-04-01: Vocabulary expansion disabled. Vocab migrated to concept nodes
    # which surface through normal recall. Regex expansion added noise without measurable
    # recall improvement (confirmed by decode funnel — 0% impact from vocab expansion).
    from .pipeline_contract import PIPELINE as _PL
    enriched = user_message[:_PL['user_message_query']]
    pt.mark('setup')

    # Recall — logging happens inside brain.recall() (single source of truth)
    try:
        from .pipeline_contract import CANDIDATES_FILE as _CF
        # Pass ctx directly — already loaded above, skip the redundant DB
        # lookup recall would otherwise do. ctx mutations (fatigue, segment)
        # are saved at turn end in post_response_common.
        result = brain.recall(query=enriched, limit=_CF['max_candidates'],
                              ctx=ctx, source='hook')
    except Exception as e:
        brain._log_error('recall_first_attempt', e, 'hook_recall')
        from .pipeline_contract import CANDIDATES_FILE as _CF
        result = brain.recall(query=enriched, limit=_CF['max_candidates'],
                              ctx=ctx, source='hook')

    results = result.get("results", [])
    pt.mark('recall')

    # recall_ref removed — use stop counter for tmp file naming and trace refs
    recall_ref = '%s-%s' % (session_id[:8], _current_stop)

    # Segment boundary detection
    segment_note = None
    try:
        query_emb = result.get("_query_embedding")
        if query_emb:
            seg = brain.check_segment_boundary(query_emb, session_id)
            if seg.get("is_boundary"):
                segment_note = "--- CONTEXT SHIFT (segment %d, sim=%.2f) ---" % (
                    seg["segment_id"], seg["similarity"])
            for r in results:
                brain.add_to_segment(r.get("id", ""), session_id)
    except Exception as e:
        brain._log_error('segment_boundary', e, 'hook_recall')
    pt.mark('segment')

    # Gap detection (needed by candidates file + later logging)
    gap = result.get('_gap') if isinstance(result, dict) else None

    # Write candidates to file for recall agent hook (LLM distillation + encoding agent)
    # Encoding agent needs ALL candidates (including previously surfaced) for revision.
    # Session dedup happens only at distiller stage, not here.
    try:
        from .pipeline_contract import CANDIDATES_FILE
        candidates_path = '/tmp/brain-{}-recall-candidates.json'.format(session_id)
        capped = results[:CANDIDATES_FILE['max_candidates']]

        # Batch enrichment: one call, 5 queries for all 25 nodes
        node_ids = [r.get("id", "") for r in capped if r.get("id")]
        rich_nodes = brain.get_node(node_ids)

        # Edge selection: query-aware scoring (strategy D)
        # Get query embedding + prior turn embeddings for multi-turn blend
        import numpy as np
        _query_emb = result.get("_query_embedding")
        _query_vec = None
        _prior_vecs = []
        if _query_emb is not None:
            _query_vec = np.frombuffer(_query_emb, dtype=np.float32) if isinstance(_query_emb, bytes) else np.array(_query_emb, dtype=np.float32)
            # Get prior user messages for multi-turn context
            try:
                _prior_turns = brain._trace_dal.get_session_turns(session_id, limit=4)
                _user_turns = [t for t in _prior_turns if t.get('role') == 'user'][:2]
                for _t in _user_turns:
                    _text = (_t.get('content') or '')[:500]
                    if _text and len(_text) > 5:
                        _blob = embed_query(_text)
                        if _blob:
                            _prior_vecs.append(np.frombuffer(_blob, dtype=np.float32))
            except Exception as _e:
                brain._log_error('edge_select_prior_vecs', _e, 'embedding prior turns')

        candidates_data = []
        for r in capped:
            nid = r.get("id", "")
            node_data = rich_nodes.get(nid)
            if not node_data:
                # Fallback if node disappeared between recall and enrichment
                node_data = {
                    "id": nid, "type": r.get("type", ""),
                    "title": r.get("title", ""), "content": r.get("content", ""),
                    "confidence": r.get("confidence", 0), "locked": r.get("locked", False),
                    "created_at": r.get("created_at"), "revised_at": r.get("revised_at"),
                }
            # Query-aware edge selection (S1 intelligence)
            # Encoding agent gets all connections (no select_edges call).
            # Render gets the selected subset via edge_limit in config.
            if _query_vec is not None and node_data.get('connections'):
                from .scales.s1.surface_contract import select_edges
                node_data['connections'] = select_edges(
                    node_data['connections'], _query_vec,
                    limit=10,  # keep 10, render truncates to 3
                    prior_vecs=_prior_vecs, brain_conn=brain.conn, brain=brain)
            # Attach recall-specific fields (not in DB — from scoring pipeline)
            node_data["score"] = r.get("effective_activation", 0)
            node_data["discovery"] = r.get("_discovery", "embedding")
            # Include full graph neighborhood for encoding agent
            # (encoding agent reads from /tmp file, sees all connections)
            node_data["_all_connections"] = rich_nodes.get(nid, {}).get('connections', [])
            graph = r.get("_graph", {})
            if graph:
                node_data["_graph"] = graph
            elif r.get("_neighbors"):
                node_data["_graph"] = {"degree_1": r["_neighbors"], "degree_2": [], "degree_3": []}
            candidates_data.append(node_data)
        # v8.8: Include vocab context — connectors surfaced separately
        # DEPRECATED 2026-04-01: vocab_context removed (vocab → concept migration)

        pt.mark('candidates')

        # Recent messages for surface context — from traces
        recent_messages = []
        try:
            turns = brain._trace_dal.get_session_turns(session_id, limit=5)
            recent_messages = [{"role": t['role'], "content": (t['content'] or '')[:_PL['recent_message_content']]}
                               for t in turns]
        except Exception as _e:
            brain._log_error('surface_recent_messages', _e, 'fetching recent messages from traces')
        pt.mark('traces')

        # Session context from last encoding agent run (per-session — no leak)
        session_context = brain.session_context_for(session_id)
        pt.mark('session_ctx')

        with open(candidates_path, 'w') as f:
            json.dump({
                "user_message": user_message,
                "session_context": session_context,
                "candidates": candidates_data,
                "segment_note": segment_note,
                "gap": gap.get("query") if gap else None,
                "recent_messages": recent_messages,
                "recall_ref": recall_ref,
            }, f, default=str)
        pt.mark('file_write')
    except Exception as e:
        brain._log_error('recall_candidates_write', e, 'Failed to write candidates file')

    if not results:
        brain.save()
        pt.log(brain, 'hook_recall', n_results=0)
        return {"json": {"decision": "approve"}}

    # Gap detection: log gaps for trend analysis
    if gap:
        try:
            from .dal import LogsDAL
            LogsDAL(brain.logs_conn).log_gap(gap['query'], gap.get('top_score', 0), session_id)
        except Exception as e:
            brain._log_error('hook_recall_gap_log', e, 'Failed to log recall gap')

    # ── PRODUCE: seed the signal queue ──
    from .dal_signal_queue import SignalQueueDAL
    from .surface_assembler import SurfaceAssembler
    from .signal_producers import (
        produce_reminders, produce_encoding_gap,
        produce_system_health, produce_integrity,
    )

    sq_dal = SignalQueueDAL(brain.logs_conn)
    produce_reminders(brain, sq_dal)
    produce_encoding_gap(brain, sq_dal, ctx=ctx)
    produce_system_health(brain, sq_dal)
    produce_integrity(brain, sq_dal)
    pt.mark('signals')

    # ── ASSEMBLE: budget-aware output ──
    assembler = SurfaceAssembler(sq_dal, budget_chars=6000)
    # Command hook: write candidates file, return approve + session_id.
    # The thin client reads the file, calls LLM to distill, returns context.
    # Dashboard logging happens in the thin client — one source of truth.
    brain.save()

    # ── S1 Surface: push relevant memories into awareness ──
    # 2026-05-02 (Frame Phase 2): Frame is the canonical session prior.
    # ctx.get_frame(brain) builds it from brain state + this session's
    # encoder journal. Surface receives it as the "Partnership context:"
    # block. If Frame Constructor raises, the error is logged loudly and
    # surface runs without partnership context (explicit degraded mode,
    # no silent fallback to a different layout). See docs/FRAME-DESIGN.md.
    additional_context = None
    try:
        try:
            _frame = ctx.get_frame(brain)
        except Exception as _frame_err:
            brain._log_error('frame_build_failed', _frame_err,
                             'Frame Constructor failed — surface runs without partnership context')
            _frame = ''
        pt.mark('frame')
        # Thread the same PhaseTimer into surface so it splits the surface
        # phase into haiku / id_resolve / spread / render / trace marks
        # on the same final log line. Without this we only know "surface
        # took N ms" without knowing which sub-phase ate the budget.
        additional_context = _run_surface(
            brain, ctx, candidates_data, user_message,
            recent_messages=recent_messages if 'recent_messages' in dir() else [],
            result=result, enriched=enriched, results=results,
            recall_ref=recall_ref, session_id=session_id,
            graph_changes=graph_changes,
            query_vec=_query_vec, prior_vecs=_prior_vecs,
            frame=_frame, pt=pt)
    except Exception as _surface_err:
        brain._log_error('daemon_surface', _surface_err,
                         'S1 Surface failed in daemon (query=%s)' % user_message[:100])

    pt.log(brain, 'hook_recall', n_results=len(results))
    if additional_context:
        return {"json": {"additionalContext": additional_context}, "session_id": session_id}
    else:
        return {"json": {"decision": "approve"}, "session_id": session_id}






def _hebbian_strengthen(brain, session_id, stop_counter):
    """Strengthen co_accessed edges between surface-selected nodes.

    Only nodes the S1 Surface selected get edges — meaningful co-activation.

    Every invocation emits an outcome counter to brain stats — previous
    "return silently" paths hid a filename bug for months. Now every call
    has a visible tally, so "why did Hebbian never run?" becomes answerable.

    `stop_counter` is the same counter surface.py used when writing the
    file — both producer and consumer must agree on the path so consecutive
    turns don't read each other's files.
    """
    outcome = {'file_missing': 0, 'few_ids': 0, 'unresolved': 0, 'edges': 0}
    surface_path = '/tmp/brain-%s-%d-surface-selected.json' % (session_id, stop_counter)
    try:
        if not os.path.exists(surface_path):
            outcome['file_missing'] = 1
            return

        with open(surface_path) as f:
            surface_ids = json.load(f).get('selected_ids', [])
        if len(surface_ids) < 2:
            outcome['few_ids'] = 1
            return

        # Resolve short IDs to full IDs
        from servers.dal import NodeDAL
        dal = NodeDAL(brain.conn)
        full_ids = []
        for sid in surface_ids:
            full_id = dal.resolve_id(sid)
            if full_id:
                full_ids.append(full_id)
        if len(full_ids) < 2:
            outcome['unresolved'] = 1
            return

        from .brain_constants import LEARNING_RATE
        from .dal import GraphDAL
        gdal = GraphDAL(brain.conn)
        for i in range(len(full_ids)):
            for j in range(i + 1, min(len(full_ids), i + 8)):
                try:
                    # Stage 1B: connect_typed is now idempotent upsert (no
                    # auto-strengthen). For Hebbian co-access we explicitly
                    # ensure the edge exists, then strengthen its weight.
                    brain.connect_typed(full_ids[i], full_ids[j],
                                        relation='co_accessed', weight=LEARNING_RATE * 0.15,
                                        edge_type='co_accessed', description='surface-selected')
                    gdal.strengthen_relation(full_ids[i], full_ids[j], 'co_accessed')
                    outcome['edges'] += 1
                except Exception as e:
                    brain._log_error('hebbian_edge', e, 'creating co_accessed edge')
    finally:
        # Durable tally so "did Hebbian run?" is answerable without a debugger.
        # If the log itself fails we surface that — silent pass would defeat
        # the whole point of the visibility this block was added for.
        try:
            brain.log_debug('hebbian_run', 'post_response_common', **outcome)
        except Exception as _le:
            try:
                brain._log_error('hebbian_log_outcome', _le,
                                 'log_debug failed; outcome=%s' % outcome)
            except Exception:
                pass  # last-resort safety; if even _log_error fails, swallow


def _s1e_chain_id(session_id, counter):
    """Generate S1 encode chain ID for delta trace."""
    from .session_context import SessionContext
    return SessionContext(session_id=session_id, stop_counter=counter).s1e_chain()


def post_response_common(brain, session_id, user_message, assistant_response):
    """Shared post-response path: S0 traces, Hebbian strengthening, heartbeat,
    stop counter increment. Used by prod Stop hook and by the eval harness —
    same code, same ordering, one source of truth.

    Returns the SessionContext after increment.
    """
    from .pipeline_contract import PIPELINE as _PL
    ctx = brain.get_or_create_session(session_id)
    assistant_response = (assistant_response or "")[:_PL['assistant_response_store']]

    # S0 traces (using SessionContext for chain IDs)
    try:
        recall_chain = ctx.s1r_chain()
        brain._trace_dal.append(
            chain_id=ctx.s0_chain(), scale='s0', event_type='K',
            ref_type='user_message',
            summary=user_message[:200] if user_message else '',
            metadata={'content': user_message[:4000] if user_message else '',
                      'recall_chain': recall_chain} if user_message else None,
            session_id=session_id)
        brain._trace_dal.append(
            chain_id=ctx.s0_chain(), scale='s0', event_type='delta',
            ref_type='assistant_message',
            summary=assistant_response[:200] if assistant_response else '',
            metadata={'content': assistant_response[:4000]} if assistant_response else None,
            session_id=session_id)
    except Exception as e:
        brain._log_error('trace_s0', e, 'post_response_common')

    # Hebbian strengthening — pass current stop_counter so it reads the
    # surface file written by THIS turn's recall (counter hasn't been
    # incremented yet — that happens below).
    try:
        _hebbian_strengthen(brain, session_id, ctx.stop_counter)
    except Exception as e:
        brain._log_error('hebbian_surface_selected', e, 'post_response_common')

    # Heartbeat — mutates ctx in memory; autosave loop persists
    try:
        brain.record_message(ctx)
    except Exception as e:
        brain._log_error('record_message', e, 'post_response_common')

    # Stop counter increment — in-memory; autosave persists every minute.
    # If the daemon crashes mid-turn, at most ~60s of stop_counter / fatigue
    # is lost; encoding gates are approximate so this is acceptable.
    ctx.increment_stop()
    return ctx


def hook_post_response_track(brain, args, graph_changes):
    """Stop event — store exchange, write traces, Hebbian strengthening, gate encoder.

    Flow:
    1. Shared post-response path (post_response_common)
    2. Gate encoding agent (every 5th stop, background thread) — prod-only
    """
    ctx = post_response_common(
        brain,
        args.get('session_id', ''),
        args.get("prompt", "") or args.get("message", ""),
        args.get("last_assistant_message", "") or "",
    )
    session_id = ctx.session_id
    encoding_status = ""
    # acquired_for_spawn tracks the window between lock.acquire() and the
    # successful return of run_in_background (which transfers lock ownership
    # to the spawned thread, whose finally releases it). If we acquire but
    # then fail before the transfer (import error, Thread.start() raises),
    # the outer finally below recovers the lock — otherwise the lock would
    # be held indefinitely and ALL future encoding cycles would silently skip.
    acquired_for_spawn = False
    try:
        counter = ctx.stop_counter
        position = counter % 5

        if position == 0:
            if not _encoding_lock.acquire(blocking=False):
                encoding_status = "encoding skipped (previous still running)"
                print("[brain-hooks] Encoding agent skipped — previous run still active", flush=True)
            else:
                acquired_for_spawn = True
                from .scales.runner import run_in_background
                from .scales.s1.encode import run_encoding
                run_in_background(
                    name='s1e', brain_db_path=brain.db_path,
                    session_id=session_id, counter=counter,
                    lock=_encoding_lock, run_fn=run_encoding,
                    trace_scale='s1', trace_chain_fn=_s1e_chain_id)
                # Thread.start() returned → ownership transferred. Thread's
                # finally is now responsible for the release.
                acquired_for_spawn = False
                encoding_status = "encoding started (background)"
        else:
            encoding_status = "encoding %d/5" % position
    except Exception as e:
        brain._log_error('encoding_agent_gate', e, 'Stop hook')
        encoding_status = "encoding error: %s" % str(e)[:50]
    finally:
        # Recovery release: we acquired the lock but never successfully
        # handed ownership to a background thread. Without this release,
        # the daemon's encoder would be permanently jammed.
        if acquired_for_spawn:
            try:
                _encoding_lock.release()
            except Exception:
                pass
            brain._log_error(
                'encoding_lock_leak_recovered',
                RuntimeError("encoding lock acquired but spawn failed; released to prevent permanent jam"),
                'session=%s counter=%s' % (session_id, ctx.stop_counter))

    brain.save()
    return {"output": "(stored + %s)" % encoding_status}








def hook_idle_maintenance(brain, args, graph_changes):
    """Idle maintenance — dream, consolidate, heal, tune, reflect.

    Fires on Notification(idle_prompt). Output stored as pending message.
    """
    import datetime
    start_time = datetime.datetime.now()
    print("[brain-hooks] idle_maintenance STARTED at %s" % start_time.isoformat(), flush=True)
    output = []

    # 1. Dream — DISABLED 2026-04-08
    # dream() created random-walk emergent_bridge edges and thought/intuition nodes.
    # S2 community detection + small-cluster linking replaces this with
    # structure-aware connections instead of random walks.

    # 2. Consolidate — DISABLED 2026-04-08
    # consolidate() boosted well-connected nodes and ran auto_discover_evolutions
    # (already paused). S2 confidence recalibration replaces node boosting.
    # S2 dedup/synthesis replaces evolution discovery.

    # 3. Self-healing — DISABLED 2026-04-08
    # auto_heal() was proto-S2: dedup, auto-lock, correction consolidation,
    # confidence adjustments. S2 integration units replace all of these.
    # Keeping the code in brain_evolution.py for reference during S2 build.

    # 3b. Vocab cleanup — prune junk vocabulary nodes that pollute recall
    try:
        # Strategy 1: auto-detected junk (title or content has "auto-detected")
        junk_vocab = brain.conn.execute("""
            SELECT id, title FROM nodes
            WHERE type = 'vocabulary' AND archived = 0
            AND (content LIKE '%auto-detected%' OR title LIKE '%auto-detected%')
        """).fetchall()

        # Strategy 2: single-word vocab nodes with no real definition
        # These match everything in cosine similarity and bury real results
        single_word_junk = brain.conn.execute("""
            SELECT id, title FROM nodes
            WHERE type = 'vocabulary' AND archived = 0
            AND title NOT LIKE '% %'
            AND (content IS NULL OR content = '' OR LENGTH(content) < 30)
            AND confidence < 0.5
        """).fetchall()

        all_junk = {nid: title for nid, title in junk_vocab + single_word_junk}
        if all_junk:
            from .dal import NodeDAL
            _node_dal = NodeDAL(brain.conn)
            for nid in all_junk:
                _node_dal.purge(nid)
            output.append("VOCAB CLEANUP: pruned %d junk nodes" % len(all_junk))
            graph_changes.append("VOCAB_CLEANUP: %d pruned" % len(all_junk))
    except Exception as e:
        output.append("VOCAB CLEANUP ERROR: %s" % e)

    # 3c. Auto-tune — DISABLED 2026-04-08
    # auto_tune() adjusted brain parameters adaptively. S2 interaction
    # evolution units will replace this with trace-informed optimization.

    # 3d. Edge decay — apply half-life decay to auto-generated edges
    try:
        from .dal import GraphDAL
        graph_dal = GraphDAL(brain.conn)
        decay_result = graph_dal.decay_edges()
        decayed = decay_result.get('decayed', 0)
        pruned = decay_result.get('pruned', 0)
        if decayed or pruned:
            parts = []
            if decayed:
                parts.append("%d edges decayed" % decayed)
            if pruned:
                parts.append("%d edges pruned" % pruned)
            output.append("EDGE DECAY: " + ", ".join(parts))
            graph_changes.append("EDGE_DECAY: %s" % ", ".join(parts))
            for rel, stats in decay_result.get('by_type', {}).items():
                output.append("  %s: %d decayed, %d pruned" % (rel, stats['decayed'], stats['pruned']))
    except Exception as e:
        output.append("EDGE DECAY ERROR: %s" % e)

    # consolidation detection — REMOVED 2026-04-05 (pending_consolidation table dropped)

    # message_stream expiry REMOVED 2026-04-05 — table deleted, traces are source of truth

    # 3e. S2: Graph integration (coordinator decides what runs)
    try:
        from .scales.s2.coordinator import run_s2
        s2_results = run_s2(brain)
        for unit_name, result in s2_results.items():
            # Per-unit isolation — a buggy formatter for one unit shouldn't
            # swallow output from other units.
            try:
                if not isinstance(result, dict):
                    output.append("S2 %s: unexpected result shape (%s)" % (
                        unit_name.upper(), type(result).__name__))
                    continue
                if result.get('error'):
                    output.append("S2 %s ERROR: %s" % (unit_name.upper(), result['error']))
                    continue
                if result.get('skipped'):
                    output.append("S2 %s: skipped (%s)" % (unit_name.upper(), result['skipped']))
                    continue

                actions = result.get('actions', result.get('classified', 0)) or 0

                if unit_name == 'edge_family_integration':
                    classified = result.get('classified', 0) or 0
                    families = result.get('families', 0) or 0
                    if classified > 0:
                        output.append("S2 EDGE FAMILIES: classified %d new types into %d families" % (
                            classified, families))

                elif unit_name == 'consolidation':
                    clusters = result.get('clusters', 0)
                    # Accept either int count or list of cluster dicts.
                    cluster_count = len(clusters) if isinstance(clusters, (list, tuple)) else int(clusters or 0)
                    if cluster_count > 0:
                        output.append("S2 CONSOLIDATION: %d clusters found" % cluster_count)
                        stats = result.get('stats') or {}
                        class_counts = stats.get('class_counts') or {}
                        if class_counts:
                            output.append("  %s" % ', '.join(
                                '%d %s' % (v, k) for k, v in class_counts.items()))

                elif unit_name == 'community_detection':
                    communities = result.get('communities', 0) or 0
                    if actions > 0:
                        output.append("S2 COMMUNITY: %d communities, %d actions" % (
                            communities, actions))
                        graph_changes.append("S2_COMMUNITY: %d communities" % communities)
                    else:
                        output.append("S2 COMMUNITY: no changes (%d communities)" % communities)

                elif unit_name == 'healer':
                    nodes_healed = result.get('nodes_healed', 0) or 0
                    fields_written = result.get('fields_written', 0) or 0
                    skipped = result.get('skipped', 0) or 0
                    proposals = result.get('proposals', 0) or 0
                    if nodes_healed > 0 or fields_written > 0:
                        output.append("S2 HEALER: %d nodes healed, %d fields written (skipped %d)" % (
                            nodes_healed, fields_written, skipped))
                        graph_changes.append("S2_HEALER: %d fields" % fields_written)
                    elif proposals > 0:
                        output.append("S2 HEALER: %d proposals, 0 written" % proposals)

                else:
                    # Unknown unit — render a terse default rather than silently dropping it.
                    if actions > 0:
                        output.append("S2 %s: %d actions" % (unit_name.upper(), actions))
            except Exception as fmt_err:
                output.append("S2 %s format error: %s" % (unit_name.upper(), fmt_err))
    except Exception as e:
        output.append("S2 ERROR: %s" % e)

    # 4. Reflection prompts — DISABLED 2026-04-08
    # prompt_reflection() and auto_generate_self_reflection() were proto-S2.
    # S2 encoding and S3 reasoning replace reflection with trace-based analysis.

    # 5. Self-reflection — DISABLED 2026-04-08 (see above)

    # 6. Backfill summaries
    try:
        backfill = brain.backfill_summaries(batch_size=50)
        bf_count = backfill.get("updated", 0)
        if bf_count > 0:
            output.append("SUMMARIES: backfilled %d nodes" % bf_count)
    except Exception as e:
        brain._log_error('backfill_summaries', e, 'idle_maintenance')

    # 7. Backfill ALL vectors (primary, situation, title, high_meta, other_meta, edge_context)
    # v23: unified backfill replaces backfill_embeddings. Runs single-threaded after S2.
    try:
        vec_result = brain.backfill_vectors(batch_size=30)
        if isinstance(vec_result, dict) and not vec_result.get('error'):
            total = sum(v for k, v in vec_result.items() if isinstance(v, int))
            if total > 0:
                parts = ['%s:%d' % (k, v) for k, v in vec_result.items() if isinstance(v, int) and v > 0]
                output.append("VECTORS: backfilled %d (%s)" % (total, ', '.join(parts)))
                graph_changes.append("VECTORS: %d backfilled" % total)
    except Exception as e:
        brain._log_error('backfill_vectors', e, 'idle_maintenance')

    # 8. prune_irrelevant_quotes removed 2026-04-13 — fix at encoding time, not after.

    # 9. DB maintenance (prune old logs, clean orphans)
    try:
        from servers.dal import LogsDAL
        logs_dal = LogsDAL(brain.logs_conn)
        maint = logs_dal.run_maintenance(graph_conn=brain.conn)
        total_pruned = maint.get('total_pruned', 0)
        total_orphans = maint.get('total_orphans', 0)
        if total_pruned > 0 or total_orphans > 0:
            parts = []
            if total_pruned:
                parts.append("%d log rows pruned" % total_pruned)
            if total_orphans:
                parts.append("%d orphans cleaned" % total_orphans)
            output.append("DB MAINTENANCE: " + ", ".join(parts))
            # Log details in debug mode
            for k, v in maint.items():
                if v > 0 and k not in ('total_pruned', 'total_orphans'):
                    output.append("  %s: %d" % (k, v))
    except Exception as e:
        output.append("DB MAINTENANCE ERROR: %s" % e)

    # 10. assess_session_health removed 2026-04-13 — information not action.

    # 11. Deep integrity audit
    try:
        from .signal_producers import deep_integrity_audit
        findings = deep_integrity_audit(brain)
        if findings:
            severe = [f for f in findings if f.get("severity") in ("high", "medium")]
            if severe:
                output.append("")
                output.append("INTEGRITY AUDIT (%d finding(s), %d need attention):" % (len(findings), len(severe)))
                for f in severe[:5]:
                    output.append("  [%s] %s: %s" % (f["severity"], f["type"], f["message"]))
                graph_changes.append("INTEGRITY: %d findings" % len(findings))
    except Exception as e:
        output.append("INTEGRITY AUDIT ERROR: %s" % e)

    # Log to dashboard (not additionalContext — idle maintenance is operational, not conversational)
    if output:
        try:
            from hooks.scripts.hook_common import log_hook_output
            log_hook_output("IDLE", output_text="\n".join(output))
        except Exception as e:
            brain._log_error('log_idle_output', e, 'idle_maintenance')

    # Log so we can verify idle fires and what it does
    import datetime
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print("[brain-hooks] idle_maintenance COMPLETED in %.1fs — %d output lines" % (elapsed, len(output)), flush=True)
    for line in output:
        print("[brain-hooks]   idle: %s" % line, flush=True)

    brain.save()
    return {"output": ""}  # Notification stdout invisible


def hook_pre_edit(brain, args, graph_changes):
    """PreToolUse(Edit|Write) — surface brain rules before file edits.

    Returns JSON {"decision":"approve","reason":"..."}.
    """
    filename = args.get("filename", "")
    tool_name = args.get("tool_name", "Edit")

    if not filename:
        return {"json": {"decision": "approve"}}

    sid = args.get("session_id", "")
    ctx = brain.get_or_create_session(sid) if sid else None
    try:
        data = brain.pre_edit(file=filename, tool_name=tool_name, ctx=ctx)
    except Exception as e:
        brain._log_error('pre_edit', e, 'hook_pre_edit')
        return {"json": {"decision": "approve"}}

    suggestions = data.get("suggestions", [])
    procedures = data.get("procedures", [])
    context_files = data.get("context_files", [])
    encoding = data.get("encoding", {})
    debug_enabled = data.get("debug_enabled", False)

    # Change impact maps
    change_impacts = []
    try:
        change_impacts = brain.get_change_impact(filename)
    except Exception as e:
        brain._log_error('get_change_impact', e, 'hook_pre_edit')

    encoding_warning = _format_encoding_warning(encoding)

    if not suggestions and not procedures and not context_files and not change_impacts:
        if encoding_warning:
            return {"json": {"decision": "approve", "reason": encoding_warning}}
        return {"json": {"decision": "approve"}}

    context = _format_suggestions(filename, suggestions, procedures, context_files,
                                  change_impacts, encoding_warning)

    # Debug logging
    if debug_enabled:
        try:
            node_ids = [s.get("id", "") for s in suggestions if s.get("type") != "procedure"]
            node_ids += [p.get("id", "") for p in procedures]
            brain.log_debug(
                event_type="pre_edit",
                source="hook",
                file_target=filename,
                suggestions_served=len([s for s in suggestions if s.get("type") != "procedure"]),
                procedures_served=len(procedures),
                node_ids_served=json.dumps(node_ids),
                metadata=json.dumps({"tool": tool_name}),
            )
        except Exception as e:
            brain._log_error('debug_log_pre_edit', e, 'hook_pre_edit')

    brain.save()
    return {"json": {"decision": "approve", "reason": context}}


def hook_pre_bash_safety(brain, args, graph_changes):
    """PreToolUse(Bash) — safety check for destructive commands.

    Returns JSON {"decision":"approve"|"block","reason":"..."}.
    """
    command = args.get("command", "")

    try:
        result = brain.safety_check(command)
    except Exception as e:
        brain._log_error('safety_check', e, 'hook_pre_bash_safety')
        return {"json": {
            "decision": "approve",
            "reason": "[BRAIN] \u26a0\ufe0f Safety check failed — proceed carefully. [/BRAIN]",
        }}

    critical_matches = result.get("critical_matches", [])
    warnings = result.get("warnings", [])

    if critical_matches:
        lines = ["[BRAIN] \u26a0\ufe0f SAFETY: This command may affect critical brain-tracked resources:"]
        lines.append("")
        for cm in critical_matches[:5]:
            title = cm.get("title", "")[:80]
            content = cm.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append("  [%s] %s" % (cm.get("type", "?"), title))
            lines.append("    %s" % content)
            lines.append("")
        lines.append("Review the above before proceeding. This command has been BLOCKED.")
        lines.append("[/BRAIN]")

        # Log brain-Claude conflict
        try:
            rule_title = critical_matches[0].get("title", "")[:120] if critical_matches else "safety rule"
            brain.log_conflict(
                hook_name="pre_bash_safety",
                brain_decision="block",
                rule_node_id=critical_matches[0].get("id") if critical_matches else None,
                rule_title=rule_title,
                claude_action=command[:200],
            )
        except Exception as e:
            brain._log_error('log_conflict', e, 'hook_pre_bash_safety')

        return {"json": {"decision": "block", "reason": "\n".join(lines)}}

    elif warnings:
        lines = ["[BRAIN] \u26a0\ufe0f WARNING: Destructive command detected. Relevant brain context:"]
        lines.append("")
        for w in warnings[:5]:
            title = w.get("title", "")[:80]
            content = w.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append("  [%s] %s" % (w.get("type", "?"), title))
            lines.append("    %s" % content)
            lines.append("")
        lines.append("Proceed carefully — verify this command is intentional.")
        lines.append("[/BRAIN]")
        return {"json": {"decision": "approve", "reason": "\n".join(lines)}}

    else:
        return {"json": {
            "decision": "approve",
            "reason": "[BRAIN] \u26a0\ufe0f Destructive command detected. No safety rules match, but proceed carefully. [/BRAIN]",
        }}


def hook_session_end(brain, args, graph_changes):
    """SessionEnd — session synthesis + reflection + consolidation + clean shutdown."""
    # synthesize_session removed 2026-04-13

    # reflect_for_next_claude removed 2026-04-13 — boot nodes nothing read.
    # consolidate() removed 2026-04-13 — wrote to deprecated stability field, created noise.

    # Final save + drop the cached SessionContext for this session so
    # the in-memory cache doesn't accumulate ended-session entries.
    sid = args.get("session_id", "")
    if sid:
        try:
            brain.discard_session_context(sid)
        except Exception as e:
            brain._log_error('session_context_discard', e, 'hook_session_end')

    brain.save()
    # Note: the hook client sends "shutdown" separately after this returns
    return {"output": ""}


def hook_stop_failure_log(brain, args, graph_changes):
    """StopFailure — logs API failures. miss_log table dropped, use debug_log."""
    error_type = args.get("error", "unknown")
    error_details = args.get("error_details", "")
    try:
        brain.log_debug("stop_failure", "API error: %s — %s" % (error_type, str(error_details)[:200]))
    except Exception:
        pass
    return {"output": ""}


def hook_config_change_host(brain, args, graph_changes):
    """ConfigChange — detects host environment changes.

    Stdout invisible. Stores output as pending message.
    """
    source = args.get("source", "unknown")
    file_path = args.get("file_path", "")

    try:
        env_result = brain.scan_host_environment()
        changes = env_result.get("changes", {}) if env_result else {}

        if changes:
            output_lines = ["[BRAIN] HOST ENVIRONMENT CHANGED:"]
            for key, change in changes.items():
                old_val = change.get("old", "?")
                new_val = change.get("new", "?")
                output_lines.append("  %s: %s \u2192 %s" % (key, old_val, new_val))
            output_lines.append("  Trigger: config change in %s" % source)
            if file_path:
                output_lines.append("  File: %s" % file_path)
            output_lines.append("")
            output_lines.append("Review arch_constraint and capability nodes that may be affected.")
            output_lines.append("[/BRAIN]")

            try:
                from hooks.scripts.hook_common import log_hook_output
                log_hook_output("HOST_ENV", output_text="\n".join(output_lines))
            except Exception as e:
                brain._log_error('log_host_env_output', e, 'hook_config_change_host')
            graph_changes.append("HOST: environment changed (%d items)" % len(changes))
            brain.save()
    except Exception as e:
        brain._log_error('scan_host_environment', e, 'hook_config_change_host')

    return {"output": ""}


def hook_post_bash_host_check(brain, args, graph_changes):
    """PostToolUse(Bash) — detects env changes after pip/brew/etc.

    Logs to dashboard, not additionalContext.
    """
    try:
        env_result = brain.scan_host_environment()
        changes = env_result.get("changes", {}) if env_result else {}

        if changes:
            command = args.get("command", "")
            output_lines = ["HOST ENVIRONMENT CHANGED (after bash):"]
            for key, change in changes.items():
                old_val = change.get("old", "?")
                new_val = change.get("new", "?")
                output_lines.append("  %s: %s \u2192 %s" % (key, old_val, new_val))
            output_lines.append("  Command: %s" % command[:100])

            try:
                from hooks.scripts.hook_common import log_hook_output
                log_hook_output("HOST_ENV", output_text="\n".join(output_lines))
            except Exception as e:
                brain._log_error('log_host_env_output', e, 'hook_post_bash_host_check')
            graph_changes.append("HOST: env changed after bash (%d items)" % len(changes))
            brain.save()
    except Exception as e:
        brain._log_error('scan_host_environment', e, 'hook_post_bash_host_check')

    return {"output": ""}


def hook_worktree_context(brain, args, graph_changes):
    """WorktreeCreate — tracks git branch/worktree info in brain."""
    worktree_name = args.get("name", "unknown")
    cwd = args.get("cwd", "")

    # Detect git branch from cwd
    branch = "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
    except Exception as e:
        brain._log_error('git_branch_detect', e, 'hook_worktree_context')

    brain.set_config("current_worktree", worktree_name)
    brain.set_config("current_branch", branch)
    brain.set_config("current_cwd", cwd)

    try:
        brain.scan_host_environment()
    except Exception as e:
        brain._log_error('scan_host_environment', e, 'hook_worktree_context')

    graph_changes.append("WORKTREE: created %s (branch: %s)" % (worktree_name, branch))

    output_lines = [
        "[BRAIN] GIT CONTEXT:",
        "  Worktree: " + worktree_name,
        "  Branch: " + branch,
        "  CWD: " + cwd,
        "[/BRAIN]",
    ]

    brain.save()
    return {"output": "\n".join(output_lines)}


def hook_worktree_cleanup(brain, args, graph_changes):
    """WorktreeRemove — clears worktree context from brain config."""
    old_worktree = brain.get_config("current_worktree", "")
    brain.set_config("current_worktree", "")
    brain.set_config("current_branch", "")
    brain.set_config("current_cwd", "")
    if old_worktree:
        graph_changes.append("WORKTREE: removed %s" % old_worktree)
    brain.save()
    return {"output": ""}
