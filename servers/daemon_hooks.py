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
from datetime import datetime, timezone

# Encoding agent: at most 1 running at a time. Non-blocking acquire — skip if busy.
_encoding_lock = threading.Lock()

# ── Constants (canonical definitions in brain_voice.py) ──

from servers.brain_voice import BrainVoice

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


from .scales.dispatch import daemon_tcp_send as _daemon_tcp_send
from .scales.s1.surface import run_surface as _run_surface







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
    """
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

    # Store last user message for operator voice capture
    try:
        from .pipeline_contract import PIPELINE as _PL
        brain.set_config("last_user_message", user_message[:_PL['user_message_store']])
    except Exception as e:
        brain._log_error('set_last_user_message', e, 'hook_recall')

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
    enriched = user_message[:_PL['user_message_query']]

    # Recall — logging happens inside brain.recall() (single source of truth)
    try:
        from .pipeline_contract import CANDIDATES_FILE as _CF
        result = brain.recall(query=enriched, limit=_CF['max_candidates'],
                              session_id=session_id, source='hook')
    except Exception as e:
        brain._log_error('recall_first_attempt', e, 'hook_recall')
        from .pipeline_contract import CANDIDATES_FILE as _CF
        result = brain.recall(query=enriched, limit=_CF['max_candidates'],
                              session_id=session_id, source='hook')

    results = result.get("results", [])

    # recall_ref removed — use stop counter for tmp file naming and trace refs
    recall_ref = '%s-%s' % (session_id[:8], _current_stop)

    # Segment boundary detection
    segment_note = None
    try:
        query_emb = result.get("_query_embedding")
        if query_emb:
            seg = brain.check_segment_boundary(query_emb)
            if seg.get("is_boundary"):
                segment_note = "--- CONTEXT SHIFT (segment %d, sim=%.2f) ---" % (
                    seg["segment_id"], seg["similarity"])
            for r in results:
                brain.add_to_segment(r.get("id", ""))
    except Exception as e:
        brain._log_error('segment_boundary', e, 'hook_recall')

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
                from .scales.s1.surface_contract import select_edges
                _prior_turns = brain._trace_dal.get_session_turns(session_id, limit=4)
                _user_turns = [t for t in _prior_turns if t.get('role') == 'user'][:2]
                from servers.embedder import embed as _emb
                for _t in _user_turns:
                    _text = (_t.get('content') or '')[:500]
                    if _text and len(_text) > 5:
                        _blob = _emb(_text)
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
                    session=ctx, limit=10,  # keep 10, render truncates to 3
                    prior_vecs=_prior_vecs, brain_conn=brain.conn)
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

        # Recent messages for surface context — from traces
        recent_messages = []
        try:
            turns = brain._trace_dal.get_session_turns(session_id, limit=5)
            recent_messages = [{"role": t['role'], "content": (t['content'] or '')[:_PL['recent_message_content']]}
                               for t in turns]
        except Exception as _e:
            brain._log_error('surface_recent_messages', _e, 'fetching recent messages from traces')

        # Session context from last encoding agent run (S1 Surface needs this)
        session_context = brain.session_context

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
    except Exception as e:
        brain._log_error('recall_candidates_write', e, 'Failed to write candidates file')

    if not results:
        brain.save()
        return {"json": {"decision": "approve"}}

    # Priming check removed 2026-04-13 — queried dropped tables.
    priming_note = None

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
    produce_encoding_gap(brain, sq_dal)
    produce_system_health(brain, sq_dal)
    produce_integrity(brain, sq_dal)

    # ── ASSEMBLE: budget-aware output ──
    assembler = SurfaceAssembler(sq_dal, budget_chars=6000)
    # Command hook: write candidates file, return approve + session_id.
    # The thin client reads the file, calls LLM to distill, returns context.
    # Dashboard logging happens in the thin client — one source of truth.
    brain.save()

    # ── S1 Surface: push relevant memories into awareness ──
    additional_context = None
    try:
        additional_context = _run_surface(
            brain, ctx, candidates_data, user_message,
            session_context=brain.session_context,
            recent_messages=recent_messages if 'recent_messages' in dir() else [],
            result=result, enriched=enriched, results=results,
            recall_ref=recall_ref, session_id=session_id,
            graph_changes=graph_changes)
    except Exception as _surface_err:
        brain._log_error('daemon_surface', _surface_err,
                         'S1 Surface failed in daemon (query=%s)' % user_message[:100])

    if additional_context:
        return {"json": {"additionalContext": additional_context}, "session_id": session_id}
    else:
        return {"json": {"decision": "approve"}, "session_id": session_id}






def _hebbian_strengthen(brain, session_id):
    """Strengthen co_accessed edges between surface-selected nodes.

    Only nodes the S1 Surface selected get edges — meaningful co-activation.
    """
    surface_path = '/tmp/brain-%s-surface-selected.json' % session_id
    if not os.path.exists(surface_path):
        return

    with open(surface_path) as f:
        surface_ids = json.load(f).get('selected_ids', [])
    if len(surface_ids) < 2:
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
        return

    from .brain_constants import LEARNING_RATE
    for i in range(len(full_ids)):
        for j in range(i + 1, min(len(full_ids), i + 8)):
            try:
                brain.connect_typed(full_ids[i], full_ids[j],
                                    relation='co_accessed', weight=LEARNING_RATE * 0.15,
                                    edge_type='co_accessed', description='surface-selected')
            except Exception as e:
                brain._log_error('hebbian_edge', e, 'creating co_accessed edge')


def _s1e_chain_id(session_id, counter):
    """Generate S1 encode chain ID for delta trace."""
    from .session_context import SessionContext
    return SessionContext(session_id=session_id, stop_counter=counter).s1e_chain()


def hook_post_response_track(brain, args, graph_changes):
    """Stop event — store exchange, write traces, Hebbian strengthening, gate encoder.

    Flow:
    1. Read recall data from tmp files (written by recall hook)
    2. Store exchange in message_stream (legacy, for escalation)
    3. Write S0 traces (K=user_message, delta=assistant_message)
    4. Hebbian strengthen surface-selected co_accessed edges
    5. Gate encoding agent (every 5th stop, background thread)
    6. Record message heartbeat
    """
    from .pipeline_contract import PIPELINE as _PL
    ctx = brain.get_or_create_session(args.get('session_id', ''))
    session_id = ctx.session_id
    user_message = args.get("prompt", "") or args.get("message", "")
    assistant_response = (args.get("last_assistant_message", "") or "")[:_PL['assistant_response_store']]

    # message_stream writes REMOVED 2026-04-05 — content lives in S0 traces.
    # Escalation tracking was redundant (encoding agent reads from traces, not pending queue).

    # 2. Write S0 traces (using SessionContext for chain IDs)
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
        brain._log_error('trace_s0', e, 'Stop hook')

    # 4. Hebbian strengthening
    try:
        _hebbian_strengthen(brain, session_id)
    except Exception as e:
        brain._log_error('hebbian_surface_selected', e, 'Stop hook')

    # 5. Encoding agent gate (every 5th stop)
    ctx.increment_stop()
    ctx.save(brain.logs_conn)
    encoding_status = ""
    try:
        counter = ctx.stop_counter
        position = counter % 5

        if position == 0:
            if not _encoding_lock.acquire(blocking=False):
                encoding_status = "encoding skipped (previous still running)"
                print("[brain-hooks] Encoding agent skipped — previous run still active", flush=True)
            else:
                from .scales.runner import run_in_background
                from .scales.s1.encode import run_encoding
                run_in_background(
                    name='s1e', brain_db_path=brain.db_path,
                    session_id=session_id, counter=counter,
                    lock=_encoding_lock, run_fn=run_encoding,
                    trace_scale='s1', trace_chain_fn=_s1e_chain_id)
                encoding_status = "encoding started (background)"
        else:
            encoding_status = "encoding %d/5" % position
    except Exception as e:
        brain._log_error('encoding_agent_gate', e, 'Stop hook')
        encoding_status = "encoding error: %s" % str(e)[:50]

    # 6. Heartbeat
    try:
        brain.record_message()
    except Exception as e:
        brain._log_error('record_message', e, 'Stop hook')

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
            if result.get('error'):
                output.append("S2 %s ERROR: %s" % (unit_name.upper(), result['error']))
            elif result.get('skipped'):
                output.append("S2 %s: skipped (%s)" % (unit_name.upper(), result['skipped']))
            else:
                # Unit-specific formatting
                actions = result.get('actions', result.get('classified', 0))
                if unit_name == 'edge_family_integration' and result.get('classified', 0) > 0:
                    output.append("S2 EDGE FAMILIES: classified %d new types into %d families" % (
                        result['classified'], result['families']))
                elif unit_name == 'consolidation' and result.get('clusters'):
                    output.append("S2 CONSOLIDATION: %d clusters found" % len(result['clusters']))
                    stats = result.get('stats', {})
                    class_counts = stats.get('class_counts', {})
                    if class_counts:
                        output.append("  %s" % ', '.join(
                            '%d %s' % (v, k) for k, v in class_counts.items()))
                elif unit_name == 'community_detection':
                    communities = result.get('communities', 0)
                    if actions > 0:
                        output.append("S2 COMMUNITY: %d communities, %d actions" % (
                            communities, actions))
                        graph_changes.append("S2_COMMUNITY: %d communities" % communities)
                    else:
                        output.append("S2 COMMUNITY: no changes (%d communities)" % communities)
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

    # 7. Backfill embeddings
    try:
        emb_count = brain.backfill_embeddings(batch_size=20)
        if isinstance(emb_count, dict):
            emb_count = emb_count.get("count", 0)
        if emb_count and emb_count > 0:
            output.append("EMBEDDINGS: backfilled %d nodes" % emb_count)
    except Exception as e:
        brain._log_error('backfill_embeddings', e, 'idle_maintenance')

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


def hook_post_compact_reboot(brain, args, graph_changes):
    """Post-compact reboot — re-inject brain context after compaction.

    PostCompact stdout IS visible. This is the safety net.
    COMPUTE phase gathers data, then delegates to BrainVoice for FORMAT.
    """
    user = brain.get_config("default_user", "User")
    project = brain.get_config("default_project", "default")

    # ── COMPUTE: gather all data ──

    # synthesize_session removed 2026-04-13
    synthesis_info = {}

    # Re-run lightweight boot
    boot = brain.context_boot(user=user, project=project, task="post-compaction reboot")

    # Locked rules
    locked_nodes = boot.get("locked", [])
    locked_rules = [n for n in locked_nodes if n.get("type") == "rule"]

    # Last synthesis — open questions with age
    try:
        synth_row = brain.conn.execute(
            """SELECT open_questions, decisions_made, corrections_received, created_at
               FROM session_syntheses ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
        if synth_row and synth_row[3]:
            try:
                synth_time = datetime.fromisoformat(synth_row[3].replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                age_minutes = (now - synth_time).total_seconds() / 60

                oq = synth_row[0]
                if oq:
                    try:
                        questions = json.loads(oq)
                        if questions:
                            synthesis_info["open_questions"] = questions
                            synthesis_info["age_minutes"] = age_minutes
                    except Exception as e:
                        brain._log_error('parse_open_questions', e, 'hook_post_compact_reboot')
                elif age_minutes >= 30:
                    synthesis_info["open_questions"] = []
                    synthesis_info["age_minutes"] = age_minutes
            except Exception as e:
                brain._log_error('parse_synth_time', e, 'hook_post_compact_reboot')
    except Exception as e:
        brain._log_error('fetch_last_synthesis', e, 'hook_post_compact_reboot')

    # Consciousness signals (migrated to signal queue — minimal stub for boot)
    signals = {"reminders": brain.get_due_reminders()}

    # assess_developmental_stage removed 2026-04-13.
    dev_stage = None

    # Recall context related to recent work
    recall_results = []
    try:
        recall_query_parts = []
        recent_rows = brain.conn.execute(
            "SELECT title FROM nodes WHERE created_at > datetime('now', '-2 hours') ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
        for row in recent_rows:
            if row[0]:
                recall_query_parts.append(row[0])

        if synth_row:
            for field_idx in (1, 2):
                val = synth_row[field_idx]
                if val and val != "[]":
                    recall_query_parts.append(str(val)[:150])

        if recall_query_parts:
            recall_query = " ".join(recall_query_parts)[:500]
            try:
                result = brain.recall(query=recall_query, limit=8, source='hook')
            except Exception as e:
                brain._log_error('reboot_recall_first_attempt', e, 'hook_post_compact_reboot')
                result = brain.recall(query=recall_query, limit=8, source='hook')

            all_recall = result.get("results", [])
            recent_ids = {r[0] for r in brain.conn.execute(
                "SELECT id FROM nodes WHERE created_at > datetime('now', '-2 hours')"
            ).fetchall()}
            recall_results = [r for r in all_recall if r.get("id") not in recent_ids]
    except Exception as e:
        brain._log_error('reboot_recall', e, 'hook_post_compact_reboot')

    # Find transcript for rehydration hint
    transcript_path = None
    db_dir_env = os.environ.get("BRAIN_DB_DIR", "")
    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT", ".")
    try:
        home = os.path.expanduser("~")
        claude_projects = os.path.join(home, ".claude", "projects")
        if os.path.isdir(claude_projects):
            candidates = []
            for pdir in os.listdir(claude_projects):
                ppath = os.path.join(claude_projects, pdir)
                if not os.path.isdir(ppath):
                    continue
                for fname in os.listdir(ppath):
                    if fname.endswith(".jsonl"):
                        fpath = os.path.join(ppath, fname)
                        candidates.append(fpath)
            if candidates:
                transcript_path = max(candidates, key=os.path.getmtime)
    except Exception as e:
        brain._log_error('find_transcript', e, 'hook_post_compact_reboot')

    # ── FORMAT via BrainVoice ──
    voice = BrainVoice(brain)
    rendered = voice.render_reboot(
        boot_context=boot,
        synthesis_info=synthesis_info,
        locked_rules=locked_rules,
        signals=signals,
        dev_stage=dev_stage,
        recall_results=recall_results,
        pending_messages=None,
        transcript_path=transcript_path,
        db_dir_env=db_dir_env,
        plugin_root=plugin_root,
    )

    brain.save()
    merged = voice.wrap_for_hook(rendered['for_claude'], rendered.get('for_operator'))
    result = {"output": merged}
    return result


def hook_pre_edit(brain, args, graph_changes):
    """PreToolUse(Edit|Write) — surface brain rules before file edits.

    Returns JSON {"decision":"approve","reason":"..."}.
    """
    filename = args.get("filename", "")
    tool_name = args.get("tool_name", "Edit")

    if not filename:
        return {"json": {"decision": "approve"}}

    try:
        data = brain.pre_edit(file=filename, tool_name=tool_name)
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


def hook_pre_compact_save(brain, args, graph_changes):
    """PreCompact — synthesize session + compaction boundary + save.

    Must always return {"decision":"approve"} — never block compaction.
    """
    # synthesize_session removed 2026-04-13

    # Write compaction boundary as S0 trace (not a node — was creating 21+ duplicate nodes)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        ctx = brain.get_or_create_session(args.get('session_id', ''))
        brain._trace_dal.append(
            chain_id=ctx.s0_chain(), scale='s0', event_type='delta',
            ref_type='compaction_boundary',
            summary='Context compacted at %s' % ts,
            session_id=ctx.session_id)
    except Exception as e:
        brain._log_error('compaction_trace', e, 'writing compaction boundary trace')
    graph_changes.append("COMPACTION: boundary trace at %s" % ts)

    brain.save()
    return {"json": {"decision": "approve"}}


def hook_session_end(brain, args, graph_changes):
    """SessionEnd — session synthesis + reflection + consolidation + clean shutdown."""
    # synthesize_session removed 2026-04-13

    # reflect_for_next_claude removed 2026-04-13 — boot nodes nothing read.
    # consolidate() removed 2026-04-13 — wrote to deprecated stability field, created noise.

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
