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






def _get_precision(brain):
    """Get or create a RecallPrecision instance cached on the brain object.

    Caching avoids re-running _ensure_columns() on every hook invocation.
    The instance is garbage collected if the brain object is replaced
    (e.g., daemon restart).
    """
    if not hasattr(brain, '_precision') or brain._precision is None:
        from servers.brain_precision import RecallPrecision
        brain._precision = RecallPrecision(brain.logs_conn, brain.conn,
                                            logs_dal=getattr(brain, '_logs_dal', None))
        # Lazy-load BART for precision evaluation (stays warm for daemon lifetime)
        try:
            from servers.recall_scorer import load_bart
            load_bart()
        except Exception as e:
            brain._log_error('load_bart', e, '_get_precision')
    return brain._precision


# ══════════════════════════════════════════════════════════════════════════════
# HOOK FUNCTIONS — one per hook event
# ══════════════════════════════════════════════════════════════════════════════


def hook_recall(brain, args, graph_changes):
    """Pre-response recall — surfaces brain context before Claude responds.

    Fires on UserPromptSubmit. Returns JSON with additionalContext.
    The richest hook: vocab expansion, recall, segment boundaries, priming,
    aspirations, hypotheses, tensions, instincts, pending messages, graph changes.
    """
    user_message = args.get("prompt", "") or args.get("message", "")
    session_id = brain.get_config("session_id", "ses_unknown")

    # Store last user message for operator voice capture
    try:
        brain.set_config("last_user_message", user_message[:500])
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
            precision = _get_precision(brain)
            # Find the most recent ask_operator recall
            _ask_row = brain.logs_conn.execute(
                """SELECT id FROM recall_log
                   WHERE followup_signal = 'ask_operator' AND explicit_feedback IS NULL
                   ORDER BY created_at DESC LIMIT 1""").fetchone()
            if _ask_row:
                precision.receive_feedback(_ask_row[0], _matched_feedback, source="operator")
                brain.log_debug("precision_feedback", "Operator feedback: %s on recall %d" % (
                    _matched_feedback, _ask_row[0]))
    except Exception as e:
        brain._log_error('explicit_feedback', e, 'hook_recall')

    # ── Table-driven precision: evaluate ALL pending followups ──
    # The user's current message is the "followup" signal for PREVIOUS recalls.
    # Query the table for all recalls awaiting evaluation (Stage 2 → 3),
    # not just the last one. This fixes the 68% evaluation loss.
    try:
        dal = getattr(brain, '_logs_dal', None)
        if dal and user_message:
            pending = dal.get_pending_followups(session_id, limit=5)
            if pending:
                precision = _get_precision(brain)
                for p in pending:
                    try:
                        precision.evaluate_followup(p['id'], user_message)
                    except Exception as e:
                        brain._log_error('precision_evaluate_followup', e,
                                         'recall_log_id=%s' % p['id'])
        else:
            # Fallback: single-slot config handoff (remove once DAL always available)
            prev_log_id = brain.get_config("last_evaluated_recall_id", "")
            if prev_log_id and user_message:
                precision = _get_precision(brain)
                precision.evaluate_followup(int(prev_log_id), user_message)
                brain.set_config("last_evaluated_recall_id", "")
    except Exception as e:
        brain._log_error('precision_evaluate_followup', e, 'table-driven')

    # Vocabulary expansion
    expansions = []
    try:
        candidates = set()
        candidates.update(
            t.strip().lower() for t in
            re.findall(r"\bthe\s+([\w][\w\s-]{2,25})\b", user_message, re.IGNORECASE)
        )
        candidates.update(
            t.strip().lower() for t in
            re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", user_message)
            if len(t) > 4
        )
        for term in candidates:
            resolved = brain.resolve_vocabulary(term)
            if resolved:
                if resolved.get("ambiguous"):
                    for m in resolved.get("mappings", []):
                        expansions.append(m.get("content", ""))
                else:
                    expansions.append(resolved.get("content", ""))
    except Exception as e:
        brain._log_error('vocab_expansion', e, 'hook_recall')

    enriched = user_message[:500]
    if expansions:
        enriched += " " + " ".join(expansions)[:200]

    # Recall
    try:
        result = brain.recall(query=enriched, limit=8)
    except Exception as e:
        brain._log_error('recall_first_attempt', e, 'hook_recall')
        result = brain.recall(query=enriched, limit=8)

    results = result.get("results", [])

    # ── Precision: log recall through the precision module ──
    # Previously, logging was buried inside recall() via _log_recall().
    # Now the hook calls precision.log_recall() explicitly, storing full context
    # (titles, snippets, embeddings_used flag) for future evaluation.
    if results:
        try:
            precision = _get_precision(brain)
            from .pipeline_contract import PRECISION
            recalled_titles = {r.get("id"): r.get("title", "")[:PRECISION['title_limit']] for r in results}
            recalled_snippets = {r.get("id"): (r.get("content") or "")[:PRECISION['snippet_limit']] for r in results}
            embeddings_used = result.get("_recall_mode") != "keyword_only_DEGRADED"
            recall_log_id = precision.log_recall(
                session_id=session_id,
                query=enriched[:500],
                returned_ids=[r.get("id") for r in results],
                recalled_titles=recalled_titles,
                recalled_snippets=recalled_snippets,
                embeddings_used=embeddings_used,
            )
            # Table-driven: row IS the state (Stage 1: LOGGED). No config key needed.
            # Fallback config set for backward compat if DAL not available.
            if not getattr(brain, '_logs_dal', None):
                brain.set_config("last_recall_log_id", str(recall_log_id))
        except Exception as e:
            brain._log_error('precision_log_recall', e, 'query=%s' % enriched[:100])

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
        session_id = brain.get_config('session_id', 'unknown')
        candidates_path = '/tmp/brain-{}-recall-candidates.json'.format(session_id)
        import json as _json
        content_limit = CANDIDATES_FILE['content_limit']
        candidates_data = []
        for r in results[:CANDIDATES_FILE['max_candidates']]:
            node_data = {
                "id": r.get("id", ""),
                "type": r.get("type", ""),
                "title": r.get("title", ""),
                "content": (r.get("content") or "")[:content_limit],
                "confidence": r.get("confidence", 0),
                "locked": r.get("locked", False),
                "score": r.get("effective_activation", 0),
                "revised_at": r.get("revised_at"),
                "created_at": r.get("created_at"),
                "discovery": r.get("_discovery", "embedding"),
            }
            # Include 3-degree graph neighborhood (full — encoding agent needs it)
            graph = r.get("_graph", {})
            if graph:
                node_data["_graph"] = graph
            elif r.get("_neighbors"):
                node_data["_graph"] = {"degree_1": r["_neighbors"], "degree_2": [], "degree_3": []}
            candidates_data.append(node_data)
        # v8.8: Include vocab context — connectors surfaced separately
        vocab_context = result.get('vocab_context', []) if isinstance(result, dict) else []
        with open(candidates_path, 'w') as f:
            _json.dump({
                "user_message": user_message,
                "candidates": candidates_data,
                "segment_note": segment_note,
                "gap": gap.get("query") if gap else None,
                "vocab_context": [{"id": v.get("id", ""), "title": v.get("title", ""),
                                   "content": v.get("content", "")[:200]}
                                  for v in vocab_context[:5]],
            }, f, default=str)
    except Exception as e:
        brain._log_error('recall_candidates_write', e, 'Failed to write candidates file')

    if not results:
        brain.save()
        return {"json": {"decision": "approve"}}

    # Priming check
    priming_note = None
    try:
        primes = brain.get_active_primes()
        if primes:
            match = brain.check_priming(user_message[:500], primes)
            if match:
                priming_note = (
                    'PRIMED TOPIC: "%s" (source: %s, sim: %.2f) '
                    '— this conversation touches an active concern.' % (
                        match["topic"][:80], match["source"], match["similarity"]))
    except Exception as e:
        brain._log_error('priming_check', e, 'hook_recall')

    # Gap detection: log gaps for trend analysis
    if gap:
        try:
            from .dal import LogsDAL
            session_id = brain.get_config('session_id', '')
            LogsDAL(brain.logs_conn).log_gap(gap['query'], gap.get('top_score', 0), session_id)
        except Exception as e:
            brain._log_error('hook_recall_gap_log', e, 'Failed to log recall gap')

    # ── PRODUCE: seed the signal queue ──
    from .dal_signal_queue import SignalQueueDAL
    from .surface_assembler import SurfaceAssembler
    from .signal_producers import (
        produce_reminders, produce_encoding_gap,
        produce_vocabulary_gap, produce_system_health,
        produce_integrity,
    )

    sq_dal = SignalQueueDAL(brain.logs_conn)
    produce_reminders(brain, sq_dal)
    produce_encoding_gap(brain, sq_dal)
    produce_vocabulary_gap(brain, sq_dal)
    produce_system_health(brain, sq_dal)
    produce_integrity(brain, sq_dal)

    # ── ASSEMBLE: budget-aware output ──
    assembler = SurfaceAssembler(sq_dal, budget_chars=6000)
    # Command hook: write candidates file, return approve + session_id.
    # The thin client reads the file, calls LLM to distill, returns context.
    # Dashboard logging happens in the thin client — one source of truth.
    brain.save()
    session_id = brain.get_config('session_id', '')
    return {"json": {"decision": "approve"}, "session_id": session_id}





def hook_post_response_track(brain, args, graph_changes):
    """Stop event — store exchange + gate encoding agent.

    1. store_exchange: persist conversation to message stream
    2. encoding agent gating: set stop_agent_prompt every 5th stop
    3. record_message: heartbeat counter
    """
    user_message = args.get("prompt", "") or args.get("message", "")
    assistant_response = (args.get("last_assistant_message", "") or "")[:4000]

    # Store conversation in message stream (encoding agent reads from this)
    try:
        session_id = brain.get_config('session_id', '')
        brain.store_exchange(user_message, assistant_response, session_id)
    except Exception as e:
        brain._log_error('store_exchange', e, 'Stop hook: failed to store exchange')

    # Encoding agent — fires every 5th stop via Sonnet API.
    # Runs INLINE (not background thread) — SQLite single-connection deadlocks
    # if encoding thread and pool worker access brain.conn concurrently.
    # Stop hook runs after Claude responds, so user doesn't wait.
    encoding_status = ""
    try:
        counter = int(brain.get_config('stop_counter', '0') or '0') + 1
        brain.set_config('stop_counter', str(counter))
        position = counter % 5  # Fire encoding every 5th stop

        if position == 0:
            from .encoding_agent import run_encoding
            from .daemon_dispatch import COMMAND_TABLE
            import time as _t
            _enc_t0 = _t.time()
            print("[brain-hooks] ENCODING AGENT STARTING (counter=%d)" % counter, flush=True)
            def dispatch(cmd, cmd_args):
                entry = COMMAND_TABLE.get(cmd)
                if entry:
                    return entry.handler(brain, cmd_args, [])
                return {"ok": False, "error": "Unknown: %s" % cmd}
            try:
                enc_result = run_encoding(brain, dispatch, counter)
                _enc_ms = int((_t.time() - _enc_t0) * 1000)
                actions = enc_result.get('actions', 0) if isinstance(enc_result, dict) else 0
                encoding_status = "ENCODING RAN: %d actions in %dms" % (actions, _enc_ms)
                print("[brain-hooks] ENCODING AGENT DONE: %s" % encoding_status, flush=True)
            except Exception as enc_e:
                _enc_ms = int((_t.time() - _enc_t0) * 1000)
                encoding_status = "ENCODING FAILED after %dms: %s" % (_enc_ms, enc_e)
                print("[brain-hooks] ENCODING AGENT FAILED: %s" % enc_e, flush=True)
                brain._log_error('encoding_agent_run', enc_e, 'Encoding failed after %dms' % _enc_ms)
        else:
            encoding_status = "encoding %d/5" % position
    except Exception as e:
        brain._log_error('encoding_agent_gate', e, 'Stop hook')
        encoding_status = "encoding error: %s" % str(e)[:50]

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

    # 1. Dream
    try:
        dream_result = brain.dream()
        dream_count = dream_result.get("count", 0)
        if dream_count > 0:
            output.append("DREAM: %d dream(s) generated" % dream_count)
            for d in dream_result.get("dreams", [])[:2]:
                output.append("  - " + d.get("title", "untitled"))
            graph_changes.append("DREAM: %d new dream node(s)" % dream_count)
    except Exception as e:
        output.append("DREAM ERROR: %s" % e)

    # 2. Consolidate
    try:
        consolidate_result = brain.consolidate()
        cons_count = consolidate_result.get("consolidated", 0)
        output.append("CONSOLIDATE: %d nodes boosted" % cons_count)
        if cons_count > 0:
            graph_changes.append("CONSOLIDATE: %d nodes boosted" % cons_count)

        discoveries = consolidate_result.get("discoveries", {})
        total = discoveries.get("total", 0)
        if total > 0:
            output.append("\nBRAIN DISCOVERED (%d evolution(s)):" % total)
            graph_changes.append("DISCOVERED: %d evolution(s)" % total)
            active = brain.get_active_evolutions()
            auto_discovered = [e for e in active if "auto-discovered" in (e.get("content") or "")]
            for evo in auto_discovered[:5]:
                etype = evo["type"].upper()
                title = evo["title"]
                eid = evo["id"]
                content = evo.get("content", "")
                action = ""
                if "ACTION:" in content:
                    action = content.split("ACTION:")[-1].strip()
                output.append("  %s: %s" % (etype, title))
                if action:
                    output.append("    -> " + action)
                eid_short = eid[:8]
                output.append('    [confirm: brain.confirm_evolution("%s...")]' % eid_short)
                output.append('    [dismiss: brain.dismiss_evolution("%s...")]' % eid_short)
    except Exception as e:
        output.append("CONSOLIDATE ERROR: %s" % e)

    # 3. Self-healing
    try:
        heal_result = brain.auto_heal()
        resolved = heal_result.get("resolved", [])
        tuned = heal_result.get("tuned", [])
        cleaned = heal_result.get("cleaned", {})

        if resolved:
            output.append("\nBRAIN HEALED (%d action(s)):" % len(resolved))
            graph_changes.append("HEALED: %d action(s)" % len(resolved))
            for r in resolved[:5]:
                action = r.get("action", "unknown")
                if action == "merge_duplicate":
                    output.append('  MERGED: "%s" into "%s" (sim %s)' % (
                        r.get("archived", ""), r.get("kept", ""), r.get("sim", "")))
                elif action == "auto_lock":
                    output.append('  LOCKED: "%s" (%d accesses)' % (
                        r.get("title", ""), r.get("access_count", 0)))
                else:
                    output.append("  %s: %s" % (action, r))

        if tuned:
            output.append("\nBRAIN TUNED (%d parameter(s)):" % len(tuned))
            for t in tuned[:5]:
                output.append("  %s: %s" % (t.get("param", ""), t.get("reason", "")))

        archived = cleaned.get("archived", 0)
        edges_created = cleaned.get("edges_created", 0)
        edges_normalized = cleaned.get("edges_normalized", 0)
        merged = cleaned.get("merged", 0)
        locked = cleaned.get("locked", 0)
        if any([archived, edges_created, edges_normalized, merged, locked]):
            parts = []
            if merged: parts.append("%d merged" % merged)
            if locked: parts.append("%d locked" % locked)
            if archived: parts.append("%d archived" % archived)
            if edges_created: parts.append("%d edges created" % edges_created)
            if edges_normalized: parts.append("%d edges normalized" % edges_normalized)
            output.append("  HYGIENE: " + ", ".join(parts))
    except Exception as e:
        output.append("HEAL ERROR: %s" % e)

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
            for nid in all_junk:
                brain.conn.execute("DELETE FROM node_enrichments WHERE node_id = ?", (nid,))
                brain.conn.execute("DELETE FROM node_vectors WHERE node_id = ?", (nid,))
                brain.conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (nid, nid))
                brain.conn.execute("DELETE FROM nodes WHERE id = ?", (nid,))
            brain.conn.commit()
            output.append("VOCAB CLEANUP: pruned %d junk nodes" % len(all_junk))
            graph_changes.append("VOCAB_CLEANUP: %d pruned" % len(all_junk))
    except Exception as e:
        output.append("VOCAB CLEANUP ERROR: %s" % e)

    # 3c. Auto-tune
    try:
        tune_result = brain.auto_tune()
        tuned = tune_result.get("tuned", [])
        if tuned:
            output.append("\nBRAIN AUTO-TUNED (%d parameter(s)):" % len(tuned))
            graph_changes.append("TUNED: %d parameter(s)" % len(tuned))
            for t in tuned[:5]:
                output.append("  %s: %s" % (t.get("param", ""), t.get("reason", t.get("note", ""))))
    except Exception as e:
        brain._log_error('auto_tune', e, 'idle_maintenance')

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

    # 3e. Consolidation detection — find overlapping nodes for LLM-driven merging
    try:
        consolidation_count = brain.detect_consolidation_candidates()
        if consolidation_count:
            output.append("CONSOLIDATION: %d new candidate pair(s) queued" % consolidation_count)
            graph_changes.append("CONSOLIDATION: %d pair(s) detected" % consolidation_count)
    except Exception as e:
        brain._log_error('idle_consolidation', e, 'Consolidation detection failed')
        output.append("CONSOLIDATION ERROR: %s" % e)

    # 3f. Expire old pending messages (> 48h = stale, resolve silently)
    try:
        from .dal_message_stream import MessageStreamDAL
        msg_dal = MessageStreamDAL(brain.logs_conn)
        expired = msg_dal.expire_old(max_age_hours=48)
        if expired:
            output.append("MSG_STREAM: expired %d old pending messages" % expired)
    except Exception as e:
        brain._log_error('idle_expire_messages', e, 'Failed to expire old messages')

    # 4. Reflection prompts
    try:
        reflections = brain.prompt_reflection()
        if reflections:
            output.append("")
            output.append("REFLECT (transferable insights from this session?):")
            for r in reflections[:3]:
                output.append("  " + r)
            output.append("")
    except Exception as e:
        brain._log_error('prompt_reflection', e, 'idle_maintenance')

    # 5. Self-reflection
    try:
        reflection = brain.auto_generate_self_reflection()
        ref_count = sum(1 for v in reflection.values() if v)
        if ref_count > 0:
            output.append("SELF-REFLECTION: %d reflection(s) generated" % ref_count)
    except Exception as e:
        brain._log_error('self_reflection', e, 'idle_maintenance')

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

    # 8. Prune irrelevant auto-captured quotes
    try:
        prune_result = brain.prune_irrelevant_quotes(batch_size=30)
        if prune_result.get("pruned", 0) > 0:
            output.append("QUOTE PRUNING: %d/%d checked, %d irrelevant removed" % (
                prune_result["pruned"], prune_result["checked"], prune_result["pruned"]))
            graph_changes.append("PRUNED: %d irrelevant quotes" % prune_result["pruned"])
            for p in prune_result.get("pruned_nodes", [])[:3]:
                output.append('  pruned: "%s" (sim %.2f) from: %s' % (
                    p["quote"][:50], p["similarity"], p["title"][:40]))
    except Exception as e:
        brain._log_error('prune_quotes', e, 'idle_maintenance')

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

    # 10. Session health check
    try:
        health = brain.assess_session_health()
        if health and health.get("overall") == "concerning":
            output.append("")
            output.append("SESSION HEALTH CHECK: %s" % health["overall"])
            top = health.get("top_prompt")
            if top:
                output.append("  %s" % top)
            for g in health.get("gaps", [])[:2]:
                if g["signal"] != top:
                    output.append("  [%s] %s" % (g["dimension"], g["signal"][:100]))
    except Exception as e:
        brain._log_error('session_health', e, 'idle_maintenance')

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

    # Safety net: check if pre-compact synthesis ran
    synthesis_info = {}
    synth_row = None
    try:
        last_synth = brain.conn.execute(
            "SELECT created_at FROM session_syntheses ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        session_start = brain.get_config("session_start_at", "")

        synth_ran = False
        if last_synth and session_start:
            synth_ran = last_synth[0] >= session_start

        if not synth_ran:
            try:
                synthesis = brain.synthesize_session()
                parts = []
                for key in ("decisions", "corrections", "open_questions"):
                    val = synthesis.get(key)
                    if val:
                        parts.append("%s %s" % (val, key))
                synthesis_info = {"just_ran": True, "parts": parts}
            except Exception as e:
                synthesis_info = {"error": str(e)}
    except Exception as e:
        brain._log_error('synthesis_safety_net', e, 'hook_post_compact_reboot')

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

    # Developmental stage
    dev_stage = None
    try:
        dev_stage = brain.assess_developmental_stage()
    except Exception as e:
        brain._log_error('assess_dev_stage', e, 'hook_post_compact_reboot')

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
                result = brain.recall(query=recall_query, limit=8)
            except Exception as e:
                brain._log_error('reboot_recall_first_attempt', e, 'hook_post_compact_reboot')
                result = brain.recall(query=recall_query, limit=8)

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
    # Synthesize session
    try:
        synthesis = brain.synthesize_session()
        parts = []
        for key in ("decisions", "corrections", "teaching_arcs", "open_questions"):
            val = synthesis.get(key)
            if val:
                parts.append("%s %s" % (val, key))
    except Exception as e:
        brain._log_error('synthesize_session', e, 'hook_pre_compact_save')

    # Write compaction boundary marker
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    brain.remember(
        type="context",
        title="Compaction boundary at %s" % ts,
        content="Context compacted. Synthesis ran. Post-compact reboot will re-inject context.",
        keywords="compaction boundary session handoff",
        locked=False,
    )
    graph_changes.append("COMPACTION: boundary marker created at %s" % ts)

    brain.save()
    return {"json": {"decision": "approve"}}


def hook_session_end(brain, args, graph_changes):
    """SessionEnd — session synthesis + reflection + consolidation + clean shutdown."""
    # Synthesize
    try:
        brain.synthesize_session()
    except Exception as e:
        brain._log_error('synthesize_session', e, 'hook_session_end')

    # Reflect for next Claude — create boot node with session handoff
    try:
        brain.reflect_for_next_claude()
    except Exception as e:
        brain._log_error('reflect_for_next_claude', e, 'hook_session_end')

    # Consolidate
    try:
        brain.consolidate()
    except Exception as e:
        brain._log_error('consolidate', e, 'hook_session_end')

    brain.save()
    # Note: the hook client sends "shutdown" separately after this returns
    return {"output": ""}


def hook_stop_failure_log(brain, args, graph_changes):
    """StopFailure — logs API failures to brain for pattern detection."""
    error_type = args.get("error", "unknown")
    error_details = args.get("error_details", "")
    session_id = args.get("session_id", "")

    try:
        brain.log_miss(
            session_id=session_id,
            signal="api_failure",
            query="API error: %s" % error_type,
            expected_node_id=None,
            context=str(error_details)[:500],
        )
        brain.save()
    except Exception as e:
        brain._log_error('log_miss', e, 'hook_stop_failure_log')

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
