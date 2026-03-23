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
import time
import traceback
from datetime import datetime, timezone

# ── Constants (canonical definitions in brain_voice.py) ──

from servers.brain_voice import (
    EVOLUTION_TYPES, ENGINEERING_TYPES, CODE_COGNITION_TYPES,
    BrainVoice,
)

# Backwards-compatible function aliases — delegate to BrainVoice static methods
_format_recall_results = BrainVoice.format_recall_results
_format_encoding_warning = BrainVoice.format_encoding_warning
_format_suggestions = BrainVoice.format_suggestions

CHECKPOINT_CYCLE = [
    "UNCERTAINTY: What don't you fully understand from the last few exchanges? "
    "Encode at least one brain.remember_uncertainty(). Honest 'I don't know' is "
    "more valuable than thin facts.",

    "CONNECTIONS: What connections did you discover? Use brain.connect() between "
    "related nodes and brain.remember_impact(if_changed, must_check, because) for "
    "dependencies. Orphan nodes die.",

    "DECISIONS + LESSONS: What was decided or learned? brain.remember(type='decision', "
    "locked=True) with full reasoning. brain.remember_lesson() for any bugs or mistakes. "
    "Include WHY, not just WHAT.",

    "BLAST RADIUS: What could break if this code changes? "
    "brain.remember_impact(if_changed='component', must_check=['dependents'], "
    "because='reason'). Map the ripple effects you noticed.",

    "PATTERNS: What patterns or conventions did you observe? "
    "brain.remember_convention() for coding patterns. "
    "brain.remember(type='mental_model') for architectural insights. Name the pattern.",
]

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

def _store_pending(brain, message):
    """Store a pending message for next UserPromptSubmit to drain."""
    try:
        existing = brain.get_config("pending_hook_messages", "[]")
        pending = json.loads(existing) if existing else []
    except Exception:
        pending = []
    pending.append(message)
    pending = pending[-5:]  # cap at 5
    brain.set_config("pending_hook_messages", json.dumps(pending))


def _drain_pending(brain):
    """Read and clear pending messages. Returns list of strings."""
    try:
        existing = brain.get_config("pending_hook_messages", "[]")
        pending = json.loads(existing) if existing else []
        if pending:
            brain.set_config("pending_hook_messages", "[]")
    except Exception:
        pending = []
    return pending


def _drain_debug_logs(brain):
    """Read and clear hook_debug entries from brain_logs.db.

    Uses a high-water mark (last_drained_debug_id in brain_meta) so each
    debug message is surfaced exactly once. Returns list of formatted strings.
    """
    try:
        last_id = int(brain.get_config("last_drained_debug_id", "0") or 0)
        rows = brain.logs_conn.execute(
            "SELECT id, source, metadata, created_at FROM debug_log "
            "WHERE event_type = 'hook_debug' AND id > ? ORDER BY id LIMIT 20",
            (last_id,),
        ).fetchall()
        if not rows:
            return []
        messages = []
        max_id = last_id
        for row in rows:
            rid, source, metadata, ts = row
            if rid > max_id:
                max_id = rid
            try:
                meta = json.loads(metadata) if metadata else {}
            except Exception:
                meta = {}
            msg = meta.get("message", metadata or "")
            short_ts = ts[11:19] if ts and len(ts) > 19 else ts or ""
            messages.append("[%s] %s: %s" % (short_ts, source, msg))
        brain.set_config("last_drained_debug_id", str(max_id))
        return messages
    except Exception:
        return []


def _drain_graph_changes(graph_changes):
    """Drain and return graph changes, clearing the list."""
    if not graph_changes:
        return []
    changes = list(graph_changes)
    graph_changes.clear()
    return changes


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
        except Exception:
            pass  # Graceful degradation: regex+embeddings still work
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
    except Exception:
        pass

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
    except Exception:
        pass

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
    except Exception:
        pass

    enriched = user_message[:500]
    if expansions:
        enriched += " " + " ".join(expansions)[:200]

    # Recall
    try:
        result = brain.recall_with_embeddings(query=enriched, limit=8)
    except Exception:
        result = brain.recall(query=enriched, limit=8)

    results = result.get("results", [])

    # ── Precision: log recall through the precision module ──
    # Previously, logging was buried inside recall_with_embeddings() via _log_recall().
    # Now the hook calls precision.log_recall() explicitly, storing full context
    # (titles, snippets, embeddings_used flag) for future evaluation.
    if results:
        try:
            precision = _get_precision(brain)
            recalled_titles = {r.get("id"): r.get("title", "")[:100] for r in results}
            recalled_snippets = {r.get("id"): (r.get("content") or "")[:150] for r in results}
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
    except Exception:
        pass

    # Drain pending messages
    pending_messages = _drain_pending(brain)

    # Drain debug logs (hook_debug entries from brain_logs.db)
    debug_messages = _drain_debug_logs(brain)

    # Drain graph changes
    recent_graph_changes = _drain_graph_changes(graph_changes)

    # Awareness heartbeat — urgent signals from consciousness layer
    urgent_signals = []
    try:
        urgent_signals = brain.get_urgent_signals()
    except Exception:
        pass

    if not results and not pending_messages and not urgent_signals and not recent_graph_changes and not debug_messages:
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
    except Exception:
        pass

    # ── Precision: request feedback if uncertain about previous recall ──
    precision_feedback = None
    try:
        prev_eval_id = brain.get_config("last_evaluated_recall_id", "")
        if prev_eval_id:
            precision = _get_precision(brain)
            precision_feedback = precision.request_feedback(int(prev_eval_id))
    except Exception:
        pass

    # ── DECIDE + FORMAT via BrainVoice ──
    voice = BrainVoice(brain)
    prompt_signals = voice.select_prompt_signals(user_message, results)
    rendered = voice.render_prompt(
        results=results,
        prompt_signals=prompt_signals,
        urgent_signals=urgent_signals,
        segment_note=segment_note,
        priming_note=priming_note,
        graph_changes=recent_graph_changes,
        pending_messages=pending_messages,
        debug_messages=debug_messages,
        precision_feedback=precision_feedback,
    )

    brain.save()
    merged = voice.wrap_for_hook(rendered['for_claude'], rendered.get('for_operator'))
    result_json = {"additionalContext": merged}
    return {"json": result_json}


def hook_post_response_track(brain, args, graph_changes):
    """Post-response tracker: precision evaluation + vocab gap detection + encoding checkpoints.

    Fires on UserPromptSubmit AND Stop.
    """
    user_message = args.get("prompt", "") or args.get("message", "")
    has_user_message = user_message and len(user_message) >= 10
    is_user_prompt = bool(args.get("prompt"))

    # ── Precision: evaluate Claude's response (Signal 1) ──
    # The Stop event provides last_assistant_message — Claude's actual response.
    # We store it on the recall_log row for future LLM-based evaluation.
    # This is the WEAK signal (biased — Claude absorbs injected context).
    assistant_response = args.get("last_assistant_message", "")
    if not assistant_response:
        # Fallback for backward compat / events that don't provide it
        assistant_response = ""
    assistant_response = assistant_response[:4000]

    session_id = brain.get_config("session_id", "ses_unknown")
    if not session_id.startswith("ses_"):
        session_id = "ses_%s" % session_id

    # Table-driven: find the most recent recall awaiting response storage
    recall_log_id = None
    try:
        dal = getattr(brain, '_logs_dal', None)
        if dal:
            recall_log_id = dal.get_pending_response(session_id)
        else:
            # Fallback: config-slot (remove once DAL is always available)
            recall_log_id = brain.get_config("last_recall_log_id", "")
            if recall_log_id:
                recall_log_id = int(recall_log_id)
    except Exception:
        pass

    if recall_log_id and assistant_response and len(assistant_response) >= 20:
        try:
            precision = _get_precision(brain)
            precision.evaluate_response(int(recall_log_id), assistant_response)
            # No config keys to set — table row IS the state (Stage 2: RESPONSE_STORED)
        except Exception as e:
            brain._log_error('precision_evaluate_response', e, 'recall_log_id=%s' % recall_log_id)
    elif recall_log_id and (not assistant_response or len(assistant_response) < 20):
        # Log diagnostic — empty response means precision evaluation can't run.
        # This helps detect if the hook communication is broken.
        try:
            brain._logs_dal.write_debug(
                "precision",
                "Empty/short assistant response in Stop event — precision evaluation skipped",
                session_id=session_id,
                metadata=json.dumps({
                    "args_keys": list(args.keys()),
                    "response_len": len(assistant_response or ""),
                    "recall_log_id": recall_log_id,
                }),
            )
        except Exception:
            pass

    # ── Vocab gap detection (only when we have user message) ──
    if has_user_message:
        try:
            from servers.text_processing import filter_domain_terms

            # ── Strategy 1: Quoted terms ──
            quoted = re.findall(r'["]([\w\s-]{3,30})["]', user_message)

            # ── Strategy 2: "the/a/this X" with expanded suffix list ──
            the_patterns = re.findall(
                r"\b(?:the|a|an|this|that|our|my|your)\s+"
                r"([\w][\w\s-]{2,25}(?:hook|script|table|file|function|method|class|"
                r"module|layer|loop|sequence|pipeline|system|engine|server|db|database|"
                r"config|schema|signal|node|type|map|graph|cache|queue|log|test|spec|"
                r"worker|adapter|pattern|protocol|daemon|scorer|mixin|handler|resolver|"
                r"builder|encoder|decoder|parser|formatter|validator|serializer|"
                r"middleware|endpoint|route|trigger|listener|callback|factory|strategy|"
                r"observer|wrapper|proxy|bridge|decorator|registry|repository|mapper|"
                r"transformer|dispatcher|emitter|collector|aggregator|provider|consumer|"
                r"subscriber|publisher|context|session|token|metric|monitor|tracer|"
                r"profiler|compiler|runtime|kernel|driver|plugin|extension|toolkit|"
                r"library|framework|platform|screen|component|service|client))\b",
                user_message, re.IGNORECASE,
            )

            # ── Strategy 3: Verb-object patterns (expanded) ──
            action_context = re.findall(
                r"\b(?:fix|update|change|modify|check|look at|review|debug|test|refactor|"
                r"rewrite|add|remove|delete|move|rename|split|merge|clean|implement|"
                r"configure|deploy|migrate|optimize|integrate|initialize|bootstrap|"
                r"instrument|validate|authenticate|provision|dispatch|schedule|monitor|"
                r"benchmark|evaluate|classify|extract|transform|aggregate|normalize|"
                r"cache|batch|rollback|seed|stub|mock|patch|inject|bind|resolve|"
                r"register|subscribe|emit|consume|publish)\s+(?:the\s+)?"
                r"([\w]+(?:\s+[\w]+){0,3}?)(?:\s*(?:and|or|but|also|then|,|;|\.|!|\?)|\s*$)",
                user_message, re.IGNORECASE,
            )
            action_context = [t.strip() for t in action_context if len(t.strip()) >= 3]

            # ── Strategy 4: Hyphenated compounds ──
            hyphenated = re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", user_message)
            hyphenated = [h for h in hyphenated if len(h) > 4 and h.lower() not in (
                "re-run", "re-do", "non-null", "non-zero", "up-to-date", "e-g",
                "pre-existing", "co-authored-by",
            )]

            # ── Strategy 5: Capitalized mid-sentence terms (product names, entities) ──
            capitalized = re.findall(
                r'(?<=[a-z.,;:!?\s])\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                user_message,
            )
            # Filter out sentence-start words and common words
            capitalized = [c.strip() for c in capitalized
                          if c.strip() and c.strip() not in (
                              'I', 'The', 'This', 'That', 'It', 'We', 'You',
                              'He', 'She', 'They', 'But', 'And', 'Or', 'So',
                              'If', 'When', 'What', 'How', 'Why', 'Where',
                              'Yes', 'No', 'Ok', 'Sure', 'Please', 'Thanks',
                          )]

            # ── Strategy 6: Backtick-wrapped code terms ──
            backtick = re.findall(r'`([^`]{2,40})`', user_message)

            # ── Strategy 7: Acronyms (2-6 uppercase letters) ──
            acronyms = re.findall(r'\b([A-Z]{2,6})\b', user_message)

            # ── Collect and filter candidates ──
            skip_words = {
                "the", "a", "an", "this", "that", "it", "them", "is", "are",
                "was", "were", "be", "been", "do", "does", "did", "have", "has",
                "can", "could", "will", "would", "should", "may", "might",
                "yes", "no", "ok", "sure", "please", "thanks", "code", "file",
                "thing", "stuff", "something", "everything", "nothing",
                "just", "also", "very", "really", "actually", "basically",
                "like", "about", "some", "more", "here", "there", "now", "then",
            }
            raw_candidates = set()
            for term in (quoted + the_patterns + action_context + hyphenated +
                        capitalized + backtick + acronyms):
                term = term.strip()
                if len(term) < 2 or len(term) > 40:
                    continue
                words = term.lower().split()
                if all(w in skip_words for w in words):
                    continue
                raw_candidates.add(term)

            # Apply domain-specificity filter
            candidates = set(filter_domain_terms(list(raw_candidates)))

            if candidates:
                try:
                    vocab_rows = brain.conn.execute(
                        "SELECT LOWER(title), LOWER(content) FROM nodes WHERE type = ? AND archived = 0",
                        ("vocabulary",),
                    ).fetchall()
                except Exception:
                    vocab_rows = []

                known_terms = set()
                for title, content in vocab_rows:
                    if " \u2192 " in title:
                        known_terms.add(title.split(" \u2192 ")[0].strip())
                    elif " -> " in title:
                        known_terms.add(title.split(" -> ")[0].strip())
                    else:
                        known_terms.add(title.strip())

                try:
                    all_titles = brain.conn.execute(
                        "SELECT LOWER(title) FROM nodes WHERE archived = 0"
                    ).fetchall()
                    title_set = {r[0] for r in all_titles}
                except Exception:
                    title_set = set()

                unmapped = []
                for term in candidates:
                    if term in known_terms:
                        continue
                    if term in title_set:
                        continue
                    if any(term in kt or kt in term for kt in known_terms):
                        continue
                    words = term.split()
                    if len(words) == 1 and any(term in t for t in title_set):
                        continue
                    unmapped.append(term)

                if unmapped:
                    existing = brain.get_config("vocabulary_gaps", "[]")
                    try:
                        gaps = json.loads(existing)
                    except Exception:
                        gaps = []
                    existing_terms = {g.get("term") if isinstance(g, dict) else g for g in gaps}
                    for term in unmapped[:5]:
                        if term not in existing_terms:
                            gaps.append({"term": term, "message_preview": user_message[:80]})
                    gaps = gaps[-20:]
                    brain.set_config("vocabulary_gaps", json.dumps(gaps))
        except Exception:
            pass

    # ── Encoding heartbeat with rotating checkpoints ──
    output = ""
    try:
        brain.record_message()
        nudge = brain.get_encoding_heartbeat()
        if nudge:
            msg = nudge.get("message", "")

            # Get and rotate checkpoint
            try:
                idx = int(brain.get_config("checkpoint_index", "0"))
            except Exception:
                idx = 0
            focus = CHECKPOINT_CYCLE[idx % len(CHECKPOINT_CYCLE)]
            next_idx = str((idx + 1) % len(CHECKPOINT_CYCLE))
            try:
                brain.set_config("checkpoint_index", next_idx)
            except Exception:
                pass

            # Session encoding stats
            try:
                node_count = brain.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE created_at > datetime('now', '-2 hours')"
                ).fetchone()[0]
                uncert_count = brain.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type = 'uncertainty' AND created_at > datetime('now', '-2 hours')"
                ).fetchone()[0]
                connect_count = brain.conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE created_at > datetime('now', '-2 hours')"
                ).fetchone()[0]
                stats = "Session stats: %d nodes, %d uncertainties, %d connections." % (
                    node_count, uncert_count, connect_count)
            except Exception:
                stats = ""

            checkpoint_lines = ["[BRAIN] ENCODING CHECKPOINT: " + msg]
            if stats:
                checkpoint_lines.append(stats)
            checkpoint_lines.append(focus)
            checkpoint_lines.append("[/BRAIN]")
            checkpoint_text = "\n".join(checkpoint_lines)

            if is_user_prompt:
                output = "\n" + checkpoint_text + "\n"
            else:
                # Stop event — store as pending (invisible)
                _store_pending(brain, checkpoint_text)
    except Exception:
        pass

    brain.save()
    return {"output": output}


def hook_idle_maintenance(brain, args, graph_changes):
    """Idle maintenance — dream, consolidate, heal, tune, reflect.

    Fires on Notification(idle_prompt). Output stored as pending message.
    """
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

    # 3b. Auto-tune
    try:
        tune_result = brain.auto_tune()
        tuned = tune_result.get("tuned", [])
        if tuned:
            output.append("\nBRAIN AUTO-TUNED (%d parameter(s)):" % len(tuned))
            graph_changes.append("TUNED: %d parameter(s)" % len(tuned))
            for t in tuned[:5]:
                output.append("  %s: %s" % (t.get("param", ""), t.get("reason", t.get("note", ""))))
    except Exception:
        pass

    # 4. Reflection prompts
    try:
        reflections = brain.prompt_reflection()
        if reflections:
            output.append("")
            output.append("REFLECT (transferable insights from this session?):")
            for r in reflections[:3]:
                output.append("  " + r)
            output.append("")
    except Exception:
        pass

    # 5. Self-reflection
    try:
        reflection = brain.auto_generate_self_reflection()
        ref_count = sum(1 for v in reflection.values() if v)
        if ref_count > 0:
            output.append("SELF-REFLECTION: %d reflection(s) generated" % ref_count)
    except Exception:
        pass

    # 6. Backfill summaries
    try:
        backfill = brain.backfill_summaries(batch_size=50)
        bf_count = backfill.get("updated", 0)
        if bf_count > 0:
            output.append("SUMMARIES: backfilled %d nodes" % bf_count)
    except Exception:
        pass

    # 7. Backfill embeddings
    try:
        emb_count = brain.backfill_embeddings(batch_size=20)
        if isinstance(emb_count, dict):
            emb_count = emb_count.get("count", 0)
        if emb_count and emb_count > 0:
            output.append("EMBEDDINGS: backfilled %d nodes" % emb_count)
    except Exception:
        pass

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
    except Exception:
        pass

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
    except Exception:
        pass

    # Store as pending message (Notification stdout is invisible)
    if output:
        summary = "[BRAIN] IDLE MAINTENANCE:\n" + "\n".join(output) + "\n[/BRAIN]"
        _store_pending(brain, summary)

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
    except Exception:
        pass

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
                    except Exception:
                        pass
                elif age_minutes >= 30:
                    synthesis_info["open_questions"] = []
                    synthesis_info["age_minutes"] = age_minutes
            except Exception:
                pass
    except Exception:
        pass

    # Consciousness signals
    signals = brain.get_consciousness_signals()

    # Developmental stage
    dev_stage = None
    try:
        dev_stage = brain.assess_developmental_stage()
    except Exception:
        pass

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
                result = brain.recall_with_embeddings(query=recall_query, limit=8)
            except Exception:
                result = brain.recall(query=recall_query, limit=8)

            all_recall = result.get("results", [])
            recent_ids = {r[0] for r in brain.conn.execute(
                "SELECT id FROM nodes WHERE created_at > datetime('now', '-2 hours')"
            ).fetchall()}
            recall_results = [r for r in all_recall if r.get("id") not in recent_ids]
    except Exception:
        pass

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
    except Exception:
        pass

    # Drain pending messages
    pending = _drain_pending(brain)

    # ── FORMAT via BrainVoice ──
    voice = BrainVoice(brain)
    rendered = voice.render_reboot(
        boot_context=boot,
        synthesis_info=synthesis_info,
        locked_rules=locked_rules,
        signals=signals,
        dev_stage=dev_stage,
        recall_results=recall_results,
        pending_messages=pending if pending else None,
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
    except Exception:
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
    except Exception:
        pass

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
        except Exception:
            pass

    brain.save()
    return {"json": {"decision": "approve", "reason": context}}


def hook_pre_bash_safety(brain, args, graph_changes):
    """PreToolUse(Bash) — safety check for destructive commands.

    Returns JSON {"decision":"approve"|"block","reason":"..."}.
    """
    command = args.get("command", "")

    try:
        result = brain.safety_check(command)
    except Exception:
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
        except Exception:
            pass

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
    except Exception:
        pass

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
    """SessionEnd — session synthesis + consolidation + clean shutdown."""
    # Synthesize
    try:
        synthesis = brain.synthesize_session()
    except Exception:
        pass

    # Consolidate
    try:
        brain.consolidate()
    except Exception:
        pass

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

            _store_pending(brain, "\n".join(output_lines))
            graph_changes.append("HOST: environment changed (%d items)" % len(changes))
            brain.save()
    except Exception:
        pass

    return {"output": ""}


def hook_post_bash_host_check(brain, args, graph_changes):
    """PostToolUse(Bash) — detects env changes after pip/brew/etc.

    Stdout invisible. Stores output as pending message.
    """
    try:
        env_result = brain.scan_host_environment()
        changes = env_result.get("changes", {}) if env_result else {}

        if changes:
            command = args.get("command", "")
            output_lines = ["[BRAIN] HOST ENVIRONMENT CHANGED (after bash):"]
            for key, change in changes.items():
                old_val = change.get("old", "?")
                new_val = change.get("new", "?")
                output_lines.append("  %s: %s \u2192 %s" % (key, old_val, new_val))
            output_lines.append("  Command: %s" % command[:100])
            output_lines.append("[/BRAIN]")

            _store_pending(brain, "\n".join(output_lines))
            graph_changes.append("HOST: env changed after bash (%d items)" % len(changes))
            brain.save()
    except Exception:
        pass

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
    except Exception:
        pass

    brain.set_config("current_worktree", worktree_name)
    brain.set_config("current_branch", branch)
    brain.set_config("current_cwd", cwd)

    try:
        brain.scan_host_environment()
    except Exception:
        pass

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
