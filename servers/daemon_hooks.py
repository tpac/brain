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

# ── Constants ──

EVOLUTION_TYPES = {"tension", "hypothesis", "pattern", "catalyst", "aspiration"}

ENGINEERING_TYPES = {"purpose", "mechanism", "impact", "constraint", "convention",
                     "lesson", "vocabulary"}
CODE_COGNITION_TYPES = {"fn_reasoning", "param_influence", "code_concept",
                        "arch_constraint", "causal_chain", "bug_lesson",
                        "comment_anchor"}

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
        brain._precision = RecallPrecision(brain.logs_conn, brain.conn)
    return brain._precision


def _format_recall_results(results, lines):
    """Format recall results into output lines."""
    evolution_results = [r for r in results if r.get("type") in EVOLUTION_TYPES]
    regular_results = [r for r in results if r.get("type") not in EVOLUTION_TYPES]

    if evolution_results:
        lines.append("ACTIVE EVOLUTION (brain is tracking these):")
        for r in evolution_results[:3]:
            title = r.get("title", "")[:80]
            content = r.get("content", "")
            if len(content) > 150:
                content = content[:150] + "..."
            lines.append("  " + title)
            lines.append("    " + content)
            lines.append("")

    for r in regular_results[:5]:
        typ = r.get("type", "?")
        title = r.get("title", "")[:60]
        content = r.get("content", "")
        locked = "LOCKED " if r.get("locked") else ""
        score = r.get("effective_activation", 0)
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append("  [%s] %s%s (score: %.2f)" % (typ, locked, title, score))
        lines.append("    " + content)
        lines.append("")


def _format_encoding_warning(encoding):
    """Generate encoding health warning if needed."""
    health = encoding.get("health", "OK")
    edits_gap = encoding.get("edits_since_last_remember", 0)
    mins_since = encoding.get("minutes_since_last_remember", 0)
    session_min = encoding.get("session_minutes", 0)

    if health == "NONE" and session_min > 3:
        return (
            "ENCODING ALERT: You have not stored ANY learnings in the brain this session. "
            "If decisions were made, corrections happened, or the user gave feedback — "
            "call /remember NOW before continuing. The brain cannot learn from what you do not store."
        )
    elif health == "STALE":
        if edits_gap > 15:
            return (
                "ENCODING WARNING: %d edits since your last /remember call "
                "(%d min ago). If anything worth remembering happened in that span — "
                "a decision, a correction, a pattern, feedback — store it now." % (edits_gap, mins_since)
            )
        elif edits_gap > 8:
            return (
                "ENCODING CHECK: %d edits since last /remember. "
                "Anything worth storing? Decisions, corrections, lessons?" % edits_gap
            )
    return ""


def _format_suggestions(filename, suggestions, procedures, context_files, change_impacts, encoding_warning):
    """Format brain suggestions into readable output for pre-edit hook."""
    lines = ["BRAIN AUTO-SUGGEST for %s:" % filename, ""]

    eng_nodes = [s for s in suggestions if s.get("type") in ENGINEERING_TYPES]
    code_nodes = [s for s in suggestions if s.get("type") in CODE_COGNITION_TYPES]
    other_nodes = [s for s in suggestions if (
        s.get("type") not in ENGINEERING_TYPES
        and s.get("type") not in CODE_COGNITION_TYPES
        and s.get("type") != "procedure"
        and not (s.get("type") == "file" and "[ctx:" in s.get("title", ""))
    )]

    if change_impacts:
        lines.append("CHANGE IMPACT WARNING:")
        lines.append("")
        for ci in change_impacts[:5]:
            ci_title = ci.get("title", "")[:80]
            ci_content = ci.get("content", "")
            if len(ci_content) > 300:
                ci_content = ci_content[:300] + "..."
            lines.append("  [impact] " + ci_title)
            lines.append("    " + ci_content)
            lines.append("")

    if eng_nodes:
        lines.append("ENGINEERING MEMORY (read carefully — these describe what you are about to edit):")
        lines.append("")
        for s in eng_nodes:
            typ = s.get("type", "?")
            title = s.get("title", "")[:80]
            content = s.get("content", "")
            locked = "LOCKED " if s.get("locked") else ""
            if len(content) > 350:
                content = content[:350] + "..."
            lines.append("  [%s] %s%s" % (typ, locked, title))
            lines.append("    " + content)
            lines.append("")

    if code_nodes:
        lines.append("CODE KNOWLEDGE:")
        lines.append("")
        for s in code_nodes:
            typ = s.get("type", "?")
            title = s.get("title", "")[:80]
            content = s.get("content", "")
            locked = "LOCKED " if s.get("locked") else ""
            if len(content) > 350:
                content = content[:350] + "..."
            lines.append("  [%s] %s%s" % (typ, locked, title))
            lines.append("    " + content)
            lines.append("")

    if other_nodes:
        if code_nodes:
            lines.append("OTHER RULES & DECISIONS:")
        lines.append("")
        for s in other_nodes:
            typ = s.get("type", "?")
            title = s.get("title", "")[:80]
            content = s.get("content", "")
            locked = "LOCKED " if s.get("locked") else ""
            if len(content) > 250:
                content = content[:250] + "..."
            lines.append("  [%s] %s%s" % (typ, locked, title))
            lines.append("    " + content)
            lines.append("")

    if procedures:
        lines.append("TRIGGERED PROCEDURES:")
        for p in procedures[:3]:
            lines.append("  [procedure] " + p.get("title", ""))
            psteps = p.get("steps", "")
            if len(psteps) > 300:
                psteps = psteps[:300] + "..."
            lines.append("    " + psteps)
            lines.append("")

    if context_files:
        lines.append("CONTEXT FILES (read before editing — may contain detailed requirements):")
        for cf in context_files[:2]:
            cftopic = cf.get("topic", "")
            cftitle = cf.get("title", "")
            cfupdated = str(cf.get("last_updated", ""))[:10]
            cfsummary = str(cf.get("summary", ""))[:150]
            lines.append("  [%s] %s (updated %s)" % (cftopic, cftitle, cfupdated))
            lines.append("    " + cfsummary)
            lines.append("")
        lines.append("IMPORTANT: If the context file conflicts with current work, flag the conflict.")
        lines.append("")

    if encoding_warning:
        lines.append("")
        lines.append(encoding_warning)
        lines.append("")

    locked_ids = [s.get("id", "") for s in suggestions if s.get("locked")]
    if locked_ids:
        lines.append("BRAIN->HOST: If you follow locked rules above, call brain.log_communication(node_id, 'high_priority', True).")
        lines.append("If you must deviate, call brain.log_communication(node_id, 'high_priority', False, reason).")
        lines.append("")

    lines.append("Review these constraints before proceeding with the edit.")
    return "\n".join(lines)


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

    # ── Two-turn precision: evaluate previous turn's followup ──
    # The user's current message is the "followup" signal for the PREVIOUS
    # turn's recall. If there's a pending evaluated recall, feed the user's
    # message to evaluate_followup() before starting the new recall cycle.
    try:
        prev_log_id = brain.get_config("last_evaluated_recall_id", "")
        if prev_log_id and user_message:
            precision = _get_precision(brain)
            precision.evaluate_followup(int(prev_log_id), user_message)
            brain.set_config("last_evaluated_recall_id", "")
    except Exception:
        pass

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
            brain.set_config("last_recall_log_id", str(recall_log_id))
        except Exception:
            pass

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

    # Drain graph changes
    recent_graph_changes = _drain_graph_changes(graph_changes)

    # Awareness heartbeat — urgent signals from consciousness layer
    urgent_signals = []
    try:
        urgent_signals = brain.get_urgent_signals()
    except Exception:
        pass

    if not results and not pending_messages and not urgent_signals and not recent_graph_changes:
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

    # Format output
    lines = []

    # Urgent signals go FIRST — these are the brain's alarm system
    if urgent_signals:
        lines.append("BRAIN AWARENESS:")
        for sig in urgent_signals:
            lines.append("  " + sig)
        lines.append("")

    # Graph changes — what happened since last prompt
    if recent_graph_changes:
        lines.append("GRAPH ACTIVITY (since last prompt):")
        for change in recent_graph_changes[-10:]:  # cap display at 10
            lines.append("  " + change)
        lines.append("")

    if results:
        lines.append("BRAIN RECALL (auto-surfaced for this conversation):")
        lines.append("")

    if segment_note:
        lines.append(segment_note)
        lines.append("")

    if priming_note:
        lines.append(priming_note)
        lines.append("")

    _format_recall_results(results, lines)

    # Aspiration compass
    try:
        relevant_aspirations = brain.get_relevant_aspirations(user_message[:200], limit=2)
        recalled_ids = set(r.get("id") for r in results)
        new_aspirations = [a for a in relevant_aspirations if a.get("id") not in recalled_ids]
        if new_aspirations:
            lines.append("ASPIRATION COMPASS (relevant to this conversation):")
            for a in new_aspirations:
                lines.append("  " + a.get("title", "")[:80])
            lines.append("")
    except Exception:
        pass

    # Hypothesis validation
    try:
        hyp = brain.check_hypothesis_relevance(user_message[:200])
        if hyp and hyp.get("id") not in set(r.get("id") for r in results):
            hconf = hyp.get("confidence", 0)
            lines.append("HYPOTHESIS TO VALIDATE (confidence: %.1f):" % hconf)
            lines.append("  " + hyp.get("title", "")[:80])
            hcontent = str(hyp.get("content", ""))[:120]
            if hcontent:
                lines.append("    " + hcontent)
            lines.append("  Does this conversation confirm or deny this? "
                         "Use brain.confirm_evolution() or brain.dismiss_evolution().")
            lines.append("")
    except Exception:
        pass

    # Active tensions
    try:
        active = brain.get_active_evolutions(["tension"])
        recalled_ids = set(r.get("id") for r in results)
        unrecalled = [a for a in active if a.get("id") not in recalled_ids]
        if unrecalled:
            lines.append("BRAIN AGENDA (active tensions):")
            for a in unrecalled[:2]:
                lines.append("  " + a.get("title", "")[:80])
            lines.append("")
    except Exception:
        pass

    # Host instinct check
    try:
        nudge = brain.get_instinct_check(user_message[:500])
        if nudge:
            lines.insert(0, nudge)
            lines.insert(1, "")
    except Exception:
        pass

    # Append pending messages
    if pending_messages:
        lines.append("")
        lines.append("--- QUEUED MESSAGES (from background hooks) ---")
        for pm in pending_messages:
            lines.append(str(pm))
            lines.append("")

    # ── Precision: request feedback if uncertain about previous recall ──
    try:
        prev_eval_id = brain.get_config("last_evaluated_recall_id", "")
        if prev_eval_id:
            precision = _get_precision(brain)
            fb = precision.request_feedback(int(prev_eval_id))
            if fb:
                lines.append("")
                lines.append(fb)
    except Exception:
        pass

    lines.append("Use this context to inform your response. Call /remember for new decisions.")
    context = "\n".join(lines)

    brain.save()
    return {"json": {"additionalContext": context}}


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
    recall_log_id = brain.get_config("last_recall_log_id", "")

    if recall_log_id and assistant_response and len(assistant_response) >= 20:
        try:
            precision = _get_precision(brain)
            precision.evaluate_response(int(recall_log_id), assistant_response)
            # Store for followup evaluation on next user message
            brain.set_config("last_evaluated_recall_id", recall_log_id)
            # Clear to prevent double-evaluation
            brain.set_config("last_recall_log_id", "")
        except Exception:
            pass
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
            # Extract candidate terms
            quoted = re.findall(r'["]([\w\s-]{3,30})["]', user_message)
            the_patterns = re.findall(
                r"\bthe\s+([\w][\w\s-]{2,25}(?:hook|script|table|file|function|method|class|"
                r"module|layer|loop|sequence|pipeline|system|engine|server|db|database|config|"
                r"schema|signal|node|type|map|graph|cache|queue|log|test|spec))\b",
                user_message, re.IGNORECASE,
            )
            action_context = re.findall(
                r"\b(?:fix|update|change|modify|check|look at|review|debug|test|refactor|"
                r"rewrite|add|remove|delete|move|rename|split|merge|clean)\s+(?:the\s+)?"
                r"([\w]+(?:\s+[\w]+){0,3}?)(?:\s*(?:and|or|but|also|then|,|;|\.|!|\?)|\s*$)",
                user_message, re.IGNORECASE,
            )
            action_context = [t.strip() for t in action_context if len(t.strip()) >= 3]
            hyphenated = re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", user_message)
            hyphenated = [h for h in hyphenated if len(h) > 4 and h.lower() not in (
                "re-run", "re-do", "non-null", "non-zero", "up-to-date", "e-g",
                "pre-existing", "co-authored-by",
            )]

            skip_words = {
                "the", "a", "an", "this", "that", "it", "them", "is", "are",
                "was", "were", "be", "been", "do", "does", "did", "have", "has",
                "can", "could", "will", "would", "should", "may", "might",
                "yes", "no", "ok", "sure", "please", "thanks", "code", "file",
                "thing", "stuff", "something", "everything", "nothing",
            }
            candidates = set()
            for term in quoted + the_patterns + action_context + hyphenated:
                term = term.strip().lower()
                if len(term) < 3 or len(term) > 40:
                    continue
                words = term.split()
                if all(w in skip_words for w in words):
                    continue
                candidates.add(term)

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

            checkpoint_lines = ["ENCODING CHECKPOINT: " + msg]
            if stats:
                checkpoint_lines.append(stats)
            checkpoint_lines.append(focus)
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

    # 9. Session health check
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
        summary = "IDLE MAINTENANCE:\n" + "\n".join(output)
        _store_pending(brain, summary)

    brain.save()
    return {"output": ""}  # Notification stdout invisible


def hook_post_compact_reboot(brain, args, graph_changes):
    """Post-compact reboot — re-inject brain context after compaction.

    PostCompact stdout IS visible. This is the safety net.
    """
    output = ["BRAIN POST-COMPACTION REBOOT:", ""]

    user = brain.get_config("default_user", "User")
    project = brain.get_config("default_project", "default")

    # Safety net: check if pre-compact synthesis ran
    try:
        last_synth = brain.conn.execute(
            "SELECT created_at FROM session_syntheses ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        session_start = brain.get_config("session_start_at", "")

        synth_ran = False
        if last_synth and session_start:
            synth_ran = last_synth[0] >= session_start

        if not synth_ran:
            output.append("NOTE: Pre-compact synthesis did not run. Running now...")
            try:
                synthesis = brain.synthesize_session()
                parts = []
                for key in ("decisions", "corrections", "open_questions"):
                    val = synthesis.get(key)
                    if val:
                        parts.append("%s %s" % (val, key))
                if parts:
                    output.append("  Synthesis: " + ", ".join(parts))
                else:
                    output.append("  Synthesis: no notable events captured")
            except Exception as e:
                output.append("  Synthesis failed: %s" % e)
            output.append("")
    except Exception:
        pass

    # Re-run lightweight boot
    boot = brain.context_boot(user=user, project=project, task="post-compaction reboot")

    # Locked rules
    locked_nodes = boot.get("locked", [])
    locked_rules = [n for n in locked_nodes if n.get("type") == "rule"]
    if locked_rules:
        output.append("LOCKED RULES (%d active):" % len(locked_rules))
        for rule in locked_rules[:15]:
            output.append("  %s" % rule.get("title", "")[:80])
        output.append("")

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

                if age_minutes < 30:
                    oq = synth_row[0]
                    if oq:
                        try:
                            questions = json.loads(oq)
                            if questions:
                                output.append("OPEN QUESTIONS (from synthesis %d min ago):" % int(age_minutes))
                                for q in questions[:5]:
                                    output.append("  ? %s" % str(q)[:100])
                                output.append("")
                        except Exception:
                            pass
                else:
                    output.append("NOTE: Last synthesis was %.0f hours ago - open questions may be resolved." % (age_minutes / 60))
                    output.append("  Use brain.recall_with_embeddings() for current context instead.")
                    output.append("")
            except Exception:
                pass
    except Exception:
        pass

    # Consciousness signals
    signals = brain.get_consciousness_signals()
    for sig_key, sig_label in [("reminders", "REMINDERS"), ("evolutions", "EVOLUTIONS")]:
        items = signals.get(sig_key, [])
        if items:
            output.append("%s:" % sig_label)
            for item in items[:5]:
                output.append("  %s" % item.get("title", "")[:80])
            output.append("")

    # Developmental stage
    try:
        dev = brain.assess_developmental_stage()
        if dev:
            output.append("STAGE: %s (%.0f%%)" % (dev.get("stage_name", "?"), dev.get("maturity_score", 0) * 100))
            output.append("")
    except Exception:
        pass

    # Recall context related to recent work
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

            recall_results = result.get("results", [])
            recent_ids = {r[0] for r in brain.conn.execute(
                "SELECT id FROM nodes WHERE created_at > datetime('now', '-2 hours')"
            ).fetchall()}
            new_recall = [r for r in recall_results if r.get("id") not in recent_ids]

            if new_recall:
                output.append("RECALLED CONTEXT (related to recent work):")
                for r in new_recall[:6]:
                    typ = r.get("type", "?")
                    title = r.get("title", "")[:70]
                    content = r.get("content", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    locked = "LOCKED " if r.get("locked") else ""
                    output.append("  [%s] %s%s" % (typ, locked, title))
                    output.append("    %s" % content)
                    output.append("")
    except Exception:
        pass

    # Find transcript for rehydration hint
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
                db_dir_env = os.environ.get("BRAIN_DB_DIR", "")
                plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT", ".")
                output.append("")
                output.append("TRANSCRIPT AVAILABLE FOR REHYDRATION:")
                output.append("  Path: %s" % transcript_path)
                output.append("  To recover lost context, run:")
                output.append("    BRAIN_DB_DIR=%s python3 %s/hooks/scripts/extract-session-log.py --last-n-hours 4" % (
                    db_dir_env, plugin_root))
                output.append("  Or read the transcript directly to find what you lost.")
    except Exception:
        pass

    # Drain pending messages
    pending = _drain_pending(brain)
    if pending:
        output.append("")
        output.append("--- QUEUED MESSAGES (from background hooks) ---")
        for pm in pending:
            output.append(str(pm))
            output.append("")

    output.append("Brain is live. Context was compacted — you lost conversation history.")
    output.append("The brain persists. Use brain.recall_with_embeddings() to recover context.")

    brain.save()
    return {"output": "\n".join(output)}


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
            "reason": "\u26a0\ufe0f Destructive command detected. Safety check failed — proceed carefully.",
        }}

    critical_matches = result.get("critical_matches", [])
    warnings = result.get("warnings", [])

    if critical_matches:
        lines = ["\u26a0\ufe0f BRAIN SAFETY: This command may affect critical brain-tracked resources:"]
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
        return {"json": {"decision": "block", "reason": "\n".join(lines)}}

    elif warnings:
        lines = ["\u26a0\ufe0f BRAIN WARNING: Destructive command detected. Relevant brain context:"]
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
        return {"json": {"decision": "approve", "reason": "\n".join(lines)}}

    else:
        return {"json": {
            "decision": "approve",
            "reason": "\u26a0\ufe0f Destructive command detected. No brain safety rules match, but proceed carefully.",
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
            output_lines = ["HOST ENVIRONMENT CHANGED (detected by brain):"]
            for key, change in changes.items():
                old_val = change.get("old", "?")
                new_val = change.get("new", "?")
                output_lines.append("  %s: %s \u2192 %s" % (key, old_val, new_val))
            output_lines.append("  Trigger: config change in %s" % source)
            if file_path:
                output_lines.append("  File: %s" % file_path)
            output_lines.append("")
            output_lines.append("Review arch_constraint and capability nodes that may be affected.")

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
            output_lines = ["HOST ENVIRONMENT CHANGED (after bash command):"]
            for key, change in changes.items():
                old_val = change.get("old", "?")
                new_val = change.get("new", "?")
                output_lines.append("  %s: %s \u2192 %s" % (key, old_val, new_val))
            output_lines.append("  Command: %s" % command[:100])

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
        "GIT CONTEXT (brain is tracking):",
        "  Worktree: " + worktree_name,
        "  Branch: " + branch,
        "  CWD: " + cwd,
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
