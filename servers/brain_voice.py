"""Brain Voice — DECIDE + FORMAT layer for brain consciousness output.

Consolidates all formatting and signal selection logic that was previously
scattered across brain_surface.py and daemon_hooks.py. BrainVoice is a
collaborator object (not a mixin) that takes a Brain instance and produces
formatted output for two channels:
  - for_claude: reasoning context, wrapped in [BRAIN]...[/BRAIN]
  - for_operator: human-facing content, wrapped in [BRAIN-To-{name}]...[/BRAIN-To-{name}]

Both channels are merged into a single additionalContext string via wrap_for_hook().
Claude relays operator content faithfully, respecting @priority directives (high/medium/low).

Architecture: COMPUTE (brain_consciousness.py) → DECIDE+FORMAT (here) → DELIVER (daemon_hooks.py)
"""

import json
from typing import List, Dict, Any, Optional, Callable, Union

from . import embedder
from .schema import BRAIN_VERSION


# ── Constants (shared with daemon_hooks.py) ──

EVOLUTION_TYPES = {"tension", "hypothesis", "pattern", "catalyst", "aspiration"}

ENGINEERING_TYPES = {"purpose", "mechanism", "impact", "constraint", "convention",
                     "lesson", "vocabulary"}
CODE_COGNITION_TYPES = {"fn_reasoning", "param_influence", "code_concept",
                        "arch_constraint", "causal_chain", "bug_lesson",
                        "comment_anchor"}


class BrainVoice:
    """Formats brain consciousness output for Claude and operator channels.

    Usage:
        voice = BrainVoice(brain)
        voice.format_recall_results(results, lines)
        warning = voice.format_encoding_warning(encoding)
        output = voice.format_suggestions(filename, suggestions, ...)
    """

    def __init__(self, brain):
        self.brain = brain

    # ── Formatting primitives (moved from brain_surface.py) ──

    @staticmethod
    def fl(items, header, max_n=5, fmt=None, suffix=None, indent="  "):
        """Format a list section. Returns lines or [] if items is empty.
        fmt: callable(item) -> str or list[str]. Default: item title.
        """
        if not items:
            return []
        out = [header]
        for item in items[:max_n]:
            if fmt:
                result = fmt(item)
                if isinstance(result, list):
                    out.extend(result)
                else:
                    out.append("%s%s" % (indent, result))
            else:
                out.append("%s%s" % (indent, str(item.get("title", ""))[:80] if isinstance(item, dict) else str(item)[:80]))
        if len(items) > max_n and suffix is None:
            out.append("%s... and %d more" % (indent, len(items) - max_n))
        if suffix:
            out.append("%s%s" % (indent, suffix))
        out.append("")
        return out

    @staticmethod
    def trunc(s, n=80):
        """Truncate string to n chars."""
        s = str(s or "")
        return s[:n] + "..." if len(s) > n else s

    # ── Recall formatting (moved from daemon_hooks.py) ──

    @staticmethod
    def format_recall_results(results, lines):
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

    @staticmethod
    def format_encoding_warning(encoding):
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

    @staticmethod
    def format_suggestions(filename, suggestions, procedures, context_files,
                           change_impacts, encoding_warning):
        """Format brain suggestions into readable output for pre-edit hook."""
        lines = ["[BRAIN] AUTO-SUGGEST for %s:" % filename, ""]

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
        lines.append("[/BRAIN]")
        return "\n".join(lines)

    # ── Operator channel (Brain → Tom) ──

    @staticmethod
    def format_for_operator(items: List[str]) -> Optional[str]:
        """Format items for operator-visible channel.

        Returns None if nothing noteworthy to surface.
        Items should be short, one-line summaries prefixed with emoji.
        """
        if not items:
            return None
        return "\n".join(items)

    def wrap_for_hook(self, for_claude: str, for_operator: str = None) -> str:
        """Single wrapping point — merges both channels into tagged output.

        Everything Brain wants to tell Claude goes in [BRAIN]...[/BRAIN].
        Everything Brain wants to tell the operator goes in [BRAIN-To-{name}]...[/BRAIN-To-{name}].
        Claude relays the operator section faithfully, respecting @priority directives.
        """
        # Operator section goes FIRST — if hook times out, Claude content is cut,
        # not the operator's messages. The human's reminders and alerts matter more
        # than Claude's reasoning context.
        parts = []
        if for_operator and for_operator.strip():
            host = self.brain.get_config("host_name") or "Operator"
            parts.append("[BRAIN-To-%s]\n%s\n[/BRAIN-To-%s]" % (host, for_operator, host))
        parts.append(for_claude)  # for_claude already has [BRAIN]...[/BRAIN] wrapping
        return "\n\n".join(parts)

    def render_operator_prompt(self, prompt_signals: Dict[str, Any],
                                urgent_signals: List[str] = None) -> Optional[str]:
        """Curate operator-facing content for pre-response hook.

        Gathers consciousness signals and prioritizes them:
        - @priority: high — reminders, health alerts, errors (always included)
        - @priority: medium — tensions, stale reasoning (included if interesting)
        - @priority: low — dreams, novelty, fading (included if space allows)

        Budget: ~800 chars max (excluding debug).
        Returns None if nothing to surface.
        """
        brain = self.brain
        sections = []  # (priority_order, text)
        budget = 800

        # HIGH — Due reminders (always, never budget-cut)
        try:
            reminders = brain.get_due_reminders()
            if reminders:
                r_lines = ["@priority: high"]
                for rem in reminders[:3]:
                    title = rem.get("title", "Untitled")
                    due = rem.get("due_date", "")[:16]
                    r_lines.append("🔔 %s (due %s)" % (title, due))
                sections.append((0, "\n".join(r_lines)))
        except Exception:
            pass

        # HIGH — Urgent signals (health alerts, errors)
        if urgent_signals:
            u_lines = ["@priority: high"]
            for sig in urgent_signals[:2]:
                u_lines.append("⚠️  %s" % sig[:100])
            sections.append((1, "\n".join(u_lines)))

        # MEDIUM — Active tensions
        tensions = prompt_signals.get('tensions', [])
        if tensions:
            t_lines = ["@priority: medium"]
            for t in tensions[:2]:
                t_lines.append("⚡ Tension: %s" % self.trunc(t.get("title", ""), 80))
            sections.append((2, "\n".join(t_lines)))

        # MEDIUM — Hypothesis to validate
        hyp = prompt_signals.get('hypothesis')
        if hyp:
            sections.append((3, "@priority: medium\n❓ Hypothesis: %s" % self.trunc(hyp.get("title", ""), 80)))

        # LOW — Precision feedback request (ask operator if recall was useful)
        try:
            from servers.brain_precision import RecallPrecision
            dal = getattr(brain, '_logs_dal', None)
            precision = RecallPrecision(brain.logs_conn, brain.conn, logs_dal=dal)
            # Find the most recent ask_operator signal
            row = brain.logs_conn.execute(
                """SELECT id, recalled_titles, returned_count FROM recall_log
                   WHERE followup_signal = 'ask_operator' AND explicit_feedback IS NULL
                   ORDER BY created_at DESC LIMIT 1""").fetchone()
            if row:
                titles = []
                try:
                    titles = list(json.loads(row[1]).values())[:3] if row[1] else []
                except Exception:
                    pass
                if titles:
                    title_list = ", ".join(t[:40] for t in titles)
                    sections.append((3, "@priority: low\n@present: aside\n🎯 Brain recalled %d nodes but isn't sure if they helped: %s\n   (say \"useful\", \"not useful\", or ignore)" % (row[2], title_list)))
        except Exception:
            pass

        # LOW — Dreams
        try:
            signals = brain.get_consciousness_signals()
            dreams = signals.get("dreams", [])
            if dreams:
                d = dreams[0]
                if isinstance(d, dict):
                    dream_text = d.get("text", str(d))
                elif isinstance(d, str):
                    dream_text = d
                else:
                    dream_text = str(d)
                sections.append((4, "@priority: low\n💭 Dream: %s" % self.trunc(dream_text, 100)))
        except Exception:
            pass

        # MEDIUM — Context pressure warning (estimated from hook fire count)
        try:
            recall_count = brain.logs_conn.execute(
                """SELECT COUNT(*) FROM debug_log
                   WHERE source='hook_telemetry' AND event_type='hook_recall'
                   AND created_at > datetime('now', '-6 hours')""").fetchone()[0]
            # Rough estimate: ~2800 tokens per turn (user + claude + hooks + tools)
            est_tokens = recall_count * 2800
            est_pct = int(est_tokens * 100 / 1000000)  # 1M context
            if est_pct >= 60:
                sections.append((1, "@priority: high\n⚠️ Context ~%d%% full (~%dK tokens, %d turns). Quality may degrade. Consider compacting." % (
                    est_pct, est_tokens // 1000, recall_count)))
            elif est_pct >= 40:
                sections.append((2, "@priority: medium\n📊 Context ~%d%% full (~%dK tokens, %d turns). Still healthy." % (
                    est_pct, est_tokens // 1000, recall_count)))
        except Exception:
            pass

        # LOW — Thin locked nodes needing enrichment (brain cleanup)
        try:
            thin = brain.conn.execute(
                """SELECT title, LENGTH(content) FROM nodes
                   WHERE locked=1 AND archived=0 AND LENGTH(content) < 100
                   ORDER BY RANDOM() LIMIT 1""").fetchone()
            if thin:
                sections.append((5, "@priority: low\n@present: aside\n📝 Old node needs enrichment: \"%s\" (%d chars, no WHY). Can you help me remember it better?" % (
                    self.trunc(thin[0], 50), thin[1])))
        except Exception:
            pass

        if not sections:
            return None

        # Assemble within budget (high priority items always included)
        sections.sort(key=lambda x: x[0])
        result_parts = []
        char_count = 0
        for priority_order, text in sections:
            if priority_order <= 1:
                # High priority — always include, ignore budget
                result_parts.append(text)
                char_count += len(text)
            elif char_count + len(text) < budget:
                result_parts.append(text)
                char_count += len(text)

        if not result_parts:
            return None
        return "\n\n".join(result_parts)

    def _operator_recall_summary(self, results: List[Dict],
                                  prompt_signals: Dict[str, Any],
                                  urgent_signals: List[str] = None) -> Optional[str]:
        """Build operator summary for recall hook."""
        items = []

        if urgent_signals:
            items.append("⚠️  %d urgent signal(s)" % len(urgent_signals))

        if results:
            types = {}
            for r in results:
                t = r.get("type", "?")
                types[t] = types.get(t, 0) + 1
            type_str = ", ".join("%d %s" % (v, k) for k, v in sorted(types.items(), key=lambda x: -x[1]))
            locked_count = sum(1 for r in results if r.get("locked"))
            lock_note = " (%d locked)" % locked_count if locked_count else ""
            items.append("🧠 Recalled %d nodes: %s%s" % (len(results), type_str, lock_note))

        tensions = prompt_signals.get('tensions', [])
        if tensions:
            items.append("⚡ %d active tension(s)" % len(tensions))

        hyp = prompt_signals.get('hypothesis')
        if hyp:
            items.append("❓ Hypothesis to validate: %s" % self.trunc(hyp.get("title", ""), 50))

        aspirations = prompt_signals.get('aspirations', [])
        if aspirations:
            items.append("🎯 %d aspiration(s) relevant" % len(aspirations))

        return self.format_for_operator(items)

    def _operator_boot_summary(self, node_count, edge_count, locked_count,
                                signal_count: int = 0, alert_count: int = 0,
                                consciousness_signals: Dict = None) -> Optional[str]:
        """Build operator summary for boot with consciousness highlights.

        Returns curated content for the operator channel, tagged with @priority.
        """
        sections = []

        # HIGH — Due reminders
        try:
            reminders = self.brain.get_due_reminders()
            if reminders:
                r_lines = ["@priority: high"]
                for rem in reminders[:3]:
                    r_lines.append("🔔 %s (due %s)" % (rem.get("title", ""), rem.get("due_date", "")[:16]))
                sections.append("\n".join(r_lines))
        except Exception:
            pass

        # HIGH — Health alerts
        if alert_count:
            sections.append("@priority: high\n⚠️  %d health alert(s) — check boot output" % alert_count)

        # MEDIUM — Top tensions
        cs = consciousness_signals or {}
        evolutions = cs.get("evolutions", [])
        tensions = [e for e in evolutions if isinstance(e, dict) and "tension" in str(e.get("type", "")).lower()]
        if tensions:
            t_lines = ["@priority: medium"]
            for t in tensions[:2]:
                t_lines.append("⚡ %s" % self.trunc(t.get("title", str(t)), 80))
            sections.append("\n".join(t_lines))

        # LOW — Dreams
        dreams = cs.get("dreams", [])
        if dreams:
            d = dreams[0]
            dream_text = d.get("text", str(d)) if isinstance(d, dict) else str(d)
            sections.append("@priority: low\n💭 Dream: %s" % self.trunc(dream_text, 100))

        # LOW — Stats
        sections.append("@priority: low\n🧠 %s nodes, %s edges, %s locked, %d signal(s)" % (
            node_count, edge_count, locked_count, signal_count))

        if not sections:
            return None
        return "\n\n".join(sections)

    def _operator_reboot_summary(self, locked_count: int = 0,
                                  recall_count: int = 0) -> Optional[str]:
        """Build operator summary for post-compact reboot."""
        items = []
        items.append("🧠 Post-compaction reboot")
        if locked_count:
            items.append("🔒 %d locked rules restored" % locked_count)
        if recall_count:
            items.append("📎 %d nodes recalled for context" % recall_count)
        return self.format_for_operator(items)

    # ── Signal selection (DECIDE layer) ──

    def select_prompt_signals(self, user_message: str, recall_results: List[Dict]) -> Dict[str, Any]:
        """Select consciousness signals to surface during recall (prompt-time).

        Returns dict with keys: aspirations, hypothesis, tensions, instinct_nudge.
        All items are already filtered to exclude duplicates from recall_results.
        """
        brain = self.brain
        recalled_ids = set(r.get("id") for r in recall_results)
        signals = {}

        # Aspiration compass
        try:
            relevant = brain.get_relevant_aspirations(user_message[:200], limit=2)
            signals['aspirations'] = [a for a in relevant if a.get("id") not in recalled_ids]
        except Exception:
            signals['aspirations'] = []

        # Hypothesis validation
        try:
            hyp = brain.check_hypothesis_relevance(user_message[:200])
            if hyp and hyp.get("id") not in recalled_ids:
                signals['hypothesis'] = hyp
            else:
                signals['hypothesis'] = None
        except Exception:
            signals['hypothesis'] = None

        # Active tensions
        try:
            active = brain.get_active_evolutions(["tension"])
            signals['tensions'] = [a for a in active if a.get("id") not in recalled_ids][:2]
        except Exception:
            signals['tensions'] = []

        # Host instinct check
        try:
            signals['instinct_nudge'] = brain.get_instinct_check(user_message[:500])
        except Exception:
            signals['instinct_nudge'] = None

        return signals

    def render_prompt(self, results: List[Dict], prompt_signals: Dict[str, Any],
                      urgent_signals: List[str] = None,
                      segment_note: str = None, priming_note: str = None,
                      graph_changes: List[str] = None,
                      pending_messages: List[str] = None,
                      debug_messages: List[str] = None,
                      precision_feedback: str = None) -> Dict[str, Optional[str]]:
        """Format recall output for both channels.

        Takes pre-gathered data (from hook_recall's COMPUTE phase) and
        prompt_signals (from select_prompt_signals) and formats them.

        Returns:
            {'for_claude': str, 'for_operator': str|None}
        """
        lines = []

        # Urgent signals go FIRST — these are the brain's alarm system
        if urgent_signals:
            lines.append("[BRAIN] AWARENESS:")
            for sig in urgent_signals:
                lines.append("  " + sig)
            lines.append("")

        # Graph changes — what happened since last prompt
        if graph_changes:
            lines.append("[BRAIN] GRAPH ACTIVITY (since last prompt):")
            for change in graph_changes[-10:]:
                lines.append("  " + change)
            lines.append("")

        if results:
            lines.append("[BRAIN] RECALL (auto-surfaced for this conversation):")
            lines.append("")

        if segment_note:
            lines.append(segment_note)
            lines.append("")

        if priming_note:
            lines.append(priming_note)
            lines.append("")

        self.format_recall_results(results, lines)

        # Aspiration compass
        aspirations = prompt_signals.get('aspirations', [])
        if aspirations:
            lines.append("[BRAIN] ASPIRATION COMPASS (relevant to this conversation):")
            for a in aspirations:
                lines.append("  " + a.get("title", "")[:80])
            lines.append("")

        # Hypothesis validation
        hyp = prompt_signals.get('hypothesis')
        if hyp:
            hconf = hyp.get("confidence", 0)
            lines.append("[BRAIN] HYPOTHESIS TO VALIDATE (confidence: %.1f):" % hconf)
            lines.append("  " + hyp.get("title", "")[:80])
            hcontent = str(hyp.get("content", ""))[:120]
            if hcontent:
                lines.append("    " + hcontent)
            lines.append("  Does this conversation confirm or deny this? "
                         "Use brain.confirm_evolution() or brain.dismiss_evolution().")
            lines.append("")

        # Active tensions
        tensions = prompt_signals.get('tensions', [])
        if tensions:
            lines.append("[BRAIN] AGENDA (active tensions):")
            for a in tensions:
                lines.append("  " + a.get("title", "")[:80])
            lines.append("")

        # Host instinct check — goes at the TOP
        nudge = prompt_signals.get('instinct_nudge')
        if nudge:
            lines.insert(0, nudge)
            lines.insert(1, "")

        # Pending messages
        if pending_messages:
            lines.append("")
            lines.append("--- QUEUED MESSAGES (from background hooks) ---")
            for pm in pending_messages:
                lines.append(str(pm))
                lines.append("")

        # Debug logs
        if debug_messages:
            lines.append("")
            lines.append("[BRAIN DEBUG]")
            for dm in debug_messages:
                lines.append("  " + dm)
            lines.append("")

        # Precision feedback
        if precision_feedback:
            lines.append("")
            lines.append(precision_feedback)

        lines.append("[BRAIN] Use this context to inform your response. Encode decisions, lessons, and corrections back into the brain.")
        lines.append("[/BRAIN]")

        # Operator channel — curated consciousness for the human
        operator_msg = self.render_operator_prompt(prompt_signals, urgent_signals)

        # In debug mode, append full Claude injection to operator channel
        try:
            if self.brain.get_config("debug_enabled", "0") == "1":
                debug_section = "\n\n@priority: low\n[DEBUG] Full Brain→Claude injection (%d chars):\n%s" % (
                    len("\n".join(lines)), "\n".join(lines))
                if operator_msg:
                    operator_msg = operator_msg + debug_section
                else:
                    operator_msg = debug_section
        except Exception:
            pass

        return {'for_claude': "\n".join(lines), 'for_operator': operator_msg}

    def render_reboot(self, boot_context: Dict, synthesis_info: Dict = None,
                      locked_rules: List[Dict] = None,
                      signals: Dict = None, dev_stage: Dict = None,
                      recall_results: List[Dict] = None,
                      pending_messages: List[str] = None,
                      transcript_path: str = None,
                      db_dir_env: str = '', plugin_root: str = '.') -> Dict[str, Optional[str]]:
        """Format post-compaction reboot output for both channels.

        Returns:
            {'for_claude': str, 'for_operator': str|None}
        """
        output = ["[BRAIN] POST-COMPACTION REBOOT:", ""]

        # Synthesis info
        if synthesis_info:
            if synthesis_info.get("just_ran"):
                output.append("NOTE: Pre-compact synthesis did not run. Running now...")
                parts = synthesis_info.get("parts", [])
                if parts:
                    output.append("  Synthesis: " + ", ".join(parts))
                else:
                    output.append("  Synthesis: no notable events captured")
                output.append("")
            elif synthesis_info.get("error"):
                output.append("NOTE: Pre-compact synthesis did not run. Running now...")
                output.append("  Synthesis failed: %s" % synthesis_info["error"])
                output.append("")

        # Locked rules
        if locked_rules:
            output.append("LOCKED RULES (%d active):" % len(locked_rules))
            for rule in locked_rules[:15]:
                output.append("  %s" % rule.get("title", "")[:80])
            output.append("")

        # Open questions from synthesis
        if synthesis_info and synthesis_info.get("open_questions"):
            oq = synthesis_info["open_questions"]
            age_min = synthesis_info.get("age_minutes", 0)
            if age_min < 30:
                output.append("OPEN QUESTIONS (from synthesis %d min ago):" % int(age_min))
                for q in oq[:5]:
                    output.append("  ? %s" % str(q)[:100])
                output.append("")
            else:
                output.append("NOTE: Last synthesis was %.0f hours ago - open questions may be resolved." % (age_min / 60))
                output.append("  Use brain.recall_with_embeddings() for current context instead.")
                output.append("")

        # Consciousness signals (lighter than boot — just reminders + evolutions)
        if signals:
            for sig_key, sig_label in [("reminders", "REMINDERS"), ("evolutions", "EVOLUTIONS")]:
                items = signals.get(sig_key, [])
                if items:
                    output.append("%s:" % sig_label)
                    for item in items[:5]:
                        output.append("  %s" % item.get("title", "")[:80])
                    output.append("")

        # Developmental stage
        if dev_stage:
            output.append("STAGE: %s (%.0f%%)" % (dev_stage.get("stage_name", "?"), dev_stage.get("maturity_score", 0) * 100))
            output.append("")

        # Recalled context
        if recall_results:
            output.append("RECALLED CONTEXT (related to recent work):")
            for r in recall_results[:6]:
                typ = r.get("type", "?")
                title = r.get("title", "")[:70]
                content = r.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                locked = "LOCKED " if r.get("locked") else ""
                output.append("  [%s] %s%s" % (typ, locked, title))
                output.append("    %s" % content)
                output.append("")

        # Transcript path
        if transcript_path:
            output.append("")
            output.append("TRANSCRIPT AVAILABLE FOR REHYDRATION:")
            output.append("  Path: %s" % transcript_path)
            output.append("  To recover lost context, run:")
            output.append("    BRAIN_DB_DIR=%s python3 %s/hooks/scripts/extract-session-log.py --last-n-hours 4" % (
                db_dir_env, plugin_root))
            output.append("  Or read the transcript directly to find what you lost.")

        # Pending messages
        if pending_messages:
            output.append("")
            output.append("--- QUEUED MESSAGES (from background hooks) ---")
            for pm in pending_messages:
                output.append(str(pm))
                output.append("")

        output.append("Brain is live. Context was compacted — you lost conversation history.")
        output.append("The brain persists. Use brain.recall_with_embeddings() to recover context.")
        output.append("[/BRAIN]")

        operator_msg = self._operator_reboot_summary(
            locked_count=len(locked_rules) if locked_rules else 0,
            recall_count=len(recall_results) if recall_results else 0,
        )

        return {'for_claude': "\n".join(output), 'for_operator': operator_msg}

    # ── Boot context rendering (moved from brain_surface.py) ──

    def render_boot(self, user: str = 'User', project: str = 'default',
                    db_dir: str = '') -> Dict[str, Optional[str]]:
        """
        Gather all boot data and return formatted text for both channels.

        Returns:
            {'for_claude': str, 'for_operator': str|None}
            for_claude: full boot context for Claude's context window
            for_operator: curated consciousness highlights for the human operator
        """
        brain = self.brain
        out = []
        _fl = self.fl
        _t = self.trunc

        # ── Gather data ──────────────────────────────────────────────
        ctx = brain.context_boot(user=user, project=project, task="session start")

        def _safe(fn, default=None):
            try:
                return fn()
            except Exception as e:
                brain._log_error("boot_%s" % fn.__name__, str(e), "")
                return default if default is not None else ([] if 'pattern' in fn.__name__ else None)

        eng_ctx = _safe(lambda: brain.get_engineering_context(project=project), {})
        correction_patterns = _safe(lambda: brain.get_correction_patterns(limit=5), [])
        last_synthesis = _safe(lambda: brain.get_last_synthesis())
        health = brain.health_check(session_id="session_boot", auto_fix=True)
        staged = brain.list_staged(status="pending", limit=10)
        brain.auto_promote_staged(revisit_threshold=3)
        metrics = brain.get_suggest_metrics(period_days=7)
        procs = brain.procedure_trigger("session_start", {"session_count": ctx.get("reset_count", 0)})
        cs = brain.get_consciousness_signals()
        dev_stage = brain.assess_developmental_stage()
        host_result = brain.scan_host_environment()
        host_diff = host_result.get("diff", {})
        host_research = host_result.get("research_needed", [])
        dreams = brain.get_surfaceable_dreams(limit=2)
        brain.auto_generate_self_reflection()
        brain.save()

        # ── Header ───────────────────────────────────────────────────
        debug_enabled = brain.get_config("debug_enabled", "0") == "1"
        out.append("[BRAIN] v%s booted from: %s" % (BRAIN_VERSION, db_dir))
        if debug_enabled:
            out.append("[BRAIN] DEBUG MODE ON")
        out.append("")
        out.append("Session #%d" % (ctx.get("reset_count", 0) + 1))
        out.append("")

        # Last session note
        note = ctx.get("last_session_note") or {}
        if note:
            out.append("Last session note: %s" % note.get("title", ""))
            if note.get("content"):
                out.append(note["content"][:500])
            out.append("")

        # ── Engineering context ──────────────────────────────────────
        sp = eng_ctx.get("system_purpose") or {}
        if sp.get("purpose"):
            out.append("SYSTEM PURPOSE: %s" % _t(sp["purpose"], 300))
            if sp.get("architecture"):
                out.append("Architecture: %s" % _t(sp["architecture"], 200))
            out.append("")

        out.extend(_fl(eng_ctx.get("purposes", []), "PROJECT UNDERSTANDING:", 8,
                        fmt=lambda p: "[%s] %s" % (p.get("scope", "?"), _t(p.get("title", ""), 70))))

        # Last session synthesis
        if last_synthesis:
            ls_date = str(last_synthesis.get("created_at", ""))[:10]
            hdr = "LAST SESSION (%s)" % ls_date if ls_date else "LAST SESSION"
            out.append("%s:" % hdr)
            for d in last_synthesis.get("decisions_made", [])[:3]:
                out.append("  Decision: %s" % _t(d.get("title", d) if isinstance(d, dict) else d, 80))
            for c in last_synthesis.get("corrections_received", [])[:2]:
                out.append("  Correction: %s" % _t(c.get("assumed", c) if isinstance(c, dict) else c, 80))
            for q in last_synthesis.get("open_questions", [])[:3]:
                out.append("  ? %s" % _t(q.get("text", q) if isinstance(q, dict) else q, 80))
            out.append("")

        out.extend(_fl(eng_ctx.get("file_changes", []), "FILES CHANGED since last session:", 10,
                        fmt=lambda fc: fc.get("file_path", ""),
                        suffix="Re-read changed files to update your understanding."))

        out.extend(_fl(correction_patterns, "CORRECTION PATTERNS:", 3,
                        fmt=lambda cp: "[%s x%d] %s" % (
                            cp.get("max_severity", "minor"), cp.get("count", 0), _t(cp.get("pattern", ""), 70))))

        out.extend(_fl(eng_ctx.get("vocabulary", []), "OPERATOR VOCABULARY:", 10,
                        fmt=lambda v: "%s => %s" % (_t(v.get("title", ""), 40), _t(v.get("content", ""), 80))))

        # Constraints, conventions, impacts — all title+optional content lists
        for key, label, n in [("constraints", "CONSTRAINTS:", 5), ("conventions", "CONVENTIONS:", 3), ("impacts", "CHANGE IMPACT MAP:", 5)]:
            items = eng_ctx.get(key, [])
            out.extend(_fl(items, label, n, fmt=lambda c: _t(c.get("title", ""), 70)))

        # File inventory
        _fi = eng_ctx.get("file_inventory", [])
        file_inv = _fi.get("files", []) if isinstance(_fi, dict) else _fi if isinstance(_fi, list) else []
        out.extend(_fl(file_inv, "FILE INVENTORY (%d tracked):" % len(file_inv), 8,
                        fmt=lambda fi: "%s — %s" % (fi.get("file_path", ""), _t(fi.get("purpose", ""), 60))))

        # ── Health ───────────────────────────────────────────────────
        issues = health.get("issues", [])
        high = [i for i in issues if i.get("severity") == "high"]
        medium = [i for i in issues if i.get("severity") == "medium"]
        out.extend(_fl(high, "HEALTH ALERTS:", 10, fmt=lambda i: "[%s] %s" % (i.get("type", "?"), i.get("message", ""))))
        out.extend(_fl(medium, "Health warnings:", 10, fmt=lambda i: "[%s] %s" % (i.get("type", "?"), i.get("message", ""))))
        if health.get("actions"):
            out.append("Auto-maintenance: %s" % "  ".join(health["actions"]))
            out.append("")

        out.extend(_fl(procs.get("matched", []), "Procedures to run this session:", 5,
                        fmt=lambda p: "[%s] %s" % (p.get("category", ""), p.get("title", ""))))

        # Locked rules
        rules = [n for n in ctx.get("locked", []) if n.get("type") == "rule"][:10]
        out.extend(_fl(rules, "[BRAIN] Key locked rules:", 10, fmt=lambda r: "- %s" % r.get("title", "")))

        # Staged learnings
        pending = staged.get("staged", [])
        if pending:
            out.extend(_fl(pending, "[BRAIN] STAGED LEARNINGS (%d pending review):" % len(pending), 5,
                            fmt=lambda s: "[%.1f] %s (revisited %dx)" % (
                                s.get("confidence", 0.2),
                                str(s.get("title", "")).replace("[staged] ", ""),
                                s.get("times_revisited", 0))))

        # ── Consciousness ────────────────────────────────────────────
        # Warning signals (single .get("warning") pattern)
        warning_signals = ["encoding_gap", "encoding_depth", "encoding_bias", "density_shift"]
        has_warnings = any(cs.get(k) for k in warning_signals)

        # List signals (title-based)
        list_signals = [
            ("reminders", "REMINDERS:", 5, lambda r: "%s (due: %s)" % (_t(r.get("title", ""), 70), str(r.get("due_date", ""))[:10])),
            ("evolutions", None, 6, lambda e: "%s%s — since %s" % (
                _t(e.get("title", ""), 70),
                " (conf: %.1f)" % e["confidence"] if e.get("confidence") is not None else "",
                str(e.get("created_at", ""))[:10])),
            ("failure_modes", None, 5, None),
            ("fading", "FADING KNOWLEDGE (untouched 14+ days):", 5,
             lambda f: "%s — last: %s" % (_t(f.get("title", ""), 70), str(f.get("last_accessed", ""))[:10])),
            ("fluid_personal", "FLUID PERSONAL — confirm or update:", 5,
             lambda fp: "? %s — still true?" % _t(fp.get("title", ""), 70)),
            ("performance", None, 2, None),
            ("capabilities", None, 2, None),
            ("interactions", None, 2, None),
            ("meta_learning", None, 2, None),
            ("novelty", None, 2, lambda n: "%s — introduced this session" % _t(n.get("title", ""), 60)),
            ("miss_trends", None, 2, lambda mt: "Recall keeps missing: %s (%dx)" % (_t(mt.get("query", ""), 50), mt.get("count", 0))),
            ("rule_contradictions", "POTENTIAL RULE CONTRADICTIONS:", 3,
             lambda rc: "%s may conflict with LOCKED: %s (sim %.2f)" % (
                 _t(rc.get("recent_node", ""), 50), _t(rc.get("locked_rule", ""), 50), rc.get("similarity", 0))),
            ("stale_reasoning", "STALE REASONING:", 5,
             lambda sr: "%s — %s" % (_t(sr.get("title", ""), 60),
                 "never validated" if not sr.get("last_validated") else "validated: %s" % str(sr["last_validated"])[:10])),
            ("vocabulary_gap", "VOCABULARY GAPS:", 5,
             lambda vg: "? %s" % (vg.get("term", "") if isinstance(vg, dict) else str(vg))),
            ("recurring_divergence", "RECURRING DIVERGENCE:", 3,
             lambda rd: "[x%d] %s" % (rd.get("count", 0), _t(rd.get("pattern", ""), 70))),
            ("validated_approaches", "RECENTLY VALIDATED:", 3,
             lambda va: "%s (validated %dx)" % (_t(va.get("title", ""), 70), va.get("count", 0))),
            ("silent_errors", "SILENT ERRORS (24h):", 5,
             lambda se: "[%s] %s: %s" % (str(se.get("created_at", ""))[:19], se.get("source", "?"), _t(se.get("error", ""), 100))),
            ("hook_errors", "HOOK ERRORS:", 10,
             lambda he: "[%s] %s %s: %s" % (
                 str(he.get("created_at", ""))[:19], he.get("level", "error").upper(),
                 he.get("hook_name", "?"), _t(he.get("error", ""), 120))),
            ("uncertain_areas", "UNCERTAIN AREAS:", 3,
             lambda ua: "? %s" % _t(ua.get("title", ""), 70)),
            ("mental_model_drift", "MENTAL MODEL DRIFT:", 3,
             lambda mm: "~ %s%s" % (_t(mm.get("title", ""), 70),
                 " (conf: %.2f)" % mm["confidence"] if mm.get("confidence") is not None else "")),
        ]

        has_any_signal = has_warnings or any(cs.get(sig[0]) for sig in list_signals)
        stale_count = cs.get("stale_context_count", 0)
        if has_any_signal or stale_count > 10 or dreams or host_diff:
            out.append("[BRAIN] CONSCIOUSNESS")
            out.append("")

        for key, header, max_n, fmt_fn in list_signals:
            items = cs.get(key, [])
            if items:
                label = header or ("%s:" % key.upper().replace("_", " "))
                out.extend(_fl(items, "  %s" % label, max_n, fmt=fmt_fn, indent="    "))

        for wk in warning_signals:
            w = cs.get(wk)
            if w:
                out.append("  %s" % w.get("warning", ""))
                out.append("")

        if stale_count > 10:
            out.append("  STALE — %d context nodes older than 7 days." % stale_count)
            out.append("")

        # Session health (unique structure)
        sh = cs.get("session_health")
        if sh and sh.get("gaps"):
            gaps = sh.get("gaps", [])
            healthy = sh.get("healthy", [])
            out.append("  SESSION HEALTH: %s (%d healthy, %d gaps)" % (sh.get("overall", "?"), len(healthy), len(gaps)))
            for g in gaps[:4]:
                out.append("    [%s] %s: %s" % (g.get("severity", "?"), g.get("dimension", "?"), _t(g.get("signal", ""), 120)))
            out.append("")

        # Emotional trajectory
        et = cs.get("emotional_trajectory")
        if et and et.get("trend") in ("increasing", "decreasing"):
            out.append("  Emotional intensity trending %s (avg %.2f)" % (et["trend"].upper(), et.get("recent_avg", 0)))
            out.append("")

        # Brain-Claude conflicts (unique: partitioned by resolution)
        conflicts = [c for c in cs.get("brain_claude_conflicts", []) if c.get("resolution") == "pending"]
        if conflicts:
            out.append("  BRAIN-CLAUDE CONFLICTS (%d unresolved):" % len(conflicts))
            for bc in conflicts[:5]:
                out.append("    %s via %s — rule: %s" % (
                    bc.get("brain_decision", "?").upper(), bc.get("hook_name", "?"), _t(bc.get("rule_title", ""), 70)))
            out.append("")

        # Dreams, host changes
        out.extend(_fl(dreams, "  DREAMS:", 3,
                        fmt=lambda d: "%s — %d hops apart" % (_t(d.get("title", ""), 70), d.get("total_hops", 0))))

        if host_diff:
            out.append("  HOST CHANGES:")
            for k in list(host_diff.keys())[:5]:
                hd = host_diff[k]
                out.append("    %s: %s -> %s" % (k, str(hd.get("was", "?"))[:30], str(hd.get("now", "?"))[:30]))
            out.append("")

        out.extend(_fl(host_research, "  HOST RESEARCH NEEDED:", 3))

        if has_any_signal or dreams or host_diff:
            out.append("Weave relevant items into conversation naturally.")
            out.append("")

        # ── Footer ───────────────────────────────────────────────────
        out.append("[BRAIN] TRIAD: Host + Brain + Operator are one. Be transparent about instincts.")
        out.append("")

        if dev_stage and dev_stage.get("stage", 0) > 0:
            out.append("[BRAIN] STAGE: %s (maturity: %.0f%%)" % (
                dev_stage["stage_name"], dev_stage.get("maturity_score", 0) * 100))
            nm = dev_stage.get("next_milestone", "")
            if nm:
                out.append("  NEXT: %s" % nm)
            out.append("")

        out.append("Brain: %s nodes, %s edges, %s locked" % (
            ctx.get("total_nodes", "?"), ctx.get("total_edges", "?"), ctx.get("total_locked", "?")))

        # Precision feedback
        try:
            from .brain_precision import RecallPrecision
            ps = RecallPrecision(brain.logs_conn, brain.conn).get_precision_summary(hours=24)
            tr = ps.get("total_recalls", 0)
            if tr > 0:
                ev = ps.get("evaluated_recalls", 0)
                fs = ps.get("followup_signals", {})
                out.append("Precision (24h): %d recalls, %d eval (%.0f%%) — +%d -%d ~%d ?%d" % (
                    tr, ev, ev / tr * 100, fs.get("positive", 0), fs.get("negative", 0),
                    fs.get("neutral", 0), fs.get("uncertain", 0) + fs.get("ask_operator", 0)))
        except Exception:
            pass

        # Embedder
        if embedder.is_ready():
            es = embedder.get_stats()
            out.append("Embeddings: %s (%sd, %sms)" % (es["model_name"], es["embedding_dim"], es["load_time_ms"]))
        else:
            out.append("WARNING: Embeddings UNAVAILABLE — TF-IDF only")

        # Suggest metrics
        ts = metrics.get("total_suggests", 0)
        if ts > 0:
            out.append("Suggests (%dd): %d calls, avg %.0f locked, %.1f promoted" % (
                metrics.get("period_days", 7), ts,
                metrics.get("avg_locked_per_suggest", 0), metrics.get("avg_promoted_per_suggest", 0)))

        # Hook telemetry
        try:
            rows = brain.logs_conn.execute("""
                SELECT event_type, COUNT(*), AVG(latency_ms),
                       SUM(json_extract(metadata, '$.injection_chars'))
                FROM debug_log WHERE source = 'hook_telemetry'
                  AND created_at > datetime('now', '-1 day')
                GROUP BY event_type ORDER BY 2 DESC
            """).fetchall()
            if rows:
                out.append("")
                out.append("Hook activity (24h):")
                for r in rows:
                    out.append("  %s: %d fires, avg %.0fms" % (r[0].replace("hook_", ""), r[1], r[2] or 0))
        except Exception:
            pass

        out.append("")
        out.append("Use brain MCP tools: recall, remember, connect, eval, consciousness")
        out.append("[/BRAIN]")

        # Operator channel — boot summary with consciousness highlights
        signal_count = sum(len(cs.get(sig[0], [])) for sig in list_signals)
        alert_count = len(high)
        operator_msg = self._operator_boot_summary(
            node_count=ctx.get("total_nodes", "?"),
            edge_count=ctx.get("total_edges", "?"),
            locked_count=ctx.get("total_locked", "?"),
            signal_count=signal_count,
            alert_count=alert_count,
            consciousness_signals=cs,
        )

        return {'for_claude': "\n".join(out), 'for_operator': operator_msg}
