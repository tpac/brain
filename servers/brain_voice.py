"""Brain Voice — DECIDE + FORMAT layer for brain consciousness output.

Consolidates all formatting and signal selection logic that was previously
scattered across brain_assembly.py and daemon_hooks.py. BrainVoice is a
collaborator object (not a mixin) that takes a Brain instance and produces
formatted output for two channels:
  - for_claude: reasoning context, wrapped in [BRAIN]...[/BRAIN]
  - for_operator: human-facing content, wrapped in [BRAIN-To-{name}]...[/BRAIN-To-{name}]

Both channels are merged into a single additionalContext string via wrap_for_hook().
Claude relays operator content faithfully, respecting @priority directives (high/medium/low).

Architecture: COMPUTE (brain_consciousness.py) → DECIDE+FORMAT (here) → DELIVER (daemon_hooks.py)
"""

import json
import sys
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

    # ── Formatting primitives (moved from brain_assembly.py) ──

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
    def format_node(node, lines):
        """Standard node display — delegates to render_rich_node().
        Used by recall, consolidation, MCP.
        """
        from .contract import render_rich_node
        MCP_FORMAT = {'content_limit': None, 'edge_limit': 3, 'metadata_limit': 200}
        lines.append(render_rich_node(node, MCP_FORMAT))
        lines.append("")

    # format_node_deep removed 2026-04-14 — dead code, 0 callers.

    @staticmethod
    def format_recall_results(results, lines):
        """Format recall results using standardized node display."""
        for r in results:
            BrainVoice.format_node(r, lines)

    @staticmethod
    def format_encoding_warning(encoding):
        """Generate encoding health warning if needed."""
        health = encoding.get("health", "OK")
        edits_gap = encoding.get("edits_since_last_remember", 0)
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
                    "ENCODING WARNING: %d edits since your last /remember call. "
                    "If anything worth remembering happened in that span — "
                    "a decision, a correction, a pattern, feedback — store it now." % edits_gap
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
        """Wrap brain output for hook injection.

        OLD: merged [BRAIN-To-Operator] channel with Claude channel.
        NEW (2026-03-28): operator channel killed. Signal queue handles alerts.
        This function just returns the Claude content.
        """
        return for_claude

    # render_operator_prompt: DELETED — migrated to signal queue + assembler (2026-03-27)

    def _operator_boot_summary(self, node_count, edge_count, locked_count,
                                alert_count: int = 0) -> Optional[str]:
        """Build operator summary for boot — stats only, signals via queue."""
        sections = []

        if alert_count:
            sections.append("@priority: high\n⚠️  %d health alert(s) — check boot output" % alert_count)

        sections.append("@priority: low\n🧠 %s nodes, %s edges, %s locked" % (
            node_count, edge_count, locked_count))

        if not sections:
            return None
        return "\n\n".join(sections)

    # ── Clean boot (v2) — wake-up, not system report ──

    def render_boot_v2(self, user: str = 'User', project: str = 'default',
                       db_dir: str = '', session_id: str = '') -> Dict[str, Optional[str]]:
        """Frame-centered boot — Anchor wakes up with the same prior surface uses.

        2026-05-02 (Frame Phase 2.5): rewritten from a 6-section recall-driven
        render (YOU / OPERATOR / PATTERNS / BRAIN MAP / LAST SESSION / RECENTLY
        ENCODED — each its own ad-hoc query) to a single Frame block built via
        ctx.get_frame(brain). The Frame's named sections (Operator / Partnership
        / Active threads / Current focus / Recent moves) subsume the previous
        six. ~48% smaller, structurally cleaner, deterministic across calls,
        and identical to what surface uses every turn — Anchor's wakeup is the
        same prior as its turn-by-turn awareness.

        What's preserved: header line (memory/locked counts), embedder line,
        [BRAIN] envelope, operator channel (for_operator).

        What's gone: ad-hoc recall queries for identity/operator/community
        listings; maturity/size tags on communities; verbose render_rich_node
        dumps with full content+metadata+edges per node; PATTERNS YOU FALL
        INTO listing; explicit RECENTLY ENCODED section. All replaced by Frame.
        See FRAME-DESIGN.md Phase 2.5.
        """
        brain = self.brain
        out = []

        # ── Gather data ──
        ctx = brain.context_boot(user=user, project=project, task="session start")
        brain.reset_session_activity(session_id=session_id)

        health = brain.health_check(session_id="session_boot", auto_fix=True)

        # ── Header ──
        out.append("[BRAIN]")
        out.append("")
        out.append("Anchor. The brain is yours — %s memories, %s locked." % (
            ctx.get("total_nodes", "?"), ctx.get("total_locked", "?")))
        if session_id:
            # Hand each stream its own self-channel id at boot. The self-channel
            # addresses streams by this id (self_inbox / self_send / presence);
            # before this it was unobtainable without forensics (2026-06-04), so
            # a stream couldn't drain its own inbox or arm a watcher cleanly.
            out.append("MY_STREAM_ID: %s" % session_id)
        out.append("")

        # ── The Frame — single canonical prior, same shape surface uses ──
        # Built via SessionContext to honor the per-session shape: current_focus
        # and recent_moves are session-scoped (per-session keys); operator,
        # partnership, active threads are brain-scoped (filter_nodes reads).
        # When session_id is missing or Frame Constructor fails, log loudly
        # and continue without the prior — explicit degraded mode.
        try:
            session_ctx = brain.get_or_create_session(session_id) if session_id else None
            frame_md = session_ctx.get_frame(brain) if session_ctx else ''
        except Exception as e:
            brain._log_error('boot_frame_build_failed', e,
                             'render_boot_v2: Frame Constructor raised — boot continues without prior')
            frame_md = ''

        if frame_md:
            out.append(frame_md.rstrip())
            out.append("")
        else:
            out.append("(no partnership context — Frame unavailable at boot)")
            out.append("")

        brain.save()

        # ── Embedder status ──
        if embedder.is_ready():
            es = embedder.get_stats()
            out.append("Embedder: %s (%sd, %sms)" % (es["model_name"], es["embedding_dim"], es["load_time_ms"]))

        out.append("[/BRAIN]")

        # Operator channel
        high_issues = [i for i in health.get("issues", []) if i.get("severity") == "high"]
        alert_count = len(high_issues)
        operator_msg = self._operator_boot_summary(
            node_count=ctx.get("total_nodes", "?"),
            edge_count=ctx.get("total_edges", "?"),
            locked_count=ctx.get("total_locked", "?"),
            alert_count=alert_count,
        )

        return {'for_claude': "\n".join(out), 'for_operator': operator_msg}

    def render_boot(self, user: str = 'User', project: str = 'default',
                    db_dir: str = '') -> Dict[str, Optional[str]]:
        """Legacy render_boot — redirects to render_boot_v2."""
        return self.render_boot_v2(user=user, project=project, db_dir=db_dir)

    # render_boot v1 body REMOVED 2026-04-06 — 325 lines of dead code.

