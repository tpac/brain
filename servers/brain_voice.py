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

    @staticmethod
    def format_node_deep(node, lines, conn=None, max_d1=3, max_d2=3, max_d3=3):
        """3-degree node display for encoding agent context.
        Reads field selection and truncation from pipeline_contract.

        Degree 0: Full node header + content
        Degree 1: Rich — header, content_summary, relation, confidence
        Degree 2: Title, type, relation
        Degree 3: Title, id only
        """
        from .dal import GraphDAL
        from .pipeline_contract import (
            format_node_header, format_neighbor_d1, format_neighbor_d2,
            NEIGHBOR_TRUNCATION,
        )

        node_id = node.get("id", "")

        # Degree 0: full node
        lines.append(format_node_header(node))
        lines.append(node.get("content", ""))

        # Degree 1: rich neighbor display
        # Use pre-attached graph data, or fetch via rich query
        graph = node.get("_graph", {})
        d1_neighbors = graph.get("degree_1", node.get("_neighbors") or [])[:max_d1]
        seen_ids = {node_id}
        if not d1_neighbors and conn and node_id:
            try:
                graph_dal = GraphDAL(conn)
                d1_neighbors = graph_dal.get_neighbors_rich(
                    node_id, limit=max_d1, exclude_node_ids=seen_ids)
            except Exception as e:
                print('[brain_voice] ERROR format_node_deep d1_neighbors: %s' % e, file=sys.stderr)

        for nb in d1_neighbors:
            nb_id = nb.get("id", "")
            seen_ids.add(nb_id)
            lines.append(format_neighbor_d1(nb))

            # Degree 2
            d2_neighbors = []
            d2_from_graph = graph.get("degree_2", [])
            if d2_from_graph:
                d2_neighbors = [n for n in d2_from_graph if n.get("id") not in seen_ids][:max_d2]
            elif conn and nb_id:
                try:
                    graph_dal = GraphDAL(conn)
                    d2_neighbors = graph_dal.get_neighbors_rich(
                        nb_id, limit=max_d2, exclude_node_ids=seen_ids)
                except Exception as e:
                    print('[brain_voice] ERROR format_node_deep d2_neighbors: %s' % e, file=sys.stderr)

            for nb2 in d2_neighbors:
                nb2_id = nb2.get("id", "")
                seen_ids.add(nb2_id)
                lines.append("     ↳ %s" % format_neighbor_d2(nb2))

                # Degree 3
                d3_neighbors = []
                d3_from_graph = graph.get("degree_3", [])
                if d3_from_graph:
                    d3_neighbors = [n for n in d3_from_graph if n.get("id") not in seen_ids][:max_d3]
                elif conn and nb2_id:
                    try:
                        d3_neighbors = GraphDAL(conn).get_neighbors_rich(
                            nb2_id, limit=max_d3, exclude_node_ids=seen_ids)
                    except Exception as e:
                        print('[brain_voice] ERROR format_node_deep d3_neighbors: %s' % e, file=sys.stderr)

                t = NEIGHBOR_TRUNCATION
                for nb3 in d3_neighbors:
                    nb3_id = nb3.get("id", "")
                    seen_ids.add(nb3_id)
                    lines.append("        ↳ \"%s\" (id:%s)" % (
                        str(nb3.get("title", ""))[:t['d3_title']], nb3_id[:t['d3_id']]))

        lines.append("")

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
        """Wrap brain output for hook injection.

        OLD: merged [BRAIN-To-Operator] channel with Claude channel.
        NEW (2026-03-28): operator channel killed. Signal queue handles alerts.
        This function just returns the Claude content.
        """
        return for_claude

    # render_operator_prompt: DELETED — migrated to signal queue + assembler (2026-03-27)

    def _operator_boot_summary(self, node_count, edge_count, locked_count,
                                signal_count: int = 0, alert_count: int = 0,
                                consciousness_signals: Dict = None) -> Optional[str]:
        """Build operator summary for boot — stats only, signals via queue."""
        sections = []

        if alert_count:
            sections.append("@priority: high\n⚠️  %d health alert(s) — check boot output" % alert_count)

        sections.append("@priority: low\n🧠 %s nodes, %s edges, %s locked" % (
            node_count, edge_count, locked_count))

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

    # render_prompt: DELETED — replaced by SurfaceAssembler (2026-03-27)

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
                output.append("  Use brain.recall() for current context instead.")
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
        output.append("The brain persists. Use brain.recall() to recover context.")
        output.append("[/BRAIN]")

        operator_msg = self._operator_reboot_summary(
            locked_count=len(locked_rules) if locked_rules else 0,
            recall_count=len(recall_results) if recall_results else 0,
        )

        return {'for_claude': "\n".join(output), 'for_operator': operator_msg}

    # ── Clean boot (v2) — wake-up, not system report ──

    def render_boot_v2(self, user: str = 'User', project: str = 'default',
                       db_dir: str = '') -> Dict[str, Optional[str]]:
        """Dynamic boot — pulls personality from the brain, not static text.

        Every section is a live query. Different brain, different boot.
        """
        brain = self.brain
        _t = self.trunc
        out = []

        # ── Gather data ──
        ctx = brain.context_boot(user=user, project=project, task="session start")
        brain.reset_session_activity()

        def _recall(query, filt=None, limit=3):
            r = brain.recall(query=query, filter=filt, limit=limit, source='internal')
            return (r.get('results', r) if isinstance(r, dict) else r)[:limit]

        self_knowledge = brain.fetch_self_knowledge(limit=5)
        session_context = brain.get_config('session_context', '')
        recent_nodes = ctx.get("recent", [])[:5]
        cs = {"reminders": brain.get_due_reminders()}
        health = brain.health_check(session_id="session_boot", auto_fix=True)
        brain.save()

        # ── Identity (one line) ──
        out.append("[BRAIN]")
        out.append("")
        out.append("Anchor. The brain is yours — %s memories, %s locked." % (
            ctx.get("total_nodes", "?"), ctx.get("total_locked", "?")))
        out.append("")

        # ── Your words (get_rich_node + render) ──
        from .pipeline_contract import get_rich_node
        from .contract import render_rich_node

        BOOT_FORMAT = {'content_limit': 400, 'edge_limit': 3, 'metadata_limit': 150, 'time_format': 'relative'}

        anchor_nodes = _recall("who I am, what I've learned, my identity", limit=5)
        if anchor_nodes:
            out.append("YOU:")
            for n in anchor_nodes:
                rich = get_rich_node(brain, n['id'])
                if rich:
                    out.append(render_rich_node(rich, BOOT_FORMAT))
                    out.append("")
            out.append("")

        # ── Operator's words ──
        operator_nodes = _recall("operator partnership vision corrections teaching", limit=3)
        if operator_nodes:
            out.append("%s:" % user.upper())
            for n in operator_nodes:
                rich = get_rich_node(brain, n['id'])
                if rich:
                    out.append(render_rich_node(rich, BOOT_FORMAT))
                    out.append("")
            out.append("")

        # ── Patterns you fall into (self-knowledge) ──
        if self_knowledge:
            out.append("PATTERNS YOU FALL INTO:")
            for sk in self_knowledge[:3]:
                out.append("  %s" % _t(sk.get('title', ''), 100))
            out.append("")

        # ── What the brain knows (community map) ──
        # Communities compress 10-30 nodes into one narrative.
        # Loading them here gives Anchor the SHAPE of everything it knows.
        try:
            communities = brain.conn.execute('''
                SELECT n.id, n.title, n.content
                FROM nodes n
                WHERE n.type = 'community' AND n.archived = 0
                ORDER BY n.confidence DESC, n.updated_at DESC
            ''').fetchall()

            if communities:
                # Load maturity + member count per community
                comm_items = []
                for cid, ctitle, ccontent in communities:
                    meta = dict(brain.conn.execute(
                        "SELECT key, value FROM node_metadata_kv "
                        "WHERE node_id = ? AND key IN "
                        "('community_maturity', 'community_size', 'community_narrative')",
                        (cid,)).fetchall())
                    maturity = meta.get('community_maturity', '?')
                    size = meta.get('community_size', '?')
                    narrative = meta.get('community_narrative', '')
                    # Use narrative if available, fall back to content
                    summary = narrative or (ccontent or '')
                    comm_items.append((maturity, size, ctitle, summary))

                if comm_items:
                    # Show top communities with narrative, rest as titles only
                    # Settled/active first (they're the most stable knowledge)
                    maturity_order = {'settled': 0, 'active': 1, 'forming': 2, 'corridor': 3}
                    comm_items.sort(key=lambda x: (
                        maturity_order.get(x[0], 4),
                        -(int(x[1]) if x[1] and x[1] != '?' else 0)))

                    TOP_WITH_NARRATIVE = 20
                    out.append("BRAIN MAP (%d communities):" % len(comm_items))
                    for i, (maturity, size, title, summary) in enumerate(comm_items):
                        mat_tag = maturity[:1].upper() if maturity and maturity != '?' else '?'
                        if i < TOP_WITH_NARRATIVE:
                            out.append("  [%s|%s] %s" % (mat_tag, size, _t(title, 70)))
                            if summary:
                                out.append("    %s" % _t(summary, 120))
                        else:
                            out.append("  [%s|%s] %s" % (mat_tag, size, _t(title, 70)))
                    out.append("")
        except Exception as e:
            brain._log_error('boot_community_map', e, 'loading community map')

        # ── Where we left off ──
        if session_context:
            out.append("LAST SESSION:")
            out.append("  %s" % _t(session_context, 400))
            out.append("")

        # ── Recently encoded ──
        if recent_nodes:
            out.append("RECENTLY ENCODED:")
            for rn in recent_nodes:
                out.append("  [%s] %s" % (rn.get('type', '?'), _t(rn.get('title', ''), 100)))
            out.append("")

        if embedder.is_ready():
            es = embedder.get_stats()
            out.append("Embedder: %s (%sd, %sms)" % (es["model_name"], es["embedding_dim"], es["load_time_ms"]))

        out.append("[/BRAIN]")

        # Operator channel
        signal_count = sum(len(cs.get(k, [])) for k in ["evolutions", "silent_errors", "uncertain_areas"])
        high_issues = [i for i in health.get("issues", []) if i.get("severity") == "high"]
        alert_count = len(high_issues)
        operator_msg = self._operator_boot_summary(
            node_count=ctx.get("total_nodes", "?"),
            edge_count=ctx.get("total_edges", "?"),
            locked_count=ctx.get("total_locked", "?"),
            signal_count=signal_count,
            alert_count=alert_count,
            consciousness_signals=cs,
        )

        return {'for_claude': "\n".join(out), 'for_operator': operator_msg}

    def render_boot(self, user: str = 'User', project: str = 'default',
                    db_dir: str = '') -> Dict[str, Optional[str]]:
        """Legacy render_boot — redirects to render_boot_v2."""
        return self.render_boot_v2(user=user, project=project, db_dir=db_dir)

    # render_boot v1 body REMOVED 2026-04-06 — 325 lines of dead code.

