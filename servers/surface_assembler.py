"""Surface Assembler — replaces render_prompt() with budget-aware assembly.

## Architecture
Two-phase assembly:

Phase A (Reactive): Query-dependent content computed fresh per turn
  1. Preempt check — if critical signals exist, return ONLY those
  2. Recall results — formatted with truncated content
  3. Reactive signals — gap detection, segment note, priming note

Phase B (Proactive): From signal_queue, pulled by priority within budget
  4. Calculate remaining budget after Phase A
  5. Pull from queue by priority
  6. Append to output

## Budget Model
Total budget defaults to 6000 chars. Recall gets first claim.
Queue fills the rest. If recall returns 8 verbose nodes (4000 chars),
queue gets 2000. If recall returns nothing, queue gets 6000.

## Design Principles
- No hardcoded boilerplate text (no review menus, no footers)
- Node content truncated to ~150 chars with ID for drill-down
- Proactive signals labeled by producer for monitoring
- Assembler output logged to dashboard for comparison with old path
"""
import logging
from typing import Dict, List, Optional

from .dal_signal_queue import SignalQueueDAL

log = logging.getLogger(__name__)

# Truncation limits
_NODE_CONTENT_MAX = 200
_NODE_NEIGHBOR_MAX = 2


class SurfaceAssembler:
    """Budget-aware assembly of brain output for Claude's context.

    Args:
        signal_dal: SignalQueueDAL for reading proactive signals
        budget_chars: max total chars for the assembled output
    """

    def __init__(self, signal_dal: SignalQueueDAL, budget_chars: int = 6000):
        self.signal_dal = signal_dal
        self.budget = budget_chars

    def assemble(self, recall_results: List[Dict],
                 segment_note: str = None,
                 priming_note: str = None,
                 gap: Dict = None) -> Dict[str, Optional[str]]:
        """Assemble the final brain output.

        Returns:
            {'for_claude': str, 'for_operator': str|None, 'stats': dict}
        """
        # Phase 0: Preempt check
        preempts = self.signal_dal.pull_preempt()
        if preempts:
            return self._format_preempt(preempts)

        lines = []
        lines.append("[BRAIN]")

        # Phase A: Reactive content
        reactive_lines = []

        if segment_note:
            reactive_lines.append(segment_note)
            reactive_lines.append("")

        if priming_note:
            reactive_lines.append(priming_note)
            reactive_lines.append("")

        # Recalled nodes — truncated
        if recall_results:
            reactive_lines.append("RECALLED:")
            for node in recall_results:
                self._format_node_compact(node, reactive_lines)
            reactive_lines.append("")

        # Gap detection
        if gap:
            reactive_lines.append('UNKNOWN TOPIC: "%s"' % gap.get('query', ''))
            reactive_lines.append("No brain knowledge on this. Encode if discussed.")
            reactive_lines.append("")

        lines.extend(reactive_lines)
        reactive_chars = sum(len(line) + 1 for line in reactive_lines)

        # Phase B: Proactive signals from queue
        remaining_budget = max(0, self.budget - reactive_chars - 50)  # 50 chars for wrapper
        proactive_items = self.signal_dal.pull(budget_chars=remaining_budget, limit=5)

        if proactive_items:
            lines.append("SIGNALS:")
            for item in proactive_items:
                lines.append("  [%s] %s (dismiss: %s)" % (item['producer'], item['content'], item['id']))
            lines.append("")

        lines.append("[/BRAIN]")

        assembled = "\n".join(lines)

        stats = {
            'reactive_chars': reactive_chars,
            'proactive_items': len(proactive_items),
            'proactive_chars': sum(item['content_chars'] for item in proactive_items),
            'total_chars': len(assembled),
            'recall_count': len(recall_results),
            'budget': self.budget,
            'budget_used_pct': round(len(assembled) / self.budget * 100, 1) if self.budget else 0,
        }

        log.info("assembler: %d reactive + %d proactive signals = %d chars (%.0f%% of %d budget)",
                 len(recall_results), len(proactive_items), len(assembled),
                 stats['budget_used_pct'], self.budget)

        return {
            'for_claude': assembled,
            'for_operator': None,  # Phase 2: operator channel from queue
            'stats': stats,
        }

    def _format_preempt(self, preempts: List[Dict]) -> Dict[str, Optional[str]]:
        """Format preempt-level signals. These replace ALL other output."""
        lines = ["[CRITICAL — RELAY TO USER IMMEDIATELY]"]
        for p in preempts:
            lines.append("  [%s] %s (dismiss: %s)" % (p['producer'], p['content'], p['id']))
        lines.append("DO NOT continue without informing the user.")
        lines.append("Dismiss with: dismiss_signal(signal_id=\"...\")")
        lines.append("[/CRITICAL]")
        assembled = "\n".join(lines)

        # Operator channel — make sure Tom sees it directly
        operator_lines = ["@priority: high"]
        for p in preempts:
            lines_op = "⚠️ %s" % p['content']
            operator_lines.append(lines_op)
        operator_msg = "\n".join(operator_lines)

        log.warning("assembler: PREEMPT — %d critical signals, skipping recall", len(preempts))

        return {
            'for_claude': assembled,
            'for_operator': operator_msg,
            'stats': {
                'reactive_chars': 0, 'proactive_items': len(preempts),
                'proactive_chars': len(assembled), 'total_chars': len(assembled),
                'recall_count': 0, 'budget': self.budget,
                'budget_used_pct': round(len(assembled) / self.budget * 100, 1),
                'preempt': True,
            },
        }

    @staticmethod
    def _format_node_compact(node: Dict, lines: List[str]):
        """Format a recalled node — unified format matching format_recall_results().

        Format:
          [type] LOCKED Title
          id:abc123 | revised:never | conf:0.95 | created:2026-03-21
          First 200 chars of content...
            ↳ relation: "Neighbor title" (type, id:...)
        """
        typ = node.get("type", "?")
        title = node.get("title", "")
        locked = "LOCKED " if node.get("locked") else ""

        lines.append("[%s] %s%s" % (typ, locked, title))

        # Metadata line — matches proven format used by challenge system
        node_id = node.get("id", "")
        conf = node.get("confidence") or node.get("effective_activation", 0) or 0
        revised = node.get("revised_at") or "never"
        created = (node.get("created_at") or "")[:10]
        lines.append("id:%s | revised:%s | conf:%.2f | created:%s" % (
            node_id, revised, conf, created))

        content = node.get("content", "")
        if content:
            truncated = content[:_NODE_CONTENT_MAX]
            if len(content) > _NODE_CONTENT_MAX:
                truncated += "..."
            lines.append(truncated)

        # Limited neighbors
        for nb in node.get("_neighbors", [])[:_NODE_NEIGHBOR_MAX]:
            nb_type = nb.get("type", "")
            nb_id = nb.get("id", "")[:12]
            lines.append("  ↳ %s: \"%s\" (%s, id:%s)" % (
                nb.get("relation", "related"),
                nb.get("title", ""),
                nb_type, nb_id))
        lines.append("")
