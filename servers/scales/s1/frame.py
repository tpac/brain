"""Frame Constructor — Anchor's structured awareness object.

Phase 2 of the Frame architecture (see docs/FRAME-DESIGN.md). Produces a
markdown text Frame composed of five sections, deterministically queried
from the brain's existing API. No LLM call. No new SQL. Pure composition
over `brain.filter_nodes`, `brain.session_context_for(session_id)`,
`brain.get_recent_encoding_journal(session_id)`, and `brain.aspects`
(the AspectRegistry — Step 11 of unified-aspects).

Sections:
    Operator      — locked identity-bearing nodes (principle/identity/vision)
    Partnership   — three layers:
                    integrated (top communities by access_count)
                    permanent  (locked moments)
                    warm       (recent moments/insights, last 7 days)
    Active state  — open + tension nodes
    Current focus — encoder's session_context blob
    Recent moves  — encoder's recent journal entries

Same Frame is rendered at boot (via SessionStart hook) and during
conversation (via hook_recall → run_surface). Unified per Tom's call
2026-05-02 — splits into boot/live variants only if measurement says so.

Aspect routing:
    brain.aspects.identity_bearing → Operator section
    brain.aspects.episodic_anchor  → Partnership.permanent
    brain.aspects.episodic_anchor + lesson_insight → Partnership.warm (union)
    brain.aspects.active_thread    → Active threads
"""

from typing import Dict, List


# ── Layer caps ──
# Token budget targets ~1500-2000 total. SQL fetches are wider than render
# to leave headroom for type filtering in Python.
TOP_COMMUNITIES = 8
LOCKED_PULL_LIMIT = 200        # 342 total locked exist — pull wide so type-filter has options
ACTIVE_THREAD_PULL = 10
WARM_PULL = 8


def _short_content(node: dict, limit: int = 180) -> str:
    """Prefer content_summary; fall back to content snippet."""
    cs = node.get('content_summary') or ''
    if cs:
        return cs[:limit].rstrip()
    return (node.get('content') or '')[:limit].rstrip()


def _render_operator(brain, locked_nodes: List[dict]) -> str:
    """Operator section — who the operator is, what locked principles Anchor lives by.

    Reads `brain.aspects.identity_bearing.node_types` to know which types
    qualify (principle, identity, vision, rule, operator, capability).
    """
    operator_types = set(brain.aspects.identity_bearing.node_types)
    operator_nodes = [n for n in locked_nodes
                      if n.get('type') in operator_types]
    if not operator_nodes:
        return "## Operator\n(no locked operator/principle/identity nodes)\n"

    lines = ["## Operator"]
    for n in operator_nodes[:10]:  # cap render even if pull is bigger
        snippet = _short_content(n, 160)
        lines.append("- **%s** — %s" % (n.get('title', ''), snippet))
    return "\n".join(lines) + "\n"


def _render_partnership(brain, locked_nodes: List[dict]) -> str:
    """Partnership section — three layers: integrated, permanent, warm.

    Permanent uses `brain.aspects.episodic_anchor.node_types`. Warm unions
    types from episodic_anchor + lesson_insight via `aspects.types_in(...)`.
    """
    lines = ["## Partnership"]

    # Integrated: top communities by RECENCY (last_accessed). Pure access_count
    # rewards historic obsessions — if we've spent 10 sessions on hooks, those
    # communities dominate forever. Recency is fluid — what's currently in mind.
    comms_result = brain.filter_nodes(
        field='type', include=['community'],
        sort_by='last_accessed', sort_order='desc',
        limit=TOP_COMMUNITIES, rich=True)
    comms = comms_result.get('nodes', []) if isinstance(comms_result, dict) else []
    if comms:
        lines.append("\n**Integrated (top communities):**")
        for c in comms:
            snippet = _short_content(c, 140)
            lines.append("- **%s** — %s" % (c.get('title', ''), snippet))

    # Permanent: locked moments / identity (episodic_anchor types only)
    permanent_types = set(brain.aspects.episodic_anchor.node_types)
    permanent = [n for n in locked_nodes
                 if n.get('type') in permanent_types]
    if permanent:
        lines.append("\n**Permanent (locked moments):**")
        for n in permanent[:10]:
            snippet = _short_content(n, 140)
            lines.append("- **%s** — %s" % (n.get('title', ''), snippet))

    # Warm: recently-accessed nodes from episodic + lesson aspects. Recency-
    # first sort means what's been touched recently rises naturally — no
    # separate cutoff needed because the sort already orders by what matters.
    warm_types = list(brain.aspects.types_in(
        ['episodic_anchor', 'lesson_insight']))
    warm_result = brain.filter_nodes(
        field='type', include=warm_types,
        sort_by='last_accessed', sort_order='desc',
        limit=WARM_PULL, rich=True)
    warm = warm_result.get('nodes', []) if isinstance(warm_result, dict) else []
    if warm:
        lines.append("\n**Warm (recently active):**")
        for n in warm:
            snippet = _short_content(n, 120)
            lines.append("- **%s** — %s" % (n.get('title', ''), snippet))

    if len(lines) == 1:  # only the heading
        lines.append("(no partnership context yet)")
    return "\n".join(lines) + "\n"


def _render_active_threads(brain, arc_text: str = '') -> str:
    """Open work — types in the active_thread aspect.

    When arc_text is non-empty (the session's current focus blob), the result
    is relevance-ranked against it — top half by relevance to today's arc,
    remainder by raw recency. Lifts threads semantically connected to current
    work above unrelated brain-wide noise. See FRAME-DESIGN.md Phase 2.5.
    """
    active_types = list(brain.aspects.active_thread.node_types)
    res = brain.filter_nodes(
        field='type', include=active_types,
        sort_by='last_accessed', sort_order='desc',
        limit=ACTIVE_THREAD_PULL, rich=True,
        relevance_query=arc_text or None)
    nodes = res.get('nodes', []) if isinstance(res, dict) else []
    # Only keep unresolved — skip nodes with resolved_at set
    open_nodes = [n for n in nodes if not n.get('resolved_at')]
    if not open_nodes:
        return "## Active threads\n(none open)\n"
    lines = ["## Active threads"]
    for n in open_nodes[:8]:
        snippet = _short_content(n, 140)
        lines.append("- **%s** [%s] — %s" % (
            n.get('title', ''), n.get('type', ''), snippet))
    return "\n".join(lines) + "\n"


def _render_current_focus(brain, session_id: str) -> str:
    """Encoder's rolling session arc — per-session key (no parallel-session leak).

    2026-05-02 (Frame Phase 2.5): switched from `brain.session_context`
    (global, leaked across parallel sessions) to `brain.session_context_for
    (session_id)`. Cross-session continuity is now a deliberate query at
    boot, not an accidental side-effect of a leaky global.
    """
    ctx = (brain.session_context_for(session_id) or '').strip()
    if not ctx:
        return "## Current focus\n(fresh session)\n"
    return "## Current focus\n%s\n" % ctx


def _render_recent_moves(brain, session_id: str) -> str:
    """Encoder's recent journal — what Anchor has done this session."""
    journal = (brain.get_recent_encoding_journal(session_id) or '').strip()
    if not journal:
        return "## Recent moves\n(fresh session)\n"
    return "## Recent moves\n%s\n" % journal


def build_frame(brain, session_id: str) -> str:
    """Construct the Frame as markdown text.

    Five sections, deterministic SQL via brain.filter_nodes, no LLM call.
    Type-routing for sections reads `brain.aspects` (the AspectRegistry —
    single source of truth for which node types qualify as identity-bearing,
    episodic, lesson-insight, active-thread).

    Reuses one locked-nodes pull for both Operator and Partnership-permanent
    sections.

    Args:
        brain: Brain instance
        session_id: current session ID (for recent_moves)
    Returns:
        Markdown string. ~1500-2000 tokens typical.
    """
    # One pull, two consumers — reuse locked-nodes between operator + permanent.
    # Sort by last_accessed: locked nodes are curated, but recency reveals which
    # are still ALIVE in the partnership vs which have gone dormant.
    locked_result = brain.filter_nodes(
        field='locked', include=[1],
        sort_by='last_accessed', sort_order='desc',
        limit=LOCKED_PULL_LIMIT, rich=True)
    locked_nodes = (locked_result.get('nodes', [])
                    if isinstance(locked_result, dict) else [])

    # Arc text: this session's compressed-down current focus. Used as the
    # relevance pivot for Active threads (and later, Warm/Integrated). When
    # empty (fresh session), relevance ranking falls back to pure recency.
    arc_text = brain.session_context_for(session_id)

    sections = [
        _render_operator(brain, locked_nodes),
        _render_partnership(brain, locked_nodes),
        _render_active_threads(brain, arc_text=arc_text),
        _render_current_focus(brain, session_id),
        _render_recent_moves(brain, session_id),
    ]
    return "\n".join(sections).strip() + "\n"
