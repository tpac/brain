"""Frame Constructor — Anchor's structured awareness object.

Phase 2 of the Frame architecture (see docs/FRAME-DESIGN.md). Produces a
markdown text Frame composed of five sections, deterministically queried
from the brain's existing API. No LLM call. No new SQL. Pure composition
over `brain.filter_nodes`, `brain.session_context_for(session_id)`, and
`brain.get_recent_encoding_journal(session_id)`.

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
"""

from typing import Dict, List


# ── Layer caps ──
# Token budget targets ~1500-2000 total. SQL fetches are wider than render
# to leave headroom for type filtering in Python.
TOP_COMMUNITIES = 8
LOCKED_PULL_LIMIT = 200        # 342 total locked exist — pull wide so type-filter has options
ACTIVE_THREAD_PULL = 10
WARM_PULL = 8

# ── Family routing ──
# Type lists no longer inline — Frame reads them from the s2_node_families
# interaction (parallel to s2_edge_families). The future S2 maintenance unit
# updates the families from observed data; Frame and other consumers just
# read names. See servers/scales/s2/node_families_v1.json for v1 seed.
OPERATOR_FAMILY = 'identity_bearing'
PERMANENT_FAMILY = 'episodic_anchor'
WARM_FAMILIES = ['episodic_anchor', 'lesson_insight']
ACTIVE_FAMILY = 'active_thread'

# Fallback type lists — used if interaction is missing or empty (defensive,
# also lets the module work in tests with an unseeded brain).
_FALLBACK_FAMILIES = {
    'identity_bearing': ['principle', 'identity', 'vision', 'rule', 'operator'],
    'episodic_anchor': ['moment', 'anchor_quote', 'user_quote', 'quote'],
    'active_thread': ['open', 'tension', 'hypothesis', 'aspiration'],
    'lesson_insight': ['lesson', 'insight', 'reflection'],
}


def _family_members(brain, family_name: str) -> List[str]:
    """Resolve a family name to its member type strings.

    Reads from `s2_node_families` interaction; falls back to hardcoded defaults
    if the interaction is absent (fresh brain that hasn't been seeded yet, or
    test environments).
    """
    from servers.scales.s2.edge_families import iter_families
    config = brain.get_interaction_config('s2_node_families') or {}
    for fam, members, _meaning in iter_families(config):
        if fam == family_name:
            return list(members)
    # Fallback
    return list(_FALLBACK_FAMILIES.get(family_name, []))


def _members_in_families(brain, family_names: List[str]) -> List[str]:
    """Union of member types across multiple families."""
    seen = []
    for fname in family_names:
        for m in _family_members(brain, fname):
            if m not in seen:
                seen.append(m)
    return seen


def _short_content(node: dict, limit: int = 180) -> str:
    """Prefer content_summary; fall back to content snippet."""
    cs = node.get('content_summary') or ''
    if cs:
        return cs[:limit].rstrip()
    return (node.get('content') or '')[:limit].rstrip()


def _render_operator(brain, locked_nodes: List[dict]) -> str:
    """Operator section — who Tom is, what locked principles Anchor lives by."""
    operator_types = set(_family_members(brain, OPERATOR_FAMILY))
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
    """Partnership section — three layers: integrated, permanent, warm."""
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

    # Permanent: locked moments / identity
    permanent_types = set(_family_members(brain, PERMANENT_FAMILY))
    permanent = [n for n in locked_nodes
                 if n.get('type') in permanent_types]
    if permanent:
        lines.append("\n**Permanent (locked moments):**")
        for n in permanent[:10]:
            snippet = _short_content(n, 140)
            lines.append("- **%s** — %s" % (n.get('title', ''), snippet))

    # Warm: recently-accessed nodes from episodic + lesson families. Recency-
    # first sort means what's been touched recently rises naturally — no
    # separate cutoff needed because the sort already orders by what matters.
    warm_types = _members_in_families(brain, WARM_FAMILIES)
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


def _render_active_threads(brain) -> str:
    """Open work — types in the active_thread family, sorted by recency."""
    active_types = _family_members(brain, ACTIVE_FAMILY)
    res = brain.filter_nodes(
        field='type', include=active_types,
        sort_by='last_accessed', sort_order='desc',
        limit=ACTIVE_THREAD_PULL, rich=True)
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

    sections = [
        _render_operator(brain, locked_nodes),
        _render_partnership(brain, locked_nodes),
        _render_active_threads(brain),
        _render_current_focus(brain, session_id),
        _render_recent_moves(brain, session_id),
    ]
    return "\n".join(sections).strip() + "\n"
