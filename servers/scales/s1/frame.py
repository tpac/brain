"""Frame Constructor — Anchor's structured awareness object.

Produces a markdown Frame, deterministically composed from the brain's
existing API (no LLM call, no new SQL). Three sections:

    What I've learned — the `wisdom` aspect (insight / lesson / principle /
                        vision / reflection / meta_learning / philosophy): the
                        generative understanding that shapes how Anchor thinks.
                        Focus-adaptive — relevance-ranked against the session
                        arc when present, influence-sampled at boot.
    Current focus     — the encoder's rolling per-session arc.
    Recent moves      — the encoder's recent per-session journal.

Same Frame renders at boot (SessionStart hook) and during conversation
(hook_recall → run_surface).

Operator and Partnership are deliberately absent: they were recency-by-type
pulls that, for a mature brain whose work is software, filled with engineering
nodes (a dev changelog wearing partnership labels) and polluted Haiku's
per-turn prior. Seed-brain operator/identity scaffolding lives in the
conditional Zero-Memory boot block (docs/DISTRIBUTION-READINESS.md §7); the
rest is recoverable via recall() on demand.

Aspect routing:
    brain.aspects.wisdom → What I've learned
"""

import random
from typing import List


# ── Caps ──
WISDOM_RENDER = 5      # wisdom nodes surfaced
WISDOM_POOL = 200      # wide skinny candidate pull, degree-ranked in Python at boot
WISDOM_HUB_CAP = 30    # structural degree past which influence is dampened, so
                       # over-connected hubs don't crowd out genuinely deep nodes


def _influence_sample(brain, pool: List[dict], k: int, seed: str = '') -> List[dict]:
    """Rank wisdom candidates by STRUCTURAL graph degree, dampen runaway hubs,
    then sample k from the top tier — seeded per session (varied across
    sessions, stable within one, so the list doesn't reshuffle each arc-less
    turn).

    Boot has no session arc to relevance-rank against, so we surface a *varied*
    set of high-influence wisdom rather than a fixed top-N. Degree comes from
    the brain's structural-degree cache (excludes Hebbian co_accessed /
    emergent_bridge edges — topology, not churn — and is already built at
    warm_up). Hub-dampening keeps over-connected nodes from crowding out
    genuinely deep ones. A failed cache build logs loud (in _ensure_*) and
    degrades to an unranked sample — never silent.
    """
    if len(pool) <= k:
        return pool

    brain._ensure_structural_degree_cache()
    degree = getattr(brain, '_structural_degree_cache', {}) or {}

    def _influence(n: dict) -> float:
        d = degree.get(n.get('id'), 0)
        if d <= WISDOM_HUB_CAP:
            return float(d)
        return WISDOM_HUB_CAP * WISDOM_HUB_CAP / d  # dampen past the cap

    top = sorted(pool, key=_influence, reverse=True)[:max(k * 2, k)]
    if len(top) <= k:
        return top
    rng = random.Random(seed) if seed else random
    return rng.sample(top, k)


def _snippet(node: dict, limit: int = 200) -> str:
    """Short content for the wisdom render — prefer content_summary, whitespace-
    collapsed to one line and trimmed at a word boundary (no mid-word cuts)."""
    text = ' '.join((node.get('content_summary') or node.get('content') or '').split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(' ', 1)[0].rstrip() + '…'


def _render_wisdom(brain, arc_text: str = '', session_id: str = '') -> str:
    """What I've learned — the inspiring, generative wisdom layer.

    Pulls the `wisdom` aspect (insight / lesson / principle / vision /
    reflection / meta_learning / philosophy) — explicitly NOT operational rules
    or tactical record-keeping.

    Focus present (mid-session): relevance-rank the wisdom nodes against the
    session's current focus, so the surfaced wisdom tracks the topic and
    refreshes every encode (the encoder updates current_focus; the Frame reads
    it at the next turn).

    No focus (boot/fresh): influence-sample (see _influence_sample) so waking
    surfaces a varied set of high-influence wisdom rather than a fixed list.
    """
    # by_name (not attribute access) so a missing aspect degrades gracefully —
    # surface builds the Frame every turn and isn't wrapped, so a raise here
    # would crash recall, not just boot.
    wis = brain.aspects.by_name('wisdom')
    if wis is None:
        # Loud: a REQUIRED aspect is missing — surface as an error, not a quiet
        # degrade. _log_error self-protects (never re-raises), so call unguarded.
        brain._log_error(
            'frame_wisdom',
            Exception('wisdom aspect missing from registry — Frame rendered without it'),
            'required aspect absent')
        return ""
    wisdom_types = list(wis.node_types)
    if not wisdom_types:
        return ""

    # Both branches rank cheaply (skinny / embedding), then enrich ONLY the
    # <=WISDOM_RENDER winners for their content — the render shows a snippet,
    # but we never correction-enrich the whole candidate pool.
    if arc_text:
        # Focus present → relevance-rank; rank-then-enrich (brain_recall) means
        # rich=True here enriches only the <=5 winners, not the pool.
        res = brain.filter_nodes(
            field='type', include=wisdom_types,
            sort_by='last_accessed', sort_order='desc',
            limit=WISDOM_RENDER, rich=True, relevance_query=arc_text)
        nodes = res.get('nodes', []) if isinstance(res, dict) else []
    else:
        # No focus (boot) → degree-rank a wide skinny pool, seeded sample, then
        # enrich only the sampled winners for their content.
        res = brain.filter_nodes(
            field='type', include=wisdom_types,
            sort_by='created_at', sort_order='desc',
            limit=WISDOM_POOL, rich=False)
        pool = res.get('nodes', []) if isinstance(res, dict) else []
        sampled = _influence_sample(brain, pool, WISDOM_RENDER, seed=session_id)
        rich_map = brain.get_node([n['id'] for n in sampled]) if sampled else {}
        nodes = [rich_map[n['id']] for n in sampled if n.get('id') in rich_map]

    if not nodes:
        return "## What I've learned\n(nothing yet)\n"
    lines = ["## What I've learned"]
    for n in nodes:
        snip = _snippet(n)
        if snip:
            lines.append("- **%s** — %s" % (n.get('title', ''), snip))
        else:
            lines.append("- **%s**" % n.get('title', ''))
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
    """Construct the Frame as markdown text — three sections, no LLM call.

    Type-routing for the wisdom section reads `brain.aspects.wisdom` (the
    AspectRegistry — single source of truth for which node types are the
    generative-wisdom subset).

    Args:
        brain: Brain instance
        session_id: current session ID (for focus + recent_moves)
    Returns:
        Markdown string.
    """
    # This session's compressed arc — the relevance pivot for the wisdom
    # section. Empty (fresh session) → wisdom falls back to influence-sampling.
    arc_text = brain.session_context_for(session_id)

    sections = [
        _render_wisdom(brain, arc_text=arc_text, session_id=session_id),
        _render_current_focus(brain, session_id),
        _render_recent_moves(brain, session_id),
    ]
    return "\n".join(s for s in sections if s).strip() + "\n"


# ── Standing items — boot-only, operator-extensible type injection ──
# NOT a Frame section: the Frame renders on every recall turn; standing items
# are a wake-up ritual (see them once, act, archive). Injected by
# render_boot_v2 after the Frame.
#
# BRAIN_BOOT_INJECT_TYPES (~/.config/brain/env, comma-separated) picks which
# node types surface at boot. Ships with `journals-escalation` — the landing
# type for journal open-items the encoders promoted (trace_contract
# JOURNAL_ESCALATION_TYPE) — so escalations reach a human by default. Users
# extend their boot with their own types the same way.

BOOT_INJECT_TYPES_DEFAULT = 'journals-escalation'
BOOT_INJECT_CAP = 10


def render_standing_items(brain) -> str:
    """Every live node of the configured types, newest first, capped.

    Empty string when nothing qualifies — a clean boot adds nothing. The
    lifecycle exit is on the node: handle it, then archive (or revise) it and
    it leaves the next boot.
    """
    import os
    raw = os.environ.get('BRAIN_BOOT_INJECT_TYPES', BOOT_INJECT_TYPES_DEFAULT)
    types = [t.strip() for t in raw.split(',') if t.strip()]
    if not types:
        return ''
    result = brain.filter_nodes(field='type', include=types,
                                limit=BOOT_INJECT_CAP + 1, rich=False)
    if (result or {}).get('error'):
        # Loud by default — an error dict is not "nothing standing".
        brain._log_error('boot_standing_items_filter',
                         ValueError(str(result.get('error'))),
                         'render_standing_items: filter_nodes errored for types=%r' % types)
        return ''
    nodes = (result or {}).get('nodes') or []
    if not nodes:
        return ''
    overflow = len(nodes) > BOOT_INJECT_CAP
    nodes = nodes[:BOOT_INJECT_CAP]
    lines = ['## Standing items']
    for n in nodes:
        lines.append('- [%s] %s (id:%s, since %s)' % (
            n.get('type', '?'), n.get('title', '?'),
            (n.get('id') or '')[:8], (n.get('created_at') or '')[:10]))
    if overflow:
        lines.append("- (+more — filter_nodes(field='type', include=%r))" % types)
    return '\n'.join(lines)
