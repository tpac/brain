"""Surface Contract — S1 surface (Haiku) prompt building, candidate formatting, output formatting.

S1 Surface pushes relevant memories into awareness. This contract defines:
- What the surfacer sees (SURFACE config, CANDIDATE_POOL, neighbor fields)
- How candidates are formatted (format_candidate_for_surface)
- How the prompt is assembled (build_surface_prompt)
- How output is formatted for Anchor (format_surface_output_activation)

Correction enrichment lives on Brain (brain.correction_enrich) — it's a
graph join, not a surface concern. Every canonical node pull (brain.get_node)
attaches `_corrections` automatically.

Interaction: 'surface' in interactions table. Prompt is learnable.
"""

import hashlib
import os
import random
import threading
from datetime import datetime, timezone
from collections import OrderedDict

from servers.daemon_config import brain_tmp_dir
# Hoisted: any embedder API drift fails at daemon boot, not 16s into hook_recall.
from servers.embedder import embed_batch
from servers import embedder as _embedder

# Bounded cache for edge-description embeddings. Same descriptions recur
# across turns and across sessions — caching eliminates redundant embed calls.
#
# Key: (model_name, description_text). Model-scoped so a model swap doesn't
# hand out vectors from the wrong geometry. Values are L2-normalized blobs.
# LRU eviction via OrderedDict; capped at 5000 entries (~15MB at 3KB/vector).
# All reads/writes guarded by _DESC_CACHE_LOCK so concurrent embed callers
# (future multi-threaded recall/backfill) can't corrupt LRU order.
_DESC_VEC_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_DESC_VEC_CACHE_MAX = 5000
_DESC_CACHE_LOCK = threading.Lock()
# Note (2026-05-09): the hit/miss counters and the get_desc_cache_stats()
# diagnostic that read them were removed when edges got first-class stored
# embeddings (schema v26). The cache below is now a fallback path only —
# fired for edges whose stored embedding is NULL — so the hit-rate metric
# became uninteresting in production. Re-add if/when there's a consumer.


def _desc_vecs_batched(descs):
    """Resolve descriptions → normalized blobs, using cache + one batched embed.

    Returns list of blobs aligned with descs input. Empty strings return None.
    Keys the cache by active model name so model swaps don't hand out stale
    geometry. Dedupes within a single call so a batch with duplicates only
    pays one embed per unique text.

    Post-v26 this function is FALLBACK-ONLY for edges whose stored
    `edge_relations.embedding` is NULL — production write paths populate
    the column at write time. If the cache is fielding many calls in
    production, that's a signal that backfill is incomplete or the
    write-path embed hook is failing.
    """
    if not descs:
        return []

    model = _embedder.stats.get('model_name') or ''
    out = [None] * len(descs)

    # Partition under lock: cached → out[i]; uncached → unique_texts + index map
    # (unique_texts is deduped so duplicate descriptions in one call only embed once)
    unique_texts: list = []
    text_to_unique_idx: dict = {}
    indices_for_unique: list = []  # list of lists of original indices per unique text
    with _DESC_CACHE_LOCK:
        for i, d in enumerate(descs):
            if not d:
                continue
            key = (model, d)
            blob = _DESC_VEC_CACHE.get(key)
            if blob is not None:
                out[i] = blob
                _DESC_VEC_CACHE.move_to_end(key)
                continue
            u_idx = text_to_unique_idx.get(d)
            if u_idx is None:
                u_idx = len(unique_texts)
                text_to_unique_idx[d] = u_idx
                unique_texts.append(d)
                indices_for_unique.append([i])
            else:
                indices_for_unique[u_idx].append(i)

    # Embed outside the lock — embedder call may be slow; we hold no state.
    if unique_texts:
        fresh = embed_batch(unique_texts, kind='document')
        # Populate out[] + cache under lock
        with _DESC_CACHE_LOCK:
            for u_idx, blob in enumerate(fresh):
                if blob is None:
                    continue
                for i in indices_for_unique[u_idx]:
                    out[i] = blob
                _DESC_VEC_CACHE[(model, unique_texts[u_idx])] = blob
            while len(_DESC_VEC_CACHE) > _DESC_VEC_CACHE_MAX:
                _DESC_VEC_CACHE.popitem(last=False)

    return out


def _relative_time(iso_str, now=None, fine=False):
    """Convert UTC ISO timestamp to relative time label.

    Returns human-readable age: 'just now', 'today', 'yesterday', '3d ago', '2w ago', '1mo ago'.
    Both surface and Anchor see this instead of raw UTC timestamps.

    `fine` (encoder catalog): sub-day steps — '25m ago' / '3h ago' instead of
    'just now' / 'today'. Mid-session the hour is the signal ("revised 20m ago
    → my own recent write" vs "today" which swallows the whole working day).
    Default off — surface keeps its coarse vocabulary.

    `now` — the as-of instant (tz-aware). Replays must pass conversation time
    explicitly (the same rule as iso_cutoff — see tests/test_time_window_contract);
    None → operator wall-clock, correct for live renders only.
    """
    if not iso_str:
        return None
    try:
        ts_str = str(iso_str).replace('Z', '+00:00')
        if '+' not in ts_str and ts_str.count('-') <= 2:
            ts_str += '+00:00'
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if now is None:
            # brain_now() in operator's TZ — relative-time labels ("today",
            # "yesterday") must reflect the operator's day boundaries, not the
            # daemon host's UTC day boundaries. See servers/clock.py.
            from servers.clock import brain_now
            now = brain_now()
        delta = now - ts
        hours = delta.total_seconds() / 3600
        days = delta.days

        if fine and hours < 24:
            if delta.total_seconds() < 120:
                return "just now"
            if hours < 1:
                return "%dm ago" % (delta.total_seconds() // 60)
            return "%dh ago" % hours
        if hours < 1:
            return "just now"
        elif hours < 24:
            return "today"
        elif hours < 48:
            return "yesterday"
        elif days < 7:
            return "%dd ago" % days
        elif days < 30:
            return "%dw ago" % (days // 7)
        else:
            months = days // 30
            return "%dmo ago" % months
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════
# SURFACE CONFIG
# ═══════════════════════════════════════════════════════════════

# Haiku's fetch tools — the discovery values that mean "came from a tool
# call" (vs recall MODES like laf_v1/embedding, which are engine provenance).
# Must stay in sync with fetch_tools.TOOL_DEFINITIONS; pinned by
# test_surface_transitions.test_fetch_tool_names_in_sync.
SURFACE_FETCH_TOOLS = ('recall_topical', 'recall_by_time')

# Candidate pool — how many nodes recall pulls for the surface funnel.
# Production reads only max_candidates; content_limit and include_metadata
# are read by the eval harnesses (judge_eval, surface_ab_eval) when they
# rebuild candidates.
CANDIDATE_POOL = {
    'content_limit': 1000,
    'max_candidates': 25,
    # Over-fetch so the seen-dedup filter (already-surfaced nodes dropped
    # from the pool) has ranked replacements to backfill from. Without it
    # the filter only SHRINKS the pool — recall truncates at its limit
    # before the filter runs (a98143f review, finding 1). Sized to the
    # seen-set ceiling: the window holds ~4 selections × ~4 turns.
    'seen_dedup_headroom': 15,
    'include_metadata': True,   # situation, reasoning, user_raw_quote
}


def surface_selected_path(session_id, stop_counter):
    """Canonical path of the per-turn surfaced-ids file.

    Writer: surface.py (once per run_surface, post liveness gate).
    Reader: daemon_hooks._hebbian_strengthen (Stop hook).
    Single source of truth — a writer/reader format drift here is a
    proven bug class (a test once wrote the counter-less format and
    Hebbian silently read nothing: file_missing on every Stop).
    """
    return os.path.join(brain_tmp_dir(), 'brain-%s-%d-surface-selected.json' % (
        session_id, stop_counter))

# The model used by the S1 surface step. Single source of truth — read by:
#   - surface.py:_call_surface (the actual selection call)
#   - brain.py:warm_up (the boot-time ping that pre-pays SDK + TLS + Haiku
#     route cold-start so the user's first prompt doesn't carry it)
# Kept as a flat constant rather than buried in SURFACE so the warmup path
# doesn't have to reach into surface-call config to know which model to ping.
SURFACE_MODEL = 'claude-haiku-4-5'

# Anthropic prompt-cache minimum cacheable prefix for SURFACE_MODEL.
# Floors are model-specific: Haiku 4.5 + Opus 4.x = 4096; Sonnet 4.6 +
# Fable 5 = 2048; Sonnet 4.5 and older = 1024. This constant is scoped to
# SURFACE_MODEL (Haiku 4.5 → 4096); the Sonnet encoders live above that
# floor at 2048. Prompts under the floor are silently not cached by the API.
# The agentic loop's cache-miss tripwire gates on it so sub-minimum prompts
# (tests, tiny brains) don't warn.
CACHE_MIN_PREFIX_TOKENS = 4096

# Judge (Haiku) — selects relevant nodes with reasoning
# v9: max_candidates 25→20, Anchor truncation 150→400, recent_messages 5→7
SURFACE = {
    'content_limit': 300,           # shorter per node since more candidates
    'max_candidates': 30,           # 2026-05-01: was 20. FTS5 adds up to 5 more = 35 max total.
                                    # Bumped to give Haiku more axis-of-context room while
                                    # multi-axis candidate generation is being designed.
    'max_selected': 5,              # Haiku picks at most this many (was 8 — reduced for 10K hook cap)
    'user_message_limit': 300,
    'anchor_message_limit': 400,    # v9: was 150. Anchor responses carry design context
    'recent_messages': 8,           # previous MESSAGES (~4 exchanges when turns alternate;
                                    # interrupted turns leave lone user messages, so pairing
                                    # is not guaranteed). Single source of truth: daemon_hooks
                                    # pulls exactly this many from traces (current chain
                                    # excluded via exclude_trace_id, wake envelopes filtered);
                                    # build_surface_prompt slices the same number. The current
                                    # prompt is NOT in this window — it renders as its own
                                    # "Current message" block.
    'recent_recalls_messages': 5,   # Aligned with recent_messages: the dedup window matches
                                    # the conversation window Haiku sees. Was 10 — meant any
                                    # node surfaced in the last 10 selections was rendered as
                                    # "do not re-pick", which exceeded the conversation context.
    'session_context_limit': 800,   # shared with ENCODING_AGENT — full session journey
    'session_context_tail': 800,  # 2026-05-02 (Frame Phase 1): was 200. Surface now gets
                                  # the full session_context blob, not just the tail.
                                  # Encoder writes ~768 chars; surface was seeing ~25%.
                                  # See docs/FRAME-DESIGN.md Phase 1.
    'max_tokens': 600,              # Haiku output cap
}


def recall_score(recall_result):
    """The ONE score semantic for surface candidates.

    brain.recall() results carry no 'score' key — the surface pipeline's
    score is `effective_activation` from the scoring pipeline. Both the
    hook's cosine pool (daemon_hooks) and fetch_tools.recall_topical MUST
    read it through this function: the agentic admission floor compares
    tool-fetched scores against the pool median, so the two sides forking
    on field name silently zeroes one of them (2026-07-02: recall_topical
    read 'score' → 0.0 for every fetch → floor dropped 100% for 3 weeks).
    """
    return recall_result.get('effective_activation') or 0


# ═══════════════════════════════════════════════════════════════
# NEIGHBOR FIELDS — what to show for graph neighbors
# ═══════════════════════════════════════════════════════════════

# Degree 1 neighbors (closest — richest display)
NEIGHBOR_D1_FIELDS = {
    'id', 'type', 'title', 'relation', 'confidence',
    'locked', 'revised_at', 'content_summary', 'weight',
}

# Degree 2 neighbors (breadcrumbs — moderate display)
NEIGHBOR_D2_FIELDS = {
    'id', 'type', 'title', 'relation',
}

# Degree 3 neighbors (hints — minimal display)
NEIGHBOR_D3_FIELDS = {
    'id', 'title',
}

# Truncation per neighbor field
NEIGHBOR_TRUNCATION = {
    'd1_title': 60,
    'd1_id': 12,
    'd1_content_summary': 150,
    'd2_title': 35,
    'd2_id': 8,
    'd3_title': 30,
    'd3_id': 8,
}


# DEPRECATED: use PIPELINE['recall_log_*'] instead.
PRECISION = {
    'title_limit': 100,       # → PIPELINE['recall_log_title']
    'snippet_limit': 150,     # → PIPELINE['recall_log_snippet']
    'query_limit': 500,       # → PIPELINE['recall_log_query']
}


# ═══════════════════════════════════════════════════════════════
# CANDIDATE ENRICHMENT — DEPRECATED 2026-04-07
# Replaced by get_rich_node() in pipeline_contract.py.
# Candidates now use unified get_rich_node shape everywhere.
# ═══════════════════════════════════════════════════════════════

def enrich_candidate_metadata(brain, node_id, node_data, config):
    """DEPRECATED 2026-04-07: Use brain.get_node() instead.

    Kept as stub for backward compatibility with eval scripts.
    """
    rich = brain.get_node(node_id)
    if rich:
        node_data.update({k: v for k, v in rich.items() if k not in node_data})


# ═══════════════════════════════════════════════════════════════
# FORMATTERS
# ═══════════════════════════════════════════════════════════════

def format_candidate_for_surface(c, index, layout='legacy',
                                 scope=None):
    """Format a single candidate for the surface prompt.

    Thin wrapper around render_rich_node(HAIKU_FORMAT) — adds recall-specific
    fields (score, discovery) that aren't in the node itself.

    Candidates must be in get_rich_node() shape (with _metadata, _corrections, connections).

    Render format: BRAIN_HAIKU_RENDER env ∈ {'lean' (default), 'full'} picks
    HAIKU_FORMAT_LEAN vs HAIKU_FORMAT. Default flipped full→lean 2026-06-12
    after the ablation (ab_render_ablation.py): gold-neutral at −41% tokens,
    pick-divergence (J=0.64) at the same-prompt noise floor (full-vs-full
    J=0.72). Covers cosine candidates AND tool results (fetch_tools renders
    through this same function).

    layout='xml_v13': wraps the same node body in a <candidate> element —
    id as attribute (the pick key), locked / source_tool as flags. No score
    header: exposing match floats invites anchoring on our numbers instead
    of semantic judgment (v13 decision). Since 2026-07-14 (§20.12 A2) the
    menu order is shuffled per turn, so in xml_v13 the menu carries NO rank
    signal at all — Haiku judges on meaning alone.
    """
    import os as _os_hr
    from servers.contract import render_rich_node

    fmt = HAIKU_FORMAT \
        if _os_hr.environ.get('BRAIN_HAIKU_RENDER', 'lean').strip().lower() == 'full' \
        else HAIKU_FORMAT_LEAN
    if scope is not None:
        # Differential project exposure (never mutate the shared constant):
        # foreign-project candidates render a mismatch mark, same-project
        # render nothing — see render_rich_node.
        fmt = dict(fmt, scope=scope)

    discovery = c.get("discovery", "")

    if layout == 'xml_v13':
        attrs = ['id="%s"' % str(c.get('id', ''))[:8]]
        if c.get('locked'):
            attrs.append('locked="true"')
        # Strict tool-name match: `discovery` also carries recall MODES
        # (laf_v1, embedding+keyword, ...) which are not tool provenance —
        # the legacy "anything non-embedding" predicate mislabeled them.
        if discovery in SURFACE_FETCH_TOOLS:
            attrs.append('source_tool="true"')
        return '<candidate %s>\n%s\n</candidate>' % (
            ' '.join(attrs), render_rich_node(c, fmt))

    # Recall-specific header (score + discovery — not part of the node)
    score_parts = []
    score = c.get("score", 0)
    if score:
        display_score = min(score, 1.0)
        score_str = "match:%.2f" % display_score
        if score > 1.0:
            score_str += ",boosted"
        score_parts.append(score_str)
    if discovery and discovery not in ("embedding", "embedding_only", "embedding+keyword"):
        score_parts.append("via:%s" % discovery)

    header = "#%d" % index
    if score_parts:
        header += " (%s)" % ", ".join(score_parts)

    return header + "\n" + render_rich_node(c, fmt)


def _dedup_candidates(candidates):
    """Remove near-duplicate candidates by title similarity.
    Keeps the higher-scored candidate when two titles share >80% words."""
    if len(candidates) <= 1:
        return candidates
    seen_titles = {}  # normalized title words → candidate
    result = []
    for c in candidates:
        title_words = set(c.get("title", "").lower().split())
        duplicate = False
        for seen_key, seen_c in list(seen_titles.items()):
            seen_words = set(seen_key.split())
            if not title_words or not seen_words:
                continue
            overlap = len(title_words & seen_words) / max(len(title_words), len(seen_words))
            if overlap > 0.8:
                # Keep the higher scored one
                if (c.get("score", 0) or 0) > (seen_c.get("score", 0) or 0):
                    result.remove(seen_c)
                    seen_titles[" ".join(sorted(title_words))] = c
                    result.append(c)
                duplicate = True
                break
        if not duplicate:
            key = " ".join(sorted(title_words))
            seen_titles[key] = c
            result.append(c)
    return result


def _group_turns(msgs):
    """Group a flat oldest-first message list into user+assistant turns.

    A user message always opens a new turn; an assistant message closes the
    open turn or opens its own (lone assistant — e.g. the user half was a
    filtered wake envelope). No strict-alternation assumption.
    """
    groups = []
    for m in msgs:
        if m.get('role') == 'user':
            groups.append({'user': m})
        elif groups and 'assistant' not in groups[-1]:
            groups[-1]['assistant'] = m
        else:
            groups.append({'assistant': m})
    return groups


def _build_user_content_xml(candidates, user_message, recent_messages,
                            retrieval_stats, frame, cfg,
                            scope=None):
    """v13 XML user content — one grammar, oldest first, current message last.

    Sections: <partnership_context> (Frame), <conversation> (turns with
    per-turn <shown> dedup elements; final turn is current_msg="true", user
    only), <candidates> (id-attribute grammar, no score floats), and a
    conditional <note> when retrieval is weak (the threshold logic stays in
    code; Haiku sees the conclusion, never our numbers).

    Replaces the legacy separate "Recently surfaced" section: already-shown
    memories sit inline on the turn they surfaced for, so out-of-scope is
    structural (<shown> vs <candidate>), not a rule about a distant block.
    """
    parts = []
    parts.append('<partnership_context>\n%s\n</partnership_context>' % (
        frame.strip() if frame
        else '(none — fresh session or Frame unavailable)'))

    lines = ['<conversation>']
    n = 0
    for g in _group_turns((recent_messages or [])[-(cfg['recent_messages']):]):
        n += 1
        lines.append('<turn n="%d">' % n)
        u = g.get('user')
        if u:
            lines.append('<user>%s</user>'
                         % (u.get('content') or '')[:cfg['user_message_limit']])
            for s in (u.get('surfaced') or []):
                lines.append('<shown id="%s">%s</shown>' % (
                    str(s.get('id', ''))[:8], (s.get('title') or '')[:80]))
        a = g.get('assistant')
        if a:
            lines.append('<assistant>%s</assistant>'
                         % (a.get('content') or '')[:cfg['anchor_message_limit']])
        lines.append('</turn>')
    n += 1
    lines.append('<turn n="%d" current_msg="true">' % n)
    lines.append('<user>%s</user>' % (
        user_message[:cfg['user_message_limit']] if user_message
        else '(no message)'))
    lines.append('</turn>')
    lines.append('</conversation>')
    parts.append('\n'.join(lines))

    cand_lines = ['<candidates n="%d">' % len(candidates)]
    for c in candidates:
        cand_lines.append('')
        cand_lines.append(format_candidate_for_surface(
            c, 0, layout='xml_v13', scope=scope))
    cand_lines.append('</candidates>')
    parts.append('\n'.join(cand_lines))

    if retrieval_stats:
        from servers.brain_constants import RETRIEVAL_LOW_CONFIDENCE
        if retrieval_stats.get('top_score', 0) < RETRIEVAL_LOW_CONFIDENCE:
            parts.append('<note>Retrieval is weak for this message — the '
                         'brain likely has nothing relevant. Prefer '
                         'selecting 0.</note>')

    return '\n\n'.join(parts), cfg['max_tokens']


def presentation_shuffle_seed(session_id, user_message):
    """Deterministic per-turn seed for the candidate presentation shuffle.

    Derived from (session_id, user_message) — the two per-turn inputs
    _call_surface already holds — so a replay of the same turn reproduces
    production's presentation order exactly. sha256, not hash(): Python
    salts hash() per process, and the seed must be stable across daemon
    restarts and replay processes.
    """
    key = ('%s|%s' % (session_id or '', user_message or '')).encode(
        'utf-8', 'replace')
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def prepare_presented_candidates(candidates, shuffle_seed=None, cfg=None):
    """Slice + dedup + presentation shuffle — the exact candidate list
    build_surface_prompt renders, in the order Haiku sees it.

    The shuffle (2026-07-14, RECALL-SR-REDESIGN.md §20.12 A2): Haiku shows
    a mild pure position bias (1.19× top-vs-bottom pick rate within
    identical-score tie groups, 92k candidates), so the PRESENTATION order
    is randomized per turn. The scorer's ranking is untouched everywhere
    else — the O-trace cand_detail and the walker's rank_in_pool keep
    recording it; only what Haiku sees shuffles. Side effect: every
    picked/dropped log row becomes randomization-grade training data for
    the LAF P3 fit (exact uniform propensities, no estimation).

    Slice + dedup run BEFORE the shuffle so top-N membership stays
    scorer-determined. Seeded RNG so the capture/replay byte-contract
    holds; shuffle_seed=None skips the shuffle (legacy behavior — evals
    and tests that don't pass a seed are bit-identical to pre-shuffle).
    """
    cfg = cfg or SURFACE
    cands = _dedup_candidates(candidates[:cfg['max_candidates']])
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(cands)
    return cands


def build_surface_prompt(candidates, user_message,
                       recent_messages=None, recently_recalled=None,
                       retrieval_stats=None, frame="",
                       layout='legacy', shuffle_seed=None,
                       scope=None):
    """Build the S1 recall surface USER message — per-turn delta only.

    v11 (2026-05-03, Frame Phase 2.5 / surface prompt v2): instructions
    moved to the cached system block (in `_call_surface`). This function
    now builds ONLY the per-turn user content: Frame, conversation,
    recently surfaced, retrieval stats, candidates. The registered
    `surface` interaction template is the system block; this is the user
    block. Two parts → two-block API call → caching becomes possible.

    Operator name is rendered generically as "Operator:" — the brain plugin
    ships to different operators, prompts must not hardcode personal names.

    Args:
        candidates: List of candidate node dicts (enriched with metadata)
        user_message: The operator's latest message — rendered as its own
            "Current message" block (user only, no reply yet)
        recent_messages: List of {"role": str, "content": str} — PREVIOUS
            turns only; the caller excludes the current turn's chain
            (daemon_hooks passes exclude_trace_id to get_session_turns)
        recently_recalled: List of {"id": str, "title": str} from last N recalls
        retrieval_stats: Dict with brain_size, top_score, median_score, source_breakdown
        frame: Markdown Frame (Anchor's prior). When non-empty becomes the
            "Partnership context:" block. When empty, explicit degraded marker.
        shuffle_seed: When not None, presentation order is a seeded shuffle
            of the deduped pool (see prepare_presented_candidates). None →
            scorer order, bit-identical to pre-shuffle rendering.

    Returns: (user_content_string, max_tokens)
    """
    cfg = SURFACE

    # Slice + dedup + presentation shuffle (§20.12 A2) — one prep path,
    # shared with the presented-order record _call_surface writes to traces.
    candidates = prepare_presented_candidates(candidates, shuffle_seed, cfg)

    # v13 XML layout — selected by the active `surface` interaction config
    # ({"layout": "xml_v13"}), so template and renderer flip atomically.
    # Ignores `recently_recalled` (replaced by per-turn <shown> elements).
    if layout == 'xml_v13':
        return _build_user_content_xml(
            candidates, user_message, recent_messages, retrieval_stats,
            frame, cfg, scope=scope)

    # Format conversation context (both roles, asymmetric truncation).
    # PREVIOUS turns only — the caller excludes the current chain upstream
    # (get_session_turns exclude_trace_id); the current message renders below
    # as its own block, never inside the history window.
    # 2026-05-03: "Operator:" is generic — prompts ship to different operators.
    conversation = ""
    if recent_messages:
        for msg in recent_messages[-(cfg['recent_messages']):]:
            role = msg.get("role", "?")
            if role == "user":
                label = "Operator"
                content = (msg.get("content") or "")[:cfg['user_message_limit']]
            else:
                label = "Anchor"
                content = (msg.get("content") or "")[:cfg['anchor_message_limit']]
            conversation += "%s: %s\n" % (label, content)

    # Current user message — the one Haiku surfaces for. Rendered as its own
    # labeled block so it never blurs into the history window (user only:
    # the assistant hasn't replied yet). Explicit degraded marker when empty
    # so Haiku is never told to surface for a blank line.
    current_block = ("Operator: %s" % user_message[:cfg['user_message_limit']]) \
        if user_message else "(no message)"

    # 2026-05-02 (Frame Phase 2): Frame is the canonical session prior.
    # When non-empty, it's the "Partnership context:" block — rich content
    # carrying operator + partnership + active-threads + current focus +
    # recent moves. When empty (Frame Constructor failed), surface runs
    # without any partnership context — explicit degraded mode, logged
    # upstream by daemon_hooks. No silent fallback to a different layout.
    partnership_block = ""
    if frame:
        partnership_block = "Partnership context (your prior — what's currently in awareness):\n" + frame

    # Format recently recalled (lightweight — id + title only)
    recalled_text = ""
    if recently_recalled:
        for r in recently_recalled:
            recalled_text += "%s \"%s\"\n" % (str(r.get("id", ""))[:8], r.get("title", "")[:60])

    # v9: Build retrieval context block (always present when stats available)
    retrieval_context = ""
    if retrieval_stats:
        rs = retrieval_stats
        top = rs.get('top_score', 0)
        median = rs.get('median_score', 0)
        brain_sz = rs.get('brain_size', 0)
        n_candidates = rs.get('candidates_after_floor', 0)
        breakdown = rs.get('source_breakdown', {})

        retrieval_context = "Retrieval: %d candidates from %d memories. Top: %.2f, median: %.2f." % (
            n_candidates, brain_sz, top, median)

        # Source breakdown (non-zero only)
        src_parts = []
        for src, count in breakdown.items():
            if count > 0:
                src_parts.append("%d %s" % (count, src))
        if src_parts:
            retrieval_context += " Sources: %s." % ", ".join(src_parts)

        # v9: Dynamic guidance based on distribution
        from servers.brain_constants import RETRIEVAL_LOW_CONFIDENCE
        if top < RETRIEVAL_LOW_CONFIDENCE:
            retrieval_context += (
                "\nNOTE: Top score %.2f is low for %d memories — "
                "brain likely has nothing relevant. Prefer selecting 0." % (top, brain_sz))

    # Format candidates
    candidates_text = ""
    for i, c in enumerate(candidates, 1):
        candidates_text += format_candidate_for_surface(
            c, i, scope=scope) + "\n\n"

    # User content — per-turn delta only. Instructions live in the cached
    # system block (the registered `surface` interaction template), assembled
    # in _call_surface. This function builds ONLY what changes per turn.
    user_content = """%s

Conversation (previous turns, oldest first):
%s
Current message (the one to surface for — the assistant has not replied yet):
%s

Recently surfaced (OUT OF SCOPE — Anchor has already seen these in the last 5 turns; do NOT select any ID from this block):
%s
%s
%d candidates follow. Select 0-%d.

Candidates:

%s""" % (
        partnership_block or "(no partnership context — fresh session or Frame unavailable)",
        conversation or "(none — first message of the session)\n",
        current_block,
        recalled_text or "(none)",
        retrieval_context,
        len(candidates),
        cfg['max_selected'],
        candidates_text,
    )

    return user_content, cfg['max_tokens']


# ═══════════════════════════════════════════════════════════════
# RENDER FORMATS — per-mode constants for the surface OUTPUT
# ═══════════════════════════════════════════════════════════════

# `format_surface_output_activation` is the live path. It picks a mode
# based on Haiku's per-pick `mode` annotation, looks up the matching
# SURFACE_*_FORMAT constant, then resolves it to a concrete
# render_rich_node cfg via resolve_surface_format(fmt, budget).
#
# Constants here use *proportions* (of the per-node budget) rather than
# absolute char limits — the budget allocator decides how many chars
# each node gets; the format decides the within-node split. Floors
# (min_*_chars) prevent micro-budgets from producing empty renders.

# Arc mode — DEFAULT for surfaced nodes. State-of-mind framing: Anchor
# reads the gist + situation + voice fields; low-activation fields are
# masked out so noise drops cleanly.
SURFACE_ARC_FORMAT = {
    'content_proportion':  0.60,
    'metadata_proportion': 0.15,
    'min_content_chars':   50,
    'min_metadata_chars':  40,
    'edge_limit':          3,
    'time_format':         'relative',
    'show_confidence':     False,
    'show_encoding_source': False,
    'extra_skip_keys':     ('question',),   # recall scaffold
    'correction_render':   'balanced',
}

# Fact mode — verbatim content. Used when Haiku tags a pick as carrying
# a specific value/quote/date the operator literally asked for. Larger
# content budget; no field masking.
SURFACE_FACT_FORMAT = {
    'content_proportion':  0.75,
    'metadata_proportion': 0.15,
    'min_content_chars':   120,
    'min_metadata_chars':  60,
    'edge_limit':          3,
    'time_format':         'relative',
    'show_confidence':     False,
    'show_encoding_source': False,
    'extra_skip_keys':     ('question',),
    'correction_render':   'balanced',
}

# Background mode — title + 1-line situation only. Cheap context.
# Doesn't go through render_rich_node (too minimal); inline-rendered.
SURFACE_BACKGROUND_FORMAT = {
    'situation_max_chars': 200,
}

# Valid render modes Haiku may emit in selection JSON. Default 'arc'
# when a pick has no `mode` field.
SURFACE_MODES = ('arc', 'fact', 'background')
SURFACE_MODE_DEFAULT = 'arc'


def resolve_surface_format(fmt, budget, scope=None):
    """Resolve a SURFACE_*_FORMAT contract into a concrete render_rich_node cfg.

    Translates proportional fields (content_proportion, metadata_proportion)
    into absolute char limits using the per-node budget. Honours min_*
    floors so micro-budgets don't produce empty renders. Returns a cfg
    dict ready to pass to `render_rich_node(node, cfg)`.

    `scope`: the session's declared scope dimensions — injected here (the
    cfg is already a fresh dict) so callers never hand-clone a shared format
    constant; a forgotten clone would leak one session's scope into every
    other session's renders.
    """
    cfg = {k: v for k, v in fmt.items()
           if k not in ('content_proportion', 'metadata_proportion',
                        'min_content_chars', 'min_metadata_chars',
                        'situation_max_chars')}
    if 'content_proportion' in fmt:
        cfg['content_limit'] = max(
            fmt.get('min_content_chars', 50),
            int(budget * fmt['content_proportion']))
    if 'metadata_proportion' in fmt:
        cfg['metadata_limit'] = max(
            fmt.get('min_metadata_chars', 30),
            int(budget * fmt['metadata_proportion']))
    if scope:
        cfg['scope'] = scope
    return cfg


# ═══════════════════════════════════════════════════════════════
# PICKER RENDER — what Haiku reads when selecting 3-5 from 25
# ═══════════════════════════════════════════════════════════════

# Distinct from the SURFACE_*_FORMAT constants above: HAIKU_FORMAT is
# the IN (rendered candidate menu Haiku reads), the SURFACE_*_FORMATs
# are the OUT (what Anchor receives after Haiku's selection).
HAIKU_FORMAT = {
    'content_limit': 300, 'edge_limit': 3, 'metadata_limit': 120,
    'time_format': 'relative',
    # Haiku surface receives correction context at balanced fidelity:
    # relation + edge_description + content excerpt (~150 chars). Enough
    # for picks to factor in superseded knowledge without bloating the
    # 25-candidate prompt.
    'correction_render': 'balanced',
}

# Selection-grade lean render (Area 2, 2026-06-12). The selector's job is
# "pick 3-5 relevant ids", not "read the node" — injection still renders
# the SELECTED nodes at full richness, so no information leaves the
# pipeline; it's relocated to the stage that uses it. Recon numbers
# (eval/oracle_audit/ab_render_recon.py, 150 candidates): full render
# 385 tok/cand — edges-with-descriptions 29%, content 23%, situation 10%,
# reasoning 10%, question 6%, corrections 6%, quotes 5%. Lean keeps every
# SELECTION signal in cheapest sufficient form (~60% cut):
#   • situation kept whole — it IS the selection question ("when relevant")
#   • content kept at 300 (operator call, 2026-06-12)
#   • encoding_source kept in header (future: guide Haiku to prefer
#     src:anchor manual encodings)
#   • edges → oneline (direction + relation + title; descriptions are
#     injection payload). Edge CHOICE stays query-aware via select_edges.
#   • corrections → lean flag (the "superseded" signal, not the payload)
#   • dropped: reasoning, question, quotes — encoder/recall
#     scaffolding and voice; zero hypothesized selection value (the
#     inject path already drops question for the same reason).
# DEFAULT since 2026-06-12 (ablation-cleared: gold-neutral, −41% tokens,
# divergence at the same-prompt noise floor — ab_render_ablation.py).
# BRAIN_HAIKU_RENDER=full reverts to the heavy render.
# Open follow-on: lean_noedge scored equal-or-better at −53% — edge lines
# carry no measurable selection signal TODAY, but the cut is deferred until
# the aspect-aligned edge-choice experiment decides whether edges can earn
# their place in selection (operator's aspect-traversal thread).
HAIKU_FORMAT_LEAN = {
    'content_limit': 300, 'edge_limit': 3, 'metadata_limit': 120,
    'time_format': 'relative',
    'correction_render': 'lean',
    'edge_style': 'oneline',
    'extra_skip_keys': ('question', 'reasoning', 'user_raw_quote',
                        'anchor_raw_quote'),
}


# ═══════════════════════════════════════════════════════════════
# OUTPUT SCHEMA — what Haiku must produce (Anthropic Structured Outputs)
# ═══════════════════════════════════════════════════════════════

# Enforced during generation via output_config={'format':{'type':'json_schema',
# 'schema': SURFACE_SELECTION_SCHEMA}} on the final agentic-round API call.
# See surface.py:_call_surface_agentic.
# id pattern: hex-only kills the observed corruption class — Haiku copying a
# candidate id as its BPE chunks with spaces leaked in ('9 9a 2e ' for
# 99a2e…) — by masking the space token during constrained decoding. Length
# {4,8}, not {8}: Haiku is known to emit 7-char leading-zero-dropped ids, and
# a hard 8 would force it to guess digits it doesn't know; short-but-honest
# fragments are recovered downstream (surface.py unique-prefix match).
# No per-pick rationale field by design: generation time tracks output
# tokens (~20ms/token), a why per pick cost ~150-200 tokens/recall, and
# nothing rendered it. The recall-level `reason` (kept, cheap) is the
# journal signal — persisted as selection_reason in the K trace.
SURFACE_SELECTED_ID_PATTERN = "^[0-9a-f]{4,8}$"

SURFACE_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "string",
                             "pattern": SURFACE_SELECTED_ID_PATTERN},
                    "mode": {"type": "string", "enum": list(SURFACE_MODES)},
                },
                "required": ["id", "mode"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["selected"],
    "additionalProperties": False,
}


# ═══════════════════════════════════════════════════════════════
# SPREAD_ACTIVATION — the kernel that decides what surfaces
#
# One mechanism, three granularities:
#   • Nodes receive activation  (max over per-field cosines with query)
#   • Edges transmit activation (cosine of query to enriched edge text)
#   • Fields within a node hold per-field activation (for render budget)
#
# Depth is emergent — high-coefficient edges keep propagating, low-
# coefficient edges stop on the first hop. Saturation (tanh) bounds
# accumulation. The per-hop transmission gate is self-calibrating:
# when no edges in the batch exceed the brain's noise floor, propagation
# halts — the brain is allowed to be quiet.
#
# No static blend weights. No multiplicative family boost (family meaning
# rides inside the enriched edge text). No count-based fatigue. The
# activation map IS the ranking, the tiebreaker, and the render-budget
# allocator, all in one.
# ═══════════════════════════════════════════════════════════════

import numpy as np

# Safety cap — if activation hasn't settled by 3 hops, force stop.
# Real termination comes from the transmission gate + saturation, not this.
_SPREAD_MAX_STEPS = 3

# Per-source neighbor cap during spread expansion. Held at 50 after
# 2026-05-01 partial-rollback: lim=30 was tried (variant C, iter_M
# 14/15 on seed=42) but iter_O on a different sample (seed=1) revealed
# the narrowing starves muster recall during ingestion → encoder
# misses specific facts → quality drops 80%→60% on that sample. The
# narrower lim hurt 5 items on seed=1, helped 0. Reverted.
#
# The latency win lives entirely in HOP_SCRUTINY_DEFAULT (variant D in
# the bench): scrutiny alone reduces avg recall 11.2s → 6.3s (44%) by
# narrowing only at hop 3, where the work doesn't bite encoder context.
# Lim stays at production-original 50.
SPREAD_NEIGHBOR_LIMIT_DEFAULT = 50

# Per-hop scrutiny: at hop 3+ (step >= 2), only the top half of currently
# active nodes by activation propagate further. Distribution-derived —
# the threshold is the median of the current activation set, not a static
# number. Forces narrowing as depth increases.
#
# Production default = OFF (2026-05-01): scrutiny was shipped on the
# strength of LongMemEval iter_M (14/15) but the eval is N=15 with ~20pp
# variance, and scrutiny was never A/B'd in isolation against baseline.
# Real-world conversational use (where ambient context lives 2-3 hops out)
# regressed — see the ex.co class of failure: the brain held rich operator
# context but it lived past the scrutiny cut and stopped surfacing.
# Override BRAIN_SPREAD_HOP_SCRUTINY=on to re-enable for benchmarking.
HOP_SCRUTINY_DEFAULT = False

# Outer propagation gate: if the max transmission coefficient in the batch
# falls below the brain's noise floor, nothing meaningful to spread —
# halt rather than forcing below-noise edges through. Imported from
# brain_constants to stay in sync with recall()'s own noise definition.
from servers.brain_constants import NOISE_FLOOR_THRESHOLD as _SPREAD_NOISE_FLOOR


def _batch_load_field_vectors(brain_conn, node_ids):
    """One SQL round-trip: all field-cohort + legacy vectors for given nodes.

    Returns: {full_node_id: {vector_type: numpy.ndarray}}
    Short-prefix matching handled by caller if needed; here we read full IDs.
    """
    if not brain_conn or not node_ids:
        return {}
    ids = list(node_ids)
    placeholders = ','.join('?' * len(ids))
    rows = brain_conn.execute(
        "SELECT node_id, vector_type, embedding FROM node_enrichments "
        "WHERE node_id IN (%s) AND embedding IS NOT NULL" % placeholders,
        ids).fetchall()
    out = {}
    for nid, vtype, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        out.setdefault(nid, {})[vtype] = vec
    return out


def _field_cosines_for_node(query_vec, node_vectors, norm_q=None):
    """Cosines against query, per semantic field.

    `node_vectors` = {vector_type: vec} from _batch_load_field_vectors.
    For each field type in the field-cohort stable order, try the per-field
    vector first; fall back to the blended legacy vector via
    FIELD_VECTOR_FALLBACK. Missing field → not in the result (no cosine
    fabricated).

    Returns: {field_name: cosine} — order matches field_vector_types().
    """
    from servers.pipeline_contract import field_vector_types, FIELD_VECTOR_FALLBACK
    out = {}
    for field_type in field_vector_types():
        # Field-cohort vector available?
        vec = node_vectors.get(field_type)
        # Fall back through legacy chain
        if vec is None:
            for fb in FIELD_VECTOR_FALLBACK.get(field_type, []):
                vec = node_vectors.get(fb)
                if vec is not None:
                    break
        if vec is None:
            continue
        out[field_type] = _cosine_nonneg(query_vec, vec, norm_a=norm_q)
    return out


def _bulk_load_stored_edge_embeddings(brain_conn, edge_keys,
                                       brain=None, error_label=None):
    """Bulk-load (edge_id, relation) → embedding blob, filtered by active model.

    Used by both `_build_edge_coeffs` (spread) and `select_edges`
    (per-candidate render). Single SQL round-trip per chunk via
    `(edge_id, relation) IN (VALUES (?, ?), ...)`. Filters out:
      - Rows with NULL embedding (predates v26 backfill / write-path failure)
      - Rows where embedding_model ≠ active model (stale after model swap;
        same staleness pattern node_enrichments uses)

    Stale and NULL rows fall through to the on-demand embed path in the
    caller — correctness preserved at the cost of a fastembed call.

    Args:
        brain_conn: sqlite3 connection.
        edge_keys: iterable of (edge_id, relation) tuples — duplicates fine.
        brain: optional Brain — used only for error logging via _log_error.
            Function works without it (errors silently swallowed) but
            production callers should pass it.
        error_label: brain._log_error tag for failures (e.g.
            'edge_embedding_read_spread'). None disables the log.

    Returns:
        Dict[(edge_id, relation), bytes]. Empty if no input or all stale.
    """
    if not edge_keys:
        return {}
    valid = [(eid, rel) for eid, rel in edge_keys if eid and rel]
    if not valid:
        return {}

    from servers import embedder as _embedder_mod
    active_model = _embedder_mod.stats.get('model_name') or ''

    out: dict = {}  # (edge_id, relation) -> embedding blob
    # SQLite parameter limit is 999 (default; some builds 32766). With
    # 2 params per pair + 1 for active_model, chunk_size=400 → 801 params,
    # safely under 999.
    chunk_size = 400
    keys = list(valid)
    try:
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i:i + chunk_size]
            ph = ','.join(['(?, ?)'] * len(chunk))
            params = [v for pair in chunk for v in pair] + [active_model]
            rows = brain_conn.execute(
                'SELECT edge_id, relation, embedding FROM edge_relations '
                'WHERE (edge_id, relation) IN (VALUES %s) '
                'AND embedding IS NOT NULL '
                'AND embedding_model = ?' % ph, params).fetchall()
            for eid, rel, blob in rows:
                if blob:
                    out[(eid, rel)] = blob
    except Exception as e:
        # Stored-embedding lookup failure: caller's loop falls through to
        # on-demand embed for everything it didn't get back here. Log if
        # we have a brain to log against; otherwise swallow (defensive).
        if brain is not None and error_label:
            try:
                brain._log_error(
                    error_label, e,
                    'bulk fetch of stored edge embeddings — '
                    'falling back to on-demand embed')
            except Exception:
                pass
    return out


def _build_edge_coeffs(brain, activated_nodes, query_vec,
                      cached_edge_coeffs):
    """Collect outgoing edges from all currently activated nodes; compute
    their transmission coefficients.

    Returns list of (source_id, target_id, coeff, edge_dict).
    Uses cached_edge_coeffs as an in-kernel memo so repeat edges in later
    hops don't recompute cosine.

    Implementation note (schema v26+): edges have stored embeddings on
    `edge_relations.embedding`, computed ASYNC by the embed_queue worker
    (`Brain.backfill_edge_embeddings`) — write paths only invalidate (NULL the
    blob) + enqueue_edge. For each edge we look up the stored blob by `edge_id`
    and skip fastembed entirely. Edges still NULL (pre-v26, or written since the
    last worker drain) fall through to the legacy on-demand embed path; recall
    stays correct, just pays the per-edge fastembed cost. The worker repopulates
    NULLs on its next drain.

    Enriched-text composition delegates to
    `brain.aspects.compose_edge_text(relation, description)` — single
    source of truth for the embed format.

    Bulk SQL: get_neighbors_bulk for all sources in one round-trip (3 SQL
    queries per recall instead of 200+). Per-owner neighbor cap enforced
    in Python.
    """
    # brain's held instance — the old brain_conn param existed only to avoid
    # admitting the brain dependency this function already has (brain.aspects,
    # brain._log_error), and rebuilt a GraphDAL once per spread hop.
    gdal = brain._graph
    excluded = set(brain.aspects.traversal_exclusions)

    # Per-source neighbor cap. Default from contract (SPREAD_NEIGHBOR_LIMIT_DEFAULT).
    # Env override: BRAIN_SPREAD_NEIGHBOR_LIMIT — used by eval variants.
    import os as _os
    _SPREAD_LIMIT = int(_os.environ.get(
        'BRAIN_SPREAD_NEIGHBOR_LIMIT', str(SPREAD_NEIGHBOR_LIMIT_DEFAULT)))

    # ── ONE SQL round-trip for all sources ──
    bulk = gdal.get_neighbors_bulk(
        activated_nodes, exclude_relations=excluded)

    # Collect every edge first; we'll fetch stored embeddings in bulk.
    pending = []  # list of (source_id, target_id, edge_dict, enriched_text)
    edge_keys = set()  # set of (edge_id, relation) for the bulk lookup
    for source_id in activated_nodes:
        rows = bulk.get(source_id, [])
        if len(rows) > _SPREAD_LIMIT:
            rows = sorted(rows, key=lambda r: r.get('weight') or 0,
                          reverse=True)[:_SPREAD_LIMIT]
        for r in rows:
            target_id = r.get('id', '')
            # Compose enriched text via AspectRegistry (single source of
            # truth for the format — see aspects.py:compose_edge_text).
            # Used as the in-call memo key + as the input to the on-demand
            # embed fallback if the stored blob is NULL.
            enriched = brain.aspects.compose_edge_text(
                r.get('relation', ''),
                r.get('edge_description') or '')
            pending.append((source_id, target_id, r, enriched))
            eid = r.get('edge_id')
            rel = r.get('relation') or ''
            if eid and rel:
                edge_keys.add((eid, rel))

    # Bulk-load stored embeddings (schema v26+) via the shared helper.
    # See _bulk_load_stored_edge_embeddings docstring for SQL/staleness
    # semantics. Empty result on any failure — caller falls through to
    # on-demand embed for all edges (correctness preserved).
    stored_embeddings = _bulk_load_stored_edge_embeddings(
        brain.conn, edge_keys, brain=brain,
        error_label='edge_embedding_read_spread')

    edges_out = []
    enriched_to_embed = []  # texts still needing embedding (NULL in DB)
    enriched_keys = []      # parallel: (source, target, edge, text)
    norm_q = float(np.linalg.norm(query_vec))

    for source_id, target_id, r, enriched in pending:
        # Per-call memo (in-kernel cache across hops in this recall) —
        # avoids cosine recomputation when the same edge appears as a
        # source-target reverse in a later hop.
        cached = cached_edge_coeffs.get(enriched)
        if cached is not None:
            edges_out.append((source_id, target_id, cached, r))
            continue

        # Stored embedding (schema v26+): skip fastembed entirely. Populated
        # ASYNC by the embed_queue worker (Brain.backfill_edge_embeddings) after
        # a write invalidates+enqueues — see brain_connections.py.
        eid = r.get('edge_id')
        rel = r.get('relation') or ''
        blob = stored_embeddings.get((eid, rel)) if eid and rel else None
        if blob is not None:
            vec = np.frombuffer(blob, dtype=np.float32)
            coeff = _cosine_nonneg(query_vec, vec, norm_a=norm_q)
            cached_edge_coeffs[enriched] = coeff
            edges_out.append((source_id, target_id, coeff, r))
            continue

        # Fallback: row predates v26 backfill or write failed to populate.
        # Queue for the legacy on-demand embed path. After
        # `scripts/backfill_edge_embeddings.py` runs against the brain,
        # this branch should be cold.
        enriched_to_embed.append(enriched)
        enriched_keys.append((source_id, target_id, r, enriched))

    # Batch-embed queued enriched texts (hits _desc_vecs_batched cache).
    # On a fully-backfilled brain this list is empty and the fastembed
    # call is skipped entirely.
    if enriched_to_embed:
        blobs = _desc_vecs_batched(enriched_to_embed)
        for (src, tgt, edge, text), blob in zip(enriched_keys, blobs):
            if blob is None:
                coeff = 0.0
            else:
                vec = np.frombuffer(blob, dtype=np.float32)
                coeff = _cosine_nonneg(query_vec, vec, norm_a=norm_q)
            cached_edge_coeffs[text] = coeff
            edges_out.append((src, tgt, coeff, edge))

    return edges_out


def spread_activation(seed_ids, query_vec, brain, prior_vecs=None):
    """Spreading activation from seed nodes through the graph.

    One mechanism does what select_edges + _graph_expand + compute_shared_context
    + fatigue were doing separately:
      • Activation originates at each seed from its max field-cosine with query
      • It flows through edges, weighted by the edge's own enriched-text cosine
      • Nodes receiving activation from multiple paths accumulate (tanh-saturated)
      • Depth is emergent — a strong-matching edge propagates far, a weak one
        stops on hop 1. Mutual traversal is automatic: two seeds whose paths
        meet at a neighbor boost that neighbor above singleton reach.

    Args:
        seed_ids: List of full node IDs to start activated (typically the set
            Haiku selected from the 25 candidates).
        query_vec: numpy array (768d) — current query embedding.
        brain: Brain instance — needed for interaction_config (family map)
            and conn access.
        prior_vecs: Optional list of prior-turn query embeddings for
            multi-turn blending (dampens one-word query ambiguity).

    Returns:
        dict with:
          'node_activation':  {node_id: float in [0,1]}
          'field_activation': {node_id: {field_name: float in [0,1]}}
          'trace': list of per-step {step, new_nodes, edges_considered,
                                     edges_transmitted, max_act}
    """
    if not seed_ids or query_vec is None:
        return {'node_activation': {}, 'field_activation': {}, 'trace': []}

    # ── Multi-turn query blend (existing pattern, kept) ──
    if prior_vecs:
        ws = TURN_WEIGHTS[:len([query_vec] + prior_vecs)]
        total = sum(ws)
        ws = [w / total for w in ws]
        blended = sum(w * v for w, v in zip(ws, [query_vec] + prior_vecs))
        nm = np.linalg.norm(blended)
        if nm > 0:
            blended = blended / nm
    else:
        blended = query_vec
    norm_q = float(np.linalg.norm(blended))

    # ── Lineage relations (union across the structural-lineage aspects) ──
    # brain.aspects.lineage_relations — derived at registry load from the
    # per-aspect `structural_lineage` fact in aspects_v1.json. Consulted only
    # under the 'lineage' recall variant (_LINEAGE_PASS, below): these edges
    # bypass the per-hop median gate. It is a UNION across the lineage
    # aspects, so a verb in several aspects rides along if ANY of them
    # declares structural_lineage.
    lineage_relations = frozenset()
    try:
        lineage_relations = brain.aspects.lineage_relations
    except Exception as _e:
        # Optional, but a registry read-failure is worth surfacing — without
        # it lineage ride-along silently degrades to "nothing rides".
        brain._log_error('spread_aspect_config', _e,
                         'loading lineage relations in spread_activation')

    # ── Step 0: seeds' own activations (per-field cosines vs query) ──
    all_touched_ids = list(seed_ids)
    node_vectors = _batch_load_field_vectors(brain.conn, all_touched_ids)
    # Silent-failure guard: if we got NO vectors for a seed, the kernel can't
    # produce activation for it. This happened at N=15 once (seed had no
    # vectors at all), which is the failure mode worth surfacing.
    seeds_missing_vectors = [sid for sid in seed_ids if sid not in node_vectors]
    if seeds_missing_vectors:
        brain._log_error('spread_seed_no_vectors',
                         RuntimeError('seeds without any vectors'),
                         'seeds=%s (kernel will produce zero activation for these)' %
                         ','.join(s[:8] for s in seeds_missing_vectors[:5]))
    node_activation = {}
    field_activation = {}

    for nid in seed_ids:
        vecs = node_vectors.get(nid, {})
        field_cos = _field_cosines_for_node(blended, vecs, norm_q=norm_q)
        field_activation[nid] = field_cos
        # MAX over signals, per TRIZ — best discriminating field wins
        node_activation[nid] = max(field_cos.values()) if field_cos else 0.0

    # ── Spread loop ──
    trace_steps = []
    cached_edge_coeffs = {}  # memo across hops: enriched_text → coeff

    # Variant 'thickness' (eval): multiply cosine by accumulated edge weight
    # before any gating. Edges accumulate weight via explicit Hebbian
    # strengthening (servers/dal.py:GraphDAL.strengthen_relation — bumps
    # weight by LEARNING_RATE × 0.5 per call, capped at MAX_WEIGHT).
    # Called by daemon_hooks for co_accessed edges per surface event.
    # Today that signal is unused in spread; this variant uses it.
    import os as _os
    _THICKNESS = 'thickness' in _os.environ.get('BRAIN_RECALL_VARIANT', '').lower()

    # Hop scrutiny: at hop 3+ (step >= 2), only the top half of currently-
    # active nodes by activation propagate further. Distribution-derived —
    # the threshold is the median of the current activation set, not a
    # static number. Default from contract (HOP_SCRUTINY_DEFAULT = True).
    # Opt-out via BRAIN_SPREAD_HOP_SCRUTINY=off for A/B comparisons.
    _scrutiny_env = _os.environ.get('BRAIN_SPREAD_HOP_SCRUTINY')
    if _scrutiny_env is None:
        _HOP_SCRUTINY = HOP_SCRUTINY_DEFAULT
    else:
        _HOP_SCRUTINY = _scrutiny_env.lower() == 'on'

    for step in range(_SPREAD_MAX_STEPS):
        # Source nodes at this hop: whichever are currently activated above zero
        raw_active = [(n, a) for n, a in node_activation.items() if a > 0]
        if not raw_active:
            break

        # Per-hop scrutiny on the jump to hop 3+. The median of current
        # activations becomes the floor — only above-median sources
        # propagate further. The cost of going deeper is that you only
        # follow strong activations, not all weak ones.
        if _HOP_SCRUTINY and step >= 2 and len(raw_active) > 4:
            sorted_acts = sorted([a for _, a in raw_active], reverse=True)
            scrutiny_floor = sorted_acts[len(sorted_acts) // 2]
            active_sources = [n for n, a in raw_active if a >= scrutiny_floor]
        else:
            active_sources = [n for n, _ in raw_active]

        edges = _build_edge_coeffs(
            brain, active_sources, blended,
            cached_edge_coeffs)

        if not edges:
            break

        # Apply thickness: cosine × edge_weight. A confirmed edge (weight=1.0)
        # transmits at full cosine; a single-mention edge (weight=0.5)
        # transmits at half. Weakened/contradicted edges (weight<0.3) drop out
        # of the gate naturally without needing a separate suppression rule.
        if _THICKNESS:
            edges = [
                (s, t, c * float(e.get('weight') or 0.5), e)
                for (s, t, c, e) in edges
            ]

        # Outer gate: any meaningfully-matching edges at all?
        max_coeff = max(e[2] for e in edges)
        if max_coeff < _SPREAD_NOISE_FLOOR:
            trace_steps.append({'step': step, 'new_nodes': 0,
                               'edges_considered': len(edges),
                               'edges_transmitted': 0,
                               'max_act': max(node_activation.values())
                                   if node_activation else 0.0,
                               'halted': 'below_noise_floor'})
            break

        # Per-hop self-calibrating threshold: median of positive coeffs.
        # Ensures only better-than-typical edges in THIS batch transmit.
        threshold = float(np.percentile([e[2] for e in edges], 50))

        # Variant 'lineage' (eval): lineage-family edges bypass the median
        # gate. The relation type itself encodes structural meaning the
        # enriched-text cosine can't capture (corrects, extends, supersedes).
        # This is opt-in via env var so production behavior is unchanged.
        import os as _os
        _LINEAGE_PASS = 'lineage' in _os.environ.get(
            'BRAIN_RECALL_VARIANT', '').lower()

        # Accumulate contributions for each target
        contributions = {}
        transmitted_count = 0
        for src, tgt, coeff, _edge in edges:
            if coeff < threshold:
                if _LINEAGE_PASS:
                    relation = (_edge.get('relation') or '').strip()
                    if relation not in lineage_relations:
                        continue
                    # Else fall through — lineage edge bypasses the gate
                else:
                    continue
            transmitted_count += 1
            source_act = node_activation.get(src, 0)
            # Target hasn't been computed yet? Skip source-act lookup failures.
            transferred = source_act * coeff
            contributions[tgt] = contributions.get(tgt, 0.0) + transferred

        if not contributions:
            trace_steps.append({'step': step, 'new_nodes': 0,
                               'edges_considered': len(edges),
                               'edges_transmitted': 0,
                               'max_act': max(node_activation.values()),
                               'halted': 'all_below_threshold'})
            break

        # Load field vectors for any newly-touched targets (one round-trip)
        new_target_ids = [t for t in contributions.keys() if t not in node_activation]
        if new_target_ids:
            fresh_vecs = _batch_load_field_vectors(brain.conn, new_target_ids)
            node_vectors.update(fresh_vecs)
            all_touched_ids.extend(new_target_ids)

        # Apply saturation — tanh blends old activation + incoming contribution.
        # Also seed target's own field-cosines (against query) so rendering has
        # field-level signal for newly-activated targets.
        pre_activation = dict(node_activation)
        new_nodes_added = 0
        for target, incoming in contributions.items():
            prior = node_activation.get(target, 0.0)
            saturated = float(np.tanh(prior + incoming))
            node_activation[target] = saturated
            if target not in field_activation:
                new_nodes_added += 1
                vecs = node_vectors.get(target, {})
                field_activation[target] = _field_cosines_for_node(
                    blended, vecs, norm_q=norm_q)

        trace_steps.append({'step': step, 'new_nodes': new_nodes_added,
                           'edges_considered': len(edges),
                           'edges_transmitted': transmitted_count,
                           'max_act': max(node_activation.values()),
                           'threshold': threshold})

        # No-change check: if nothing moved meaningfully, stop early
        delta = max((node_activation[n] - pre_activation.get(n, 0.0)
                    for n in node_activation), default=0.0)
        if delta < 0.01:
            break

    return {
        'node_activation': node_activation,
        'field_activation': field_activation,
        'trace': trace_steps,
    }


# ═══════════════════════════════════════════════════════════════
# SPREAD_ACTIVATION_CLUSTER — variant under eval (2026-04-30)
#
# Three principled changes from spread_activation, each derived from a
# distribution or taxonomy already present in the brain:
#
#  1. Distribution-derived per-hop gate. Replace the median-threshold
#     ("always 50% pass") and earlier draft's hardcoded fractions with a
#     z-score gate within this hop's edge coefficients: pass iff
#     coeff > μ_hop + k_hop·σ_hop. k narrows per hop (more strict deeper).
#     If the distribution is tight (everything similar), the gate
#     naturally lets more through; if wide (clear winners), it cuts hard.
#     The cut is the derivative of the situation, per c2730676.
#
#  2. Aspect-aware ride-along. Instead of a hardcoded relation allowlist,
#     read brain.aspects (the AspectRegistry) and treat edges whose aspect
#     declares structural_lineage — they ride along even when
#     their enriched-text cosine is weak. New aspects emerging via
#     AspectIntegration inherit the behavior automatically (when they map
#     to one of the lineage names). Floor for lineage transmission is the
#     per-hop 25th percentile of edge coefficients — distribution-derived,
#     not 0.4.
#
#  3. Convergence as a tag, not a multiplier. When a target is reached by
#     ≥2 distinct sources, mark it convergent and increment a per-node
#     convergence count. Activation stays scalar (sum + tanh as before).
#     The render layer can read the convergence map and prioritize cluster
#     boundaries — that's qualitative information, not amplitude.
#
# Same return shape as spread_activation + an additional 'convergence'
# field. Render layer can ignore it for compat.
# ═══════════════════════════════════════════════════════════════

# Structural-lineage ride-along: aspects whose semantic role is lineage
# rather than topical-similarity declare `structural_lineage: true` in
# aspects_v1.json — the first-class flag on the aspect itself, which is the
# drift-proof end state (an earlier hardcoded name list here held five
# aspect names that had stopped existing, so lineage silently stopped
# riding until 2026-06-08). Edges in these aspects ride along even with
# weak enriched-text cosine, because the relation type itself carries the
# meaning, not the description embedding.
#
# brain.aspects.lineage_relations is the UNION of relations across the
# flagged aspects, derived once at registry load — a verb that belongs to
# several aspects (e.g. `revises` ∈ correction_improvement AND
# temporal_sequence) rides as lineage as long as ANY of its aspects carries
# the flag. New verbs AspectIntegration adds to flagged aspects inherit
# the behavior automatically.
#
# The seed flags four aspects: corrections (correction_improvement);
# extension / refinement / evolution (extension_refinement); composition +
# versioning / supersedes (hierarchical_structure); dependency /
# prerequisite (dependency_flow). Kept narrow on purpose —
# temporal_sequence is unflagged so generic ordering (before/after/during)
# doesn't turn this into "everything rides."

# Per-hop z-gate strictness. k controls how many σ above hop mean an edge
# must be to transmit. Larger k = stricter cut = narrower spread. Hop 0
# casts wide (k=0, anything above mean), hop 1 narrows (k=0.5), hop 2
# only outliers (k=1.0). This is the only place a "tuning constant"
# survives — and it's a strictness schedule, not a static threshold.
# The actual cut value is computed from the hop's distribution.
_CLUSTER_K_SCHEDULE = (0.0, 0.5, 1.0)

# Safety cap on hops. Real termination comes from "no convergence + no
# new high-activation" check inside the loop.
_CLUSTER_MAX_STEPS = 3


def spread_activation_cluster(seed_ids, query_vec, brain, prior_vecs=None):
    """Cluster-completion variant of spread_activation.

    See module-level comment above for the design rationale. Returns the
    same shape as spread_activation plus a 'convergence' map that the
    render layer can use to prioritize cluster-boundary nodes.
    """
    if not seed_ids or query_vec is None:
        return {'node_activation': {}, 'field_activation': {},
                'convergence': {}, 'trace': []}

    # Multi-turn query blend (same pattern as the original kernel)
    if prior_vecs:
        ws = TURN_WEIGHTS[:len([query_vec] + prior_vecs)]
        total = sum(ws)
        ws = [w / total for w in ws]
        blended = sum(w * v for w, v in zip(ws, [query_vec] + prior_vecs))
        nm = np.linalg.norm(blended)
        if nm > 0:
            blended = blended / nm
    else:
        blended = query_vec
    norm_q = float(np.linalg.norm(blended))

    # Aspect map — single source of truth via brain.aspects.
    # rel_to_family (single-valued, last-writer-wins) labels each edge with a
    # family for convergence tagging. lineage_relations is the UNION of
    # relations across the structural-lineage aspects — used for the
    # lineage/semantic split so a multi-aspect verb (e.g. `revises`) still
    # rides as lineage even when its single family label is non-lineage.
    rel_to_family = {}
    lineage_relations = frozenset()
    try:
        for name, aspect in brain.aspects.all().items():
            for r in aspect.edge_relations:
                rel_to_family[r] = name
        lineage_relations = brain.aspects.lineage_relations
    except Exception as _e:
        brain._log_error('cluster_spread_aspect_config', _e,
                         'loading brain.aspects in spread_activation_cluster')

    # Seed activation (same as original)
    all_touched_ids = list(seed_ids)
    node_vectors = _batch_load_field_vectors(brain.conn, all_touched_ids)
    seeds_missing_vectors = [sid for sid in seed_ids if sid not in node_vectors]
    if seeds_missing_vectors:
        brain._log_error('cluster_seed_no_vectors',
                         RuntimeError('seeds without any vectors'),
                         'seeds=%s' % ','.join(s[:8] for s in seeds_missing_vectors[:5]))
    node_activation = {}
    field_activation = {}
    for nid in seed_ids:
        vecs = node_vectors.get(nid, {})
        field_cos = _field_cosines_for_node(blended, vecs, norm_q=norm_q)
        field_activation[nid] = field_cos
        node_activation[nid] = max(field_cos.values()) if field_cos else 0.0

    # Spread loop with distribution-gated narrowing + family-aware lineage
    # + convergence tagging
    trace_steps = []
    cached_edge_coeffs = {}
    convergence_count = {}  # node_id → number of distinct sources that reached it

    for step in range(_CLUSTER_MAX_STEPS):
        active_sources = [n for n, a in node_activation.items() if a > 0]
        if not active_sources:
            break

        edges = _build_edge_coeffs(
            brain, active_sources, blended,
            cached_edge_coeffs)

        if not edges:
            break

        # Classify: lineage = ride-along by family; semantic = subject to
        # distribution-derived gate.
        lineage = []
        semantic = []
        for src, tgt, coeff, edge in edges:
            relation = (edge.get('relation') or '').strip()
            family = rel_to_family.get(relation, '')
            if relation in lineage_relations:
                lineage.append((src, tgt, coeff, edge, family))
            else:
                semantic.append((src, tgt, coeff, edge, family))

        # Distribution-derived gate on semantic edges. Compute mean+std
        # from this hop's coefficients and require coeff > μ + k·σ.
        # If σ is tiny (everything similar), only just-above-mean pass —
        # which is fine, those are the cluster's coherent fringe. If σ is
        # wide (mixed quality), the cut is harder.
        if semantic:
            sem_coeffs = np.array([c for _, _, c, _, _ in semantic])
            mu = float(np.mean(sem_coeffs))
            sigma = float(np.std(sem_coeffs))
            k = _CLUSTER_K_SCHEDULE[min(step, len(_CLUSTER_K_SCHEDULE) - 1)]
            cut = mu + k * sigma
            transmitting_sem = [(s, t, c, e, f) for s, t, c, e, f in semantic
                                if c >= cut]
        else:
            mu = sigma = 0.0
            transmitting_sem = []

        # Distribution-derived floor for lineage: per-hop p25 of all edge
        # coefficients (semantic + lineage). Lineage with raw coeff below
        # this floor transmits AT the floor — the family carries the
        # meaning. With no semantic edges, lineage transmits at its own
        # coeff or a nominal small value, whichever is greater.
        all_coeffs = [e[2] for e in edges]
        floor = float(np.percentile(all_coeffs, 25)) if all_coeffs else 0.0
        transmitting_lin = [
            (s, t, max(c, floor), e, f)
            for s, t, c, e, f in lineage
        ]

        # Outer halt: if neither path will transmit anything meaningful,
        # the brain's allowed to be quiet.
        if not transmitting_sem and not transmitting_lin:
            trace_steps.append({'step': step, 'new_nodes': 0,
                               'edges_considered': len(edges),
                               'edges_transmitted': 0,
                               'mu': round(mu, 3), 'sigma': round(sigma, 3),
                               'cut': round(mu + (_CLUSTER_K_SCHEDULE[min(step, len(_CLUSTER_K_SCHEDULE)-1)] * sigma), 3),
                               'max_act': max(node_activation.values())
                                   if node_activation else 0.0,
                               'halted': 'gate_closed'})
            break

        transmitting = transmitting_sem + transmitting_lin

        # Accumulate contributions per target. Track distinct sources for
        # convergence tagging (NOT amplification).
        contributions = {}
        sources_per_target = {}
        families_per_target = {}
        for src, tgt, coeff, _edge, family in transmitting:
            source_act = node_activation.get(src, 0)
            transferred = source_act * coeff
            contributions[tgt] = contributions.get(tgt, 0.0) + transferred
            sources_per_target.setdefault(tgt, set()).add(src)
            if family:
                families_per_target.setdefault(tgt, set()).add(family)

        # Convergence: count distinct sources reaching each target. Tag,
        # don't amplify — render layer reads this map.
        for tgt, srcs in sources_per_target.items():
            if len(srcs) >= 2:
                convergence_count[tgt] = convergence_count.get(tgt, 0) + len(srcs)

        # Load field vectors for new targets (one round-trip)
        new_target_ids = [t for t in contributions.keys() if t not in node_activation]
        if new_target_ids:
            fresh_vecs = _batch_load_field_vectors(brain.conn, new_target_ids)
            node_vectors.update(fresh_vecs)
            all_touched_ids.extend(new_target_ids)

        # Apply with tanh saturation; seed targets' field activations
        pre_activation = dict(node_activation)
        new_nodes_added = 0
        for target, incoming in contributions.items():
            prior = node_activation.get(target, 0.0)
            saturated = float(np.tanh(prior + incoming))
            node_activation[target] = saturated
            if target not in field_activation:
                new_nodes_added += 1
                vecs = node_vectors.get(target, {})
                field_activation[target] = _field_cosines_for_node(
                    blended, vecs, norm_q=norm_q)

        n_converged = sum(1 for s in sources_per_target.values() if len(s) >= 2)
        trace_steps.append({'step': step,
                           'new_nodes': new_nodes_added,
                           'edges_considered': len(edges),
                           'edges_transmitted': len(transmitting),
                           'edges_lineage': len(transmitting_lin),
                           'edges_semantic': len(transmitting_sem),
                           'mu': round(mu, 3), 'sigma': round(sigma, 3),
                           'cut': round(mu + (_CLUSTER_K_SCHEDULE[min(step, len(_CLUSTER_K_SCHEDULE)-1)] * sigma), 3),
                           'floor_p25': round(floor, 3),
                           'targets_converged': n_converged,
                           'max_act': max(node_activation.values())})

        # Convergence-stop: cluster boundary reached when no target this
        # hop saw ≥2 sources AND no new high-activation node added.
        any_new_high = any(node_activation[t] > 0.3 and t not in pre_activation
                           for t in contributions)
        if n_converged == 0 and not any_new_high and step > 0:
            break

    return {
        'node_activation': node_activation,
        'field_activation': field_activation,
        'convergence': convergence_count,
        'trace': trace_steps,
    }


# ═══════════════════════════════════════════════════════════════
# EDGE SELECTION — per-node edge ranking for DISPLAY
#
# Used when a single activated node is being rendered and we need to pick
# which of its edges to show (typically 3). Scoring is the same MAX formula
# as the spreading-activation kernel uses for transmission — one rule at
# two granularities (node-level and edge-level).
#
#   edge_score = max(
#       cos(query, target_node_embedding),
#       cos(query, enriched_edge_text)
#   )
#
# No blend weights. No family_boost term (family meaning is inlined into
# the enriched text, so it rides inside the second signal). No fatigue
# discount (novelty is handled at the activation-set level, not per edge).
# No weight tiebreaker (edge weights are static ~0.6 across the graph
# today — they're not signal yet).
#
# Multi-turn query blending is still applied: when a prior turn's query
# vector is available, the effective query is a weighted average that
# smooths single-word follow-ups.
# ═══════════════════════════════════════════════════════════════

# Multi-turn query blending (current turn, previous, two-back)
TURN_WEIGHTS = [0.6, 0.3, 0.1]


# `_compose_enriched_edge_text` removed 2026-05-09 — the format lives in
# AspectRegistry.compose_edge_text (see servers/aspects.py). Keeping two
# implementations of the same string format created drift risk; the
# AspectRegistry version owns it. `_build_edge_coeffs` and `select_edges`
# call `brain.aspects.compose_edge_text(relation, description)` directly.


def _cosine_nonneg(a, b, norm_a=None):
    """Non-negative cosine. Computes norm_b on the fly; caller may pre-pass
    norm_a since it's invariant across edges in a single select_edges call."""
    import numpy as np
    if a is None or b is None or len(a) != len(b):
        return 0.0
    if norm_a is None:
        norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, float(np.dot(a, b)) / (norm_a * norm_b))


def select_edges(connections, query_vec, limit=3, prior_vecs=None,
                 brain_conn=None, brain=None):
    """Pick the top-N edges of a single node for display rendering.

    Used inside the activation renderer when laying out an activated node —
    we have space for ~3 edges under the node header, and this function
    picks the 3 that best fit the query.

    Scoring: max(cos(query, target_node_embedding), cos(query, enriched_edge_text)).
    Family meaning is composed into the enriched text via
    AspectRegistry.compose_edge_text (see aspects.py), so it contributes
    automatically to the second term without a separate signal. After
    schema v26, the edge text vector is read from edge_relations.embedding;
    only rows with NULL embedding fall through to live compose + fastembed.

    Args:
        connections: list of edge dicts (from get_rich_node().connections OR
            from GraphDAL.get_neighbors()). Required keys: id, relation;
            useful keys: title/target_title, description.
        query_vec: numpy array (768d) — current query embedding.
        limit: max edges to return.
        prior_vecs: prior-turn query embeddings for multi-turn blending.
        brain_conn: sqlite3 connection — used to load stored target node
            embeddings. Without it, node_rel falls back to 0 and scoring
            leans on enriched_edge only.
        brain: Brain instance — used to load `s2_edge_families` config so
            family meaning can be composed into enriched edge text.

    Returns:
        list of edge dicts (top N by score), same shape as input.
        On empty input or missing query_vec, returns the input unchanged
        (truncated to limit) — no graceful reordering.
    """
    import numpy as np

    if not connections or query_vec is None:
        return connections[:limit]

    # `brain` is required: fallback path calls brain.aspects.compose_edge_text
    # when a stored edge embedding is missing. Production callers
    # (daemon_hooks.py, format_surface_output_activation, frame_replay)
    # always pass brain. The silent fallback that produced a parts-list
    # without family meaning was removed 2026-05-09 — it produced a
    # different text than the AspectRegistry composer, so the embedding
    # geometry would silently drift. Loud-by-default: crash if missing.
    if brain is None:
        raise ValueError(
            "select_edges requires a Brain instance — the fallback edge-text "
            "composer used to silently drop family meaning, producing a "
            "different vector than the AspectRegistry composer. If you're "
            "calling this from a test or eval script, pass the same brain you "
            "loaded the connections from.")

    # Multi-turn blend — fold prior queries into effective query vector
    if prior_vecs:
        ws = TURN_WEIGHTS[:len([query_vec] + prior_vecs)]
        total = sum(ws)
        ws = [w / total for w in ws]
        blended = sum(w * v for w, v in zip(ws, [query_vec] + prior_vecs))
        nm = np.linalg.norm(blended)
        if nm > 0:
            blended = blended / nm
    else:
        blended = query_vec
    norm_q = float(np.linalg.norm(blended))

    # Family meaning composition is owned by AspectRegistry — see
    # aspects.py:compose_edge_text. The fallback path below calls
    # brain.aspects.compose_edge_text(rel, desc) directly when a stored
    # edge embedding is missing; no per-call dict-building needed.

    # Batch-load target node embeddings (one SQL round-trip).
    # Connections carry full UUIDs, so use IN with full IDs — indexed
    # lookup. Previous version used `LIKE prefix% OR LIKE prefix% ...`
    # which couldn't use the node_enrichments index efficiently
    # (2026-05-09 candidates phase 4-8s, removed). Dedupe so two
    # connections to the same target only consume one SQL parameter.
    full_target_ids = list({c.get('id', '') for c in connections
                            if c.get('id')})
    stored_embeddings = {}
    if brain_conn is not None and full_target_ids:
        try:
            ph = ','.join(['?'] * len(full_target_ids))
            rows = brain_conn.execute(
                "SELECT node_id, embedding FROM node_enrichments "
                "WHERE vector_type = '_primary' AND node_id IN (%s)" % ph,
                full_target_ids).fetchall()
            for full_id, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                # Index by both 8-char prefix (legacy) and full id so the
                # scoring loop below works either way.
                stored_embeddings[full_id[:8]] = vec
                stored_embeddings[full_id] = vec
        except Exception as _e:
            # node_enrichments query failure: log and proceed without
            # node-cosine signal. Scoring falls back to enriched-edge
            # cosine alone. Symmetric with the edge_relations fetch
            # below — both are best-effort, neither breaks recall.
            try:
                brain._log_error(
                    'select_edges_node_embed', _e,
                    'bulk fetch of target node embeddings — '
                    'falling back to enriched-edge-only scoring')
            except Exception:
                pass

    # Bulk-load stored edge embeddings (schema v26+) via the shared
    # helper. See _bulk_load_stored_edge_embeddings docstring.
    edge_keys = [(c.get('edge_id'), c.get('relation') or '')
                 for c in connections]
    stored_edge_embeddings = _bulk_load_stored_edge_embeddings(
        brain_conn, edge_keys, brain=brain,
        error_label='edge_embedding_read_select_edges')

    # Edges WITHOUT a stored embedding fall through to live compose +
    # fastembed (legacy path). Build the enriched text only for those.
    # After `scripts/backfill_edge_embeddings.py` runs against the brain,
    # this loop's `else` branch is cold.
    enriched_blobs = [None] * len(connections)
    needs_embed_idx = []
    needs_embed_text = []
    for i, c in enumerate(connections):
        key = (c.get('edge_id'), c.get('relation') or '')
        stored = stored_edge_embeddings.get(key)
        if stored is not None:
            enriched_blobs[i] = stored
        else:
            # Live compose for the rare missing-blob case (predates
            # backfill, or write-path skipped this edge). Connections
            # come from get_neighbors (key 'edge_description') OR
            # get_rich_node (key 'description'); accept either.
            rel = c.get('relation', '') or ''
            desc = (c.get('description') or
                    c.get('edge_description') or '')
            text = brain.aspects.compose_edge_text(rel, desc)
            needs_embed_idx.append(i)
            needs_embed_text.append(text)
    if needs_embed_text:
        live_blobs = _desc_vecs_batched(needs_embed_text)
        for idx, blob in zip(needs_embed_idx, live_blobs):
            enriched_blobs[idx] = blob

    # Score each edge: MAX of node-target and enriched-edge cosines
    scored = []
    for c, enriched_blob in zip(connections, enriched_blobs):
        tid = c.get('id', '')[:8]

        target_vec = stored_embeddings.get(tid)
        node_rel = (_cosine_nonneg(blended, target_vec, norm_a=norm_q)
                    if target_vec is not None else 0.0)

        if enriched_blob is not None:
            enriched_vec = np.frombuffer(enriched_blob, dtype=np.float32)
            enriched_rel = _cosine_nonneg(blended, enriched_vec, norm_a=norm_q)
        else:
            enriched_rel = 0.0

        scored.append((max(node_rel, enriched_rel), c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:limit]]


# ═══════════════════════════════════════════════════════════════
# ACTIVATION-DRIVEN RENDERING
#
# Once spread_activation has produced {node_activation, field_activation},
# format_surface_output_activation renders nodes by activation rank, with
# each node's token budget proportional to its activation and each field
# appearing only when its per-field activation clears a minimum threshold.
#
# No SURFACE_FORMAT whitelist. No hardcoded "show content but not reasoning."
# Fields with high activation surface; fields with low activation don't.
# Tom's framing: "fading means less and less data surfaced."
# ═══════════════════════════════════════════════════════════════

# Minimum per-node budget — below this, we stop rendering further nodes
# rather than emit a stub that can't carry meaning.
_MIN_NODE_BUDGET_CHARS = 150

# Hard byte cap on the inject. Claude Code spills additionalContext to a
# file above ~10k chars, and Anchor doesn't read that file path back. Cap
# below the ceiling with headroom for any wrapper bytes Claude Code adds.
_MAX_INJECT_CHARS = 9500


def _allocate_budget_softmax(activations, total_budget):
    """Softmax-weighted budget per node. High-activation nodes get more share;
    saturated-at-1.0 nodes split evenly; weak nodes still get minimum viable.
    Returns list of int budgets aligned with activations input.
    """
    if not activations:
        return []
    arr = np.array(activations, dtype=np.float64)
    # Subtract max for numerical stability, then softmax
    exps = np.exp(arr - arr.max())
    weights = exps / exps.sum()
    return [max(_MIN_NODE_BUDGET_CHARS, int(w * total_budget)) for w in weights]


def _event_time_line(node):
    """If node has event_time in kv metadata, return a structured render line
    exposing it (absolute + relative). Added 2026-05-11 to address v15.8
    L4 bottleneck: encoder writes event_time kv but answerer never sees the
    date as a structured field — it lives buried in content prose.

    Returns '' when no event_time is present, so callers can unconditionally
    prepend without guarding.

    Future generalization (per Tom): query-aware kv field promotion. A query
    asking 'what did X say' should promote user_raw_quote / anchor_raw_quote
    similarly; 'when' queries promote event_time/created_at. Current scope:
    event_time only — surgical change to test L4-fix in isolation.
    """
    kv = (node.get('metadata_kv') or node.get('kv') or {})
    if not isinstance(kv, dict):
        return ''
    et = kv.get('event_time') or kv.get('event_date')
    if not et:
        # Also peek into _metadata where some render paths copy kv
        m = node.get('_metadata') or {}
        if isinstance(m, dict):
            et = m.get('event_time') or m.get('event_date')
    if not et:
        return ''
    et_str = str(et).strip()
    if not et_str:
        return ''
    # Absolute date only — no relative-to-wall-clock gloss.
    # The relative would anchor to brain_now() (real today) which is the
    # WRONG reference frame for the answerer: in eval the conversation's
    # question_date is its "now"; in production it's wall-clock and the
    # answerer can derive the delta from created_at/updated_at if needed.
    # Adding "X mo ago" relative-to-wall-clock is noise, not signal.
    return '  Event date: %s' % et_str


def _render_node_activation(node, budget, activation,
                             query_vec=None, brain=None, mode='arc',
                             seen_root_ids=None, scope=None):
    """Render a single activated node within a char budget.

    Mode controls layout depth; the encoder's attached fields are trusted
    in every mode (no cosine masking — the encoder picked these fields for
    a reason, the renderer doesn't second-guess).

      - 'arc' (default): full render via SURFACE_ARC_FORMAT; edges
        re-ranked by query relevance via select_edges. State-of-mind /
        identity nodes.
      - 'fact': verbatim content, larger budget; SURFACE_FACT_FORMAT.
        Specific values / quotes / exact wording.
      - 'background': title + 1-line situation only; SURFACE_BACKGROUND_FORMAT.
        Low-weight framing context.

    Common behavior:
      • Budget scales content / metadata / edge limits proportionally.
      • Structured event_time kv (when present) is rendered as a dedicated
        line via _event_time_line, just after the title.
    """
    from servers.contract import render_rich_node
    event_line = _event_time_line(node)

    def _inject_event_line(body):
        """Place the structured event_time line right after the title."""
        if not event_line:
            return body
        body_lines = body.split('\n', 1)
        if len(body_lines) == 2:
            return body_lines[0] + '\n' + event_line + '\n' + body_lines[1]
        return body + '\n' + event_line

    if mode == 'background':
        # Background — title + 1-line situation only. No render_rich_node
        # round-trip (output is too minimal to benefit). Reads
        # SURFACE_BACKGROUND_FORMAT for the situation char cap.
        title = node.get('title', '')
        kv = node.get('metadata_kv') or node.get('kv') or {}
        sit_max = SURFACE_BACKGROUND_FORMAT['situation_max_chars']
        situation = (kv.get('situation') or '')[:sit_max]
        body_lines = ['[%s] %s' % (node.get('type', '?'), title)]
        if event_line:
            body_lines.append(event_line)
        # Differential scope marks even at minimal depth — a foreign node
        # is foreign regardless of render budget (mark, don't hide).
        if scope:
            from servers.contract import scope_marks
            body_lines.extend(scope_marks(node, scope, meta=kv))
        if situation:
            body_lines.append('  ' + situation)
        return '\n'.join(body_lines)

    if mode == 'fact':
        cfg = resolve_surface_format(SURFACE_FACT_FORMAT, budget, scope=scope)
        return _inject_event_line(render_rich_node(node, cfg))

    # 'arc' (default) — full encoder-attached fields render; edges
    # re-ranked by query relevance so the most pertinent bridges appear
    # under the node header.
    arc_node = dict(node)
    connections = arc_node.get('connections') or []
    # Dedup edge-lines that point to a node already rendered as a root in this
    # inject: the neighbor is shown in full elsewhere, so the edge-line is a
    # redundant restatement that wastes budget. Conservative — seen_root_ids
    # only ever holds nodes rendered BEFORE this one (higher activation), so a
    # dropped edge always has its target shown above; an edge to a node that
    # never renders is kept (its only appearance).
    if seen_root_ids and connections:
        connections = [c for c in connections
                       if c.get('id') not in seen_root_ids]
        arc_node['connections'] = connections
    if query_vec is not None and connections:
        arc_node['connections'] = select_edges(
            connections, query_vec, limit=10,
            brain_conn=brain.conn if brain is not None else None,
            brain=brain)

    cfg = resolve_surface_format(SURFACE_ARC_FORMAT, budget, scope=scope)
    return _inject_event_line(render_rich_node(arc_node, cfg))


def format_surface_output_activation(node_activation, field_activation,
                                      rich_nodes, selected_mode=None,
                                      query_vec=None, brain=None,
                                      total_budget=7000,
                                      scope=None):
    """Render activated nodes as additionalContext.

    Args:
        node_activation:  {node_id: float} from spread_activation — drives
                          ranking + softmax budget weighting.
        field_activation: {node_id: {field_name: float}} from spread_activation —
                          used only for sort-tie-breaking (mean across fields).
                          Per-field masking removed 2026-05-17 — the renderer
                          trusts the encoder's attached fields.
        rich_nodes:       {node_id: rich_node_dict} from brain.get_node(ids)
        selected_mode:    {node_id: str} — the Haiku-selected seed ids (dict
                          keys) → per-seed render mode (fact/arc/background).
                          Omitted → 'arc'.
        query_vec:        query embedding, used to re-rank each node's edges
        brain:            Brain instance — for select_edges + overflow logging
        total_budget:     soft target for the total inject; per-node budgets
                          are softmax-allocated from this. Hard exit cap is
                          _MAX_INJECT_CHARS.
    """
    if not node_activation:
        return ""

    selected_mode = selected_mode or {}

    # Rank the inject. Two modes (BRAIN_SURFACE_RANK_MODE, default 'activation'):
    #   'activation' — (node_activation, mean_field_activation): the historical
    #                  default. node_activation saturates (tanh, :1154) so this
    #                  ranks by graph CONNECTIVITY once many nodes hit 1.0.
    #   'cosine'     — (mean_field_activation, node_activation): rank by honest
    #                  query relevance; activation only breaks ties. Spread still
    #                  expands the pool + allocates budget — we just stop letting
    #                  connectivity bury relevance at the inject point.
    # Inject-precision A/B (finding 29f0f385): 'activation' buried essentials at
    # rank ~36 / 81% noise; cosine-rank put them at ~4 / 55% noise. Flag-gated,
    # off by default — production byte-identical until flipped after the A/B.
    import os
    _cosine_rank = os.environ.get('BRAIN_SURFACE_RANK_MODE', 'activation').lower() == 'cosine'

    def sort_key(item):
        nid, act = item
        fa = field_activation.get(nid, {})
        mean_fa = (sum(fa.values()) / len(fa)) if fa else 0.0
        if _cosine_rank:
            return (mean_fa, act, nid)   # relevance primary, activation tiebreak
        return (act, mean_fa, nid)       # activation primary (historical default)

    ranked = sorted(node_activation.items(), key=sort_key, reverse=True)

    # Filter to nodes we have full rich data for
    ranked = [(nid, act) for nid, act in ranked if nid in rich_nodes]

    if not ranked:
        return ""

    # Softmax budget allocation
    acts = [a for _, a in ranked]
    budgets = _allocate_budget_softmax(acts, total_budget)

    lines = ['Brain activated %d memories:' % len(ranked), '']
    remaining = total_budget - len(lines[0])
    seen_root_ids = set()  # nodes already rendered as roots — dedup their edge-lines

    for (nid, activation), budget in zip(ranked, budgets):
        if remaining < _MIN_NODE_BUDGET_CHARS:
            break

        node = rich_nodes[nid]
        mode = selected_mode.get(nid, 'arc')

        # Fact-mode gets a 1.5× budget bump — ensures verbatim content
        # survives truncation when many candidates compete for budget.
        effective_budget = min(int(budget * 1.5) if mode == 'fact' else budget,
                                remaining)
        rendered = _render_node_activation(
            node, effective_budget, activation,
            query_vec=query_vec, brain=brain, mode=mode,
            seen_root_ids=seen_root_ids, scope=scope)

        lines.append(rendered)
        lines.append('')  # blank line between nodes
        remaining -= len(rendered) + 2
        seen_root_ids.add(nid)  # now a rendered root — later nodes dedup edges to it

    result = '\n'.join(lines)
    # Hard byte cap. Claude Code spills additionalContext to a file path
    # above ~10k chars, and Anchor doesn't read that path back — the inject
    # would be effectively lost. Truncate at a clean line boundary so the
    # tail isn't a half-rendered field.
    if len(result) > _MAX_INJECT_CHARS:
        if brain is not None:
            try:
                brain._log_error(
                    'surface_inject_overflow',
                    ValueError('inject %d > cap %d' % (len(result), _MAX_INJECT_CHARS)),
                    'ranked=%d primary=%d; truncated at byte cap' % (
                        len(ranked), len(selected_mode)))
            except Exception:
                pass
        result = result[:_MAX_INJECT_CHARS].rsplit('\n', 1)[0]
    return result
