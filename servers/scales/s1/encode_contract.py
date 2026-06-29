"""Encoding Contract — S1 turn encoder (Sonnet) config and catalog building.

The encoding agent reads conversation turns and creates/revises brain nodes.
This contract defines:
- What the encoder sees (ENCODING_AGENT config)
- How the node catalog is built (build_node_catalog)

Node formatting uses render_rich_node() from servers.contract.
Interaction: 's1e' in interactions table. Prompt is learnable.
"""

from servers.contract import render_rich_node

# How many conversational turns accumulate before the S1 Scribe fires. The gate
# is a LEVEL trigger on turns-since-last-encode (read live from traces), not a
# modular counter — a skipped run (lock busy) isn't lost; the next turn re-checks.
ENCODE_EVERY = 5

# turns-since-last-encode at/above this means the Scribe is WEDGED: it should have
# fired at ENCODE_EVERY but the backlog kept growing (lock jammed, or runs erroring
# before they write their encoding_prompt trace). The gate logs a loud error here —
# this is the monitor that would have caught the 20h encode-drought on hour one.
# 4× the cadence: well past normal, so a rapid burst mid-run can't trip it.
SCRIBE_STARVATION_TURNS = 4 * ENCODE_EVERY


def scribe_is_starved(turns_since: int) -> bool:
    """True when the Scribe is wedged and the gate should emit a loud signal.
    Level condition, rate-limited to one alert per ENCODE_EVERY turns of continued
    starvation (fires at 20, 25, 30… not every turn) so the error log isn't spammed."""
    return turns_since >= SCRIBE_STARVATION_TURNS and turns_since % ENCODE_EVERY == 0


# ── Idle-tail trigger (the second clause of the Scribe gate) ──
# A session that goes quiet below the ENCODE_EVERY threshold would otherwise
# never have its last turns encoded (the 5+ trigger never fires, and the Stop
# hook is gone). The poll-driven reactor catches that tail: once a session has
# been idle this long AND has more than SCRIBE_TAIL_MIN_TURNS unencoded turns,
# fire one final encode. An hour is long enough that it only ever fires on a
# genuinely abandoned session; the >2-turns guard skips trivial tails not worth
# a Sonnet call.
SCRIBE_TAIL_IDLE_SECONDS = 3600
SCRIBE_TAIL_MIN_TURNS = 2

# The 5+ clause fires only for a session whose last turn is within this window —
# i.e. one ACTIVELY conversing. Without it, the poll would `5+`-encode every
# recently-present session (90-min window) regardless of how long ago it went
# quiet — a backlog sweep, worst on restart. A session that goes quiet below the
# active window waits for the 1h idle tail instead (and re-qualifies for `5+`
# the moment it takes another turn). This is the "is this a live conversation?"
# bound the old Stop-hook trigger had implicitly (it only fired on a turn).
SCRIBE_ACTIVE_WINDOW_SECONDS = 600

# The reactor only sees sessions present within this wall-clock window, so it
# must outrun the tail threshold (1h) — otherwise a session that crossed 1h idle
# would age out before the tail could fire. Set to 5h so a tail survives a normal
# same-day work pause (lunch, a meeting, a short sleep) + a daemon restart, not
# just a bare margin. NOT longer: we stamp writes at now() (transaction-time, not
# the conversation's time), so resurrecting a genuinely-stale conversation would
# date its nodes "today" — the unbounded catch-up is bi-temporal work (deferred).
SCRIBE_CANDIDATE_WINDOW_MIN = 300   # 5 hours

# The reactor re-evaluates every few seconds. A session whose encode FAILS or
# SKIPS never advances the cadence, so it stays "due" — without a guard the poll
# would re-fire it every tick (a tight machine-paced retry the old human-paced
# Stop gate never had). The reactor records each attempt and won't re-fire the
# same session within this cooldown — bounding a failing session to one retry per
# interval, and giving the poll the cheap per-session gate it otherwise lacks. A
# SUCCESSFUL encode clears the entry immediately (the cadence reset already
# prevents a re-fire), so healthy sessions are never throttled.
SCRIBE_RETRY_COOLDOWN_SECONDS = 120

# Consecutive re-fires of the same session WITHOUT the cadence advancing means
# the encode is wedged (crashing before its encoding_prompt trace, or skipping).
# Escalate loudly at this count — the starvation alarm can't see this case (turns
# is frozen at a fixed value, so its `% ENCODE_EVERY` rate-limit never trips).
SCRIBE_MAX_FAILED_RETRIES = 3

# ═══════════════════════════════════════════════════════════════
# ENCODING AGENT CONFIG
# ═══════════════════════════════════════════════════════════════

# Encoding agent v3.2 (Sonnet) — split node catalog + timeline with references
ENCODING_AGENT = {
    'message_content_limit': 2500,    # per message stored in message_stream (both roles equally)
    'message_display_limit': 2500,    # per message in timeline (both roles — shared learnings, not just Tom's words)
    'max_messages': 20,               # last N messages (~10 turns)
    'recall_candidates_limit': 5,     # candidates per turn (pre-attached)
    'max_rounds': 5,                  # Sonnet API round limit (target: 2-3)
    'journal_max_chars': 8000,        # encoding journal truncation limit
    'max_d1': 3,                      # degree 1 neighbors shown
    'max_d2': 3,                      # degree 2
    'max_d3': 3,                      # degree 3
    'recall_on_create_limit': 5,      # max related_nodes returned per remember()
    'recall_on_create_content_limit': 500,  # chars of content per related node
    'recall_on_create_query_limit': 200,    # chars of content used in recall query
    'journal_entry_limit': 2000,      # max chars per journal entry
    'max_tokens': 12288,              # Sonnet API output cap (raised from 4096)
    'timeline_snippet_limit': 500,    # chars of recalled content shown in timeline (fallback only)
    'session_context_limit': 800,     # session context chars (additive within session, editable by S2)

    # Node catalog: full rich nodes shown once at top, referenced by ID in timeline
    'node_content_limit': None,       # full content — no truncation for encoder
    'node_edge_limit': 5,             # structural edges per node (with descriptions)
}

# Lived-sequence timeline (S1E code-half piece 1): how many recent s0 events to
# pull when assembling the messages+actions interleave. Bounded by EPISODE_MAX_LIMIT
# (=500) — recall_episodes/filter_events clamps anything larger — so 500 IS the max
# a single pull can return; the result is then trimmed to the control arm's turn count.
LIVED_SEQUENCE_PULL = 500


# ═══════════════════════════════════════════════════════════════
# NODE CATALOG — uses system format_node() with S1 config
# ═══════════════════════════════════════════════════════════════

# S1 encoder node config — full depth, no truncation.
# Correction render: 'heavy' — full corrector content + reasoning +
# user_raw_quote. The 2026-05-17 three-way A/B (lean vs balanced vs heavy)
# on 3 items showed heavy was actually CHEAPER than balanced on both
# encoder time (+8.1% vs lean, vs balanced's +11.4%) AND answerer tokens
# (+1.0% vs balanced's +17.5%) at equal pass rate. Correction context is
# targeted signal, not noise — heavy lets the encoder converge faster, not
# flood it. The locked principle (id:eaf833c5) "more signal ≠ better
# encoding" still holds for ARBITRARY signal flooding; correction-aspect
# context is the structured exception that demonstrably helps.
# Healer also uses 'heavy' (set in HealerEncoder._format_batch).
# Surface (HAIKU/SURFACE) stays at 'balanced' — latency-critical, smaller pool gain.
S1_NODE_CONFIG = {
    'content_limit': ENCODING_AGENT.get('node_content_limit'),
    'edge_limit': ENCODING_AGENT.get('node_edge_limit', 5),
    'correction_render': 'heavy',
    # encoding_source is a technical attribution field (encoder:sonnet / anchor /
    # s2:*) — noise to the encoder, which shouldn't reason about who wrote a node.
    # render_rich_node defaults show_encoding_source=True, so hide it explicitly.
    'show_encoding_source': False,
}


def build_node_catalog(judge_outputs, brain):
    """Build deduplicated node catalog from surface outputs across multiple turns.

    Uses system format_node() with S1 config for full rich nodes.
    Adds correction chain annotations on top.

    Args:
        judge_outputs: list of surface_output strings (one per turn, may be None)
        brain: Brain instance

    Returns:
        (catalog_text, node_id_set) — formatted catalog + set of IDs for reference
    """
    import re
    conn = getattr(brain, 'conn', brain)  # tests may pass raw conn
    # Extract all node IDs from surface outputs (pattern: id:XXXXXXXX)
    # Supports both hex IDs (d7d1ddfa) and typed-prefix IDs (con_1c0v)
    seen_ids = set()
    for jo in judge_outputs:
        if not jo or jo == '(no selection)':
            continue
        for match in re.finditer(r'id:([a-z0-9_]{6,8})', jo):
            seen_ids.add(match.group(1))

    if not seen_ids:
        return '', set()

    # Skip community nodes — S2CE manages communities, S1E encodes from conversation.
    # S1E still sees "SURFACED: community node" in the timeline but doesn't get
    # the full content in the catalog. This prevents S1E from revising, correcting,
    # or connecting to community nodes instead of their members.
    community_ids = set()
    if seen_ids:
        placeholders = ','.join('?' * len(seen_ids))
        for row in conn.execute(
                "SELECT id FROM nodes WHERE id IN (%s) AND type = 'community'" % placeholders,
                list(seen_ids)):
            community_ids.add(row[0])

    # Fetch + format: brain.get_node() for data, render_rich_node() for presentation
    catalog_ids = seen_ids - community_ids
    lines = ['Node Catalog (%d nodes surfaced this session)' % len(catalog_ids), '']
    formatted_ids = set()
    for nid in catalog_ids:
        node = brain.get_node(nid)
        if node:
            formatted = render_rich_node(node, S1_NODE_CONFIG)
            if formatted:
                lines.append(formatted)
                lines.append('')
                formatted_ids.add(nid)

    return '\n'.join(lines), formatted_ids


# Backward compat alias
build_encoder_node_catalog = build_node_catalog
