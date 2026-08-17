"""Encoder view policy — what the S1 Scribe is SHOWN, not what is captured.

POLICY, not mechanism: pure predicates + constants over plain values — no I/O,
no brain, no env reads beyond the single flag reader. encode.py and
encode_contract.py render; this file decides what the render feeds. The
boundary rule: numbers shared with other consumers stay in encode_contract
(ENCODING_AGENT is re-exported through pipeline_contract and read by
brain_remember); single-consumer feeding decisions live here.

FILTER AT RENDER, NEVER AT CAPTURE: the traces keep recording everything —
the dashboard, episodes and recall_episodes read them. This policy only shapes
the encoder's prompt view. Every filter marks itself in place (a stubbed
<actions> says "trimmed", an aged catalog entry says how to expand), so
absence can never be misread as "nothing happened" — which is why no prompt
version registration rides this change: there are zero new how-to-read lines.

Flag: BRAIN_S1E_VIEW_POLICY, OFF by default. Off = the assembled prompt is
byte-identical to the pre-policy path — the A/B control arm
(eval/longmem/ab_encode.py flips the flag per arm; an ungated change would
have no control arm and could not be measured).

Measured basis (id:155ddb64, re-measured 2026-08-16 at 80-89% catalog share):
a catalog node renders at ~4-5K chars of which content is only ~22% — edges
and heavy corrections dominate. So aging drops edges + heavy corrections and
keeps a content head; cutting content alone saves almost nothing.
"""
import os


def view_policy_enabled():
    """The view-policy flag (see module docstring). run_encoding resolves it
    once per run and threads it down, mirroring _lived_sequence_enabled — no
    torn state if the env flips mid-run."""
    return os.environ.get('BRAIN_S1E_VIEW_POLICY', '') in ('1', 'true', 'True')


# ── Catalog aging (id:f3302000 / id:f011dc76 — the ~80-89% lever) ──

# Newest N encode rounds render full depth: the encoder keeps seeing complete
# bodies for what it wrote most recently (the quality-bar feedback loop,
# id:3e245ff7). Everything older trims to a stub.
CATALOG_FULL_ROUNDS = 2

# Content head kept on an aged entry. Nodes lead with the claim — 2-3 sentences
# carry the thesis. Deliberately NOT the load-bearing number in this spec: the
# savings come from AGED_NODE_CONFIG dropping edges + heavy corrections.
AGED_CONTENT_CHARS = 400

# The trimmed entry (~700 chars vs ~4,500 full): id + type + title (recognition
# + dedup key), situation (the "does this already exist" anchor — top-level in
# render_rich_node, survives metadata_limit=0), content head, one ⚠ correction
# line. NO edges, NO reasoning/quotes/KV. Trimming is reversible: get_nodes is
# in ENCODING_TOOLS, so the encoder can expand any id on demand.
AGED_NODE_CONFIG = {
    'content_limit': AGED_CONTENT_CHARS,
    'edge_limit': 0,               # the heaviest render weight — dropped
    'correction_render': 'lean',   # one ⚠ header line, not the corrector body
    'metadata_limit': 0,           # no reasoning / quotes / generic KV
    'show_encoding_source': False,
}

# Marks an aged entry in the catalog; the header line the catalog renders when
# any entry is aged explains the tag once (entries mark themselves in place).
AGED_TAG = '[aged]'

# Catalog header time render: relative with sub-day steps ('25m ago', '3h ago')
# — the encoder's questions are recency-shaped and mid-session the hour is the
# signal ("revised 20m ago" = my own recent write; an absolute date can't say
# that). Matches surface's time vocabulary (one system, one clock). Side
# effect by design: relative mode suppresses render_rich_node's duplicate
# absolute `Created:` line. The caller supplies `time_now` (conversation time
# — replay-safe) and `this_session_ids` (the ids this session WROTE: encoded ∪
# authored; reads deliberately don't qualify — reading isn't ownership).
CATALOG_TIME_CONFIG = {'time_format': 'relative', 'time_fine': True}


def timeline_now_attr(now):
    """The <timeline now="…"> stamp — the absolute anchor that makes every
    relative label in the prompt invertible, and the current-time declaration
    the encoder's date resolution never had (only the scouts got a
    current_date). `now` is conversation time (replay-safe); renders UTC to
    match the Frame's 'Now:' vocabulary. Returns '' when unstampable."""
    try:
        from datetime import timezone
        return ' now="%s"' % now.astimezone(timezone.utc).strftime(
            '%Y-%m-%d %H:%M UTC')
    except Exception:
        return ''


def aging_cutoff(run_stops, full_rounds=CATALOG_FULL_ROUNDS):
    """The stop at/after which catalog ids stay full: the Nth-newest encode
    run's stop. None = no aging yet (fewer than N runs this session — nothing
    is "before the last N rounds")."""
    rs = sorted(set(run_stops or ()))
    if len(rs) < full_rounds:
        return None
    return rs[-full_rounds]


def catalog_view(ids, stops, run_stops, protected=()):
    """Order + tier the catalog: returns (ordered_ids, aged_id_set).

    ordered_ids — oldest→newest by each id's last-touched stop, so the most
    recent encodes sit last, adjacent to the timeline (id:f3302000). Ids with
    no known stop are the CURRENT window's surfaced nodes — they sort last and
    never age. Ties break on id for a deterministic render.

    aged — ids last touched strictly before the cutoff, minus `protected`
    (ids surfaced for the current window: the encoder's likeliest dedup/revise
    targets keep full bodies even when their last write is old)."""
    stops = stops or {}
    cutoff = aging_cutoff(run_stops)
    aged = set()
    if cutoff is not None:
        protected = set(protected or ())
        aged = {i for i in ids
                if i not in protected and stops.get(i, cutoff) < cutoff}
    ordered = sorted(ids, key=lambda i: (stops.get(i, float('inf')), i))
    return ordered, aged


# ── Actions (the <actions> block inside timeline turns) ──

# Brain node-op tools whose lines leave <actions> (Tom's ruling, id:27db2472),
# split by what the drop would lose:
#   DROPPED — the turn's <provenance> already carries everything actionable
#     (verb + ids + titles for writes; for get_node[s]/enrich the arguments ARE
#     the ids provenance shows). brain_batch is dropped WITH the known cost
#     that its batched connect sub-ops vanish (accepted: the timeline is not
#     the edge ledger).
#   STUBBED — search tools render a trimmed line keeping the QUERY head:
#     provenance shows result ids only, so a full drop loses the search intent
#     — and a search that found nothing would vanish entirely, though "Anchor
#     looked for X and the brain had nothing" is encodeable signal.
# Edge tools (connect / disconnect / revise_edge) stay fully VISIBLE:
# anchor_touched is node-ids only, deliberately — never edges. Unknown tools
# default to visible.
DROPPED_ACTION_TOOLS = frozenset({
    'remember', 'remember_batch', 'revise', 'revise_batch', 'brain_batch',
    'get_node', 'get_nodes', 'enrich',
})
STUBBED_ACTION_TOOLS = frozenset({
    'recall', 'recall_batch', 'find_node_by_title', 'filter_nodes',
})

# Kept head of a stubbed search line — enough for the query, not the args blob.
ACTION_STUB_HEAD = 60


def action_mode(tool_name):
    """'full' | 'stub' | 'drop' for a tool_result line. Keys on the raw tool
    name the trace metadata carries (`mcp__<server>__<tool>` — post_tool_trace
    records CC's name verbatim), matching any brain MCP server registration
    (plugin or user-scope). Non-brain and unknown tools render full."""
    name = str(tool_name or '')
    if not name.startswith('mcp__'):
        return 'full'
    parts = name.split('__')
    if len(parts) < 3 or 'brain' not in parts[1]:
        return 'full'
    if parts[-1] in DROPPED_ACTION_TOOLS:
        return 'drop'
    if parts[-1] in STUBBED_ACTION_TOOLS:
        return 'stub'
    return 'full'


def action_stub(summary):
    """The trimmed line for a stubbed search action: bare tool name + the
    query head + where the results went. Whitespace-collapsed; the caller
    XML-escapes (the pointer says 'provenance' in words so escaping can't
    mangle it)."""
    s = ' '.join(str(summary or '').split())
    tool, _, args = s.partition(': ')
    base = tool.split('__')[-1]
    head = args[:ACTION_STUB_HEAD] + ('…' if len(args) > ACTION_STUB_HEAD else '')
    return '%s: %s → results in provenance' % (base, head)


def actions_stub_line(n_actions):
    """The <actions> body for an already-encoded turn. The element renders with
    this stub rather than disappearing — absence would read as "nothing
    happened this turn"; the stub states the filter (Tom's design). First
    person — the prompt speaks as the encoder, and the reader of this line IS
    the one who read them last run."""
    return ('trimmed — %d action(s) recorded on this turn; I already read '
            'them in a previous run' % n_actions)


# ── Encoded-turn message text ──

# No trim on already-encoded turns: <other> and <me> keep the full display
# limit. Reverses encoded_turn_trim (Tom, 2026-08-15: "trimming defeats the
# purpose and we found other areas that make much more sense to trim on") —
# the contract key stays for the flag-off control arm and retires with it at
# activation. None = render_lived_sequence's "no cap" (full display limit).
ENCODED_TURN_MESSAGE_CAP = None


# ── Provenance verb split ──

# Node-op categories rendered per turn when the policy is on, replacing the
# merged `encoded(Anchor)` (= created∪revised) with the verbs, plus the
# categories the merged form dropped (Tom's ruling, id:27db2472). Labels use
# the timeline's identity vocabulary — (me), matching <me>/<other>. Each entry
# is (label, link-keys): recalled(me) merges the by-id reads (`recalled`,
# catalog-folded) with the search-tool results (`looked_up`, provenance-only)
# into ONE line — the reader's question is "what did I look at", not which
# tool answered it. Keys match nodes_for_traces' link dict; the substrate
# (anchor_touched → ANCHOR_TOUCHED_KEYS) has kept them separate all along.
PROVENANCE_SPLIT = (
    ('created(me)',  ('created',)),
    ('revised(me)',  ('revised',)),
    ('recalled(me)', ('recalled', 'looked_up')),
    ('archived(me)', ('archived',)),
)

# The run-attribution label for prior encode runs. One identity, two stations:
# (me) = my session half (Anchor in conversation), (scribe) = my encoding half
# (S1E runs). The legacy arm's 'encoded(S1S)' is jargon; 'Anchor' is an
# internal name the operator may not even use — neither belongs in
# encoder-facing text.
ENCODED_RUN_LABEL = 'encoded(scribe)'

# Catalog provenance tags, view-policy arm — SAME vocabulary as the provenance
# labels above, so the catalog and the timeline read as one system. Keys +
# priority order match encode_contract.PROVENANCE_TAGS (the control arm's
# tags); surfaced stays untagged (lowest priority) on both arms.
PROVENANCE_TAGS_VIEW = (
    ('authored', '[authored(me)]'),
    ('recalled', '[recalled(me)]'),
    ('encoded',  '[encoded(scribe)]'),
)
