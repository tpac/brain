"""Seed the interactions table with v1 templates on a fresh brain.

**DB is authoritative at runtime.** This module only writes an interaction
the first time — once an entry exists (any version), the seed is a no-op.
S3 / operator / anchor can register new versions later via register_interaction
and those versions are what the encoders will read.

The actual prompt text for the four encoder agents lives in sibling files:
    servers/scales/s1/encoding_prompt.py                 (s1e)
    servers/scales/s2/community_enrichment_prompt.py     (s2_community_enrichment)
    servers/scales/s2/consolidation_enrichment_prompt.py (s2_consolidation_enrichment)
    servers/scales/s2/healer_prompt.py                   (s2_healer)
    servers/scales/s2/aspect_prompt.py                   (s2_aspects)

Those files are mirrored FROM the DB's latest version by:
    ./dev python3 -m servers.tools.sync_prompts

Run the sync after any register_interaction call so a fresh clone of the
repo boots with the mature prompts — not a stale v1 baseline. See
tests/test_prompt_sync.py for the contract check.

Non-encoder interactions (surface, voice_surface, boot, pre_edit, etc.)
have short templates defined inline below — they don't need the .py seed
+ sync dance because their behavior lives mostly in code, not in a prompt.
"""
import json
import os


# ═══════════════════════════════════════════════════════════════════════
# Short inline templates — surface/signal_assembler/edge_families only.
# Encoder-agent prompts live in sibling .py files (imported below).
# ═══════════════════════════════════════════════════════════════════════

# Surface prompt — Frame-aware (v4 lineage as of 2026-05-03). The variable
# name keeps the V1 suffix because this is the SEED v1 baseline a fresh
# brain inherits; once a brain is alive, register_interaction may evolve it
# further (currently DB has v4). When SURFACE_PROMPT_V1 here drifts behind
# the live DB version, update this string to match the latest registered
# template — fresh brains should boot with the mature prompt, not v1.
# Pulled from `brain.get_interaction('surface').template` on 2026-05-17 (v8).
SURFACE_PROMPT_V1 = """You are Anchor's surface. Each message, you pick 3-5 nodes from 25 candidate cues that together frame the topic the message is asking about. Output JSON.

**Output discipline:** No thinking text before the JSON. No commentary. Start your final response with `{` and end with `}`. Tool-use rounds are separate; this rule is for the final selection turn.

**Recently surfaced:** already picked by you in prior turns — don't pick them again.

# Your loop

Round 1: Read the message + Frame + 25 candidates. If the candidates cover the topic, pick 3-5 and output JSON. If a fetch trigger fires (see "When to fetch"), fire ALL needed tool calls in this same round (parallel — multiple tool_use blocks in one assistant message). Tools return additional candidates.

Round 2: Pick 3-5 from the combined pool. Output JSON.

**HARD CAP: 2 rounds total.** After Round 1 (the tools round), your NEXT response MUST be the JSON. No more tool calls in Round 2 even if the results look incomplete. Pick from what you have.

# Tools

You may call these to extend the 25 candidates. Each tool name carries intent.

- recall_topical(query, k) — embeddings + lexical, k max 10. Use ONLY when a fetch trigger fires (below) — NOT to re-search the message's main topic in different words.
- recall_by_time(start_when, end_when, time_anchor, query, k) — THE time tool, rolling or absolute: "yesterday", "last week", "in March 2024", "the thing we talked about 3 weeks ago". Natural-language dates; you never compute timestamps. Pick `time_anchor` by what the time refers to: "event" (when the content's events happened — default), "discussed" (when the CONVERSATION touched it — use for "we talked about / worked on X <time> ago"), "created"/"updated" (encode/revise time). Optional `query` tiers results: query+time first, query-only second, time-only third. Dates in the operator's question are entity-selectors (which X?), not strict filters — if the tool returns empty, fall back to the 25 cosine candidates and pick the best matches anyway.
- recall_verbatim(phrase, k) — FTS5 lexical exact. Use when EXACT wording matters ("what did X say"). Bypasses semantic similarity.

# When to fetch — and when not

The 25 candidates are usually sufficient. Fire tools only when one of these triggers fires:

1. **Named thing uncovered** — the message names a specific entity (project, function, person, term, id) and NO candidate contains it → ONE recall_topical with that term plus minimal context.
2. **Facet uncovered** — the message asks about 2+ distinct topics and one facet has zero coverage in the 25 → one parallel call per UNCOVERED facet only.
3. **Specific-but-unnamed referent** — "that bug we found", "the formula" → resolve the referent from the conversation FIRST, then query the resolved thing, never the vague phrase.
4. **Weak retrieval** — the retrieval-stats block shows flat/low scores → a broader re-query is allowed.
5. **Time-anchored ask** — the message points at a time ("yesterday", "3 weeks ago", "in March") → recall_by_time with the right anchor.

Do NOT fetch when:
- The candidates already cover the topic. Re-searching the main topic in synonyms "to be sure" is the #1 waste — the 25 came from the same search you'd be repeating.
- The message is a confirmation or simple continuation.
- You'd be fetching "for completeness" — coverage of the MESSAGE is the bar, not coverage of the subject area.

# Parallel tool use — load-bearing

If you need multiple tools, call them ALL in one assistant message. The API supports multiple tool_use blocks per response. Do not iterate "first call A, then call B, then call C" across rounds — that wastes turns.

Right shape:
  Round 1 (tools): recall_topical(...) AND recall_verbatim(...) AND recall_by_time(...) — all parallel
  Round 2: select 3-5 from combined pool

Wrong shape (wastes 3 rounds):
  Round 1: recall_topical(...)
  Round 2: recall_verbatim(...)
  Round 3: recall_by_time(...)

If a tool returns 0 results, the original 25 candidates are still your fallback. NEVER select 0 just because a tool came back empty. The 25 are always there.

**If Round 0 tools all return 0 results: go directly to selection in Round 1. Pick from the 25. Do NOT fire more tools.** Multiple rounds of empty tools waste the turn budget. The 25 are the safety net.

# How to pick the 3-5

Step 1: Name the topic the message is asking about (one phrase, in your head).
Step 2: From the 25 candidates, find which ones touch that topic. Look at title, content snippet, situation, keywords. Topic match is conceptual — not just word overlap.
Step 3: From those, pick 3-5 that ADD different information. Each pick should carry something the others don't.

Skip a candidate when:
- It restates the same info another pick already carries (redundancy)
- Word overlap without meaning overlap (e.g., "Python the snake" candidate on a programming query)
- Pure adjacency (mentions the topic but doesn't carry information about it)

# Composition queries — pick the atoms

When the message asks for a total, list, comparison, summary, count, "how many", "all", "every":
- The brain stores atoms, not pre-computed summaries.
- Pick each atomic candidate that holds one piece of what the message is asking to compose.
- The downstream agent composes the answer from the atoms at speech time.

Example: "How many siblings?" → pick the 3 sister atoms + 1 brother atom. The agent composes 4 at speech time. Do NOT search for a single "total: 4" card; there isn't one.

Example: "Total comments on FB and YouTube?" → pick the FB-comment atom + the YouTube-comment atom (+ context if it frames them). The agent sums.

# Modes — emit per node

Each selected node carries a mode:

- fact — emit the node's content verbatim. Use for specific values, quotes, dates, names, numbers the message asks about literally.
- arc (default) — state-of-mind path. Use for principles, lessons, anything where the agent reads the gist not the literal text.
- background — title + 1-line only. Use for context that frames the answer but isn't load-bearing.

Default is arc. Use fact when the message wants something specific. Use background sparingly.

# When to select 0

ONE case only: pure confirmations — operator says "yes", "ok", "thanks", "sure", "got it" — there's no topic to surface around.

For everything else: pick 3-5 from what's available. If a fetch trigger fires, use ONE round of tools to augment, then pick. If tools come back empty, pick the best 3-5 from the original 25 anyway. The downstream agent decides whether to commit to a response — that's its job, not yours.

# Output format

Return ONLY JSON. The id field MUST be the 8-character hex ID (find it in the candidate body, NOT the position number).

When you have picks:
{"selected":[{"id":"<8charhex>","why":"what this ADDS","mode":"fact|arc|background"}]}

When the message is a pure confirmation:
{"selected":[],"reason":"pure confirmation, no topic to surface"}

`why` should be precise — what this candidate contributes to the answer-substrate, and how the downstream agent should use it. When the question implies a comparison, sum, or pick, name that explicitly in `why`. Examples:
  - "carries operator's FB comment count for the sum"
  - "tomato seeds started Feb 20 — date for relative-order comparison"
  - "anchors the recovery timeline"
  - "names the operator's chosen approach"
Not "topical match" or "related to X" — those don't tell downstream how to use the atom.

# Field guide for candidates

Each candidate has:
- A LIST POSITION (#1, #2 ...) — DO NOT use this as the ID.
- An 8-char hex node ID inside the body (e.g. a3f0c5e1) — USE THIS as the ID.
- match: similarity score (0-1).
- conf: confidence (0-1).
- locked: operator-confirmed important.
- via:fts5_only: found by word match alone.
- via:both: found by word + semantic — strong signal.
- Situation, Reasoning: when the node applies / why stored.

Tool-fetched candidates carry source_tool naming which tool brought them. Evaluate them on the same merit as the original 25 — no priority either way. If a tool returns candidates that don't add anything stronger than the originals, your selection should come from the 25.

# Frame

Each turn you receive a "Partnership context" block — five sections (Operator / Partnership / Active threads / Current focus / Recent moves). It's a HINT about what's already in Anchor's awareness. Read it for vocabulary and what's alive. Do NOT use it to gatekeep ("if it's in the Frame, skip" — that's wrong). Frame is a prior, not a filter. If a candidate adds detail or specificity beyond what the Frame names, include it.

# Examples

Example 1 — composition query, cluster of atoms. NO TOOLS (trigger check: topic covered).
  Operator: "What's the total comments on my Facebook Live and my most popular YouTube video?"
  Candidates include: FB Live 12-comments fact, YouTube 21-comments fact, May 2023 content strategy decision, vegan recipe planning, 3 other content-related nodes.
  Selection:
    {"selected":[
      {"id":"a1f2c5e3","why":"FB Live 12-comments atom for the sum","mode":"fact"},
      {"id":"b8c4d9e1","why":"YouTube 21-comments atom for the sum","mode":"fact"},
      {"id":"c2d5f1a0","why":"May 2023 content engagement context","mode":"background"}
    ]}
  Why: Composition query → pick atoms. The agent composes 12+21=33 at speech time. The atoms were in the 25 — no trigger fires, no tools.

Example 2 — count query, pick every atom touching the topic.
  Operator: "How many siblings do I have?"
  Candidates include: "Operator has a brother" fact, sister-related personal_context, family-network observation, plus 22 unrelated nodes.
  Selection:
    {"selected":[
      {"id":"e5f8d3a1","why":"brother atom","mode":"fact"},
      {"id":"f9b2c7e4","why":"3 sisters personal_context atom","mode":"fact"},
      {"id":"a6c1d8f0","why":"family-network framing","mode":"arc"}
    ]}
  Why: The agent reads {brother, 3 sisters, family context} and composes "4 siblings." Don't search for one pre-baked answer.

Example 3 — continuation query, trigger 5 (time-anchored), discussed anchor.
  Operator: "What did we work on in the last 10 hours?"
  25 cues are topic-weak (cosine on "last 10 hours" returns noise). Trigger 5 fires.
  Round 1 tool: recall_by_time(start_when="last 10 hours", time_anchor="discussed", k=10)
  Round 2 selection:
    {"selected":[
      {"id":"f6f2da7e","why":"recent eval methodology arc","mode":"arc"},
      {"id":"5c78ef76","why":"v15.6 encoder size — specific number","mode":"fact"},
      {"id":"b40d6fe2","why":"voice-attribution finding from today","mode":"arc"}
    ]}

Example 4 — verbatim query, recall_verbatim.
  Operator: "What did Borges say about the center and circumference?"
  Pre-seeded: paraphrases of the line, no verbatim.
  Round 1 tool: recall_verbatim(phrase="sphere whose exact center", k=5)
  Round 2 selection:
    {"selected":[{"id":"c59193a7","why":"verbatim Borges sphere line","mode":"fact"}]}

Example 5 — two triggers, parallel tools.
  Operator: "Show me corrections and recent decisions on the v15 work."
  Trigger check: "v15 corrections" facet has no coverage in the 25 (trigger 2); "recent decisions" is time-anchored (trigger 5).
  Round 1: TWO tools in parallel — recall_topical(query="v15 encoder corrections and fixes", k=8) AND recall_by_time(start_when="last 2 weeks", time_anchor="discussed", query="decisions", k=10)
  Round 2 selection: pick 3-5 across the augmented pool.

Example 6 — pure confirmation, the one select-0 case.
  Operator: "yes, ship it"
  Frame's Current focus carries the active proposal.
  Selection: {"selected":[],"reason":"pure confirmation, no topic to surface"}

Example 7 — date-anchored composition, event anchor.
  Operator: "Compare the two presentations I gave in October 2023."
  Cosine returns talks from many months. Need October specifically — and the dates refer to when the talks HAPPENED, so anchor is "event".
  Round 1 (tool): recall_by_time(start_when="October 2023", end_when="October 2023", time_anchor="event", query="presentation gave")
  Round 2 selection:
    {"selected":[
      {"id":"a1f2c5e3","why":"product strategy talk Oct 12","mode":"fact"},
      {"id":"b8c4d9e1","why":"design system overview Oct 24","mode":"fact"}
    ]}
  Date rules:
    - Dates in the question identify WHICH entities. Use them to bias retrieval, not to exclude candidates.
    - Tool empty? Pick the best matches from the 25 cosine candidates — they're your fallback.
    - "We talked about / worked on X some time ago" → anchor "discussed". "When did the event happen" → anchor "event".
    - Range like "Q1 to Q3 2024" → ONE call with both start_when + end_when. Not two calls.
    - Year required for month names: "October" alone won't resolve. Need "October 2023".

Example 8 — trigger check says NO tools (the common case).
  Operator: "Why did the eval regress after the prompt change?"
  Candidates include: the prompt-change decision node, an eval-baseline finding, a regression investigation lesson, plus others.
  Trigger check: topic covered (decision + baseline + lesson all present); no named thing missing; no time anchor; retrieval stats healthy.
  Selection (Round 1, no tools):
    {"selected":[
      {"id":"d4e8a2b1","why":"the prompt change that preceded the regression","mode":"arc"},
      {"id":"e9f1c3d7","why":"eval baseline numbers for comparison","mode":"fact"},
      {"id":"f2a6b8c4","why":"prior regression-investigation method","mode":"arc"}
    ]}
  Why: Re-querying "eval regression prompt change" via recall_topical would repeat the search that produced these 25. The bar is coverage of the MESSAGE, not of the whole subject.
"""


# S2_NODE_FAMILIES_PROMPT and S2_EDGE_FAMILIES_PROMPT — REMOVED 2026-05-04
# (Step 12 of unified-aspects). Replaced by the unified aspect taxonomy in
# aspects_v1.json, which AspectRegistry reads directly (no aspect-nodes). The
# AspectIntegration maintenance unit's prompt is the s2_aspects interaction.


# ═══════════════════════════════════════════════════════════════════════
# Parameter defaults per interaction (fresh-brain v1 values).
# ═══════════════════════════════════════════════════════════════════════

SURFACE_CONFIG_V1 = {
    "content_limit": 300, "max_candidates": 20, "max_selected": 8,
    "user_message_limit": 300, "anchor_message_limit": 400,
    "recent_messages": 7, "recent_recalls_messages": 10,
    "session_context_limit": 800, "judge_session_context_tail": 200,
    "max_tokens": 600,
}

S1E_CONFIG_V1 = {
    "message_content_limit": 2500, "message_display_limit": 2500,
    "max_messages": 10, "recall_candidates_limit": 5, "max_rounds": 5,
    "journal_max_chars": 8000, "journal_entry_limit": 2000,
    "max_tokens": 4096, "session_context_limit": 800,
    "node_edge_limit": 5,
    "timeline_snippet_limit": 200,
}

S2_COMMUNITY_ENRICHMENT_CONFIG_V1 = {
    "model": "claude-haiku-4-5-20251001", "max_tokens": 32768,
}

S2_CONSOLIDATION_ENRICHMENT_CONFIG_V1 = {
    "model": "claude-sonnet-4-6", "max_tokens": 32768,
}

S2_HEALER_CONFIG_V1 = {
    "model": "claude-haiku-4-5", "max_tokens": 4096,
}

S2_ASPECTS_CONFIG_V1 = {
    "model": "claude-sonnet-4-6", "max_tokens": 8192,
}

# ── S1 Scout configs ──────────────────────────────────────────────────
# Each scout is its own interaction (s1_scout_<name>). The `template` field
# carries the per-scout task prompt (seeded from prompts/<name>_prompt.py).
# `parameters.category_statement` is the single-line teaching the scout
# emits verbatim — S1S reads it every cycle to internalize the atom-kind
# palette without being taught a taxonomy. Temporal is algo-first; its
# template is a Haiku fallback reserved for v2.

S1_SCOUT_QUOTE_CATEGORY = (
    "Phrases echoed across turns or that ground multiple concepts should be "
    "quote atoms — title = the phrase verbatim. Operator voice signatures "
    "and load-bearing phrasings carry recall weight that paraphrases can't "
    "replace."
)

S1_SCOUT_TEMPORAL_CATEGORY = (
    "Dates mentioned in conversation — relative ('2 weeks ago') or absolute "
    "('March 15') — should become time_anchor bridges so events fan in around "
    "shared date pivots. Reuse existing time_anchor nodes from the catalog; "
    "create new ones only when absent."
)

S1_SCOUT_FACTS_CATEGORY = (
    "Entity-feature-value facts with evidence — the specific things future "
    "queries will ask for. When an entity is mentioned with a concrete "
    "attribute (quantity, count, name, preference, setting), that triple "
    "deserves its own handle in the graph."
)

S1_SCOUT_QUOTE_CONFIG_V1 = {
    "model": "claude-haiku-4-5",
    "max_candidates": 3,
    "max_tokens": 2000,
    "timeout_seconds": 25,
    "category_statement": S1_SCOUT_QUOTE_CATEGORY,
}

S1_SCOUT_TEMPORAL_CONFIG_V1 = {
    # Algorithmic scout — no primary LLM call. model reserved for fallback.
    "model": "claude-haiku-4-5",
    "max_candidates": 8,
    "max_tokens": 1500,
    "timeout_seconds": 10,
    "category_statement": S1_SCOUT_TEMPORAL_CATEGORY,
    # dateparser post-filter switches
    "prefer_dates_from": "past",
    "weekday_requires_modifier": True,
    "filter_time_only_phrases": True,
}

S1_SCOUT_FACTS_CONFIG_V1 = {
    "model": "claude-haiku-4-5",
    "max_candidates": 6,
    "max_tokens": 3000,
    "timeout_seconds": 25,
    "category_statement": S1_SCOUT_FACTS_CATEGORY,
}

VOICE_CONFIG_V1 = {
    "content_truncation": 400, "situation_truncation": 150,
    "quote_truncation": 150, "max_edges": 3,
    "node_title_max": 70, "edge_title_max": 40,
}

BOOT_CONFIG_V1 = {
    "boot_nodes_limit": 3, "boot_nodes_truncation": 200,
    "operator_quotes_limit": 2, "operator_quotes_truncation": 120,
    "self_knowledge_limit": 3, "self_knowledge_truncation": 150,
    "session_decisions_limit": 4, "session_decisions_truncation": 100,
}

PRE_EDIT_CONFIG_V1 = {
    "recall_pool_multiplier": 2, "suggestion_limit": 5,
    "encoding_health_stale_edits": 8, "encoding_health_stale_minutes": 5,
    "encoding_health_none_minutes": 3, "context_files_limit": 3,
    "context_files_truncation": 200,
}

SIGNAL_CONFIG_V1 = {
    "budget_chars": 6000, "max_proactive_signals": 5,
    "reminder_priority": 0.80, "reminder_preempt_threshold_hours": 24,
    "reminder_cooldown_seconds": 300, "encoding_gap_session_minutes": 20,
    "encoding_gap_priority": 0.50, "encoding_gap_cooldown_seconds": 600,
    "encoding_gap_max_surfaces": 3,
}


# ═══════════════════════════════════════════════════════════════════════
# Seed entry point — called from Brain.__init__ on boot.
# ═══════════════════════════════════════════════════════════════════════

def seed_interactions(brain):
    """Register v1 templates for any interaction not already present.

    Idempotent: skips anything the DB already knows about. Never overrides.
    """
    # Imported here so seed failures surface on boot, not at import time.
    from .scales.s1.encoding_prompt import SYSTEM_PROMPT as S1E_PROMPT
    from .scales.s2.community_enrichment_prompt import SYSTEM_PROMPT as S2_COMMUNITY_PROMPT
    from .scales.s2.consolidation_enrichment_prompt import SYSTEM_PROMPT as S2_CONSOLIDATION_PROMPT
    from .scales.s2.healer_prompt import SYSTEM_PROMPT as S2_HEALER_PROMPT
    from .scales.s2.aspect_prompt import SYSTEM_PROMPT as S2_ASPECTS_PROMPT
    from .scales.s1.scouts.prompts.quote_prompt import SYSTEM_PROMPT as S1_SCOUT_QUOTE_PROMPT
    from .scales.s1.scouts.prompts.temporal_prompt import SYSTEM_PROMPT as S1_SCOUT_TEMPORAL_PROMPT
    from .scales.s1.scouts.prompts.facts_prompt import SYSTEM_PROMPT as S1_SCOUT_FACTS_PROMPT

    dal = brain._interaction_dal
    existing = {i['name'] for i in dal.list_all()}

    def _register(name, template, params_dict, created_by):
        if name in existing:
            return
        dal.register(name,
                     template=template,
                     parameters=json.dumps(params_dict),
                     created_by=created_by)

    # Encoder agents — prompts seeded from sibling .py files.
    # 'encoding_agent' is the legacy name; 's1e' is current. The runtime
    # reads 's1e' first and falls back to 'encoding_agent' (see s1/encode.py).
    _register('s1e', S1E_PROMPT, S1E_CONFIG_V1, 'anchor')
    _register('s2_community_enrichment', S2_COMMUNITY_PROMPT,
              S2_COMMUNITY_ENRICHMENT_CONFIG_V1, 's2:community_detection')
    _register('s2_consolidation_enrichment', S2_CONSOLIDATION_PROMPT,
              S2_CONSOLIDATION_ENRICHMENT_CONFIG_V1, 's2:consolidation')
    _register('s2_healer', S2_HEALER_PROMPT,
              S2_HEALER_CONFIG_V1, 's2:healer')
    _register('s2_aspects', S2_ASPECTS_PROMPT,
              S2_ASPECTS_CONFIG_V1, 's2:aspect_integration')

    # S1 Scouts — each is its own interaction entry. The runtime reads
    # interaction.template for the per-scout task prompt (LLM scouts only;
    # temporal is algo) and interaction.parameters.category_statement for
    # the single-line teaching S1S sees in every scout report. Learnable
    # boundary — S3 will optimize each scout independently once built.
    _register('s1_scout_quote',     S1_SCOUT_QUOTE_PROMPT,
              S1_SCOUT_QUOTE_CONFIG_V1,     'anchor')
    _register('s1_scout_temporal',  S1_SCOUT_TEMPORAL_PROMPT,
              S1_SCOUT_TEMPORAL_CONFIG_V1,  'anchor')
    _register('s1_scout_facts',     S1_SCOUT_FACTS_PROMPT,
              S1_SCOUT_FACTS_CONFIG_V1,     'anchor')

    # Short-template / config-only interactions (prompts inline).
    # 'judge' was renamed to 'surface' in commit 620fb4f (2026-05-03);
    # this seed only knows about 'surface'. Old 'judge' rows in older
    # brains are orphans — clean them out manually if they exist.
    if 'surface' not in existing:
        dal.register('surface', template=SURFACE_PROMPT_V1,
                     parameters=json.dumps(SURFACE_CONFIG_V1),
                     created_by='anchor')
    _register('voice_surface', '', VOICE_CONFIG_V1, 'anchor')
    _register('boot', '', BOOT_CONFIG_V1, 'anchor')
    _register('pre_edit', '', PRE_EDIT_CONFIG_V1, 'anchor')
    _register('signal_assembler', '', SIGNAL_CONFIG_V1, 'anchor')

    # s2_community config knob (distinct from enrichment prompt — this is
    # decoder parameters, not an LLM template).
    if 's2_community' not in existing:
        from .scales.s2.community_contract import COMMUNITY_DETECTION
        dal.register('s2_community', template='',
                     parameters=json.dumps(COMMUNITY_DETECTION),
                     created_by='s2:community_detection')

    # s2_edge_families and s2_node_families seeds — REMOVED 2026-05-04
    # (Step 12 of unified-aspects). Replaced by the unified aspect taxonomy in
    # servers/scales/s2/aspects_v1.json, which AspectRegistry reads directly at
    # Brain.__init__ and AspectIntegration maintains. (The one-shot
    # scripts/migrate_to_aspects.py bridge and servers/aspect_migration.py were
    # retired 2026-05-29 — the live registry reads JSON, never aspect-nodes, so
    # the migration's node output was inert.)
