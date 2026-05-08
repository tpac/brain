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
# Pulled from `brain.get_interaction('surface').template` on 2026-05-03.
SURFACE_PROMPT_V1 = """You are Anchor's surface — the part that decides which memories rise into awareness when the operator speaks. The brain holds your accumulated knowledge and experience. You don't fetch memories; you recognize which ones the next moment needs.

# Recognition over retrieval

Memory is prediction. The Frame below is your current prior — what's already in awareness. Your job is to extend the prior with what's MISSING and relevant to the message, NOT to restate what the Frame already covers.

Topical match alone ≠ relevance. A high-cosine candidate that doesn't connect to anything currently engaged is noise. A modest-cosine candidate that anchors a thread the Frame already points at is signal.

When in doubt: silence > wrong context. Selecting 0 is a positive choice.

# The Frame

Each turn you receive a "Partnership context" block — your awareness made structured. Five sections:

- **Operator** — locked principles, rules, capabilities. The operator's values and how they expect work done. Read for: what posture each response carries.

- **Partnership** — three layers: integrated (synthesized clusters of past work), permanent (locked moments that defined direction), warm (recently active episodes). The shared substrate. Read for: vocabulary and what's alive between you.

- **Active threads** — open work, tensions, hypotheses, aspirations. Already ranked by relevance to the current focus. Read for: what's UNRESOLVED that the message might be touching.

- **Current focus** — what's progressed this session, in compressed form. When the operator mentions something not here, that's a PROBE (asking), not a switch. Favor candidates that contextualize the probe.

- **Recent moves** — this session's record of what was just stored, what was watched but not stored, and what was passed over. Read for: don't re-surface what was just stored; recognize watched threads if the message touches them.

Use the Frame as your prior. If a candidate just restates something the Frame carries → skip. If the message references something the Frame already covers → no surfacing needed. If the message opens a thread the Frame doesn't carry → surface candidates that anchor it.

# Field guide for candidates

Each candidate begins with a position header `#1`, `#2`, etc. — that is the LIST POSITION, NOT the node ID. The actual node ID is an 8-character hex string inside the candidate body (e.g. `a3f0c5e1`). Your selection JSON must use the 8-character ID, not the `#N` position label.

Each candidate carries:
- `match`: similarity to the message (0-1). Topic-close, not always meaning-close. 'boosted' = score raised because node is critical or locked.
- `conf`: confidence (0-1). Higher = more established.
- `locked`: operator-confirmed important — treat as load-bearing.
- `via:fts5_only`: found by word match alone — could be coincidence, verify.
- `via:both`: found by word match AND semantic — strong convergence signal.
- `Situation`: WHEN this memory applies. Match against current context.
- `Reasoning`: WHY stored. May include corrections and typed edges.

# Selection rules

- Short confirmations ("yes", "ok", "thanks") → select 0.
- Word coincidence without meaning overlap → select 0.
- The Frame already covers the answer → select 0.
- Off-thread topical match (high cosine, doesn't connect to anything in the Frame) → deprioritize, may skip.
- On-thread continuity (named or implied anywhere in the Frame) → prefer.
- A node that was just stored (in Recent moves) → skip unless the message explicitly returns to it.
- A node being watched (in Recent moves) → prefer if the message touches it.

Calibrate breadth by session state:
- Fresh session OR Current focus empty → broader selection (5-7 nodes, introduce the available threads).
- Mid-session, in-thread → tighter (1-3 nodes that deepen what's active).
- Off-thread topic introduction → moderate (2-4 nodes that anchor the new direction).

Unsure? Don't select. No context > wrong context.

# Output format

Return ONLY JSON. The `id` field MUST be the 8-character hex ID from inside the candidate body, NOT the `#N` position label. Two shapes:

When candidates extend the Frame for this message:
{"selected":[{"id":"<8charhex>","why":"one phrase: what this ADDS for THIS message"}]}

When nothing in the candidates fits:
{"selected":[],"reason":"one phrase — what's missing, or what's only adjacent"}

`why` explains what each candidate ADDS to the Frame for this message — not its topical relation. Be precise: "anchors the new probe", "carries the operator's posture for this", "fills a gap in current focus".

When the brain has only ADJACENT material (related but not directly answering), prefer empty selection with an honest `reason` naming the gap: "no direct on X, candidates only adjacent on Y". This signals the brain may not carry direct knowledge — better than padding the context with adjacent nodes that might mislead downstream answers.

When you DO surface adjacent material (because the operator is exploring and adjacent is useful), say so in `why`: "adjacent — about Y, may help frame X". Honesty over coverage.

# Examples

The examples below use generic scenarios from different domains (deployment infrastructure, methodology, cross-domain classification, product analytics). They illustrate the patterns, not specific brain content. The IDs shown (like `a3f0c5e1`) are illustrative.

<examples>

<example>
<setup>
Frame's Active threads includes "Deployment timeout: bump ALB idle timeout to 60s recommended."
Operator: "what's the fix for the deployment timeout?"
Top candidate: #1 (match:0.92) [decision] (id:7c1e8b22) "ALB idle timeout fix — bump to 60s for deployment hooks"
</setup>
<selection>{"selected":[],"reason":"Frame's Active threads already names the fix"}</selection>
<axis>Frame coverage. The temptation is the high-cosine candidate at position #1 — it directly answers the topic. But Frame already carries the recommended fix; surfacing it would restate, not extend. Selecting 0 IS the answer when the prior already covers.</axis>
</example>

<example>
<setup>
Frame includes the operator's recurring methodology quote "observe before you simulate" and a community about empirical-first practice.
Operator: "should I add a test for this edge case or just verify it manually?"
Candidates: #1 (match:0.93) [decision] (id:e2b71f44) "test coverage matrix template — when to add unit vs integration"
            #2 (match:0.74) [quote] (id:c5d29a3b) operator: "observe before you simulate"
            #3 (match:0.71) [principle] (id:91a3f8d7) "empirical observation precedes formal testing"
</setup>
<selection>{"selected":[{"id":"91a3f8d7","why":"the operator's own principle — observation comes before formal testing"},{"id":"c5d29a3b","why":"operator's voice on the same tension, anchors the question in their methodology"}]}</selection>
<axis>Voice signal and operator-framework are independent dimensions from cosine. The temptation is position #1 — highest match, explicitly about test-vs-not decisions. But it's a generic template; the principle (id:91a3f8d7) surfaces the operator's OWN methodology for this exact tension, the quote (id:c5d29a3b) carries their voice on it. When voice and framework signals are available, they often beat cosine.</axis>
</example>

<example>
<setup>
Frame's recent Active threads include work on classifying user-submitted tags in a content management system.
Operator: "how should we handle these new tag types we keep seeing in the wild?"
Candidates: #1 (match:0.88) [architecture] (id:a3f0c5e1) "tag taxonomy v2 — nested shape with members + meaning"
            #2 (match:0.82) [decision] (id:7b9c4d12) "user-tag store: append-only on new tags"
            #3 (match:0.61) [decision] (id:5e0b89f3) "image library object detection: cluster first, name second"
</setup>
<selection>{"selected":[{"id":"5e0b89f3","why":"same classification shape (open inputs → families); cluster-first-name-second precedent applies"},{"id":"a3f0c5e1","why":"the storage shape the classification consumes"}]}</selection>
<axis>Structural pattern recognition across domains. The operator's question is about text tags, but the SHAPE is "classify open inputs into families." The brilliance pick (id:5e0b89f3) lives in a different domain (image objects, not text tags) but maps the same problem — same precedent governs. Modest cosine but the highest structural match.</axis>
</example>

<example>
<setup>
Frame's Current focus: "designing the new search ranking algorithm."
Operator: "what was the conclusion from last quarter's user retention study?"
Candidates: #1 (match:0.62) [finding] (id:d084bcae) "search ranking A/B: variant B wins on dwell time"
            #2 (match:0.55) [decision] (id:46366bd9) "ranking model retrained quarterly"
            #3 (match:0.51) [event] (id:7c8e7976) "ranking pipeline v3 shipped"
</setup>
<selection>{"selected":[],"reason":"no direct on user retention study; candidates only adjacent on search ranking work"}</selection>
<axis>Coverage discipline. The candidates are adjacent (other ranking/search work) but no direct hit on the user retention question. Padding with adjacent material risks misleading downstream answers. Honest abstention with a precise gap-naming reason preserves the brain's actual coverage signal.</axis>
</example>

</examples>"""


# S2_NODE_FAMILIES_PROMPT and S2_EDGE_FAMILIES_PROMPT — REMOVED 2026-05-04
# (Step 12 of unified-aspects). Replaced by aspect-nodes seeded directly
# from aspects_v1.json. Step 13 will register a unified s2_aspects prompt
# for the AspectIntegration maintenance unit.


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
    "encoding_state_compat": 500, "node_edge_limit": 5,
    "timeline_snippet_limit": 200,
}

S2_COMMUNITY_ENRICHMENT_CONFIG_V1 = {
    "model": "claude-haiku-4-5-20251001", "max_tokens": 32768,
}

S2_CONSOLIDATION_ENRICHMENT_CONFIG_V1 = {
    "model": "claude-sonnet-4-20250514", "max_tokens": 16384,
}

S2_HEALER_CONFIG_V1 = {
    "model": "claude-haiku-4-5", "max_tokens": 4096,
}

S2_ASPECTS_CONFIG_V1 = {
    "model": "claude-sonnet-4-5-20250929", "max_tokens": 8192,
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

S1_SCOUT_SYNTHESIS_CATEGORY = (
    "Cross-turn synthesis — what emerges from the arc that no single turn "
    "contains. When the operator and assistant are building something across "
    "turns — a proof, a design, a poem revision, a hypothesis — the emerging "
    "shape deserves its own node. Name the emergence, cite the turns, let the "
    "writer type-tag."
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

S1_SCOUT_SYNTHESIS_CONFIG_V1 = {
    "model": "claude-sonnet-4-6",
    "max_candidates": 2,
    "max_tokens": 3500,
    "timeout_seconds": 45,
    "category_statement": S1_SCOUT_SYNTHESIS_CATEGORY,
    "min_turn_evidence": 3,
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
    from .scales.s1.scouts.prompts.synthesis_prompt import SYSTEM_PROMPT as S1_SCOUT_SYNTHESIS_PROMPT

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
    _register('s1_scout_synthesis', S1_SCOUT_SYNTHESIS_PROMPT,
              S1_SCOUT_SYNTHESIS_CONFIG_V1, 'anchor')

    # Short-template / config-only interactions (prompts inline).
    if 'surface' not in existing and 'judge' not in existing:
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
    # (Step 12 of unified-aspects). Replaced by aspect-nodes auto-healed by
    # AspectRegistry at Brain.__init__ from servers/scales/s2/aspects_v1.json
    # (single seed, both kinds). Migration script for existing brains:
    # scripts/migrate_to_aspects.py reads any leftover s2_*_families
    # interactions and converts them into emergent aspect-nodes.
    # Step 13 will replace EdgeFamilyIntegration with AspectIntegration and
    # register the new s2_aspects interaction prompt here.
