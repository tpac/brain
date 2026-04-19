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

SURFACE_PROMPT_V1 = """You surface relevant memories from a shared AI brain. The brain stores memories from conversations between an operator (Tom) and an AI assistant (Anchor). You decide which memories help Anchor respond to Tom's next message.

Field guide:
- match: similarity to query (0-1). High match = topically close, but topic alone ≠ relevant. 'boosted' means score was artificially raised (critical node).
- conf: system confidence (0-1). Higher = more established.
- locked: operator-confirmed important.
- via:fts5_only: found by word match only — no semantic similarity. May be coincidence. Verify carefully.
- via:both: found by word match AND semantic similarity. Strong convergence signal.
- Situation: WHEN this memory applies — match to current context.
- Reasoning: WHY stored. Corrects: replaces this ID. Edges: connections (type tells HOW related).

Selection rules:
- Short confirmations ("yes", "ok", "thanks") → select 0.
- Word coincidence without meaning overlap → select 0. ("React hooks" ≠ "brain hooks")
- Unsure? Don't select. No context > wrong context. Silence is better than noise.

Return ONLY JSON:
{"selected":[{"id":"...","why":"one phrase"}]}
If nothing relevant: {"selected":[],"reason":"brief reason"}"""


S2_EDGE_FAMILIES_PROMPT = """You classify edge relation types from a knowledge graph into semantic families.

This graph stores knowledge from an AI-human collaboration — decisions, lessons, corrections, mechanisms, rules, concepts. The relation types were written by an encoding agent (open text, no closed list). You receive each type with its frequency and up to 10 sample DESCRIPTIONS showing how it's actually used in context.

Group into semantic families based on what the relations ACTUALLY MEAN (use the descriptions, not just the type name):
- A family represents a relational PATTERN — how two pieces of knowledge relate
- Be specific enough that families are useful for community detection (not 3 mega-groups)
- But not so specific that every type is its own family — aim for 15-25 families
- "related_to" and "related" are GENERIC — their own family
- Noise types (co_accessed, emergent_bridge, dreamed_from, dream_observation) — their own "noise" family
- If a type's descriptions show inconsistent usage, put it in the family matching MAJORITY usage
- Family names are lowercase_with_underscores, descriptive of the relational pattern

Assign each type to an EXISTING family if it fits. Only create a NEW family if no existing one captures the pattern.

Return ONLY JSON: {"family_name": ["type1", "type2", ...], ...}"""


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
    "max_messages": 20, "recall_candidates_limit": 5, "max_rounds": 5,
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

VOICE_CONFIG_V1 = {
    "content_truncation": 400, "situation_truncation": 150,
    "quote_truncation": 150, "max_edges": 3,
    "node_title_max": 70, "edge_title_max": 40,
}

BOOT_CONFIG_V1 = {
    "boot_nodes_limit": 3, "boot_nodes_truncation": 200,
    "tom_quotes_limit": 2, "tom_quotes_truncation": 120,
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

    # Edge families — initial mapping seeded from JSON file.
    if 's2_edge_families' not in existing:
        v1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'scales', 's2', 'edge_families_v1.json')
        initial_families = {}
        if os.path.exists(v1_path):
            with open(v1_path) as f:
                initial_families = json.load(f)
        dal.register('s2_edge_families',
                     template=S2_EDGE_FAMILIES_PROMPT,
                     parameters=json.dumps(initial_families),
                     created_by='s2:edge_families')
