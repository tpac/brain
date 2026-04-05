"""Seed interactions table with v1 from current hardcoded values.

Run on first boot or when interactions table is empty.
The current code IS v1. After seeding, code reads from the table.
"""
import json
import os


# ── Judge v1: the instruction portion of the prompt ──
JUDGE_PROMPT_V1 = """You are a memory relevance judge for a shared AI brain. The brain stores memories from conversations between an operator (Tom) and an AI assistant (Anchor). You decide which memories help Anchor respond to Tom's next message.

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

JUDGE_CONFIG_V1 = {
    "content_limit": 300,
    "max_candidates": 20,
    "max_selected": 8,
    "user_message_limit": 300,
    "anchor_message_limit": 400,
    "recent_messages": 7,
    "recent_recalls_messages": 10,
    "session_context_limit": 800,
    "judge_session_context_tail": 200,
    "max_tokens": 600,
}


# ── Encoding Agent v1: loaded from hooks/prompts/encoding-agent-v3.md ──
def _load_encoding_prompt():
    """Load the current encoding agent prompt from file."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'hooks', 'prompts', 'encoding-agent-v3.md')
    try:
        with open(prompt_path) as f:
            return f.read()
    except Exception:
        return ''

ENCODING_CONFIG_V1 = {
    "message_content_limit": 2500,
    "message_display_limit": 2500,
    "max_messages": 10,
    "recall_candidates_limit": 5,
    "max_rounds": 5,
    "journal_max_chars": 8000,
    "journal_entry_limit": 2000,
    "max_tokens": 4096,
    "session_context_limit": 800,
    "encoding_state_compat": 500,
    "node_edge_limit": 5,
    "timeline_snippet_limit": 200,
}


# ── Voice Surface v1 ──
VOICE_CONFIG_V1 = {
    "content_truncation": 400,
    "situation_truncation": 150,
    "quote_truncation": 150,
    "max_edges": 3,
    "node_title_max": 70,
    "edge_title_max": 40,
}


# ── Boot v1 ──
BOOT_CONFIG_V1 = {
    "boot_nodes_limit": 3,
    "boot_nodes_truncation": 200,
    "tom_quotes_limit": 2,
    "tom_quotes_truncation": 120,
    "self_knowledge_limit": 3,
    "self_knowledge_truncation": 150,
    "session_decisions_limit": 4,
    "session_decisions_truncation": 100,
}


# ── Pre-Edit v1 ──
PRE_EDIT_CONFIG_V1 = {
    "recall_pool_multiplier": 2,
    "suggestion_limit": 5,
    "encoding_health_stale_edits": 8,
    "encoding_health_stale_minutes": 5,
    "encoding_health_none_minutes": 3,
    "context_files_limit": 3,
    "context_files_truncation": 200,
}


# ── Signal Assembler v1 ──
SIGNAL_CONFIG_V1 = {
    "budget_chars": 6000,
    "max_proactive_signals": 5,
    "reminder_priority": 0.80,
    "reminder_preempt_threshold_hours": 24,
    "reminder_cooldown_seconds": 300,
    "encoding_gap_session_minutes": 20,
    "encoding_gap_priority": 0.50,
    "encoding_gap_cooldown_seconds": 600,
    "encoding_gap_max_surfaces": 3,
}


def seed_interactions(brain):
    """Seed all 6 interactions with v1 from current hardcoded values.

    Idempotent: skips interactions that already exist.
    """
    dal = brain._interaction_dal
    existing = {i['name'] for i in dal.list_all()}

    if 'judge' not in existing:
        dal.register('judge',
                     template=JUDGE_PROMPT_V1,
                     parameters=json.dumps(JUDGE_CONFIG_V1),
                     created_by='anchor')

    if 'encoding_agent' not in existing:
        dal.register('encoding_agent',
                     template=_load_encoding_prompt(),
                     parameters=json.dumps(ENCODING_CONFIG_V1),
                     created_by='anchor')

    if 'voice_surface' not in existing:
        dal.register('voice_surface',
                     template='',
                     parameters=json.dumps(VOICE_CONFIG_V1),
                     created_by='anchor')

    if 'boot' not in existing:
        dal.register('boot',
                     template='',
                     parameters=json.dumps(BOOT_CONFIG_V1),
                     created_by='anchor')

    if 'pre_edit' not in existing:
        dal.register('pre_edit',
                     template='',
                     parameters=json.dumps(PRE_EDIT_CONFIG_V1),
                     created_by='anchor')

    if 'signal_assembler' not in existing:
        dal.register('signal_assembler',
                     template='',
                     parameters=json.dumps(SIGNAL_CONFIG_V1),
                     created_by='anchor')
