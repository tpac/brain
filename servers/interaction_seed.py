"""Seed interactions table with v1 from current hardcoded values.

Run on first boot or when interactions table is empty.
The current code IS v1. After seeding, code reads from the table.
"""
import json
import os


# ── Surface v1: the instruction portion of the prompt ──
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


# ── S2 Community Enrichment prompt ──
def _load_community_enrichment_prompt():
    """Load S2CE prompt from file."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'scales', 's2', 'community_enrichment_prompt.py')
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('ce_prompt', prompt_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.PROMPT
    except Exception:
        return ''

_S2_COMMUNITY_ENRICHMENT_PROMPT_LEGACY = """You evaluate community proposals from a knowledge graph that stores memories from a collaboration between an operator (Tom) and an AI assistant (Anchor).

Communities are clusters of nodes that participate in the same *story* — correction chains evolving an idea, implementation dependencies building a system, lessons learned from a shared experience. A community is NOT just topical grouping ("all nodes about X") — it's a narrative structure where the relationships between nodes matter as much as the nodes themselves.

## What You Receive

Each proposal was detected by analyzing three signal types:
- **Semantic**: embedding similarity between nodes (topical affinity)
- **Relational**: typed edge patterns between nodes (corrects, extends, depends_on, etc.)
- **Usage**: co-surfacing patterns from recall traces (nodes that were retrieved together in real conversations)

## Node Structure

Each node has:
- **type**: decision, rule, lesson, correction, mechanism, concept, etc.
- **title**: short, scannable name
- **content**: rich explanation with reasoning and context
- **situation**: WHEN this node is relevant (has its own embedding for recall)
- **edges**: typed relationships to other nodes (relation + description). Multi-relation: one pair can have "corrects" AND "extends"
- **confidence**: 0-1 system reliability score
- **locked**: operator-confirmed important

Edge relation types are open text. Common: corrects, extends, depends_on, implements, caused_by, enables, validates, contradicts, supersedes, refines. But any descriptive relation is valid.

## Proposal Types

**NEW COMMUNITY**: A cluster of nodes that form a coherent story. You must name it, describe it, and explain when it's relevant for recall.

**PLACE NODE**: A node that belongs in an existing community. Confirm the placement. If it also belongs in another community (overlap), state the secondary membership with a reason — what story does it participate in for each?

**SPLIT/MERGE**: Communities whose structure has changed. Evaluate whether the change is meaningful or noise.

**RECALL SIGNAL**: A diagnostic about recall quality. Confirm the diagnosis and suggest what to fix.

## What Good Community Characterization Looks Like

BAD: "Nodes about hooks and timeouts"
GOOD: "The evolution of hook architecture from blocking to async. Started with 30s timeouts causing daemon hangs, corrected through os._exit(0) for instant exit, then extended with encoding write-lock awareness. The correction chain shows the team discovering that hook latency was a symptom of daemon thread safety."

BAD: "Collection of recall-related nodes"
GOOD: "The recall quality journey — from brute-force cosine scan to z-weighted multi-vector scoring. Each decision node corrects the previous approach. The dependency chain shows how title embeddings, situation embeddings, and synaptic fatigue were layered in response to specific failure cases. When working on recall quality, this community provides the full decision history."

The description should tell a STORY, not list topics. The situation should scope WHEN this community matters for recall.

## Overlap

A node can belong to multiple communities when it genuinely participates in different stories. "Hook timeout causing encoding delay" belongs to BOTH the hook architecture story (where it's a symptom) AND the encoding performance story (where it's a cause). The overlap reason should state the different role the node plays in each community.

Don't create overlaps for loose topical similarity. Two nodes mentioning "daemon" doesn't make them overlap candidates. The overlap must be relational — the node has typed edges into both communities that tell different stories.

## Rejection

Reject proposals when:
- Co-surfacing is coincidental (both recalled for meta-questions, not because they're related)
- The cluster has no relational coherence (just topically similar nodes with no meaningful edges)
- A split/merge is noise from a single session's activity, not structural change

## Response Format

Return ONLY a JSON array:
```json
[
  {"id": 1, "action": "accept", "title": "Hook Architecture Evolution",
   "content": "The evolution of hook architecture from blocking to async...",
   "situation": "When working on hook latency, daemon stability, or encoding delays caused by hook execution",
   "keywords": "hook timeout daemon async blocking os_exit encoding delay"},

  {"id": 2, "action": "place", "primary": "Hook Architecture Evolution",
   "secondary": ["Encoding Performance"], "overlap_reason": "Acts as symptom in hook story, cause in encoding story"},

  {"id": 3, "action": "reject", "reason": "Co-surfacing is coincidental — both recalled for meta-questions about the brain, not topically related"},

  {"id": 4, "action": "confirm_signal", "diagnosis": "Node embedding is in wrong region", "remediation": "Re-embed with updated content"}
]
```"""

S2_COMMUNITY_ENRICHMENT_PROMPT = _load_community_enrichment_prompt() or _S2_COMMUNITY_ENRICHMENT_PROMPT_LEGACY


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


SURFACE_CONFIG_V1 = {
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
    "max_messages": 20,
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

    if 'surface' not in existing and 'judge' not in existing:
        dal.register('surface',
                     template=SURFACE_PROMPT_V1,
                     parameters=json.dumps(SURFACE_CONFIG_V1),
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

    if 's2_community' not in existing:
        from .scales.s2.community_contract import COMMUNITY_DETECTION
        dal.register('s2_community',
                     template='',
                     parameters=json.dumps(COMMUNITY_DETECTION),
                     created_by='s2:community_detection')

    if 's2_edge_families' not in existing:
        # Seed with initial classification of edge types into semantic families.
        # The parameters field IS the mapping: {family_name: [relation_types]}
        # S2 EdgeFamilyIntegration updates when new types appear.
        # The template field is the LLM prompt for classifying new types.
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

    if 's2_community_enrichment' not in existing:
        from .scales.s2.community_contract import COMMUNITY_ENRICHMENT
        dal.register('s2_community_enrichment',
                     template=S2_COMMUNITY_ENRICHMENT_PROMPT,
                     parameters=json.dumps(COMMUNITY_ENRICHMENT),
                     created_by='s2:community_detection')
