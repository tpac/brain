"""Pipeline Contract — shared constants and utilities for the recall/encoding pipeline.

This file contains cross-boundary constants used by multiple stages.
Boundary-specific logic lives in dedicated contracts:
- scales/s1/recall_contract.py — S1 recall judge (Haiku selection, formatting, enrichment)
- scales/s1/encode_contract.py — S1 turn encoder (Sonnet config, node formatting, catalog)

Key names are re-exported here for convenience. Boundary-specific code should
import from the specific contract directly.

contract.py defines what fields a node HAS.
pipeline_contract.py defines what fields FLOW at each stage.
"""

from .scales.s1.recall_contract import _relative_time  # noqa: F401 — canonical definition in judge_contract


# ═══════════════════════════════════════════════════════════════
# NODE FIELDS — what to include at each stage
# ═══════════════════════════════════════════════════════════════

# Core fields present in every pipeline stage
NODE_CORE_FIELDS = {
    'id', 'type', 'title', 'content', 'confidence', 'locked',
    'revised_at', 'created_at',
}

# Additional fields for richer contexts
NODE_EXTENDED_FIELDS = NODE_CORE_FIELDS | {
    'access_count', 'encoding_source', 'content_summary',
    'emotion', 'emotion_label', 'updated_at',
}


# ═══════════════════════════════════════════════════════════════
# EMBEDDING GROUPS — multi-vector architecture for recall
# ═══════════════════════════════════════════════════════════════

EMBEDDING_GROUPS = {
    # Group 1: Title — the diagnostic pointer. Always exists. Highest weight.
    'title': {
        'weight': 1.00,
        'fields': ['title'],
        'vector_type': 'title',
        'always_compute': True,
    },
    # Group 2: Blend — existing title+content embedding. Lives in node_embeddings.
    'blend': {
        'weight': 0.85,
        'fields': ['title', 'content'],
        'vector_type': '_primary',
        'always_compute': True,
    },
    # Group 3: High-priority metadata — when is this relevant + who said it.
    'high_meta': {
        'weight': 0.70,
        'fields': ['situation', 'user_raw_quote', 'anchor_raw_quote'],
        'vector_type': 'high_meta',
        'always_compute': False,
    },
    # Group 4: Other metadata — why was this stored + behavioral patterns + emergent.
    'other_meta': {
        'weight': 0.40,
        'fields': ['reasoning', 'correction_pattern', 'source_context', '_emergent'],
        'vector_type': 'other_meta',
        'always_compute': False,
    },
}

# Scoring method for combining group vectors
EMBEDDING_SCORING_METHOD = 'top2_avg'

# KV fields to skip when building embedding text (not semantic content)
EMBEDDING_SKIP_FIELDS = {
    'metadata_created_at', 'validation_count', 'last_validated',
    'alternatives', 'change_impacts',
}

# Max chars per field when building group embedding text
EMBEDDING_FIELD_CHAR_LIMIT = 300


def get_group_fields(group_name):
    """Get the field names for an embedding group. Used by remember() and revise()."""
    group = EMBEDDING_GROUPS.get(group_name, {})
    return [f for f in group.get('fields', []) if f != '_emergent']


def get_group_weight(vector_type):
    """Get the z-index weight for a vector type. Used by recall scoring."""
    for group in EMBEDDING_GROUPS.values():
        if group.get('vector_type') == vector_type:
            return group['weight']
    return EMBEDDING_GROUPS['other_meta']['weight']  # default for unknown types


# ═══════════════════════════════════════════════════════════════
# TRUNCATION LIMITS — per stage, per field
# ═══════════════════════════════════════════════════════════════

PIPELINE = {
    'user_message_store': 500,
    'user_message_query': 500,
    'assistant_response_store': 4000,
    'recent_message_content': 300,
    'recall_log_query': 500,
    'recall_log_title': 80,
    'recall_log_snippet': 150,
    'encoding_state_compat': 2000,
}


# ═══════════════════════════════════════════════════════════════
# SURFACE CONFIG — MCP output and pre-edit suggestions
# ═══════════════════════════════════════════════════════════════

# MCP tool output (direct recall by Claude)
MCP_OUTPUT = {
    'content_limit': None,
    'max_results': 20,
    'enrich_top_n': 3,
}

# Pre-edit suggestions
PRE_EDIT = {
    'title_limit': 80,
    'content_limit_engineering': 350,
    'content_limit_code': 350,
    'content_limit_other': 250,
    'content_limit_impact': 300,
}


# ═══════════════════════════════════════════════════════════════
# SHARED FORMATTERS — used across boundaries
# ═══════════════════════════════════════════════════════════════

def format_node_header(node, id_length=8):
    """Standard one-line node header used across all stages."""
    locked = "LOCKED " if node.get("locked") else ""
    return "[%s] %s%s (id:%s, conf:%.2f, revised:%s, created:%s)" % (
        node.get("type", "?"),
        locked,
        node.get("title", "?"),
        str(node.get("id", ""))[:id_length],
        node.get("confidence") or 0,
        node.get("revised_at") or "never",
        str(node.get("created_at") or "")[:10],
    )


def format_neighbor_d1(nb):
    """Standard degree-1 neighbor line."""
    from .scales.s1.recall_contract import NEIGHBOR_TRUNCATION
    t = NEIGHBOR_TRUNCATION
    locked = "LOCKED " if nb.get("locked") else ""
    line = "  → %s: %s\"%s\" (%s, id:%s, conf:%.2f, revised:%s)" % (
        nb.get("relation", "related"),
        locked,
        str(nb.get("title", ""))[:t['d1_title']],
        nb.get("type", "?"),
        str(nb.get("id", ""))[:t['d1_id']],
        nb.get("confidence") or 0,
        nb.get("revised_at") or "never",
    )
    summary = nb.get("content_summary") or ""
    if summary:
        line += "\n      %s" % summary[:t['d1_content_summary']]
    return line


def format_neighbor_d2(nb):
    """Standard degree-2 neighbor breadcrumb."""
    from .scales.s1.recall_contract import NEIGHBOR_TRUNCATION
    t = NEIGHBOR_TRUNCATION
    return "\"%s\" (%s, id:%s)" % (
        str(nb.get("title", ""))[:t['d2_title']],
        nb.get("type", "?"),
        str(nb.get("id", ""))[:t['d2_id']],
    )


# ═══════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — re-exports from split contracts
# New code should import from the specific contract directly.
# ═══════════════════════════════════════════════════════════════

from .scales.s1.recall_contract import (  # noqa: E402, F401
    JUDGE,
    CANDIDATES_FILE,
    NEIGHBOR_D1_FIELDS,
    NEIGHBOR_D2_FIELDS,
    NEIGHBOR_D3_FIELDS,
    NEIGHBOR_TRUNCATION,
    PRECISION,
    format_candidate_for_judge,
    build_judge_prompt,
    format_judge_output,
    enrich_candidate_metadata,
    correction_enrich,
)

from .scales.s1.encode_contract import (  # noqa: E402, F401
    ENCODING_AGENT,
    format_node_for_encoder,
    build_encoder_node_catalog,
)
