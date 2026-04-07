"""Pipeline Contract — shared constants and utilities for the recall/encoding pipeline.

This file contains cross-boundary constants used by multiple stages.
Boundary-specific logic lives in dedicated contracts:
- scales/s1/surface_contract.py — S1 Surface (relevance surfacing, formatting, enrichment)
- scales/s1/encode_contract.py — S1 turn encoder (Sonnet config, node formatting, catalog)

Key names are re-exported here for convenience. Boundary-specific code should
import from the specific contract directly.

contract.py defines what fields a node HAS.
pipeline_contract.py defines what fields FLOW at each stage.
"""

from .scales.s1.surface_contract import _relative_time  # noqa: F401 — canonical definition in surface_contract


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
    from .scales.s1.surface_contract import NEIGHBOR_TRUNCATION
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
    from .scales.s1.surface_contract import NEIGHBOR_TRUNCATION
    t = NEIGHBOR_TRUNCATION
    return "\"%s\" (%s, id:%s)" % (
        str(nb.get("title", ""))[:t['d2_title']],
        nb.get("type", "?"),
        str(nb.get("id", ""))[:t['d2_id']],
    )


# ═══════════════════════════════════════════════════════════════
# NODE REFERENCE — the bare minimum for any node mention
# ═══════════════════════════════════════════════════════════════
#
# Whenever a node appears as a connection, correction, neighbor,
# or any reference — it carries at least these fields.

NODE_REF_FIELDS = ('id', 'type', 'title', 'confidence', 'locked', 'created_at', 'revised_at')

EDGE_REF_FIELDS = ('relation', 'weight', 'description')

# Connections = NODE_REF_FIELDS + EDGE_REF_FIELDS
# Corrections = NODE_REF_FIELDS + direction + content
# Traverse neighbors = NODE_REF_FIELDS + EDGE_REF_FIELDS + content (truncated) + seed_id


# ═══════════════════════════════════════════════════════════════
# TRAVERSE — the enriched cluster atom
# ═══════════════════════════════════════════════════════════════
#
# Given seed node IDs, light them up: follow edges (excluding co_accessed),
# attach correction chains, attach metadata. One function, every consumer.
#
# Callers: boot (S0), MCP recall, S1 Surface, get_node/get_nodes.

TRAVERSE_EXCLUDED_EDGES = {'co_accessed', 'emergent_bridge'}


def traverse(brain, seed_ids, depth=1, limit_per_seed=3):
    """The atom: seed IDs → enriched cluster.

    Returns: {
        'neighbors': [{id, type, title, content, edge_type, edge_weight,
                        edge_description, confidence, seed_id}, ...],
        'corrections': {node_id: [{id, title, direction}, ...]},
        'metadata': {node_id: {key: value, ...}},
    }
    """
    from .dal import NodeDAL
    from .dal_metadata import MetadataDAL
    from .scales.s1.surface_contract import correction_enrich

    conn = brain.conn
    ndal = NodeDAL(conn)
    mdal = MetadataDAL(conn)

    # Resolve short IDs
    resolved_ids = set()
    for sid in seed_ids:
        full = ndal.resolve_id(sid) if len(str(sid)) < 16 else sid
        if full:
            resolved_ids.add(full)

    if not resolved_ids:
        return {'neighbors': [], 'corrections': {}, 'metadata': {}}

    # ── Graph expansion (exclude co_accessed + emergent_bridge) ──
    seen = set(resolved_ids)
    neighbors = []
    excluded = TRAVERSE_EXCLUDED_EDGES
    excl_placeholders = ','.join('?' for _ in excluded)

    for full_id in resolved_ids:
        rows = conn.execute("""
            SELECT n.id, n.type, n.title, substr(n.content, 1, 300),
                   e.edge_type, e.weight, e.description,
                   n.confidence, n.locked, n.created_at, n.revised_at
            FROM edges e
            JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id = ? OR e.target_id = ?) AND n.archived = 0
            AND n.id != ?
            AND e.edge_type NOT IN ({excl})
            ORDER BY e.weight DESC LIMIT ?
        """.format(excl=excl_placeholders),
            [full_id, full_id, full_id, full_id] + list(excluded) + [limit_per_seed]).fetchall()

        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                neighbors.append({
                    "id": r[0], "type": r[1], "title": r[2],
                    "content": r[3], "edge_type": r[4],
                    "edge_weight": r[5], "edge_description": r[6] or "",
                    "confidence": r[7], "locked": r[8] == 1,
                    "created_at": r[9], "revised_at": r[10],
                    "seed_id": full_id,
                })

    # ── Correction chains (seeds + neighbors) ──
    all_ids = set()
    for sid in resolved_ids:
        all_ids.add(sid)
        all_ids.add(sid[:8])
    for nb in neighbors:
        all_ids.add(nb['id'])
        all_ids.add(nb['id'][:8])

    corrections = correction_enrich(all_ids, conn)

    # ── Metadata for seeds ──
    metadata = {}
    for sid in resolved_ids:
        meta = mdal.get(sid)
        if meta:
            metadata[sid] = meta
            metadata[sid[:8]] = meta

    return {
        'neighbors': neighbors,
        'corrections': corrections,
        'metadata': metadata,
    }


def get_rich_node(brain_or_conn, node_id):
    """One node, fully assembled: content + metadata + correction chain + light connections.

    - Full content and all metadata on the node itself
    - Correction chain: follows corrects/corrected_by with full content
    - Connections: titles only (id, type, title, relation, weight) — no expansion

    Accepts brain object or raw db connection.
    """
    from .dal import NodeDAL
    from .dal_metadata import MetadataDAL
    from .scales.s1.surface_contract import correction_enrich

    conn = getattr(brain_or_conn, 'conn', brain_or_conn)
    ndal = NodeDAL(conn)
    mdal = MetadataDAL(conn)

    # Resolve short ID
    full_id = ndal.resolve_id(node_id) if len(str(node_id)) < 16 else node_id
    if not full_id:
        return None

    node = ndal.get_node(full_id)
    if not node:
        return None

    # ── Metadata ──
    meta = mdal.get(full_id)
    if meta:
        node['_metadata'] = meta

    # ── Situation (stored in node_embeddings, not nodes) ──
    sit = conn.execute(
        "SELECT situation_text FROM node_embeddings WHERE node_id = ?",
        (full_id,)).fetchone()
    if sit and sit[0]:
        node['situation'] = sit[0]

    # ── Correction chain (both directions, with content) ──
    corrections = correction_enrich({full_id, full_id[:8]}, conn)
    node_corrs = corrections.get(full_id, []) or corrections.get(full_id[:8], [])
    if node_corrs:
        # Fetch full content for each correction node
        for corr in node_corrs:
            corr_node = ndal.get_node(ndal.resolve_id(corr['id']) or corr['id'])
            if corr_node:
                corr['content'] = corr_node.get('content', '')
                corr['type'] = corr_node.get('type', '')
        node['_corrections'] = node_corrs

    # ── Light connections (titles + dates + description, exclude co_accessed) ──
    rows = conn.execute("""
        SELECT n.id, n.type, n.title, e.edge_type, e.weight, e.description,
               n.created_at, n.revised_at, n.confidence, n.locked
        FROM edges e
        JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
        WHERE (e.source_id = ? OR e.target_id = ?) AND n.archived = 0
        AND n.id != ?
        AND e.edge_type NOT IN ('co_accessed', 'emergent_bridge')
        ORDER BY e.weight DESC LIMIT 10
    """, [full_id, full_id, full_id, full_id]).fetchall()

    node['connections'] = [
        {"id": r[0], "type": r[1], "title": r[2], "relation": r[3], "weight": r[4],
         "description": r[5] or "", "created_at": r[6], "revised_at": r[7],
         "confidence": r[8], "locked": r[9] == 1}
        for r in rows
    ]

    return node


# ═══════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — re-exports from split contracts
# New code should import from the specific contract directly.
# ═══════════════════════════════════════════════════════════════

from .scales.s1.surface_contract import (  # noqa: E402, F401
    SURFACE,
    CANDIDATES_FILE,
    NEIGHBOR_D1_FIELDS,
    NEIGHBOR_D2_FIELDS,
    NEIGHBOR_D3_FIELDS,
    NEIGHBOR_TRUNCATION,
    PRECISION,
    format_candidate_for_surface,
    build_surface_prompt,
    format_surface_output,
    enrich_candidate_metadata,
    correction_enrich,
)
# Backward compat aliases
JUDGE = SURFACE
format_candidate_for_judge = format_candidate_for_surface
build_judge_prompt = build_surface_prompt
format_judge_output = format_surface_output

from .scales.s1.encode_contract import (  # noqa: E402, F401
    ENCODING_AGENT,
    format_node_for_encoder,
    build_encoder_node_catalog,
)
