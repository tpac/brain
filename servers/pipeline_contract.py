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
                   er.relation, e.weight, er.description,
                   n.confidence, n.locked, n.created_at, n.revised_at,
                   CASE WHEN e.source_id = ? THEN 'outgoing' ELSE 'incoming' END as direction
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            WHERE (e.source_id = ? OR e.target_id = ?) AND n.archived = 0
            AND n.id != ?
            AND er.relation NOT IN ({excl})
            ORDER BY e.weight DESC LIMIT ?
        """.format(excl=excl_placeholders),
            [full_id, full_id, full_id, full_id, full_id] + list(excluded) + [limit_per_seed]).fetchall()

        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                neighbors.append({
                    "id": r[0], "type": r[1], "title": r[2],
                    "content": r[3], "edge_type": r[4],
                    "edge_weight": r[5], "edge_description": r[6] or "",
                    "confidence": r[7], "locked": r[8] == 1,
                    "created_at": r[9], "revised_at": r[10],
                    "direction": r[11],
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


def get_rich_node(brain_or_conn, node_id_or_ids):
    """Fully assembled node(s): content + metadata + correction chain + connections.

    Accepts a single node_id (str) or a list of node_ids.
    - Single ID → returns one rich node dict, or None if not found.
    - List of IDs → returns dict {node_id: rich_node_dict}. Missing nodes omitted.

    When given a list, uses batched queries (5 queries total instead of N×4).
    """
    from .dal import NodeDAL
    from .dal_metadata import MetadataDAL
    from .scales.s1.surface_contract import correction_enrich

    conn = getattr(brain_or_conn, 'conn', brain_or_conn)
    ndal = NodeDAL(conn)

    # ── Dispatch: single vs batch ──
    single = isinstance(node_id_or_ids, str)
    raw_ids = [node_id_or_ids] if single else list(node_id_or_ids)

    if not raw_ids:
        return None if single else {}

    # Resolve short IDs
    full_ids = []
    for nid in raw_ids:
        full = ndal.resolve_id(nid) if len(str(nid)) < 16 else nid
        if full:
            full_ids.append(full)

    if not full_ids:
        return None if single else {}

    # ── 1. Batch fetch all nodes ──
    ph = ','.join('?' for _ in full_ids)
    cols = [desc[0] for desc in conn.execute('SELECT * FROM nodes LIMIT 0').description]
    rows = conn.execute(
        'SELECT * FROM nodes WHERE id IN (%s)' % ph, full_ids
    ).fetchall()

    nodes = {}
    for row in rows:
        d = dict(zip(cols, row))
        for bf in ('locked', 'archived', 'critical'):
            d[bf] = d.get(bf) == 1
        d['emotion'] = d.get('emotion') or 0
        d['emotion_label'] = d.get('emotion_label') or 'neutral'
        nodes[d['id']] = d

    if not nodes:
        return None if single else {}

    found_ids = list(nodes.keys())
    ph = ','.join('?' for _ in found_ids)

    # ── 2. Batch fetch all metadata ──
    meta_rows = conn.execute(
        'SELECT node_id, key, value FROM node_metadata_kv WHERE node_id IN (%s)' % ph,
        found_ids
    ).fetchall()
    meta_by_node = {}
    for nid, key, value in meta_rows:
        meta_by_node.setdefault(nid, {})[key] = value
    for nid in found_ids:
        if nid in meta_by_node:
            nodes[nid]['_metadata'] = meta_by_node[nid]

    # ── 3. Batch fetch all situations ──
    sit_rows = conn.execute(
        'SELECT node_id, situation_text FROM node_embeddings WHERE node_id IN (%s)' % ph,
        found_ids
    ).fetchall()
    for nid, sit in sit_rows:
        if sit:
            nodes[nid]['situation'] = sit

    # ── 4. Batch corrections (already set-based) ──
    all_ids_for_corrections = set()
    for fid in found_ids:
        all_ids_for_corrections.add(fid)
        all_ids_for_corrections.add(fid[:8])
    corrections = correction_enrich(all_ids_for_corrections, conn)
    for nid in found_ids:
        node_corrs = corrections.get(nid, []) or corrections.get(nid[:8], [])
        if node_corrs:
            for corr in node_corrs:
                corr_full = ndal.resolve_id(corr['id']) or corr['id']
                corr_node = ndal.get_node(corr_full)
                if corr_node:
                    corr['content'] = corr_node.get('content', '')
                    corr['type'] = corr_node.get('type', '')
            nodes[nid]['_corrections'] = node_corrs

    # ── 5. Batch fetch all connections ──
    # Read from edge_relations (source of truth) via edge_id JOIN.
    # Single-direction storage: query both directions, detect outgoing/incoming.
    edge_rows = conn.execute("""
        SELECT e.source_id, e.target_id, e.weight,
               er.relation, er.description, er.weight as rel_weight,
               n1.id, n1.type, n1.title, n1.created_at, n1.revised_at, n1.confidence, n1.locked,
               n2.id, n2.type, n2.title, n2.created_at, n2.revised_at, n2.confidence, n2.locked
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        JOIN nodes n1 ON n1.id = e.target_id
        JOIN nodes n2 ON n2.id = e.source_id
        WHERE (e.source_id IN ({ph}) OR e.target_id IN ({ph}))
        AND er.relation NOT IN ('co_accessed', 'emergent_bridge')
        AND n1.archived = 0 AND n2.archived = 0
    """.format(ph=ph), found_ids + found_ids).fetchall()

    # Group by (owner_node, neighbor_node) — collect all relations per neighbor
    edges_by_node = {}  # {owner_id: {neighbor_id: {node_data, relations: [...]}}}
    found_set = set(found_ids)
    for row in edge_rows:
        src, tgt = row[0], row[1]
        agg_weight = row[2]
        rel = row[3] or 'related'
        desc = row[4] or ''
        rel_weight = row[5] or agg_weight
        # n1 = target node, n2 = source node
        n1_data = {"id": row[6], "type": row[7], "title": row[8],
                   "created_at": row[9], "revised_at": row[10],
                   "confidence": row[11], "locked": row[12] == 1}
        n2_data = {"id": row[13], "type": row[14], "title": row[15],
                   "created_at": row[16], "revised_at": row[17],
                   "confidence": row[18], "locked": row[19] == 1}
        relation_entry = {"relation": rel, "description": desc, "weight": rel_weight}

        # For source node looking outward → neighbor is target (n1) — outgoing
        if src in found_set and tgt != src:
            key = (src, n1_data['id'])
            if key not in edges_by_node.setdefault(src, {}):
                edges_by_node[src][key] = {**n1_data, "weight": agg_weight,
                                            "direction": "outgoing", "relations": []}
            edges_by_node[src][key]["relations"].append(relation_entry)

        # For target node looking outward → neighbor is source (n2) — incoming
        if tgt in found_set and src != tgt:
            key = (tgt, n2_data['id'])
            if key not in edges_by_node.setdefault(tgt, {}):
                edges_by_node[tgt][key] = {**n2_data, "weight": agg_weight,
                                            "direction": "incoming", "relations": []}
            edges_by_node[tgt][key]["relations"].append(relation_entry)

    for nid in found_ids:
        conns = list(edges_by_node.get(nid, {}).values())
        # Sort by aggregate weight, set 'relation' to highest-weight relation for compat
        for c in conns:
            rels = sorted(c['relations'], key=lambda r: r.get('weight', 0), reverse=True)
            c['relations'] = rels
            c['relation'] = rels[0]['relation'] if rels else 'related'
            c['description'] = rels[0]['description'] if rels else ''
        conns.sort(key=lambda x: x.get('weight', 0), reverse=True)
        nodes[nid]['connections'] = conns

    # ── Return ──
    if single:
        return nodes.get(full_ids[0])
    return nodes


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
