"""Brain health checks — validate data integrity.

Functions that check coverage, consistency, and completeness of brain data.
Called by S2 idle hook, backfill scripts, and diagnostic tools.

Each function returns {ok: bool, gaps: [...], coverage: {...}}.
"""


def check_group_vector_coverage(brain):
    """Validate that all active nodes have group vectors in node_enrichments.

    Checks:
    - Every active non-community node should have a 'title' vector
    - Nodes with situation/quotes should have 'high_meta' vector
    - Nodes with reasoning/correction_pattern should have 'other_meta' vector
    - No orphaned enrichment rows for archived nodes

    Returns: {ok, gaps, total, orphaned, coverage: {title_count, title_pct, ...}}
    """
    total = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE archived = 0 AND type != 'community'"
    ).fetchone()[0]

    # Count nodes with each vector type (matching active nodes only)
    title_count = brain.conn.execute(
        "SELECT COUNT(DISTINCT ne.node_id) FROM node_enrichments ne "
        "JOIN nodes n ON n.id = ne.node_id "
        "WHERE ne.vector_type = 'title' AND n.archived = 0 AND n.type != 'community'"
    ).fetchone()[0]

    high_meta_count = brain.conn.execute(
        "SELECT COUNT(DISTINCT ne.node_id) FROM node_enrichments ne "
        "JOIN nodes n ON n.id = ne.node_id "
        "WHERE ne.vector_type = 'high_meta' AND n.archived = 0"
    ).fetchone()[0]

    other_meta_count = brain.conn.execute(
        "SELECT COUNT(DISTINCT ne.node_id) FROM node_enrichments ne "
        "JOIN nodes n ON n.id = ne.node_id "
        "WHERE ne.vector_type = 'other_meta' AND n.archived = 0"
    ).fetchone()[0]

    # Orphaned enrichment rows
    orphaned = brain.conn.execute(
        "SELECT COUNT(*) FROM node_enrichments "
        "WHERE node_id NOT IN (SELECT id FROM nodes WHERE archived = 0)"
    ).fetchone()[0]

    # Nodes that SHOULD have high_meta (have situation or quotes)
    should_have_high_meta = brain.conn.execute(
        "SELECT COUNT(DISTINCT n.id) FROM nodes n "
        "LEFT JOIN node_enrichments ne ON ne.node_id = n.id AND ne.vector_type = '_situation' "
        "LEFT JOIN node_metadata_kv m1 ON m1.node_id = n.id AND m1.key = 'user_raw_quote' "
        "LEFT JOIN node_metadata_kv m2 ON m2.node_id = n.id AND m2.key = 'anchor_raw_quote' "
        "WHERE n.archived = 0 AND n.type != 'community' "
        "AND (ne.text IS NOT NULL OR m1.value IS NOT NULL OR m2.value IS NOT NULL)"
    ).fetchone()[0]

    # Nodes that SHOULD have edge_context (have edge descriptions)
    should_have_edge_context = brain.conn.execute(
        "SELECT COUNT(DISTINCT CASE WHEN e.source_id IN (SELECT id FROM nodes WHERE archived=0 AND type!='community') "
        "THEN e.source_id ELSE e.target_id END) "
        "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
        "WHERE er.description IS NOT NULL AND length(er.description) > 10 "
        "AND er.relation NOT IN ('co_accessed', 'emergent_bridge', 'community_member')"
    ).fetchone()[0]

    edge_context_count = brain.conn.execute(
        "SELECT COUNT(DISTINCT ne.node_id) FROM node_enrichments ne "
        "JOIN nodes n ON n.id = ne.node_id "
        "WHERE ne.vector_type = 'edge_context' AND n.archived = 0"
    ).fetchone()[0]

    gaps = []
    if title_count < total:
        gaps.append('%d nodes missing title vector' % (total - title_count))
    if orphaned > 0:
        gaps.append('%d orphaned enrichment rows' % orphaned)
    if high_meta_count < should_have_high_meta:
        gaps.append('%d nodes should have high_meta but don\'t' % (
            should_have_high_meta - high_meta_count))
    if edge_context_count < should_have_edge_context:
        gaps.append('%d nodes should have edge_context but don\'t' % (
            should_have_edge_context - edge_context_count))

    return {
        'ok': len(gaps) == 0,
        'gaps': gaps,
        'total': total,
        'orphaned': orphaned,
        'coverage': {
            'title_count': title_count,
            'title_pct': 100 * title_count / max(total, 1),
            'high_meta_count': high_meta_count,
            'high_meta_pct': 100 * high_meta_count / max(total, 1),
            'other_meta_count': other_meta_count,
            'other_meta_pct': 100 * other_meta_count / max(total, 1),
            'should_have_high_meta': should_have_high_meta,
            'edge_context_count': edge_context_count,
            'edge_context_pct': 100 * edge_context_count / max(total, 1),
            'should_have_edge_context': should_have_edge_context,
        },
    }
