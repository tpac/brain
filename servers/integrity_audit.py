"""Deep integrity audit — runs during idle maintenance.

Returns a list of findings for the IDLE hook log:
duplicate-title clusters, emergent types, cold zones, isolated nodes,
revision stats, edge-type imbalance, sparse metadata.
"""

from .clock import iso_cutoff


# Types the system actively queries by name — these get special treatment
STRUCTURAL_TYPES = {
    'vocabulary', 'rule', 'decision', 'mechanism', 'lesson', 'impact',
    'convention', 'pattern', 'constraint', 'correction', 'purpose', 'tension',
}


def deep_integrity_audit(brain):
    """Full brain health audit — runs during idle maintenance.

    Returns a list of findings, each: {type, severity, message, details}
    """
    findings = []

    try:
        # 1. Duplicate detection — find nodes with very similar titles
        rows = brain.conn.execute("""
            SELECT SUBSTR(title, 1, 35) as prefix, COUNT(*) as cnt,
                   GROUP_CONCAT(id, ',') as ids
            FROM nodes WHERE archived=0
            GROUP BY prefix HAVING cnt >= 2
            ORDER BY cnt DESC LIMIT 20
        """).fetchall()
        for r in rows:
            findings.append({
                "type": "duplicate_cluster",
                "severity": "medium" if r[1] >= 4 else "low",
                "message": "\"%s...\" × %d nodes" % (r[0], r[1]),
                "node_ids": r[2].split(",")[:5],
            })

        # 2. Emergent types
        rows = brain.conn.execute("""
            SELECT type, COUNT(*) as cnt FROM nodes
            WHERE archived=0 GROUP BY type ORDER BY cnt DESC
        """).fetchall()
        for r in rows:
            if r[0] not in STRUCTURAL_TYPES and r[1] >= 5:
                findings.append({
                    "type": "emergent_type",
                    "severity": "info" if r[1] < 10 else "medium",
                    "message": "Type \"%s\" has %d nodes — no system behavior defined" % (r[0], r[1]),
                    "count": r[1],
                })

        # 3. Cold zones — nodes not accessed in 14+ days
        cold = brain.conn.execute("""
            SELECT COUNT(*) FROM nodes WHERE archived=0
            AND (last_accessed IS NULL OR last_accessed < ?)
        """, (iso_cutoff(days=14),)).fetchone()[0]
        total = brain.conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
        if total > 0:
            cold_pct = cold / total * 100
            if cold_pct > 30:
                findings.append({
                    "type": "cold_zone",
                    "severity": "medium",
                    "message": "%d of %d nodes (%.0f%%) not accessed in 14+ days" % (cold, total, cold_pct),
                })

        # 4. Isolated nodes (no edges)
        isolated = brain.conn.execute("""
            SELECT COUNT(*) FROM nodes n WHERE n.archived=0
            AND NOT EXISTS (SELECT 1 FROM edges WHERE source_id=n.id OR target_id=n.id)
        """).fetchone()[0]
        if isolated > 5:
            findings.append({
                "type": "isolated_nodes",
                "severity": "medium",
                "message": "%d nodes with zero connections — invisible to graph traversal" % isolated,
            })

        # 5. Revision stats
        revised = brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived=0 AND revised_at IS NOT NULL"
        ).fetchone()[0]
        findings.append({
            "type": "revision_stats",
            "severity": "info",
            "message": "%d of %d nodes revised (%.0f%%)" % (revised, total, revised / max(1, total) * 100),
        })

        # (Edge-type distribution check removed with the co_accessed
        # retirement — no relation family can dominate by mechanism anymore.)

        # 7. Metadata sparseness (via KV DAL)
        _meta_dal = brain._meta_kv
        meta_total = _meta_dal.total_nodes()
        if meta_total > 0:
            for field in ['reasoning', 'user_raw_quote', 'source_context']:
                filled = _meta_dal.nodes_with_field(field)
                pct = filled / meta_total * 100
                if pct < 30:
                    findings.append({
                        "type": "sparse_metadata",
                        "severity": "low",
                        "message": "metadata.%s: only %.0f%% filled (%d of %d)" % (field, pct, filled, meta_total),
                    })
        else:
            findings.append({
                "type": "no_metadata",
                "severity": "medium",
                "message": "No metadata in KV store — no reasoning, quotes, or corrections tracked",
            })

    except Exception as e:
        findings.append({"type": "audit_error", "severity": "high", "message": str(e)})

    return findings
