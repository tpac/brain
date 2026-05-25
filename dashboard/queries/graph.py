"""3D force graph data — nodes, edges, communities for the Graph tab."""

from ..db import brain_db_path, direct_query

# Bright saturated palette — communities pop on the dark background.
_COMMUNITY_PALETTE = [
    '#FF4444', '#00E5CC', '#2196F3', '#4CAF50', '#FFD600',
    '#E040FB', '#00BCD4', '#FF9800', '#9C27B0', '#03A9F4',
    '#FF5722', '#00E676', '#F44336', '#448AFF', '#CE93D8',
    '#26C6DA', '#FFAB40', '#5C6BC0', '#69F0AE', '#FF8A80',
    '#B388FF', '#64FFDA', '#FFE57F', '#82B1FF', '#FF80AB',
]

_TYPE_FALLBACK = {
    'lesson': '#4a9eff', 'correction': '#ff6666', 'interaction': '#33ff88',
    'rule': '#ffaa33', 'decision': '#aa66ff', 'mental_model': '#33dddd',
    'mechanism': '#dddd33', 'community': '#ffffff',
}


def query_graph3d():
    """All non-archived nodes + community-aware coloring + deduped edges."""
    db = brain_db_path()

    rows = direct_query(
        "SELECT id, type, title, locked, confidence, access_count, "
        "encoding_source, created_at, emotion, critical "
        "FROM nodes WHERE archived = 0",
        db_path=db,
    )

    # Community membership from active community_member edges
    comm_edges = direct_query(
        "SELECT e.source_id, e.target_id, n.title "
        "FROM edges e "
        "JOIN edge_relations er ON er.edge_id = e.edge_id "
        "JOIN nodes n ON n.id = e.source_id AND n.type = 'community' AND n.archived = 0 "
        "WHERE er.relation = 'community_member' "
        "AND er.archived = 0",
        db_path=db,
    )
    member_to_community = {}
    community_titles = {}
    for src, tgt, title in comm_edges:
        member_to_community[tgt] = src
        community_titles[src] = title

    unique_comms = sorted(set(member_to_community.values()))
    comm_color = {c: _COMMUNITY_PALETTE[i % len(_COMMUNITY_PALETTE)]
                  for i, c in enumerate(unique_comms)}

    node_ids = set()
    nodes = []
    for r in rows:
        nid, ntype = r[0], r[1]
        node_ids.add(nid)
        comm = member_to_community.get(nid)
        is_comm = ntype == 'community'
        color = comm_color.get(comm, comm_color.get(nid, _TYPE_FALLBACK.get(ntype, '#555')))
        if is_comm:
            color = comm_color.get(nid, '#ffffff')
        nodes.append({
            "id": nid,
            "name": (r[2] or nid[:8])[:80],
            "type": ntype,
            "locked": bool(r[3]),
            "confidence": r[4] or 1.0,
            "access_count": r[5] or 1,
            "created_at": r[7],
            "community": comm,
            "community_title": community_titles.get(comm, ''),
            "color": color,
            "val": (1.5 if not is_comm
                    else max(8, len([m for m, c in member_to_community.items() if c == nid]) * 0.6)),
            "hub": is_comm,
        })

    edges = []
    if node_ids:
        placeholders = ",".join("?" * len(node_ids))
        id_list = list(node_ids)
        edge_rows = direct_query(
            "SELECT e.source_id, e.target_id, er.relation, e.weight "
            "FROM edges e "
            "JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE e.source_id IN (%s) AND e.target_id IN (%s) "
            "AND er.archived = 0" % (placeholders, placeholders),
            id_list * 2, db_path=db,
        )
        # Dedupe — keep the strongest relation per (src, tgt) pair.
        seen = {}
        for src, tgt, rel, w in edge_rows:
            key = src + ':' + tgt
            if key not in seen or w > seen[key][3]:
                seen[key] = (src, tgt, rel, w)
        edges = [
            {"source": v[0], "target": v[1], "relation": v[2], "weight": v[3]}
            for v in seen.values()
        ]

    communities = []
    for cid in unique_comms:
        title = community_titles.get(cid, 'Community')
        member_count = len([m for m, c in member_to_community.items() if c == cid])
        communities.append({
            "id": cid, "hub_id": cid, "name": title[:60],
            "color": comm_color.get(cid, '#555'), "count": member_count,
        })

    return {
        "nodes": nodes, "edges": edges, "communities": communities,
        "stats": {"nodes": len(nodes), "edges": len(edges), "communities": len(communities)},
    }
