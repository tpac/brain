"""Community structural fields — derived from community_member edges.

Five community fields are pure arithmetic over the graph, NOT judgment:
  community_size, community_members, community_internal_fraction,
  community_is_corridor, community_dominant_type.

They are computed here — never authored by the encoder LLM — and stamped
onto community nodes as a second, *algorithmic* Δ in CommunityEncoder.run()
(after the agent's judgment Δ and after reconcile, so the edges are final).

`structural_metrics` is the SINGLE shared definition of the internal/external
cohesion formula: the decoder calls it for its own fresh computation and this
helper calls it when stamping, so a stamped value can never disagree with the
decoder's. The corridor thresholds live in community_contract for the same
reason. See docs/COMMUNITY-METADATA-DENORMALIZATION.md.
"""

from collections import Counter, defaultdict

from .community_contract import (
    CORRIDOR_INTERNAL_FRACTION_MAX, CORRIDOR_MIN_SIZE,
    ADJACENCY_EXCLUDED_RELATIONS, ADJACENCY_SKIP_ASPECTS)


# Aspect names whose edges never count as cohesion (shared with the decoder
# via the contract; mirrors community_decoder._build_typed_adjacency).
_SKIP_FAMS = frozenset(ADJACENCY_SKIP_ASPECTS)
_SCOPED_BIND_LIMIT = 800  # above this, scan whole-graph instead of an IN-scope


def structural_metrics(members, edges_by_node):
    """Internal/external edge counts, internal_fraction, is_corridor for one
    community's member set over a typed-edge adjacency.

    `edges_by_node` is the symmetric adjacency the decoder builds: each
    surviving typed edge_relation appears in BOTH endpoints' lists as
    (neighbor, aspect, relation), with community/archived nodes and
    noise/structural relations already excluded. Internal edges are
    double-counted (//2); external are counted once from the member side.

    internal_fraction is RAW (unrounded) — callers round only at the
    display/stamp boundary, so the decoder's exact threshold comparisons are
    preserved. This is the single source of the cohesion formula; both the
    decoder and the structural stamp route through it.
    """
    ms = members if isinstance(members, set) else set(members)
    internal = sum(1 for n in ms
                   for nbr, _, _ in edges_by_node.get(n, [])
                   if nbr in ms) // 2
    external = sum(1 for n in ms
                   for nbr, _, _ in edges_by_node.get(n, [])
                   if nbr not in ms)
    int_frac = internal / (internal + external) if (internal + external) else 0.0
    is_corridor = (int_frac < CORRIDOR_INTERNAL_FRACTION_MAX
                   and len(ms) > CORRIDOR_MIN_SIZE)
    return {
        'internal': internal,
        'external': external,
        'internal_fraction': int_frac,
        'is_corridor': is_corridor,
    }


def build_member_adjacency(brain, member_ids=None):
    """Typed-edge adjacency identical to the decoder's, optionally scoped.

    Mirrors community_decoder._build_typed_adjacency exactly: excludes
    archived/community nodes, the three non-cohesion relations, and any
    relation whose aspect is noise/generic_relation. Symmetric — each
    surviving edge_relation row is added to BOTH endpoints, so a multi-relation
    pair is counted once per relation, matching the decoder.

    member_ids scopes the scan to edges incident to those nodes (cheap for the
    per-encode stamp) when the set is within SQLite's bind limit; otherwise the
    whole-graph scan runs (the one-time fill). Extra non-member adjacency is
    harmless — structural_metrics only walks member nodes' lists.
    """
    rel_to_fam = {
        relation: name
        for name, aspect in brain.aspects.all().items()
        for relation in aspect.edge_relations
    }

    excl = ','.join('?' * len(ADJACENCY_EXCLUDED_RELATIONS))
    sql = """
        SELECT e.source_id, e.target_id, er.relation
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        JOIN nodes ns ON ns.id = e.source_id AND ns.archived = 0
            AND ns.type != 'community'
        JOIN nodes nt ON nt.id = e.target_id AND nt.archived = 0
            AND nt.type != 'community'
        WHERE er.archived = 0
        AND er.relation NOT IN (%s)
    """ % excl
    params = tuple(ADJACENCY_EXCLUDED_RELATIONS)
    ids = [m for m in (member_ids or []) if m]
    if ids and len(ids) <= _SCOPED_BIND_LIMIT:
        ph = ','.join('?' * len(ids))
        sql += " AND (e.source_id IN (%s) OR e.target_id IN (%s))" % (ph, ph)
        params = (*params, *ids, *ids)

    edges_by_node = defaultdict(list)
    for src, tgt, rel in brain.conn.execute(sql, params).fetchall():
        fam = rel_to_fam.get(rel, 'unclassified')
        if fam in _SKIP_FAMS:
            continue
        edges_by_node[src].append((tgt, fam, rel))
        edges_by_node[tgt].append((src, fam, rel))
    return edges_by_node


def compute_community_structural(brain, community_ids):
    """Derive the structural fields for each community from its member edges.

    Returns {community_id: {
        'community_size': int,
        'community_internal_fraction': float (rounded 3dp),
        'community_is_corridor': bool,
        'community_dominant_type': str | None,
    }}. Communities with no live member edges get size 0.

    Pure arithmetic over community_member edges + the typed adjacency — no LLM.
    Members are the LIVE ones (archived nodes are not members). internal_fraction
    matches the decoder's fresh computation exactly — archived members carry no
    edges in the (archived-excluding) adjacency, so they can't shift it; only
    `size`/`is_corridor`'s member count uses the live set. (The stored
    `community_members` list stays agent-authored — the orphan-recovery seed for
    reconcile — and is deliberately not derived here.)
    """
    ids = [c for c in (community_ids or []) if c]
    if not ids:
        return {}

    members_by_comm = brain._graph.get_members_bulk(ids)
    union = {m['id'] for ms in members_by_comm.values() for m in ms}
    edges_by_node = build_member_adjacency(brain, union if union else None)

    out = {}
    for cid in ids:
        members = members_by_comm.get(cid, [])
        ms = {m['id'] for m in members}
        metrics = structural_metrics(ms, edges_by_node)
        types = [m.get('type') for m in members if m.get('type')]
        if types:
            counts = Counter(types)
            # Deterministic: highest count, ties broken lexically.
            dominant = min(counts, key=lambda t: (-counts[t], t))
        else:
            dominant = None
        out[cid] = {
            'community_size': len(ms),
            'community_internal_fraction': round(metrics['internal_fraction'], 3),
            'community_is_corridor': metrics['is_corridor'],
            'community_dominant_type': dominant,
        }
    return out
