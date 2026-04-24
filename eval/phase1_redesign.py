#!/usr/bin/env python3
"""Phase 1 Recall Redesign: Recognition-Based Activation.

Wide activation → chain detection → loudest signal wins.
Two discovery channels, edge-family-aware chain scoring, standalone nodes compete.

Eval-only: does not modify brain_recall.py.
"""

import os
import sys
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from servers import embedder
from servers.recall_scoring import unified_score
from servers.brain_constants import (
    TITLE_MATCH_BOOST, CRITICAL_BOOST, NOISE_FLOOR_THRESHOLD,
    RELEVANCE_FLOOR_PRIMARY, RELEVANCE_FLOOR_ENRICHED,
)
from eval.phase1_contract import (
    QUERY, DISCOVERY, CHAIN, CHAIN_EDGE_FAMILY_WEIGHTS,
    DEFAULT_EDGE_WEIGHT, RANKING,
)


# ═══════════════════════════════════════════════════
# STEP 1: Query Understanding
# ═══════════════════════════════════════════════════

def step1_query_understanding(brain, user_message, session_id):
    """Produce community_vec and node_vec from conversation context.

    Returns: {
        'community_vec': np.array or None,
        'node_vec': np.array or None,
        'current_vec': bytes (raw embedding),
        'is_garbage': bool,
        'debug': {...}
    }
    """
    debug = {'turns_found': 0, 'community_turns_used': 0, 'node_turns_used': 0}

    # Embed current message
    current_raw = embedder.embed(user_message[:500])
    if not current_raw:
        return {'community_vec': None, 'node_vec': None, 'current_vec': None,
                'is_garbage': True, 'debug': debug}

    current_vec = np.frombuffer(current_raw, dtype=np.float32)

    # Garbage check: short message + low best community score
    is_garbage = False
    if len(user_message.strip()) < QUERY['garbage_max_chars']:
        community_embs = brain.conn.execute(
            "SELECT ne.embedding FROM node_embeddings ne "
            "JOIN nodes n ON n.id = ne.node_id "
            "WHERE n.type = 'community' AND n.archived = 0 AND ne.embedding IS NOT NULL"
        ).fetchall()
        best = 0.0
        for (blob,) in community_embs:
            sim = embedder.cosine_similarity(current_raw, blob)
            best = max(best, sim)
        if best < QUERY['garbage_floor']:
            is_garbage = True
            debug['garbage_best_community'] = round(best, 3)
            return {'community_vec': None, 'node_vec': None, 'current_vec': current_raw,
                    'is_garbage': True, 'debug': debug}

    # Get prior user messages from traces
    try:
        turns = brain._trace_dal.get_session_turns(session_id, limit=QUERY['community_window'] + 2)
        user_turns = [t for t in turns if t.get('role') == 'user']
        debug['turns_found'] = len(user_turns)
    except Exception:
        user_turns = []

    # Embed prior turns
    turn_vecs = [current_vec]  # current is always index 0
    for t in user_turns[:QUERY['community_window'] - 1]:
        text = (t.get('content') or '')[:500]
        if text and len(text.strip()) > 5:
            blob = embedder.embed(text)
            if blob:
                turn_vecs.append(np.frombuffer(blob, dtype=np.float32))

    # Community vector: blend up to 10 turns
    comm_weights = QUERY['community_weights'][:len(turn_vecs)]
    total_w = sum(comm_weights)
    comm_weights_norm = [w / total_w for w in comm_weights]
    community_vec = sum(w * v for w, v in zip(comm_weights_norm, turn_vecs))
    community_vec = community_vec / (np.linalg.norm(community_vec) + 1e-10)
    debug['community_turns_used'] = len(turn_vecs)

    # Node vector: blend up to 3 turns
    node_count = min(len(turn_vecs), QUERY['node_window'])
    node_weights = QUERY['node_weights'][:node_count]
    total_nw = sum(node_weights)
    node_weights_norm = [w / total_nw for w in node_weights]
    node_vec = sum(w * v for w, v in zip(node_weights_norm, turn_vecs[:node_count]))
    node_vec = node_vec / (np.linalg.norm(node_vec) + 1e-10)
    debug['node_turns_used'] = node_count

    return {
        'community_vec': community_vec,
        'node_vec': node_vec,
        'current_vec': current_raw,
        'is_garbage': False,
        'debug': debug,
    }


# ═══════════════════════════════════════════════════
# STEP 2: Candidate Discovery
# ═══════════════════════════════════════════════════

def _community_vec_as_bytes(vec):
    """Convert numpy vec to bytes for cosine_similarity."""
    return vec.astype(np.float32).tobytes()


def step2_discover(brain, community_vec, node_vec, current_vec):
    """Two-channel candidate discovery.

    Returns: {
        'candidates': {node_id: {score, source, community_id, ...}},
        'activated_communities': [(id, score, title)],
        'debug': {...}
    }
    """
    candidates = {}  # node_id → candidate dict
    debug = {'community_members': 0, 'node_direct': 0, 'overlap': 0}

    # 2a: Community channel
    comm_bytes = _community_vec_as_bytes(community_vec)
    community_rows = brain.conn.execute(
        "SELECT n.id, n.title, ne.embedding FROM nodes n "
        "JOIN node_embeddings ne ON ne.node_id = n.id "
        "WHERE n.type = 'community' AND n.archived = 0 AND ne.embedding IS NOT NULL"
    ).fetchall()

    activated_communities = []
    for cid, ctitle, cemb in community_rows:
        sim = embedder.cosine_similarity(comm_bytes, cemb)
        if sim >= DISCOVERY['community_floor']:
            activated_communities.append((cid, sim, ctitle))

    activated_communities.sort(key=lambda x: x[1], reverse=True)

    # Pull members of activated communities
    for cid, cscore, ctitle in activated_communities:
        member_rows = brain.conn.execute(
            "SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END "
            "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE (e.source_id = ? OR e.target_id = ?) "
            "AND er.relation = 'community_member' "
            "AND er.archived = 0",
            (cid, cid, cid)
        ).fetchall()
        for (mid,) in member_rows:
            if mid not in candidates:
                candidates[mid] = {
                    'score': 0, 'source': 'community',
                    'community_id': cid, 'community_score': cscore,
                }
            debug['community_members'] += 1

    # 2b: Node channel — cosine scan all non-community nodes
    node_bytes = _community_vec_as_bytes(node_vec)
    node_rows = brain.conn.execute(
        "SELECT ne.node_id, ne.embedding, ne.situation_embedding, "
        "n.title, n.type, n.confidence, n.critical, n.created_at, "
        "n.emotion, n.access_count "
        "FROM node_embeddings ne JOIN nodes n ON n.id = ne.node_id "
        "WHERE n.archived = 0 AND n.type != 'community'"
    ).fetchall()

    for row in node_rows:
        nid, emb, sit_emb = row[0], row[1], row[2]
        if not emb:
            continue
        sim = embedder.cosine_similarity(node_bytes, emb)
        if sim < DISCOVERY['node_floor']:
            continue

        # Unified score modulation
        score = unified_score(
            semantic_score=sim,
            created_at=row[7], emotion=row[8] or 0,
            access_count=row[9] or 0, confidence=row[5],
        )

        # Situation boost
        if sit_emb and current_vec:
            sit_sim = embedder.cosine_similarity(current_vec, sit_emb)
            if sit_sim >= DISCOVERY['situation_threshold']:
                score += DISCOVERY['situation_weight'] * sit_sim

        # Title match boost
        title = (row[3] or '').lower()
        # Use current message for title matching (not blended)
        # Approximate: just check first few query words
        # (Full implementation would parse user_message, but we don't have it here)

        if nid in candidates:
            # Node found by BOTH channels — keep max score, mark overlap
            if score > candidates[nid].get('score', 0):
                candidates[nid]['score'] = score
            candidates[nid]['source'] = 'both'
            debug['overlap'] += 1
        else:
            candidates[nid] = {
                'score': score, 'source': 'node',
                'community_id': None, 'community_score': 0,
            }
            debug['node_direct'] += 1

        # Store metadata for later steps
        candidates[nid].update({
            'title': row[3] or '', 'type': row[4] or '',
            'confidence': row[5], 'critical': row[6] or 0,
            'cosine': round(sim, 4),
        })

    # Score community-sourced candidates and apply member_floor
    member_floor = DISCOVERY['member_floor']
    for nid, cand in list(candidates.items()):
        if cand['source'] == 'community' and cand['score'] == 0:
            # Embed-score this community member against node_vec
            emb_row = brain.conn.execute(
                "SELECT ne.embedding, n.title, n.type, n.confidence, n.critical, "
                "n.created_at, n.emotion, n.access_count "
                "FROM node_embeddings ne JOIN nodes n ON n.id = ne.node_id "
                "WHERE ne.node_id = ?", (nid,)
            ).fetchone()
            if emb_row and emb_row[0]:
                sim = embedder.cosine_similarity(node_bytes, emb_row[0])
                if sim < member_floor:
                    del candidates[nid]
                    continue
                score = unified_score(
                    semantic_score=sim, created_at=emb_row[5],
                    emotion=emb_row[6] or 0, access_count=emb_row[7] or 0,
                    confidence=emb_row[3],
                )
                candidates[nid].update({
                    'score': score, 'cosine': round(sim, 4),
                    'title': emb_row[1] or '', 'type': emb_row[2] or '',
                    'confidence': emb_row[3], 'critical': emb_row[4] or 0,
                })
            else:
                del candidates[nid]

    debug['total_raw_pool'] = len(candidates)
    return {
        'candidates': candidates,
        'activated_communities': activated_communities,
        'debug': debug,
    }


# ═══════════════════════════════════════════════════
# STEP 3.5: Chain Detection
# ═══════════════════════════════════════════════════

def _load_edge_families(brain):
    """Load edge family classification from interactions table.
    Shape-agnostic (handles legacy list + new nested-dict via iter_families)."""
    from servers.scales.s2.edge_families import get_reverse_map
    config = brain.get_interaction_config('s2_edge_families')
    if not config:
        return {}
    rel_to_family = get_reverse_map(config)
    return rel_to_family


def _find_inter_candidate_edges(brain, candidate_ids):
    """Find all meaningful edges where BOTH endpoints are in candidate set.

    Returns: [(source_id, target_id, relation, weight)]
    """
    if len(candidate_ids) < 2:
        return []

    id_list = list(candidate_ids)
    ph = ','.join('?' * len(id_list))
    excluded = CHAIN['excluded_relations']
    excl_ph = ','.join('?' * len(excluded))

    rows = brain.conn.execute("""
        SELECT e.source_id, e.target_id, er.relation, e.weight
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE e.source_id IN ({ids}) AND e.target_id IN ({ids})
        AND er.relation NOT IN ({excl})
        AND er.archived = 0
    """.format(ids=ph, excl=excl_ph),
        id_list + id_list + list(excluded)
    ).fetchall()

    return [(r[0], r[1], r[2], r[3] or 0.5) for r in rows]


def _find_connected_components(edges, node_ids):
    """Union-find to detect chains (connected components) in candidate subgraph."""
    parent = {nid: nid for nid in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for src, tgt, _, _ in edges:
        if src in parent and tgt in parent:
            union(src, tgt)

    components = defaultdict(set)
    for nid in node_ids:
        components[find(nid)].add(nid)

    return list(components.values())


def step35_chain_detection(brain, candidates):
    """Detect chains among candidates using typed edges.

    Returns: {
        'chains': [{members: set, edges: [...], chain_score, edge_families: {...}}],
        'chain_membership': {node_id: chain_index},
        'chain_bonuses': {node_id: bonus},
        'debug': {...}
    }
    """
    candidate_ids = set(candidates.keys())
    debug = {'edges_found': 0, 'chains_found': 0, 'chain_members': 0, 'standalone': 0}

    # Load edge family mapping
    rel_to_family = _load_edge_families(brain)

    # Find inter-candidate edges
    edges = _find_inter_candidate_edges(brain, candidate_ids)
    debug['edges_found'] = len(edges)

    if not edges:
        debug['standalone'] = len(candidates)
        return {'chains': [], 'chain_membership': {}, 'chain_bonuses': {}, 'debug': debug}

    # Find connected components
    # Only include nodes that actually have edges
    edge_nodes = set()
    for src, tgt, _, _ in edges:
        edge_nodes.add(src)
        edge_nodes.add(tgt)

    components = _find_connected_components(edges, edge_nodes)

    # Score each chain
    chains = []
    chain_membership = {}
    chain_bonuses = {}

    for i, component in enumerate(components):
        if len(component) < CHAIN['min_chain_size']:
            continue
        members = set(list(component)[:CHAIN['max_chain_size']])

        # Collect edges within this chain
        chain_edges = [(s, t, r, w) for s, t, r, w in edges
                       if s in members and t in members]

        # Score: embedding strength across chain
        member_scores = [candidates[nid]['score'] for nid in members if nid in candidates]
        if not member_scores:
            continue
        chain_embedding_strength = sum(member_scores) / len(member_scores)

        # Score: edge family strength across chain
        edge_family_weights = []
        edge_families_found = defaultdict(int)
        for _, _, rel, weight in chain_edges:
            family = rel_to_family.get(rel)
            fam_weight = CHAIN_EDGE_FAMILY_WEIGHTS.get(family, DEFAULT_EDGE_WEIGHT) if family else DEFAULT_EDGE_WEIGHT
            edge_family_weights.append(fam_weight)
            if family:
                edge_families_found[family] += 1

        chain_edge_strength = (sum(edge_family_weights) / len(edge_family_weights)
                               if edge_family_weights else DEFAULT_EDGE_WEIGHT)

        # Combined chain score
        chain_score = chain_embedding_strength * (1.0 + chain_edge_strength)

        # Distribute bonus proportionally to individual relevance
        for nid in members:
            if nid not in candidates:
                continue
            member_score = candidates[nid]['score']
            # Proportional share: members with higher relevance get more bonus
            proportion = member_score / chain_embedding_strength if chain_embedding_strength > 0 else 1.0
            # Chain bonus capped: max 25% of the member's own score
            # Prevents mega-chains from overwhelming standalone high-scorers
            raw_bonus = chain_score * 0.3 * proportion
            bonus = min(raw_bonus, candidates[nid]['score'] * 0.25)
            chain_bonuses[nid] = bonus
            chain_membership[nid] = i

        chains.append({
            'index': i,
            'members': members,
            'edges': chain_edges,
            'chain_score': round(chain_score, 4),
            'embedding_strength': round(chain_embedding_strength, 4),
            'edge_strength': round(chain_edge_strength, 4),
            'edge_families': dict(edge_families_found),
            'size': len(members),
        })

    debug['chains_found'] = len(chains)
    debug['chain_members'] = len(chain_membership)
    debug['standalone'] = len(candidates) - len(chain_membership)

    return {
        'chains': chains,
        'chain_membership': chain_membership,
        'chain_bonuses': chain_bonuses,
        'debug': debug,
    }


# ═══════════════════════════════════════════════════
# STEP 4: Unified Ranking
# ═══════════════════════════════════════════════════

def step4_rank(candidates, chain_bonuses, limit=25):
    """Rank candidates: chain bonus + standalone, loudest signal wins.

    Returns: [{id, title, type, score, final_score, source, chain_index, ...}]
    """
    ranked = []
    for nid, cand in candidates.items():
        relevance = cand.get('score', 0)
        bonus = chain_bonuses.get(nid, 0)
        final = relevance + bonus

        if final < RANKING['relevance_floor']:
            continue

        ranked.append({
            'id': nid,
            'title': cand.get('title', ''),
            'type': cand.get('type', ''),
            'score': round(final, 4),
            'relevance': round(relevance, 4),
            'chain_bonus': round(bonus, 4),
            'cosine': cand.get('cosine', 0),
            'confidence': cand.get('confidence'),
            'critical': cand.get('critical', 0),
            'source': cand.get('source', 'unknown'),
            'community_id': cand.get('community_id'),
        })

    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked[:limit]


# ═══════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════

def phase1_recall(brain, query, limit=25, session_id=None):
    """Full Phase 1 redesign: query understanding → discovery → chains → rank.

    Drop-in replacement for baseline_recall in eval.
    """
    sid = session_id or getattr(brain, 'session_id', '') or ''

    # Step 1: Query understanding
    q = step1_query_understanding(brain, query, sid)
    if q['is_garbage'] or q['community_vec'] is None:
        return {
            'results': [],
            'activated_communities': [],
            'chains': [],
            'is_garbage': q['is_garbage'],
            '_debug': q['debug'],
        }

    # Step 2: Candidate discovery
    disc = step2_discover(brain, q['community_vec'], q['node_vec'], q['current_vec'])

    # Step 3: Enrichment scoring
    # (Enrichment vectors reuse is deferred — would need EnrichmentDAL scan.
    #  For eval, the node-channel scores already include unified_score modulation.
    #  Full enrichment integration is a production concern.)

    # Step 3.5: Chain detection
    chain_result = step35_chain_detection(brain, disc['candidates'])

    # Step 4: Unified ranking
    results = step4_rank(disc['candidates'], chain_result['chain_bonuses'], limit)

    return {
        'results': results,
        'activated_communities': disc['activated_communities'],
        'chains': chain_result['chains'],
        'is_garbage': False,
        '_debug': {
            'query': q['debug'],
            'discovery': disc['debug'],
            'chains': chain_result['debug'],
        },
    }


def baseline_recall(brain, query, limit=25, **kwargs):
    """Current recall system, normalized output."""
    result = brain.recall(query=query, limit=limit, source='eval')
    recalled = result.get('results', [])
    return {
        'results': [{
            'id': r.get('id', ''), 'title': r.get('title', ''),
            'type': r.get('type', ''), 'score': round(r.get('effective_activation', 0), 4),
            'cosine': round(r.get('embedding_similarity', 0) or 0, 4),
            'confidence': r.get('confidence'), 'critical': r.get('critical', 0),
            'source': r.get('_source', 'baseline'), 'chain_bonus': 0,
        } for r in recalled],
        'activated_communities': [],
        'chains': [],
        'is_garbage': False,
    }


# ═══════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    from tests.isolated_brain import IsolatedBrain

    queries = [
        ("Specific technical", "I want to optimize the recall scoring pipeline"),
        ("Destructive op", "Let me clean up by deleting all the archived nodes"),
        ("Continuation", "perfect. what about the decoding side?"),
        ("Emotional/vague", "something about this architecture feels off"),
        ("Garbage", "great"),
    ]

    print("Phase 1 Redesign — Standalone Test")
    with IsolatedBrain() as env:
        for label, q in queries:
            print(f"\n{'='*70}")
            print(f"[{label}] {q}")

            r = phase1_recall(env.brain, q, limit=10, session_id='test-session')

            if r['is_garbage']:
                print(f"  → GARBAGE (skipped)")
                continue

            d = r.get('_debug', {})
            qd = d.get('query', {})
            dd = d.get('discovery', {})
            cd = d.get('chains', {})

            print(f"  Turns: {qd.get('community_turns_used',0)} community, {qd.get('node_turns_used',0)} node")
            print(f"  Pool: {dd.get('total_raw_pool',0)} candidates "
                  f"({dd.get('community_members',0)} community, "
                  f"{dd.get('node_direct',0)} node, {dd.get('overlap',0)} overlap)")
            print(f"  Chains: {cd.get('chains_found',0)} found, "
                  f"{cd.get('chain_members',0)} members, "
                  f"{cd.get('standalone',0)} standalone")
            print(f"  Edges between candidates: {cd.get('edges_found',0)}")

            if r['activated_communities']:
                print(f"\n  Communities activated:")
                for cid, cscore, ctitle in r['activated_communities'][:5]:
                    print(f"    [{cscore:.3f}] {ctitle[:55]}")

            if r['chains']:
                print(f"\n  Chains:")
                for ch in r['chains'][:3]:
                    fams = ', '.join(f'{f}:{c}' for f, c in ch['edge_families'].items())
                    print(f"    [{ch['chain_score']:.3f}] {ch['size']} nodes, "
                          f"emb={ch['embedding_strength']:.3f}, "
                          f"edge={ch['edge_strength']:.3f} — {fams}")

            print(f"\n  Top 8:")
            for i, res in enumerate(r['results'][:8]):
                bonus = f" +{res['chain_bonus']:.3f}" if res['chain_bonus'] > 0 else ""
                src = res['source'][:8]
                print(f"    {i+1}. [{res['score']:.3f}{bonus}] ({src:8}) "
                      f"[{res['type']:12}] {res['title'][:48]}")

    print("\nDone.")
