#!/usr/bin/env python3
"""Network activation recall: the graph lights up like a neural network.

A query creates an activation pattern across the knowledge graph.
Activation spreads through typed edges, modulated by edge type and weight.
Communities act as soft membranes — activation flows more easily within them.
Situations overlay behavioral rules by WHEN, not WHAT.

The activated subgraph IS the recalled memory.
"""

import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from servers.recall_scoring import unified_score
from servers import embedder

# ── Activation Parameters ──

SPREAD_ROUNDS = 2            # How many rounds of activation spreading
DECAY_PER_HOP = 0.45         # Activation decays by this factor each hop
INITIAL_THRESHOLD = 0.15     # Min cosine to receive initial activation
SPREAD_THRESHOLD = 0.10      # Min activation to spread from
COMMUNITY_MEMBRANE = 1.25    # Within-community spread factor (soft focus)
SITUATION_THRESHOLD = 0.40   # Min situation cosine to activate a rule
SITUATION_INJECTION = 0.20   # How much situation contributes to activation
CLUSTER_BONUS = 0.03         # Bonus per activated neighbor (rewards clusters)

# Edge type activation channels — how strongly each type carries activation
EDGE_CHANNELS = {
    # Knowledge evolution: strong channel — corrections and extensions matter
    'corrects': 0.9, 'supersedes': 0.85, 'extends': 0.8,
    'refines': 0.75, 'resolves': 0.8, 'evolved_from': 0.7,
    # Structural dependency: strong channel
    'depends_on': 0.8, 'enables': 0.75, 'blocks': 0.7,
    'implements': 0.75, 'contributes_to': 0.65,
    # Validation: moderate channel
    'validates': 0.6, 'challenges': 0.7, 'contradicts': 0.75,
    'answers': 0.7, 'demonstrates': 0.55, 'supports': 0.6,
    # Weak/generic: low channel
    'related_to': 0.35, 'related': 0.35,
    'exemplifies': 0.5, 'strengthens': 0.5, 'produced': 0.4,
    'example_of': 0.45,
}
DEFAULT_CHANNEL = 0.4

# Edges that don't carry meaningful activation
DEAD_EDGES = {'co_accessed', 'emergent_bridge', 'community_member'}


def _cosine(query_vec, blob):
    return embedder.cosine_similarity(query_vec, blob)


def _load_graph_edges(conn):
    """Load all meaningful edges into an adjacency structure.

    Returns: {node_id: [(neighbor_id, relation, weight, description)]}
    """
    rows = conn.execute("""
        SELECT e.source_id, e.target_id, er.relation, e.weight, er.description
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        JOIN nodes n1 ON n1.id = e.source_id AND n1.archived = 0
        JOIN nodes n2 ON n2.id = e.target_id AND n2.archived = 0
        WHERE er.archived = 0
    """).fetchall()

    adj = defaultdict(list)
    for src, tgt, rel, weight, desc in rows:
        if rel in DEAD_EDGES:
            continue
        adj[src].append((tgt, rel, weight or 0.5, desc or ''))
        adj[tgt].append((src, rel, weight or 0.5, desc or ''))
    return adj


def _load_community_membership(conn):
    """Load which nodes belong to which communities.

    Returns: {node_id: set(community_ids)}
    """
    rows = conn.execute("""
        SELECT e.source_id as community_id, e.target_id as member_id
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE er.relation = 'community_member'
        AND er.archived = 0
    """).fetchall()

    membership = defaultdict(set)
    for cid, mid in rows:
        membership[mid].add(cid)
    return membership


def _initial_activation(conn, query_vec):
    """Cosine scan → initial activation for every node.

    Returns: {node_id: activation}, {node_id: node_info}
    """
    rows = conn.execute('''
        SELECT ne.node_id, ne.embedding, n.title, n.type,
               n.confidence, n.critical, n.created_at,
               n.emotion, n.access_count
        FROM node_embeddings ne
        JOIN nodes n ON n.id = ne.node_id
        WHERE n.archived = 0 AND n.type != 'community'
    ''').fetchall()

    activation = {}
    info = {}
    for r in rows:
        blob = r[1]
        if not blob:
            continue
        sim = _cosine(query_vec, blob)
        if sim < INITIAL_THRESHOLD:
            continue

        score = unified_score(
            semantic_score=sim,
            created_at=r[6],
            emotion=r[7] or 0,
            access_count=r[8] or 0,
            confidence=r[4],
        )
        activation[r[0]] = score
        info[r[0]] = {
            'title': r[2], 'type': r[3], 'confidence': r[4],
            'critical': r[5] or 0, 'cosine': sim,
        }

    return activation, info


def _spread_activation(activation, adj, membership, rounds=SPREAD_ROUNDS):
    """Spread activation as CONFIRMATION, not creation.

    Key insight: spreading MULTIPLIES existing activation, doesn't add to it.
    A node with 0 initial relevance stays at 0 regardless of neighbor activity.
    A node with high initial relevance gets CONFIRMED by active neighbors.

    This prevents hub amplification — hubs with low relevance don't get boosted
    just because they have many edges.
    """
    spread_log = []
    initial = dict(activation)  # Save initial state

    for round_num in range(rounds):
        # For each node, compute a "confirmation score" from neighbors
        confirmations = {}

        for node_id, act in activation.items():
            if act < SPREAD_THRESHOLD:
                continue

            neighbors = adj.get(node_id, [])
            for neighbor_id, relation, edge_weight, _desc in neighbors:
                if neighbor_id not in initial:
                    continue  # Only confirm nodes that had initial activation

                channel = EDGE_CHANNELS.get(relation, DEFAULT_CHANNEL)

                # Community membrane
                membrane = 1.0
                node_comms = membership.get(node_id, set())
                neighbor_comms = membership.get(neighbor_id, set())
                if node_comms & neighbor_comms:
                    membrane = COMMUNITY_MEMBRANE

                # Confirmation amount
                conf = DECAY_PER_HOP * channel * min(edge_weight, 1.0) * membrane
                confirmations[neighbor_id] = confirmations.get(neighbor_id, 0) + conf

        # Apply confirmations as MULTIPLIERS, degree-normalized
        # Hub with 30 edges shouldn't get 10x more confirmation than
        # peripheral node with 3 edges — normalize by sqrt(degree)
        applied = 0
        for nid, conf_total in confirmations.items():
            base = initial.get(nid, 0)
            if base <= 0:
                continue
            degree = len(adj.get(nid, []))
            normalizer = max(degree ** 0.5, 1.0)
            normalized = conf_total / normalizer
            boost = min(normalized, 0.5)  # max 50% boost from network
            activation[nid] = base * (1.0 + boost)
            applied += 1

        spread_log.append({'round': round_num + 1, 'nodes_confirmed': applied})

    # Cluster bonus: nodes with 2+ activated neighbors in same community
    for node_id in list(activation.keys()):
        if activation[node_id] < SPREAD_THRESHOLD:
            continue
        neighbors = adj.get(node_id, [])
        active_neighbors = sum(1 for n, _, _, _ in neighbors
                               if activation.get(n, 0) > SPREAD_THRESHOLD)
        if active_neighbors >= 2:
            activation[node_id] *= (1.0 + CLUSTER_BONUS * active_neighbors)

    return spread_log


def _situation_overlay(conn, query_vec, activation, info):
    """Inject activation for rules/principles based on SITUATION match.

    Rules should activate by WHEN they're relevant, not WHAT they say.
    """
    rows = conn.execute('''
        SELECT ne.node_id, ne.situation_embedding,
               n.title, n.type, n.confidence
        FROM node_embeddings ne
        JOIN nodes n ON n.id = ne.node_id
        WHERE n.archived = 0
        AND n.type IN ('rule', 'principle', 'lesson')
        AND ne.situation_embedding IS NOT NULL
    ''').fetchall()

    activated = []
    for r in rows:
        nid, sit_blob, title, ntype, conf = r
        if not sit_blob:
            continue
        sit_sim = _cosine(query_vec, sit_blob)
        if sit_sim < SITUATION_THRESHOLD:
            continue

        injection = SITUATION_INJECTION * sit_sim
        old = activation.get(nid, 0)
        activation[nid] = old + injection

        if nid not in info:
            info[nid] = {
                'title': title, 'type': ntype,
                'confidence': conf, 'critical': 0, 'cosine': 0,
            }

        activated.append({
            'id': nid, 'title': title, 'type': ntype,
            'situation_score': round(sit_sim, 4),
        })

    return activated


def _community_frame(conn, top_node_ids, membership):
    """Build community context frame from the most-activated nodes."""
    community_counts = defaultdict(int)
    for nid in top_node_ids:
        for cid in membership.get(nid, set()):
            community_counts[cid] += 1

    top_communities = sorted(community_counts.items(),
                             key=lambda x: x[1], reverse=True)[:3]

    frames = []
    for cid, count in top_communities:
        row = conn.execute(
            "SELECT title FROM nodes WHERE id = ? AND archived = 0",
            (cid,)).fetchone()
        if not row:
            continue
        meta_rows = conn.execute("""
            SELECT key, value FROM node_metadata_kv
            WHERE node_id = ? AND key IN (
                'community_narrative', 'community_open_questions',
                'community_key_decisions', 'community_maturity'
            )
        """, (cid,)).fetchall()
        meta = {k.replace('community_', ''): v for k, v in meta_rows}
        frames.append({
            'id': cid, 'title': row[0],
            'members_in_results': count, **meta,
        })
    return frames


# ─────────────────────────────────────────
# Main entry points
# ─────────────────────────────────────────

def recognition_recall(brain, query, limit=25):
    """Network activation recall — ADDITIVE on top of existing pipeline.

    Strategy: keep the baseline results intact (don't re-rank).
    Use reserved slots to ADD what the baseline misses:
    - Network-connected nodes (graph neighbors of top seeds)
    - Situation-activated behavioral rules (by WHEN, not WHAT)
    - Community framing (territory context, not a result — structural output)
    """
    query_vec = embedder.embed(query)
    if not query_vec:
        return {'results': [], 'clusters': [], 'situation_activated': [],
                'community_frame': []}

    # Step 0: Baseline results (fully featured pipeline) — PRESERVED
    baseline = brain.recall(query=query, limit=limit, source='eval')
    recalled = baseline.get('results', [])

    # Build activation map + results from baseline
    activation = {}
    info = {}
    baseline_ids = set()
    results = []

    for r in recalled:
        nid = r.get('id', '')
        score = r.get('effective_activation', 0)
        activation[nid] = score
        baseline_ids.add(nid)
        info[nid] = {
            'title': r.get('title', ''),
            'type': r.get('type', ''),
            'confidence': r.get('confidence'),
            'critical': r.get('critical', 0),
            'cosine': r.get('embedding_similarity', 0) or 0,
        }
        results.append({
            'id': nid,
            'title': r.get('title', ''),
            'type': r.get('type', ''),
            'score': round(score, 4),
            'cosine': round(r.get('embedding_similarity', 0) or 0, 4),
            'confidence': r.get('confidence'),
            'critical': r.get('critical', 0),
            'source': 'baseline',
        })

    # Step 1: Graph structure for spreading
    adj = _load_graph_edges(brain.conn)
    membership = _load_community_membership(brain.conn)

    # Step 2: Spread activation from baseline seeds
    spread_log = _spread_activation(activation, adj, membership)

    # Find the best network-discovered nodes NOT in baseline
    network_additions = []
    for nid, act in sorted(activation.items(), key=lambda x: x[1], reverse=True):
        if nid in baseline_ids:
            continue
        if nid not in info:
            # Load info for network-discovered nodes
            row = brain.conn.execute(
                "SELECT title, type, confidence FROM nodes WHERE id = ?",
                (nid,)).fetchone()
            if row:
                info[nid] = {'title': row[0], 'type': row[1],
                             'confidence': row[2], 'critical': 0, 'cosine': 0}
            else:
                continue
        if act > 0.25:
            network_additions.append({
                'id': nid,
                'title': info[nid].get('title', ''),
                'type': info[nid].get('type', ''),
                'score': round(act, 4),
                'cosine': 0.0,
                'confidence': info[nid].get('confidence'),
                'critical': 0,
                'source': 'network',
            })
        if len(network_additions) >= 2:
            break

    # Step 3: Situation-activated behavioral rules
    situation_activated = _situation_overlay(
        brain.conn, query_vec, activation, info)

    used_ids = baseline_ids | {r['id'] for r in network_additions}
    situation_additions = []
    for sr in situation_activated:
        if sr['id'] in used_ids:
            continue
        situation_additions.append({
            'id': sr['id'],
            'title': sr['title'],
            'type': sr['type'],
            'score': round(SITUATION_INJECTION * sr['situation_score'], 4),
            'cosine': 0.0,
            'confidence': None,
            'critical': 0,
            'source': 'situation',
        })
        if len(situation_additions) >= 3:
            break

    # Selective replacement: situation/network nodes replace the WEAKEST
    # baseline results only if they're genuinely more valuable.
    # This preserves baseline quality while allowing high-value additions.
    additions = situation_additions + network_additions
    if additions and results:
        # Sort additions by score descending
        additions.sort(key=lambda x: x['score'], reverse=True)
        # Replace weakest baseline results with stronger additions
        for add in additions:
            if len(results) < limit:
                results.append(add)
            elif add['score'] > results[-1]['score']:
                results[-1] = add
            # Re-sort to keep weakest at the end for next replacement
            results.sort(key=lambda x: x['score'], reverse=True)

    # Step 4: Community frame
    top_ids = [r['id'] for r in results[:15]]
    community_frame = _community_frame(brain.conn, top_ids, membership)

    return {
        'results': results,
        'clusters': [],
        'situation_activated': situation_activated,
        'community_frame': community_frame,
        '_spread_log': spread_log,
        '_network_additions': len(network_additions),
        '_situation_additions': len(situation_additions),
    }


def baseline_recall(brain, query, limit=25):
    """Current recall system, normalized for comparison."""
    result = brain.recall(query=query, limit=limit, source='eval')
    recalled = result.get('results', [])

    normalized = []
    for r in recalled:
        normalized.append({
            'id': r.get('id', ''),
            'title': r.get('title', ''),
            'type': r.get('type', ''),
            'score': round(r.get('effective_activation', 0), 4),
            'cosine': round(r.get('embedding_similarity', 0) or 0, 4),
            'confidence': r.get('confidence'),
            'critical': r.get('critical', 0),
            'source': r.get('_source', r.get('_discovery', 'unknown')),
        })

    return {
        'results': normalized,
        'clusters': [],
        'situation_activated': [],
        'community_frame': [],
    }


if __name__ == '__main__':
    from tests.isolated_brain import IsolatedBrain

    print("Testing network activation recall...")
    with IsolatedBrain() as env:
        queries = [
            "I want to optimize the recall scoring pipeline",
            "The encoder keeps creating disconnected nodes",
            "Let me clean up by deleting all archived nodes",
            "something about this architecture feels off",
            "If a new Anchor woke up, what matters most?",
        ]
        for q in queries:
            print(f"\n{'='*60}")
            print(f"Query: {q}")
            r = recognition_recall(env.brain, q, limit=10)

            print(f"  Activated: {r.get('_total_activated', 0)} nodes")
            for log in r.get('_spread_log', []):
                print(f"  Spread round {log['round']}: {log['nodes_boosted']} boosted")

            if r['situation_activated']:
                print(f"\n  Situation-activated rules:")
                for s in r['situation_activated'][:3]:
                    print(f"    [{s['situation_score']:.3f}] {s['title'][:55]}")

            if r['community_frame']:
                print(f"\n  Community frame:")
                for f in r['community_frame']:
                    print(f"    [{f['members_in_results']}] {f['title'][:55]}")

            print(f"\n  Top 8:")
            for res in r['results'][:8]:
                print(f"    [{res['score']:.3f}] ({res['source']:<14}) [{res['type']}] "
                      f"{res['title'][:50]}")

    print("\nDone.")
