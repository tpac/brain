"""Embedding redistribution — pulls node embeddings toward their graph neighborhood.

Architecture:
  Each node's active embedding = blend of frozen original + weighted neighbor influence.
  new_vector = ratio × frozen_original + (1 - ratio) × weighted_avg(neighbor_vectors)

  The frozen original is NEVER overwritten. Every cycle blends from it (idempotent).
  Fidelity = cosine(active, frozen) tracks drift. Auto-resets below threshold.

Blend ratios (from node status):
  locked:          90/10 (authoritative — barely moves)
  high confidence:  80/20
  normal:           70/30
  low confidence:   60/40

Bridge nodes (structural edges to 2+ communities with similar weight) skip redistribution.

Called from: scheduled sleep task (unlimited time).
Stores: frozen originals in embedding_fidelity table, updated embeddings in node_embeddings.

Tested: [date] via eval/brain_eval.py before/after comparison.
"""

import sqlite3
import struct
import math
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

BLEND_RATIOS = {
    'locked': 0.90,
    'high_confidence': 0.80,   # confidence >= 0.85
    'normal': 0.70,
    'low_confidence': 0.60,    # confidence <= 0.5
}

CONFIDENCE_HIGH = 0.85
CONFIDENCE_LOW = 0.5

FIDELITY_RESET_THRESHOLD = 0.50   # Below this: auto-reset to frozen original
BRIDGE_DOMINANCE_RATIO = 2.0      # If strongest community edge < 2× second, it's a bridge


# ═══════════════════════════════════════════════════════════════
# CORE
# ═══════════════════════════════════════════════════════════════

def ensure_fidelity_table(conn: sqlite3.Connection):
    """Create the fidelity tracking table if it doesn't exist."""
    conn.execute("""CREATE TABLE IF NOT EXISTS embedding_fidelity (
        node_id TEXT PRIMARY KEY,
        original_embedding BLOB NOT NULL,
        fidelity REAL DEFAULT 1.0,
        last_redistributed TEXT,
        redistribution_count INTEGER DEFAULT 0,
        pinned INTEGER DEFAULT 0,
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
    )""")
    conn.commit()


def freeze_originals(conn: sqlite3.Connection) -> int:
    """Store frozen copies of current embeddings for nodes that don't have one yet.

    Called once at setup, then incrementally for new nodes.
    Returns count of newly frozen embeddings.
    """
    ensure_fidelity_table(conn)

    rows = conn.execute("""
        SELECT ne.node_id, ne.embedding
        FROM node_embeddings ne
        WHERE ne.embedding IS NOT NULL
        AND ne.node_id NOT IN (SELECT node_id FROM embedding_fidelity)
    """).fetchall()

    count = 0
    for node_id, embedding in rows:
        conn.execute(
            "INSERT OR IGNORE INTO embedding_fidelity (node_id, original_embedding) VALUES (?, ?)",
            (node_id, embedding))
        count += 1

    conn.commit()
    return count


def get_blend_ratio(node: Dict) -> float:
    """Determine blend ratio based on node status."""
    if node.get('locked'):
        return BLEND_RATIOS['locked']
    conf = node.get('confidence')
    if conf is not None:
        if conf >= CONFIDENCE_HIGH:
            return BLEND_RATIOS['high_confidence']
        elif conf <= CONFIDENCE_LOW:
            return BLEND_RATIOS['low_confidence']
    return BLEND_RATIOS['normal']


def is_bridge_node(conn: sqlite3.Connection, node_id: str,
                   communities: Dict[str, int]) -> bool:
    """Check if a node has strong edges to 2+ communities without one dominating.

    Uses community_member edges to determine community membership.
    The `communities` dict parameter is kept for backward compat but
    we rebuild from edges if empty.
    """
    # Build community membership from community_member edges if needed
    if not communities:
        try:
            rows = conn.execute("""
                SELECT e2.target_id as member_id,
                       e2.source_id as community_id
                FROM edges e2
                JOIN edge_relations er2 ON er2.edge_id = e2.edge_id
                WHERE er2.relation = 'community_member'
                UNION
                SELECT e2.source_id as member_id,
                       e2.target_id as community_id
                FROM edges e2
                JOIN edge_relations er2 ON er2.edge_id = e2.edge_id
                JOIN nodes n ON n.id = e2.target_id AND n.type = 'community'
                WHERE er2.relation = 'community_member'
            """).fetchall()
            for member_id, comm_id in rows:
                communities[member_id] = comm_id
        except Exception:
            return False

    if node_id not in communities:
        return False

    # Get structural edges with their target communities (both directions)
    edges = conn.execute("""
        SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as neighbor,
               e.weight
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE (e.source_id = ? OR e.target_id = ?)
        AND er.relation NOT IN ('co_accessed', 'emergent_bridge', 'community_member')
        AND e.weight > 0.1
        GROUP BY neighbor
    """, (node_id, node_id, node_id)).fetchall()

    # Sum edge weight per community
    community_weights = {}
    for target_id, weight in edges:
        target_comm = communities.get(target_id)
        if target_comm is not None and target_comm != communities.get(node_id):
            community_weights[target_comm] = community_weights.get(target_comm, 0) + (weight or 0.5)

    if len(community_weights) < 2:
        return False

    # Check if any community dominates by BRIDGE_DOMINANCE_RATIO
    sorted_weights = sorted(community_weights.values(), reverse=True)
    if sorted_weights[0] < BRIDGE_DOMINANCE_RATIO * sorted_weights[1]:
        return True  # No clear winner — it's a bridge

    return False


def compute_weighted_neighbor_centroid(
    conn: sqlite3.Connection,
    node_id: str,
    frozen_embeddings: Dict[str, bytes]
) -> Optional[bytes]:
    """Compute weighted average of neighbor embeddings (structural edges only).

    Uses frozen originals for neighbors (not their redistributed vectors).
    Weight = edge weight. Stronger edges pull harder.
    """
    edges = conn.execute("""
        SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as neighbor,
               e.weight
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE (e.source_id = ? OR e.target_id = ?)
        AND er.relation NOT IN ('co_accessed', 'emergent_bridge')
        AND e.weight > 0.1
        GROUP BY neighbor
    """, (node_id, node_id, node_id)).fetchall()

    if not edges:
        return None

    # Collect weighted vectors
    dim = None
    weighted_sum = None
    total_weight = 0.0

    for target_id, edge_weight in edges:
        vec_bytes = frozen_embeddings.get(target_id)
        if not vec_bytes:
            continue

        w = edge_weight or 0.5
        vec = struct.unpack(f'{len(vec_bytes) // 4}f', vec_bytes)

        if dim is None:
            dim = len(vec)
            weighted_sum = [0.0] * dim
        elif len(vec) != dim:
            continue

        for i in range(dim):
            weighted_sum[i] += w * vec[i]
        total_weight += w

    if weighted_sum is None or total_weight == 0:
        return None

    # Normalize
    centroid = [x / total_weight for x in weighted_sum]
    return struct.pack(f'{dim}f', *centroid)


def blend_vectors(original: bytes, neighbor_centroid: bytes, ratio: float) -> bytes:
    """Blend original embedding with neighbor centroid.

    result = ratio × original + (1 - ratio) × neighbor_centroid
    """
    dim = len(original) // 4
    orig = struct.unpack(f'{dim}f', original)
    neigh = struct.unpack(f'{dim}f', neighbor_centroid)

    blended = [ratio * orig[i] + (1 - ratio) * neigh[i] for i in range(dim)]

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in blended))
    if norm > 0:
        blended = [x / norm for x in blended]

    return struct.pack(f'{dim}f', *blended)


def cosine_sim(a: bytes, b: bytes) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dim = len(a) // 4
    va = struct.unpack(f'{dim}f', a)
    vb = struct.unpack(f'{dim}f', b)

    dot = sum(va[i] * vb[i] for i in range(dim))
    norm_a = math.sqrt(sum(x * x for x in va))
    norm_b = math.sqrt(sum(x * x for x in vb))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def redistribute(conn: sqlite3.Connection, dry_run: bool = False) -> Dict:
    """Run one redistribution cycle on all eligible nodes.

    Args:
        conn: Brain database connection
        dry_run: If True, compute but don't write. Returns stats.

    Returns:
        Stats dict: nodes_processed, nodes_skipped, nodes_reset,
                     avg_fidelity_before, avg_fidelity_after, bridge_nodes
    """
    ensure_fidelity_table(conn)

    # Freeze any new nodes that don't have frozen originals yet
    newly_frozen = freeze_originals(conn)

    # Load all frozen originals
    frozen = {}
    for row in conn.execute("SELECT node_id, original_embedding FROM embedding_fidelity WHERE original_embedding IS NOT NULL"):
        frozen[row[0]] = row[1]

    # Load community assignments from community_member edges (source of truth)
    communities = {}
    try:
        rows = conn.execute("""
            SELECT CASE WHEN e.source_id = n.id THEN e.target_id ELSE e.source_id END as member_id,
                   n.id as community_id
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes n ON (n.id = e.source_id OR n.id = e.target_id)
                AND n.type = 'community' AND n.archived = 0
            WHERE er.relation = 'community_member'
        """).fetchall()
        for member_id, comm_id in rows:
            if member_id != comm_id:  # Don't map community to itself
                communities[member_id] = comm_id
    except Exception:
        pass  # No community data — skip bridge detection

    # Load node metadata for blend ratio decisions
    nodes = {}
    for row in conn.execute("SELECT id, locked, confidence FROM nodes WHERE archived = 0"):
        nodes[row[0]] = {'locked': row[1], 'confidence': row[2]}

    # Load pinned nodes
    pinned = set()
    for row in conn.execute("SELECT node_id FROM embedding_fidelity WHERE pinned = 1"):
        pinned.add(row[0])

    stats = {
        'newly_frozen': newly_frozen,
        'nodes_processed': 0,
        'nodes_skipped_no_neighbors': 0,
        'nodes_skipped_bridge': 0,
        'nodes_skipped_pinned': 0,
        'nodes_reset': 0,
        'fidelities_before': [],
        'fidelities_after': [],
    }

    now = __import__('datetime').datetime.utcnow().isoformat() + 'Z'

    for node_id, original in frozen.items():
        node = nodes.get(node_id)
        if not node:
            continue

        # Skip pinned
        if node_id in pinned:
            stats['nodes_skipped_pinned'] += 1
            continue

        # Skip bridge nodes
        if communities and is_bridge_node(conn, node_id, communities):
            stats['nodes_skipped_bridge'] += 1
            continue

        # Compute neighbor centroid
        centroid = compute_weighted_neighbor_centroid(conn, node_id, frozen)
        if centroid is None:
            stats['nodes_skipped_no_neighbors'] += 1
            continue

        # Get blend ratio
        ratio = get_blend_ratio(node)

        # Blend
        new_embedding = blend_vectors(original, centroid, ratio)

        # Compute fidelity
        fidelity = cosine_sim(new_embedding, original)
        stats['fidelities_after'].append(fidelity)

        # Check current fidelity (before this cycle)
        current_emb = conn.execute(
            "SELECT embedding FROM node_embeddings WHERE node_id = ?",
            (node_id,)).fetchone()
        if current_emb and current_emb[0]:
            fidelity_before = cosine_sim(current_emb[0], original)
            stats['fidelities_before'].append(fidelity_before)

        # Auto-reset if fidelity too low
        if fidelity < FIDELITY_RESET_THRESHOLD:
            if not dry_run:
                conn.execute(
                    "UPDATE node_embeddings SET embedding = ? WHERE node_id = ?",
                    (original, node_id))
                conn.execute(
                    "UPDATE embedding_fidelity SET fidelity = 1.0, last_redistributed = ? WHERE node_id = ?",
                    (now, node_id))
            stats['nodes_reset'] += 1
            continue

        # Write new embedding
        if not dry_run:
            conn.execute(
                "UPDATE node_embeddings SET embedding = ? WHERE node_id = ?",
                (new_embedding, node_id))
            # Update fidelity tracking
            conn.execute("""
                INSERT OR REPLACE INTO embedding_fidelity
                (node_id, original_embedding, fidelity, last_redistributed, redistribution_count, pinned)
                VALUES (?, ?, ?, ?, COALESCE(
                    (SELECT redistribution_count + 1 FROM embedding_fidelity WHERE node_id = ?), 1
                ), COALESCE(
                    (SELECT pinned FROM embedding_fidelity WHERE node_id = ?), 0
                ))
            """, (node_id, original, fidelity, now, node_id, node_id))

        stats['nodes_processed'] += 1

    if not dry_run:
        conn.commit()

    # Compute summary stats
    if stats['fidelities_before']:
        stats['avg_fidelity_before'] = sum(stats['fidelities_before']) / len(stats['fidelities_before'])
    else:
        stats['avg_fidelity_before'] = 1.0

    if stats['fidelities_after']:
        stats['avg_fidelity_after'] = sum(stats['fidelities_after']) / len(stats['fidelities_after'])
    else:
        stats['avg_fidelity_after'] = 1.0

    # Clean up large lists from stats
    del stats['fidelities_before']
    del stats['fidelities_after']

    return stats
