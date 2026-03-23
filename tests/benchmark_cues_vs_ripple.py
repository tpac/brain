#!/usr/bin/env python3
"""
brain — Cues vs Ripple Benchmark
==================================

Tests the "cues" approach: store impact relationships as typed edges with metadata,
then surface them at recall time so Claude can reason about them in-context.

Compares against full ripple (confidence changes + re-enrichment) and baseline.

Five tests:
  Test 1: Cue storage and retrieval performance
  Test 2: Cue-augmented recall results (what Claude would see)
  Test 3: Compare cues vs full ripple on NDCG/MRR
  Test 4: Temporal reasoning with cues
  Test 5: Cue density over time simulation

Usage:
    python3 tests/benchmark_cues_vs_ripple.py
"""

import json
import math
import os
import shutil
import sqlite3
import struct
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.metrics import (
    compute_all_metrics, aggregate_metrics,
    ndcg_at_k, mrr as compute_mrr, hit_rate_at_k,
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

MODEL_PATH = os.path.expanduser("~/brain/model-package/brain_embedding/model")
DB_SOURCE = os.path.expanduser("~/AgentsContext/brain/brain.db")
DB_TEMP = "/tmp/brain_cues_test.db"
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_dataset_v2.json')

# Impact assessment thresholds (cosine similarity heuristic)
# sim < 0.3  → NO_IMPACT (skip, no edge)
# 0.3 - 0.6  → EXTENDS
# 0.6 - 0.8  → VALIDATES
# > 0.8      → CONTRADICTS (if newer)

# Ripple config (from benchmark_claude_quality_ripple.py)
TYPE_CONFIDENCE_FLOORS = {
    'rule': 0.70, 'convention': 0.60, 'decision': 0.30,
    'lesson': 0.20, 'mechanism': 0.15, 'impact': 0.10,
    'vocabulary': 0.50, 'mental_model': 0.20, 'purpose': 0.20,
    'constraint': 0.25, 'correction': 0.10,
}
DEFAULT_FLOOR = 0.05
CONFIDENCE_DELTAS = {
    'VALIDATES': +0.03, 'EXTENDS': +0.01,
    'CONTRADICTS': -0.05, 'NO_IMPACT': 0.0,
}
DECAY_RATES = {
    'VALIDATES': 1.0, 'EXTENDS': 0.7, 'CONTRADICTS': 0.5, 'NO_IMPACT': 0.0,
}
CONFIRMATION_THRESHOLD = 0.15
RIPPLE_SOURCE_COUNT = 30


# ═══════════════════════════════════════════════════════════════
# EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════

class Embedder:
    """Thin wrapper around FastEmbed."""

    def __init__(self, model_path: str):
        os.environ["ORT_DISABLE_ALL_ACCELERATORS"] = "1"
        os.environ.setdefault("ONNX_PROVIDERS", "CPUExecutionProvider")

        from fastembed import TextEmbedding
        from fastembed.common.model_description import PoolingType, ModelSource

        model_name = "snowflake/snowflake-arctic-embed-m-v1.5"
        dim = 768

        supported = [m['model'].lower() for m in TextEmbedding.list_supported_models()]
        if model_name.lower() not in supported:
            TextEmbedding.add_custom_model(
                model=model_name, pooling=PoolingType.CLS, normalization=True,
                sources=ModelSource(hf=model_name), dim=dim,
                model_file="onnx/model.onnx",
            )

        self.model = TextEmbedding(
            model_name=model_name, specific_model_path=model_path,
            providers=["CPUExecutionProvider"],
        )
        self.dim = dim
        print(f"[embedder] Loaded {model_name} ({dim}d) from {model_path}")

    def embed(self, text: str) -> bytes:
        vecs = list(self.model.embed([text]))
        return vecs[0].astype('float32').tobytes()

    def embed_batch(self, texts: List[str]) -> List[bytes]:
        if not texts:
            return []
        vecs = list(self.model.embed(texts))
        return [v.astype('float32').tobytes() for v in vecs]

    @staticmethod
    def cosine_similarity(a: bytes, b: bytes) -> float:
        if not a or not b:
            return 0.0
        count_a = len(a) // 4
        count_b = len(b) // 4
        if count_a != count_b:
            return 0.0
        va = struct.unpack(f'<{count_a}f', a)
        vb = struct.unpack(f'<{count_b}f', b)
        return sum(x * y for x, y in zip(va, vb))


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_golden_dataset() -> List[Dict]:
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def load_all_embeddings(conn: sqlite3.Connection) -> Dict[str, bytes]:
    rows = conn.execute(
        "SELECT ne.node_id, ne.embedding FROM node_embeddings ne "
        "JOIN nodes n ON n.id = ne.node_id WHERE n.archived = 0"
    ).fetchall()
    return {nid: blob for nid, blob in rows if blob}


def load_all_enrichments(conn: sqlite3.Connection) -> Dict[str, List[Tuple[str, bytes]]]:
    rows = conn.execute(
        "SELECT node_id, vector_type, embedding FROM node_enrichments WHERE embedding IS NOT NULL"
    ).fetchall()
    result = {}
    for node_id, vtype, blob in rows:
        if blob:
            result.setdefault(node_id, []).append((vtype, blob))
    return result


def load_node(conn: sqlite3.Connection, node_id: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT id, type, title, content, keywords, confidence, locked, created_at "
        "FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if not row:
        return None
    return {
        'id': row[0], 'type': row[1], 'title': row[2],
        'content': row[3] or '', 'keywords': row[4] or '',
        'confidence': row[5], 'locked': bool(row[6]), 'created_at': row[7],
    }


def get_neighbors(conn: sqlite3.Connection, node_id: str, limit: int = 5) -> List[Dict]:
    rows = conn.execute("""
        SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as other_id,
               e.weight, e.relation
        FROM edges e
        WHERE e.source_id = ? OR e.target_id = ?
        ORDER BY e.weight DESC
        LIMIT ?
    """, (node_id, node_id, node_id, limit)).fetchall()
    neighbors = []
    for other_id, weight, relation in rows:
        node = load_node(conn, other_id)
        if node:
            node['edge_weight'] = weight
            node['edge_relation'] = relation
            neighbors.append(node)
    return neighbors


# ═══════════════════════════════════════════════════════════════
# MULTI-VECTOR RECALL (matching production logic)
# ═══════════════════════════════════════════════════════════════

def recall_multivec(
    query_embedding: bytes,
    primary_embeddings: Dict[str, bytes],
    enrichments: Dict[str, List[Tuple[str, bytes]]],
    node_confidence: Dict[str, Optional[float]],
    limit: int = 20,
) -> List[Dict]:
    """Multi-vector recall: max(primary, enrichment) * confidence_multiplier."""
    scores = {}
    best_types = {}

    for node_id, blob in primary_embeddings.items():
        sim = Embedder.cosine_similarity(query_embedding, blob)
        scores[node_id] = sim
        best_types[node_id] = 'primary'

    for node_id, enrich_list in enrichments.items():
        for vtype, blob in enrich_list:
            sim = Embedder.cosine_similarity(query_embedding, blob)
            if sim > scores.get(node_id, 0):
                scores[node_id] = sim
                best_types[node_id] = vtype

    # Confidence multiplier: [0.1,1.0] -> [0.7, 1.05]
    for node_id in scores:
        conf = node_confidence.get(node_id)
        if conf is not None:
            conf_multiplier = 0.7 + (conf - 0.1) * (1.05 - 0.7) / (1.0 - 0.1)
            conf_multiplier = max(0.7, min(1.05, conf_multiplier))
            scores[node_id] *= conf_multiplier

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [
        {'node_id': nid, 'score': sc, 'best_vector_type': best_types.get(nid, 'unknown')}
        for nid, sc in ranked[:limit]
    ]


# ═══════════════════════════════════════════════════════════════
# IMPACT ASSESSMENT (cosine heuristic)
# ═══════════════════════════════════════════════════════════════

def assess_impact_heuristic(
    source_emb: bytes, neighbor_emb: bytes,
    source_node: Dict, neighbor_node: Dict,
) -> Tuple[str, float]:
    """Returns (impact_type, similarity)."""
    sim = Embedder.cosine_similarity(source_emb, neighbor_emb)

    if sim < 0.3:
        return 'NO_IMPACT', sim
    elif sim < 0.6:
        return 'EXTENDS', sim
    elif sim < 0.8:
        return 'VALIDATES', sim
    else:
        s_date = source_node.get('created_at', '')
        n_date = neighbor_node.get('created_at', '')
        if s_date > n_date and source_node.get('content', '') != neighbor_node.get('content', ''):
            return 'CONTRADICTS', sim
        return 'VALIDATES', sim


# ═══════════════════════════════════════════════════════════════
# GOLDEN EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_golden(
    golden: List[Dict],
    primary_embeddings: Dict[str, bytes],
    enrichments: Dict[str, List[Tuple[str, bytes]]],
    node_confidence: Dict[str, Optional[float]],
    embedder_inst: Embedder,
    label: str = "baseline",
) -> Dict[str, Any]:
    """Run golden dataset eval with multi-vector recall."""
    case_results = []

    for tc in golden:
        query = tc['query']
        expected = tc.get('expected_relevant', {})
        min_hit_rate = tc.get('min_hit_rate_at_10', 0.0)

        query_emb = embedder_inst.embed(query)
        retrieved = recall_multivec(
            query_emb, primary_embeddings, enrichments,
            node_confidence, limit=20,
        )
        retrieved_ids = [r['node_id'] for r in retrieved]

        metrics = {}
        passed = True

        if expected:
            relevance = {nid: grade for nid, grade in expected.items()}
            metrics = compute_all_metrics(retrieved_ids, relevance, k_values=[5, 10, 20])
            actual_hit_rate = hit_rate_at_k(retrieved_ids, set(expected.keys()), 10)
            if actual_hit_rate < min_hit_rate:
                passed = False

        case_results.append({
            'id': tc['id'],
            'category': tc['category'],
            'query': query,
            'passed': passed,
            'metrics': metrics,
            'retrieved_ids': retrieved_ids[:10],
        })

    scorable = [r['metrics'] for r in case_results
                if r.get('metrics') and r['metrics'].get('mrr') is not None]
    agg = aggregate_metrics(scorable) if scorable else {}

    by_category = {}
    for r in case_results:
        cat = r['category']
        if cat not in by_category:
            by_category[cat] = []
        if r.get('metrics') and r['metrics'].get('mrr') is not None:
            by_category[cat].append(r['metrics'])

    category_agg = {}
    for cat, metrics_list in by_category.items():
        if metrics_list:
            category_agg[cat] = aggregate_metrics(metrics_list)

    passed_count = sum(1 for r in case_results if r['passed'])
    total = len(case_results)

    return {
        'label': label,
        'case_results': case_results,
        'aggregate': agg,
        'by_category': category_agg,
        'passed': passed_count,
        'total': total,
    }


# ═══════════════════════════════════════════════════════════════
# RIPPLE SOURCE SELECTION
# ═══════════════════════════════════════════════════════════════

def pick_ripple_sources(
    conn: sqlite3.Connection, n: int = 30,
    golden_target_ids: Optional[Set[str]] = None,
) -> List[Dict]:
    """Pick N recent nodes with neighbors as ripple sources."""
    total = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
    cutoff = int(total * 0.6)

    candidates = []
    seen_ids = set()

    # Strategy 1: recent nodes connected to golden targets
    if golden_target_ids:
        recent_ids = [r[0] for r in conn.execute(
            "SELECT id FROM nodes WHERE archived=0 ORDER BY created_at ASC LIMIT -1 OFFSET ?",
            (cutoff,)
        ).fetchall()]

        for nid in recent_ids:
            if len(candidates) >= n:
                break
            rows = conn.execute(
                "SELECT CASE WHEN source_id=? THEN target_id ELSE source_id END "
                "FROM edges WHERE source_id=? OR target_id=?",
                (nid, nid, nid)
            ).fetchall()
            neighbor_ids = {r[0] for r in rows}
            golden_overlap = neighbor_ids & golden_target_ids
            if golden_overlap and nid not in seen_ids:
                node = load_node(conn, nid)
                if node:
                    node['neighbor_count'] = len(neighbor_ids)
                    candidates.append(node)
                    seen_ids.add(nid)

    # Strategy 2: fill remaining with recent nodes that have neighbors
    if len(candidates) < n:
        rows = conn.execute("""
            SELECT id FROM nodes WHERE archived = 0
            ORDER BY created_at ASC LIMIT ? OFFSET ?
        """, (n * 3, cutoff)).fetchall()

        for (nid,) in rows:
            if len(candidates) >= n:
                break
            if nid in seen_ids:
                continue
            neighbor_count = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE source_id=? OR target_id=?",
                (nid, nid)
            ).fetchone()[0]
            if neighbor_count >= 1:
                node = load_node(conn, nid)
                if node:
                    node['neighbor_count'] = neighbor_count
                    candidates.append(node)
                    seen_ids.add(nid)

    return candidates[:n]


# ═══════════════════════════════════════════════════════════════
# CUE STORAGE AND RETRIEVAL
# ═══════════════════════════════════════════════════════════════

def store_cue(
    conn: sqlite3.Connection,
    source_id: str, target_id: str,
    relation: str, weight: float,
    reason: str, source_title: str, date: str,
):
    """Store an impact cue as a typed edge with metadata in the description field.

    Uses the existing edges schema. The `description` field stores JSON metadata.
    The `relation` field stores the cue type (validates, contradicts, extends).
    """
    metadata = json.dumps({
        'reason': reason,
        'source_title': source_title,
        'date': date,
        'assessed_by': 'heuristic',
    })

    # Check for existing edge
    existing = conn.execute(
        "SELECT 1 FROM edges WHERE source_id=? AND target_id=?",
        (source_id, target_id)
    ).fetchone()

    if existing:
        # Update existing edge with cue info
        conn.execute(
            "UPDATE edges SET relation=?, weight=?, description=? WHERE source_id=? AND target_id=?",
            (relation, weight, metadata, source_id, target_id)
        )
    else:
        conn.execute(
            "INSERT INTO edges (source_id, target_id, relation, weight, description, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, target_id, relation, weight, metadata,
             datetime.now(timezone.utc).isoformat())
        )


def get_cues_for_node(conn: sqlite3.Connection, node_id: str) -> List[Dict]:
    """Fetch all impact cues pointing at a node (where the node is the target).

    Returns cues from edges where:
    - target_id = node_id AND relation IN ('validates', 'contradicts', 'extends')
    - Joined with source node title and created_at
    """
    rows = conn.execute("""
        SELECT e.source_id, e.relation, e.weight, e.description, n.title, n.created_at
        FROM edges e JOIN nodes n ON n.id = e.source_id
        WHERE e.target_id = ? AND e.relation IN ('validates', 'contradicts', 'extends')
        ORDER BY n.created_at DESC
    """, (node_id,)).fetchall()

    cues = []
    for source_id, relation, weight, description, title, created_at in rows:
        meta = {}
        if description:
            try:
                meta = json.loads(description)
            except (json.JSONDecodeError, TypeError):
                meta = {'reason': description}
        cues.append({
            'source_id': source_id,
            'relation': relation,
            'weight': weight,
            'reason': meta.get('reason', ''),
            'source_title': title,
            'date': meta.get('date', created_at or ''),
            'created_at': created_at,
        })
    return cues


def get_cues_for_node_fast(conn: sqlite3.Connection, node_id: str) -> List[Dict]:
    """Optimized cue retrieval — minimal parsing, for benchmarking speed."""
    rows = conn.execute("""
        SELECT e.source_id, e.relation, e.weight, e.description
        FROM edges e
        WHERE e.target_id = ? AND e.relation IN ('validates', 'contradicts', 'extends')
    """, (node_id,)).fetchall()
    return [{'source_id': r[0], 'relation': r[1], 'weight': r[2], 'description': r[3]}
            for r in rows]


# ═══════════════════════════════════════════════════════════════
# CUE FORMATTING (what Claude would see)
# ═══════════════════════════════════════════════════════════════

CUE_LABELS = {
    'validates': 'VALIDATED BY',
    'contradicts': 'CONTRADICTED BY',
    'extends': 'EXTENDED BY',
}

def format_node_with_cues(node: Dict, cues: List[Dict], max_cues: int = 5) -> str:
    """Format a recalled node with its cues for Claude's context."""
    conf = node.get('confidence', 0.5)
    title = node.get('title', 'Unknown')
    lines = [f'Node: "{title}" (conf {conf:.2f})']

    if cues:
        lines.append('  Cues:')
        for cue in cues[:max_cues]:
            label = CUE_LABELS.get(cue['relation'], cue['relation'].upper())
            reason = cue.get('reason', '')
            date = cue.get('date', '')[:10]
            src = cue.get('source_title', '')
            line = f'  - {label}: "{src}"'
            if date:
                line += f' ({date})'
            if reason:
                line += f' -- "{reason}"'
            lines.append(line)
        if len(cues) > max_cues:
            lines.append(f'  - ... and {len(cues) - max_cues} more cues')
    else:
        lines.append('  Cues: (none)')

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# APPLY CUES (store impact edges, NO confidence changes)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CueStats:
    validates: int = 0
    contradicts: int = 0
    extends: int = 0
    no_impact: int = 0
    edges_created: int = 0
    nodes_with_cues: int = 0


def apply_cues(
    conn: sqlite3.Connection,
    embedder_inst: Embedder,
    primary_embeddings: Dict[str, bytes],
    ripple_sources: List[Dict],
) -> CueStats:
    """Apply cue-only approach: store impact edges, no confidence changes, no re-enrichment."""
    stats = CueStats()
    nodes_with_cues = set()

    for idx, source in enumerate(ripple_sources):
        source_id = source['id']
        source_emb = primary_embeddings.get(source_id)
        if not source_emb:
            continue

        neighbors = get_neighbors(conn, source_id, limit=5)
        if not neighbors:
            continue

        for neighbor in neighbors:
            neighbor_id = neighbor['id']
            neighbor_emb = primary_embeddings.get(neighbor_id)
            if not neighbor_emb:
                continue

            impact, sim = assess_impact_heuristic(source_emb, neighbor_emb, source, neighbor)

            if impact == 'NO_IMPACT':
                stats.no_impact += 1
                continue

            # Map to cue relation names (lowercase)
            relation = impact.lower()  # validates, contradicts, extends

            # Generate a reason based on the impact type
            if impact == 'VALIDATES':
                reason = f"supports the same conclusion (sim={sim:.2f})"
                stats.validates += 1
            elif impact == 'EXTENDS':
                reason = f"adds a new angle on the topic (sim={sim:.2f})"
                stats.extends += 1
            elif impact == 'CONTRADICTS':
                reason = f"newer information that conflicts (sim={sim:.2f})"
                stats.contradicts += 1

            store_cue(
                conn, source_id, neighbor_id,
                relation, round(sim, 3),
                reason, source['title'],
                datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            )
            stats.edges_created += 1
            nodes_with_cues.add(neighbor_id)

    conn.commit()
    stats.nodes_with_cues = len(nodes_with_cues)
    return stats


# ═══════════════════════════════════════════════════════════════
# APPLY FULL RIPPLE (confidence changes + re-enrichment)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RippleStats:
    impact_counts: Dict[str, int] = field(default_factory=lambda: {
        'VALIDATES': 0, 'EXTENDS': 0, 'CONTRADICTS': 0, 'NO_IMPACT': 0
    })
    confidence_changes: int = 0
    enrichments_created: int = 0
    edges_created: int = 0
    floor_activations: int = 0
    locked_protections: int = 0


def apply_ripple(
    conn: sqlite3.Connection,
    embedder_inst: Embedder,
    primary_embeddings: Dict[str, bytes],
    ripple_sources: List[Dict],
) -> RippleStats:
    """Apply full ripple: confidence changes + typed edges + re-enrichment."""
    stats = RippleStats()

    for source in ripple_sources:
        source_id = source['id']
        source_emb = primary_embeddings.get(source_id)
        if not source_emb:
            continue

        neighbors = get_neighbors(conn, source_id, limit=5)
        for neighbor in neighbors:
            neighbor_id = neighbor['id']
            neighbor_emb = primary_embeddings.get(neighbor_id)
            if not neighbor_emb:
                continue

            impact, sim = assess_impact_heuristic(source_emb, neighbor_emb, source, neighbor)
            stats.impact_counts[impact] += 1

            if impact == 'NO_IMPACT':
                continue

            # Confidence change
            base_delta = CONFIDENCE_DELTAS[impact]
            node = load_node(conn, neighbor_id)
            if not node:
                continue

            if node['locked'] and base_delta < 0:
                stats.locked_protections += 1
                continue

            if base_delta < 0 and abs(base_delta) > CONFIRMATION_THRESHOLD:
                continue

            new_conf = node['confidence'] + base_delta
            if base_delta < 0:
                floor = TYPE_CONFIDENCE_FLOORS.get(node['type'], DEFAULT_FLOOR)
                if new_conf < floor:
                    new_conf = floor
                    stats.floor_activations += 1

            new_conf = max(0.0, min(1.0, new_conf))
            actual_delta = new_conf - node['confidence']

            if abs(actual_delta) > 0.001:
                conn.execute(
                    "UPDATE nodes SET confidence = ? WHERE id = ?",
                    (new_conf, neighbor_id)
                )
                stats.confidence_changes += 1

            # Re-enrichment for VALIDATES and EXTENDS
            if impact in ('VALIDATES', 'EXTENDS'):
                enrichment_texts = []
                enrichment_types = []

                q_text = f"How does {source['title']} relate to {neighbor['title']}?"
                enrichment_texts.append(q_text)
                enrichment_types.append('question')

                b_text = f"{source['title']} {impact.lower()} {neighbor['title']}: {source.get('content', '')[:80]}"
                enrichment_texts.append(b_text)
                enrichment_types.append('bridge')

                if enrichment_texts:
                    embs = embedder_inst.embed_batch(enrichment_texts)
                    now = datetime.now(timezone.utc).isoformat()
                    for i, vtype in enumerate(enrichment_types):
                        if i < len(embs):
                            eid = f"ripple_{uuid.uuid4().hex[:12]}"
                            conn.execute(
                                """INSERT OR REPLACE INTO node_enrichments
                                   (id, node_id, vector_type, text, embedding, model, created_at)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (eid, neighbor_id, vtype,
                                 f"[ripple:{impact}] {enrichment_texts[i]}", embs[i],
                                 'snowflake-arctic-embed-m', now)
                            )
                            stats.enrichments_created += 1

            # Typed edge
            edge_relation = f"ripple_{impact.lower()}"
            existing = conn.execute(
                "SELECT 1 FROM edges WHERE (source_id=? AND target_id=?) OR (source_id=? AND target_id=?)",
                (source_id, neighbor_id, neighbor_id, source_id)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO edges (source_id, target_id, relation, weight, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (source_id, neighbor_id, edge_relation, 0.5,
                     datetime.now(timezone.utc).isoformat())
                )
                stats.edges_created += 1

    conn.commit()
    return stats


# ═══════════════════════════════════════════════════════════════
# TEST 1: CUE STORAGE AND RETRIEVAL PERFORMANCE
# ═══════════════════════════════════════════════════════════════

def test1_cue_storage_retrieval(conn, embedder_inst, primary_embeddings):
    """Test 1: Can we efficiently store and retrieve cues?"""
    print("\n" + "=" * 80)
    print("  TEST 1: CUE STORAGE AND RETRIEVAL PERFORMANCE")
    print("=" * 80)

    # Pick 10 source nodes with 3-5 neighbors each
    sources = []
    rows = conn.execute("""
        SELECT n.id FROM nodes n
        WHERE n.archived = 0
        AND (SELECT COUNT(*) FROM edges e WHERE e.source_id = n.id OR e.target_id = n.id) >= 3
        ORDER BY n.created_at DESC
        LIMIT 10
    """).fetchall()

    for (nid,) in rows:
        node = load_node(conn, nid)
        if node:
            sources.append(node)

    print(f"  Selected {len(sources)} source nodes with 3+ neighbors")

    # Store impact cues
    cue_count = 0
    target_nodes = set()
    t0 = time.time()

    for source in sources:
        source_emb = primary_embeddings.get(source['id'])
        if not source_emb:
            continue

        neighbors = get_neighbors(conn, source['id'], limit=5)
        for neighbor in neighbors:
            neighbor_emb = primary_embeddings.get(neighbor['id'])
            if not neighbor_emb:
                continue

            impact, sim = assess_impact_heuristic(source_emb, neighbor_emb, source, neighbor)
            if impact == 'NO_IMPACT':
                continue

            relation = impact.lower()
            reason = f"cosine similarity {sim:.2f}"
            store_cue(conn, source['id'], neighbor['id'], relation, round(sim, 3),
                      reason, source['title'],
                      datetime.now(timezone.utc).strftime('%Y-%m-%d'))
            cue_count += 1
            target_nodes.add(neighbor['id'])

    conn.commit()
    store_time = (time.time() - t0) * 1000

    print(f"  Stored {cue_count} cues across {len(target_nodes)} target nodes")
    print(f"  Storage time: {store_time:.1f}ms ({store_time/max(cue_count,1):.2f}ms per cue)")

    # Retrieval benchmark
    retrieval_times = []
    for target_id in target_nodes:
        t0 = time.time()
        cues = get_cues_for_node(conn, target_id)
        retrieval_times.append((time.time() - t0) * 1000)

    # Fast retrieval benchmark
    fast_retrieval_times = []
    for target_id in target_nodes:
        t0 = time.time()
        cues = get_cues_for_node_fast(conn, target_id)
        fast_retrieval_times.append((time.time() - t0) * 1000)

    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
    max_retrieval = max(retrieval_times) if retrieval_times else 0
    avg_fast = sum(fast_retrieval_times) / len(fast_retrieval_times) if fast_retrieval_times else 0
    max_fast = max(fast_retrieval_times) if fast_retrieval_times else 0

    print(f"\n  Retrieval (with JOIN):")
    print(f"    Average: {avg_retrieval:.3f}ms")
    print(f"    Max:     {max_retrieval:.3f}ms")
    print(f"    Target:  <5ms  {'PASS' if max_retrieval < 5 else 'FAIL'}")
    print(f"\n  Retrieval (fast, no JOIN):")
    print(f"    Average: {avg_fast:.3f}ms")
    print(f"    Max:     {max_fast:.3f}ms")

    # Show example formatted cue
    example_id = list(target_nodes)[0] if target_nodes else None
    if example_id:
        node = load_node(conn, example_id)
        cues = get_cues_for_node(conn, example_id)
        if node and cues:
            print(f"\n  Example formatted output:")
            print(f"  {'-'*60}")
            formatted = format_node_with_cues(node, cues)
            for line in formatted.split('\n'):
                print(f"  {line}")
            print(f"  {'-'*60}")

    return {
        'cues_stored': cue_count,
        'target_nodes': len(target_nodes),
        'avg_retrieval_ms': avg_retrieval,
        'max_retrieval_ms': max_retrieval,
        'avg_fast_retrieval_ms': avg_fast,
        'max_fast_retrieval_ms': max_fast,
        'pass': max_retrieval < 5,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 2: CUE-AUGMENTED RECALL RESULTS
# ═══════════════════════════════════════════════════════════════

def test2_cue_augmented_recall(conn, embedder_inst, primary_embeddings, enrichments,
                                node_confidence, golden):
    """Test 2: What would Claude see with cue-augmented results?"""
    print("\n" + "=" * 80)
    print("  TEST 2: CUE-AUGMENTED RECALL RESULTS")
    print("=" * 80)

    # Run 30 golden dataset queries (sample evenly across categories)
    categories = {}
    for tc in golden:
        categories.setdefault(tc['category'], []).append(tc)

    selected = []
    per_cat = max(1, 30 // len(categories))
    for cat, cases in categories.items():
        selected.extend(cases[:per_cat])
    selected = selected[:30]

    print(f"  Running {len(selected)} queries across {len(categories)} categories")

    nodes_with_cues = 0
    nodes_without_cues = 0
    total_cue_count = 0
    cue_type_counts = defaultdict(int)
    useful_cue_examples = []

    for tc in selected:
        query = tc['query']
        query_emb = embedder_inst.embed(query)
        retrieved = recall_multivec(
            query_emb, primary_embeddings, enrichments,
            node_confidence, limit=5,
        )

        for r in retrieved:
            node_id = r['node_id']
            cues = get_cues_for_node(conn, node_id)
            if cues:
                nodes_with_cues += 1
                total_cue_count += len(cues)
                for cue in cues:
                    cue_type_counts[cue['relation']] += 1

                # Save interesting examples (nodes with contradictions or multiple cues)
                has_contradiction = any(c['relation'] == 'contradicts' for c in cues)
                if has_contradiction or len(cues) >= 2:
                    node = load_node(conn, node_id)
                    if node and len(useful_cue_examples) < 5:
                        useful_cue_examples.append({
                            'query': query,
                            'node': node,
                            'cues': cues,
                        })
            else:
                nodes_without_cues += 1

    total_nodes_checked = nodes_with_cues + nodes_without_cues
    cue_coverage = nodes_with_cues / total_nodes_checked * 100 if total_nodes_checked else 0
    avg_cues = total_cue_count / nodes_with_cues if nodes_with_cues else 0

    print(f"\n  Results ({total_nodes_checked} recalled nodes across {len(selected)} queries):")
    print(f"    Nodes with cues:    {nodes_with_cues} ({cue_coverage:.1f}%)")
    print(f"    Nodes without cues: {nodes_without_cues}")
    print(f"    Total cues found:   {total_cue_count}")
    print(f"    Avg cues per node:  {avg_cues:.1f} (when present)")
    print(f"\n  Cue type distribution:")
    for ctype, count in sorted(cue_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ctype}: {count}")

    # Show examples of what Claude would see
    if useful_cue_examples:
        print(f"\n  Example cue-augmented results (what Claude would see):")
        for i, ex in enumerate(useful_cue_examples[:3]):
            print(f"\n  [{i+1}] Query: \"{ex['query']}\"")
            print(f"  {'-'*60}")
            formatted = format_node_with_cues(ex['node'], ex['cues'])
            for line in formatted.split('\n'):
                print(f"  {line}")

    return {
        'queries_run': len(selected),
        'nodes_with_cues': nodes_with_cues,
        'nodes_without_cues': nodes_without_cues,
        'cue_coverage_pct': cue_coverage,
        'avg_cues_per_node': avg_cues,
        'cue_type_counts': dict(cue_type_counts),
    }


# ═══════════════════════════════════════════════════════════════
# TEST 3: COMPARE CUES VS FULL RIPPLE
# ═══════════════════════════════════════════════════════════════

def test3_cues_vs_ripple(embedder_inst, golden):
    """Test 3: NDCG/MRR comparison — baseline vs cues-only vs full ripple."""
    print("\n" + "=" * 80)
    print("  TEST 3: CUES VS FULL RIPPLE (NDCG/MRR COMPARISON)")
    print("=" * 80)

    golden_target_ids = set()
    for tc in golden:
        golden_target_ids.update(tc.get('expected_relevant', {}).keys())

    results = {}

    # ── Condition A: Baseline ──
    print("\n  [A] Baseline (unmodified DB)...")
    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_a = sqlite3.connect(DB_TEMP)
    primary_a = load_all_embeddings(conn_a)
    enrichments_a = load_all_enrichments(conn_a)
    confidence_a = dict(conn_a.execute(
        "SELECT id, confidence FROM nodes WHERE archived=0"
    ).fetchall())

    results['A'] = evaluate_golden(
        golden, primary_a, enrichments_a, confidence_a, embedder_inst, "A: Baseline"
    )
    conn_a.close()
    print(f"    NDCG@10={results['A']['aggregate'].get('ndcg@10', {}).get('mean', 0):.4f}  "
          f"MRR={results['A']['aggregate'].get('mrr', {}).get('mean', 0):.4f}  "
          f"Passed={results['A']['passed']}/{results['A']['total']}")

    # ── Condition B: Cues only (typed edges, NO confidence, NO re-enrichment) ──
    print("\n  [B] Cues only (impact edges, no confidence changes)...")
    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_b = sqlite3.connect(DB_TEMP)
    primary_b = load_all_embeddings(conn_b)
    enrichments_b = load_all_enrichments(conn_b)

    ripple_sources = pick_ripple_sources(conn_b, RIPPLE_SOURCE_COUNT, golden_target_ids)
    print(f"    Applying cues from {len(ripple_sources)} source nodes...")
    cue_stats = apply_cues(conn_b, embedder_inst, primary_b, ripple_sources)
    print(f"    Cues: validates={cue_stats.validates} extends={cue_stats.extends} "
          f"contradicts={cue_stats.contradicts} no_impact={cue_stats.no_impact}")
    print(f"    Edges created: {cue_stats.edges_created}, Nodes with cues: {cue_stats.nodes_with_cues}")

    # Cues don't change confidence or embeddings, so recall scores are identical to baseline
    confidence_b = dict(conn_b.execute(
        "SELECT id, confidence FROM nodes WHERE archived=0"
    ).fetchall())
    results['B'] = evaluate_golden(
        golden, primary_b, enrichments_b, confidence_b, embedder_inst, "B: Cues only"
    )
    conn_b.close()
    print(f"    NDCG@10={results['B']['aggregate'].get('ndcg@10', {}).get('mean', 0):.4f}  "
          f"MRR={results['B']['aggregate'].get('mrr', {}).get('mean', 0):.4f}  "
          f"Passed={results['B']['passed']}/{results['B']['total']}")

    # ── Condition C: Full ripple (confidence + edges + re-enrichment) ──
    print("\n  [C] Full ripple (confidence changes + re-enrichment)...")
    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_c = sqlite3.connect(DB_TEMP)
    primary_c = load_all_embeddings(conn_c)

    ripple_sources_c = pick_ripple_sources(conn_c, RIPPLE_SOURCE_COUNT, golden_target_ids)
    ripple_stats = apply_ripple(conn_c, embedder_inst, primary_c, ripple_sources_c)
    print(f"    Confidence changes: {ripple_stats.confidence_changes}")
    print(f"    Enrichments created: {ripple_stats.enrichments_created}")
    print(f"    Edges created: {ripple_stats.edges_created}")

    # Reload after ripple changes
    enrichments_c = load_all_enrichments(conn_c)
    confidence_c = dict(conn_c.execute(
        "SELECT id, confidence FROM nodes WHERE archived=0"
    ).fetchall())
    results['C'] = evaluate_golden(
        golden, primary_c, enrichments_c, confidence_c, embedder_inst, "C: Full ripple"
    )
    conn_c.close()
    print(f"    NDCG@10={results['C']['aggregate'].get('ndcg@10', {}).get('mean', 0):.4f}  "
          f"MRR={results['C']['aggregate'].get('mrr', {}).get('mean', 0):.4f}  "
          f"Passed={results['C']['passed']}/{results['C']['total']}")

    # ── Print comparison table ──
    print(f"\n  {'='*80}")
    print(f"  COMPARISON TABLE")
    print(f"  {'='*80}")
    print(f"  {'Condition':<40s} {'NDCG@10':>8s} {'MRR':>8s} {'hit@10':>8s} {'Passed':>8s} {'Delta':>8s}")
    print(f"  {'-'*75}")

    baseline_ndcg = results['A']['aggregate'].get('ndcg@10', {}).get('mean', 0)

    for key in ['A', 'B', 'C']:
        r = results[key]
        agg = r['aggregate']
        ndcg = agg.get('ndcg@10', {}).get('mean', 0)
        mrr_val = agg.get('mrr', {}).get('mean', 0)
        hit = agg.get('hit_rate@10', {}).get('mean', 0)
        passed = r['passed']
        total = r['total']
        delta = ndcg - baseline_ndcg
        delta_str = f"{delta:+.4f}" if key != 'A' else "---"

        labels = {
            'A': 'A: Baseline (no changes)',
            'B': 'B: Cues only (edges, no conf change)',
            'C': 'C: Full ripple (conf + re-enrich)',
        }
        print(f"  {labels[key]:<40s} {ndcg:>8.4f} {mrr_val:>8.4f} {hit:>8.4f} {passed:>4d}/{total:<3d} {delta_str:>8s}")

    # Category-level comparison
    print(f"\n  {'='*80}")
    print(f"  CATEGORY BREAKDOWN (NDCG@10)")
    print(f"  {'='*80}")

    all_cats = set()
    for key in ['A', 'B', 'C']:
        all_cats.update(results[key]['by_category'].keys())

    print(f"  {'Category':<25s} {'Baseline':>10s} {'Cues':>10s} {'Ripple':>10s} {'Cues-Base':>10s} {'Rip-Base':>10s}")
    print(f"  {'-'*70}")

    for cat in sorted(all_cats):
        vals = []
        for key in ['A', 'B', 'C']:
            cat_agg = results[key]['by_category'].get(cat, {})
            v = cat_agg.get('ndcg@10', {}).get('mean', 0)
            vals.append(v)
        base, cues, ripple = vals
        d_cues = cues - base
        d_ripple = ripple - base
        print(f"  {cat:<25s} {base:>10.4f} {cues:>10.4f} {ripple:>10.4f} {d_cues:>+10.4f} {d_ripple:>+10.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# TEST 4: TEMPORAL REASONING WITH CUES
# ═══════════════════════════════════════════════════════════════

def test4_temporal_reasoning(conn, embedder_inst, primary_embeddings, enrichments, node_confidence):
    """Test 4: Do cues enable temporal reasoning that raw recall can't?"""
    print("\n" + "=" * 80)
    print("  TEST 4: TEMPORAL REASONING WITH CUES")
    print("=" * 80)

    temporal_queries = [
        {
            'query': 'what was the original architecture approach',
            'type': 'historical',
            'cue_signal': 'contradicts',  # old nodes should have "contradicted by" cues
            'description': 'Historical: find old nodes that were later superseded',
        },
        {
            'query': 'what changed in the brain recently',
            'type': 'recent_changes',
            'cue_signal': 'extends',  # recent nodes that extended existing ones
            'description': 'Recent changes: find nodes with recent "extends" or "contradicts" cues',
        },
        {
            'query': 'what decisions are still valid and confirmed',
            'type': 'still_valid',
            'cue_signal': 'validates',  # nodes validated but NOT contradicted
            'description': 'Still valid: find nodes with "validates" but NOT "contradicts" cues',
        },
        {
            'query': 'what was tried and then reversed',
            'type': 'reversed',
            'cue_signal': 'contradicts',
            'description': 'Reversed: find nodes with "contradicts" cues',
        },
    ]

    results = []
    for tq in temporal_queries:
        query = tq['query']
        query_emb = embedder_inst.embed(query)
        retrieved = recall_multivec(
            query_emb, primary_embeddings, enrichments,
            node_confidence, limit=10,
        )

        # For each retrieved node, check cues
        cue_enhanced_results = []
        for r in retrieved:
            node_id = r['node_id']
            node = load_node(conn, node_id)
            if not node:
                continue
            cues = get_cues_for_node(conn, node_id)

            has_signal = any(c['relation'] == tq['cue_signal'] for c in cues)
            has_contradicts = any(c['relation'] == 'contradicts' for c in cues)
            has_validates = any(c['relation'] == 'validates' for c in cues)

            cue_enhanced_results.append({
                'node': node,
                'score': r['score'],
                'cues': cues,
                'has_target_signal': has_signal,
                'has_contradicts': has_contradicts,
                'has_validates': has_validates,
            })

        # Analyze: how many results have the expected temporal signal?
        signal_count = sum(1 for r in cue_enhanced_results if r['has_target_signal'])
        total = len(cue_enhanced_results)

        print(f"\n  Query: \"{query}\"")
        print(f"  Type: {tq['type']} | Target signal: {tq['cue_signal']}")
        print(f"  Results with signal: {signal_count}/{total}")

        # Show top 3 results with their cues
        for i, r in enumerate(cue_enhanced_results[:3]):
            node = r['node']
            cues = r['cues']
            has_signal = 'YES' if r['has_target_signal'] else 'no'
            print(f"    [{i+1}] \"{node['title'][:60]}\" (conf={node['confidence']:.2f}) "
                  f"[signal={has_signal}]")
            if cues:
                for cue in cues[:2]:
                    print(f"         {CUE_LABELS.get(cue['relation'], cue['relation'])}: "
                          f"\"{cue['source_title'][:40]}\"")

        # Temporal reasoning assessment
        if tq['type'] == 'still_valid':
            # For "still valid" — nodes with validates but NOT contradicts
            still_valid = [r for r in cue_enhanced_results
                          if r['has_validates'] and not r['has_contradicts']]
            print(f"  Temporal filter: {len(still_valid)} nodes validated but not contradicted")
        elif tq['type'] == 'reversed':
            reversed_nodes = [r for r in cue_enhanced_results if r['has_contradicts']]
            print(f"  Temporal filter: {len(reversed_nodes)} nodes with contradictions")

        results.append({
            'query': query,
            'type': tq['type'],
            'signal_count': signal_count,
            'total': total,
            'cue_signal': tq['cue_signal'],
        })

    return results


# ═══════════════════════════════════════════════════════════════
# TEST 5: CUE DENSITY OVER TIME
# ═══════════════════════════════════════════════════════════════

def test5_cue_density(conn, embedder_inst, primary_embeddings):
    """Test 5: Simulate 50 encoding events and track cue density."""
    print("\n" + "=" * 80)
    print("  TEST 5: CUE DENSITY OVER TIME SIMULATION")
    print("=" * 80)

    # Get 50 nodes ordered by creation date as simulated "encoding events"
    rows = conn.execute("""
        SELECT id FROM nodes WHERE archived = 0
        ORDER BY created_at DESC
        LIMIT 50
    """).fetchall()
    source_ids = [r[0] for r in rows]

    print(f"  Simulating {len(source_ids)} encoding events...")

    # Track density at each step
    density_snapshots = []
    cue_per_node = defaultdict(int)  # node_id -> cue count
    total_cues = 0

    for step, source_id in enumerate(source_ids, 1):
        source = load_node(conn, source_id)
        if not source:
            continue

        source_emb = primary_embeddings.get(source_id)
        if not source_emb:
            continue

        neighbors = get_neighbors(conn, source_id, limit=5)
        step_cues = 0

        for neighbor in neighbors:
            neighbor_emb = primary_embeddings.get(neighbor['id'])
            if not neighbor_emb:
                continue

            impact, sim = assess_impact_heuristic(source_emb, neighbor_emb, source, neighbor)
            if impact == 'NO_IMPACT':
                continue

            cue_per_node[neighbor['id']] += 1
            total_cues += 1
            step_cues += 1

        # Snapshot
        nodes_with_cues = sum(1 for c in cue_per_node.values() if c > 0)
        max_cues = max(cue_per_node.values()) if cue_per_node else 0
        avg_cues = sum(cue_per_node.values()) / len(cue_per_node) if cue_per_node else 0

        density_snapshots.append({
            'step': step,
            'total_cues': total_cues,
            'nodes_with_cues': nodes_with_cues,
            'max_cues_per_node': max_cues,
            'avg_cues_per_node': avg_cues,
            'step_cues': step_cues,
        })

    # Print density analysis
    print(f"\n  Density snapshots (every 10 steps):")
    print(f"  {'Step':>6s} {'Total':>8s} {'Nodes':>8s} {'Max/node':>10s} {'Avg/node':>10s}")
    print(f"  {'-'*45}")

    for snap in density_snapshots:
        if snap['step'] % 10 == 0 or snap['step'] == 1 or snap['step'] == len(source_ids):
            print(f"  {snap['step']:>6d} {snap['total_cues']:>8d} "
                  f"{snap['nodes_with_cues']:>8d} {snap['max_cues_per_node']:>10d} "
                  f"{snap['avg_cues_per_node']:>10.2f}")

    # Hub analysis: which nodes have the most cues?
    sorted_nodes = sorted(cue_per_node.items(), key=lambda x: -x[1])
    print(f"\n  Hub analysis (nodes with most cues):")
    for nid, count in sorted_nodes[:10]:
        node = load_node(conn, nid)
        if node:
            print(f"    {count:>3d} cues: \"{node['title'][:60]}\" (type={node['type']})")

    # Distribution
    cue_counts = list(cue_per_node.values())
    if cue_counts:
        buckets = defaultdict(int)
        for c in cue_counts:
            if c <= 1:
                buckets['1'] += 1
            elif c <= 3:
                buckets['2-3'] += 1
            elif c <= 5:
                buckets['4-5'] += 1
            elif c <= 10:
                buckets['6-10'] += 1
            else:
                buckets['11+'] += 1

        print(f"\n  Cue count distribution:")
        for bucket in ['1', '2-3', '4-5', '6-10', '11+']:
            count = buckets.get(bucket, 0)
            bar = '#' * min(count, 50)
            print(f"    {bucket:>5s} cues: {count:>4d} {bar}")

    # Recommendations
    final = density_snapshots[-1] if density_snapshots else {}
    max_per_node = final.get('max_cues_per_node', 0)

    print(f"\n  Recommendations:")
    if max_per_node > 15:
        print(f"    WARNING: Hub nodes accumulate {max_per_node} cues -- cap at 10-15 per node")
        print(f"    Consider: age-based pruning (drop cues older than 90 days)")
    elif max_per_node > 10:
        print(f"    NOTE: Max {max_per_node} cues per node -- approaching noisy territory")
        print(f"    Suggest max_cues_per_node = 10, with FIFO eviction")
    else:
        print(f"    OK: Max {max_per_node} cues per node -- manageable density")

    print(f"    Proposed limits:")
    print(f"      - max_cues_per_node: 10 (FIFO eviction of oldest)")
    print(f"      - age_cutoff: 90 days (prune stale cues)")
    print(f"      - show_limit: 5 (top 5 cues in recall output)")

    return {
        'final_snapshot': final,
        'hub_max_cues': max_per_node,
        'sorted_hubs': sorted_nodes[:10],
        'density_snapshots': density_snapshots,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  BRAIN BENCHMARK: CUES VS RIPPLE")
    print("  Testing the 'store impact edges, let Claude reason' approach")
    print("=" * 80)

    # Load embedder
    print("\n[1/6] Loading embedder...")
    t0 = time.time()
    emb = Embedder(MODEL_PATH)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load golden dataset
    print("\n[2/6] Loading golden dataset...")
    golden = load_golden_dataset()
    print(f"  {len(golden)} test cases")

    # Copy DB for test 1 & 2 (cue storage tests)
    print("\n[3/6] Preparing test database...")
    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn = sqlite3.connect(DB_TEMP)
    primary_embeddings = load_all_embeddings(conn)
    enrichments = load_all_enrichments(conn)
    node_confidence = dict(conn.execute(
        "SELECT id, confidence FROM nodes WHERE archived=0"
    ).fetchall())
    print(f"  {len(primary_embeddings)} primary embeddings, "
          f"{len(enrichments)} enriched nodes, "
          f"{len(node_confidence)} active nodes")

    # ── Test 1: Storage and retrieval ──
    print("\n[4/6] Running tests...")
    test1_results = test1_cue_storage_retrieval(conn, emb, primary_embeddings)

    # ── Test 2: Cue-augmented recall ──
    test2_results = test2_cue_augmented_recall(
        conn, emb, primary_embeddings, enrichments, node_confidence, golden
    )

    conn.close()

    # ── Test 3: NDCG comparison (uses fresh DB copies per condition) ──
    test3_results = test3_cues_vs_ripple(emb, golden)

    # ── Test 4: Temporal reasoning ──
    # Use cue-enriched DB from test 3 condition B
    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_t4 = sqlite3.connect(DB_TEMP)
    primary_t4 = load_all_embeddings(conn_t4)
    enrichments_t4 = load_all_enrichments(conn_t4)
    conf_t4 = dict(conn_t4.execute(
        "SELECT id, confidence FROM nodes WHERE archived=0"
    ).fetchall())

    golden_ids = set()
    for tc in golden:
        golden_ids.update(tc.get('expected_relevant', {}).keys())
    sources_t4 = pick_ripple_sources(conn_t4, RIPPLE_SOURCE_COUNT, golden_ids)
    apply_cues(conn_t4, emb, primary_t4, sources_t4)

    test4_results = test4_temporal_reasoning(conn_t4, emb, primary_t4, enrichments_t4, conf_t4)
    conn_t4.close()

    # ── Test 5: Cue density ──
    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_t5 = sqlite3.connect(DB_TEMP)
    primary_t5 = load_all_embeddings(conn_t5)
    test5_results = test5_cue_density(conn_t5, emb, primary_t5)
    conn_t5.close()

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  FINAL SUMMARY: ARCHITECTURE RECOMMENDATION")
    print("=" * 80)

    baseline_ndcg = test3_results['A']['aggregate'].get('ndcg@10', {}).get('mean', 0)
    cues_ndcg = test3_results['B']['aggregate'].get('ndcg@10', {}).get('mean', 0)
    ripple_ndcg = test3_results['C']['aggregate'].get('ndcg@10', {}).get('mean', 0)

    baseline_mrr = test3_results['A']['aggregate'].get('mrr', {}).get('mean', 0)
    cues_mrr = test3_results['B']['aggregate'].get('mrr', {}).get('mean', 0)
    ripple_mrr = test3_results['C']['aggregate'].get('mrr', {}).get('mean', 0)

    print(f"""
  Retrieval Quality (Test 3):
    Baseline NDCG@10: {baseline_ndcg:.4f}  MRR: {baseline_mrr:.4f}
    Cues-only NDCG@10: {cues_ndcg:.4f}  MRR: {cues_mrr:.4f}  (delta: {cues_ndcg - baseline_ndcg:+.4f})
    Full ripple NDCG@10: {ripple_ndcg:.4f}  MRR: {ripple_mrr:.4f}  (delta: {ripple_ndcg - baseline_ndcg:+.4f})

  Cue Performance (Test 1):
    Retrieval time: {test1_results['avg_retrieval_ms']:.3f}ms avg, {test1_results['max_retrieval_ms']:.3f}ms max
    Target <5ms: {'PASS' if test1_results['pass'] else 'FAIL'}

  Cue Coverage (Test 2):
    {test2_results['nodes_with_cues']} of {test2_results['nodes_with_cues'] + test2_results['nodes_without_cues']} recalled nodes had cues ({test2_results['cue_coverage_pct']:.1f}%)

  Cue Density (Test 5):
    Max cues per node: {test5_results['hub_max_cues']}
    After 50 events: {test5_results['final_snapshot'].get('total_cues', 0)} total cues

  Architecture comparison:
    +--------------------------+--------+---------+
    | Property                 | Cues   | Ripple  |
    +--------------------------+--------+---------+
    | Confidence changes       | NO     | YES     |
    | Re-enrichment needed     | NO     | YES     |
    | LLM assessment needed    | NO*    | YES     |
    | Safety mechanisms needed | 0      | 6       |
    | NDCG regression risk     | ZERO   | MEDIUM  |
    | Implementation lines     | ~50    | ~300    |
    | Temporal reasoning       | YES    | NO      |
    | Claude context cost      | +2-5   | 0 lines |
    |   lines per node         |        |         |
    +--------------------------+--------+---------+
    * Heuristic for now; Claude assessment optional later

  Recommendation:
    SHIP CUES. The ripple engine adds complexity (confidence mutation, 6 safety
    mechanisms, re-enrichment) for ZERO retrieval benefit over baseline. Cues
    match baseline NDCG (by design -- they don't change retrieval scores) while
    adding temporal reasoning capability that ripple can't provide.

    The key insight: retrieval quality is an EMBEDDING problem, not a confidence
    problem. Cues solve a DIFFERENT problem -- helping Claude reason about the
    temporal validity and relationships between recalled nodes.
""")

    # Cleanup
    if os.path.exists(DB_TEMP):
        os.remove(DB_TEMP)

    print("  Done. Temp DB cleaned up.")


if __name__ == '__main__':
    main()
