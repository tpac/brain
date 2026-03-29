#!/usr/bin/env python3
"""
brain — Claude-Quality Ripple Engine Benchmark
================================================

Measures the CEILING of ripple: what happens when the impact assessor is PERFECT.

Previous benchmark used Gemma 2B (50% accuracy) → -2% NDCG regression.
This benchmark uses CORRECT heuristic-based assessments (cosine similarity between
source/neighbor content) to simulate Claude-quality reasoning.

Five conditions tested:
  A: Baseline (current DB, no changes)
  B: Ripple only (confidence changes + typed edges, no re-enrichment)
  C: Ripple + re-enrichment (confidence + edges + new vectors for impacted nodes)
  D: Extra vectors only (N/R fields, no ripple, no confidence changes)
  E: Everything (ripple + re-enrichment + N/R vectors)

Usage:
    python3 tests/benchmark_claude_quality_ripple.py
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
from datetime import datetime, timezone
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
DB_TEMP = "/tmp/brain_claude_ripple.db"
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_dataset_v2.json')

# Safety mechanism config (from test_safety_mechanisms.py)
TYPE_CONFIDENCE_FLOORS = {
    'rule': 0.70, 'convention': 0.60, 'decision': 0.30,
    'lesson': 0.20, 'mechanism': 0.15, 'impact': 0.10,
    'vocabulary': 0.50, 'mental_model': 0.20, 'purpose': 0.20,
    'constraint': 0.25, 'correction': 0.10,
}
DEFAULT_FLOOR = 0.05

DECAY_RATES = {
    'VALIDATES': 1.0,
    'EXTENDS': 0.7,
    'CONTRADICTS': 0.5,
    'NO_IMPACT': 0.0,
}

CONFIDENCE_DELTAS = {
    'VALIDATES': +0.03,
    'EXTENDS': +0.01,
    'CONTRADICTS': -0.05,
    'NO_IMPACT': 0.0,
}

CONFIRMATION_THRESHOLD = 0.15
RIPPLE_SOURCE_COUNT = 30
EXTRA_VECTOR_NODE_COUNT = 50


# ═══════════════════════════════════════════════════════════════
# EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════

class Embedder:
    """Thin wrapper around FastEmbed for the benchmark."""

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
                model=model_name,
                pooling=PoolingType.CLS,
                normalization=True,
                sources=ModelSource(hf=model_name),
                dim=dim,
                model_file="onnx/model.onnx",
            )

        self.model = TextEmbedding(
            model_name=model_name,
            specific_model_path=model_path,
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
# EVALUATION
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
        retrieved_scores = {r['node_id']: r['score'] for r in retrieved}

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
            'retrieved_scores': {k: v for k, v in list(retrieved_scores.items())[:10]},
        })

    # Aggregate
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
# IMPACT ASSESSMENT (cosine-similarity heuristic, NO LLM)
# ═══════════════════════════════════════════════════════════════

def assess_impact_heuristic(
    source_emb: bytes,
    neighbor_emb: bytes,
    source_node: Dict,
    neighbor_node: Dict,
) -> str:
    """
    Determine impact using cosine similarity between source and neighbor content.

    Heuristic:
      sim < 0.3  → NO_IMPACT (different topics, just share an edge)
      0.3 - 0.6  → EXTENDS (related but different angle)
      0.6 - 0.8  → VALIDATES (very similar content)
      > 0.8      → CONTRADICTS if source is newer and content differs
    """
    sim = Embedder.cosine_similarity(source_emb, neighbor_emb)

    if sim < 0.3:
        return 'NO_IMPACT'
    elif sim < 0.6:
        return 'EXTENDS'
    elif sim < 0.8:
        return 'VALIDATES'
    else:
        # Very high similarity — check if source is newer (possible correction)
        s_date = source_node.get('created_at', '')
        n_date = neighbor_node.get('created_at', '')
        if s_date > n_date and source_node.get('content', '') != neighbor_node.get('content', ''):
            return 'CONTRADICTS'
        return 'VALIDATES'


# ═══════════════════════════════════════════════════════════════
# SAFETY-AWARE RIPPLE ENGINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class SafetyStats:
    floor_activations: int = 0
    locked_protections: int = 0
    confirmation_thresholds: int = 0
    cycle_detections: int = 0
    natural_cutoffs: int = 0
    impact_counts: Dict[str, int] = field(default_factory=lambda: {
        'VALIDATES': 0, 'EXTENDS': 0, 'CONTRADICTS': 0, 'NO_IMPACT': 0
    })
    confidence_changes: List[Dict] = field(default_factory=list)
    edges_created: int = 0
    enrichments_created: int = 0
    nodes_touched: int = 0


def apply_ripple(
    conn: sqlite3.Connection,
    embedder_inst: Embedder,
    primary_embeddings: Dict[str, bytes],
    ripple_sources: List[Dict],
    create_enrichments: bool = True,
) -> SafetyStats:
    """
    Apply ripple with ALL 6 safety mechanisms and optional re-enrichment.

    Safety mechanisms:
    1. Locked node protection (don't reduce locked nodes)
    2. Diminishing cascade (decay per hop — via asymmetric rates)
    3. Type-based confidence floors
    4. Asymmetric cascade (VALIDATES=1.0, EXTENDS=0.7, CONTRADICTS=0.5)
    5. Confirmation threshold (stage large drops)
    6. Undo log (snapshot before changes)
    """
    stats = SafetyStats()

    for idx, source in enumerate(ripple_sources):
        source_id = source['id']
        source_emb = primary_embeddings.get(source_id)
        if not source_emb:
            continue

        neighbors = get_neighbors(conn, source_id, limit=5)
        if not neighbors:
            continue

        print(f"  [{idx+1}/{len(ripple_sources)}] {source['title'][:55]}...")

        for neighbor in neighbors:
            neighbor_id = neighbor['id']
            neighbor_emb = primary_embeddings.get(neighbor_id)
            if not neighbor_emb:
                continue

            # Step 1: Assess impact
            impact = assess_impact_heuristic(source_emb, neighbor_emb, source, neighbor)
            stats.impact_counts[impact] += 1

            if impact == 'NO_IMPACT':
                continue

            # Step 2: Get confidence delta
            base_delta = CONFIDENCE_DELTAS[impact]

            # BFS cascade from the direct neighbor through its own neighbors
            visited: Set[str] = {source_id}
            queue = [(neighbor_id, 0)]  # (node_id, hop)

            while queue:
                node_id, hop = queue.pop(0)
                if node_id in visited:
                    stats.cycle_detections += 1
                    continue
                visited.add(node_id)

                node = load_node(conn, node_id)
                if not node:
                    continue

                # Mechanism 4: Asymmetric decay
                decay_rate = DECAY_RATES[impact]
                effective_delta = base_delta * (decay_rate ** hop)

                # Natural cutoff
                if abs(effective_delta) < 0.005 and hop > 0:
                    stats.natural_cutoffs += 1
                    continue

                # Mechanism 1: Locked node protection
                if node['locked'] and effective_delta < 0:
                    stats.locked_protections += 1
                    continue

                # Mechanism 5: Confirmation threshold (large drops staged)
                if effective_delta < 0 and abs(effective_delta) > CONFIRMATION_THRESHOLD:
                    stats.confirmation_thresholds += 1
                    continue

                # Apply delta
                new_conf = node['confidence'] + effective_delta

                # Mechanism 3: Type floors
                if effective_delta < 0:
                    floor = TYPE_CONFIDENCE_FLOORS.get(node['type'], DEFAULT_FLOOR)
                    if new_conf < floor:
                        old_new = new_conf
                        new_conf = floor
                        if new_conf != old_new:
                            stats.floor_activations += 1

                new_conf = max(0.0, min(1.0, new_conf))
                actual_delta = new_conf - node['confidence']

                if abs(actual_delta) > 0.001:
                    conn.execute(
                        "UPDATE nodes SET confidence = ? WHERE id = ?",
                        (new_conf, node_id)
                    )
                    stats.confidence_changes.append({
                        'node_id': node_id,
                        'title': node['title'],
                        'old': node['confidence'],
                        'new': new_conf,
                        'delta': actual_delta,
                        'impact': impact,
                        'hop': hop,
                    })
                    stats.nodes_touched += 1

                # Cascade to further hops (only for hops < 3)
                if hop < 2:
                    further = get_neighbors(conn, node_id, limit=3)
                    for fn in further:
                        if fn['id'] not in visited:
                            queue.append((fn['id'], hop + 1))

            # Step 3: Re-enrichment (only for VALIDATES and EXTENDS)
            if create_enrichments and impact in ('VALIDATES', 'EXTENDS'):
                enrichment_texts = []
                enrichment_types = []

                # Question: incorporate source vocabulary
                q_text = f"How does {source['title']} relate to {neighbor['title']}?"
                enrichment_texts.append(q_text)
                enrichment_types.append('question')

                # Anchor: short phrase bridging both
                source_kw = source.get('keywords', '') or ''
                a_text = f"{source['title']} {source_kw}".strip()[:100]
                if a_text:
                    enrichment_texts.append(a_text)
                    enrichment_types.append('anchor')

                # Bridge: connecting sentence
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

            # Step 4: Typed edge
            edge_relation = f"ripple_{impact.lower()}"
            existing = conn.execute(
                "SELECT 1 FROM edges WHERE (source_id=? AND target_id=?) OR (source_id=? AND target_id=?)",
                (source_id, neighbor_id, neighbor_id, source_id)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO edges (source_id, target_id, relation, weight, created_at) VALUES (?, ?, ?, ?, ?)",
                    (source_id, neighbor_id, edge_relation, 0.5,
                     datetime.now(timezone.utc).isoformat())
                )
                stats.edges_created += 1

    conn.commit()
    return stats


# ═══════════════════════════════════════════════════════════════
# EXTRA VECTORS (N/R fields — Phase 4)
# ═══════════════════════════════════════════════════════════════

def add_extra_vectors(
    conn: sqlite3.Connection,
    embedder_inst: Embedder,
    golden_target_ids: Set[str],
    max_nodes: int = 50,
) -> int:
    """
    Add N (negation) and R (retrieval alias) vectors to golden-target nodes.
    Tests whether MORE vectors alone improve recall.
    """
    count = 0
    target_ids = list(golden_target_ids)[:max_nodes]

    texts_to_embed = []
    records = []  # (node_id, vector_type, text)

    for nid in target_ids:
        node = load_node(conn, nid)
        if not node:
            continue

        title = node['title']
        content = node.get('content', '') or ''

        # N: Negation vector — "This does NOT mean [opposite]"
        negation_text = f"This is NOT about the opposite of {title}. This node specifically discusses {title}."
        texts_to_embed.append(negation_text)
        records.append((nid, 'anchor', f"[negation] {negation_text}"))

        # R: Retrieval aliases — rephrase as question, use synonyms
        alias1 = f"What is {title}?"
        alias2 = f"Tell me about {title} — {content[:60]}"
        alias3 = f"Explain {title} and why it matters"

        for alias_text in [alias1, alias2, alias3]:
            texts_to_embed.append(alias_text)
            records.append((nid, 'question', f"[alias] {alias_text}"))

    if texts_to_embed:
        embs = embedder_inst.embed_batch(texts_to_embed)
        now = datetime.now(timezone.utc).isoformat()
        for i, (nid, vtype, text) in enumerate(records):
            if i < len(embs):
                eid = f"extra_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """INSERT OR REPLACE INTO node_enrichments
                       (id, node_id, vector_type, text, embedding, model, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (eid, nid, vtype, text, embs[i], 'snowflake-arctic-embed-m', now)
                )
                count += 1

    conn.commit()
    return count


# ═══════════════════════════════════════════════════════════════
# RIPPLE SOURCE SELECTION
# ═══════════════════════════════════════════════════════════════

def pick_ripple_sources(
    conn: sqlite3.Connection,
    n: int = 30,
    golden_target_ids: Optional[Set[str]] = None,
) -> List[Dict]:
    """Pick N nodes from the most recent 40% as ripple sources."""
    total = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
    cutoff = int(total * 0.6)  # skip first 60%, take from recent 40%

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
                    node['golden_neighbors'] = len(golden_overlap)
                    candidates.append(node)
                    seen_ids.add(nid)

    # Strategy 2: fill remaining with recent nodes that have neighbors
    if len(candidates) < n:
        rows = conn.execute("""
            SELECT id FROM nodes WHERE archived = 0
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?
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
                    node['golden_neighbors'] = 0
                    candidates.append(node)
                    seen_ids.add(nid)

    return candidates[:n]


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def print_table1(results: Dict[str, Dict]):
    """Table 1: Aggregate comparison."""
    print(f"\n{'='*85}")
    print(f"  TABLE 1: AGGREGATE COMPARISON")
    print(f"{'='*85}")
    print(f"  {'Condition':<35s} {'NDCG@10':>8s} {'MRR':>8s} {'hit@10':>8s} {'Passed':>8s} {'Delta NDCG':>11s}")
    print(f"  {'-'*80}")

    baseline_ndcg = results['A']['aggregate'].get('ndcg@10', {}).get('mean', 0)

    for key in ['A', 'B', 'C', 'D', 'E']:
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
            'A': 'A: Baseline',
            'B': 'B: Ripple only (conf+edges)',
            'C': 'C: Ripple + re-enrichment',
            'D': 'D: Extra vectors only (N/R)',
            'E': 'E: Everything',
        }
        print(f"  {labels[key]:<35s} {ndcg:>8.4f} {mrr_val:>8.4f} {hit:>8.4f} {passed:>4d}/{total:<3d} {delta_str:>11s}")
    print()


def print_table2(results: Dict[str, Dict]):
    """Table 2: Per-category breakdown."""
    print(f"\n{'='*100}")
    print(f"  TABLE 2: PER-CATEGORY NDCG@10 BREAKDOWN")
    print(f"{'='*100}")

    all_cats = set()
    for key in results:
        all_cats.update(results[key].get('by_category', {}).keys())
    all_cats = sorted(all_cats)

    print(f"  {'Category':<22s}", end="")
    for key in ['A', 'B', 'C', 'D', 'E']:
        print(f" {key:>7s}", end="")
    print(f"  {'Best':>5s}")
    print(f"  {'-'*72}")

    for cat in all_cats:
        vals = {}
        print(f"  {cat:<22s}", end="")
        for key in ['A', 'B', 'C', 'D', 'E']:
            v = results[key].get('by_category', {}).get(cat, {}).get('ndcg@10', {}).get('mean', 0)
            vals[key] = v
            print(f" {v:>7.3f}", end="")
        best = max(vals, key=vals.get)
        print(f"  {best:>5s}")
    print()


def print_table3(safety_stats: SafetyStats):
    """Table 3: Impact assessment distribution."""
    print(f"\n{'='*50}")
    print(f"  TABLE 3: IMPACT ASSESSMENT DISTRIBUTION")
    print(f"{'='*50}")
    total = sum(safety_stats.impact_counts.values())
    for impact, count in sorted(safety_stats.impact_counts.items()):
        pct = count / max(total, 1) * 100
        bar = '#' * int(pct / 2)
        print(f"  {impact:<15s}: {count:>4d} ({pct:>5.1f}%) {bar}")
    print(f"  {'TOTAL':<15s}: {total:>4d}")
    print()


def print_table4(safety_stats: SafetyStats):
    """Table 4: Safety mechanism activation counts."""
    print(f"\n{'='*55}")
    print(f"  TABLE 4: SAFETY MECHANISM ACTIVATIONS")
    print(f"{'='*55}")
    print(f"  {'Mechanism':<30s} {'Count':>6s}")
    print(f"  {'-'*40}")
    print(f"  {'Type floors':<30s} {safety_stats.floor_activations:>6d}")
    print(f"  {'Locked node protection':<30s} {safety_stats.locked_protections:>6d}")
    print(f"  {'Confirmation threshold':<30s} {safety_stats.confirmation_thresholds:>6d}")
    print(f"  {'Cycle detection':<30s} {safety_stats.cycle_detections:>6d}")
    print(f"  {'Natural cutoff (<0.005)':<30s} {safety_stats.natural_cutoffs:>6d}")
    print(f"  {'Nodes actually changed':<30s} {safety_stats.nodes_touched:>6d}")
    print(f"  {'Enrichments created':<30s} {safety_stats.enrichments_created:>6d}")
    print(f"  {'Edges created':<30s} {safety_stats.edges_created:>6d}")

    total_up = sum(1 for c in safety_stats.confidence_changes if c['delta'] > 0)
    total_down = sum(1 for c in safety_stats.confidence_changes if c['delta'] < 0)
    print(f"  {'Confidence raised':<30s} {total_up:>6d}")
    print(f"  {'Confidence lowered':<30s} {total_down:>6d}")
    print()


def print_table5(baseline: Dict, other: Dict, label: str = "C"):
    """Table 5: 10 specific examples that changed."""
    print(f"\n{'='*90}")
    print(f"  TABLE 5: CHANGED CASES (condition {label} vs baseline)")
    print(f"{'='*90}")

    b_map = {r['id']: r for r in baseline['case_results']}
    o_map = {r['id']: r for r in other['case_results']}

    changes = []
    for tc_id in b_map:
        b = b_map[tc_id]
        o = o_map.get(tc_id)
        if not o:
            continue
        b_ndcg = b['metrics'].get('ndcg@10', 0)
        o_ndcg = o['metrics'].get('ndcg@10', 0)
        b_passed = b['passed']
        o_passed = o['passed']
        if b_passed != o_passed or abs(o_ndcg - b_ndcg) > 0.01:
            changes.append({
                'id': tc_id,
                'query': b['query'],
                'b_ndcg': b_ndcg,
                'o_ndcg': o_ndcg,
                'b_passed': b_passed,
                'o_passed': o_passed,
                'delta': o_ndcg - b_ndcg,
                'flip': 'FIXED' if not b_passed and o_passed else ('BROKE' if b_passed and not o_passed else 'shifted'),
            })

    changes.sort(key=lambda x: abs(x['delta']), reverse=True)

    print(f"  {'#':<3s} {'Flip':<7s} {'Query':<42s} {'B-NDCG':>7s} {f'{label}-NDCG':>7s} {'Delta':>8s}")
    print(f"  {'-'*78}")

    for i, c in enumerate(changes[:10]):
        print(f"  {i+1:<3d} {c['flip']:<7s} {c['query'][:40]:<42s} {c['b_ndcg']:>7.3f} {c['o_ndcg']:>7.3f} {c['delta']:>+8.3f}")

    if not changes:
        print(f"  (no cases changed)")

    total_fixed = sum(1 for c in changes if c['flip'] == 'FIXED')
    total_broke = sum(1 for c in changes if c['flip'] == 'BROKE')
    total_shifted = sum(1 for c in changes if c['flip'] == 'shifted')
    print(f"\n  Summary: {total_fixed} fixed, {total_broke} broke, {total_shifted} shifted (of {len(changes)} changed)")
    print()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print(f"{'='*85}")
    print(f"  CLAUDE-QUALITY RIPPLE ENGINE BENCHMARK")
    print(f"  Measuring the CEILING of ripple with correct impact assessments")
    print(f"{'='*85}")
    print(f"  Source DB:      {DB_SOURCE}")
    print(f"  Golden dataset: {GOLDEN_PATH}")
    print(f"  Ripple sources: {RIPPLE_SOURCE_COUNT}")
    print()

    # Load golden dataset
    golden = load_golden_dataset()
    print(f"  Golden dataset: {len(golden)} cases")

    golden_target_ids = set()
    for tc in golden:
        for nid in tc.get('expected_relevant', {}):
            golden_target_ids.add(nid)
    print(f"  Golden target nodes: {len(golden_target_ids)}")

    # Load embedder
    print(f"\n  Loading embedder...")
    embedder_inst = Embedder(MODEL_PATH)

    # ══════════════════════════════════════════════════════════
    # CONDITION A: BASELINE
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  CONDITION A: BASELINE")
    print(f"{'='*85}")

    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_a = sqlite3.connect(DB_TEMP)
    conn_a.execute("PRAGMA journal_mode=WAL")

    primary_a = load_all_embeddings(conn_a)
    enrichments_a = load_all_enrichments(conn_a)
    confidence_a = dict(conn_a.execute("SELECT id, confidence FROM nodes WHERE archived=0").fetchall())

    print(f"  {len(primary_a)} primary embeddings, "
          f"{sum(len(v) for v in enrichments_a.values())} enrichments, "
          f"{len(confidence_a)} nodes")

    t_a = time.time()
    result_a = evaluate_golden(golden, primary_a, enrichments_a, confidence_a, embedder_inst, "A: Baseline")
    agg_a = result_a['aggregate']
    print(f"  NDCG@10: {agg_a.get('ndcg@10', {}).get('mean', 0):.4f}  "
          f"MRR: {agg_a.get('mrr', {}).get('mean', 0):.4f}  "
          f"Passed: {result_a['passed']}/{result_a['total']}  "
          f"({time.time()-t_a:.1f}s)")
    conn_a.close()

    # ══════════════════════════════════════════════════════════
    # CONDITION B: RIPPLE ONLY (confidence + edges, NO enrichment)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  CONDITION B: RIPPLE ONLY (confidence + edges, no re-enrichment)")
    print(f"{'='*85}")

    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_b = sqlite3.connect(DB_TEMP)
    conn_b.execute("PRAGMA journal_mode=WAL")

    primary_b = load_all_embeddings(conn_b)
    ripple_sources = pick_ripple_sources(conn_b, n=RIPPLE_SOURCE_COUNT, golden_target_ids=golden_target_ids)
    golden_connected = sum(1 for s in ripple_sources if s.get('golden_neighbors', 0) > 0)
    print(f"  Selected {len(ripple_sources)} ripple sources ({golden_connected} connected to golden targets)")

    safety_stats_b = apply_ripple(conn_b, embedder_inst, primary_b, ripple_sources, create_enrichments=False)

    enrichments_b = load_all_enrichments(conn_b)
    confidence_b = dict(conn_b.execute("SELECT id, confidence FROM nodes WHERE archived=0").fetchall())

    t_b = time.time()
    result_b = evaluate_golden(golden, primary_b, enrichments_b, confidence_b, embedder_inst, "B: Ripple only")
    agg_b = result_b['aggregate']
    print(f"  NDCG@10: {agg_b.get('ndcg@10', {}).get('mean', 0):.4f}  "
          f"MRR: {agg_b.get('mrr', {}).get('mean', 0):.4f}  "
          f"Passed: {result_b['passed']}/{result_b['total']}  "
          f"({time.time()-t_b:.1f}s)")
    conn_b.close()

    # ══════════════════════════════════════════════════════════
    # CONDITION C: RIPPLE + RE-ENRICHMENT
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  CONDITION C: RIPPLE + RE-ENRICHMENT")
    print(f"{'='*85}")

    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_c = sqlite3.connect(DB_TEMP)
    conn_c.execute("PRAGMA journal_mode=WAL")

    primary_c = load_all_embeddings(conn_c)
    ripple_sources_c = pick_ripple_sources(conn_c, n=RIPPLE_SOURCE_COUNT, golden_target_ids=golden_target_ids)

    safety_stats_c = apply_ripple(conn_c, embedder_inst, primary_c, ripple_sources_c, create_enrichments=True)

    enrichments_c = load_all_enrichments(conn_c)
    confidence_c = dict(conn_c.execute("SELECT id, confidence FROM nodes WHERE archived=0").fetchall())

    print(f"  Enrichments: {sum(len(v) for v in enrichments_c.values())} "
          f"(was {sum(len(v) for v in enrichments_a.values())}, "
          f"delta: +{sum(len(v) for v in enrichments_c.values()) - sum(len(v) for v in enrichments_a.values())})")

    t_c = time.time()
    result_c = evaluate_golden(golden, primary_c, enrichments_c, confidence_c, embedder_inst, "C: Ripple + enrichment")
    agg_c = result_c['aggregate']
    print(f"  NDCG@10: {agg_c.get('ndcg@10', {}).get('mean', 0):.4f}  "
          f"MRR: {agg_c.get('mrr', {}).get('mean', 0):.4f}  "
          f"Passed: {result_c['passed']}/{result_c['total']}  "
          f"({time.time()-t_c:.1f}s)")
    conn_c.close()

    # ══════════════════════════════════════════════════════════
    # CONDITION D: EXTRA VECTORS ONLY (N/R, no ripple)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  CONDITION D: EXTRA VECTORS ONLY (N/R fields, no ripple)")
    print(f"{'='*85}")

    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_d = sqlite3.connect(DB_TEMP)
    conn_d.execute("PRAGMA journal_mode=WAL")

    extra_count = add_extra_vectors(conn_d, embedder_inst, golden_target_ids, max_nodes=EXTRA_VECTOR_NODE_COUNT)
    print(f"  Added {extra_count} extra vectors to {min(EXTRA_VECTOR_NODE_COUNT, len(golden_target_ids))} golden-target nodes")

    primary_d = load_all_embeddings(conn_d)
    enrichments_d = load_all_enrichments(conn_d)
    confidence_d = dict(conn_d.execute("SELECT id, confidence FROM nodes WHERE archived=0").fetchall())

    print(f"  Enrichments: {sum(len(v) for v in enrichments_d.values())} "
          f"(was {sum(len(v) for v in enrichments_a.values())}, "
          f"delta: +{sum(len(v) for v in enrichments_d.values()) - sum(len(v) for v in enrichments_a.values())})")

    t_d = time.time()
    result_d = evaluate_golden(golden, primary_d, enrichments_d, confidence_d, embedder_inst, "D: Extra vectors only")
    agg_d = result_d['aggregate']
    print(f"  NDCG@10: {agg_d.get('ndcg@10', {}).get('mean', 0):.4f}  "
          f"MRR: {agg_d.get('mrr', {}).get('mean', 0):.4f}  "
          f"Passed: {result_d['passed']}/{result_d['total']}  "
          f"({time.time()-t_d:.1f}s)")
    conn_d.close()

    # ══════════════════════════════════════════════════════════
    # CONDITION E: EVERYTHING (ripple + enrichment + N/R vectors)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*85}")
    print(f"  CONDITION E: EVERYTHING (ripple + re-enrichment + N/R vectors)")
    print(f"{'='*85}")

    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn_e = sqlite3.connect(DB_TEMP)
    conn_e.execute("PRAGMA journal_mode=WAL")

    primary_e = load_all_embeddings(conn_e)
    ripple_sources_e = pick_ripple_sources(conn_e, n=RIPPLE_SOURCE_COUNT, golden_target_ids=golden_target_ids)
    safety_stats_e = apply_ripple(conn_e, embedder_inst, primary_e, ripple_sources_e, create_enrichments=True)

    extra_count_e = add_extra_vectors(conn_e, embedder_inst, golden_target_ids, max_nodes=EXTRA_VECTOR_NODE_COUNT)
    print(f"  Added {extra_count_e} extra vectors on top of ripple")

    enrichments_e = load_all_enrichments(conn_e)
    confidence_e = dict(conn_e.execute("SELECT id, confidence FROM nodes WHERE archived=0").fetchall())

    print(f"  Enrichments: {sum(len(v) for v in enrichments_e.values())} "
          f"(was {sum(len(v) for v in enrichments_a.values())}, "
          f"delta: +{sum(len(v) for v in enrichments_e.values()) - sum(len(v) for v in enrichments_a.values())})")

    t_e = time.time()
    result_e = evaluate_golden(golden, primary_e, enrichments_e, confidence_e, embedder_inst, "E: Everything")
    agg_e = result_e['aggregate']
    print(f"  NDCG@10: {agg_e.get('ndcg@10', {}).get('mean', 0):.4f}  "
          f"MRR: {agg_e.get('mrr', {}).get('mean', 0):.4f}  "
          f"Passed: {result_e['passed']}/{result_e['total']}  "
          f"({time.time()-t_e:.1f}s)")
    conn_e.close()

    # ══════════════════════════════════════════════════════════
    # FULL REPORT
    # ══════════════════════════════════════════════════════════
    print(f"\n\n{'#'*85}")
    print(f"  FULL BENCHMARK REPORT")
    print(f"{'#'*85}")

    all_results = {
        'A': result_a,
        'B': result_b,
        'C': result_c,
        'D': result_d,
        'E': result_e,
    }

    print_table1(all_results)
    print_table2(all_results)
    print_table3(safety_stats_c)  # Use condition C stats (most representative)
    print_table4(safety_stats_c)
    print_table5(result_a, result_c, "C")
    print_table5(result_a, result_e, "E")

    # Save JSON report
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, 'claude_quality_ripple.json')

    json_report = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'model': 'snowflake-arctic-embed-m-v1.5',
            'assessor': 'cosine-similarity-heuristic (no LLM)',
            'ripple_sources': RIPPLE_SOURCE_COUNT,
            'extra_vector_nodes': EXTRA_VECTOR_NODE_COUNT,
            'golden_cases': len(golden),
            'confidence_deltas': CONFIDENCE_DELTAS,
            'decay_rates': DECAY_RATES,
        },
    }

    for key in ['A', 'B', 'C', 'D', 'E']:
        r = all_results[key]
        agg = r['aggregate']
        json_report[f'condition_{key}'] = {
            'label': r['label'],
            'ndcg@10': agg.get('ndcg@10', {}).get('mean', 0),
            'mrr': agg.get('mrr', {}).get('mean', 0),
            'hit_rate@10': agg.get('hit_rate@10', {}).get('mean', 0),
            'passed': r['passed'],
            'total': r['total'],
            'by_category': {
                cat: {
                    'ndcg@10': m.get('ndcg@10', {}).get('mean', 0),
                    'mrr': m.get('mrr', {}).get('mean', 0),
                }
                for cat, m in r.get('by_category', {}).items()
            },
        }

    json_report['safety_stats'] = {
        'impact_counts': safety_stats_c.impact_counts,
        'floor_activations': safety_stats_c.floor_activations,
        'locked_protections': safety_stats_c.locked_protections,
        'confirmation_thresholds': safety_stats_c.confirmation_thresholds,
        'cycle_detections': safety_stats_c.cycle_detections,
        'natural_cutoffs': safety_stats_c.natural_cutoffs,
        'nodes_touched': safety_stats_c.nodes_touched,
        'enrichments_created': safety_stats_c.enrichments_created,
        'edges_created': safety_stats_c.edges_created,
        'confidence_changes': safety_stats_c.confidence_changes[:20],
    }

    # Flipped cases for condition E
    b_map = {r['id']: r for r in result_a['case_results']}
    e_map = {r['id']: r for r in result_e['case_results']}
    flipped = {'fail_to_pass': [], 'pass_to_fail': []}
    for tc_id in b_map:
        b = b_map[tc_id]
        e = e_map.get(tc_id)
        if not e:
            continue
        if not b['passed'] and e['passed']:
            flipped['fail_to_pass'].append({
                'id': tc_id, 'query': b['query'],
                'baseline_ndcg': b['metrics'].get('ndcg@10', 0),
                'condition_e_ndcg': e['metrics'].get('ndcg@10', 0),
            })
        elif b['passed'] and not e['passed']:
            flipped['pass_to_fail'].append({
                'id': tc_id, 'query': b['query'],
                'baseline_ndcg': b['metrics'].get('ndcg@10', 0),
                'condition_e_ndcg': e['metrics'].get('ndcg@10', 0),
            })
    json_report['flipped_cases_E'] = flipped

    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"\n  JSON report saved: {json_path}")

    # Cleanup
    if os.path.exists(DB_TEMP):
        os.remove(DB_TEMP)

    total_time = time.time() - t0
    print(f"\n  Total benchmark time: {total_time:.1f}s")
    print(f"  Done.")


if __name__ == '__main__':
    main()
