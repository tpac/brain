#!/usr/bin/env python3
"""
brain — Ripple Engine Simulation Benchmark

Simulates a "ripple engine" that propagates new information backward to related
nodes, re-enriching them and adjusting confidence. Measures impact on recall
quality via the golden dataset.

The ripple engine works like this: when a new node is encoded, the brain:
1. Finds related existing nodes (via edges + embedding similarity)
2. For each related node, assesses impact: VALIDATES, CONTRADICTS, or EXTENDS
3. Adjusts confidence: validated +0.05, contradicted -0.10
4. RE-ENRICHES impacted nodes — generates new Q/A/B/K vectors
5. Creates typed edges (validates, contradicts, extends)

Usage:
    python benchmark_ripple_simulation.py
"""

import json
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.metrics import (
    compute_all_metrics, aggregate_metrics,
    ndcg_at_k, mrr as compute_mrr, hit_rate_at_k,
)

OLLAMA_BIN = "/Applications/Ollama.app/Contents/Resources/ollama"
OLLAMA_MODEL = "gemma2:2b"
MODEL_PATH = os.path.expanduser("~/brain/model-package/brain_embedding/model")
DB_SOURCE = os.path.expanduser("~/AgentsContext/brain/brain.db")
DB_TEMP = "/tmp/brain_ripple_test.db"
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')


# ═══════════════════════════════════════════════════════════════
# EMBEDDING ENGINE (standalone, no Brain dependency)
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
# LLM (Ollama)
# ═══════════════════════════════════════════════════════════════

def ollama_generate(prompt: str, timeout: int = 30) -> str:
    """Call ollama CLI to generate text."""
    try:
        result = subprocess.run(
            [OLLAMA_BIN, "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  [ollama] Error: {result.stderr[:200]}", file=sys.stderr)
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"  [ollama] Timeout after {timeout}s", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  [ollama] Exception: {e}", file=sys.stderr)
        return ""


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_golden_dataset() -> List[Dict]:
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def load_all_embeddings(conn: sqlite3.Connection) -> Dict[str, bytes]:
    """Load all node_embeddings (primary vectors)."""
    rows = conn.execute(
        "SELECT ne.node_id, ne.embedding FROM node_embeddings ne "
        "JOIN nodes n ON n.id = ne.node_id WHERE n.archived = 0"
    ).fetchall()
    return {nid: blob for nid, blob in rows if blob}


def load_all_enrichments(conn: sqlite3.Connection) -> Dict[str, List[Tuple[str, bytes]]]:
    """Load all enrichment embeddings. Returns {node_id: [(vector_type, embedding), ...]}."""
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
        "SELECT id, type, title, content, keywords, confidence, created_at "
        "FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if not row:
        return None
    return {
        'id': row[0], 'type': row[1], 'title': row[2],
        'content': row[3] or '', 'keywords': row[4] or '',
        'confidence': row[5], 'created_at': row[6],
    }


def get_neighbors(conn: sqlite3.Connection, node_id: str, limit: int = 5) -> List[Dict]:
    """Get neighbors via edges table with full node data."""
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
# RECALL ENGINE (multi-vector, matching production logic)
# ═══════════════════════════════════════════════════════════════

def recall_multivec(
    query_embedding: bytes,
    primary_embeddings: Dict[str, bytes],
    enrichments: Dict[str, List[Tuple[str, bytes]]],
    node_confidence: Dict[str, Optional[float]],
    embedder_inst: Embedder,
    limit: int = 20,
) -> List[Dict]:
    """
    Multi-vector recall matching production recall_with_embeddings logic.
    For each node: max(primary_sim, enrichment_sims...) * confidence_multiplier.
    Returns list of {node_id, score, best_vector_type}.
    """
    scores = {}
    best_types = {}

    # Primary embeddings
    for node_id, blob in primary_embeddings.items():
        sim = embedder_inst.cosine_similarity(query_embedding, blob)
        scores[node_id] = sim
        best_types[node_id] = 'primary'

    # Enrichment embeddings
    for node_id, enrich_list in enrichments.items():
        for vtype, blob in enrich_list:
            sim = embedder_inst.cosine_similarity(query_embedding, blob)
            if sim > scores.get(node_id, 0):
                scores[node_id] = sim
                best_types[node_id] = vtype

    # Apply confidence multiplier (matching production: [0.1,1.0] -> [0.7, 1.05])
    for node_id in scores:
        conf = node_confidence.get(node_id)
        if conf is not None:
            conf_multiplier = 0.7 + (conf - 0.1) * (1.05 - 0.7) / (1.0 - 0.1)
            conf_multiplier = max(0.7, min(1.05, conf_multiplier))
            scores[node_id] *= conf_multiplier

    # Sort and return
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
            node_confidence, embedder_inst, limit=20,
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

    # Aggregate
    scorable = [r['metrics'] for r in case_results
                if r.get('metrics') and r['metrics'].get('mrr') is not None]
    agg = aggregate_metrics(scorable) if scorable else {}

    # By category
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
# RIPPLE ENGINE SIMULATION
# ═══════════════════════════════════════════════════════════════

def pick_ripple_sources(conn: sqlite3.Connection, n: int = 20,
                        golden_target_ids: Optional[Set[str]] = None) -> List[Dict]:
    """Pick N nodes from the most recent 30% as ripple sources.

    Strategy: pick sources that have edges to golden dataset targets,
    so the ripple actually affects nodes that will be queried.
    Falls back to recent nodes with any neighbors.
    """
    candidates = []
    seen_ids = set()

    # Strategy 1: Find recent nodes connected to golden targets
    if golden_target_ids:
        total = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
        cutoff = int(total * 0.7)

        # Get all recent node IDs
        recent_ids = [r[0] for r in conn.execute(
            "SELECT id FROM nodes WHERE archived=0 ORDER BY created_at ASC LIMIT -1 OFFSET ?",
            (cutoff,)
        ).fetchall()]

        for nid in recent_ids:
            if len(candidates) >= n:
                break
            # Check if this node connects to any golden target
            neighbor_ids = set()
            rows = conn.execute(
                "SELECT CASE WHEN source_id=? THEN target_id ELSE source_id END "
                "FROM edges WHERE source_id=? OR target_id=?",
                (nid, nid, nid)
            ).fetchall()
            for r in rows:
                neighbor_ids.add(r[0])

            golden_overlap = neighbor_ids & golden_target_ids
            if golden_overlap and nid not in seen_ids:
                node = load_node(conn, nid)
                if node:
                    node['neighbor_count'] = len(neighbor_ids)
                    node['golden_neighbors'] = len(golden_overlap)
                    candidates.append(node)
                    seen_ids.add(nid)

    # Strategy 2: Fill remaining slots with recent nodes that have neighbors
    if len(candidates) < n:
        total = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
        offset = int(total * 0.7)
        rows = conn.execute("""
            SELECT id, type, title, content, keywords, confidence, created_at
            FROM nodes WHERE archived = 0
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?
        """, (n * 3, offset)).fetchall()

        for row in rows:
            if len(candidates) >= n:
                break
            nid = row[0]
            if nid in seen_ids:
                continue
            neighbor_count = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE source_id=? OR target_id=?",
                (nid, nid)
            ).fetchone()[0]
            if neighbor_count >= 1:
                node = {
                    'id': row[0], 'type': row[1], 'title': row[2],
                    'content': row[3] or '', 'keywords': row[4] or '',
                    'confidence': row[5], 'created_at': row[6],
                    'neighbor_count': neighbor_count,
                    'golden_neighbors': 0,
                }
                candidates.append(node)
                seen_ids.add(nid)

    return candidates[:n]


def assess_impact(new_node: Dict, neighbor: Dict) -> str:
    """Use LLM to assess impact: VALIDATES, CONTRADICTS, or EXTENDS."""
    prompt = f"""Given these two brain memories, classify the relationship.

New memory: "{new_node['title']}"
Content: "{new_node['content'][:150]}"

Existing memory: "{neighbor['title']}"
Content: "{neighbor['content'][:150]}"

Does the new memory VALIDATE (confirm/support), CONTRADICT (conflict with/correct), or EXTEND (add new aspects to) the existing memory?

Answer with exactly one word: VALIDATES, CONTRADICTS, or EXTENDS
"""
    raw = ollama_generate(prompt, timeout=20)
    raw = raw.strip().upper()

    if 'CONTRADICT' in raw:
        return 'CONTRADICTS'
    elif 'VALIDATE' in raw or 'CONFIRM' in raw:
        return 'VALIDATES'
    else:
        return 'EXTENDS'


def generate_re_enrichment(
    neighbor: Dict,
    new_node: Dict,
    impact_type: str,
    other_neighbors: List[Dict],
) -> Dict[str, str]:
    """Generate new Q/A/B/K enrichment vectors for a neighbor impacted by a ripple."""
    other_text = ""
    for on in other_neighbors[:5]:
        other_text += f"- {on['title']} (keywords: {on.get('keywords', 'none')})\n"
    if not other_text:
        other_text = "- (no other neighbors)\n"

    prompt = f"""New context has arrived that affects this memory.

This memory: "{neighbor['title']}"
Content: "{neighbor['content'][:200]}"

New information: "{new_node['title']}" — {impact_type} this memory.

Its other neighbors:
{other_text}
Given this new context, regenerate:
Q: [one question a user would naturally ask that leads to this memory, considering the new information]
A: [3-5 word phrase using words from neighbors]
B: [one sentence connecting this memory to the new information]
K: [5 comma-separated keywords that now apply]
"""
    raw = ollama_generate(prompt, timeout=30)
    if not raw:
        return {}

    result = {}
    for line in raw.strip().split('\n'):
        line = line.strip()
        # Try exact prefix match first, then fuzzy
        if line.startswith('Q:') or line.startswith('Q '):
            result['question'] = re.sub(r'^Q[:\s]+', '', line).strip().strip('"[]')
        elif line.startswith('A:') or line.startswith('A '):
            result['anchor'] = re.sub(r'^A[:\s]+', '', line).strip().strip('"[]')
        elif line.startswith('B:') or line.startswith('B '):
            result['bridge'] = re.sub(r'^B[:\s]+', '', line).strip().strip('"[]')
        elif line.startswith('K:') or line.startswith('K '):
            result['keywords'] = re.sub(r'^K[:\s]+', '', line).strip().strip('"[]')
        # Also match **Q:** markdown format
        elif line.startswith('**Q'):
            result['question'] = re.sub(r'^\*\*Q\*?\*?[:\s]*', '', line).strip().strip('"[]')
        elif line.startswith('**A'):
            result['anchor'] = re.sub(r'^\*\*A\*?\*?[:\s]*', '', line).strip().strip('"[]')
        elif line.startswith('**B'):
            result['bridge'] = re.sub(r'^\*\*B\*?\*?[:\s]*', '', line).strip().strip('"[]')
        elif line.startswith('**K'):
            result['keywords'] = re.sub(r'^\*\*K\*?\*?[:\s]*', '', line).strip().strip('"[]')

    # If we got nothing from structured parsing, try to extract useful text anyway
    if not result and raw:
        lines = [l.strip() for l in raw.strip().split('\n') if l.strip() and len(l.strip()) > 10]
        if lines:
            result['question'] = lines[0][:200]
            if len(lines) > 1:
                result['bridge'] = lines[1][:200]
            if len(lines) > 2:
                result['keywords'] = lines[2][:200]

    return result


def simulate_ripple(
    conn: sqlite3.Connection,
    embedder_inst: Embedder,
    ripple_sources: List[Dict],
) -> Dict[str, Any]:
    """
    Simulate ripple propagation for each source node.
    Returns stats and modifies the DB in place.
    """
    stats = {
        'total_sources': len(ripple_sources),
        'total_neighbors_touched': 0,
        'total_enrichments_created': 0,
        'total_edges_created': 0,
        'confidence_changes': [],
        'impact_counts': {'VALIDATES': 0, 'CONTRADICTS': 0, 'EXTENDS': 0},
        'per_source': [],
        'examples': [],
    }

    for idx, source in enumerate(ripple_sources):
        source_id = source['id']
        print(f"\n  [{idx+1}/{len(ripple_sources)}] Rippling from: {source['title'][:60]}...")

        neighbors = get_neighbors(conn, source_id, limit=5)
        if not neighbors:
            print(f"    No neighbors found, skipping.")
            continue

        source_stats = {
            'source_id': source_id,
            'source_title': source['title'],
            'neighbors_touched': 0,
            'enrichments_created': 0,
            'edges_created': 0,
            'impacts': [],
        }

        for neighbor in neighbors[:5]:
            neighbor_id = neighbor['id']

            # Step 1: Assess impact
            impact = assess_impact(source, neighbor)
            stats['impact_counts'][impact] += 1
            print(f"    -> {neighbor['title'][:50]}... [{impact}]")

            # Step 2: Adjust confidence
            old_conf = neighbor.get('confidence') or 0.7
            if impact == 'VALIDATES':
                new_conf = min(1.0, old_conf + 0.05)
            elif impact == 'CONTRADICTS':
                new_conf = max(0.1, old_conf - 0.10)
            else:  # EXTENDS
                new_conf = old_conf  # No change for extensions

            conf_delta = new_conf - old_conf
            if conf_delta != 0:
                conn.execute(
                    "UPDATE nodes SET confidence = ? WHERE id = ?",
                    (new_conf, neighbor_id)
                )
                stats['confidence_changes'].append({
                    'node_id': neighbor_id,
                    'title': neighbor['title'],
                    'old': old_conf,
                    'new': new_conf,
                    'delta': conf_delta,
                    'impact': impact,
                })

            # Step 3: Get other neighbors for context
            other_neighbors = get_neighbors(conn, neighbor_id, limit=5)
            other_neighbors = [n for n in other_neighbors if n['id'] != source_id]

            # Step 4: Generate re-enrichment
            enrichment = generate_re_enrichment(neighbor, source, impact, other_neighbors)
            if enrichment:
                texts_to_embed = []
                types_to_store = []
                for vtype in ['question', 'anchor', 'bridge', 'keywords']:
                    if vtype in enrichment and enrichment[vtype]:
                        texts_to_embed.append(enrichment[vtype])
                        types_to_store.append(vtype)

                if texts_to_embed:
                    embeddings = embedder_inst.embed_batch(texts_to_embed)
                    now = datetime.now(timezone.utc).isoformat()
                    for i, vtype in enumerate(types_to_store):
                        if i < len(embeddings):
                            eid = uuid.uuid4().hex[:16]
                            # Use standard vector_type (CHECK constraint enforces it)
                            # Store ripple provenance in text prefix instead
                            conn.execute(
                                """INSERT OR REPLACE INTO node_enrichments
                                   (id, node_id, vector_type, text, embedding, model, created_at)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (f"ripple_{eid}", neighbor_id, vtype,
                                 f"[ripple:{impact}] {enrichment[vtype]}", embeddings[i],
                                 'snowflake-arctic-embed-m', now)
                            )
                            source_stats['enrichments_created'] += 1
                            stats['total_enrichments_created'] += 1

            # Step 5: Create/update typed edge
            edge_relation = f"ripple_{impact.lower()}"
            existing = conn.execute(
                "SELECT 1 FROM edges WHERE (source_id=? AND target_id=?) OR (source_id=? AND target_id=?)",
                (source_id, neighbor_id, neighbor_id, source_id)
            ).fetchone()
            if existing:
                # Update existing edge relation
                conn.execute(
                    "UPDATE edges SET relation=?, weight=MAX(weight, 0.5) WHERE (source_id=? AND target_id=?) OR (source_id=? AND target_id=?)",
                    (edge_relation, source_id, neighbor_id, neighbor_id, source_id)
                )
                # Edge already existed, just updated relation
            else:
                conn.execute(
                    "INSERT INTO edges (source_id, target_id, relation, weight, created_at) VALUES (?, ?, ?, ?, ?)",
                    (source_id, neighbor_id, edge_relation, 0.5,
                     datetime.now(timezone.utc).isoformat())
                )
                source_stats['edges_created'] += 1
                stats['total_edges_created'] += 1

            source_stats['neighbors_touched'] += 1
            stats['total_neighbors_touched'] += 1
            source_stats['impacts'].append({
                'neighbor_id': neighbor_id,
                'neighbor_title': neighbor['title'],
                'impact': impact,
                'conf_delta': conf_delta,
                'enrichment_types': list(enrichment.keys()) if enrichment else [],
            })

        stats['per_source'].append(source_stats)

        # Collect examples (first 5)
        if len(stats['examples']) < 5:
            stats['examples'].append({
                'source': source['title'],
                'neighbors_touched': source_stats['neighbors_touched'],
                'impacts': source_stats['impacts'],
                'enrichments_created': source_stats['enrichments_created'],
            })

    conn.commit()
    return stats


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def print_comparison(baseline: Dict, rippled: Dict):
    """Print baseline vs rippled comparison."""
    b_agg = baseline['aggregate']
    r_agg = rippled['aggregate']

    print(f"\n{'='*75}")
    print(f"  BASELINE vs RIPPLED COMPARISON")
    print(f"{'='*75}")
    print(f"  {'Metric':<20s} {'Baseline':>10s} {'Rippled':>10s} {'Delta':>10s} {'Change':>10s}")
    print(f"  {'-'*65}")

    for metric in ['ndcg@10', 'mrr', 'hit_rate@10', 'precision@5', 'recall@10']:
        b_val = b_agg.get(metric, {}).get('mean', 0)
        r_val = r_agg.get(metric, {}).get('mean', 0)
        delta = r_val - b_val
        pct = (delta / b_val * 100) if b_val > 0 else 0
        print(f"  {metric:<20s} {b_val:>10.4f} {r_val:>10.4f} {delta:>+10.4f} {pct:>+9.1f}%")

    print(f"  {'-'*65}")
    b_p = baseline['passed']
    r_p = rippled['passed']
    b_t = baseline['total']
    print(f"  {'Passed':<20s} {b_p:>7d}/{b_t} {r_p:>7d}/{b_t}   {r_p-b_p:>+7d}")
    print()


def print_category_comparison(baseline: Dict, rippled: Dict):
    """Print per-category comparison."""
    b_cats = baseline['by_category']
    r_cats = rippled['by_category']
    all_cats = sorted(set(list(b_cats.keys()) + list(r_cats.keys())))

    print(f"  {'Category':<22s} {'B-NDCG':>8s} {'R-NDCG':>8s} {'Delta':>8s}  {'B-MRR':>8s} {'R-MRR':>8s} {'Delta':>8s}")
    print(f"  {'-'*75}")

    for cat in all_cats:
        b_ndcg = b_cats.get(cat, {}).get('ndcg@10', {}).get('mean', 0)
        r_ndcg = r_cats.get(cat, {}).get('ndcg@10', {}).get('mean', 0)
        d_ndcg = r_ndcg - b_ndcg
        b_mrr = b_cats.get(cat, {}).get('mrr', {}).get('mean', 0)
        r_mrr = r_cats.get(cat, {}).get('mrr', {}).get('mean', 0)
        d_mrr = r_mrr - b_mrr
        print(f"  {cat:<22s} {b_ndcg:>8.3f} {r_ndcg:>8.3f} {d_ndcg:>+8.3f}  {b_mrr:>8.3f} {r_mrr:>8.3f} {d_mrr:>+8.3f}")
    print()


def print_flipped_cases(baseline: Dict, rippled: Dict):
    """Find cases that flipped from fail to pass (or vice versa)."""
    b_map = {r['id']: r for r in baseline['case_results']}
    r_map = {r['id']: r for r in rippled['case_results']}

    fail_to_pass = []
    pass_to_fail = []

    for tc_id in b_map:
        b = b_map[tc_id]
        r = r_map.get(tc_id)
        if not r:
            continue
        if not b['passed'] and r['passed']:
            fail_to_pass.append(tc_id)
        elif b['passed'] and not r['passed']:
            pass_to_fail.append(tc_id)

    print(f"  FLIPPED CASES:")
    print(f"    Fail -> Pass: {len(fail_to_pass)}")
    for tc_id in fail_to_pass[:10]:
        b = b_map[tc_id]
        r = r_map[tc_id]
        b_mrr = b['metrics'].get('mrr', 0)
        r_mrr = r['metrics'].get('mrr', 0)
        print(f"      + [{tc_id}] {b['query'][:50]} (MRR {b_mrr:.3f} -> {r_mrr:.3f})")

    print(f"    Pass -> Fail: {len(pass_to_fail)}")
    for tc_id in pass_to_fail[:10]:
        b = b_map[tc_id]
        r = r_map[tc_id]
        b_mrr = b['metrics'].get('mrr', 0)
        r_mrr = r['metrics'].get('mrr', 0)
        print(f"      - [{tc_id}] {b['query'][:50]} (MRR {b_mrr:.3f} -> {r_mrr:.3f})")
    print()


def print_ripple_stats(stats: Dict):
    """Print ripple statistics."""
    print(f"\n{'='*75}")
    print(f"  RIPPLE ENGINE STATISTICS")
    print(f"{'='*75}")
    print(f"  Sources processed:     {stats['total_sources']}")
    print(f"  Neighbors touched:     {stats['total_neighbors_touched']}")
    print(f"  Enrichments created:   {stats['total_enrichments_created']}")
    print(f"  Edges created:         {stats['total_edges_created']}")
    print(f"  Avg neighbors/source:  {stats['total_neighbors_touched']/max(stats['total_sources'],1):.1f}")
    print()
    print(f"  Impact distribution:")
    for impact, count in stats['impact_counts'].items():
        pct = count / max(sum(stats['impact_counts'].values()), 1) * 100
        print(f"    {impact:<15s}: {count:>4d} ({pct:.0f}%)")
    print()

    # Confidence changes
    changes = stats['confidence_changes']
    if changes:
        up = [c for c in changes if c['delta'] > 0]
        down = [c for c in changes if c['delta'] < 0]
        avg_delta = sum(abs(c['delta']) for c in changes) / len(changes)
        print(f"  Confidence changes:")
        print(f"    Upward:   {len(up)} nodes")
        print(f"    Downward: {len(down)} nodes")
        print(f"    Avg |delta|: {avg_delta:.3f}")
    print()


def print_examples(stats: Dict, baseline: Dict, rippled: Dict):
    """Print 5 specific ripple examples with before/after."""
    b_map = {r['id']: r for r in baseline['case_results']}
    r_map = {r['id']: r for r in rippled['case_results']}

    print(f"  RIPPLE EXAMPLES (5 sources):")
    print(f"  {'='*70}")
    for ex in stats.get('examples', [])[:5]:
        print(f"\n  Source: {ex['source'][:65]}")
        print(f"  Neighbors touched: {ex['neighbors_touched']}, Enrichments: {ex['enrichments_created']}")
        for imp in ex.get('impacts', [])[:3]:
            print(f"    -> {imp['neighbor_title'][:50]}  [{imp['impact']}]  conf_delta={imp['conf_delta']:+.2f}")
            if imp['enrichment_types']:
                print(f"       enrichments: {', '.join(imp['enrichment_types'])}")
    print()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0_total = time.time()

    # ── Setup ──
    print(f"[ripple] Source DB: {DB_SOURCE}")
    print(f"[ripple] Temp DB:   {DB_TEMP}")

    # Copy fresh DB
    shutil.copy2(DB_SOURCE, DB_TEMP)
    conn = sqlite3.connect(DB_TEMP)
    conn.execute("PRAGMA journal_mode=WAL")

    golden = load_golden_dataset()
    print(f"[ripple] Golden dataset: {len(golden)} cases")

    # Init embedder
    print(f"[ripple] Loading embedding model...")
    embedder_inst = Embedder(MODEL_PATH)

    # ── Phase 1: Baseline ──
    print(f"\n{'='*75}")
    print(f"  PHASE 1: BASELINE EVALUATION")
    print(f"{'='*75}")

    t_baseline = time.time()
    primary_embeddings = load_all_embeddings(conn)
    enrichments = load_all_enrichments(conn)
    node_confidence = dict(conn.execute(
        "SELECT id, confidence FROM nodes WHERE archived=0"
    ).fetchall())

    print(f"  {len(primary_embeddings)} primary embeddings, "
          f"{sum(len(v) for v in enrichments.values())} enrichment vectors, "
          f"{len(node_confidence)} nodes")

    baseline_result = evaluate_golden(
        golden, primary_embeddings, enrichments, node_confidence,
        embedder_inst, label="baseline"
    )
    b_agg = baseline_result['aggregate']
    print(f"  Baseline NDCG@10: {b_agg.get('ndcg@10', {}).get('mean', 0):.4f}")
    print(f"  Baseline MRR:     {b_agg.get('mrr', {}).get('mean', 0):.4f}")
    print(f"  Baseline Passed:  {baseline_result['passed']}/{baseline_result['total']}")
    print(f"  Baseline time:    {time.time()-t_baseline:.1f}s")

    # ── Phase 2: Ripple Simulation ──
    print(f"\n{'='*75}")
    print(f"  PHASE 2: RIPPLE SIMULATION (20 nodes)")
    print(f"{'='*75}")

    t_ripple = time.time()
    golden_target_ids = set()
    for tc in golden:
        for nid in tc.get('expected_relevant', {}):
            golden_target_ids.add(nid)
    ripple_sources = pick_ripple_sources(conn, n=20, golden_target_ids=golden_target_ids)
    golden_connected = sum(1 for s in ripple_sources if s.get('golden_neighbors', 0) > 0)
    print(f"  Selected {len(ripple_sources)} ripple sources ({golden_connected} connected to golden targets)")

    ripple_stats = simulate_ripple(conn, embedder_inst, ripple_sources)
    print(f"\n  Ripple simulation time: {time.time()-t_ripple:.1f}s")

    # ── Phase 3: Post-Ripple Evaluation ──
    print(f"\n{'='*75}")
    print(f"  PHASE 3: POST-RIPPLE EVALUATION")
    print(f"{'='*75}")

    t_rippled = time.time()
    # Reload everything from the modified DB
    primary_embeddings_2 = load_all_embeddings(conn)
    enrichments_2 = load_all_enrichments(conn)
    node_confidence_2 = dict(conn.execute(
        "SELECT id, confidence FROM nodes WHERE archived=0"
    ).fetchall())

    print(f"  {len(primary_embeddings_2)} primary, "
          f"{sum(len(v) for v in enrichments_2.values())} enrichments (was {sum(len(v) for v in enrichments.values())}), "
          f"delta: +{sum(len(v) for v in enrichments_2.values()) - sum(len(v) for v in enrichments.values())}")

    rippled_result = evaluate_golden(
        golden, primary_embeddings_2, enrichments_2, node_confidence_2,
        embedder_inst, label="rippled"
    )
    r_agg = rippled_result['aggregate']
    print(f"  Rippled NDCG@10: {r_agg.get('ndcg@10', {}).get('mean', 0):.4f}")
    print(f"  Rippled MRR:     {r_agg.get('mrr', {}).get('mean', 0):.4f}")
    print(f"  Rippled Passed:  {rippled_result['passed']}/{rippled_result['total']}")
    print(f"  Rippled eval time: {time.time()-t_rippled:.1f}s")

    # ── Phase 4: Report ──
    print(f"\n{'='*75}")
    print(f"  PHASE 4: COMPARISON REPORT")
    print(f"{'='*75}")

    print_comparison(baseline_result, rippled_result)
    print_category_comparison(baseline_result, rippled_result)
    print_flipped_cases(baseline_result, rippled_result)
    print_ripple_stats(ripple_stats)
    print_examples(ripple_stats, baseline_result, rippled_result)

    # ── Save JSON results ──
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, 'ripple_simulation.json')

    json_report = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'model': 'snowflake-arctic-embed-m-v1.5',
            'llm': OLLAMA_MODEL,
            'ripple_sources': len(ripple_sources),
            'golden_cases': len(golden),
        },
        'baseline': {
            'ndcg@10': b_agg.get('ndcg@10', {}).get('mean', 0),
            'mrr': b_agg.get('mrr', {}).get('mean', 0),
            'hit_rate@10': b_agg.get('hit_rate@10', {}).get('mean', 0),
            'passed': baseline_result['passed'],
            'total': baseline_result['total'],
            'by_category': {
                cat: {
                    'ndcg@10': m.get('ndcg@10', {}).get('mean', 0),
                    'mrr': m.get('mrr', {}).get('mean', 0),
                }
                for cat, m in baseline_result.get('by_category', {}).items()
            },
        },
        'rippled': {
            'ndcg@10': r_agg.get('ndcg@10', {}).get('mean', 0),
            'mrr': r_agg.get('mrr', {}).get('mean', 0),
            'hit_rate@10': r_agg.get('hit_rate@10', {}).get('mean', 0),
            'passed': rippled_result['passed'],
            'total': rippled_result['total'],
            'by_category': {
                cat: {
                    'ndcg@10': m.get('ndcg@10', {}).get('mean', 0),
                    'mrr': m.get('mrr', {}).get('mean', 0),
                }
                for cat, m in rippled_result.get('by_category', {}).items()
            },
        },
        'ripple_stats': {
            'total_sources': ripple_stats['total_sources'],
            'total_neighbors_touched': ripple_stats['total_neighbors_touched'],
            'total_enrichments_created': ripple_stats['total_enrichments_created'],
            'total_edges_created': ripple_stats['total_edges_created'],
            'impact_counts': ripple_stats['impact_counts'],
            'confidence_changes': ripple_stats['confidence_changes'],
            'examples': ripple_stats['examples'],
        },
        'flipped_cases': {
            'fail_to_pass': [],
            'pass_to_fail': [],
        },
    }

    # Compute flipped cases
    b_map = {r['id']: r for r in baseline_result['case_results']}
    r_map = {r['id']: r for r in rippled_result['case_results']}
    for tc_id in b_map:
        b = b_map[tc_id]
        r = r_map.get(tc_id)
        if not r:
            continue
        if not b['passed'] and r['passed']:
            json_report['flipped_cases']['fail_to_pass'].append({
                'id': tc_id, 'query': b['query'],
                'baseline_mrr': b['metrics'].get('mrr', 0),
                'rippled_mrr': r['metrics'].get('mrr', 0),
            })
        elif b['passed'] and not r['passed']:
            json_report['flipped_cases']['pass_to_fail'].append({
                'id': tc_id, 'query': b['query'],
                'baseline_mrr': b['metrics'].get('mrr', 0),
                'rippled_mrr': r['metrics'].get('mrr', 0),
            })

    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"\n[ripple] JSON report saved: {json_path}")

    # Cleanup
    conn.close()
    total_time = time.time() - t0_total
    print(f"[ripple] Total time: {total_time:.1f}s")
    print(f"[ripple] Done.")


if __name__ == '__main__':
    main()
