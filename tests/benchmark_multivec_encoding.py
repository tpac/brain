#!/usr/bin/env python3
"""
brain — Multi-Vector Encoding Benchmark

Tests whether storing additional "question" embeddings per node improves recall.
For each target node in the golden dataset, generates 3 questions via LLM,
embeds them, and tests whether matching against all embeddings (original + questions)
produces better retrieval metrics.

3 prompt variations tested:
  V1 (bare):       Just asks for questions
  V2 (structured): Provides related nodes + structured fields
  V3 (motivated):  Same as V2 + motivational framing

Usage:
    python benchmark_multivec_encoding.py [brain.db path]
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
import tempfile
import time
from typing import Dict, List, Optional, Set, Tuple, Any

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.metrics import (
    compute_all_metrics, aggregate_metrics,
    ndcg_at_k, mrr as compute_mrr, hit_rate_at_k,
)
from tests.brain_test_base import copy_brain_for_testing

OLLAMA_BIN = "/Applications/Ollama.app/Contents/Resources/ollama"
OLLAMA_MODEL = "gemma2:2b"
MODEL_PATH = os.path.expanduser("~/brain/model-package/brain_embedding/model")

# Baseline control numbers
BASELINE = {"ndcg@10": 0.204, "mrr": 0.202, "passed": 34, "total": 104}


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

        # Register custom model
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


def parse_questions(raw: str) -> List[str]:
    """Extract up to 3 questions from messy LLM output."""
    lines = raw.strip().split('\n')
    questions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip numbering, bullets, checkboxes
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-*•□]\s*', '', line)
        line = line.strip()
        if not line:
            continue
        if line.endswith('?'):
            questions.append(line)

    # If fewer than 3 questions, take first non-empty lines
    if len(questions) < 3:
        for line in lines:
            line = line.strip()
            line = re.sub(r'^[\d]+[.)]\s*', '', line)
            line = re.sub(r'^[-*•□]\s*', '', line)
            line = line.strip()
            if line and line not in questions:
                questions.append(line)
            if len(questions) >= 3:
                break

    return questions[:3]


# ═══════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════

def prompt_v1(title: str, content: str, related: List[Dict]) -> str:
    """V1 (bare): Just ask for questions."""
    return f"""List 3 questions this node answers. Just the questions, one per line.
Node: "{title}"
Content: "{content[:200]}"
"""


def prompt_v2(title: str, content: str, related: List[Dict]) -> str:
    """V2 (structured): Related nodes + structured fields."""
    related_text = ""
    if related:
        related_lines = []
        for r in related:
            related_lines.append(f"- {r['title']} ({r['type']}, confidence {r.get('confidence', 'N/A')})")
        related_text = "\n".join(related_lines)
    else:
        related_text = "(none found)"

    return f"""The brain found these related memories:
{related_text}

New node: "{title}"
Content: "{content[:200]}"

Answer these (one per line, no numbering):
□ 3 questions a user would ask that this node answers
□ Which related memories does this validate, contradict, or extend?
□ Key vocabulary terms that should link to this node
"""


def prompt_v3(title: str, content: str, related: List[Dict]) -> str:
    """V3 (motivated): Same as V2 + motivational framing."""
    related_text = ""
    if related:
        related_lines = []
        for r in related:
            related_lines.append(f"- {r['title']} ({r['type']}, confidence {r.get('confidence', 'N/A')})")
        related_text = "\n".join(related_lines)
    else:
        related_text = "(none found)"

    return f"""You are the brain's encoding engine. Every unencoded connection is lost forever. The human will search for this memory using their own words — not yours. Think about how THEY would ask.

The brain already knows:
{related_text}

New memory arriving: "{title}"
Content: "{content[:200]}"

□ 3 questions the human would naturally ask that lead to this memory (use THEIR vocabulary, not technical abstractions)
□ Which existing memories does this change? validate? contradict?
□ What words would the human use that should find this?
"""


PROMPT_VARIANTS = {
    "v1_bare": prompt_v1,
    "v2_structured": prompt_v2,
    "v3_motivated": prompt_v3,
}


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_golden_dataset() -> List[Dict]:
    path = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
    with open(path) as f:
        return json.load(f)


def get_target_node_ids(golden: List[Dict]) -> Set[str]:
    """Extract all unique expected node IDs from golden dataset."""
    ids = set()
    for tc in golden:
        for nid in tc.get('expected_relevant', {}):
            ids.add(nid)
    return ids


def load_node_data(conn: sqlite3.Connection, node_id: str) -> Optional[Dict]:
    """Load node title/content/type/confidence from DB."""
    row = conn.execute(
        "SELECT id, type, title, content, confidence FROM nodes WHERE id = ?",
        (node_id,)
    ).fetchone()
    if not row:
        return None
    return {
        'id': row[0],
        'type': row[1],
        'title': row[2],
        'content': row[3] or '',
        'confidence': row[4],
    }


def load_related_nodes(conn: sqlite3.Connection, node_id: str, limit: int = 5) -> List[Dict]:
    """Find related nodes via edges table."""
    rows = conn.execute("""
        SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as other_id,
               e.weight, e.relation
        FROM edges e
        WHERE e.source_id = ? OR e.target_id = ?
        ORDER BY e.weight DESC
        LIMIT ?
    """, (node_id, node_id, node_id, limit)).fetchall()

    related = []
    for other_id, weight, relation in rows:
        node = conn.execute(
            "SELECT id, type, title, confidence FROM nodes WHERE id = ?",
            (other_id,)
        ).fetchone()
        if node:
            related.append({
                'id': node[0],
                'type': node[1],
                'title': node[2],
                'confidence': node[3],
                'weight': weight,
                'relation': relation,
            })
    return related


def load_original_embeddings(conn: sqlite3.Connection) -> Dict[str, bytes]:
    """Load all node embeddings from DB."""
    rows = conn.execute(
        "SELECT ne.node_id, ne.embedding FROM node_embeddings ne "
        "JOIN nodes n ON n.id = ne.node_id WHERE n.archived = 0"
    ).fetchall()
    return {nid: blob for nid, blob in rows if blob}


# ═══════════════════════════════════════════════════════════════
# ENRICHMENT: Generate question embeddings
# ═══════════════════════════════════════════════════════════════

def enrich_targets(
    conn: sqlite3.Connection,
    embedder: Embedder,
    target_ids: Set[str],
) -> Dict[str, Dict[str, List]]:
    """
    For each target node, generate questions via all 3 prompt variants
    and embed them.

    Returns:
        {variant_name: {node_id: [embedding_bytes, ...]}}
    """
    enrichments = {v: {} for v in PROMPT_VARIANTS}
    example_enrichments = []  # For reporting

    total = len(target_ids)
    print(f"\n{'='*70}")
    print(f"  ENRICHING {total} TARGET NODES")
    print(f"{'='*70}\n")

    for idx, node_id in enumerate(sorted(target_ids)):
        node = load_node_data(conn, node_id)
        if not node:
            print(f"  [{idx+1}/{total}] SKIP {node_id[:12]}... (not found in DB)")
            continue

        related = load_related_nodes(conn, node_id)
        title = node['title']
        content = node['content']

        print(f"  [{idx+1}/{total}] {title[:60]}...")
        if related:
            print(f"           {len(related)} related nodes found")

        is_example = len(example_enrichments) < 5

        for variant_name, prompt_fn in PROMPT_VARIANTS.items():
            prompt = prompt_fn(title, content, related)
            raw = ollama_generate(prompt, timeout=45)

            if not raw:
                print(f"           {variant_name}: no LLM output")
                enrichments[variant_name][node_id] = []
                continue

            questions = parse_questions(raw)
            if not questions:
                print(f"           {variant_name}: no questions parsed")
                enrichments[variant_name][node_id] = []
                continue

            # Embed the questions
            q_embeddings = embedder.embed_batch(questions)
            enrichments[variant_name][node_id] = q_embeddings

            if variant_name == "v1_bare":
                print(f"           questions: {questions}")

            # Collect example for report
            if is_example and variant_name == "v1_bare":
                example_enrichments.append({
                    'node_id': node_id,
                    'title': title,
                    'related': [r['title'] for r in related[:3]],
                    'questions': {v: [] for v in PROMPT_VARIANTS},
                })

        # Fill in all variant questions for examples
        if is_example and example_enrichments and example_enrichments[-1]['node_id'] == node_id:
            for variant_name in PROMPT_VARIANTS:
                prompt = PROMPT_VARIANTS[variant_name](title, content, related)
                # Re-use the raw output we already generated (wasteful to call again)
                # Actually we already called it above, so just parse again
                # For examples, re-generate to capture questions per variant
                raw = ollama_generate(PROMPT_VARIANTS[variant_name](title, content, related), timeout=45)
                qs = parse_questions(raw)
                example_enrichments[-1]['questions'][variant_name] = qs

    print(f"\n  Enrichment complete.\n")
    return enrichments, example_enrichments


def enrich_targets_optimized(
    conn: sqlite3.Connection,
    embedder_inst: Embedder,
    target_ids: Set[str],
) -> Tuple[Dict[str, Dict[str, List]], List[Dict]]:
    """
    Optimized version: generate all prompts, call LLM once per (node, variant),
    collect examples from the first pass (no double-calling).
    """
    enrichments = {v: {} for v in PROMPT_VARIANTS}
    example_enrichments = []

    total = len(target_ids)
    sorted_ids = sorted(target_ids)

    print(f"\n{'='*70}")
    print(f"  ENRICHING {total} TARGET NODES")
    print(f"{'='*70}\n")

    t0 = time.time()

    for idx, node_id in enumerate(sorted_ids):
        node = load_node_data(conn, node_id)
        if not node:
            print(f"  [{idx+1}/{total}] SKIP {node_id[:12]}... (not in DB)")
            continue

        related = load_related_nodes(conn, node_id)
        title = node['title']
        content = node['content']

        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (total - idx - 1) / rate if rate > 0 else 0
        print(f"  [{idx+1}/{total}] {title[:55]}... ({len(related)} edges, ETA {eta:.0f}s)")

        is_example = len(example_enrichments) < 5
        example_entry = None
        if is_example:
            example_entry = {
                'node_id': node_id,
                'title': title,
                'related': [r['title'][:50] for r in related[:3]],
                'questions': {},
            }

        for variant_name, prompt_fn in PROMPT_VARIANTS.items():
            prompt = prompt_fn(title, content, related)
            raw = ollama_generate(prompt, timeout=45)

            questions = parse_questions(raw) if raw else []

            if questions:
                q_embeddings = embedder_inst.embed_batch(questions)
                enrichments[variant_name][node_id] = q_embeddings
            else:
                enrichments[variant_name][node_id] = []

            if example_entry is not None:
                example_entry['questions'][variant_name] = questions

        if example_entry is not None:
            example_enrichments.append(example_entry)

    elapsed = time.time() - t0
    print(f"\n  Enrichment done in {elapsed:.1f}s ({total} nodes x 3 variants = {total*3} LLM calls)\n")
    return enrichments, example_enrichments


# ═══════════════════════════════════════════════════════════════
# MODIFIED RECALL: Multi-vector scoring
# ═══════════════════════════════════════════════════════════════

def multivec_recall(
    query_embedding: bytes,
    original_embeddings: Dict[str, bytes],
    extra_embeddings: Dict[str, List[bytes]],
    embedder_inst: Embedder,
    limit: int = 20,
) -> List[str]:
    """
    Recall with multi-vector scoring.
    For each node, take max similarity across original + question embeddings.
    Return top-k node IDs.
    """
    scores = {}

    for node_id, orig_blob in original_embeddings.items():
        sim = embedder_inst.cosine_similarity(query_embedding, orig_blob)
        scores[node_id] = sim

        # Check question embeddings for this node
        if node_id in extra_embeddings:
            for q_blob in extra_embeddings[node_id]:
                q_sim = embedder_inst.cosine_similarity(query_embedding, q_blob)
                if q_sim > scores[node_id]:
                    scores[node_id] = q_sim

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked[:limit]]


def baseline_recall(
    query_embedding: bytes,
    original_embeddings: Dict[str, bytes],
    embedder_inst: Embedder,
    limit: int = 20,
) -> List[str]:
    """Standard single-vector recall for baseline comparison."""
    scores = {}
    for node_id, orig_blob in original_embeddings.items():
        sim = embedder_inst.cosine_similarity(query_embedding, orig_blob)
        scores[node_id] = sim

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked[:limit]]


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_variant(
    golden: List[Dict],
    original_embeddings: Dict[str, bytes],
    extra_embeddings: Dict[str, List[bytes]],
    embedder_inst: Embedder,
    variant_name: str,
) -> Dict[str, Any]:
    """Run golden eval with multi-vector recall for a specific variant."""
    case_results = []

    for tc in golden:
        query = tc['query']
        expected = tc.get('expected_relevant', {})
        min_hit_rate = tc.get('min_hit_rate_at_10', 0.0)

        query_emb = embedder_inst.embed(query)

        if extra_embeddings is not None:
            retrieved_ids = multivec_recall(
                query_emb, original_embeddings, extra_embeddings, embedder_inst, limit=20
            )
        else:
            retrieved_ids = baseline_recall(
                query_emb, original_embeddings, embedder_inst, limit=20
            )

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

    passed = sum(1 for r in case_results if r['passed'])
    total = len(case_results)

    return {
        'variant': variant_name,
        'case_results': case_results,
        'aggregate': agg,
        'by_category': category_agg,
        'passed': passed,
        'total': total,
    }


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def print_variant_report(result: Dict, delta_baseline: Dict):
    """Print results for a single variant."""
    agg = result['aggregate']
    variant = result['variant']
    passed = result['passed']
    total = result['total']

    ndcg = agg.get('ndcg@10', {}).get('mean', 0)
    mrr_val = agg.get('mrr', {}).get('mean', 0)
    hit_rate = agg.get('hit_rate@10', {}).get('mean', 0)

    d_ndcg = ndcg - delta_baseline.get('ndcg@10', 0)
    d_mrr = mrr_val - delta_baseline.get('mrr', 0)
    d_passed = passed - delta_baseline.get('passed', 0)

    print(f"\n  ─── {variant} ───")
    print(f"    NDCG@10:    {ndcg:.3f}  ({d_ndcg:+.3f})")
    print(f"    MRR:        {mrr_val:.3f}  ({d_mrr:+.3f})")
    print(f"    Hit@10:     {hit_rate:.3f}")
    print(f"    Passed:     {passed}/{total}  ({d_passed:+d})")

    # Category breakdown
    cat_agg = result.get('by_category', {})
    if cat_agg:
        print(f"    ─── Categories ───")
        for cat, metrics in sorted(cat_agg.items()):
            cat_ndcg = metrics.get('ndcg@10', {}).get('mean', 0)
            cat_mrr = metrics.get('mrr', {}).get('mean', 0)
            cat_n = metrics.get('mrr', {}).get('count', 0)
            print(f"      {cat:>20s}: NDCG={cat_ndcg:.3f}  MRR={cat_mrr:.3f}  (n={cat_n})")


def print_comparison_table(results: Dict[str, Dict], baseline_result: Dict):
    """Print side-by-side comparison of all variants."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Variant':<20s} {'NDCG@10':>10s} {'MRR':>10s} {'Hit@10':>10s} {'Passed':>10s}")
    print(f"  {'-'*60}")

    # Baseline
    b_agg = baseline_result['aggregate']
    b_ndcg = b_agg.get('ndcg@10', {}).get('mean', 0)
    b_mrr = b_agg.get('mrr', {}).get('mean', 0)
    b_hit = b_agg.get('hit_rate@10', {}).get('mean', 0)
    b_passed = baseline_result['passed']
    b_total = baseline_result['total']
    print(f"  {'baseline':<20s} {b_ndcg:>10.3f} {b_mrr:>10.3f} {b_hit:>10.3f} {b_passed:>4d}/{b_total}")

    for vname, result in sorted(results.items()):
        agg = result['aggregate']
        ndcg = agg.get('ndcg@10', {}).get('mean', 0)
        mrr_val = agg.get('mrr', {}).get('mean', 0)
        hit_rate = agg.get('hit_rate@10', {}).get('mean', 0)
        passed = result['passed']
        total = result['total']
        d_ndcg = ndcg - b_ndcg
        print(f"  {vname:<20s} {ndcg:>10.3f} {mrr_val:>10.3f} {hit_rate:>10.3f} {passed:>4d}/{total}  ({d_ndcg:+.3f})")

    print(f"  {'-'*60}")
    print(f"  Control baseline:  NDCG={BASELINE['ndcg@10']}, MRR={BASELINE['mrr']}, {BASELINE['passed']}/{BASELINE['total']} passed")
    print()


def print_example_enrichments(examples: List[Dict]):
    """Print sample enrichments."""
    if not examples:
        return
    print(f"\n{'='*70}")
    print(f"  EXAMPLE ENRICHMENTS (5 nodes)")
    print(f"{'='*70}")

    for ex in examples:
        print(f"\n  Node: {ex['title']}")
        if ex['related']:
            print(f"  Related: {', '.join(ex['related'][:3])}")
        else:
            print(f"  Related: (none)")

        for variant, questions in ex.get('questions', {}).items():
            qs = questions[:3] if questions else ['(none)']
            print(f"    {variant}:")
            for q in qs:
                print(f"      - {q}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]

    # Find brain.db
    db_path = None
    for arg in args:
        if arg.endswith('.db') and os.path.exists(arg):
            db_path = arg
            break
    if not db_path:
        db_path = os.path.expanduser("~/AgentsContext/brain/brain.db")
    if not os.path.exists(db_path):
        print(f"ERROR: brain.db not found at {db_path}")
        sys.exit(1)

    print(f"[benchmark] Brain DB: {db_path}")

    # Copy to temp
    tmp_dir, tmp_db = copy_brain_for_testing(db_path)
    print(f"[benchmark] Working on temp copy: {tmp_db}")

    conn = sqlite3.connect(tmp_db)
    conn.execute("PRAGMA journal_mode=WAL")

    # Load golden dataset and identify targets
    golden = load_golden_dataset()
    target_ids = get_target_node_ids(golden)
    print(f"[benchmark] Golden dataset: {len(golden)} cases, {len(target_ids)} unique target nodes")

    # Initialize embedder
    print(f"[benchmark] Loading embedding model...")
    embedder_inst = Embedder(MODEL_PATH)

    # Load all original embeddings
    print(f"[benchmark] Loading original embeddings...")
    original_embeddings = load_original_embeddings(conn)
    print(f"[benchmark] {len(original_embeddings)} node embeddings loaded")

    # ─── Phase 1: Baseline (pure embedding, no enrichment) ───
    print(f"\n{'='*70}")
    print(f"  PHASE 1: BASELINE (pure semantic, no enrichment)")
    print(f"{'='*70}")
    baseline_result = evaluate_variant(
        golden, original_embeddings, None, embedder_inst, "baseline"
    )
    b_agg = baseline_result['aggregate']
    print(f"  NDCG@10: {b_agg.get('ndcg@10', {}).get('mean', 0):.3f}")
    print(f"  MRR:     {b_agg.get('mrr', {}).get('mean', 0):.3f}")
    print(f"  Passed:  {baseline_result['passed']}/{baseline_result['total']}")

    baseline_delta = {
        'ndcg@10': b_agg.get('ndcg@10', {}).get('mean', 0),
        'mrr': b_agg.get('mrr', {}).get('mean', 0),
        'passed': baseline_result['passed'],
    }

    # ─── Phase 2: Enrich target nodes ───
    enrichments, examples = enrich_targets_optimized(conn, embedder_inst, target_ids)

    # ─── Phase 3: Evaluate each variant ───
    print(f"\n{'='*70}")
    print(f"  PHASE 3: EVALUATING VARIANTS")
    print(f"{'='*70}")

    variant_results = {}
    for variant_name in PROMPT_VARIANTS:
        print(f"\n  Evaluating {variant_name}...")
        result = evaluate_variant(
            golden, original_embeddings,
            enrichments[variant_name],
            embedder_inst, variant_name,
        )
        variant_results[variant_name] = result
        print_variant_report(result, baseline_delta)

    # ─── Phase 4: Report ───
    print_comparison_table(variant_results, baseline_result)
    print_example_enrichments(examples)

    # Save JSON results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, 'multivec_benchmark.json')

    json_report = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'model': 'snowflake-arctic-embed-m-v1.5',
            'llm': OLLAMA_MODEL,
            'target_nodes': len(target_ids),
            'golden_cases': len(golden),
        },
        'baseline': {
            'ndcg@10': b_agg.get('ndcg@10', {}).get('mean', 0),
            'mrr': b_agg.get('mrr', {}).get('mean', 0),
            'passed': baseline_result['passed'],
            'total': baseline_result['total'],
        },
        'variants': {},
        'examples': examples,
    }
    for vname, vresult in variant_results.items():
        vagg = vresult['aggregate']
        json_report['variants'][vname] = {
            'ndcg@10': vagg.get('ndcg@10', {}).get('mean', 0),
            'mrr': vagg.get('mrr', {}).get('mean', 0),
            'hit_rate@10': vagg.get('hit_rate@10', {}).get('mean', 0),
            'passed': vresult['passed'],
            'total': vresult['total'],
            'by_category': {
                cat: {
                    'ndcg@10': m.get('ndcg@10', {}).get('mean', 0),
                    'mrr': m.get('mrr', {}).get('mean', 0),
                }
                for cat, m in vresult.get('by_category', {}).items()
            },
        }

    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"\n[benchmark] JSON report saved: {json_path}")

    # Cleanup
    conn.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[benchmark] Done.")


if __name__ == '__main__':
    main()
