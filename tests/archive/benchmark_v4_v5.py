#!/usr/bin/env python3
"""
brain — Multi-Vector Encoding Benchmark: V2 vs V4 vs V5

Compares 3 multi-vector encoding strategies:
  V2 (control):      Question-based enrichment (proven NDCG 0.704, 91/104)
  V4 (anchor-bridge): Anchor phrases + bridge sentence + keyword vector
  V5 (hybrid):       Question + anchor + bridge + keyword vector

Each variant generates extra embedding vectors per node via LLM, then
recall uses max(cosine) across all vectors for ranking.

Usage:
    python benchmark_v4_v5.py [brain.db path]
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

# Control numbers from previous runs
CONTROL = {
    "baseline": {"ndcg@10": 0.325, "passed": 56},
    "v2": {"ndcg@10": 0.704, "passed": 91},
    "cross_encoder_best": {"ndcg@10": 0.518, "passed": 61},
}


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
# PARSERS
# ═══════════════════════════════════════════════════════════════

def strip_markdown(text: str) -> str:
    """Remove common markdown formatting."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)       # *italic*
    text = re.sub(r'`(.*?)`', r'\1', text)         # `code`
    return text.strip()


def parse_v2_questions(raw: str) -> List[str]:
    """Extract up to 3 questions from V2 output (same as original benchmark)."""
    lines = raw.strip().split('\n')
    questions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip numbering, bullets, checkboxes
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-*\u2022\u25a1]\s*', '', line)
        line = strip_markdown(line)
        if not line:
            continue
        if line.endswith('?'):
            questions.append(line)

    # If fewer than 3 questions, take first non-empty lines
    if len(questions) < 3:
        for line in lines:
            line = line.strip()
            line = re.sub(r'^[\d]+[.)]\s*', '', line)
            line = re.sub(r'^[-*\u2022\u25a1]\s*', '', line)
            line = strip_markdown(line)
            if line and line not in questions:
                questions.append(line)
            if len(questions) >= 3:
                break

    return questions[:3]


def parse_labeled_line(raw: str, prefix: str) -> Optional[str]:
    """Find a line containing 'PREFIX:' and return the text after it."""
    for line in raw.split('\n'):
        line = line.strip()
        # Look for prefix anywhere in the line (robust to leading chars)
        idx = line.find(f'{prefix}:')
        if idx != -1:
            text = line[idx + len(prefix) + 1:].strip()
            text = strip_markdown(text)
            # Remove surrounding brackets if present
            if text.startswith('[') and text.endswith(']'):
                text = text[1:-1].strip()
            if text:
                return text
    return None


def parse_v4(raw: str) -> Dict[str, Optional[str]]:
    """Parse V4 output: A1, A2, B, K lines."""
    return {
        'A1': parse_labeled_line(raw, 'A1'),
        'A2': parse_labeled_line(raw, 'A2'),
        'B': parse_labeled_line(raw, 'B'),
        'K': parse_labeled_line(raw, 'K'),
    }


def parse_v5(raw: str) -> Dict[str, Optional[str]]:
    """Parse V5 output: Q, A, B, K lines."""
    return {
        'Q': parse_labeled_line(raw, 'Q'),
        'A': parse_labeled_line(raw, 'A'),
        'B': parse_labeled_line(raw, 'B'),
        'K': parse_labeled_line(raw, 'K'),
    }


def keywords_to_phrase(k_text: str) -> str:
    """Convert comma-separated keywords to a single embeddable phrase."""
    parts = [p.strip() for p in k_text.split(',') if p.strip()]
    return ' '.join(parts)


# ═══════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════

def prompt_v2(title: str, content: str, related: List[Dict]) -> str:
    """V2 (structured): Related nodes + structured fields."""
    if related:
        related_lines = []
        for r in related:
            related_lines.append(
                f"- {r['title']} ({r['type']}, confidence {r.get('confidence', 'N/A')})"
            )
        related_text = "\n".join(related_lines)
    else:
        related_text = "(none found)"

    return f"""The brain found these related memories:
{related_text}

New node: "{title}"
Content: "{content[:200]}"

Answer these (one per line, no numbering):
\u25a1 3 questions a user would ask that this node answers
\u25a1 Which related memories does this validate, contradict, or extend?
\u25a1 Key vocabulary terms that should link to this node
"""


def prompt_v4(title: str, content: str, related: List[Dict]) -> str:
    """V4 (anchor-and-bridge): Anchor phrases from neighbors."""
    if related:
        neighbor_lines = []
        for r in related:
            kw = r.get('keywords', '') or ''
            neighbor_lines.append(f"- {r['title']} (keywords: {kw})")
        neighbor_text = "\n".join(neighbor_lines)
    else:
        neighbor_text = "(none)"

    return f"""New node: "{title}"
Content: "{content[:200]}"

Neighbors:
{neighbor_text}

Generate exactly these lines, no explanations:
A1: [3-5 word phrase using words from the neighbors above]
A2: [3-5 word phrase using words from the neighbors above]
B: [one sentence connecting this node to its most important neighbor]
K: [5 comma-separated keywords borrowed from neighbors that also describe this node]
"""


def prompt_v5(title: str, content: str, related: List[Dict]) -> str:
    """V5 (hybrid): Question + anchor + bridge + keywords."""
    if related:
        neighbor_lines = []
        for r in related:
            kw = r.get('keywords', '') or ''
            neighbor_lines.append(f"- {r['title']} (keywords: {kw})")
        neighbor_text = "\n".join(neighbor_lines)
    else:
        neighbor_text = "(none)"

    return f"""New node: "{title}"
Content: "{content[:200]}"

Neighbors:
{neighbor_text}

Generate exactly these lines, no explanations:
Q: [one question a user would naturally ask that leads to this node]
A: [3-5 word phrase using words from the neighbors above]
B: [one sentence connecting this node to its most important neighbor]
K: [5 comma-separated keywords borrowed from neighbors that also describe this node]
"""


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_golden_dataset() -> List[Dict]:
    path = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
    with open(path) as f:
        return json.load(f)


def get_target_node_ids(golden: List[Dict]) -> Set[str]:
    ids = set()
    for tc in golden:
        for nid in tc.get('expected_relevant', {}):
            ids.add(nid)
    return ids


def load_node_data(conn: sqlite3.Connection, node_id: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT id, type, title, content, confidence, keywords FROM nodes WHERE id = ?",
        (node_id,)
    ).fetchone()
    if not row:
        return None
    return {
        'id': row[0], 'type': row[1], 'title': row[2],
        'content': row[3] or '', 'confidence': row[4],
        'keywords': row[5] or '',
    }


def load_related_nodes(conn: sqlite3.Connection, node_id: str, limit: int = 5) -> List[Dict]:
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
            "SELECT id, type, title, confidence, keywords FROM nodes WHERE id = ?",
            (other_id,)
        ).fetchone()
        if node:
            related.append({
                'id': node[0], 'type': node[1], 'title': node[2],
                'confidence': node[3], 'keywords': node[4] or '',
                'weight': weight, 'relation': relation,
            })
    return related


def load_original_embeddings(conn: sqlite3.Connection) -> Dict[str, bytes]:
    rows = conn.execute(
        "SELECT ne.node_id, ne.embedding FROM node_embeddings ne "
        "JOIN nodes n ON n.id = ne.node_id WHERE n.archived = 0"
    ).fetchall()
    return {nid: blob for nid, blob in rows if blob}


# ═══════════════════════════════════════════════════════════════
# ENRICHMENT
# ═══════════════════════════════════════════════════════════════

def embed_v2_output(raw: str, embedder_inst: Embedder) -> Tuple[List[bytes], List[str]]:
    """Parse V2 output, embed questions. Returns (embeddings, descriptions)."""
    questions = parse_v2_questions(raw)
    if not questions:
        return [], []
    embeddings = embedder_inst.embed_batch(questions)
    return embeddings, questions


def embed_v4_output(raw: str, embedder_inst: Embedder) -> Tuple[List[bytes], Dict[str, str]]:
    """Parse V4 output, embed each piece. Returns (embeddings, parsed_fields)."""
    parsed = parse_v4(raw)
    texts = []
    labels = []

    for key in ['A1', 'A2', 'B']:
        if parsed[key]:
            texts.append(parsed[key])
            labels.append(key)

    if parsed['K']:
        phrase = keywords_to_phrase(parsed['K'])
        if phrase:
            texts.append(phrase)
            labels.append('K')

    if not texts:
        return [], parsed

    embeddings = embedder_inst.embed_batch(texts)
    return embeddings, parsed


def embed_v5_output(raw: str, embedder_inst: Embedder) -> Tuple[List[bytes], Dict[str, str]]:
    """Parse V5 output, embed each piece. Returns (embeddings, parsed_fields)."""
    parsed = parse_v5(raw)
    texts = []
    labels = []

    for key in ['Q', 'A', 'B']:
        if parsed[key]:
            texts.append(parsed[key])
            labels.append(key)

    if parsed['K']:
        phrase = keywords_to_phrase(parsed['K'])
        if phrase:
            texts.append(phrase)
            labels.append('K')

    if not texts:
        return [], parsed

    embeddings = embedder_inst.embed_batch(texts)
    return embeddings, parsed


VARIANTS = {
    'v2_structured': {
        'prompt_fn': prompt_v2,
        'embed_fn': embed_v2_output,
    },
    'v4_anchor_bridge': {
        'prompt_fn': prompt_v4,
        'embed_fn': embed_v4_output,
    },
    'v5_hybrid': {
        'prompt_fn': prompt_v5,
        'embed_fn': embed_v5_output,
    },
}


def enrich_all_variants(
    conn: sqlite3.Connection,
    embedder_inst: Embedder,
    target_ids: Set[str],
) -> Tuple[Dict[str, Dict[str, List[bytes]]], List[Dict]]:
    """
    For each target node, generate enrichments for all 3 variants.

    Returns:
        (enrichments_by_variant, example_enrichments)
        enrichments_by_variant: {variant_name: {node_id: [embedding_bytes, ...]}}
    """
    enrichments = {v: {} for v in VARIANTS}
    example_enrichments = []

    total = len(target_ids)
    sorted_ids = sorted(target_ids)

    print(f"\n{'='*70}")
    print(f"  ENRICHING {total} TARGET NODES  (3 variants x {total} = {total*3} LLM calls)")
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
                'outputs': {},
            }

        for variant_name, variant_cfg in VARIANTS.items():
            prompt = variant_cfg['prompt_fn'](title, content, related)
            raw = ollama_generate(prompt, timeout=45)

            if not raw:
                enrichments[variant_name][node_id] = []
                if example_entry:
                    example_entry['outputs'][variant_name] = "(no LLM output)"
                continue

            embs, parsed = variant_cfg['embed_fn'](raw, embedder_inst)
            enrichments[variant_name][node_id] = embs

            vec_count = len(embs)
            if variant_name == 'v2_structured':
                print(f"           {variant_name}: {vec_count} question vectors")
            else:
                print(f"           {variant_name}: {vec_count} vectors")

            if example_entry:
                example_entry['outputs'][variant_name] = parsed

        if example_entry:
            example_enrichments.append(example_entry)

    elapsed = time.time() - t0
    print(f"\n  Enrichment done in {elapsed:.1f}s\n")
    return enrichments, example_enrichments


# ═══════════════════════════════════════════════════════════════
# RECALL
# ═══════════════════════════════════════════════════════════════

def multivec_recall(
    query_embedding: bytes,
    original_embeddings: Dict[str, bytes],
    extra_embeddings: Dict[str, List[bytes]],
    embedder_inst: Embedder,
    limit: int = 20,
) -> List[str]:
    """Max-similarity across original + extra vectors per node."""
    scores = {}

    for node_id, orig_blob in original_embeddings.items():
        sim = embedder_inst.cosine_similarity(query_embedding, orig_blob)
        scores[node_id] = sim

        if node_id in extra_embeddings:
            for q_blob in extra_embeddings[node_id]:
                q_sim = embedder_inst.cosine_similarity(query_embedding, q_blob)
                if q_sim > scores[node_id]:
                    scores[node_id] = q_sim

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked[:limit]]


def baseline_recall(
    query_embedding: bytes,
    original_embeddings: Dict[str, bytes],
    embedder_inst: Embedder,
    limit: int = 20,
) -> List[str]:
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
    extra_embeddings: Optional[Dict[str, List[bytes]]],
    embedder_inst: Embedder,
    variant_name: str,
) -> Dict[str, Any]:
    """Run golden eval for a variant. extra_embeddings=None means baseline."""
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

def print_comparison_table(results: Dict[str, Dict], baseline_result: Dict):
    """Print side-by-side comparison of all variants + baseline."""
    b_agg = baseline_result['aggregate']
    b_ndcg = b_agg.get('ndcg@10', {}).get('mean', 0)
    b_mrr = b_agg.get('mrr', {}).get('mean', 0)
    b_hit = b_agg.get('hit_rate@10', {}).get('mean', 0)
    b_passed = baseline_result['passed']
    b_total = baseline_result['total']

    print(f"\n{'='*78}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*78}")
    print(f"  {'Variant':<22s} {'NDCG@10':>8s} {'MRR':>8s} {'Hit@10':>8s} {'Passed':>10s} {'dNDCG':>8s}")
    print(f"  {'-'*68}")

    print(f"  {'baseline':<22s} {b_ndcg:>8.3f} {b_mrr:>8.3f} {b_hit:>8.3f} "
          f"{b_passed:>4d}/{b_total:<4d}  {'---':>8s}")

    for vname in ['v2_structured', 'v4_anchor_bridge', 'v5_hybrid']:
        if vname not in results:
            continue
        result = results[vname]
        agg = result['aggregate']
        ndcg = agg.get('ndcg@10', {}).get('mean', 0)
        mrr_val = agg.get('mrr', {}).get('mean', 0)
        hit_rate = agg.get('hit_rate@10', {}).get('mean', 0)
        passed = result['passed']
        total = result['total']
        d_ndcg = ndcg - b_ndcg
        print(f"  {vname:<22s} {ndcg:>8.3f} {mrr_val:>8.3f} {hit_rate:>8.3f} "
              f"{passed:>4d}/{total:<4d}  {d_ndcg:>+8.3f}")

    print(f"  {'-'*68}")
    print(f"  Control numbers (prior runs):")
    print(f"    Baseline:        NDCG ~{CONTROL['baseline']['ndcg@10']}, ~{CONTROL['baseline']['passed']}/104")
    print(f"    V2 (previous):   NDCG  {CONTROL['v2']['ndcg@10']},  {CONTROL['v2']['passed']}/104")
    print(f"    Cross-encoder:   NDCG  {CONTROL['cross_encoder_best']['ndcg@10']},  {CONTROL['cross_encoder_best']['passed']}/104")
    print()


def print_category_breakdown(results: Dict[str, Dict]):
    """Print per-category NDCG for each variant."""
    # Collect all categories
    all_cats = set()
    for result in results.values():
        all_cats.update(result.get('by_category', {}).keys())
    all_cats = sorted(all_cats)

    if not all_cats:
        return

    print(f"\n{'='*78}")
    print(f"  PER-CATEGORY NDCG@10")
    print(f"{'='*78}")

    header = f"  {'Category':<22s}"
    for vname in ['v2_structured', 'v4_anchor_bridge', 'v5_hybrid']:
        header += f" {vname:>18s}"
    print(header)
    print(f"  {'-'*78}")

    for cat in all_cats:
        line = f"  {cat:<22s}"
        for vname in ['v2_structured', 'v4_anchor_bridge', 'v5_hybrid']:
            if vname in results:
                cat_metrics = results[vname].get('by_category', {}).get(cat, {})
                cat_ndcg = cat_metrics.get('ndcg@10', {}).get('mean', 0)
                cat_n = cat_metrics.get('ndcg@10', {}).get('count', 0)
                line += f"  {cat_ndcg:>7.3f} (n={cat_n:<2d})"
            else:
                line += f"  {'---':>14s}"
        print(line)
    print()


def print_example_enrichments(examples: List[Dict]):
    """Print sample enrichments showing all 3 variants."""
    if not examples:
        return
    print(f"\n{'='*78}")
    print(f"  EXAMPLE ENRICHMENTS ({len(examples)} nodes)")
    print(f"{'='*78}")

    for ex in examples:
        print(f"\n  Node: {ex['title']}")
        if ex['related']:
            print(f"  Related: {', '.join(ex['related'][:3])}")
        else:
            print(f"  Related: (none)")

        for variant in ['v2_structured', 'v4_anchor_bridge', 'v5_hybrid']:
            output = ex.get('outputs', {}).get(variant, '(not run)')
            print(f"    --- {variant} ---")
            if isinstance(output, str):
                print(f"      {output}")
            elif isinstance(output, list):
                # V2: list of questions
                for q in output[:3]:
                    print(f"      Q: {q}")
            elif isinstance(output, dict):
                # V4/V5: parsed fields
                for key, val in output.items():
                    if val:
                        print(f"      {key}: {val}")
                    else:
                        print(f"      {key}: (missing)")
            else:
                print(f"      {output}")
    print()


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
    print(f"[benchmark] LLM: {OLLAMA_MODEL}")
    print(f"[benchmark] Variants: V2 (control), V4 (anchor-bridge), V5 (hybrid)")

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

    # ─── Phase 1: Baseline ───
    print(f"\n{'='*70}")
    print(f"  PHASE 1: BASELINE (pure cosine, no enrichment)")
    print(f"{'='*70}")
    baseline_result = evaluate_variant(
        golden, original_embeddings, None, embedder_inst, "baseline"
    )
    b_agg = baseline_result['aggregate']
    b_ndcg = b_agg.get('ndcg@10', {}).get('mean', 0)
    b_mrr = b_agg.get('mrr', {}).get('mean', 0)
    print(f"  NDCG@10: {b_ndcg:.3f}")
    print(f"  MRR:     {b_mrr:.3f}")
    print(f"  Passed:  {baseline_result['passed']}/{baseline_result['total']}")

    # ─── Phase 2: Enrich target nodes ───
    enrichments, examples = enrich_all_variants(conn, embedder_inst, target_ids)

    # Count vectors per variant
    for vname, venrich in enrichments.items():
        total_vecs = sum(len(v) for v in venrich.values())
        nodes_with = sum(1 for v in venrich.values() if v)
        print(f"  {vname}: {total_vecs} vectors across {nodes_with}/{len(target_ids)} nodes")

    # ─── Phase 3: Evaluate each variant ───
    print(f"\n{'='*70}")
    print(f"  PHASE 3: EVALUATING VARIANTS")
    print(f"{'='*70}")

    variant_results = {}
    for variant_name in ['v2_structured', 'v4_anchor_bridge', 'v5_hybrid']:
        print(f"\n  Evaluating {variant_name}...")
        result = evaluate_variant(
            golden, original_embeddings,
            enrichments[variant_name],
            embedder_inst, variant_name,
        )
        variant_results[variant_name] = result
        agg = result['aggregate']
        ndcg = agg.get('ndcg@10', {}).get('mean', 0)
        mrr_val = agg.get('mrr', {}).get('mean', 0)
        print(f"    NDCG@10: {ndcg:.3f}  MRR: {mrr_val:.3f}  "
              f"Passed: {result['passed']}/{result['total']}")

    # ─── Phase 4: Report ───
    print_comparison_table(variant_results, baseline_result)
    print_category_breakdown(variant_results)
    print_example_enrichments(examples)

    # ─── Phase 5: Find cases where V4/V5 beat V2 (and vice versa) ───
    print(f"\n{'='*78}")
    print(f"  CASE-LEVEL DELTAS (V4/V5 vs V2)")
    print(f"{'='*78}")

    v2_cases = {r['id']: r for r in variant_results['v2_structured']['case_results']}
    for compare_name in ['v4_anchor_bridge', 'v5_hybrid']:
        wins = []
        losses = []
        for r in variant_results[compare_name]['case_results']:
            cid = r['id']
            v2_r = v2_cases.get(cid)
            if not v2_r or not r.get('metrics') or not v2_r.get('metrics'):
                continue
            my_ndcg = r['metrics'].get('ndcg@10', 0)
            v2_ndcg = v2_r['metrics'].get('ndcg@10', 0)
            delta = my_ndcg - v2_ndcg
            if delta > 0.05:
                wins.append((cid, delta, r['query'][:60]))
            elif delta < -0.05:
                losses.append((cid, delta, r['query'][:60]))

        print(f"\n  {compare_name} vs v2_structured:")
        print(f"    Wins (NDCG > +0.05):  {len(wins)}")
        for cid, delta, query in sorted(wins, key=lambda x: -x[1])[:5]:
            print(f"      {delta:+.3f}  {query}")
        print(f"    Losses (NDCG < -0.05): {len(losses)}")
        for cid, delta, query in sorted(losses, key=lambda x: x[1])[:5]:
            print(f"      {delta:+.3f}  {query}")

    # Save JSON results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, 'benchmark_v4_v5.json')

    json_report = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'model': 'snowflake-arctic-embed-m-v1.5',
            'llm': OLLAMA_MODEL,
            'target_nodes': len(target_ids),
            'golden_cases': len(golden),
            'variants': list(VARIANTS.keys()),
        },
        'control_numbers': CONTROL,
        'baseline': {
            'ndcg@10': b_ndcg,
            'mrr': b_mrr,
            'passed': baseline_result['passed'],
            'total': baseline_result['total'],
        },
        'variants': {},
        'examples': [
            {
                'node_id': ex['node_id'],
                'title': ex['title'],
                'related': ex['related'],
                'outputs': {
                    k: v if isinstance(v, (str, list)) else
                    {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else str(v)
                    for k, v in ex.get('outputs', {}).items()
                },
            }
            for ex in examples
        ],
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
                    'count': m.get('mrr', {}).get('count', 0),
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
