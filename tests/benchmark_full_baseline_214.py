#!/usr/bin/env python3
"""
Full Baseline Benchmark — 214 Golden Dataset v2 Cases

Methodology:
  For each case, embed query with Arctic v1.5, scan ALL node_embeddings
  (primary vectors) and ALL node_enrichments (enrichment vectors) via cosine
  similarity.  final_score = max(primary, best_enrichment).  Top 10 results.

  Positive cases: check expected nodes in top 10, compute NDCG@10, MRR, hit_rate.
  Negative cases: check top_score < max_acceptable_top_score (0.5).

Output:
  - JSON results:  tests/results/full_baseline_214.json
  - Text summary:  tests/results/full_baseline_214_summary.txt
"""

import os, sys, json, struct, time, math, sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Force CPU-only ONNX
os.environ["ORT_DISABLE_ALL_ACCELERATORS"] = "1"
os.environ.setdefault("ONNX_PROVIDERS", "CPUExecutionProvider")

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = "/tmp/brain_full_baseline.db"
GOLDEN_PATH = os.path.join(SCRIPT_DIR, "golden_dataset_v2.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model-package", "brain_embedding", "model")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "full_baseline_214.json")
RESULTS_TXT = os.path.join(RESULTS_DIR, "full_baseline_214_summary.txt")

# ─── Embedding ───
def load_embedder():
    """Load Arctic v1.5 via fastembed with local ONNX."""
    from fastembed import TextEmbedding
    from fastembed.common.model_description import PoolingType, ModelSource

    model_name = "Snowflake/snowflake-arctic-embed-m-v1.5"
    supported = [m['model'].lower() for m in TextEmbedding.list_supported_models()]
    if model_name.lower() not in supported:
        TextEmbedding.add_custom_model(
            model=model_name, pooling=PoolingType.CLS,
            normalization=True, sources=ModelSource(hf=model_name),
            dim=768, model_file="onnx/model.onnx",
        )
    model = TextEmbedding(
        model_name=model_name,
        specific_model_path=MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    return model


def embed_text(model, text: str) -> List[float]:
    """Embed a single text, return list of floats."""
    result = list(model.embed([text]))
    return result[0].tolist()


def blob_to_vec(blob: bytes) -> List[float]:
    """Convert BLOB (768 * 4 bytes) to float list."""
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ─── DB Loading ───
def load_primary_embeddings(db: sqlite3.Connection) -> Dict[str, List[float]]:
    """Load all primary embeddings: node_id -> vector."""
    rows = db.execute("SELECT node_id, embedding FROM node_embeddings").fetchall()
    result = {}
    for node_id, blob in rows:
        if blob and len(blob) >= 4:
            result[node_id] = blob_to_vec(blob)
    print(f"  Loaded {len(result)} primary embeddings")
    return result


def load_enrichment_embeddings(db: sqlite3.Connection) -> Dict[str, List[Tuple[str, str, List[float]]]]:
    """Load all enrichment embeddings: node_id -> [(enrichment_id, vector_type, vector), ...]."""
    rows = db.execute(
        "SELECT id, node_id, vector_type, embedding FROM node_enrichments WHERE embedding IS NOT NULL"
    ).fetchall()
    result = {}
    for eid, node_id, vtype, blob in rows:
        if blob and len(blob) >= 4:
            vec = blob_to_vec(blob)
            result.setdefault(node_id, []).append((eid, vtype, vec))
    total = sum(len(v) for v in result.values())
    print(f"  Loaded {total} enrichment vectors across {len(result)} nodes")
    return result


def load_node_titles(db: sqlite3.Connection) -> Dict[str, str]:
    """Load node_id -> title mapping."""
    rows = db.execute("SELECT id, title FROM nodes").fetchall()
    return {r[0]: r[1] for r in rows}


def load_node_confidence(db: sqlite3.Connection) -> Dict[str, float]:
    """Load node_id -> confidence."""
    rows = db.execute("SELECT id, confidence FROM nodes").fetchall()
    return {r[0]: (r[1] if r[1] is not None else 1.0) for r in rows}


# ─── Recall ───
def recall_top10(
    query_vec: List[float],
    primary: Dict[str, List[float]],
    enrichments: Dict[str, List[Tuple[str, str, List[float]]]],
) -> List[Dict[str, Any]]:
    """
    For each node: final_score = max(primary_score, best_enrichment_score).
    Return top 10 sorted descending.
    """
    scores = {}  # node_id -> {score, source, enrichment_type}

    # Primary scan
    for node_id, vec in primary.items():
        sim = cosine_sim(query_vec, vec)
        scores[node_id] = {
            'score': sim,
            'primary_score': sim,
            'best_enrichment_score': 0.0,
            'best_enrichment_type': None,
            'won_by': 'primary',
        }

    # Enrichment scan
    for node_id, enrich_list in enrichments.items():
        best_escore = 0.0
        best_etype = None
        for eid, vtype, vec in enrich_list:
            sim = cosine_sim(query_vec, vec)
            if sim > best_escore:
                best_escore = sim
                best_etype = vtype

        if node_id in scores:
            scores[node_id]['best_enrichment_score'] = best_escore
            scores[node_id]['best_enrichment_type'] = best_etype
            if best_escore > scores[node_id]['primary_score']:
                scores[node_id]['score'] = best_escore
                scores[node_id]['won_by'] = f'enrichment:{best_etype}'
        else:
            # Node has enrichments but no primary embedding
            scores[node_id] = {
                'score': best_escore,
                'primary_score': 0.0,
                'best_enrichment_score': best_escore,
                'best_enrichment_type': best_etype,
                'won_by': f'enrichment:{best_etype}',
            }

    # Sort and take top 10
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)[:10]
    return [{'node_id': nid, **info} for nid, info in ranked]


# ─── Metrics ───
def ndcg_at_k(retrieved_ids: List[str], relevance: Dict[str, int], k: int) -> float:
    """NDCG@K."""
    def dcg(ids, rel, k):
        s = 0.0
        for i, rid in enumerate(ids[:k]):
            r = rel.get(rid, 0)
            s += (2 ** r - 1) / math.log2(i + 2)
        return s
    actual = dcg(retrieved_ids, relevance, k)
    ideal_order = sorted(relevance.keys(), key=lambda x: -relevance[x])
    ideal = dcg(ideal_order, relevance, k)
    return actual / ideal if ideal > 0 else 0.0


def mrr_score(retrieved_ids: List[str], relevant: set) -> float:
    """MRR: 1/rank of first relevant hit."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate(retrieved_ids: List[str], relevant: set, k: int) -> float:
    """1.0 if any relevant in top-K, else 0.0."""
    for rid in retrieved_ids[:k]:
        if rid in relevant:
            return 1.0
    return 0.0


# ─── Main Benchmark ───
def run_benchmark():
    print("=" * 70)
    print("FULL BASELINE BENCHMARK — 214 Golden Dataset v2 Cases")
    print(f"DB: {DB_PATH}")
    print(f"Golden: {GOLDEN_PATH}")
    print(f"Model: Arctic v1.5 from {MODEL_PATH}")
    print("=" * 70)

    # Load golden dataset
    with open(GOLDEN_PATH) as f:
        cases = json.load(f)
    print(f"\nLoaded {len(cases)} test cases")

    # Load embedder
    print("\nLoading embedder...")
    t0 = time.time()
    model = load_embedder()
    print(f"  Embedder loaded in {(time.time()-t0)*1000:.0f}ms")

    # Load DB data
    print("\nLoading DB data...")
    db = sqlite3.connect(DB_PATH)
    primary = load_primary_embeddings(db)
    enrichments = load_enrichment_embeddings(db)
    titles = load_node_titles(db)
    confidences = load_node_confidence(db)

    # Run all cases
    print(f"\nRunning {len(cases)} cases...")
    results = []
    t_start = time.time()

    for i, case in enumerate(cases):
        t0 = time.time()
        query = case['query']
        query_vec = embed_text(model, query)
        top10 = recall_top10(query_vec, primary, enrichments)
        elapsed_ms = (time.time() - t0) * 1000

        retrieved_ids = [r['node_id'] for r in top10]
        retrieved_scores = {r['node_id']: r['score'] for r in top10}
        top_score = top10[0]['score'] if top10 else 0.0

        is_negative = 'max_acceptable_top_score' in case
        expected = case.get('expected_relevant', {})

        result = {
            'id': case['id'],
            'query': query,
            'category': case['category'],
            'description': case.get('description', ''),
            'is_negative': is_negative,
            'elapsed_ms': round(elapsed_ms, 1),
            'top_score': round(top_score, 4),
            'top10': [
                {
                    'node_id': r['node_id'],
                    'title': titles.get(r['node_id'], '?'),
                    'score': round(r['score'], 4),
                    'primary_score': round(r['primary_score'], 4),
                    'best_enrichment_score': round(r['best_enrichment_score'], 4),
                    'best_enrichment_type': r['best_enrichment_type'],
                    'won_by': r['won_by'],
                }
                for r in top10
            ],
        }

        if is_negative:
            threshold = case['max_acceptable_top_score']
            passed = top_score < threshold
            result['threshold'] = threshold
            result['passed'] = passed
        else:
            # Positive case
            rel_set = set(expected.keys())
            result['expected_relevant'] = expected
            result['ndcg_10'] = round(ndcg_at_k(retrieved_ids, expected, 10), 4)
            result['mrr'] = round(mrr_score(retrieved_ids, rel_set), 4)
            result['hit_rate_10'] = round(hit_rate(retrieved_ids, rel_set, 10), 4)
            min_hr = case.get('min_hit_rate_at_10', 0.0)
            result['min_hit_rate'] = min_hr
            actual_hr = result['hit_rate_10']
            result['passed'] = actual_hr >= min_hr

            # Where expected nodes actually ranked
            for nid in expected:
                if nid in retrieved_scores:
                    rank = retrieved_ids.index(nid) + 1
                    result.setdefault('expected_ranks', {})[nid] = {
                        'rank': rank,
                        'score': round(retrieved_scores[nid], 4),
                        'title': titles.get(nid, '?'),
                    }
                else:
                    result.setdefault('expected_ranks', {})[nid] = {
                        'rank': None,
                        'score': None,
                        'title': titles.get(nid, '?'),
                    }

        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cases)} done...")

    total_time = time.time() - t_start
    print(f"\nAll {len(cases)} cases completed in {total_time:.1f}s "
          f"({total_time/len(cases)*1000:.0f}ms avg)")

    db.close()

    # ─── Compute Aggregates ───
    positive_results = [r for r in results if not r['is_negative']]
    negative_results = [r for r in results if r['is_negative']]

    pos_passed = sum(1 for r in positive_results if r['passed'])
    neg_passed = sum(1 for r in negative_results if r['passed'])

    pos_ndcg = [r['ndcg_10'] for r in positive_results]
    pos_mrr = [r['mrr'] for r in positive_results]
    pos_hr = [r['hit_rate_10'] for r in positive_results]
    neg_top_scores = [r['top_score'] for r in negative_results]

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def stdev(xs):
        if len(xs) < 2: return 0.0
        m = mean(xs)
        return math.sqrt(sum((x-m)**2 for x in xs) / len(xs))

    aggregate = {
        'positive': {
            'count': len(positive_results),
            'passed': pos_passed,
            'pass_rate': round(pos_passed / len(positive_results), 4) if positive_results else 0,
            'ndcg_10_mean': round(mean(pos_ndcg), 4),
            'ndcg_10_std': round(stdev(pos_ndcg), 4),
            'mrr_mean': round(mean(pos_mrr), 4),
            'mrr_std': round(stdev(pos_mrr), 4),
            'hit_rate_10_mean': round(mean(pos_hr), 4),
            'hit_rate_10_std': round(stdev(pos_hr), 4),
        },
        'negative': {
            'count': len(negative_results),
            'passed': neg_passed,
            'pass_rate': round(neg_passed / len(negative_results), 4) if negative_results else 0,
            'avg_top_score': round(mean(neg_top_scores), 4),
            'max_top_score': round(max(neg_top_scores), 4) if neg_top_scores else 0,
            'min_top_score': round(min(neg_top_scores), 4) if neg_top_scores else 0,
        },
        'all': {
            'count': len(results),
            'passed': pos_passed + neg_passed,
            'pass_rate': round((pos_passed + neg_passed) / len(results), 4),
        },
        'timing': {
            'total_seconds': round(total_time, 1),
            'avg_ms': round(total_time / len(cases) * 1000, 1),
        },
    }

    # ─── Per-Category Breakdown ───
    categories = {}
    for r in results:
        cat = r['category']
        categories.setdefault(cat, []).append(r)

    cat_summary = {}
    for cat, cat_results in sorted(categories.items()):
        cat_pos = [r for r in cat_results if not r['is_negative']]
        cat_neg = [r for r in cat_results if r['is_negative']]
        cat_passed = sum(1 for r in cat_results if r['passed'])
        cat_scores = [r['top_score'] for r in cat_results]

        worst = min(cat_results, key=lambda r: (1 if r['passed'] else 0, -r['top_score']))

        entry = {
            'count': len(cat_results),
            'positive': len(cat_pos),
            'negative': len(cat_neg),
            'passed': cat_passed,
            'pass_rate': round(cat_passed / len(cat_results), 4),
            'avg_top_score': round(mean(cat_scores), 4),
            'worst_case': {
                'id': worst['id'],
                'query': worst['query'][:60],
                'passed': worst['passed'],
                'top_score': worst['top_score'],
            },
        }
        if cat_pos:
            entry['ndcg_10_mean'] = round(mean([r['ndcg_10'] for r in cat_pos]), 4)
            entry['mrr_mean'] = round(mean([r['mrr'] for r in cat_pos]), 4)
            entry['hit_rate_10_mean'] = round(mean([r['hit_rate_10'] for r in cat_pos]), 4)

        cat_summary[cat] = entry

    # ─── Enrichment Analysis ───
    enrichment_stats = {
        'won_by_primary': 0,
        'won_by_enrichment': 0,
        'enrichment_type_wins': {'question': 0, 'anchor': 0, 'bridge': 0, 'keywords': 0},
        'negative_enrichment_bleed': [],  # negative cases where top result won via enrichment
    }

    for r in results:
        for hit in r.get('top10', [])[:1]:  # Just top-1 result per case
            if hit['won_by'] == 'primary':
                enrichment_stats['won_by_primary'] += 1
            else:
                enrichment_stats['won_by_enrichment'] += 1
                etype = hit['won_by'].replace('enrichment:', '')
                if etype in enrichment_stats['enrichment_type_wins']:
                    enrichment_stats['enrichment_type_wins'][etype] += 1

        if r['is_negative']:
            for hit in r.get('top10', [])[:1]:
                if hit['won_by'] != 'primary':
                    enrichment_stats['negative_enrichment_bleed'].append({
                        'case_id': r['id'],
                        'query': r['query'],
                        'top_score': r['top_score'],
                        'won_by': hit['won_by'],
                        'title': hit['title'],
                    })

    # Full enrichment analysis across ALL top-10 results
    enrich_all = {'primary': 0}
    for r in results:
        if r['is_negative']:
            continue
        for hit in r.get('top10', []):
            if hit['won_by'] == 'primary':
                enrich_all['primary'] += 1
            else:
                etype = hit['won_by'].replace('enrichment:', '')
                enrich_all[etype] = enrich_all.get(etype, 0) + 1
    enrichment_stats['all_top10_won_by'] = enrich_all

    # ─── Score Distribution & Threshold Sweep ───
    pos_top_scores = [r['top_score'] for r in positive_results]
    neg_top_scores_list = [r['top_score'] for r in negative_results]

    # Histogram buckets
    buckets = [(i/20, (i+1)/20) for i in range(20)]  # 0.00-0.05, 0.05-0.10, ...
    pos_hist = [0]*20
    neg_hist = [0]*20
    for s in pos_top_scores:
        idx = min(int(s * 20), 19)
        pos_hist[idx] += 1
    for s in neg_top_scores_list:
        idx = min(int(s * 20), 19)
        neg_hist[idx] += 1

    score_distribution = {
        'pos_top_scores': {
            'mean': round(mean(pos_top_scores), 4),
            'min': round(min(pos_top_scores), 4) if pos_top_scores else 0,
            'max': round(max(pos_top_scores), 4) if pos_top_scores else 0,
            'std': round(stdev(pos_top_scores), 4),
        },
        'neg_top_scores': {
            'mean': round(mean(neg_top_scores_list), 4),
            'min': round(min(neg_top_scores_list), 4) if neg_top_scores_list else 0,
            'max': round(max(neg_top_scores_list), 4) if neg_top_scores_list else 0,
            'std': round(stdev(neg_top_scores_list), 4),
        },
        'histogram': [
            {
                'range': f"{lo:.2f}-{hi:.2f}",
                'positive': pos_hist[i],
                'negative': neg_hist[i],
            }
            for i, (lo, hi) in enumerate(buckets)
        ],
    }

    # Threshold sweep
    threshold_sweep = []
    for threshold_int in range(50, 96, 5):  # 0.50 to 0.95
        threshold = threshold_int / 100.0
        # Positive: passes if top_score >= threshold (we want results ABOVE floor)
        # Actually: for RELEVANCE_FLOOR, positive cases pass if their expected node
        # has score >= threshold. But simpler: positive passes if hit_rate > 0 (unchanged).
        # Negative passes if top_score < threshold.
        # The question is: what floor maximizes positive passes - negative failures?
        # But the real question is: at this floor, how many positive cases would
        # still find their expected nodes?

        # For positive: check if the expected nodes' scores are >= threshold
        pos_pass = 0
        for r in positive_results:
            expected_ranks = r.get('expected_ranks', {})
            # Pass if ANY expected node scored >= threshold
            found = False
            for nid, info in expected_ranks.items():
                if info.get('score') is not None and info['score'] >= threshold:
                    found = True
                    break
            if found:
                pos_pass += 1

        # For negative: pass if top_score < threshold
        neg_pass = sum(1 for r in negative_results if r['top_score'] < threshold)
        neg_fail = len(negative_results) - neg_pass

        net = pos_pass - neg_fail
        threshold_sweep.append({
            'threshold': threshold,
            'pos_pass': pos_pass,
            'pos_total': len(positive_results),
            'neg_pass': neg_pass,
            'neg_total': len(negative_results),
            'neg_fail': neg_fail,
            'net_score': net,
        })

    best_threshold = max(threshold_sweep, key=lambda x: x['net_score'])
    score_distribution['threshold_sweep'] = threshold_sweep
    score_distribution['optimal_threshold'] = best_threshold

    # ─── Build Full Output ───
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'db_path': DB_PATH,
            'golden_path': GOLDEN_PATH,
            'model': 'Snowflake/snowflake-arctic-embed-m-v1.5',
            'model_path': MODEL_PATH,
            'total_primary_embeddings': len(primary),
            'total_enrichment_nodes': len(enrichments),
            'total_enrichment_vectors': sum(len(v) for v in enrichments.values()),
        },
        'aggregate': aggregate,
        'category_summary': cat_summary,
        'enrichment_analysis': enrichment_stats,
        'score_distribution': score_distribution,
        'case_results': results,
    }

    # ─── Save JSON ───
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved: {RESULTS_JSON}")

    # ─── Generate Text Summary ───
    lines = []
    lines.append("=" * 78)
    lines.append("FULL BASELINE BENCHMARK — 214 Golden Dataset v2")
    meta = output['metadata']
    lines.append("Timestamp: %s" % meta['timestamp'])
    lines.append("DB: %s (%d primary, %d enrichment vectors)" % (
        DB_PATH, len(primary), meta['total_enrichment_vectors']))
    lines.append("=" * 78)

    # Table 1: Aggregate
    lines.append("\n" + "─" * 78)
    lines.append("TABLE 1: AGGREGATE METRICS")
    lines.append("─" * 78)
    ap = aggregate['positive']
    an = aggregate['negative']
    aa = aggregate['all']
    lines.append("%-25s %-25s %-25s %s" % (
        "Metric", "Positive (n=%d)" % ap['count'],
        "Negative (n=%d)" % an['count'], "All (n=%d)" % aa['count']))
    lines.append("-" * 78)
    lines.append("%-25s %-25s %-25s %s" % ("Pass count", ap['passed'], an['passed'], aa['passed']))
    lines.append("%-25s %-25.4f %-25.4f %.4f" % ("Pass rate", ap['pass_rate'], an['pass_rate'], aa['pass_rate']))
    lines.append("%-25s %-25.4f %-25s" % ("NDCG@10 mean", ap['ndcg_10_mean'], "N/A"))
    lines.append("%-25s %-25.4f %-25s" % ("NDCG@10 std", ap['ndcg_10_std'], "N/A"))
    lines.append("%-25s %-25.4f %-25s" % ("MRR mean", ap['mrr_mean'], "N/A"))
    lines.append("%-25s %-25.4f %-25s" % ("MRR std", ap['mrr_std'], "N/A"))
    lines.append("%-25s %-25.4f %-25s" % ("Hit rate@10 mean", ap['hit_rate_10_mean'], "N/A"))
    lines.append("%-25s %-25s %-25.4f" % ("Avg top score", "N/A", an['avg_top_score']))
    lines.append("%-25s %-25s %-25.4f" % ("Max top score", "N/A", an['max_top_score']))
    lines.append("%-25s %-25s %-25.4f" % ("Min top score", "N/A", an['min_top_score']))
    t_total = aggregate['timing']['total_seconds']
    t_avg = aggregate['timing']['avg_ms']
    lines.append("\nTiming: %d cases in %.1fs (%.1fms avg)" % (aa['count'], t_total, t_avg))

    # Table 2: Per-Category
    lines.append("\n" + "─" * 78)
    lines.append("TABLE 2: PER-CATEGORY BREAKDOWN")
    lines.append("─" * 78)
    lines.append(f"{'Category':<28} {'Cnt':>4} {'Pass':>5} {'Rate':>6} "
                 f"{'NDCG':>6} {'MRR':>6} {'AvgScr':>7}  Worst Case")
    lines.append("-" * 78)
    for cat in sorted(cat_summary.keys()):
        c = cat_summary[cat]
        ndcg_val = c.get('ndcg_10_mean', 0)
        ndcg_str = "%.3f" % ndcg_val if 'ndcg_10_mean' in c else "  N/A"
        mrr_val = c.get('mrr_mean', 0)
        mrr_str = "%.3f" % mrr_val if 'mrr_mean' in c else "  N/A"
        w = c['worst_case']
        pf_label = "PASS" if w['passed'] else "FAIL"
        worst_str = "%s %s" % (pf_label, w['id'])
        lines.append(f"{cat:<28} {c['count']:>4} {c['passed']:>5} "
                     f"{c['pass_rate']:>6.2f} {ndcg_str:>6} {mrr_str:>6} "
                     f"{c['avg_top_score']:>7.4f}  {worst_str}")

    # Table 3: Context Bleed Analysis
    lines.append("\n" + "─" * 78)
    lines.append("TABLE 3: CONTEXT BLEED ANALYSIS (Negative Cases)")
    lines.append("─" * 78)
    lines.append(f"{'Score':>6} {'P/F':>4}  {'Query':<40} Top Result Title")
    lines.append("-" * 78)
    neg_sorted = sorted(negative_results, key=lambda r: -r['top_score'])
    for r in neg_sorted:
        pf = "PASS" if r['passed'] else "FAIL"
        top_title = r['top10'][0]['title'][:35] if r['top10'] else "?"
        query_short = r['query'][:40]
        lines.append(f"{r['top_score']:>6.4f} {pf:>4}  {query_short:<40} {top_title}")

    # Table 4: Positive Case Failures
    lines.append("\n" + "─" * 78)
    lines.append("TABLE 4: POSITIVE CASE FAILURES")
    lines.append("─" * 78)
    pos_failures = [r for r in positive_results if not r['passed']]
    if not pos_failures:
        lines.append("  (none — all positive cases passed)")
    else:
        for r in pos_failures:
            lines.append(f"\n  Case: {r['id']}")
            lines.append(f"  Query: {r['query']}")
            lines.append(f"  Category: {r['category']}")
            lines.append(f"  NDCG@10={r['ndcg_10']:.4f}  MRR={r['mrr']:.4f}  "
                         f"HitRate@10={r['hit_rate_10']:.4f}")
            lines.append(f"  Expected nodes:")
            for nid, info in r.get('expected_ranks', {}).items():
                if info['rank'] is not None:
                    lines.append(f"    [{nid[:12]}] rank={info['rank']} "
                                 f"score={info['score']:.4f} — {info['title']}")
                else:
                    lines.append(f"    [{nid[:12]}] NOT IN TOP 10 — {info['title']}")
            lines.append(f"  Actual top 3:")
            for hit in r['top10'][:3]:
                lines.append(f"    score={hit['score']:.4f} "
                             f"won_by={hit['won_by']} — {hit['title'][:60]}")

    # Table 5: Enrichment Vector Analysis
    lines.append("\n" + "─" * 78)
    lines.append("TABLE 5: ENRICHMENT VECTOR ANALYSIS")
    lines.append("─" * 78)
    es = enrichment_stats
    lines.append(f"  Top-1 results won by primary:    {es['won_by_primary']}")
    lines.append(f"  Top-1 results won by enrichment: {es['won_by_enrichment']}")
    lines.append(f"\n  Enrichment type wins (top-1):")
    for etype, count in sorted(es['enrichment_type_wins'].items(), key=lambda x: -x[1]):
        lines.append(f"    {etype:<12}: {count}")
    lines.append(f"\n  All top-10 results (positive cases) won by:")
    for source, count in sorted(es['all_top10_won_by'].items(), key=lambda x: -x[1]):
        lines.append(f"    {source:<12}: {count}")
    lines.append(f"\n  Negative cases where top result won via enrichment "
                 f"({len(es['negative_enrichment_bleed'])} cases):")
    for bleed in sorted(es['negative_enrichment_bleed'], key=lambda x: -x['top_score'])[:15]:
        lines.append(f"    score={bleed['top_score']:.4f} {bleed['won_by']:<20} "
                     f"query={bleed['query'][:30]}")

    # Table 6: Score Distribution
    lines.append("\n" + "─" * 78)
    lines.append("TABLE 6: SCORE DISTRIBUTION")
    lines.append("─" * 78)
    sd = score_distribution
    ps = sd['pos_top_scores']
    ns = sd['neg_top_scores']
    lines.append("  Positive top scores: mean=%.4f min=%.4f max=%.4f std=%.4f" % (
        ps['mean'], ps['min'], ps['max'], ps['std']))
    lines.append("  Negative top scores: mean=%.4f min=%.4f max=%.4f std=%.4f" % (
        ns['mean'], ns['min'], ns['max'], ns['std']))

    lines.append(f"\n  Histogram (top scores):")
    lines.append(f"  {'Range':<12} {'Pos':>4} {'Neg':>4}  Bar")
    for bucket in sd['histogram']:
        total = bucket['positive'] + bucket['negative']
        if total == 0:
            continue
        pos_bar = '#' * bucket['positive']
        neg_bar = '.' * bucket['negative']
        lines.append(f"  {bucket['range']:<12} {bucket['positive']:>4} "
                     f"{bucket['negative']:>4}  {pos_bar}{neg_bar}")

    lines.append(f"\n  Threshold sweep (RELEVANCE_FLOOR optimization):")
    lines.append(f"  {'Thresh':>7} {'Pos Pass':>9} {'Neg Pass':>9} "
                 f"{'Neg Fail':>9} {'Net':>5}")
    lines.append(f"  {'-'*42}")
    for ts in sd['threshold_sweep']:
        marker = "  <-- OPTIMAL" if ts['threshold'] == best_threshold['threshold'] else ""
        lines.append(f"  {ts['threshold']:>7.2f} {ts['pos_pass']:>5}/{ts['pos_total']:<3} "
                     f"{ts['neg_pass']:>5}/{ts['neg_total']:<3} "
                     f"{ts['neg_fail']:>9} {ts['net_score']:>5}{marker}")

    lines.append(f"\n  Optimal threshold: {best_threshold['threshold']:.2f} "
                 f"(pos_pass={best_threshold['pos_pass']}/{best_threshold['pos_total']}, "
                 f"neg_fail={best_threshold['neg_fail']}, "
                 f"net={best_threshold['net_score']})")

    # Write summary
    summary_text = '\n'.join(lines)
    with open(RESULTS_TXT, 'w') as f:
        f.write(summary_text)
    print(f"Summary saved: {RESULTS_TXT}")

    # Print summary to stdout
    print("\n" + summary_text)

    return output


if __name__ == '__main__':
    run_benchmark()
