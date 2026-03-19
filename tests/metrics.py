"""
brain — Retrieval Quality Metrics

Standard information retrieval metrics for evaluating recall quality:
  - NDCG@K  (Normalized Discounted Cumulative Gain)
  - MRR     (Mean Reciprocal Rank)
  - Precision@K
  - Recall@K
  - MAP     (Mean Average Precision)
  - Hit Rate@K

All functions take:
  - retrieved: ordered list of node IDs returned by recall
  - relevant: set/dict of relevant node IDs (with optional graded relevance)
"""

import math
from typing import List, Dict, Set, Union


def precision_at_k(retrieved: List[str], relevant: Union[Set[str], Dict[str, int]], k: int) -> float:
    """
    Fraction of top-K results that are relevant.

    Args:
        retrieved: Ordered list of returned node IDs
        relevant: Set of relevant IDs, or dict of ID→relevance_grade
        k: Cutoff position

    Returns:
        Precision@K (0.0–1.0)
    """
    if k <= 0:
        return 0.0

    rel_set = set(relevant) if isinstance(relevant, dict) else relevant
    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for rid in top_k if rid in rel_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Union[Set[str], Dict[str, int]], k: int) -> float:
    """
    Fraction of all relevant items found in top-K.

    Args:
        retrieved: Ordered list of returned node IDs
        relevant: Set of relevant IDs, or dict of ID→relevance_grade
        k: Cutoff position

    Returns:
        Recall@K (0.0–1.0)
    """
    rel_set = set(relevant) if isinstance(relevant, dict) else relevant
    if not rel_set:
        return 1.0  # vacuously true

    top_k = retrieved[:k]
    hits = sum(1 for rid in top_k if rid in rel_set)
    return hits / len(rel_set)


def mrr(retrieved: List[str], relevant: Union[Set[str], Dict[str, int]]) -> float:
    """
    Reciprocal rank of the first relevant result.

    Args:
        retrieved: Ordered list of returned node IDs
        relevant: Set/dict of relevant IDs

    Returns:
        1/rank of first hit, or 0.0 if no hit
    """
    rel_set = set(relevant) if isinstance(relevant, dict) else relevant
    for i, rid in enumerate(retrieved):
        if rid in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved: List[str], relevance: Dict[str, int], k: int) -> float:
    """
    Discounted Cumulative Gain at position K.

    Uses the formula: sum( (2^rel - 1) / log2(i+2) ) for i in 0..K-1

    Args:
        retrieved: Ordered list of returned node IDs
        relevance: Dict of ID→relevance_grade (0=not relevant, 1=relevant, 2=highly relevant)
        k: Cutoff position

    Returns:
        DCG@K score
    """
    score = 0.0
    for i, rid in enumerate(retrieved[:k]):
        rel = relevance.get(rid, 0)
        score += (2 ** rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(retrieved: List[str], relevance: Dict[str, int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at position K.

    NDCG = DCG / ideal_DCG, where ideal_DCG is the best possible ranking.

    Args:
        retrieved: Ordered list of returned node IDs
        relevance: Dict of ID→relevance_grade (0=not relevant, 1=relevant, 2=highly relevant)
        k: Cutoff position

    Returns:
        NDCG@K (0.0–1.0)
    """
    actual_dcg = dcg_at_k(retrieved, relevance, k)

    # Ideal ranking: sort by relevance descending
    ideal_order = sorted(relevance.keys(), key=lambda x: -relevance[x])
    ideal_dcg = dcg_at_k(ideal_order, relevance, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def average_precision(retrieved: List[str], relevant: Union[Set[str], Dict[str, int]]) -> float:
    """
    Average Precision: mean of Precision@K at each relevant position.

    Args:
        retrieved: Ordered list of returned node IDs
        relevant: Set/dict of relevant IDs

    Returns:
        AP score (0.0–1.0)
    """
    rel_set = set(relevant) if isinstance(relevant, dict) else relevant
    if not rel_set:
        return 1.0

    hits = 0
    sum_precision = 0.0
    for i, rid in enumerate(retrieved):
        if rid in rel_set:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / len(rel_set)


def hit_rate_at_k(retrieved: List[str], relevant: Union[Set[str], Dict[str, int]], k: int) -> float:
    """
    Binary: 1.0 if any relevant item in top-K, else 0.0.
    """
    rel_set = set(relevant) if isinstance(relevant, dict) else relevant
    for rid in retrieved[:k]:
        if rid in rel_set:
            return 1.0
    return 0.0


def compute_all_metrics(retrieved: List[str], relevance: Dict[str, int],
                        k_values: List[int] = None) -> Dict[str, float]:
    """
    Compute all metrics at once for a single query.

    Args:
        retrieved: Ordered list of returned node IDs
        relevance: Dict of ID→relevance_grade
        k_values: List of K cutoffs (default: [5, 10, 20])

    Returns:
        Dict of metric_name→value
    """
    if k_values is None:
        k_values = [5, 10, 20]

    results = {
        'mrr': mrr(retrieved, relevance),
        'average_precision': average_precision(retrieved, relevance),
    }

    for k in k_values:
        results[f'precision@{k}'] = precision_at_k(retrieved, relevance, k)
        results[f'recall@{k}'] = recall_at_k(retrieved, relevance, k)
        results[f'ndcg@{k}'] = ndcg_at_k(retrieved, relevance, k)
        results[f'hit_rate@{k}'] = hit_rate_at_k(retrieved, relevance, k)

    return results


def aggregate_metrics(all_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple queries.

    Returns dict of metric_name → {mean, min, max, std}.
    """
    if not all_results:
        return {}

    keys = all_results[0].keys()
    agg = {}

    for key in keys:
        values = [r[key] for r in all_results if key in r]
        if not values:
            continue

        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0

        agg[key] = {
            'mean': mean,
            'min': min(values),
            'max': max(values),
            'std': variance ** 0.5,
            'count': n,
        }

    return agg
