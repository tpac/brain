"""Shared KPI computation for the eval platform.

Three scoring domains:
  - Encoding: coverage, precision, type accuracy, richness, dedup, volume
  - Decoding: recall@K, MRR, latency, distillation fidelity
  - E2E: round-trip accuracy, cross-topic recall, synergy score

Reuses tests/metrics.py for standard IR metrics (recall@K, MRR, NDCG).
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.metrics import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k,
    hit_rate_at_k, compute_all_metrics, aggregate_metrics,
)


# ── Encoding KPIs ──

def score_encoding_run(encoded_nodes: List[Dict], ground_truth_targets: List[Dict]) -> Dict:
    """Score an encoding run against ground truth targets.

    Args:
        encoded_nodes: List of nodes created by the encoding agent.
            Each: {"type": str, "title": str, "content": str, "keywords": str, ...}
        ground_truth_targets: Expected encoding targets from corpus.
            Each: {"type": str, "topic": str, "exchange_range": [int, int]}

    Returns:
        Dict with encoding KPIs.
    """
    if not ground_truth_targets:
        # No ground truth — can only measure volume/richness
        return {
            "coverage": None,
            "precision": None,
            "type_accuracy": None,
            "richness": _richness_score(encoded_nodes),
            "volume": len(encoded_nodes),
            "volume_ok": len(encoded_nodes) <= 5,
        }

    # Coverage: how many ground truth targets were encoded?
    matched_targets = set()
    matched_nodes = set()
    for i, target in enumerate(ground_truth_targets):
        for j, node in enumerate(encoded_nodes):
            if _topic_match(node, target):
                matched_targets.add(i)
                matched_nodes.add(j)
                break

    coverage = len(matched_targets) / len(ground_truth_targets) if ground_truth_targets else 0

    # Precision: how many encoded nodes matched a target?
    precision = len(matched_nodes) / len(encoded_nodes) if encoded_nodes else 1.0

    # Type accuracy: of matched pairs, how many have correct type?
    type_correct = 0
    type_total = 0
    for i, target in enumerate(ground_truth_targets):
        if i not in matched_targets:
            continue
        for j, node in enumerate(encoded_nodes):
            if j in matched_nodes and _topic_match(node, target):
                type_total += 1
                if _type_match(node.get('type', ''), target.get('type', '')):
                    type_correct += 1
                break

    type_accuracy = type_correct / type_total if type_total else None

    return {
        "coverage": coverage,
        "precision": precision,
        "type_accuracy": type_accuracy,
        "richness": _richness_score(encoded_nodes),
        "volume": len(encoded_nodes),
        "volume_ok": len(encoded_nodes) <= 5,
        "targets_total": len(ground_truth_targets),
        "targets_matched": len(matched_targets),
        "nodes_matched": len(matched_nodes),
    }


def _topic_match(node: Dict, target: Dict) -> bool:
    """Fuzzy check if a node matches a ground truth target by topic."""
    topic_lower = target.get('topic', '').lower()
    if not topic_lower:
        return False

    # Check title and content for topic keywords
    title = (node.get('title', '') or '').lower()
    content = (node.get('content', '') or '').lower()

    # Split topic into words, check if most appear in title or content
    words = [w for w in topic_lower.split() if len(w) > 2]
    if not words:
        return False

    text = title + ' ' + content
    matches = sum(1 for w in words if w in text)
    return matches >= len(words) * 0.5


def _type_match(encoded_type: str, expected_type: str) -> bool:
    """Check if encoded type matches expected, with common aliases."""
    aliases = {
        'correction': {'correction', 'divergence'},
        'divergence': {'correction', 'divergence'},
        'lesson': {'lesson', 'insight'},
        'insight': {'lesson', 'insight'},
        'decision': {'decision'},
        'mechanism': {'mechanism'},
        'convention': {'convention', 'pattern'},
        'pattern': {'convention', 'pattern'},
        'vocabulary': {'vocabulary'},
    }
    expected_set = aliases.get(expected_type, {expected_type})
    return encoded_type in expected_set


def _richness_score(nodes: List[Dict]) -> float:
    """Compute encoding richness (0-100) from encoded nodes."""
    if not nodes:
        return 0.0

    score = 0.0

    # Content depth (0-30): average content length
    avg_content = sum(len(n.get('content', '')) for n in nodes) / len(nodes)
    score += min(30, avg_content / 10)

    # Type variety (0-20): unique types
    types = set(n.get('type', 'unknown') for n in nodes)
    score += min(20, len(types) * 5)

    # Keywords present (0-15)
    has_keywords = sum(1 for n in nodes if n.get('keywords'))
    score += min(15, (has_keywords / len(nodes)) * 15)

    # Connections (0-20)
    connections = sum(1 for n in nodes if n.get('connections'))
    score += min(20, connections * 10)

    # Reasonable volume bonus (0-15): 1-3 nodes is ideal
    if 1 <= len(nodes) <= 3:
        score += 15
    elif len(nodes) <= 5:
        score += 10
    else:
        score += 5

    return min(100.0, score)


# ── Decoding KPIs ──

def score_decoding_query(retrieved_ids: List[str], expected_ids: Set[str],
                         latency_ms: float = 0) -> Dict:
    """Score a single decoding query.

    Args:
        retrieved_ids: Ordered node IDs from recall
        expected_ids: Set of expected node IDs
        latency_ms: Query latency in milliseconds
    """
    return {
        "recall_at_3": recall_at_k(retrieved_ids, expected_ids, 3),
        "recall_at_8": recall_at_k(retrieved_ids, expected_ids, 8),
        "mrr": mrr(retrieved_ids, expected_ids),
        "hit_at_3": hit_rate_at_k(retrieved_ids, expected_ids, 3),
        "hit_at_8": hit_rate_at_k(retrieved_ids, expected_ids, 8),
        "latency_ms": latency_ms,
    }


def score_decoding_suite(query_results: List[Dict]) -> Dict:
    """Aggregate decoding results across all queries."""
    if not query_results:
        return {}

    agg = {}
    for key in ['recall_at_3', 'recall_at_8', 'mrr', 'hit_at_3', 'hit_at_8', 'latency_ms']:
        values = [r[key] for r in query_results if key in r]
        if values:
            agg[key] = sum(values) / len(values)

    agg['queries_tested'] = len(query_results)
    return agg


# ── E2E KPIs ──

def score_e2e_run(encoding_scores: Dict, decoding_scores: Dict) -> Dict:
    """Compute end-to-end synergy score from encoding + decoding results.

    Args:
        encoding_scores: Output from score_encoding_run
        decoding_scores: Output from score_decoding_suite
    """
    enc_coverage = encoding_scores.get('coverage') or 0
    dec_recall = decoding_scores.get('recall_at_8') or 0

    return {
        "encoding": encoding_scores,
        "decoding": decoding_scores,
        "round_trip_accuracy": dec_recall,
        "synergy_score": (enc_coverage * dec_recall) ** 0.5 if enc_coverage and dec_recall else 0,
    }
