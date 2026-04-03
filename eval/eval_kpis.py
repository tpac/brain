"""KPI calculation functions for the brain evaluation framework.

Each KPI function takes standardized inputs and returns a score + metadata.
All KPIs are independent — you can run any subset.

KPI Groups:
- enc_*  : Encoding quality (did we store the right things?)
- dec_*  : Decoding quality (can we find what was stored?)
- loop_* : Loop quality (does encoding+decoding work together?)
"""

import math
from typing import Dict, List, Optional, Any
from collections import Counter


# ═══════════════════════════════════════════════════════════════
# GROUP 1: ENCODING KPIS
# ═══════════════════════════════════════════════════════════════

def enc_completeness(encoded_nodes: List[Dict], key_facts: List[Dict],
                     match_fn=None) -> Dict[str, Any]:
    """What % of annotated key facts got encoded?

    Args:
        encoded_nodes: List of node dicts created during encoding
        key_facts: Annotated facts that SHOULD have been encoded
        match_fn: Optional custom matcher(node, fact) -> bool.
                  Default: title substring match on topic.

    Returns:
        score: 0.0-1.0 (fraction matched)
        matched: list of (fact, matching_node) pairs
        missed: list of facts with no matching node
    """
    if not key_facts:
        return {"score": 1.0, "matched": [], "missed": [], "total_facts": 0}

    if match_fn is None:
        def match_fn(node, fact):
            topic = fact.get("topic", "").lower()
            title = (node.get("title", "") or "").lower()
            content = (node.get("content", "") or "").lower()
            # Check if any significant word from topic appears in title or content
            topic_words = [w for w in topic.split() if len(w) > 3]
            if not topic_words:
                return topic in title or topic in content
            matches = sum(1 for w in topic_words if w in title or w in content)
            return matches >= len(topic_words) * 0.5

    matched = []
    missed = []
    for fact in key_facts:
        found = False
        for node in encoded_nodes:
            if match_fn(node, fact):
                matched.append({"fact": fact, "node_id": node.get("id"), "node_title": node.get("title")})
                found = True
                break
        if not found:
            missed.append(fact)

    score = len(matched) / len(key_facts)
    return {"score": score, "matched": matched, "missed": missed, "total_facts": len(key_facts)}


def enc_overencoding(encoded_nodes: List[Dict], key_facts: List[Dict],
                     match_fn=None) -> Dict[str, Any]:
    """What % of encoded nodes don't map to any key fact?

    Lower is better. Some overencoding is acceptable (the encoder may find
    things the annotator missed), but >50% suggests noise.
    """
    if not encoded_nodes:
        return {"score": 0.0, "unmatched_nodes": [], "total_nodes": 0}

    if match_fn is None:
        def match_fn(node, fact):
            topic = fact.get("topic", "").lower()
            title = (node.get("title", "") or "").lower()
            content = (node.get("content", "") or "").lower()
            topic_words = [w for w in topic.split() if len(w) > 3]
            if not topic_words:
                return topic in title or topic in content
            matches = sum(1 for w in topic_words if w in title or w in content)
            return matches >= len(topic_words) * 0.5

    unmatched = []
    for node in encoded_nodes:
        has_match = any(match_fn(node, fact) for fact in key_facts)
        if not has_match:
            unmatched.append({"node_id": node.get("id"), "node_title": node.get("title")})

    score = len(unmatched) / len(encoded_nodes)
    return {"score": score, "unmatched_nodes": unmatched, "total_nodes": len(encoded_nodes)}


def enc_dedup(encoded_nodes: List[Dict], similarity_fn=None,
              threshold: float = 0.85) -> Dict[str, Any]:
    """Are there semantically duplicate nodes in the same encoding run?

    Args:
        encoded_nodes: Nodes with 'embedding' field (bytes)
        similarity_fn: cosine_similarity(a, b) -> float
        threshold: Above this = duplicate
    """
    if not similarity_fn or len(encoded_nodes) < 2:
        return {"score": 0.0, "duplicates": [], "total_pairs": 0}

    duplicates = []
    n = len(encoded_nodes)
    for i in range(n):
        for j in range(i + 1, n):
            emb_a = encoded_nodes[i].get("embedding")
            emb_b = encoded_nodes[j].get("embedding")
            if emb_a and emb_b:
                sim = similarity_fn(emb_a, emb_b)
                if sim >= threshold:
                    duplicates.append({
                        "node_a": encoded_nodes[i].get("title", "")[:50],
                        "node_b": encoded_nodes[j].get("title", "")[:50],
                        "similarity": sim,
                    })

    score = len(duplicates)  # Count, not fraction — 0 is the target
    return {"score": score, "duplicates": duplicates, "total_pairs": n * (n - 1) // 2}


def enc_metadata_richness(encoded_nodes: List[Dict],
                          required_fields: List[str] = None) -> Dict[str, Any]:
    """What % of nodes have key metadata fields filled?"""
    if required_fields is None:
        required_fields = ["situation", "reasoning"]

    if not encoded_nodes:
        return {"score": 0.0, "per_field": {}, "total_nodes": 0}

    per_field = {}
    for field in required_fields:
        filled = sum(1 for n in encoded_nodes
                     if n.get(field) and str(n[field]).strip())
        per_field[field] = filled / len(encoded_nodes)

    score = sum(per_field.values()) / len(per_field) if per_field else 0.0
    return {"score": score, "per_field": per_field, "total_nodes": len(encoded_nodes)}


# ═══════════════════════════════════════════════════════════════
# GROUP 2: DECODING KPIS
# ═══════════════════════════════════════════════════════════════

def dec_recall_at_k(results: List[Dict], k: int = 8) -> Dict[str, Any]:
    """For each query, is at least one expected node in top-k?

    Args:
        results: List of {"query": str, "expected_ids": [...], "returned_ids": [...]}
    """
    if not results:
        return {"score": 0.0, "hits": 0, "total": 0, "per_query": []}

    hits = 0
    per_query = []
    for r in results:
        expected = r.get("expected_ids", [])
        returned = r.get("returned_ids", [])[:k]
        if not expected:
            continue

        found = any(
            any(ret.startswith(exp) or exp.startswith(ret) for exp in expected)
            for ret in returned
        )
        hits += int(found)
        per_query.append({"query": r.get("query", "")[:60], "found": found})

    total = len([r for r in results if r.get("expected_ids")])
    score = hits / max(total, 1)
    return {"score": score, "hits": hits, "total": total, "per_query": per_query}


def dec_mrr(results: List[Dict], max_k: int = 25) -> Dict[str, Any]:
    """Mean Reciprocal Rank — how quickly do we find the first relevant hit?"""
    if not results:
        return {"score": 0.0, "total": 0}

    rr_sum = 0.0
    count = 0
    for r in results:
        expected = r.get("expected_ids", [])
        returned = r.get("returned_ids", [])[:max_k]
        if not expected:
            continue
        count += 1
        for rank, ret_id in enumerate(returned, 1):
            if any(ret_id.startswith(exp) or exp.startswith(ret_id) for exp in expected):
                rr_sum += 1.0 / rank
                break

    score = rr_sum / max(count, 1)
    return {"score": score, "total": count}


def dec_hub_concentration(results: List[Dict], k: int = 8,
                          top_n: int = 5) -> Dict[str, Any]:
    """What % of all recall slots do the top-N most frequent nodes consume?"""
    if not results:
        return {"score": 0.0, "top_hubs": [], "total_slots": 0}

    counts = Counter()
    total_slots = 0
    for r in results:
        for nid in r.get("returned_ids", [])[:k]:
            counts[nid] += 1
            total_slots += 1

    top_hubs = counts.most_common(top_n)
    top_total = sum(c for _, c in top_hubs)
    score = top_total / max(total_slots, 1)

    return {
        "score": score,
        "top_hubs": [{"id": nid, "count": c} for nid, c in top_hubs],
        "total_slots": total_slots,
        "unique_nodes": len(counts),
    }


def dec_false_positive_rate(results: List[Dict], k: int = 8,
                            relevance_threshold: float = 0.3) -> Dict[str, Any]:
    """For queries that should return nothing, how often do we return results?

    Args:
        results: List with expected_ids=[] for false positive queries
                 and returned_scores for actual scores
    """
    fp_queries = [r for r in results if not r.get("expected_ids")]
    if not fp_queries:
        return {"score": 0.0, "false_positives": 0, "total_null_queries": 0}

    false_positives = 0
    for r in fp_queries:
        scores = r.get("returned_scores", [])[:k]
        # If any result scores above threshold, it's a false positive
        if any(s > relevance_threshold for s in scores):
            false_positives += 1

    score = false_positives / len(fp_queries)
    return {"score": score, "false_positives": false_positives,
            "total_null_queries": len(fp_queries)}


def dec_cross_topic_contamination(results: List[Dict], k: int = 8) -> Dict[str, Any]:
    """For topic-specific queries, how many results are from a different topic?

    Args:
        results: List with "query_topic" and returned nodes with "node_topic"
    """
    if not results:
        return {"score": 0.0, "contaminated": 0, "total_slots": 0}

    contaminated = 0
    total = 0
    for r in results:
        query_topic = r.get("query_topic", "")
        if not query_topic:
            continue
        for node in r.get("returned_nodes", [])[:k]:
            node_topic = node.get("project") or node.get("topic", "")
            total += 1
            if node_topic and node_topic != query_topic:
                contaminated += 1

    score = contaminated / max(total, 1)
    return {"score": score, "contaminated": contaminated, "total_slots": total}


# ═══════════════════════════════════════════════════════════════
# GROUP 3: LOOP KPIS
# ═══════════════════════════════════════════════════════════════

def loop_encode_then_recall(encode_results: List[Dict],
                            recall_results: List[Dict],
                            k: int = 8) -> Dict[str, Any]:
    """After encoding, can we recall the encoded facts?

    Uses dec_recall_at_k but specifically on queries designed to find
    the just-encoded content.
    """
    return dec_recall_at_k(recall_results, k=k)


def loop_sequential_contamination(results_after_both: List[Dict],
                                  conv_a_node_ids: set,
                                  k: int = 8) -> Dict[str, Any]:
    """After encoding A then B, do B's queries pull A's nodes?

    Args:
        results_after_both: Recall results for conv B queries after both A and B encoded
        conv_a_node_ids: Set of node IDs from conversation A
    """
    if not results_after_both:
        return {"score": 0.0, "contaminated_slots": 0, "total_slots": 0}

    contaminated = 0
    total = 0
    for r in results_after_both:
        for nid in r.get("returned_ids", [])[:k]:
            total += 1
            if nid in conv_a_node_ids:
                contaminated += 1

    score = contaminated / max(total, 1)
    return {"score": score, "contaminated_slots": contaminated, "total_slots": total}


# ═══════════════════════════════════════════════════════════════
# AGGREGATE
# ═══════════════════════════════════════════════════════════════

def compute_all_decode_kpis(results: List[Dict]) -> Dict[str, Any]:
    """Convenience: compute all decoding KPIs from a standard result set."""
    return {
        "recall@8": dec_recall_at_k(results, k=8),
        "recall@25": dec_recall_at_k(results, k=25),
        "mrr": dec_mrr(results),
        "hub_concentration": dec_hub_concentration(results),
        "false_positive_rate": dec_false_positive_rate(results),
    }


def format_kpi_summary(kpis: Dict[str, Any]) -> str:
    """One-line-per-KPI summary for console output."""
    lines = []
    for name, result in kpis.items():
        if isinstance(result, dict) and "score" in result:
            score = result["score"]
            if isinstance(score, float):
                lines.append(f"  {name:<30s} {score:.1%}")
            else:
                lines.append(f"  {name:<30s} {score}")
        else:
            lines.append(f"  {name:<30s} {result}")
    return "\n".join(lines)
