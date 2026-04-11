"""S2 Consolidation contract — config and thresholds.

Phase 1: decoder only. Finds and characterizes convergent clusters.
Phase 2 (future): encoder config (model, max_tokens, prompt).
"""

CONSOLIDATION = {
    # ── Embedding scan ──
    'similarity_threshold': 0.82,       # Minimum cosine for candidate pair
    'max_cluster_size': 5,              # Larger = topic, not convergence

    # ── Behavioral trace lookback ──
    'co_recall_lookback_hours': 168,    # 7 days of S1R traces
    'encoding_lookback_hours': 168,     # 7 days of S1E traces

    # ── Suppression ──
    # Pairs with these edge relations are skipped (already reviewed)
    'suppression_relations': {'similar_to', 'consolidated_into'},

    # ── Pre-classification thresholds ──
    'likely_consolidate_cosine': 0.90,  # Above this + structural signal = likely consolidate
}
