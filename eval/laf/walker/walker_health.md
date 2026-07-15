# walker_health — build report
**Overall: GREEN** — 7 PASS / 0 WARN / 0 FAIL


## 0 · Build provenance (stamps) — PASS

- `extract_version` = `v5-microturns-noTSS-manifesthash` ✓
- `embed_version` = `v2-qvec-incremental` ✓
- `scores_lanes_version` = `v3-qvec-j0|title,_primary,high_meta,other_meta,edge_context,question` ✓
- gold manifest hashes match live corpus ✓

## 1 · Fill-rate matrix (column × month) — PASS

| month | op_text | anchor_text | query_stored | op_vec | anchor_vec | op_trace_id | project | gap_seconds | (turns) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04 | 100% | 100% | 100% | 100% | 88% | 88% | 0% | 96% | 2014 |
| 2026-05 | 100% | 100% | 100% | 100% | 85% | 85% | 0% | 96% | 2183 |
| 2026-06 | 100% | 100% | 100% | 100% | 82% | 97% | 9% | 93% | 1557 |
| 2026-07 | 100% | 100% | 100% | 100% | 88% | 100% | 84% | 96% | 712 |

| month | node_id | outcome | fetched_by | pool_score | node_created_at | used_next_1 | (cands) |
|---|---|---|---|---|---|---|---|
| 2026-04 | 100% | 100% | 0% | 100% | 100% | 100% | 12532 |
| 2026-05 | 100% | 100% | 0% | 100% | 100% | 100% | 44158 |
| 2026-06 | 100% | 100% | 0% | 100% | 100% | 100% | 37550 |
| 2026-07 | 100% | 100% | 4% | 100% | 100% | 100% | 15130 |

## 2 · Join conservation ledger — PASS

- Δ rows (independent recount): **6643**
- labeled 4514 + empty 1313 + gold/synthetic 777 + unpaired 1 + no-candidates 0 + no-O 0 + text-disagree 38 = **6643**
- drift: +0 rows (post-build live accretion)
- NOTE: recount runs against the LIVE logs db — rows written after the walker build appear in the recount only; a small positive drift (recount > accounted) is expected.

## 3 · Achieved-window histogram — PASS

| prior turns available | labeled turns |
|---|---|
| 0 | 210 |
| 1 | 206 |
| 2 | 187 |
| 3 | 177 |
| 4 | 165 |
| 5 | 162 |
| 6 | 149 |
| 7 | 137 |
| 8+ | 3121 |

- 3121/4514 labeled turns (69%) have the full K=8 window.

## 4 · Lane sensitivity (anti-dead-operator) — PASS

| lane | within-turn std (a) | per-node cross-turn std (b) | verdict |
|---|---|---|---|
| v_title_op | 0.1432 | 0.1725 | alive |
| v_primary_op | 0.0904 | 0.1920 | alive |
| v_high_meta_op | 0.1713 | 0.1882 | alive |
| v_other_meta_op | 0.1509 | 0.1717 | alive |
| v_edge_context_op | 0.1527 | 0.1749 | alive |
| v_question_op | 0.1899 | 0.1989 | alive |
| sit_op | 0.1558 | 0.1870 | alive |
| idf_op | 0.1827 | 0.3430 | alive |

## 5 · Replay sanity (pool separability + within-pool rank) — PASS

- Pool-vs-random separability (v_primary, q_vec, 40 turns): median AUC **0.920**, p25 0.855
- Within-pool Spearman vs live rank_in_pool (informational, 474 turns): median 0.319 — gap attributable to episodic lanes joining at sweep time (0.8/2.8 gain mass), per-lane z over full field vs pool, and fatigue/floors applied after the scorer.

## 6 · Embedding spot-audit (recipe/prefix drift) — PASS

- 20 stored vectors re-embedded from their stored render text
- cosine(stored, fresh): min **0.9628**, mean 0.9907
- Measured noise floor: the quantized ONNX embedder is batch-shape nondeterministic — the same text solo vs in a batch of 5 gives ~0.982 cosine (worker embeds in batches of 5; this audit re-embeds solo). The check therefore gates on catastrophic drift (wrong model/prefix/recipe), not bit-identity.

