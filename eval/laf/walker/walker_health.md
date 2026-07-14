# walker_health — build report
**Overall: GREEN** — 6 PASS / 0 WARN / 0 FAIL


## 1 · Fill-rate matrix (column × month) — PASS

| month | op_text | anchor_text | query_stored | op_vec | anchor_vec | op_trace_id | project | gap_seconds | (turns) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04 | 100% | 100% | 100% | 31% | 17% | 86% | 0% | 96% | 2051 |
| 2026-05 | 100% | 100% | 100% | 100% | 84% | 84% | 0% | 96% | 2208 |
| 2026-06 | 100% | 100% | 100% | 100% | 81% | 97% | 9% | 93% | 1564 |
| 2026-07 | 100% | 100% | 100% | 100% | 88% | 100% | 83% | 96% | 611 |

| month | node_id | outcome | fetched_by | pool_score | node_created_at | used_next_1 | (cands) |
|---|---|---|---|---|---|---|---|
| 2026-04 | 100% | 100% | 0% | 100% | 100% | 100% | 12682 |
| 2026-05 | 100% | 100% | 0% | 100% | 100% | 100% | 44594 |
| 2026-06 | 100% | 100% | 0% | 100% | 100% | 100% | 37725 |
| 2026-07 | 100% | 100% | 3% | 100% | 100% | 100% | 13479 |

## 2 · Join conservation ledger — PASS

- Δ rows (independent recount): **6579**
- labeled 4486 + empty 1313 + gold/synthetic 777 + unpaired 1 + no-candidates 0 + no-O 0 = **6577**
- drift: +2 rows (post-build live accretion)
- NOTE: recount runs against the LIVE logs db — rows written after the walker build appear in the recount only; a small positive drift (recount > accounted) is expected.

## 3 · Achieved-window histogram — PASS

| prior turns available | labeled turns |
|---|---|
| 0 | 212 |
| 1 | 200 |
| 2 | 184 |
| 3 | 173 |
| 4 | 166 |
| 5 | 160 |
| 6 | 148 |
| 7 | 138 |
| 8+ | 3105 |

- 3105/4486 labeled turns (69%) have the full K=8 window.

## 4 · Lane sensitivity (anti-dead-operator) — PASS

| lane | within-turn std (a) | per-node cross-turn std (b) | verdict |
|---|---|---|---|
| v_title_op | 0.1429 | 0.1716 | alive |
| v_primary_op | 0.0897 | 0.1916 | alive |
| v_high_meta_op | 0.1716 | 0.1877 | alive |
| v_other_meta_op | 0.1508 | 0.1711 | alive |
| v_edge_context_op | 0.1520 | 0.1733 | alive |
| v_question_op | 0.1898 | 0.1979 | alive |
| sit_op | 0.1561 | 0.1864 | alive |
| idf_op | 0.1822 | 0.3426 | alive |

## 5 · Replay sanity (pool separability + within-pool rank) — PASS

- Pool-vs-random separability (v_primary, q_vec, 40 turns): median AUC **0.907**, p25 0.840
- Within-pool Spearman vs live rank_in_pool (informational, 408 turns): median 0.311 — gap attributable to episodic lanes joining at sweep time (0.8/2.8 gain mass), per-lane z over full field vs pool, and fatigue/floors applied after the scorer.

## 6 · Embedding spot-audit (recipe/prefix drift) — PASS

- 20 stored vectors re-embedded from their stored render text
- cosine(stored, fresh): min **0.9634**, mean 0.9965
- Measured noise floor: the quantized ONNX embedder is batch-shape nondeterministic — the same text solo vs in a batch of 5 gives ~0.982 cosine (worker embeds in batches of 5; this audit re-embeds solo). The check therefore gates on catastrophic drift (wrong model/prefix/recipe), not bit-identity.

