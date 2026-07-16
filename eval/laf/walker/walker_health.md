# walker_health — build report
**Overall: GREEN** — 7 PASS / 0 WARN / 0 FAIL


## 0 · Build provenance (stamps) — PASS

- `extract_version` = `v6-machineturns-noTSS-manifesthash` ✓
- `embed_version` = `v2-qvec-incremental` ✓
- `scores_lanes_version` = `v3-qvec-j0|title,_primary,high_meta,other_meta,edge_context,question` ✓
- gold manifest hashes match live corpus ✓

## 1 · Fill-rate matrix (column × month) — PASS

| month | op_text | anchor_text | query_stored | op_vec | anchor_vec | op_trace_id | project | gap_seconds | (turns) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04 | 100% | 100% | 100% | 95% | 88% | 84% | 0% | 96% | 2014 |
| 2026-05 | 100% | 100% | 100% | 74% | 85% | 69% | 0% | 96% | 2183 |
| 2026-06 | 100% | 100% | 100% | 91% | 82% | 88% | 9% | 93% | 1557 |
| 2026-07 | 100% | 100% | 100% | 77% | 88% | 77% | 84% | 96% | 726 |
- op_vec fill threshold computed over non-machine turns (machine turns have no operator side by design).

| month | node_id | outcome | fetched_by | pool_score | node_created_at | used_next_1 | (cands) |
|---|---|---|---|---|---|---|---|
| 2026-04 | 100% | 100% | 0% | 100% | 100% | 100% | 11474 |
| 2026-05 | 100% | 100% | 0% | 100% | 100% | 100% | 34360 |
| 2026-06 | 100% | 100% | 0% | 100% | 100% | 100% | 33950 |
| 2026-07 | 100% | 100% | 4% | 100% | 100% | 100% | 13155 |

## 2 · Join conservation ledger — PASS

- Δ rows (independent recount): **6669**
- labeled 3736 + empty 1313 + gold/synthetic 777 + unpaired 1 + no-candidates 0 + no-O 0 + text-disagree 29 + machine 799 = **6655**
- drift: +14 rows (post-build live accretion)
- NOTE: recount runs against the LIVE logs db — rows written after the walker build appear in the recount only; a small positive drift (recount > accounted) is expected.

## 3 · Achieved-window histogram — PASS

| prior turns available | labeled turns |
|---|---|
| 0 | 204 |
| 1 | 196 |
| 2 | 172 |
| 3 | 159 |
| 4 | 154 |
| 5 | 145 |
| 6 | 135 |
| 7 | 125 |
| 8+ | 2446 |

- 2446/3736 labeled turns (65%) have the full K=8 window.

## 4 · Lane sensitivity (anti-dead-operator) — PASS

| lane | within-turn std (a) | per-node cross-turn std (b) | verdict |
|---|---|---|---|
| v_title_op | 0.1459 | 0.1728 | alive |
| v_primary_op | 0.0939 | 0.1928 | alive |
| v_high_meta_op | 0.1736 | 0.1906 | alive |
| v_other_meta_op | 0.1524 | 0.1729 | alive |
| v_edge_context_op | 0.1540 | 0.1761 | alive |
| v_question_op | 0.1962 | 0.2003 | alive |
| sit_op | 0.1592 | 0.1899 | alive |
| idf_op | 0.2084 | 0.3592 | alive |

## 5 · Replay sanity (pool separability + within-pool rank) — PASS

- Pool-vs-random separability (v_primary, q_vec, 40 turns): median AUC **0.915**, p25 0.877
- Within-pool Spearman vs live rank_in_pool (informational, 437 turns): median 0.323 — gap attributable to episodic lanes joining at sweep time (0.8/2.8 gain mass), per-lane z over full field vs pool, and fatigue/floors applied after the scorer.

## 6 · Embedding spot-audit (recipe/prefix drift) — PASS

- 20 stored vectors re-embedded from their stored render text
- cosine(stored, fresh): min **0.9634**, mean 0.9942
- Measured noise floor: the quantized ONNX embedder is batch-shape nondeterministic — the same text solo vs in a batch of 5 gives ~0.982 cosine (worker embeds in batches of 5; this audit re-embeds solo). The check therefore gates on catastrophic drift (wrong model/prefix/recipe), not bit-identity.

