# walker_health — build report
**Overall: GREEN** — 6 PASS / 0 WARN / 0 FAIL


## 1 · Fill-rate matrix (column × month) — PASS

| month | op_text | anchor_text | query_stored | op_vec | anchor_vec | op_trace_id | project | gap_seconds | (turns) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04 | 100% | 100% | 100% | 100% | 88% | 88% | 0% | 96% | 2014 |
| 2026-05 | 100% | 100% | 100% | 100% | 85% | 85% | 0% | 96% | 2183 |
| 2026-06 | 100% | 100% | 100% | 100% | 82% | 97% | 9% | 93% | 1557 |
| 2026-07 | 100% | 100% | 100% | 100% | 88% | 100% | 82% | 96% | 650 |

| month | node_id | outcome | fetched_by | pool_score | node_created_at | used_next_1 | (cands) |
|---|---|---|---|---|---|---|---|
| 2026-04 | 100% | 100% | 0% | 100% | 100% | 100% | 12532 |
| 2026-05 | 100% | 100% | 0% | 100% | 100% | 100% | 44158 |
| 2026-06 | 100% | 100% | 0% | 100% | 100% | 100% | 37550 |
| 2026-07 | 100% | 100% | 4% | 100% | 100% | 100% | 14229 |

## 2 · Join conservation ledger — PASS

- Δ rows (independent recount): **6609**
- labeled 4478 + empty 1313 + gold/synthetic 777 + unpaired 1 + no-candidates 0 + no-O 0 + text-disagree 38 = **6607**
- drift: +2 rows (post-build live accretion)
- NOTE: recount runs against the LIVE logs db — rows written after the walker build appear in the recount only; a small positive drift (recount > accounted) is expected.

## 3 · Achieved-window histogram — PASS

| prior turns available | labeled turns |
|---|---|
| 0 | 209 |
| 1 | 205 |
| 2 | 186 |
| 3 | 176 |
| 4 | 164 |
| 5 | 160 |
| 6 | 147 |
| 7 | 136 |
| 8+ | 3095 |

- 3095/4478 labeled turns (69%) have the full K=8 window.

## 4 · Lane sensitivity (anti-dead-operator) — PASS

| lane | within-turn std (a) | per-node cross-turn std (b) | verdict |
|---|---|---|---|
| v_title_op | 0.1430 | 0.1721 | alive |
| v_primary_op | 0.0900 | 0.1918 | alive |
| v_high_meta_op | 0.1713 | 0.1878 | alive |
| v_other_meta_op | 0.1508 | 0.1711 | alive |
| v_edge_context_op | 0.1524 | 0.1744 | alive |
| v_question_op | 0.1897 | 0.1985 | alive |
| sit_op | 0.1558 | 0.1867 | alive |
| idf_op | 0.1817 | 0.3424 | alive |

## 5 · Replay sanity (pool separability + within-pool rank) — PASS

- Pool-vs-random separability (v_primary, q_vec, 40 turns): median AUC **0.908**, p25 0.848
- Within-pool Spearman vs live rank_in_pool (informational, 438 turns): median 0.313 — gap attributable to episodic lanes joining at sweep time (0.8/2.8 gain mass), per-lane z over full field vs pool, and fatigue/floors applied after the scorer.

## 6 · Embedding spot-audit (recipe/prefix drift) — PASS

- 20 stored vectors re-embedded from their stored render text
- cosine(stored, fresh): min **0.9773**, mean 0.9968
- Measured noise floor: the quantized ONNX embedder is batch-shape nondeterministic — the same text solo vs in a batch of 5 gives ~0.982 cosine (worker embeds in batches of 5; this audit re-embeds solo). The check therefore gates on catastrophic drift (wrong model/prefix/recipe), not bit-identity.

