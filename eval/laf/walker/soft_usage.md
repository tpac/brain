# soft_usage — label build + quality audit (§20.12 A4)

version: v1-j0anchor-minseqafter|title,primary,high_meta,other_meta,edge_context,question

## 1 · picks correlation (AUC selected-vs-dropped)

| slice | n_sel | n_drop | AUC soft_max | AUC soft_mean |
|---|---|---|---|---|
| ALL | 7492 | 94057 | 0.6473 | 0.6342 |
| own-anchor | 6827 | 83027 | 0.6415 | 0.6275 |
| stop-resolved | 665 | 11030 | 0.6833 | 0.6724 |
| era pre-2026-06-08 | 4184 | 54045 | 0.6600 | 0.6435 |
| era post-2026-06-08 | 3308 | 40012 | 0.6329 | 0.6243 |

## 2 · gold agreement (median soft_max, gold vs rest)

- comparable cue-turns (gold node among labeled rows): 0
- gold-median beats rest-median: 0/0
- NOTE: 0 is BY CONSTRUCTION, not a broken join — the walker excludes all gold-cue sessions at extraction (anti-leak; ledger: extract_sessions_gold_excluded=16). "Correlate with gold where both exist" (A4) has no population: both never exist. The gold check lives in the sweep's reach-Δ leg instead.

## 3 · distribution sanity

- labeled rows: 101578; NULL rows: 7758
- soft_max percentiles p1/p25/p50/p75/p99: 0.590/0.700/0.732/0.761/0.831
- std: 0.0490

**Pre-declared bar: AUC(ALL, soft_max) > 0.55 → PASS (measured 0.6473)**
