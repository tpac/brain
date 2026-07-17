# soft_usage — label build + quality audit (§20.12 A4)

version: v1-j0anchor-minseqafter|title,primary,high_meta,other_meta,edge_context,question

## 1 · picks correlation (AUC selected-vs-dropped)

| slice | n_sel | n_drop | AUC soft_max | AUC soft_mean |
|---|---|---|---|---|
| ALL | 7262 | 81209 | 0.6304 | 0.6189 |
| own-anchor | 6651 | 74663 | 0.6337 | 0.6212 |
| stop-resolved | 611 | 6546 | 0.6016 | 0.5986 |
| era pre-2026-06-08 | 3938 | 43630 | 0.6316 | 0.6175 |
| era post-2026-06-08 | 3324 | 37579 | 0.6333 | 0.6255 |

## 2 · gold agreement (median soft_max, gold vs rest)

- comparable cue-turns (gold node among labeled rows): 0
- gold-median beats rest-median: 0/0
- NOTE: 0 is BY CONSTRUCTION, not a broken join — the walker excludes all gold-cue sessions at extraction (anti-leak; ledger: extract_sessions_gold_excluded=16). "Correlate with gold where both exist" (A4) has no population: both never exist. The gold check lives in the sweep's reach-Δ leg instead.

## 3 · distribution sanity

- labeled rows: 88512; NULL rows: 6458
- soft_max percentiles p1/p25/p50/p75/p99: 0.625/0.706/0.736/0.764/0.833
- std: 0.0442

**Pre-declared bar: AUC(ALL, soft_max) > 0.55 → PASS (measured 0.6304)**
