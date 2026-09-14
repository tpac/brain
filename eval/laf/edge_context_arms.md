# edge_context producer arms

- revision `78ce34e` · model `nomic-ai/nomic-embed-text-v1.5-Q` · sqlite `3.47.1` · positive control reach@1 100.0%
- corpus corpus_v2 valid, cutoff 2026-05-11 → **792 cues**, 11645 embedded nodes
- live `edge_context` config on the copy: `{'top_k': 15}`
- noise aspect (10): co_anchored, co_member, community_member, correction_improvement, dream_observation, dreamed_from, extension_refinement, member, temporal_sequence, validation_evidence
- MaxSim base views: title, _primary, high_meta, other_meta, question (+ edge_context per arm)

| arm | top_k | excl | eligible | chars | r@1 | r@5 | r@10 | r@25 | SOLO r@5 | Δr@5 vs OLD [95% CI] | b/c | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_old  (k=5,  cm-only) | 5 | 1 | 9512 | 542 | 15.15 | 33.46 | 46.97 | 70.08 | 15.66 | +0.00 [+0.00, +0.00] | 0/0 | 1.000 |
| A_new  (k=15, noise) | 15 | 10 | 9363 | 740 | 15.03 | 33.46 | 47.10 | 69.82 | 18.56 | +0.00 [-0.76, +0.76] | 5/5 | 1.000 |
| C      (k=5,  noise) | 5 | 10 | 9363 | 557 | 15.03 | 33.46 | 46.72 | 70.45 | 16.67 | +0.00 [-0.38, +0.38] | 1/1 | 1.000 |
| D      (k=15, cm-only) | 15 | 1 | 9512 | 743 | 15.03 | 33.21 | 47.10 | 69.82 | 18.06 | -0.25 [-1.01, +0.38] | 3/5 | 0.727 |
| Z_no_lane (control) | — | — | 0 | 0 | 14.77 | 33.59 | 48.36 | 70.08 | — | +0.13 [-0.88, +1.14] | 9/8 | 1.000 |
