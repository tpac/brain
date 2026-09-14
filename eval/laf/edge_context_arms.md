# edge_context producer arms

- revision `6406a72` · model `nomic-ai/nomic-embed-text-v1.5-Q` · isolated copy `/var/folders/p8/v9_c_hfj0pzbyj2shlzjsj1c0000gq/T/brain_test_o2hv6xbv`
- corpus corpus_v2 valid, cutoff 2026-05-11 → **792 cues**, 11605 embedded nodes
- live `edge_context` config on the copy: `{'top_k': 15}`
- noise aspect (10): co_anchored, co_member, community_member, correction_improvement, dream_observation, dreamed_from, extension_refinement, member, temporal_sequence, validation_evidence
- MaxSim base views: title, _primary, high_meta, other_meta, question (+ edge_context per arm)

| arm | top_k | excl | eligible | vectors | chars | r@1 | r@5 | r@10 | r@25 | med rank | Δr@5 vs OLD [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B_old  (k=5,  cm-only) | 5 | 1 | 9479 | 9479 | 541 | 15.03 | 33.84 | 47.35 | 69.57 | 12 | +0.00 [+0.00, +0.00] |
| A_new  (k=15, noise) | 15 | 10 | 9331 | 9331 | 738 | 15.03 | 33.84 | 47.47 | 69.95 | 12 | +0.00 [-0.76, +0.63] |
| C      (k=5,  noise) | 5 | 10 | 9331 | 9331 | 556 | 14.77 | 34.09 | 47.22 | 69.70 | 12 | +0.25 [+0.00, +0.63] |
| D      (k=15, cm-only) | 15 | 1 | 9479 | 9479 | 742 | 15.03 | 33.84 | 47.47 | 69.82 | 12 | +0.00 [-0.76, +0.76] |
| Z_no_lane (control) | — | — | 0 | 0 | 0 | 14.77 | 33.59 | 48.36 | 70.08 | 11 | -0.25 [-1.26, +0.76] |
