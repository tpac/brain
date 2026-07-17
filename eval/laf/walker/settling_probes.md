# settling_probes — renorm control, graph spread, MMR

## A — renorm confound control (K8-exp0.7, op-cue, val)
| arm | sel@1 | sel-in-5 | AUC | soft_r |
|---|---|---|---|---|
| linear | 0.614 | 0.659 | 0.8360 | 0.343 |
| renorm | 0.466 | 0.536 | 0.7674 | 0.397 |
| renorm_shuffled | 0.531 | 0.603 | 0.8126 | 0.317 |
| fitted_linear | 0.545 | 0.596 | 0.8000 | 0.410 |

- coverage (single-message turns, linear vs renorm order mismatches): 0
- fitted per-message weights: [0.4414, 0.0459, -0.0725, -0.0544, -0.0774, -0.0189, -0.1461, 0.0029, 0.0599, 0.527, 0.3383, 0.1666, 0.319, 0.1854, 0.1492, 0.1521, 0.0279]

## B — graph-spread settling (full field, K2-exp0.7, 120 turns)
| arm | sel@1 | sel@5 | sel@25 | median | soft_r(pool) |
|---|---|---|---|---|---|
| base | 0.110 | 0.408 | 0.843 | 8 | 0.308 |
| spread h1 b0.3 | 0.100 | 0.311 | 0.732 | 12 | 0.335 |
| spread h1 b0.5 | 0.074 | 0.241 | 0.605 | 17 | 0.338 |
| spread h2 b0.3 | 0.104 | 0.358 | 0.756 | 10 | 0.325 |
- spread h1 b0.3 @25: brought 2 / lost 35 selected nodes vs base

## C — MMR delivery (top-5 from base top-50)
| λ | sel-in-5 (share) | delivered redundancy | base redundancy |
|---|---|---|---|
| 0.3 | 0.428 | 0.773 | 0.773 |
| 0.7 | 0.428 | 0.772 | 0.773 |

