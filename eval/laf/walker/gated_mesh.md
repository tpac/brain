# gated_mesh — field-agreement pivot detector + gated mesh

- base K8-exp0.7 · gate = spearman(F0, Fj) over the pool · hard-gate thresh 0.10

## step 1 — detector validation (Δ target-rank = moment − j0-only; negative = moment helped)
| agreement quartile | mean ρ | n | mean Δrank | %turns moment hurt (Δ>+2) | %helped (Δ<−2) |
|---|---|---|---|---|---|
| Q1 | -0.034 | 412 | -4.71 | 11% | 55% |
| Q2 | 0.114 | 411 | -2.90 | 15% | 46% |
| Q3 | 0.211 | 412 | -3.19 | 12% | 44% |
| Q4 | 0.363 | 410 | -2.03 | 11% | 36% |

- named eyeball MOMENT-HURT cases (ρ percentile in population): ad249ee4→ρ=0.13 (p41), 9ec0b4e8→ρ=0.36 (p89), 124cf35a→ρ=0.18 (p54), c2244e8e→ρ=0.23 (p66)

## step 2 — gated mesh vs linear vs fitted (val turns)
| arm | sel@1 | sel-in-5 | AUC | soft_r |
|---|---|---|---|---|
| linear | 0.614 | 0.659 | 0.8360 | 0.343 |
| fitted (reweight champion) | 0.545 | 0.596 | 0.8000 | 0.410 |
| gate-cont | 0.615 | 0.675 | 0.8522 | 0.271 |
| gate-hard | 0.606 | 0.649 | 0.8299 | 0.298 |
| gate-fitted | 0.592 | 0.655 | 0.8411 | 0.308 |

- coverage: turns with no history score identically across all arms by construction (gate acts on w[1:]).
