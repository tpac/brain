# moment_influence — placement, turn arms, inhibition

## pool placement of Haiku selections (val turns; @25 ≡ pool size)
| arm | sel@1 | sel-in-3 | sel-in-5 | MRR | AUC | soft_r | soft@top1 |
|---|---|---|---|---|---|---|---|
| K0 | 0.552 | 0.471 | 0.620 | 0.694 | 0.8297 | 0.201 | 0.764 |
| winner K1 | 0.674 | 0.562 | 0.711 | 0.790 | 0.8756 | 0.290 | 0.769 |
| winner anchor-only hist | 0.623 | 0.523 | 0.677 | 0.750 | 0.8571 | 0.300 | 0.770 |
| winner op-only (no anchor) | 0.624 | 0.525 | 0.681 | 0.752 | 0.8611 | 0.210 | 0.764 |
| K0 + inhibit δ=0.1 | 0.552 | 0.471 | 0.620 | 0.694 | 0.8297 | 0.201 | 0.764 |
| winner + inhibit δ=0.1 | 0.674 | 0.562 | 0.711 | 0.790 | 0.8756 | 0.290 | 0.769 |
| K0 + inhibit δ=0.3 | 0.552 | 0.470 | 0.620 | 0.694 | 0.8297 | 0.201 | 0.764 |
| winner + inhibit δ=0.3 | 0.674 | 0.562 | 0.711 | 0.791 | 0.8756 | 0.290 | 0.769 |
| K0 + inhibit δ=0.6 | 0.551 | 0.470 | 0.619 | 0.693 | 0.8296 | 0.201 | 0.764 |
| winner + inhibit δ=0.6 | 0.674 | 0.562 | 0.711 | 0.791 | 0.8756 | 0.290 | 0.769 |
| K0 + inhibit δ=1.0 | 0.551 | 0.470 | 0.619 | 0.693 | 0.8296 | 0.201 | 0.764 |
| winner + inhibit δ=1.0 | 0.674 | 0.562 | 0.711 | 0.791 | 0.8756 | 0.290 | 0.769 |

## op_len conditioning — j1-op / anchor value per operator-message-length quartile (winner arm)
| quartile | op_len | n | Δj1-op AUC | Δj1-op soft_r | Δanchor AUC | Δanchor soft_r |
|---|---|---|---|---|---|---|
| Q1 | 0-43 | 482 | +0.0233 | -0.011 | +0.0220 | +0.108 |
| Q2 | 43-93 | 468 | +0.0206 | -0.009 | +0.0124 | +0.083 |
| Q3 | 93-196 | 467 | +0.0164 | -0.011 | +0.0163 | +0.086 |
| Q4 | 196-4000 | 466 | +0.0151 | -0.006 | +0.0091 | +0.070 |

## full-field placement of Haiku selections (120 sampled val turns, all eligible nodes)
- selected nodes unresolvable in today's brain (archived since): 2 turns skipped
| arm | sel@1 | sel@5 | sel@25 | median rank | n sel |
|---|---|---|---|---|---|
| K0 | 0.154 | 0.512 | 0.880 | 5 | 299 |
| winner K1 | 0.181 | 0.589 | 0.936 | 4 | 299 |
| winner anchor-only hist | 0.181 | 0.528 | 0.933 | 5 | 299 |
| winner + inhibit δ=0.3 | 0.181 | 0.592 | 0.936 | 4 | 299 |

