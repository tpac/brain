# p3_fit — the composition fit (§20.13 P3.1)

- substrate: current (P3.0 verdict); features: 30 lane-slots + M_e_f; intercept ≡ 0 (cancels in pairwise diffs)
- train April–May / validate June+; pairs: picked 81545, soft 14461 (margin 0.10)

## metrics (June+ validation, pooled sel-vs-drop AUC)
| arm | val AUC | train AUC | val normal | val flagged | soft_r |
|---|---|---|---|---|---|
| A_full_picked | 0.9698 | 0.9282 | 0.9612 | 0.9732 | 0.130 |
| B_full_soft | 0.7695 | 0.7709 | 0.7835 | 0.7664 | 0.427 |
| C_ablate_echo | 0.7102 | 0.7159 | 0.7230 | 0.7056 | 0.243 |
| D_j0_only | 0.9279 | 0.7128 | 0.9335 | 0.9259 | 0.088 |
| E_j0_ablate | 0.6947 | 0.6919 | 0.7165 | 0.6864 | 0.159 |
| F_soft_ablate | 0.6448 | 0.6681 | 0.6558 | 0.6417 | 0.430 |
| G_j0_soft | 0.7887 | 0.7182 | 0.8092 | 0.7811 | 0.177 |
| H_j0_soft_ablate | 0.6845 | 0.6854 | 0.7098 | 0.6749 | 0.169 |
| K0-static | 0.8226 |  |  |  | 0.173 |
| winner-static | 0.8676 |  |  |  | 0.273 |
| K0-static content-only (pick/enc=0) | 0.6921 |  |  |  |  |

- λ sensitivity (arm A, val AUC): {"0.1": 0.9698, "1.0": 0.9698, "10.0": 0.9698}

## coefficients — arm A (full, picked), z-space gains
| lane | j0-op | j1-op | j1-anchor | j2-op | j2-anchor | tail |
|---|---|---|---|---|---|---|
| maxsim | +0.356 | -0.078 | +0.070 | -0.005 | +0.048 | -0.039 |
| sit | +0.198 | +0.061 | -0.065 | +0.052 | +0.143 | -0.012 |
| idf | +0.198 | -0.073 | -0.094 | -0.014 | -0.076 | -0.115 |
| pick | +1.371 | +1.002 | +0.863 | +0.150 | +0.347 | -0.152 |
| enc | +0.093 | -0.004 | +0.077 | +0.110 | -0.040 | +0.081 |
- M_e_f: -1.2887 (−δ in production terms)

## P3a candidates (j0-only fitted gains)
- **D_j0_only**: maxsim +0.320 · sit +0.200 · idf +0.308 · pick +2.098 · enc +0.139 · M_e_f -0.024
- **G_j0_soft**: maxsim +0.506 · sit +0.394 · idf +0.173 · pick +0.222 · enc +0.341 · M_e_f +1.280
- **H_j0_soft_ablate**: maxsim +0.558 · sit +0.428 · idf +0.165 · M_e_f +1.992

## echo read
- full fit ΔAUC vs K0-static: +0.1472; content-only (no pick/enc): -0.1124 — echo share of the fit gain: 176%
