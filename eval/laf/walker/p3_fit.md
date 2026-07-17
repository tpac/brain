# p3_fit — the composition fit (§20.13 P3.1)

- substrate: current (P3.0 verdict); features: 30 lane-slots + M_e_f; intercept ≡ 0 (cancels in pairwise diffs)
- train April–May / validate June+; pairs: picked 82094, soft 14644 (margin 0.10)

## metrics (June+ validation, pooled sel-vs-drop AUC)
| arm | val AUC | train AUC | val normal | val flagged | soft_r |
|---|---|---|---|---|---|
| A_full_picked | 0.9458 | 0.9563 | 0.9242 | 0.9539 | 0.159 |
| B_full_soft | 0.7287 | 0.7570 | 0.7383 | 0.7265 | 0.440 |
| C_ablate_echo | 0.7040 | 0.7222 | 0.7162 | 0.6997 | 0.308 |
| D_j0_only | 0.8885 | 0.7399 | 0.8974 | 0.8851 | 0.176 |
| E_j0_ablate | 0.6916 | 0.6951 | 0.7134 | 0.6834 | 0.170 |
| F_soft_ablate | 0.6496 | 0.6744 | 0.6590 | 0.6470 | 0.441 |
| G_j0_soft | 0.7304 | 0.7089 | 0.7526 | 0.7220 | 0.189 |
| H_j0_soft_ablate | 0.6828 | 0.6921 | 0.7056 | 0.6742 | 0.176 |
| K0-static | 0.8226 |  |  |  | 0.173 |
| winner-static | 0.8676 |  |  |  | 0.273 |
| K0-static content-only (pick/enc=0) | 0.6921 |  |  |  |  |

- λ sensitivity (arm A, val AUC): {"0.1": 0.9458, "1.0": 0.9458, "10.0": 0.9459}

## coefficients — arm A (full, picked), z-space gains
| lane | j0-op | j1-op | j1-anchor | j2-op | j2-anchor | tail |
|---|---|---|---|---|---|---|
| maxsim | +0.588 | -0.077 | +0.179 | -0.074 | +0.052 | +0.094 |
| sit | +0.035 | -0.031 | +0.051 | -0.003 | +0.075 | +0.023 |
| idf | -0.070 | -0.111 | -0.360 | -0.109 | -0.153 | -0.171 |
| pick | +0.117 | +0.689 | +1.927 | +0.161 | +0.643 | +0.020 |
| enc | +0.087 | -0.060 | -0.099 | -0.051 | +0.174 | +0.116 |
- M_e_f: -1.3194 (−δ in production terms)

## P3a candidates (j0-only fitted gains)
- **D_j0_only**: maxsim +0.478 · sit +0.136 · idf +0.210 · pick +0.455 · enc +0.205 · M_e_f +4.293
- **G_j0_soft**: maxsim +0.651 · sit +0.156 · idf +0.144 · pick +0.080 · enc +0.194 · M_e_f +3.224
- **H_j0_soft_ablate**: maxsim +0.678 · sit +0.165 · idf +0.153 · M_e_f +3.328

## echo read
- full fit ΔAUC vs K0-static: +0.1232; content-only (no pick/enc): -0.1186 — echo share of the fit gain: 196%
