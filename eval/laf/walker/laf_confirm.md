# Confirmatory test — does the winning arm replicate?

The 10-arm run was EXPLORATORY (best-of-9 → ~2.6pp bar). This is a PRE-REGISTERED CONFIRMATORY run: 3 arms only, so the bar returns to ~1.9pp; 3 session→fold permutations; 2 corpora; per stratum.

**Pass criterion (pre-declared):** C vs A must exclude 0 in BOTH corpora AND across every fold seed. **Attribution:** C vs B decides whether the LANE earns new production code or whether retuning alone (a K-store value change) captures it.

## quality (≥2026-05-11) — PRIMARY · n=707

| fold seed | A shipped | B refit 5 | C +enrichment | **B−A (95% CI)** | C−A (95% CI) | C−B (95% CI) |
|---|---|---|---|---|---|---|
| 0 | 51.2% | 51.9% | 52.9% | +0.72 [-0.99, +2.40] | +1.70 [-0.28, +3.68] | +0.99 [-0.85, +2.83] |
| 1 | 51.2% | 51.3% | 51.6% | +0.15 [-1.56, +1.84] | +0.44 [-1.70, +2.55] | +0.29 [-1.27, +1.84] |
| 2 | 51.2% | 50.9% | 51.1% | -0.28 [-2.26, +1.70] | -0.15 [-1.98, +1.70] | +0.12 [-1.56, +1.84] |
| 3 | 51.2% | 51.6% | 53.0% | +0.43 [-1.41, +2.40] | +1.84 [-0.14, +3.82] | +1.41 [-0.57, +3.39] |
| 4 | 51.2% | 50.8% | 52.9% | -0.40 [-2.40, +1.41] | +1.71 [-0.28, +3.68] | +2.11 [+0.42, +3.96] |

- **B (retune) vs A excludes 0 in 0/5 fold seeds** · C vs A in 0/5 · C vs B in 1/5

### Per stratum (fold seeds pooled)

| stratum | n | A shipped | C +enrichment | Δ |
|---|---|---|---|---|
| cue | 306 | 51.0% | 55.1% | +4.1pp |
| window | 295 | 57.3% | 53.9% | -3.4pp |
| session | 106 | 34.9% | 39.8% | +4.9pp |

## wide (all valid golds) — SENSITIVITY · n=912

| fold seed | A shipped | B refit 5 | C +enrichment | **B−A (95% CI)** | C−A (95% CI) | C−B (95% CI) |
|---|---|---|---|---|---|---|
| 0 | 50.3% | 52.0% | 51.2% | +1.66 [+0.00, +3.29] | +0.88 [-0.88, +2.63] | -0.78 [-2.30, +0.77] |
| 1 | 50.3% | 52.2% | 51.4% | +1.88 [+0.33, +3.51] | +1.09 [-0.55, +2.74] | -0.78 [-2.19, +0.66] |
| 2 | 50.3% | 51.8% | 50.4% | +1.43 [-0.33, +3.18] | +0.12 [-1.54, +1.86] | -1.31 [-2.85, +0.22] |
| 3 | 50.3% | 51.5% | 51.4% | +1.22 [-0.33, +2.74] | +1.11 [-0.55, +2.85] | -0.11 [-1.54, +1.32] |
| 4 | 50.3% | 51.4% | 50.7% | +1.11 [-0.55, +2.85] | +0.34 [-1.32, +1.97] | -0.77 [-2.19, +0.66] |

- **B (retune) vs A excludes 0 in 1/5 fold seeds** · C vs A in 0/5 · C vs B in 0/5

### Era split — is the retune an OLD-gold effect?

The retune looked ~0 on the quality corpus and +1.5–1.9pp on the wide one; the wide corpus differs only by adding pre-2026-05-11 golds. If the gain sits in that slice, the retune is an old-gold effect (consistent with age being the strongest miss predictor, −0.570).

| era | n | A shipped | B refit 5 | Δ (B−A) |
|---|---|---|---|---|
| pre-cutoff | 205 | 47.3% | 49.7% | +2.3pp |
| post-cutoff | 707 | 51.2% | 52.4% | +1.2pp |

### Per stratum (fold seeds pooled)

| stratum | n | A shipped | C +enrichment | Δ |
|---|---|---|---|---|
| cue | 388 | 50.3% | 53.2% | +2.9pp |
| window | 371 | 57.4% | 54.8% | -2.6pp |
| session | 153 | 33.3% | 36.5% | +3.1pp |

