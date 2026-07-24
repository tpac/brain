# Router investigation — clean valids ≥2026-05-11, n=707

## A. Per-lane descriptive (op0) — HITS vs MISSES

support = # nodes with nonzero raw activation ("size of what came back"). peak/mean/std/gap = z-field stats. grank = gold rank in that lane alone. sole% = lane ALONE reaches gold ≤5.

| lane | grp | support | peak z | mean | std | top2-gap | gold rank≤5 |
|---|---|---|---|---|---|---|---|
| maxsim | hit | 5835 | 4.67 | -0.000 | 1.00 | 0.42 | 53% |
| maxsim | miss | 5561 | 4.41 | 0.000 | 1.00 | 0.32 | 10% |
| sit | hit | 5835 | 4.24 | -0.000 | 1.00 | 0.35 | 32% |
| sit | miss | 5561 | 4.09 | 0.000 | 1.00 | 0.29 | 6% |
| idf | hit | 518 | 5.70 | 0.000 | 0.26 | 1.09 | 23% |
| idf | miss | 522 | 5.96 | -0.000 | 0.28 | 1.28 | 2% |
| pick | hit | 162 | 3.72 | -0.000 | 0.17 | 0.00 | 46% |
| pick | miss | 160 | 3.53 | 0.000 | 0.17 | 0.00 | 3% |
| enc | hit | 59 | 2.04 | -0.000 | 0.10 | 0.02 | 10% |
| enc | miss | 58 | 2.02 | 0.000 | 0.10 | 0.04 | 1% |

## B. Lane relationships — pairwise top-25 Jaccard (agreement)

How much do lanes agree on their top-25 nodes? Low Jaccard = independent lanes (composition adds reach); high = redundant.

| pair | mean Jaccard |
|---|---|
| maxsim∩sit | 0.16 |
| maxsim∩idf | 0.06 |
| maxsim∩pick | 0.05 |
| sit∩idf | 0.04 |
| maxsim∩enc | 0.03 |
| sit∩pick | 0.03 |
| pick∩enc | 0.03 |
| idf∩pick | 0.03 |
| sit∩enc | 0.02 |
| idf∩enc | 0.02 |

## B2. Sole-reacher census — which lane ALONE reaches golds no other does

Of golds reached (≤5) by exactly ONE lane, which lane? This is the "one lane great where others fail" pattern, measured.

| lane | sole-reacher count | as %% of all golds |
|---|---|---|
| maxsim | 62 | 9% |
| pick | 46 | 7% |
| M_h | 29 | 4% |
| sit | 25 | 4% |
| idf | 21 | 3% |
| enc | 3 | 0% |

- 186/707 golds (26%) are reached by exactly one lane — the conditional-lane population. 226 reached by ≥2 lanes (redundant), 295 by none.

## C. Held-out per-message router fit (4 guards)

| config | reach@5 | vs best-fixed |
|---|---|---|
| best FIXED λ=0.7 | 52.1% | — |
| held-out router (CV) | 48.8% | -3.3pp |
| SHUFFLE control (10 perms) | 49.0%±0.4 | -3.1pp |
| oracle-λ (ceiling) | 56.7% | +4.7pp |

**Cross-population transfer (the hallucination killer):**
- fit door-1 → apply door-2: 51.9% (door-2 best-fixed 52.4%)
- fit door-2 → apply door-1: 46.4%
- learned-weight SIGN FLIPS across doors: 2 / 8 features (β stable → real signal)

## Verdict
- router beats best-fixed by -3.3pp on held-out; shuffle by -3.1pp. Router ≈ shuffle or ≤ best-fixed → NO cheap router; lever is a new reach signal + the confidence gate.
