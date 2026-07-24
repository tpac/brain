# Graph-lane refinements — rescue vs noise (maxsim-top5 seeds, neighbor-level)

misses: 345 · neighbor rows: 12175 (rescue 71 · noise 12104) · blind fan-out 35.1 nodes/turn

## R1. Edge-why cosine (semantic edges w/ embedding)

| group | n | median cos | p25 | p75 |
|---|---|---|---|---|
| rescue | 51 | 0.661 | 0.593 | 0.692 |
| noise | 5679 | 0.605 | 0.533 | 0.656 |

| cos ≥ τ | rescues kept | noise kept |
|---|---|---|
| 0.30 | 100% | 100% |
| 0.35 | 100% | 100% |
| 0.40 | 100% | 98% |
| 0.45 | 98% | 92% |
| 0.50 | 94% | 83% |

## R2. Convergence (reached from ≥2 seeds)

| n_seeds | rescue rate | share of rows |
|---|---|---|
| =1 | 0.5% | 91% |
| ≥2 | 1.5% | 9% |
| ≥3 | 3.4% | 2% |

## R3. co_accessed channel

- rescue rows via co_accessed: 16 · noise: 1632
- co_access_count median: rescue 0 vs noise 0
- days since last_strengthened: rescue 4 vs noise 4

## R4. Target-node priors

- rescue sizes median 1072 vs noise 893
- noise-heavy target types (noise%%/rescue%%): decision 15%/7%, community 12%/8%, principle 10%/7%, finding 9%/11%, architecture 7%/14%, lesson 6%/21%

## Cumulative filter curve (neighbor kept if ANY channel passes)

| walk spec | rescues kept | noise kept | noise nodes/turn |
|---|---|---|---|
| blind +1hop (all edges) | 100% (71/71) | 100% | 35.1 |
| base union: co_acc ∪ sem(desc≥80) | 73% (52/71) | 46% | 16.1 |
| + why-cos≥0.40 on sem channel | 73% (52/71) | 46% | 16.1 |
| + co_acc needs fresh≤14d OR n≥2 | 69% (49/71) | 43% | 15.1 |
| + convergence override (any edge, ≥2 seeds) | 70% (50/71) | 47% | 16.5 |

