# Gold anatomy — what separates the golds we find from the ones we miss

n=707 clean valids ≥2026-05-11 · shipped gains · tie-fair ranks. "found@k" = gold ranked ≤k in the full field.

## The miss distribution — how badly do we miss?

| band | n | share |
|---|---|---|
| rank 1–5 | 362 | 51% |
| rank 6–10 | 76 | 11% |
| rank 11–25 | 113 | 16% |
| rank 26–100 | 105 | 15% |
| rank 101–500 | 44 | 6% |
| rank 501+ | 7 | 1% |

- reach: @5 51.2% · @10 62.0% · @25 77.9% · median miss rank 64 (of misses@25: 156)

**The @5→@25 jump is the cheapest read on how much is "nearly there" vs "structurally unreachable".**

## Found@5 vs missed@5

### By node type (@5)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| decision | 113 | 77 | 68% | +17pp |
| finding | 95 | 51 | 54% | +2pp |
| lesson | 79 | 33 | 42% | -9pp |
| architecture | 65 | 31 | 48% | -4pp |
| principle | 61 | 31 | 51% | -0pp |
| mechanism | 35 | 16 | 46% | -5pp |
| correction | 27 | 9 | 33% | -18pp |
| fact | 26 | 15 | 58% | +6pp |
| community | 24 | 5 | 21% | -30pp |
| bug | 23 | 14 | 61% | +10pp |
| milestone | 17 | 8 | 47% | -4pp |
| event | 17 | 12 | 71% | +19pp |
| insight | 16 | 6 | 38% | -14pp |
| rule | 15 | 3 | 20% | -31pp |
| open | 10 | 4 | 40% | -11pp |
| reflection | 9 | 4 | 44% | -7pp |
| design | 9 | 7 | 78% | +27pp |

### By age at recall time (@5)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| 0–7d | 289 | 193 | 67% | +16pp |
| 8–30d | 212 | 84 | 40% | -12pp |
| 31–90d | 192 | 74 | 39% | -13pp |
| 90d+ | 14 | 11 | 79% | +27pp |

### By connectivity — active edge degree (@5)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| 9–20 | 271 | 159 | 59% | +7pp |
| 3–8 | 229 | 97 | 42% | -9pp |
| 21+ | 165 | 93 | 56% | +5pp |
| 0–2 | 42 | 13 | 31% | -20pp |

### By content length (@5)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| 500–1500 | 560 | 282 | 50% | -1pp |
| 1500–3000 | 124 | 72 | 58% | +7pp |
| 3000+ | 12 | 4 | 33% | -18pp |
| <500 | 11 | 4 | 36% | -15pp |

### By encoding_source (@5)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| encoder:sonnet | 346 | 204 | 59% | +8pp |
| (none) | 199 | 89 | 45% | -6pp |
| anchor | 99 | 43 | 43% | -8pp |
| s2:community_detection | 28 | 7 | 25% | -26pp |
| recovery:trace_reconstruct | 13 | 5 | 38% | -13pp |
| manual | 11 | 7 | 64% | +12pp |
| s2:consolidation | 10 | 7 | 70% | +19pp |

### By encoding completeness (@5)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| situation+question | 707 | 362 | 51% | +0pp |

### By stratum (@5)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| cue | 306 | 156 | 51% | -0pp |
| window | 295 | 169 | 57% | +6pp |
| session | 106 | 37 | 35% | -16pp |

### By cue sharpness quartile (@5)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| Q1 | 177 | 76 | 43% | -8pp |
| Q3 | 177 | 89 | 50% | -1pp |
| Q4 | 177 | 110 | 62% | +11pp |
| Q2 | 176 | 87 | 49% | -2pp |

## Found@10 vs missed@10

### By node type (@10)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| decision | 113 | 83 | 73% | +11pp |
| finding | 95 | 60 | 63% | +1pp |
| lesson | 79 | 44 | 56% | -6pp |
| architecture | 65 | 41 | 63% | +1pp |
| principle | 61 | 36 | 59% | -3pp |
| mechanism | 35 | 22 | 63% | +1pp |
| correction | 27 | 15 | 56% | -6pp |
| fact | 26 | 17 | 65% | +3pp |
| community | 24 | 8 | 33% | -29pp |
| bug | 23 | 19 | 83% | +21pp |
| milestone | 17 | 10 | 59% | -3pp |
| event | 17 | 13 | 76% | +15pp |
| insight | 16 | 7 | 44% | -18pp |
| rule | 15 | 6 | 40% | -22pp |
| open | 10 | 4 | 40% | -22pp |
| reflection | 9 | 5 | 56% | -6pp |
| design | 9 | 7 | 78% | +16pp |

### By age at recall time (@10)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| 0–7d | 289 | 222 | 77% | +15pp |
| 8–30d | 212 | 104 | 49% | -13pp |
| 31–90d | 192 | 101 | 53% | -9pp |
| 90d+ | 14 | 11 | 79% | +17pp |

### By connectivity — active edge degree (@10)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| 9–20 | 271 | 194 | 72% | +10pp |
| 3–8 | 229 | 119 | 52% | -10pp |
| 21+ | 165 | 108 | 65% | +4pp |
| 0–2 | 42 | 17 | 40% | -21pp |

### By content length (@10)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| 500–1500 | 560 | 344 | 61% | -1pp |
| 1500–3000 | 124 | 83 | 67% | +5pp |
| 3000+ | 12 | 6 | 50% | -12pp |
| <500 | 11 | 5 | 45% | -16pp |

### By encoding_source (@10)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| encoder:sonnet | 346 | 239 | 69% | +7pp |
| (none) | 199 | 116 | 58% | -4pp |
| anchor | 99 | 52 | 53% | -9pp |
| s2:community_detection | 28 | 10 | 36% | -26pp |
| recovery:trace_reconstruct | 13 | 6 | 46% | -16pp |
| manual | 11 | 8 | 73% | +11pp |
| s2:consolidation | 10 | 7 | 70% | +8pp |

### By encoding completeness (@10)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| situation+question | 707 | 438 | 62% | +0pp |

### By stratum (@10)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| cue | 306 | 190 | 62% | +0pp |
| window | 295 | 198 | 67% | +5pp |
| session | 106 | 50 | 47% | -15pp |

### By cue sharpness quartile (@10)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| Q1 | 177 | 95 | 54% | -8pp |
| Q3 | 177 | 111 | 63% | +1pp |
| Q4 | 177 | 128 | 72% | +10pp |
| Q2 | 176 | 104 | 59% | -3pp |

## Found@25 vs missed@25

### By node type (@25)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| decision | 113 | 95 | 84% | +6pp |
| finding | 95 | 77 | 81% | +3pp |
| lesson | 79 | 62 | 78% | +1pp |
| architecture | 65 | 47 | 72% | -6pp |
| principle | 61 | 46 | 75% | -3pp |
| mechanism | 35 | 25 | 71% | -7pp |
| correction | 27 | 22 | 81% | +4pp |
| fact | 26 | 19 | 73% | -5pp |
| community | 24 | 13 | 54% | -24pp |
| bug | 23 | 22 | 96% | +18pp |
| milestone | 17 | 15 | 88% | +10pp |
| event | 17 | 13 | 76% | -1pp |
| insight | 16 | 11 | 69% | -9pp |
| rule | 15 | 9 | 60% | -18pp |
| open | 10 | 9 | 90% | +12pp |
| reflection | 9 | 6 | 67% | -11pp |
| design | 9 | 8 | 89% | +11pp |

### By age at recall time (@25)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| 0–7d | 289 | 257 | 89% | +11pp |
| 8–30d | 212 | 151 | 71% | -7pp |
| 31–90d | 192 | 131 | 68% | -10pp |
| 90d+ | 14 | 12 | 86% | +8pp |

### By connectivity — active edge degree (@25)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| 9–20 | 271 | 221 | 82% | +4pp |
| 3–8 | 229 | 171 | 75% | -3pp |
| 21+ | 165 | 132 | 80% | +2pp |
| 0–2 | 42 | 27 | 64% | -14pp |

### By content length (@25)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| 500–1500 | 560 | 430 | 77% | -1pp |
| 1500–3000 | 124 | 105 | 85% | +7pp |
| 3000+ | 12 | 9 | 75% | -3pp |
| <500 | 11 | 7 | 64% | -14pp |

### By encoding_source (@25)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| encoder:sonnet | 346 | 286 | 83% | +5pp |
| (none) | 199 | 151 | 76% | -2pp |
| anchor | 99 | 72 | 73% | -5pp |
| s2:community_detection | 28 | 16 | 57% | -21pp |
| recovery:trace_reconstruct | 13 | 10 | 77% | -1pp |
| manual | 11 | 9 | 82% | +4pp |
| s2:consolidation | 10 | 7 | 70% | -8pp |

### By encoding completeness (@25)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| situation+question | 707 | 551 | 78% | +0pp |

### By stratum (@25)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| cue | 306 | 237 | 77% | -0pp |
| window | 295 | 239 | 81% | +3pp |
| session | 106 | 75 | 71% | -7pp |

### By cue sharpness quartile (@25)

| bucket | n | found | rate | vs overall |
|---|---|---|---|---|
| Q1 | 177 | 121 | 68% | -10pp |
| Q3 | 177 | 142 | 80% | +2pp |
| Q4 | 177 | 156 | 88% | +10pp |
| Q2 | 176 | 132 | 75% | -3pp |

## Which lane holds the gold — found@10 vs missed@10

| best lane for the gold | found | missed | found rate |
|---|---|---|---|
| maxsim | 199 | 152 | 57% |
| sit | 65 | 50 | 57% |
| pick | 87 | 12 | 88% |
| idf | 51 | 29 | 64% |
| graph | 24 | 18 | 57% |
| enc | 12 | 8 | 60% |

## The deep tail (rank >100, n=51) — the structurally hard golds

| property | tail median/share | all-golds median/share |
|---|---|---|
| degree (median) | 8 | 11 |
| co_degree (median) | 0 | 3 |
| content_len (median) | 979 | 1078 |
| age_days (median) | 29 | 12 |
| access_count (median) | 239 | 136 |
| has_situation (share) | 100% | 100% |
| has_question (share) | 100% | 100% |
| has_quote (share) | 61% | 71% |
| gold_in_graph (share) | 4% | 21% |

- deep-tail types: principle 7, finding 5, decision 5, mechanism 4, lesson 4, architecture 3

- deep-tail strata: cue 22, window 17, session 12

## Multivariate — which property actually predicts found@10?

Session-grouped 5-fold logistic (standardized on train). Coefficients are log-odds per SD; AUC is held-out. This separates "correlated with" from "carries independent signal".

| feature | mean coef (log-odds/SD) | sd across folds | stable? |
|---|---|---|---|
| age_days | -0.570 | 0.042 | yes |
| degree | +0.340 | 0.072 | yes |
| cur_maxz | +0.335 | 0.032 | yes |
| access_count | +0.120 | 0.027 | yes |
| gold_in_graph | +0.108 | 0.056 | no |
| content_len | -0.052 | 0.041 | no |
| co_degree | -0.018 | 0.019 | no |
| has_situation | +0.000 | 0.000 | no |
| has_question | +0.000 | 0.000 | no |

- held-out AUC of node properties alone: **0.683** (0.5 = properties carry nothing; the lanes' own AUC is ~0.81 on the pool substrate)

