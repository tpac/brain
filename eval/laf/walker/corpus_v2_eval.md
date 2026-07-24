# Corpus-v2 eval — decomposed honest baselines (cutoff 2026-05-11)

Valid golds only, turn-date ≥ 2026-05-11. n=707. Static λ=0.65 mix.

## 1. Reach — by door and stratum

| slice | n | reach@5 | reach@25 | median mix |
|---|---|---|---|---|
| cue | 306 | 51% | 77% | 5 |
| window | 295 | 57% | 81% | 4 |
| session | 106 | 35% | 71% | 12 |
| **DOOR-1** | 306 | 51% | 77% | 5 |
| **DOOR-2** | 401 | 51% | 78% | 5 |

## 2. What got in — carrier lane on the 362 HITS

Which held field/lane ranks the gold best (what recall is leaning on when it succeeds).

| carrier | door-1 | door-2 | total |
|---|---|---|---|
| maxsim | 72 | 59 | 131 |
| F0 | 43 | 71 | 114 |
| M_h | 10 | 32 | 42 |
| sit | 15 | 20 | 35 |
| idf | 11 | 10 | 21 |
| pick | 5 | 11 | 16 |
| enc | 0 | 3 | 3 |

**Hit@5 by gold type (n≥15):**

| gtype | n | hit@5 |
|---|---|---|
| decision | 113 | 68% |
| finding | 95 | 54% |
| lesson | 79 | 42% |
| architecture | 65 | 48% |
| principle | 61 | 51% |
| mechanism | 35 | 46% |
| correction | 27 | 33% |
| fact | 26 | 58% |
| community | 24 | 21% |
| bug | 23 | 61% |
| milestone | 17 | 47% |
| event | 17 | 71% |
| insight | 16 | 38% |
| rule | 15 | 20% |

**Hit@5 by gold age:**

| age | n | hit@5 |
|---|---|---|
| ≤1d | 144 | 75% |
| 1-7d | 128 | 63% |
| 7-21d | 162 | 37% |
| 21-45d | 128 | 41% |
| >45d | 145 | 42% |

## 3. What didn't — miss anatomy on the 345 MISSES

Best rank the gold achieves in ANY held field/lane → is it a RANKING problem (reachable, machinery can fix) or an ENCODING problem (buried everywhere)?

| miss class | door-1 | door-2 | total | meaning |
|---|---|---|---|---|
| REACHABLE | 44 | 45 | 89 | best≤5 — remix/compose problem |
| ALMOST | 65 | 113 | 178 | best≤25 — selection problem |
| BURIED | 35 | 30 | 65 | best≤100 — calibration/crowding |
| BARELY | 6 | 7 | 13 | best>100 — encode-side, no signal |

**Recoverable misses (best≤25): which lane holds the gold (n=267)**

| holding lane | count | share |
|---|---|---|
| maxsim | 113 | 42% |
| sit | 40 | 15% |
| F0 | 35 | 13% |
| M_h | 24 | 9% |
| idf | 22 | 8% |
| pick | 20 | 7% |
| enc | 13 | 5% |

**Encode-limited tail (best>100 = BARELY) by door & stratum:**

| stratum | BARELY n | share of stratum misses |
|---|---|---|
| cue | 6 | 4% |
| window | 4 | 3% |
| session | 3 | 4% |

## 4. Door-2 ceiling verdict

- door-2 valid golds: 401 · hit@5 51% · misses 195
- of the misses: 158 (81%) are REACHABLE/ALMOST (best≤25 — a ranking/composition fix recovers them), 7 (4%) BARELY (encode-side, machinery can't help)
- **read:** door-2 is RANKING-limited — worth building the running field

