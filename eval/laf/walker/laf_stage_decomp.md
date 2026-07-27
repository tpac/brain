# Where is the loss — field ORDERING or surface SELECTION?

n=707 clean valids ≥2026-05-11 · pool median 25 candidates · Haiku picks median 2

**What this corpus can answer.** The gold is in the pool BY CONSTRUCTION (gold_i indexes cand_rows), so P(gold in pool)=1 and pool-ENTRY failure is NOT measurable here — the 78% reach@25 quoted earlier is a counterfactual about a LAF field cut at 25, not production's real pool. What IS measurable is P(gold PICKED | in pool), from the real `sel`.

**Causal caveat.** The pool and `sel` came from the CHAMPION path, so Haiku saw the champion ordering, not the LAF ordering computed here. Field-rank↔pick relations below are ASSOCIATION, not causation; the causal test is the frame_replay A/B.

## The selection loss

| quantity | value |
|---|---|
| P(gold picked \| gold in pool) | **43.6%** |
| chance rate (n_picks / pool_size) | 9.2% |
| gold in field-top-2 of pool | 48.7% |
| gold in field-top-5 of pool | 68.6% |

- Haiku selects ~2.3 of ~25 candidates, so even a perfect field ordering leaves a hard selection: the gold must land in those few picks.

## Does the field ordering align with what Haiku picks?

If P(picked) rises steeply with the field's in-pool rank, ordering is worth improving. If it is flat, the field is not what drives selection and the lever is the surface.

| gold field-rank in pool | turns | picked | P(picked) |
|---|---|---|---|
| 1 | 244 | 181 | 74% |
| 2 | 100 | 63 | 63% |
| 3 | 64 | 25 | 39% |
| 4–5 | 77 | 19 | 25% |
| 6–10 | 128 | 15 | 12% |
| 11–25 | 94 | 5 | 5% |

- **P(picked | field ranks gold #1 in pool) = 74%** vs overall 44%. If perfect field ordering were achievable, that difference (+31pp) bounds what reordering could buy — an upper bound, since Haiku saw a different ordering.

## What distinguishes a picked gold from an unpicked one

### By stratum

| bucket | n | picked | P(picked) |
|---|---|---|---|
| cue | 306 | 138 | 45% |
| window | 295 | 136 | 46% |
| session | 106 | 34 | 32% |

### By gold type

| bucket | n | picked | P(picked) |
|---|---|---|---|
| decision | 113 | 67 | 59% |
| finding | 95 | 46 | 48% |
| lesson | 79 | 22 | 28% |
| architecture | 65 | 23 | 35% |
| principle | 61 | 34 | 56% |
| mechanism | 35 | 16 | 46% |
| correction | 27 | 12 | 44% |
| fact | 26 | 14 | 54% |
| community | 24 | 2 | 8% |
| bug | 23 | 8 | 35% |
| milestone | 17 | 4 | 24% |
| event | 17 | 9 | 53% |
| insight | 16 | 6 | 38% |
| rule | 15 | 2 | 13% |
| open | 10 | 3 | 30% |

### By age at recall

| bucket | n | picked | P(picked) |
|---|---|---|---|
| 0–7d | 289 | 169 | 58% |
| 8–30d | 212 | 65 | 31% |
| 31–90d | 192 | 65 | 34% |
| 90d+ | 14 | 9 | 64% |

### By soft-usage label

| bucket | n | picked | P(picked) |
|---|---|---|---|
| soft ≥0.7 | 707 | 308 | 44% |

### By cue sharpness

| bucket | n | picked | P(picked) |
|---|---|---|---|
| Q2–Q3 | 353 | 153 | 43% |
| Q1 (flat) | 177 | 69 | 39% |
| Q4 (sharp) | 177 | 86 | 49% |

## The surface's own misses (field ranked gold ≤5 in pool, not picked)

- **197 turns (28% of all)** — the field did its job and the gold still was not selected. This population is what surface work would target; it is invisible to any reach metric.

| slice | count |
|---|---|
| type=lesson | 28 |
| type=decision | 25 |
| type=architecture | 23 |
| type=finding | 22 |
| type=bug | 13 |
| type=community | 11 |
| type=correction | 10 |
| type=mechanism | 10 |
| stratum=window | 89 |
| stratum=cue | 74 |
| stratum=session | 34 |

## Verdict

- selection loss: gold is in the pool 100% of the time (by construction) yet picked only **43.6%** of the time.
- field ordering is strongly associated with picks: P(picked) goes from 74% at in-pool rank #1 to 5% at ranks 11–25.
- **direction**: field ordering still carries real signal into selection — improving in-pool rank is worth it, AND the surface has its own independent loss (see the misses table).

