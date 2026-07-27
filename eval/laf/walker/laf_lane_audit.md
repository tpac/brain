# LAF composition audit — graph as a 6th lane (reach substrate)

n=707 clean valids ≥2026-05-11 · composition f0=Σgain·z(lane), mix=0.65·zn(f0)+0.35·zn(mh) · tie-fair ranks

CROSS-CHECK baseline reach@5 = 51% (committed 51%) — MATCH

## RECORD — per-lane descriptive (hits vs misses), 6 lanes

support = #nonzero z · gold≤5 = lane ALONE ranks gold ≤5 (tie-fair) · sole = lane is the only ≤5 reacher

| lane | grp | support | gold z (mean) | gold≤5 |
|---|---|---|---|---|
| maxsim | hit | 5835 | 3.89 | 53% |
| maxsim | miss | 5561 | 2.90 | 10% |
| sit | hit | 5835 | 2.93 | 32% |
| sit | miss | 5561 | 1.92 | 6% |
| idf | hit | 517 | 2.09 | 23% |
| idf | miss | 522 | 0.40 | 2% |
| pick | hit | 162 | 2.50 | 46% |
| pick | miss | 160 | 0.21 | 3% |
| enc | hit | 59 | 0.33 | 10% |
| enc | miss | 58 | 0.03 | 1% |
| graph | hit | 16 | 0.09 | 9% |
| graph | miss | 16 | 0.18 | 8% |

**Sole-reacher census** (gold reached ≤5 by exactly one lane):

| lane | sole count | %% of valids |
|---|---|---|
| pick | 69 | 10% |
| maxsim | 63 | 9% |
| sit | 29 | 4% |
| graph | 25 | 4% |
| idf | 19 | 3% |
| enc | 4 | 1% |

## GAINS? (1) graph-gain sweep — reach@5 per stratum

| gain_graph | all | cue | window | session |
|---|---|---|---|---|
| 0.00 | 51.2% | 51.0% | 57.3% | 34.9% |
| 0.10 | 51.3% | 51.0% | 57.3% | 35.8% |
| 0.20 | 51.6% | 51.3% | 57.6% | 35.8% |
| 0.30 | 51.9% | 52.3% | 56.9% | 36.8% |
| 0.40 | 51.9% | 52.6% | 56.6% | 36.8% |
| 0.50 | 52.6% | 53.6% | 56.9% | 37.7% |
| 0.60 | 52.5% | 53.3% | 56.9% | 37.7% |
| 0.80 | 52.2% | 52.9% | 56.6% | 37.7% |
| 1.00 | 52.3% | 53.3% | 56.6% | 37.7% |
| 1.25 | 52.3% | 53.3% | 56.6% | 37.7% |
| 1.50 | 52.1% | 53.3% | 55.9% | 37.7% |

- best fixed gain_graph = **0.50** → reach@5 52.6% (baseline 51.2%, +1.4pp)

- rescuable misses (gold ∈ base-union graph neighbors): **52** / 345 misses — cross-check vs committed 52: MATCH
- REACH CEILING (oracle graph, every rescuable miss converted): 58.6% (+7.4pp) — the prize; gain/scoring decides real conversion

## GAINS? (2) leave-one-out — does each lane earn its place?

reach@5 with one lane zeroed (graph at best fixed gain=0.50). Big drop = load-bearing; ~0 = dead weight; rise = harmful.

| lane zeroed | reach@5 | Δ vs full |
|---|---|---|
| (none — full+graph) | 52.6% | — |
| maxsim | 43.4% | -9.2pp |
| sit | 52.2% | -0.4pp |
| idf | 49.5% | -3.1pp |
| pick | 49.4% | -3.3pp |
| enc | 52.8% | +0.1pp |
| graph | 51.2% | -1.4pp |

## GAINS? (3) joint refit — does adding graph shift the others?

MULTI-START coordinate ascent (keep best over diverse inits, 4 passes each) on reach@5, grid {0,.25,.5,.75,1,1.25,1.5}. Greedy ascent from one start is unreliable (coupled gains → local optima); multi-start + a clean "graph on the tuned base" row guard it. Non-additive: cd74b974.

| arm | maxsim | sit | idf | pick | enc | graph | reach@5 |
|---|---|---|---|---|---|---|---|
| current (shipped) | 1.00 | 0.50 | 0.50 | 0.50 | 0.30 | 0.00 | 51.2% |
| refit, no graph | 1.50 | 0.00 | 0.50 | 0.25 | 0.25 | 0.00 | 53.2% |
| no-graph optimum + graph on top | 1.50 | 0.00 | 0.50 | 0.25 | 0.25 | 0.00 | 53.2% |
| refit + graph (joint) | 1.25 | 0.00 | 0.25 | 0.25 | 0.00 | 0.50 | 53.3% |

## SETTLE — per-lane displacement (mean gold-rank pull)

Δrank = rank(without lane) − rank(full+graph), over turns where gold rankable. Positive = the lane pulls the gold UP the field (b6a4dc6b: a field is measured by how it moves anchors).

| lane | median Δrank | mean Δrank | % turns helped |
|---|---|---|---|
| maxsim | +2.0 | +138.9 | 64% |
| sit | +0.0 | -1.6 | 32% |
| idf | +0.0 | +1.6 | 27% |
| pick | +0.0 | +1.3 | 23% |
| enc | +0.0 | -0.1 | 10% |
| graph | +0.0 | +0.7 | 11% |

