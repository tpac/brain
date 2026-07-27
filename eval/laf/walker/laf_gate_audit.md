# Door 1 + Door 2 — held-out gains, then graph as a gated action

n=707 clean valids ≥2026-05-11 · 5-fold session-grouped CV · tie-fair ranks · fast scorer parity vs audited path |Δ| 0.000pp

Shipped-gain baseline reach@5 = **51.2%** (cross-check vs committed 51%: MATCH)

## DOOR 1 — is the fixed-gain retune real out-of-sample?

### T1. Fold-by-fold held-out reach@5 (gains fit on TRAIN only)

| fold | train n | test n | shipped | refit (held-out) | Δ |
|---|---|---|---|---|---|
| 0 | 545 | 162 | 50.0% | 51.9% | +1.9pp |
| 1 | 574 | 133 | 51.1% | 53.4% | +2.3pp |
| 2 | 564 | 143 | 53.1% | 54.5% | +1.4pp |
| 3 | 548 | 159 | 53.5% | 52.8% | -0.6pp |
| 4 | 597 | 110 | 47.3% | 48.2% | +0.9pp |
| **pooled** | — | 707 | **51.2%** | **52.3%** | **+1.1pp** |

- paired turn bootstrap (×2000) of the held-out Δ: **+1.14pp** (sd 0.91, 95% CI [-0.57, +2.97]) → INSIDE NOISE (CI spans 0)

- in-sample refit (the earlier number): 53.2% (+2.0pp). Optimism = in-sample − held-out = **+0.8pp** — the inflation that made Door 2 look worthless.

### T2. Per-fold learned gains — stability is the generalization tell

| fold | maxsim | sit | idf | pick | enc |
|---|---|---|---|---|---|
| 0 | 1.25 | 0.25 | 0.50 | 0.50 | 0.00 |
| 1 | 1.50 | 0.25 | 0.50 | 0.25 | 0.00 |
| 2 | 1.50 | 0.00 | 0.50 | 0.25 | 0.25 |
| 3 | 1.50 | 0.00 | 0.50 | 0.25 | 0.25 |
| 4 | 1.50 | 0.00 | 0.50 | 0.25 | 0.25 |
| **shipped** | 1.00 | 0.50 | 0.50 | 0.50 | 0.30 |
| **in-sample** | 1.50 | 0.00 | 0.50 | 0.25 | 0.25 |

| lane | fold mean | fold sd | shipped | verdict |
|---|---|---|---|---|
| maxsim | 1.45 | 0.10 | 1.00 | STABLE |
| sit | 0.10 | 0.12 | 0.50 | STABLE |
| idf | 0.50 | 0.00 | 0.50 | STABLE |
| pick | 0.30 | 0.10 | 0.50 | STABLE |
| enc | 0.15 | 0.12 | 0.30 | STABLE |

### T3. Held-out reach@5 by stratum

| stratum | n | shipped | refit (held-out) | Δ |
|---|---|---|---|---|
| cue | 306 | 51.0% | 53.6% | +2.6pp |
| window | 295 | 57.3% | 55.3% | -2.0pp |
| session | 106 | 34.9% | 40.6% | +5.7pp |

### T4. Held-out leave-one-out (fit without the lane, test)

Per fold: refit the remaining gains on TRAIN with the lane forced to 0, evaluate on TEST. Honest "does this lane earn its place".

| lane zeroed | held-out reach@5 | Δ vs full refit |
|---|---|---|
| (none) | 52.3% | — |
| maxsim | 30.6% | -21.8pp |
| sit | 52.6% | +0.3pp |
| idf | 50.8% | -1.6pp |
| pick | 50.6% | -1.7pp |
| enc | 52.5% | +0.1pp |

### T5. Gain sensitivity curves (others held at shipped)

reach@5 as ONE gain sweeps. Flat = the dial barely matters; peaked = it does. Shows WHY an argmax can be noise.

| lane | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 |
|---|---|---|---|---|---|---|---|
| maxsim | 42.3 | 46.1 | 48.4 | 50.2 | 51.2 | 51.2 | 51.3 |
| sit | 51.8 | 51.8 | 51.2 | 49.8 | 48.8 | 48.1 | 46.8 |
| idf | 48.4 | 49.4 | 51.2 | 50.6 | 48.5 | 45.1 | 42.3 |
| pick | 49.2 | 50.5 | 51.2 | 49.9 | 48.9 | 48.8 | 46.5 |
| enc | 51.5 | 51.2 | 50.6 | 50.5 | 50.4 | 49.6 | 49.1 |
| graph | 51.2 | 51.9 | 52.6 | 52.2 | 52.3 | 52.3 | 52.1 |

## DOOR 2 — graph as a gated action (vs always-on lane)

### T7. Gate calibration — cur_maxz quartiles (is the moment reach-starved?)

| quartile | cur_maxz range | n | shipped reach@5 | graph-rescuable | graph support (median) | gold∈graph |
|---|---|---|---|---|---|---|
| Q1 | 2.80–3.97 | 177 | 42.9% | 7 (7% of its misses) | 13 | 11% |
| Q2 | 3.97–4.37 | 176 | 49.4% | 11 (12% of its misses) | 13 | 14% |
| Q3 | 4.37–5.01 | 177 | 50.3% | 11 (12% of its misses) | 13 | 19% |
| Q4 | 5.01–8.83 | 177 | 62.1% | 23 (34% of its misses) | 15 | 39% |

- Q1→Q4 shipped spread: +19.2pp (8b3ef4f4 measured +19pp — replication check)

### T8. Gated vs always-on graph — held-out

ALWAYS-ON: one gain for every turn. GATED: fire graph only when cur_maxz ≤ threshold (fit threshold+gain on TRAIN, apply on TEST). Base gains = the held-out refit per fold.

| arm | held-out reach@5 | Δ vs refit base | fires on |
|---|---|---|---|
| refit base (no graph) | 52.3% | — | — |
| + graph ALWAYS-ON | 52.2% | -0.1pp | 100% |
| + graph GATED | 52.2% | -0.1pp | 100% |

- always-on vs base: -0.14pp (95% CI [-0.71, +0.28])
- gated vs base: -0.14pp (95% CI [-0.71, +0.28])
- gated vs always-on: +0.00pp (95% CI [+0.00, +0.00])
- per-fold chosen (threshold, gated gain, always gain): (none, 0.75, 0.75); (none, 0.00, 0.00); (none, 0.00, 0.00); (none, 0.25, 0.25); (none, 0.00, 0.00)

### T9. Global vs conditional lift — the washout test

Graph reach@5 delta measured WITHIN each cur_maxz quartile. If value concentrated in LOW quartiles while the global average is ~0, the lane washed out (768e827a pattern) and gating would recover it.

- optimizer's best fixed graph gain on the tuned base: **0.00**
- characterization gain used below: **0.50** (forced nonzero — the optimizer wanted 0, so a 0-vs-0 table would say nothing about the lane's behaviour)

| quartile | n | base | +graph | Δ |
|---|---|---|---|---|
| Q1 | 177 | 45.2% | 43.5% | -1.7pp |
| Q2 | 176 | 48.9% | 48.9% | +0.0pp |
| Q3 | 177 | 52.5% | 50.8% | -1.7pp |
| Q4 | 177 | 66.1% | 66.1% | +0.0pp |
| **all** | 707 | 53.2% | 52.3% | -0.8pp |

### T9b. Forced gate arms (gain 0.50) — held-out base, gate not fit

Fire graph only below a cur_maxz threshold. Gate NOT fitted here — each threshold reported directly, so a tie at gain 0 cannot hide the comparison.

| gate | fires on | reach@5 | Δ vs no-graph |
|---|---|---|---|
| always-on | 100% | 52.3% | -0.8pp |
| cur_maxz ≤ Q1 | 25% | 52.8% | -0.4pp |
| cur_maxz ≤ Q2 | 50% | 52.8% | -0.4pp |
| cur_maxz ≤ Q3 | 75% | 52.3% | -0.8pp |

### T12. Rank movement — what reach@5 cannot see

Gold rank with graph off vs on (tuned base, characterization gain 0.50). reach@5 only counts crossings of 5; this shows the whole distribution — a lane can move golds a lot and score 0.0pp.

| metric | value |
|---|---|
| turns where graph moved the gold at all | 164 (23%) |
| median Δrank (improvement, moved turns) | -1.0 |
| mean Δrank (all turns) | +0.58 |
| p90 improvement | +1 |
| p10 (worst regression) | -1 |
| golds GAINED into @5 | 4 |
| golds LOST from @5 | 10 |
| net @5 | -6 |

### T14. Rescue anatomy — which golds are graph-reachable

| slice | rescuable golds | share of slice misses |
|---|---|---|
| stratum=cue | 22 | 15% of 150 |
| stratum=window | 17 | 13% of 126 |
| stratum=session | 13 | 19% of 69 |
| gold type=lesson | 14 | 30% of 46 |
| gold type=architecture | 9 | 26% of 34 |
| gold type=finding | 7 | 16% of 44 |
| gold type=mechanism | 4 | 21% of 19 |
| gold type=rule | 3 | 25% of 12 |
| gold type=principle | 3 | 10% of 30 |
| gold type=insight | 2 | 20% of 10 |
| gold type=correction | 2 | 11% of 18 |

- gold convergence among rescuable (how many seeds reached it): 1 seeds: 36, 2 seeds: 9, 3 seeds: 5, 4 seeds: 2

### T10. SORTING substrate — within-pool ordering (held-out CV)

Does graph improve the order Haiku consumes? Pairwise logistic on candidate lane-z, session-CV. picked = echo-prone; soft-usage = answer-need (the honest target).

| lanes | picked-AUC | soft-AUC |
|---|---|---|
| 5 lanes (no graph) | 0.8903 | 0.8107 |
| 6 lanes (+graph) | 0.8904 | 0.8116 |

### T11. SORTING inside the fire regime (Q1 cur_maxz only, n=177)

The gated hypothesis: graph earns its place where the moment is reach-starved. Same CV, restricted to Q1.

| lanes | picked-AUC | soft-AUC |
|---|---|---|
| 5 lanes (no graph) | 0.8586 | 0.7788 |
| 6 lanes (+graph) | 0.8591 | 0.7748 |

### T13. Eyeball — turns where graph moved the gold most

| Δrank | rank off→on | stratum | cur_maxz | gold type | seeds | gold title |
|---|---|---|---|---|---|---|
| +251 | 419→168 | cue | 3.64 | architecture | 1 | Endo surface: recognition off Anchor's own loop — reflexive, |
| +57 | 80→23 | cue | 3.56 | lesson | 1 | Anchor held merge: checked dirty main for other-stream work  |
| +26 | 45→19 | session | 4.58 | lesson | 1 | Anchor held merge: checked dirty main for other-stream work  |
| +20 | 27→7 | cue | 5.26 | lesson | 4 | Anchor held merge: checked dirty main for other-stream work  |
| +15 | 21→6 | cue | 5.56 | architecture | 2 | /watch-live: Monitor-driven event-wake listener — peek_inbox |
| +13 | 55→42 | cue | 3.85 | architecture | 1 | Anchor working-memory visibility to encoder: layered provena |
| +12 | 39→27 | session | 4.20 | milestone | 2 | Dashboard Logs tab overhaul: unified feed, error grouping, f |
| +12 | 21→9 | session | 5.80 | event | 3 | Branch claude/busy-dirac-902ca7: parallel-session correctnes |
| -5 | 52→57 | cue | 4.35 | mechanism | 1 | Session handoff protocol: commit + doc + brain node (doc is  |
| -5 | 9→14 | window | 3.31 | community | 1 | Session Identity Across Lifecycle: From Daemon Restart to Re |
| -7 | 22→29 | session | 5.09 | open | 1 | Open: revisit Frame in S1E prompt later — noted by Tom in Tu |
| -21 | 100→121 | window | 4.49 | rule | 1 | Four-guide stream contract: worktree = identity + boundary,  |

## Verdict

- **Door 1** (fixed-gain retune, held-out): +1.14pp, 95% CI [-0.57, +2.97] → **NOISE**. In-sample optimism +0.8pp.
- **Door 2 always-on graph**: -0.14pp (CI [-0.71, +0.28])
- **Door 2 gated graph**: -0.14pp (CI [-0.71, +0.28]); gated vs always-on +0.00pp (CI [+0.00, +0.00]) → gating not separable from always-on

