# Corpus-v2 synthesis (2026-07-21)

Semantic judge pass over the walker gold corpus. n=2151 turns (1 dead-gold dropped). Rubric v3, Sonnet judges, Opus-audited.

## 1. Verdict distribution

| verdict | n | share |
|---|---|---|
| valid | 1004 | 47% |
| echo_mislabel | 1104 | 51% |
| ambiguous | 43 | 2% |

## 2. Per-stratum honest baselines (VALID golds only, echoes excluded)

reach@k = static λ=0.65 mix rank ≤ k. Door-1 = cue; Door-2 = window + session (Moments).

| stratum | n | reach@5 | reach@25 | median mix-rank |
|---|---|---|---|---|
| cue | 388 | 50% | 77% | 5 |
| window | 371 | 57% | 81% | 4 |
| session | 153 | 33% | 66% | 14 |
| **DOOR-1 (cue)** | 388 | **50%** | 77% | — |
| **DOOR-2 (window+session)** | 524 | **50%** | 77% | — |

Contrast: the OLD blended reach@5 counted all 2151 golds as one population.
- blended reach@5 over ALL golds: 31%
- reach@5 over VALID golds only: 50%

## 3. Echo-mislabel rate vs strong tier

strong = soft-gold AND Haiku-picked that turn. Protocol prediction: mislabels concentrate OFF the strong tier.

| tier | n | echo% | valid% |
|---|---|---|---|
| strong | 562 | 22% | 75% |
| non-strong | 1589 | 62% | 37% |

## 4. Failure-mode codebook — bridge devices on VALID misses (n=453)

Which held/missing device the judge named as the bridge. Multi-label (a bridge can name several).

| device | count | share of valid-misses |
|---|---|---|
| graph-walk | 201 | 44% |
| situation-lane / re-enrich | 179 | 40% |
| lexical / idf | 151 | 33% |
| episodic recency / same-session | 122 | 27% |
| node-class prior | 70 | 15% |
| style-recall (Tom-pattern) | 50 | 11% |
| conversation-window (M_h) | 35 | 8% |
| running session field | 25 | 6% |
| (other / unbucketed) | 16 | 4% |
| query segmentation | 3 | 1% |

## 5. Recency cross-tab — verdict & valid-reach by gold age

| age band | n | echo% | valid% | valid reach@5 |
|---|---|---|---|---|
| ≤1d | 339 | 26% | 68% | 70% |
| 1-7d | 321 | 44% | 55% | 60% |
| 7-21d | 567 | 55% | 44% | 38% |
| 21-45d | 530 | 63% | 36% | 41% |
| >45d | 393 | 57% | 41% | 42% |

## 6. Mechanical v0 stratum vs semantic verdict

Where the old strata_v0 mechanical bins land under semantic judgment.

| v0 stratum | n | echo% | valid% | of valids: cue/win/sess |
|---|---|---|---|---|
| CUE-SUFF | 1054 | 38% | 60% | 281/271/81 |
| MOMENT-DEP | 156 | 48% | 47% | 16/42/16 |
| NEITHER | 785 | 73% | 26% | 91/58/56 |

## Provenance

- rubric mix: {'v3': 1605, 'v2': 546}
- The 546 v2-rubric rows are front-half VALIDS (kept un-re-judged: loosening only moves echo→valid, never reverse). Their VERDICTS are safe; their STRATA were assigned pre-v3 (anaphora rule was already present in v2, so drift is second-order). Re-judging them for stratum-perfect consistency = ~3.6M more Sonnet tokens if wanted.

## Verdict — what the corpus says

**Echo-mislabel is the dominant defect: 51% of walker "gold" is response-echo, not helpful recall.** The soft-labeler minted a label whenever a node shared vocabulary with Anchor's NEXT response; half the time that node did not serve the moment.

**Honest reach is 50%, not 31%.** The blended 31% was dragged ~19pp by echo-mislabels no recall system should surface. Over genuine golds, static-mix reach@5 is 50%.

**The strong tier is the trustworthy sub-corpus** (22% echo / 75% valid vs 62% / 37% off-tier) — soft ∩ Haiku-picked is a ~75%-clean label, exactly the protocol prediction. Build and measure against the strong tier, not the blend.

**Door-2 is real and under-served.** Valid golds split cue 388 / window 371 / session 153. Session-stratum Moments reach only 33% (median rank 14) vs 50-57% for cue/window — the genuine Moment golds are exactly what the static mix cannot reach. That is the running-field case, now quantified on clean golds.

**Recency verdict: cohort-drift + episodic-recency, NOT topic-drift.** Two separable effects: (a) echo-rate CLIMBS with age (26%→63%) — old nodes accreted content that spuriously matches random future responses, so old golds are disproportionately mislabels (cohort/labeling artifact, not recall failure); (b) among VALID golds, reach@5 STEPS DOWN 70%(≤1d)→60%(1-7d)→~40%(>7d) then PLATEAUS — the fresh-gold advantage is episodic-lane presence (same-session recency), and once it decays reach settles at the semantic/lexical floor. The plateau past 21d (not monotonic decay) rules out topic-drift as the driver — it is a binary episodic-lane on/off, not gradual topical distance.

**Mechanical v0 was mostly wrong about NEITHER.** The v0 NEITHER bin (0% reach) is 73% echo-mislabel — its "0% reach" was largely an INVALID exam (non-golds), not mechanism failure, confirming 7e8f82e3. CUE-SUFF is 60% valid; the honest door-1 exam lives there minus its echoes.

**Top recoverable levers (bridge codebook on valid misses):** graph-walk (44% — the inert graph operator, 54777ca7) and situation-lane/re-enrichment (40%) dominate; episodic recency (27%) third. Style-recall (11%) is the door-2/terse-cue lever.

