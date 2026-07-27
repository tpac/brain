# Depth probe (@5/@10/@25) + the enc⊕pick merge

n=707 clean valids ≥2026-05-11 · fast scorer (parity-checked in laf_gate_audit) · tie-fair ranks

## D1. Depth expansion — the same arms at @5 / @10 / @25

Held-out arms use the per-fold gains fit on TRAIN only (same CV as laf_gate_audit T1). Graph arms use the tuned base.

| arm | @5 | @10 | @25 |
|---|---|---|---|
| shipped gains | 51.2% | 62.0% | 77.9% |
| refit (held-out) | 52.3% | 63.5% | 78.9% |
| Δ vs shipped | +1.1pp | +1.6pp | +1.0pp |
| ⤷ bootstrap 95% CI @5 | +1.14pp [-0.57, +2.97] → NOISE |
| ⤷ bootstrap 95% CI @10 | +1.58pp [-0.28, +3.54] → NOISE |
| ⤷ bootstrap 95% CI @25 | +1.04pp [-0.99, +3.11] → NOISE |

### D1b. Graph lane at three depths (tuned base, gain swept)

The heavy-tail test: a rescue landing at rank 23 is invisible at @5 and counts at @25.

| gain_graph | @5 | @10 | @25 |
|---|---|---|---|
| 0.00 | 53.2% | 64.5% | 79.3% |
| 0.25 | 53.2% | 64.8% | 79.5% |
| 0.50 | 52.3% | 64.8% | 79.8% |
| 0.75 | 52.2% | 64.9% | 79.6% |
| 1.00 | 51.9% | 65.2% | 79.3% |
| 1.50 | 51.6% | 64.8% | 79.1% |

- best graph gain per depth: @5 → 0.25 (+0.0pp) · @10 → 1.00 (+0.7pp) · @25 → 0.50 (+0.4pp)

## D2. Per-lane solo reach (lane alone ranks gold ≤ k)

| lane | support (med) | @5 | @10 | @25 |
|---|---|---|---|---|
| maxsim | 5897 | 32% | 45% | 68% |
| sit | 5897 | 20% | 28% | 40% |
| idf | 387 | 13% | 19% | 28% |
| pick | 155 | 25% | 31% | 35% |
| enc | 57 | 6% | 9% | 15% |
| graph | 13 | 8% | 10% | 10% |

## D3. Is there a lane to lean on when the cosine field is flat?

Per-lane solo reach@10 by cur_maxz quartile. If every lane degrades together, no reweighting can rescue a vague cue — they all read the same flat geometry.

| lane | Q1 (flattest) | Q2 | Q3 | Q4 (sharpest) | Q1−Q4 |
|---|---|---|---|---|---|
| maxsim | 34% | 39% | 47% | 60% | -27pp |
| sit | 15% | 24% | 31% | 40% | -24pp |
| idf | 11% | 18% | 18% | 28% | -17pp |
| pick | 23% | 31% | 34% | 37% | -14pp |
| enc | 5% | 7% | 13% | 12% | -8pp |
| graph | 6% | 3% | 11% | 21% | -15pp |

## D4. Does `enc` hurt because it is too sparse? (the merge test)

Fuse pick+enc on RAW activations then support-z ONCE, so the union carries larger support. Merging after z would just be the additive sum two separate gains already express.

### D4a. Mechanism — does fusing actually fix the sparsity?

| lane | median support | median peak z | mean gold z |
|---|---|---|---|
| pick | 155 | 3.60 | +1.38 |
| enc | 57 | 1.87 | +0.18 |
| epi_max | 208 | 3.89 | +1.57 |
| epi_sum | 208 | 5.90 | +1.11 |

### D4b. Arms — separate vs merged (best gain per arm, in-sample)

| arm | best gains | @5 | @10 | @25 |
|---|---|---|---|---|
| shipped (0.5/0.3) | 0.50/0.30 | 51.2% | 62.0% | 77.9% |
| pick only (enc=0) | 0.50/0 | 51.5% | 61.8% | 77.8% |
| separate (pick,enc) | 0.50/0.00 | 51.5% | 63.2% | 77.8% |
| merged epi (max) | 0.50 | 50.6% | 63.2% | 77.5% |
| merged epi (sum) | 0.25 | 49.6% | 62.5% | 77.4% |

- NOTE: these are in-sample best-gain arms (a fair arm-vs-arm comparison, but each has the same in-sample optimism ~+0.8pp as Door 1). Only a difference LARGER than that is interesting.

### D4c. Held-out: merged vs separate (fit gains on TRAIN, @10)

| arm | held-out reach@10 |
|---|---|
| separate (5 lanes) | 63.4% |
| merged epi max (4 lanes) | 64.2% |

