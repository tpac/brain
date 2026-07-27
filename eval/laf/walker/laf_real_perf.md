# Real performance of every candidate at @5 (endo target)

n=707 clean valids ≥2026-05-11 · held-out session-grouped 5-fold · EVERY arm jointly refit on TRAIN (so no arm is scored under a rival's tuning) · paired bootstrap ×4000

**Power.** paired-delta sd ≈ 0.96pp → **MDE(95%) ≈ 1.9pp**. A null here means "no effect ≥ 1.9pp", NOT "no effect". Every candidate dismissed earlier today had a delta below this bar.

**Reading it.** reach@5 is the DECISION metric (endo takes top-5). MRR and mean Δrank are SENSITIVITY metrics — they use the whole rank, so they see movement that never crosses 5. An arm that moves MRR but not reach@5 is a real component, just not shippable alone.

**Two baselines, because one column cannot separate the two effects.** "Δ vs shipped" mixes *this lane set is better* with *refitting at all helps*. "Δ vs refit-5" holds refitting constant and isolates the lane-set change — that is the comparison that says whether a lane earns its place.

**Multiplicity.** 9 arms are compared here, each with a ~±1.9pp CI. Picking the largest and quoting its own CI overstates it (winner's curse); a best-of-9 needs roughly a 2.6pp effect to mean what a single 1.9pp CI would.

| arm | reach@5 | Δ vs shipped | 95% CI | Δ vs refit-5 | MRR | Δ MRR (95% CI) | verdict |
|---|---|---|---|---|---|---|---|
| baseline (shipped gains, no refit) | 51.2% | — | — | — | 0.3604 | — | reference |
| refit 5 lanes | 52.3% | +1.14pp | [-0.71, +2.97] | — | 0.3661 | +0.0059 [-0.0045, +0.0166] | not measurable (<MDE) |
| drop enc | 52.6% | +1.42pp | [+0.00, +2.97] | +0.28 [-0.99, +1.56] | 0.3669 | +0.0066 [-0.0017, +0.0149] | not measurable (<MDE) |
| drop sit+enc | 52.8% | +1.57pp | [-0.28, +3.54] | +0.43 [-1.13, +1.98] | 0.3628 | +0.0025 [-0.0087, +0.0137] | not measurable (<MDE) |
| epi_max fuse | 52.2% | +1.01pp | [-0.85, +2.97] | -0.13 [-1.70, +1.56] | 0.3635 | +0.0033 [-0.0070, +0.0138] | not measurable (<MDE) |
| epi_sum fuse | 49.6% | -1.55pp | [-3.68, +0.57] | -2.69 [-4.53, -0.85] | 0.3489 | -0.0112 [-0.0254, +0.0028] | not measurable (<MDE) |
| enrichment K=5 | 50.9% | -0.26pp | [-2.12, +1.56] | -1.41 [-3.11, +0.42] | 0.3693 | +0.0091 [-0.0029, +0.0213] | not measurable (<MDE) |
| enrichment K=20 | 52.3% | +1.14pp | [-0.71, +2.97] | -0.00 [-1.84, +1.84] | 0.3644 | +0.0042 [-0.0075, +0.0157] | not measurable (<MDE) |
| enrichment K=20 + corridors | 53.7% | +2.55pp | [+0.71, +4.38] | +1.41 [-0.42, +3.39] | 0.3610 | +0.0007 [-0.0112, +0.0126] | **REAL @5** |
| epi_max + enrichment K=20+corr | 53.5% | +2.26pp | [+0.28, +4.24] | +1.11 [-0.71, +2.97] | 0.3670 | +0.0069 [-0.0054, +0.0191] | **REAL @5** |

## Mean Δrank vs baseline (negative = gold ranked better)

| arm | mean Δrank | median Δrank | turns improved |
|---|---|---|---|
| refit 5 lanes | -4.85 | +0.0 | 36% |
| drop enc | -4.71 | +0.0 | 35% |
| drop sit+enc | -3.56 | +0.0 | 35% |
| epi_max fuse | -3.26 | +0.0 | 32% |
| epi_sum fuse | -4.15 | +0.0 | 37% |
| enrichment K=5 | -5.06 | +0.0 | 34% |
| enrichment K=20 | -5.54 | +0.0 | 36% |
| enrichment K=20 + corridors | -5.35 | +0.0 | 34% |
| epi_max + enrichment K=20+corr | -4.41 | +0.0 | 33% |

## Learned gains per arm (fold mean ± sd)

Fold sd is the generalization tell: a gain that swings across folds is fitting the fold, not the signal.

| arm | learned gains |
|---|---|
| baseline (shipped gains, no refit) | maxsim 1.00±0.00 · sit 0.50±0.00 · idf 0.50±0.00 · pick 0.50±0.00 · enc 0.30±0.00 |
| refit 5 lanes | maxsim 1.45±0.10 · sit 0.10±0.12 · idf 0.50±0.00 · pick 0.30±0.10 · enc 0.15±0.12 |
| drop enc | maxsim 1.30±0.10 · sit 0.25±0.00 · idf 0.50±0.00 · pick 0.45±0.10 |
| drop sit+enc | maxsim 1.25±0.00 · idf 0.50±0.00 · pick 0.45±0.10 |
| epi_max fuse | maxsim 1.35±0.12 · sit 0.00±0.00 · idf 0.60±0.12 · epi_max 0.35±0.12 |
| epi_sum fuse | maxsim 1.30±0.19 · sit 0.20±0.19 · idf 0.30±0.10 · epi_sum 0.25±0.00 |
| enrichment K=5 | maxsim 1.15±0.20 · sit 0.20±0.19 · idf 0.40±0.20 · pick 0.40±0.12 · enc 0.05±0.10 · enr_k5 0.55±0.19 |
| enrichment K=20 | maxsim 1.25±0.00 · sit 0.25±0.00 · idf 0.50±0.16 · pick 0.50±0.16 · enc 0.00±0.00 · enr_k20 0.55±0.10 |
| enrichment K=20 + corridors | maxsim 1.25±0.00 · sit 0.25±0.00 · idf 0.50±0.00 · pick 0.45±0.10 · enc 0.00±0.00 · enr_k20_corr 0.70±0.10 |
| epi_max + enrichment K=20+corr | maxsim 1.25±0.16 · sit 0.10±0.12 · idf 0.55±0.10 · epi_max 0.35±0.12 · enr_k20_corr 0.60±0.12 |

