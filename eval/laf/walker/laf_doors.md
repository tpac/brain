# Door-separated evaluation — each mechanism on its own exam

Door-1 (`cue`) and door-2 (`window`+`session`) are DEDICATED corpora, not slices of one. Every earlier arm in this arc was fit AND scored on the blend, which diluted a door-1 mechanism by its door-2 cost (enrichment: cue +2.9/+4.1pp, window −2.6/−4.0pp, blend ≈0) and tuned the gains to a mixture of two exams. Here the gains are **fit on each door's own population**.

5 session→fold permutations · 2 corpora · paired bootstrap ×4000 · pass = CI excludes 0 across every seed.

## quality (≥2026-05-11) · door-1 (cue) — contextless recall · n=306

- paired sd ≈ 1.62pp → **MDE(95%) ≈ 3.2pp** at this n

| seed | A shipped | B refit | C +enrichment | B−A (95% CI) | C−A (95% CI) | C−B (95% CI) |
|---|---|---|---|---|---|---|
| 0 | 51.0% | 52.0% | 54.9% | +0.98 [-1.96, +3.92] | +3.86 [+0.00, +7.52] | +2.89 [-0.65, +6.54] |
| 1 | 51.0% | 52.9% | 54.9% | +1.95 [-0.98, +5.23] | +3.89 [+0.98, +7.19] | +1.94 [-1.31, +5.23] |
| 2 | 51.0% | 52.9% | 53.3% | +1.91 [-0.98, +4.90] | +2.25 [-0.98, +5.56] | +0.34 [-2.95, +3.59] |
| 3 | 51.0% | 51.6% | 54.2% | +0.61 [-2.61, +3.92] | +3.25 [+0.33, +6.21] | +2.64 [-0.65, +6.21] |
| 4 | 51.0% | 52.0% | 54.6% | +0.94 [-2.29, +3.92] | +3.56 [+0.65, +6.54] | +2.62 [-0.65, +5.88] |

- **B−A excludes 0 in 0/5 seeds · C−A in 3/5 · C−B in 0/5**

- **fitted λ** (weight on the current message; 1.0 = moment OFF) — B refit gains+λ: 0.65/0.80 · C refit gains+λ + enrichment: 0.65/0.80

## quality (≥2026-05-11) · door-2 (window+session) — conversation-state carry · n=401

- paired sd ≈ 1.00pp → **MDE(95%) ≈ 2.0pp** at this n

| seed | A shipped | B refit | C +enrichment | B−A (95% CI) | C−A (95% CI) | C−B (95% CI) |
|---|---|---|---|---|---|---|
| 0 | 51.4% | 50.4% | 49.9% | -0.99 [-2.99, +1.00] | -1.51 [-3.24, +0.00] | -0.52 [-2.24, +1.25] |
| 1 | 51.4% | 50.4% | 51.4% | -0.98 [-3.49, +1.50] | -0.00 [-2.00, +2.00] | +0.98 [-1.75, +3.49] |
| 2 | 51.4% | 49.1% | 51.9% | -2.24 [-4.74, +0.00] | +0.49 [-1.75, +2.74] | +2.72 [+0.50, +4.99] |
| 3 | 51.4% | 50.6% | 49.4% | -0.73 [-3.24, +1.75] | -1.97 [-4.74, +0.50] | -1.24 [-3.24, +0.75] |
| 4 | 51.4% | 50.4% | 50.6% | -0.99 [-2.74, +0.75] | -0.75 [-2.99, +1.75] | +0.25 [-2.00, +2.49] |

- **B−A excludes 0 in 0/5 seeds · C−A in 0/5 · C−B in 1/5**

- **fitted λ** (weight on the current message; 1.0 = moment OFF) — B refit gains+λ: 0.65 · C refit gains+λ + enrichment: 0.65

## wide (all valid golds) · door-1 (cue) — contextless recall · n=388

- paired sd ≈ 1.05pp → **MDE(95%) ≈ 2.1pp** at this n

| seed | A shipped | B refit | C +enrichment | B−A (95% CI) | C−A (95% CI) | C−B (95% CI) |
|---|---|---|---|---|---|---|
| 0 | 50.3% | 52.6% | 52.3% | +2.30 [+0.26, +4.38] | +2.04 [-0.52, +4.64] | -0.26 [-2.84, +2.06] |
| 1 | 50.3% | 51.8% | 52.1% | +1.51 [-1.03, +4.38] | +1.79 [-0.77, +4.38] | +0.28 [-2.32, +2.84] |
| 2 | 50.3% | 53.1% | 53.1% | +2.81 [+0.26, +5.41] | +2.82 [+0.00, +5.67] | +0.01 [-2.32, +2.32] |
| 3 | 50.3% | 51.5% | 54.6% | +1.27 [-1.55, +4.12] | +4.36 [+1.80, +7.22] | +3.09 [+0.52, +5.93] |
| 4 | 50.3% | 52.1% | 53.6% | +1.78 [-1.03, +4.64] | +3.35 [+0.52, +6.19] | +1.57 [-1.03, +4.12] |

- **B−A excludes 0 in 2/5 seeds · C−A in 2/5 · C−B in 1/5**

- **fitted λ** (weight on the current message; 1.0 = moment OFF) — B refit gains+λ: 0.65/0.80 · C refit gains+λ + enrichment: 0.65

## wide (all valid golds) · door-2 (window+session) — conversation-state carry · n=524

- paired sd ≈ 0.92pp → **MDE(95%) ≈ 1.8pp** at this n

| seed | A shipped | B refit | C +enrichment | B−A (95% CI) | C−A (95% CI) | C−B (95% CI) |
|---|---|---|---|---|---|---|
| 0 | 50.4% | 49.8% | 51.0% | -0.56 [-2.29, +1.15] | +0.57 [-1.72, +2.86] | +1.13 [-1.15, +3.44] |
| 1 | 50.4% | 50.6% | 49.4% | +0.19 [-1.91, +2.29] | -0.95 [-3.24, +1.15] | -1.14 [-3.24, +0.95] |
| 2 | 50.4% | 49.6% | 48.5% | -0.75 [-2.86, +1.34] | -1.91 [-3.82, +0.00] | -1.16 [-3.24, +0.76] |
| 3 | 50.4% | 49.4% | 48.3% | -0.94 [-3.05, +1.34] | -2.11 [-4.39, +0.00] | -1.16 [-2.86, +0.38] |
| 4 | 50.4% | 50.4% | 49.2% | +0.01 [-1.72, +1.72] | -1.14 [-3.05, +0.76] | -1.15 [-3.05, +0.57] |

- **B−A excludes 0 in 0/5 seeds · C−A in 0/5 · C−B in 0/5**

- **fitted λ** (weight on the current message; 1.0 = moment OFF) — B refit gains+λ: 0.65 · C refit gains+λ + enrichment: 0.65

