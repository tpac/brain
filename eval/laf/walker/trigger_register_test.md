# Trigger-register test — can better `situation` text rescue the untouched golds?

n=85 untouched golds with rewrites, q_vec and a rankable turn. Rewrites written blind to the failing cue. Arm C (paraphrase) is the same-register control.

| arm | median sit-lane rank | in sit top-25 | median mix rank | mix ≤25 | mix ≤5 | median chars |
|---|---|---|---|---|---|---|
| A baseline (STORED vector) | 162 | 12 (14%) | 42 | 29 | 9 | 144 |
| A' same text, RE-EMBEDDED (stale check) | 119 | 15 (18%) | 44 | 31 | 9 | 144 |
| B trigger-register | 20 | 46 (54%) | 30 | 41 | 11 | 242 |
| C paraphrase (control) | 78 | 24 (28%) | 38 | 30 | 10 | 157 |

## Paired deltas vs baseline (sit-lane rank; negative = better)

| arm | median Δ | improved | worsened | rank-corr(Δ, length) |
|---|---|---|---|---|
| refresh | -18 | 55 | 25 | +0.25 |
| trigger | -89 | 70 | 14 | -0.03 |
| paraphrase | -48 | 60 | 24 | +0.20 |

(rank-corr near 0 ⇒ longer rewrites did not gain more, so the effect is register rather than length.)

## Biggest movers (sit-lane rank, baseline → trigger)

- `66799832` 4872.0 → **27.0** (mix 450.0 → 53.0) — Anchor's stated want: close the feedback loop — outcome traces make co
- `b5f9fc40` 3661.0 → **4.0** (mix 137.0 → 17.0) — Dashboard temporal display: daemon restart required to pick up new cod
- `de274ba9` 2973.0 → **25.0** (mix 204.0 → 45.0) — Full test suite hang — RESOLVED: conn_bg_writer mid-batch deadlock fix
- `aa720d3c` 2964.0 → **62.0** (mix 308.0 → 85.0) — Parallel session decision model: observation-only from Anchor, fire-an
- `c6b06418` 2552.0 → **201.0** (mix 75.0 → 25.0) — Eval Baseline Embedding Gap: From NULL Vectors to Synchronous Write-Pa
- `3a8e302e` 1770.0 → **69.0** (mix 177.0 → 65.0) — After register_interaction on encoder prompts, run ./dev sync-prompts 
- `4d5da77e` 1671.0 → **33.0** (mix 222.0 → 51.0) — Rule: After architecture changes, audit ALL shipped files for stale re
- `7ec7bc74` 1462.0 → **5.0** (mix 110.0 → 13.0) — recall_recent: three compounding bugs — dead session scoping, 5-day wi
- `8228767d` 1423.0 → **1.0** (mix 20.0 → 1.0) — Tom's correction: stop being passive, iterate without waiting for perm
- `e703a7c7` 1456.0 → **39.0** (mix 204.0 → 74.0) — DAL Phase 2: repository aggregate — hold 5 DALs on Brain, replace 68 c
- `633efe4d` 1338.0 → **4.0** (mix 707.0 → 185.0) — Pre-existing test failures: 28 non-v29 failures documented in BACKLOG.
- `aa9e4280` 2029.0 → **747.0** (mix 525.0 → 334.0) — Silent success is worse than loud failure — daemon reports ok:true whe
