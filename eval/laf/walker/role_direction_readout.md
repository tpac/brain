# Direction read-out — stored direction vs gold rescue

> ⚠️ **Read the AGGREGATE and the walk-policy table only.** Most per-verb rows
> below have n<5 rescues; individual verb directions (e.g. `corrects` 0-out/2-in)
> are NOT evidence. The load-bearing result is the aggregate 31-out / 34-in / 11-both
> split and the policy table showing direction keeps 77% of rescues vs 79% of noise.


misses: 345 · rescued turns: 76 — reached via OUT-edge only: 31, IN-edge only: 34, both: 11

## DIRECTED-CANDIDATE verbs

rescue hops: out 39 / in 35 · noise hops: out 2945 / in 3326

| verb | rescue out | rescue in | noise out | noise in |
|---|---|---|---|---|
| extends | 7 | 5 | 329 | 308 |
| grounds | 4 | 4 | 222 | 283 |
| implements | 5 | 2 | 310 | 344 |
| resolves | 4 | 0 | 132 | 128 |
| validates | 3 | 1 | 137 | 235 |
| instantiates | 1 | 3 | 50 | 52 |
| after | 3 | 0 | 32 | 31 |
| refines | 0 | 3 | 114 | 166 |
| contextualizes | 1 | 1 | 181 | 193 |
| corrects | 0 | 2 | 51 | 63 |
| supersedes | 0 | 2 | 102 | 63 |
| addresses | 2 | 0 | 81 | 31 |
| strengthens | 1 | 0 | 76 | 70 |
| challenges | 1 | 0 | 27 | 38 |
| before | 1 | 0 | 7 | 21 |
| same_domain_as | 0 | 1 | 3 | 8 |
| configures | 0 | 1 | 12 | 28 |
| enables | 0 | 1 | 81 | 72 |
| operationalizes | 0 | 1 | 19 | 36 |
| specializes | 0 | 1 | 1 | 1 |
| synthesizes | 0 | 1 | 16 | 11 |
| abstracts | 0 | 1 | 20 | 24 |
| constrains | 1 | 0 | 35 | 37 |
| produces | 1 | 0 | 78 | 51 |
| completes_phase | 0 | 1 | 0 | 0 |
| motivates | 0 | 1 | 23 | 47 |
| depends_on | 0 | 1 | 58 | 59 |
| targets | 1 | 0 | 0 | 3 |
| extends_fix_for | 1 | 0 | 1 | 0 |
| differs_from | 1 | 0 | 0 | 0 |
| scopes | 1 | 0 | 2 | 3 |
| maps | 0 | 1 | 1 | 0 |
| formalizes | 0 | 1 | 8 | 16 |

## SYMMETRIC verbs (direction = accident by design)

rescue hops: out 38 / in 38 · noise hops: out 4243 / in 5144

| verb | rescue out | rescue in | noise out | noise in |
|---|---|---|---|---|
| co_accessed | 19 | 25 | 1581 | 1660 |
| co_anchored | 10 | 2 | 411 | 481 |
| community_member | 4 | 4 | 793 | 1589 |
| similar_to | 2 | 5 | 109 | 107 |
| related_to | 1 | 2 | 1019 | 1026 |
| related | 2 | 0 | 325 | 279 |

## Walk-policy table (directed verbs honor policy; symmetric always both ways)

| policy | rescue hops kept | noise hops kept |
|---|---|---|
| both ways (direction ignored) | 150 (100%) | 15658 (100%) |
| out only (honor stored direction) | 115 (77%) | 12332 (79%) |
| in only (reverse) | 111 (74%) | 12713 (81%) |
