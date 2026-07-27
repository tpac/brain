# Enrichment support widening — can the ceiling move?

n=707 clean valids ≥2026-05-11 · misses@5 = 345 · **configuration-free** (no gains, no mix — pure support membership)

ceiling = P(gold ∈ lit set); rescuable@5 = misses whose gold is in the lit set (the reach a perfect scorer could convert).

## Policies

MARGINAL TRADE = extra rescuable golds per extra support-node/turn, measured against the K=5 base row. This is the column that decides: a policy that doubles the lit set to buy one gold is a bad trade no matter how much its ceiling rises.

| policy | support (nodes/turn) | ceiling | rescuable@5 | Δsupport | Δrescuable | marginal trade |
|---|---|---|---|---|---|---|
| seeds K=5 (base union) | 16.3 | 21% | 52 | +0.0 | +0 | — |
| seeds K=10 (base union) | 30.2 | 28% | 69 | +13.9 | +17 | 1.22 golds / node |
| seeds K=20 (base union) | 56.6 | 37% | 97 | +40.2 | +45 | 1.12 golds / node |
| seeds K=40 (base union) | 105.5 | 45% | 128 | +89.2 | +76 | 0.85 golds / node |
| K=5 + community(≥1 seed) | 159.2 | 47% | 112 | +142.8 | +60 | 0.42 golds / node |
| K=5 + community(≥2 seeds) | 41.7 | 26% | 66 | +25.4 | +14 | 0.55 golds / node |
| K=20 + community(≥1 seed) | 467.1 | 67% | 195 | +450.7 | +143 | 0.32 golds / node |
| K=20 + community(≥2 seeds) | 168.8 | 49% | 137 | +152.4 | +85 | 0.56 golds / node |
| K=5 + corridor(≥1 seed) | 18.4 | 23% | 53 | +2.1 | +1 | 0.48 golds / node |
| K=20 + corridor(≥1 seed) | 62.8 | 39% | 102 | +46.4 | +50 | 1.08 golds / node |
| community(≥1) ONLY, no 1-hop | 153.4 | 43% | 99 | +137.1 | +47 | 0.34 golds / node |
| community(≥2) ONLY, no 1-hop | 28.7 | 17% | 36 | +12.3 | -16 | -1.30 golds / node |

- reference row: K=5 base union → support 16.3, ceiling 21%, rescuable 52 (the committed 52-gold figure).

## Does widening reach the ABSTRACT types we systematically miss?

Gold type among rescuable misses, K=5 base vs the widest policy. The anatomy (ff93cce8) found rule/community/insight/lesson are the blind spot — if widening only adds concrete types it does not address it.

| gold type | rescuable @ K=5 base | rescuable @ K=20+community(≥1) |
|---|---|---|
| lesson | 14 | 31 |
| finding | 7 | 26 |
| architecture | 9 | 25 |
| principle | 3 | 19 |
| decision | 2 | 16 |
| correction | 2 | 10 |
| mechanism | 4 | 9 |
| insight | 2 | 7 |
| milestone | 2 | 7 |
| bug | 0 | 6 |
| rule | 3 | 5 |
| fact | 0 | 5 |
| open | 1 | 3 |
| reflection | 0 | 3 |
| concept | 1 | 2 |
| pattern | 0 | 2 |
| reference | 0 | 2 |
| quote | 0 | 2 |
| community | 1 | 2 |
| event | 1 | 2 |
| mental_model | 0 | 1 |
| diagnosis | 0 | 1 |
| vocabulary | 0 | 1 |
| bug_lesson | 0 | 1 |
| idea | 0 | 1 |
| research | 0 | 1 |
| plan | 0 | 1 |
| project | 0 | 1 |
| fix | 0 | 1 |
| craft_rule | 0 | 1 |
| design | 0 | 1 |

## Rescuable@5 by stratum

| stratum | misses | K=5 base | K=20+community(≥1) |
|---|---|---|---|
| cue | 150 | 22 | 88 |
| window | 126 | 17 | 63 |
| session | 69 | 13 | 44 |

