# Is the relation registry worth building? — walk-policy curve

Seed lane **sit** top-25 (T2's best seed) · miss population n=345 · verb coverage 92.7% of edge-relations

EFF = rescues per 100 nodes of fan-out. Registry arms (R*) must beat the cheap data-quality baselines (B2/B3) to justify the build.

| policy | hop1 rescues | new reach | sorting | mean fanout | EFF |
|---|---|---|---|---|---|
| B0 all verbs, both ways | 99 | **50** | 49 | 140 | **0.20** |
| B1 complementary-only (T2) | 50 | **24** | 26 | 42 | **0.35** |
| B2 desc≥80 only | 70 | **34** | 36 | 66 | **0.31** |
| B3 non-hub only (deg≤60) | 98 | **50** | 48 | 137 | **0.21** |
| R1 sign=reinforcing | 85 | **45** | 40 | 126 | **0.20** |
| R2 sign + symmetry-aware direction | 74 | **41** | 33 | 111 | **0.19** |
| R3 sign + desc≥80 | 58 | **30** | 28 | 53 | **0.32** |
| R4 sign + desc + non-hub | 57 | **30** | 27 | 52 | **0.32** |
| R5 sign + desc + non-hub + sym-dir | 43 | **25** | 18 | 36 | **0.35** |

## Transitivity as the multi-hop guardrail (§1.4)

hop2 from the R4 frontier, on turns R4 did not already rescue.

| hop2 policy | extra rescues | mean fanout | EFF |
|---|---|---|---|
| ungated | 37 | 131 | **0.08** |
| transitive-only | 9 | 28 | **0.09** |

## Verb coverage

classified 32757/35348 edge-relations (92.7%). Top unclassified: [('opens', 126), ('explains', 101), ('produced', 77), ('emergent_bridge', 76), ('qualifies', 70), ('part_of', 66), ('caused_by', 58), ('follows', 57), ('produced_by', 48), ('contrasts_with', 41)]

