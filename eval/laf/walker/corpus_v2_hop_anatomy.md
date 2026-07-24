# 1-hop anatomy — rescue vs noise character (clean valid misses)

misses analyzed: 345 · gold within 1 hop of a top-5 seed: 75 (22%)
mean fan-out per turn: 36 nodes (the noise a blind +1-hop adds)

rescue rate by stratum: cue 23% · window 14% · session 33%

## Edge-relation character — rescue share vs noise share (lift)

| relation | rescue n | rescue % | noise % | LIFT |
|---|---|---|---|---|
| co_accessed | 43 | 28.9% | 20.1% | 1.4× |
| co_anchored | 12 | 8.1% | 5.7% | 1.4× |
| extends | 12 | 8.1% | 4.1% | 2.0× |
| grounds | 8 | 5.4% | 3.2% | 1.7× |
| community_member | 8 | 5.4% | 15.3% | 0.4× |
| similar_to | 7 | 4.7% | 1.4% | **3.4× |
| implements | 7 | 4.7% | 4.2% | 1.1× |
| resolves | 4 | 2.7% | 1.7% | 1.6× |
| validates | 4 | 2.7% | 2.4% | 1.1× |
| instantiates | 4 | 2.7% | 0.7% | **4.1× |
| after | 3 | 2.0% | 0.4% | **5.0× |
| related_to | 3 | 2.0% | 13.2% | 0.2× |
| refines | 3 | 2.0% | 1.8% | 1.1× |
| contextualizes | 2 | 1.3% | 2.4% | 0.6× |

## Scalar character — rescue vs noise medians

| axis | rescue | noise |
|---|---|---|
| edge description length | 118 | 22 |
| neighbor (target) degree | 21 | 14 |
| seed rank carrying the hop | 3 | — |
| edge age at turn (days) | 7 | — |
| gold (target) content size | 1059 | — |

seed-rank distribution of rescuing hops: r1:26 · r2:31 · r3:42 · r4:28 · r5:22

## Node-type character

rescued-gold types: lesson 44, architecture 20, finding 20, decision 17, correction 9, rule 9, mechanism 6, community 4

noise-neighbor types (top): decision 2199, community 1715, finding 1585, principle 1548, architecture 1279, lesson 872, fact 761, insight 584

## Filter curve — %% rescues kept vs %% fan-out cut

| filter | rescues kept | fan-out kept |
|---|---|---|
| ALL edges (blind +1hop) | 100% (149/149) | 100% |
| drop co_accessed | 71% (106/149) | 80% |
| drop co_accessed+community+related* | 62% (93/149) | 48% |
| semantic only + desc≥80 chars | 53% (79/149) | 36% |
| semantic + desc≥80 + non-hub (deg≤60) | 52% (78/149) | 35% |

