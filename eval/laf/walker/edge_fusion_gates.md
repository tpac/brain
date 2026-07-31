# T1 echo control + T2 provenance-marked reach

## T1. Degree-matched diffs (band 6–25) under echo controls

| population | matched turns | cell | episodic | current | history |
|---|---|---|---|---|---|
| ALL (audited baseline) | 142 | complementary | +7.7 [+0.8, +14.8] ** | +9.2 [+2.7, +15.8] ** | +8.3 [+0.3, +16.4] ** |
| ALL (audited baseline) | 142 | hebbian | +0.9 [-4.4, +6.3] | -0.7 [-8.3, +6.9] | -1.1 [-8.1, +6.0] |
| ALL (audited baseline) | 142 | similarity | +1.2 [-2.2, +5.4] | -2.5 [-9.7, +5.4] | -0.8 [-7.6, +5.7] |
| ALL (audited baseline) | 142 | corrective_strict | -1.8 [-3.2, -0.7] ** | +1.8 [-3.9, +8.2] | +2.3 [-3.3, +8.3] |
| NOT-PICKED-BEFORE | 101 | complementary | +8.0 [-0.5, +16.7] | +9.2 [+1.1, +16.7] ** | +9.6 [-0.0, +19.0] |
| NOT-PICKED-BEFORE | 101 | hebbian | -5.1 [-10.9, +0.8] | -3.6 [-12.5, +6.0] | -1.3 [-10.2, +8.5] |
| NOT-PICKED-BEFORE | 101 | similarity | +0.3 [-3.5, +4.5] | -3.8 [-11.8, +4.6] | -1.1 [-8.9, +7.2] |
| NOT-PICKED-BEFORE | 101 | corrective_strict | -2.2 [-3.9, -0.8] ** | +1.4 [-5.6, +8.4] | +1.7 [-4.8, +8.5] |
| NEVER-PICKED (strict) | 91 | complementary | +6.5 [-2.2, +15.0] | +8.0 [-0.8, +16.0] | +6.1 [-3.4, +15.8] |
| NEVER-PICKED (strict) | 91 | hebbian | -7.2 [-12.3, -1.9] ** | -6.6 [-16.4, +3.3] | -5.8 [-15.6, +3.7] |
| NEVER-PICKED (strict) | 91 | similarity | +0.8 [-3.2, +5.2] | -1.8 [-10.7, +7.3] | -0.5 [-8.8, +8.3] |
| NEVER-PICKED (strict) | 91 | corrective_strict | -2.1 [-3.9, -0.6] ** | +1.1 [-6.3, +8.9] | +2.5 [-4.1, +10.0] |
| ALL, ENC-ONLY episodic | 142 | complementary | -0.2 [-4.3, +4.1] | +9.2 [+2.7, +15.8] ** | +8.3 [+0.3, +16.4] ** |
| ALL, ENC-ONLY episodic | 142 | hebbian | +0.7 [-2.8, +4.8] | -0.7 [-8.3, +6.9] | -1.1 [-8.1, +6.0] |
| ALL, ENC-ONLY episodic | 142 | similarity | -0.1 [-1.8, +2.1] | -2.5 [-9.7, +5.4] | -0.8 [-7.6, +5.7] |
| ALL, ENC-ONLY episodic | 142 | corrective_strict | -0.8 [-1.7, -0.2] ** | +1.8 [-3.9, +8.2] | +2.3 [-3.3, +8.3] |

## T2. Per-lane provenance-marked reach — miss population (n=320)

organic = gold already in that lane's own top-25 (no hop). hop1 = reached by one complementary edge from an organic seed. hop2 = reached ONLY from a hop1 node (traversed-from-traversed — the excludable double-count). EFF = rescues per 100 nodes of fan-out.

| lane | organic | hop1 comp | of which NEW REACH | sorting | fanout | EFF | hop1 all-verbs | EFF | hop2 extra | fanout |
|---|---|---|---|---|---|---|---|---|---|---|
| maxsim | 181 (57%) | 25 | **16** | 9 | 51 | 0.15 | 51 | 0.11 | 14 | 150 |
| sit | 76 (24%) | 56 | **23** | 33 | 51 | 0.35 | 94 | 0.21 | 35 | 139 |
| idf | 41 (13%) | 37 | **9** | 28 | 58 | 0.20 | 53 | 0.10 | 18 | 186 |
| pick | 31 (10%) | 44 | **10** | 34 | 74 | 0.19 | 73 | 0.12 | 39 | 189 |
| enc | 28 (9%) | 12 | **3** | 9 | 28 | 0.13 | 27 | 0.11 | 27 | 77 |
| mh | 58 (18%) | 32 | **10** | 22 | 55 | 0.18 | 51 | 0.12 | 29 | 150 |
