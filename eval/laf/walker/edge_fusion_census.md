# Edge-conditioned fusion census — cross-lane edge context vs goldness

> ⚠️ **RETRACTED IN PART — read `edge_fusion_audit.md` first.**
> The risk-ratio "lift" columns below are NOT base-rate invariant. Lane-group
> support is wildly asymmetric (`current` lights ~1414 nodes/turn, `episodic`
> ~21), so dense cells sit near presence-ceiling and compress toward 1.0 while
> sparse cells have room to move. The headline "episodic carries 1.7-2.4x" was
> substantially that artifact: on odds ratios the divergence shrinks, and
> degree-matched the flagship `complementary` cell REVERSES (current +9.2pp vs
> episodic +7.7pp). What survives is lane-AGNOSTIC complementary adjacency plus
> ~1-2.5pp sign flips on the redundancy verbs. Brain node id:c31256e5 carries
> the corrected verdict; id:d48faec6 carries the echo-controlled gates.


turns: 707 · gold inside top-200 activated set: 682 (96%) · LIT_Z=1.0

Presence rate = share of nodes having ANY lit neighbour in that (verb class × neighbour lane) cell, gold vs non-gold, WITHIN a base-rank band. LIFT > 1 → gold-enriched (excitation licensed); < 1 → gold-depleted (inhibition licensed).

## base-rank band 1–5  (golds here: 362)

| verb class | neighbour lane | gold n | gold has | non-gold has | LIFT | gold mean z | non mean z |
|---|---|---|---|---|---|---|---|
| temporal | episodic | 8/362 | 2.2% | 1.0% | **2.17×** | 0.05 | 0.03 |
| structural | episodic | 15/362 | 4.1% | 2.4% | **1.73×** | 0.13 | 0.08 |
| corrective_soft | episodic | 24/362 | 6.6% | 4.0% | **1.66×** | 0.18 | 0.13 |
| corrective_strict | episodic | 14/362 | 3.9% | 2.6% | **1.47×** | 0.11 | 0.08 |
| temporal | current | 21/362 | 5.8% | 4.0% | **1.44×** | 0.17 | 0.11 |
| temporal | history | 19/362 | 5.2% | 3.9% | **1.36×** | 0.16 | 0.10 |
| hebbian | current | 193/362 | 53.3% | 45.7% | **1.17×** | 1.83 | 1.51 |
| hebbian | history | 179/362 | 49.4% | 42.8% | **1.15×** | 1.88 | 1.73 |
| similarity | episodic | 33/362 | 9.1% | 7.9% | **1.15×** | 0.26 | 0.23 |
| complementary | episodic | 103/362 | 28.5% | 25.0% | **1.14×** | 0.97 | 0.85 |
| complementary | current | 293/362 | 80.9% | 72.2% | **1.12×** | 2.72 | 2.20 |
| complementary | history | 252/362 | 69.6% | 63.6% | **1.09×** | 2.31 | 2.16 |
| corrective_soft | current | 62/362 | 17.1% | 15.9% | **1.08×** | 0.50 | 0.44 |
| structural | current | 166/362 | 45.9% | 49.2% | **0.93×** | 1.22 | 1.21 |

## base-rank band 6–25  (golds here: 189)

| verb class | neighbour lane | gold n | gold has | non-gold has | LIFT | gold mean z | non mean z |
|---|---|---|---|---|---|---|---|
| corrective_strict | episodic | 0/189 | 0.0% | 1.3% | **0.00×** | 0.00 | 0.04 |
| corrective_soft | episodic | 6/189 | 3.2% | 1.5% | **2.05×** | 0.10 | 0.04 |
| complementary | episodic | 41/189 | 21.7% | 12.5% | **1.73×** | 0.63 | 0.38 |
| temporal | history | 10/189 | 5.3% | 3.4% | **1.56×** | 0.12 | 0.08 |
| hebbian | episodic | 31/189 | 16.4% | 11.4% | **1.44×** | 0.46 | 0.38 |
| similarity | episodic | 11/189 | 5.8% | 4.1% | **1.41×** | 0.13 | 0.11 |
| temporal | current | 11/189 | 5.8% | 4.5% | **1.30×** | 0.22 | 0.10 |
| structural | episodic | 5/189 | 2.6% | 2.1% | **1.27×** | 0.07 | 0.06 |
| corrective_soft | current | 33/189 | 17.5% | 14.3% | **1.22×** | 0.49 | 0.34 |
| corrective_strict | history | 17/189 | 9.0% | 7.6% | **1.18×** | 0.21 | 0.18 |
| structural | current | 106/189 | 56.1% | 48.4% | **1.16×** | 1.54 | 1.11 |
| corrective_soft | history | 26/189 | 13.8% | 12.0% | **1.15×** | 0.39 | 0.28 |
| similarity | current | 51/189 | 27.0% | 30.7% | **0.88×** | 0.90 | 0.78 |
| complementary | history | 122/189 | 64.6% | 57.7% | **1.12×** | 1.80 | 1.50 |

## base-rank band 26–100  (golds here: 105)

| verb class | neighbour lane | gold n | gold has | non-gold has | LIFT | gold mean z | non mean z |
|---|---|---|---|---|---|---|---|
| corrective_soft | episodic | 0/105 | 0.0% | 0.7% | **0.00×** | 0.00 | 0.02 |
| temporal | episodic | 1/105 | 1.0% | 0.2% | **5.72×** | 0.04 | 0.00 |
| hebbian | episodic | 10/105 | 9.5% | 4.0% | **2.39×** | 0.36 | 0.13 |
| structural | episodic | 3/105 | 2.9% | 1.3% | **2.27×** | 0.07 | 0.04 |
| complementary | episodic | 13/105 | 12.4% | 6.1% | **2.03×** | 0.36 | 0.17 |
| corrective_strict | episodic | 1/105 | 1.0% | 0.5% | **1.96×** | 0.04 | 0.01 |
| corrective_strict | current | 14/105 | 13.3% | 8.9% | **1.49×** | 0.37 | 0.19 |
| temporal | current | 6/105 | 5.7% | 3.9% | **1.48×** | 0.19 | 0.08 |
| structural | history | 43/105 | 41.0% | 30.8% | **1.33×** | 0.77 | 0.58 |
| structural | current | 60/105 | 57.1% | 44.9% | **1.27×** | 1.38 | 0.92 |
| temporal | history | 4/105 | 3.8% | 3.0% | **1.27×** | 0.09 | 0.06 |
| corrective_strict | history | 8/105 | 7.6% | 6.5% | **1.17×** | 0.17 | 0.13 |
| corrective_soft | current | 16/105 | 15.2% | 13.3% | **1.14×** | 0.35 | 0.28 |
| hebbian | current | 35/105 | 33.3% | 29.3% | **1.14×** | 1.08 | 0.71 |

## base-rank band 101–200  (golds here: 26)

| verb class | neighbour lane | gold n | gold has | non-gold has | LIFT | gold mean z | non mean z |
|---|---|---|---|---|---|---|---|
| corrective_strict | history | 0/26 | 0.0% | 5.2% | **0.00×** | 0.00 | 0.09 |
| corrective_soft | episodic | 0/26 | 0.0% | 0.4% | **0.00×** | 0.00 | 0.01 |
| corrective_strict | episodic | 0/26 | 0.0% | 0.3% | **0.00×** | 0.00 | 0.01 |
| temporal | episodic | 0/26 | 0.0% | 0.1% | **0.00×** | 0.00 | 0.00 |
| hebbian | episodic | 3/26 | 11.5% | 2.1% | **5.42×** | 0.34 | 0.06 |
| similarity | episodic | 2/26 | 7.7% | 1.6% | **4.79×** | 0.11 | 0.04 |
| structural | episodic | 1/26 | 3.8% | 0.9% | **4.44×** | 0.09 | 0.02 |
| temporal | current | 3/26 | 11.5% | 3.4% | **3.38×** | 0.31 | 0.07 |
| temporal | history | 2/26 | 7.7% | 2.7% | **2.83×** | 0.26 | 0.05 |
| corrective_strict | current | 1/26 | 3.8% | 7.6% | **0.51×** | 0.10 | 0.15 |
| hebbian | current | 4/26 | 15.4% | 25.3% | **0.61×** | 0.51 | 0.55 |
| similarity | history | 9/26 | 34.6% | 21.3% | **1.62×** | 0.59 | 0.39 |
| similarity | current | 11/26 | 42.3% | 29.1% | **1.45×** | 1.11 | 0.62 |
| structural | current | 15/26 | 57.7% | 41.2% | **1.40×** | 1.20 | 0.79 |

## Cross-lane test — does neighbour-lane change the verdict?

Same verb class, different neighbour lane. If the lifts track each other, provenance is decoration; if they diverge (especially in sign), lane-conditioned fusion carries information post-fusion spread cannot.

| band | verb class | lift(current) | lift(episodic) | lift(history) |
|---|---|---|---|---|
| 1–5 | complementary | 1.12× | 1.14× | 1.09× |
| 1–5 | corrective_soft | 1.08× | 1.66× | 1.07× |
| 1–5 | corrective_strict | 1.00× | 1.47× | 0.98× |
| 1–5 | hebbian | 1.17× | 1.05× | 1.15× |
| 1–5 | similarity | 0.96× | 1.15× | 1.00× |
| 1–5 | structural | 0.93× | 1.73× | 0.95× |
| 1–5 | temporal | 1.44× | 2.17× | 1.36× |
| 6–25 | complementary | 1.10× | 1.73× | 1.12× |
| 6–25 | corrective_soft | 1.22× | 2.05× | 1.15× |
| 6–25 | corrective_strict | 1.01× | 0.00× | 1.18× |
| 6–25 | hebbian | 1.06× | 1.44× | 1.08× |
| 6–25 | similarity | 0.88× | 1.41× | 0.93× |
| 6–25 | structural | 1.16× | 1.27× | 1.06× |
| 6–25 | temporal | 1.30× | 0.90× | 1.56× |
| 26–100 | complementary | 1.11× | 2.03× | 1.03× |
| 26–100 | corrective_soft | 1.14× | 0.00× | 0.99× |
| 26–100 | corrective_strict | 1.49× | 1.96× | 1.17× |
| 26–100 | hebbian | 1.14× | 2.39× | 1.03× |
| 26–100 | similarity | 1.06× | 0.88× | 0.97× |
| 26–100 | structural | 1.27× | 2.27× | 1.33× |
| 26–100 | temporal | 1.48× | 5.72× | 1.27× |
| 101–200 | complementary | 1.25× | 1.02× | 1.00× |
| 101–200 | corrective_soft | 1.28× | 0.00× | 1.26× |
| 101–200 | corrective_strict | 0.51× | 0.00× | 0.00× |
| 101–200 | hebbian | 0.61× | 5.42× | 0.87× |
| 101–200 | similarity | 1.45× | 4.79× | 1.62× |
| 101–200 | structural | 1.40× | 4.44× | 0.85× |
| 101–200 | temporal | 3.38× | 0.00× | 2.83× |

## Door split (band 6–25, the promotable zone)

| door | verb class | neighbour lane | gold has | non-gold has | LIFT |
|---|---|---|---|---|---|
| door-1 | corrective_strict | episodic | 0.0% | 1.4% | **0.00×** |
| door-1 | structural | episodic | 4.9% | 2.1% | **2.34×** |
| door-1 | complementary | episodic | 25.9% | 11.7% | **2.22×** |
| door-1 | similarity | episodic | 9.9% | 4.5% | **2.19×** |
| door-1 | corrective_soft | episodic | 3.7% | 1.7% | **2.14×** |
| door-1 | hebbian | episodic | 21.0% | 11.2% | **1.87×** |
| door-2 | corrective_strict | episodic | 0.0% | 1.2% | **0.00×** |
| door-2 | temporal | episodic | 0.0% | 0.5% | **0.00×** |
| door-2 | structural | episodic | 0.9% | 2.1% | **0.45×** |
| door-2 | corrective_soft | episodic | 2.8% | 1.4% | **1.98×** |
| door-2 | temporal | history | 5.6% | 3.3% | **1.70×** |
| door-2 | corrective_strict | history | 11.1% | 7.9% | **1.41×** |
