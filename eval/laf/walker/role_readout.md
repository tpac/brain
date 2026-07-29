# Role readout — do conn/auth reach the golds?

## quality (≥2026-05-11) · door-1 (cue) — contextless recall · n=306

| shipped-rank bucket | n | gold conn>0 | gold auth>0 | gold pick>0 | gold enc>0 | conn lane-rank≤5 | median lane support |
|---|---|---|---|---|---|---|---|
| hit@5 | 156 | 31 (20%) | 5 (3%) | 114 (73%) | 51 (33%) | 12 (8%) | 79 |
| 6–10 | 34 | 7 (21%) | 0 (0%) | 18 (53%) | 6 (18%) | 1 (3%) | 81 |
| 11–25 | 47 | 11 (23%) | 2 (4%) | 16 (34%) | 8 (17%) | 4 (9%) | 69 |
| beyond-25 | 69 | 10 (14%) | 3 (4%) | 27 (39%) | 7 (10%) | 1 (1%) | 87 |

- misses (rank>5): 150; gold conn-touched: 28 (19%); of those, conn-lane-rank≤5: 6 — the lane-oracle rescue ceiling

## quality (≥2026-05-11) · door-2 (window+session) — conversation-state carry · n=401

| shipped-rank bucket | n | gold conn>0 | gold auth>0 | gold pick>0 | gold enc>0 | conn lane-rank≤5 | median lane support |
|---|---|---|---|---|---|---|---|
| hit@5 | 206 | 43 (21%) | 6 (3%) | 138 (67%) | 60 (29%) | 12 (6%) | 80 |
| 6–10 | 42 | 8 (19%) | 1 (2%) | 13 (31%) | 12 (29%) | 3 (7%) | 78 |
| 11–25 | 66 | 10 (15%) | 1 (2%) | 20 (30%) | 8 (12%) | 3 (5%) | 84 |
| beyond-25 | 87 | 15 (17%) | 2 (2%) | 25 (29%) | 13 (15%) | 2 (2%) | 91 |

- misses (rank>5): 195; gold conn-touched: 33 (17%); of those, conn-lane-rank≤5: 8 — the lane-oracle rescue ceiling

## wide (all valid golds) · door-1 (cue) — contextless recall · n=388

| shipped-rank bucket | n | gold conn>0 | gold auth>0 | gold pick>0 | gold enc>0 | conn lane-rank≤5 | median lane support |
|---|---|---|---|---|---|---|---|
| hit@5 | 195 | 44 (23%) | 5 (3%) | 132 (68%) | 63 (32%) | 19 (10%) | 73 |
| 6–10 | 44 | 10 (23%) | 0 (0%) | 19 (43%) | 8 (18%) | 3 (7%) | 69 |
| 11–25 | 59 | 13 (22%) | 3 (5%) | 20 (34%) | 9 (15%) | 4 (7%) | 65 |
| beyond-25 | 90 | 12 (13%) | 3 (3%) | 30 (33%) | 9 (10%) | 1 (1%) | 82 |

- misses (rank>5): 193; gold conn-touched: 35 (18%); of those, conn-lane-rank≤5: 8 — the lane-oracle rescue ceiling

## wide (all valid golds) · door-2 (window+session) — conversation-state carry · n=524

| shipped-rank bucket | n | gold conn>0 | gold auth>0 | gold pick>0 | gold enc>0 | conn lane-rank≤5 | median lane support |
|---|---|---|---|---|---|---|---|
| hit@5 | 264 | 59 (22%) | 7 (3%) | 161 (61%) | 75 (28%) | 20 (8%) | 72 |
| 6–10 | 60 | 12 (20%) | 2 (3%) | 18 (30%) | 14 (23%) | 4 (7%) | 69 |
| 11–25 | 78 | 11 (14%) | 1 (1%) | 20 (26%) | 11 (14%) | 3 (4%) | 81 |
| beyond-25 | 122 | 16 (13%) | 2 (2%) | 30 (25%) | 16 (13%) | 2 (2%) | 82 |

- misses (rank>5): 260; gold conn-touched: 39 (15%); of those, conn-lane-rank≤5: 9 — the lane-oracle rescue ceiling

