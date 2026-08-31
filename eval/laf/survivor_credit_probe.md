# Survivor-credit A/B — absorbed-id drop in the LAF episodic lanes

23 cues · master 10246 nodes · shipped laf_v1 gains (maxsim 1.0 · pick 0.5 · enc 0.3 · idf 0.5 · sit 0.5), ±1-turn moments

## Footprint — did the fix fire at all?

| harvested role ids | dead (no live row) | credited to a survivor | cues touched | cues with a survivor gold node | credits landing ON gold |
|---|---|---|---|---|---|
| 6146 | 568 (9.2%) | 275 (4.5%) | 23/23 | 12/23 | 2 (in 2/23 cues) |

`credits landing ON gold` is the load-bearing column: zero would mean the corpus holds no case this bug hurts — a null INSTRUMENT, not a null effect.

Those are OCCURRENCES (an id harvested by 5 cues counts 5×) — the honest measure of how much evidence moves. The DISTINCT nodes behind them:

| distinct harvested | distinct dead | distinct credited | distinct survivors receiving credit |
|---|---|---|---|
| 1946 | 171 (8.8%) | 99 | 85 |

### Why each dead id had no live row

| reason | distinct ids |
|---|---|
| archived + survivor pointer | 108 |
| absent from nodes table | 47 |
| archived, retired (no survivor) | 16 |

## need@k — z_norm='support' (production K-store)

| subset | cues | need@5 OFF → ON | need@25 OFF → ON | brought | lost |
|---|---|---|---|---|---|
| ALL | 23 | 16% → 17% (+1.4pp) | 27% → 27% (+0.0pp) | +0 | −0 |
| credit-touched | 23 | 16% → 17% (+1.4pp) | 27% → 27% (+0.0pp) | +0 | −0 |
| gold-survivor | 12 | 13% → 16% (+2.8pp) | 28% → 28% (+0.0pp) | +0 | −0 |
| gold-credited | 2 | 12% → 12% (+0.0pp) | 29% → 29% (+0.0pp) | +0 | −0 |

## need@k — z_norm='current' (historical probe default)

| subset | cues | need@5 OFF → ON | need@25 OFF → ON | brought | lost |
|---|---|---|---|---|---|
| ALL | 23 | 17% → 17% (+0.0pp) | 27% → 27% (+0.0pp) | +0 | −0 |
| credit-touched | 23 | 17% → 17% (+0.0pp) | 27% → 27% (+0.0pp) | +0 | −0 |
| gold-survivor | 12 | 14% → 14% (+0.0pp) | 25% → 25% (+0.0pp) | +0 | −0 |
| gold-credited | 2 | 12% → 12% (+0.0pp) | 33% → 33% (+0.0pp) | +0 | −0 |

## Gold-rank movement — every gold node, both arms

`near-top` = gold reaching rank ≤ 50 in either arm (the only band where a move can change what recall surfaces); `tail` is everything deeper.

| z_norm | band | gold nodes | improved | worsened | unchanged | median Δrank | best Δ | worst Δ |
|---|---|---|---|---|---|---|---|---|
| support | near-top | 29 | 3 | 3 | 23 | +0 | +1 | -1 |
| support | tail | 66 | 15 | 11 | 40 | +0 | +32 | -9 |
| current | near-top | 29 | 2 | 10 | 17 | +0 | +1 | -4 |
| current | tail | 66 | 7 | 44 | 15 | -2 | +1 | -13 |

### need@k boundary crossings — z_norm='support'

| k | direction | cue | need | OFF rank | ON rank |
|---|---|---|---|---|---|
| 5 | gained | operator_msg_0978 | What distinguishes a partner from a good ass | 6 | 5 |

### need@k boundary crossings — z_norm='current'

None — no need changed side at k=5 or k=25. The lanes moved; nothing crossed.

### Largest gold-rank moves — z_norm='support'

| Δrank | cue | gold node | OFF | ON |
|---|---|---|---|---|
| +32 | operator_msg_0183 | 6a964255 | 2341 | 2309 |
| +27 | operator_msg_1572 | 0d3d7771 | 745 | 718 |
| -9 | operator_msg_1162 | 0cd2f9cb | 1538 | 1547 |
| -8 | anchor_turn_1106 | 3bcc506c | 656 | 664 |
| +3 | anchor_turn_0538 | 1472bf53 | 843 | 840 |
| +3 | operator_msg_0183 | 495f4aca | 1372 | 1369 |
| -3 | operator_msg_0718 | b7958bad | 360 | 363 |
| -2 | anchor_turn_0087 | 0b0ae0e7 | 1184 | 1186 |
| +2 | anchor_turn_1106 | e56dc13b | 294 | 292 |
| +2 | operator_msg_0183 | 83873db2 | 3775 | 3773 |
| +2 | operator_msg_1558 | b4d6f876 | 3732 | 3730 |
| +2 | operator_msg_1558 | 09bea718 | 583 | 581 |

### Largest gold-rank moves — z_norm='current'

| Δrank | cue | gold node | OFF | ON |
|---|---|---|---|---|
| -13 | operator_msg_0183 | 83873db2 | 868 | 881 |
| -13 | operator_msg_1162 | 0cd2f9cb | 181 | 194 |
| -10 | operator_msg_1537 | afc4311c | 65 | 75 |
| -9 | operator_msg_0718 | bd967982 | 151 | 160 |
| -9 | operator_msg_1572 | 0d3d7771 | 151 | 160 |
| -6 | anchor_turn_0345 | 203b06d6 | 136 | 142 |
| -6 | operator_msg_1558 | b4d6f876 | 417 | 423 |
| -5 | anchor_turn_0906 | 67e31fbe | 572 | 577 |
| -5 | operator_msg_1014 | 35f16063 | 63 | 68 |
| -4 | anchor_turn_1106 | 3bcc506c | 136 | 140 |
| -4 | operator_msg_0183 | 092df713 | 1195 | 1199 |
| -4 | operator_msg_0183 | 87bb8718 | 2675 | 2679 |
