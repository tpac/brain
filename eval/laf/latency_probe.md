# P1 latency probe — episodic field on the live path

24 cues × 3 repeats · IsolatedBrain (6854 master nodes) · one-time: warmup 1.2s, field matrices 0.3s

## Stage decomposition (per-cue medians → p50/p95 across cues, ms)

| condition | total | embed | ep_scan | ep_hydrate | gather | join | vec | sessions p50 | rows p50 | moments p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| A_eval_cutoff | 251/516 | 57/345 | 37/43 | 0/0 | 119/154 | 1/1 | 3/3 | 7 | 529 | 14 |
| C_live_fullhist | 245/519 | 56/345 | 55/56 | 0/0 | 109/152 | 1/1 | 3/3 | 6 | 402 | 15 |

## Comparators

| path | p50 | p95 |
|---|---|---|
| production brain.recall(limit=25) | 1359 | 2010 |
| maxsim_field (winner's other half) | 2 | 2 |

Cues with ZERO moments on the live shape (empty ≠ truth — honesty note): none

## Per-cue detail — C_live_fullhist (median of 3 repeats, ms)

| cue | total | gather | sessions | rows | moments |
|---|---|---|---|---|---|
| anchor_turn_0087 | 527 | 112 | 6 | 422 | 15 |
| anchor_turn_0132 | 388 | 58 | 3 | 227 | 14 |
| anchor_turn_0345 | 202 | 75 | 4 | 329 | 14 |
| anchor_turn_0421 | 386 | 107 | 6 | 362 | 15 |
| anchor_turn_0538 | 464 | 134 | 7 | 500 | 15 |
| anchor_turn_0764 | 471 | 94 | 5 | 402 | 15 |
| anchor_turn_0906 | 340 | 158 | 9 | 562 | 15 |
| anchor_turn_1106 | 413 | 117 | 6 | 470 | 15 |
| anchor_turn_1120 | 311 | 97 | 5 | 402 | 15 |
| anchor_turn_1224 | 460 | 139 | 7 | 438 | 15 |
| operator_msg_0094 | 239 | 154 | 8 | 517 | 14 |
| operator_msg_0183 | 166 | 94 | 5 | 444 | 15 |
| operator_msg_0191 | 200 | 117 | 6 | 395 | 13 |
| operator_msg_0483 | 475 | 97 | 5 | 407 | 15 |
| operator_msg_0622 | 216 | 129 | 7 | 477 | 15 |
| operator_msg_0718 | 210 | 126 | 7 | 361 | 14 |
| operator_msg_0978 | 169 | 91 | 5 | 392 | 12 |
| operator_msg_1014 | 198 | 128 | 7 | 331 | 14 |
| operator_msg_1162 | 183 | 114 | 6 | 461 | 14 |
| operator_msg_1313 | 189 | 106 | 6 | 370 | 14 |
| operator_msg_1369 | 147 | 78 | 4 | 295 | 14 |
| operator_msg_1537 | 250 | 132 | 7 | 462 | 15 |
| operator_msg_1558 | 168 | 81 | 4 | 376 | 12 |
| operator_msg_1572 | 581 | 81 | 4 | 358 | 15 |
