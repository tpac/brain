# P1 ship gate — laf_v1 vs champion through the real recall path

23 cues · limit=25 · IsolatedBrain · both arms rank today's full brain, post-cutoff results dropped identically at scoring

| arm | need@5 | need@25 | warm p50 | warm p95 | mode |
|---|---|---|---|---|---|
| champion | 11% | 17% | 1496ms | 2384ms | embeddings_first |
| laf_v1 | 14% | 23% | 819ms | 1501ms | laf_v1 |

laf_v1 first call (engine cache build): 1237ms

## Per-cue (need@5 / need@25)

| cue | champion | laf_v1 | Δ@5 | Δ@25 |
|---|---|---|---|---|
| anchor_turn_0087 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| anchor_turn_0132 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| anchor_turn_0345 | 0.25 / 0.25 | 0.25 / 0.50 | +0.00 | +0.25 |
| anchor_turn_0421 | 0.00 / 0.17 | 0.17 / 0.50 | +0.17 | +0.33 |
| anchor_turn_0538 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| anchor_turn_0764 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| anchor_turn_0906 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| anchor_turn_1106 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| anchor_turn_1120 | 0.33 / 0.33 | 0.00 / 0.33 | -0.33 | +0.00 |
| anchor_turn_1224 | 0.50 / 0.50 | 0.50 / 0.50 | +0.00 | +0.00 |
| operator_msg_0094 | 0.67 / 1.00 | 0.67 / 1.00 | +0.00 | +0.00 |
| operator_msg_0183 | 0.00 / 0.00 | 0.00 / 0.14 | +0.00 | +0.14 |
| operator_msg_0191 | 0.00 / 0.00 | 0.00 / 0.25 | +0.00 | +0.25 |
| operator_msg_0483 | 0.00 / 0.25 | 0.25 / 0.25 | +0.25 | +0.00 |
| operator_msg_0622 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| operator_msg_0718 | 0.25 / 0.25 | 0.25 / 0.50 | +0.00 | +0.25 |
| operator_msg_0978 | 0.00 / 0.33 | 0.33 / 0.33 | +0.33 | +0.00 |
| operator_msg_1014 | 0.33 / 0.33 | 0.33 / 0.33 | +0.00 | +0.00 |
| operator_msg_1162 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| operator_msg_1313 | 0.00 / 0.33 | 0.33 / 0.33 | +0.33 | +0.00 |
| operator_msg_1537 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| operator_msg_1558 | 0.00 / 0.00 | 0.00 / 0.00 | +0.00 | +0.00 |
| operator_msg_1572 | 0.12 / 0.12 | 0.12 / 0.25 | +0.00 | +0.12 |
