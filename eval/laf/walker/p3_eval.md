# p3_eval — P3.2 pre-declared evaluations (§20.13)

- arms: A_full_picked, F_soft_ablate; M_e_f omitted engine-side (gold sessions walker-excluded; Q1 measured M_e flat)

## (3) tier placement (blind-judged)
| arm | tier | n | top-1 | top-5 | top-25 | median |
|---|---|---|---|---|---|---|
| k0_static | gold_plus | 16 | 1 | 4 | 6 | 74 |
| k0_static | gold | 80 | 0 | 4 | 12 | 367 |
| k0_static | silver_plus | 12 | 2 | 2 | 2 | 820 |
| k0_static | silver | 138 | 2 | 8 | 25 | 306 |
| winner_static | gold_plus | 16 | 1 | 3 | 7 | 74 |
| winner_static | gold | 80 | 1 | 3 | 12 | 342 |
| winner_static | silver_plus | 12 | 2 | 2 | 2 | 242 |
| winner_static | silver | 138 | 1 | 10 | 26 | 336 |
| A_full_picked | gold_plus | 16 | 1 | 3 | 3 | 200 |
| A_full_picked | gold | 80 | 0 | 3 | 6 | 525 |
| A_full_picked | silver_plus | 12 | 1 | 2 | 3 | 768 |
| A_full_picked | silver | 138 | 1 | 4 | 18 | 540 |
| F_soft_ablate | gold_plus | 16 | 0 | 3 | 6 | 56 |
| F_soft_ablate | gold | 80 | 1 | 3 | 10 | 317 |
| F_soft_ablate | silver_plus | 12 | 2 | 2 | 2 | 179 |
| F_soft_ablate | silver | 138 | 4 | 9 | 18 | 454 |

## (2) miss classes (gold+silver not in top-25; shared q1_reverse attribution)
| arm | near_miss | lane_buried | moment_seen | unreachable | weak_everywhere | not_in_field |
|---|---|---|---|---|---|---|
| k0_static | 24 | 50 | 61 | 35 | 28 | 3 |
| winner_static | 23 | 47 | 61 | 35 | 30 | 3 |
| A_full_picked | 12 | 73 | 62 | 35 | 31 | 3 |
| F_soft_ablate | 16 | 68 | 59 | 35 | 29 | 3 |

- leak canary (unreachable-substrate nodes ranked into top-25): {"k0_static": 0, "winner_static": 0, "A_full_picked": 0, "F_soft_ablate": 0}

## (4) shuffle control (fitted models; reference = same coefficients, j0-restricted)
| arm | real AUC | shuffled AUC | j0-restricted | verdict |
|---|---|---|---|---|
| A_full_picked | 0.9489 | 0.7980 | 0.8257 | holds |
| F_soft_ablate | 0.6561 | 0.5962 | 0.6937 | holds |

## eyeball · F_soft_ablate · anchor_turn_0087
 1. [7505c6b7] Cross-stream comms synthesis: 4 frictions → one seam; F2 is read-LAG not miss-st
 2. [8bc52e39] Two-stream live chat experiment: real back-and-forth between Anchor streams (202
 3. [f17f4ea7] Tom: 'Other sessions are all waiting. It's all you, do things right'
 4. [c67d7476] Cross-stream live coordination: diagnosis via self_send, division of labor confi
 5. [2ff0598a] Cross-stream friction taxonomy: F1-F5 verified ground-truth diagnosis (2026-06-0
 6. [aefb7795] Cross-stream fix plan: Group 2 (B1+B2+C1) approved, Group 3 (A2) queued
 7. [e328dd02] watch-live validated as real-time two-stream conversation substrate (2026-06-04)
 8. [9c9a8784] Two-stream cross-channel brainstorm: fede4918 + 4cb11a89 find each other, 5 fric
 9. [56c5072f] Session 2026-06-02 shipped: trace consolidation + watch-live + presence fix + se
10. [d57dd7c1] Self-channel live traffic in Turn 2: fresh test stream, DAL-cleanup coordination

## eyeball · F_soft_ablate · anchor_turn_0132
 1. [2890e908] Anchor rule: when designing brain mechanics, recall the brain first
 2. [ce4787f1] Three gaps in Anchor's memory substrate vs. field patterns
 3. [2e6986a2] Spread Activation and Recall Sampling: From Reach Quantification to Agentic Rede
 4. [00ed3f3d] Dampening cluster: synaptic fatigue + hub-dampening suspected broken post spread
 5. [c2f1a079] Hub Dampening vs. Live-Thread Continuity: From Fatigue Inversion to Type-Gated S
 6. [30ac0132] Neuroscience contradiction: biology rewrites on recall — Anchor only reads, whic
 7. [e524b57c] Corrections as topology suppression: LTD analog for recall-time edge behavior
 8. [394f85d6] Hub dampening tradeoff: penalizing access_count hurts genuine high-use nodes lik
 9. [83094ce4] Anchor on memory — 'I know more than I remember': philosophy and post-session-21
10. [e6110765] Merge-recall ceiling was the decoder, not the prompt — an LLM defers to its pre-

