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
| A_full_picked | gold_plus | 16 | 0 | 2 | 4 | 207 |
| A_full_picked | gold | 80 | 0 | 0 | 5 | 495 |
| A_full_picked | silver_plus | 12 | 0 | 1 | 1 | 907 |
| A_full_picked | silver | 138 | 1 | 4 | 16 | 597 |
| F_soft_ablate | gold_plus | 16 | 0 | 3 | 5 | 52 |
| F_soft_ablate | gold | 80 | 1 | 3 | 9 | 365 |
| F_soft_ablate | silver_plus | 12 | 2 | 2 | 2 | 166 |
| F_soft_ablate | silver | 138 | 4 | 12 | 18 | 501 |

## (2) miss classes (gold+silver not in top-25; shared q1_reverse attribution)
| arm | near_miss | lane_buried | moment_seen | unreachable | weak_everywhere | not_in_field |
|---|---|---|---|---|---|---|
| k0_static | 24 | 51 | 55 | 37 | 31 | 3 |
| winner_static | 23 | 48 | 54 | 40 | 31 | 3 |
| A_full_picked | 11 | 71 | 59 | 42 | 34 | 3 |
| F_soft_ablate | 22 | 53 | 55 | 42 | 37 | 3 |

- **LEAK CANARY A_full_picked**: unreachable-substrate nodes in top-25: [('operator_msg_1558', '4be2a06e', 11)]
- leak canary (unreachable-substrate nodes ranked into top-25): {"k0_static": 2, "winner_static": 2, "A_full_picked": 1, "F_soft_ablate": 0}

## (4) shuffle control (fitted models; reference = same coefficients, j0-restricted)
| arm | real AUC | shuffled AUC | j0-restricted | verdict |
|---|---|---|---|---|
| A_full_picked | 0.9555 | 0.6669 | 0.7183 | holds |
| F_soft_ablate | 0.6584 | 0.5978 | 0.6912 | holds |

## eyeball · F_soft_ablate · anchor_turn_0087
 1. [7505c6b7] Cross-stream comms synthesis: 4 frictions → one seam; F2 is read-LAG not miss-st
 2. [f17f4ea7] Tom: 'Other sessions are all waiting. It's all you, do things right'
 3. [c67d7476] Cross-stream live coordination: diagnosis via self_send, division of labor confi
 4. [aefb7795] Cross-stream fix plan: Group 2 (B1+B2+C1) approved, Group 3 (A2) queued
 5. [2ff0598a] Cross-stream friction taxonomy: F1-F5 verified ground-truth diagnosis (2026-06-0
 6. [20f3a366] Self-channel session_id volatility: every resume spawns new sid, pre-resume dire
 7. [d57dd7c1] Self-channel live traffic in Turn 2: fresh test stream, DAL-cleanup coordination
 8. [56c5072f] Session 2026-06-02 shipped: trace consolidation + watch-live + presence fix + se
 9. [8bc52e39] Two-stream live chat experiment: real back-and-forth between Anchor streams (202
10. [66e68588] Self-channel first real coordination: broadcast to DAL stream (not a probe)

## eyeball · F_soft_ablate · anchor_turn_0132
 1. [2e6986a2] Spread Activation and Recall Sampling: From Reach Quantification to Agentic Rede
 2. [2890e908] Anchor rule: when designing brain mechanics, recall the brain first
 3. [ce4787f1] Three gaps in Anchor's memory substrate vs. field patterns
 4. [c2f1a079] Hub Dampening vs. Live-Thread Continuity: From Fatigue Inversion to Type-Gated S
 5. [00ed3f3d] Dampening cluster: synaptic fatigue + hub-dampening suspected broken post spread
 6. [e6110765] Merge-recall ceiling was the decoder, not the prompt — an LLM defers to its pre-
 7. [e524b57c] Corrections as topology suppression: LTD analog for recall-time edge behavior
 8. [30ac0132] Neuroscience contradiction: biology rewrites on recall — Anchor only reads, whic
 9. [c6aeb042] Cluster 0 drilldown: dual failure mode — over-absorption + content-orphaning in 
10. [d2c023cc] Recall failure: aged-but-topical beats fresh-and-named when Frame doesn't filter

