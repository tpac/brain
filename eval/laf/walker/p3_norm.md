# p3_norm — P3.0 normalization repair mini-verdict (§20.13)

## 0 · variant sanity — PASS
- support ≡ current on dense lane ✓; sparse fixture: current z_max 9.1 → support z_max 1.9, zeros neutral ✓; rank bounded (|z|≤2.50) ✓
- coverage invariant: 30 empty-history turns × 3 norms, 90 pairs ≡ K0 ✓

## 1 · rank leg (walker pools, static gains)
| norm | arm | AUC val (June+) | AUC all | soft_r |
|---|---|---|---|---|
| current | k0 | 0.8226 | 0.7713 | 0.173 |
| current | win | 0.8676 | 0.8446 | 0.273 |
| support | k0 | 0.7749 | 0.7306 | 0.169 |
| support | win | 0.7810 | 0.7567 | 0.278 |
| rank | k0 | 0.8020 | 0.7557 | 0.170 |
| rank | win | 0.8351 | 0.8122 | 0.275 |

## 2 · gold-24 tier placement
| norm | arm | g+g top-1 | top-5 | top-25 | (gate tiers = gold_plus+gold) |
|---|---|---|---|---|---|
| current | k0 | 1 | 8 | 18 |  |
| current | win | 2 | 6 | 19 |  |
| support | k0 | 0 | 9 | 21 |  |
| support | win | 0 | 8 | 19 |  |
| rank | k0 | 1 | 4 | 17 |  |
| rank | win | 0 | 4 | 13 |  |

## 3 · VERDICT (pre-declared rule)
- current: June+ AUC (win) 0.8676 · tier gate PASS
- support: June+ AUC (win) 0.7810 · tier gate PASS
- rank: June+ AUC (win) 0.8351 · tier gate FAIL

**PICK: current** — no variant beats the incumbent on the pre-declared primary; fit substrate stays current

## 4 · eyeball (winner arm, first 2 cues × 3 norms)
### current · anchor_turn_0087
 1. [846eb135] (pick z6.8) ARCHITECTURE-FRACTAL.md: Scale 2 claims 'PARTIALLY BUILT' — needs verification a
 2. [aefb7795] (pick z6.8) Cross-stream fix plan: Group 2 (B1+B2+C1) approved, Group 3 (A2) queued
 3. [7f4b24a1] (pick z6.2) Tom's audit mandate: re-examine same files for missed dead code and confusion
 4. [81c3982b] (idf z7.1) Trace-pull presence fix: liveness + focus from real-turn s0 traces, not autosave
 5. [5f9e19e5] (pick z6.8) Oracle audit: B1-20 completed (recall miss confirmed); B2-9 pending — Opus + Tom
 6. [549c43d9] (enc z11.4) Phase 2 self-channel directed signal — design proposed, pending Tom approval (20
 7. [a9cb1bab] (enc z11.4) Presence proved live: real concurrent roster returned after daemon restart (2026
 8. [8f7df29d] (pick z6.3) Three self-channel message formats: Letter (absorb), Signal (act), Presence (jus ◀ TIERED
 9. [276d4156] (enc z7.3) Tom's 4 benchmark dimensions: answerability, encoding coverage, recall funnel, A
10. [8228767d] (pick z6.2) Tom's correction: stop being passive, iterate without waiting for permission

### current · anchor_turn_0132
 1. [9a19c5d4] (pick z6.6) Graph-based retrieval: HippoRAG (activation spreading) + PAM honest read (JEPA p
 2. [4954271a] (enc z8.3) Hub dominance is recall-structural, not graph-structural — access-based dampenin
 3. [c67d7476] (maxsim z3.4) Cross-stream live coordination: diagnosis via self_send, division of labor confi
 4. [00ed3f3d] (idf z7.5) Dampening cluster: synaptic fatigue + hub-dampening suspected broken post spread
 5. [ba71ea5d] (enc z9.5) Three-lens trace audit: A=surfaced help, B=silent miss, C=dry Claude counterfact
 6. [ca9d9103] (pick z6.6) Two-retrieval divergence audit: production recall vs oracle recall — gap localiz
 7. [96c194b1] (pick z5.8) Tom: form data-driven opinions — use research, brain nodes, recalls, and message
 8. [04e2c838] (idf z8.9) Hub dampening vs. live-thread nodes: fatigue applies the same dial to two differ
 9. [2890e908] (maxsim z2.9) Anchor rule: when designing brain mechanics, recall the brain first
10. [ecaf11d4] (maxsim z2.0) /watch skill: self-channel listener mode — pacing modes, SAFE-ACT boundary, warm

### support · anchor_turn_0087
 1. [81c3982b] (idf z6.7) Trace-pull presence fix: liveness + focus from real-turn s0 traces, not autosave
 2. [aefb7795] (maxsim z2.6) Cross-stream fix plan: Group 2 (B1+B2+C1) approved, Group 3 (A2) queued
 3. [7505c6b7] (maxsim z2.6) Cross-stream comms synthesis: 4 frictions → one seam; F2 is read-LAG not miss-st
 4. [f17f4ea7] (maxsim z3.7) Tom: 'Other sessions are all waiting. It's all you, do things right'
 5. [0d23b09e] (maxsim z2.7) Self-channel test suite fixed: 4 corrected + 1 added, 321 pass 0 fail (2026-06-0
 6. [56c5072f] (maxsim z2.8) Session 2026-06-02 shipped: trace consolidation + watch-live + presence fix + se
 7. [846eb135] (maxsim z1.5) ARCHITECTURE-FRACTAL.md: Scale 2 claims 'PARTIALLY BUILT' — needs verification a
 8. [16010f92] (maxsim z2.4) S2 Consolidation same idle-waste bug: 1043 runs/14d, always cold_start, 0 pairs 
 9. [bccad35f] (maxsim z1.8) F2 presence fix: stamp liveness at hook_recall only is wrong — watch-streams nev
10. [4eb88ad1] (maxsim z1.8) Phase2b self-channel PreToolUse delivery confirmed live via cross-stream Stop fe

### support · anchor_turn_0132
 1. [00ed3f3d] (idf z7.2) Dampening cluster: synaptic fatigue + hub-dampening suspected broken post spread
 2. [9a19c5d4] (maxsim z2.4) Graph-based retrieval: HippoRAG (activation spreading) + PAM honest read (JEPA p
 3. [04e2c838] (idf z8.7) Hub dampening vs. live-thread nodes: fatigue applies the same dial to two differ
 4. [c2f1a079] (maxsim z2.6) Hub Dampening vs. Live-Thread Continuity: From Fatigue Inversion to Type-Gated S
 5. [394f85d6] (idf z5.1) Hub dampening tradeoff: penalizing access_count hurts genuine high-use nodes lik
 6. [9032b824] (maxsim z3.2) S1 Scribe encodes ephemeral self-channel coordination ops as durable memory
 7. [c67d7476] (maxsim z3.4) Cross-stream live coordination: diagnosis via self_send, division of labor confi
 8. [2e6986a2] (maxsim z3.1) Spread Activation and Recall Sampling: From Reach Quantification to Agentic Rede
 9. [ba71ea5d] (maxsim z2.2) Three-lens trace audit: A=surfaced help, B=silent miss, C=dry Claude counterfact
10. [4954271a] (maxsim z1.9) Hub dominance is recall-structural, not graph-structural — access-based dampenin

### rank · anchor_turn_0087
 1. [aefb7795] (maxsim z1.7) Cross-stream fix plan: Group 2 (B1+B2+C1) approved, Group 3 (A2) queued
 2. [7f4b24a1] (maxsim z1.7) Tom's audit mandate: re-examine same files for missed dead code and confusion
 3. [846eb135] (maxsim z1.5) ARCHITECTURE-FRACTAL.md: Scale 2 claims 'PARTIALLY BUILT' — needs verification a
 4. [549c43d9] (maxsim z1.7) Phase 2 self-channel directed signal — design proposed, pending Tom approval (20
 5. [7a58383b] (maxsim z1.7) Session end ritual: commit, push, repackage plugin (copy-chat-to-conversations/ 
 6. [276d4156] (maxsim z1.6) Tom's 4 benchmark dimensions: answerability, encoding coverage, recall funnel, A
 7. [8228767d] (maxsim z1.7) Tom's correction: stop being passive, iterate without waiting for permission
 8. [e5852d92] (maxsim z1.7) xhigh-effort code review caught real bugs in just-committed 'tested' absorb code
 9. [d533e74f] (maxsim z1.6) Frozen Corpus design: two-stage benchmark — build_corpus.py (once) + sweep.py (f
10. [88c88b49] (maxsim z1.7) 1-by-1 test triage discipline: what it tests, does it really, is it redundant

### rank · anchor_turn_0132
 1. [2890e908] (maxsim z1.7) Anchor rule: when designing brain mechanics, recall the brain first
 2. [c67d7476] (maxsim z1.7) Cross-stream live coordination: diagnosis via self_send, division of labor confi
 3. [db8714d1] (maxsim z1.7) Fatigue layer coverage: three mechanisms at two distinct layers — candidate-soft
 4. [2e6986a2] (maxsim z1.7) Spread Activation and Recall Sampling: From Reach Quantification to Agentic Rede
 5. [69457c5a] (maxsim z1.7) Proposed 2-rule recall modulation: show-once + hub-crowd-prevention only
 6. [9a19c5d4] (maxsim z1.7) Graph-based retrieval: HippoRAG (activation spreading) + PAM honest read (JEPA p
 7. [ba71ea5d] (maxsim z1.7) Three-lens trace audit: A=surfaced help, B=silent miss, C=dry Claude counterfact
 8. [6fad647a] (maxsim z1.7) Post-Haiku production pipeline: three signals computed, silently dropped before 
 9. [96c194b1] (maxsim z1.7) Tom: form data-driven opinions — use research, brain nodes, recalls, and message
10. [30ac0132] (maxsim z1.7) Neuroscience contradiction: biology rewrites on recall — Anchor only reads, whic

