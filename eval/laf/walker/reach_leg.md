# reach_leg — gold-24 self-checks + K0 (§20.5 reach leg)

- cues: 24; base-parity (K0 ≡ engine.scores(as_of), top-25 sequence): 24/24 PASS
- K0 need-reach: @5 8.3% (8/96) · @25 18.8% (18/96)

## positive control — ±1-turn episodic reproduction (9634cce9, ORIGINAL newest-500 recipe, envelope ±4.2pp)
- pick-only: @5 6.5% (expect 8%) · @25 13.8% (expect 16%) → PASS   [engine recipe, uncapped: @5 2.5% · @25 9.8%]
- enc-only: @5 6.1% (expect 6%) · @25 14.4% (expect 14%) → PASS   [engine recipe, uncapped: @5 1.6% · @25 8.9%]

**Overall: PASS — reach harness trusted**
