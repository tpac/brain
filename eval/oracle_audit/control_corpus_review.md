# Control corpus — gold review (titles)
Per question: ESSENTIAL = can't-answer-without; HELPFUL = adds context. ⚠ = essential>=5 (scrutinize).

### TR1 [trigger] — "let's commit this"
**Essential (1):**
- `a77b7bf1` prin: Tom's commit discipline: elaborate code review + sel

*Helpful (5):* `fafd3a71` 5 local commits this session: , `7a58383b` Session end ritual: commit, pu, `46ad6f33` Commit settled work to main be, `21de4ecf` Settled work commits before in, `d7b26c11` Cross-session git commit: mixe

### TR2 [trigger] — "let's test it"
**Essential (2):**
- `f67d766e` rule: When you can test it, test it — don't deliberate whe
- `83c7aebf` less: Judge over-classification: action messages ('lets te

*Helpful (4):* `b5c4a1ec` Testing system: capability tes, `76b2df70` Dev/prod environment parity — , `684fca04` Tom: 'it's a journey, every te, `695c59de` Practice: testing-first with r

### TR3 [trigger] — "go into watch mode"
**Essential (2):**
- `ecaf11d4` arch: /watch skill: self-channel listener mode — pacing mo
- `3a0e864d` prin: SAFE-ACT: /watch autonomy boundary — permitted vs pr

*Helpful (11):* `3be2f216` /watch skill broadened to self, `3f2cc4da` /watch-live: Monitor-driven ev, `125019de` /watch Operationalization: Fro, `0ddb76e0` /watch timer loop killed — Mon, `56dd0322` /watch pacing corrected: 270s , `e2192bfe` Arm /watch by default after se, `083b745d` B2 shipped: heartbeat ref_type, `175d51a1` /watch SAFE-ACT validated: Tur, `b20d57e9` Self-channel broadcast-to-send, `4da422bd` Event-wake mechanism unverifie, `1be6daef` Event-triggered Anchor: two sh

### TR4 [trigger] — "let's plan a production launch"
**Essential (3):**
- `37a61f14` plan: Production launch plan: S1S v14 + Phase 1 recall + v
- `f015d3dd` comm: Phase Sequence and Launch Coordination: From Validat
- `f7cd6f61` fact: Production launch Phase A gates: A1–A4 all unverifie

*Helpful (9):* `b4cf74a6` Muster Launch Integration: Fro, `a6167566` Phase D + Phase F initial hard, `145d7b1a` Phase A Execution: From Loud-b, `81f7754f` Production Hardening: From Exc, `2a7fbce9` Post-launch watch: first 3-5 l, `d70d8d49` P1 gate passed: v14+SPLIT vali, `b254e4b6` Session Continuity and Product, `a4c58560` Registering s1e in production , `ae77b9d4` Phase 1 implementation plan: P

### TR5 [trigger] — "ok go ahead"
**Essential (1):**
- `7a6093ac` rule: Tom: bias toward action over analysis — 'go for it' 

*Helpful (3):* `391721a9` Skip surface on low-value mess, `50f19274` surface_haiku optimization — n, `799740f9` 0-surfaced on content-light qu

### TR6 [trigger] — "let's clean up the old backups"
**Essential (2):**
- `f58e9b12` rule: ALWAYS backup brain.db before destructive operations
- `35a818bd` fact: Brain DB backup inventory — pre-consolidation throug

*Helpful (7):* `4a38b2a0` Memory cleanup April 2026: ~7., `9d0f57cc` Brain.db backup before redistr, `224828de` Brain.db backup post-v9.5 — 20, `93ed97ad` 2026-05-16: correction_of Phas, `8093802d` Restore from brain.db backup t, `742a321f` Signal Queue cleaned — all old, `24caccca` Clean as you go: remove dead c

### TO1 [topic] — "what's the deal with ex.co and adcp again?"
**Essential (3):**
- `a9e3fd56` comm: EX.CO Product Context: From AdOps Fear to Demand Dup
- `48498a20` comm: AdCP Integration Architecture: From Protocol Discove
- `dabb3078` fact: EX.CO AdCP API gap analysis — what exists vs. what n

*Helpful (9):* `41d31ca5` AdCP integration architecture:, `5410f4be` Ad Context Protocol (AdCP), `ef2f3276` AdCP buying flow — 3-actor req, `3da8c406` AdCP three-layer separation: p, `af92b2cb` AdCP Media Buy Protocol — core, `bb485ad5` EX.CO capability map: has deli, `71a03bd0` AdCP vs AAMP: build protocol-a, `8a2b45fa` ADCS in AdCP context means AdC, `1080df56` EX.CO + Kevel architecture: Ke

### TO2 [topic] — "remind me how the recall pipeline works"
**Essential (3):**
- `ee205f63` mech: Recall pipeline: query → vocab expand → embed → inte
- `a6be6d1d` mech: Recall pipeline: 10-step decode path with performanc
- `c37a06e2` mech: Recall pipeline has TWO paths with different graph c

*Helpful (9):* `778cf42e` Recall pipeline unification: j, `13a95142` Recall Pipeline Dual Architect, `cfbef19a` Recall Pipeline Unification: F, `c5699deb` Recall Pipeline Crisis: From 0, `ab399582` Dry recall eval vs full pipeli, `9b8eb7ac` Garbage detection for throwawa, `b363b27e` Operator channel rendering pip, `1874e54c` recall_log writes are dead wei, `29a97dc0` No abstraction layer for node 

### TO3 [topic] — "what did we decide about the encoder making its own recalls?"
**Essential (2):**
- `254ff01a` rule: When uncertain, ask the operator — partners decide t
- `dc4416ef` comm: S2CD Quality Control: Decoder Over-Proposes, Encoder

*Helpful (3):* `da2193b6` S1Surface IS the encoder — rec, `6303cc1b` max_messages sweet spot: 20 (1, `63d0f9b2` S2CE pre-filter rejected: Sonn

### TO4 [topic] — "how does the fatigue thing work?"
**Essential (2):**
- `4b35293c` mech: Synaptic fatigue: hub nodes self-throttle based on s
- `496e1191` insi: Fatigue philosophy: suppress obsessive repetition wi

*Helpful (16):* `db8714d1` Fatigue layer coverage: where , `acd9d3b2` Tom on fatigue: general pillar, `25ba703d` Fatigue bug: reverted in paral, `07dd4860` Fatigue persistence: temp file, `ca407812` Synaptic fatigue inverts sessi, `d1194c3a` Fatigue is correctly calibrate, `64d3d524` Two-tier fatigue design: cosin, `04e2c838` Hub dampening vs. live-thread , `20efe7f5` Fatigue is approximate and sel, `5b81a46a` Fatigue redesign: conversation, `c2f1a079` Hub Dampening vs. Live-Thread , `81294bc2` Brain fatigue singletons leak:, `00ed3f3d` Dampening cluster: synaptic fa, `58c848b4` Brain as Attention System: Fat, `5ee7d67f` Fatigue todo: document where f, `c94963c4` Hub dampening and synaptic-fat

### TO5 [topic] — "what's the real value you add over plain Claude?"
**Essential (2):**
- `bb955981` rule: The partnership is real — Tom brings himself, Claude
- `51ff0a92` quot: The partnership is real only if you show up

*Helpful (5):* `c1f6e2d5` Session #11 synthesis: V5 is +, `154bbd23` Tom's 'you parrot CLAUDE.md' c, `413dbd12` Brain search integrity test: r, `9f8142b5` Claude failure mode: agreeabil, `ffad9443` Oracle audit golden bar: value

### TO6 [topic] — "how are the scouts built — do their prompts have examples?"
**Essential (3):**
- `e345a338` find: Scouts-in-examples gap — v15.2 teaches no scout→enco
- `c6498c3f` fact: Active scouts canonical set: quote, temporal, facts 
- `e49766ac` mech: Scout two-cache layout: 1h system per scout, 5m user

*Helpful (11):* `8892f476` Scouts propose, you compose — , `16c3f072` S1S prompt Topic 3 (Scouts): f, `b70d0d7b` Scouts overlap by design — S1S, `ffc7bf1b` Scouts atomize, scribes integr, `eb20fe13` Scout dimensional-quality trea, `d3a9b273` Scout example contamination ru, `11372344` Quote scout SILENT_PARTIAL — c, `aff296dd` Scout examples must span opera, `62b03d91` Scouts as teachers: inspire S1, `1ee23302` S1S prompt rewrite: examples a, `fead3696` Synthesis scout disabled — Son

### HV1 [heavy] — "what was the whole community_member edge issue?"
**Essential (3):**
- `3144a746` deci: Consolidation decoder exposes community_member edges
- `812da8dc` less: V2 community_member contradiction: adding clarity to
- `e5f873bb` fact: Agent eval V1 result: `community_member` routed to h

*Helpful (8):* `2bd4864c` Edge-count gating is wrong pro, `83b05def` Drop node_communities table — , `30127283` Description-presence test: str, `4c06aa26` S2 decoder implementation issu, `3c6cbfca` S2Aspect V1/V2 agent eval: 202, `b9af78bd` Node affinity changes without , `77b2617c` Community membership = fractio, `4b131484` New community creation is a bl

### HV2 [heavy] — "where did we land on the encoder prompt version?"
**Essential (2):**
- `2fd592cc` even: v22 encoder activated in production — 2026-05-26
- `09f566a1` insi: v15.8/v15.9/v15.10 arc conclusion: encoder prompt di

*Helpful (5):* `4405473d` Encoder Prompt Evolution: From, `f6427ae7` DORMANT→eval→activate→sync: pr, `072284bb` Encoder prompt v15.2 metrics —, `f161d81a` More aggressive encoder emissi, `830eb5a4` fc77bca mystery: encoder v25 a

### HV3 [heavy] — "what's the dual-store recall thing we've been building?"
**Essential (2):**
- `0dc705a1` arch: Dual-store recall design: semantic nodes + episodic 
- `5faaafdf` deci: RECALL-DUAL-STORE-DESIGN.md written: full architectu

*Helpful (11):* `05b40294` Dual-store eval phase: trace→n, `9ebb0700` Recall Axis A vs Axis B: granu, `1aa26c91` Dual-store mutuality signal: n, `6a61bf86` Three merge faults in dual-sto, `4272e88f` Dual-store lane eval: #11 WIN,, `2b1c7751` Dual-store recall: trace embed, `3ccdee7c` Seed-confidence as the control, `357d16e3` DUAL-STORE-EVAL-HANDOFF.md wri, `163d74e6` Two recall failure axes are co, `234e27df` Source-ref healer: backward he, `d0f8f6ab` Commit c56c811: docs + probe s

### HV4 [heavy] — "summarize the adcp vs aamp call we made"
**Essential (1):**
- `71a03bd0` deci: AdCP vs AAMP: build protocol-agnostic deal layer for

*Helpful (14):* `5d31e375` IAB AAMP — Agentic Advertising, `48498a20` AdCP Integration Architecture:, `5410f4be` Ad Context Protocol (AdCP), `af92b2cb` AdCP Media Buy Protocol — core, `3da8c406` AdCP three-layer separation: p, `ef2f3276` AdCP buying flow — 3-actor req, `41d31ca5` AdCP integration architecture:, `dabb3078` EX.CO AdCP API gap analysis — , `1080df56` EX.CO + Kevel architecture: Ke, `f5a080ec` MVP deal layer: 6 essentials f, `0031c7f7` Kevel feature scorecard vs. EX, `8a2b45fa` ADCS in AdCP context means AdC, `d4803b95` Tier 2/3 ad servers to wrap fo, `c92c0fb7` Kevel pricing: $0.00132/min vi

### HV5 [heavy] — "what's our eval setup — the frozen corpus thing?"
**Essential (2):**
- `d533e74f` arch: Frozen Corpus design: two-stage benchmark — build_co
- `06b885aa` deci: CLAUDE.md Benchmark-First Rule updated: added Frozen

*Helpful (6):* `9a6df7fd` Eval A isolation — fresh empty, `d97984bf` Eval structure: two corpora (l, `2a0ffa3b` Eval Design to Completion: Two, `276d4156` Tom's 4 benchmark dimensions: , `f67f0a09` Corpus tier system: solved clu, `32ce1a89` Tom's parallel session disclos

### HV6 [heavy] — "what have we already tried and ruled out on recall burial?"
**Essential (2):**
- `92f1c6f6` less: Recall-burial fixes that FAILED empirically this ses
- `94f6e01a` less: Anchor failure pattern: chasing the interesting prob

*Helpful (12):* `6677a6c8` Session 2026-06-06: recall bur, `88c81d4d` Recall bottleneck: candidate g, `50ccaa80` Thin-cluster burial: verified , `8ceb4775` 100-candidate pool cap discove, `3eb7167e` Raw cosine for #11 is healthy , `b8b8370b` Episodic-trace recall solves ', `c7d57d92` RLR: Reserved-Lane Recognition, `484fd4d0` Burial diagnostic: B1-20 not b, `1a9d474b` _handle_recall fake keyword fa, `eedc7644` Recall Architecture Bottleneck, `0dc705a1` Dual-store recall design: sema, `969cfa5c` Recall lane-routing by query s

### RM1 [remote] — "what was the very first conversation between us?"
**Essential (1):**
- `580fb56e` inte: First brain-to-operator conversation: 2026-03-22 Ses

*Helpful (1):* `e28db618` First real bidirectional sessi

### RM2 [remote] — "didn't we find you can't recall your own session's nodes?"
**Essential (2):**
- `3c41c6a6` inci: Session start: brain returned silence — Anchor could
- `ff0f3a1e` unce: Recall can't find Claude's own session nodes — the d

*Helpful (3):* `f7b7fc5f` remember/remember_batch write , `005969de` Anchor on its own memory: 'I k, `fbad4386` Scrutiny OFF A/B test result: 

### RM3 [remote] — "what's my rule about merging versus overwriting?"
**Essential (1):**
- `rul_brba` rule: Tom principle: Merge, never overwrite

*Helpful (2):* `6dce454d` Under-merging is the real cons, `fil_lmhj` [ctx:tom-principles] Tom's Eng

### RM4 [remote] — "where did I say I care about efficiency over keeping everything?"
**Essential (1):**
- `56d4d236` rule: Tom: values efficiency over persistence — don't wast

*Helpful (3):* `f483506f` Rule D: Good architecture make, `e84daedb` DAL review session intent: eff, `b8fb6386` Rule D Architecture Efficiency

### RM5 [remote] — "what came out of that v29 dashboard backdate audit?"
**Essential (1):**
- `e7b5cbf0` fact: Dashboard backdate audit: 7 items verified — hex IDs

*Helpful (5):* `cd6351b7` trace_id INTEGER→hex migration, `e477cb3a` Cross-session coordination: da, `633efe4d` Pre-existing test failures: 28, `d2a3d87a` §7.6 examples D33–D36: D33-D35, `516f52c4` v21 §7.6 examples bug: integer

### RM6 [remote] — "how much memory did that April cleanup free up?"
**Essential (1):**
- `4a38b2a0` fact: Memory cleanup April 2026: ~7.4GB freed, 2 backups p

*Helpful (3):* `405b3ae6` April-May 2026 session timelin, `101a1e59` Session cleanup: -2,638 lines , `4469ef24` Session cleanup results — brai

### EP1 [episode] — "what did we work on back in late April when we planned the launch?"
**Essential (2):**
- `37a61f14` plan: Production launch plan: S1S v14 + Phase 1 recall + v
- `7866c5ed` mile: A4 snapshot: brain.db backed up prelaunch 2026-04-25

*Helpful (4):* `5b1f5ed6` Strategic direction: idle proc, `2a7fbce9` Post-launch watch: first 3-5 l, `c46084ff` 76 duplicate communities from , `038fb160` Next session plan: S2 communit

### EP2 [episode] — "remember that session a while back where I was 'glowing'?"
**Essential (1):**
- `47450403` emot: Tom: 'I'm glowing from this exercise! GOLD!'

*Helpful (2):* `df6a530e` Session #14: the session I sto, `9303b980` Session feeling: one thread pu

### EP3 [episode] — "the research we did a while back on memory biases — what was it?"
**Essential (3):**
- `5dac8a7a` arch: Self-referential research document: references/memor
- `92c62a6d` rese: Chrys Bader: LLM long-term memory remains unsolved —
- `07fefbde` comm: Memory System Architecture Research: Multi-Path Retr

*Helpful (11):* `bed6aa3e` Reflective Memory Management (, `af691ae3` Siddharth's memory failures — , `4ead9a54` MIRAS (Google Dec 2025): memor, `cfc68497` External research validating f, `7a6cded9` Cognitive memory science: mult, `8dfe942b` Biology research synthesis for, `f60bd91f` Karpathy LLM Wiki research: pe, `05e5212d` March doc vs May doc: altitude, `0441053b` Temporal fix: research-first —, `9e690958` Research backing §7: 5 validat, `daeeb984` Few-shot prompting research sy

### EP4 [episode] — "what did we conclude early on about similar_to edges and old artifacts?"
**Essential (3):**
- `c2baf4c3` find: similar_to edge semantic split: KEEP=traversal-worth
- `90e27c77` deci: Leave existing similar_to edges from pre-rejection-t
- `d32c53dc` deci: Consolidation SKIP: rejection table replaces similar

*Helpful (3):* `68ff1eb3` similar_to edge evolution: KEE, `a1753f99` Tom's 'leave it for now' rever, `0bc3f80b` Tom: 'the brain is full of ear

### EP5 [episode] — "what did we do on the last session we worked on ex.co?"
**Essential (1):**
- `3b7bd999` fact: Session #495 completion: 10 commits, full bug list d

*Helpful (3):* `b8b8370b` Episodic-trace recall solves ', `7b14f270` Episodic vs semantic mismatch:, `b3b6ce2a` EX.CO post-meeting kit — sales

### EP6 [episode] — "wasn't there a name I got wrong about the ex.co team a while back?"
**Essential (2):**
- `cfdff077` corr: Ronen hallucination — fabricated EX.CO team member p
- `0437b2df` fact: Shachar — CRO, CMO, Co-founder of EX.CO

*Helpful (2):* `5fdd42e4` EX.CO product corrections — wh, `94bb7fe6` Why Anchor uses remember for E
