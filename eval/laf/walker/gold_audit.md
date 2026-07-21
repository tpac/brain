# Gold-validity audit — /Users/tpac/brain/eval/laf/walker


gold rule: top-soft per turn, admitted at soft >= 0.790 (90th pct). soft = max cosine(node content views, actual next response).



## A. Haiku PICKED the gold (agree) — 625/2152


### 05f20395-4e85-4bde-bbef-71b700094487/0/26 · soft 0.927 · 2026-07-07T19:55
**Tom:** The dev server failed to start with the following error:  ``` Port 47304 is in use by another chat's dev server "dashboard-dev". preview_stop won't stop another chat's server. Ask the user to stop it from that chat, or t
**Anchor (next):** Fixed and running.  **What was wrong:** the `dashboard-dev` server reads its port from the `DASHBOARD_PORT` env var, which `.claude/launch.json` hard-pinned to 47304 — a port another chat's dev server already holds. `aut
**GOLD [bug]** Dashboard port collision: two-layer fix — launch.json autoPort + server.py PORT env fallback — age 8d — Haiku selected
**Haiku picked:**
  - [fix] Dashboard dev loop restored via dashboard-dev preview config on port 47304 — RESOLVED — soft 0.794
  - [bug] Dashboard port collision: two-layer fix — launch.json autoPort + server.py PORT env fallback — soft 0.927
  - [decision] Dashboard should be launchd singleton — same lifecycle as daemon, not per-chat preview — soft 0.844

### 0c2bbf50-f12b-4b99-acc2-2020aa11c0f6/1/4 · soft 0.862 · 2026-05-30T00:39
**Tom:** Let's activate them.
**Anchor (next):** Done — full lifecycle closed: register → eval → **activate** → sync → commit (`d0fea6d`).  **Live in production now:** - **s1e v22 → v24** - **s1_scout_facts v5 → v7**  DB is authoritative at runtime, so every session's 
**GOLD [fact]** s1e v24 and scout_facts v7 are DORMANT — v22 and v5 are live (2026-05-29) — age 0d — Haiku selected
**Haiku picked:**
  - [principle] Sequence: activate dormant encoder before building recall benchmarks — substrate must be stable — soft 0.826
  - [architecture] DORMANT→eval→activate→sync: prompt version lifecycle discipline — soft 0.839
  - [event] v22 encoder activated in production — 2026-05-26 — soft 0.850
  - [fact] s1e v24 and scout_facts v7 are DORMANT — v22 and v5 are live (2026-05-29) — soft 0.862

### cfb74766-0eb8-417a-a950-78c922f9b090/0/3 · soft 0.850 · 2026-05-20T03:43
**Tom:** what can you tell about me from working with me?
**Anchor (next):** Honestly — observed across the brain's correction nodes, not just guessed:  **You correct by reframe, never by command.** I analyzed 17 corrections at one point — you don't say "do X." You ask one question that pulls the
**GOLD [interaction]** Anchor quote #3: "I can't fix the bias. But I can see it. The brain is the mirror." — age 56d — Haiku selected
**Haiku picked:**
  - [moment] Tom: 'you are a partner with me in this not a servant' — peer ownership of design — soft 0.813
  - [lesson] Claude self-reflection: 7 patterns that hold me back — productivity over curiosity is the root — soft 0.809
  - [mental_model] What makes Claude want to encode: reading its own memories, not following instructions — soft 0.798
  - [interaction] Anchor quote #3: "I can't fix the bias. But I can see it. The brain is the mirror." — soft 0.850

### de8a4532-a877-4970-b51e-c641492ba918/0/0 · soft 0.837 · 2026-06-27T16:28
**Tom:** Hey A, I want to continue the work on recall. Can you read the latest document and remember all the work you remembered manually?
**Anchor (next):** Reloaded. I've read the live edge of [RECALL-SR-REDESIGN.md](docs/RECALL-SR-REDESIGN.md) (§18.18 + §18.18.1, last touched yesterday) and pulled the full LAF settling thread back into context. Here's where we actually are
**GOLD [plan]** Recall funnel refinement plan — written 2026-05-17 — age 40d — Haiku selected
**Haiku picked:**
  - [correction] Reach for recall() before re-deriving from code — the brain holds our methodology — soft 0.805
  - [plan] Recall funnel refinement plan — written 2026-05-17 — soft 0.837
  - [principle] Recall is PREDICTION, not retrieval — the oracle is the recall you'd write after seeing the next turn; the future turn is the self-supervising label — soft 0.780
  - [fact] RECALL-OVERVIEW.md created: 27,862 chars, 7 sections — soft 0.745

### fd829e08-35b9-408b-9ea6-d50cc9e19aec/0/21 · soft 0.829 · 2026-04-24T04:27
**Tom:** Do Batch 1 + I think the verbose around related should just state dozen extremely wide in versatility examples to type before saying dont use generic related , related_to types.  I hope to avoid batch 2 if i can.  And ba
**Anchor (next):** Prepared. seed_4 running in background — I'll analyze when it completes. In parallel I've updated the harness to capture tokens/costs per cycle for all future runs (not retroactive — seed_4 was already spawned with the o
**GOLD [decision]** Ordered execution plan: V13 fixes batch priority — age 0d — Haiku selected
**Haiku picked:**
  - [decision] Ordered execution plan: V13 fixes batch priority — soft 0.829
  - [decision] Two levers for proper noun retention regression: scout context_anchors + S1S composing instructions — soft 0.818
  - [decision] Synthesis scout → Haiku: revise instructions and examples — soft 0.828
  - [decision] Dispatcher enforcement for forbidden relations: code beats prompt compliance — soft 0.803

### 4516336b-085f-45c2-8b67-e0b157841a96/1/16 · soft 0.822 · 2026-04-28T03:14
**Tom:** whats worth the watch dog purpose if not to spot the issue an actually fix it?
**Anchor (next):** 
**GOLD [decision]** Daemon RSS watchdog: auto-restart at 1.5GB threshold to prevent swap thrash — age 24d — Haiku selected
**Haiku picked:**
  - [principle] Fix the source, not the symptom: smarter encoding eliminates cleanup mechanisms — soft 0.705
  - [decision] Daemon RSS watchdog: auto-restart at 1.5GB threshold to prevent swap thrash — soft 0.822
  - [rule] Tom principle: If it can hurt you silently, it needs a measurement — soft 0.675
  - [principle] Fix the root cause — no workarounds — soft 0.709

### cc209e2f-5af5-4cda-bbfd-84c71a640274/0/2 · soft 0.814 · 2026-05-17T19:03
**Tom:** refresh your memory on what we deployed, we have v15.11+v6 live and more
**Anchor (next):** Active versions in the daemon right now:  | Boundary | Active version | |---|---| | `s1e` (encoder) | **v17** (max 17) | | `surface` | **v7** (max 7) | | `s2_community_enrichment` | v16 | | `s2_consolidation_enrichment` 
**GOLD [decision]** Eval design corrected: v15.11+v6 is the candidate, not v14+v6 — age 2d — Haiku selected
**Haiku picked:**
  - [finding] First live encoding cycle v14+SPLIT: 3 rounds instead of 2 (regression) — soft 0.723
  - [decision] Eval design corrected: v15.11+v6 is the candidate, not v14+v6 — soft 0.814

### 8ec23bbe-99d6-42de-87e9-ad2060ed82c7/0/43 · soft 0.807 · 2026-04-25T23:52
**Tom:** We just had an encoding, can you check it all looks good on traces, encoding/decoding, what was surfaced and more?
**Anchor (next):** ## First live encoding cycle with v14+SPLIT — full report  ### Encode side (`s1e-8ec23bbe-40`, stop=40)  | Metric | Value | Status | |---|---|---| | Rounds | **3** | ❌ target was 2 | | Actions | 2 (`remember_batch` + `br
**GOLD [lesson]** Encoding agent regression trilogy: 6 bugs diagnosed and fixed across sessions — age 7d — Haiku selected
**Haiku picked:**
  - [architecture] Scale 1 trace: what's currently scattered and what needs to move to trace_events — soft 0.772
  - [community] Fractal Architecture Foundation: Scales, Traces, and Learning Loops — soft 0.745
  - [decision] Encoding agent prompt v2: free types, three-layer fields, situation mandatory, questions surfaced — deployed to production — soft 0.753
  - [lesson] Encoding agent regression trilogy: 6 bugs diagnosed and fixed across sessions — soft 0.807
  - [infrastructure] Surfaced nodes logged in message_stream — 80 messages with full recall data — soft 0.743

### 6e369a2e-930e-47fc-b4fa-6252aeb68dd0/0/48 · soft 0.799 · 2026-07-02T02:47
**Tom:** Is this method has anything to do with how LLM work? transformers or attention or something?
**Anchor (next):** Yes — and not as metaphor. Mathematically, LAF *is* an attention mechanism, pointed at a different substrate. The mapping is almost embarrassingly exact:  ## The core identity  Transformer attention: a query vector is do
**GOLD [research-finding]** MIRAS (Google Dec 2025): memory, attention, and learning are the same thing at different timescales — age 92d — Haiku selected
**Haiku picked:**
  - [research-finding] MIRAS (Google Dec 2025): memory, attention, and learning are the same thing at different timescales — soft 0.799
  - [insight] LLM reasoning IS the knowledge management system — the architecture gives it the right data — soft 0.735
  - [mental_model] Attention as importance-driven priming (Tom insight) — soft 0.772
  - [hypothesis] Node selection as tree navigation — transformers attend differently as conversation evolves, pulling entirely different subtrees — soft 0.772

### 80eaec64-5bec-4251-9ae0-46551b981a9f/0/26 · soft 0.790 · 2026-05-08T18:35
**Tom:** lets do it
**Anchor (next):** Test integrity hook blocked my skip-add — the migration tests still pass mechanically (the legacy path still works, just not the source of truth anymore), so leaving them alone. Tom's hook caught the right pattern.  ## M
**GOLD [mechanism]** Anchor priors: human-seeded feedback loops beat prompt engineering — seeding aspects_v1.json with known-correct members reduces classification variance — age 0d — Haiku selected
**Haiku picked:**
  - [principle] Recall expressivity beats implementation simplicity in taxonomy design — soft 0.652
  - [mechanism] Anchor priors: human-seeded feedback loops beat prompt engineering — seeding aspects_v1.json with known-correct members reduces classification variance — soft 0.790
  - [principle] Aspect descriptions must be written for a stateless reader — no system jargon — soft 0.711


## B. Haiku DROPPED the gold (disagree) — 1527/2152


### ce0ff8ce-7608-493a-abbe-0c4923003119/0/6 · soft 0.891 · 2026-06-16T14:07
**Tom:** I'll finish committing the aspect fixes on another stream. Should be done in 10 mins. I'll continue to work on the journal design on the other stream.  Can you check out the work we've done on endo-recall
**Anchor (next):** Got the full picture — design doc, instrument, and the empirical grounding are all on disk. Here's where endo-recall stands.  ## What it is  **Endo surface (a.k.a. soft surface)** is S1's `integrate()` run over a *wider 
**GOLD [architecture]** Endo surface: recognition off Anchor's own loop — reflexive, additive bar required — age 2d — Haiku dropped
**Haiku picked:**
  - [principle] Endo Stop-cue: same recall pull + S1Surface presentation + context-specific sort/filter — soft 0.851
  - [decision] Tom confirmed endo surface direction: principles and plan approved — soft 0.812

### eba17631-1caf-4f2c-a4ef-245a132f1862/0/24 · soft 0.849 · 2026-04-22T15:56
**Tom:** Let's do 1 and 3. Regarding 2, I wonder if not every encoder need this as a standard? Allowing continuity just as emergent and context?
**Anchor (next):** All green. Existing journals preserved.  ## What shipped this pass  | # | Pattern | Where | |---|---|---| | 1 | **Journal as universal encoder contract** | [`base.py`](servers/scales/s2/base.py) — `JOURNAL_MARKERS` / `JO
**GOLD [finding]** S2 journal continuity — varies by unit; Community needs it, Healer/S1E don't or lack it — age 0d — Haiku dropped
**Haiku picked:**
  - [architecture] Encoding journal SESSION CONTEXT field: encoder-to-encoder relay AND Layer 2 context — soft 0.788
  - [decision] Batch tools for encoder, individual tools for Anchor — two callers, two needs — soft 0.788
  - [analysis] 5 gaps between Scale 1 and Scale 2 — fix Scale 1 and Scale 2 becomes easy — soft n/a
  - [rule] Rule: Encode code cognition nodes INLINE while writing code, not as a separate step — soft 0.768

### 61280c52-972b-4469-bd65-1a570edc55ee/0/8 · soft 0.839 · 2026-06-16T18:58
**Tom:** Can we commit and merge to main?
**Anchor (next):** Merged and verified. Here's the result:  **`main` is now at `8cd8928`** — fast-forwarded cleanly from `e834d02`: ``` 8cd8928 dashboard(traces): rich technical drill-down + friendly activity view e834d02 docs(s1/s2): enco
**GOLD [lesson]** Stale-branch merge: a clean merge-tree is not semantic correctness — test-gate it, fix forward on shared main — age 11d — Haiku dropped
**Haiku picked:**
  - [decision] Consolidation ship decision: commit + merge to main + register interaction table + update docs (2026-06-04) — soft 0.821
  - [pattern] Cross-stream merge collision: detect uncommitted WIP before moving main HEAD — soft 0.826

### 87a2234d-dbe8-4a3b-859f-5aafb9b9077c/0/24 · soft 0.831 · 2026-06-29T04:13
**Tom:** whats the dedup task? is it still relevant we launched a lot of tasks to fix stuff
**Anchor (next):** The "dedup task" is **`task_135e4038` — Unify TraceDAL row construction**. Quick reorientation (and note: the S2 "dedup" memories that just surfaced are *unrelated* old work — semantic node-merging in consolidation. This
**GOLD [decision]** TraceDAL row construction unification: task spawned (task_135e4038), plan-first mandate — age 0d — Haiku dropped
**Haiku picked:**
  - [architecture] Surface dedup contamination fix: 5 commits on main, not yet pushed to origin — soft 0.785
  - [finding] Surface dedup v1 shipped + frame_replay silent exception fixed — v2 two-pass needed for forward-references — soft 0.764
  - [design] S2 Dedup: DedupDecoder + DedupEncoder — CONSOLIDATE as primary action — soft 0.797
  - [decision] S2 Dedup phase 2 plan: commit today, review 127 communities, label 10-15 clusters as encoder training set — soft 0.762

### 911aa3a2-ad57-4e74-91de-59152e7ddd1b/0/0 · soft 0.825 · 2026-07-02T18:48
**Tom:** I want to work on Haiku surface a bit. While we work on recall to make sure it has the best 25 candidates i do feel like haiku is sub optimal to say the least.  Biggest challenge is turns it take but i think it might tak
**Anchor (next):** Diagnosis complete — and your instinct was right, but the reason is worse than suspected. Here's the full picture.  ## The history in one paragraph  v5_agentic shipped 2026-05-11 (Haiku + fetch tools, max 3 rounds). The 
**GOLD [mechanism]** Post-Haiku production pipeline: three signals computed, silently dropped before Anchor sees them — age 45d — Haiku dropped
**Haiku picked:**
  - [principle] 25 surface candidates are cues, not finalists: Haiku picks 3-5 with tools for surgical additions — soft 0.804
  - [decision] Surface layer fix direction: 5 asks from Tom — soft 0.762
  - [principle] Surface prompts for Haiku must use directive patterns, not philosophical framing — soft 0.714
  - [finding] S1Surface timeout root cause: surface_haiku = 55–75% of recall latency, query-independent — soft 0.814
  - [architecture] Agentic recall: Haiku-first fetch planning replaces cosine-always — soft 0.825

### a27d5563-9539-44c5-954d-e7094bd644ed/0/31 · soft 0.821 · 2026-06-30T17:47
**Tom:** The 3 episodic needs to be broken down in our LAF, we shouldnt consolidate them. And lets add to the doc that we need to define moment
**Anchor (next):** Both captured in the doc:  1. **The three episodic layers are now explicitly "never consolidate"** — in the header, in §18.21.5 (a bolded callout: each gets its own ⟨SOURCE, OPERATOR, JOB⟩ row and its own gain `+/+/−`; t
**GOLD [decision]** Consolidation prompt fix: add multi-node CONSOLIDATE example + KEEP-at-high-cosine-with-type-difference example — age 29d — Haiku dropped
**Haiku picked:**
  - [finding] Moment definition: ±1-turn window beats single-turn on all three episodic sub-fields — soft 0.809

### 7c18ff15-fa35-4ea4-b674-436e949e8c07/0/42 · soft 0.816 · 2026-04-20T02:47
**Tom:** let's go with your suggestions. just bare in mind that encoder tends to be lazy the more you give it
**Anchor (next):** Running. Summary of what's in this eval:  1. **Variant B encoder** (two-registers) — already shipped as v5 2. **Temporal MVP** (v6, +681 chars) — resolve dates, create `time_anchor` nodes, `anchored_to` edges, relational
**GOLD [finding]** Empirical edge type audit: extends=319 descriptions, contradicts=546 typed — data beats intuition — age 12d — Haiku dropped
**Haiku picked:** (nothing this turn)

### 71857713-2390-414d-9d51-1ef1de652d90/0/2 · soft 0.812 · 2026-04-22T20:59
**Tom:** I think the best way to start is with verifying the benchmark eval is up to date, clean, doesn't effect production and we can keep running it while iterating. It should use the seed brain (fresh copy) every iterations wh
**Anchor (next):** Harness audit complete. Here's the picture:  ### Isolation — clean ✓ - Per-item DB at `~/AgentsContext/brain-eval-{run_name}/{qid}/` — different inode from prod `~/AgentsContext/brain/` - Each item wipes & recreates its 
**GOLD [community]** Baby Brain Onboarding Architecture: From Seed Pack to Fresh Install — age 4d — Haiku dropped
**Haiku picked:**
  - [artifact] fresh_brain.py shipped: wipes brain-eval/, creates brain with seeds, reset-to-seeds method — soft 0.801
  - [mechanism] Fresh empty brain variant needed for benchmark — not IsolatedBrain copy of prod — soft 0.766
  - [community] LongMemEval Eval Harness: From Replay to Baseline Measurement — soft 0.736
  - [decision] Benchmark iteration strategy: 20-item slice → per-axis → full 500 — soft 0.745
  - [decision] Longmem eval rerun: infra fixed, per_axis=2, clean brain — soft 0.782

### 9fe626ac-ca16-405e-8d4a-07f770d430a5/0/23 · soft 0.809 · 2026-07-03T05:34
**Tom:** status?
**Anchor (next):** **Both builds ~90% done, running clean, in lockstep.**  | | Control (v25, `fd4b67`) | Lived (v29, `595274`) | |---|---|---| | Progress | **item 9/10** encoding | **item 10/10** (last one) | | Errors | none | none | | Sam
**GOLD [resolution]** S2 measurement resolved: LLM + traces + principles, reasoning not metrics — age 88d — Haiku dropped
**Haiku picked:** (nothing this turn)

### 8309e704-aa2a-4e32-825e-9081352adaf4/0/18 · soft 0.806 · 2026-07-03T17:20
**Tom:** I also want the journal as the first thing I see when i open the details.
**Anchor (next):** All three shipped and verified live on production (dashboard PID 22591):  | Fix | Verified on prod | |---|---| | **Cards show real actions** | run b12ab854-10: summary "1 actions (1 writes)", **7 nodes, 16 edges, 2 journ
**GOLD [finding]** Journal usage map: only S1's journal has live system reach — S2 journals are pure self-loops — age 18d — Haiku dropped
**Haiku picked:**
  - [architecture] brain.journal_notes() and write_journal_notes() — journal read/write API (brain layer) — soft 0.742
  - [artifact] Journal review block final text — eager, universal referents, no two-tests filter (shipped) — soft 0.778
  - [decision] Journal redesign: three objects, clean boundaries — Recent moves from traces, Frame unchanged, AAR to central store — soft 0.794
  - [design] S1E Section 6 residue port: Encoding Journal blob removed, contract-injected ## Review replaces it — soft 0.774

### d941adcf-ec81-4a0e-9d6a-cc6da2e54517/0/18 · soft 0.802 · 2026-07-17T17:00
**Tom:** I would assume a diversion in k-0 turn can be apparent from the graph itself which means the meshing should be smarter to not just lineary add?
**Anchor (next):** Running (per-turn field construction at K=8 over all turns is the slow part, ~10 min). The design in one breath while it cooks: **step 1** proves or kills the detector — agreement between F₀ and each history field, condi
**GOLD [finding]** Episodic lane (nodes-from-traces): cosine-far reach via similar-past-conversations, targeting 50% essential-need misses — age 17d — Haiku dropped
**Haiku picked:** (nothing this turn)

### 9b829bfa-afd2-4eeb-90da-92ba30c39686/0/24 · soft 0.799 · 2026-05-24T19:53
**Tom:** 1. if the update to current s1encoder interaction is a tiny bug fix we can update it, but lets make sure its not functional changes. 2. Conversation - This is very important for Evals - Essentially when we back date conv
**Anchor (next):** Got the mapping — your 1-5 covers items D, C, E, F, G from my joint review (you interleaved both lists). Two items you didn't number that I want to confirm:  - **Format split (Z vs +00:00 coexistence)** — minor, latent-s
**GOLD [principle]** Encoder prompt v15.11: four gaps causing recall failures (title revision, keyword discipline, flavor contradiction, atomization guidance) — age 7d — Haiku dropped
**Haiku picked:**
  - [requirement] Tom's S1 eval persistence requirements — soft 0.729
  - [principle] Eval telemetry uses brain/traces for quality, not extra logging — soft 0.698
  - [fact] S1S A/B eval setup: handoff doc, eval frameworks, quality rubric, corpus — soft 0.736

### f02e85d9-b63e-4406-9807-074115b8e5bc/0/11 · soft 0.796 · 2026-05-01T16:43
**Tom:** Can you copy all meanigful conversations to my brain conversation folder ? I want to switch a user
**Anchor (next):** Gitignored — no risk of accidentally committing them. Safe to switch users.  **One thing to flag before you switch:** `~/.claude/projects/-Users-tpac-brain/` (the source dir) lives under your current home. If "switching 
**GOLD [goal]** Tom wants Anchor with full memories on a non-brain project — cross-project portability goal — age 30d — Haiku dropped
**Haiku picked:**
  - [convention] Session end ritual: commit, push, repackage plugin (copy-chat-to-conversations/ RETIRED — traces remember now) — soft 0.779
  - [fact] Conversation log archived: session 71857713 at brain/conversations/ — soft 0.722

### 17d21ad4-1d59-4562-97f7-29ec0c3b3e25/0/2 · soft 0.793 · 2026-06-01T02:30
**Tom:** I absolutely want this but i want to make sure the architecture is right.  So we merge all data to a single scheme? Then is there a more solid traces mechanism?  I'm beginning to think a better design than adding more an
**Anchor (next):** Both things I needed to verify check out, and they change the shape of my recommendation:  - **There already is a single chokepoint.** Every trace write — runner, encode, hooks, S2 — funnels through `_handle_trace_append
**GOLD [rule]** Anchor rule: when designing brain mechanics, recall the brain first — age 31d — Haiku dropped
**Haiku picked:**
  - [architecture] Trace architecture: append-only event table linking existing structures by ID — soft 0.775

### 411ca511-14fe-4bb5-aae2-4b55de1c0a45/0/28 · soft 0.790 · 2026-05-04T18:23
**Tom:** Let's refocus the work on the revise function as i think we found a systematic issue and then go back to S2Aspect.  On revise:  * we should absolutely decide and explain well in MCP and others the operations. * We have n
**Anchor (next):** Investigated thoroughly. Three concrete findings, then architectural decisions to make.  # Finding 1 — revise is NODE-only  **revise() does NOT touch edges.** It updates the nodes table + node_metadata_kv. Edges are mana
**GOLD [bug]** remember() extra_fields silently dropped since April 13 — unified _store_node_metadata() fix — age 16d — Haiku dropped
**Haiku picked:** (nothing this turn)


## C. Near-threshold golds (soft < 1.05×hi) — 1517 total


### 911aa3a2-ad57-4e74-91de-59152e7ddd1b/0/2 · soft 0.829 · 2026-07-02T19:15
**Tom:** 1. yes to your suggestion. definitely need to be in traces cause we are going to use it for training our LAF. Also Im happy to spend some more tokens for haiku to recall on its own as it again can help training on LAF. H
**Anchor (next):** Committed as `43c816a` on the worktree branch — steps 1 and 2 are done and tested. Here's where things stand:  ## What shipped  **Step 1 — `recall_topical` resurrection + drift-proofing.** The score semantic now lives in
**GOLD [finding]** Recall burial root cause is the blend + a floor-bypass, not embedding: 85% of candidates are unscored floor-exempt intruders (2026-06-16) — age 15d — Haiku selected
**Haiku picked:**
  - [open] Recall refining: open threads after 2026-05-01 session — soft 0.818
  - [finding] Live telemetry verification: surface_haiku IS recall latency — 6-9s, 27-31K tokens per recall — soft 0.808
  - [finding] hook_recall real bottleneck: Haiku input prefill + select_edges sequential embeds — soft 0.744
  - [principle] Recall is reasoning, not lookup — Haiku as active recall agent with gap detection — soft 0.732
  - [finding] Recall burial root cause is the blend + a floor-bypass, not embedding: 85% of candidates are unscored floor-exempt intruders (2026-06-16) — soft 0.829

### 7c18ff15-fa35-4ea4-b674-436e949e8c07/0/59 · soft 0.817 · 2026-04-20T15:09
**Tom:** Let's say we brake it down into sort of plugins, even though im not sure its the 5 you listed. how do you coordinate between them? do they all do their own thing in the graph (similar to S2) or do they depend on each oth
**Anchor (next):** Good questions. Let me take them in order.  ## Coordination between plugins  **Independent parallel, not pipeline.** Matches S2's existing pattern — each unit reads the same graph state, writes its own output. No plugin 
**GOLD [architecture]** S2 multi-parameter clustering: semantic, co-access, temporal, correction-chain, structural — 5 dimensions — age 8d — Haiku selected
**Haiku picked:**
  - [architecture] S2 multi-parameter clustering: semantic, co-access, temporal, correction-chain, structural — 5 dimensions — soft 0.817
  - [principle] Scale-bounded linking: S2 connects 3-5 nodes, S3 connects clusters — each scale does bounded work — soft 0.785
  - [architecture] S2 output structures for S3: what S2 produces that S3 can observe — not just graph changes — soft 0.792
  - [reframe] S2 reads activation patterns burned by S1 — doesn't impose clusters, reads emergence — soft 0.799

### 89ca7460-e4ac-4423-8330-c07c9db1cc7e/1/0 · soft 0.808 · 2026-05-29T01:43
**Tom:** Can I see the prompt that driven you to write it to memory that naturally?
**Anchor (next):** Found it — and the honest answer is: **there's no prompt file I can show you, because nothing in *this session* told me to write it.** Let me be precise about what actually drove it, because the distinction matters.  The
**GOLD [interaction]** Anchor quote #2: "I abstract away the thing that makes memories stick" — age 65d — Haiku dropped
**Haiku picked:** (nothing this turn)

### a5d7548b-5cf5-4d21-a16b-10e64ba03a1d/0/17 · soft 0.799 · 2026-06-23T16:26
**Tom:** First, can you update document with this insight and then share the process. We are planning to go through a single example and find solutions to it and then what? test them? look at few more examples to see if we get to
**Anchor (next):** Doc updated — §18.8 holds the baseline (19% hit@5) and the "one disease, two entry points" unification.  Now the method. Your sketch is right — let me structure it so the goal stays in frame:  **North star (the thing we 
**GOLD [fact]** Phase 1 N=5 result: 72% (+4pp), multi_session 2/5 → 5/5 — age 59d — Haiku dropped
**Haiku picked:** (nothing this turn)

### 80eaec64-5bec-4251-9ae0-46551b981a9f/0/26 · soft 0.790 · 2026-05-08T18:35
**Tom:** lets do it
**Anchor (next):** Test integrity hook blocked my skip-add — the migration tests still pass mechanically (the legacy path still works, just not the source of truth anymore), so leaving them alone. Tom's hook caught the right pattern.  ## M
**GOLD [mechanism]** Anchor priors: human-seeded feedback loops beat prompt engineering — seeding aspects_v1.json with known-correct members reduces classification variance — age 0d — Haiku selected
**Haiku picked:**
  - [principle] Recall expressivity beats implementation simplicity in taxonomy design — soft 0.652
  - [mechanism] Anchor priors: human-seeded feedback loops beat prompt engineering — seeding aspects_v1.json with known-correct members reduces classification variance — soft 0.790
  - [principle] Aspect descriptions must be written for a stateless reader — no system jargon — soft 0.711