# Recall — Dual-Store Seed-and-Spread (Design)

> **Status:** design, converged 2026-06-06 (Tom + Anchor). NOT built. Execution-ready.
> **Reading order:** §0 spine → §1 the reframe → §3 architecture → §4 gates → §5 build order.
> **Supersedes nothing; complements:** `RECALL-OVERVIEW.md` (current pipeline = the "Current method"
> baseline), `RECALL-BURIAL-HANDOFF.md` (the verified burial diagnosis + the parallel semantic-lane
> workstream, §6 here), `EPISODIC-REFERENCES.md` (the source_ref substrate this consumes).
> **Discipline:** every load-bearing claim tagged **[VERIFIED]** / **[SUGGESTED]** / **[HYPOTHESIS]**.
> Built substrate is cited by commit; recall consumers are unbuilt.

---

## 0. The spine, in one paragraph

The brain has **two stores** — semantic (the node graph: *what is X*) and episodic (the S0 trace
record: *what did we do, when*). Today recall searches only the semantic store, which is why
episodic queries like #11 (*"what did we do last session on ex.co"*) are buried. The architecture:
**run an episodic lane in parallel** (cosine over `trace_embeddings`) so a confidently-matched trace
becomes a **seed**; **spread from that seed across the store boundary** into the node graph (via
`source_refs`/`co_anchored` edges where they exist, and via the trace's own vector as a re-query
where they don't); the spread **rescues buried nodes** the direct query cosine missed; package the
result as **bundles** (a node-or-trace plus its corroborating cross-store partners) and hand them to
**Haiku-the-defender**, which recognizes against the Frame as it does today. Agentic tool-orchestration
is the **override** for the rare query the deterministic lanes can't serve — not the default. This is
hippocampal indexing / CLS implemented: episodic fast store + semantic slow store + `source_refs` as
the index/replay bridge.

---

## 1. The reframe — two axes, not three rival concepts

Three "concepts" were on the table; they're points on **two orthogonal axes**, not mutually-exclusive
designs.

**Axis A — granularity + who drives** (the three concepts live here):

| | unit | who drives | who defends | cost/turn | precision | reach |
|---|---|---|---|---|---|---|
| **Current** | whole node | algo | Haiku (selector) | low | node-level (all-or-nothing) | bounded by what algo puts in the 25 |
| **Particles** | field/sentence/fragment | algo | Haiku (selector) | low–med | sub-node (the gold sentence) | higher (small units → more fit) |
| **Orchestrator** | tool-results | **Haiku** | Haiku (itself) | high (N rounds) | tool-dependent | maximal |

**Axis B — which store** (the thing none of the three named, and the thing `source_refs` was built for):
semantic only / episodic only / **both, bridged by `source_refs`**.

**Resolution:**
- Axis A → stay on the **Current↔Particles** side; Haiku is the **defender**, not the driver.
  Orchestrator is the *override*, gated. *Rationale (Tom):* agentic orchestration is a crutch for weak
  retrieval — "if the algo were better, Haiku could not do turns." Recognition (System 1, every turn,
  cheap) is the default; deliberate tool-search (System 2, costly, rare) is the exception.
- Axis B → **both stores**. This is non-negotiable per the research (§2) and the empirical win (§3.2).

---

## 2. Neuroscience grounding (why this is the correct architecture, not a heuristic)

- **Complementary Learning Systems** (`0bb0fa97`): fast hippocampal episodic store + slow neocortical
  semantic store; the hippocampus *replays into* the cortex. "They aren't optional — they need each
  other." → S0 traces = hippocampal layer; node graph = neocortical layer; `source_refs` = replay bridge.
- **Hippocampal Indexing Theory** (`ec2b4c1c`, `f152c597`): the hippocampus stores *indices, not
  content* — pointers binding distributed fragments; a partial cue reactivates the index → pattern
  completion reconstitutes the content. "Our nodes try to be both index AND content — a 500-char field
  is neither." → `source_refs` is the index; **seed-and-spread is pattern completion**; **particles
  (Axis-A) are the distributed fragments**.
- **3-D coordinate model** (`07f61bc6`): episodic/semantic is not a binary — temporal-specificity,
  content-format, self-reference are independent gradients. → a node has coordinates; the same recall
  machinery serves both ends of each gradient; store choice is per-query, not per-node-type.

The architecture below was discovered in biology ~2000 and fMRI-validated ~2021. We're implementing a
25-year-old theory, not inventing one.

**Why the episodic lane "leaks" into semantic queries — and why that's correct.** A cue *about* a topic
reactivates that topic's index and pattern-completes **both** stores (`ec2b4c1c`); a *temporal* framing
makes the episodic component dominant, a *definitional* framing makes the semantic dominant — same index,
different magnitude (the temporal-specificity gradient, `07f61bc6`). So the episodic lane partially
covering semantic queries is the stores cross-activating as CLS requires, not a leak. **The one
deliberate divergence from biology:** biology tolerates lossy/probabilistic recall; we want
*deterministic* coverage — so we do NOT rely on episodic spillover to cover the semantic burial, we still
fix the semantic lane (§6). Same stance as `2101f567` ("biology can't control graduation; our system can
via recall gating") — engineer determinism where biology leaves it to chance.

---

## 3. Architecture — components

> **⚠ READ THIS LINE FIRST — retrieval vs presentation (Tier-1 result, 2026-06-06).** The burial fix is
> **RETRIEVAL ONLY**: the **trace-chain (§3.2 + §3.3 form 1, semantic) + reserved-tail merge (§4)**.
> That alone took #11 from 0→8 EX.CO *in the pool* — proven, no other step required. Everything below
> tagged **[PRESENTATION]** — graph-edge spread, bundle assembly, activation-weighted render — is the
> SEPARATE **"smart surfacing"** problem (`e3c8f671`): it improves how the defender *attends to* what the
> chain already delivered, applies to ALL recall (not the EX.CO fix), and is **gated on Tier-2 showing the
> defender actually needs help.** Do NOT bundle presentation into the burial fix — that's the over-scope
> this banner exists to prevent. Build the small retrieval fix; treat smart surfacing as its own project.

### 3.1 Two stores (both already exist)

- **Semantic** — `nodes` + embeddings + edges. Searched today. **[VERIFIED — live]**
- **Episodic** — `trace_embeddings` (brain_logs.db): trace_id, vector, text, recency. **25,915 embedded
  (30.6%), s0-only, 82% `tool_result`; embedding depth ~2 weeks** (embed worker:
  `EAGER_TRACE_SCALES=('s0',)`, 30-day window, runs forward). But the **raw `trace_events` run from
  2026-04-05 and are never pruned** (S0 retention guarantee) — ~3,500 s0 dialogue traces sit un-embedded.
  **Built, populated, NO recall consumer.** **[VERIFIED — Exp C + trace census]** (`b8b8370b`)

### 3.2 The episodic lane (BUILD FIRST)

Additive candidate source, parallel to the semantic channels. Cosine over `trace_embeddings` for the
query; matched traces enter candidate assembly.

- **Why first:** highest leverage, lowest risk. Additive (a new candidate source, not a re-rank) so it
  **cannot scramble the brain-dev controls** the way RRF/z-score did. Categorically unlocks a query
  *class* instead of incrementally tuning one. **[VERIFIED it answers #11 — `episodic_probe.py`: rank 1–2.
  CAVEAT (Exp A): the win is a *recent self-echo* (query echoed back + a re-stated answer), NOT the
  original episode (0 April traces embedded). Real but fragile — see §4 hygiene + §8.6 horizon.]**
- **Trace render format (NEW, required):** raw conversation is long and noisy; Haiku's selection prompt
  is tuned for node-shaped candidates. Render the **matched span** (not the whole turn), **labeled as
  episodic evidence** ("said 2026-04-21" vs a concluded principle) — analog of the existing verbatim
  `SURFACE_FACT_FORMAT`. Filter this-session meta-traces (our own recall calls rank high — measured
  caveat from `episodic_probe.py`).

### 3.3 The trace→node hop — TWO forms, and they are NOT equals

"Seed-and-spread" sounded like one mechanism; the Tier-1 result split it into a retrieval fix and a
presentation option. The trace→node hop has two forms — pick by what they cost and what they're proven to do:

1. **Semantic chain — [THE FIX, retrieval].** Use the matched trace's vector as a second query into the
   node store (query→trace→node, all cosine). Coverage-free, no edges. This is **literally what Tier-1
   ran** and it took #11 from 0→8 EX.CO in the pool. The stored doc-vector beats re-embed (§8.4). This is
   the burial fix — a *second cosine retrieval*, NOT a graph operation. **[VERIFIED — `dual_store_merge_probe.py`]**
2. **Structural chain / graph-edge spread — [PRESENTATION, separate, unproven-need].** Walk
   `source_refs`/`co_anchored` edges from the trace into the graph and spread (the `30dbe1c8` mutual-
   traversal generalized across the store boundary). Its *unique* value is reaching nodes that are
   structurally central but semantically dissimilar — which **no cosine path finds**. BUT: it's
   **coverage-dead pre-v22** (`episodic_probe.py`: 0/15 top #11 traces had node links; ~13% coverage), the
   generic post-selection spread **already exists** (pipeline step 10, on Haiku's picks), and the cosine
   chain already got 8/12 *without* it — so its marginal value here is **unproven**. A forward mechanism
   that strengthens as the Source-Ref Healer (§3.7) fills coverage; not part of the fix today.

→ **The fix is form 1.** Form 2 is an optional later enhancement, gated on Tier-2 showing the cosine chain
left structurally-relevant nodes unreached. Don't build form 2 to fix the burial — form 1 already did.

**On `graph_rerank_probe.py`** (EX.CO intra-degree 2.0 vs 0.4; structure-rerank 37→1): that motivated form
2, but the merge probe showed the *cosine* chain reaches those nodes directly — so the structure signal is
a presentation nicety, not the retrieval lever it looked like.

### 3.4 The merge (retrieval) vs bundles (presentation)

**Merge — [THE FIX, retrieval].** The parallel lanes (direct semantic + FTS5 + trace-chain) combine into
one pool: dedup by node-id, trace-chain rescues go to the **reserved tail** (§4), fill-on-demand. This is
part of the burial fix and is what `dual_store_merge_probe.py` exercised.

**Bundles — [PRESENTATION, selection-aid, Tier-2-gated].** Optionally group a center (node or trace) with
its cross-store partners and present as a unit (trace-led for episodic, node-led for semantic; "cluster
not node", `RECALL-OVERVIEW` principle 3). Its *value* is at the **selection** layer — if a rescued node
lands at pool-rank 24, does Haiku attend to it, or does bundling it with the answer-trace make it
pickable? That's a Tier-2 question (`frame_replay`). It does **not** get more nodes into the pool — the
merge already did — so it is NOT part of the retrieval fix. Build only if Tier-2 shows the defender
misses rescued nodes presented flat.

### 3.5 Haiku as defender; agentic as override

- **Defender (default):** Haiku selects 0–8 candidates from the merged pool, recognizing against the Frame
  — unchanged role from today's pipeline (step 9). The better the lanes, the less Haiku must reach.
  (Candidates are bare nodes/traces by default; bundling them is the optional §3.4 presentation step.)
- **Override (gated, rare):** the Orchestrator (v5_agentic, tools) is reserved for queries the
  deterministic lanes flag as un-served. Every turn agentic is a tax; don't pay it by default.
  Open question on trigger — see §8.

### 3.6 Particles — [PRESENTATION, separate project]

Not part of the burial fix. Move the **select/render unit** from whole-node to fragment (field/sentence). The *scoring* machinery
already operates at sub-node granularity — z-weighted vector groups (title/question/_primary/high_meta)
are separately-embedded fragments; spread activation flows on `cosine(query, edge_text)` over fragments.
**[VERIFIED scoring is fragment-level — live]**. What's unbuilt is rendering/selecting at fragment
granularity (today: whole node, char-budget truncated). This is pattern completion's "distributed
fragments" (§2). After the episodic lane, because it's a bigger build with a real risk (a fragment
without node-context can mislead) and the episodic lane is the higher-leverage win.

### 3.7 Source-Ref Healer (S2 unit — feeds the structural chain backward)

The structural chain (§3.3 form 1) is forward-only today: pre-v22 nodes have no `source_refs`, so the
EX.CO cluster can't be reached structurally (`episodic_probe.py`: 0/15 #11 traces had node links). A
**Source-Ref Healer** backfills it: over idle S2 cycles, for a node lacking `source_refs`, find candidate
traces by **timestamp proximity** (node `created_at` ≈ trace `created_at`) **+ content similarity** (node
vector vs trace vector) and, above a conservative confidence bar, write the ref. This turns the
structural chain from forward-only into **self-healing backward** — the historical EX.CO cluster
*acquires* refs and the structural chain eventually lights up for #11. Slots next to the existing S2
Healer; gated/suppressed like every S2 unit (see CLAUDE.md S2 section).

**Caveats (a wrong ref is a false provenance claim that pollutes the index):**
- Mark inferred refs distinctly (`encoding_source=s2:sourceref_healer`); the structural chain weights
  inferred refs **lower** than encoder-authored ones.
- Conservative threshold; backfill the high-confidence matches only, leave the rest unlinked (the
  semantic chain still covers them).
- It's a recall problem feeding a recall mechanism — reuse the trace-as-vector machinery for the matching.

---

## 4. The gates — rescue the *right* buried nodes, not junk

"Increase nodes that got low ranking" is the goal **and** the failure mode. Indiscriminate rescue is
exactly what scrambled the brain-dev controls before (z-score, RRF). The spread must be precision-gated:

- **Seed earns its rank from the query, not from a node.** Resolves the circularity concern: if the
  trace is pulled by the *query* independently, a node reachable from it gets a *legitimate* boost ("the
  query matched the conversation this node came from"). The only circular case — node corroborating its
  *own* source trace where the trace wasn't query-matched — is excluded by construction. **node ↔ its own
  source trace = presentation value (show the receipt), NOT a ranking boost.**
- **One mechanism per similarity (failure point 1, confirmed).** The semantic chain pulls node N
  *because* N is similar to trace T — a "N and T converge" bonus would count that one similarity twice.
  Rule: a node pulled by the **query** that *also* matches a **query-pulled** trace = independent
  convergence = bonus. A node pulled by the **trace-chain** = rescue; its score is
  `trace_cos × hop_decay × node↔trace_cos`, **no** convergence bonus on top.
- **Rescue is additive to a reserved tail, never a reorder (failure point 2, confirmed — the
  shippability gate).** Rescued nodes occupy **reserved tail slots** (like the existing `fts5_only`
  lane); they never enter the top of the semantic ranking. Additive survives controls; global re-rank
  scrambles them. **[NON-NEGOTIABLE]**
- **NO query-intent classifier — control-safety is hygiene + reserved-tail + defender, eval-arbitrated
  (Tom, 2026-06-06).** The recall is just cosine + FTS5 over *both* stores → traverse/spread → surface the
  activation smartly (Haiku). We do NOT pre-classify "is this an episodic query" — that reintroduces the
  discriminator the *recognition-over-retrieval* principle exists to kill; the activation pattern encodes
  relevance and the defender reads it. Defender-distraction (failure point 5) was mostly the *poison*
  (recall-echo `tool_result`s); hygiene (next bullet) removes it, so the episodic candidates left on a
  control query are on-topic dialogue — fine to surface. **Arbiter:** if the always-on spread passes the
  control eval (top-5 unmoved at selection level), query-intent is *proven* unnecessary; add gating ONLY
  if the eval fails, never speculatively. Keep the ≤3–5 episodic-slot cap as cheap insurance.
- **⚠ Match-strength is NOT a valve (FALSIFIED 2026-06-06 — `dual_store_validation_probe.py` Exp B).**
  Top-trace cosine does NOT separate episodic from control queries: EX.CO top-1 ∈ [0.787, 0.852], controls
  ∈ [0.721, **0.889**] — control #9 outscores every EX.CO query. This killed the "calibrate a confidence
  threshold" idea. The right conclusion is **no valve** (above), not a different valve. → the weight moves
  to **"surface activation smartly to awareness"** (§3.4 bundles + activation-weighted render) — the real
  unsolved lever.
- **⚠ Trace hygiene is mandatory (CONFIRMED 2026-06-06 — Exp A + C).** The episodic store is **82%
  `tool_result`** and shallow (effective depth ~2 weeks; embeddings only exist from 2026-05-24; **0
  April/EX.CO-era traces embedded**). For EX.CO queries the top hits are **recent self-echoes** (the query
  echoed back; a prior answer re-stated) and **this-session recall-call `tool_result`s** (query #2's entire
  top-3 was our own `mcp__...recall` output). Before the lane is usable: (1) drop recall-machinery
  `tool_result` traces; (2) drop *this-session* traces at recall time; (3) likely restrict the lane to
  `user_message`+`assistant_message`. The b8b8370b "rank 1-2 win" was real but a *recent echo*, not the
  original episode — fragile, depends on having recently re-discussed the topic.
- **Dedup + no cross-store loops.** A node pulled by both the semantic lane and the trace-chain must
  **merge to one candidate** (max-provenance, not two rows). The visited-set must span **both stores** so
  `trace→node→co_anchored→node→source_ref→trace` can't cycle. Within-store gates already exist in
  `30dbe1c8`; the new requirement is one **global** visited set, merge-on-collision.
- **Short chains + per-hop decay** — reuse spread activation's existing per-hop median gate; 1–2 hops,
  not 5. A node four hops from a trace is noise.
- **Control-safety is the hard gate** — any change that moves a brain-dev control's surfacer top-5
  **FAILS**, measured at the *selection* level (not candidate ranking). Same gate that killed RRF and
  z-score. **[NON-NEGOTIABLE]**
- **Loud, not silent** — a node rescued by spread must carry *why* (the chain), so a bad rescue is
  visible in traces, not hidden as rank inflation.

---

## 5. Build order — the FIX is small; everything else is a separate, gated project

**THE BURIAL FIX (retrieval — build this, it's the whole fix):**
1. **Episodic lane + trace-chain** (§3.2 + §3.3 form 1) — trace-cosine candidate source (hygiene: s0
   dialogue only, drop `tool_result`) + the semantic chain (query→trace→node, stored doc-vector) + trace
   render format. **[Tier-1 proven: #11 0→8.]**
2. **Merge** (§3.4 first half + §4 gates) — dedup, reserved-tail (fill-on-demand), no-loops, no-double-count.
3. **Gate at Tier-2** — `frame_replay`: does the always-on chain hold the 9 controls' *selections*? This
   decides the reserved-tail size K and whether any presentation help is needed. **Do not build §§below
   before this gate says they're needed.**

**SMART SURFACING (presentation — SEPARATE project, build only if Tier-2 shows the defender needs it):**
- **Bundles** (§3.4 second half) — group trace + its nodes for the selection layer.
- **Graph-edge spread** (§3.3 form 2) — reach structurally-central, semantically-dissimilar nodes; today
  coverage-dead + already-exists post-selection; unproven marginal value.
- **Particles** (§3.6) — fragment-granularity select/render.

**SUPPORTING (independent, low priority):**
- **Backfill** (§8.6) — embed the ~3,500 un-embedded historical s0 dialogue traces; recovers true episodic
  depth (and de-confounds the #11 number). Precondition: April traces carry concrete identity.
- **Source-Ref Healer** (§3.7) — backfills `source_refs` so the (presentation) structural chain heals
  backward; idle S2 unit, compounds over time.

**Reused (most of it):** trace cosine, z-weighted fragment scoring, `co_anchored` edges, per-hop spread
gates. **New for the fix:** the trace-chain lane, trace render format, the merge/reserved-tail.

**Reused (most of it):** trace cosine (lane), mutual traversal `30dbe1c8`, per-hop decay gates,
trace-as-vector path (probed), z-weighted fragment scoring, `co_anchored` edges (`07ab3f1`).
**New:** the cross-store first hop, the trace render format, bundle scoring, the gates.

---

## 6. Parallel workstream — the semantic lane (independent, do not block on it)

The episodic lane is **orthogonal** to the burial diagnosis in `RECALL-BURIAL-HANDOFF.md`. The episodic
lane fixes *episodic* queries; the semantic-lane burial still hurts *semantic* queries ("what is EX.CO").
Two independent threads:

- **▶ DO FIRST (semantic lane, from the burial handoff):** confirm the z-weighted top-2-average is what
  drops the best EX.CO node (`8359cf1d`, raw cosine rank 3) out of the candidate pool — dump its
  per-vector-group cosines. **[SUGGESTED, not VERIFIED — extend `score_decomp_probe.py`.]** Do not build
  the semantic-lane fix (MAX-not-AVG `673783e4`; prevalence-weighted title-boost `608e23b2`) before this.
- The two lanes meet only at bundle assembly (§3.4); neither blocks the other's build.

---

## 6b. Tier-1 measured result (2026-06-06, `dual_store_merge_probe.py`)

Offline A/B against the REAL `brain.recall()` baseline on the 12-corpus. **The mechanism works**:

| Query | baseline EX.CO | rescued (stored / re-embed) | pool after |
|---|---|---|---|
| #11 last-session (episodic) | **0/12** | **+8 / +7** | 8/12 |
| #2 deal-layer | 6/12 | +4 / +3 | 10/12 |
| #12 people-names | 6/12 | +1 / +1 | 7/12 (the *people*-relevant node) |

- **Stored-vector ≥ re-embed in all 3** → §8.4 free symmetric path validated (cheaper AND better).
- **Sweep:** K=3→7 rescued/27 displaced; K=5→10/45; K=8→13/72. No saturation; displacement = K×9.
- **⚠ CORRECTION — the chain is always-ACTIVE on controls, NOT inert.** Earlier hand-wave ("self-targets,
  dedups to nothing where direct succeeds") is **FALSIFIED**: every control yields ≥8 fresh inserts.
  Control-safety rests ENTIRELY on reserved-tail + defender, not on the chain going quiet. → Tier-2 gate
  is load-bearing. Mitigation observed: inserts are often *relevant* (e.g. #4 "embedding leaked" inserts
  the memory-leak/ONNX nodes), but not always (#10 inserts identity nodes for a test query).
- **⚠ #11's 0→8 is proof-of-mechanism, NOT the production number.** Two strongest drivers are eval-replay
  echoes (session cfb74766 where #11 was previously asked+answered); the genuine April episode isn't
  embedded (backfill pending). Clean re-measure needs backfill + drop eval-replay sessions.
- **Process scar:** the first run reported 0 rescue everywhere — a column-index bug (`scale` was r[5], the
  filter checked r[4]=session_id) emptied the trace set. The 4-agent verification MISSED it (validated the
  filter's intent, not its positional correctness); the geometry diagnostic (`models=set()`) caught it.
  Lesson: adversarial verification ≠ a runtime canary. Every reimplementation needs a "did it load data"
  assertion that fires loudly.

## 7. How we'll know (eval discipline)

- **Verify against the REAL pipeline's own numbers**, not a reimplementation (`IsolatedBrain`, the
  pipeline's returned `embedding_similarity`/`effective_activation`). The standalone probes are
  reimplementations — trustworthy for direction, not for ship.
- **Define the pass before running.** The episodic-lane pass: ≥1 relevant EX.CO bundle in top-25 for #11
  AND zero brain-dev control top-5 movement. The spread pass: buried EX.CO node (`598d78a8`, r37) rises
  into top-25 via a trace seed, controls unmoved.
- **Rank is a proxy; outcome is ground truth** — does Anchor *answer* the query (the oracle/recognition
  measure, `ORACLE-AUDIT-SPEC.md`). Re-anchor on it; we under-used it last session.
- **The meta-trap** (`94f6e01a`): chasing the interesting hard problem over the boring easy win. The
  episodic lane is the boring easy win sitting in plain sight — take it first.

---

## 8. Open questions / forks

1. **Episodic-lane trigger** — (a) always-on parallel lane / (b) cheap query-type gate / (c) Haiku reaches
   for it as a tool. **Decided: (a) always-on, efficiency later (Tom).** Bounded cost: ~one extra cosine
   pass over ~13K trace vectors/turn (~2–3× current node-cosine, likely sub-100ms, vectorizable). Build
   (b) only if measurement says the always-on cost is real.
2. **Bundle score formula** — how trace-cosine, node-cosine, and the (independent-only) convergence bonus
   combine, and where it slots into the STEP-6 blend. UNSPECIFIED — design next. Constrained by the §4
   "one mechanism per similarity" rule.
3. **NO query-intent classifier (decided, Tom).** Match-strength can't gate (Exp B), and query-intent
   reintroduces the discriminator recognition-over-retrieval kills. Recall = cosine + FTS5 (both stores) →
   spread → smart surface. Control-safety = hygiene + reserved-tail + defender; the **control eval is the
   arbiter** (add gating only if it fails). ≤3–5 episodic-slot cap as cheap insurance. **The real open
   problem this relocates to: "surface activation smartly to awareness"** — how the post-spread activation
   map becomes the 25 bundles (§3.4). THIS is the hard lever, not query classification. UNSPECIFIED.
6. **Episodic horizon = an un-run backfill, NOT data loss (RESOLVED, from the trace census).** The raw
   `trace_events` run continuously from **2026-04-05** and are **never pruned** (S0 retention guarantee;
   `run_maintenance` deletes only debug_log/hook_errors). Effective *embedding* depth is ~2 weeks only
   because the `trace_embeddings` table was created 2026-05-24 (Phase A) and the recency-first + 30-day
   worker never backfilled the older history. **~3,500 s0 dialogue traces (incl. the EX.CO Apr 21–22
   session) exist un-embedded.** → FIX: a **one-time backfill embed** (drop the 30-day `since`; worker
   already exists; minutes of compute). **Precondition:** verify the April traces carry concrete identity
   (Tom/Anchor) from the Phase-A identity migration — the 30-day window was originally a guard against
   embedding pre-migration `OPERATOR`/`ANCHOR` sentinel text. *(Note: Exp C's "0 April embedded" measured
   embedding write-date, not trace event-date — 312 April events are in fact embedded; corrected.)*
   **Orphan caveat (Tom):** backfilled April traces have NO `source_ref` to nodes (pre-v22 cluster), so
   the *structural* chain stays dead for them. Two resolutions: (a) the *semantic* chain (`trace_as_vector`)
   seeds a spread with no link needed — works now; (b) the **Source-Ref Healer (§3.7) reconnects them, and
   the backfill is its fuel** — the Healer can't content-match a trace until it's embedded. Sequence:
   **backfill → heal.** Orphan state is temporary, not permanent.
7. **Trace hygiene (NEW, from Exp A+C).** 82% of the store is `tool_result`; recall-machinery and
   this-session traces dominate EX.CO top hits. Decide the filter: exclude recall-tool `tool_result`s +
   this-session at recall; possibly restrict the lane to dialogue (`user`+`assistant`) only.
4. **Semantic-chain embedding mode (cost, failure point 3 — decided: no realtime effort).** Use the
   **stored** trace vector (`trace_embeddings.vector`) directly — no re-embed. That makes it document↔document
   (symmetric content) similarity; the probe used `embed_query` (asymmetric query mode, validated rank 1).
   **Test the free symmetric path first**; only if it underperforms do we re-embed-as-query — and that
   becomes a realtime-embed **infra ask** (Tom: "let me know if we need more infra"). Caching/precompute
   for that path is the infra in question.
5. **Particle coherence** — how to keep a surfaced fragment from misleading without its node context
   (§3.6 risk). UNSPECIFIED — design when particles get built.

---

## 9. Key nodes / probes / cross-refs

**Nodes:** `b8b8370b` episodic-lane validated (the salvage) · `30dbe1c8` multi-seed mutual traversal
(the spread mechanism, ignored by Surface) · `0bb0fa97` CLS · `ec2b4c1c` hippocampal indexing ·
`f152c597` index-vs-content · `07f61bc6` 3-D coordinate model · `1a2b641b` negotiation between structures
· `861a6b30` agreement-across-independent-paths = confidence · `598d78a8` buried EX.CO node (r37) ·
`8359cf1d` best EX.CO node (raw r3, dropped) · `94f6e01a` the meta-trap.

**Probes** (`eval/oracle_audit/`, all isolated): `dual_store_validation_probe.py` (**the validation run** —
provenance/separability/census/coverage; falsified match-strength valve, confirmed echo+poison+horizon) ·
`episodic_probe.py` (lane + dead structural hop) · `trace_as_vector_probe.py` (semantic chain, rank 1) ·
`graph_rerank_probe.py` (structure-rescue, 37→1) · `score_decomp_probe.py` (semantic-lane DO-FIRST).
Corpus: `meshed_top10.json` (EX.CO = ranks 2/11/12; controls = 1, 3–10).

**Substrate commits:** `c144ddf` v22 (source_refs teaching) · `07ab3f1` co_anchored auto-edge ·
EPISODIC-REFERENCES.md §0 for the full Phase A/B log.
