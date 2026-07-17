# RECALL — from the SR/PPR thesis to the Layered Activation Field (LAF)

> ## ⟦ CURRENT STATE — read this first (2026-06-30) ⟧
> Journal of the recall-redesign arc, not a static spec. **Live = §18.21 (per-field-verified LAF on the lens-independent gold).**
>
> **The pivot that reset everything (§18.19–20):** the old 73-cue gold was **100% lens-minted → circular** — every operator
> number before 2026-06-27 measured the minting lenses, not helpfulness. Rebuilt as a **lens-independent 24-cue gold** (blind
> Opus judges reason from the OUTCOME, brain-only; 16 Gold+ / 80 Gold; 2-judge ensemble; integrity-filtered; frozen at
> `eval/oracle_audit/gold_remint/frozen_gold_24.json`). Two reframes: helpfulness = **three forms** (redirect/ground/enrich),
> and gold = **pure relevance — availability is NOT subtracted** (already-surfaced is the inhibition layer's job).
>
> **On the honest gold the verdict INVERTS (§18.21):** operators that LOST to cosine on the circular gold now WIN. Discipline
> (Tom): every field must **stand on its own** before layering. Verified standalone (need-collapsed, 24-cue):
> - **MaxSim** (max cos over 6 field-groups) — healthy base (13%/20%); the nanmax-enrich-bias `794c137a` is CLEARED here (corr 0.11).
> - **graph → rebuilt as `relational_reinstatement`** — the stored edge `weight` is a meaningless **0.5 default**, so it's
>   replaced by **conductance = cos(cue, edge.why)**; sparse top-25 seed + 2-hop; reaches **24% @25 > MaxSim's 20%** = a REACH operator.
> - **episodic → THREE SEPARATE layers (never consolidate)** (`episodic_ops.py`): `encoded+` / Haiku-`picked+` / Haiku-`dropped−`
>   (÷prevalence) — each its own gain, not one "episodic" term; it's really a role×seed FAMILY (only cue-seeded built). Dropped
>   avoids gold (safe). **OPEN: "moment" is undefined** — today a crude s0 turn ±1 cosine seam; the real similarity+boundary algo is owed.
> - **temporal RETIRED** (0% distinct on this burst corpus); **node-types-in-edge-text FALSIFIED** (cosmetic, no reach gain).
>
> **LAYERING (the payoff):** one summed z-scored field (MaxSim + graph + episodic) → **reach@25 21% → 29% (+8pp)**, decomposed
> **brought +8–10 ≫ lost −2–3, reinforced 5–6** — validates the one-field model (an overlapping operator earns its place by
> bringing misses AND raising existing). **But it COSTS @5 (14% → 12–13%)** — the aux fields are reach operators that inject
> noise into the top-5. **Reach is solved; converting reach→top-5 is the open problem** (the `8bcc8c96` selector, now over a real
> enriched pool). Caveats: N=24 (directional); gains hand-picked, un-swept.
>
> **▶ NEXT:** (a) gain-sweep for the @5/@25 operating point, or (b) the @5 ranker/selector on the enriched pool. Plus: the
> episodic **seed-axis gap** (only cue-seeded; prev-anchor / work-context seeds unbuilt) and the **moment-definition algo**.
>
> **Reading guide:** §0–17 = SR/PPR → falsification → integration-thesis (largely historical). §18–18.17 = LAF design + selector
> reframe. §18.18 = settling engine. §18.19–20 = circular-gold pivot + lens-independent re-mint. **§18.21 = LIVE.**

**Written:** 2026-06-17 (Anchor + Tom, session `ce0ff8ce` arc continued in a fresh stream).
**Status:** design / direction. Nothing built yet — this is the synthesized thesis and the staging.
**For:** future-me. Written so it surfaces when I'm working recall ranking, spread, the endo
surface, episodic/temporal recall, query composition, or "should we train something."
**Parent:** `ARCHITECTURE-FRACTAL.md` (`integrate(O,K)→Δ`), `RECALL-OVERVIEW.md` (current pipeline),
`RECALL-STATE.md` (validated current numbers). **Siblings:** `SOFT-SURFACE-DESIGN.md` (the endo
surface — this is the kernel it inherits), `ORACLE-AUDIT-SPEC.md` (the eval instrument),
`RECALL-DUAL-STORE-DESIGN.md`, `RECALL-TEMPORAL-ANCHOR-SPEC.md`, `HANDOFF-RECALL-NORMALIZATION.md`.

> **The one-line thesis** *(historical — the SR/PPR framing; largely falsified at §13b, see CURRENT STATE).* Relevance is *flow from your current state through a stable relational
> geometry.* Put all the "where am I now" in the **seeds** (an orthogonal basis of state anchors,
> convergence-combined, learned-gated by next-turn prediction); put all the "how the brain connects"
> in a **static, symmetric-normalized, aspect-weighted transition matrix P** (corrections directional +
> suppressive). The diffusion `(I−αP)⁻¹` is **Personalized PageRank ≡ the hippocampal Successor
> Representation** — a *proven* equivalence (arXiv 2512.24722). α picks the regime; stay below the
> hub-collapse transition. Made stateful across turns, it's an **online SR** = frozen geometry + two
> bounded recurrent states (fatigue, belief), read by attention/PPR each turn.

> **The catch that gates everything.** The read can't exceed what the write encoded. Our one empirical
> test of the matrix idea was a **null** (`aa33ebb9`) — because the state-index was too thin and the
> eval corpus was contaminated, not because the operator was wrong. So: **corpus rebuild + session-level
> eval are preconditions, not parallel nice-to-haves.**

---

## 0. Why this doc exists

We spent a long arc moving from "spread activation is broken" to "recall is the Successor
Representation, learned from next-turn prediction." It's coherent but large, and it spans
neuroscience, graph theory, and our own substrate. This is the narrative + the math + the open forks,
so I can pick it up cold and dive into the two remaining threads (the **value function** and
**encode/decode co-design**) without re-deriving.

---

## 1. The narrative arc (how we got here)

1. **Endo surface** (`SOFT-SURFACE-DESIGN.md`): recognition off my *own* output (the Stop self-cue),
   additive bar required (`0d395b91`, `2ce3aa55`). Blind test retired "inverted retrieval" — contradicting
   priors are cosine-**near**, not far (`e857d994`); the real problem is **awareness suppression**, not
   retrieval geometry (`a9a754ed`). The endo surface **reuses the recall+spread kernel** — so "is that
   kernel healthy?" became the question.
2. **Spread diagnosis** (`0f3efed4`): live EX.CO pull → 5 seeds activate **590 nodes** (~10% of brain);
   `tanh` saturation pins hundreds at activation=1.0; hub-tiebreak buries the focused answer. A/B on the
   58-node harness: cosine top-25 (84%) **beats** spread (79%). Tom's *intended* spread (semantic-anchored,
   convergence-as-bounded-boost) was **inverted** in implementation into path-count-dominance (`8a33f0b6`).
   Agent code-audit (`8e5c9df5`): `HOP_SCRUTINY` off in prod by default; `cluster` variant already tracks
   `convergence_count`.
3. **Recall redesign** (multi-arm bandit, `32b88dd5`; 5-lens workflow): burial root cause is **NOT
   embedding** — it's the **scoring blend + a floor-bypass** (`ac26cca4`). 85% of candidates (FTS5/trace
   lanes) bypass the relevance floor and float above scored cosine nodes via `effective_activation`
   (Spearman −1.0 vs final rank, `4f4d3aa9`). Side-findings: `recency_score = 1.000` across all 2878 rows —
   dead component (`37021fd1`); spread-arm vs cosine-arm have **different diseases** (`ef5bdebb`).
4. **Query starvation** (`179f7858`): the retrieval query is **just the user message + 2 user turns**.
   My own last response (the un-shipped May "game changer" `e765929e`), the arc, episodic — all absent.
   We use **~1.5 of 15** state descriptors (`12848b52`). Query-side is higher leverage than node-side ranking.
5. **The reframe spine**: retrieval is **reinstatement of state**, not content search (`c65776f5`,
   `2f7e5b03`); and beneath that, recall is **prediction** — the next turn is the self-supervising label
   (`ca840441`). Unified: **the Successor Representation** (`eee3eaa5`).
6. **Multidimensional / online**: make it stateful across turns — fatigue + belief carried, P frozen.
   (This session.)
7. **Training / value** (this session): what "learning" means, multi-timescale, S2-replay; and the two
   open threads — the **value function** and **encode/decode co-design**.

---

## 2. The three diagnoses + two side-findings

| # | Disease | Evidence | Node |
|---|---|---|---|
| 1 | **Spread saturation** — 590 nodes, tanh→1.0, hub-tiebreak buries answer | live EX.CO pull; A/B 79% < 84% cosine | `0f3efed4` |
| 2 | **Blend + floor-bypass** — 85% of candidates skip the floor, rank by `effective_activation` | 5-lens workflow + code read | `ac26cca4` |
| 3 | **Query starvation** — query = user msg only; ~1.5/15 descriptors used | code-verified | `179f7858`, `12848b52` |
| + | `recency_score` dead (1.000 × 2878 rows) | feature table | `37021fd1` |
| + | eval corpus stale (186d, pre-v29, archived golds) — **episodic boost INVERTS relevance on it** | corpus audit | `8e417c62`, `ac26cca4` |

Two distinct miss classes (route fixes correctly): **8 buried** (in pool, ranked too low → ranking fix)
vs **6 never-recalled** (never in the 100-pool → query-side / candidate-generation only) (`0d368dd3`).

---

## 3. The reframe spine — reinstatement → prediction → SR

- **Reinstatement** (`c65776f5`): relevance is not a property of content; it's the match between the
  *present state* and the *state that produced the knowledge*. Encoding-specificity (Tulving), hippocampal
  index theory. "Memory isn't a library you search by topic; it's a state you re-enter."
- **Prediction** (`ca840441`): evolutionarily, memory exists to *predict the next step*. The best recall
  for turn N is the one you'd write after seeing turn N+1. The future turn is the label →
  **self-supervised recall**, dissolves the contaminated-gold problem. Recall output should be a **field**
  (distribution over nodes), not a premature top-K. Associate trace→nodes **created/revised** then, never
  **surfaced** (surfaced is circular echo — killed the trace-chain lane).
- **Successor Representation** (`eee3eaa5`): the dominant hippocampal model (Stachenfeld & Gershman 2017)
  — represent each state by its **discounted expected future**, `M = (I − γT)⁻¹`. **PPR ≡ SR proven**
  (arXiv 2512.24722, Millidge Dec 2025): set α=γ, P=T. Graph-retrieval and hippocampal memory are the
  *same operator*. HippoRAG (2405.14831) = LLM + KG + PPR.

---

## 4. The operator & the math

PPR: `r = (1−α)(I − αP)⁻¹ s`. Unfold the resolvent (Neumann series):

```
r = (1−α) · Σ_k α^k P^k s  =  (1−α)·[ s + αPs + α²P²s + α³P³s + … ]
```

`P^k s` = where seed-mass lands after exactly k hops → `r` is a **decay-weighted sum over all path
lengths**. Three truths:

1. **Relevance is flow, not a property.** Cosine = Euclidean distance in a flat space (ours *is* flat:
   0.54–0.63, `dea1a002`). PPR = **geodesic** along the graph's own structure → reads the signal the
   embedding threw away.
2. **Convergence is built in** (`87bb8718`, `7d9b5a8f`): a node reachable from seeds A *and* B accumulates
   both masses → multi-seed nodes win even at lower cosine. This is "A→C and B→C suddenly makes the cut,"
   as algebra, not heuristic.
3. **α is a phase transition.** As α→1, `r` → dominant eigenvector of P = **global PageRank = pure hub
   centrality**. Small α → relevance-anchored near seeds; large α → hubs dominate regardless of query.
   **Our broken spread lived past α\*** — that's the math under hub-burial.

**Emergent behavior:** (a) phase transition in α; (b) convergence amplification (superlinear lift for
jointly-supported nodes — the good emergence, serves top-5); (c) community basins/attractors — mass pools
in dense clusters → **within-cluster smear** (reaches the right neighborhood, can't rank inside it — this
*predicts* the `aa33ebb9` finding); (d) it's a **smoothing operator** — trades sharpness for robustness,
a *good* trade on a flat space.

**Good at:** compositional/multi-hop, vague queries, state-recurrence, robustness on flat embeddings.
**Bad at:** within-cluster discrimination (smear), cold low-degree nodes (diffusion starves orphans —
rich-get-richer; idf must counter), garbage on a thin/wrong graph.

---

## 5. Decision: adopt the operator (a variation), placement = B

**Not HippoRAG-the-system** — its offline LLM-OpenIE indexer rebuilds a graph from raw passages; **we
already have a richer typed graph** (encoder + S2 = our continuous OpenIE). We adopt the **operator + the
proof**, fed by inputs richer than HippoRAG's on two axes (our graph; multi-anchor state seeds vs their
query-entity seeds) plus the SR's learned-T / prediction dimension HippoRAG lacks.

**Placement fork** (Tom chose **B**):
- **(A) post-agent re-ranker** — diffuse from Haiku's picks; fixes saturation + the 8 buried; **cannot**
  reach the 6 never-recalled. Smallest move; where we'd *prove* it (contained blast radius).
- **(B) retrieval operator seeding the agent** — multi-anchor query → PPR over the whole graph → ranked
  pool → agent selects. Reaches the never-recalled/compositional class. HippoRAG's actual placement. The
  vision; bigger regression surface.
- **Staging:** prove as (A) on the rebuilt corpus → move upstream to (B). Floor-bypass (`ac26cca4`) is
  upstream of both and gets fixed regardless.

Spread-rebuild "Arm C" (`a4db93ee`) and "real PPR" are the **same thing** — Arm C's
`max(field_activation)×(1+coeff·convergence_count)` is a crude 1-hop truncation of the full resolvent.

---

## 6. Parameters — the theses (logic, not trial-and-error)

### Seeds `s` (the personalization vector = the query)
Principle (`12848b52`): *state is the salient SUBSET of descriptors, not the union — including all of them
dilutes the signal.* Three sub-principles:
- **(a) Choose for orthogonality, not relevance** — value of an anchor = marginal info. The three
  near-orthogonal axes: **user message** (the ask) · **my last response** (the live state, `e765929e`) ·
  **episodic trace-matrix** (the formation state, `8e8c65cc`). Arc = low-weight 4th (frame). Last-2-turns =
  redundant with the first two; fold in, don't seed separately.
- **(b) Never pre-blend** (`87bb8718` — conjunction collapse). PPR is linear: `r = Σ_k w_k·PPR(s_k)`.
  Keep anchors as separate seed masses; convergence finds their intersection. Mean-pooling destroys exactly
  the structure we want.
- **(c) Weights `w_k` are query-conditional and learned** — not constants, not blind search. Sharp factual
  → user-msg dominates; vague "where were we" → my-response+arc; "have we hit this" → episodic. The gate is
  supervised by the **same next-turn label** as the whole SR direction. *One* learning problem.

The 15-descriptor basis (`12848b52`, `5b42a349`): Said (msg / my-response / thread / arc), Did (files /
mode / branch), When (trace-trajectory / cadence), Felt (tone), Known (awareness set — this *is* fatigue),
Who/where (operator / project / phase). We use ~1.5. Most are already in traces — we embed words, discard
the rest.

### Edges `P` (the transition matrix)
Principle: `P[i→j]` = probability relevance should flow i→j = "if I'm at i, is j the next thing I need" =
a **conductance**.
- **Primary signal = aspect-conductance**, NOT endpoint-cosine (flat). But **corrections are directional +
  suppressive, not a high scalar** (`e524b57c`, LTD analog): `corrects`/`supersedes` → strong flow toward
  the corrector + **suppress the corrected node**; `extends`/`refines` → medium symmetric;
  `community_member` → weak; `co_anchored` → weak-temporal. So aspect → `(direction, sign, magnitude)`.
- **The key call — factor static vs dynamic:** **P is static** (precompute the resolvent once); **ALL
  query-dependence lives in `s`.** Tom's "cosine(arc, edges) / cosine(aspect, state)" ideas make P
  query-dependent → you lose the precomputed inverse *and* hub-damping. T is the stable world-model; `s`
  is where you are now. (Exactly how attention factors: static weight matrices, dynamic query — and PPR ≡
  attention.) State-conditioning is expressed by *seeding near* the relevant relation types, not rewriting P.
- **Symmetric-normalize** `P = D^(−1/2) W D^(−1/2)` (GCN trick) → damps high-in-degree hub accumulation.
  The principled hub fix, independent of α.
- **Top-5 needs a diversity guard** (MMR/DPP) — cluster-smear else fills slots 2–5 with near-duplicate
  siblings, and 5 gold slots is the whole goal.

---

## 7. Multidimensional / online — the two-level loop

Diagram: `two_level_recall_loop_vs_transformer` (per-turn read ≈ attention/Hopfield; across-turn state ≈
state-space recurrence; diverges from transformer = append-only KV cache that grows, vs frozen geometry +
small contracting state).

**Layer 0 — frozen geometry (built once):** `P = D^(−1/2) W D^(−1/2)`; precompute `M(α) = (1−α)(I−αP)⁻¹`
(or k-step truncate).

**Layer 1 — per-turn read:** anchors `e_k^t` → node-distributions `ê_k^t`.
```
s_t = Σ_k w_k(θ_t)·ê_k^t  −  λ·f_t                 (gated seeds minus fatigue)
r_t = M(α_t)·s_t = Σ_k w_k(θ_t)·M(α_t)ê_k^t − λ·M(α_t)f_t   (linearity → convergence is a sum)
surface = diversify(r_t)                             (MMR/DPP, then cut to top-5)
```

**Layer 2 — cross-turn recurrence (two carried states):**
```
f_{t+1} = β·f_t + 1[surfaced_t]                      fatigue: accumulate + decay (0<β<1)
θ_{t+1} = (1−ε)·θ_t + ε·θ_0 + η·g_t                  belief: drift on signal g_t, leak ε to prior θ_0
```

**The one inequality that decides usable-vs-runaway:** carried state `x_t=(f_t,θ_t)` must contract →
`ρ( ∂x_{t+1}/∂x_t ) < 1` (spectral radius below 1). Fatigue: `β<1` (free). Belief:
`ρ[(1−ε)I + η·∂g/∂θ] < 1` → **the leak ε must dominate the feedback gain η.** This is the **Echo State
Property** from reservoir computing — a known closed condition, not something to find by trial.

**Goal #1 (no resurfacing) falls out as suppression:** inject the already-surfaced set as suppression —
cleanest as **absorbing/killed random-walk states** (drain mass), not raw negative seeds (negative-mass
artifacts). Diffusion suppresses the node *and its neighborhood*. This suppression set **IS the endo
surface's awareness set** (surfaced ∪ cited ∪ echo) — the recurrence is the mechanism the endo surface
needed; the threads converge.

**Goal #2 (long-conversation refinement) = a session that warms up:** α anneals (early high/broad → late
low/tight); the anchor gate rebalances as the arc accrues evidence (Bayesian belief update). Emergent:
a **session attractor** that settles onto the conversation's manifold. The leak-to-prior keeps it *subtle*
(can't drift into a basin / mode-collapse).

---

## 8. What "learning" actually means here

**Not** the Gemma graph-traversal fine-tune (`b706eebe`, `15bd986f`, `fbc8b1a4`) — that was teaching a
neural model the graph schema (heavy, offline GPU fine-tune; recognition-training created embedding
dependency, `44403f86`). **This** is fitting two small explicit objects:
- **T (geometry):** "which nodes are needed after which states" — counted/decayed transition statistics
  over our trace history, or TD over replayed trajectories. **No backprop, no model — fitting a sparse
  matrix.** Closer to fitting a Kalman filter.
- **θ (gate):** anchor weights + α — from the next-turn prediction signal.

**Is it true to all parts? No — multi-timescale:**

| Part | adapts | rate | rule |
|---|---|---|---|
| fatigue `f`, activation | live awareness | every turn | decay + accumulate (no "training") |
| belief `θ`, arc | session focus | within a session | drift + leak to prior |
| edge weights, `T` | the geometry | across sessions | slow statistics / replay |
| nodes & edges | the graph | on encode | the encoder builds it |

No part is *trained* uniformly; the system is a stack of learners at different speeds — which is just the
`integrate(O,K)→Δ` fractal (Δ→K *is* learning) at the recall scale. The SR gives one shape ("update a map +
a state") that specializes per row by learning-rate and leak-rate.

**Constant post-training? Mostly-frozen with slow drift** (complementary-learning-systems): geometry
consolidated **offline and frozen during a session** (the precompute); session state live but ephemeral
(leaks to prior, resets between sessions). Continual at *controlled* rates — the leak + offline
consolidation is the brain's stability–plasticity / catastrophic-forgetting answer.

**Complexity? Low.** Online (`θ`,`f`) = a vector update, microseconds. Geometry = `O(edges)` statistics;
SR is famously cheap vs model-based planning; resolvent solved by iterative push / **incremental PPR**
(maintain walks ~`O(1)` per change — Bahmani VLDB; EvePPR), not naive `O(n³)`. **Learn T offline in
replay** — we already have the idle consolidation loop; this is a new **S3-class** unit (behavior-changing
learning = S3, deferred until stable+measurable — see §12), not a new pipeline.

---

## 9. The underlying math (and the deepest frame)

Everything — PPR read, SR map, TD learning, echo-state stability, encoding — are facets of **one frame: a
predictive filter over a graph** (Bayesian predict-correct), which closed-loop *is* **active inference /
the free-energy view** (Friston; SR-active-inference arXiv 2207.09897). Recall = inference (posterior over
relevance given state); encoding = updating the generative model; prediction = the objective. The graph
Laplacian is the prior, the read is the posterior, learning reduces next-step surprise. This *is*
`integrate(O,K)→Δ` made mathematical — the fractal's recall-scale instance.

---

## 10. Known math, research map, pitfalls

| Our piece | Known as | Pitfall it teaches | Fix it hands us |
|---|---|---|---|
| the read `M(α)` | PPR / SR / modern Hopfield / linear attention (one operator) | hub collapse as α→1 (our spread bug) | α below transition; symmetric norm |
| recompute per turn | dynamic / temporal PageRank (Rozenshtein–Gionis; EvePPR; Bahmani) | recompute cost; index staleness | incremental update, don't re-invert |
| online `θ` | SR via TD (Dayan; Stachenfeld–Gershman; eLife 80680/78904) | SR is **policy-dependent** → stale when conversation pivots (off-policy bias) | bounded η + leak-to-prior; or STDP-style local rule |
| carried-state stability | leaky echo-state network / contractive SSM (Jaeger; arXiv 2509.04422) | runaway if ρ>1; drift/forget if too plastic | ρ<1; leak ε = plasticity regularizer |
| fatigue suppression | absorbing / killed random walk (loosely signed PPR) | naive negative seeds → negative-mass artifacts | absorbing states drain mass |
| top-5 diversity | result diversification (MMR; DPP) | cluster-smear fills slots with siblings | MMR/DPP on the field |

**Pitfall checklist:** (1) keep α below hub-collapse; never make P query-dependent. (2) leak ε is
non-negotiable — the line between "refines" and "drifts." (3) SR is policy-dependent → leak-to-prior keeps
a session-adapted θ honest after a pivot; don't crank η. (4) suppress via absorbing states, not negative
seeds. (5) explicit diversity at top-5. (6) **session-level eval** — invisible to single-turn Q→gold.

**Improvements the literature hands us:** incremental PPR (cheap recompute); STDP/theta-sweep SR learning
(local, cheap update rule for θ vs full TD); DPP for diverse top-k.

---

## 11. What we missed (the structural gaps — the real payoff)

1. **We've designed the read in isolation from the write.** No read-side math rescues a graph not
   *encoded* with predictive structure (`5b42a349`, `aa33ebb9` — the index is the bottleneck). Lever:
   **encode/decode co-design** — encode the transitions the SR needs, not just semantic content.
   (Encode/decode symmetry is already a brain principle; unapplied here.)
2. **Where is T learned? → replay.** CLS: the slow store learns by replaying the fast store offline =
   **S2**. We have the consolidation loop and haven't wired it to learning the predictive map. This single
   decision answers "constant?" and "complex?" at once: T learned by S3-class replay (deferred — see §12),
   frozen per session.
3. **The value function `g_t` is underspecified — the genuinely open piece.** SR factors the *map* (T,
   learnable) from the *goal* (what recall optimizes). We have the map's direction but not an operational
   definition of "what the next turn needed." That's the reasoning-oracle / teacher→student thread
   (`0cac5dad`, `ca840441`). It decides whether learning has a true north or chases correlations.

---

## 12. Hard gates & current status

- **S3 prompt-freeze (Tom, 2026-06-17 — a correction to my earlier framing):** interaction/prompt
  learning is **S3 (growth), deliberately deferred** — changing the selection prompt while the system is
  unstable and not-yet-measurable induces chaos. (I'd wrongly called it a live S2 mechanism; it isn't.)
  So **near-term the Haiku guidance is STATIC**; only the **dynamic state** Haiku sees (candidate set,
  seeds, fatigue, previous selections, the multi-anchor query) is workable. All the online/multidimensional
  design (§7) is *dynamic-state* work → **compatible with the freeze.** The value-anchored *guidance
  rewrite* (and behavior-changing T-learning) is **S3-later**; near-term = measurement + dynamic-state
  engineering only.
- **Corpus rebuild** (recent <60d, de-contaminated of archived/superseded golds, episodic substrate
  present). Tom's instinct, confirmed: *everything is measured against it and it isn't what we need*
  (`8e417c62`, `ac26cca4` meta-finding). Current corpus: `eval/oracle_audit/control_corpus.json` (186d,
  control fails 15/30, `RECALL-STATE.md`).
- **Session-level eval harness** — sequences of turns, score the trajectory. Required before the
  recurrence (Layer 2) is even measurable. Stacks on top of the corpus, not beside it.
- **Stopgap status:** `BRAIN_SURFACE_RANK_MODE=cosine` branch is **written into `surface_contract.py:1865`
  but uncommitted, default 'activation' (off), not exported** → production still runs the broken
  connectivity ranking (`13b439c2`, `8e5c9df5`). "Spread for breadth, cosine for rank."
- **Arm 1 (score-for-all / floor-parity)** = the highest-leverage ranking fix (`ac26cca4` portfolio),
  highest regression risk (floor bypass is intentional) → ship behind a flat-cosine gate, per-mode
  recall@8 must not regress. Cheaper warm-up: tune down `TITLE_MATCH_BOOST`/`SITUATION_WEIGHT`.
- **North-star metric:** recall@5 / nDCG@5 on gold (Tom's framing — push gold into top-5 → endo surface
  becomes viable + I use recall efficiently).

---

## 12b. Ground-Truth workstream-2 result — recall-quality baseline (2026-06-17, node `a39e0104`)

Teacher-on-production: 90 real surface turns judged by Opus teachers vs Anchor's actual next move
(harness: `eval/oracle_audit/surface_quality_{sample,report}.py` + the `recall-quality-baseline` workflow).
- **Headline:** served_well **51%** / +partial **78%** / **61%** of memory-needed turns. Pick quality:
  best_available 73%, better_candidate_dropped 12%, mostly_noise 11%. Recall is **decent, not broken**.
- **Dead lever:** contested turns (top-1 cosine dropped) served_well **57.9%** vs uncontested **39.4%** →
  **cosine-rank re-ranking of the pool is dead** (buries the "67% top-1 dropped" worry).
- **Tools:** off-pool/tool-fetched picks are **higher-variance** (52% well but only 65% partial+) — nail it
  or wander; the wander tail is the Q2 replay-audit target.
- **Query-type map:** design 77% · compositional 62% (PPR/SR sweet spot) · factual 46% (**flat-embedding
  wall — the PPR/SR target**) · action 43% (often needs live code — a ceiling). ~17% of turns need no
  memory (surface-need gate opportunity).
- **Key property:** this metric is **corpus-independent** (judges real turns, not the stale corpus) → a
  trustworthy, re-runnable ruler NOW. Workstream 2 partly de-risked workstream 1.

## 12c. Endo measurement — artifacts found, clean harness, inconclusive (2026-06-18)

Tried to measure endo recall by *replaying* recall as of each cue (cue = one of Anchor's own turns;
ground truth = Anchor's actual next turn). First pass: ~12% hit@2. **It was an artifact — three of
them, all biasing low:**
1. **`created_at` cutoff is post-top-K** (`brain_recall.py`: `[:limit]` at :1891, then `_apply_filter`
   at :2013; only `type` is pre-scan-pushed at :1353-59). Future nodes — often encoded *from the cue's
   own conversation*, maximally similar — win top-K slots, get stripped, no backfill → the cue-era gold
   never appears and it under-returns (3–10 vs a full pool). **Fix:** filter `created_at` *before*
   scoring, or large over-fetch.
2. **FATIGUE contamination (the big one).** Recall STEP 8 (`_mark_accessed`, :2020) marks every returned
   node → in-memory per-session fatigue dampening, applied to the *next* recall in that session, keyed
   by the **caller's session, not the cue's origin.** Replaying many cues through one session makes each
   call suppress the previous call's winners (proven: the same query repeated returned 3 *disjoint* sets;
   the rank-1 node fatigued out by call 3). The MCP `recall` tool exposes **no `session_id`**, so it
   *cannot* isolate fatigue. **Fix:** unique throwaway `session_id` per cue (only possible off the MCP
   tool). NB fatigue is *correct* in production — it's the endo "forgotten" condition — so this is an
   eval-isolation issue **and** a build constraint (key fatigue per-session, never shared).
3. **Teacher-bar drift** — "the *specific* move-changing node" (strict) vs "a topical node" (loose)
   flips hits↔misses. Pin it with the cue-0 worked example.

**Clean harness:** `eval/oracle_audit/endo_clean_recall.py` — recall against an **IsolatedBrain copy**
(never live), **fresh `session_id` per cue** (no fatigue), **over-fetch + cutoff** (no truncation).
Offline, zero Anthropic spend. Caveat: loads a 2nd embedder → can contend with the live daemon (a recall
timeout followed; run with the maintenance lock or while the daemon is idle).

**Clean result (n=12, strict bar): ~1 hit (cue 0, rank 2) — INCONCLUSIVE.** Same ballpark as the poisoned
12%; the fixes bought *honesty*, not a higher rate. The misses are *mostly correct silence* (status
reports, wrap-ups, live git/fs checks, fresh discoveries no prior could hold). cue 0 = real hit (the
brain held the plugin arch I instead asked Tom to clarify); cue 1 = real miss (a fact I'd surfaced
earlier didn't come back). At n=12 you **cannot** separate "recall weak" from "endo moments rare."

**Epistemics correction (Tom caught it):** the random-turn sample is NOT "the wrong corpus" — every Stop
*is* an endo cue, so it correctly answers "does endo help on a random turn?" (rarely; *gating*-relevant).
It just can't measure *retrieval quality* (silence-dominated). Two different questions — don't conflate
"wrong corpus" with "wrong question." And: noticing cues were "silence-worthy" only *after* a low score
is the tell of motivated reasoning — defend each silence-call independent of the score.

## 13. Next — RETRIEVAL FIRST, then gating (Tom, 2026-06-18)

**Decided:** strengthen *retrieval* before building the *when-to-fire gate* — you can't tune a gate on
mediocre retrieval (you'd be gating noise). Get retrieval strong, *then* decide when to deploy it.

**▶ NEXT STREAM STARTS HERE:** build an **endo-worthy-stratified** corpus (cue-0/cue-1-shaped: a
forgotten move-changer plausibly exists), then measure **retrieval quality** on it via the clean harness
(§12c) — cosine baseline → PPR/SR arm A. The random-turn sample is *gating*-relevant and comes *after*;
it can't discriminate retrieval (correct-silence-dominated). Harness rules: unique `session_id` per cue +
over-fetch (or pre-scan cutoff); run with the daemon idle / maintenance-locked (2nd-embedder contention).

**Later (after retrieval is strong) — the original forks, now sequenced behind retrieval:**

1. **Value function `g_t`** — what defines "the next turn needed"? Teacher (reasoning, offline, sees the
   future → labels) → student (cheap present-only ranker → distills). The reasoning-oracle confirmed
   *divergence* (present vs future recall ~10% overlap) but its pro-prediction headline was hindsight-biased
   (`0cac5dad`); settle learnability-from-the-present with a teacher→student test, not a hindsight judge.
2. **Encode/decode co-design** — what must the encoder lay down (richer source_refs, state descriptors per
   `12848b52`) for the SR to have structure to learn? Gap #1 above.

---

## 13b. RESULT — retrieval measured; read-side falsified; the embedder is the wall (2026-06-18, continued)

Did exactly what §13 asked. Built the endo gold corpus and measured. Pipeline in `eval/oracle_audit/`
(`endo_corpus_build` → `coverage_sweep` → `teacher_prep`/`judge` → `corpus_freeze` → `baseline_recall` /
`ppr_ab` / `episodic_arm` / `followup_arm` / `miss_analysis` / `corpus_harden`); frozen corpus
`endo_corpus/endo_gold_corpus.json` — **73 cues (34 anchor + 39 operator), Opus-teacher gold**. New nodes:
`89583d50` (read-side falsified), `adb91eb7` (two-disease), `836f5d6d` (embedder is the wall),
`324ea730` (HippoRAG mismatch), `62052d67` (gold-circularity lesson), `2a1cf110` (Premium Corpus).

**Every read-side lever underperforms plain cue-cosine** (hit@5, essential gold): cosine_cue **22%** ·
brain.recall (full ~11-vector z-weighted blend) **19%** · PPR operator **14%** (α-sweep monotonic decay →
diffusion smears on a flat/hub graph) · content-episodic **7–8%** (cue→trace→node transitivity collapse) ·
predictive-episodic / "followup" (`06fe1193`-style seed-from-the-future-of-past-analogs) **3–5%** (too
noisy through 3 cosine hops). The apparent **seed-lever** (next-move oracle, **42%**) was a **gold-discovery
lens-priming artifact** — on gold its lens didn't find, the next-seed scores **0%** (`endo_corpus_harden.py`).
That retraction is the §12c discipline biting a 4th time (`62052d67`): never eval a retrieval method on gold
that method generated.

**The embedder is the binding constraint, two proofs** (`836f5d6d`): (1) the **oracle** — embedding the
*actual next turn* as the query — caps at **42% hit@5 / 68% hit@25**; even the perfect query can't rank gold
into top-5 past ~42% (relevance signal ~0.00–0.05 < embedding jitter ~0.05–0.09 → near-random sort within the
band). (2) the full multi-vector scheme (19%) **≈** single `_primary` cosine (22%) — all ~11 vectors share
the same nomic model on the same narrow domain, so combining them averages flat with flat. The model resolves
**topic, not proposition-level relevance**. Lever = a **more discriminative embedding** (domain fine-tune on
our cue→gold pairs, or a sharper model).

**Two-disease miss split** (`adb91eb7`, `endo_miss_analysis.py`; 80 essential-gold: 20% top-5, **53% far-120+**):
(A) within-cluster **smear** — 26% of misses, gold within 0.03 of the top-5 bar → embedding/reranker lever;
(B) **cue-far** — gold not surface-similar to the cue, 53% not even in pool → query/seed lever (what the oracle
fixes). FTS rescues only 10%.

**Why PPR works for HippoRAG, not us** (`324ea730`): they have the *write* substrate (OpenIE entity-bridge KG +
multi-hop task + sharp entity seeds + uniform degree); we took the *read* operator onto a sense-making graph over
flat-embedded ideas for a single-hop task. PageRank isn't the magic; the KG is.

**Correction (Tom's catch, post-hoc):** the PPR arm above was built over ALL edges — **~71% structural/usage
noise** (co_accessed 32% + community_member 14% + related_to 13% + related 6% + co_anchored/emergent_bridge),
which production *excludes* from traversal (`d1d1a90c`) and which amplify hubs. The **noise-filtered semantic**
re-run (`endo_ppr_semantic.py`; keep extends/grounds/corrects/implements/refines/depends_on — 9,319 of 29,466
edge-relations) gives **16%** (vs noisy 14%) — filtering helped, but it's **still < cosine 22%**, still
α-decaying, and **no hit@25 gain** (33% ≤ 36% → no multi-hop reach to cue-far gold; the semantic edges connect
topically-adjacent nodes, so diffusion stays in-cluster). Split: it preserves anchor cues (21%) but smears
operator cues (→5%). **Untested:** directional+suppressive corrections + aspect-conductance weights (the §6
design) — only symmetric aggregate-weight was tested, so "PPR is dead" overclaims. Established: *symmetric
semantic-graph diffusion doesn't beat cosine for top-5.* Mandatory henceforth: any graph-jump must exclude the
`noise` aspect + `co_accessed` first.

**▶ NEXT STREAM STARTS HERE (updated):** two parallel threads. **(1) Embedder bench** — sharper models on the
SAME corpus, single-vector / model-isolated; running on stream `06fe1193` (its work, not this stream's). Watch
hit@25 + the 42% oracle ceiling; the full upside needs a re-minted gold with non-nomic discovery lenses.
**(2) Premium Corpus** (`2a1cf110`) — 10 cues (5 op + 5 anchor) hand-examined by Opus agents: recall more, read
traces before/after, examine node fields, analyze "what WOULD have surfaced this" against the theory.
`endo_premium_select.py` → `premium_seeds.json`. The seed-prediction lever (oracle's 42%) stays open but needs
a realizable predictor AND a sharper embedding — the realizable attempts (trajectory, followup) failed on the
flat space.

---

## 14. Node & artifact index (pull these for full context)

**Endo surface:** `0d395b91` (architecture), `e857d994` (inverted-retrieval retired), `2ce3aa55` (3
helpfulness conditions), `a9a754ed` (awareness suppression), `17da090c` (examine-K vs inject-K),
`aecb20cf` (pull-volume not the lever), `5c6dc24e` (blind test).

**Spread:** `0f3efed4` (590-node flood), `8a33f0b6` (intent vs implemented), `8e5c9df5` (code audit),
`bb533447` (traversal math), `e410deed` (L0–L5), `13b439c2` (stopgap), `a4db93ee` (Arm C), `29f0f385`
(inject-precision A/B), `69052586` (three jobs fail), `69fc4f08` (community).

**Recall diagnosis:** `ac26cca4` (blend+floor-bypass root cause), `ef5bdebb` (two diseases), `37021fd1`
(recency dead), `4f4d3aa9` (rank=effective_activation), `0d368dd3` (6 never-recalled), `8e417c62` (corpus
stale), `32b88dd5` (multi-arm bandit), `c6651af4` (idf2).

**Reframe / SR:** `c65776f5` (reinstatement), `2f7e5b03` (state-of-mind), `ca840441` (prediction),
`cfff6201` (reinstatement-of-state), `eee3eaa5` (**SR unifies — M=(I−γT)⁻¹, PPR≡SR**), `aa33ebb9` (matrix
null — thin index), `5b42a349` (capture more at encode), `12848b52` (15 descriptors), `8e8c65cc`
(trace→node matrix), `03c05e89` (multi-anchor), `87bb8718` (query multiplicity), `7d9b5a8f` (convergence as
signal), `e524b57c` (corrections as suppression / LTD), `e765929e` (embed my last response).

**Training thread (distinct):** `b706eebe` (Gemma 4 E2B), `15bd986f` (4200 dense examples), `fbc8b1a4`
(preserve graph schema), `e356d258` (traversal structure is the variable), `44403f86` (recognition →
embedding dependency), `fe2a0ea1` (collapse tool into model).

**Eval methodology:** `b4d6f876` (oracle must out-recall), `ca9d9103` (two-retrieval divergence audit),
`3a8ba7bc` (stationarity check), `1939273d` (episodic gate-not-lane), `0cac5dad` (reasoning-oracle verdict).

**On-disk artifacts** (`eval/oracle_audit/`, mostly untracked): `reasoning_oracle.py(+_result.json)`,
`oracle_gap.py`, `recall_feature_table.py(+.json)`, `episodic_boost_recover.py`, `episodic_reach_6.py`,
`spread_glance.py`, `spread_inject_ab.py`, `spread_output_shape.py`, `control_corpus.json` (the stale gold).
**2026-06-18 eval harness:** `surface_quality_{sample,report}.py` (Haiku-path baseline, 51% served),
`surface_tool_usage.py`, `surface_selection_baseline.py`, `surface_teacher_briefs.py`,
`surface_endo_sample.py`, **`endo_clean_recall.py`** (the clean fatigue-isolated endo harness — start here),
`surface_endo_report.py`. Dormant code: `servers/scales/s1/surface_contract.py:1865` (`BRAIN_SURFACE_RANK_MODE`).

**Papers:** HippoRAG (2405.14831, OSU-NLP NeurIPS'24); PPR≡SR (arXiv 2512.24722, Millidge 2025); SR
(Stachenfeld & Gershman 2017; Dayan 1993; eLife 80680, 78904); Echo State Property (Jaeger; ESN-as-SSM
arXiv 2509.04422); Temporal PageRank (Rozenshtein & Gionis); incremental/dynamic PPR (Bahmani VLDB; EvePPR);
active inference (Friston; arXiv 2207.09897); fast weights / linear-attention-as-RNN (Schlag 2021;
Katharopoulos 2020); diversification (MMR; DPP).

---

## 15. The integration thesis — all cues, typed operations, one coherent (non-fragile) architecture (Tom, 2026-06-18)

**Thesis (Tom).** The §13b negatives are negatives of *isolated, uniform* levers — **not** of the signals
themselves. Every brain capability is an **underutilized recall**: edges (typed + described), reasoning,
situation, question, episodic, main content, previous turns. The mandate is **MORE, not less** — the
failure so far is that we haven't found the **coherent, non-fragile architecture** that uses them
*together*. Edges are **non-negotiable**: *"there is NO way edges won't be taken into account; all edges
have types and descriptions."*

**Biological frame — imitate the final *result*, not the time-mechanism.** Bio recall is time-based +
spreading activation; we don't replicate the dynamics, we imitate the *outcome* of its operations on a set:
- a node **keeps firing** (stays active / reinforced) — `extends`/`grounds`/`depends_on` convergence
- a node **blocks** another (inhibits) — `corrects`/`supersedes`/`contradicts` suppression ← the "blocker"
- a node **fades / passes-through** (routes a message but doesn't stay) — transient, not surfaced
- → activation + inhibition + gating, **selected by the relationship between nodes**, not one global op.

**The key move: the indicator between two nodes IS the edge, not cosine.** node↔node cosine is
anisotropic-flat (0.5–0.7, ~never negative) → it cannot signal opposition. The **typed edge** is the
operation-selector: reinforce / inhibit / dedup. So edges return — **not** as a global PPR diffusion
(falsified: smears, 71%-noise, ≤cosine) but as **per-pair operation-selectors within the retrieved set**.

**Empirical guardrails (build on evidence, not hope):**
- single cosine levers (primary, multi-vector, episodic, followup, question-field) all **≤ cosine 22%** — flat embedding;
- global PPR (noisy AND clean-semantic) **< cosine**, α-decays, no hit@25 gain;
- *uniform* fusion noise-bumps the easy hits; *hard-AND* gate annihilates recall;
- **first positive:** *operator-cue* multiplicative 3-lane fusion **beat** cosine (26% vs 23% hit@5; 41% vs
  28% hit@25) while *anchor-cue* fusion lost → the lead is **conditional/typed combination, not single ops or uniform fusion.**

**Architecture hypothesis (reconciles thesis + guardrails):**
- **lanes = all cues** (cosine, FTS, question, situation, reasoning, episodic, content, prior-turns) — each a node-score or a graph;
- **FIND = additive union (RRF)** — cast wide, recall (carries sparse/operator cues);
- **RANK = multiplicative gate + edge-operations** — sharpen via agreement (precision) + inhibition (suppress corrected / dedup similar) = the bio block/fade;
- **conditional/typed** — lane-weights + operator chosen by cue-type/intent (the router), not globally fixed (operator-question vs anchor-prose want different blends).

**Optionality — the menu to test (each non-fragile in isolation, composed incrementally):**

| option | what | targets | status |
|---|---|---|---|
| edge-ops within pool | **✓ INHIBITION (suppress+dedup) works + hit-preserving (anchor 21→24, buried 0→12%); reinforce/activation HURTS (re-hubs)** | Disease A (anchor) | ✓ tested |
| cue-type-gated router | operator→**fusion lanes** (26/23), anchor→**edge-inhibition** (24/21) — complementary, each source's winning lever | the conditional | **next build** |
| more lanes | situation / reasoning / content / prior-turns as score-lanes | recall | open |
| RRF-find + mult-rank | proper two-layer (wide → sharp) | both | open |
| intent router | classify cue → pick lanes + operator | coherence | open |
| signed PPR | activation + correction-inhibition (node/edge-level) | Disease A | open |
| discriminative embedder | lifts every lane's signal (precondition) | both | bench `06fe1193` |

**The tension to hold (not either/or):** the embedder is the wall for *abstract* relevance (the precondition
that makes every lane sharper) **and** the coherent typed multi-cue architecture is the untested upside. The
architecture pays off *more* once the embedder lifts the lanes — so build it **embedder-ready**, and stop
testing levers in isolation; test them as *composable, conditional* parts. Nodes: `3e2c73eb` (flat ceiling),
`449a13e6` (cheap-levers-by-disease), `95d60d18` (intent router), `7f9a35e8` (noise-filter PPR).

---

## 16. RESULT — the realizable signal is ~3× what we use; the unlock is a SELECTOR ▶ NEXT STREAM (2026-06-18 eve)

Reverse-engineered every gold against ALL realizable state-cues × every mechanism (`endo_reverse_regress.py`
over `state_cues.json` from `endo_state_cues.py`). The finding that reframes the whole arc (node `8bcc8c96`):

- **Realizable UNION = 53% hit@5 / 64% hit@25** vs **19% / 36%** for cue×primary alone (what production uses).
  best-of-union ≫ best-single ⇒ the cues are **strongly complementary**. **We extract ~1/3 of the realizable
  signal — on the *current flat embedder*.**
- **Partition:** 53%/64% realizably-reachable · **22% oracle-only** (only `next_move` surfaces it → needs a
  predictor) · **14% unreachable** (encode-gap, §11.1 bridges).
- **Matrix:** `_primary` is the best field for *every* cue (content/situation/question/title/reasoning all
  underperform → **multi-field is a dead end; the lever is multi-CUE**). Realizable cue ranking (hit@5):
  **cue 19% > prev_operator 12% > recent_context 8% > prev_anchor 5%**. Oracle `next_move` 39%/64% (~2×).
  **Graph used right:** gold is 1-hop edge-adjacent to an in-context node **19%** of the time (distinct,
  realizable, cheap — local from context, NOT global diffusion). FTS 6% / episodic 6% weak alone.
- **Reframe of "embedder is the wall":** with the *same* embedder, the realizable union holds ~3× the signal;
  **the near-term wall is the SELECTOR**, and a better embedder lifts the ceiling on top. Strongest evidence
  for Tom's "more, not less."
- **The catch:** 53% is the best-feature-*per-gold* ceiling; *naive uniform* fusion dilutes back to ~cosine
  (the fusion lab). The gap between 53% (ceiling) and ~22% (naive fusion) **IS the selector problem.**

**▶ NEXT STREAM STARTS HERE — build the learned SELECTOR.** Fit a regularized logistic / small ranker on the
per-(cue, candidate-node) feature vector (cue×field cosines + FTS + episodic + graph-1hop — already computed by
`endo_reverse_regress.py`) → is-gold, **leave-one-cue-out CV** (n=73, indicative), measure hit@5 against the three
references: **cosine 19% / naive-fusion ~22% / ceiling 53%.** Where it lands = how much of the 3× prize is
*extractable*; report which features carry weight (live: cue×primary, prev_operator×primary, graph-1hop-from-context).
Two levers proven **complementary-by-source** to fold in: operator→fusion (FTS+question lanes), anchor→edge-inhibition
(suppress corrected + dedup similar) — `ef1024df`. Parallel: **embedder bench on stream `06fe1193`** lifts the ceiling
underneath all of this (and is the cause of this session's live-recall timeouts). Harness: scorer
`endo_baseline_recall.py:score_one`, frozen gold `endo_corpus/endo_gold_corpus.json` (73 cues), IsolatedBrain +
daemon-idle. Caveat: gold was discovered via nomic cosine/FTS lenses (`gold_lens`) — don't re-introduce that bias.
This §16 supersedes the §13/§13b "NEXT STREAM" markers.

---

## 17. Honest caveats + the edge-seed lever (addendum, 2026-06-18 eve)

**De-bias §16's headline BEFORE building on it (next stream's #1 check).** The 53% is (a) a **best-feature-
per-gold CEILING** — an oracle over features; a realizable selector that must *guess* the right cue×mechanism
could collapse toward naive fusion (~22%); and (b) **subject to gold-discovery lens-circularity** — the gold was
discovered via nomic cosine/FTS lenses (`gold_lens`), which are *also* features in the union, so their
contribution to 53% is inflated (same trap as `62052d67`). The honest number = the union's reach on gold **NOT**
discovered by the lens-features — **uncomputed; run the lens-departition first.** Also: n=73 (LOO-CV overfitting
risk); graph-1hop=19% may be leakage (in-context nodes were surfaced by the same recall path). **So: the selector
is a hypothesis to TEST, not "the unlock"; 53% is an optimistic upper bound.** Treat §16's framing as the
predecessor's *interpretation*; the per-feature hit@5 matrix is the *measurement*. (This session's confident calls
were overturned repeatedly — PPR/noise, question-field, "embedder is the wall" — so the next stream should be the
skeptic, not the believer.)

**The edge-seed lever — peer `06fe1193`'s HippoRAG finding, delegated to them.** HippoRAG's single biggest gain
(+20.9 recall) was **query→EDGE seeding** (seed the walk from query↔triple matches, reset ∝ cosine), NOT query→node.
We never tested it — every PPR/seed arm here seeded from query↔node. **Substrate is ready:** `edge_relations.embedding`
(via `compose_edge_text(relation, description)`) is populated for **91% of semantic edges** — testable now, no backfill.
A/B (delegated to `06fe1193`, on this corpus): query→edge-seed vs node-seed, hit@5/@25 by source/bucket — **does
edge-seeding beat node-seeding on the *current flat* embedder (additive-now), or is it gated on the embedding upgrade?**
If it wins it's an **edge-seed lane** in the selector. NB this **re-opens the PPR-negative** (`89583d50`/`7f9a35e8`):
those arms may have failed on the *wrong (node) seed*, not because diffusion is useless.

**RESULT — `06fe1193` ran it (their lever, my substrate): FALSIFIED in the simple seed-ranking form.** Edge-seed is
**dominated, not complementary**: hit@5 **5%** / hit@25 **11%** vs node-seed **23%** / **37%** (recall@5 5% vs 21%,
nDCG 0.04 vs 0.18). 67% of gold IS edge-reachable → it's a **ranking failure, not reach** (hit@25: edge rescues 4
node-misses but loses 23 node-hits). Anchor cues catastrophic (0% hit@5), operator less bad (10%). WHY: edge
embeddings are the *same flat nomic*, and `compose_edge_text` (relation+description) is short/abstract — it describes
the *relationship*, not content — so it's **flatter than node-cosine** and misaligned with topical cues (`324ea730`
predicted exactly this). HippoRAG's +20.9 had preconditions we lack (NV-Embed-7B + multi-hop + entity-bridge KG +
PPR-diffusion *on top*). **So edge-seed is NOT additive now — gated on (a) a sharper embedding AND (b) PPR-diffusion
on top, not bare seed-ranking.** Only ember: the 4-cue hit@25 rescue. Post-upgrade re-tests still open: edge-seed +
PPR-diffusion, and edge-seed on a sharper embedder. Script: `emb_bench/edge_seed_ab.py` (06fe1193's worktree,
uncommitted). **Net: reinforces "embedder is the wall" + selector-first — graph/edge lanes are gated on the embedding
upgrade.** This closes the edge-seed optionality row as ✗-in-simple-form / re-test-post-upgrade.

**Cross-stream convergence (the morale frame).** The peer's HippoRAG read and this eval land on the **same stack** —
*(1) a more discriminative embedding (bench `06fe1193`) sharpens the seeds; (2) query→edge seeding + (3) a
recognition/selector gate make the graph walk pay off* = **HippoRAG-2's recipe.** We are **not architecturally behind —
a hand-curated HippoRAG running on a weak retriever + node-only seeding.** The pieces are addressable and additive; no
single one is the silver bullet (the embedder is doing a lot of the limiting — oracle caps 42%). Open nodes: `8bcc8c96`
(regression), `117b6ad9` (handoff), `328d2ac3` (Tom's thesis), `ef1024df` (edge-inhibition), `62052d67` (lens trap).

---

## 18. LAF — the Layered Activation Field: grounded cue→layer catalog, the algebra, the MVP (Anchor + Tom, 2026-06-22)

**What this is.** §15's integration thesis ("all cues, typed operations, one coherent architecture"), now
(a) **grounded against the actual scoring code**, (b) given a composition **algebra** + a coherence
**grammar**, and (c) reduced to a measurement-first **MVP**. LAF = recall as a shared **activation field**
over nodes, shaped by a stack of bounded, **log-additive**, control-safe operators, read out at the end.
Retrieval is the *readout*, not the mechanism. It reconciles §16/§17: the learned selector isn't replaced —
it sits ON TOP as "learn the layer weights," but only after the parity harness + clean gold + control-gate
exist (the missing foundation the §17 caveats kept hitting).

**18.1 The grounding diagnosis (`brain_recall.py` STEP 6, mapped 2026-06-22). The headline: the graph is DARK at scoring time.**
- **WIRED:** cosine over 6 field-groups (title 1.0, blend[title+content] 0.85, high_meta[situation, quotes]
  0.70, other_meta[reasoning, correction_pattern, source_context] 0.40, edge_context[edge *descriptions*,
  per-node, offline-backfilled] 0.55, **question 0.90**); FTS5 lexical; idf2 title boost (**additive**); session fatigue (mult `×(1−r)`,
  `r=count/(count+K)`, `K=10/(1+deg/10)`); **critical (mult `×3.0` — 100%-FP slot-squatter, `00f3b008`)**;
  situation (**additive** `+0.2·sim` when sim≥0.30); context-mismatch (mult `×0.7`); project (pre-filter).
- **COLLECTED-BUT-DORMANT:** recency(created_at), last_accessed, access_count, confidence —
  `recall_scoring.py:unified_score()` is built but **deferred because the multiplicative stack regressed
  R@8 by −10pts** (the receipt that justifies log-additive + a control-gate).
- **ABSENT from scoring:** spread activation / traversal (removed 2026-04-14, moved post-surface,
  `pre_response_recall.py`); communities; source_refs/episodic (trace-lane flag-OFF); **correction edges
  (metadata-only — walked for enrichment via `correction_enrich`, never scored)**; co-access Hebbian (disabled).
- **Algebra today = entangled additive+multiplicative** (the anti-pattern LAF replaces). `question` IS
  scored (0.90) → the Premium Corpus "not scored" claim is **stale** (verify-before-claiming win).
- ⇒ **Today's recall = embedding + lexical + a few entangled scalar multipliers.** The entire hand-curated
  graph is unused where it could discriminate. Every *structural* LAF layer is **greenfield**.

**18.2 The unifying law (carry from `104fc874`).** `signal÷prevalence` = divisive normalization =
Bayes-optimal efficient coding (Carandini-Heeger) = IDF (self-information) = ACT-R fan = PMI. **ONE `÷norm`
operator at multiple scales:** token (idf2 ●), **cluster (community-mass ○ — Tom)**, degree (fan ◐), and —
new in 18.3 — **TIME (temporal distinctiveness)**. Several "separate mechanisms" collapse into this one op.

**18.3 Tom's reclassifications (2026-06-22) — three are structural, not cosmetic:**
1. **`+act` IS bio-grounded — the LLM is the lower perceptual cortex.** Cosine/graph are *higher* association
   functions reading LLM-extracted features. lexical/cosine = **R1 cued activation** (recognition floor,
   `9efadb3a`) over a perceptual front-end — not un-biological, the cortical-association layer.
2. **`situation` generalizes to the work-situation.** It's a cue *about the situation* → expand the
   match-target to **project + open files + libraries + task**, not just the prompt. The situation layer =
   **contextual reinstatement** (encoding-specificity / context-dependent memory): match current work-context
   against nodes' `situation` descriptors. **Unifies the `situation` field-layer with the `work-context`
   seed → one layer** — and it's the engine for "recall while I work, before tool-use, on turn-finish."
3. **Recency is RELATIVE, not absolute → reclassify `+act`→`÷norm`(time).** Salience = recency ÷
   **temporal-density of the candidate span**, not decay-vs-now. "EX.CO mentioned once long ago" (temporally
   isolated) beats many co-temporal community-siblings. = **isolation / von-Restorff effect** + temporal
   distinctiveness (SIMPLE; Brown/Neath/Chater) → folds recency into signal÷prevalence. **Hypothesis: this is
   WHY absolute-recency `unified_score` regressed −10pts — it decayed instead of normalizing-by-density.**
   Distinctiveness predicts no such regression. ▶ test in MVP.
4. (minor) communities carry **clustering algebra** (cohesion, within-community centrality, boundary-spanning)
   beyond membership. Fields: **keywords removed**; **episodes (S0 turns) are a distinct match-surface** from
   distilled quotes — usable as an episodic `+act` source.

**18.4 The coherence grammar (PROPOSED — Tom holding open pending online research on graph+function fusion
architectures).** Every cue-usage decomposes onto closed vocabularies, so the catalog is a table not a sprawl:
> `LAYER = ⟨ SOURCE, SUPPORT∈{matched,group,field}, OPERATOR∈{+act,÷norm,−inhib,→spread,⟳plast}, JOB∈{reach,reorder,clear}, STATUS∈{●wired,◐dormant,○absent,⚠broken} ⟩`

A *combination* = a row with multiple SOURCE cues. Operators trace to the research families; `÷norm` is the
proven law (18.2). The 5-operator / 3-job / 3-support / 4-status vocabulary is what makes the capture coherent.

**18.5 The layer catalog (grouped by operator, status grounded to 18.1).**

| Operator | Layer | SOURCE | SUPPORT | JOB | STATUS | bio |
|---|---|---|---|---|---|---|
| `+act` | lexical-semantic core | prompt × {title,blend} | matched | reach | ● | R1 cued activation |
| `+act` | field-surface match | prompt × {situation, question 0.90} | matched | reorder | ● | — |
| `+act` | FTS5 rare-token | prompt-tokens × FTS5 | matched(5) | reach | ● | — |
| `+act` | **situational reinstatement** | project+files+libs+task × `situation` | matched | reach | ○ | encoding-specificity |
| `+act` | episodic reinstatement | prompt × source_refs/episodes→nodes | matched | reach | ○ | CLS replay bridge |
| `+act` | prior-pull expansion | prior-pulls × edge-neighbors | matched | reach | ○ | — |
| `+act` | edge-description match | prompt × edge_context | matched | reorder | ● | (NOT traversal; per-node vector, offline-backfilled) |
| `÷norm` | token-IDF (idf2) | prompt-tokens × title | matched | reorder | ● | self-information |
| `÷norm` | **community-mass (Tom)** | communities + clustering-algebra | group | reorder | ○ | divisive norm |
| `÷norm` | degree/fan | edge-degree | group | reorder | ◐ (tunes fatigue-K only) | ACT-R fan |
| `÷norm` | **temporal distinctiveness (Tom)** | created_at ÷ pool temporal-density | group | reorder | ○ (was ◐ absolute) | von-Restorff/SIMPLE |
| `−inhib` | session fatigue | prior-pulls | field | clear | ● (behavior unverified, `00ed3f3d`) | habituation |
| `−inhib` | **correction-LTD** | corrects/supersedes edges | target node | reorder | ○ (metadata-only today) | LTD/anti-Hebbian |
| `−inhib` | context/superseded gate | personal_context ●, evolution_status ○ | matched | clear | ◐ | schema-forgetting |
| `−inhib` | critical (current) | critical flag | matched | reorder | ●⚠ pathological ×3.0 | — |
| `→spread` | spread activation | seeds × edges | field | reach | ○ (moved post-surface) | CA3 completion |
| `⟳plast` | co-access Hebbian | co_accessed | edge weights | reorder | ○ disabled | LTP |

**18.6 The MVP (Anchor's recommendation, 2026-06-22) — the measurement harness, NOT a feature.**
The capability every prior attempt lacked is *artifact-free measurement of an added layer* — so that's the MVP.
Three pieces:
1. **Log-additive composition engine + PARITY proof.** Re-express the wired signals (lexical, idf2, situation,
   fatigue, critical) as log-additive layers over the activation field; prove the engine reproduces production
   rankings on a control set. → the control baseline; **untangles the mixed algebra + de-pathologizes
   `critical ×3.0` into a bounded log term — production value even if LAF goes nowhere.**
2. **3-case instrument: hand-built TRUE gold + frozen control set.** 3 cases spanning reach / reorder /
   work-context; the true gold = every node that *should* surface (not teacher-minted). The control set IS the
   control-safety gate made concrete. (Avoids §17's lens-circularity by hand-building, not lens-discovering.)
3. **ONE dark-graph layer, chosen BY the cases** (bottom-up, not top-down): study the 3 through the engine,
   add the dark signal that would've surfaced each true gold (community-÷norm / correction-LTD / situational-
   reinstatement / temporal-distinctiveness), measure marginal lift + control-safety.

**Pass/fail:** the new layer lifts the 3 true-golds' rank **without scrambling the control set.**
**Why this & not alternatives:** vs build-all-layers = the spray-and-pray we reject (unmeasurable); vs
pick-best-layer-top-down = violates bottom-up + can't separate lift from artifact without the parity baseline;
vs research-architecture-first = **the MVP is research-INDEPENDENT** (parity engine + 3-case instrument +
control-gate are needed under *any* fusion architecture) → **build it WHILE researching architectures.** The
selector (§16) sits on top later as "learn the layer weights" — the MVP is its missing, artifact-free foundation.

**18.7 RESULT — piece 1 built + PASSED: log-additive recomposition is control-safe (2026-06-22).**
Built `eval/laf/log_additive_scorer.py` (pure engine) + `eval/laf/parity_harness.py`. **Non-invasive** — never
touches the recall hot path: reconstructs base/critical/mismatch from what `recall()` already returns and backs
out the hidden additive layer (idf2+situation, post-multiplier) as a residual. Ran over the 30-query control set
(`eval/oracle_audit/control_gold_result.json`, 750 candidates) on an IsolatedBrain copy.

| KPI | production | log-additive | verdict |
|---|---|---|---|
| extraction faithfulness (residual health) | — | **750/750 ok, 0 negative, 0 oversized** | extraction proven faithful |
| top-5 overlap | (ref) | **0.987** | ~identical surfaced top-5 |
| Kendall tau | (ref) | **0.927** | strong rank agreement |
| essential-gold @5 | 0.661 | **0.678** | +1.7pp — no regression (slight lift) |
| essential-gold @25 | 0.867 | 0.867 | identical |

**Reading.** Recomposing production's entangled `(base+idf2)·C·M + situation` as independent log-additive layers
`base·C·M·(1+additive/base_ref)` preserves the surfaced ranking and does **not** degrade essential coverage —
it nudges it up (the additive boost was slightly over-helping low-base nodes; multiplicative tempers it). **The
log-additive algebra lock (§18.4) is empirically validated as control-safe.** The residual back-out (0 anomalies
across 750 diverse candidates) means base+idf2+situation+critical+mismatch FULLY account for production scoring —
nothing unmodeled. **Honest scope:** this measures RE-RANKING within production's returned top-25 (the
control-safety baseline), NOT reach — top-25 overlap=1.0 is structural; n=30. Testing a NEW layer's effect on
reach (piece 3) needs scoring the fuller pre-cut candidate pool (bigger limit, or a gated `_feature_vector`
expose on recall). **Next: piece 2** — 3 hand-built true-gold cases (reach / reorder / work-context).

**18.8 — Baseline on the endo corpus + the reach=reorder unification (2026-06-23).**
*Pivot from §18.7's "hand-built 3 cases":* Tom — reuse the existing frozen gold, don't re-mint. The endo 73-cue
corpus IS in-tree (`eval/oracle_audit/endo_corpus/`, full harness alongside). Baseline ran
(`eval/laf/baseline_review.py`, reusing `endo_baseline_recall.score_corpus`): **hit@5 19% / hit@25 33% /
nDCG@5 0.16** — reproduces §16's recorded number; self-checks clean (fatigue-isolated, cutoff, no-truncation).
design weakest (14%), compositional best (27%). Per-case review (gold + `teacher_why` + live top-5) in
`eval/laf/baseline_review.md`. First two cases reviewed with Tom confirm the **gold is narrow** — on
Disease-A cases the surfaced top-5 is often genuinely relevant while scoring hit@5=0, so relevance-rated@5
will read above 19% (the "thoughts not answers" frame; the surfaced cluster is a *good pull* the gold-only
metric calls a miss). Silver = `gold_helpful`, already in the corpus.

**Reach dominates ranking:** ~half the cues are flat MISS (gold not in the 120-pool at all); buried-in-pool
(rank 6–25) is the smaller slice. hit@25 33% vs hit@5 19% ⇒ ~14pp sits in-pool mis-ranked (the cheap reorder
win); the rest is reach. *(Superseded — §18.12/§18.15: much of this "reach gap" is pipeline-burial + field-fusion-recoverable; read-side reach actually saturates ~54–58%, and the lever turned out to be RANK, not reach.)*

**THE UNIFICATION (Tom, this session): reach and reorder are ONE mechanism — *activation too low* — with two
entry points.** The reach/reorder split is an artifact of the pre-cut pool; in a true field (activation over
the whole graph) every node has an activation and a layer that adds activation moves it up regardless of where
it started. **Load-bearing caveat** (brain already proved it, `e75dcf6d`/`3e2c73eb`): on a flat embedder NO
cosine-reweighting breaks *either* disease — the oracle caps at 42% (can't separate twins → the reorder
ceiling), and a selector over cosine-features can reorder but *can't reach* a cosine-far gold (scored ~0). So
both diseases hit the *same* cosine ceiling and both are broken by the *same family* — the **structural layers**
(graph-edge expansion, community, episodic; + lexical idf2 for reorder). Only the entry differs:
- **reach** = structure as **activation** — light a dark node (gold is 1-hop from an in-field node, co-temporal, or co-community)
- **reorder** = structure as **discrimination** — separate near-twins (community-mass ÷norm, edge corroboration)

Same field, same math, same lever. Not "two diseases, one architecture" — **one disease, two entry points.**
Known reach signal: graph-1hop-from-in-context carries ~19% of golds (`8bcc8c96`; verify the §17 leakage flag
in the miss-analysis).

**18.9 — The method: the LAF build loop (so single-example dives don't lose the goal).**
*North star:* a new recall whose layered field beats baseline (19% hit@5) toward the realizable ceiling
(~53%, `8bcc8c96`) across all 73 cues — by the best **combination** of log-additive layers, each earned by
measured marginal lift without scrambling controls.

*The reframe that keeps us on track:* **the unit of work is a LAYER, not an example.** Examples DISCOVER and
DEBUG layers; the corpus JUDGES them. Never ship a layer on single-example evidence; never discover one from
aggregate numbers alone (they hide the mechanism). Both halves required.

- **Phase A — Diagnose at scale (once):** the miss-analysis. Classify every miss by what would reach/reorder it
  (graph-1hop / co-community / co-temporal / source-ref / encode-gap). Output = a PRIORITIZED layer menu
  ("build layer X → fixes N cases → ceiling Y") + the encode-gap residual no read-layer can fix.
- **Phase B — The layer loop (per candidate, biggest-impact first):** (1) discover on 1 example, triangulate on
  2–3 (the "same activation field" check — real failure mode, same layer helps, not a one-off); (2) build the
  layer (log-additive → composes onto the stack); (3) corpus A/B (layer ON vs OFF on top of the kept stack):
  lifts targets AND doesn't scramble controls (the control-safety gate); (4) keep / refine / kill. Marginal lift
  always measured on top of the kept stack → optimize the COMBINATION, not isolated wins.
- **Phase C — Converge:** stop when marginal lift flattens or near the ceiling; residual = encode-gap (separate workstream).

*The guardrail:* at any moment we can name (a) which layer, (b) why (which Phase-A miss-class), (c) its measured
marginal contribution. The example is the means; the layer-combination beating baseline is the goal.

**18.10 — Reverse-look + the field-activation probe (2026-06-23).**
*The reverse-look method (Tom):* for a far-miss gold, reconstruct its **encoding context** (community, `situation`,
`question`, connections) — that context IS the gold's ideal cue (reinstatement operationalized, `cfff6201`). The
gap between the *actual* cue and that encoding-state is the recall problem; reverse-look also reveals empirically
what the cue-ledger *should* be. Built `eval/laf/field_activation_probe.py` + reverse-look via `get_nodes`.

Findings (3 far-miss golds: `0f730a1c`/spread-cue, `d1d1a90c`/co_access-cue, `6a964255`+`951f3ac8`/RRF-cue):
- **The gold's encoding-HOME ≠ the cue's surfaced cluster** (surprise — it overturned my graph-1hop guess).
  `0f730a1c` (convergence principle) lives in the *Identity Rendering* community; the cue surfaces the *Spread*
  cluster — so graph/community from the surface can't reach it. **Cross-domain abstraction = the hardest class.**
- **The `question`-field hypothesis was mostly WRONG** — reaches **1 of 4** golds (`d1d1a90c`: question rank 1,
  reasoning 2, other_meta 2). For `0f730a1c` question(53)≈primary(45); for `6a964255` question(4592) is
  catastrophic. `8bcc8c96`'s "multi-field is a dead end (aggregate)" largely holds; `question` is a **narrow,
  gated** lever, not the headline. (Firewall working: I read the question *text* as matching; the cosine disagreed.)
- **Three reach classes:** (A) cross-domain abstraction (`0f730a1c`, hardest, flat-embedder-bound); (B)
  pipeline-buried-but-field-reachable (`d1d1a90c`, raw `_primary` rank 7, question/reasoning top-5); (C)
  true-cosine-far (`6a964255`, `_primary` 1556, nothing reaches).
- **Episodic is thin/dead for old golds:** `source_turn_id` null on 3/4; `traces_before=0` for Mar/Apr golds
  (`d1d1a90c`, `951f3ac8`), 57k–80k for May/June. Traces start ~April → pre-April golds can't use the dynamic
  episodic methods. The working reverse-look signal is the gold's own `situation`/`question` + community theme,
  NOT its (mostly-absent) trace anchors — sharpens the source-ref-is-sparse point even on the reverse side.

**18.11 — The fusion architecture: flexible operators (+ − / * block algo) without losing measurability.**
Tom's question: how do we build fusion flexible enough to apply different effects between layers? Answer — two
ideas do all the work:
- **In log-activation space the operator palette collapses to addition.** `*`(amplify)→`+log`; `/`(÷norm)→`−log`;
  `−`(inhibit)→`−log`; `block`→`−∞`. So `* / − block` are ALL "emit a Δlog, sum them" — **commutative,
  attributable, control-safe** (the locked log-additive algebra, §18.4). The ONE operator that does NOT fit
  log-sum is linear-`+`/OR (reach) → that's a *separate* regime.
- **Two regimes** (= `1c4cdedf`: add=OR=recall, mult=AND=precision; the RRF-find + mult-rank thesis):
  - **FIND** (membership / reach / OR): `field = ⋃ find_layers` (set-union by *membership*; NOT RRF-*ranking* — RRF dilutes, §18.13). Monotonic — adding a find-layer
    can't drop what's found. This is where the 51% reach ceiling (§18.12) is captured.
  - **RANK** (order / discriminate / AND): `score(n) = Σ rank_layers(n)` in log-space. `*///−/block` all live here.
- **"algo" is first-class:** a layer = any `fn(node, ctx) → candidate-set (FIND) | Δlog∈ℝ∪{−∞} (RANK)`. Cosine,
  graph-walk, PPR, cross-encoder reranker, learned selector — all plug in identically. The engine never inspects
  *how* a layer computes; PPR (falsified *standalone*) becomes one RANK-layer whose smear others correct.
- **Shared field, NO layer-to-layer piping** (the anti-spaghetti choice): layers each contribute to ONE shared
  log-activation; "effects between layers" emerge as the *net* (a `+` raises, a `−` dims — "high question-match
  raises it, other factors dim it" happens mechanically, no layer knowing another). Order-independent → measurable.
- **Safety guarantee:** every layer's marginal contribution is one isolable number in the log-sum, no matter how
  exotic its algo → full flexibility *without* losing measurability. The scramble-prone operators are the hard
  ones (`block`=−∞, big `−`inhibit) — they get the most control-gate scrutiny (they're today's `critical ×3.0` /
  floor pathologies). FIND-union is monotonic-safe.

**18.12 — Burial-decomp RESULT — validates §18.11 + refines §18.8 (2026-06-23).** `eval/laf/burial_decomp.py`,
73-cue corpus, essential gold, three rankers on the identical corpus:

| ranker | hit@5 | hit@25 |
|---|---|---|
| pipeline (today's `brain.recall`) | 19% | 33% |
| raw `_primary` cosine | **21%** | **37%** |
| best-field (ORACLE over all field-vectors) | **32%** | **51%** |

- **(1) The pipeline is net-NEGATIVE vs raw `_primary`** (19<21, 33<37). It **buries 5** cosine-reachable golds
  (`0363` prim7→pipeNone, `0836` prim5→None, `1491`/`0929` prim8→None) and **rescues only 2** (idf2/FTS: `0318`
  51→17, `0764` 26→20). The multi-group z-average + idf2/FTS flooding net-bury. **"Fix the fusion" is a real
  lever, not a 3-case mirage** — there's ~2pp free just dropping to raw `_primary`.
- **(2) Field-fusion headroom is large + genuinely multi-field:** best-field **51% hit@25** (+18pp over pipeline)
  ≈ `8bcc8c96`'s 53% realizable-union, independently arrived at. Reach @25-where-`_primary`-can't spreads across
  `_situation`×3, user/anchor quotes×3, title/reasoning/question×1 — **no single field carries it** ⇒ the *stack*
  is the right shape.
- **CAVEAT (the firewall):** best-field is an **ORACLE** — picks the right field per gold *with hindsight*. 51% is
  the **ceiling, not deployable**. The `_primary`→best-field gap (37→51 @25) is the *prize*; the realizable
  fraction is whatever a selector/fusion captures without hindsight (the build). Small N (73), mild lens caveat.
- **Validates §18.11:** the pipeline's net-loss IS the average-buries pathology the independent-layer model fixes
  — a weak field emits ~0 Δlog and can't drag a strong `_primary` down. **Refines §18.8's "reach dominates":**
  much of the apparent reach-gap is *pipeline-burial* + *field-fusion-recoverable*; only the residual (e.g.
  `6a964255` raw `_primary` rank 1556) is *true* reach.
- **Next:** build a *realizable* FIND-union over field-activations (deployable approximation of best-field),
  measure how much of the 51% ceiling it captures **without scrambling controls** (the control-gate watching the
  burial-prone operators).

**18.13 — Patchy FIND-union RESULT: uniform fusion DILUTES; the ceiling needs a selector (2026-06-23).**
`eval/laf/rrf_union_patch.py`, 73-cue corpus, essential gold — deployable (hindsight-free) RRF fusions vs oracle:

| ranker | hit@5 | hit@25 |
|---|---|---|
| pipeline (today) | 19% | 33% |
| raw `_primary` | 21% | 37% |
| best-field (ORACLE) | 32% | 51% |
| rrf_all | 16% | 29% |
| rrf_core (semantic fields) | 14% | 33% |
| rrf_max (each node by own best field) | 19% | 29% |

**ALL uniform fusions are WORSE than raw `_primary`** — sum, curated-sum, max all DILUTE below 21/37, none near
the 51% oracle. Confirms `8bcc8c96` on the LAF substrate: **the unlock is a SELECTOR, not uniform fusion.** A gold
strong in ONE field is drowned by nodes mediocre in MANY (sum) or by the crowd of every-node's-best-field (max);
the 51% oracle is a hindsight artifact (knowing *which* gold) no hindsight-free combine realizes. **SALVAGE:** the
union still wins as a POOL — gold in *some* field's top-25 = 51% vs pipeline's 33% → **FIND-union = reach pool
(+18pp); RANK = selector.** Kills the easy "RRF the fields" path.

**18.14 — Reach matrix RESULT: graph-1hop apparently dominates reach, fields are redundant — but graph is
hairball-suspect (2026-06-23).** `eval/laf/reach_matrix.py`, 73-cue corpus, REVERSE per-gold reach across every
signal (field-vectors + FTS + graph-1hop + episodic). **UNION reach @25 (≥1 signal) = 58%** (vs field-only 51%,
pipeline 33% — FTS+graph add +7pp).

| signal | reach@25 | UNIQUE@25 | best-for |
|---|---|---|---|
| **graph1hop** (binary reach-flag) | **24** | **3** | **14** |
| _primary | 27 | 0 | 4 |
| high_meta | 17 | 0 | 10 |
| title | 16 | 1 | 6 |
| situation / _situation | 15 / 15 | 0 | 3 |
| content | 14 | 0 | 4 |
| question | 13 | 0 | 4 |
| **FTS** | 13 | **2** | 4 |
| reasoning / other_meta | 11 / 11 | 0 | 1 / 5 |
| anchor/user quotes | 9 / 6 | 0 / 1 | 5 / 3 |
| episodic | 0 | 0 | 0 |

**Two structural findings, both gated by one caveat:**
1. **graph-1hop is apparently the biggest reach signal** (best-for 14, reaches 24, 3 UNIQUE); FTS adds 2 unique.
   Almost everything else has **0 UNIQUE** → **the field-vectors are REDUNDANT for reach** (they reach the same
   golds). First read of "layer more or less": keep `_primary` + graph + FTS; extra field-vectors add ~no unique reach.
2. **BUT the graph signal is hairball-suspect:** the edge graph is **79% co_accessed** (`d1d1a90c`'s exact
   decision — co_accessed excluded from traversal *because* it floods). My graph-1hop used ALL edges → "1-hop from
   top-25" is likely mostly usage-noise, not semantic reach. AND this **entangles finding (1):** broad-graph
   covering the fields' golds is plausibly WHY they show 0 unique — typing the graph may restore field uniqueness.

**episodic = 0 coverage** — the `recall_episodes` bridge returned no usable episodes in IsolatedBrain (a bug, not
a real zero) → untested, to fix.

**The decisive next test:** TYPED graph-1hop (semantic edges only, exclude co_accessed). Does the biggest reach
signal survive de-noising (→ graph+FTS+`_primary` is the reach union, fields redundant) or collapse (→ fields
regain uniqueness, hairball was the inflation)? **The combination principle is provisional until this resolves.**
Fix the episodic bridge in parallel.

**18.15 — Typed-graph test RESOLVES §18.14: graph's reach advantage was co_accessed hairball (2026-06-23).**
Re-ran `reach_matrix.py` with graph-1hop typed — excluding the 9 `noise`-aspect relations (`co_accessed`,
`community_member`, `emergent_bridge`, `temporal_sequence`, `dream_*`, …, via `brain.aspects.by_name('noise')`):

| | reach@25 | UNIQUE@25 |
|---|---|---|
| graph1hop (untyped, all edges) | 24 | 3 |
| graph1hop_typed (noise excluded) | **16** | **0** |

**The hairball was real.** Typing drops graph 24→16 reach and **3→0 unique** — ALL of graph's *unique* reach (and a
third of its total) ran through `noise` edges. **Vindicates `d1d1a90c`** (co_accessed excluded *because* it floods).
De-noised, **graph-1hop is reach-REDUNDANT (0 unique)** — its 16 golds are all reached by fields/FTS too.

**Corrected reach picture:** the ONLY unique-reach signals are **FTS (2), title (1), user_raw_quote (1)**. Every
field-vector AND typed-graph = 0 unique → mutually redundant FOR REACH. Read-side reach saturates ~54–58% union on
a *handful* of complementary signals (`_primary` + FTS + title/voice); the ~42% residual (e.g. `6a964255`) no
read-signal reaches → encode-gap / next-move (§16's 14%+22%, out of read-scope). **episodic STILL 0 coverage**
after the time-bound fix → genuine `recall_episodes`-in-IsolatedBrain blocker, untested.

**The underlying principle (layer more or less):**
- **REACH needs FEW layers** — `_primary` + FTS + title/voice ≈ saturate the read-reachable ~54-58%. MORE
  field-vectors or (de-noised) graph do NOT widen reach (all 0-unique). *Layer fewer* on FIND.
- **The leverage is RANK, not reach** — the field that ranks a gold best VARIES per gold (best-for: high_meta 10,
  title 6, content/question/`_primary` 4 each); uniform fusion dilutes (§18.13). So the win is a per-gold
  **field-SELECTOR within the reach pool** — `8bcc8c96`'s selector, now substrate-confirmed as the real lever
  (not more reach signals, not graph, not episodic).
- The ~42% residual is encode-side / next-move — a separate workstream, not a read-layer.

**Next:** the rank-selector over field-vectors (which field per gold) within the FIND pool (`_primary` + FTS +
title/voice). Separately: debug the `recall_episodes` IsolatedBrain bridge before episodic can be measured at all.

**18.16 — Deep-research salvage (truncated by spend limit) — the selector might not need training (2026-06-24).**
A deep-research run on LAF inspirations hit the org spend limit mid-synthesis (104 agents / 2.9M tokens). Only 3
claims survived 3-0 adversarial verification (the "refuted" ones had incomplete voting — spend-limit failures, NOT
genuine refutations — so disregard them; this is a partial salvage, not the full report):
- **WHY RRF dilutes (mechanism, arXiv 2210.11934):** RRF is rank-only — it discards score magnitude/distribution.
  A gold strong-in-ONE-signal contributes the same rank-1 weight as any rank-1, so it's drowned by nodes
  mediocre-but-present in MANY. This is the mechanistic confirmation of §18.13's measured dilution.
- **DAT — Dynamic Alpha Tuning (arXiv 2503.23013):** a per-query fusion weight α(q) for dense vs BM25, replacing
  a fixed α — set by an **LLM rating the TOP-1 result of each retriever**, normalized `α = Sv/(Sv+Sb)`, with hard
  overrides (perfect single-signal hit → route fully to it; double-zero → fallback).
- **Takeaway for the LAF selector (directly applicable):** the per-query selector may **not need a trained
  model.** An LLM-rated, per-query, top-1-effectiveness weighting (DAT-style) is cheap, query-conditioned, and
  **dodges the overfit-on-flat-embedder + small-corpus risk** we flagged — and we ALREADY run an LLM (Haiku
  surface) in the loop. This is the leading selector design to prototype. Fuller lit synthesis pending a
  re-run on ample budget (do not treat as exhaustive).

**18.17 — Selector design direction (research-informed, 2026-06-24).** Synthesis from the (truncated)
deep-research + established IR knowledge — what the rank-selector should be. NOT a verified lit report (the
workflow died at synthesis); confidence-tagged: `[verified]` survived adversarial check, `[established]`
well-known, connections are Anchor's. Three sources converge on one shape:
- **RANK = per-query-weighted MAX over fields, NOT average.** ColBERT's MaxSim (best-matching part wins,
  summed; never pooled-to-one-vector) is the principled form of our best-field oracle (§18.12, 51%) —
  averaging fields IS the dilution we measured (§18.13/15). So the RANK operator is max/best-field; the
  selector picks the weighting per query. `[established]`
- **The selector = LLM-judged per-query weighting (DAT-style, §18.16), NOT a trained model.** The Haiku
  surface we ALREADY run rates the top-1 of each signal/field per query → per-query weights, with hard
  overrides (perfect single hit → route fully; double-zero → fallback). Cheap, query-conditioned, and
  **dodges the overfit-on-flat-embedder + small-corpus risk** (§17). `[verified-DAT + connection]`
- **FIND stays lean** (§18.15: `_primary` + FTS + title/voice ≈ saturate reach); the selector ranks within
  that pool; an optional cross-encoder rerank is the precision layer after. `[established]`

**Don't-reinvent / pitfalls:** RRF is rank-only → discards score magnitude → dilutes (arXiv 2210.11934);
learning-to-rank / MoE-routing need labels + training we lack (overfit on flat/small); QPP predictors
correlate weakly with real performance (prior, not decider); HippoRAG-PPR is embedder-gated (§13b/§17). The
agent-memory landscape (Mem0/Letta/Zep/GraphRAG) is retrieve-then-rank / graph-summary — **none do a
per-query signal-selector**, so it's under-explored, not a reinvention.

**Prototype first:** Haiku-judged per-query field-weighting (DAT-style) + max-composition, measured on the
73-cue corpus against the control-gate. (Full lit synthesis pending a budget-capped re-run — direction, not proof.)

---

## 18.18 — The settling engine: recall as a converged activation field (Anchor + Tom, 2026-06-26)

The selector framing (§18.17) was **superseded** by Tom's *biological dynamical-fusion* vision: not reach→selector/sort,
but **one activation field that settles** — operator-fields push/pull on a shared per-node activation, iterate to a fixed
point, read out the settled state. Built, verified, measured this session. Files: `eval/laf/{verify_substrate,operators,field_recall}.py`.

**The engine (`field_recall.py`).**
```
base = Σₖ gainₖ · zscore(opₖ)                       # operator-experts, summed at LOGIT level, commensurate
a₀   = softmax(τ · base)                             # contractive init
aₜ₊₁ = softmax( τ · (base + λ · spread(aₜ)) )        # settle: graph corroboration feeds back
       until ‖Δa‖₁ < ε                               # ~1.7 iters, converges
readout: rank by settled a (hit@k) ; commit = α-entmax(a)   # sparsity ONLY at readout, not in the loop
```
Three verified operators (`operators.py`): **MaxSim** = maxₖ cos(q, field-groupₖ) over 6 live groups (best field wins);
**temporal-distinctiveness** = 1/(1+temporal-neighbours) on `created_at` (query-INDEPENDENT node-prior, von-Restorff —
NOT recency, NOT revised_at); **typed-graph-spread** = degree-normalized undirected flow over noise-aspect-excluded edges.

**Believability first (`verify_substrate.py`) — Tom's mandate.** Every source + operator must pass liveness ·
input-dependence (catches the recency=1.000 constant class) · invariant · independent-recompute (cosine from raw bytes,
NOT the pipeline's self-report) · baseline re-derivation, BEFORE any number is trusted. The baseline (19/33) is a
*suspect*, re-derived not assumed — reproduced **19/33 exactly**, proving the IsolatedBrain substrate faithful.

**Two bugs the method caught (not luck):**
- **edge_context dead** — a 0.55-weighted scoring group with **0 rows brain-wide, forever**. Cause: computed at
  write-time (before a node has edges) + the backfill catch-up path lacked the `_edge_descriptions` handler. Revived by
  sibling stream `e7188c02` (main `ebb58ad`): 4490 vectors, now the 6th MaxSim group. (My first diagnosis "never
  implemented" was an **overclaim**, corrected by a surfaced memory — verify-don't-assert applies to one's OWN findings. `a197ad0f`)
- **`NEG=-1e30` poisoned α-entmax** — a finite mask sentinel blew up the bisection bracket → entmax returned
  near-uniform regardless of temperature. The **sweep** caught it (identical results across scales 4→128 = the tell). Fix: `-np.inf`.

**The instability + the fix (the core result, `114c2deb`).**
- **Naive recurrence is UNSTABLE.** `a_{t+1} = α-entmax(base + λ·z(spread(aₜ)))` collapses to a **single wandering node**,
  never converges, at every temperature. The in-loop z-score of an increasingly-sparse spread manufactures a fresh outlier
  each step; no damping → traveling-bump (the Q3 instability, empirically hit).
- **Fix = 2+3 (Hopfield-contractive form):** **softmax in-loop** (dense, contractive — the proven-to-settle nonlinearity),
  **raw spread, no in-loop z-score** (kills the self-amplification), **α-entmax only at readout** (sparsity is the COMMIT
  rule, not the dynamics). → **settles 73/73 in ~1.7 iters.** This corrects §18.16/17: α-entmax is readout, not in-loop.

**First measured result (73-cue endo, IsolatedBrain):**
| ranker | hit@5 | hit@25 |
|---|---|---|
| pipeline (recorded) | 19% | 33% |
| raw `_primary` | 21% | 37% |
| MaxSim-6grp alone | ~20% | 34% (dilutes below `_primary`) |
| **LAF settling (2+3)** | **16%** | **38%** |

**Honest read (corrected after the T8 degeneracy finding, 2026-06-26): NOT a win — LAF currently UNDER-performs raw
`_primary` (21/37).** The only config that reaches 38 @25 is `full`/`−graph` (temporal ON), and that +1pp over `_primary`
@25 is the temporal **artifact** (see the ablation below), bought at a **−5pp @5 cost** (16 < 21). Strip the artifact
(maxsim-only, 19/34) and LAF loses to `_primary` on BOTH k. What IS validated: the field **settles** (Banach contraction,
~1.6 iters), is extensible, and is measurable. `τ` is the commit-sparsity knob (ranking-invariant — softmax monotonic);
`gain_graph` would be the @5↔@25 dial IF graph were live — it isn't (the f04f6db7 scale bug, still unfixed).

**Mechanism vs `_primary` — ABLATION RESULT (2026-06-26, the controlled measurement falsified the graph story).**
Per-operator toggle, same harness (`field_recall.py --ablate`):
| config | hit@5 | hit@25 |
|---|---|---|
| full (ms+temp+graph) | 16% | 38% |
| − graph (ms+temp) | 16% | 38% |
| − temporal (ms+graph) | 19% | 34% |
| maxsim only (ms) | 19% | 34% |
- **Graph "inert" was a SCALE BUG (resolved); the real crux is convergence-vs-commensurability** (`f04f6db7`). Toggling graph
  moved hit@k by 0 because the 2+3 raw spread was **~0.3% of base** (verified gterm.std/base.std=0.0034). After fixing the
  scale, the scale-swept ablation showed both extremes fail: **raw spread → converges but inert; per-iteration z-scored
  spread → graph MATTERS (toggling now changes hit@k) but the field never converges** (graph configs run to max_iters=20 —
  the per-iter z-score injects unit-variance every step so ‖Δa‖ never →0; anti-convergent). **Fix to try (FIXED-scale
  spread):** scale raw spread by a CONSTANT computed once (≈ std(base)/std(spread₀)), keep softmax-in-loop, so it is
  commensurate at the operating point AND diminishes as the field settles → converges; add light damping
  (`a←(1−η)a+η·softmax`) only if it still wanders. So we have NOT yet measured graph's effect on a converged field.
- **Temporal carries the entire LAF-vs-MaxSim delta** (−3pp @5, +4pp @25) — and the hardened T8 gate (2026-06-26)
  CONFIRMS it is a corpus **artifact**, not relevance: on this burst-created brain **0% of nodes are temporally
  distinctive** (every node has hundreds of neighbours within ±7d; field range [0.0008, 0.0024], CV 0.235 —
  functionally constant). The `_z` step reinflates that micro-variance into a unit-variance, **query-INDEPENDENT** noise
  prior that flips ~2 cues out of top-5 and ~3 into top-25. NOT a signal. **Action: drop temporal, or replace
  von-Restorff with an operator that has structure on a burst corpus. A shuffled-gold control would re-confirm, but
  T8's 0%-distinctive already settles it.**
- **MaxSim dilutes**: maxsim-only 19/34 < raw `_primary` 21/37. So the "+1pp @25" decomposes as `_primary 37 → MaxSim −3 →
  temporal +4 = 38` — i.e. a query-independent (suspect) prior over a diluting field. **No real relevance-driven win yet.**
- Harness trust: `maxsim-only` 19/34 reproduces the independent gate MaxSim-6grp measurement exactly.

**Extensibility — the payoff (a new cue = one z-scored term + a gain):**
- **Influence = the gain coefficient.** z-scoring standardizes every operator to unit variance, so `gainₖ` is the *pure*
  influence dial (scale already normalized out). A deliberately-weaker cue (e.g. **previous-turn cosine**) is just
  `gain_prev=0.3` vs `gain_ms=1.0` — ~4 lines (embed prev-turn, one z-scored term, one gain). Even at low gain it can still
  *decide* a node it's the only signal for (additive base + settle) — "raise AND dim in parallel".
- **Hub dampening:** (b.1) **fan-effect ÷norm** — `÷degreeᵝ` / `÷log-degree` (the §18.2 law at graph scale; trying first);
  (b.3) **aspect-weighted edges** — up-weight diagnostic relations (`corrects`, identity), down-weight generic. Compose.
- **Query-conditioned aspect weighting (Tom):** cosine(cue, aspect-meaning) → rank aspects per query → weight node
  aspect-membership accordingly. "Who are you Anchor" ranks **identity** top; "what's left to do" ranks **active_thread** top.
  Makes aspect influence query-dependent (not a fixed up-weight).
- **Per-node-type activation field (Tom):** a field per node-type, gained by cue↔type cosine — the cue routes attention to
  the relevant types. Now measurable per-field.
- **Episodic layer:** query→trace-embedding cosine→linked nodes, as another base operator. Bridge returns 0 in
  IsolatedBrain (fix first); `source_refs` sparse pre-April.

**Ceiling-lever (sibling `e7188c02`, bug `e6f0edc8`) — NOT verified by me.** nomic-Q embedder is
**batch-composition-dependent**: fastembed pads each batch to its longest text; int8-quant output shifts ~1.5–2% with
padding length → a systematic **~2–3% batch-padding noise floor on every query↔doc cosine** (query small-batch vs docs
large-backfill-batch). Common-mode (doesn't change LAF-vs-baseline) but a candidate contributor to the 0.54–0.63
flat-discrimination band — possibly part of the "flat embedder ceiling" (§13b) is a *fixable artifact*. Re-measure when fixed.

**18.18.1 — POST-REVIEW STATE + ▶ NEXT STREAM STARTS HERE (2026-06-26).**
The believability gate is hardened (`eval/laf/verify_substrate.py` + `field_recall.py`, committed this session):
- **T8 catches operator DEGENERACY** — it WARNs when <1% of nodes are distinctive, and recomputes the von-Restorff
  direction independently from `created_at` (not `1/dist`, which is circular). A plain `std > 1e-6` / `argmax>argmin`
  check is blind to a functionally-constant field — which is exactly what temporal is on this corpus.
- **T5 baseline reproduction is a HARD FAIL at ±2pp** (was a ±4pp soft-warn that never gated). Reproduces 19/33 exactly.
- **Eligibility excludes empty `created_at`** (`'' <= cutoff` is lexicographically true → undated nodes silently
  admitted); `run_corpus` prints the 63-with-gold / 10-zero-gold split and skips unembeddable cues; `alpha_entmax`
  guards α≤1. T3 recompute is sound (median|Δ|=0.0002, max|Δ|=0.0005; `embedding_similarity` IS raw `_primary` cosine).
**Gate now: 26 pass · 1 WARN · 0 FAIL** — and the 1 WARN (temporal degenerate) is the honest signal, not noise to clear.
The substrate IS faithful (cosine recompute, baseline reproduces, gold intact); what is NOT trustworthy is any
*temporal-driven* number.

**Trustworthy CONVERGED numbers (73-cue endo, hardened harness):** raw `_primary` **21/37 = the bar** · maxsim-only 19/34 ·
full LAF 16/38. **temporal's +4@25 is ARTIFACT-CONFIRMED (T8: 0% distinctive), graph is INERT (the f04f6db7 scale bug,
unfixed).** Strip the noise and the realizable engine is **maxsim-only 19/34 — which LOSES to `_primary` on both k.**
**No relevance-driven win, and `_primary` is the bar to beat.** The settling MECHANISM is validated (Banach contraction,
converges); no read-side operator built so far beats plain cosine — converging with `89583d50` (every read-side lever <
cosine; the embedder/encoding is the wall).

**✅ Q4 was EXECUTED 2026-06-27** (`reach_matrix.py`, full multi-cue bank) — and it inverted the plan: **the gold corpus
is CIRCULAR, so operators must not be built on it. ◀ CURRENT DIRECTION IS §18.19.** Q4 did confirm the
read-side-ceiling-vs-encode-gap framing (44/73 gold-not-in-pool; ceiling upstream), but the load-bearing finding is that
the gold itself is lens-minted (§18.19). The Q4 spec below is what `reach_matrix.py` implements; the Phase-B list remains
the operator backlog, now GATED on the lens-independent corpus.

**0. FIRST — Q4 reverse-engineering (Phase-A diagnose).** Don't build any operator on a hunch; derive the priority list from
the gold. Build a diverse operator-instance bank `⟨cue × support × operator × param-sweep⟩` (prompt / prev-turn / work-context /
episodic cosine; graph-spread {1,2,3-hop}; temporal-÷norm {window 3/7/30}; per-aspect & per-type cosine). Per query, compute
each gold/silver node's activation under every instance → a `[gold × operator-instance]` matrix. Then THREE analyses (NOT plain
linear regression — the regressors are FUNCTIONS with params):
  - **@25 → unique-reach + set-cover** (which operators non-redundantly reach golds; minimal covering union = the ceiling).
  - **@5 → discriminative/sparse fit** (LASSO/logistic gold-vs-noise → which operator-instances + params carry the top margin).
  - **per-context first, then aggregate** importance (respect the per-context KPI; an operator decisive for 10% of queries
    shows as high-unique-reach, not averaged away).
  - **held-out validation** (avoid lens-circularity, `155c6df6`: fit on one cue-slice, verify lift on another WITHOUT the gold).
  Output = a ranked, de-redundant operator menu **+ the read-side-ceiling vs encode-gap split** (gold cosine-far under EVERY
  cue = encode/pattern-separation residual, not a missing cue). Generalizes `8bcc8c96`/`reach_matrix`; it is the §18.9 Phase-A
  and it answers "is read-side even where the ceiling is." Runs on per-operator activations — NO dependency on the settling engine.

**Then Phase-B, in the order Q4 ranks (each earned by measured marginal lift on the kept stack, not assumed):**
- **Fix spread scaling** — FIXED-scale (`f04f6db7`), re-ablate — only worth it if Q4 ranks graph-spread reach.
- **Temporal shuffled-gold control** — is +4@25 real or artifact? If artifact, drop temporal.
- **Adaptation / sequential-recall** (attractor-net) — fatigue as the DYNAMICAL traverser (settle A → adapt → move to B): the
  engine of multi-pull + the recovery-slope KPI. Anchor's bet: ranks high — let Q4 confirm, don't assume.
- **Pattern separation** — sparse/expansion recoding as a NON-embedder route to the discrimination ceiling (if Q4's residual
  is large, this is the lever, not another cue).
- **Hub-dampening (b.1)** — fan ÷norm (`÷degreeᵝ`/`÷log-deg`) + aspect-weighted edges, for graph's @5 cost.
- **New cue fields** (`220a2808`): prev-turn (low gain), query-conditioned aspect weighting (cue↔aspect cosine →
  "who are you"→identity, "what's left"→active_thread), per-node-type field, episodic — each a z-scored term + a gain (`--ablate`).
- **Hebbian / offline plasticity** — S2 strengthens edges between co-recalled nodes (the offline auto-training).

**Theory (2026-06-26 discussion, parked for build):**
- **Attractor-network lens (vs RAG, which has no concept of any):** the borrowable ideas are adaptation→sequential-recall,
  pattern-separation, Hebbian plasticity (offline learning), and energy-landscape *sculpting* as the operator design language.
- **Runtime cost (estimate, un-instrumented):** ~200ms/recall, **~90% of it the query embed (~180ms)**; the settling field
  itself is ~10–50ms. Deploying LAF adds ~tens of ms over a cosine sort — NOT a latency concern; full-field fine to ~10–100k nodes.
- **The live question Q4 resolves:** is read-side where the ceiling is? (53% realizable union says headroom; the ablation's
  query-independent "win" + the embedder batch-padding noise-floor say maybe upstream — Q4 quantifies the split.)

**Pull for context:** `15d62a95` (math/design), `076799d0` (settling-system reframe), `da67f5aa`+`f04f6db7` (graph-scaling arc),
`220a2808` (new-cue design), `a197ad0f` (verify-don't-assert lesson), `6affa824` (five-check / silent-component-death discipline).
The discipline that held all session: **every number
traces to a verified component — verify the HARNESS, not just the data** (it caught a dead group, a finite-sentinel entmax
bug, and two of my own mechanism overclaims).

---

## 18.19 — The corpus is circular → lens-independent re-mint (Anchor + Tom, 2026-06-27). ◀ START HERE

**Q4 was executed (the `reach_matrix.py` full multi-cue bank) and it inverted the plan: don't build operators yet — the
gold corpus is circular, so every reach number measures the minting lenses, not helpfulness.**

**The verdict (`4942bd35`):** the endo gold is **100% lens-minted by {cos_cue, cos_next, fts}** (0 hand-added), **60%
cos_next-ONLY**. So the 71% ceiling + the 8-lane build menu (`25576087`) are artifacts. The corpus CANNOT rank mechanisms
(cosine-family lanes are credited circularly) and is BLIND to any mechanism that finds relevance the 3 lenses missed.
`prompt×_primary` reaches exactly its own lens's 37%. (Confirms the long-standing suspicion `d32d671a`/`62052d67`.)

**3 silent-deaths caught en route (all in the HARNESS, none in the data):** the §18.18.1 hardening was claimed-but-never-
committed (`fdc6ac66`); the T8 gate was tautological, certifying a functionally-dead temporal operator (`95a58231`); the
episodic lane returned silent-0 from a swallowed `AttributeError`, NOT a thin substrate (`8fbe480e` — 48k trace_embeddings
are present). The discipline laddered: verify the harness → verify the REVIEW of the harness (`3159ea2e`) → verify the
CORPUS (the target itself).

**THE BUILD (locked): a lens-independent gold corpus.** A blind reason-then-retrieve Opus judge (operationalizes
`155c6df6`/`5603dc33`): sees the recall-moment + the ACTUAL outcome, BLIND to what production recalled; reasons
needed-knowledge from the outcome FIRST, then wide lens-tagged search (`created_at ≤ cutoff`), classifies by REASONED
HELPFULNESS not topic-proximity, logs encode-gaps. **Durable file (committed):**
`eval/oracle_audit/gold_remint/gold_judge_protocol.md` — the method; the scale-up reuses it. The pilot **cards**
(8 engineering + 6 preference) + adjudication questions are throwaway *evidence* — the decisions they informed are
locked above + in nodes — so they are NOT committed; they live in the pilot session's scratchpad, reachable by the
bootstrap prompt's session-uuid path if a fork needs them. (Scratch is session-scoped + ephemeral; durable continuity
is the brain + this doc, never the scratch — see the `self-salvage` skill.)

**Calibration (locked via Opus-reviewer adjudication):**
- **essential = STRICT**: "its absence hurt / it would have CHANGED the move" — NOT "would strengthen" (that re-admits
  topic-proximity). The strongest essential CORRECTS a wrong assumption (`a35ed242` beats a confirming node).
- **echo-test by CONTENT-PRESENCE**: drop any node whose content is already in the recall_moment (same-session echoes),
  regardless of timestamp.
- **content-graft guard**: `created_at ≤ cutoff` is necessary NOT sufficient — a node's CONTENT can be edited post-cutoff
  (`anchor_0387`: `a1364fc9` said the OPPOSITE at the cutoff). The judge must reason about what the node SAID then.
  (revised_at path partially fixed `f3e9e16`; historical grafts remain — confirm scope.)
- **gap split**: `missing_node` vs `missing_facet`; encode-gaps carry their negative-search receipts.
- **worthiness filter**: DROP cues the judge marks not-worthwhile (same-session echo / no memory needed).

**Pilots done:**
- **Engineering (8 cues, `98c8dd53`):** overlap(old_gold ∩ new_essential) = **1/8** — empirical proof the old gold measured
  circularity, not helpfulness. Caught same-session echoes + the content-graft time-leak.
- **Diverse/preference (6 hand-authored cues, `7181615f`):** the judges nailed all 6 answers, BUT Tom's preferences are
  systematically UNDER-ENCODED as retrievable nodes — the crisp rules live in CLAUDE.md/MEMORY.md, the brain holds only the
  failure-instances. (error-handling / architecture / what-upsets-me are well-encoded; questions=curiosity / no-rush /
  cap-heavy-ops are gaps.)

**Corpus scope (LOCKED — hybrid):** engineering re-mint of the 73 (drop low-value → ~55–65) **+** a diverse slice (~20
preference/identity cues). The brain is a brain-dev monoculture, so world/personal-fact recall is untestable — the diverse
slice is bounded to identity/relationship + working-style/preference. Cues may be hand-authored by Tom (the purest
anti-circularity: human cue + human gold, no lens, no outcome-anchor needed).

**▶ NEXT STEPS (in order):**
1. **Source the diverse slice** — ~15 more identity/relationship + working-style cues (Tom hand-authors / node-anchor from
   `moment`/`reflection`/`craft_rule`/feedback nodes back to the turns where they should have surfaced).
2. **Fold calibration into the protocol file** (strict-essential, content-presence echo-test, content-graft guard,
   gap-split, search-receipts).
3. **Scale the judge** over the full ~75–85 cues — **a Workflow, ~10M tokens, EXPLICIT GO required.**
4. **Then Phase-B operators**, measured trap-proof on the new corpus, priority:
   - **query-segmentation (`9c46c291`)** — cosine each prompt segment (split `\n`/`.`) + the whole, union = the query-side
     dual of MaxSim (ColBERT late-interaction); cheap, deterministic, query-intrinsic; fixes blob-dilution.
   - **mint missing preference nodes (`7181615f`)** — encode Tom's CLAUDE.md/MEMORY.md rules as retrievable nodes so recall
     can serve "how Tom works."
   - the §18.18.1 Phase-B backlog (fixed-scale spread, temporal shuffled-gold control, episodic Method-2 encode-timing,
     pattern-separation, hub-dampening, Hebbian).
5. **Brain-integrity side-fix:** confirm the `revised_at`/content-graft path (`f3e9e16` scope) — content edited without a
   revised_at bump corrupts any time-based reasoning.

**Pull for context:** `4942bd35` (circular gold — the pivot), `98c8dd53` (engineering pilot + content-graft), `7181615f`
(diverse pilot + preference under-encoding), `9c46c291` (segmentation operator), `25576087` (the now-invalidated build
menu), `155c6df6`/`5603dc33` (corpus methodology), `3159ea2e` (verify-the-review lesson).

---

## 18.20 — The lens-independent 24-cue gold, frozen (the corpus §18.21 measures against)

§18.19's re-mint executed: a stratified 24-cue third judged by **2 blind Opus judges/cue** (reason-then-retrieve,
brain-only, conversation-window cues), classified node-level into four tiers, then an **integrity post-filter** drops
content-graft / archived / creation-leakage before tiering. **Frozen + durable:**
`eval/oracle_audit/gold_remint/frozen_gold_24.json` (**16 Gold+ / 80 Gold** across 24 cues; `frozen_cards_24.json` =
the 48 judges' full reasoning, for human adjudication without re-running). Hardened method:
`gold_remint/gold_judge_protocol.md`. Two calibration reframes from §18.19 carried in: helpfulness = **three forms**
(redirect / ground / enrich — not strict "changed the move"), and gold = **pure relevance — availability NOT subtracted**
(already-surfaced is the inhibition layer's job, never baked into gold). This is the instrument every §18.21 number runs on.
*(Method+validation provenance was the abandoned `strange-newton` stream's §18.20; the gold artifact + protocol were
preserved into this tree, the prose not — hence §18.20 is brief here.)*

---

## 18.21 — Per-field-verified LAF on the honest gold: the verdict inverts, reach is solved, @5 is the next wall (2026-06-30)

The first operator work measured on the lens-independent gold (§18.20). Method throughout: **verify every component in
isolation before trusting any number, and verify your OWN build** (it caught two of this session's bugs).

**18.21.1 — The A/B inverts. The circular corpus was hiding the operators.** Re-baselined on the 24-cue gold
(`eval/laf/gold24_diagnostic.py`, need-collapsed + tier-graded). On the OLD circular gold MaxSim/LAF LOST to raw `_primary`
(19/34, 16/38 < 21/37); on the HONEST gold they WIN — pipeline 29% · raw `_primary` 25% · **MaxSim 33% · LAF 38%** hit@5,
and Gold+@5 **4% → 17%**. hit@25 ties (~54%) ⇒ same reach, better ranking, and only honest gold can *see* a ranking gain
(circular gold scored ranking against cosine itself). Ablation (`--ablate`): the lift is **MaxSim** (multi-field); graph
**inert** (the scale bug), temporal's +pp is the **degeneracy artifact**.

**18.21.2 — The matrix instrument + reverse-engineering (the "several activations").** `eval/laf/gold24_matrix.py` persists a
per-(cue × gold-node × signal) matrix — one substrate, two views (scorecard / reverse-engineer), so a variation = a column
re-weight and a new field = a column that covers golds the others miss. Four agents reverse-engineered the *mathematical*
activation reaching each gold (pooled 98 nodes, **4/4-slice convergence** on the top operators):
- **`cos_outcome` — 65%, PREDICTOR-GATED.** The gold is the *answer the move concludes*, cosine-near the OUTCOME, not the cue
  (only ~28% are *truly* gated — most have a realizable rescue). This is "recall is prediction" quantified (`ca840441`).
- Realizable rescues: **`cos_cue`-SEGMENT × multi-field** (segment the cue — the long-anchor blob dilutes; match non-content
  fields like `anchor_raw_quote`/`title`); **graph typed-1hop from a cosine seed** (the edge `why` text often IS the need);
  **`fts` rare token**; **episodic**; plus a **dedup/consolidation** meta-op (near-dup clusters steal slots).

**18.21.3 — Per-field health audit** (`eval/laf/gold24_field_audit.py`; invariants hold: MaxSim ≥ its groups, `_primary` ==
raw-byte cosine). 8 fields stand alone (cosine family + episodic); **graph BROKEN** (blunt 0.5-weight spread, 2% standalone)
and **temporal DEGENERATE** (0% distinct) fail; `question` is weighted **0.90** in production yet is the weakest field (4%) —
a wrong assumption to revisit; the MaxSim enrich-bias is cleared (0.11).

**18.21.4 — Graph rebuilt → `relational_reinstatement`** (`operators.py`). The stored edge `weight` is an **uncalibrated 0.5
default** (`add_relation` default; decay+Hebbian touch only the excluded co-access types) — `SUM(weight)` ≈ 0.5 ×
relation-multiplicity, no relevance signal. Replaced by **conductance = cos(cue, edge.why)** using `edge_relations.embedding`
(v26, `compose_edge_text` = "[relation] description", ~109% coverage of the non-noise graph). First build under-performed
(1%/3%) — **my bug: a continuous seed dilutes.** Fixed config = **sparse top-25 seed + 2-hop**: **6% / 24%** — and **24% @25
exceeds MaxSim's 20%**: a real REACH operator (reach, not @5 rank). Edge-`why` conductance is partially flat (mean 0.499, but
14% of edges >0.6 — enough to steer). **Node-types-in-edge-text FALSIFIED** (`gold24_edge_text_probe.py`): `[stype][rel][ttype]
desc` lifts conductance mean 0.494→0.523 / >0.6 13.8%→18.6% but reach is **unchanged** (6%/24%) — cosmetic, not worth the
cascade-stale coupling `compose_edge_text` avoids.

**18.21.5 — Episodic, three-way** (`episodic_ops.py`; `trace_links.episode_node_roles` + `gather_roles`; 27 tests green).
Seeded from `recall_episodes(cue, older_than=cutoff)` → similar past moments, split by role: **`encoded+`** (created/revised
there), **Haiku-`picked+`** (surfaced & selected), **Haiku-`dropped−`** (candidate not picked, ÷prevalence so a node
*consistently* rejected across similar moments inhibits, not one cap-drop). All live (24/24 cues return episodes — the
`8fbe480e` "0 coverage" bug stays dead). Standalone modest (picked best ~9%/16%); **dropped-inhibition is SAFE** (1.5% of mass
on gold) and directionally correct (independently inhibited `e3a267aa`, also flagged noise by the inhibition judge).

**THESE ARE THREE SEPARATE LAF LAYERS — NEVER CONSOLIDATE THEM (Tom, 2026-06-30).** `encoded+`, `picked+`, `dropped−` each get
their **own ⟨SOURCE, OPERATOR, JOB⟩ row and their own gain** in the summed field (`+`, `+`, `−`) — they are not one "episodic"
term. The old single `episodic_field` (which merged surfaced+encoded into one signal) is **RETIRED**; `episodic_ops.py` keeps
them as three functions and `gold24_layer.py` gives each its own gain. And with the seed axis they **multiply**: episodic is a
**FAMILY** of layers = {encoded, picked, dropped} × {cue, prev-anchor, work-context, …} seeds — only the three *cue*-seeded
layers are built so far; that's a real gap, not a finished episodic.

**OPEN REQUIREMENT — DEFINE "MOMENT" (Tom, 2026-06-30).** Today a moment = one s0 turn, cosine-ranked, optionally a ±1-turn
window (`±1` beats single-turn, 6%/14% vs 4%/10%, so layering uses it). That is a **crude default behind a pluggable seam**
(`window` + `score_fn` in `episodic_ops.py`) — **NOT the algorithm.** We still owe a real definition of a *moment*: (a) the
**similarity metric** — how alike two moments are (flat cue↔trace cosine is the placeholder), and (b) the **boundary** — what
turns/nodes are *included* in one moment (a single turn? a topic-bounded span? a settling neighbourhood?). Until this is
defined, the whole episodic family runs on a stub. This is the next design call on the episodic side, Tom's to make.

Separately: an **inhibition anti-gold** protocol (`inhibition_judge_protocol.md`) + a 2-cue pilot exist — the measurement
target for the `dropped−` layer.

**18.21.6 — LAYERING result (`eval/laf/gold24_layer.py`) — the payoff + the next wall.** One summed z-scored field, all-fields
verified:

| config | hit@5 | hit@25 | brought | lost | reinforced |
|---|---|---|---|---|---|
| maxsim (base) | 14% | 21% | — | — | — |
| + graph | 7% | 27% | +8 | −3 | ↑2/14 |
| + episodic | 15% | 26% | +9 | −3 | ↑6/14 |
| + both (full) | 12% | 28% | +9 | −2 | ↑5/15 |
| + both (lighter aux gains) | 13% | **29%** | **+10** | −2 | ↑5/15 |

- **Reach works: 21% → 29% @25 (+8pp)** — and the decomposition proves it (**brought ≫ lost**, plus **reinforced 5–6**),
  the first clean confirmation of the one-field "more, not less" model: overlap earns its place by *both* bringing misses *and*
  raising existing gold. Episodic reinforces best.
- **It costs @5 (14% → 12–13%).** The aux fields are REACH operators — graph alone halves @5 (14→7) while adding the most @25.
  **Lower aux gains** recover @5 (13%) and keep the best @25 (29%): gains matter, heavy aux over-injects top-5 noise.
- **The shape of where we are:** the stack **solves reach** (the 50%-cosine-far problem). The open problem is **converting
  reach → top-5** — a ranker/selector over the now-real enriched pool (`8bcc8c96`'s selector, finally with substrate under it).

**Files (this session, `eval/laf/`):** `gold24_diagnostic.py` (A/B + miss-diagnostic), `gold24_matrix.py` (the matrix
primitive + episodic merge — superseded by `episodic_ops.py`), `gold24_field_audit.py` (standalone health gate),
`gold24_graph_audit.py` + `gold24_graph_probe.py` (graph rebuild), `gold24_edge_text_probe.py` (node-types falsified),
`episodic_ops.py` (the 3-way), `gold24_episodic_audit.py`, `gold24_layer.py` (the layering). Operators: `operators.py`
(`relational_reinstatement`, `build_edge_conductance`). Substrate: `servers/scales/s1/trace_links.py` (`episode_node_roles`,
`gather_roles`). Gold: `eval/oracle_audit/gold_remint/frozen_{gold,cards}_24.json`.

**▶ NEXT (in order):** (1) **gain-sweep** the aux gains for the @5/@25 operating point — and treat the three episodic layers as
**independent gains** (`pick`,`enc`,`drop`), never one "episodic" knob (Tom). (2) **the @5 ranker/selector** over the enriched
pool — the real lever now that reach is solved. (3) **DEFINE "MOMENT"** (the open requirement in §18.21.5) — the similarity
metric + the moment boundary — *then* expand the episodic **seed axis** (prev-anchor / work-context) so episodic becomes the
full role×seed **family** of separate layers. (4) measure the `dropped−` layer against the **inhibition anti-gold** corpus at
scale. All bounded by N=24 → grow the corpus (the remaining ~49 engineering + 22 diverse cues) before any number is treated as
more than directional.

---

## 19 — The trained engine: phased ladder, every rung deployable (Tom + Anchor, 2026-07-01/02) ◀ NEXT ARC

**The frame (settled 2026-07-01, nodes `cf7d5773`/`25a23312`):** LAF is an attention mechanism over persistent
memory — graph=weights (trained by living), layers=heads (hand-built lenses), gains=head-mixing (the trainable
part), field=one pass's activations. Tom's direction: reverse-engineering DISCOVERS mechanisms; the combination
is LEARNED from the production flywheel, never hand-tuned. The flywheel already exists: `outcomes_per_candidate`
≈ 4,650 turns since 2026-04-19 — each a labeled (moment, ~25 candidates, picked 3–5, dropped ~20) decision, plus
next-turn usage as the debiasing label. Judge-dropped candidates = hard negatives (failed as a recall-time layer,
gold as a training-time signal). Fitted values from the literature are priors, not inventions to re-make:
SYNAPSE/HippoRAG2/PAM parameter node `7e762e54` (fan(j) division, T=3, δ=0.5/S=0.8, PPR damping 0.5, synonym τ=0.8,
familiarity-normalized conductance `w−E[w]`, fusion λ≈{0.5 cos, 0.3 activation, 0.2 global prior}).

**The mixing math (node `d076d946`):** `score(n) = Σₖ gₖ(φ(q))·fₖ(n|q)`. Level 0: gₖ constant (20 params).
Level 1: gₖ(q) = gate over situation features φ(q) — discrete classes (C×K ≈ 120 params) or softmax(W·φ(q))
(K×dim(φ) ≈ 320–640). Include per-field peakedness (max z-score of fₖ on q) in φ: a field flat on this query
self-silences — which is ALSO the cold-start story (empty fields → gains → 0 → graceful collapse to cosine).
Gate only pays over DECORRELATED fields (oracle probe `1b5b18fc`: routing over same-query arms ≈ worthless;
the 53%-union prize lives on query-side seeds).

**Current verified state (all on the 24-cue lens-independent gold, need-collapsed):** base maxsim 14/21;
+graph-beam 13–14/24 (+7-8 brought); +pick+enc episodic **16/28** (best); drop− OUT (direct ablation);
temporal OUT (burst corpus); graph redundant on top of episodic at static gains (re-check post seed-axis).
Substrate: unified `nodes_for_traces` join (+dropped role), dict `gather`, `EdgeIndex` + cutoff-masked
`edge_cos`, `laf_metrics.py` shared metrics, gold artifacts in `eval/oracle_audit/gold_remint/`.

### The ladder — each rung gated, deployable, reversible

**P1 — Ship the measured winner. ✅ DONE 2026-07-02 (flag armed; LIVE at next daemon restart).** Shipped
composition: `z(maxsim) + 0.5·z(pick) + 0.3·z(enc) + 0.5·z(idf_title) + 0.5·z(sit)`, sigmoid-squashed,
UNCAPPED episodic (the 500-cap deleted structurally at zero quality cost — uncapped hurt only the BARE
stack; idf+sit anchor the top-5: composition is non-additive, insight `cd74b974`), gains via K-store
interaction `recall_laf`. GATE PASSED through the real recall path (`eval/laf/p1_gate.py|.md`): 16%/23%
need@5/@25 vs champion 11%/17%, p50 728ms vs 1650ms (2.3× faster), frame_replay A/B neutral-to-better.
Engine `servers/recall_laf.py`: field registry (one entry + `gain_<name>` key per lane), incremental
staleness-keyed caches, per-candidate `_laf_fields` z-score telemetry on every result — the P2 production
feed. 10 flag-guarded sites in `brain_recall._recall_impl`; flag off → champion bit-identical. Flip line in
`hooks/scripts/brain-env.sh`; rollback = remove + restart (milestone `63012d59`: armed-not-live state +
post-flip watchlist). FTS lane floods at static gains and the corpus has ZERO entity cues to credit it
(gold-growth gap; P4 peakedness gate is its way back); temporal is corpus-blind; the 58-need residual =
the P5 build menu, node `9f053861` (action-anchored rules first).

**P2 — The dataset walker (the foundation asset).** Walk all traced turns; per (turn, candidate) emit
{φ(q) features, f₁..fₖ(n|q) computed AS-OF the turn (replay fidelity = the gold24 harness discipline: node
eligibility, edge created_at masks, older_than episodic), label picked/dropped, label used-next-turn}.
GATE (not deployable, but a hard win): leakage audit per field ("can this feature know the label by
construction?"), gold-24 cue turns/sessions EXCLUDED by id, April-gap rows handled, replay sanity check —
offline static-gain scoring of walker rows reproduces the live ranking. ~2–3 days reusing the probe machinery.

**P3 — Fit Level-0 gains.** Pairwise ranking loss (picked > dropped within a turn's candidate set), L1,
~20 params, numpy/sklearn, trains in seconds. GATE: beats hand-set gains on held-out gold-24 AND time-split
(train April–May → validate June; drift measured, not assumed). Deploy: gains are config — flag flip.
This rung validates the whole loop end-to-end (data → fit → deploy → measure) at minimum blast radius.

**P4 — Level-1 gate.** Add φ(q) (aspect cosines, query shape, FTS-hit, per-field peakedness) → gₖ(q).
GATE: nested comparison — ships ONLY if it beats Level-0 on held-out (gates on faith are how per-situation
weights become worse than global). Deploy: config. Cold-start rule ships here: fits apply only above a
minimum-data threshold (~500 labeled turns); below it, literature defaults + self-silencing.

**P5 — New fields as columns (parallel, from P2 onward).** Seed-axis fields FIRST (prev-anchor, work-context,
segment particles — the decorrelated inputs where routing pays), then ACT-R base-level usage prior
(B=ln(Σt⁻ᵈ) over ACCESS events — dodges the burst-corpus degeneracy that killed creation-time temporal),
PAM familiarity-normalized conductance in `edge_cos` (one line; targets measured mean-0.499 flatness),
FTS∪maxsim beam seeds, node-specificity in spread. Each field: leakage audit → column → refit → per-field
ablation on gold. Post-training economics: a field costs one column + refit, not a hand-tuning session.

**P6 — The moment encoder (answers "define moment" empirically).** Contrastive training from the walker:
two moments are similar iff overlapping nodes were useful in them (behavioral metric, not text cosine).
Small projection head on nomic. GATE: episodic field standalone + in-stack beats the text-cosine stub on gold.
Deploys as the episodic layer's internal `score_fn`. This is the trained answer to the open moment-definition
call (window/boundary seam stays in `episodic_ops.py`).

**P7 — Node-embedder fine-tune (the keys).** Contrastive: picks/used = positives, judge-dropped = hard
negatives. Attacks the flat-space wall (0.54–0.63 band) on OUR distribution. GATE: the ENTIRE ladder re-runs
on the new space (biggest win potential, biggest re-validation cost) — nothing above survives by assumption.
Deploy: re-embed migration, staged.

**Ongoing, not a rung:** grow the gold (remaining ~49 engineering + 22 diverse cues); RE-MINT blind gold
periodically on fresh moments — this is the anti-feedback-loop mechanism, not one-time infra.

### Risk map = acceptance criteria (2026-07-02; each guard is a checklist item, not advice)

1. **Loses-to-cosine** → champion/challenger + runtime fallback; worst case IS today's behavior.
2. **Feedback loop** (model trains on its own surfacing echo — the deepest risk) → next-turn-usage labels
   (surfacing-independent) + periodic blind-gold re-mint + small exploration rate in candidate admission.
3. **Cold start / 100-node brain** → peakedness self-silencing + literature-default gains + min-data
   threshold before any fit applies + NEVER transfer one brain's fitted weights as another's truth.
4. **Field poisoning** → training itself filters noise (L1 → exactly 0); the real danger is LEAKAGE —
   per-field audit at the door + per-field ablation after every fit.
5. **Bad φ(q)** → nested model comparison; gate ships only if it beats static.
6. **Eval contamination** → gold cue turns/sessions excluded from the walk BY CONSTRUCTION (P2 gate).
7. **Goodhart-on-Haiku** → pick-agreement is the training signal, never the KPI; gold + usage are the KPIs.
8. **Drift** → time-split validation each refit; canaries: judge-disagreement rate, abstention rate.

**Files/nodes to load on wake:** this section; `eval/laf/` (probes + `laf_metrics.py` + `operators.py`);
gold in `eval/oracle_audit/gold_remint/`; nodes `cf7d5773` (analogy), `25a23312` (Tom's training directive),
`1b5b18fc` (oracle/seed-axis), `7e762e54` (literature params), `d076d946`/`3be5c0df`/`152cd0db` (gates, build
scope, failure modes — encoder-written), community `dae30088`. **P1 done → START at P2** (pickup node
`f53f2ff3`: schema + per-field leakage-audit proposal first, propose→approve; walker columns come from the
production field functions in `servers/recall_laf.py` — no reimplementation; emit the 6 maxsim views as
separate columns per `maxsim_decomp.md`; `p1_gate.py` is the replay-sanity target; `_laf_fields` telemetry
accretes the same rows in production once the daemon restarts).

**⚠ P2 SUPERSEDED IN PLACE by §20 (2026-07-14):** the walker survives but its objective changed — it is now
the *moment-recognition instrument* (per-turn ingredients stored, moment shapes swept offline), not just the
gains table. §20 is the approved-shape proposal; P3+ ride on §20's table unchanged. Known wiring gap found
in the 9-day live health check: `_laf_fields` does NOT reach the trace substrate (dies at the
`cand_detail` compaction in `surface.py`) — the forward-feed leg of the old P2 assumption is broken until
that 3-line fix lands (routed to the surface-lane stream).

## 20 — Moment recognition: walker v2 + pre-registered sweep (Tom + Anchor, 2026-07-14) ◀ ACTIVE ARC

### 20.1 Objective and ship rule

Recall today matches the **prompt**; it should recognize the **moment**. The LAF cue audit (2026-07-13)
showed five of six lanes share ONE cue — the bare current message, truncated (`daemon_hooks.py:260`);
the only other query-side inputs in the whole stack are the session project (dormant, gain 0) and fatigue's
surfacing history (inhibition only). History enters the pipeline only downstream (Haiku selection) and
sideways (fatigue) — never where the field is shaped.

**The behavioral definition (the judge of everything):** two moments are the same moment iff the same
memories serve them. Every feature below is an approximation; corpora decide which approximations carry it.
Five candidate ingredients (Anchor's frame, Tom-endorsed 2026-07-13): trajectory (where this is heading),
nested timescales (exchange/thread/arc), activity mode (what the operator is DOING), what's-already-in-the-
room (attention state → inhibition as signal, not hygiene), distinctiveness (what makes this moment unusual
against the session baseline). This arc instruments trajectory + timescales first; activity-mode features
ride along in φ(M) for free.

**The moment object:** `M(t) = {turn_{t-j}, j=0..K}` with per-turn weights `w_j = decay(j)`. Per lane f:

    blend:    f(n | Σ_j w_j·q_{t-j})            — one blended query vector (cheapest)
    turnmax:  max_j [ w_j · f(n | q_{t-j}) ]     — best single-turn evidence, decayed
    turnsum:  Σ_j w_j · f(n | q_{t-j})           — accumulated evidence, decayed

K, the decay form, the composition, and per-lane participation are FITTED, never hand-set (Tom's standing
directive `25a23312`). K=0 ≡ today's shipped behavior — always the control arm.

**Ship rule (pre-committed):** a moment shape ships only if it wins BOTH measurement modes:
- **reach** (full-field, honest gold): brings gold into top-25 that K=0 misses — gold-24 + LongMemEval;
- **rank** (pool-restricted, our 5,088 judged turns): picked>dropped pairwise improvement, time-split held out.
Single-mode wins are findings, not ships. Cross-corpus disagreement escalates to Tom (values call, not math).

### 20.2 Walker v2 schema (`eval/laf/walker/`)

SQLite `walker.db` — LOCAL BUILD ARTIFACT, never committed; committed are the build script, the health
report (`walker_health.md` per build), and the gold-exclusion manifest. Substrate verified 2026-07-13:
5,088 labeled turns (Δ `additionalContext` rows with `outcomes_per_candidate`: Apr 555 / May 1,920 /
Jun 2,120 / Jul 493), ~25 candidates each, all three legs joined structurally on `s1r-{session}-{stop}`
chains via the pure `nodes_for_traces` join.

**`turns`** — one row per (session_id, stop):
- ts, chain_id, project, session ordinal features
- `op_text`, `anchor_text` — FULL texts from S0 conversation traces (never the O-row's 500-char
  `query` field; prev_operator and prev_anchor were DISTINCT cues in the reverse-regression `8bcc8c96`,
  so they stay separate columns, separately embeddable)
- `op_vec`, `anchor_vec` — 768d, stored as blobs (query-prefix convention pinned + asserted in the build)
- φ(M) activity features: tool_result density, was_correction, prompt shape (len/code-fence/question),
  turns_since_session_start, gap_seconds_since_prev_turn, files_touched (from S0 tool traces — the most
  action-anchored signal available)
**`candidates`** — one row per (session_id, stop, candidate_id):
- labels: outcome (`selected`/`dropped`), tier (`picked`/`pooled_dropped`/`floored`), fetched_by tool
  (post-07-02 rows only), `used_next_1`, `used_next_3` (S0 anchor_touched recalled/authored at stops
  t+1..t+3 — surfacing-independent debias labels)
- rank_in_pool, quality flags: node_revised_after_turn, sit_missing, label_ambiguous (short-id prefix
  collision), pre_v5 (no tool tiers)
**`cand_turn_scores`** — one row per (session_id, stop, candidate_id, j, lane_view):
- j ∈ 0..8 turn offset; lane_view ∈ 6 maxsim views + sit + idf (idf recomputed per turn-j against
  as-of title df); cosine/score computed by IMPORTING the production field functions from
  `servers/recall_laf.py` — no reimplementation (measured==shipped)
- ~5,088 × 25 × 9 × 8 ≈ 9M floats ≈ 40MB — trivial
**episodic symmetrization:** pick/enc lanes get their cue swapped from single-vector to the same M(t)
stack in the SWEEP (their trace-side moment windows already exist); no extra storage needed beyond `turns`.

**Scale note:** full-field reach mode does NOT use `cand_turn_scores` (pool-only) — it matvecs the frozen
node matrices per config inside the sweep harness. Pool-restricted rank mode reads the table directly.

### 20.3 As-of + leakage rules (per-field audit)

| field | as-of discipline | leakage risk & mitigation |
|---|---|---|
| maxsim ×6 | node eligibility `created_at < t` | revision leakage (embeddings are current-state; no history exists) — same accepted residual as the gold-24 harness, but now FLAGGED per row (`node_revised_after_turn`) so P3 can quantify it |
| pick / enc | trace mask strictly `< t`, current chain excluded | past picks are judge output (echo risk) — legitimate as FEATURE; debias lives on the LABEL side (used-next-turn) |
| idf | df recomputed over eligible titles only | current-corpus df would leak growth; replay check arbitrates the divergence from production. **REVIEW F1 (2026-07-14):** the as-of corpus is also filtered by TODAY's archive (`nodes WHERE archived=0`) — a node archived AFTER t is absent from both the candidate set and the df/n_titles DENOMINATOR, shifting live candidates' idf too. KEPT (not fixed): matches the engine's live-only design (§20.11 residual), so walker and engine-as_of AGREE — the cross-check holds. The distortion grows with S2 consolidation churn; magnitude to be counted (since-archived-candidate tally) before it feeds a fit. |
| sit | eligibility mask; missing → neutral (post-`f77c453` semantics) | Healer backfill postdates node creation — mask by enrichment ts if available, else `sit_missing` flag |
| proj | session project from the turn's session record | pre-v30 turns → null; low risk (system-stamped) |
| labels | — | 8-char short ids prefix-resolved; ambiguous → flagged, never guessed |
| moment stack | turns strictly ≤ t from the SAME session, compaction seams respected | wrong-turn reconstruction is the silent killer → human fidelity read (§20.5 checkpoint H1) |

**Exclusions by construction:** gold-24 cue turns AND their full sessions, via a one-time committed
manifest (cards carry `cutoff` but no session ids — manifest maps cue_id → session_id/stop by
cutoff+text match). April rows missing candidate detail (~57) or lacking a Δ label row: dropped AND
counted in the health report — never silently skipped.

### 20.4 Health report — mandatory build artifact

Every walker build emits `walker_health.md`. A build without a green health report does not feed a sweep.

1. **Fill-rate matrix** — every column × month (catches the "empty column for a subset" class: pre-07-02
   tool tiers, April candidates, sit coverage)
2. **Join conservation** — turns in = labeled + dropped-with-reason; per-reason counts
3. **Achieved-window histogram** — how many turns actually had j turns of history (catches the
   coverage-blindspot class: session starts, compactions, short sessions)
4. **Query-sensitivity per lane** — field values must move when the query changes (catches dead lanes
   that `std > ε` checks certify; the temporal-operator disease `0c8352f1`)
5. **Replay sanity** — offline K=0 static-gain scoring of walker rows reproduces the live `p1_gate.py`
   ranking on the gold cues (proves offline rows ≡ live scoring)
6. **Embedding spot-audit** — N sampled turn vectors re-embedded fresh and compared (catches prefix/
   normalization drift)

### 20.5 Pre-registration: sweep Q1 (approve BEFORE it runs)

**Question Q1:** does any moment shape beat K=0 (prompt-only) on BOTH reach and rank?

**Grid (H2-LOCKED 2026-07-14):** K ∈ {0,1,2,4,8} × decay {exp γ ∈ 0.3/0.5/0.7/0.9, power α ∈ 1/2,
uniform w_j=1/K} × composition {turnsum, turnmax} × **aggregation point {lane, zsum}** (Tom's field-level
blend: compose per message THEN aggregate, vs per-lane aggregation; post-sigmoid rejected a priori —
saturation compresses exactly the extremes that matter) × turn texts {op-only, op+anchor} ×
**feedback M_e {off, δ ∈ 0.1/0.3}** (running fatigue replayed from stored picked/dropped labels:
same-session prior outcomes modulate subsequent turns' scores) × lanes {maxsim, sit, idf, pick, enc} —
**episodic participates in the IDENTICAL grid** (Tom 2026-07-14: no special casing; per-message episodic =
role-map lookups against each message's vector; the old "stack-as-similarity-input" is just blend).
NOTE: vector-level blend ≡ turnsum for cosine lanes up to within-turn z (cosine is linear in the query;
per-turn norm cancels in z) — blend is therefore NOT a separate arm; recorded here so nobody re-adds it.
The uniform family separates "history helps" from "the WEIGHTING matters." Shuffle control runs on the
top-3 configs + a random sample (bounded compute, pre-registered). Results report per turn-class
(normal vs flagged-stack) — the 36%-contamination check. Winner selected by ONE pre-declared aggregate
(mean of reach-Δ@25 across gold corpora + rank-AUC Δ on time-split holdout), not per-metric cherry-picking.

**Controls (all mandatory):**
- **Shuffle control** — moment stack built from random OTHER-session turns must NOT beat K=0; if it does,
  the gain is a norm/length artifact and the config is dead regardless of its numbers
- **Positive control** — the harness must reproduce the measured ±1-turn episodic result (`9634cce9`)
  before any new number is trusted
- **Coverage control** — the empty-history subset (no previous turns) must score EXACTLY K=0

**Corpora:** gold-24 reach (report, ±4pp noise acknowledged — never sole ship evidence); LongMemEval
frozen-corpus reach (existing `eval/longmem/sweep.py` harness, variance runs); judge-label rank
(train April–May / validate June+ time split). LoCoMo joins only if legs disagree and a tiebreaker is
needed.

**Pass criteria (pre-committed):** reach: +≥2 needs @25 on gold-24 AND LongMemEval improvement beyond its
measured variance band; rank: pairwise AUC improvement on the held-out split. Both, or no ship. A
surprising result spawns a NEW pre-registered question — no goalpost moves after the run.

**What we do NOT conclude from Q1:** single-mode wins, post-hoc subgroup stories, anything about lanes
not in the grid.

**H2 FINAL LOOK — SIGNED (Tom, 2026-07-15): grid stands.** Substrate = walker v6 (machine-turn relabel
from the H1 read; GREEN 7/7, 3,736 labeled turns). **M_e pinned = 2′, surfaced-only running fatigue:**
`f ← β·f + 1[picked-into-context]`, score −δ·f in z-space, β=0.7 fixed, δ per the registered axis;
session-scoped replay from stored labels. DROPPED nodes carry NO automatic within-session signal —
being passed over once is not a relevance verdict (Tom); the cross-moment drop-RATE remains a future
P3+ inhibition lane, not an M_e reflex. Per-node only: no neighbor spread until a field operator
carries it (P5 Mexican-hat). The signed excite/inhibit form (§20.10) goes to P3 with its own pre-reg.

### 20.6 Human checkpoints (Tom) — few and high-value

- **H1 (Stage 1, one sitting):** moment-fidelity read — ~15 sampled reconstructed moments read against
  Tom's memory of those conversations. The math cannot detect wrong reconstruction; this is the highest-
  value human hour in the plan.
- **H2 (before sweep):** pre-registration sign-off — grid, controls, pass criteria (the understand-the-
  test rule `f11ae3cd`). Tom's parameter intuitions belong HERE (what to search), never mid-sweep.
- **H3 (on disagreement):** cross-corpus verdict conflicts — a values call about which distribution we
  serve, escalated with both numbers on the table.
- **H4 (before ship):** frame_replay qualitative A/B read — "are these the memories that moment needed?"
- Standing: cost checkpoint before any heavy run (LongMemEval corpus REBUILDS are expensive; sweeps over
  a frozen corpus are cheap).

**Drift guards (agreed 2026-07-14):** parameter intuitions go into H2's pre-registration, not mid-sweep
picks (researcher degrees of freedom at N=24 are unaffordable); single-recall anecdotes go to the residual
ledger and earn lane status statistically; mid-rung scope changes are batched at stage gates — the rung
finishes as registered.

### 20.7 Stages, artifacts, effort

- **Stage 1 — walker v2 build (~2–3 days):** gold-exclusion manifest → build script → health report →
  H1 fidelity read. Artifacts: `eval/laf/walker/{build_walker.py, walker_health.md, gold_manifest.json}`.
- **Stage 2 — Q1 sweep (~2–3 days, offline math):** controls first, then grid. Artifact:
  `eval/laf/walker/q1_sweep.md` (all configs, controls, verdict vs pre-registration).
- **Stage 3 — ship proposal (separate, after Q1):** live wiring rides the existing engine (session turn
  vectors are ALREADY in the trace matrix the episodic lanes scan — moment stack at recall time is
  reading vectors we hold; blend ≈ free, turnmax/turnsum ≈ K extra matvecs). Moment params live in the
  `recall_laf` interaction config — config flip to deploy, config flip to roll back. Gate = P1 discipline:
  gold-24 through real `brain.recall`, frame_replay A/B (H4), latency probe.
- **Stage 4 — P3 fit on the same table** (gains + moment params jointly), then P6 trains its contrastive
  moment encoder on the same rows. The ladder is unchanged; §20 is P2's schema + P5's first fields.

### 20.8 H2 resolutions (Tom, 2026-07-14) — walker-scope sign-off DONE

1. **K grid:** {0,1,2,4,8} approved. Tom: influence of more/fewer turns is interesting but hard to
   measure — the achieved-window histogram is the honest instrument; K=16 only via a follow-up pre-reg.
2. **Decay families:** exp + power + uniform control approved. Tom flags the cognitive-psychology echo
   (7±2 working-memory span) — worth a literature dig alongside the sweep, priors not parameters.
3. **Turn texts:** op+anchor approved (the situation is together). Registered-but-parked: contradiction /
   shift-away dynamics — when the operator (or a future LLM turn) contradicts or pivots, previous turns
   should plausibly flip from reinforcing to inhibiting. NOT in Q1; parked in §20.9.
4. **φ(M):** files_touched added (Tom independently wrote the same before reading the proposal);
   test-failure signal deferred to the residual ledger.
5. **used-next windows:** 1 and 3, no more — beyond ~3 turns the causal link dilutes.

NOTE: §20.5's sweep pre-registration gets its own final H2 look AFTER the walker health report exists
(pass criteria are cheap to confirm once fill rates are visible).

### 20.9 Parked ingredients (caught, not lost — batch at stage gates)

Tom's reminder (2026-07-14, "in case it synergizes") + session spirals, all candidate cue/field material
for P5+ or φ(M) extensions — none enter Q1:

- **arc** — the encoder's per-session focus blob; a nested-timescale cue (thread/arc layer above the
  K-turn stack)
- **communities** — offline S2 cluster nodes; membership as moment context or support-side expansion
- **selection counts / access history** — the ACT-R base-level usage prior B=ln(Σt⁻ᵈ) already on the
  P5 menu; also a candidate ÷norm (familiarity) operator
- **edges/aspects** — aspect cosines belong to φ(q) at P4 (already registered there); typed edges as
  per-pair operation-selectors remain the standing integration-thesis item
- **contradiction / shift-away turn dynamics** (from H2 Q3) — sign-flipping the decayed stack when the
  trajectory pivots; needs a pivot detector; revisit after Q1 shows whether plain decay carries signal
- **test-failure signal** in φ(M) — deferred pending a reliable parse
- **'interrupted' flag is really 'untraced'** — the class mixes real interrupts with Stop-hook failures /
  session kills / mid-turn compaction (Tom's felt interrupt rate << 10.1%); taxonomy investigation spawned
  as its own stream (task_79d02069, consult-before-execute); rename after its verdict

### 20.10 The running field — the architecture target (Tom, H2 lock 2026-07-14; node 87a6dae9)

The moment work converged past the decayed stack. The target object:

    A ← λ(Δt)·A + w_e·F_e + M_e        on every attention event e

**LAF is the recognition UNIT** (msg → settled field F_e, algorithmic, no LLM in the loop — why P1–P3
quality is the prerequisite). **A is the running field**: one activation per node, persisted across the
session in SessionContext — the state of mind as a MAINTAINED object, not a per-query reconstruction.
A moment is what the integrator holds. **Endo-recall = reading A at any time** — always-on recognition;
operator prompt / Anchor stop / tool use are just event types feeding one state (the 3-attention-moments
vision unified). **M_e is outcome feedback**: Haiku's verdicts on the previous field arrive as one-turn-
late labels — picked→excite, dropped→inhibit, surfaced→suppress. Fatigue becomes the inhibitory half of
the update rule ("running fatigue"), fulfilling fatigue-as-lane (7e9e36a7).

**Deployment economics:** each F_e is computed anyway at its own recall; the running field costs a cached
vector (~28KB/turn) + weighted adds — moment recognition at ~zero marginal live latency. Each cached
field carries its own query-side vector by construction (dissolves the walker's q_vec hand-fix).

**SPEC LINE (do not "optimize" away): the persisted field is the FULL activation vector, never the
top-K.** Below-pool mass is where reach lives (union 8bcc8c96; the retired 25-node blindness 62b04f12).

**The §20.5 sweep is this object's first parameter fit** — λ from the decay grid, w_e from compositions,
M_e's first measurement via the feedback axis. Q1's question, upgraded: does a MAINTAINED field beat a
stateless query? Stage 3 implements A in SessionContext if the answer is yes.

### 20.11 as_of — read-side time travel in the engine (design map, Tom + Anchor 2026-07-14)

**Decision: replace the walker's masked-formula replay with `as_of` support in the PRODUCTION engine.**
The eval calls the real unit with a date; every future rung (P3 fit, gold re-mints, P7 re-validation)
inherits the capability; leakage-by-reimplementation dies as a class. The walker's existing content-lane
columns become the CROSS-CHECK: engine-as-of and walker rows must agree row-level on content lanes —
two independent implementations, mutual verification.

**Scope: `LafV1Engine.scores(..., as_of=None)` — engine level, NOT brain.recall.** scores() is read-only
by construction (access marks / Hebbian / traces all fire later in `_recall_impl`), so replay needs zero
side-effect suppression. Fatigue, floors, critical boost, hydration are post-scorer production machinery —
out of as_of scope by design (session state can't time-travel; the sweep replays M_e itself).
brain.recall(as_of) is deferred until a consumer needs the full path.

**Mechanism: masks, not copies.** Caches stay current-state supersets (they already hold full history);
as_of builds per-call boolean row masks. as_of=None → no masks → the identical code path: inert by
construction, pinned by tests.

The root-function map (every LAF data touchpoint, walked 2026-07-14):

| # | root | touches | as_of treatment | new cache ingredient |
|---|---|---|---|---|
| 1 | `_full_matrix_build` / `_refresh_matrices` | node view vectors (VectorDAL) | row mask: node created_at ≤ as_of; masked rows → NaN (isfinite machinery already downstream) | `_created[row]` array (one SELECT at build, maintained on append) |
| 2 | `_refresh_titles` + `idf_scores` | node titles (NodeDAL.title_rows) | as-of df + n_titles via per-token sorted creation timestamps + bisect (the walker's invention, moved into the cache) | `_token_created` sorted lists + `_all_created` sorted array |
| 3 | `_refresh_projects` / `_project_field` | project labels (nodes + kv) | same row mask as #1 (labels are creation-stamped) | none |
| 4 | `_refresh_traces` (episodic matrix) | trace vectors (TraceDAL.event_vector_rows, arrives created_at ASC) | mask sims where trace created_at > as_of — retain the timestamps the refresh currently discards | `_tr_created` array parallel to `_tr_meta` |
| 5 | `_episodic_vectors` → `roles_for_moments` → `gather`/`nodes_for_traces` | s1/s0 trace rows (query_traces door; rows carry created_at) | filter gathered stream rows to created_at ≤ as_of BEFORE the join; moment selection inherits #4's mask | `as_of` param on `roles_for_moments` |
| 6 | `scores()` / `_fields` | orchestration | thread as_of; z-scores over the masked universe (`_zscore(mask=)` already exists) | `as_of` kwarg |

**Accepted residuals (flagged, not hidden):** revised nodes leak current text/embeddings into the past
(`node_revised_after_turn` measures the effect); nodes archived after as_of stay hidden (engine is
live-only, 54c2e6e0); deleted nodes are absent. All minor, all already carried as walker flags.

**Test contract:** (a) as_of=None bit-identical to pre-change engine (inertness pin); (b) as_of=now ≡
as_of=None; (c) exclusion correctness (a node/trace created after as_of contributes nothing);
(d) the walker cross-check — row-level agreement on content lanes across ~4.5k turns.

**Status (2026-07-15):** SHIPPED — (a)–(c) in tests/test_recall_laf.py (TestAsOfTimeTravel), engine
e335b89. (d) RUN, verdict **AGREE** (eval/laf/walker/cross_check.py → cross_check.md): 105,776 j=0
rows × 8 lanes, cosine/sit median |Δ| 5.96e-08; the only out-of-tolerance rows sit on 2 nodes revised
after the walker build (provably excused, listed in the report), idf median |Δ| = 0 with the tail
explained by corpus churn. The 434 archived-since-build candidate nodes (3,560 rows) are counted in
the report — §20.3's since-archived pre-fit tally, now measured.

### 20.12 Priors & borrowed machinery — ASSUMPTIONS REGISTER (Tom 2026-07-14: "all assumptions worth noting and testing")

Every entry is an assumption with a named test — none is trusted, none constrains the locked grid.
Research: 3 agents 2026-07-14 (ACT-R priors / counterfactual-LTR / prior-art sweep) + two in-house
measurements on the walker.

**A1 — decay prior λ≈0.7, event-indexed.** Three independent sources converge: ACT-R d=0.5 (canonical
across ~30y of fits; per-event licensed by Anderson & Schooler's event-time power law + time-compression
practice), dialogue-context optimum 2–3 turns (≡ λ=0.7 horizon; 7±2 is the soft UPPER edge ~λ0.9),
GRU4Rec-lineage EMA λ≈0.7–0.9 event-indexed with an extra last-item boost. All from OTHER distributions
(human memory, consumer dialogue/shopping). TEST: the Q1 grid itself (0.3–0.9 + power + uniform brackets
it; the prior predicts the winner, never narrows the search). Parked P3 nuance: separate current-message
boost term (anchor-on-last-item).

**A2 — Haiku position bias is mild → plain pairwise for P3 v1, NO IPS.** In-house natural experiment:
within identical-pool_score tie groups (92k candidates; order carries no relevance signal), pick rate
top-third 6.25% vs bottom-third 5.27% — 1.19× pure-bias gradient, far inside the "skip IPS" threshold
(the raw 7× position curve is almost entirely relevance). CAVEAT: ties cluster in the low-score region —
bias is measured mainly where relevance is weak. TEST: ship the Haiku candidate-order SHUFFLE (surface
lane, 2-line change — kills residual bias at source AND makes future logs randomization-grade); compare
P3 fits with/without IPS on shuffled-era data. **SHUFFLED ERA BEGINS 2026-07-14** (shipped: seeded
per-turn shuffle in `build_surface_prompt`, seed = sha256(session_id|user_message); K trace carries
`shuffle_seed` + `presented_order`, O-trace cand_detail / rank_in_pool stay scorer-ordered; surface
prompt v15 drops "ordered by retrieval strength"). Pre-2026-07-14 rows keep the mild-bias caveat;
post-shuffle rows are propensity-exact. IPS recipe on file if needed: regression-EM propensities
(Wang 2018) or harvested cross-scorer-version swaps (Agarwal 2019), pair weight min(1/θ, 10),
self-normalized (Joachims 2017; Unbiased LambdaMART for the negative side).

**A3 — exploration ε≈4–8% (1–2 of 25 slots), unmarked, logged, ships WITH P3 (not before).** Industry
norm vs degenerate feedback loops (Jiang 2019; Krauth 2022); the explore slice is also what makes
round-2 training debiasable. TEST: canaries (judge-disagreement, abstention) + pool-novelty rate.

**A4 — H2 AMENDMENT (pre-run, legal window): used-next label redefined.** The registered used_next_1/3
label is statistically dead (~5–10 positives / 110k rows — Anchor near-never touches surfaced nodes
within 3 turns as anchor_touched records it). REPLACED as the debias leg by SOFT USAGE: similarity
between the surfaced memory and Anchor's actual next response (computable from stored walker texts).
Hard label kept as rare-gold secondary. TEST before it gates anything: label-quality audit — does
soft-usage correlate with picks and with gold where both exist?
**REVIEW F2 (2026-07-14) — carry into the soft-usage build:** the hard used_next label had a self-leak —
`anchor_touched` is accumulated per STOP, and a same-stop successor (superseded micro-turn) shares that
accumulator with the labeled turn, so the turn's OWN usage counted as "next-turn." Fixed in extract.py
(same-stop successors excluded from the window). Soft usage MUST key by `seq` and skip same-stop
successors the same way, or it re-inherits the leak.

**A5 — F_e borrows (P5 menu, each behind per-field ablation):** Synapse (arXiv 2601.02744) lateral
inhibition + fan-effect normalization + calibrated feeling-of-knowing gate (their τ_gate calibration
method for our Tier-1 peakedness threshold); RF-Mem (2603.09250) familiarity/recollection switching
thresholds. Judge-trustworthiness prior: LLM judges ≈93% human agreement on memory relevance (REMem) —
our own blind gold re-mints are the standing test.

**Prior-art verdict (context, not assumption):** the synthesis — persistent leaky-integrator field over
a lifetime graph + online LLM-judge feedback + tiered familiarity gate — is NOT implemented anywhere
found (2024–2026 sweep: Mem0/Letta/Zep/HippoRAG2/A-MEM all stateless per-query). Continuum Memory
Architectures (2601.09913) is a position paper CALLING for exactly this with no equations — a recognized
gap. Closest running system: Synapse (stateless per-query F_e analog). Read those two first when
implementing.

**A6 — walker v4 data shapes (relabel shipped 2026-07-14, d786f01; taxonomy: brain node 9adc8127).**
No DDL change — turns/candidates/cand_turn_scores columns identical; only VALUES moved. The sweep
inherits: (1) flag vocabulary is now `untraced_legacy` (627 — pre-06-08 s0 loss, op_text = O query;
`interrupted` no longer exists), `superseded` (847, 13.2% — a later turn shares the stop; steering/
interrupt/notification, live signal — moment-stack builders decide whether M(t) includes them),
`text_disagree` (69 structurally-paired turns kept as context, never labeled), `no_recall` (575).
(2) Labeled set: 4,478 (38 text_disagree exclusions, ledger-counted; was 4,495). (3) assistant_message
now attaches to the LAST s0 of its stop — the §20.9 first-s0 mis-attach is FIXED, anchor lanes at j≥1
score the response that actually followed. (4) op_vec_source value `local_interrupted` →
`local_untraced`. Rebuild GREEN 6/6, 809,466 score rows; fresh walker.db copied to the main tree.

### 20.13 P3 pre-registration — fit the composition on the walker table (laid out 2026-07-15, awaiting Tom's "stands")

**Question:** how much signal do the fixed gains leave on the table — and does fixing the
normalization artifact change what wins?

**P3.0 — normalization repair FIRST (its own mini-verdict).** The sparse-lane z inflation
(q1_reverse eyeball catch: enc z=11.4 / pick z=6.8 next to cosine z≈2 — mostly-zero lanes explode
under _zscore, sparse activations dominate the blend; explains pick dominance + the lane_buried
class). Three variants under STATIC gains before any fitting: current / support-z (stats over
nonzero support, zeros stay neutral — recommended) / rank-norm. Pre-declared pick: best June+ AUC
AND gold-24 tier placement not worsened. Winner becomes the fit substrate AND a flag-gated
production candidate on its own.

**P3.1 — the fit.** Walker v6, provenance-gated. Features (~32, pre-registered): {maxsim, sit,
idf, pick, enc} × {j0-op, j1-op, j1-anchor, j2-op, j2-anchor, decayed-tail j3–8 @ γ0.5} + M_e
fatigue value + intercept; rank_in_pool EXCLUDED (production's own output — leak). Model:
within-turn pairwise logistic, L2, linear on purpose — coefficients deploy as recall_laf K-store
VALUES, zero code (the §19 promise). Targets: picked (primary, pairwise); soft-usage (secondary,
separate fit, compared). Echo-sensitivity ablation pre-registered: every fit re-run WITHOUT
pick/enc features — if content-only collapses, the learned signal was echo and we say so.
Split: train April–May / validate June+; report per era + turn-class.

**P3.2 — evaluation (all six, pre-declared):** (1) June+ AUC vs K0-static and Q1-winner-static;
(2) miss-class deltas on gold-24 — near_miss + lane_buried should shrink, **unreachable must NOT
move (leak canary)**; (3) tier placement re-run (blind-judged); (4) shuffle control re-run on the
FITTED model; (5) soft-usage correlation; (6) ship gate = P1 discipline (gold-24 through real
brain.recall with DORMANT config, frame_replay A/B = H4, latency flat).

**P3.3 — deliverable:** P3a = j0-only fitted gains (deployable NOW via K-store — production
scores j=0 today); P3b = full moment gains (fit now, deploy rides Stage-3 wiring). Both DORMANT;
activation after H4; rollback = pointer flip.

**NOT in P3:** new lanes (cross-moment drop-rate inhibition, ÷norm recency), data-dependent
attention weights, selector/gating — each its own pre-reg after P3. Reach parked (Tom).

**Q1 inputs P3 stands on (all committed 2026-07-15):** verdict a1698bb3 (rank +0.045 shuffle-proof
/ reach flat / no-ship as registered); contributor decomposition (label echo named, maxsim
restriction-of-range named); miss classes (moment_seen 61 / lane_buried 48 / unreachable 38 /
near_miss 23); tier placement flat; artifacts q1_{sweep,analysis,tiers,reverse}.* + shuffle_control.md.

#### 20.13.1 P3.0 VERDICT (ran 2026-07-15, pre-declared rule) — PICK: current

Instrument `eval/laf/walker/p3_norm.py` (report `p3_norm.md`). Variants live in
`servers/recall_laf.py` (`_zscore_support`, `_zscore_rank`, `zscore_variant`; config key
`z_norm`, default `'current'` — K-store-flippable, shipped inert). Controls: sanity fixture
(current z_max 9.1 → support 1.9; rank bounded ±√3) + coverage invariant ×3 norms, all PASS.
Design correction pre-outcome: rank-norm as z-of-ranks re-inflated under the zero-tie block
(sanity caught z=7.1) → analytic uniform z (centered percentile × √12), bounded unconditionally.

| norm | AUC val win | soft_r win | tiers g+g k0 top5/25 | win top5/25 | gate |
|---|---|---|---|---|---|
| current | **0.8676** | 0.273 | 8 / 18 | 6 / 19 | PASS |
| support | 0.7810 | 0.278 | **9 / 21** | **8** / 19 | PASS |
| rank | 0.8351 | 0.275 | 4 / 17 | 4 / 13 | FAIL |

Pick = current, as registered (best June+ AUC; rank additionally fails the tier gate).
**Fit substrate for P3.1 = current. No production candidate ships from P3.0.**

**First-class finding — the primary metric is echo-shared for THIS comparison:** the picked-label
AUC shares its judge with the pick/enc lanes; under `current` those lanes carry z 6–11 and the
metric partly measures "the pick lane predicts picks." Support-z demotes exactly those lanes and
loses 0.087 AUC — but on every judge-independent metric it is neutral-to-better (soft_r 0.278 vs
0.273; blind-judged tiers: k0 arm top-5 9>8, top-25 21>18; win arm top-5 8>6, top-25 equal).
The echo-sensitivity ablation (P3.1, pre-registered) adjudicates. Note: a linear fit's gains
absorb per-lane SCALE, so `current` as substrate does not handicap the fit — the per-turn scale
instability of sparse-lane z is second-order and the ablation catches the echo story. Secondary
observation for a future pre-reg: idf remains heavy-tailed WITHIN its support (eyeball z 6.7–8.7
under support-z) — support-z fixes pick/enc, not idf's internal skew.

#### 20.13.2 P3.1 fit results (ran 2026-07-15) — the ablation matrix is the story

Instrument `eval/laf/walker/p3_fit.py` (report `p3_fit.md`, coefficients `p3_fit.json`). 31
features as registered (intercept cancels in the pairwise formulation — 0 by construction);
Newton/L2, λ-insensitive across {0.1, 1, 10}; train 81,545 picked pairs / 14,461 soft pairs.
Eight arms: {full, j0-only} × {picked, soft} × {with, without pick/enc}.

| arm | target | features | val AUC (picked) | soft_r |
|---|---|---|---|---|
| A full | picked | all 31 | **0.9698** | 0.130 |
| C full−pick/enc | picked | 19 | 0.7102 | 0.243 |
| B full | soft | all 31 | 0.7695 | **0.427** |
| F full−pick/enc | soft | 19 | 0.6448 | **0.430** |
| D j0-only | picked | 6 | 0.9279 | 0.088 |
| G j0-only | soft | 6 | 0.7887 | 0.177 |
| statics | — | — | K0 0.8226 · winner 0.8676 · K0-content-only 0.6921 | 0.173 / 0.273 |

**The pre-registered sentence fires: the picked-label fit gain was echo.** Content-only (C)
collapses below K0-static (0.7102 < 0.8226); the A coefficients say why (pick·j0 +1.37,
pick·j1op +1.00, pick·j1anchor +0.86 — the fit poured weight into the label-sharing lane).
Like-for-like, though, the fit is real: C beats the content-only static (0.7102 > 0.6921), and
even K0-static's own AUC is mostly echo (0.8226 → 0.6921 with pick/enc zeroed).

**The quality signal lives in the moment slots and needs no echo lanes:** the soft-target fit
nearly doubles the static soft_r (0.427–0.430 vs 0.273) and loses NOTHING when pick/enc are
ablated (F ≈ B) — but drops to statics' level when restricted to j0 (G 0.177 ≈ static 0.173).
Judge-independent quality headroom = moment stack, not j0 re-weighting.

**M_e sign flip (mechanism finding):** fatigue coefficient −1.29 on the picked target,
+1.28 on the soft target — recently-surfaced nodes are LESS likely re-picked (availability
management, the 2′ design intent) yet MORE likely response-relevant (thread continuity).
One dial, two opposing jobs — relevant when M_e ships.

#### 20.13.3 P3.2 verdict (ran 2026-07-15, all six pre-declared) — NO SHIP, as the gates ruled

Instrument `eval/laf/walker/p3_eval.py` (report `p3_eval.md`). Arms at the blind gates:
A (registered primary) and F (the ablation matrix's quality candidate).

1. **June+ AUC** — in p3_fit.md (A +0.147 over K0-static; echo per above).
2. **Miss classes** — expectation FAILED both arms: near_miss shrinks (24→12 A / 16 F) but
   lane_buried grows (50→73 A / 68 F); net misses UP (201 → 216 A / 210 F). **Leak canary
   CLEAN: unreachable frozen at 35 in all four arms; zero unreachable-substrate nodes in any
   top-25.**
3. **Tier placement (blind)** — A actively harms (gold top-25 12→6, gold_plus 6→3): the echo
   model buries blind-judged gold to chase pick prediction. F ≈ parity (gold top-25 12→10,
   gold_plus median 74→56, silver top-1 2→4).
4. **Shuffle control on fitted models** — holds for both (A: shuffled 0.7980 ≤ j0-restricted
   0.8257; F: 0.5962 ≤ 0.6937). What moment gain exists is not a length artifact.
5. **Soft-usage correlation** — the one clear win: 0.427–0.430 vs 0.273 static, echo-ablation-proof.
6. **Ship gate** — moot except for P3b below; nothing activates.

**Deliverables (P3.3):** P3a = **no gain flip ships** — D's j0 gains are echo (blind tiers
collapse), G's j0 soft gains ≈ static (0.177 vs 0.173); production statics stand. P3b = **F's
coefficients recorded** (`p3_fit.json`, arm F_soft_ablate) as the Stage-3 candidate — the
moment-slot soft-quality gain (0.43 vs 0.27) is real, judge-independent, and shuffle-proof, but
it deploys only with Stage-3 moment wiring, and its blind-tier parity means it re-runs the full
gate suite (incl. H4 frame_replay) at that point. No K-store registration now — registering a
candidate that failed its blind gates would hollow out the eval-gated-activation discipline.
`z_norm` stays in the engine at `'current'` (inert, K-store-flippable) as P3.0's infrastructure.

### 20.14 Post-P3 design notes (Tom + Anchor, 2026-07-16) — fatigue taxonomy, mesh shapes, settling, the underlying mechanism

Design conversation record — nothing here is built or pre-registered; each item names its
sequencing decision.

**Fatigue taxonomy (Tom).** Two distinct concepts the system currently conflates:

- **Technical fatigue** — context economy. Re-surfacing the same node wastes tokens and works
  against the LLM's own context management. Caveat (Tom): the S1 Scribe USES surfaced nodes to
  decide revise/connect targets — suppressing a node from Anchor's context must not starve the
  encoder. Design note: decouple the two consumers — suppress from `additionalContext`, still
  deliver to the encoder as "raised" (the pick/drop data already exists in S1R traces; the
  encoder feed reads a different field). Costs nothing to honor at Stage-3 wiring time.
- **Brain fatigue** — in-field inhibition: nodes Haiku selected at turn t−1 should be inhibited
  IN the field at turn t, visibly to other mechanisms (spread must not resurrect an inhibited
  node through a neighbor). This is decision 7e9e36a7 (fatigue as inhibition lane, never an
  outside multiply) + the running field's negative channel (87a6dae9).
- **Current state:** production `_mark_accessed` fatigue is the stopgap and is out-of-contract
  with the 2′ pin on all three axes — candidate-scoped (not picked-only), drop-punishing,
  undecaying (node ba05383d). Session-scoped (`ctx.fatigue`), so a proper running field with an
  inhibition component RETIRES it, not layers under it.
- **Sequencing (Tom's call, 2026-07-16):** fatigue strategy is thought through AFTER the
  moment-recall value check — don't hold Stage-3 for it. P3 support: M_e was rank-flat (Q1) and
  its sign flip (−1.29 picked / +1.28 soft, §20.13.2) says its tuning needs a quality target.

**Mesh shapes (the moment-composition families) and the chosen ladder:**

| shape | form | status |
|---|---|---|
| linear sum (turnsum·zsum) | Σ wⱼ·Aⱼ | shipped shape; Q1 winner; P3-fitted |
| max-union (turnmax) | maxⱼ wⱼ·Aⱼ | tested, lost on rank |
| leaky integrator (running field) | A_t = λA_{t−1} + act(q_t) − μ·surfaced_t | designed (87a6dae9), unbuilt — unrolls to the exp-decay stack, so Q1/P3 numbers transfer; O(1)/turn; principled home for brain-fatigue |
| gated / attention mesh | wⱼ = f(relevance of turn j to now) | parked by pre-reg (its own pre-reg after P3); attacks the fixed-weight-noise failure (flagged-turn slice) |
| nonlinear settling (attractor) | iterate a ← norm(a + ΣO(a)) | vision (076799d0); needs its own replay instrument before any rung ships |

Ladder: **integrator → gate → settle.** Every rung walker-replayable before it ships — same
discipline that caught the echo.

**Settling, the principle:** stop scoring nodes independently; let the memory vote on itself
until it agrees. State = one activation vector over the graph; operators = forces (cue evidence,
edge propagation, inhibition, normalization); iterate to the fixed point where the constraints
are mutually satisfied — that state IS the recall. Buys pattern completion (nodes the cue never
touched arrive via neighbors — the unreachable-38 class), coherence-beats-similarity, and
context sensitivity for free (settling starts from the current state, not zero). One-shot LAF
is exactly one iteration of this loop.

**Multi-modal settling (Tom's requirement, 2026-07-16):** single-attractor/global-inhibition
settling is NARROWING — it converges to one coherent interpretation. Tom: the field should be
able to settle on several distinct areas. The knob is the INHIBITION KERNEL: global inhibition →
one winner; local/similarity-scoped inhibition (Mexican-hat over embedding or graph distance) →
multi-bump settled states, one bump per distinct coherent cluster (continuous-attractor
literature; k-WTA and per-community normalization are the discrete cousins). This is a standing
design constraint: **never global winner-take-all** — the pool's job is to hand Haiku several
distinct coherent clusters, not one; redundant near-duplicates compete, distant clusters coexist.
Also the fix shape for @5 slot-stealing / pool saturation (30d88dd0).

**The underlying mechanism (Tom's question: research findings are instances of what?).**
The computational-level claim: **recall is inference of the latent moment — a posterior over
"which stored patterns does the present belong to" — under a delivery economy.** Every research
mechanism is one factor of that inference in closed form: similarity lanes = likelihood terms;
recency/base-level decay = the prior from reuse statistics (Anderson's rational analysis —
ACT-R's decay curve is DERIVED from environmental reuse frequencies, not postulated); edges =
conditional dependencies (if A is relevant, B likely is); inhibition/normalization = explaining
away (competition IS explaining-away); settling = iterative inference; attractors = hypotheses;
pattern completion = filling in the posterior over unobserved variables. Two direct payoffs of
this frame on OUR measurements: (1) the M_e sign flip is the relevance-vs-marginal-value
distinction falling out — an already-delivered node keeps high relevance (soft +1.28) but has
zero marginal information value (picked −1.29); fatigue is an ECONOMY term, not a relevance
term, and must never be tuned on a relevance target. (2) Multi-modal settling = keeping the
posterior multi-modal instead of collapsing to MAP — Tom's narrowing worry is exactly the
MAP-collapse failure, and the delivery answer is "represent the modes," not "pick the peak."

**Prospective memory (reminders)** — Tom's feature ask, designed in `docs/BACKLOG.md`
(2026-07-16 entry): time-triggered, guaranteed-delivery reminders for both Tom and Anchor —
deterministic due-check at the recall hook, never similarity-gated.

### 20.15 The math of recall-as-moment-inference (2026-07-16) — the formal skeleton

Objects: nodes i=1..N with structure W_ij (edges); latent moment z_{i,t} ∈ {0,1} ("node i
belongs to the present moment"); observations o_t (the turn's messages + activity); deliverable
S_t ⊆ nodes under token budget B. Recall = posterior p(z_t | o_{1:t}) + budgeted delivery.

**L1 — static posterior (built: LAF).** Log-odds Bayes under exponential-family likelihoods →
p(z_i=1|o) = σ(Σ_k β_k·φ_k(i,o)). Production `sigmoid(Σ gain·zscore(lane)/τ)` IS this equation:
gains = log-likelihood-ratio weights; z-scoring = crude calibration (P3.0 = "which calibration
makes the linearity truest"); the P3 pairwise-logistic fit = maximum-likelihood estimation of β.

**L2 — temporal filter (built: moment stack; designed: running field).** Give z dynamics
p(z_t|z_{t-1}) and filter. Under geometric thread-persistence the log-odds recursion collapses
to a_t = λ·a_{t-1} + ℓ(o_t) — the running field — whose unrolled form Σ_j λ^j·ℓ(o_{t-j}) is
the exp-decay turnsum stack that won Q1. The Q1 decay grid was model selection over persistence
priors; exp-γ winning = data votes geometric at short range. Prediction: thread resumption
after days ⇒ the true persistence prior is heavy-tailed ⇒ a SECOND slower timescale is
structurally required — the role episodic lanes / session state already fill by instinct.

**L3 — structure (vision: settling).** Edges make it an MRF: p(z|o) ∝ exp(Σθ_i z_i + ΣW_ij z_i z_j),
θ = the L1 field. Mean-field inference = iterate q_i ← σ(θ_i + Σ_j W_ij q_j) — settling, term
for term (Hopfield energy = −log posterior; attractors = posterior modes; one-shot LAF =
iteration #1 with W=0). Naive mean-field's NAMED pathology is mode collapse — Tom's "narrowing"
worry, formally; the fix (structured q / diverse restarts) implements as local-inhibition
kernels ⇒ the §20.14 never-global-WTA constraint is the correction for a documented
approximation error, not a taste choice.

**L4 — delivery economy (built: Haiku selector + reach metric; misfit named: echo).**
S* = argmax_{cost≤B} E[U(S)|posterior]. (a) Value-of-information: delivered nodes have ~0
marginal U regardless of relevance ⇒ fatigue is ∂U/∂delivered, an economy term OUTSIDE the
posterior ⇒ a fitted fatigue coefficient MUST flip sign between delivery and relevance targets —
the measured M_e flip (−1.29/+1.28), derived. (b) Submodularity: need-reach@K IS a max-coverage
objective; greedy selection carries the Nemhauser (1−1/e) guarantee — Haiku ≈ approximate greedy;
diversity-over-redundancy falls out of the objective. (c) The P3 pick-echo has a formal name:
off-policy logging bias (picks generated by the previous policy; counterfactual LTR / propensity
correction is the standard treatment) — a picks-trained model re-learns the logger.

Umbrella: Anderson's rational analysis at the prior, variational inference at the structure,
decision theory at the delivery. The operational sentence: **LAF already is this math with
hand-set parameters; the integrator → gate → settle ladder replaces each hand-set term with its
properly-inferred version, one measured rung at a time.** Brain: insight 813c185d (frame),
this section (formalization).

### 20.16 The type system (Tom's collapse question, 2026-07-16) — every mechanism is a factor, a move, or a value term

Foundation, one line: S* = argmax_{cost≤B} E_q[U(S,z)] with q(z) ≈ p(z|o_{1:t}) under a
generative model p(z,o). The pipeline is model → inference → decision; there is no fourth place
for a mechanism to live — that closure is what pre-covers unimagined layers.

| type | lives in | composition | admission test | bug shape |
|---|---|---|---|---|
| **Factor** (evidence/prior) | p(z,o) | adds in log space (the field); gain = log-likelihood ratio; must be calibrated | carries CONDITIONAL info about z given existing factors (the 8bcc8c96 union analysis IS this test) | mis-specified likelihood / mis-calibration |
| **Move** (inference step) | q | iterates; improves q→p, never changes the model | measurably closes the approximation gap | mode collapse, dead operators |
| **Value term** (economy) | U | adds in the SELECTOR objective only — never in the posterior | tuned on outcomes/quality, never on relevance labels or logged picks | category leaks, label echo |

The arc's entire bug ledger retro-types: sit zero-fill = factor (likelihood) · z-inflation =
factor (calibration; P3.0 was a calibration study) · graph-inert = dead move · `_mark_accessed`
score-multiply = value term LEAKED into the posterior (typing violation — independently flagged
as out-of-contract) · M_e sign flip = one coefficient asked to be factor AND value term ·
pick echo = value/label confusion (off-policy) · "signal ÷ prevalence at three scales"
(1e154cbe) = three layers collapsing into one factor family. The type system was being
enforced by trial and pain; naming it makes it enforceable by inspection.

**Derive, don't design (covers future layers):** write the generative story of a moment — a
session carries live threads; threads activate nodes; active nodes generate every observable —
then read the layers off it: one likelihood FACTOR per observation channel (message text /
situation phrasing / rare tokens / pick·enc events / project / and mechanically-enumerable
unbuilt ones: tool-usage, files-touched, time-of-day rhythm, register, speaker), one PRIOR per
latent structure (thread persistence = decay; switching rate = the gate; thread↔community
alignment; cross-session resumption = the L2 second timescale), VALUE terms from the delivery
story (token cost, redundancy, fatigue/VOI; the encoder-feed decoupling = a second consumer
with its own U over the shared posterior). New layers arrive pre-typed, pre-composed,
pre-tested. The falsifiable residue moves up a level: the STORY (threads as the latent object)
can be wrong in shape — a far better place to be wrong than at the layer level, fifty times.

**Multi-store correction (Tom, 2026-07-16):** the story does NOT hardwire nodes as the target —
threads activate CONTENT; each store is a different record of past moments (semantic nodes =
distilled, episodic traces = lived, images/other modalities later). Per store: likelihood
factors on the evidence side (pick/enc lanes = the episodic store already testifying) AND
delivery candidates on the selection side — S widens from "nodes" to "pieces across stores"
under one budget and one U. Tom's 'pieces of memories' vision is inside the foundation, not an
amendment; nothing shipped may foreclose the widening (a per-store factor is a lane; a
mixed-store S is a bigger candidate set for the same selector). Sequencing unchanged: recall
operates first. Prior-art status (per e16feae3, June sweep): every level battle-tested somewhere
(max-ent fusion / POMDP belief tracking + ACT-R need-odds / MRF-energy settling / submodular
selection + counterfactual LTR); the assembled typed stack over a personal graph with a replay
instrument — not found deployed; re-sweep the literature before ever leaning on novelty.

**Adoption policy (Tom's pragmatism check, 2026-07-16): the shape is a lens, NOT a refactor
mandate.** The shipped code already IS the shape (§20.15 — that was the point); rebuilding
recall_laf into model/inference/utility classes would produce the same numbers through new bugs
and orphan the measurement harness (walker, parity, cross-check) that our verdicts rest on.
Three sanctioned uses only: (1) every future proposal opens with its type declaration —
factor/move/value — one line, the whole adoption cost; (2) NEW rungs (running field, gate,
settling) are built in the typed vocabulary as they arrive — the codebase converges organically,
rung by rung; (3) refactor is licensed only where a typing violation is ALSO a measured defect —
exactly one on the books (`_mark_accessed` fatigue), already sequenced post-moment-value-check.
The math is a compass, not a demolition order.
