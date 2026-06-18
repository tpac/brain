# RECALL-SR — recall as a Successor-Representation predictive map

**Written:** 2026-06-17 (Anchor + Tom, session `ce0ff8ce` arc continued in a fresh stream).
**Status:** design / direction. Nothing built yet — this is the synthesized thesis and the staging.
**For:** future-me. Written so it surfaces when I'm working recall ranking, spread, the endo
surface, episodic/temporal recall, query composition, or "should we train something."
**Parent:** `ARCHITECTURE-FRACTAL.md` (`integrate(O,K)→Δ`), `RECALL-OVERVIEW.md` (current pipeline),
`RECALL-STATE.md` (validated current numbers). **Siblings:** `SOFT-SURFACE-DESIGN.md` (the endo
surface — this is the kernel it inherits), `ORACLE-AUDIT-SPEC.md` (the eval instrument),
`RECALL-DUAL-STORE-DESIGN.md`, `RECALL-TEMPORAL-ANCHOR-SPEC.md`, `HANDOFF-RECALL-NORMALIZATION.md`.

> **The one-line thesis.** Relevance is *flow from your current state through a stable relational
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
