# RECALL — from the SR/PPR thesis to the Layered Activation Field (LAF)

> ## ⟦ CURRENT STATE — read this first (2026-06-26) ⟧
> This doc is a **journal** of the recall-redesign arc, not a static spec. It records a real evolution
> through four framings, and **the head is now historical.** Live direction = **§18.18 (the settling engine)**.
>
> **Live design — recall as a SETTLING ACTIVATION FIELD** (Tom's biological dynamical-fusion vision; supersedes
> the §18.17 selector framing). One shared per-node activation; operator-fields add **z-scored, gained** signals to a
> `base`; a **softmax-in-loop recurrence** lets graph corroboration feed back and **settle** to a fixed point; read
> out the settled field (α-entmax only at the commit). **A new cue = one z-scored term + a gain — influence IS the gain.**
> - **Operators (built + verified):** MaxSim (max cos over 6 field-groups) · temporal-distinctiveness (created_at,
>   von-Restorff) · typed-graph-spread (degree-norm, noise-excluded). Next: prev-turn, query-conditioned aspect
>   weighting, per-node-type field, episodic.
> - **Stability:** the naive α-entmax-every-step recurrence COLLAPSES (wandering single node); the **Hopfield-contractive
>   form** (softmax in-loop + raw spread + sparse-only-at-readout) **settles 73/73** (§18.18).
>
> **Numbers (73-cue endo, IsolatedBrain):** pipeline 19/33 · raw `_primary` 21/37 · MaxSim-6grp 19/34 · LAF settling 16/38.
> **ABLATION + REVIEW (§18.18 / §18.18.1):** harness code-reviewed & hardened — **gate 27 pass / 0 / 0**. Converged numbers:
> maxsim 19/34 · +temporal 16/38 (temporal = query-independent +4@25, **ARTIFACT-SUSPECT**) · MaxSim dilutes below raw
> `_primary` 21/37. **Graph's effect is still UNMEASURED on a converged field**: fixing the scale made graph *matter* but
> per-iteration z-score is anti-convergent (`f04f6db7`) — next is a FIXED-scale spread. **No relevance-driven win yet — on a
> now-verified harness, for understood reasons.** Architecture validated (settles, extensible, MEASURABLE). **▶ NEXT STREAM =
> §18.18.1** — **START WITH Q4** (reverse-engineer the operator bank → ranked menu + read-side-ceiling-vs-encode-gap split),
> which ORDERS the rest: fixed-scale re-ablate · temporal shuffled-gold control · adaptation/sequential-recall · hub-dampening · new cues (`220a2808`).
>
> **Reading guide:** **§0–12 = the SR/PPR thesis — largely FALSIFIED at §13b** (PPR-standalone < cosine; the embedder
> limits abstract relevance). **§13b–17 + §18–18.17** = falsification arc + multi-cue "integration thesis" (§15) +
> the selector reframe (§16–18.17, **now superseded**). **§18.18 = LIVE** (the settling field). All earlier
> "▶ NEXT STREAM" markers and the §18.17 selector are superseded by §18.18.

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

The settled field **recovers MaxSim's dilution and beats both baselines @25** (+1pp over `_primary`, +5pp over pipeline)
at a **precision@5 cost** (16 < 21). **Honest read: the architecture is validated (settles, extensible, measurable);
the number is NOT a win yet.** `τ` is the commit-sparsity knob (ranking-invariant — softmax is monotonic); **`gain_graph`
is the @5↔@25 dial.**

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
- **Temporal carries the entire LAF-vs-MaxSim delta** (−3pp @5, +4pp @25) — the operator dismissed as "near-flat"; z-scoring
  gave the tiny-raw signal teeth. **BUT temporal is query-INDEPENDENT** → it reshuffles the same nodes every query → its @25
  lift is a **corpus-artifact candidate** (gold happens to be temporally distinctive here), not relevance. Needs a
  shuffled-gold control to confirm real-vs-artifact.
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
The harness was code-reviewed (high effort) and hardened — fixes in `eval/laf/{operators,field_recall,verify_substrate}.py`:
α-entmax `α≤1→softmax` guard; null/empty-`created_at` now excluded from eligibility (was wrongly eligible — `'' <= cutoff`);
`qv=None` cues skipped; **T5 baseline-reproduction is now a HARD FAIL at ±2pp** (was a no-op warn — the gate's master check
didn't gate); **T8 verifies the von-Restorff DIRECTION** via an independent neighbour recompute (was tautological with the
non-constant check); T6 batched its per-node SQL + embeds once; MaxSim's **unweighted-max coverage bias documented** (more-
enriched nodes get more "max lottery tickets"; edge_context's 71% makes it active; real fix = the weighted/"smarter fields"
step). **Gate: 27 pass / 0 warn / 0 FAIL** — substrate + operators trustworthy. Refuted finder claims: T3 recompute is sound
(max|Δ|=0.0005, `embedding_similarity` IS raw `_primary` cosine); `primary_field` intentionally reproduces the recorded
raw-`_primary` 21/37 reference.

**Trustworthy CONVERGED numbers (73-cue endo):** maxsim-only 19/34 · +temporal 16/38 (temporal = query-independent +4@25,
ARTIFACT-SUSPECT). Graph's signal is UNCONVERGED → not bankable (apparent +2@25/−4@5, helps @25 only with temporal). MaxSim
dilutes below raw `_primary` (21/37). **NO relevance-driven win yet — but on a now-verified harness, for understood reasons.**

**▶ Next stream — START WITH Q4 (diagnose-before-build); it ORDERS everything else (Anchor's recommendation, 2026-06-26):**

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
