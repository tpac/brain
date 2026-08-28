# S2 Community — Design

Two parts, on purpose:

- **Part I — Persistent anchors + continuous evidence (design, awaiting
  ratification).** The target architecture, rewritten 2026-08-28 after a day
  of measurement refuted the previous draft. Nothing in it is built.
- **Part II — The pipeline as it runs today.** Kept verbatim below. It is
  stale in places (dated 2026-04-11); correcting it is C3's job, scheduled for
  after the work lands, so the doc does not go stale against itself mid-flight.
  Where Part I contradicts Part II, Part I is a proposal and Part II is what
  production does today.

---

# Part I — Persistent Anchors, Continuous Membership

**Status: DESIGN. Not built, not ratified.** Every number below was measured
on 2026-08-28 against an `IsolatedBrain` copy of production; every claim about
existing code carries a `file:line` opened the same day.

## 0. What this replaces, and why

The previous Part I (commit `13466fd`) specified **reshape-diff**: re-derive
the whole-graph partition every run, diff it against the stored communities,
and have the encoder judge only the structural delta. Six measurements
retired it. The decisive one:

> **A 2% edge cut moves 23.4% of the production seeder's communities. A 0.5%
> cut moves 19%.** Every algorithm tested amplifies input change by more than
> an order of magnitude.

A partition is not a stable object for this graph, so it cannot be the thing
you diff. That is not a bug in our seeder and it is not fixed by adopting a
better algorithm — Louvain and SLPA were measured and are *worse* on this axis.

What replaces it is a different division of labour, and it is the answer to
*"known methods never had an agent, we do — how do we close the gap?"*:

> Classical community detection must re-partition from scratch every run
> **because it has no way to persist a decision.** We have an agent that can
> create a durable, named, meaningful object and stand behind it. So: **the
> agent owns a persistent discrete layer; the algorithm supplies continuous
> evidence against it.** Nothing is ever re-partitioned.

The measured consequence: under the identical 2% perturbation, membership
evidence against a fixed anchor flips **1.3%** of decisions where the
partition moved **23.4%** — a ~20× reduction in churn.

## 1. Evidence base

Instruments: `eval/community_reshape_probe.py` (rerun unmodified) plus
scratchpad probes reusing its own step functions and the production decoder's.
The scratchpad probes are not committed; §9-P2 lands the load-bearing ones.

| # | measurement | result |
|---|---|---|
| **M1** | probe rerun, weekly velocity | **22 splits, 0 merges, 3 dispersals, 43 births, 97% stable.** The checklist F18 and node `7fbad66b` record *2 splits / 99%* — same instrument, same day. **Do not build on the "2".** |
| **M2** | A/A, one graph, two processes | same `PYTHONHASHSEED` → 1,168/1,168 identical (100%); seed 0 vs 1 → **93.2% identical**, which the probe's own differ calls 16 splits + 12 merges + 1 dispersal, **0 births**. Mechanism traced: a str-keyed set materialised at `community_decoder.py:765` fixes dict order, which breaks ties in the stable sort at `community_decoder.py:824`. |
| **M3** | z-score tie density | **14,830 seed pairs collapse onto 60 distinct z values; largest tie group 2,366 pairs.** The score is not ranking pairs, it is bucketing them; order within a bucket — which the greedy walk consumes — is arbitrary. |
| **M4** | recursive clustering (clusters-of-clusters) | a meso layer **is** derivable: **174 groups**, median 23 nodes, max 149, 65% coverage, typed internal fraction **0.49–0.69**. |
| **M5** | hybrid substrate (typed ∪ mutual-kNN-10 on centred embeddings) | **1,500 of 9,232 nodes have zero typed edges** and are unreachable at any threshold. With a semantic layer, coverage 67% → **90%**, and **92% of the 1,500** get clustered. |
| **M6** | algorithm bake-off | Louvain hierarchy reaches **14 communities, median 645, covering 99%** (the project-scale layer). At matched granularity Louvain beats production on modularity (**Q +0.547 vs +0.473**). But cross-run stability: production **98%**, SLPA 82%, Louvain typed 83.5%, **Louvain hybrid 50%**. |
| **M7** | consensus clustering ×9 | lifts stability to **90–94%** and then plateaus — τ 0.5→0.9 does not stabilise further (94.0→93.9) but destroys coverage (64%→46%). Best operating point: hybrid τ=0.5, 94% coverage, Q +0.549. |
| **M8** | **perturbation sensitivity** | % of communities surviving a cut, hybrid substrate, fixed seeds: <br>`0.5% cut / 2% / 5%` → production **81.1 / 76.6 / 69.3**, Louvain **77.6 / 76.3 / 74.1**, Louvain-consensus **90.0 / 87.0 / 81.4**. |
| **M9** | **evidence discrimination** (2,500 member pairs vs 2,500 non-member, leave-one-out) | AUC: `edge_frac` 0.976 ⚠circular · `sem_centred` **0.949** · `edge_sem` 0.893 · `knn_share` 0.860 · `co_surface` 0.689 · `co_anchored` 0.620. |
| **M10** | **evidence stability**, same 2% cut | decisions flipped: `edge_frac` **1.3%**, `edge_sem` **0.9%**, embedding signals 0.0% (true by construction). Against the partition's **23.4%**. |
| **M11** | placement rank of true anchor among 707 | full evidence: top-1 87%, top-3 97%, top-10 98% · **semantic only (a new, edgeless node): top-1 47%, top-3 68%, top-10 83%** · hub-scale pick: full 93%, **semantic 65%**. |
| **M12** | operating context (`query_traces`, 168h) | **31 decode runs/week (4.4/day)**; communities 744 → 787 (**+43/week**); unplaced population 1,947–1,989 with 9–47 pending → the rest gate suppresses **97.6–99.5%** every run. Encoder cost: 9 consecutive runs at 4 actions / 2 writes, output tokens 2,903–7,397 (mean **5,308**). |

⚠ **`edge_frac`'s 0.976 must never be cited as validation.** These communities
were *built* from typed edges, so members are members largely because they had
edges to members. `edge_sem` inherits some of it. The non-circular signals are
`sem_centred`, `knn_share`, `co_surface`, `co_anchored`.

## 2. The eight first principles this must satisfy

Stated by Tom on 2026-08-28 (`17abce4d`) after watching the measurements land.

1. **Purpose is threefold** — activation areas (divisive normalisation: a
   crowd dims, a lone node shines, `5e60e0a0`), project scoping (`6ee28032`),
   and boot-time recognition (`531d1831`). Three consumers, three resolutions.
2. **Aggregation gives more than a summary** — a community is a *reusable
   computation*: a prevalence denominator, a region-level prior, a scoping
   handle, context compression, and a change detector.
3. **Nodes do not need edges to belong.** Overturns `77b2617c` as a universal
   gate — edge fraction survives as a signal, not as the definition.
4. **A new node joins a hub first, and differentiates later** as more nodes
   arrive around it.
5. **Communities split when there is a better definition** that groups them.
6. **Lean on a hybrid substrate**, not edges alone.
7. **Reduce recalculation to real change without killing plasticity.**
8. **Slowly heal current state** — correction over time, not migration. True
   for this brain and for the fleet.

## 3. The architecture

### 3.1 Two layers, one of them persistent

| layer | what it is | owner | persistence |
|---|---|---|---|
| **anchor** | a `type='community'` node — title, narrative, situation, question | the agent | permanent until the agent retires it. **Never recomputed.** |
| **evidence** | per `(node, anchor)` continuous scores | the algorithm | recomputed every cycle, cheap, zero tokens |

There is no third object. Micro-clusters, fresh partitions and carry-forward
cluster identities are all gone — they were the thing that churned.

**The 790 existing communities are the initial anchor set.** There is no
migration and no reorganisation (principle 8). The giants heal by being named
apart over time, a piece at a time.

### 3.2 The evidence vector

Six signals, recomputed per cycle for every `(node, anchor)` pair with any
non-zero term:

| signal | what it is | AUC (M9) | coverage | notes |
|---|---|---|---|---|
| `edge_frac` | fraction of the node's typed neighbours inside the anchor | 0.976 ⚠ | 95% | circular for validation; still the strongest operational term |
| `sem_centred` | centred cosine to the anchor's centroid | **0.949** | 97% | the strongest *independent* signal; reproduces the P0 baseline (0.349 vs 0.000) |
| `edge_sem` | node's pooled edge-embedding vs the anchor's pooled internal-edge embedding | 0.893 | 98% | uses `edge_relations.embedding` (v26, `[relation] description`) — **built and never read by community detection** |
| `knn_share` | fraction of top-10 semantic neighbours inside | 0.860 | 73% | reaches nodes edges cannot |
| `co_surface` | fraction of co-surfaced partners inside | 0.689 | 38% | high-precision / low-recall — 100× lift over base rate |
| `co_anchored` | fraction of `co_anchored` neighbours inside | 0.620 | 24% | currently filtered as `noise` |

`sem_centred` and `knn_share` are what make principle 3 real: they score the
1,500 edgeless nodes that `edge_frac` cannot see at all.

⚠ `co_surface` was predicted to be the strongest "activation areas" signal and
measured the weakest. It is corroboration, not foundation.

### 3.3 Membership

**Membership is a threshold on evidence against a fixed anchor** — algorithmic,
no agent call (principle 7; Tom: *"an agent connecting to a big community isn't
really necessary if the algo is good"*).

- **Multi-membership is the default.** Evidence is computed against every
  anchor independently, so a node joins each anchor it clears. That is
  `df292d31`'s overlap made native rather than permitted.
- **The edge carries confidence as weight.** `GraphDAL.add_relation`
  (`dal_graph.py:999`) already takes `weight`; a node placed at 0.65 evidence
  contributes proportionally less to the anchor's activation mass and its
  narrative, so a weak placement costs little while it waits to be corrected.
- **Provenance** is `encoding_source='s2:community_reshape'`, distinct from the
  agent's `s2:community_detection`. Everything the mechanism ever wrote is one
  query — which is what makes staged rollout and rollback tractable.
- **Writer:** one owner, `servers/scales/s2/community_reshape.py`. Edge writes
  route through `GraphDAL.add_relation`; removals through the DAL's archive
  path. No raw SQL against `edges` / `edge_relations`.
- **When:** after the agent's Δ in the same cycle, the position and pattern of
  the existing structural stamp (`community_encoder.py:266-277`), which then
  runs after it on the final edge state as it does today
  (`community_encoder.py:318-369`).

**Loud at the write boundary:**
1. post-write parity — the anchor's live member set equals the thresholded
   evidence set; a mismatch logs `_log_error` with the diff;
2. a churn ceiling that **refuses the write and logs** rather than silently
   rewriting an anchor's identity in one cycle;
3. never remove membership from a `locked`/`critical` anchor — pre-filter as
   `_auto_archive_dead` already does (`community.py:264-268`);
4. `consecutive_failures` per the house S2 pattern.

### 3.4 Placement lifecycle (principle 4)

Measured operating characteristic (M11), for a node with **no edges yet**:
top-1 47%, top-10 83%, correct hub 65%.

That is not enough to *assign* and it does not need to be:

1. A new node is placed into the best **coarse** anchor its semantic evidence
   clears — right about 65% of the time.
2. Evidence is recomputed every cycle. As the node accumulates typed edges,
   `edge_frac` and `edge_sem` come online and the full-evidence arm reaches
   87% top-1 / 97% top-3.
3. **A wrong early guess self-corrects**, because membership is fluid and the
   anchor is fixed. This is the exact inverse of today's system, where
   placement is frozen the moment it happens (F17,
   `community_decoder.py:118-121, 383, 390-400`).

Nobody is ever communityless, and no claim is made that isn't backed by
evidence: a node with weak evidence sits in a coarse anchor at low weight.

### 3.5 What summons the agent

The agent is scarce (`e40ebfee`). It is called only for what cannot be computed:

| event | trigger | the agent's job | failure mode |
|---|---|---|---|
| **birth (fission)** | a dense, separable sub-region *inside* an existing anchor, persisting N cycles | name the child, write its narrative, add lineage to the parent | fission along algorithmic fineness rather than a real seam |
| **split gate** | same trigger | **decide.** Separability is necessary; **nameability is sufficient.** If the agent cannot articulate what makes the sub-region a different thing, there is no split, however good the numbers | — |
| **dispersal** | an anchor's total evidence mass collapses, persisting N cycles | retire, keep ("a loose community is not a dead one"), or convert to an umbrella | archiving a live community whose members merely spread out (F4 today archives blind) |
| **naming** | any new anchor | title, situation, question, narrative — the retrieval handle | — |

Births are **fission from a parent**, never a whole-graph discovery. That is
principle 4 read backwards, and it is far more stable: *"this sub-region of
anchor H got dense and distinct"* is a local judgment against a fixed
reference, not a global re-derivation. It also matches the one clean channel in
M2 — **0 phantom births** — because a birth only has to be right once, after
which it becomes an anchor and stops being re-derived.

**Fresh clustering survives for exactly one job: finding fission candidates
inside an anchor.** Never across the whole graph.

## 4. What retires, with evidence

| # | what | evidence |
|---|---|---|
| **R-1** | **Orphan placement pipeline** — `_compute_orphan_affinities` (`community_decoder.py:985-1029`), `embedding_placement_threshold: 0.50` (`community_contract.py:42`) | F13: 0.50 raw cosine passes **100%** of random pairs. **And its output is already inert:** it emits `type='node_affinities'` (`:1268-1277`), which is not in the encoder's actionable set (`community_encoder.py:63-66`) — so every run performs a per-member N+1 centroid scan (`:990-993`) that is then dropped. Replaced by `sem_centred` / `knn_share` as first-class evidence terms. |
| **R-2** | **Cross-cutting proposals** — `_detect_cross_cutting` (`:971-981`) | Same inertness: `type='cross_cutting'` is not actionable. The render branches at `community_encoder.py:632` and `:641` are unreachable in production. |
| **R-3** | **Unplaceable rest gate** — `community_decoder.py:127-145`, `community.py:325-356` | M12: suppresses 97.6–99.5% of the unplaced population every run. Under anchors there is no "unplaced" population — every node has evidence against every anchor, and being below threshold is a normal state, not a deferred decision. |
| **R-4** | **Drift proposals** — Step 5c (`community_decoder.py:437-498`), prompt `## DRIFT`, plus per-node `_sys_drift_threshold` state (`:444`) | F5: the encoder moves a node between two communities seeing neither one's membership. Under §3.3 a node "drifts" when its evidence crosses a threshold — computed, not judged. |
| **R-5** | **>60% overlap-conversion** — `cluster_overlap_threshold` (`community_contract.py:49`), conversion at `community_decoder.py:1194-1222` | F18: **39% of adds are converted births** — the mechanism that fed would-be communities into the masses. `_seed_clusters` never unions two clusters (`:832-845`), so a 206-member community is unreachable by clustering; conversion is how it was built. |
| **R-6** | **`add_to_existing`** (quota 12/run) + Step 5b emitter (`:416-435`) + Step 9c contract (`:1288-1355`) | Replaced wholesale by §3.3. F1: judged through title + 150 chars, 0 edges, 0 metadata; `3e499972`: 26.6% of accepted placements sit below the noise line. |
| **R-7** | **`health_update`** (quota 3/run) | Replaced by the dispersal event. The deterministic auto-archive below the cohesion floor (`community.py:241-289`) is **kept** — structural, not judged. |
| **R-8** | **`merge_communities` decoder detection** — `_detect_merge_candidates` (`:928-967`) | Merge becomes two anchors whose evidence profiles converge — visible continuously rather than detected by stored-membership overlap. |
| **R-9** | **Corridor pre-filter** — `community_encoder.py:79-90` | A corridor is a region whose evidence never concentrates. One mechanism, not a special case. |
| **R-10** | **`community_members` / `community_key_decisions` metadata** | F6: zero code readers except `reconcile_community_membership` reading `community_members` as an orphan seed (`dal_graph.py:525-527`). Under §3.3 reconciliation is against the evidence set, so the seed goes dead — **retirement depends on §3.3 landing first.** |

### Two checklist rows resolved by reading the code

- **§P5's ⚠ "the parent→child verb probably must join `non_cohesion_relations`
  or it distorts `internal_fraction`" is FALSE.** Both cohesion adjacency
  builders require `type != 'community'` on *both* endpoints —
  `community_decoder.py:732-735` and `community_structural.py:89-92` — so an
  edge between two community nodes never enters the adjacency. Nothing to add.
- **F15's stated harm is not reachable through the path I opened.** The claim
  that 11 community-as-member edges *"inflate `community_size`, can set
  `community_dominant_type='community'`"* fails against
  `get_community_members` (`dal_graph.py:701`) and `get_members_bulk`
  (`dal_graph.py:742`), which both filter `member.type != 'community'`; the
  structural stamp reads through the latter (`community_structural.py:135`).
  The edges are still junk worth removing; C2 should narrow its rationale.

## 5. Lineage verbs

F14: 12 improvised inter-community verbs, only `absorbed_into` (20) systematic.
**Extend before creating — the existing taxonomy already covers every case:**

| need | verb | aspect | edit |
|---|---|---|---|
| umbrella → child | `part_of` (child → umbrella) | `hierarchical_structure` | none |
| fission lineage | `derived_from` (child → parent) | `extension_refinement` | none |
| merge lineage | `absorbed_into` | `survivor_lineage` | none |

**Explicitly ruled out:** putting fission lineage in `survivor_lineage`.
`resolve_live` walks that aspect, so a living child pointing at its living
parent would make every reference to the child resolve to the parent.

If fission lineage should be legible in its own right, the minimum edit is one
string (`split_from`) in `hierarchical_structure.edge_relations`.
`aspects_v1.json` is a **human edit** and this doc does not make it. No
`REQUIRED_ASPECTS` change is involved either way — that list
(`servers/aspect_store.py:23`) gates adding an *aspect*, not a relation.

## 6. Success metric

Not modularity and not internal fraction — both are proxies for a consumer.
The measurable target is the stated purpose (`6ee28032`): **how many anchors
must you suppress to mute a project?**

Measured today on the brain's real labels (10,099 `brain`, 255 `ex.co`, 3
`abadai`): the ex.co nodes spread across **38–49** communities at production
granularity, and **5–6** at a coarse level. Fewer handles is better, and it is
a number that moves when the architecture improves.

Second-order: journal noise classes → 0, and the F11 umbrella rate.

## 7. What is NOT measured

Stated plainly so nothing here is mistaken for settled:

- **Whether fission candidates inside a giant are detectable and stable.** The
  whole birth/split path rests on it. Next probe.
- **Per-cycle wall-clock cost** of computing evidence for every
  `(node, anchor)` pair with a non-zero term.
- **Whether the agent judges these renders better than the current ones.**
  That needs an encoder eval on production-faithful batches.
- **Whether a coarse anchor layer should be seeded algorithmically** (M6 shows
  Louvain's top level reaches it) or grown by fission from what exists.

## 8. Proposals — changes this doc does not make

| | proposal |
|---|---|
| **P1** | Total sort key `(-z, -is_direct, a, b)` at `community_decoder.py:824`. M3 shows 100% of seed pairs are tied; the ordering is currently arbitrary. Needed for *any* reproducible derivation, including fission-candidate detection. |
| **P2** | Commit the load-bearing probes as instruments: evidence discrimination + stability (M9/M10) and perturbation sensitivity (M8). R0 measured with scratchpad code; gates need instruments in the repo. |
| **P3** | New module `servers/scales/s2/community_reshape.py` — evidence computation + membership reconciliation. |
| **P4** | Wire `edge_relations.embedding` into community evidence — the artifact exists (v26, 23,272 edges re-embedded at `f359f6f`) and community detection has never read it. |
| **P5** | Decide whether `co_anchored` leaves the `noise` aspect for evidence purposes, or is read directly. It is locked into `noise` in `aspects_v1.json` and skipped by `_build_typed_adjacency`. |
| **P6** | Re-run the probe after P1 and re-encode `7fbad66b` / F18 with the corrected weekly velocity (M1). |

## 9. Open decisions

1. **Evidence thresholds** — one per signal, or a combined score? Relative to
   the run's own distribution (fleet-safe) rather than absolute constants; a
   literal tuned on a 10,008-node brain is the same bug class as the inert
   0.50 (`3e499972`).
2. **Coarse anchor seeding** — algorithmic (M6's Louvain top level) or grown
   by fission?
3. **Confirmation window** — how many cycles a fission or dispersal signal must
   persist before the agent is called. At 4.4 runs/day (M12), N=2 is ~5–11h.

---

# Part II — The pipeline as it runs today

*(2026-04-11; stale in places — C3 corrects it at R6, after the work lands.)*

## Status: SHIPPED TO PRODUCTION

114 communities live on production brain (cold start rerun 2026-04-11, 12 orphans self-healed, 1 duplicate merged).
Decoder/encoder split shipped. Dashboard 3D graph with legend click-to-focus. Merge detection with adaptive threshold for young brains. Community split NOT BUILT (roadmap).

## Architecture: S2CD / S2CE Decoder-Encoder Pair

Same pattern as S1R/S1E. Decoder finds structure, encoder characterizes it.

### S2CD — Community Decoder (algorithmic, <1s)

```
_decode()
├── _build_typed_adjacency()         → non-noise edges by aspect (via brain.aspects)
├── _compute_pair_scores()           → z-scores by degree bucket
├── _seed_clusters()                 → z≥1.0 + direct edges
├── _validate_clusters()             → dissolve fragments, flag corridors
├── _compute_affinities()            → every node → {cluster: affinity}
├── _detect_cross_cutting()          → high-degree thin-spread nodes
├── _compute_orphan_affinities()     → embedding centroid matching
├── _analyze_ties()                  → overlap vs split signal
├── _build_proposals()               → community + affinity + drift proposals
└── Incremental: add_to_existing, drift detection, health updates
```

**No static thresholds** — all scoring is relative:
- Z-score pair scoring within degree buckets (adapts to graph density)
- Adaptive grow threshold (median of all affinities)
- Ratio-based overlap (secondary/primary ≥ 0.5)
- Cross-cutting detection (degree ≥ 15, top affinity < 35%)

**Incremental path**: placed nodes excluded from seeding. New nodes matched against existing communities. Drift detected when node's foreign affinity > home affinity × 1.5. Health updates when internal fraction degrades.

### S2CE — Community Encoder (Sonnet, agentic)

Uses `brain_batch` tool to create community nodes directly. Same `run_llm_loop` as S1E.

Prompt (v6, `s2_community_enrichment` interaction):
- "What pattern do these nodes reveal that no single node names?"
- "What would change how the next you approaches this area?"
- FLAT → RICH transformations (summary→insight, history→wisdom, technical→relational)
- Field guide: reads proposal data (int_frac, edge signature, timeline) as story signals

Tools: `brain_batch` (create + revise + connect), `get_nodes` (read).
Target: 1-2 rounds. Batched at 10 proposals per Sonnet call.

### Community Node Structure

Type `community`, encoding_source `s2:community_detection`.

**Visible to Anchor** (in render_rich_node):
- content: narrative with node references and dates
- situation: discriminating recall trigger
- community_narrative: 2-4 sentence arc
- community_key_decisions: "id: title" pairs
- community_open_questions: what's unresolved
- community_latest_development: most recent change
- community_maturity: forming/active/settled/corridor
- community_dominant_type: most common member type
- community_members: "id: title" pairs
- Open fields: community_learning_arc, community_tension, community_risk, etc.

**Hidden from Anchor** (machine-readable for S2/S3):
- community_internal_fraction, community_internal_edges, community_external_edges
- community_centroid, community_is_corridor, community_size
- community_growth_rate, community_run_count

**Source of truth**: `community_member` edges. Metadata is denormalized cache.

## Supporting Infrastructure

### Aspects (`servers/aspects.py`, `servers/scales/s2/aspects_v1.json`)
- Unified taxonomy for both node TYPES and edge RELATIONS (replaces the
  retired `EdgeFamilyIntegration` + `s2_edge_families` interaction)
- Required aspects (`REQUIRED_ASPECTS`) locked + auto-healed at boot; emergent aspects
  discovered by `AspectIntegration` (planned, see SESSION-HANDOFF-2026-05-04)
- Decoder uses `brain.aspects.all()` to build `{relation: aspect_name}` map
  for typed adjacency; `noise` + `generic_relation` skipped as low-signal
- Single API: `brain.aspects.<name>.edge_relations`, `brain.aspects.relations_in([...])`

### Shared S2 Base (`base.py`)
- `_has_new_traces()`, `_read_traces_since()`, `_last_run_timestamp()`
- `_call_llm()` — learnable prompt from interactions table
- `_extract_json()` — robust JSON parsing from LLM responses
- Reusable by all future S2 units (dedup, confidence, etc.)

### Trace Contract (updated)
- S2 O: `s1_delta`, `graph_structure`
- S2 K: `community_proposals`
- S2 delta: `community_enriched`, `community_created`, `recall_quality_signal`

### Eval Harness (`eval/s2_community_decoder_eval.py`)
- Runs the production decoder on an isolated brain copy with simulated
  encoder acceptance — backlog convergence, fingerprint suppression,
  proposal mix across cycles
- `run_decoder()` is the shared seam the community sims
  (`sim_community_journal`, `sim_community_structural`,
  `diag_community_encode`) build on

## Bug Fixes Shipped

1. **Dispatch open fields bug** — `_handle_remember` and `_handle_remember_batch` silently dropped all fields not in contract whitelist. S1E's open fields (`assumed`, `reality`, `trigger`, etc.) were lost for months. Fixed: all fields pass through.

2. **Runner trace enrichment** — S1E delta traces now carry structured `{created, revised, connected}` metadata + full tool input. Previously only summaries.

3. **S1E community node exclusion** — community nodes excluded from S1E's node catalog. S2CE owns community membership, S1E encodes from conversation.

4. **Metadata render filtering** — structural community metrics (internal_fraction, centroid, etc.) hidden from Anchor's view. Only human-readable metadata shown.

5. **register_interaction MCP tool** — added for managing learnable boundaries from conversation.

## What's Next

### Immediate
- **Fatigue redesign** — message-distance-based fatigue replacing count-based. Doc: `docs/FATIGUE-REDESIGN.md`
- **Dashboard polish** — S2 decode/encode display in Live tab, scale filter
- **S2 Weaver/Healer** — connect orphan nodes (635 with no typed edges)

### Short-term
- **Incremental production runs** — idle hook triggers S2CD/S2CE on new S1 traces
- **Boot integration** — community summaries at session start
- **S1R integration** — community_key_decisions prioritized in graph expansion
- **Community merge** — detect and merge converging communities

### Medium-term
- **Other S2 units** — dedup, confidence recalibration, correction chain resolution
- **S3 foundation** — reads S2 traces, evaluates community quality over time
- **Progressive simulation eval** — replay brain history to validate incremental path

## Files Changed

| File | Change |
|------|--------|
| `servers/scales/s2/community.py` | Full rewrite — S2CD + S2CE pipeline |
| `servers/scales/s2/community_contract.py` | Config, metadata schema, node rendering |
| `servers/scales/s2/community_enrichment_prompt.py` | S2CE Sonnet prompt (v6) |
| ~~`servers/scales/s2/edge_families.py`~~ | Edge type classification unit (DELETED 2026-05-04 — replaced by aspects) |
| ~~`servers/scales/s2/edge_families_v1.json`~~ | Initial 21-family classification (DELETED 2026-05-04 — replaced by `aspects_v1.json`) |
| `servers/aspects.py` | AspectRegistry + Aspect value object (post-2026-05-04) |
| `servers/scales/s2/aspects_v1.json` | Unified seed for the required aspects (post-2026-05-04) |
| `servers/scales/s2/base.py` | Shared S2 infrastructure |
| `servers/scales/runner.py` | Structured trace metadata + full tool input logging |
| `servers/scales/s1/encode_contract.py` | Exclude community nodes from S1E catalog |
| `servers/daemon_dispatch.py` | Open fields pass-through fix + register_interaction |
| `servers/daemon_hooks.py` | Simplified idle hook |
| ~~`servers/interaction_seed.py`~~ (DELETED 2026-08-23) | S2CE interaction seeding (legacy s2_*_families seeds removed 2026-05-04; the whole module fell to the override collapse — code owns defaults now) |
| `servers/trace_contract.py` | S2 community ref_types |
| `servers/contract.py` | Metadata render filtering for community nodes |
| `servers/redistribution.py` | Edge-based community lookup |
| `servers/brain_mcp.py` | register_interaction MCP tool |
| `dashboard/brain_dashboard_standalone.py` | 3D graph, Decoding/Encoding tabs, scale filter |
| `eval/s2_community_decoder_eval.py` | Decoder eval harness (production decoder + simulated encoder) |
| `scripts/recover_s1e_open_fields.py` | Metadata recovery script |
| `docs/FATIGUE-REDESIGN.md` | Fatigue redesign spec |
| `dashboard/Dashboard-nextwork.md` | Dashboard roadmap |
