# S2 Community — Design

Two parts, on purpose:

- **Part I — Reshape-diff (R0 design, awaiting ratification).** The target
  architecture. Nothing in it is built. Written 2026-08-28 against the plan of
  record in `docs/S2-COMMUNITY-CHECKLIST.md` §PX.
- **Part II — The pipeline as it runs today.** Kept verbatim below. It is
  stale in places (dated 2026-04-11); correcting it is C3's job, scheduled for
  R6 *after* the work lands, so the doc does not go stale against itself
  mid-flight. Where Part I contradicts Part II, Part I is a proposal and
  Part II is what production does today.

---

# Part I — Reshape-Diff Architecture (R0)

**Status: DESIGN. Not built, not ratified.** Three decisions are Tom's:
the naming bar (§1), hysteresis (§2), migration pacing (§8).

The inversion, in one line: *the algorithm decides who clusters with whom;
the agent decides which of those clusters deserve a name, what they mean, and
what a structural change to one signifies.* Matched structure costs zero
tokens. Agent tokens buy naming, narrative, and structural judgment — nothing
else (`e40ebfee`).

## 0. Evidence base

Every number below was measured **this session (2026-08-28)** unless it cites
a checklist F-row. Two instruments:

- **P** — `eval/community_reshape_probe.py`, rerun unmodified.
- **A/A** — a throwaway harness importing that probe's own `fresh_partition`,
  `diff_against_stored` and `merge_and_birth`, run in separate processes over
  one `IsolatedBrain` copy. Not committed; R1 must land it as a real gate
  (§9-P1).

| # | measurement | value |
|---|---|---|
| **M1** | fresh whole-graph partition (P) | 7,722 edge-connected nodes → **1,168 clusters**, median 4, >10: 74, >40: 2, max 48 (corridors 25, dissolved 10, absorbed 0). 790 live communities, 7,084 of their members edge-connected |
| **M2** | stored vs fresh identity (P) | 770 scored / 20 too sparse. Jaccard ≥0.3: 395 (51%) · ≥0.5: 266 (35%) · ≥0.7: 134 (17%). **Split into ≥2 parts: 216 (28%) · dispersed: 86 (11%) · merge events: 27 · births: 17** |
| **M3** | weekly velocity (P, arm D) | 1,068 old clusters scored; stable ≥0.5: 1,041 (**97%**); **22 splits, 0 merges, 3 dispersals, 43 births**; 489 new connected nodes |
| **M4** | **A/A churn floor — same graph, two processes** | seed 0 vs seed 0: **1,168/1,168 identical (100%)**. seed 0 vs seed 1: **1,088/1,168 identical (93.2%)**, stable ≥0.5 1,140 (98%), and the probe's own differ reports **16 phantom splits, 12 phantom merges, 1 phantom dispersal, 0 phantom births** |
| **M5** | fresh-partition size histogram (A/A dump) | size 3: 409 · ≥4: 759 (65%) covering 4,959 nodes (64% of connected) · ≥5: **501 (43%) covering 3,927 (51%)** · ≥6: 344 (29%)/3,142 (41%) · ≥8: 191 (16%)/2,162 (28%) · ≥10: 104 (9%)/1,438 (19%) · ≥15: 28/571 · ≥20: 7/221 · ≥40: 2/90 |
| **M6** | run cadence (`query_traces` s2 `s1_delta`, 168h) | **31 decode runs / 7 days = 4.4/day**, one per ~5.4h. Community count in those traces: **744 → 787 (+43/week)**. Unplaced population 1,947–1,989; pending after the rest gate 9–47 → **the rest gate suppresses 97.6–99.5% of it every run** |
| **M7** | encoder cost (`community_enriched` COMPLETE, 9 consecutive successful runs 08-25→08-28) | every run: **4 actions / 2 writes / 4 rounds across 2 batches**. Output tokens 2,903–7,397 (mean **5,308**, median 5,661); input 8 + `cache_read` ~71K; wall 76.6–176.6s. One run in the window lost both batches to a connection error |

### Two corrections to the checklist, from M3 and M4

**(a) The weekly split count is not ~2.** F18 / `7fbad66b` record *2 splits,
99% stable*. The same instrument, unmodified, on the same day reports **22
splits, 97% stable**. Do not build on the "2".

**(b) The fresh partition is not a pure function of graph state — and this is
the single most consequential fact in the design.** M4 isolates it: two
processes over *one* `IsolatedBrain` copy agree perfectly when they share a
`PYTHONHASHSEED` and diverge on 6.8% of clusters when they don't. The
mechanism is traced, not guessed: `_compute_pair_scores` materialises a
str-keyed set as a list (`community_decoder.py:765`), which fixes the
insertion order of `raw_shared`/`pair_zscores`, which breaks ties in the
stable sort at `community_decoder.py:824` (`key=(-z, -is_direct)` — exact z
ties are common, because `raw_shared` counts are small integers within a
degree bucket).

Consequence: **phantom structural events (16+12+1 = 29) exceed the real
weekly structural signal (22+0+3 = 25).** Births are the one clean channel:
0 phantom against 43 real.

And hysteresis cannot fix it. Within one daemon process the hash seed is
fixed, so a phantom partition *persists* across runs and any consecutive-run
counter will happily **confirm** it; it flips only at daemon restart — which,
in this repo, is every deploy. Determinism is therefore a hard prerequisite
for cutover, not a tuning nicety (§9-P1).

## 1. The naming bar

### The two layers

| layer | what it is | who owns it | persistence | cost |
|---|---|---|---|---|
| **micro-cluster** | a cluster in the fresh whole-graph partition | the algorithm | a row in the stability table (§5) — no node, no edges | 0 tokens |
| **named community** | a `type='community'` node with title, narrative, situation, question | the agent | node + `community_member` edges | the naming event |
| **umbrella** | a named community anchored to ≥2 micro-clusters | the agent, editorially | same as above + lineage edges to children | the umbrella event |

An umbrella is not a separate node type — it is a named community whose
anchor set has more than one member. This is what preserves Tom's overlap
principle (`df292d31`: *"a piece of knowledge is not part of a single
community"*) and what gives today's giants somewhere to land in migration
without a destructive re-partition.

### The bar: three gates in series

A micro-cluster earns a name only if it clears all three.

1. **Size** — `size >= naming_min_size`. The clustering floor is already 3
   (`min_community_size`, `community_contract.py:16`, enforced at
   `community_decoder.py:855-859`); the *naming* floor is a separate,
   higher dial. This is Decision 1.
2. **Cross-run stability** — the cluster has been matched to itself for
   `confirm_runs` consecutive decode runs at Jaccard ≥ `match_jaccard` (§2, §5).
3. **Narrative-worth** — the agent's judgment, and it must be able to answer
   *no*. A decline is recorded against the membership it was declined for
   (§5), so it neither re-fires next run nor silences the same region forever
   once it has genuinely grown.

Gate 3 is the one that cannot be automated and the reason the agent encoder
stays (`f3cb2f52`). Gates 1 and 2 exist to keep gate 3's inbox small enough
to be worth an agent's attention.

### What each size floor buys (M5)

| `naming_min_size` | named objects if every cluster qualified | node coverage | vs today's 790 communities |
|---|---|---|---|
| 3 | 1,168 | 80% | +48% |
| **5** | **501** | **51%** | **−37%** |
| 8 | 191 | 28% | −76% |

Coverage is the ceiling, not the outcome — gates 2 and 3 cut it further, and
nodes not in any named community are a *correct* resting state under this
architecture, not a backlog.

**Recommendation: 5.** Size-3 clusters are 35% of the partition (409 of
1,168) and are three-node fragments; a 3-node arc rarely has a story that its
three members do not already tell. At 5 the named layer covers half the
connected graph with fewer named objects than exist today — the giants'
membership redistributed into things that were actually derived rather than
accreted (F18).

⚠ **What is not measured:** the *birth rate* at each floor. The probe's birth
definition is hard-coded at `len(fmem) >= 4` and ≥70% novel
(`eval/community_reshape_probe.py:126-128`), giving 43/week (M3). R1's differ
must emit births per candidate floor before the floor is finally set; the
number above is the population, not the flow.

## 2. Hysteresis

### Move 1 — determinism first (prerequisite, not hysteresis)

Given M4, smoothing must not be asked to hide a defect. `_seed_clusters`'s
sort key must be total: `(-z, -is_direct, a, b)` at
`community_decoder.py:824`. Gate: A/A across processes with differing
`PYTHONHASHSEED` yields **100% identical partitions**. Proposal only — §9-P1.

### Move 2 — confirmation counting on identity, not on diff magnitude

The thing that must be stable before an agent is summoned is *a cluster's
identity*, not the size of a delta. So hysteresis is a counter on the
micro-cluster row (§5), not a threshold on a Jaccard difference.

| parameter | value | why this number |
|---|---|---|
| `match_jaccard` | **0.5** | the probe's own identity threshold (`eval/community_reshape_probe.py:216`), so instrument and production speak one vocabulary. 97% of clusters hold it week-over-week (M3) and 98% hold it even under tie-break churn (M4) |
| `confirm_runs` | **2** | at 4.4 runs/day (M6) that is ~5–11h of confirmation. A cluster that forms and dissolves inside half a day is graph motion, not structure |
| `event_min_size` | = `naming_min_size` | one dial, not two |

### The resulting budget

43 births/week ÷ 31 runs/week = **~1.4 birth events per run**; 25 structural
events/week = **~0.8 per run**. Against today's measured *actual* encoder
throughput of 2 writes per run at ~5.3K output tokens (M7), the reshape
event stream is the same order of work the encoder already does — it just
buys structure instead of one-node adds.

The independent cross-check: the current pipeline created **+43 communities
in the last 7 days** (M6), and the probe projects **43 births/week** (M3).
Different definitions, same order — reshape-diff is not a spend increase, it
is a spend *redirection*.

**If Tom ships before the determinism fix**, `confirm_runs` must not be the
mitigation — it does not work on this failure mode (§0(b)). The only
pre-determinism mitigation is to enable **births only** (0 phantom, M4) and
hold splits/merges/dispersals until the A/A gate is green.

## 3. Event taxonomy

Notation: *M* = a confirmed micro-cluster, *C* = a named community, `cover(M,C)
= |M ∩ C| / |C|`. Thresholds reuse the probe's constants so the offline
instrument and the daemon cannot disagree: `COVER_PART = 0.25`,
`COVER_CONTAIN = 0.5`, `BIRTH_NOVEL = 0.7`
(`eval/community_reshape_probe.py:41-43`).

**The default event is no event.** A named community whose anchor set is
unchanged is silent: no render, no call, no tokens. That is the majority of
every run (97% week-over-week stability, M3).

### 3.1 Birth

| | |
|---|---|
| **Trigger** | *M* confirmed ≥ `confirm_runs`, `size ≥ naming_min_size`, and ≥70% of *M* lies outside the membership of every named community |
| **Render** | today's `new_community` payload is already the right shape (timeline / hubs / edge signature / all members, built at `community_decoder.py:1224-1239`, formatted at `community_encoder.py:597`) **plus** the named communities *M*'s members already belong to, rendered in full through the R2 disclosure-invariant render |
| **Encoder ops** | `remember` type=`community` + `connect_to` `community_member` for every member (as today, prompt line 42) — **or decline**, which is recorded (§5) |
| **Failure mode** | **duplicate naming of an existing arc** — the cross-run duplicate the encoder journals today (checklist §P5: *"created today… was created last run on the same topic"*). Today it cannot see the target: `S2CE_COMMUNITY_FORMAT` is content 150 / edges 0 / metadata 0 (`community_contract.py:275-280`) and the community set comes from a member-title recall that need not contain the relevant one (`community_encoder.py:538-591`, F2). The mitigation is the render, not the prompt |

### 3.2 Split

| | |
|---|---|
| **Trigger** | *C*'s membership covered by ≥2 confirmed micro-clusters, each with `cover ≥ 0.25` — the probe's own `diff_against_stored` classification (`eval/community_reshape_probe.py:96-103`) — persisted `confirm_runs` |
| **Render** | *C*'s full narrative + per-part bounded, disclosed member slice derived from edges + the remainder count. The checklist's P5 sketch is the shape; ~16 titles, never the parent's full membership |
| **Encoder ops** | **additive** (`df292d31`): `remember` the child community/communities, connect their members, add lineage child→parent (§6), and `revise` the parent — either narrow its narrative to the surviving core, or promote it to umbrella. Never a destructive re-partition |
| **Failure mode** | **splitting along algorithmic fineness rather than a real seam.** F18's own ⚠ caveat: the conservative seeder fragments genuinely coherent regions, so "N parts" overstates decomposition. Mitigations: the ≥2-parts-each-≥25% trigger, confirmation, and an explicit *"this is one story — keep it"* verdict that is recorded so it stops re-firing |

### 3.3 Merge

| | |
|---|---|
| **Trigger** | one confirmed *M* contains ≥50% of the membership of ≥2 named communities (`eval/community_reshape_probe.py:123-125`), persisted `confirm_runs` |
| **Render** | **both** narratives in full + shared/unique member counts + the containing cluster |
| **Encoder ops** | today's MERGE (revise survivor content, connect unique members, archive the smaller — prompt lines 132-141) **or** the editorial alternative: create an umbrella over both and archive neither |
| **Failure mode** | **destructive rewrite from partial visibility** (F3: `## MERGE` tells the encoder to replace a 1,038–1,626-char narrative it can see 150 chars of). This event must not be enabled before the R2 full-content render lands. That ordering is a correctness constraint, not a preference |

### 3.4 Dispersal

| | |
|---|---|
| **Trigger** | no confirmed micro-cluster covers ≥25% of *C* (`eval/community_reshape_probe.py:99-101`), persisted `confirm_runs` |
| **Render** | *C*'s full narrative + **where the members went** — the micro-clusters and named communities that now hold them |
| **Encoder ops** | `archive` with reason · **keep** (*"a loose community is not a dead one"*, today's HEALTH UPDATE keep branch) · or convert to umbrella if its members now sit under several named children |
| **Failure mode** | **archiving a live community whose members merely spread out.** This is F4 — today's `## HEALTH UPDATE` makes a membership judgment from zero members and its accept branch archives the whole community. The "where the members went" render is what makes the judgment possible at all; a *keep* verdict must be recorded (§5) or the event re-fires every run forever. The deterministic auto-archive below the cohesion floor (`community.py:241-289`) is a *separate*, structural seam and survives unchanged |

### 3.5 Umbrella-shift

| | |
|---|---|
| **Trigger** | *C*'s anchor set changes — a child cluster leaves, a new one arrives, or a one-cluster community becomes multi-cluster (or back) — while *C* still has ≥1 part at `cover ≥ 0.25`, persisted `confirm_runs`, and the arriving/leaving cluster is itself ≥ `naming_min_size` |
| **Render** | *C* + its anchor set before/after + the arriving/leaving cluster's members |
| **Encoder ops** | `revise` the narrative and `community_latest_development`; optionally add/remove lineage edges to child communities |
| **Failure mode** | **this becomes the new `add_to_existing` noise channel.** It is structurally the highest-frequency event and the one most able to consume the whole budget. Mitigations: the minimum-shift size gate above, the hardest quota of the five, and — non-negotiable — it is the *last* event type enabled at cutover, after its live rate has been observed in shadow mode |

## 4. Membership write policy

**The agent decides which communities exist and what they mean. The algorithm
decides who is in them.**

- A named community *C* anchored to micro-clusters {M₁…Mₖ} has membership
  `∪ Mᵢ` — computed, never judged. There is no per-node placement decision
  left to make, which is what retires `add_to_existing`, `drift`, and the
  orphan path in one stroke (§7).
- **Who writes:** one new owner, `servers/scales/s2/community_reshape.py`.
  Edge creation routes through `GraphDAL.add_relation` (`dal_graph.py:999`)
  with `encoding_source='s2:community_reshape'`; removal routes through the
  DAL's archive path with the same tag. No raw SQL against `edges` /
  `edge_relations` (CLAUDE.md: route, don't reach).
- **When:** after the agent's judgment Δ, in the same run, so communities born
  this run get their membership immediately — the position and pattern of the
  existing structural stamp (`community_encoder.py:266-277`). The structural
  stamp then runs *after* it, on the final edge state, exactly as it does
  today (`community_encoder.py:318-369`).
- **Provenance:** `s2:community_reshape` is a distinct `category:process` tag
  from the agent's `s2:community_detection`. That single distinction makes the
  whole mechanism reversible — *everything this mechanism ever wrote* is one
  query — which is what makes staged cutover (R4) and paced migration (R5)
  safe to attempt.

### What makes it loud (checks at the write boundary, per `feedback_loud_at_write_boundary`)

1. **Post-write parity.** For every touched *C*, the live member set must
   equal the anchored union. A mismatch logs
   `_log_error('s2_community_reshape_membership', …)` carrying the diff.
2. **Churn ceiling.** If a single run would change more than
   `max_membership_churn` of a community's members, **refuse the write and
   log** rather than silently rewriting a community's identity. A named
   community whose membership turns over wholesale is a split, a dispersal,
   or a bug — all three deserve an event, none deserves a quiet UPDATE.
3. **Locked communities.** Never remove membership from a `locked` or
   `critical` community; pre-filter as `_auto_archive_dead` already does
   (`community.py:264-268`).
4. **`consecutive_failures`** per the house S2 pattern, so a stuck reshaper
   surfaces rather than degrades.

## 5. Cross-run micro-cluster stability — new state

**This does not exist today.** Micro-clusters have no persistent identity:
cluster ids are per-run integers from a counter (`community_decoder.py:828`).

### Owner and location

Rejected: `s2_rejections` (a fingerprint suppression table keyed on proposal
parameters, `rejection_table.py:46` — different lifecycle, and the reshape
design retires most of its current uses); `brain_meta` (one key/value, and
1,168 rows in a blob is a write-amplification and concurrency hazard);
community-node metadata (micro-clusters are precisely the things that are
*not* nodes).

**Recommendation: a new table `s2_micro_clusters` in `brain.db`**, declared in
`servers/schema.py` on the migration ladder with a version bump, with all its
SQL owned by one module — the exact precedent `s2_rejections` set
(`tests/test_raw_sql_guardrail.py:43` records it as an allowed exception:
*"owns all s2_rejections SQL"*). That ratchet gets one new entry with a
one-line why; growing it silently is what the test exists to prevent.

Keep the store inside `community_reshape.py` until a second consumer appears
(CLAUDE.md: extend before creating; a new module is a structural commitment).

### Shape

| column | meaning |
|---|---|
| `cluster_key` | PK. **Carried forward**, not derived from membership: a fresh cluster matching a stored row at Jaccard ≥ `match_jaccard` inherits its key; an unmatched cluster mints a new one. Membership-derived keys are useless here — they change the moment a member joins |
| `members` / `member_hash` | JSON sorted member ids + a hash for cheap unchanged-detection |
| `size` | current member count |
| `first_seen_at` / `last_seen_at` | ISO, via `clock.iso_now()` — transaction-time, wall-clock anchored (CLAUDE.md time rule) |
| `run_count` / `consecutive_runs` | total runs matched / consecutive runs matched. `consecutive_runs` is the hysteresis counter; it resets to 0 on a miss, and the row survives |
| `named_community_id` | the named community this cluster anchors, or NULL. The anchor relation is many-to-one: an umbrella has several rows pointing at it |
| `encoder_verdict` | `named` · `declined` · `keep` · NULL |
| `verdict_at` / `verdict_member_hash` | **the verdict is scoped to the membership it was made about.** When membership moves materially past a threshold, the verdict expires and the cluster can be proposed again |

That last row is the honest replacement for fingerprint suppression: a
decline about 5 nodes should not silence a proposal about 12, and today's
mechanism cannot tell the difference.

### Lifecycle

- **Written** once per decode run, after the fresh partition, before the
  encoder — so the differ reads the previous run's state and writes this one's.
- **Unmatched stored row:** `consecutive_runs = 0`, row retained. Clusters
  vanish and come back; forgetting them costs confirmation history.
- **Pruned** when `last_seen_at` is older than 30 days *and*
  `named_community_id IS NULL`. Without a prune the table grows without bound.
- **Dangling anchor:** if `named_community_id` points at an archived
  community, clear the pointer — the community died, the cluster did not.
- **Disposable by design.** Dropping the table costs only confirmation
  history: every cluster restarts at `consecutive_runs = 1` and no event
  fires for `confirm_runs` runs. State this property in the module docstring;
  it is what makes the whole mechanism safe to reset during migration.
- **Loud:** a fresh cluster matching ≥2 stored rows that anchor *different*
  named communities is logged — that is either a merge signal or an identity
  bug, and both want to be seen.

## 6. Lineage verbs

F14 measured the current state: 12 improvised inter-community verbs, mostly
singletons, only `absorbed_into` (20) systematic, and no sanctioned
parent/child verb.

### ⚠ A checklist row resolved — no `non_cohesion_relations` entry is needed

Checklist §P5 warns (⚠ UNVERIFIED) that a parent→child verb *"probably must
join `non_cohesion_relations` or it distorts `internal_fraction`."* **Traced
this session: false.** Both cohesion adjacency builders require
`type != 'community'` on **both** endpoints — the decoder's at
`community_decoder.py:732-735` and the structural stamp's at
`community_structural.py:89-92`. An edge between two community nodes never
enters the adjacency at all, so it cannot move `internal_fraction`. Nothing
to add to `non_cohesion_relations` (`community_contract.py:91-93`).

### The verbs

**Extend before creating — the existing taxonomy already covers both cases:**

| need | verb | aspect | edit required |
|---|---|---|---|
| umbrella → child | `part_of` (child → umbrella) | `hierarchical_structure` — *"Edges expressing CONTAINMENT"* | **none** |
| split lineage | `derived_from` (child → parent) | `extension_refinement` — *"the target gets richer, not changed"* | **none** |
| merge lineage | `absorbed_into` | `survivor_lineage` | **none** — already systematic |

**Explicitly ruled out:** putting split lineage in `survivor_lineage`. That
aspect's contract is *"where an archived node's knowledge SURVIVED… the
source was absorbed into the living target"*, and `resolve_live` walks it. A
living split child pointing at its living parent through that aspect would
make every reference to the child resolve to the parent. Actively harmful.

**If Tom wants split lineage legible in its own right** — and there is a fair
argument that "was carved out of" is not "was derived from" — the minimum
edit is one string in `hierarchical_structure.edge_relations`:

```json
"edge_relations": ["part_of", "includes", "contains", "supersedes",
                   "superseded_by", "section_of", "split_from", "…"]
```

`aspects_v1.json` is a **human edit** and this doc does not make it. No
`REQUIRED_ASPECTS` change is involved either way — that list
(`servers/aspect_store.py:23`) gates adding an *aspect*, not a relation. My
recommendation is to start with `derived_from` and mint `split_from` only if
R1's real split inventory shows the distinction carrying weight.

## 7. Retire list, with evidence per item

| # | what | evidence | retires when |
|---|---|---|---|
| **R-1** | **Orphan placement pipeline** — `_compute_orphan_affinities` (`community_decoder.py:985-1029`), `embedding_placement_threshold: 0.50` (`community_contract.py:42`) | F13: 0.50 raw cosine passes **100%** of random pairs; raw has no viable operating point anywhere. `3e499972`: the gate is a max-over-601 decision — 97.9% of sleepers clear any pair-calibrated cut. **And its output is already inert:** it emits `type='node_affinities'` (`:1268-1277`), which is not in the encoder's actionable set (`community_encoder.py:63-66`), so every run computes a per-member N+1 centroid scan (`:990-993`) that is then dropped. Under reshape-diff "orphan" is not a category: an unclustered node is simply in no named community, which is the right answer | R6 (kNN/centred survive as matching + evidence machinery) |
| **R-2** | **Cross-cutting proposals** — `_detect_cross_cutting` (`:971-981`) | Same inertness: `type='cross_cutting'` is not actionable (`community_encoder.py:63-66`). The render branches at `community_encoder.py:632` and `:641` are unreachable from the production path | R6, with R-1 |
| **R-3** | **Unplaceable rest gate** — `community_decoder.py:127-145`, `community.py:325-356` | Measured (M6): 9–47 pending of 1,947–1,989 unplaced across 11 sampled runs — **the gate suppresses 97.6–99.5% of the population every run**. Its purpose (don't re-examine an unmoved neighborhood) is served for free by §5: an unchanged cluster produces no event. PX already expects the ~3,694 rows to retire with the machinery rather than be cleared and re-woken | R6 |
| **R-4** | **Drift proposals** — Step 5c (`community_decoder.py:437-498`), prompt `## DRIFT`, quota 2/run, plus the per-node `_sys_drift_threshold` state (`:444`) | F5: the encoder moves a node between two communities while seeing neither one's membership. Under §4 membership is algorithmic — a node "drifts" when its cluster's anchoring changes, which is an umbrella-shift or a split, not a per-node judgment | R4(b), with algorithmic membership |
| **R-5** | **>60% overlap-conversion** — `cluster_overlap_threshold: 0.60` (`community_contract.py:49`), conversion at `community_decoder.py:1194-1222` | F18 / `7fbad66b`: **39% of adds are converted births** — this is the mechanism that fed would-be new communities into the masses. `_seed_clusters` never unions two existing clusters (`:832-845`), so a 206-member community is unreachable by clustering; conversion is how it was built | R4(a), with births |
| **R-6** | **`add_to_existing`** (quota 12/run) and its Step 5b emitter (`:416-435`) + Step 9c contract (`:1288-1355`) | Replaced wholesale by §4. F1: the encoder judges these seeing the target as title + 150 chars, 0 edges, 0 metadata. `3e499972`: 26.6% of accepted placements sit below the pair-noise line | R4(b) |
| **R-7** | **`health_update`** (quota 3/run) | Replaced by the dispersal event, which renders where the members went. F4: today's accept branch archives a community from a zero-member judgment. The deterministic auto-archive seam below the cohesion floor (`community.py:241-289`) is **kept** — it is structural, not judged | R4(c) |
| **R-8** | **`merge_communities` decoder detection** — `_detect_merge_candidates` (`:928-967`) | Replaced by the merge event, which is derived from the fresh partition rather than from stored-membership overlap | R4(c) |
| **R-9** | **Corridor pre-filter** — `community_encoder.py:79-90` | A corridor is a micro-cluster that fails the naming bar. One mechanism instead of a special case | R4(a) |
| **R-10** | **`community_members` / `community_key_decisions` metadata** | F6: zero code readers, except `reconcile_community_membership` reading `community_members` as a creation-time orphan seed (`dal_graph.py:525-527`). Under §4 reconciliation is against the anchored union, so the seed goes dead — **this item's retirement depends on §4 landing first** | R6 (P1's text edits may land earlier) |

### One checklist claim that needs re-checking before C2 cites it

F15 says the 11 community-as-member edges *"inflate `community_size`, can set
`community_dominant_type='community'`."* Both member readers used by the
structural stamp exclude them: `get_community_members` filters
`member.type != 'community'` (`dal_graph.py:701`) and `get_members_bulk` does
the same (`dal_graph.py:742`), and `compute_community_structural` reads
through the latter (`community_structural.py:135`). The 11 edges are still
junk worth removing, but the stated harm is not reachable through the path I
opened. C2 should re-derive its rationale or narrow its claim.

## 8. Migration pacing (Decision 3)

The first reshape is a one-time reorganisation, not a swap. M2 is the bill:
only **35%** of stored communities match a fresh cluster at Jaccard ≥0.5;
**28% (216) split**, **11% (86) disperse**, 27 merge events, 17 births. The
giants do not decompose cleanly — `bc639843` (206 connected members) has
exactly **one** fresh part covering ≥10% of it, at 13% coverage; `2e6986a2`
(174) has **none**.

That last fact sets the constraint: for the largest communities, reshape-diff
has no opinion to offer. Their membership is not derivable structure (F18),
so "re-anchor them to their fresh clusters" would strand 85%+ of their
members. The migration is therefore **agent-led decomposition, paced**, not a
mechanical re-anchoring:

- **K giants per run**, largest first, each one an explicit event batch.
- The encoder re-narrates children and **chooses** whether the parent becomes
  an umbrella or narrows to its core.
- Lineage edges preserve identity in both directions (§6).
- `backup_before_destructive` per batch — no exceptions (CLAUDE.md).
- Communities that match cleanly (35%) migrate silently: anchor set assigned,
  membership reconciled, no agent call.

Pacing is Decision 3 below.

## 9. Proposals — changes this doc does not make

| | proposal | why it is here and not done |
|---|---|---|
| **P1** | Total sort key `(-z, -is_direct, a, b)` at `community_decoder.py:824` | Touching `servers/` is outside R0. This is the prerequisite for R1's "zero churn on unchanged graph" gate (§0(b)) |
| **P2** | Land the A/A harness as a committed instrument alongside `community_reshape_probe.py`, with the two-seed run as its gate | R0 measured with a throwaway; a gate needs an instrument in the repo |
| **P3** | New table `s2_micro_clusters` + `servers/schema.py` ladder entry + version bump + one `ALLOWED` entry in `tests/test_raw_sql_guardrail.py` | §5; a schema change is R1 work |
| **P4** | New module `servers/scales/s2/community_reshape.py` (differ + membership reconciliation + state store) | §4, §5 |
| **P5** | Optional `split_from` in `hierarchical_structure.edge_relations` | `aspects_v1.json` is a human edit (§6) |
| **P6** | Re-run the probe **after** P1 and re-encode `7fbad66b` / F18 with the corrected weekly velocity | The recorded "2 splits / 99% stable" is wrong (§0(a)) and is a stated gate in R3 |

## 10. Decisions for ratification

1. **Naming bar** — `naming_min_size` ∈ {3, 5, 8}. Recommendation **5**.
2. **Hysteresis** — `confirm_runs` and whether determinism (P1) gates cutover.
   Recommendation: `confirm_runs = 2`, `match_jaccard = 0.5`, and **yes** —
   P1 gates every event type except births.
3. **Migration pacing** — K giants per run and whether migration begins before
   or after steady-state cutover.

Until these are ratified, nothing is built (checklist §PX, R0).

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
