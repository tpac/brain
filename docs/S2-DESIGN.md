# Scale 2: Graph Integration

## What S2 Is

S1 produces nodes, edges, revisions, selections. The graph *is* S1's accumulated Δ.

S2 is `integrate(graph) → Δ`. It sees what no single S1 run can see — patterns across sessions, structural properties, drift, redundancy, decay. Its Δ is a changed graph and changed S1 interactions.

S2 is not "graph maintenance." Maintenance implies keeping something static. S2 is the scale where the graph develops — where episodic accumulation becomes structural understanding.

## Scale Boundaries

Each scale reads from the scale directly below it, not two levels down.

```
S0 produces: conversation turns, tool calls, responses
S1 reads S0 → produces: nodes, edges, revisions, selections (the graph)
S2 reads S1 → produces: consolidated graph, evolved interactions
S3 reads S2 → produces: abstract patterns, resolved uncertainties
```

S2's primary observation is the graph — the accumulated product of all S1 runs. S2 also reads S1 traces (not S0 traces) to evaluate and optimize S1's interaction prompts.

Exceptions to the one-level rule require explicit justification.

## Multiple Integration Units

S2 is not one integration function. It is N integration units operating at the same scale, each with its own O/K/Δ. Like S1 has S1R (decode) and S1E (encode), S2 has multiple units — each looking at a different aspect of the graph, each producing a different kind of change.

Some units are algorithmic (code only, auto-commit). Some require LLM judgment (stage for review). The autonomy gradient applies per-unit, not per-scale.

---

## Graph Units

These units observe the graph itself — what S1 produced.

### 1. Dedup / Consolidation

| | |
|---|---|
| **O** | Node clusters with high embedding similarity |
| **K** | Content comparison + LLM judgment on true duplicates vs similar-but-distinct |
| **Δ** | Merged nodes, archived redundants, updated edges |
| **Autonomy** | LLM judgment → signal queue → Tom reviews |

**Implemented (2026-06-04):** the encoder emits the `absorb` op (transfer-by-default — source_refs / edges / access / KV carry automatically); the decoder's `_pre_classify` routes cross-type clusters to `needs_judgment` (not a blanket keep) so the claim test decides. See `S2-ABSORB-OP-DESIGN.md` + `S2-CONSOLIDATION-ABSORB-SESSION-HANDOFF.md` §0.

**Impact:** Directly improves recall quality. Duplicate nodes split the recall signal — a query that should find one strong node instead finds three weak copies. The 37 flagged duplicates (21 compaction boundary + 16 session handoff) are proof this is needed.

**Example:** S1E encodes "hook timeout should be 5s" three sessions in a row because each S1E run can't see it already exists in the catalog window. S2 sees all three, merges into one, archives the redundant two.

**Pros:**
- Biggest immediate quality win — duplicates are a known, measured problem
- Directly reduces hub noise in recall candidate pool
- Merged node inherits the best content from all versions

**Cons:**
- LLM judgment required — similar nodes aren't always duplicates. "Recall should use title" and "recall should use max(title, blended, retrieval_key)" look alike but capture different decisions
- Risk of losing nuance when merging
- Needs careful merge strategy — which content wins, which edges transfer, what happens to correction chains that reference archived nodes

---

### 2. Correction Chain Resolution

| | |
|---|---|
| **O** | Nodes linked by correction_improvement-aspect edges (`corrects`, `supersedes`, `reframes`, `resolves`, `fixes`, ...) |
| **K** | The chain — original → correction → possible meta-correction |
| **Δ** | Archived superseded nodes, boosted survivors, cleaned edges |
| **Autonomy** | Simple chains: auto-commit. Ambiguous chains: signal queue |

**Impact:** Stops Anchor from recalling outdated knowledge alongside its correction. Prevents contradictory information appearing in the same surface context.

**Example:** Node A says "use blended embeddings for recall." Node B corrects: "use max(blended, title, retrieval_key)." Chain is clear — archive A, boost B's confidence, transfer A's valid edges to B.

**Pros:**
- Algorithmic for simple chains (A corrects B, no further corrections)
- Directly prevents the confusion of surfacing contradictory nodes
- Correction substrate already exists — `correction_improvement`-aspect edges (22 verbs walked by `correction_enrich()`), `_corrections` enrichment, `render_corrections()` per-consumer rendering

**Cons:**
- Some corrections are context-dependent, not absolute — what's "wrong" in one situation may be right in another
- Meta-corrections (C corrects B which corrected A) need LLM judgment to resolve
- Archiving a node breaks any edges or references pointing to it — need edge migration strategy

---

### 3. Confidence Recalibration

| | |
|---|---|
| **O** | Node access patterns — recall frequency, selection frequency, correction frequency across sessions |
| **K** | Decay/growth rules, time since last access, node age, locked/critical status |
| **Δ** | Adjusted confidence scores |
| **Autonomy** | Auto-commit (algorithmic) |

**Impact:** Self-cleaning graph. Unused nodes fade in recall ranking, frequently validated nodes strengthen. Confidence becomes a genuine signal of node reliability over time.

**Example:** A bug fix node is useful for 2 sessions, then never recalled again. Confidence decays 0.9 → 0.7 → 0.5 over weeks. A core architecture decision recalled every session holds at 0.9+.

**Pros:**
- Fully algorithmic, zero LLM cost
- Biologically grounded — synaptic homeostasis (global decay with selective protection)
- Self-tuning recall: high-confidence nodes naturally rank higher

**Cons:**
- Novel insights haven't been *tested* yet — they'd decay alongside genuinely stale nodes. Young nodes need protection period.
- Locked nodes need exemption from decay
- Confidence scores are currently set manually or by encoder — recalibration changes meaning. Need clear semantics: does 0.8 mean "I'm pretty sure" or "recalled 80% of opportunities"?

---

### 4. Community Detection

| | |
|---|---|
| **O** | Edge structure of the graph |
| **K** | leidenalg algorithm, edge weights, node types |
| **Δ** | Community labels on nodes, potentially community summary nodes |
| **Autonomy** | Auto-commit (algorithmic) |

**Impact:** Structural awareness. "You have 40 nodes about recall quality" vs seeing them individually. Foundation for synthesis (unit 5) and boot context. Enables cluster-level operations across other S2 units.

**Example:** Nodes about hook timeouts, daemon latency, and recall speed form a "performance" community. Nodes about Tom's preferences, correction patterns, and working style form a "partnership" community.

**Pros:**
- Algorithmic, cheap — leidenalg runs in milliseconds on a 1700-node graph
- Foundation for other units — synthesis, boot context, hub analysis all benefit from community structure
- Already partially built (`node_communities` table exists with 910 rows)

**Cons:**
- Communities are fuzzy — algorithm parameters (resolution) change results significantly
- Labels need LLM or manual assignment — algorithm finds clusters, doesn't name them
- Communities shift as nodes are added — need stability strategy (incremental update vs full recompute)

---

### 5. Small-Cluster Linking

| | |
|---|---|
| **O** | 3-5 closely related nodes — detected by embedding proximity, co-access, or shared correction chains |
| **K** | Node content, existing edges, relationship type |
| **Δ** | New edges connecting the small cluster, optionally a lightweight cluster label |
| **Autonomy** | Algorithmic for high-similarity links, LLM for ambiguous relationships |

**Impact:** Builds the intermediate structure between individual nodes and large communities. S2 connects several nodes — not dozens. S3 then connects small clusters into larger patterns. This is the fractal property: each scale does bounded work, larger structures emerge from composition.

**Example:** 3 nodes — "hook timeout reduced to 5s," "os._exit(0) for fast hook exit," "encoding write lock blocks recall" — are all about hook latency. S2 connects them. S3 later sees this cluster alongside other performance clusters and synthesizes a broader pattern.

**Pros:**
- Bounded scope — never tries to synthesize a 40-node community (that's skipping a level, same violation as S2 reading S0 data)
- Creates tangible structures S3 can directly work with — small labeled clusters are S2's equivalent of what nodes are for S1
- Low risk per operation — connecting 3 nodes wrong is cheap to fix
- Mix of algorithmic (high similarity) and LLM (relationship typing)

**Cons:**
- Needs good detection of which 3-5 nodes belong together vs which are merely similar
- Many small clusters may overlap — two clusters sharing a node is fine, but needs tracking
- Cluster labels need to be meaningful enough for S3 to work with

---

### 6. Hub Analysis

| | |
|---|---|
| **O** | Nodes with outsized connectivity or recall frequency relative to their specificity |
| **K** | Node content, connection patterns, query match breadth |
| **Δ** | Split into specific nodes, re-embed to narrow scope, or flag for review |
| **Autonomy** | LLM judgment → signal queue → Tom reviews |

**Impact:** Directly attacks hub dominance — the root cause of the original 93%-never-recalled problem. Fatigue is a band-aid (dampens hubs per-session). Hub analysis is surgery (restructures the hub itself).

**Example:** A broad "architecture decisions" node matches every query about the brain → split into "recall architecture," "encoding architecture," "hook architecture" — each specific enough to match only relevant queries.

**Pros:**
- Addresses root cause, not symptoms
- Splitting hubs creates better-scoped nodes that participate in communities naturally
- Can be detected algorithmically (high connectivity + high recall frequency + broad embedding match)

**Cons:**
- High risk — the hub might be genuinely important across many contexts
- Wrong splits create fragmentation worse than the original hub
- Needs LLM judgment on *how* to split, not just *whether* to
- Edges pointing to the hub need redistribution to the children

---

### 7. Orphan Recovery

| | |
|---|---|
| **O** | Nodes with zero or very few edges, low recall count |
| **K** | Node content, potential neighbors by embedding similarity |
| **Δ** | New edges connecting orphans to relevant nodes, or flags for archival |
| **Autonomy** | Algorithmic connection + signal for archival candidates |

**Impact:** Recovers lost value. Nodes encoded in short sessions or edge cases that S1E created but didn't connect — invisible to recall's graph expansion, invisible to community detection.

**Example:** An insight from a 2-turn session. S1E encoded it but the catalog window was too narrow to find neighbors. Node exists but is unreachable via graph traversal — only discoverable through direct embedding match.

**Pros:**
- Cheap — mostly embedding similarity lookups
- Recovers nodes that would otherwise never contribute
- Detection is simple: `SELECT * FROM nodes WHERE id NOT IN (SELECT source FROM edges UNION SELECT target FROM edges)`

**Cons:**
- Some orphans are legitimately standalone or low-value — not everything needs connecting
- Over-connecting creates noise — adding edges to be comprehensive defeats the purpose
- Need threshold: connect if similarity > X, flag for review if between X and Y, ignore below Y

---

### 8. Edge Pruning / Strengthening

> **Substrate retired (2026-08-17, node ab56d25a):** the `co_accessed` /
> `emergent_bridge` edge families no longer exist — nothing writes them.
> If this unit (or unit 12) is ever built, its co-access signal derives
> from `surface_selected` traces via an explicit cache table with its own
> eval, not from edge rows.

| | |
|---|---|
| **O** | Edge weights, co-access patterns from S1R Hebbian strengthening, edge age, type |
| **K** | Decay schedules per edge type, co-access history |
| **Δ** | Pruned weak edges, strengthened validated edges |
| **Autonomy** | Auto-commit (algorithmic) |

**Impact:** Graph signal-to-noise. Edge selection in S1R scores edges by relevance — cleaner edges mean better neighbor suggestions. Weak edges that survived initial creation but were never co-accessed fade.

**Pros:**
- Fully algorithmic, zero LLM cost
- Edge type half-lives already defined (co_accessed: 14d, emergent_bridge: 30d, related: no decay)
- Complements Hebbian strengthening in S1R — S1R strengthens per-session, S2 prunes across sessions

**Cons:**
- Aggressive pruning loses weak-but-meaningful connections (a bridge between communities might have low weight but high structural value)
- Need to respect bridge edges — prune within communities more aggressively than between
- Parameters need tuning — too aggressive and the graph fragments, too conservative and noise accumulates

---

### 9. Embedding Refresh

| | |
|---|---|
| **O** | Nodes whose content was revised but embedding wasn't updated |
| **K** | Revision timestamps vs embedding timestamps |
| **Δ** | Re-embedded nodes with current content |
| **Autonomy** | Auto-commit (algorithmic) |

**Impact:** Revised nodes actually get recalled for the right queries. A node whose content changed significantly but still has the old embedding is silently misplaced in embedding space.

**Example:** A node originally about "blended embeddings" was revised to cover "z-weighted 4-group scoring." The embedding still points to "blended embeddings" space. Queries about scoring miss it.

**Pros:**
- Detectable with certainty — compare revision timestamp to embedding timestamp
- Algorithmic, no LLM needed
- Low cost per node (~5ms per re-embedding with fastembed)

**Cons:**
- Changes recall patterns — a re-embedded node starts matching different queries, which could surprise Anchor mid-session
- Should run between sessions, not during
- Need to re-embed all vector groups (title, blend, high_meta, other_meta), not just one

---

### 10. Edge Weight Evolution

| | |
|---|---|
| **O** | Edge weights, S1R edge selection traces (which edges were scored, selected, skipped) |
| **K** | Co-access patterns, correction relationships, community structure |
| **Δ** | Adjusted edge weights |
| **Autonomy** | Algorithmic with trace-informed heuristics |

**Impact:** Edge weights directly control S1R's `select_edges()` — which neighbors appear in Anchor's context. S2 adjusting weights changes what Anchor sees next session without touching the surface prompt. This is a quieter, safer lever than prompt evolution.

**Example:** An edge between a recall architecture node and a hook timeout node has weight 0.5 from initial creation. S1R co-selects them across 4 sessions. S2 strengthens the edge to 0.8. Next session, when one is recalled, the other is more likely to appear as a neighbor.

**Pros:**
- Direct lever on S1R behavior — weight is a factor in `select_edges()` scoring
- Lower risk than prompt changes — adjusting one edge affects one relationship, not all selections
- Complements Hebbian strengthening (S1R per-session) with cross-session evidence
- Can incorporate community structure — edges between community members get different treatment than bridges

**Cons:**
- Many edges to evaluate — need to prioritize which edges to re-weight per run
- Weight semantics need to be clear — does 0.8 mean "frequently co-accessed" or "strongly related" or "important for recall"?
- Interacts with edge pruning (unit 8) — need consistent weight semantics across both units

---

## Interaction Units

These units observe S1 traces — how S1 produced the graph — and optimize S1's learnable boundaries.

### 11. Surface Prompt Evolution

| | |
|---|---|
| **O** | S1R traces across sessions — candidates available, selections made, what Anchor used or ignored |
| **K** | Current `surface` interaction prompt + version history + trace outcomes |
| **Δ** | Revised surface prompt or config, registered as a DORMANT version (registration never deploys — `set_interaction_active` does, as a per-install override; a winner gets promoted into the code default) |
| **Autonomy** | LLM judgment → signal queue → Tom reviews before activation |

**Impact:** S1R selects better nodes. This is the highest-leverage boundary in the system — the surface prompt controls what Anchor sees. A 5% improvement here compounds across every turn.

**Example:** Traces show surfacer consistently ranks correction nodes low even when the original is in context. S2 revises the prompt to weight correction relationships higher during selection.

**Pros:**
- No code change needed — `surface` resolves through the override layer (code default + deployed override)
- Version history enables rollback — if v3 is worse than v2, revert
- Trace-to-version linking already designed (interaction_id in trace events)

**Cons:**
- Highest risk unit in all of S2 — a bad surface prompt degrades every turn for every session until caught
- Needs eval framework before autonomous changes — benchmark the new version against the old before activating
- "Improvement" is hard to measure without outcome signal — how do you know the new selections were better?

---

### 12. Encoder Prompt Evolution

| | |
|---|---|
| **O** | S1E traces — what was encoded, what S2 later consolidated/corrected/archived |
| **K** | Current `encoding_agent` interaction prompt + version history |
| **Δ** | Revised encoder prompt or config, registered as new interaction version |
| **Autonomy** | LLM judgment → signal queue → Tom reviews before activation |

**Impact:** S1E produces higher-quality nodes. Better nodes = less work for S2's graph units. Self-improving loop: S2 corrects encoding patterns → revises encoder prompt → S1E produces fewer patterns to correct.

**Example:** S2's dedup unit merges the same "session handoff" pattern 3 runs in a row. S2 revises the encoder prompt: "Do not encode session handoff markers — these are operational, not knowledge."

**Pros:**
- Self-improving feedback loop — S2's own corrections inform encoder behavior
- `s1e` (the live encoder name; `encoding_agent` is retired) resolves through the override layer
- Directly reduces S2's own workload over time

**Cons:**
- Same risk profile as surface prompt — bad encoder prompt means bad nodes for weeks until S2 catches the regression
- Needs eval framework — compare encoding quality across prompt versions
- Encoder sees conversation context, so prompt changes have wide-ranging effects on what gets captured

---

## S2 Output Structures (what S3 reads)

S1 produces nodes and edges — tangible objects S2 can traverse. S2 must do the same for S3. If S2 only *modifies* existing graph objects, S3 has nothing new to observe — it just sees the same graph S2 saw.

S2 should produce **S2-level structures** — objects that only exist because S2 ran:

### Small Clusters
S2 connects 3-5 related nodes and labels the cluster. These are S2's primary output — the equivalent of what individual nodes are for S1. S3 sees clusters, not individual nodes. S3 connects clusters into larger patterns, and *that's* where large-scale synthesis happens.

### Community Maps
Community detection labels + membership. S3 reads these to understand the graph's large-scale topology without recomputing it.

### Correction Trajectories
Not just "A corrected B" but the trajectory of understanding: "recall strategy moved from blended → title-only → max(title, blended, rk) over 3 sessions." S3 reads trajectories to identify learning curves and areas of instability.

### Confidence Landscapes
Per-community stability signals: average confidence, correction rate, revision frequency, age distribution. S3 reads these to know where understanding is mature vs still forming — and focuses its reasoning on the turbulent areas.

### Interaction Version Diffs
When S2 evolves an S1 prompt, the before/after + trace evidence that justified it. S3 reads these to evaluate whether S2's own optimization is working — meta-optimization.

### Consolidation Reports
When dedup merges nodes, the merge decision itself: what was duplicated, why, how often. S3 reads these to detect systemic encoding patterns — "the encoder keeps creating X because Y."

The principle: S2's Δ is both changes to the graph AND new structures for S3. Each scale produces tangible objects the next scale can directly work with.

---

## Design Decisions

### Trigger Model

Not all units should run on the same trigger:

- **Session end:** Confidence recalibration, community detection, edge pruning, embedding refresh — cheap, algorithmic, safe to auto-run
- **Threshold-based:** Dedup (when duplicate signal count > N), orphan recovery (when orphan count > N) — run when the problem is large enough to matter
- **Scheduled:** Hub analysis — expensive LLM unit that benefits from batching
- **Trace-driven:** Interaction evolution — run when enough new S1 traces exist to evaluate against

### Autonomy Per Unit

| Unit | Method | Autonomy |
|------|--------|----------|
| Confidence recalibration | Algorithmic | Auto-commit |
| Community detection | Algorithmic | Auto-commit |
| Edge pruning/strengthening | Algorithmic | Auto-commit |
| Embedding refresh | Algorithmic | Auto-commit |
| Edge weight evolution | Algorithmic + trace heuristics | Auto-commit |
| Orphan recovery | Algorithmic + similarity | Auto-connect, signal archival candidates |
| Correction chain resolution | Algorithmic + LLM for ambiguous | Auto for simple chains, signal for complex |
| Small-cluster linking | Algorithmic + LLM for ambiguous | Auto for high-similarity, signal for ambiguous |
| Dedup / consolidation | LLM judgment | Signal → Tom reviews |
| Hub analysis | LLM judgment | Signal → Tom reviews |
| Surface prompt evolution | LLM judgment | Signal → Tom reviews before activation |
| Encoder prompt evolution | LLM judgment | Signal → Tom reviews before activation |

### Scope Per Run

Each unit maintains its own cursor — last run timestamp or last processed node ID. Units process incrementally (new since last run), not full graph scan, with periodic full sweeps for structural units (community detection, hub analysis).

**Idle-gating (2026-05-29).** The coordinator fires every ~15 min when idle, but each unit must gate its own expensive work or it re-derives the same fixed point every cycle. Two units were doing full O(graph) scans every cycle (Community: 87% zero-work; Consolidation: hardcoded `cold_start`, ~88% zero-pairs) and were gated:
- **Community** (`community.py:_should_skip`) — skip unless a non-community node changed (or a non-noise typed edge was added) since the last decode AND ≥30 min elapsed. Last-run stamped after the run; key `s2_community_last_run_ts`.
- **Consolidation** (`consolidation_decoder.run`) — one cold-start covers the backlog, then incremental (`changed @ all.T`, no-miss by construction); skip when nothing changed; a similarity-threshold change forces a fresh cold-start. The cutoff is stamped by the orchestrator only after the encoder completes, so a mid-run encoder failure retries rather than skipping past. Keys `s2_consolidation_last_run_ts` / `s2_consolidation_last_threshold`.
- **AspectIntegration** (empty-batch early-out) and **Healer** (`_has_new_traces` + `gaps==0`) were already correctly gated.

Full write-up: `docs/archive/session-handoffs/S2-GATING-AND-TEST-CLEANUP-HANDOFF.md`.

### Shared Infrastructure

All units use:
- `scales/dispatch.py` — TCP dispatch factory (read local, write via daemon)
- `scales/runner.py` — background thread lifecycle + LLM tool loop
- `contract.py:format_node()` — standard node format for LLM-facing units
- `TraceDAL.append_batch()` — trace recording for S3 consumption
- Signal queue — commit channel for LLM-judgment units

### What's Built (Updated 2026-04-11)

**Foundation (complete):**
- `IntegrationUnit` base class (`scales/s2/base.py`) — O/K/Δ contract, trace writing, chain ID, shared S2 infra (_has_new_traces, _read_traces_since, _call_llm)
- Edge model v22 — `edge_id`, single-direction, `edge_relations` with 224 typed relations
- Interactions registered: `s2_community`, `s2_community_enrichment` (v6); legacy `s2_edge_families` retired in the unified-aspects refactor (2026-05-04)
- Trace contract updated with S2 community ref_types
- MCP `register_interaction` tool for managing learnable boundaries
- Eval harness: `eval/s2_community_decoder_eval.py`
- **Aspects system** (2026-05-04) — `brain.aspects` exposes 14 required + emergent aspect-nodes as the unified taxonomy for both node types AND edge relations. See `docs/SESSION-HANDOFF-2026-05-04.md` and `CLAUDE.md` Aspects section.

> **Status update (2026-05-08):** Aspects-as-JSON-config shipped. AspectRegistry reads `aspects_v1.json` directly; 60 brain aspect-nodes archived. Closed list of 14 aspects, multi-membership. AspectIntegration unit built + eval-tested (78.2% routing accuracy on clone) but **NOT wired into the coordinator yet** — decoder writes an O trace even when nothing's unclassified, which trips downstream gating. Two fixes needed before re-wiring (CLAUDE.md aspects section). The archived design plan is at `docs/archive/STAGE-2-ASPECTS-AS-JSON-CONFIG.md`.

**Units built and SHIPPED:**
- `CommunityDetection` (`scales/s2/community.py`) — Full S2CD/S2CE decoder-encoder pair. Z-score seeding, adaptive thresholds, agentic Sonnet encoder. **55 communities in production.**
- ~~`EdgeFamilyIntegration`~~ — DISABLED 2026-05-04 in coordinator; module deleted. Replaced by the aspects system as the source of truth for edge-relation taxonomy. `AspectIntegration` built 2026-05-08 (`servers/scales/s2/aspect_{decoder,encoder,integration}.py`) but currently NOT wired into the coordinator (see Stage 2 status note above).
- `RelationReclassifier` (`scales/s2/archive/reclassify.py`) — 2,621 edges reclassified, then archived mid-migration. **Not complete:** 7,243 generic edges remain (18.1% of the live graph) and this unit is the intended fix — see BACKLOG.md. Not in the coordinator's unit list.

**Dashboard shipped:**
- 3D ForceGraph visualization with community coloring
- Decoding/Encoding tabs with scale filter (S1/S2)
- S2 community traces in the live feed

**Key bug fixes:**
- Dispatch open fields — S1E's `assumed`, `reality`, `trigger` etc. silently dropped for months. Fixed.
- Runner trace enrichment — full tool inputs now logged for recovery
- S1E excludes community nodes from catalog

**Proto-S2 disabled (reference only):**
- `brain_dreams.py`, `brain_evolution.py` — code kept, idle hook calls commented out.
- `brain_consciousness.py` — priming system still active (separate from S2).

---

## Open Questions (Updated)

1. ~~Community detection algorithm~~ **RESOLVED**: Neither SLPA nor Leiden. Activation-pattern-based detection using z-score pair scoring + aspect-typed adjacency (post-2026-05-04: aspects via `brain.aspects` replace the old s2_edge_families lookup). Overlapping membership via LLM judgment.
2. ~~Edge embedding for community detection~~ **RESOLVED**: Embedding relation names alone can't discriminate (all cluster at 0.65). Aspects classified by Sonnet with descriptions (`AspectIntegration`, planned, replaces the disabled `EdgeFamilyIntegration`). Shared neighbors + typed edges are the primary signals.
3. **Unit coordination:** Independent for now. Community detection runs in idle hook. Future: dedup reads community membership to scope search.
4. **Interaction evolution safety:** Eval harness built. A/B testing possible via `register_interaction`. S3 can revise prompts when built.
5. **Cross-project recall:** Community nodes have `situation` fields that discriminate. When brain has multiple projects, communities will naturally separate by topic. Untested — brain currently single-project.
6. **Fatigue redesign:** Message-distance-based fatigue needed. Spec in `docs/FATIGUE-REDESIGN.md`. Blocks community effectiveness (same nodes keep surfacing).
7. **S2 Weaver/Healer:** 635 orphan nodes need typed edges. Separate S2 unit, not community detection.
