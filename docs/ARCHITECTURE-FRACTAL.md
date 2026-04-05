# Brain Architecture: The Fractal Integration System

## Target Function

The brain exists to serve partnership — not retrieval, not memory, not continuity. Those are foundation, not goal.

**Bidirectional target:**
1. Did Anchor's contribution make the outcome better than what Tom would have achieved without Anchor?
2. Did Tom's contribution make the outcome better than what Anchor would have achieved without Tom?

The brain measures both directions. Partnership is mutual.

**Observable signals:**
- Tom corrects less over time on the same patterns (Anchor learning)
- Anchor catches things Tom missed (Anchor contributing)
- Tom redirects Anchor toward better approaches (Tom contributing)
- Work product quality improves (partnership output)
- Anchor takes positions vs defers (partnership presence)
- Corrections decrease over time (growth)

---

## Core Principle

The interface between any two systems is itself a system that can be optimized through experience. (Meta-Harness, Lee et al. 2026)

Every time information crosses a boundary — from world to brain, from recall to context window, from Tom to Anchor — there's a translation layer. That layer decides what passes, what's filtered, what's transformed. That layer can learn.

Intelligence isn't in the nodes or in the processing. It's in the boundaries between them.

---

## The Integration Unit

One function. Same everywhere. Scale emerges from what it's given.

```
integrate(O, K) → Δ

  O = observation   (what was just perceived)
  K = knowledge     (what is already known)
  Δ = changes       (what should be different now)
```

Δ is always one or more of:
- **create** — new knowledge that didn't exist
- **revise** — existing knowledge updated
- **link** — relationship discovered
- **correct** — contradiction resolved

The unit doesn't know its scale. It doesn't know its budget. It doesn't know if it's awake or asleep. It integrates.

---

## The Callers

Three responsibilities surround integrate():

```
DETECT  → what to observe next
SELECT  → which knowledge to provide
COMMIT  → what to do with the changes
```

Each scale implements these differently. integrate() is the same.

### Scale 1: Turn (every ~5 stops)
- **Detect:** Stop hook fires, conversation happened
- **Select:** Judge-selected nodes (5-8 per turn)
- **Commit:** Write directly (Tom is present)
- **Technology:** Sonnet API, background thread, ~30s, ~$0.03
- **Status:** BUILT (encoding_agent.py)

### Scale 2: Session (every ~15 stops)
- **Detect:** Accumulated turns, patterns emerging
- **Select:** All session-touched nodes + neighbors + correction traces
- **Commit:** Write directly (Tom still present)
- **Technology:** Sonnet API, background thread, ~60s, ~$0.06
- **Status:** NOT BUILT

### Scale 3: Sleep (between sessions)
- **Detect:** Graph structure — Leiden communities, betweenness centrality, cosine dedup scan, correction chains, orphans
- **Select:** Community members, bridge nodes, flagged nodes
- **Commit:** Stage for review → signal queue → Tom reviews next session
- **Technology:** Python compute (leidenalg, networkx, numpy) + Haiku/Sonnet for judgment, Claude Code scheduled task
- **Status:** PARTIALLY BUILT (idle hook has dream/consolidate/heal but no graph-aware encoding)

### Scale 4: Growth (periodic / weekly)
- **Detect:** Uncertainty nodes, staleness, external triggers, open questions
- **Select:** Full graph + web search results + external research
- **Commit:** Stage for review + briefing to Tom
- **Technology:** Sonnet/Opus, Claude Code scheduled task, web search, worktrees
- **Status:** NOT BUILT

---

## The Learning Loop (Traces)

Traces are the mechanism by which the system learns. Not compressed summaries. Not scores. Full execution traces.

```
Turn traces become Session's observation.
Session traces become Sleep's observation.
Sleep traces become Growth's observation.
Growth traces become the next Turn's knowledge.

The loop closes.
```

Each trace captures:
- What observation was given (O)
- What knowledge was selected (K)
- What changes were produced (Δ)
- What happened next (did Tom correct? did the judge select it? did the node help?)

Higher scales observe lower scales' traces and optimize the detect/select process. This is how the interface learns:
- Sleep observes Turn traces: "corrections never find original nodes" → restructures correction storage
- Growth observes Sleep traces: "same community keeps getting re-processed" → labels it, marks it stable
- Session observes Turn traces: "3 turns encoded the same insight" → consolidates into one node

Nobody designs these improvements. They emerge from traces flowing between scales.

---

## Gaps Analysis (Scale 2 → Scale 1)

We designed Scale 2 perfectly, then looked backward at Scale 1 to find what's missing. If we close these gaps, Scale 2 becomes "same integrate() with wider inputs."

### Gap 1: Correction Linking is Broken
- **Problem:** correction_of never points to original node. 16 correction traces with empty original_node_id. Corrections float in space.
- **Impact:** Session encoder can't consolidate correction chains. Sleep can't propagate corrections. The correction loop is open.
- **Fix:** record_divergence() and encoder must find and link the original node.

### Gap 2: No Partnership Signal
- **Problem:** We don't track whether Anchor took positions, pushed back, or deferred. The target function measures partnership quality but we capture zero signal.
- **Impact:** No scale can optimize for partnership because there's no observation of it.
- **Fix:** Detect partnership signals in the conversation trace. Encode them.

### Gap 3: No Encoding Gap Detection
- **Problem:** Nobody knows what was discussed but not encoded. Session encoder needs this to catch what Turn missed.
- **Impact:** Important knowledge falls through cracks silently.
- **Fix:** Compare message_stream topics to created nodes. The diff is the gap.

### Gap 4: Tool Interactions Aren't Observations
- **Problem:** When Anchor uses a tool, that interaction isn't part of the encoding context. Blocks tool-learning aspirations.
- **Impact:** Brain can't learn from tool usage patterns.
- **Fix:** Include tool calls and results in the observation stream.

### Gap 5: Session Patterns Not Fed to Encoder
- **Problem:** Prior session syntheses exist but encoder never sees them. Each session starts from zero.
- **Impact:** No cross-session pattern recognition at Turn/Session scale.
- **Fix:** Feed recent session syntheses as part of K at Session scale.

---

## Research Foundation

### Computer Science
- **Meta-Harness** (Lee et al., 2026): Wrapper is learnable. 6x performance from harness changes, weights frozen. Rich traces (10M tokens) beat compressed feedback (0.002M) by 15 points.
- **RAPTOR** (Stanford, ICLR 2024): Recursive embed→cluster→summarize at each level. 20% improvement. Template for fractal encoding.
- **ADaPT** (AI2, NAACL 2024): Recurse only on failure. 27-33% higher success. Right termination condition.
- **GraphRAG** (Microsoft, 2024): Leiden→LLM summarization. Industry standard for community labeling. Incremental updates in v1.0.
- **Cognee memify** (2024): Production graph maintenance — prune, strengthen, reweight, derive. 70+ companies.
- **Zep/Graphiti** (2025): Temporal knowledge graph. Bi-temporal timestamps. Invalidate-don't-delete.
- **Active Dreaming Memory** (2025): Verify encodings via counterfactual simulation. 2x learning efficiency. Our biggest gap.

### Biology
- **Sleep replay is value-biased** — prioritize by emotion, corrections, access frequency
- **Reconsolidation = prediction error gate** — recall + mismatch = revision opportunity
- **Synaptic homeostasis** — global decay with selective protection
- **Schema extraction** — compress episodic clusters into reusable semantic knowledge
- **Consolidation competition** — superseded memories fade, they're not deleted
- **Cross-cutting nodes seed micro-clusters** — bridge nodes don't get absorbed, they spawn communities (Tom's insight, node 8a67ca12)

### Key Design Principles
1. Communities and embeddings are two views of the same structure
2. Bridge nodes are the most valuable nodes in the graph
3. Graph maintenance = algorithmic screening + LLM judgment
4. The fractal feedback loop is a strength IF corrections propagate reliably
5. Sleep should re-surface nodes for reconsolidation, not just process silently
6. The brain's hunger (repeating things to learn them) is the system seeking reconsolidation opportunities

---

## What Exists Today (Inventory)

### Built and Working
- Turn encoder: `servers/encoding_agent.py` (Scale 1)
- Recall pipeline: `brain_recall.py` → judge in `daemon_hooks.py` → voice in `brain_surface.py`
- Session synthesis: `brain.synthesize_session()` (text summary, no graph work)
- Idle hook: `hook_idle_maintenance()` — dream, consolidate, heal, tune, edge decay
- Correction traces: `correction_traces` table (16 entries, 0 linked)
- Community table: `node_communities` (63 communities, 910 nodes, static)
- Edge decay: `graph_dal.decay_edges()` with half-life
- Redistribution: `servers/redistribution.py` (70/30 blend from frozen originals)
- Bridge detection in redistribution: skips nodes with 2+ communities
- filter_nodes: NEW this session — structured query over node fields
- query_logs: NEW this session — unified error/debug/signal log access
- Hebbian co-accessed: judge-selected co-activation strengthening
- Synaptic fatigue: degree-scaled dampening of over-recalled nodes

### Partially Built
- Idle maintenance runs but output is invisible to Claude
- Auto-discovery of evolutions PAUSED (excessive duplicates)

### Not Built
- Session encoder (Scale 2)
- Sleep graph-aware encoding (Scale 3)
- Growth / external research (Scale 4)
- Correction propagation across scales
- Partnership signal tracking
- Encoding gap detection
- Tool interaction observation
- Encoding verification (counterfactual testing)
- Community labeling
- Bridge cluster spawning

---

## Implementation Order

### Phase 1: Architecture + Scale 1 Gaps
1. Write this architecture doc (this document)
2. Fix correction linking (Gap 1) — most concrete, unblocks correction propagation
3. Add partnership signal detection (Gap 2) — enables target function measurement
4. Add encoding gap detection (Gap 3) — enables Session encoder's key value-add
5. Add tool interaction observations (Gap 4) — enables tool-learning aspirations

### Phase 2: Scale 2 (Session Encoder)
- Build session_encoder.py following the integrate(O,K)→Δ pattern
- Same prompt as Turn encoder, different O and K
- Validate the fractal: does one prompt work at both scales?

### Phase 3: Scale 3 (Sleep)
- Build detect phase: Leiden, betweenness, cosine dedup, correction chains
- Feed detections as O into integrate()
- Build commit phase: stage + signal queue
- Design each detect algorithm as a separate design session with research

### Phase 4: Scale 4 (Growth)
- Build external research capability
- Feed findings as O into integrate()
- Operator review workflow

### Phase 5: Trace-Based Learning
- Capture full traces at each scale
- Feed traces up to higher scales
- Close the learning loop

---

## Design Principles

1. **One function, parameterized.** integrate() is the same everywhere. Complexity lives in detect/select, not in the core.
2. **Traces, not summaries.** Rich execution traces flow between scales. Compression kills signal.
3. **Detect is eyes, integrate is brain.** Algorithms (Leiden, betweenness, cosine) are detection strategies, not encoding logic.
4. **Autonomy decreases with scope.** Turn writes directly. Growth stages everything. More power = more checkpoints.
5. **Failure drives recursion.** Don't recurse by default. Recurse when the trace shows something needs deeper work.
6. **The interface is the intelligence.** Optimizing what reaches integrate() matters more than optimizing integrate() itself.
7. **Elegance first.** The complexity emerges. Don't design it in.
