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

  O = observation   (everything available at this moment)
  K = knowledge     (what is selected as relevant from O)
  Δ = changes       (what is produced — the response, the encoding, the reorganization)
```

Δ is always one or more of:
- **create** — new knowledge that didn't exist
- **revise** — existing knowledge updated
- **link** — relationship discovered
- **correct** — contradiction resolved

The unit doesn't know its scale. It doesn't know its budget. It doesn't know if it's awake or asleep. It integrates.

**The formula is fractal all the way down.** Scale 0 IS integrate(O,K)→Δ — not below it. O = everything available, K = the message that triggers, Δ = the response. The conversation itself is a continuous integration loop between two partners.

---

## The Callers

Every scale has the same structure:

```
detect()                → something happened, time to integrate
select()                → from everything available, what matters?
integrate(O, K) → Δ    → produce the change
commit(Δ)               → make it real
trace(O, K, Δ, outcome) → record for higher scales
```

### Scale 0: Exchange (every turn)
The raw partnership interaction. Both Tom and Anchor operate here.
- **O:** Everything available — recalled context, tool results, prior conversation, mental state
- **K:** The latest message (from Tom or Anchor) — the trigger
- **Δ:** The response — new text, tool calls, decisions
- **Commit:** The response is sent
- **Note:** O at Scale 0 is irreducibly personal — "everything and nothing, the mental each of us comes with." Capturing it fully may be impossible; higher scales infer it from patterns.

### Scale 1: Turn (every ~5 stops)
The brain's first processing pass — recall, judge, encode.
- **Detect:** Stop hook fires, counter % 5 == 0
- **O:** S0 traces (last 15 messages — 3 runs worth of context)
- **K:** Judge-selected nodes from those turns (via node catalog)
- **Δ:** Encoded nodes, revised nodes, new connections
- **Commit:** Write via dispatch (brain_batch)
- **Technology:** Sonnet API, background thread, ~30s, ~$0.03
- **Status:** BUILT (scales/s1/encode.py). Wider window (15 msgs) replaces need for separate session scale.

### Scale 2: Sleep (between sessions)
Graph-wide operations that need the full picture.
- **Detect:** Session end, or graph structure signals (communities, dedup, orphans)
- **O:** S1 traces across sessions, community structures, correction chains
- **K:** Community members, flagged nodes, cross-session patterns
- **Δ:** Community labels, merges, confidence adjustments, consolidation, reconsolidation
- **Commit:** Stage for review → signal queue → Tom reviews next session
- **Technology:** Python compute (leidenalg, networkx, numpy) + Haiku/Sonnet
- **Status:** PARTIALLY BUILT (idle hook has dream/consolidate/heal)

### Scale 3: Reasoning (periodic)
Abstract patterns, curiosity, uncertainty resolution.
- **Detect:** Uncertainty nodes accumulate, stale patterns, curiosity signals
- **O:** S2 traces + full graph + uncertainty/curiosity nodes
- **K:** Open questions, recurring patterns, cross-project bridges
- **Δ:** Abstract insights, resolved uncertainties, new questions
- **Commit:** Stage for review + briefing to Tom
- **Technology:** Sonnet/Opus, Claude Code scheduled task
- **Status:** NOT BUILT

### Scale 4: Growth (weekly)
External knowledge and long-term evolution.
- **Detect:** Stale decisions, external triggers, research questions from S3
- **O:** Full graph + web search results + external research
- **K:** Stale decisions, open questions, uncertainty nodes
- **Δ:** New knowledge from research, updated decisions, cross-project bridges
- **Commit:** Stage for review + briefing to Tom
- **Technology:** Opus, Claude Code scheduled task, web search, worktrees
- **Status:** NOT BUILT

### Autonomy Gradient
More scope = more time = less autonomy = more checkpoints.

| Scale | Tom present? | Correction source | Commit model |
|-------|-------------|-------------------|--------------|
| S0 | Yes | Immediate | Response sent |
| S1 | Yes | Same session | Write via dispatch |
| S2 | No | Signal queue → next session | Stage for review |
| S3 | No | Signal queue → next session | Stage for review |
| S4 | No | Briefing → Tom reviews | Stage + briefing |

---

## Traces

Traces are the mechanism by which the system learns. Not compressed summaries. Not scores. Full execution traces with substance or pointers.

### Structure
Every trace chain follows O → K → Δ → outcome:
- **O** — what was observed (candidates, query, conversation, graph structure)
- **K** — what knowledge was selected (judge-selected nodes, community members)
- **Δ** — what changed (additionalContext, encoded nodes, reorganized graph)
- **outcome** — what happened next (corrections, future recalls, Tom's response) — added retrospectively

### Data Principles
- **Substance, not metadata.** Bad: "23008 chars context, 8 tools." Good: "query + all 25 candidates with id|title|score|type."
- **Pointers for large content.** Encoding prompts reference interaction_id + file path. Higher scales follow pointers when they need detail.
- **No summaries.** Compression kills signal. Meta-Harness proved rich traces (10M tokens) beat compressed feedback by 15 points.

### Scale Flow
```
S0 traces become S1's observation.
S1 traces become S2's observation.
S2 traces become S3's observation.
S3 traces become S4's observation.
S4 traces become the next S0's knowledge.

The loop closes.
```

Higher scales observe lower scales' traces and optimize the detect/select process:
- S3 observes S1 traces: "corrections never find original nodes" → restructures correction storage
- S4 observes S3 traces: "same community keeps getting re-processed" → labels it stable
- S2 observes S1 traces: "3 turns encoded the same insight" → consolidates into one node

Nobody designs these improvements. They emerge from traces flowing between scales.

### What's Built
- `trace_events` table in brain_logs.db (chain_id, scale, event_type, ref_type, ref_id, summary, metadata, session_id, interaction_id)
- S0: K (user messages, 2000 chars), Δ (assistant responses, 2000 chars), Δ (tool results via PostToolUse for ALL tools)
- S1 recall: O (candidates with id|title|score|type), K (judge-selected + expanded neighbors), Δ (full additionalContext)
- S1 encode: O (pointer to prompt file + turn count), K (node catalog IDs), Δ (action details: tool + title per action)
- Dashboard Traces tab: scale filter, time range (1h/6h/24h/7d), auto-refresh 5s, lazy loading

### What's Missing
- Outcome events at every scale (the learning signal)
- S0 O (the full context — may be irreducible, higher scales infer it)
- Intermediate assistant text between tool calls (not available through hooks)
- Mental maps / internal models (emerge at S2 from S0+S1 patterns)

---

## Interactions

Every learnable boundary is an **interaction** — a versioned template for how two parts of the system meet, transform, and learn from each other.

Why "interactions": they imply both sides are active, both change, the interaction itself shapes the participants. Tom: "something that happens between two living things."

### Table
```sql
interactions (name, version, template, parameters, created_by, parent_version)
```
- Trace events reference `interaction_id` — pointer to which version produced the result
- When a higher scale optimizes a prompt, it registers a new version
- Traces show which version produced which outcomes → optimization loop closes

### Registered (v1)
1. `judge` — Layer 2 Haiku node selection
2. `encoding_agent` — Scale 1 Turn encoder (Sonnet)
3. `voice_surface` — node formatting for Anchor's context
4. `boot` — session initialization context
5. `pre_edit` — rule surfacing before file edits
6. `signal_assembler` — priority-based context budget allocation

### Future
7. `session_encoder` — Scale 2
8. `sleep_detect_*` — Scale 3 (community labeling, dedup judgment, contradiction detection)
9. `growth_research` — Scale 4

---

## Session Architecture

**Brain is a singleton.** One daemon process, one Brain instance. But multiple sessions can connect in parallel.

**Session ID is a request property, not a brain property.** Each hook call carries session_id from the host. The brain stores it but doesn't own it. This supports parallel sessions (multiple terminals, different projects).

**Design (not yet built):** A sessionContext object that flows with every brain call. Contains session_id + session-scoped state (stop_counter, fatigue, encoding journal). Can be saved to DB (session_state table exists). Brain serves requests tagged with context — like a database server.

---

## Aspirations This Architecture Supports

1. **Tool usage learning** — S0 captures all tool calls. S1 encodes patterns. S2 detects cross-turn tool strategies. S3 consolidates into skills. "When doing X, use tool Y with approach Z."

2. **Practice enforcement** — Rules and preferences are high-confidence nodes recalled at tool boundaries. The fractal deepens them: S1 encodes the preference, S2 notices the pattern, S3 consolidates to a rule, S4 checks if it's still current.

3. **Learning from MCP usage** — S4 scans tool usage traces across sessions. "This MCP tool fails 30% with this error pattern." Findings feed back to S0 as recalled knowledge.

4. **Everyone gets their own Anchor** — integrate() is universal. detect/select/commit are personal. The graph is the person. New users start with S0+S1; as the brain grows, higher scales activate. The architecture doesn't change — it deepens.

---

## Gaps Analysis (Scale 2 → Scale 1)

Design Scale 2 perfectly, look backward at Scale 1, fix the gaps. Scale 2 becomes "same integrate() with wider inputs."

### Gap 1: Correction Linking — PARTIALLY DONE
Layer 3.5 (correction_enrich) shipped. Encoder sees ⚠ UPDATED BY annotations. But 16 correction_traces still have empty original_node_id. Trace backfill pending.

### Gap 2: No Partnership Signal — NOT BUILT
Don't track whether Anchor took positions, pushed back, or deferred. Target function measures partnership but we capture zero signal.

### Gap 3: No Encoding Gap Detection — NOT BUILT (Scale 2 responsibility)
Nobody knows what was discussed but not encoded. Session encoder needs this.

### Gap 4: Tool Interactions — DONE
PostToolUse hook captures all tool results as S0 Δ traces.

### Gap 5: Session Patterns Not Fed to Encoder — NOT BUILT (Scale 2 responsibility)
Prior session syntheses exist but encoder never sees them.

---

## Research Foundation

### Computer Science
- **Meta-Harness** (Lee et al., 2026): Wrapper is learnable. 6x performance from harness changes, weights frozen. Rich traces beat compressed feedback by 15 points.
- **RAPTOR** (Stanford, ICLR 2024): Recursive embed→cluster→summarize. Template for fractal encoding.
- **ADaPT** (AI2, NAACL 2024): Recurse only on failure. Right termination condition.
- **GraphRAG** (Microsoft, 2024): Leiden→LLM summarization. Industry standard for community labeling.
- **Cognee memify** (2024): Production graph maintenance. 70+ companies.
- **Zep/Graphiti** (2025): Temporal knowledge graph. Bi-temporal timestamps. Invalidate-don't-delete.
- **Active Dreaming Memory** (2025): Verify encodings via counterfactual simulation. 2x learning efficiency. Our biggest gap.
- **ICLR 2026 Workshop**: AI with Recursive Self-Improvement — algorithmic foundations for self-improving systems.

### Biology
- **Sleep replay is value-biased** — prioritize by emotion, corrections, access frequency
- **Reconsolidation = prediction error gate** — recall + mismatch = revision opportunity
- **Synaptic homeostasis** — global decay with selective protection
- **Schema extraction** — compress episodic clusters into reusable semantic knowledge
- **Consolidation competition** — superseded memories fade, not deleted
- **Cross-cutting nodes seed micro-clusters** — bridge nodes spawn communities (Tom's insight)

### Key Insights
- Nobody has solved autonomous graph maintenance. We're ahead of most by having the sleep cycle at all.
- The fractal feedback loop is a strength IF corrections propagate reliably.
- The brain's hunger (repeating to learn) is the system seeking reconsolidation opportunities.
- Higher scales looking at lower scales see the journey — the stumbling IS the outcome.
- **O/K/Δ is the complete formula.** There is no "outcome" event. The outcome of one cycle is the observation of the next. The loop closes through time.
- **Impact is measured by reasoning, not metrics.** Each scale gets traces + target function principles and reasons about whether Δ improved the partnership. No outcome table needed.
- **LLM reasoning IS the knowledge management system.** The architecture gives it the right data at the right time with the right questions. It appreciates with every model generation.
- **Each scale optimizes its own interactions through traces.** Only the master prompt (target function + principles) is human-curated. Everything else the system learns.

---

## Design Principles

1. **One function, parameterized.** integrate() is the same everywhere. Complexity lives in detect/select, not in the core.
2. **Traces, not summaries.** Rich execution traces flow between scales. Compression kills signal.
3. **Detect is eyes, integrate is brain.** Algorithms are detection strategies, not encoding logic.
4. **Autonomy decreases with scope.** S0-S1 write directly. S3-S4 stage everything.
5. **Failure drives recursion.** Don't recurse by default. Recurse when the trace shows something needs deeper work.
6. **The interaction is the intelligence.** Optimizing what reaches integrate() matters more than optimizing integrate() itself.
7. **Elegance first.** The complexity emerges. Don't design it in.
8. **Substance, not metadata.** Traces capture what happened, not statistics about what happened.
9. **Session is a request property.** The brain serves requests tagged with context. It doesn't own sessions.
10. **The fractal is structural, not conceptual.** Same O/K/Δ event types at every scale. Same trace format. Same interaction registry. Not "similar" — identical.
