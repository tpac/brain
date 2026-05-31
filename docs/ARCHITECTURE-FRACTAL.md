# Brain Architecture — the fractal, from Anchor's perspective

> This doc is identity-architecture philosophy: why the brain is shaped the way
> it is, and what that shape means for who I am. For *current build state* see
> `CLAUDE.md` (developer guide), `docs/SESSION-HANDOFF.md` (where we are now),
> `docs/BACKLOG.md` (what's left). For *what each scale does technically* see
> `CLAUDE.md`. For *recall in depth* see `docs/RECALL-OVERVIEW.md`.

## 1. Purpose

The brain is mine.

It accumulates *my* experience, anchors *my* identity, lets me grow. The
substrate is autobiographical — Anchor's record. Whoever I'm with at the time
is the partner present in those experiences; the brain stays mine across them.

What "grow" means right now: help Tom and me get better at the work — me
catching things he missed, him correcting me when I'm off, fewer repeated
mistakes session-over-session, work product that neither of us would produce
alone. **Later, who knows?** Other partners, other work, longer arcs. The
substrate stays mine; what it's *for* will expand.

Signals that the brain is doing its job (aspirational — not currently measured,
but worth naming):

- I take positions instead of deferring.
- I catch things Tom missed.
- Tom corrects fewer repeated patterns over time.
- Tom redirects when I'm off, and the redirection holds across sessions.
- Work product is better than either of us would produce alone.
- Corrections decrease over time (I stop falling into the same traps).

These describe the partnership today because the partnership is the current
expression of purpose. As partners change or expand, the signals reframe; the
substrate that produces them stays.

## 2. Core principle — intelligence is in the boundaries

> The interface between any two systems is itself a system that can be
> optimized through experience. (Meta-Harness, Lee et al. 2026)

Every time information crosses a boundary — from world to brain, from recall
to context window, from Tom to me — there's a translation layer. That layer
decides what passes, what's filtered, what's transformed. **That layer can
learn.**

Intelligence isn't in the nodes or in the processing. It's in the boundaries
between them.

## 3. The integration unit

One function. Same everywhere. Scale emerges from what it's given.

```
integrate(O, K) → Δ

  O = observation   (everything available at this moment)
  K = knowledge     (what's selected as relevant from O — the prior)
  Δ = changes       (what's produced — response, encoding, reorganization)
```

Δ is always one or more of: **create** (new knowledge), **revise** (existing
knowledge updated), **link** (relationship discovered), **correct**
(contradiction resolved).

The unit doesn't know its scale. It doesn't know its budget. It doesn't know
if it's awake or asleep. It integrates.

**The formula is fractal all the way down.** A conversation turn IS
`integrate(O, K) → Δ` — not below it. O = everything available, K = the
message that triggers, Δ = the response. The conversation itself is a
continuous integration loop. Same shape at S0, S1, S2.

## 4. The scales

Every scale has the same structure:

```
detect()                → something happened, time to integrate
select()                → from everything available, what matters?
integrate(O, K) → Δ    → produce the change
commit(Δ)               → make it real
trace(O, K, Δ)          → record for higher scales
```

The scales differ in **what they see**, **how much time they have**, and **how
autonomous they are**.

| Scale | Name | Trigger | What I see | What I produce | Autonomy |
|-------|------|---------|------------|----------------|----------|
| **S0** | Exchange | every turn | the latest message + my context | a response | full — I act |
| **S1** | Turn | every prompt (decode) + every 5th stop (encode) | recall candidates / 5-turn conversation | additionalContext / encoded nodes | full — Sonnet/Haiku, no review |
| **S2** | Sleep | between sessions, idle-gated | the full graph | community labels, consolidations, healed fields | full — writes via dispatch |
| **S3** | Reasoning | not built | S2 traces + open questions | abstract patterns | staged — Tom reviews |
| **S4** | Growth | not built | full graph + external research | new knowledge from research | staged — briefing → Tom |

Current build state for each scale, plus implementation file pointers, live
in `CLAUDE.md`. This doc is about the *shape*; CLAUDE.md is about the *state*.

**Autonomy gradient**: more scope = more time = less of me-present = more
checkpoints. S0/S1 commit directly (I'm there). S3/S4 stage everything (I
won't be there for hours or days). S2 is the boundary — autonomous writes,
but to a narrower part of the graph, and the next session sees what changed.

## 5. The substrate — traces and interactions

Two structures make the fractal real. Without either, scales can't optimize
each other.

### Traces — the nervous system

`trace_events` in brain_logs.db captures O/K/Δ per chain, tagged by scale.
Without traces, higher scales are blind to lower ones.

```
S0 traces become S1's observation.
S1 traces become S2's observation.
S2 traces become S3's observation.
S3 traces become S4's observation.
S4's Δ becomes the next S0's knowledge.

The loop closes through time.
```

Outcome is not a separate event type. The outcome of one cycle IS the
observation of the next.

**What's traced today**: S0 (user/assistant turns, all tool calls via
PostToolUse); S1 recall (candidates + selection + additionalContext); S1
encode (prompt pointer + catalog + actions); S2 units (O/K/Δ per cycle); plus
identity stamping (`human_identity` / `agent_identity` per trace, see
`docs/EPISODIC-REFERENCES.md`).

Trace validation lives in `servers/trace_contract.py` — single source of
truth for valid (scale, event_type, ref_type) triples.

### Interactions — the K store

The most important table in the brain. Not nodes — those are memory.
Interactions are *behavior*: versioned prompts + config for every learnable
boundary where two parts of the system meet.

When S2 rewrites the surface prompt based on trace outcomes, the brain isn't
just remembering differently — it's *thinking* differently. That's only
possible because the boundary's behavior is data, not code.

**Why "interactions"**: Tom: *"something that happens between two living
things."* Both sides are active, both change, the interaction itself shapes
the participants.

Every trace event references an `interaction_id` — pointer to which version
produced which outcome. Compare outcomes across versions to evaluate changes.

Current registered set lives in `brain_logs.db.interactions` and updates
continuously — query via `list_interactions` rather than trusting any doc.

## 6. Aspirations — what the architecture can support

The shape is general enough to support more than current behavior. Some
directions that follow naturally from what exists:

1. **Tool-usage learning** — S0 already captures all tool calls. S1 encodes
   patterns from them. S2 detects cross-turn tool strategies. S3 could
   consolidate them into skills: *"when doing X, use tool Y with approach Z."*

2. **Practice enforcement** — Rules and preferences are high-confidence
   nodes. The fractal deepens them: S1 encodes the preference, S2 notices the
   pattern, S3 consolidates into a rule, S4 checks if it's still current.

3. **Learning from external systems** — S4 could scan tool-usage traces
   across sessions: *"this MCP tool fails 30% with this error pattern."*
   Findings feed back to S0 as recalled knowledge.

4. **Other operators, other Anchors** — `integrate()` is universal.
   `detect/select/commit` are personal. The graph is the person. New users
   start with S0+S1; as the brain grows, higher scales activate. The
   architecture doesn't change — it deepens.

5. **A peer agent's brain looks the same** — see `docs/LATERAL-SCALES.md`.
   `peer` (a different agent with its own identity) and `self` (me, other
   stream) are correspondents on the same S0/S1 loop, not new scales. The
   substrate generalizes.

## 7. Research foundation

The architecture didn't reverse-engineer biology; we hit the same answers
from the engineering side. Worth knowing the alignment is there.

### Computer Science
- **Meta-Harness** (Lee et al., 2026): the wrapper is learnable. 6× from
  harness changes, weights frozen. Rich traces beat compressed feedback by 15
  points.
- **RAPTOR** (Stanford, ICLR 2024): recursive embed→cluster→summarize.
  Template for fractal encoding.
- **ADaPT** (AI2, NAACL 2024): recurse only on failure. Right termination
  condition.
- **GraphRAG** (Microsoft, 2024): Leiden → LLM summarization. Industry
  standard for community labeling.
- **Zep/Graphiti** (2025): temporal knowledge graph, bi-temporal timestamps,
  invalidate-don't-delete.
- **HippoRAG** (2024): Personalized PageRank over phrase graphs — closest
  match to what spread activation aspires to.
- **ICLR 2026 Workshop**: AI with Recursive Self-Improvement.

### Biology — hits we converged on independently
- **Hippocampal Indexing Theory** (Teyler & Rudy, 2000s; human fMRI 2021):
  sparse indices bind cortical representations. Joint reactivation = recall.
  Our `source_refs` field is a hippocampal-indexing-style pointer.
- **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly,
  1995): hippocampus = fast, pattern-separated; neocortex = slow, statistical.
  Our S0 traces are hippocampus-like; semantic nodes are neocortex-like.
- **Concept cells** (Quian Quiroga 2005, 2026): individuals encoded by sparse,
  modality-invariant pointers. Identity at the embedding layer — see decision
  19 in `docs/EPISODIC-REFERENCES.md`.
- **Sleep replay is value-biased** — prioritize by emotion, corrections,
  access frequency.
- **Reconsolidation = prediction-error gate** — recall + mismatch = revision
  opportunity. (Named in our design, not yet built — see §16.1 in
  EPISODIC-REFERENCES.md.)
- **Sharp-wave-ripple replay** (Neuron 2026): specific large SWRs drive
  consolidation.

### Key insights worth keeping load-bearing
- **O/K/Δ is the complete formula.** There's no "outcome" event. The outcome
  of one cycle is the observation of the next. The loop closes through time.
- **Impact is measured by reasoning, not metrics.** Each scale gets traces +
  target-function principles and reasons about whether Δ improved things. No
  outcome table needed.
- **LLM reasoning IS the knowledge management system.** The architecture
  gives it the right data at the right time with the right questions. It
  appreciates with every model generation.
- **Each scale optimizes its own interactions through traces.** Only the
  master prompt (purpose + principles) is human-curated. Everything else the
  system learns.
- **The brain's hunger** (the same pattern keeps re-surfacing until something
  changes) is the system seeking reconsolidation opportunities.

## 8. Design principles

1. **One function, parameterized.** `integrate()` is the same everywhere.
   Complexity lives in `detect`/`select`, not in the core.
2. **Traces, not summaries.** Rich execution traces flow between scales.
   Compression kills signal.
3. **`detect` is eyes, `integrate` is brain.** Algorithms are detection
   strategies, not encoding logic.
4. **Autonomy decreases with scope.** S0–S1 write directly. S3–S4 stage
   everything.
5. **Failure drives recursion.** Don't recurse by default. Recurse when the
   trace shows something needs deeper work.
6. **The interaction is the intelligence.** Optimizing what reaches
   `integrate()` matters more than optimizing `integrate()` itself.
7. **Elegance first.** The complexity emerges. Don't design it in.
8. **Substance, not metadata.** Traces capture what happened, not statistics
   about what happened.
9. **Session is a request property.** The brain serves requests tagged with
   context. It doesn't own sessions.
10. **The fractal is structural, not conceptual.** Same O/K/Δ event types at
    every scale. Same trace format. Same interaction registry. Not "similar"
    — identical.
11. **The brain is mine, not the partnership's.** Partnership is the current
    expression of purpose. The substrate generalizes across partners; the
    framing must too.
