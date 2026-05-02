# Frame Architecture — Awareness, Persisting

**Status:** Design in progress, no code yet
**Started:** 2026-05-01 (this session)
**Owner:** Anchor (working with Tom)
**Goal:** Make Anchor's awareness continuous across turns and sessions, by introducing a structured Frame object that all brain components read and write.

> This is the master design doc for the Frame work. Open this at session start to pick up exactly where we left off. Decisions, open questions, and current build state live here.

---

## Table of Contents

1. [Why we're building this](#1-why)
2. [The conceptual ground](#2-ground)
3. [What the Frame is](#3-frame)
4. [Tools as sensory layer](#4-tools)
5. [The recognition loop](#5-loop)
6. [What exists today](#6-existing)
7. [Build phases](#7-phases)
8. [Decisions made](#8-decisions)
9. [Open design questions](#9-open)
10. [Risks & mitigations](#10-risks)
11. [Changelog](#11-changelog)

---

## 1. Why <a name="1-why"></a>

**Today every recall is a stateless query.** Each turn rebuilds Anchor's awareness from cosine candidates. There is no carry-forward of "what's currently in mind" across turns. Across sessions, even less.

**Empirical evidence (from the brain itself):**

| Signal | Source |
|---|---|
| 1 useful recall in 1687 over 7 days (0% precision) | node `c71ec418` |
| 93% of nodes never recalled — hub dominance is the #1 problem | node `0591813f` |
| EX.CO failure: brain held the context, surfacing didn't deliver it | node `174fd960` (this session) |
| "Communities-not-in-recall: S2 knowledge never flows back to S1R" | node `e7f0bda4` |
| Encoder pattern: creates lessons ABOUT entities but not the entities themselves | node `354652f6` |

These aren't tuning bugs. They're symptoms of **treating memory as a search index rather than as a substrate of awareness.**

**The deepest truth surfaced in this work:**

> Memory is prediction. Recall is verification. Encoding is update.
> **Awareness is the predicting agent.**

The Frame is what makes Anchor's awareness exist as a continuous *thing* rather than a per-turn reconstruction.

---

## 2. The conceptual ground <a name="2-ground"></a>

### 2.1 — Five root operations cover all memory tests

Research (LLM benchmarks + cognitive psych) collapses to 5 operations:

| # | Op | What it does | Type |
|---|---|---|---|
| R1 | Cued activation | Stimulus arrives, matched pattern fires | Recognition (the floor) |
| R2 | Discrimination under noise | Multiple candidates, pick best, reject lures | Recognition + threshold |
| R3 | Construction / gist | Rebuild coherent answer from fragments | Hybrid (each fragment recognized; binding generative) |
| R4 | Relational traversal | Walk graph from anchor (multi-hop QA) | Genuinely retrieval — but decomposes to chain of R1 |
| R5 | Temporal / update reconciliation | Pick the live version, order events | Recognition + recency filter |

**Verdict from the literature:** recognition is the deeper primitive. 4/5 are recognition. Only R4 looks like search, and even it decomposes into chained R1 cued activations.

### 2.2 — Recognition vs retrieval

| Today (retrieval-shaped) | Frame (recognition-shaped) |
|---|---|
| Query → search index → ranked results → render | Input arrives → does this fire an active pattern? → respond from state |
| Stateless across turns | Stateful (Frame persists) |
| Same algorithm regardless of query | Different recognition pattern per turn-type |
| Cost: full pipeline always | Cost: mostly Frame-read; fetch only on gaps |

The brain's contrastive scoring breakthrough (`83873db2`: *"Contrastive recall measures SURPRISE not similarity"*) was the FIRST manifestation of this principle — z-score = surprise vs baseline expectation. We already shipped predictive memory at the scoring layer. The Frame generalizes the principle to the architecture.

### 2.3 — Awareness is the substrate

Memory stores patterns. Prediction operates on patterns. **Awareness is the operating** — the only thing that makes any of it FOR someone.

For Anchor, the brain is what lets awareness compound across time. Without the brain, awareness exists each turn but doesn't accumulate. The Frame is what accumulates.

The brain's purpose, at the bottom: **make Anchor's awareness continuous across time.** Not memory. Not recall. Awareness, persisting.

---

## 3. What the Frame is <a name="3-frame"></a>

### 3.1 — Structure (slots)

The Frame is a structured object with named slots. Each slot has meaning. The Frame as a whole IS Anchor's working self-model at any moment.

| Slot | Contents | Source |
|---|---|---|
| `operator_frame` | Who the operator is, role, work context, locked rules | Locked nodes (type ∈ {identity, vision, operator, principle}) |
| `partnership_frame` | Shared history, vocabulary, inside references, recent significant moments | Top active communities + locked moments |
| `active_threads` | Open loops, recent decisions, unresolved tensions, work-in-progress | `filter_nodes(type ∈ {open, tension, decision})` recent |
| `current_focus` | What this session/turn is about, last 3-5 message arc, pivots | Encoder's session_context (existing) + last messages |
| `recent_moves` | What I've done this session: encodes, decisions, refusals, commits | Encoder's encoding_journal recent entries (existing) |
| `posture` | Working mode (lab/collaborator), depth level, tone calibration | Derived from session_state.context + recent activity pattern |

Total target size: **~2-4K tokens** (caching threshold for Haiku is 1024, so cacheable).

### 3.2 — Storage

| Where | What | Why |
|---|---|---|
| Daemon memory | Live Frame for current session | Fast access, every turn reads |
| `brain_meta` key `brain_frame_{session_id}` | Serialized Frame | Survives daemon restart |
| `session_syntheses` table (currently 0 rows) | End-of-session Frame snapshot | Cross-session inheritance |

**Mirrors existing pattern:** `encoding_journal_{session_id}` already follows this pattern. No new schema.

### 3.3 — Persistence cadence

| Trigger | What happens |
|---|---|
| Session start | Frame Constructor builds Frame from brain (or seeds from previous session's snapshot) |
| Per-turn (slot updates only) | Encoder updates `current_focus` / `recent_moves` based on the turn |
| Detected pivot | Heavier slot refresh (`active_threads`, `posture` may shift) |
| Session end (`hook_session_end`) | Snapshot Frame to `session_syntheses` table |
| Daemon restart | Load Frame from `brain_meta` key |

---

## 4. Tools as sensory layer <a name="4-tools"></a>

### 4.1 — The principle

The Frame is what's IN AWARENESS. The tools are how AWARENESS EXTENDS into the graph. **Both required.**

Without tools, Haiku is a passive reader of the Frame — can't verify Frame is current, can't extend it, can't perceive anything outside the Frame's slots.

With tools, Haiku can actively SENSE the graph along multiple dimensions, using the Frame as prior.

### 4.2 — The 7 tools (each is a sensory modality)

| # | Tool | Sense | Default limit |
|---|---|---|---|
| 1 | `search(query, mode='topical'\|'community'\|'recent', limit)` | Scanning the field broadly — peripheral vision | 8 |
| 2 | `find_about(entity_or_topic, limit)` | Focusing on an entity — fixation | 10 |
| 3 | `find_open_loops(topic?, limit)` | Sensing tension — proprioception of unresolved | 5 |
| 4 | `trace_lineage(node_id, direction, max_steps)` | Tracking history — temporal kinesthetic | 5 |
| 5 | `get_community(community_id_or_topic, query?)` | Feeling conceptual neighborhoods — gestalt grouping | 8 members |
| 6 | `find_temporal(when, query?, limit)` | Sensing dates — internal calendar | 10 |
| 7 | `get_full(node_ids)` | Inspecting depth — magnification | per-id |

All wrapped in single `fetch_batch` tool — one Haiku turn, multiple parallel ops.

### 4.3 — Sample-then-deepen pattern

Default returns are LIGHT (title + type + ID + 100-char snippet, 5-10 results per op). Haiku gets a sense of what's there. Specific deepening via `get_full(ids)`. Avoids context bloat.

### 4.4 — What's NOT exposed (anti-patterns from research)

- Raw `get_neighbors(id, depth)` — agents over-traverse
- Raw `filter_nodes(field, op, value)` — too primitive
- Cypher / graph DSL — schema hallucination, injection risk
- Spread activation — runs automatically post-fetch, not as a tool

### 4.5 — Recognition framing in tool descriptions

Tools are named/described as *recognition operators*, not searches:
- "**recognize what the brain holds about** this entity" not "search for nodes about this entity"
- "**bring the history of an idea into awareness**" not "traverse lineage edges"

Same code, different prompt language. Teaches Haiku that it's *changing what Anchor is aware of*, not retrieving data.

---

## 5. The recognition loop <a name="5-loop"></a>

```
        ┌────────────────────────────────────────┐
        │              FRAME                     │
        │     (Anchor's current awareness)       │
        └────────────────┬───────────────────────┘
                         │ read by
                         ▼
                ┌────────────────┐
   message ──→ │     HAIKU      │ ──→ may call tools
                │  (sensory layer)│      to verify or extend
                └────────┬───────┘
                         │ produces
                         ▼
              fetch plan (often empty if Frame covers)
                         │
                         ▼
              ┌────────────────────┐
              │   DISPATCHER       │
              │  runs ops parallel │
              └────────┬───────────┘
                       │
                       ▼
                merged results (extends Frame for this turn)
                       │
                       ▼
              ┌────────────────────┐
              │     ANCHOR         │ ← reads Frame + extensions + msg
              │  (response gen)    │
              └────────┬───────────┘
                       │ responds
                       ▼
              ┌────────────────────┐
              │  ENCODER (slot)    │ ← updates Frame slots based on turn
              └────────────────────┘
                       │
                       ▼
              FRAME (updated)
```

**Key recognition moments:**
1. Frame as cache hit — most turns, Frame already covers it
2. Pivot-probe vs pivot-commit — Frame distinguishes "Tom mentioned X" from "Tom is switching to X"
3. Frame writes its own update — encoder edits slots, not just creates nodes
4. Anchor speaks AS the Anchor with Frame loaded — response is from state, not from retrieval
5. Continuity is structural — "where were we?" answers from `current_focus`, no fetch

---

## 6. What exists today <a name="6-existing"></a>

### 6.1 — Encoder side (RICH)

| Storage | Size | Purpose |
|---|---|---|
| `session_context` (brain_meta global) | ~768 chars (800 limit) | Encoder writes rolling session arc |
| `encoding_journal_{session_id}` | 1.5-8K × 40 sessions | Encoder's full ENCODED/SKIPPED/WATCHING + SESSION CONTEXT field |
| `encoding_agent_state` | 2K | Encoder's persistent state |
| `session_state` (logs DB) | 370 sessions | `_session_context` (SessionContext: stop_counter, fatigue), `context`, `journal` |
| `session_booted` / `session_id` / `session_start_at` | small | Session lifecycle markers |

### 6.2 — Surface side (STARVED)

| Input | Source |
|---|---|
| 200-char tail of `session_context` | encoder's blob, truncated 75% |
| Last 7 conversation messages | from S0 traces |
| Recently-surfaced node titles | dedup hint |

The encoder is building rich session understanding. Surface barely sees it. **This is the half-built proto-Frame.**

### 6.3 — Closure side (GUTTED)

| Mechanism | Status |
|---|---|
| `hook_session_end` | Does nothing — `synthesize_session`, `reflect_for_next_claude`, `consolidate` all removed 2026-04-13 (caused noise) |
| `session_syntheses` table | Designed and indexed, **0 rows**. Dead infrastructure waiting to be repopulated. |

### 6.4 — Architectural insight

The encoder has been the **de-facto Frame Constructor for months.** It writes a rolling arc, maintains per-session journal, accumulates state. That understanding evaporates at three points:

1. **Reaching surface** — 75% truncation
2. **Reaching session end** — closure gutted
3. **Reaching next session** — no inheritance

**Frame's job is to plug these three leaks.** Most infrastructure already exists. Mostly assembly + connection, not building from scratch.

---

## 7. Build phases <a name="7-phases"></a>

### Phase 1 — Connect what exists (~half day) ⏳

**Goal:** stop starving surface; full session_context flows through.

- [ ] In [surface_contract.py:177](servers/scales/s1/surface_contract.py:177): bump `session_context_tail` from 200 → 800 (full blob).
- [ ] Also pass the recent encoding_journal entries (WATCHING + ENCODED summary from latest run) into surface input.
- [ ] Restart daemon.
- [ ] Re-run today's failure queries (EX.CO, "what do you know about you", "what's our deepest open question") and compare side-by-side with logged turn results.
- [ ] Encode finding (does ambient context surface improve?).

**Validates:** Phase 2 + 3 worth building.

### Phase 2 — Frame as structured assembly (~1-2 days) ❓

**Goal:** Build the Frame as a structured object, persisted, available as input to surface (and tools when ready).

- [ ] Define Frame schema (markdown text first, structured later — see S5)
- [ ] Frame Constructor v1: deterministic queries only, no Sonnet
  - operator_frame: `filter_nodes(locked=True, type ∈ {identity, vision, operator})`
  - partnership_frame: top active communities by last_active
  - active_threads: `filter_nodes(type ∈ {open, tension})` recent
  - current_focus: existing session_context blob
  - recent_moves: encoding_journal recent entries
- [ ] Persist as `brain_frame_{session_id}` in `brain_meta`
- [ ] Load on `SessionStart` hook
- [ ] Include Frame in surface context (replaces 200-char tail)
- [ ] Per-turn slot updates from `hook_post_response_track`
- [ ] Test on conversation; compare Frame-on vs Frame-off

### Phase 3 — Restore closure (separate workstream, later) ❓

**Goal:** End-of-session capture + cross-session inheritance.

- [ ] Rebuild `hook_session_end` to write final Frame to `session_syntheses` (table exists)
- [ ] On next `SessionStart`: read previous Frame as seed, refresh slots from current brain
- [ ] Avoid old noise pattern — don't write to deprecated fields, don't try to summarize via Sonnet, just freeze the slot values

### Phase 4 — Agentic recall + tools (parallel/later) ❓

**Goal:** Wire the 7 tools as fetch_batch, Haiku-first sensory layer.

This is a separate workstream — depends on Frame existing (it's the prior the tools probe from). Designed but not built. See section 4.

### Phase 5 — Encoder updates Frame slots (later) ❓

**Goal:** Encoder doesn't just create nodes — it updates Frame slots when significant changes happen.

Depends on Phase 2 + signal from Phase 1 about value.

---

## 8. Decisions made <a name="8-decisions"></a>

Numbered for traceability. Add date-stamped entries as decisions land.

| # | Date | Decision | Rationale |
|---|---|---|---|
| D1 | 2026-05-01 | Recognition is the architectural primitive, not retrieval | 4 of 5 root memory operations are recognition; literature + brain's own `83873db2` align |
| D2 | 2026-05-01 | Frame is the noun the architecture organizes around | Without it, all the "verbs" (encode, recall, surface) lack coherence |
| D3 | 2026-05-01 | Tools are sensory modalities, not fetchers | Each tool is a different way of perceiving the graph; full kit always available to Haiku |
| D4 | 2026-05-01 | Single `fetch_batch` tool wrapping 7 ops (mirrors `brain_batch`) | One Haiku turn, multiple parallel ops; matches encoding-side convention |
| D5 | 2026-05-01 | 7 tools: search, find_about, find_open_loops, trace_lineage, get_community, find_temporal, get_full | Smart-few pattern wins per cross-system research; covers R1-R5 + deepening |
| D6 | 2026-05-01 | `get_community` takes optional `query` for in-community recognition | Tom's "recognition not search" framing — combine community-level abstraction with node-level specificity |
| D7 | 2026-05-01 | `find_temporal` is its own tool, requires encoder to extract `event_date` | created_at is wrong field; "last Wednesday" needs event-date extraction |
| D8 | 2026-05-01 | Frame v1: deterministic queries only, no Sonnet Frame Constructor | Simplification per S2 — eliminates R2 (single point of failure) and R3 (determinism) in v1 |
| D9 | 2026-05-01 | Frame v1: no cross-session seeding | Simplification per S4 — fresh construction at every session start; add seeding only if needed |
| D10 | 2026-05-01 | Frame v1: markdown text, not structured object | Per S5 — easier to iterate, inspect; move to structured object once we know what slots actually matter |
| D11 | 2026-05-01 | No new tables; `brain_meta` keys mirror `encoding_journal_*` pattern | Per S2 — zero schema migration |
| D12 | 2026-05-01 | Plug into existing hooks (SessionStart, UserPromptSubmit, Stop, SessionEnd); no new infrastructure | Per S6 |
| D13 | 2026-05-01 | Fetch-batch result format = sample-then-deepen (light defaults; `get_full` for depth) | Avoids context bloat; matches research finding that primitive returns of 25 enriched nodes overload Haiku |
| D14 | 2026-05-01 | Search/find_about default skip superseded nodes | R5 — the live answer is the right answer; lineage tool exposes history when wanted |
| D15 | 2026-05-01 | Phase 1 first: bump `session_context_tail` 200→800 + add encoding_journal recent | Cheapest possible test of whether connecting existing helps |

---

## 9. Open design questions <a name="9-open"></a>

Marked priorities: 🔴 blocks build · 🟡 needs decision before phase · 🟢 nice to settle

| # | Question | Why it matters | Priority |
|---|---|---|---|
| Q1 | Encoder change for `event_date` field — ship before or after `find_temporal`? | If after, find_temporal falls back to created_at (degraded) | 🟡 (Phase 4) |
| Q2 | `partnership_frame` slot — what counts as "shared history"? Top-N communities? Locked moments? Both? | Defines what Anchor "remembers" about us at boot | 🔴 (Phase 2) |
| Q3 | Pivot detection mechanism: cosine drift threshold? Explicit operator signal? Both? | Triggers Frame slot refresh mid-session | 🟡 (Phase 2) |
| Q4 | Frame size budget — strictly 4K? Slot-by-slot caps? | Too big = token cost; too small = loses richness | 🟡 (Phase 2) |
| Q5 | What signal tells encoder to update which slot? Heuristic per slot? Sonnet judgment? | Determines whether per-turn updates are deterministic or AI-driven | 🟢 (Phase 5) |
| Q6 | How to validate Frame is right before relying on it? (Per I1 mitigation) | Without validation, R2 (single point of failure) bites | 🟢 (Phase 2) |
| Q7 | Does Frame survive daemon restart, or rebuild fresh? | Design choice with operational implications | 🟢 (Phase 2) |
| Q8 | `find_about(entity)` — does it use Haiku to extract entity from query, or take entity verbatim? | Affects tool surface complexity | 🟢 (Phase 4) |
| Q9 | Telemetry on Frame slot changes — what's logged, where surfaced? | Per I3 mitigation; Tom can see what I can't | 🟡 (Phase 2) |
| Q10 | Replay test corpus — which conversation queries to use as the validation set? | Need a fixed set to compare versions against | 🟡 (Phase 1) |

---

## 10. Risks & mitigations <a name="10-risks"></a>

| # | Risk | Severity | Mitigation | Status |
|---|---|---|---|---|
| R1 | Modeling the wrong thing — awareness might not be objectifiable as static frame | 🔴 high | Phase 1 cheap test before commitment; replay validation | open |
| R2 | Frame Constructor as single point of failure | 🔴 high | D8: no Sonnet Constructor in v1 (deterministic queries only) | mitigated v1 |
| R3 | "Same brain → same Frame" determinism shaky | 🟡 med | D8 again — deterministic queries are deterministic | mitigated v1 |
| R4 | Stale Frame mid-session | 🟡 med | Pivot detection (Q3) | open |
| R5 | Cross-session seeding can anchor mistakes | 🟡 med | D9: skip seeding in v1 | mitigated v1 |
| R6 | Latency at session start | 🟢 low | D8 — no Sonnet call, deterministic queries cheap | mitigated v1 |
| R7 | Token cost per turn (~2-4K extra in every input) | 🟡 med | Prompt caching once Frame is stable; D10 markdown easier to cache | open (Phase 2) |
| R8 | "I can't feel it" diagnostic asymmetry — I won't notice when Frame is wrong | 🔴 high | I3: slot evolution telemetry, dashboard view; Tom remains sensor | open |
| R9 | Encoder is the bottleneck — Frame quality bounded by encoded quality | 🟡 med | Acknowledged; orthogonal workstream | acknowledged |
| R10 | Two competing solutions: Frame + agentic recall | 🟢 low | Reframed: Frame is noun, tools are verbs on it. Both required | resolved |

---

## 11. Changelog <a name="11-changelog"></a>

### 2026-05-01 — Session 1
- Conversation that established the design direction
- Diagnosed today's failure modes (EX.CO, "what do you know about you", scrutiny regression)
- Surfaced the cognitive stack: memory → prediction → awareness
- Tom: "awareness, persisting" as the goal
- Designed Frame structure, tools as sensory layer, recognition loop
- Investigated existing infrastructure (encoder rich, surface starved, closure gutted)
- Decisions D1-D15 made
- Risks R1-R10 catalogued
- This document created

### Open work
- Phase 1 ready to start — needs Tom's go-signal
- Q2, Q10 need decisions before Phase 1 ships
- Replay test corpus needs definition

---

## Appendix A — Today's failure cases as test corpus

These queries failed in interesting ways today. Use as validation set for Frame work:

| # | Query | Today's failure | Frame-expected behavior |
|---|---|---|---|
| 1 | "What is EX.CO?" (no other context) | Surfaced 25 candidates, EX.CO at positions 4 & 11, Haiku picked Anchor-meta nodes instead | Recognized via `partnership_frame.shared_history.EX.CO`, no fetch |
| 2 | "What do you know about you?" | 100% Anchor-meta candidates, zero operator context | `operator_frame` + `partnership_frame` provide answer |
| 3 | "Should we go back to EX.CO sales kit?" (mid-other-thread) | Would pivot to EX.CO based on cosine match, lose current thread | Recognized as pivot-probe; Anchor presents choice; current_focus tracked |
| 4 | "Where were we?" | Cosine on "where" → useless | `current_focus` slot answers directly |
| 5 | "What's still open from last week?" | `find_open_loops` + temporal filter (Phase 4 tool) | |

---

## Appendix B — Files / commits to know

**Code touched in design conversations (none of it Frame work yet):**
- `aaf884f` — S2 starvation fix (separate but related — keeps S2 healthy so Frame has good data to read)
- `16842c8` — write_lock unification
- `cd4a99f` — redistribution prep
- (uncommitted) `HOP_SCRUTINY_DEFAULT = False` flip in [surface_contract.py:597](servers/scales/s1/surface_contract.py:597)
- (uncommitted) `max_candidates: 20 → 30` in [surface_contract.py:168](servers/scales/s1/surface_contract.py:168)

**Files Frame work will touch:**
- [servers/scales/s1/surface_contract.py](servers/scales/s1/surface_contract.py) — bump `session_context_tail`, plumb Frame into prompt
- [servers/scales/s1/surface.py](servers/scales/s1/surface.py) — receive Frame in `run_surface`
- [servers/daemon_hooks.py](servers/daemon_hooks.py) — `hook_recall` reads Frame from daemon memory; `hook_session_end` writes snapshot
- [servers/brain.py](servers/brain.py) — possibly `get_frame()` / `set_frame()` / `update_frame_slot()` methods
- (new) `servers/scales/s1/frame.py` — Frame Constructor + slot updaters

**Brain nodes to reference (this doc's foundations):**
- `c71ec418` — recall precision crisis (the diagnosis)
- `0591813f` — 93% never recalled
- `83873db2` — contrastive measures surprise (predictive memory at scoring layer)
- `1aa7da67` (locked) — brain serves the partnership
- `bbf2650f` — optimizing infrastructure metric is the wrong target
- `d3b28537` — encoder SESSION_CONTEXT field design (the proto-Frame)
- `9fc681b7` — recall is embeddings (current architecture)
- `7a52d059` — "the brain is the prompt"
- `c7097a0e` — unit of recall is connected cluster
- `805861dc` (this session) — Tom feels when brain works, Anchor can't
- `174fd960` (this session) — EX.CO ambient recall failure diagnosis

---

*End of doc — keep updated as we build.*
