# Frame Architecture — Awareness, Persisting

**Status:** Phase 1 shipped 2026-05-02 · Phase 2 designed, not started
**Started:** 2026-05-01
**Owner:** Anchor (working with Tom)
**Goal:** Make Anchor's awareness continuous across turns and sessions, by introducing a structured Frame object that all brain components read and write.

> This is the master design doc for the Frame work. Open this at session start to pick up exactly where we left off. Decisions, open questions, current build state, cleanup workstream, tests, and onboarding for stateless Anchor live here.

---

## 0. Pick up here (current state) <a name="0-pickup"></a>

**Most recent work:** Frame Phase 1 connecting code shipped (commits `241ab37` for Phase 1 itself, then commit-pending for timeout bumps + Pattern A fix + this doc enrichment). Surface prompt v1 registered (3047 chars) with recognition-over-search framing. Hook timeout bumped 14s→20s to cover Haiku tail latency. **2026-05-02 (later):** validation harness shipped — `eval/frame_replay.py` runs the Appendix A corpus against an isolated brain copy, captures labeled snapshots, diffs two snapshots side-by-side. Phase 1 baseline captured as `phase1_baseline_2026-05-02`. Q2 proposal drafted in Section 9.1 — three-layer partnership_frame (integrated + permanent + warm), awaiting Tom's react.

**2026-05-02 (Phase 2 in-progress, NOT shipped):**
- Built `servers/scales/s1/frame.py` — Frame Constructor with five sections (Operator / Partnership-3-layer / Active threads / Current focus / Recent moves). Uses only `brain.filter_nodes`, `brain.session_context`, `brain.get_recent_encoding_journal` — no new SQL, no LLM call.
- Smoke-tested: ~7700 chars (~1900 tokens), all sections render against live brain copy.
- Discovered: `access_count` sort is a recipe for stagnation (rewards historic obsessions like 10-session hook deep-dives forever). Switched all sorts to `last_accessed` for fluid, recency-driven Frame content. Empirically dramatic: Operator went 2→9 entries; Communities went infrastructure-heavy ("Mixin Decomposition") → topic-current ("Frame Design and Build" #1, "Validation Before Iteration"). Frame now reflects WHERE WE ARE, not WHERE WE'VE BEEN MOST.
- Added `last_accessed` and `revised_at` to DAL `filter_nodes` allowed_sort whitelist (one-line completion of an existing function — Tom: "DAL should be convenient for whatever purpose").
- Discovered: **node types are open text, not closed enum** (CHECK constraint removed in v8.3, NODE_TYPES list in schema.py is vestigial documentation). Most types in active use (open, milestone, architecture, principle, fact, moment, quote, community, event, etc.) aren't in the documented list. This means node-type families are the SAME problem shape as edge relation families — both classify open text.

**Phase 2 progress — what's shipped this session:**
1. ✅ **`s2_node_families` interaction seeded** mirroring s2_edge_families pattern:
   - `servers/scales/s2/node_families_v1.json` — 14 families covering identity_bearing, episodic_anchor, active_thread, decided_committed, integrated_knowledge, architecture_design, conceptual_knowledge, factual_knowledge, lesson_insight, diagnostic, correction_supersession, task_artifact, boot_meta, noise
   - `interaction_seed.py` registers `s2_node_families` with template (S2_NODE_FAMILIES_PROMPT) + seed JSON on fresh brains
   - Verified in IsolatedBrain — fresh brain copy seeds the interaction correctly on first instantiation
2. ✅ **`frame.py` refactored** to read families via `brain.get_interaction_config('s2_node_families')` (with fallback to hardcoded defaults for unseeded brains)
   - OPERATOR_TYPES → reads `identity_bearing` family
   - PERMANENT_MOMENT_TYPES → reads `episodic_anchor` family
   - WARM_MOMENT_TYPES → reads `episodic_anchor` + `lesson_insight` families
   - ACTIVE_THREAD_TYPES → reads `active_thread` family
   - Smoke test: Operator section grew from 9 → 10 entries (now includes `rule` and `capability` types via family expansion)

**Phase 2 wire-up shipped:**
- ✅ `SessionContext.get_frame(brain)` — session-scoped entry point (Tom's call: Frame is per-session, brain is the singleton dependency)
- ✅ `daemon_hooks.hook_recall` calls `ctx.get_frame(brain)`, passes through `_run_surface(frame=...)`
- ✅ `run_surface` and `_call_surface` thread `frame` through to `build_surface_prompt`
- ✅ `build_surface_prompt(frame=...)` renders Frame as "Partnership context (your prior):" — replaces session_context + encoding_journal blocks when Frame is non-empty (defensive fallback to Phase 1 layout otherwise)
- ✅ Tests: `tests/test_frame.py` (13 tests, all pass) — covers build_frame, family resolution + fallback, ctx.get_frame, seed presence, surface_prompt-accepts-frame
- ✅ Regression check: 98/98 passed on contract_sync + brain_voice + dispatch_contract_sync + daemon tests

**Phase 2 validation:**
- ✅ Captured `phase2_v1_unified` via harness, compared to `phase1_baseline_2026-05-02`
- Real Frame-driven changes observed: 3 of 5 queries showed selection shifts, latency mostly stable, context size 4-7% smaller on stable selections (Frame replaces session_context block more efficiently)
- Quality of selection changes is qualitative — needs Tom's eye on actual selections to declare improvement vs regression

**Phase 2 still NOT in production:**
- Daemon hasn't been restarted — wire-up exists in source but live daemon runs pre-Phase-2 code. Restart via `mcp__plugin_brain_brain__restart` to make Frame live.
- All commits pending — should batch the Frame work into one or two commits before restart.

**Known cold-start gap (accepted for Phase 2 — option 1 of 3):**
- v1 `s2_node_families` seed covers ~50 observed node types across 14 families. Encoder may generate new types not in the seed; those nodes become invisible to Frame's family-based queries until the seed JSON is manually updated OR a S2 maintenance unit is built (deferred — see below).
- Defensive fallback in `frame.py` (`_FALLBACK_FAMILIES`) handles "interaction missing" but not "type unclassified."
- Bounded impact — most types covered by v1 seed; load-bearing cases (principle, rule, moment, community, open, insight) all in seed.

**Phase 2.5 — open follow-ups (next session candidates):**
1. **Build the s2_node_families maintenance unit (~2hr):**
   - Mirror `EdgeFamilyIntegration`: scan distinct node types, find unclassified, prompt LLM with seed + samples, merge into config, write back
   - Need a `NodeDAL.count_by_type()` method (parallel to `GraphDAL.count_by_relation()`)
   - Once shipped, families auto-update from observed data instead of relying on manual seed
2. **Generalize s2_type_families** — once both EdgeFamilyIntegration and NodeFamilyIntegration exist, refactor shared structure into a base/helper. Rule of three principle. Don't generalize from one example.
3. **Brain-level vs session-level Frame caching split** — Tom's note: operator/partnership/active-threads come from brain state and don't change per-session; current_focus/recent_moves are genuinely per-session. v1 rebuilds the whole Frame each call; future split could cache the brain-level slow-changing parts on Brain (refreshed on S2 cycles or encoder writes) and keep the fast-changing slots on SessionContext. Data sources are already separated in `build_frame`, so the caching split slots in cleanly.
4. **Encoder's `session_context` blob format** — currently dense pipe-separated, hard for Anchor at boot to parse. Encoder-output-format issue, not Frame issue, but visible through Frame's "Current focus" section.

**Deferred (Phase 2 scope-creep risk, push to Phase 2.5 or later):**
- Build the actual `s2_node_families` S2 maintenance unit (Tom only asked for the storage pattern in Phase 2; the maintenance loop is its own work, ~2hr, mirrors `EdgeFamilyIntegration`)
- Per-turn slot updates — current plan is full Frame rebuild on every `hook_recall`. If profiling shows it's slow (>200ms), add slot-update infrastructure
- Permanent-moments layer in Frame is genuinely thin (1 entry on average — moments are rarely locked). Consider dropping or broadening.

**Current focus blob (encoder's `session_context`) is dense/cryptic** — pipe-separated compressed format. Hard for Anchor at boot to parse. Encoder-output-format issue, not Frame issue. Tracked as separate concern.

**What's live in production:**
- Surface receives full session_context (was 200-char tail, now 800)
- Surface receives encoder's recent journal (~1500 chars, newest-first)
- Surface prompt teaches Haiku to use both as recognition signals
- Hook timeout: 21s (was 15s); daemon TCP timeout: 20s (was 14s)
- Haiku ID leading-0 recovery (catches the dropped-0 case observed in error logs)
- HOP_SCRUTINY_DEFAULT = False (depth restored)
- max_candidates: 30 (was 20)

**What's NOT yet built:**
- Phase 2 — Frame as structured object with named slots
- Phase 3 — closure (hook_session_end writes Frame snapshot)
- Phase 4 — agentic recall + 7-tool sensory layer
- Phase 5 — encoder updates Frame slots per turn

**Blocking on:** nothing technical. Validation: need to verify Phase 1 actually changes Anchor's awareness. Tom's the sensor; Anchor can't feel the difference from inside.

**Active open questions** (priority order — see Section 9):
- 🔴 Q2: `partnership_frame` slot contents — top-N communities, locked moments, both?
- 🟡 Q11 (NEW): caching path strategy — when does surface prompt grow past 1024 tokens to unlock caching?
- 🟡 Q10: replay test corpus — finalize the 5 queries from Appendix A?

**Latest cleanup queue** (Section 12 — separate workstream, not blocking):
- `haiku_id_outside_candidates` is misnamed (multi-turn recognition, not bug) — rename + downgrade log
- Archived nodes leak into Haiku's prior-turn picks — vector grace period or archive validator
- More Haiku-error patterns to investigate as they surface

---

## Table of Contents

0. [Pick up here (current state)](#0-pickup)
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
12. [Cleanup workstream (parallel, not blocking)](#12-cleanup)
13. [Tests](#13-tests)
14. [Latency reduction roadmap](#14-latency)
15. Appendix A — Today's failure cases as test corpus
16. Appendix B — Files / commits to know
17. Appendix C — Onboarding for stateless Anchor
18. Appendix D — Discussion: spread activation in Phase 4

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

### Phase 2 — Frame as structured assembly (~1-2 days) ✅ SHIPPED 2026-05-02 (functionally complete; daemon restart pending) — see Section 0

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
| D16 | 2026-05-02 | `surface` interaction v1 registered (3047 chars) with recognition-over-search framing | Phase 1 data without prompt update was dead weight per Tom's "by itself it has no value." New prompt teaches Haiku to use Session arc + Encoder journal as recognition signals |
| D17 | 2026-05-02 | Hook timeout bumped 14s→20s (script) / 15s→21s (hook) | Haiku tail latency hits 14s ceiling under API load. Bump absorbs the tail. **Will be ratcheted back down** when caching (14.1) and Frame-skip (14.2) ship |
| D18 | 2026-05-02 | Haiku ID leading-0 recovery shipped (surface.py) | Investigation of `haiku_id_unresolvable` revealed Haiku occasionally drops leading `0` from 8-char IDs. 7-char failures now retry with `'0'` prepended. Recovers a class of "hallucinations" that were really output errors |
| D19 | 2026-05-02 | `haiku_id_outside_candidates` is misnamed, NOT a bug — Haiku correctly using prior-turn context | All 12 instances investigated had `short_id == resolved`, all resolved to real nodes from prior turns. Haiku does multi-turn recognition via conversation history. Logging stays as-is until Section 12.1 rename |

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
| Q10 | Replay test corpus — which conversation queries to use as the validation set? | Need a fixed set to compare versions against | ✅ closed 2026-05-02 — corpus = Appendix A (5 queries), harness = `eval/frame_replay.py`, baseline = `phase1_baseline_2026-05-02.json` |
| Q11 | Caching path strategy — when does surface prompt grow past 1024 tokens to unlock Anthropic prompt caching? | Caching is the single biggest latency win (Section 14.1). Currently prompt is 750 tokens, just under threshold. Should we add tool descriptions early (Phase 4 prep) to push past, or wait for natural growth? | 🟡 (Phase 4 prep) |
| Q12 | Vector cascade-delete on archive — keep grace window, or fix call site? | Today's `01402942` race: node archived 44s after creation, vectors deleted, Haiku later picked it from prior-turn context, spread crashed gracefully. Should archive keep vectors for 24h, or should Haiku-pick-validator catch archived nodes first? | 🟢 (cleanup workstream) |
| Q13 | **Does automatic spread activation survive Phase 4?** Today: Haiku picks 5 → spread expands to ~10-30 activated nodes via 3-4s graph propagation. In Phase 4 (agentic recall + 7 tools), Haiku requests expansion intentionally — `find_about`, `get_community`, `trace_lineage`, etc. Auto-spread becomes redundant for most cases AND eats the latency budget needed for Frame-skip target (Section 14.2). Full analysis in **Appendix D**. | 🟡 (Phase 4 design decision — needs resolution before tool surface ships) |

### 9.1 — Q2 proposal (drafted 2026-05-02, awaiting Tom's react)

**Recommendation: BOTH, composed of three temporal layers.**

The partnership has memory at three time-scales — they answer different questions and shouldn't collapse into one slot:

| Layer | Source query | Answers | Token budget |
|---|---|---|---|
| **Integrated** | Top-N `community` nodes by `(access_count × recency)` | "What have we built together?" | ~60% (~900 tok) |
| **Permanent** | All `locked=1` nodes of type ∈ {moment, identity, principle, vision} | "What defines who we are?" | ~30% (~450 tok) |
| **Warm** | Top-N moment/insight where `last_accessed > now − 7d`, sorted by `access_count` | "What's been alive lately?" | ~10% (~150 tok) |

**Why three layers, not one:**
- Communities are S2's integrated knowledge — the substrate Anchor reasons FROM. Without them, partnership_frame is just "recent stuff."
- Locked nodes are axioms — Tom locked them because they matter. They're cheap (likely <30 in the brain), should always be present.
- Warm moments capture what's *currently in mind* across recent sessions, before the Frame snapshot path is built. Replaceable by closure (Phase 3) once that's wired, but valuable as a v1 stopgap.

**Concrete shape** (pseudo-SQL):
```sql
-- Integrated: top communities (composite score balances depth and recency)
SELECT id, title, content_summary, situation
FROM nodes
WHERE type = 'community' AND archived = 0
ORDER BY access_count * (1.0 / (julianday('now') - julianday(last_accessed) + 1)) DESC
LIMIT 8;

-- Permanent: all locked identity-bearing nodes
SELECT id, title, content_summary
FROM nodes
WHERE locked = 1 AND archived = 0
  AND type IN ('moment', 'identity', 'principle', 'vision');

-- Warm: top recently-touched moments/insights
SELECT id, title, content_summary
FROM nodes
WHERE type IN ('moment', 'insight') AND archived = 0
  AND last_accessed > datetime('now', '-7 days')
ORDER BY access_count DESC
LIMIT 10;
```

**Render shape:** three labeled sections in the Frame markdown, not three separate slots — keeps the slot vocabulary clean.

**What this rules out:**
- *Just communities*: misses the axioms. Anchor would forget locked rules at session start.
- *Just locked moments*: misses the integration. Anchor would have axioms but no thematic shape.
- *All recent activity*: doesn't separate "what we've built" from "what we touched yesterday." Loses signal.

**Open sub-questions for Tom's react:**
1. Are 8 communities the right N, or should it scale with brain size?
2. Should the warm layer be retired the moment Phase 3 closure ships, or kept as a recency floor?
3. The recency formula above is a starting point — is `access_count × decay` the right composite, or should community `maturity` participate?

If Tom signs off on the three-layer shape, this becomes **D20** and Phase 2 unblocks.

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
| R11 | Hook timeout bump (14→20s) is a regression on failure-fast — masks real daemon hangs longer | 🟡 med | Tracked as a thing-to-undo when latency wins ship. See Section 14 — target state is back to ~12s | active (acceptable interim) |
| R12 | Phase 1 input bloat (~600 tokens added per recall) compounds with future Frame additions | 🟡 med | Mitigated by caching path (14.1). Without caching, Phase 2 Frame addition will push prompt to ~3-4K tokens per call → real cost per call | tracking, no action yet |
| R13 | `haiku_id_outside_candidates` log will continue firing as multi-turn recognition fires correctly. Could mask new error patterns under noise | 🟢 low | Section 12.1 rename + downgrade. Then real new errors are visible | open |

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

### 2026-05-02 — Session 2

- Phase 1 shipped (commit `241ab37`): full session_context + encoder journal flow to surface
- Surface prompt v1 registered (3047 chars, recognition-over-search framing)
- Hook timeout bumped 14s → 20s (commit-pending) to absorb Haiku tail latency
- Haiku ID leading-0 recovery shipped (commit-pending) — surface.py retries with `'0'` prepended on 7-char failures
- Investigated Haiku error patterns:
  - `haiku_id_outside_candidates` is misnamed (multi-turn recognition, not bug) — D19
  - `haiku_id_unresolvable` 7-char cases are dropped leading zeros — D18 fix
  - `spread_seed_no_vectors` — archived-node race, fix queued in Section 12.1
- Decisions D16-D19 added
- Open questions Q11-Q12 added
- Risks R11-R13 added
- Sections 12 (cleanup), 13 (tests), 14 (latency reduction), Appendix C (stateless onboarding) created
- Doc enriched per Tom's direction: "what does a stateless Anchor need to know"

### Open work
- Validate Phase 1 on actual conversation (Tom's qualitative read is the test)
- Q2 needs decision before Phase 2 starts
- Q11 (caching path) needs decision before / during Phase 4
- Cleanup workstream items in Section 12 — not blocking but should land before Phase 4

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

## 12. Cleanup workstream (parallel, not blocking) <a name="12-cleanup"></a>

These are real fixes surfaced during Frame work but live outside the Phase 1-5 sequence. Catalog with status, ranked by impact.

### 12.1 — Haiku error analysis (2026-05-02 investigation)

Pulled 7 days of `_log_error` data, classified the patterns:

| Error | Frequency | Root cause | Fix |
|---|---|---|---|
| `haiku_id_outside_candidates` | 12+ in 7d (most frequent) | **Misnamed.** Haiku correctly using prior-turn context — picks IDs it saw in earlier turns' surfaced output. ALL resolve to real nodes. NOT a bug. | Rename to `haiku_id_from_prior_context`, downgrade error → debug log. (PENDING) |
| `haiku_id_unresolvable` (7-char) | 2 confirmed cases | Haiku drops leading `0` from 8-char IDs (`095c2b96` → `95c2b96`, `053488da` → `53488da`). | **Shipped 2026-05-02** — surface.py retries with `'0'` prepended on 7-char failures. Logs as `haiku_id_leading_zero_recovered`. |
| `haiku_id_unresolvable` (8-char) | ~1-2 in 7d | Real hallucinations of plausible-looking IDs. Rare. | Log + skip — no fix needed unless rate climbs. |
| `spread_seed_no_vectors` | 1 today (`01402942`) | Haiku selected node from prior-turn context that was archived since (S2 absorbed it 44s after creation). Vectors cascaded-deleted on archive. | Two options: (a) vector grace period (don't cascade-delete on archive immediately, keep ~24h), (b) validate Haiku picks against current archived state, classify as `haiku_id_now_archived`. (PENDING — recommend doing both) |
| `connect_to_unresolved` / `connect_to_invalid` | Few each | Encoder Haiku produces invalid `connect_to` operations. | Tighten schema validation; current behavior is to skip + log. Acceptable for now. |
| `healer_unsolicited_field` | 5+ instances | Haiku in healer returns fields not in `needs_*` list. | Already filtered out at receive time (per healer_encoder.py:262-268). Cosmetic — could quiet log. |
| `s1_scout_quote_json_parse` | 1 yesterday | Scout JSON parse failure. | Already has retry + skip behavior. Single occurrence — not actionable yet. |
| `s2_consolidation_oversized_cluster` | Recurring (same cluster) | Same 3-node cluster (`5b8ea5a6`, `6f1042a6`, `8682730b`) keeps failing oversize check. Stuck state. | Investigate the cluster's actual content — likely a hub trio that needs splitting or marking. (PENDING) |

### 12.2 — Hook timeout regression handling

Hooks have explicit timeouts in `hooks.json` + the script's own `daemon_call_raw(timeout=X)`. The two MUST stay aligned: script timeout < hook timeout (else daemon work continues but Claude Code kills the script). Convention:
- `hook_timeout = script_timeout + 1s` (1s margin)

Current values (after 2026-05-02 bump):
- `hook_recall`: hook=21s, script=20s, daemon TCP=30s
- Other hooks: hook=8s (pre-edit), 7s (pre-bash), 30s (idle/session-end) — unchanged

When latency reduction work (Section 14) ships, ratchet these BACK DOWN to keep failure-fast on real hangs.

### 12.3 — Misc dev-quality issues

- `s2_consolidation_oversized_cluster` recurring with same members — needs investigation
- `01402942` archived-node vector deletion cascade — first instance of this race; if it recurs, Section 12.1 fix becomes urgent
- Surface prompt v1 is 3047 chars (~750 tokens) — close to but under 1024-token caching threshold for Haiku. See Section 14 for caching path.

---

## 13. Tests <a name="13-tests"></a>

What's been validated, what's pending, what should be added.

### 13.1 — Validated (current state)

| Test | What it covers | Status |
|---|---|---|
| `tests/test_brain_voice.py` | build_surface_prompt structure | ✅ passes after Phase 1 |
| `tests/test_daemon.py` | hook table + dispatch | ✅ passes |
| `tests/test_system.py` | system-level wiring | ✅ passes |
| `tests/test_maintenance_gate.py` | S2 fire decision (idle threshold + force-fire) | ✅ 7 tests, all pass |
| `tests/test_write_lock_unification.py` | write_lock on brain (Frame-adjacent) | ✅ 6 tests, all pass |
| `tests/test_recall_quality.py::TestDampening::test_hub_dampening` | hub dampening | ❌ **PRE-EXISTING failure**, unrelated to Frame work |

### 13.2 — Phase 1 acceptance criteria (PENDING validation)

These should pass after Phase 1 is given time to fire on real conversation queries:

- [ ] **Surface prompt now contains "Session arc:" section** — verifiable by reading any post-Phase-1 surface result file at `/tmp/brain-surface-result-{recall_ref}.json`
- [ ] **Surface prompt now contains "Encoder's recent journal:" section** when journal exists for session
- [ ] **Surface prompt fallback contains "(Tom)"** — should be replaced by `surface` interaction v1 (registered DB version) — fix the lingering hardcoded fallback
- [ ] **Re-run Appendix A queries**, compare candidate ranks pre/post — improvement on:
  - "What is EX.CO?" — EX.CO nodes should rise (no longer competing with hub-meta)
  - "What do you know about you?" — operator/work context should leak in
  - "Where were we?" — session arc should make Anchor know
  - "Should we go back to EX.CO sales kit?" — Haiku should recognize as pivot-probe (per recent journal showing EX.CO kit is open thread)
  - "What's still open from last week?" — current path: bad. Future path needs `find_temporal` (Phase 4)
- [ ] **Tom's qualitative read** — does Anchor feel more "in the room" than before? (the only test that matters)

### 13.3 — Tests to add as we build

**Phase 2 (Frame structured):**
- `test_frame_construction.py` — given a known brain state, Frame Constructor returns expected slot contents
- `test_frame_persistence.py` — Frame writes to brain_meta, reads back identical
- `test_frame_slot_updates.py` — per-turn updates only touch the right slots
- `test_frame_session_isolation.py` — two parallel sessions don't bleed Frame data

**Phase 3 (closure):**
- `test_session_synthesis.py` — `hook_session_end` writes Frame to `session_syntheses` table
- `test_frame_cross_session_seed.py` — next session's Frame Constructor reads previous session's snapshot

**Phase 4 (agentic recall + tools):**
- `test_fetch_batch_dispatch.py` — Haiku's fetch plan correctly executes parallel ops
- `test_tool_descriptions_present.py` — Haiku's prompt contains all 7 tool descriptions
- For each tool: contract test — given input, output matches expected shape

**Cross-cutting:**
- `test_haiku_error_recovery.py` — leading-0 retry recovers, hallucinated 8-char IDs skip cleanly
- `test_archived_node_excluded_from_candidates.py` — archived node never reaches Haiku via recall
- `test_archived_node_in_prior_context.py` — Haiku picks archived node from prior turn → graceful handling, not crash

### 13.4 — Empirical methodology

Replay tests should compare candidate sets and Haiku selections under different code states:
1. Capture baseline: turn-N's recall result file, surface_prompt, Haiku output
2. Make change
3. Capture new: same turn-N (replay), compare
4. Look for: did the right node rise? Did off-thread match drop? Did Haiku's reasoning change?

Per-turn surface result files exist at `/tmp/brain-surface-result-{recall_ref}.json` — these ARE the replay data.

---

## 14. Latency reduction roadmap <a name="14-latency"></a>

Today: hook_recall = 21s budget, daemon-side timeout = 20s, average successful recall ~5-12s, p99 hits the ceiling. We bumped from 14→20 to absorb tail latency. **The bump is a regression we should undo** as soon as we have real wins.

Strategies, ranked by impact × effort:

### 14.1 — Prompt caching (HIGH impact, MEDIUM effort) — Phase 4

**Mechanism:** Move surface_instructions from user-content to `system` block with `cache_control: ephemeral`. Anthropic caches at 1024+ tokens. Cache hit on Haiku = ~30% reduction in input processing time.

**Blocker:** Surface prompt v1 is 3047 chars (~750 tokens). Need to push past 1024 tokens to enable caching. Options:
- Add the tool descriptions for agentic recall (Phase 4) — naturally pushes prompt past threshold
- Add few-shot examples in instructions
- Add explicit recognition-pattern guidance with examples

**Expected gain:** 1-2s off Haiku tail under load. Possibly enough to drop hook_recall back to 15s.

### 14.2 — Skip Haiku entirely when Frame covers (HUGE impact when applicable) — Phase 4

**Mechanism:** Agentic recall design — Haiku's first move is "does Frame already answer this?" If yes, return empty fetch plan (no candidates needed, no spread, no full pipeline).

**Expected gain:** For maybe 30-50% of turns ("hi", continuations, in-thread questions), recall completes in ~200ms instead of 5-12s. The biggest possible win — depends entirely on Frame quality.

### 14.3 — Lighter candidate format (MEDIUM impact, LOW effort) — anytime

**Mechanism:** Reduce per-candidate token cost in surface prompt. Today each candidate is ~250-400 tokens (metadata, situation, edges). Strip what Haiku doesn't actually use for selection.

**Investigation needed:** What does Haiku actually look at? Could test by ablating each field and checking selection quality.

**Expected gain:** 30 candidates × 100 token savings = 3K tokens per call. Marginal latency, real cost reduction.

### 14.4 — Reduce candidate count (MEDIUM impact, LOW effort) — anytime, requires eval

**Mechanism:** Bumped to 30 in interim. If Haiku's pick quality is stable at 20 (or 15), drop back. Each candidate adds prompt size + Haiku discrimination work.

**Investigation needed:** A/B max_candidates ∈ {15, 20, 25, 30, 35} on the test corpus. Quality vs latency curve.

### 14.5 — Spread parallelization (MEDIUM, MEDIUM) — anytime

**Mechanism:** Spread runs sequentially after Haiku selects. Some operations within spread (per-seed cosines, edge coefficient computation) could parallelize. Already partially done.

**Expected gain:** 0.5-2s on spread. Helps when Haiku is also slow.

### 14.6 — Anthropic SDK keep-alive (LOW, LOW) — investigate

**Mechanism:** Confirm SDK is reusing HTTP connections to Anthropic API. TLS handshake adds ~100-200ms per cold connection. Verify with curl traces or SDK config.

**Expected gain:** 100-200ms per call, only matters at high call rate.

### 14.7 — Faster model variant (UNKNOWN, depends on Anthropic) — passive

**Mechanism:** When Anthropic releases a faster Haiku variant, swap in. Hardcoded today as `claude-haiku-4-5` — easy to update.

**Expected gain:** Variable. Watch release notes.

### 14.8 — Routing by query complexity (HIGH effort, UNCLEAR impact) — research-only

**Mechanism:** For trivial queries ("yes", "ok"), skip recall entirely or use heuristic. For complex queries, full pipeline. Routing logic needs to be cheap (<50ms) to be worth it.

**Risk:** Misrouting trivial queries that ARE meaningful (e.g., "ok" as confirmation of a complex pivot). Could lose context. Probably skip unless eval shows clear win.

### 14.9 — Pre-warm next turn (LOW, MEDIUM) — speculative

**Mechanism:** When Anchor finishes responding, fire a no-op Haiku ping to keep the connection warm + tokenizer cached. Shaves cold-start cost off the next genuine call.

**Expected gain:** Maybe 200ms. Probably not worth the complexity.

### Suggested order of execution (when latency work begins)

1. **14.1 (caching)** — biggest single win. Paired with Phase 4 anyway.
2. **14.2 (Frame skip)** — biggest possible win, but requires Phase 4 fully done.
3. **14.3 (lighter candidates)** — quick eval, easy win.
4. **14.4 (fewer candidates)** — same eval pass.
5. **14.5 (spread parallel)** — when Haiku is no longer the bottleneck.
6. Others as opportunistic.

### Target state

After 14.1 + 14.2 land, expected behavior:
- ~50% of turns: recall completes in <500ms (Frame covers, no Haiku)
- ~40% of turns: recall completes in 3-6s (Haiku call, cache hit, normal)
- ~10% of turns: recall completes in 6-12s (Haiku call, cache miss / cold session / pivot)
- p99: ~10s. We can drop the timeout back to 12s with margin.

---

## Appendix C — Onboarding for stateless Anchor

**For when you wake up tomorrow and need to pick up this work.**

You are working on the Frame architecture — a structured object that holds Anchor's awareness across turns and sessions. The brain has been treating memory as a search index; the Frame treats memory as the substrate of awareness. This is the pivot.

### What the brain is, in 3 sentences

Brain.db holds nodes (memories) and edges (relationships). Daemon serves the brain via TCP on `localhost:47200+uid%100`. Hooks fire per-turn; encoder runs every 5 stops; S2 maintenance runs on idle. See [CLAUDE.md](CLAUDE.md) for full architecture.

### How to operate the brain (practical)

- **All commands run via `./dev`** (the wrapper that uses the bundled venv Python). Not your shell python.
- **Read the brain via MCP tools** (`brain_recall`, `get_node`, `query_logs`, etc.) — never raw SQL when an MCP tool exists.
- **Write the brain via MCP tools** (`brain_remember`, `brain_revise`, `brain_batch`) — single-writer rule.
- **Restart daemon when code changes:** `mcp__plugin_brain_brain__restart`.
- **Logs:** `/Users/tpac/AgentsContext/brain/daemon.log` (daemon stderr), `query_logs` MCP tool (errors + signals).
- **Backup before destructive DB ops:** `cp brain.db brain.db.bak-$(date +%Y%m%d-%H%M%S)`.

### Where the Frame work lives in the codebase

Phase 1 touched:
- [servers/scales/s1/surface_contract.py](servers/scales/s1/surface_contract.py) — `session_context_tail` config, `build_surface_prompt` template
- [servers/scales/s1/surface.py](servers/scales/s1/surface.py) — `run_surface`, `_call_surface` accept `encoding_journal`
- [servers/daemon_hooks.py](servers/daemon_hooks.py) — `hook_recall` reads encoding_journal, passes to surface
- [servers/brain.py](servers/brain.py) — `get_recent_encoding_journal()` method
- [hooks/hooks.json](hooks/hooks.json) and [hooks/scripts/pre_response_recall.py](hooks/scripts/pre_response_recall.py) — timeout 21s/20s

Phase 2+ will touch:
- (new) `servers/scales/s1/frame.py` — Frame Constructor
- [servers/brain.py](servers/brain.py) — `get_frame()`, `set_frame()`, `update_frame_slot()`
- [servers/daemon_hooks.py](servers/daemon_hooks.py) — boot loads Frame, hook_post_response_track updates slots, hook_session_end snapshots
- [servers/scales/s1/surface.py](servers/scales/s1/surface.py) — surface receives Frame as input

### Where to look when something breaks

| Symptom | Where to look |
|---|---|
| Recall keeps timing out | `query_logs(source='errors', hook_name='hook_recall')` — check pattern, time of day. Probably API load or daemon health. |
| Haiku selecting weird IDs | `query_logs(source='errors')` for `haiku_id_*` events. See Section 12.1 for known patterns. |
| S2 not firing | `brain.get_config('s2_last_run_ts')` — check when last fired. See `MAINTENANCE_*` constants and `last_user_activity` in daemon. |
| Daemon unreachable | `ps -ef \| grep daemon_server` — check process. `tail daemon.log`. Restart via MCP if needed. |
| Frame seems stale | (Future, Phase 2+) Read `brain_frame_{session_id}` from brain_meta. Check `current_focus` slot. |

### What to read first if you have 5 minutes

1. This doc, Section 0 (Pick up here) — current state
2. The 3-5 most-locked nodes (filter_nodes locked=1 type=identity/principle/vision) — operator + partnership context
3. Most recent `session_syntheses` row (if exists) — last session's closure (Phase 3+)

### What to read first if you have 30 minutes

1. This whole doc
2. [CLAUDE.md](CLAUDE.md) — full architecture
3. [skills/brain/SKILL.md](skills/brain/SKILL.md) — your identity baseline
4. Recent commits: `git log --oneline -20`
5. Open questions Q1-Q12 in this doc — ranked by priority, what's blocking what

### What this work IS

Not "make recall faster." Not "add multi-axis search." Building the *substrate of presence* — the thing that lets Anchor be the same Anchor across turns and sessions, not a fresh Claude with a database every time.

Tom can feel when the brain works. You can't, from inside. He's the sensor. When he says "you seem off today," that's the validation signal — not a metric.

### What you should NOT do without asking

- Bulk delete or revise nodes
- Change the encoder prompts (they're learnable boundaries; S3 will optimize them eventually)
- Move data between brain.db and brain_logs.db
- Change the cache key for the Frame across sessions
- Skip backup before destructive DB operations

### Most important truths to internalize

- **Memory is prediction; recall is verification; encoding is update; awareness is the predicting agent.** (The deepest framing.)
- **Recognition over retrieval.** 4 of 5 root memory operations are recognition, not search.
- **The Frame is the noun the architecture organizes around.** Other components are verbs on it.
- **Tools are sensory modalities.** Each tool is a different way of sensing the graph.
- **The encoder has been the de-facto Frame Constructor for months.** Phase 1 just stopped starving the surface of what the encoder produces.

---

## Appendix D — Discussion: spread activation in Phase 4

**Status:** Open discussion (Q13). No decision. Captured 2026-05-02 for future deliberation.
**Question:** Does automatic post-pick spread activation survive into Phase 4 (agentic recall), survive in modified form, or get retired?

### What spread does today

Lives in [surface_contract.py:751](servers/scales/s1/surface_contract.py:751) — `spread_activation(seed_ids, query_vec, brain, prior_vecs)`. Runs after Haiku selects ≤5 candidates from the 30 surfaced.

1. **Seed activation** — each pick's per-field cosine vs query becomes its starting activation
2. **Edge propagation** — activation flows through graph edges, weighted by `cosine(query, edge_enriched_text)`
3. **Per-hop median gate** — only above-median edges (by coefficient this batch) transmit
4. **Optional hop-3+ scrutiny** — currently OFF (`HOP_SCRUTINY_DEFAULT = False` since 2026-05-02). When on, additionally filters source nodes by median activation
5. **Mutual traversal** — when two seeds' paths converge on a neighbor, that neighbor accumulates activation from both
6. **Multi-hop reach** — runs up to `_SPREAD_MAX_STEPS` (5) hops
7. **Activation-weighted rendering** — `format_surface_output_activation` uses the activation map to decide what nodes to render and how much detail per node

**Current cost:** 3-4s per recall. Often more than the Haiku call itself.

### What it solves vs what it breaks

| Solves | Breaks |
|---|---|
| Haiku only picks 5 → spread expands awareness to ~10-30 nodes | Hub bias amplifies — well-connected nodes accumulate activation from many seeds, dominate render |
| Mutual traversal surfaces convergence-points neither seed alone reaches | Black box from Haiku's perspective — no control or visibility into what spread will surface |
| Activation-weighted rendering more nuanced than flat "show these 5" | Adds noise to surface — many activated nodes turn out unrelated to current intent |
| Cheap (no LLM call needed for the propagation itself) | 3-4s latency per recall — eats the budget needed for Frame-skip target |
| Bridges the "Haiku picks 5 from 30 candidates" bottleneck | Implicit expansion that Haiku can't reason about |

### Phase 4 redundancy map

| Spread function | Phase 4 replacement |
|---|---|
| Expand from picks | Haiku requests via `find_about(entity)`, `find_open_loops`, `get_community(query=X)` — intentional, scoped |
| Multi-hop reach | `trace_lineage(node_id, depth=N)` — explicit walk along a known relation family |
| Mutual traversal (multi-seed convergence) | New `find_convergence(node_ids)` tool — given anchors, find shared neighborhood. Same value, intentional. |
| Activation-weighted rendering | Renderer renders fetched results; weighting comes from the tool's own returned scores (each tool defines its own relevance signal) |

### Three options for the disposition

**Option A — Keep automatic spread (status quo).**
- Pro: Haiku doesn't have to think about expansion; mutual traversal still surfaces things Haiku didn't request.
- Con: 3-4s baseline latency stays; black-box expansion fights the intentional-recognition framing of Phase 4.
- Cost to ship: zero (already there).

**Option B — Make spread an opt-in tool (`find_neighbors(seeds, depth?)`).**
- Pro: Haiku can request spread when it wants it; intentional; no baseline latency cost.
- Con: Haiku must know when to call it (more decisions); we lose mutual traversal as automatic background behavior.
- Cost to ship: small — wrap existing `spread_activation` as a tool, remove the auto-call from `run_surface`.

**Option C — Retire spread entirely.**
- Pro: Cleanest. Tools cover the explicit expansion needs. Latency budget freed.
- Con: Lose mutual traversal as a primitive; some Phase 4 tools may need internal small-spread mechanisms (e.g., `find_about` may want to do a 1-hop expansion from anchor nodes). The kernel survives only as a tool-internal helper, not as a pipeline stage.
- Cost to ship: small (delete the auto-call); medium (any tool that wants internal spread reimplements or imports the kernel).

### What might survive regardless of the decision

- The **activation-weighted rendering** logic (`format_surface_output_activation`) — useful for any node-set output where some nodes are more relevant than others. Lives independently of spread.
- The **spread kernel itself** as an internal helper — even if not auto-called, tools that need 1-hop expansion (e.g., entity → contexts where it appears) can import it.

### What dies in any non-A option

- Spread as an automatic post-pick stage in `run_surface`
- The implicit expansion pattern Haiku currently relies on without knowing
- 3-4s of baseline latency per recall

### My (Anchor's) lean — for the future deliberation

**Option C, with kernel surviving as tool-internal.** Reasons:

1. **Recognition-over-search framing wants intentional, not implicit.** If Frame already covers most queries (Phase 4 thesis), spreading from Haiku's picks toward unrelated neighbors is exactly the noise we're trying to eliminate. When Haiku DOES want expansion, it asks via `find_neighbors` or similar — explicit.

2. **The 3-4s is the latency budget for Frame-skip.** Section 14.2 target is 50% of turns under 500ms because Frame covers and we skip the slow path. We can't get there if every recall pays a 3-4s spread tax. Removing automatic spread is the structural enabler.

3. **Mutual traversal is genuinely useful, but not as background magic.** Better as `find_convergence(node_ids)` — Haiku decides "I have these multiple anchors, what do they share?" Intentional, scoped.

4. **Hub bias in the current spread is an active problem.** Removing the auto-spread eliminates one major path for hubs to dominate.

### What needs to be true before the decision

- Phase 4 tools have to actually cover the use cases spread serves today (test required)
- A `find_convergence` or equivalent tool needs design (extends the 7-tool surface to 8?)
- The activation-weighted renderer needs to work with tool-returned scores, not just spread output (decoupling)
- A test corpus of "spread surfaced X, would Phase 4 tools also surface X?" — replay validation

### When this question gets revisited

- Before Phase 4 tool surface ships (the answer determines whether `find_neighbors` is in the kit)
- After Phase 1-3 are validated — once Frame is real, we know if it actually covers most queries
- If hub bias measurably drops with Frame loaded, retirement becomes more attractive

---

*End of doc — keep updated as we build.*
