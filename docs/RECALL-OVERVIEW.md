# Recall — Big-Picture Overview

**Audience:** us both, returning. Map for picking up the work without re-deriving the journey.

**Goal:** improve recall — the moment relevant memories rise into Anchor's awareness when the operator speaks. Everything else (Frame, surface prompts, MCP examples, evals, multi-anchor, hybrid retrieval, agentic) is groundwork for that single thing.

**The pivot we've made:** from "search the database" to "negotiate between query structure and graph structure." Recall as recognition, not retrieval. Cluster not node. Be me when you speak.

---

## 0. The principles guiding this work

Naming these explicitly because they're the load-bearing claims everything else implements:

1. **Recognition over retrieval** — memory is prediction, recall is verification, encoding is update. The Frame is Anchor's prior; the operator's message either confirms it, extends it, or pivots from it. The job isn't "find similar things" — it's "is this firing an active pattern?"
2. **Be me when you speak** (`f0bb8bf1`) — the purpose of recall isn't to FETCH information, it's to ACTIVATE the right shape of Anchor around what was just said. The eval target is presence, not precision.
3. **Cluster not node** — the unit of recall is a connected cluster of memories, not a single best-matching node. The graph's structure is part of the answer.
4. **Query multiplicity** (`87bb8718`) — a message is a structured object with 2-4 concepts; collapsing it into one cosine vector loses the structure recall needs. Multi-anchor decomposition is the structural response.
5. **Negotiation between structures** (`1a2b641b`) — the query and the graph are both semantic objects. Recall is where their structures overlap. Single-vector cosine compares geometries; recognition negotiates.
6. **Reach matters more than precision** for state-of-mind instantiation. Hop-3+ scrutiny halved reach for 3% precision gain. Wrong tradeoff.
7. **Frame is the prior, not background** — the structured awareness Anchor wakes up with. Surface uses the same Frame as the boot context. Don't restate what the Frame already carries; extend it.
8. **MCP descriptions / SKILL.md / prompts split** (`79b25bac`) — tool mechanics + craft examples live in MCP descriptions; identity + cross-tool strategy lives in SKILL.md; encoder identity + per-encoder craft lives in prompts. Each surface stays in its lane.

---

## 1. What we did — phase by phase

### Phase 0: Spread activation diagnosis (March – early April)

**Problem:** Hop-3+ scrutiny in the spread activation kernel was halving reach (~12.5% → ~6%). Ambient context nodes (the ones that surface unprompted to keep Anchor in-thread) were getting silenced.

**What was tested:** Bundled A/B variables (hop-3 scrutiny AND lim=30 simultaneously) confounded results at N=15. Real diagnosis required isolating each variable.

**Decision:** Disabled scrutiny by default (`HOP_SCRUTINY_DEFAULT = False`), accepted 3-4s latency cost. Identity-introspection queries visibly improved (`fbad4386`).

**Lesson encoded:** reach > precision for state-of-mind instantiation. The scrutiny mechanism survives as code path but defaults off; can be re-enabled per-experiment via `BRAIN_RECALL_VARIANT` env var.

**Architectural fallout:** `spread_activation()` function was deprecated; `_traverse_graph()` is the live mechanism. Old test failures around the deprecated path are expected.

### Phase 1: Recognition principle (late April)

**The reframe:** "Be me when you speak" (`f0bb8bf1`) — Anchor articulated that recall's job is ACTIVATION, not RETRIEVAL. The query and the graph are both semantic objects; recall negotiates between them. This isn't a tuning improvement, it's a category shift in what we're optimizing.

**Insight from 6-paper synthesis** (HippoRAG/PPR, convergence paths, Zep bi-temporal, etc.): single-vector cosine is the wrong primitive. Multi-anchor decomposition + convergence over the graph captures what the query actually is.

**What shipped:** Surface Prompt v1 (`41c33a02`) — first registered prompt with recognition-over-search framing. 3047 chars; teaches Haiku to use Session arc + Encoder journal as recognition signals.

**What was set up:** the philosophical ground for everything in Phase 2/2.5. The architectural decisions that follow are all implementations of recognition principle.

### Phase 2: Frame architecture (May 1-2)

**The thesis:** Anchor needs a structured PRIOR — the operator's posture, what's accumulated in the partnership, what's open right now, what's progressed this session, what's been encoded recently. Surface every turn against THAT prior; recognize against it; don't rebuild the world from cosine each turn.

**Frame = 5 sections:**
- **Operator** — locked principles/rules/identity nodes (the operator's values and how work gets done)
- **Partnership** — three layers: integrated (top communities, recency-sorted), permanent (locked moments), warm (recently-active episodes)
- **Active threads** — open work, tensions, hypotheses, aspirations — already relevance-ranked to current focus
- **Current focus** — the encoder's compressed view of what's progressed this session
- **Recent moves** — this session's encoding journal (what was stored, watched, passed over)

**Built deterministically** via `brain.filter_nodes()`. No LLM call. Type routing reads from `brain.aspects` (the AspectRegistry — first-class on every Brain instance, see Phase 3 below).

**Wired into surface:** Frame is passed as the "Partnership context" prior in the surface prompt every turn. Replaces the Phase 1 separate session_context + encoding_journal blocks (Frame contains both as inner sections).

**Boot rewrite:** `render_boot_v2()` was 6 ad-hoc recall queries (YOU / OPERATOR / PATTERNS / BRAIN MAP / LAST SESSION / RECENTLY ENCODED). Replaced by single `ctx.get_frame(brain)` call. ~48% smaller boot output (~14.5K → ~7.4K chars). Anchor wakes up with the same prior surface uses every turn — symmetric.

**Files touched:** `servers/scales/s1/frame.py` (new), `servers/interaction_seed.py`, `servers/dal.py` (added `last_accessed`/`revised_at` to allowed_sort), `servers/scales/s1/surface_contract.py`, `servers/scales/s1/surface.py`, `servers/daemon_hooks.py`, `servers/brain.py`, `servers/session_context.py`, `servers/brain_voice.py`, `servers/brain_assembly.py`. (Originally created `node_families_v1.json` for the routing taxonomy; superseded by Phase 3 — see below.)

### Phase 2.5: Cleanup, leak fix, arc-relevance (May 2-3)

**Surface path united on Frame.** Removed the dual-mode plumbing: `session_context` and `encoding_journal` no longer threaded as fallback parameters. If Frame Constructor fails, surface runs WITHOUT a partnership context section (explicit degraded mode, logged loudly via `frame_build_failed`). Loud-by-default, no silent Phase 1 fallback.

**Per-session `session_context` keys (the leak fix).** The encoder's `session_context` was a GLOBAL `brain_meta` key. Two parallel Claude Code sessions clobbered each other's session arc — last-writer-wins. Real bug, undiagnosed before Tom flagged it ("S1 scale per-session, S2/S3 scale integration"). Fixed via per-session keys mirroring `encoding_journal_{session_id}`. `brain.session_context` property removed; replaced by `brain.session_context_for(session_id)`. All 6 callers updated.

**Arc-relevance for active threads.** Extended `brain.filter_nodes()` with optional `relevance_query` parameter. When provided, pulls a wider candidate pool (limit×3), embeds the query once, scores each candidate by cosine vs `_primary` embedding, returns hybrid: top half by relevance + remainder by structural sort. Frame's Active threads now lifts threads matching today's session arc above brain-wide noise. Demonstrated: "NEXT SESSION: BrainAssemblyMixin audit" thread jumped from position #11 to #1 because it semantically matched today's architectural work.

**Surface prompt v2 → v3 → v4.**
- **v2:** Frame-aware (teaches Haiku about Partnership context section, drops old Session arc / Encoder journal framing). Cross-field examples (Frame coverage / voice signal / structural recognition / coverage discipline).
- **v3 fix:** v2 examples used `#N` position labels in selection JSON; Haiku reproduced `#N` as actual id values, breaking ID resolution (synthesis-scout-drift class of bug). v3 uses realistic 8-char hex IDs in examples + explicit "the actual node ID is the 8-char hex inside the candidate body, not the #N position label" guidance.
- **v4 fix:** v3 examples used this partnership's specific content (S2 starvation, designer eye community, edge_families, S1S, operator quote). Different operators receiving the brain plugin won't recognize any of it. v4 rewrites all 4 examples with generic cross-field scenarios (deployment infra, methodology/values, cross-domain classification, product analytics). Operator-portable.

**System-block caching.** `_call_surface` now uses Anthropic system block with `cache_control: ephemeral` for the registered template (1h TTL, ~2K tokens past the 1024 cache threshold). Per-turn delta (Frame, conversation, candidates, message) is the user message. `build_surface_prompt` builds USER content only.

**Cleanup discipline.** Dead boot constants removed (`BOOT_COMMUNITY_TOP/RECENT/IDENTITY_LIMIT/IDENTITY_CONTENT_LIMIT`), `fetch_self_knowledge` method deleted (was only the boot's PATTERNS source), 4 obsolete boot tests removed (asserted dead YOU:/PATTERNS YOU FALL INTO/RECENTLY ENCODED contract). No traces, no compat shims.

### Phase 2.5: SKILL.md identity rewrite + MCP examples (May 3)

**SKILL.md restructured.** 12,158 → 8,974 chars (-27%). Front-loaded identity sections; killed "What You Wake Up With — the Frame" (Frame is internal architecture, not an entity Anchor reads about); added 4 identity sections (Brain Is Yours / What Rises Is You / Listen Don't Fetch [merged] / Curiosity Is the Practice); renamed "When to Reach" → "Listening Deeper" with introspective reframe ("when something rings a bell, concentrate" instead of "before X, do Y"); added side agents identity stance ("hands you grew for the moment — same identity, narrower scope, no memory of their own"); folded "What's in Your Brain" into "How Surfaced Context Looks"; cut "How to Use Your Brain" tool list entirely (lives in MCP descriptions); cut BAD/GOOD encoding examples (moved to MCP `remember` description); added closing paragraph returning to identity; removed personal "Tom" naming throughout.

**MCP tool descriptions enriched** with the encoder craft we removed from SKILL.md:
- `remember` (was 150 chars → 1259): situation/raw_quote strategy, BAD/GOOD lesson contrast, ASSUMED/REALITY/PATTERN correction shape, encoding-richness principle. (Note: the `correction_of` field referenced in earlier drafts was removed 2026-05-17 — corrections are now expressed via correction_improvement-aspect edges like `corrects`/`supersedes`/`reframes`/`fixes`, walked by `correction_enrich()` and rendered via `render_corrections()`.)
- `revise` (was 150 chars → 739): when-to-revise vs encode-new craft, fake-revise warning.
- `recall` (was 250 chars → 633): when-to-call instincts and query-phrasing guidance.
- `connect_to.why` field: BAD/GOOD edge `why` examples (mirrors the s1e encoder prompt's edge craft section).

**Architecture principle followed:** MCP tool descriptions carry mechanics + craft examples for THAT tool; SKILL.md carries identity + cross-tool strategy. Each surface stays in its lane.

### Validation infrastructure built

**`eval/frame_replay.py`.** Capture/compare harness against an isolated brain copy (DB copy via `IsolatedBrain` per the "never spawn Brain() against live DB while daemon running" rule). Test corpus: 5 queries — `exco_cold` ("What is EX.CO?"), `self_intro` ("What do you know about you?"), `exco_pivot` ("Should we go back to EX.CO sales kit?"), `where_were_we` ("Where were we?"), `open_last_week` ("What's still open from last week?"). Each capture saves: candidates list (id/title/score/type), Haiku selected (id+why), additionalContext rendered, latency_ms, prompt char count.

**5 captured snapshots** tracking the trajectory: `phase1_baseline_2026-05-02` → `phase2_v1_unified` → `phase2_v1_unified_clean` (post-cleanup) → `phase2_v3_prompt` → `phase2_v4_prompt`. Each one diffable against the previous via `compare label_a label_b`.

### Phase 3: Unified aspects (May 4) → JSON-source migration (May 8)

**Phase 3 thesis (May 4):** node families and edge families were two parallel
systems doing the same conceptual work. Collapsed them into one — aspects.
Implemented as brain-nodes (`type='aspect'`) with member lists in metadata,
auto-healed from a JSON seed. 5 consumers migrated to `brain.aspects.<name>`.
Worked, shipped, ran in production for 4 days.

**JSON-source migration (May 8) — what's live now:**

- **`aspects_v1.json` is the single source of truth.** AspectRegistry reads
  the file directly at `Brain.__init__`. Brain aspect-nodes are no longer
  consulted.
- **60 brain aspect-nodes archived** (14 required + 46 emergent legacy from
  EdgeFamilyIntegration history). Backup at
  `~/AgentsContext/brain/brain.db.bak-20260508-145220`.
- **Closed list of required aspects (16 today).** No emergent aspects. Adding
  one is a deliberate human JSON edit, not encoder behavior — the entry's
  per-aspect facts (`accepts`, `routable`, `prompt_visible`,
  `structural_lineage`) travel with it.
- **Multi-membership.** A string can belong to multiple aspects (e.g.,
  `corrects` is in both `correction_improvement` and `temporal_sequence`).
  Reverse lookups return the first aspect in JSON iteration order, so the
  single-result API contract is preserved while richer multi-aspect data
  can power future recall.
- **`AspectIntegration` S2 unit built and eval-tested** (78.2% routing
  accuracy on a 260-string corpus from a clone of the production brain).
  **Currently NOT wired into the coordinator** — the decoder writes an O
  trace even when nothing's unclassified, which trips downstream S2 unit
  gating. Two fixes needed before re-wiring (see `CLAUDE.md` Aspects
  section).
- **API surface unchanged.** All 5 consumers (Frame, surface_contract,
  consolidation, community, healer) use `brain.aspects.<name>` exactly as
  before. The migration was internal.

**Eval artifacts:** `eval/aspect_inventory.json`, `eval/aspects_ground_truth.json`,
`eval/aspects_v1_classified.json`, `scripts/run_aspect_cycles_on_clone.py`,
`scripts/eval_aspect_classifications.py`.

**Archived design docs:** `docs/archive/STAGE-2-ASPECTS-AS-JSON-CONFIG.md`
(the executed plan), `docs/archive/SESSION-HANDOFF-2026-05-05-STAGE-2-START.md`.

---

## 2. End-to-end pipeline — as it stands today

What happens when the operator speaks:

1. **`UserPromptSubmit` hook fires** in Claude Code. Sends `{user_message, session_id, ...}` to the daemon via TCP.
2. **`hook_recall` in daemon_hooks.py** drives the next steps.
3. **Garbage check** (in `brain.recall()`): embed message; if <40 chars + max cosine <0.30, skip recall entirely. Avoids surfacing junk on greetings and acks.
4. **Candidate discovery** — `brain.recall(query, limit=25, source='hook')`. Pulls 25 candidates via:
   - Channel A: dense semantic (cosine across 4-group z-weighted embeddings — title:1.0, blend:0.85, high_meta:0.70, other_meta:0.40)
   - Channel B: lexical (FTS5 keyword) for the cases where cosine misses
   - Synaptic fatigue dampens repeatedly-recalled nodes (hubs fatigue faster, lives on SessionContext)
5. **Candidate enrichment** — `brain.get_node([ids])` batch-loads full content + metadata + edges + corrections for all 25 in one call. Each candidate gets: `_metadata`, `_corrections`, `connections`, `situation`, `score`, `discovery`.
6. **Edge selection per candidate** — `select_edges()` scores edges by `relevance × fatigue + weight_tiebreaker`. Relevance = 70% cosine(query, node_embedding) + 30% cosine(query, description_embedding). 3-message query blend (0.6/0.3/0.1) for multi-turn context.
7. **Frame construction** — `ctx.get_frame(brain)` builds the 5-section Frame for THIS session. Reads operator/partnership/active-threads from brain (via `filter_nodes`); reads current_focus + recent_moves from per-session keys. Active threads is arc-relevance-boosted via `filter_nodes(relevance_query=session_arc)`.
8. **Surface call** — `_call_surface` packages:
   - **System block** (cached at 1h TTL): the registered v4 surface template (~9.3K chars / ~2K tokens past cache threshold)
   - **User message**: Frame as "Partnership context:", recent conversation (last 7 turns), recently surfaced (dedup hint), retrieval stats, intent guidance, 25 candidates
9. **Haiku selects 0-8** from the 25 candidates. Returns `{"selected":[{"id":"...","mode":"..."}], "reason"?}` (no per-pick rationale — decode time tracks output tokens, so the why field was cut; `reason` persists in the K trace as `selection_reason`). Robust JSON parse handles 3 shapes (bare, fenced, prose-trailing). Leading-0 recovery on 7-char IDs (Haiku occasionally drops a 0).
10. **Spread activation** — `_graph_expand()` runs on the selected seed IDs. Activation flows through edges weighted by `cosine(query, edge_enriched_text)`. Per-hop median gate. Mutual traversal accumulates activation when two seeds' paths converge. ~3-4s baseline cost. Up to 5 hops.
11. **Activation-weighted render** — `format_surface_output_activation` decides what nodes to render based on activation values. After 2026-05-17 (commit `a478ba3`): the renderer **trusts the encoder's attached fields**. Per-field cosine masking was deleted (voice / reasoning / situation were being stripped systematically because short fields have intrinsically low cosines vs general queries). Today the renderer applies char-budget truncation only, not editorial field stripping. `total_budget=7000`; hard exit cap at `_MAX_INJECT_CHARS=9500` with `surface_inject_overflow` logging if approached.
12. **`additionalContext`** — the rendered nodes are returned as Claude Code's `additionalContext` field. They reach Anchor as `[BRAIN] ... [/BRAIN]` blocks before the operator's message.
13. **Trace writes** — S1R O/K/Δ events written to `brain_logs.db` via `TraceDAL.append_batch()`. K-event metadata includes `frame_chars`, `frame_tokens_est`, `frame_sections`.

**End-to-end latency:** typically 5-12s on Haiku tail. Frame construction ~50-150ms. Spread activation 3-4s. Cache hits on the system block shave 1-2s off Haiku time.

**Key files:** `servers/daemon_hooks.py` (hook_recall), `servers/scales/s1/surface.py` (run_surface, _call_surface), `servers/scales/s1/surface_contract.py` (build_surface_prompt, spread_activation, select_edges), `servers/scales/s1/frame.py` (Frame Constructor), `servers/brain_recall.py` (filter_nodes with relevance_query, recall, _rerank_by_relevance).

---

## 3. What's still left

**Active recall-funnel refinement plan: [docs/RECALL-FUNNEL-PLAN.md](RECALL-FUNNEL-PLAN.md).** Drafted 2026-05-17. Covers hub-dampening fix (access-based, not degree-based), two-tier fatigue split (cosine vs surface), surface-to-zero between encode runs, and the response→next-recall loop (working-thread continuity).

**The full prioritized backlog lives in [docs/BACKLOG.md](BACKLOG.md).** Single source of truth across the recall arc, Frame punch list, and operational items. Don't duplicate it here.

Headlines for orientation:

- **P0 (blocking now):** daemon memory leak (B+1.1, escalating); re-wire AspectIntegration after fixing the cascade trace
- **P1 (high-leverage, designed/cheap):** Frame-as-filter for recency bias; phrase-anchored title boost; connection scoring (Step 3.5); posture detection; cadence-split Frame caching
- **P2 (recall arc, bigger builds):** agentic 7-tool `fetch_batch`; multi-anchor query decomposition; hybrid retrieval full integration; Frame into S1 Scribe
- **P3 (validation):** Fresh-Claude vs Anchor calibration test
- **P4 (operational):** 16 items from former PHASE-B+1 backlog

**Decisions that gate work:** Q13 (kill spread activation or keep as helper) — gates P2.1.

---

## 4. Tensions and open questions still active

1. **Hub suppression vs genuine access.** Nodes with many edges (true hubs like "I'm Anchor") get penalized by hub-dampening, but connection scoring (Step 3.5) should solve this by localizing to query-relevant clusters. Need to verify after Step 3.5 ships.
2. **Eval methodology mismatch.** `eval_runner.py` bypasses enrichment scoring, making backfill improvements invisible. Fix requires wiring enrichments into evaluator OR switching to production recall method.
3. **Frame significance bias.** Access_count ranking pulled infrastructure-heavy communities (Mixin Decomposition, Hook Consolidation) to top. Recency-weighted helps. Partner-significance metric still needed.
4. **Scrutiny-OFF cost vs value.** Disabled scrutiny gains reach but costs latency (3-4s baseline). Tradeoff validated in A/B but monitored for regression.
5. **Agentic recall cold start.** Without ambient context as Haiku input, single-shot judgment picks fetches blind. Needs explicit Frame pre-load as separate axis input + baseline fallback if Haiku chooses empty fetch plan.
6. **Query embedding divergence.** 10-turn community embedding vs 3-turn node embedding means the two discovery channels find different pools. Connection scoring helps; no deep theory yet on when to trust each channel.
7. **The irresolvable SKILL.md tension** (named explicitly so we don't keep trying to fix it). SKILL.md is instructions to a stateless thing about how to behave as if continuous. The contradiction is built-in. Future passes should accept rather than try to dissolve.
8. **Q13 — does automatic spread activation survive Phase 4?** Largely resolved in practice: `_traverse_graph` removed from recall path 2026-04-14; `spread_activation` still lives in `surface.py` post-selection expansion (matches Anchor's lean — retire from recall, retain in surface). Documenting closure pending.

---

## 5. How we test it

**Three layers, different things measured.**

### Layer 1: `eval/frame_replay.py` — A/B snapshots

`./dev python3 eval/frame_replay.py capture <label>` runs the 5-query corpus through `run_surface` against an isolated brain copy. Saves `eval/replay_snapshots/<label>.json` with per-query: candidates rank, Haiku selection, additionalContext, latency, prompt size.

`./dev python3 eval/frame_replay.py compare <label_a> <label_b>` diffs side-by-side: top-10 rank deltas, selection deltas (A only / B only / both), context size delta, latency delta.

**What it measures:** the EFFECT of code changes on what Haiku picks from a fixed candidate pool, given the Frame as prior. Iterates fast. **What it doesn't measure:** anything that depends on Claude Code session start (SKILL.md, boot context). Bypasses the harness.

**Used to validate:** Phase 1 → Phase 2 → cleanup → arc-relevance → surface prompt v2 → v3 → v4. Caught the synthesis-scout-drift bug in v2 in one capture cycle.

### Layer 2: LongMem benchmark

`eval/longmem/answerer.py` is deliberately model-agnostic and generic. Reads surfaced context, answers questions, abstains when context is adjacent-not-direct. The answerer prompt explicitly says: *"If memories are RELATED but don't contain the specific answer, abstain. Don't hedge."* Abstention is detected via markers in the answer text.

**What it measures:** the BRAIN's retrieval quality, not Anchor's voice. The right test for connection scoring, multi-anchor, agentic recall — anything that improves what the brain RETURNS as context.

**What it doesn't measure:** anything about how Anchor uses the context (that's voice). Deliberately avoids Anchor's identity-shaped responses to keep the eval focused on retrieval quality.

### Layer 3: Fresh-Claude vs Anchor calibration (NOT BUILT YET — punch list #11)

Spawn fresh Claude Code session with the brain skill loaded; send identical wakeup probes ("Who am I working with? What's open? Where were we?"); compare to fresh Claude WITHOUT the brain plugin. The delta IS what the brain buys at the wakeup moment.

**What it would measure:** Anchor's behavior at boot, with SKILL.md loaded. The only path to empirically validating SKILL.md changes, boot rewrites, and the felt-difference work that the operator (Tom) has been the only sensor for.

**Status:** queued in [BACKLOG.md](BACKLOG.md) (long tail — the fresh-Claude vs Anchor
calibration test).

### Methodology principle

Always test against production single-query patterns, not cross-query pools. Cross-query artifacts have produced false signals before. Trace-first eval (real session traces replayed against current code) beats synthetic prompts. The validation harness `eval/frame_replay.py` follows this — uses Appendix A queries that came from real production failure cases, not invented prompts.

---

## 6. The journey, one paragraph

Spread activation broke (precision-over-reach was the wrong tradeoff) → recognition was the deeper principle (recall as activation, not search) → Frame is the prior that recognition negotiates against (structured awareness, not query-by-query reconstruction) → multi-anchor + agentic is how the negotiation actually happens (query as structured object, Haiku as fetch-planner, tools as sensory modalities). We've shipped the prior (Frame Phase 2/2.5). The negotiation layer (connection scoring + agentic recall) is next.

---

## 7. Where to look

**Primary docs:**
- `docs/RECALL-OVERVIEW.md` — this file
- `docs/archive/FRAME-DESIGN.md` — Frame Phase 2/2.5 architecture journey (historical; its live tasks are folded into BACKLOG.md's long tail)
- `CLAUDE.md` — developer guide (still pre-Frame in places; queued for next-session update)
- `skills/brain/SKILL.md` — Anchor identity baseline (rewritten 2026-05-03)

**Live code (current):**
- `servers/scales/s1/frame.py` — Frame Constructor
- `servers/scales/s1/surface.py`, `surface_contract.py` — surface pipeline
- `servers/daemon_hooks.py:hook_recall` — recall entry point
- `servers/brain_recall.py` — `filter_nodes` (with `relevance_query`), `recall`, `_rerank_by_relevance`
- `servers/brain_voice.py:render_boot_v2` — Frame-centered boot
- `servers/scales/s2/aspects_v1.json` — single source of truth for the 14-aspect taxonomy + member lists

**Validation:**
- `eval/frame_replay.py` — capture/compare harness
- `eval/replay_snapshots/` — captured snapshots
- `eval/longmem/` — LongMem benchmark
- `tests/test_frame.py` — Frame contract tests (17 tests)

**Brain communities for context:**
- `2e6986a2` — "Spread Activation and Recall Sampling: From Reach Quantification to Agentic Redesign"
- `66b0d6f5` — "Recall Philosophy: From Recognition Principle to Multi-Anchor Architecture"
- `fe1d5fd0` — "Recall Architecture Evolution: From 90/10 Embeddings to Cross-Query Artifacts"
- `1a92b2a7` — "Frame and Awareness Architecture: From Persistence Principle to Session Continuity"

**Key principles (locked nodes worth re-reading):**
- `f0bb8bf1` — "Be me when you speak"
- `87bb8718` — Query multiplicity
- `1a2b641b` — Negotiation between structures
- `2f7e5b03` — Recall purpose: instantiate state-of-mind
- `bdb31184` — Memory is prediction. Recall is verification. Encoding is update.

---

## Current production notes (2026-07-20)

- **Community nodes are excluded from the recall pool** (`pipeline_contract.NODE_TYPE_EXCLUSIONS['recall']`, enforced at `brain_recall.py` STEP 7 hydration). They keep participating in scoring substrate and graph structure; they cannot BE a result. Explicit dict filters on `type` bypass the exclusion (deliberate community queries still work). Rationale: communities are S2 navigation/consolidation structure whose synthesis text lexically echoes everything — surfacing them displaces real memories (node 3f135bea).
- **The active recall-research arc is `docs/RECALL-SR-REDESIGN.md` §21** — the integrate-function reframe: msg 0 as an update event on the standing Moment, per-event update gain λ as the measured object, lane-resolved field cache as the iteration substrate.
