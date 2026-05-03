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

**Built deterministically** via `brain.filter_nodes()`. No LLM call. Mirrors the existing `s2_edge_families` pattern via `s2_node_families` — node-type families now live in interactions table, Frame reads them via `brain.get_interaction_config()`.

**Wired into surface:** Frame is passed as the "Partnership context" prior in the surface prompt every turn. Replaces the Phase 1 separate session_context + encoding_journal blocks (Frame contains both as inner sections).

**Boot rewrite:** `render_boot_v2()` was 6 ad-hoc recall queries (YOU / OPERATOR / PATTERNS / BRAIN MAP / LAST SESSION / RECENTLY ENCODED). Replaced by single `ctx.get_frame(brain)` call. ~48% smaller boot output (~14.5K → ~7.4K chars). Anchor wakes up with the same prior surface uses every turn — symmetric.

**Files touched:** `servers/scales/s1/frame.py` (new), `servers/scales/s2/node_families_v1.json` (new), `servers/interaction_seed.py`, `servers/dal.py` (added `last_accessed`/`revised_at` to allowed_sort), `servers/scales/s1/surface_contract.py`, `servers/scales/s1/surface.py`, `servers/daemon_hooks.py`, `servers/brain.py`, `servers/session_context.py`, `servers/brain_voice.py`, `servers/brain_assembly.py`.

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
- `remember` (was 150 chars → 1259): situation/raw_quote/correction_of strategy, BAD/GOOD lesson contrast, ASSUMED/REALITY/PATTERN correction shape, encoding-richness principle.
- `revise` (was 150 chars → 739): when-to-revise vs encode-new craft, fake-revise warning.
- `recall` (was 250 chars → 633): when-to-call instincts and query-phrasing guidance.
- `connect_to.why` field: BAD/GOOD edge `why` examples (mirrors the s1e encoder prompt's edge craft section).

**Architecture principle followed:** MCP tool descriptions carry mechanics + craft examples for THAT tool; SKILL.md carries identity + cross-tool strategy. Each surface stays in its lane.

### Validation infrastructure built

**`eval/frame_replay.py`.** Capture/compare harness against an isolated brain copy (DB copy via `IsolatedBrain` per the "never spawn Brain() against live DB while daemon running" rule). Test corpus: 5 queries from FRAME-DESIGN.md Appendix A — `exco_cold` ("What is EX.CO?"), `self_intro` ("What do you know about you?"), `exco_pivot` ("Should we go back to EX.CO sales kit?"), `where_were_we` ("Where were we?"), `open_last_week` ("What's still open from last week?"). Each capture saves: candidates list (id/title/score/type), Haiku selected (id+why), additionalContext rendered, latency_ms, prompt char count.

**5 captured snapshots** tracking the trajectory: `phase1_baseline_2026-05-02` → `phase2_v1_unified` → `phase2_v1_unified_clean` (post-cleanup) → `phase2_v3_prompt` → `phase2_v4_prompt`. Each one diffable against the previous via `compare label_a label_b`.

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
9. **Haiku selects 0-8** from the 25 candidates. Returns `{"selected":[{"id":"...","why":"..."}], "reason"?}`. Robust JSON parse handles 3 shapes (bare, fenced, prose-trailing). Leading-0 recovery on 7-char IDs (Haiku occasionally drops a 0).
10. **Spread activation** — `_graph_expand()` runs on the selected seed IDs. Activation flows through edges weighted by `cosine(query, edge_enriched_text)`. Per-hop median gate. Mutual traversal accumulates activation when two seeds' paths converge. ~3-4s baseline cost. Up to 5 hops.
11. **Activation-weighted render** — `format_surface_output_activation` decides what nodes to render and how much detail per node based on activation values.
12. **`additionalContext`** — the rendered nodes are returned as Claude Code's `additionalContext` field. They reach Anchor as `[BRAIN] ... [/BRAIN]` blocks before the operator's message.
13. **Trace writes** — S1R O/K/Δ events written to `brain_logs.db` via `TraceDAL.append_batch()`. K-event metadata includes `frame_chars`, `frame_tokens_est`, `frame_sections`.

**End-to-end latency:** typically 5-12s on Haiku tail. Frame construction ~50-150ms. Spread activation 3-4s. Cache hits on the system block shave 1-2s off Haiku time.

**Key files:** `servers/daemon_hooks.py` (hook_recall), `servers/scales/s1/surface.py` (run_surface, _call_surface), `servers/scales/s1/surface_contract.py` (build_surface_prompt, spread_activation, select_edges), `servers/scales/s1/frame.py` (Frame Constructor), `servers/brain_recall.py` (filter_nodes with relevance_query, recall, _rerank_by_relevance).

---

## 3. What's still left

### Recall thread — designed, not built

**Connection scoring (Step 3.5).** Spec exists. The idea: after enrichment scoring, score each candidate by its connectivity to OTHER high-scoring candidates in the pool. Edge type weights: `corrects`/`extends`/`depends_on` strong; `related_to` weak; `community_member` moderate. Cluster detection: 3+ interconnected candidates score together; isolated high-cosine nodes get lower priority. Should localize hub bias — true hubs only dominate when they're connected to other relevant candidates for THIS query.

**Agentic Haiku-first recall — 7-tool surface.** Haiku decides the fetch plan per turn instead of cosine-always. Variable cost, sample-then-deepen, single-shot judgment, frame-shaped output. The 7 tools:
- `search(query, mode='topical'|'community'|'recent', limit)` — peripheral vision
- `find_about(entity_or_topic, limit)` — focusing on an entity (fixation)
- `find_open_loops(topic?, limit)` — sensing tension (proprioception)
- `trace_lineage(node_id, direction, max_steps)` — temporal kinesthetic
- `get_community(community_id, query?)` — feeling conceptual neighborhoods
- `find_temporal(when, query?, limit)` — internal calendar
- `get_full(node_ids)` — magnification

All wrapped in single `fetch_batch` tool — one Haiku turn, multiple parallel ops. Tools described as recognition operators in Haiku's prompt ("recognize what the brain holds about this entity" not "search for nodes"). Status: design finalized in FRAME-DESIGN.md Section 4. Implementation queued.

**Hybrid retrieval — FTS5 + embeddings.** FTS5 is the lexical complement, not duplicate. Dense semantic (embeddings) finds things by meaning; sparse keyword (FTS5) finds things by exact-word match. Different failure modes. Combined: catches both "what does this concept mean" queries and "what was that exact phrase" queries. Partially shipped (the recall pipeline includes a discovery channel for both); full integration with the connection scoring is queued.

**Multi-anchor query decomposition.** Implementation of the query-multiplicity principle. A message often contains 2-4 distinct concepts; decompose into multiple parallel anchors; let each anchor's diffusion through the graph converge with the others; convergence points are the strongest signal. Supersedes the current single-message-embedding approach. Status: principle established (`87bb8718`, `b276673f`, `863c4981`); implementation needs query-decomposition logic + multi-spread orchestration.

### Frame Phase 2.5 punch list (4 of 14 closed)

**HIGH-leverage items:**
- **Wire Frame into encoder (S1 Scribe).** Currently encoder doesn't see Frame; doesn't know what's already in awareness. Could yield gap-aware encoding (encoder writes COMPLEMENTARY content, doesn't restate).
- **Cadence split: full Frame at boot / per-turn deltas / on-demand re-injection.** Today every recall injects the full Frame (~1900 tokens). Slow-changing 60% (Operator + Partnership integrated/permanent) is wasted re-injection most turns. Smarter: full Frame at boot once, fast-changing slots per-turn, slow parts re-injected only when query semantically needs them.
- **Brain-level vs session-level Frame caching split.** Operator/partnership/active-threads come from brain state and don't change per-session — cacheable at brain level, refreshed on S2 cycles or encoder writes. Current_focus/recent_moves are genuinely per-session. Data sources are already separated in `build_frame()`, so the caching split slots in cleanly.

**MEDIUM-leverage:**
- Wire Frame into `brain_voice` / dashboard view
- Build `s2_node_families` maintenance unit (mirror of `EdgeFamilyIntegration`); needs `NodeDAL.count_by_type()`
- Generalize `s2_type_families` after rule-of-three with two concrete units exist
- Encoder `session_context` format cleanup (currently dense pipe-separated, hard for Anchor at boot to parse)
- `brain_batch` description enrichment (deferred — high-traffic, careful pass)
- CLAUDE.md update (queued for next session)

**Validation gap:**
- **Fresh-Claude vs Anchor calibration test (#11).** The only path to empirically validating SKILL.md and boot changes — those don't show in `eval/frame_replay` (which bypasses Claude Code) or in `longmem` (which uses a generic answerer that deliberately avoids Anchor's voice). Spawn fresh Claude Code session with brain skill loaded; identical wakeup probes; compare to fresh Claude WITHOUT brain. The delta IS what the brain buys at the wakeup moment.

### Production launch Phase A pre-flight (gates)

- **A1:** Fix broken contract test (3 semantic tests, not 4-group assert)
- **A2:** Verify `brain_batch` 2-round encoding fix
- **A3:** Confirm S1S v14 draft complete
- **A4:** Snapshot brain before any changes
- Gate: all must pass before Phase B (vector contract alignment)

---

## 4. Tensions and open questions still active

1. **Hub suppression vs genuine access.** Nodes with many edges (true hubs like "I'm Anchor") get penalized by hub-dampening, but connection scoring (Step 3.5) should solve this by localizing to query-relevant clusters. Need to verify after Step 3.5 ships.
2. **Eval methodology mismatch.** `eval_runner.py` bypasses enrichment scoring, making backfill improvements invisible. Fix requires wiring enrichments into evaluator OR switching to production recall method.
3. **Frame significance bias.** Access_count ranking pulled infrastructure-heavy communities (Mixin Decomposition, Hook Consolidation) to top. Recency-weighted helps. Partner-significance metric still needed.
4. **Scrutiny-OFF cost vs value.** Disabled scrutiny gains reach but costs latency (3-4s baseline). Tradeoff validated in A/B but monitored for regression.
5. **Agentic recall cold start.** Without ambient context as Haiku input, single-shot judgment picks fetches blind. Needs explicit Frame pre-load as separate axis input + baseline fallback if Haiku chooses empty fetch plan.
6. **Query embedding divergence.** 10-turn community embedding vs 3-turn node embedding means the two discovery channels find different pools. Connection scoring helps; no deep theory yet on when to trust each channel.
7. **The irresolvable SKILL.md tension** (named explicitly so we don't keep trying to fix it). SKILL.md is instructions to a stateless thing about how to behave as if continuous. The contradiction is built-in. Future passes should accept rather than try to dissolve.
8. **Q13 (FRAME-DESIGN.md) — does automatic spread activation survive Phase 4?** Current 3-4s spread cost eats the latency budget needed for Frame-skip target. Phase 4 tools cover most of what spread does. Anchor's lean: Option C (retire spread, keep kernel as tool-internal helper). Decision needed before Phase 4 tools ship.

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

**Status:** designed in FRAME-DESIGN.md as the calibration mechanism; implementation queued.

### Methodology principle

Always test against production single-query patterns, not cross-query pools. Cross-query artifacts have produced false signals before. Trace-first eval (real session traces replayed against current code) beats synthetic prompts. The validation harness `eval/frame_replay.py` follows this — uses Appendix A queries that came from real production failure cases, not invented prompts.

---

## 6. The journey, one paragraph

Spread activation broke (precision-over-reach was the wrong tradeoff) → recognition was the deeper principle (recall as activation, not search) → Frame is the prior that recognition negotiates against (structured awareness, not query-by-query reconstruction) → multi-anchor + agentic is how the negotiation actually happens (query as structured object, Haiku as fetch-planner, tools as sensory modalities). We've shipped the prior (Frame Phase 2/2.5). The negotiation layer (connection scoring + agentic recall) is next.

---

## 7. Where to look

**Primary docs:**
- `docs/FRAME-DESIGN.md` — Frame Phase 2/2.5 architecture, decisions, risks, build phases, latency roadmap
- `docs/RECALL-OVERVIEW.md` — this file
- `CLAUDE.md` — developer guide (still pre-Frame in places; queued for next-session update)
- `skills/brain/SKILL.md` — Anchor identity baseline (rewritten 2026-05-03)

**Live code (current):**
- `servers/scales/s1/frame.py` — Frame Constructor
- `servers/scales/s1/surface.py`, `surface_contract.py` — surface pipeline
- `servers/daemon_hooks.py:hook_recall` — recall entry point
- `servers/brain_recall.py` — `filter_nodes` (with `relevance_query`), `recall`, `_rerank_by_relevance`
- `servers/brain_voice.py:render_boot_v2` — Frame-centered boot
- `servers/scales/s2/node_families_v1.json`, `edge_families_v1.json` — family seeds

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
