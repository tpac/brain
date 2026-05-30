# Backlog — single source of truth for what's left

**Last updated:** 2026-05-30 (P1–P5 bands code-verified this date — see the ⚠ banner under "P1"; the older "session captures" log below is append-only history, not a status source).
**Supersedes:** scattered references in `RECALL-OVERVIEW.md` §3, `PHASE-B+1-BACKLOG.md` (archived), Frame Phase 2.5 punch list, `BRAIN-CHALLENGES.md` fix sketches.

**For current session state and next priority**: see `docs/SESSION-HANDOFF.md`.

**Session captures (most recent first):**
- **2026-05-30** — **Frozen Corpus eval platform SHIPPED + v24/v7/v4 ACTIVATED.** Two-stage longmem harness: `eval/longmem/build_corpus.py` (encode once, content-addressed by s1e/ingest-surface/s2-cadence/oracle/qids + interaction-overrides; incremental manifest) → `sweep.py` (recall cheaply, many times; ~100× under a full run). Caught + fixed a `brain_batch` leaked-transaction crash in the S2 community encoder — interim guard in `dispatch_write.py`, root fix tracked at **F3 / [docs/WRITE-TXN-ISOLATION-ROOTFIX.md]**. Answerability gate stopped hard-excluding composed/multi-session answers (its single-node keyword-AND gold-scan false-negatives them — a 13-node on-topic item scanned "unanswerable") → sweep now **scores every item** with a `recall_conditional` rate + `ABSTENTION_FAIL` bucket. Also fixed harness `_snapshot_error_count` (queried a non-existent `brain_errors` table → now `debug_log`). 20-item baseline (corpus `a300d2`): **94.4% recall-conditional (17/18)**, 85% raw — the 2 clean misses are `ENCODE_MISS` (encoder wrote 0 nodes on a preference + a rejected-suggestion exchange), recall reads strong. Added `build_corpus --interaction-override` (fetches DORMANT versions from the live daemon); 2-item v24+v7 A/B showed facts-scout v7 0→6 candidates + v24 encoding nodes v22 dropped → **activated s1e v24 + s1_scout_facts v7 + s1_scout_quote v4** (`d0fea6d`, `47f7018`; seeds synced, `--check` clean). Commits: `18ac427`, `beb38ff`, `d0fea6d`, `47f7018`, `9243600` (CLAUDE.md). Refs: [docs/EVAL-PLATFORM.md], [docs/WRITE-TXN-ISOLATION-ROOTFIX.md]. Residual encode-coverage gap (encoder filters low-stake exchanges even when they hold a future answer) is a design-philosophy call, deliberately not patched.
- **2026-05-29** — S2 idle-gating + test-suite redundancy cleanup (separate thread from the encoder/episodic-refs work). S2 **Community** + **Consolidation** were doing full O(graph) scans every ~15 min 24/7 (87% / ~88% zero-work, never converging); both now gate on graph-change + skip idle cycles (Consolidation also activates its dead incremental-scan path, stamping the cutoff only after the encoder completes). AspectIntegration + Healer audited clean. Test-redundancy hunt over ~70 files: 13 inert/duplicate tests removed-or-consolidated + 1 stale RED test (`test_same_relation_strengthens`) fixed to the Stage 1B contract. 5 commits (`47ed457`, `ba89bb9`, `e37c8e5`, `694b339`, `5ab2dde`), not pushed; daemon picks up gates on next restart. **Full handoff + remaining test-cleanup map (buckets A–E): [docs/S2-GATING-AND-TEST-CLEANUP-HANDOFF.md](S2-GATING-AND-TEST-CLEANUP-HANDOFF.md).** Community Phase 2 (delta-decode) parked → memory `project_s2_community_phase2_parked`; dampening cluster (4 red tests) parked with recall.
- **2026-05-26** — Phase B Step 7 (co_anchored auto-edge) + Layer 1 validator + sync-prompts active-version fix SHIPPED. v22 active in production at 2026-05-26T00:46:54Z (commit `c144ddf`). 50-cell longmem real-test result: v22 23/25 (92%) vs v19 22/25 (88%); v22 100% source_refs coverage uniformly; +1 axis win on info_extraction where v19 zero-encoded `cc539528`. `eval/encoder_eval/` infrastructure shipped — 6 substrate-aware probes, parallel + stratified runner, scout-override capability via `apply_interaction_override` pattern. v23/v24 + scout v6/v7 + quote v3/v4 registered DORMANT after two iteration cycles (initial draft → probe-driven refinement). Targeted A/B + follow-up evals (15 cells, 10 distinct items): v24 candidate 9/10 — qualitative wins on abstention discipline (`gpt4_93159ced_abs` v22+v19 ✗ → v24 ✓) + hobby cohort retention (`5025383b` v22-this-run ✗ → v24 ✓), one regression direction on temporal hedging (`gpt4_d31cdae3` v22 ✓ → v24 ✗). Methodological finding: c2ac3c61 wasn't deterministic — v22 succeeded on re-run, confirming N=1 stochasticity floor. Eval-decision call open. 5 commits on main. 165 tests pass.
- **2026-05-25 (mid-day)** — Schema v29 trace_id INTEGER→hex migration + Phase B Steps 0+0.5+1+2+3+4 SHIPPED. Substrate complete; encoder behavior unchanged until v22 prompt drafted. 60,634 trace_events + 12,933 trace_embeddings migrated cleanly via auto-backup framework. MCP schemas, brain.remember/revise, dispatch validation all wired for source_refs. Quality contract v3 (36 dims, Group 9 example_authoring). All 7 §7.6 examples placeholder-compliant. Reviewer pass: F1 critical (revise REPLACE not APPEND) + F2/F4/F8 fixed; F3/F6/F7 deferred. 145 v29-touching tests pass. Next: Step 6 prompt v22 + Step 8 3-way eval (v22 vs v21 vs v19) gate before production flip.
- **2026-05-25 (overnight)** — Encoder quality contract v1 (32 dims) + s1e v20 SHIPPED. Fixed 4 canonical example bugs (Example 1 D31, Example 4 D23, Example 5 D7, revise ghi789 keywords drop) + appended §7.6 wave-1 block with 6 new examples (Anchor self-reference triad A6+A7+A4, correction-with-affect A2, trust formation A3, methodology/principle split B1). Built encoder eval platform at `eval/agent_introspect/encoder_contract_eval.py`. A/B-validated v20 vs v19 on 4 corpuses; D31 fix landed cleanly on structurally-different clean retest (C4). v20 active 2026-05-25 03:12 UTC.
- **2026-05-24 (late)** — Block 1 substrate cleanup SHIPPED (4 commits on top of Phase A). MCP `get_trace`/`get_traces`, `query_traces` session_id + session_ids, killed `encode_cluster` + `auto_connect` (related_to pollution source), Schema v28 dropping `nodes.keywords` + auto-extractor. Plus 327-node encoder-quality scan; see `docs/ENCODER-QUALITY-FINDINGS.md`.
- **2026-05-24 (mid)** — Time-window architecture: `iso_now`/`iso_cutoff` with `at=` parameter (conversation-anchored cutoffs); contract test banning direct `datetime.now()` in S1/S2. 12-site SQL `datetime()` fix.
- **2026-05-24 (early)** — Phase A substrate SHIPPED end-to-end (10 commits). Schema v27, DAL methods, identity stamping, embed worker, historical identity migration.
- 2026-05-23 — eval system v2 + episodic-references design complete; §15.1 closed via biology research; decision 19 revised
- 2026-05-18 — bg_writer migration + recall latency diagnosis (next-session packages A + B BOTH SHIPPED, see commits below)
- 2026-05-15 — surface v6 redesign + full A/B comparison + tooling
  - [docs/AGENT-PROBES.md](AGENT-PROBES.md) — probe family
- 2026-05-11 — temporal arc + agent introspection (ARCHIVED)
  - [docs/TEMPORAL-ARCHITECTURE.md](TEMPORAL-ARCHITECTURE.md) — full temporal handling reference

---

## ⚡ Top of queue — what's next

**Phase B substrate SHIPPED end-to-end. v22 active in production since 2026-05-26T00:46:54Z.** Step 7 (co_anchored auto-edge), Layer 1 validator (hex format soft warn + sparsity >5), and sync-prompts active-version fix all shipped. **s1e v24 + scout-facts v7 + scout-quote v4 ACTIVATED in production 2026-05-30** (commits `d0fea6d`, `47f7018`); v23 + scout v6 + quote v3 remain dormant.

**Next-session priority order**:

1. ~~v24 + scout v7 + quote v4 eval-decision~~ — **✅ RESOLVED 2026-05-30: ACTIVATED.** s1e v24 + s1_scout_facts v7 + s1_scout_quote v4 live in production (`d0fea6d`, `47f7018`; seeds synced). Eval basis: 9/10 targeted + 23/25 50-cell + a 2-item encode A/B (`build_corpus --interaction-override`) showing facts-scout v7 0→6 candidates + v24 encoding nodes v22 dropped. Residual, out of scope by design: the encoder filters low-stake / rejected-suggestion exchanges even when they hold a future question's answer (`ceb54acb`) — a stake-judgment design call, not a v24/v7 gap.

2. **Render expansion at SURFACE_FORMAT** (~0.5-1 day, pending). The recall-side joint reactivation read shape — when a source-anchored node surfaces, expand its source_refs inline. Designed in `docs/EPISODIC-REFERENCES.md §8`; not built. Depends on enough v22-encoded nodes in production to measure the surface impact (start accumulating now). **[Verified 2026-05-30: the code path is entirely unbuilt — zero `source_ref` handling in `surface_contract.py` render code. This is ground-up implementation, NOT "almost done, waiting for data." The v24 accumulation only gates *measuring* impact after the code exists.]**

3. **source_summary parallel-pathway recall scoring** (~0.5 day, pending). `docs/EPISODIC-REFERENCES.md §9.5` + decision 22. Add `source_summary` cohort to recall scoring as `max(legacy_weighted_sum, source_summary_score)`. Backwards compat by design.

4. **S2Healer source_refs cleanup** (`docs/EPISODIC-REFERENCES.md §10.6`, pending). Periodic scan for invalid trace_ids; archive orphan `co_anchored` edges when no shared trace remains.

5. **Path A ground-truth authoring** (~1.75h Tom-time, pending). 7 conversation templates scaffolded at `eval/ground_truth/` covering 5 strata (2 identity-bearing + 2 partnership voice + 1 technical correction + 1 methodology + 1 temporal). Each file has fillable YAML for ideal-node authoring. Once filled: targeted eval against ground truth (structural delta) joins the longmem oracle path (recall delta).

6. **Phase B+ quote_fidelity substring validation** (~0.5 day, deferred) — bigger fix than the identical-strings check that shipped commit `d3f6307`. Requires threading conversation context to `brain_remember.remember()` so `user_raw_quote` can be substring-matched against user_messages window and `anchor_raw_quote` against agent_messages window.

7. **S2Healer stale-node extension** (~30min) — aspect-resolved detection for status-as-fact / plan-as-executed nodes.

### Reviewer follow-ups (deferred from 2026-05-25 Phase B review)

| ID | What | Effort | Status |
|---|---|---|---|
| **F3** | GraphDAL methods manage transactions inconsistently inside/around brain_batch. **Two symptoms, same root:** (1) commit inside a batch → breaks all-or-nothing rollback; (2) leave a deferred auto-BEGIN open → next `brain_batch` `BEGIN IMMEDIATE` throws *"cannot start a transaction within a transaction"* (reproduced 2026-05-29 in the S2 community multi-pass flush via `build_corpus.py`). Interim guard SHIPPED ([dispatch_write.py:498](../servers/dispatch_write.py:498)) — `brain_batch` flushes a stale txn + logs `brain_batch_stale_txn`. Root-cause analysis + fix options (A: `_batch_mode`-aware GraphDAL; B: `isolation_level=None`): **[docs/WRITE-TXN-ISOLATION-ROOTFIX.md](WRITE-TXN-ISOLATION-ROOTFIX.md)**. | ~1 day | open (guard interim) |
| **F6** | Optional hex-format regex warning in `_validate_source_refs` | ~15 min | **shipped 2026-05-26** (commit `07ab3f1` Layer 1 validator) |
| **F7** | Move `_SOURCE_REFS_SCHEMA` from `brain_mcp.py` to `contract.py` under new `JOIN_TABLE_FIELDS` category (parallel to STRUCTURAL / PROMOTED). Makes contract-sync test implicitly cover field registration. | ~45 min | open |

### v24 experimentation thread (active)

Surfaced from forensic analysis of c2ac3c61 (multi_session precision-refinement) in 50-cell longmem run:

| Thread | Status |
|---|---|
| s1e v24 DORMANT — multi-ref anchoring sentence in §7.4 | registered, awaiting activation decision |
| s1_scout_facts v7 DORMANT — supersession-scope clarifier + Cap ranking refinement + Example 4 tightening + NEW Example 5 (parallel-entity + same-axis refinement, languages-domain) | registered, awaiting activation decision |
| s1_scout_quote v4 DORMANT — Skip-list addition for routine factual claims (facts-scout territory boundary) + mixed-content handling | registered, awaiting activation decision |
| Eval-decision call (option B: re-run gpt4_d31cdae3 ×3) | pending |
| Diff files preserved at `/tmp/{s1e_v24,s1_scout_facts_v7,s1_scout_quote_v4}_proposed.txt` and `/tmp/{*}_pre.txt` | gitignored — for inspection |

**Key methodological finding from this session**: **LLM encoders are stochastic at N=1.** c2ac3c61 failed v22 in the 50-cell run but succeeded on re-run with the same prompt. 5025383b succeeded in the 50-cell run but failed v22 on re-run. Future eval design should account for this — multi-sample per cell (N≥3) is needed to distinguish deterministic failure modes from tail outcomes.

Detailed scope per thread: see SESSION-HANDOFF.md.

---

## Open questions surfaced 2026-05-25 (post-v20 ship)

These need judgment calls before deeper work. Captured here so they don't get lost between sessions.

### Q1 — B1's generalization range
The clean C4 retest showed B1's mechanism-vs-principle teaching landed on a structurally different corpus, but N=1 corpus for "different shape." Does B1 land across the actual range of methodology-shaped conversations Tom and Anchor encounter, or only the two corpus shapes we've tested? **No way to know without production exposure.** Schedule: rerun A/B against new corpus types post-v20 (~1 week).

### Q2 — Phase B prompt rule vs example reliance
When source_refs writes ship, do we teach via:
- (a) NEW canonical examples demonstrating source_refs alongside the §7.6 wave-1 that already have them
- (b) An explicit prompt rule paragraph (the §7.4 "Anchoring nodes in the substrate" prose from EPISODIC-REFERENCES.md)
- (c) Both

Evaluator findings on Era E (D7 voice symmetry, 0/6 on identity-bearing nodes despite the rule being in v16+) suggest **examples alone are insufficient for D5/D7-level discipline.** Suggests prompt rules earn their place. Going with (c).

### Q3 — §7.6 vs canonical hierarchy in Sonnet attention
v20.1 §7.6 examples include source_refs; canonical body doesn't. A/B outputs showed Sonnet didn't write source_refs (correctly — schema doesn't accept). But this signals canonical pattern dominates §7.6 attention-wise. Should we **move §7.6 ABOVE canonical** in the prompt structure? Or keep current order? Currently §7.6 lives after canonical (line 983 vs 869). Architectural choice — defer until Phase B forces a prompt restructure anyway.

### Q4 — D22 axiom-layer carveout
Identity-bearing axioms (A7, A4 shape) naturally produce monochromatic edges (all in identity_bearing-adjacent aspects). The contract's D22 marks this as "degraded" but the encoder evaluator surfaced it as a contract gap — the rule needs an explicit carveout, OR D22 needs a new CR12 ("axiom-layer monochromatism is by design"). Defer to contract v2 alongside wave 2 examples.

### Q5 — §7.6 wave 2 priority vs Phase B priority
Both are next-session candidates. Phase B unlocks substrate value (source_refs become functional); wave 2 fills contract dimension gaps (D11 revise audit, D24 multi-aspect-pair). **Instinct: Phase B first** — substrate value compounds, wave 2 examples can use source_refs once they're functional.

### Q6 — Speaker misattribution rate in production
A/B sample showed 1/3 corpuses had the C2 shape (33%). Real production rate unknown until v20 runs for ~1 week and we measure (via the new `voice_fidelity_identical_strings` error log entries). Could be much lower — C2 had a specific operator-asks-question + anchor-articulates-principle shape that triggered it. **Schedule measurement: query brain_logs.db.brain_errors for `voice_fidelity_identical_strings` after 1 week of v20.**

### Q7 — Source_refs silent-drop pragmatics
§7.6 examples have `source_refs: [...]` scaffolding. When Sonnet writes a node matching the pattern in production today, the source_refs get silently dropped (schema doesn't accept). Should the dispatcher add a runtime warning log entry "source_refs received but write path not active until Phase B," OR is silent-drop fine until Phase B ships? **Pure pragmatics call.** Silent-drop avoids alarm-fatigue on every encoding cycle until Phase B; warning would surface premature scaffolding noise. Going silent until Phase B unless production behavior changes our read.

---

## Contract refinements identified by evaluator (10 items, defer to v2)

After 1-2 weeks of v20 production exposure, refine the contract:

1. **D22 axiom-layer carveout** — explicit rule for identity_bearing monochromatism
2. **D1 title length cap clarification** — pick 60c vs 80c, operationalize
3. **CR4 extension to novel RELATIONS** — currently covers types only; extend to verbs
4. **D24 multi-aspect verb list** — operationalize the `note` field as scoring discipline
5. **D25 vs D26 trade-off explicit rule** — when verbatim quote field is set, must its source turn be in source_refs?
6. **D31 lock-worthy N=1 criterion** — operator-named recurrence ("every time", "always") counts as multi-instance evidence
7. **D23 correction_improvement aspect strictness** — type=correction implies at least one correction_improvement aspect edge
8. **D18 conversation-time vs wall-clock check** — explicit format requirement for event_time
9. **D7 scope clarification** — technical pattern-naming reframes also earn anchor_raw_quote
10. **D8 type-aspect coherence with reasoning** — flag type-in-aspect-X-while-reasoning-invokes-aspect-Y

---

## Episodic-references execution status (re-sliced 2026-05-25)

**Phase A — substrate** ✅ **SHIPPED 2026-05-24 (early)** (10 commits). Schema v27, DAL methods, identity stamping, embed worker, historical identity migration.

**Block 1 — substrate cleanup** ✅ **SHIPPED 2026-05-24 (late)** (4 commits, see commits `24e83bc`, `c015d1b`, `fea0fef`, `8d41c8c`).

**Encoder quality contract + canonical fixes + §7.6 wave-1** ✅ **SHIPPED 2026-05-25 (overnight)** — s1e v20 registered + synced. Contract at `servers/scales/s1/quality_contract.py`. Examples at `servers/scales/s1/examples/`. Eval platform at `eval/agent_introspect/encoder_contract_eval.py`.

**Schema v29 trace_id hex migration** ✅ **SHIPPED 2026-05-25 (mid-day)** — `trace_events.id`, `trace_embeddings.trace_id`, `node_source_refs.trace_id` migrated INTEGER → TEXT (8-char hex). Auto-backup `brain.db.v28.bak` exists. DAL coercion removed (reject ints loudly per reviewer F2). MCP `get_trace`/`get_traces` string-typed.

**Quality contract v3 (Group 9 example_authoring)** ✅ **SHIPPED 2026-05-25** — 36 dims, 12 CR. D33 placeholder_syntax + D34 ref_internal_consistency + D35 voice_annotation_coverage (mechanical) + D36 turn_node_divergence (LLM-judged). All 7 §7.6 examples placeholder-compliant.

**Encoder source_refs write path** ✅ **SHIPPED 2026-05-25 (mid-day)** — MCP schemas declare `source_refs: array[string]` on remember/remember_batch/revise_batch (brain_batch inherits). `brain.remember()` accepts kwarg + persists via `GraphDAL.add_source_refs`. `brain.revise()` accepts kwarg + persists via `GraphDAL.replace_source_refs` (REPLACE semantics per unified 2-class revise contract id:`995ffeb1`). Dispatch validates list-of-strings + sparseness warns via `brain._log_warning`. `tests/test_remember_source_refs.py` covers the integration path. **Encoder prompt teaching (Step 6) still pending.**

**Render expansion at SURFACE_FORMAT** — pending (depends on Phase B substrate, which is now done).

**§7.6 wave 2 + 3** — pending. Wave 2 = shape diversity (~4-5 examples including brain_batch mix). Wave 3 = domain breadth (math/poetry/psychology/research, parallel-agent dispatched).

**Recall + eval block** — Phase B onward. `source_summary` parallel-pathway scoring, `co_anchored` writes at encode + S2Healer cleanup, weight-tuning eval (§13.6), quality_probe + source_fidelity_probe runs. Now unblocked by the MCP trace API.

---

## Identity-architecture gap-to-function items (added 2026-05-24)

From the competitive-landscape research (see `docs/IDENTITY-RESEARCH-2026-05-24.md` for the full write-up). The substrate exists; these are concrete missing functions on top of it. Listed by effort, not priority — each is independent.

| Gap | Function we can't perform | Effort |
|---|---|---|
| **Identity-eval scaffolding** | Drift detection across sessions, non-contradiction check, persistence-through-change measurement, post-model-upgrade identity verification | ~1-2 days — borrow Agent Identity Evals (arxiv:2507.17257) shape: ~30 self-reference questions, run pre/post change, judge for stylistic consistency + semantic similarity |
| **Partner-minting flow** | "Hi I'm Alice" → new-partner recognition + binding; multi-partner-over-time arc | ~0.5 day — minting writes a `partner` node, refreshes `brain.operator_name`, identity stamp picks up the new value at next trace write |
| **Identity-filter query** | "Show me everything Tom said about X"; speaker-filter in recall | ~0.5 day — extend `query_traces` with `human_identity` param; promote metadata field to indexed column if perf demands |
| **Self-narrative generation** | "Tell me about yourself" → coherent autobiography across sessions; onboarding pitch for new partner | ~1 day — Sonnet over curated subset (locked identity nodes + recent partnership traces); render-time, not encode-time |
| **Damage resilience** | Survive partial brain corruption; redundancy-backed identity (quorum of env + locked partner node + recent traces) | ~1 day — formalize identity-source quorum, detect mismatch loudly via the existing `_maybe_warn_identity_unset` write-boundary signal |
| **Plug-and-play install** | New user can install without dev setup | Larger arc — `pyproject.toml`, multi-platform daemon adapter, deferred until there's a real second user |
| **Multi-tenant SaaS** | Multiple operators on one daemon with isolation | Larger arc — auth + isolation + architectural shift; only when warranted by an actual second-user case |

Plus measurement scaffolds unique to our substrate (require no new architecture, just eval code):

- **Identity-neighborhood stability** — centroid of all "Tom said..." trace embeddings tracked over months; centroid drift signals identity-token decay
- **Source attribution accuracy** — after render-expansion ships: pick a recalled node, ask "where did this come from?", judge against encoded source_refs (cryptomnesia analog)
- **Engram cohort recall** — pick a trace, find all co_anchored nodes, run a recall query semantically near that trace, measure how many surface together
- **Bidirectional partnership impact** — the target function. Hardest to measure, most important. A/B against pre-Anchor task baselines.

Full measurement detail: `docs/IDENTITY-RESEARCH-2026-05-24.md` Part 4.

---

## Borrowing items from competitor library deep-dive (added 2026-05-24, Part 7)

Three concrete techniques from other systems that translate cleanly into Anchor's architecture. Each is small (~hours, not days). Listed by leverage.

| Borrow | Source | What it does | Effort | Leverage |
|---|---|---|---|---|
| **A.U.D.N. dedup decision** (ADD / UPDATE / DELETE / NOOP as LLM tool calls, gated by InformationContent) | Mem0 (arxiv:2504.19413) | Replaces ad-hoc consolidation prompts with a principled four-tool LLM decision. Already aligned with our `similar_to`/`supersedes` aspect edges. | ~0.5 day in S2 Consolidation encoder | High — turns a heuristic into a structured judgment; works without architectural change |
| **Node specificity weighting** (`s_i = |P_i|^-1` as IDF surrogate for spread-activation seeds) | HippoRAG (arxiv:2405.14831) | Rare nodes get more probability mass; high-degree hub nodes don't dominate. Addresses the "93% never recalled, hubs dominate" finding (node 0591813f). | ~0.5 day in spread-activation kernel | High — directly addresses an known recall problem we already measured |
| **Nucleus expansion** (after semantic nucleus match, expand ±N adjacent trace events from same session by graph adjacency) | MemMachine (arxiv:2604.04853) | Recovers context that spans turn boundaries when only one turn is embedding-similar. Natural fit for source_refs render expansion. | ~0.5 day in render path / surface_contract | Medium-high — strengthens joint reactivation (§8) without changing the encode side |

**Plus revisitable ideas (not immediate, worth tracking)**:

- **A-MEM's K/G/X mutation on link** — when a new node links to an existing one, update the existing node's structured fields. We have Healer for missing-field fill; don't mutate filled fields. The §16.1 labile-reconsolidation direction matches this.
- **Generative Agents' explicit `importance` field (LLM-rated 1-10)** — we approximate via aspect classification + locking; an explicit numeric field could complement.
- **Letta's `sleep-time agent`** — first-class concept for the same role our S2 maintenance + S1 Scribe play. Their naming is cleaner; our split across S1 Scribe and S2 Coordinator is functionally similar.
- **Letta's shared memory blocks across agents** — directly applicable when we get a second partner or a second agent personality (sub-agent flow).
- **MIRIX's procedural/resource memory split** — we don't have procedural memory (§16.5 future direction). When we do, the split into "how to do things" vs "what was looked at" is worth borrowing.

Full technical detail per library: `docs/IDENTITY-RESEARCH-2026-05-24.md` Part 7.

---

## Open questions to address over time (from 2026-05-24 research close-out)

Eight questions that emerged from the synthesis — not blocking next session's work, but worth keeping visible until each has a measured answer. Full context in `docs/IDENTITY-RESEARCH-2026-05-24.md` Part 8.

| Q | Question | Why it matters | Rough effort |
|---|---|---|---|
| **Q1** | What's the actual extraction-loss bound? (Node-alone vs trace-alone vs node + nucleus-expanded) | Tests whether our extraction earns its keep against MemMachine's verbatim-keeping (93.0% LongMemEvalS). The architectural gut check. | ~1 day for a 20-node probe |
| **Q2** | How often does source attribution actually fail? (Cryptomnesia rate) | Johnson-Hashtroudi-Lindsay's source-monitoring is uncited anywhere in LLM memory research; we have the substrate to measure it. | ~0.5 day, 50-node probe |
| **Q3** | Does identity survive a model upgrade? (Claude 4.7 → 4.8) | The killer test for the concrete-token biology-grounding claim (decision 19). | ~0.5 day to baseline, re-run on next model swap |
| **Q4** | Do identity-load-bearing nodes survive S2 consolidation? | Aspect taxonomy says they should be protected; untested. | ~0.5 day probe + S2 archive log walk |
| **Q5** | Is the spreading-activation kernel earning its complexity vs PPR + node specificity? | If PPR ties or beats, retire the kernel complexity. Decisive A/B. | ~1 day |
| **Q6** | What's the right labile-reconsolidation design? (§16.1 named but unspec'd) | Recall opens an update window in biology; our graph drifts toward original framings without this. | Design conversation; then ~1-2 day build |
| **Q7** | Does aspect-taxonomy dual-role tension cause real bugs? (Structural routing vs semantic classification) | Determines whether to split aspects into two taxonomies. | Surface-via-observation, not a probe |
| **Q8** | When the first non-Tom partner appears, what breaks? | Per-utterance binding + render reconstructive frame are designed for this; never exercised. | Synthetic Alice-session test, ~0.5 day |

**The discipline lesson** (also captured as a memory): when a substrate change ships specifically to enable a measurement, the measurement gets a task on the next session's plan automatically. Don't let substrate sit un-measured. The architecture isn't real until the eval runs.

---

## ✅ Completed since 2026-05-18

| What | Commit | When |
|---|---|---|
| Package A — Signal producer cleanup (collapse signals phase, delete reminders/encoding_gap/hook_errors/etc.) | `02f5c32` | 2026-05-19 |
| Package B — `db_maintenance.py` module (5-min checkpoint, 30-min optimize, integrity at boot, pragmas helper) | `9ce056b` | 2026-05-19 |
| Daemon: host-suspend detection (macOS sleep) + Anthropic timeout caps | `47c5907` | 2026-05-19 |
| Daemon health: BRAIN_DEV_MODE opt-out + 20s ping threshold | `6a4fe6e` | 2026-05-19 |
| brain_batch: single transaction per batch, rollback on failure | `a0434c2` | 2026-05-19 |
| bg_writer: empty-drain ticks refresh last_drain_at — kill false stalls | `0a91b43` | 2026-05-19 |
| CLAUDE.md: write topology section refresh | `b10260b` | 2026-05-19 |
| Runner: per-round diagnostic stats (ttft + per-round tokens/cache) | `a28fbc3` | 2026-05-23 |
| Eval system v2: judge reasoning + comparison enum + variance + 3 agent_introspect probes | `0b1115e` | 2026-05-23 |
| **Schema v27** — `node_source_refs` + `trace_embeddings` tables, composite index | `9015636` | 2026-05-24 |
| **DAL methods** — TraceDAL embeddings + GraphDAL source_refs primitives | `8a52164` | 2026-05-24 |
| **Identity stamping** wired at trace write (TraceDAL.set_identity + _stamp_identity) | `75075eb` | 2026-05-24 |
| **Identity activated end-to-end** — brain-env sources user config; dispatch decodes JSON-string metadata (also fixes pre-existing double-encode bug) | `65bf483` | 2026-05-24 |
| **Embed worker** — pull-reconciliation trace embedding phase, concrete-identity render | `7b5b845` | 2026-05-24 |
| **Worker review fixes** — 30-day window (architectural correctness), composite index, observability, defensive double-decode | `669ecee` | 2026-05-24 |
| **Migration**: backfill identity on all 57,546 historical trace_events; clean 22,416 double-encoded legacy rows | `5cff407` | 2026-05-24 |
| **Loud-by-default** — embedder-not-ready / vector-count-mismatch silent paths surfaced | `4288ec8` | 2026-05-24 |
| **Identity-unset signal at write boundary** — TraceDAL._maybe_warn_identity_unset (replaces boot-only check) | `987587f` | 2026-05-24 |
| **Substrate cleanups** — _decode_metadata rolled out to all TraceDAL readers, point-lookup API (`brain.get_trace`/`get_traces`), dead test removed | `d68bddc` | 2026-05-24 |
| **SQL datetime() trip-hazard** — 12 broken time-window queries fixed via `iso_cutoff` helper | `255b9de` | 2026-05-24 |
| **Time-window architecture** — `iso_now`/`iso_cutoff` with `at=` for conversation anchoring; contract test bans direct `datetime.now()` in S1/S2 | `3dd37d4` | 2026-05-24 |
| **query_traces session_id fix** — singular session_id authoritative, ignores hours, loud on empty | `24e83bc` | 2026-05-24 |
| **MCP get_trace + get_traces + auto_connect kill** — wired MCP tools (was decision 23, unstarted); deleted encode_cluster (dead code, 121 lines, 0 callers); removed auto_connect from remember_batch (source of related_to pollution) | `c015d1b` | 2026-05-24 |
| **query_traces cross-session** — `session_ids: List[str]` plural authoritative, mutually exclusive with singular | `fea0fef` | 2026-05-24 |
| **Schema v28 — keywords kill** — drop `nodes.keywords` column + rebuild `nodes_fts` without keywords + delete `_extract_keywords`/`enrich_keywords`. Auto-extractor produced near-duplicate tokenizer noise; porter stemming on title+content is cleaner | `8d41c8c` | 2026-05-24 |
| **Encoder prompt s1e v19** — example cleanup (remove `auto_connect: true` from canonical remember_batch example; no functional change) | (DB-only registration) | 2026-05-24 |
| **S2 community prompt v17** — same example cleanup + remove "`auto_connect: false` always" rule | (DB-only registration) | 2026-05-24 |

Phase A of episodic references is fully shipped: substrate live, identity stamped on every historical trace, embed worker auto-populating.
Block 1 substrate cleanup also shipped: MCP trace API live, related_to pollution source closed, keywords column retired, 327-node encoder-quality scan documented in `docs/ENCODER-QUALITY-FINDINGS.md`.

---

## ⚡ Block 1 follow-ups (deferred from 2026-05-24 late)

### Taxonomy lockdown → v19 rubric
8 emerging-quality clusters + 60+ named qualities ready in `docs/ENCODER-QUALITY-FINDINGS.md`. Collaborative session with Tom to prune/merge/rename, lock the rubric axes, decide preserve-vs-fix on each. ~1-2h. **Gates v19 examples authoring (§7.6 in EPISODIC-REFERENCES.md).** This is the highest-leverage next-session item.

### Conversation-time backdating — consumer wiring + recall feature
The helper exists (`servers/clock.py:conversation_now(at=...)`); the consumers (scouts, encoder, recall) still call `iso_now()`/`iso_cutoff()` without `at=`. Plus a new `recall(query, as_of=...)` parameter for eval/replay backdating. Strategic for evals. ~3-4h once scoped. Tom called this "very important for Evals" 2026-05-24.

### Wider quote-fidelity audit (200 nodes)
Scale the 50-node probe (`/tmp/encoder-scan/probe_quote_fidelity.py`) to 200 nodes. Hand-classify suspicious cases into {paraphrase / cross-session / fabricated / pre-trace}. Output: confident drift rate informing encode-time validation rule. ~1h.

### Reclassify scheduling check
`servers/scales/s2/reclassify.py` exists for legacy `related/related_to`-with-descriptions cleanup. Verify it's wired in the S2 coordinator's unit list and run once against the corpus. ~30min.

### Empty-description `related/related_to` archive sweep
Pre-2026-05-24 auto_connect accumulated empty-description `related_to` edges. Reclassify can't fix these (no description to read). One-off archive script — `archive_dangling_edges` pattern. ~30min after Reclassify verified.

### keywords API surface cleanup (low priority, non-breaking)
After schema v28 dropped the column, the MCP/CLI/seed_pack surfaces still advertise a `keywords` parameter (silently ignored by remember()). Clean in a follow-up pass: drop from MCP schemas, CLI flags, seed_pack node dicts. ~30min total.

### brain_dashboard.db write removal
The daemon-down INSERT into hook_log uses `datetime('now')` — marked `# sql-datetime-ok` as mid-deprecation. Per existing brain memory: `brain_dashboard.db deprecation: stop writing from log_hook_output()`. Full removal when the dashboard's deprecation actually lands.

### Contract-test line-pin cleanup
`tests/test_time_window_contract.py` has a `BRAIN_MCP_EXPECTED` dict pinning the grandfathered datetime-now site by line number. Now redundant — the inline marker covers it. Remove the dict + standardize on marker-only.

### 29 pre-existing test failures from session_context signature drift
From May 2 commit `1cdb2b8` (Frame Phase 2.5 — session_context leak fix changed `_save_session_context` signature; scout_muster/trace_system/s1_data_assembly tests still use stale signatures). Not architecture work — pure test maintenance. Address as separate cleanup pass.

### `get_node_lineage(node_id)` — proposed encoder read API
Agent 3 from the quality scan named this wish: single call returning `{creation_chain, revision_chains, related_traces}` for a node. Hold as a candidate when designing the encoder's read surface (alongside `get_traces`/`get_trace` which shipped).

---

## ⚡ Still-open items (older sessions)

### Future — `surface_haiku` 7.5s warm floor
Single Anthropic API call is the architectural floor on hook_recall latency. Options to investigate (no commitment): intent classifier to skip surface for simple queries; async surface (background Haiku while rendering recall); smaller/local model. Larger arc, separate session.

### Future — Auto-restart hung-daemon handling
`47c5907` capped Anthropic timeouts which addresses one trigger. Force-kill-then-respawn behavior for "process exists but not responding" still untouched.

### Future — Historical co_accessed trim
Pre-Phase-5 `co_accessed` edges still pollute the graph (`integrity_audit.py` flags this). Cleanup task documented as §16.8 in EPISODIC-REFERENCES.md; can run any time after episodic-refs ships.

---

## ⚡ Open items from 2026-05-11 temporal session

### Generic kv field promotion in render
Today the render hard-codes `event_time` as a promoted structured line ([surface_contract.py `_event_time_line()`](../servers/scales/s1/surface_contract.py)). Future generalization: **query-aware kv field promotion** — a temporal query promotes `event_time` / `created_at`; a "what did X say" query promotes `user_raw_quote`; a "why" promotes `reasoning`. Generalize via the existing `field_activation` scoring (the "cousin filtering" mechanism Tom referenced) so any kv field can promote when query-relevant. Tom's note: *"we have cousin filtering fields i think"*.

### UTC-internal clock refactor
Currently `brain.now()` returns operator's local TZ. UTC-internal storage + operator-TZ render at display time is the standard architecture for long-running multi-timezone systems. Today operator-TZ-default ships first since the daemon runs on the operator's machine, but if Anchor ever runs in a managed environment or multi-operator setting, UTC-internal becomes required. Brain memory `dcb5b951` notes this.

### Dispatcher enforcement for mandatory metadata fields
Encoder compliance asymmetry (brain memory `92b890e7`): restraint rules 100%, generative rules ~0-20%. v15.8 raised event_time compliance from 0% to ~5-8% — clear improvement but ceiling visible. Path B: dispatcher-level enforcement (precedent: `related/related_to` ban via dispatcher, brain memory `c39b8cc8` / `5e27a23f`). Detect at `remember_batch` dispatch: node has dated content but missing event_time → auto-extract via regex OR log loud to brain_errors. Higher reliability than prompt iteration.

### S2 Healer temporal enrichment
Healer currently fills `question` / `situation` / `reasoning`. Designed but unbuilt: dangling-anchor resolution (resolve "before the move" once "the move" is dated), implicit-sequence-edge creation (link co-occurring events with Allen relations), date propagation through sequence graph, cross-session temporal consolidation. Clean architectural slot — same idle cycle, same `Haiku + revise()` machinery.

### Agent introspection — remaining probes
4 of 6 modes built (aspect, compliance, coherence, counterfactual). Coverage probe was the open slot for "given THIS conversation, what would the agent do?" — **built 2026-05-15 as two domain-specific tools:** [`eval/agent_introspect/encoder_replay.py`](../eval/agent_introspect/encoder_replay.py) for encoder, [`eval/agent_introspect/surface_replay.py`](../eval/agent_introspect/surface_replay.py) for surface. Both replay the actual agent call against a candidate prompt without paying eval-pipeline cost (~$0.001, ~2-13s per replay). Unbuilt: **edge-case probe** (corner-case scenarios), **priority probe** (when rules conflict, which wins?). Build any when a specific iteration arc demands it.

---

## ⚡ Open items from 2026-05-15 surface arc

### v14 + v6 surface eval (the right ship-test for surface v6)
2026-05-15 12-item diverse eval showed v15.11+v6 = 8/12 (67%) vs v15.11+v5 = 6/12 (50%) — surface v6 is +2 items on the same encoder. But v14+v4 = 10/12 (83%) on the same items. Conclusion: v6 surface is a real win, v15.11 encoder is not. **Right next step: run v14 + v6 on the 24-item cohort to isolate the surface-only ship.** If v14+v6 ≥ v14+v4 by any margin, ship v6 surface alone. Cost ~$15, ~50min.

### Surface `75832dbd` render-size bug
Item `75832dbd` ("Can you recommend some recent publications or conferences") hits `stop=max_tokens` at Haiku's 8192-token output ceiling. Pre-Haiku rendered content too dense for the model to emit JSON after thinking. Likely fix in `surface_contract.py` candidate rendering — trim candidate `content` field for surface input, or use compressed rendering. Not v6-prompt issue; this is a rendering-volume issue. Failing on 1/24 items in the A/B but reproducible.

### S1S Quality Rubric — implement or drop?
Apr 24 design notes at (former) `docs/S1S-QUALITY-RUBRIC-NOTES.md` (now archived) proposed a multi-dimensional per-node + per-run Haiku-judge rubric: structural (0/1), semantic (0/1/2), atomization (0/1), correction-axis specifics. Never built. **Question for Tom: still wanted? If yes, P3 validation infrastructure; if no, drop the design intent.**

### SKILL.md Encoding Craft section reframe
2026-05-15 SKILL.md surgical edits flipped 4-5 active-encoding bullets but did not touch the "## Encoding Craft" section (title + line 109 "when you encode lessons" + Encoding Richness subsection). Decision deferred — wider reframe scope. Either delete the section (encoder owns this craft) or reframe it as "Node Craft — for revising and reading nodes" with verb flips.

### connect_to ID-shape pattern in encoder prompt
2026-05-15 bug fix (commit `08156ee`) at `_resolve_connect_to_entry` recovers when the encoder passes an 8+ hex-char ID in the `title` field. Real root cause is the encoder prompt not having a clean schema for "connect to this specific known node by id" — the `connect_to` spec is title-only. A future improvement: support `connect_to: {id: "...", relation: ..., why: ...}` shape so the encoder can express ID-based connection intent directly.

---

## Recent ship log (older — moved to archive review)

The May 8, May 10, and May 15 ship logs (encoder prompt versions v14/v15.3/v15.6/v15.8/v15.11, AspectIntegration rewire, eval artifacts subsystem, surface v6 prompt, 24-item A/B eval) are preserved in the historical context but trimmed from this backlog. Current encoder iteration moved past v15.11 to the v18-era work and is now superseded by the v19 + episodic-references arc (see EPISODIC-REFERENCES.md §7).

Key carry-forward items from those sessions (still relevant):
- **v15.11 encoder was not recommended for ship** — atomic-fact substrate didn't aggregate as well as v14 narrative bundles for multi-session synthesis. The episodic-references arc resolves this differently (source_refs preserve substrate while the encoder writes atomized framing).
- **Surface v6 candidate registration HELD** pending v14+v6 isolation test. Still held; will be re-evaluated once episodic-references ship.
- **Eval artifacts subsystem + refined-bucket analyzer + run-diff** all in production (eval/longmem/ + eval/agent_introspect/).

---

## The mission

Recall — the moment relevant memories rise into Anchor's awareness when the operator speaks. Everything in this backlog either *directly* improves that moment (P0–P2), *validates* it (P3), or is operational hygiene that prevents regressions (P4–P5).

Priority bands:
- **P0** — blocking right now
- **P1** — high-leverage recall improvements (direct user-felt impact, designed-or-cheap)
- **P2** — recall arc (bigger pieces, designed-not-built)
- **P3** — validation infrastructure
- **P4** — operational backlog (post-launch items)
- **P5** — backburner / long-tail

---

---

## P1 — High-leverage recall improvements (design-done or cheap)

> **⚠ Code-verified 2026-05-30 (Anchor).** These bands predate the Frame (Phase 2)
> + episodic-refs ships and carried stale framing. Per-item check against code:
> **P1.2–P1.5, P2.2, P2.4, P3.x confirmed NOT-BUILT (accurate as written).**
> **P1.1, P2.1, P2.5 had stale claims — corrected inline below.** Don't trust a
> band's "design-done" label cold; verify against code before picking up.

### P1.1 — Frame as filter (recency bias fix)

- **Why:** BRAIN-CHALLENGES.md entry #2. Recall consistently returns aged-but-topical clusters when the user names a recent specific arc ("Aspect encoder", "Frame", "where we left off"). Frame holds the recent arc as a structured prior but isn't biasing candidate selection. Direct user-felt failure — Tom hit this multiple times this session.
- **What:** Frame's `Active threads` and `Recent moves` sections carry node IDs. Pass that ID set into the surface scoring step. Boost candidates whose IDs match (or are 1-hop neighbors). Apply BEFORE Haiku selects from the 25-candidate pool.
- **Files:** `servers/scales/s1/frame.py` (expose `frame.frontier_ids()` returning the union), `servers/scales/s1/surface_contract.py` (the scoring function — add a `frame_match_boost` step), `servers/scales/s1/surface.py` (pass frame to scoring).
- **Effort:** ~2 h.
- **Acceptance:** capture the two failing prompts from BRAIN-CHALLENGES #2 (`aspect_encoder_pickup`, `frame_recall_resume`) as labeled queries in `eval/frame_replay.py`. Snapshot before/after. Both should surface fresh-arc nodes (not 1-month-old encoder-optimization history).

> **Verified 2026-05-30:** premise partly stale, acceptance wrong.
> (1) Frame **did** ship as a prompt-prior — injected as a "Partnership context
> (your prior)" block ([surface_contract.py:369](../servers/scales/s1/surface_contract.py)).
> That sways Haiku's choice *among* the 25 candidates. P1.1's distinct,
> still-**NOT-BUILT** lever is a candidate-**scoring** boost on Frame frontier IDs
> (no `frontier_ids()` / `frame_match_boost` in code) — it changes *which* 25 make
> the pool. The recency miss is most likely pool-composition, so P1.1 is **not**
> subsumed by the Phase-2 prior.
> (2) Acceptance is broken: `aspect_encoder_pickup` / `frame_recall_resume` do
> **not** exist in `eval/frame_replay.py` (corpus is `exco_cold`, `self_intro`,
> `exco_pivot`, `where_were_we`, `open_last_week`). First step is to ADD the
> failure queries, then re-confirm the miss still reproduces post-Phase-2 before
> building.

### P1.2 — Phrase-anchored title boost

- **Why:** Same failure family as P1.1. When user says "Aspect encoder" by name, candidates whose titles contain that exact phrase should pin to the top — not get embedded into a cosine score that buries them under broader topic matches.
- **What:** Extract proper-noun-ish phrases from query (capitalized multi-word, 2–4 tokens). FTS5 search the title field. Boost matches.
- **Files:** `servers/scales/s1/surface_contract.py`.
- **Effort:** ~1 h.
- **Acceptance:** same regression queries as P1.1.

### P1.3 — Connection scoring (Step 3.5)

- **Why:** Designed in `RECALL-OVERVIEW.md` §3, never built. Localizes hub bias — true hubs only dominate when they connect to other relevant candidates for THIS query, not always. Spec exists.
- **What:** After candidate enrichment scoring, score each candidate by connectivity to OTHER high-scoring candidates in the pool. Edge type weights via `brain.aspects` (`correction_improvement`/`extension_refinement` strong; `generic_relation`/`noise` weak; `hierarchical_structure`/`temporal_sequence` moderate). Cluster detection: 3+ interconnected candidates score together; isolated high-cosine nodes get lower priority.
- **Files:** `servers/scales/s1/surface_contract.py` (new `_connection_score()` step), `servers/brain_recall.py` (rerank pipeline integration).
- **Effort:** ~3–4 h.
- **Acceptance:** `eval/frame_replay.py` shows different ranking on `where_were_we` corpus vs baseline; hub nodes ranked lower when query doesn't match cluster.

### P1.4 — Posture detection (recent-vs-historic bias)

- **Why:** When user says "where we left off / yesterday / what we just did," surface should bias hard toward recent. Today it doesn't — same scoring regardless of intent.
- **What:** Lightweight regex/heuristic on query that detects "recency intent" → boost recency in scoring. Could be a single boolean knob passed into scoring.
- **Files:** `servers/scales/s1/surface_contract.py`.
- **Effort:** ~1 h.
- **Acceptance:** same regression queries; recent nodes (≤7d) outrank older equivalents on recency-flavored queries.

> **Verified 2026-05-30 — mostly pre-built; this is a wire-up, not a from-scratch build, and it has a regression scar.**
>
> **The math already exists.** `servers/recall_scoring.py` (Apr 2) defines
> `unified_score(semantic_score, created_at, emotion, access_count, confidence)`
> = `base * (1 + recency_boost + emotion_boost + frequency_penalty + confidence_boost)`,
> bounded ~0.81–1.55×. `freshness_from_created(created_at)` is the recency term
> (fresh nodes lifted; old nodes get boost=0, *no penalty*). The inputs are
> already collected into scope at recall time — `node_created_at` /
> `node_emotion` / `node_access_count` at [brain_recall.py:1346-1348](../servers/brain_recall.py).
>
> **Why it's OFF: it regressed R@8 by −10pts.** The deferred-work comment at
> [brain_recall.py:1712](../servers/brain_recall.py) records that wiring
> `unified_score` in *unconditionally* dampened scores that were previously
> passing the relevance floor. Its own next-step: *"likely need per-query-type
> adaptive weights rather than one fixed formula."* **That sentence IS this
> item** — the recency-intent boost = apply the existing freshness modulator
> ONLY when the query has recency intent (and relax `frequency_penalty` then),
> instead of always. So the two remaining pieces are: (1) the intent detector
> (the genuinely-new part — what phrasings count, false-positive cost), and
> (2) the conditional gate at line 1712 where `query` + `node_created_at[nid]`
> are both already in scope.
>
> **Revised effort & file:** ~half-day, and the real insertion is
> `brain_recall.py:1712` + `recall_scoring.py` — NOT (only) `surface_contract.py`.
> **Mandatory eval gate** (`decode_funnel` / `frame_replay`) — the naive
> always-on version already burned −10pts; cannot just flip on.
>
> **Orthogonal to fatigue.** Fatigue ([brain_recall.py:1562](../servers/brain_recall.py),
> `score *= 1-fatigue`) is a per-session *anti-repeat* dampener; freshness is
> node-age. They don't fight.
>
> **Caveat — freshness is a proxy.** "Where we left off" really means *this
> session's arc* (→ P1.1 Frame-frontier), not raw node age. A 3-day-old
> unrelated node also gets the freshness boost. P1.4 is the blunt lever; P1.1
> is the precise one. They stack.
>
> **Live evidence (2026-05-30 experiment):** recency intent carries zero weight
> today — query "where we left off *today* — the brain_batch invalid op fix"
> ranked today's fix node #4 behind three ~6-week-old April nodes; another
> today-node fell out of the top-25 entirely on the topical query. Reproduced
> as a **ranking/pool-composition** problem (not Haiku selection), N=2 / one
> topic — a real verdict needs the recency-query eval corpus that P1.1's
> acceptance also lacks.

### P1.5 — Cadence split: brain-level vs session-level Frame caching

- **Why:** Today every recall re-injects the full Frame (~1900 tokens). 60% of it (Operator + Partnership-integrated + Permanent) is slow-changing — wasted re-injection most turns. Cost + latency.
- **What:** Split Frame build into brain-level (cacheable, refreshed on S2 cycles or encoder writes) and session-level (current_focus + recent_moves, refreshed per turn). Two cache breakpoints in surface system block.
- **Files:** `servers/scales/s1/frame.py`, `servers/scales/s1/surface.py` (cache_control structure).
- **Effort:** ~2–3 h.

---

## P2 — Recall arc (designed, bigger builds)

### P2.1 — Agentic Haiku-first recall (7-tool `fetch_batch`)

- **Why:** Design finalized in `FRAME-DESIGN.md` §4. Replaces single-cosine candidate pull with Haiku planning the fetch per-turn. Variable cost, sample-then-deepen, frame-shaped output. The next major recall capability.
- **The 7 tools:** `search(query, mode, limit)`, `find_about(entity, limit)`, `find_open_loops(topic?, limit)`, `trace_lineage(node_id, direction, max_steps)`, `get_community(community_id, query?)`, `find_temporal(when, query?, limit)`, `get_full(node_ids)`. All wrapped in single `fetch_batch` for parallel-op single Haiku turn.
- **Files:** new `servers/scales/s1/fetch_batch.py`, surface prompt v5, `servers/daemon_dispatch.py` (new commands), tool descriptions.
- **Effort:** ~2 days.
- **Depends on:** Q13 decision (does spread activation survive Phase 4?).

> **Verified 2026-05-30:** `servers/scales/s1/fetch_batch.py` does **not** exist —
> unbuilt as described, a ground-up build. Caveat: `BRAIN_SURFACE_VARIANT=v5_agentic`
> *is* active (`brain-env.sh`), but that's the surface *rendering/selection*
> variant — NOT this 7-tool fetch module. Don't conflate them.

### P2.2 — Multi-anchor query decomposition

- **Why:** Implements the query-multiplicity principle. Messages contain 2–4 distinct concepts; collapsing into one cosine vector loses the structure. Decompose, run multiple anchors in parallel, convergence is the strongest signal.
- **What:** Query-decomposition step (heuristic or Haiku) → multi-spread orchestration → convergence scoring on overlapping nodes.
- **Files:** new `servers/brain_recall_multi.py`, surface integration.
- **Effort:** ~1–2 days.

### P2.3 — Hybrid retrieval — FTS5 + embeddings full integration

- **Why:** Partially shipped — recall pipeline has both channels but they don't combine cleanly. Full integration with connection scoring (P1.3) lets us catch both "what does this concept mean" (embeddings) and "what was that exact phrase" (FTS5).
- **Files:** `servers/brain_recall.py` (channel combiner), `servers/scales/s1/surface_contract.py`.
- **Effort:** ~1 day. Depends on P1.3 (connection scoring) to score the union.

### P2.4 — Wire Frame into S1 Scribe (gap-aware encoding)

- **Why:** Encoder doesn't currently see Frame; doesn't know what's already in awareness. Could yield complementary encoding instead of restating.
- **Files:** `servers/scales/s1/encode.py`, `servers/scales/s1/encode_contract.py` (Frame in user content).
- **Effort:** ~3 h.

### P2.5 — Q13 decision: kill spread activation or keep it?

- **Why:** Today's `_traverse_graph()` 3–4 s baseline cost eats the latency budget that Phase 4 (agentic) tools need. Anchor's lean: retire spread, keep kernel as tool-internal helper. **Decision needed before P2.1 ships.**
- **What:** This isn't a build, it's a 30-minute conversation + decision document.

> **Verified 2026-05-30:** largely **resolved in practice.** `_traverse_graph` was
> removed from the recall path 2026-04-14 (dead, 0 callers — see `brain_recall.py`
> comments); `spread_activation` still lives and runs in `surface.py` (post-selection
> expansion). The de-facto state already matches Anchor's lean — retired from recall,
> retained in surface. What's left is to *document* the decision and confirm P2.1
> doesn't still treat this as a blocking gate.

---

## P3 — Validation infrastructure

### P3.1 — Fresh-Claude vs Anchor calibration test (Frame punch list #11)

- **Why:** Only path to empirically validating SKILL.md / boot changes. `eval/frame_replay.py` bypasses Claude Code; `eval/longmem` deliberately avoids Anchor's voice. Today Tom is the only sensor for "does Anchor wake up as Anchor."
- **What:** Spawn fresh Claude Code session with brain skill loaded; identical wakeup probes ("Who am I working with? What's open? Where were we?"); compare to fresh Claude WITHOUT brain. The delta IS what the brain buys at the wakeup moment.
- **Files:** new `eval/calibration_fresh_vs_anchor.py`.
- **Effort:** ~4 h.

### P3.2 — Fix `eval_runner.py` bypass of enrichment scoring

- **Why:** `RECALL-OVERVIEW.md` §4 tension #2. Eval bypasses the enrichment scoring step that production uses. Backfill / scoring improvements are invisible to eval.
- **Files:** `eval/eval_runner.py` (wire enrichments into evaluator) OR switch to production recall method.
- **Effort:** ~2 h.

---

## P4 — Operational backlog (was PHASE-B+1)

### 🟠 HIGH

#### P4.1 — Encoding lock can hang forever (was B+1.2)
`daemon_hooks.py:456` acquires module-level `_encoding_lock` for the encoder background thread. If thread crashes before releasing, lock held forever; ALL subsequent encoding silently skips. Wrap acquire+spawn in try/finally. **20 min.**

### 🟡 MEDIUM

#### P4.2 — Build `s2_vector_healer` unit (was B+1.4)
Detects + repairs stale vectors that escaped `revise()` invalidation (kv text updated AFTER `node_enrichments.created_at`). Backstop for paths that bypass revise. **2–3 days.**

#### P4.3 — Encoder activation visibility into S1R (was B+1.5)
S1R discards activation metadata before encoder. Encoder makes revise decisions blind to which fields fired and how strongly. Pass activation through trace metadata. **4–6 h.**

#### P4.4 — Multiple format configs consolidation (was B+1.6)
Three subtly-different configs for `render_rich_node` (HAIKU_FORMAT / SURFACE_FORMAT / S1_NODE_CONFIG). Merge into a single config family with named modes. **1 h.**

#### P4.5 — Edge selection called twice per recall (was B+1.7)
`daemon_hooks.py:229` calls `select_edges()` per candidate; `surface_contract.py:1129` calls AGAIN during activation render. Cache first call's result. **30 min.**

#### P4.6 — Catalog from rendered strings, not activation results (was B+1.9)
`build_node_catalog()` regex-extracts node IDs from rendered surface text strings. Inefficient and fragile. Track surfaced node IDs in S0/S1 traces directly. **1 h.**

#### P4.7 — Healer unsolicited fields (was B+1.14)
S2 Healer asks for specific missing fields; Haiku returns ALL three. System rejects + logs loudly. Strengthen prompt (move single-field example first) OR move to `tool_choice` JSON schema. **30 min prompt / 2 h schema.**

#### P4.8 — Haiku selects IDs not in candidate menu (was B+1.15)
Surfacer Haiku given top-25 menu, sometimes returns out-of-menu IDs. Could be feature (Haiku knows IDs from training) or bug. Diagnostic first: log out-of-menu IDs, check if real brain nodes. **30 min.**

#### P4.9 — Encoder uses two tool families when one would do (was B+1.16)
First production cycle used `remember_batch` + `brain_batch` = 3 rounds when one `brain_batch` would do. Strengthen prompt's MIX rule. **20 min.**

### 🟢 LOW (cleanup, do in batches when convenient)

- **P4.10** — `find_missing()` filter naming + `source_kv_keys` semantics doc (B+1.3) — 10 min
- **P4.11** — Truncation has no ellipsis (B+1.8) — 15 min
- **P4.12** — Remaining silent-`pass` excepts (B+1.10), 4 sites — 5 min each
- **P4.13** — Hardcoded constants → interaction config (B+1.11) — 30 min
- **P4.14** — `failed_connect_to_count` in batch result (B+1.12) — 10 min
- **P4.15** — Sibling-map case-sensitivity docstring (B+1.13) — 5 min
- **P4.16** — Trace metadata bloat (B+1.17) — 30 min after a week of accumulation
- **P4.17** — Rename `judge_output` → `surface_output` across the trace metadata contract — 1–2h. The S1 surface step was renamed from "judge" → "surface" in commit `620fb4f` (2026-05-03), and the user-facing/code path has been cleaned (commit `b126d98`, 2026-05-09 — `surface.py` no longer falls back to 'judge', orphan 'judge' interaction row deleted). What remains is the trace metadata field name still carrying the legacy "judge_output" — written by `dal.py:get_user_turns` (lines 687–729) into the `judge_output` key of trace dicts, read/asserted on by `tests/test_s1_data_assembly.py`, `tests/test_okd_cycle.py`, `tests/test_scout_muster.py`, `tests/test_trace_system.py`, plus the `pipeline_contract.py` legacy aliases (`format_candidate_for_judge`, `build_judge_prompt`, `format_judge_output` — lines 509–511). One commit: rename the field, drop the aliases, update tests. No data migration needed (it's a derived field assembled from `additionalContext` traces, not stored). Defer until there's another reason to touch dal.py to keep the diff focused.

---

## P5 — Backburner

### Old standing threads (open in graph, low recent activity)

- **Telemetry / brain proprioception** (overdue since 2026-03-29, reminder `49b33c19`) — comprehensive observability layer. Likely subsumed by P0.1's memory_watchdog work + dashboard.
- **Temporal reasoning ideas** (`095cc070`) — relative time display already shipped (`format_judge_output`). Time-range retrieval is separate, harder, deferred.
- **Encoder vs Stop hook 10s timeout** (`38bc9a6f`) — addressed by background-encoding architecture; the tension node may be stale.
- **Emergent types ignored** (`c8c773b4`) — partially addressed by today's aspects work; the original "auto-promote when N nodes accumulate" hypothesis isn't built.
- **Host environment awareness** (`891f9a53`) — never worked. Low priority.
- **Brain proactively surface prior art** (`dd7b4d20`) — aspirational. Overlaps with agentic recall (P2.1).

### Stage 1C — explicitly deferred

Keywords→KV migration. Audit confirmed 0 dual-state. Pick up only if natural.

### The "irresolvable" tensions

- **SKILL.md tension** (`RECALL-OVERVIEW` §4 #7) — instructions to a stateless thing about how to behave as if continuous. Built-in contradiction. Don't try to dissolve.

---

## Decisions needed (open)

These aren't builds — they're choices that gate other work.

| # | Question | Gates | Lean |
|---|---|---|---|
| Q13 | Kill spread activation or keep as helper? | P2.1 | retire spread, keep kernel |
| Q-A | Daemon memory_watchdog: enable now or after next leak? | P0.1 | enable now |
| Q-B | AspectIntegration auto-merge or operator-review gate in production? | P0.2 | auto-merge for now |
| Q-C | Frame Phase 2.5 — wire into encoder before or after recall improvements? | P2.4 ordering | after P1.x lands |

---

## What this replaces

- `docs/PHASE-B+1-BACKLOG.md` — moved to `docs/archive/`. All items folded into P4 here.
- `docs/RECALL-OVERVIEW.md` §3 — short list pointing here; the recall arc + Frame punch list inline there will be replaced with a pointer.
- `docs/BRAIN-CHALLENGES.md` — kept (different purpose: cognitive-gap log). Items #1 + #2 fixes now live in P1 here.

## How to update this doc

- Item ships → strikethrough + `→ shipped {commit}`
- Item promoted/demoted → move between bands, note date
- New item → numbered slot in target band, brief why + acceptance
- Don't delete items — keep history visible
