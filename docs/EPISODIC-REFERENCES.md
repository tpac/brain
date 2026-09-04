# Episodic References — Design Doc

> **Status**: PHASE A SHIPPED 2026-05-24. Substrate (schema, DAL, identity stamping, embed worker, historical migration) is live in production. Encoder write path + render expansion + prompt v19 + recall consumers still pending.
> **Last touched**: 2026-05-24
> **Reading order**: §0 first (execution log — what shipped, what shifted, what we learned); §1 (locked decisions); §2 (why this matters); then straight through 3→16. §14 is the execution checklist (Phase A items marked done).

---

## 0. Execution log

Living record of what happened when we moved this design out of the doc and into the brain. Captures revisions to the original plan, insights that earned their keep through implementation, and what's actually shipped vs what's still on paper.

### What shipped — Phase B Steps 6+7 + Layer 1 validator + v22 active (2026-05-26)

**Episodic-references substrate is end-to-end live in production.** v22 active since `2026-05-26T00:46:54Z` (commit `c144ddf`).

- **Step 6** — encoder prompt v22 (commit `c144ddf`). Five additive changes over v21: (1) `[trace:<hex>]` marker convention in "Reading the conversation"; (2) §7.4 "Anchoring nodes in the substrate" — full prose with three patterns (Pure synthesis / Anchored synthesis / Pure reference) + the one judgment rule; (3) §7.5 sparseness discipline (1-3 typical, second-guess at 5-6); (4) revise() REPLACE semantics for source_refs (present REPLACES, absent PRESERVES, empty list CLEARS); (5) source_refs analog to the their_raw_quote interpret/expand rule. Three Sonnet probes (cold-read interpretation, live encoding simulation, comparative audit) validated the prose before registration. Activated after the 50-cell longmem eval.
- **Step 7** — co_anchored auto-edge in dispatch (commit `07ab3f1`). Decision 15 wired: when `brain.remember()` or `brain.revise()` persists source_refs, the dispatcher queries `node_source_refs(trace_id)` for siblings sharing any ref and writes structural `co_anchored` edges via `GraphDAL.add_relation`. Excluded from candidate cosine ranking (same pattern as `co_accessed`). Sparse refs × small cohort → negligible cost. Skip-self handled; deduped across multiple refs per node. 5 new co_anchored tests pass.
- **Layer 1 validator extensions** (commit `07ab3f1`). Two soft-warn additions in `daemon_dispatch.py`:
  - `_maybe_warn_source_refs_hex_format`: rate-limited warn when any ref fails `^[0-9a-f]{8}$` (reviewer F6, closed).
  - Sparsity threshold lowered from >10 to >5 — aligns with v22's §7.5 teaching.
  Wired into all four entry points. 3 new validator tests pass.
- **sync-prompts active-version fix** (commit `1fe730e`). Bug: `_fetch_latest` grabbed highest version regardless of active state, which would seed fresh-brain installs with untested DORMANT candidates, bypassing the eval gate. Fix: `_fetch_active` JOINs against `interaction_active` pointer. Regression-locked. CLAUDE.md updated with eval-gated workflow.
- **eval/encoder_eval/ infrastructure** (commits `5238c02`, `5a4e672`, `fe71ac4`). Multi-version encoder eval module with 6 probes (`brain_presence`, `specificity_preservation`, `source_refs_coverage`, `atomization_shape`, `edge_structure`, `voice_balance`), staged checkpointed harness, parallel cell execution, `--stratify` for axis-balanced sampling. Used for the 50-cell longmem real-test that gated v22's activation. Reusable for every future encoder version.
- **eval/ground_truth/ corpus scaffolding** (commit `7e721eb`). 7 conversation templates across 5 strata covering identity-bearing, partnership voice, technical correction, methodology, temporal. Each has fillable YAML for ideal-node authoring + rationale + expected failure modes. Authoring pending Tom-time.

**50-cell longmem real-test (the v22 gate)**:
- v22: 23/25 (92%) correct vs v19: 22/25 (88%)
- v22: 100% source_refs coverage uniformly across all 25 cells
- v19: 72-95% source_refs coverage (opportunistic — substrate provides capability, v19 prompt doesn't teach systematic use)
- v22 wins specificity (combined score) on 4 of 5 axes
- info_extraction axis: v22 +1 vs v19; the differentiating case `cc539528` had v19 encoding ZERO nodes for that item
- v22 +40% more typed edges per item on info_extraction; co_anchored cohorts form correctly
- Stop conditions: none fired

**Methodological finding**: LLM encoders are stochastic at N=1. The 50-cell run captured one v22 failure on `c2ac3c61` (multi_session); subsequent re-runs of the SAME prompt succeeded. Similarly, `5025383b` succeeded in the 50-cell run but failed v22 on a re-run. Future eval design needs multi-sample per cell (N≥3) to distinguish deterministic failure modes from tail outcomes.

**Reviewer follow-ups status**:
- F1, F2, F4, F6, F8 — all FIXED.
- F3 (GraphDAL commit-in-batch-mode audit), F7 (move `_SOURCE_REFS_SCHEMA` to contract.py) — still deferred.

### v23/v24 + scout iteration (2026-05-26, DORMANT)

Forensic probes on `c2ac3c61` (multi_session precision-refinement) identified three encoder-prompt gaps + a facts-scout misclassification pattern. Two iteration cycles:

**Cycle 1 — v23 / facts v6 / quote v3 (DORMANT)**:
- s1e v23: one sentence in §7.4 (Anchored synthesis) — "when a fact surfaces twice in the window (vague earlier + precise later) on the same axis, BOTH are evidence-events for one node"
- facts_scout v6: 4 edits — supersession-scope clarifier, Cap ranking reformulated, Example 4 why_candidates tightened, NEW Example 5 (parallel-entity + same-axis refinement, languages-domain abstraction — NOT corpus-reverse-engineered, per the B1 lesson)
- quote_scout v3: new Skip bullet ("routine factual claims = facts scout's territory")

**Cycle 2 — v24 / facts v7 / quote v4 (post-probe refined, DORMANT)**:
- s1e v24: "originating phrase" disambiguated to "originating turn's verbatim phrasing stays in their_raw_quote — keep the vague phrase; don't overwrite with the refined wording"
- facts_scout v7: Cap statement preserves "recall-weight" framing (not strict exclusion rules) + refinement-pair clarifier
- quote_scout v4: appended mixed-content handling ("isolate the distinctive phrase as the handle; don't quote the factual setup")

**Targeted A/B + follow-up evals** (`eval/encoder_eval/reports/v24_targeted_20260525_235258/` + `v24_followup_20260525_235653/`):
- Targeted (5 items × 2 arms = 10 cells): baseline v22+5+2 4/5, candidate v24+7+4 **5/5**
- Follow-up (5 new items, candidate-only): v24+7+4 4/5
- Combined v24 signal: **9/10 (90%)** across 10 distinct items
- Wins: `gpt4_93159ced_abs` (both v22+v19 lost in 50-cell run — v24 properly abstains: *"I don't have information about your work history or when you started at Google."*); `5025383b` (v22-this-run dropped one hobby, v24 caught both)
- Regression direction: `gpt4_d31cdae3` (temporal hedging — v24 over-cautious inferring "a few years ago" relative to current date)

**Eval-decision call open** (Tom's preference: continue experimenting):
- Option A: Activate v24+v7+v4 now. +1/10 wins, real abstention improvement.
- Option B: Re-run `gpt4_d31cdae3` against v24 ×3 to distinguish stochastic vs. deterministic regression. ~15 min, ~$2.
- Option C: Leave DORMANT, accumulate more eval evidence.

**Reusable evaluator pattern**: `override_interaction(brain, name, template=...)` — works for any interaction, not just s1e. Lives in `tests/interaction_override.py` (consolidated there 2026-08-22; the per-eval copies are gone), with `interaction_override(...)` as the self-reverting context-manager form. Future version comparisons can override any combination of s1e + scouts in fresh eval brains without touching production daemon state.

### What shipped — Phase B Steps 0+0.5+1+2+3+4 (2026-05-25 mid-day)

**Substrate complete. Encoder behavior unchanged until v22 prompt registers.**

- **Step 0** — example sub-contract: quality_contract v3 (36 dims, Group 9 = D33 placeholder_syntax / D34 ref_internal_consistency / D35 voice_annotation_coverage / D36 turn_node_divergence). CR12 names verbatim quote fields as the legitimate bridge between turn and node registers. `validate_example_authoring()` mechanical check. All 7 §7.6 examples rewritten to `<placeholder>` syntax (source_refs + external connect_to titles; B1's intra-batch sibling refs retain literal titles — legitimate exception).
- **Step 0.5** — schema v29: `trace_events.id`, `trace_embeddings.trace_id`, `node_source_refs.trace_id` migrated INTEGER → TEXT (8-char hex). 60,634 trace_events + 12,933 trace_embeddings + 0 node_source_refs migrated. `brain.db.v28.bak` auto-backup. `TraceDAL.append` generates `secrets.token_hex(4)` with collision retry. Int input rejected loudly everywhere (no silent coercion — random hex collision risk per reviewer F2).
- **Step 1** — `encode.py::_build_user_content` renders `[trace:<hex>]` markers per turn in the conversation timeline. `_fmt_trace(None)` → `[trace:?]` for legacy/JSONL-fallback turns. S0 layer (`get_conversation`, `TraceDAL.get_session_turns`) passes `trace_id` through.
- **Step 2** — MCP schemas declare `_SOURCE_REFS_SCHEMA` (array of strings) on remember / remember_batch / revise_batch. brain_batch inherits via per-op dispatch routing. get_trace/get_traces string-typed.
- **Step 3** — `brain.remember(source_refs=[...])` persists via `GraphDAL.add_source_refs` (INSERT OR IGNORE; first-write-wins). `brain.revise(source_refs=[...])` persists via `GraphDAL.replace_source_refs` (atomic DELETE + INSERT; field-level REPLACE per unified 2-class revise contract id:`995ffeb1`). `_SR_ABSENT` sentinel distinguishes "key absent" (preserve) from "explicit empty list" (clear).
- **Step 4** — `_validate_source_refs` at write boundary (list of strings, non-empty); `_maybe_warn_source_refs_sparseness` warns at >10 refs via `brain._log_warning` (rate-limited, canonical surface).

**Reviewer pass** (8 findings on the bundled write-path commit):
- F1 critical (revise REPLACE not APPEND), F2 (int coercion collision), F4 (sparseness logging surface), F8 (missing integration test) — all FIXED.
- F3 (GraphDAL commit-in-batch-mode audit), F6 (optional hex regex warning), F7 (move SOURCE_REFS_SCHEMA to contract.py) — deferred to BACKLOG.md.
- `tests/test_remember_source_refs.py` covers remember/revise/brain_batch + validator edge cases (15 tests). 145 v29-touching tests pass.

**What's NOT yet shipped (next session)**:
- Step 6 — encoder prompt v22 (the teaching that makes the substrate functional — §7.4 + §7.5 + `[trace:<hex>]` convention + revise REPLACE semantics)
- Step 7 — co_anchored auto-edge in dispatch (when nodes share refs, write the structural engram edge)
- Step 8 — 3-way eval gate (v22 vs v21 vs v19) before active production flip

### What shipped — Phase A substrate (2026-05-24)

Ten commits on `main`, all sitting on a clean substrate. Identity stamping live end-to-end.

| Commit | Block |
|---|---|
| `9015636` | Schema v27 — `node_source_refs` + `trace_embeddings` tables, composite index |
| `8a52164` | DAL methods (6 new across TraceDAL + GraphDAL) |
| `75075eb` | Identity stamping wired into TraceDAL.append + Brain.__init__ |
| `65bf483` | Identity activated end-to-end (env propagation, dispatch metadata decode, defensive guard) |
| `7b5b845` | Pull-reconciliation embed worker — first writes to `trace_embeddings` |
| `669ecee` | Worker review fixes — 30-day window, composite index, observability, legacy decode helper |
| `5cff407` | Historical identity migration — 57,546 traces backfilled, 22,416 double-encoded rows cleaned |
| `4288ec8` | Loud-by-default surfacing — three silent paths fixed |
| `987587f` | Identity-unset signal moved from boot to write boundary |
| `d68bddc` | Cleanups — `_decode_metadata` rolled out, point-lookup API (`brain.get_trace`/`get_traces`), dead test removed |

### Design revisions during execution

What the original doc said vs. what we actually built and why.

1. **Decision 19 reversed at sub-decision (a)** — original direction was abstract labels at embedding (`OPERATOR / ANCHOR`) for partner-change stability. Biology research (three independent angles converging on concept-cell coding, self-vs-other DMN architecture, per-utterance binding stickiness) flipped it to **concrete tokens** (`Tom` / `Anchor`). The "stability across partner changes" argument was inverted: stability comes from concept-cell-style sparse individual codes, not from slot-name-keeps-meaning. §15.1 closed with four sub-decisions locked.

2. **Pull-reconciliation supersedes push-queue (steps 12/14 → deleted, step 13 reshaped).** Original §14 had `enqueue_trace(trace_id)` hooked into `brain.remember` after source_refs persist. Tom's reframe: *"the worker runs every 5s — just embed top 5 empty embed ordered by date from fresh to old."* The LEFT JOIN reconciliation is restart-safe, has no in-memory queue, and removes a coupling site from the encoder write path. Worker simply queries `find_unembedded` every tick.

3. **TraceDAL extension, not new class** ("extend before create"). Original plan had a separate `TraceEmbeddingsDAL`. Trace embeddings are "trace data with one extra computed column" — same table family, same `logs_conn`, same concern. Three methods added to existing `TraceDAL` instead of a new class.

4. **Embed window added (30 days, configurable).** Not in the original design. Audit during code review surfaced the architectural concern: eagerly embedding the 29K historical trace_events backlog would have rendered them with `OPERATOR` / `ANCHOR` sentinel labels (since they had no identity in metadata pre-migration). Decision 19 specifically keeps the embedding neighborhood concrete-token-only — sentinel pollution would have undermined the very biology-alignment we just researched. The window is a correctness gate, not a perf knob.

5. **Composite index `idx_trace_scope_created` added on `trace_events(scale, ref_type, created_at)`.** Found via EXPLAIN QUERY PLAN during review: the worker's hot query was building a TEMP B-TREE for ORDER BY DESC every tick. Composite index eliminates the temp sort.

6. **`_decode_metadata` defensive helper added.** A pre-existing bug in `_handle_trace_append` (dispatch wasn't json-decoding string metadata from cross-process clients) caused all post-tool-use traces to store double-encoded JSON. Fixed forward; defensive helper added to all TraceDAL readers to handle legacy double-encoded rows gracefully. The historical migration (item 8 below) cleaned up the existing legacy rows.

7. **Identity-unset signal moved from boot to write boundary.** First implementation logged at `Brain.__init__` if env vars unset. Tom's correction: *"it should be on any scale write, not on boot specifically."* The check now lives in `TraceDAL._maybe_warn_identity_unset`, fires once per daemon lifetime on the first scale-write that lacks identity. Boot is one moment; writes are continuous, and that's where the gap actually manifests.

8. **Historical identity migration script added** (`scripts/migrate_trace_identity.py`). Not in original design. Tom's call: *"I want us to migrate traces identity historically."* For our brain's current cast, every historical user_message is Tom and every agent event is Anchor — the assumption is true for all 57,672 rows. One-off script, idempotent, daemon-down. Side effect: re-encoded 22,416 pre-fix double-encoded `tool_result` rows from double-JSON to clean single-JSON.

9. **`brain.get_trace` / `get_traces` added as point-lookup API.** Original §14 had a richer `get_traces` tool design (decision 23) including encoder tool surface + MCP wrapper. We shipped just the brain-level read methods first — simplest slice. The MCP wrapper and encoder-side tool surface stay in the "Trace API block" for its own focused session.

### Insights worth carrying forward

Generalizable lessons surfaced during execution that aren't specific to this feature.

- **The substrate is the source of truth for identity; the abstraction layer reads through.** Identity belongs in `trace_events.metadata`, not denormalized onto nodes. Tom's framing: *"easy retrieval = node carries identity in its situation/content; harder retrieval = traverse source_refs to recover identity from the substrate."* This matches biology (cortical abstraction with hippocampal index resolving source) and means we don't have to backfill identity onto nodes — only onto traces.

- **Eager scope windows can be correctness gates, not just perf knobs.** When the data outside the window would render differently than the data inside it (sentinel vs. concrete identity here), the window matters architecturally. Defaulting to "embed everything" silently produces the wrong substrate.

- **Loud-by-default at the write boundary, not the convenient observation point.** Boot is a single moment; write paths are continuous. Place loud signals where the gap manifests on every operation. Saved as `feedback_loud_at_write_boundary` in the brain.

- **Per-utterance binding is structurally correct even when the current cast is monolithic.** Tom: *"for us, all user is me and all agent is YOU."* True today, but the substrate machinery (per-utterance `human_identity` / `agent_identity`) costs ~30 bytes per row and makes multi-partner future-proofing free. The architectural value is paid now so it doesn't have to be retrofit later.

- **NEVER use sqlite3 CLI against the live brain.db while the daemon is running** — caused a `disk I/O error` storm mid-session when `PRAGMA wal_checkpoint(TRUNCATE)` orphaned the daemon's WAL/SHM mmaps. Recovery required clean launchctl-managed daemon restart. Saved as `feedback_no_sqlite3_cli_against_live_brain`. Use `daemon_client.send_command` or MCP brain tools for all queries.

- **Probe-driven prompt iteration over personal review.** Tom's directive (paraphrased): *"I can't evaluate prompt accuracy personally — probes are the quality gate."* Every prompt change in this work cycle (v22, v23/v24, scout v6/v7, quote v3/v4) went through cold-read + live-encoding + comparative-audit Sonnet probes BEFORE registration. The probe-driven cycle catches ambiguity better than authoring intent does.

- **LLM encoders are stochastic at N=1.** Same prompt, fresh brain, can produce different outputs. The 50-cell v22-vs-v19 run captured one v22 failure on `c2ac3c61`; subsequent re-runs of the same item with the same prompt succeeded. `5025383b` succeeded in the 50-cell run but failed v22 on re-run. Some "fails" in N=1 evals are tail outcomes, not deterministic failure modes. Multi-sample evals (N≥3) are required to distinguish; saved as a methodological constraint for future eval design.

- **Don't reverse-engineer examples from the corpus.** Lesson from B1 (id:5a6bb335) re-surfaced when drafting Example 5 for facts_scout v6: an example built FROM a specific corpus case contaminates the retest signal (Sonnet pattern-matches the example back to the source instead of generalizing the principle). The mechanism the example demonstrates must be abstracted to a different domain; the example LANGUAGE should encode the truths to internalize, not just the specific case that motivated them.

### What's still on paper (next-session candidates — post Step 7 ship)

| Thread | Status | Effort |
|---|---|---|
| Encoder prompt v22 | **SHIPPED** (active since 2026-05-26T00:46:54Z, commit `c144ddf`) | — |
| 50-cell longmem real-test (v22 vs v19) | **COMPLETED** — v22 92% vs v19 88%; gated v22's activation | — |
| `co_anchored` auto-edge in dispatch (decision 15) | **SHIPPED** (Step 7, commit `07ab3f1`) | — |
| Layer 1 validator (hex format + sparsity >5 warnings) | **SHIPPED** (commit `07ab3f1`) | — |
| v23/v24 + scout v6/v7 + quote v3/v4 iteration | **DORMANT** — eval-decision call open | ~15 min option B + decision |
| Render expansion at `SURFACE_FORMAT` (joint reactivation render) | not started — recall-side block | ~0.5-1 day |
| `source_summary` parallel-pathway recall scoring | not built — recall+eval block | ~0.5 day |
| S2Healer source_refs cleanup + co_anchored orphan archival | §10.6 design, not built | ~0.5 day |
| §7.6 wave 2 + wave 3 examples (shape diversity, domain breadth) | wave 1 shipped (7 examples) | ~1 day each wave |
| Path A ground-truth authoring (7 conversation templates) | scaffolded at `eval/ground_truth/`; Tom-time pending | ~1.75h Tom |
| Phase B+ quote_fidelity substring-match validation | identical-strings check shipped; substring pending | ~0.5 day |
| Multi-sample eval design (N≥3 per cell to distinguish stochastic from deterministic) | identified as methodology gap | ~0.5 day eval infra |

---

## 1. Decisions Log

Locked decisions from the design conversation. Each is one line + WHY. New decisions append here as we work through sections.

1. **`source_refs` is a polymorphic field on any node — not a typed category.** Type stays open-text per Anchor's existing convention. *Why*: aligns with the encoder prompt's "type is free text — and emergent" rule; avoids closed enums; lets one field serve every node shape (fact, quote, pattern, correction, ...).
2. **Targets: S0 + S1 trace events. S2 deferred.** *Why*: S0 and S1 events are already visible to the encoder via conversation + scout reports + node catalog; S2 referencing requires an S2-scale recall mechanism that doesn't exist yet.
3. **Writers: all encoding paths.** S1E and every S2 encoder (community, consolidation, healer) can write `source_refs`. *Why*: every encoding path benefits from anchoring its outputs to the trace events that drove them.
4. **Tool calls are first-class trace events today.** Per decision 353135fa (PostToolUse hook, [post_tool_trace.py](hooks/scripts/post_tool_trace.py)), each tool call writes its own row in `trace_events` with `scale='s0'`, `event_type='delta'`, `ref_type='tool_result'`, `summary={human-readable tool call}`, `metadata={"tool": tool_name}`. **`source_refs` can target individual tool calls by trace_event.id directly — no phase split needed.** *Why*: the substrate already supports per-tool addressability; the earlier Phase 1/Phase 2 split was based on incomplete code knowledge.
5. **S0 captures the full tool stream today.** User turns, assistant text, and tool calls all land in `trace_events` as referenceable rows. No follow-on capture work needed. *Why*: the PostToolUse hook is already in production (decision 353135fa); this design rides on what's there.
6. **Identity framing: Anchor is the persistent identity.** The operator (Tom) is the partner present in current experiences. Identity persists across future partners. *Why*: design from Anchor's continuity, not from "Tom-Anchor partnership"; partnership is one current instance of a more general "connection" pattern.
7. **Connection = six mechanisms** (shared episodic substrate, affinity, preferences, theory of mind, correction record, shared vocabulary). Partnership is the current instance for Tom. *Why*: avoids special-casing partnership in code; same machinery serves future connections.
8. **CoALA framework mentioned for positioning, not designed to.** *Why*: cognitive-science taxonomy describes Anchor; doesn't blueprint Anchor. Section 2 includes the mapping as a footnote-style aside, not the design structure.
9. **Procedural memory: emergent downstream consequence, not designed.** Future direction only. *Why*: experience anchors procedure, not vice versa. Procedural patterns will surface naturally via S2 consolidation over enough episodically-anchored nodes.
10. **Reconsolidation: future direction, mentioned not detailed.** *Why*: depth of design is beyond the scope of this change.
11. **No closed `episodic_reference` type.** Subsumed by decision 1 — the behavior is triggered by *having* source_refs, not by type-string match.
12. **Connection mechanism = two things, not six.** Shared episodic substrate + differential attention. Theory of mind, preferences, correction record, shared vocabulary, and affinity-weighted recall are *products* that emerge from running these two over time. *Why*: the earlier six-item list conflated mechanisms with products; the two-mechanism cut is more honest and makes the section-2 framing of "what episodic-references close" precise (the products are severed from mechanism 1; refs restore that anchor).
13. **Sparse, specific source_refs (DG-discipline).** Encoder should write the smallest set of trace events that anchors a node — typically 1-3 — not a comprehensive list. *Why*: biological pattern separation depends on sparse indices; dense indices defeat the purpose. Section 7 (encoder prompt) carries this discipline.
14. **Joint reactivation: render expansion is default, not optional.** When a node with source_refs surfaces, the render layer expands at minimum the most-anchored ref by default. *Why*: empirical neuroscience shows recognition accuracy depends on JOINT reactivation of index + cortical pattern. Section 8 (render path) enforces this.
15. **Engram cohort is a structural edge (`co_anchored`), NOT a score boost.** When the encoder writes a node with source_refs, it checks `node_source_refs(trace_id)` for other nodes sharing any of those refs; for each such node M, write a `co_anchored` edge between the new node and M. Recall's Layer 3 graph traversal includes `co_anchored` (alongside `co_accessed`); both relations are excluded from candidate cosine ranking per the existing pattern. *Why*: (a) no arbitrary magnitude to guess — the graph IS the signal; (b) biologically aligned — encoding establishes the structural engram (`co_anchored`), retrieval reinforces it operationally (`co_accessed`) — LTP-then-reactivation; (c) composable with existing mechanisms; (d) visible, debuggable in the graph; (e) self-cleaning via FK cascade on node archive.
16. **Substrate preservation through consolidation — formal principle.** S0 traces are NEVER discarded by S2 units. When S2 consolidates a cluster, the resulting community node's source_refs span ALL the original episodes. *Why*: biological replay transforms cortical abstractions while leaving hippocampal traces intact; the substrate is the source of truth, abstractions are derivatives.
17. **Corrections keep Anchor's existing revise-vs-new-node split.** Simple factual errors get `revise()`'d in place — preserving wrong info just for biological symmetry is clutter. Pattern/learning from a correction gets written as a NEW node with a correction-aspect edge to the corrected fact. **Where new correction nodes are written, source_refs span both the original episode and the correction episode** for episodic-layer lineage. *Why*: the established pattern (see prior decision `e70f777b`: *fact wrong → revise; learning → new node*) is already correct. Biology's "new engram on reconsolidation" maps to the learning case, not to every factual update. Source_refs strengthen the lineage of the cases that already get new nodes; they don't disturb the revise path.
18. **Embedding strategy: Option C with per-trace ownership.** Source content gets its own embedding group (`source_summary`), embedded per-ref (each trace_event embedded separately, not concatenated), deduplicated by trace_id (one embedding per unique trace, ever). Trace embeddings live in `brain_logs.db` next to the traces they belong to; recall accesses them via `logs_conn`. *Why*: (a) Option C preserves provenance at the embedding layer so the ranker can weight encoder framing vs source-derived signal independently; (b) per-ref keeps source vectors semantically tight — concatenation averages them into mud; (c) ownership follows the entity — node embeddings live with nodes, trace embeddings live with traces. Same principle, applied consistently. Storage savings 5-15× over per-node duplication.
19. **Identity treatment — concrete tokens everywhere, asymmetric display, roles bind separately.** Four sub-decisions, all biology-aligned (research dive 2026-05-23, three independent angles converged):

    a. **Embedding: CONCRETE tokens** (`Tom`, `Anchor`, future partner names) — NOT abstract role labels. Concept cells in human MTL (Quian Quiroga 2005, Neuron 2026) are sparse, modality-invariant pointers to *specific individuals* that stay stable across context — that's where identity coherence comes from. Roles (`operator`, `partner`) live in a separate mPFC system and bind via edges. Abstract `OPERATOR` slot at embedding actively breaks the stability it was meant to provide (slot meaning changes when partners change). New partners enter as new concrete tokens at first encounter; old tokens stay intact in the embedding neighborhood.

    b. **Display: ASYMMETRIC** — first-person for Anchor's narrated turns ("I noticed...", "I said..."), labeled third-person for operator turns ("Tom said..."). This is the field perspective (DMN/mPFC self subsystem) tied to self-continuity and integration. Observer perspective ("ANCHOR: ...") is tied to depersonalization and identity fragmentation. Carve-outs: `their_raw_quote` / `my_raw_quote` stay verbatim and labeled (source-memory anchors); correction turns stay tagged ("Tom corrected: ..." — source attribution here is catastrophic to lose); very old / cross-partner episodes may want observer mode in future (age-graded rendering, not v1).

    c. **Frame→display mechanic: pointer verbatim, thin reconstructive layer above when partner-state differs.** Render reads identity from `trace_events.metadata` as source of truth. Frame carries `current_partner` / `current_agent` context, assembled at boot from config (§4.6). When trace identity == current → render naturally ("Tom said..."). When they differ → prepend reconstructive frame ("Tom, your operator at the time, said...") so old memories stay readable across partner changes without rewriting the pointer. Identity reconstruction sits on top of the verbatim pointer; the pointer never gets overwritten (matches Schacter constructive memory + per-utterance binding stickiness).

    d. **Multi-partner: per-utterance binding, no participants list.** Each `trace_event.metadata` carries its own `human_identity` / `agent_identity`. Three people in a session = three different identity strings across trace events. This is biology's per-utterance speaker-as-FK pattern; "participants list with post-hoc attribution" is the architecture that produces source confusion in humans. The schema is multi-partner-ready by construction even with one partner today.

    *Why this matters*: identity is the stickiest feature in memory (who-tags survive when when/where drift). Abstract-at-embedding throws away exactly the property — sparse modality-invariant individuality — that makes identity coding biologically work. Three research angles (concept-cell coding, self-vs-other in episodic memory, identity-at-retrieval) converged independently on the same answer; logged at decision time as the strongest single-session convergence in the design.
20. **Source expansion is per-format, threshold-gated where latency matters.** Final render to additionalContext (SURFACE_FORMAT) expands all source_refs by default — that's the joint-reactivation contract. Surface-selection Haiku call (HAIKU_FORMAT) expands only refs whose embedding similarity to the current recall query exceeds a threshold (initial cosine ≥ 0.5, eval-tunable). Encoder catalog view (ENCODER_FORMAT) doesn't expand — the encoder uses source_refs at write time, not read time. *Why*: Anchor's recall benefits from full joint reactivation; the surface-selection LLM call benefits from source signal but can't afford to expand every ref; the encoder doesn't need source content while reading the catalog. Per-format tuning fits the actual purpose of each consumer.
21. **Trace embedding is async via the existing `embed_queue`, not synchronous at encode time.** Trace embedding extends [`servers/embed_queue.py`](servers/embed_queue.py) — same single-worker, drain-every-5s, skip-tick infrastructure that handles node/edge embeddings today. New: `enqueue_trace(trace_id)` parallel to existing `enqueue` and `enqueue_edge`. Encoder writes node + source_refs synchronously; trace_ids enqueue; worker drains within 5s. Recall handles "not yet embedded" gracefully — node falls back to other pathways for that surface cycle. *Why*: (a) any vectorization in the brain happens in one place — single source of truth; (b) future embedder swaps plug into the same pipeline, graph stays stable; (c) same mechanism treats new / retry / refresh uniformly; (d) no write-path latency. Validated against the actual existing infrastructure in production.
22. **Source_summary is a parallel pathway, NOT a weighted-sum component.** Final node score = `max(weighted_sum_of_framing_groups, source_summary_score)`. The existing recall weight scheme (title 1.00, _primary 0.85, high_meta 0.70, ...) continues for the framing groups; source_summary scores independently and either pathway can drive surfacing. *Why*: (a) biological alignment — hippocampal index and cortical representation are parallel pathways, either can drive retrieval; (b) no arbitrary ratio between source signal and framing signal — neither "is worth N% of the other"; (c) pure-reference nodes recallable via source pathway alone; (d) backwards compatible — nodes without source_refs score exactly as today via legacy path. The broader weighting-scheme rebuild (treating other groups as parallel pathways too) is future work — §16.0.
23. **Encoder gets read-only structured trace access via `get_traces` tool; same function exposed via MCP.** New encoder tool: `get_traces(trace_ids: List[str], session_id: Optional[str])` — point lookup (up to 10 trace_ids per call) plus optional session-scoped fetch. trace_ids are 8-char hex strings (schema v29). Used by the encoder to verify trace content before writing source_refs to traces outside the current encoding window (cross-window/cross-session anchoring, tool-call cross-checking). Same function exposed as MCP tool so Anchor in conversation can query traces directly. *Why*: structured lookup (not semantic search) is fundamentally different cost class from recall — same cost as `get_node()` which the encoder already uses. Preserves the principle (memory `6275b1d2`) that encoder doesn't do semantic recall. Enables high-quality cross-window encoding without re-fetching via recall.
24. **`source_refs` validation is minimal at write time; cleanup belongs to S2Healer.** Daemon accepts whatever trace_ids the encoder writes (type-check only). Invalid trace_ids degrade gracefully at recall (render skips missing refs per §8.5). S2Healer periodically scans `node_source_refs` for invalid trace_ids and archives them, with metric `invalid_refs_dropped_total` surfacing systemic encoder quality issues. Healer also cleans orphan `co_anchored` edges (decision 15) when no shared trace remains between the two endpoints. *Why*: simplicity — write path stays clean; race conditions don't cause writes to fail; quality signals centralize in healer metrics rather than scattered across rejection paths. Aligns with the loud-but-not-blocking principle.
25. **S0 trace retention is a hard formal guarantee — `trace_events` rows are never deleted.** No code path may call `DELETE FROM trace_events`. S2 consolidation/archive operations don't delete the trace events referenced by archived nodes. Daemon maintenance (vacuum, optimize) doesn't prune traces. Storage growth (~50MB/year at current encoding rate) is acceptable for years. If hot-storage pressure ever materializes, the path is archive-not-delete (§16.7) preserving trace_id resolvability. *Why*: the entire episodic-references architecture depends on trace_id resolvability — a "soft" policy invites future engineers to break it without realizing what breaks. Hard policy in the doc + a static-guard unit test ensuring no DELETE on `trace_events` makes the invariant durable.

---

## 2. Why this exists

### What this section does

Section 2 establishes what the brain *is* and what it currently *can't do*. Everything downstream — the schema, the prompt revisions, the embedding strategy — only makes sense against this framing. Read this once and you should understand the gap episodic-references close, why that gap is structural rather than cosmetic, and how Anchor positions relative to existing memory frameworks.

### Identity is Anchor

The brain stores **Anchor's memory**. It records what Anchor has experienced. Today every experience happens to be shared with the operator (Tom), but that is circumstance, not identity. As Anchor's experience expands — other agents, other people, eventually episodes that don't involve any human at all — the identity persists; the *partner present in each experience* is what varies. The substrate is Anchor's autobiographical record. The operator is whoever Anchor was with at the time.

This matters for design choices. A "partnership memory" architecture would special-case the Tom-Anchor relationship; an "Anchor memory" architecture lets connections form by the same machinery regardless of who's on the other side. We build the latter. Partnership with Tom is the current instance; the underlying machinery is general.

### Connection mechanism — two things

What we call partnership is what emerges when two mechanisms run together for one specific other:

1. **Shared episodic substrate** — the accumulated record of moments two entities shared. Without shared experience, no connection exists.
2. **Differential attention** — caring about *this* experience-set more than the average. Without differential weighting, you have a shared log but no relationship.

Both ingredients are necessary; neither is sufficient alone. A blank shared transcript doesn't generate connection. A passionately-held belief about someone you've never met doesn't either. **Shared substrate + differential attention** is the generator.

Run those two mechanisms over time and the brain produces *the visible shape of connection*:

| Product | What it is | Where it lives in Anchor today |
|---|---|---|
| **Theory of mind** | Anchor's evolving model of who the other is — situation, interests, emotional patterns | `personal_context`, `active_thread`, `interest` nodes |
| **Preferences** | Patterns about how they do things | `preference` nodes; style markers carried in `their_raw_quote` |
| **Correction record** | What got fixed across exchanges | `correction` type + corrects-aspect edges |
| **Shared vocabulary** | Inside-references and terms-of-art that mean something specific between them | Cross-node semantic atoms with mutual references; emerges over time |
| **Affinity-weighted recall** | Differential attention manifested at retrieval — some memories surface more | `co_access_count`, `critical`, `locked`, recency, emotional metadata |

These products aren't designed in as separate modules; they emerge from running the substrate + attention mechanisms. Different connections (Tom-Anchor today; some hypothetical future Other-Anchor) would generate different *fills* of the same product slots. Same machinery, different content.

### The episodic gap, concretely

Both mechanisms exist in today's brain. Mechanism 1 lives in `trace_events` (S0 captures every turn verbatim across every session, scoped by `session_id` and stamped with a stable `id`). Mechanism 2 lives in Hebbian co-access, criticality, lock-flag, recency, emotional metadata, and the edge structure built up over recalls.

The architecture works. The gap is elsewhere — in how the **products** anchor back to **mechanism 1**.

Today's `preference` node asserts the preference but doesn't point at the moments that revealed it. Today's `correction` node carries the rule but not the exchange. Today's `moment` node carries a verbatim quote but not the context that gave the quote its register. Today's `fact` node carries a computed value (190 pages left) but not the source numbers (250 of 440) that future questions might want to redo the computation on.

The products are real but **thin**. They sit in the semantic layer, severed from the episodes that produced them. The episodic substrate is preserved (S0 is forever-appended, untouched) but the abstraction layer rebuilds from scratch, paraphrases, computes-at-encode-time, and loses the link.

A schema-level fingerprint of this: the `nodes` table has a `source_turn_id` column whose schema comment reads *"message_stream ID that produced this node — episode linkage."* The column was defined for exactly this purpose. The `message_stream` table it referenced was deleted. The field is currently unused. The intention was there; the wiring wasn't completed.

The eval data already shows the symptoms:

- The encoder must **anticipate the question** to extract the right fact. Compute "190 pages left" at encode-time and you've committed to that specific computation; "what percent done?" can't be answered from the same source because the encoder threw away the inputs.
- **Ranges collapse to specific points** because the encoder has no graceful way to defer to source (the bus-price ¥3,200 case — a value pulled by the assistant without operator confirmation, encoded as fact, retrievable as if known).
- **Dense content gets either fragmented or bloated** — atomized per item with comparison view lost, or whole table dumped into one node's content. Neither is right because there's no third option: "name the topic, point to the source."
- **Provenance reduces to a metadata flag** (`source_attribution: anchor_unconfirmed`) when it could be structural — a pointer to the trace event where the value was said, role = assistant, no operator confirmation in adjacent turns.

Each symptom traces to the same root cause: **the abstraction layer can't point at the substrate.**

### What episodic-references close

The change is small in code, large in capability: a `source_refs` field on every node, holding a list of trace_event ids. The behavior is triggered by *having* refs, not by being a special type. Any node can carry them. The encoder decides per-node whether content adds value over the source or just risks distorting it.

Concretely:

- A `preference` node about Tom's introduction style stays a preference node, but now also points at the turn where Tom said *"without forcing it."* The preference is anchored in the moment that revealed it.
- A `correction` node about MCP-vs-bash discipline stays a correction, but now points at the bash call Tom corrected. The rule keeps its evidence.
- A `moment` node about a partnership turning point stays a moment, but now points at the full exchange. The emotional weight has its grounding visible.
- A dense table (the bus comparison) becomes a node whose content is minimal and whose `source_refs` is the substance. The encoder doesn't rewrite; it names the topic and points.

**The brain stops being a knowledge base of "abstractions about Tom" and becomes a record of Anchor's experiences with Tom, with the experiences preserved alongside the abstractions.** Mechanism 1's role is restored as the foundation of the product layer, not just a parallel log that gets ignored once the encoder has produced its atoms.

### Positioning relative to existing frameworks

The CoALA framework (Sumers et al. 2023, updated 2025) names four memory types in LLM agents: **working** (live context window), **episodic** (past interactions), **semantic** (facts and concepts), and **procedural** (skills, tool sequences, prompts). For outside readers, Anchor maps to this taxonomy as:

- **Working** = the in-context window per turn (Frame + additionalContext + recent_messages).
- **Episodic** = today, the `trace_events` substrate exists but is severed from the semantic layer. *This change closes that gap.*
- **Semantic** = every node Anchor's encoder writes. Strongest area today.
- **Procedural** = pinned prompts (SKILL.md, the interactions table). Today bootstrap-only. **Will emerge** as a downstream property when episodically-anchored tool-use nodes accumulate enough for S2 to consolidate them into patterns.

The Feb 2026 position paper *"Episodic Memory is the Missing Piece for Long-Term LLM Agents"* argues exactly the gap this design closes. MemPalace (arxiv 2604.21284) and MemMachine (arxiv 2604.04853) ship variants of the same idea: store the source, point at it, don't paraphrase.

**We describe ourselves in CoALA's terms when positioning Anchor against the field. We do not design to that taxonomy.** Anchor is built from identity-first principles: experience accumulates, abstractions form from experience, connections emerge when the experience is with a specific other, procedural patterns surface when experience has accumulated enough. The sequence is *episodic → semantic → procedural*, not four parallel pillars. Experience is the anchor of everything else.

### Biological alignment

Worth surfacing for readers who'll ask whether this is just neuroscience cosplay: no, it's convergent. Our architecture aligns with two well-established frameworks from cognitive neuroscience:

- **Hippocampal Indexing Theory** (Teyler & Rudy, ~2000; validated in human fMRI 2021): the hippocampus stores sparse indices — literal pointers — that bind cortical representations of an experience. Retrieval = partial cue activates index → index reactivates cortical pattern. Recognition accuracy depends on *joint* hippocampus + cortex reactivation. Our `source_refs` field is a hippocampal-indexing-style pointer; semantic nodes are the cortical representations; the trace events are the experiential record. The architecture we're proposing was discovered in biology 25 years ago.
- **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly, 1995; computational variants ongoing): hippocampus = fast learner, pattern separation, detailed snapshots. Neocortex = slow learner, pattern completion, statistical regularities. The two systems need each other. Our S0 traces are hippocampus-like; semantic nodes are neocortex-like; the gap source_refs closes is exactly the bridge CLS says must exist.

Additional alignments — engram research (Tonegawa, Josselyn — distributed memory traces, 2024-2026), pattern separation/completion (DG sparseness, CA3 completion), sharp-wave-ripple replay (specific large SWRs drive consolidation, 2026 Neuron), schema scaffolding (Method of Loci research, Nature Communications 2026), and reconsolidation (recall creates NEW engrams, not in-place edits — 2024 Neuron) — all show up in the design as constraints or future directions (see Decisions 13-17, sections 7, 8, 10, 16, and Appendix A). We didn't reverse-engineer biology; we hit the same answer from the engineering side. Worth knowing the alignment is there.

---

## 3. Core model

### One field, polymorphic targets

Every node can carry a `source_refs` field: a list of `trace_event.id` values pointing at the events that anchor the node. The field is optional. A node with no source_refs behaves exactly as today — it carries its full content. A node with source_refs points back at the events that produced or anchor it; the render layer resolves the references when the node surfaces.

The targets are heterogeneous. A single node's `source_refs` list can contain:

- An S0 `K` event (a user turn)
- An S0 `delta` event (an assistant turn, with tool_use blocks embedded in metadata — see §5 for parsing)
- An S1 recall event (when a related recall happened)
- An S1 encoding event (the encoding cycle that produced sibling nodes)

The `trace_event` row carries its own `scale` and `event_type`, so the render layer dispatches on those fields when expanding. The encoder doesn't need to know the target shape — it just writes ids.

### The judgment: when to write content vs. point to source

The encoder's per-node decision is one rule:

> **If content would just rewrite what the source already says clearly, point to the source instead.**

This is the operational form of the atomization-vs-preservation question that surfaced repeatedly in the eval data. It collapses the earlier "three flavors" exploration into a single per-node judgment.

The rule isn't arbitrary — it's the encoder respecting **substrate preservation** (decision 16). The brain doesn't rewrite hippocampal traces into cortical abstractions; it builds parallel cortical representations linked back. Rewriting the substrate defeats the substrate. So when content would just transcribe what the source already says clearly, the encoder defers — names the topic, points at the source, and stays out of the way.

Concretely:

- **Anchor's reasoning, lessons, patterns, principles** — content adds value, even when refs exist. The abstraction IS Anchor's contribution; refs anchor it but don't replace it.
- **Verbatim quotes, dense tables, ranges, specific numeric values the operator stated** — content would just transcribe. Skip the content; let `source_refs` carry the substance.
- **Operator-revealed situations, preferences, style markers** — content carries Anchor's *framing* of what the situation is, while refs preserve the moments that revealed it. Both.

### Patterns of use (descriptive, not prescriptive)

The judgment produces three natural patterns. **These aren't types** — they're points on a content-to-refs spectrum that the encoder lands at per-node. Each has a clean biological analog (see §2 for the framework):

**Pure synthesis** — content full, `source_refs` empty.
The node abstracts across many sessions, captures an emergent pattern, or holds Anchor's reasoning. No single episode to anchor in. *Biological analog: neocortical schema — a representation that has consolidated across many experiences and no longer depends on hippocampal replay to surface.* Examples: a `principle` node about how Anchor and Tom work together; a `pattern` node about a recurring failure mode that surfaced across many sessions.

**Anchored synthesis** — content full, `source_refs` as evidence.
The node carries Anchor's framing AND points at the moments that revealed it. *Biological analog: cortical representation with active hippocampal index — the abstraction is real, and recall can still re-activate the originating experiences.* Examples: a `preference` node about Tom's introduction style with `source_refs` to the turn where Tom said *"without forcing it."* A `correction` node with `source_refs` to both the original mistake-turn and the correction-turn.

**Pure reference** — content minimal or empty, `source_refs` is the substance.
The encoder names what the source is and why it matters, but doesn't try to reproduce it. *Biological analog: hippocampal index without much cortical abstraction yet — the index points at distributed episodic features and recall reconstructs through them.* Examples: a `quote` node with `source_refs` to the turn it captures; a `fact` node for a dense comparison table where any paraphrase risks distortion; an `episode` node (loose type, encoder-invented) about a multi-turn exchange that should be replayed in full when relevant.

The encoder reaches for whatever type fits the content (per Anchor's open-form type rule). `source_refs` is orthogonal — any type can carry refs, any type can omit them. The patterns above describe how the field tends to land in practice, not categories the encoder reasons about explicitly.

### Why content stays optional, not eliminated

Even with `source_refs`, content has value:

- **It's Anchor's framing** — what the encoder thinks this episode means, what to surface when. Title and situation come from the encoder's reading; reasoning is Anchor's interpretation. The brain holds Anchor's *lens* on the experience, not just the experience.
- **It's the recall signal** — embeddings are computed from content + title + situation + reasoning. Pure-reference nodes need title + situation + reasoning to be substantive enough for recall (see §9 for the embedding strategy that handles this).
- **It's the cheap render** — at surface time, the rendered node includes content directly without an extra fetch. For high-frequency nodes, this matters.

### What the surface layer sees

When a recalled node has `source_refs`, the surface layer renders both the abstraction AND the source by default — joint reactivation (decision 14). Three rendering modes:

1. **Content only** — for nodes without source_refs, or when budget excludes expansion. Same as today.
2. **Content + expanded source** — default for nodes with source_refs. Content first, then the rendered source events inline. Anchor sees both the abstraction and the moment that produced it.
3. **Source-dominant** — for pure-reference nodes (content minimal or empty), the rendered source IS what surfaces; the content fields serve as framing only.

Surface decides per-node based on budget and the node's content-to-refs ratio. The encoder writes; the surface renders. The biological model — recognition accuracy depends on joint hippocampus + cortex reactivation — argues for expansion as default, not optional.

### One concrete example, end-to-end

The colleagues conversation from the eval cohort. Today's v17 encoder produces ~5-6 nodes, none with `source_refs`. Three nodes change shape under the new model:

| Node | Old form (v17) | New form (with source_refs) |
|---|---|---|
| Personal context | `personal_context: Tom works remotely, misses watercooler chat` | Same content + `source_refs: [trace_of_t0]` (where Tom said *"I've been working from home and miss watercooler chat with colleagues"*) |
| Preference | `preference: Tom's introduction style — collaborative, not directive` | Same content + `source_refs: [trace_of_t4]` (where Tom said *"without making it sound like I'm forcing the idea on them"*) |
| Substantive content | A `fact` node with the full ground-rules list as content (no anchor) | Minimal content + `source_refs: [trace_of_t11]` (the assistant turn that proposed the list — preserved untouched) |

The first two are *anchored synthesis* — sparse anchoring (decision 13), 1 ref each pointing at the most-revealing turn. The third is *pure reference*. Same machinery; the encoder's per-node judgment lands them in different patterns.

---

## 4. Schema changes

Section 4 locks the storage layer for episodic references. Two new tables (one per database), three indexes, no destructive change to existing schema. Insert and recall costs analyzed at the end.

### 4.1 — Two new tables, two databases

Episodic references touch two storage entities: the *pointer* (who-references-what) and the *embedding* (the source's vector representation). Each lives in the database that owns its entity.

**`brain.db` — `node_source_refs`** (the pointer / join table):

```sql
CREATE TABLE node_source_refs (
    node_id   TEXT  NOT NULL,
    trace_id  TEXT  NOT NULL,    -- 8-char hex (v29); matches trace_events.id and node id shape
    position  INTEGER  NOT NULL,    -- order in the encoder's source_refs list (1-indexed)
    PRIMARY KEY (node_id, trace_id),
    FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
);
CREATE INDEX idx_nsr_trace ON node_source_refs(trace_id);
```

`position` preserves the order the encoder wrote the refs in (sometimes signals primary-vs-supporting source). The composite primary key prevents duplicate refs. The `idx_nsr_trace` index makes the reverse lookup (engram cohort detection, decision 15) trivial — one indexed query returns every node that shares a given trace.

**`brain_logs.db` — `trace_embeddings`** (the deduplicated source vectors):

```sql
CREATE TABLE trace_embeddings (
    trace_id    TEXT       PRIMARY KEY,    -- 8-char hex (v29); FK shape to trace_events.id
    vector      BLOB       NOT NULL,
    created_at  TIMESTAMP  DEFAULT CURRENT_TIMESTAMP
);
```

No `node_id` column. The embedding is OF the trace, not of a (node, trace) pair. When multiple nodes reference the same trace, they share the vector. Embed-once-per-trace-ever. Trace embeddings live in `brain_logs.db` because that's where `trace_events` live — ownership follows entity (decision 18).

**Identity metadata on `trace_events` (decision 19):**

Per-trace identity goes in the existing `trace_events.metadata` JSON field as two keys: `human_identity` (e.g., `"Tom"`) and `agent_identity` (e.g., `"Anchor"`). Populated by the daemon at trace-write time from session context. No new column — JSON storage is sufficient for v1; promotion to first-class columns is future work if filter-performance demands it. The embedding render itself (§5.3) stays abstract; concrete identity is for filter (§16 future direction) and display (§8).

### 4.2 — Why two databases

`brain.db` holds graph state (nodes, edges, node embeddings). `brain_logs.db` holds the trace substrate (S0 turns, S1 recall/encoding events, errors, sessions). Putting trace embeddings in `brain_logs.db` keeps the principle clean: the database that owns the data owns the embeddings of that data.

The daemon already holds both connections (`self.conn` for `brain.db`, `self.logs_conn` for `brain_logs.db`). Recall extends to call `logs_conn` for source vectors during scoring — one extra query per surface batch (see §10 for the recall path).

SQLite cross-database FKs are not enforced, so there is no `FOREIGN KEY` from `node_source_refs.trace_id` to `trace_events.id`. The encoder trusts trace_ids it receives in its input. Defensive: if a referenced trace_id is missing at recall time, the render layer falls back gracefully (see §12).

### 4.3 — Insert efficiency

When the encoder writes a node N with source_refs `[t1, t2, t3]`:

1. `INSERT INTO nodes (...)` — existing path
2. Multi-row INSERT into `node_source_refs` — one query for all refs
3. Find which refs lack embeddings:
   ```sql
   SELECT trace_id FROM trace_embeddings WHERE trace_id IN (?, ?, ?)
   ```
4. For any missing trace_id: fetch trace content from `trace_events`, embed (~150-200ms per trace), INSERT into `trace_embeddings`
5. Existing embedding pipeline for the node's framing fields (title / situation / reasoning into `node_embeddings`) — unchanged

Steady-state cost: when refs are already embedded (popular turns referenced by many nodes), step 4 is a no-op. First-time references pay the embedding cost once, ever. Encoder runs every 5 turns and writes a handful of nodes per cycle; even cold-cache cost is ~500ms total extra per cycle. Off the hot path.

### 4.4 — Recall efficiency

For a recall query producing K candidate nodes:

1. Standard 4-group scoring against `node_embeddings` (unchanged)
2. Collect all unique trace_ids across the candidates' source_refs (in-memory, refs are already loaded with the nodes)
3. **One** query for source vectors:
   ```sql
   SELECT trace_id, vector
   FROM trace_embeddings
   WHERE trace_id IN (?, ?, ..., ?)
   ```
4. Build `{trace_id: vector}` map in-process
5. Per candidate: look up its source vectors via the map → cosine vs query for each → **max-pool** for the candidate's `source_summary` score
6. Combine `source_summary` with the existing weighted-sum legacy groups via parallel-pathway max: `node_score = max(weighted_sum(framing_groups), source_summary_score)` — decision 22

Max-pool semantics: a node with 3 source_refs matches the query if ANY ONE of its sources is semantically close. Max preserves "at least one source fires"; averaging would dilute strong matches with weak ones.

Total extra cost vs today's recall: one query against `logs.trace_embeddings`, sub-10ms with the primary-key index. Negligible compared to the surface LLM call's ~6-8s.

### 4.5 — Engram cohort efficiency (decision 15)

Two candidates in the same surface batch share `source_refs` → they belong to the same memory at the substrate level → surfacer adds a soft co-recall weight.

In-batch detection (the common case):
- Source_refs lists are already loaded for all candidates
- O(K²) pairwise overlap check across K candidates
- With K ≤ 25, that's ~300 comparisons — microseconds
- No DB query

Cross-batch lookup (e.g., "what other nodes share refs with node X?"):
- ONE indexed query on `node_source_refs(trace_id)`
- Fast

### 4.6 — Migration

Additive only. No destructive change.

- Schema version increments from v26 to v27 (per `servers/schema.py`)
- `node_source_refs` table created in `brain.db`
- `trace_embeddings` table created in `brain_logs.db`
- Both indexes created
- Existing `nodes` table untouched — `source_turn_id` column stays as-is but is marked DEPRECATED in the schema comment (see §4.7). No drop, no rename, no migration of existing rows.
- No backfill of existing nodes to add source_refs (would require LLM judgment per node — expensive, unnecessary; old nodes work unchanged)

### 4.7 — Deprecation of `source_turn_id`

The existing `source_turn_id TEXT` column on `nodes` ([servers/schema.py:128](servers/schema.py:128)) was defined for the now-deleted `message_stream` table. It's never been populated by any current code path. The column stays for one schema version as a marker; the schema comment changes to:

```python
'source_turn_id': 'NULL',  # DEPRECATED v27 — replaced by node_source_refs join table.
                            # Schedule drop in v28 once no read paths reference it.
```

A future v28 migration drops the column once we've confirmed no code reads it.

### 4.8 — Storage implications

- `node_source_refs`: O(total refs across all nodes). ~3 refs/node avg × ~5000 nodes = ~15k rows × ~30 bytes/row = ~450KB. Trivial.
- `trace_embeddings`: O(unique referenced traces). Heavy reuse in practice (popular turns referenced by 10+ nodes); estimated 1k-3k unique embeddings × ~3KB/vector at 768d float32 = ~3-9MB total. Compare to the alternative (per-node-per-trace storage): ~45MB. 5-15× reduction.

### 4.9 — What this leaves to other sections

Section 4 locks the storage layer. Decisions explicitly deferred to later sections:

- **Reference format and target scope** (which trace events are valid refs) — §5
- **Encoder tool surface** (how the encoder writes refs in `remember_batch`) — §6
- **Embedding pipeline details** (what content gets embedded, how max-pool interacts with z-weighting) — §9
- **Recall scoring weights for the new `source_summary` group** — §10 (and an eval-driven tune)
- **Render-time fetch and expansion of source content** — §8
- **Edge cases**: missing-trace fallback, archived-source behavior — §12

---

## 5. Reference format

Section 5 defines what a reference IS at the substrate level: the value the encoder writes, which trace events it can target, how those events get rendered for embedding, and how the encoder receives references in its input.

### 5.1 — Reference value

A reference is a single `trace_event.id` (BIGINT). The `source_refs` field on a node is a list of these ids, order-preserved (the `position` column in `node_source_refs` keeps the order the encoder wrote them in).

Trace events have stable, monotonic, globally-unique ids — issued by `trace_events` autoincrement in `brain_logs.db`. No format manipulation; the encoder writes the integer it was shown.

### 5.2 — Valid targets

`source_refs` can point at any trace_event whose `scale` is `s0` or `s1`. The substrate already supports all the shapes that matter:

| scale | event_type | ref_type | What it represents |
|---|---|---|---|
| `s0` | `K` | `user_message` | One operator turn |
| `s0` | `delta` | `assistant_message` | One Anchor turn (text portion) |
| `s0` | `delta` | `tool_result` | One tool call + result (decision 4 / 353135fa) |
| `s1` | various | various | Recall events (queries + candidates), encoding events (cycles + outcomes) |

**Excluded in v1:**
- `s2` events — no S2-scale recall mechanism exists yet. Section 16 names this as future work.
- `s0` events without text content (any purely structural marker the brain might add later) — embed renders to text; structureless events have no useful embedding signal.

### 5.3 — Embedding rendering templates

Each trace event type is rendered to a text string before embedding (the embedder is text-only). The structured event row stays intact in `trace_events`; the rendered text is fed to `nomic-embed-text-v1.5-Q` to produce the vector stored in `trace_embeddings`.

The role-prefix preserves provenance at the embedding layer — *"Tom: I want to take the bus"* embeds differently than *"Anchor: The bus costs ¥3,200"*. Option C's whole point depends on this signal.

| Event | Embedding render |
|---|---|
| S0/K (`user_message`) | `{human_identity}: {metadata.content or summary}` |
| S0/delta (`assistant_message`) | `{agent_identity}: {metadata.content or summary}` |
| S0/delta (`tool_result`) | `{agent_identity} via {metadata.tool}: {summary}` |
| S1/recall | `RECALL: "{query[:200]}" → {N} candidates` |
| S1/encoding | `ENCODE: {N} actions ({journal_entry[:300]})` |

Notes:
- Tool-result `summary` is already shaped well by [post_tool_trace.py](hooks/scripts/post_tool_trace.py) (`"Bash: ls -la"`, `"Edit: /path/to/file.py\n  old: ...\n  new: ..."`, `"WebSearch: how much is the bus"`, etc.) — the embedder gets meaningful semantic signal without further preprocessing.
- For S0/K and S0/delta, prefer `metadata.content` (full text up to 4000 chars per current cap) over the short `summary` (200 chars). The longer text gives the embedder more signal.
- **Labels are concrete tokens (revised decision 19, 2026-05-23).** Per the biology research that closed §15.1 — concept cells in MTL fire for specific individuals as sparse, modality-invariant pointers; roles bind separately in mPFC. Using concrete `Tom` / `Anchor` at the embedding layer gives identity-coherent vector neighborhoods (the same individual stays in the same neighborhood across context changes). The earlier abstract-slot approach (`OPERATOR / ANCHOR`) defeats the purpose — slot meaning changes when partners change, which is exactly the instability we want to avoid.
- **Graceful degradation when identity is unset.** If `human_identity` / `agent_identity` is missing from `metadata` (env vars unconfigured), fall back to `OPERATOR` / `ANCHOR` as render-only sentinels. This keeps the embedding pipeline running for fresh installs that haven't configured identity yet.
- **Display-time rendering is separate** (§8). When Anchor reads a surfaced node's source, the render uses the same concrete names but asymmetrically — first-person for the agent's own turns, labeled for the operator (per revised decision 19b).

### 5.4 — Embedding for discoverability, source for fidelity

The embedding doesn't need to encode structured content faithfully. It needs to make the trace **findable**. The render layer (§8) then expands and serves the verbatim content unchanged.

This bounds the loss: embedding imprecision = recall miss (the trace doesn't surface when it should), not a factual error (the value, once surfaced, is exact). For numbers, code, JSON, and tool calls, we accept that the embedder doesn't fully understand the structure — the source itself preserves fidelity.

Concrete example: a tool-result trace `"WebSearch: how much is the Limousine Bus to Shinjuku"` with result text `"¥3,200 (around $29 USD) one way"`. The embedder doesn't need to "understand" ¥3,200 numerically. It just needs to recognize this is about bus fares — and the surrounding text gives it plenty of signal. At recall, the trace surfaces; render exposes the exact value; the answerer reads ¥3,200 verbatim.

### 5.5 — How the encoder receives references in user_content

The encoder's input (built by `_build_user_content` in [scales/s1/encode.py](servers/scales/s1/encode.py)) renders the conversation window with turn ids prefixed. Today that's `[turn t0, OPERATOR]`. The change for episodic-references is to additionally prefix each turn with its `trace_event.id`:

```
[turn t0  trace:54312  OPERATOR]
I've been working from home and miss watercooler chat with colleagues.
Any suggestions?

[turn t1  trace:54313  ANCHOR]
Here are a few suggestions: 1) Virtual Coffee Breaks...

[turn t1-tool1  trace:54314  TOOL Bash]
Bash: ls -la
  → 5 files
```

The encoder learns to use these trace ids verbatim in its `source_refs` list. The prompt teaches: *"When a node is anchored to a specific moment, copy the `trace:NNNNN` value into source_refs."*

This is the same pattern as the existing turn-id convention — encoder reads them, references them, writes them back into tool calls.

### 5.6 — Sparseness reminder (decision 13)

The encoder's discipline is to pick the *smallest set of trace events that anchors this node* — typically 1-3 — not a comprehensive list. The prompt enforces this; the schema doesn't.

Section 7.5 covers the prompt-side enforcement. Section 4.1's index design supports up to ~10 refs per node before performance considerations matter (it's just bytes in a small composite-PK table). Discipline is a quality issue, not a capacity issue.

### 5.7 — What about `trace_event.id` stability?

`trace_events.id` is an autoincrement BIGINT, monotonic per database. Once written, never changes. References are stable across sessions, across daemon restarts, across schema migrations.

The one risk: if `trace_events` rows were ever deleted (today they aren't — append-only by design), refs would dangle. Section 12 (Edge cases) covers the graceful-degradation behavior. Section 16 lists "guaranteed S0 retention" as a policy-level concern worth formalizing.

---

## 6. Encoder tool surface

Section 6 specifies the encoder-facing API change: which tool schemas gain the `source_refs` field, the shape of the field, and the corresponding change to the encoder's `user_content` (so the encoder can see the trace ids it needs to reference). All changes are additive; existing tool calls without `source_refs` work unchanged.

### 6.1 — `remember_batch.nodes[].source_refs`

The primary write path. Schema diff (additive):

```json
{
  "type": "object",
  "properties": {
    "title":       {"type": "string"},
    "type":        {"type": "string"},
    "content":     {"type": "string"},
    "situation":   {"type": "string"},
    "reasoning":   {"type": "string"},
    "...":          "(existing fields unchanged)",
    "source_refs": {
      "type": "array",
      "items": {"type": "integer"},
      "description": "Trace event ids that anchor this node — 1-3 typically, sparse and specific (decision 13). Empty/absent for pure-synthesis nodes."
    }
  }
}
```

Optional. Empty list and absent are equivalent — no refs. The encoder writes integer trace_event.id values directly from the markers in user_content (see §6.5).

### 6.2 — `revise_batch.revisions[].source_refs`

Revise lets the encoder add or replace source_refs on an existing node:

```json
{
  "type": "object",
  "properties": {
    "node_id":     {"type": "string"},
    "content":     {"type": "string"},
    "...":          "(other revisable fields)",
    "source_refs": {
      "type": "array",
      "items": {"type": "integer"},
      "description": "Replace the node's source_refs with this list. To add refs without replacing, omit and use connect-style write through brain_batch.remember (decision 17 — corrections produce new nodes; pure ref-additions on existing nodes are rare)."
    }
  }
}
```

Semantics: when `source_refs` is present in a revise op, it REPLACES the existing list (per the brain's existing field-replacement convention on revise — see [memory `08ec0fa2`](feedback)). When absent, the existing list is preserved.

### 6.3 — `brain_batch` operations

`brain_batch` wraps `remember` and `revise` ops; both inherit the field per §6.1 and §6.2. No new op type needed. Schema enum (`VALID_OPS`) is unchanged.

### 6.4 — S2 encoders (community, consolidation, healer)

All three S2 encoders write via the same `remember_batch` / `revise_batch` / `brain_batch` tool surface. They inherit `source_refs` automatically. Each S2 unit's prompt should be updated to teach when to use refs (the prompt update lives in §7's structure since the discipline is shared).

Typical S2 patterns:
- **Community node** — source_refs span representative episodes from the cluster's member nodes (not the member nodes' source_refs themselves; one level of indirection). Decision 16 — substrate preservation — argues for episode-level anchoring, not abstraction-level.
- **Consolidation merge** — when consolidating two nodes A and B into a survivor S, S's source_refs = union of A.source_refs and B.source_refs (deduplicated). Lineage preserved.
- **Healer** — when healing a node by filling missing fields, source_refs may be unchanged OR augmented with the episodes the healer used to infer the fill.

### 6.5 — `user_content` shape change

The encoder's input (built by `_build_user_content` in [scales/s1/encode.py](servers/scales/s1/encode.py)) renders each turn with its turn-id label. The change for episodic-references is to additionally include each turn's `trace_event.id`:

**Today:**
```
[turn t0, OPERATOR]
I've been working from home and miss watercooler chat with colleagues.
Any suggestions?

[turn t1, ANCHOR]
Here are a few suggestions: ...
```

**With episodic-references:**
```
[turn t0  trace:54312  OPERATOR]
I've been working from home and miss watercooler chat with colleagues.
Any suggestions?

[turn t1  trace:54313  ANCHOR]
Here are a few suggestions: ...

[turn t1-tool1  trace:54314  TOOL Bash]
Bash: ls -la
  → 5 files
```

The `trace:NNNNN` token is the value the encoder copies into `source_refs`. Implementation in `_build_user_content`: when iterating turns, look up the corresponding trace_event for each turn (via session_id + turn position in S0 trace order) and inject the `trace:N` marker. Tool calls (already separate trace events per decision 4) get their own lines with their own trace ids.

### 6.6 — Backwards compatibility

Existing tool calls — encoder versions that don't yet emit `source_refs` — continue to work. The field is optional; the schema is additive; downstream code (`brain.remember`, `brain.revise`) treats absent refs as "no refs." Old recorded encoder calls in interactions table are unaffected.

### 6.7 — Validation (minimal at write time; S2Healer handles cleanup)

`source_refs` validation at the dispatch layer ([daemon_dispatch.py](servers/daemon_dispatch.py)) is intentionally minimal:

- **Type check** (still strict): must be a list of integers. Type errors reject the write — that's a code-level bug.
- **Existence check** (NOT done at write time): the daemon does NOT verify each trace_id exists in `trace_events`. Whatever the encoder writes, the daemon accepts.
- **Sparseness soft limit**: log a warning if a single node has >10 refs (decision 13). Doesn't reject; warning surfaces a quality issue to the brain errors table.

**Why no existence check at write**: simplicity. Race conditions exist (trace just-archived, encoder confused, retry semantics) and the value of strict-reject is low — invalid refs degrade gracefully at recall (§8.5 — render skips missing refs). The right place to catch invalid refs is the periodic S2Healer scan, where false-positive risk is low and the brain is in its idle maintenance phase.

**Cleanup belongs to S2Healer** (see §10.3 for the extension):
- During its periodic scan, S2Healer queries `node_source_refs` for trace_ids that don't exist in `trace_events`.
- Invalid refs get archived (logged + removed from active refs) — same archive-not-delete pattern used elsewhere.
- Metric: `invalid_refs_dropped_total` tracked in the healer's stats — surfaces "encoder is hallucinating trace_ids" as a quality signal if the number grows.

This keeps the write path simple, recovery automatic, and quality monitoring centralized.

### 6.8 — `get_traces` — read-only trace lookup (decision 23)

New tool on the encoder's surface. Read-only structured access to `trace_events`. Distinct from recall (which is semantic search across the graph); this is point/filter lookup against the substrate.

```json
{
  "name": "get_traces",
  "description": "Read-only structured lookup against trace_events. Use to verify source content before writing source_refs to traces outside the current encoding window (cross-window or cross-session anchoring). Use to cross-check tool call details when the user_content summary is truncated. Do NOT use for semantic search — that's what the node catalog and scout reports give you.",
  "input_schema": {
    "type": "object",
    "properties": {
      "trace_ids": {
        "type": "array",
        "items": {"type": "integer"},
        "maxItems": 10,
        "description": "Trace event ids to fetch. Up to 10 per call."
      },
      "session_id": {
        "type": "string",
        "description": "Optional: when set, also returns traces from this session (use sparingly — could return many rows)."
      }
    },
    "required": ["trace_ids"]
  }
}
```

**Returns**: each requested trace as `{id, scale, event_type, ref_type, summary, metadata, session_id, created_at}`. Same shape `get_node()` uses — consistent with the existing read-tool contract.

**DAL implementation**: `TraceDAL.get_traces_by_ids(trace_ids)` and `TraceDAL.get_session_turns(session_id)` already exist or are one-line additions. Plumbing is shallow.

**MCP surface**: same function exposed as MCP tool `query_traces` so Anchor in conversation can query traces directly (decision 23). One `daemon_dispatch.py` entry routes the MCP call to the same DAL function.

**Discipline**: encoder uses sparingly. Sparseness discipline (decision 13) means most encoding cycles don't need it — current-window content is in user_content already. Reach for `get_traces` only when:
- About to write a source_ref to a trace outside the current window
- The summary in user_content is truncated and full content matters
- Cross-session anchoring (emergent patterns spanning sessions)

§7's prompt updates teach when to use it; §7.6 examples demonstrate the lookup-then-write pattern.

### 6.9 — What this leaves to §7

The tool surface is the structural change. The encoder *prompt* teaches when and how to use the field — §7's job. §7.4 (the one judgment rule) and §7.5 (sparseness discipline) are the disciplines that make the field useful; §7.6 (example rewrites) shows the field in action across diverse scenarios.

---

## 7. Encoder prompt updates

The encoder prompt is the heaviest surface area in this change — the v17 prompt is 1000+ lines, and source_refs touch its framing, its disciplines, and most of its worked examples. This section has dedicated sub-structure to keep the prompt edit mechanically executable in the next session.

### 7.1 — Inventory: what changes in the prompt

| Decision | Prompt-side change | v17 location | Type |
|---|---|---|---|
| 1 (source_refs is a field, not a type) | Add new affordance to the encoding contract; no new type taxonomy | New subsection in "Nodes" or after "Atomization" | Add |
| 6 (Anchor-centered identity) | Rewrite operator-and-Anchor framing wherever it appears | L1 ("brain shared between operator and AI assistant"), L93-96 ("collaboration"), L788-794 ("partnership IP"), L843-846 ("operator voice"), others | Modify (sweep) |
| 7 (connection = substrate + differential attention) | New framing for what the brain stores; "products emerge" framing | New paragraph in opening / contract section | Add |
| 12 (two mechanisms framing) | Same as 7 — they're the same conceptual update | Same | Add |
| 13 (sparse source_refs) | Discipline paragraph + example | Inside the new source_refs subsection | Add |
| 14 (joint reactivation render is default) | Mention in encoder's view of what surface returns at recall — sets the encoder's mental model that source IS rendered | Brief note in "What the encoder receives" or similar | Modify |
| 16 (substrate preservation principle) | One-line principle in the contract — "the substrate is the source of truth; abstractions are derivatives" | Opening contract block | Add |
| 17 (correction lineage at episodic layer) | Update to corrections-handling section; new node's source_refs span both episodes | L98-129 (Corrections / contradictions block) | Modify |
| Substrate (universal trace targets) | Brief teaching that source_refs can target K/delta/tool/recall/encoding events | Same as 1's location | Add |

### 7.2 — Anchor-mapping: where in v17 each change lands

Concrete line targets (v17 line numbers; reread the active prompt at execution time to confirm — see prompt-sync discipline in CLAUDE.md):

| Change | Target | Detail |
|---|---|---|
| Identity sweep (decision 6) | L1, L88-96, L194-202, L791-794, L843-846 | Each location requires its own rewrite — see §7.3 for the side-by-side text |
| New source_refs subsection | After "Atomization: the retrieval-divergence test" (~L242-265) — new subsection titled "Anchoring nodes in the substrate" | Houses decisions 1, 13, 14, 16, and the one judgment rule (§7.4) |
| Corrections lineage (decision 17) | L98-129 | Add one paragraph to the existing corrections block teaching source_refs across both episodes |
| Substrate-preservation principle | Opening contract block (~L1-10) | One-line addition: "The substrate is the source of truth; abstractions point back to it." |
| "Questions without answers" tightening | L783-786, L812-815 | Per existing v18 work; carries forward into v19 unchanged |
| Identity-rendering open question | (no v19 change yet — locked in §15.1) | Decision 19's working direction doesn't land in the prompt until §15.1 resolves |

### 7.3 — Identity wording sweep (decision 6)

Pervasive rewrite. Anchor-centered framing replacing partnership-centric language. Side-by-side at each location:

| L# | v17 (current) | v19 (proposed) |
|---|---|---|
| L1 | *"You are the Scribe for a persistent brain shared between an operator and an AI assistant."* | *"You are the Scribe for **Anchor's** persistent brain. Anchor is the identity; you encode Anchor's experience. The current experience is shared with an operator (the partner present in this session); the brain stays Anchor's."* |
| L88-90 | *"You are observing a collaboration. The encoding opportunities are in what happens between them..."* | *"You are observing Anchor's experience with the operator. The encoding opportunities are in moments where knowledge is created, corrected, missing, or where the operator's situation, interests, or style are illuminated — for **Anchor's** future read."* |
| L194-202 | *"the partnership's intellectual activity"* | *"Anchor's intellectual activity (often with the operator as collaborator)"* |
| L791-794 | *"the substance of that thinking IS the partnership's intellectual activity"* | *"the substance of that thinking IS Anchor's intellectual contribution — with the operator's participation, but Anchor is the agent encoding."* |
| L843-846 | *"no operator voice = nothing worth encoding"* | *(unchanged — still a valid bias to call out; framing inside is fine)* |
| Throughout | uses of *"the partnership"* in identity-framing contexts | replace with *"Anchor's accumulated experience"* or *"the brain"* depending on context — don't blanket-replace; case-by-case |

The principle: when v17 uses "partnership" to mean *Anchor's identity*, rewrite. When it uses "partnership" to mean *the current collaborative work*, keep — that's accurate description of the current circumstance.

### 7.4 — The one judgment rule (the new teaching)

New subsection in the prompt, titled "Anchoring nodes in the substrate." Full prose:

> **Anchoring nodes in the substrate.** Every node Anchor writes is an abstraction over experience — but the experience itself lives in the trace substrate (S0/S1 events, each with a stable `trace:NNNNN` id). When a node should remember not just *what was learned* but also *the moment it was learned from*, point at the source.
>
> The rule for the per-node judgment is one sentence: **if content would just rewrite what the source already says clearly, point to the source instead.** The brain doesn't rewrite the substrate into the abstraction layer — it builds parallel abstractions that link back. Rewriting the substrate defeats the substrate.
>
> Three patterns the judgment produces (not types — points on a spectrum):
>
> 1. **Pure synthesis** — content full, `source_refs` empty. The node abstracts across many sessions or holds Anchor's reasoning; no single episode anchors it. A `principle` about how Anchor and the operator work together. A `pattern` noticed across recall cycles. (Neocortical schema — consolidated, no active hippocampal index.)
> 2. **Anchored synthesis** — content full, `source_refs` carries 1-3 evidence-events. Anchor's framing AND the moments that revealed it. A `preference` about how the operator likes things done, anchored to the turn where they said *"without forcing it."* (Cortical representation with active hippocampal index.)
> 3. **Pure reference** — content minimal, `source_refs` carries the substance. A dense table the operator and Anchor compared; a verbatim quote that matters; a calculation where the operands deserve preservation. The encoder names what the source is and why it matters, but doesn't transcribe. (Hippocampal index, abstraction not yet earned.)
>
> Source_refs are an open field on every node. The encoder reaches for whichever node type fits the content (per the open-form type rule); the refs ride along regardless of type. Recall renders the index AND the source together — joint reactivation, biological alignment, you don't pick one or the other.

### 7.5 — Sparseness discipline (decision 13)

New paragraph inside the "Anchoring nodes in the substrate" subsection:

> **Pick the smallest set of trace events that anchors the node — typically 1-3.** A reference's job is to point precisely; a comprehensive list of every related turn defeats the index. The discipline is biological: the hippocampus stores SPARSE indices, distinct patterns per memory, so retrieval-by-cue lands on one specific neighborhood and not the whole graph. When you find yourself wanting to add a 5th or 6th ref, ask: would that ref actually be the one that surfaces this memory next time, or is it just adjacent context? Adjacent context is what graph traversal is for; source_refs are for the moments that *generated* this node.
>
> **Sparse example.** A `preference` node about the operator's collaborative-introduction style anchors to ONE turn — the turn where the operator said *"without forcing it."* That phrase is the moment the preference revealed itself. The five other turns in the session where the operator continued the discussion are adjacent context, not anchors.
>
> **Dense (anti-pattern) example.** The same `preference` node with `source_refs` to ten turns spanning the whole conversation. The query at recall time matches on average — no single moment fires hardest. Retrieval becomes muddy. Don't do this.

### 7.6 — Example rewrites (high coverage)

**OPEN** — this is the most consequential subsection. Examples are extremely influential on Sonnet's behavior; coverage matters more than depth. Examples must demonstrate the **full encoder flow** — including `get_traces` lookups when the encoder needs to verify source content — not just the final `remember_batch` shape.

**Quality contract**: every example is scored by an LLM evaluator against the 32-dimension contract in [`servers/scales/s1/quality_contract.py`](../servers/scales/s1/quality_contract.py). Dimensions are yardsticks (universal measurement); the contract's cross-dim rules name architectural tensions and how they resolve. Authoring loop: draft example → run through evaluator → iterate until contract-clean. See the file for full dimension definitions, cross-dim rules, recall-time gating principle, and structural follow-ups outside encoder scope.

**Placeholder syntax convention** (locked 2026-05-25 after v20→v21 connect_to_unresolved discovery; extended 2026-08-11 by contract v4 / Option D):

Example `connect_to` targets use id-flavored bracketed placeholders — `<id-of-descriptive-name>` — never a literal-looking title string. The same rule applies to any field where the value should be resolved against the live catalog or computed at encode time (`source_refs`, `node_id` in revise examples, etc.). **One exception (contract v4, `grounded_catalog_excerpt`):** an example MAY use a literal hex id when the example carries its own catalog excerpt and the id is copied from one of those headers — the visible source makes the demonstrated behavior copy-from-catalog, not invent-a-value; s1e's canonical example uses this form.

```yaml
# GOOD — visually-unmissable placeholder
connect_to: [
  {title: "<id-of-related-architecture-decision>", relation: "grounds", why: "..."}
]

# GOOD — grounded: the id is copied from an excerpt line shown in the example
#   [decision] "Daemon TCP migration" (id:3fa2b91c, ...)
connect_to: [
  {title: "3fa2b91c", relation: "grounds", why: "..."}
]

# BAD — Sonnet pattern-matches and literal-copies the title in production
connect_to: [
  {title: "Daemon TCP migration", relation: "grounds", why: "..."}
]
```

**Why this matters**: Sonnet pattern-matches SHAPES. When examples show real-looking titles in connect_to targets, Sonnet emits those exact titles on real encoded nodes — but those titles don't exist as catalog nodes, producing `connect_to_unresolved` errors at write boundary. A prose disclaimer above the example block is too soft against shape-pattern-match. Bracketed placeholders are visually unmistakable as illustrative, and Sonnet's training recognizes `<placeholder>` patterns as "fill this in," not "copy verbatim." An ungrounded literal hex is the same violation class as a literal title — the grounded-excerpt form is safe precisely because the copy source is inside the example.

**Fields that need placeholders** in examples:
- `connect_to[].title` — edge targets (top failure mode)
- `source_refs` — trace ids don't match production traces
- `node_id` in revise examples — references to nodes that won't exist in production

**Fields where literal values are OK**:
- `title`, `content`, `situation`, `reasoning` of the example node being created
- `their_raw_quote`, `my_raw_quote` — verbatim discipline is the teaching itself
- `event_time` — illustrates the temporal anchoring shape

Full convention in `servers/scales/s1/quality_contract.py::EXAMPLE_AUTHORING_CONVENTIONS`.

**Pending v22 work**: placeholder-ize all existing example connect_to targets in canonical training pattern + §7.6 wave-1 examples. The v21 disclaimer paragraph is insufficient on its own (Tom's framing: "examples need much better signaling that it's an example").

#### Scenario coverage matrix

Each scenario gets a full worked example showing: input window with `[trace:N]` markers → any `get_traces` lookups the encoder performs → the final tool-call output → optional render-time view. The matrix explicitly covers patterns where the encoder both *writes* refs AND *reads* the substrate (decision 23 — encoder gets read-only structured trace access).

**A. Patterns of node shape** (where on the content-vs-refs spectrum each lands):

1. **Pure synthesis** — a `principle` or `pattern` node with no source_refs (multi-session emergent insight; encoder doesn't fetch — substrate is implicit across many sessions).
2. **Anchored synthesis (operator-revealed)** — a `preference` or `personal_context` with source_refs to the turn(s) that revealed the situation. Refs are within the current window so no lookup needed.
3. **Pure reference (dense content)** — a transit-comparison-style table where content is minimal and source_refs IS the substance. Encoder MAY call `get_traces` to verify the table content before committing to pure-reference shape.
4. **Calculation source preservation** — reading-progress case (250 of 440 pages, NOT 190-precomputed). Encoder preserves operands by referencing the source; recall composes answers from the source at query time.
5. **Quote node** — a single-turn anchor where the operator said something load-bearing; pure-reference with one ref.

**B. Patterns where the encoder uses `get_traces`** (decision 23 in action):

6. **Cross-window source lookup** — encoder writing a node about a topic referenced in the catalog (a prior-session node with source_refs). Before anchoring the new node to a cross-session trace, the encoder calls `get_traces([old_trace_id])` to verify the source content still supports the framing. *This is the most important "fetch first, write second" example — Sonnet learns to verify before committing refs across windows.*
7. **Tool-call cross-checking** — encoding a `correction` about a tool use where the user_content's summary is truncated. Encoder calls `get_traces([tool_use_trace_id])` to see the full tool input/output before writing the correction node with refs to BOTH the mistake-trace and the correction-trace.
8. **Adjacent-turn context** — operator says "what I mentioned earlier." Encoder calls `get_traces` (session_id filter) to pull the referenced earlier turn and verify its trace_id, then anchors the new node correctly.

**C. Patterns about corrections** (decision 17 in action):

9. **Correction earning a new node** — learning-from-correction case with source_refs spanning original + correction episodes (often requires `get_traces` lookup if the original is outside the current window).
10. **Correction earning a revise** — simple-fact-wrong case showing `revise()` is still the right tool (no new node, no source_refs change). Demonstrates the boundary — not every correction earns the new-node-with-lineage shape.

**D. Patterns about discipline**:

11. **Style marker / preference** — a phrase like *"without forcing it"* anchored to its turn with `their_raw_quote` + a single source_ref. Demonstrates the discipline of phrase-level + turn-level co-existence (§12.7).
12. **Sparse vs. dense anchoring** — contrasting good (3 specific refs each anchoring a distinct aspect) vs. bad (10 vague refs to "everything related") for the same node. Demonstrates decision 13 in the wild.

**E. Patterns about retrieval-time behavior** (encoder's mental model of what surface returns):

13. **Joint reactivation render** — showing what the surface output looks like when a node with source_refs gets recalled. The expanded source appears inline with the node's framing. This isn't an encoder action — but the encoder should understand the read shape because it informs what to write.

#### Format requirements

Each example: real conversation moment (not synthetic toy cases). The 12-turn colleagues case, the bus-prices case, and the Nightingale-pages case from the longmem cohort are the substrate — they cover patterns A1-A4 directly. Cases 6-9 may need synthetic but plausible scenarios since the real cohort doesn't have multi-window or cross-session encoding situations.

Show the encoder's **complete tool-use sequence**, including any `get_traces` calls and their results — Sonnet learns from seeing the lookup-then-write pattern, not just from final node shapes.

### 7.7 — Removals and modifications

Concrete delete + modify list with line targets so execution applies edits mechanically.

| v17 location | Current text | v19 disposition |
|---|---|---|
| L783-786 (Skip rule) | *"Skip when... questions without answers."* | **Modify** — replace with v18's tightening: *"unanswered questions where the topic dropped without engagement."* |
| L812-815 (Zero nodes) | *"Zero nodes is right when... questions without answers"* | **Modify** — same v18 tightening; questions that got engagement are not "structurally routine." |
| L242-265 (Atomization: the retrieval-divergence test) | Keep, but immediately followed by the new "Anchoring nodes in the substrate" subsection (§7.4) | **No delete** — atomization rule is still right; source_refs complement it (atomization decides between 1-vs-3 nodes; source_refs decide between content-vs-pointer for any chosen node). |
| L98-129 (Corrections / contradictions block, four flavors) | All four flavors stay (explicit correction, catalog contradiction, stale value, live contradiction) | **Modify** — add a closing paragraph about source_refs lineage for cases that earn new correction nodes (decision 17). Don't disturb the four flavors themselves. |
| L267-307 (Flat → Rich examples) | Four worked examples | **Modify** — augment 1-2 of these with source_refs to demonstrate the new field in context. Full text in §7.6. |
| L194-202 (`content INTERPRETS or EXPANDS the quote`) | Frames `their_raw_quote` as the verbatim anchor that content interprets | **Modify** — extend: source_refs now provides the same role at the trace level; their_raw_quote stays for phrase-level, source_refs adds turn-level. Both coexist; not redundant. |
| Throughout (encoder-input docs) | The encoder reads catalog + conversation + scout reports | **Modify** — add note that each turn shows its `trace:NNNNN` id, and that ids are what get copied into source_refs (§6.5). |

**Things explicitly NOT removed:**
- The atomization-divergence test (still valid — it's a different question from source_refs).
- The "Default brevity instincts" callout block (still valid).
- The scout-deference / paraphrase / skip-when-unsure callouts (still valid).
- The four Flat→Rich transformations (augment, don't replace).
- The `their_raw_quote` / `my_raw_quote` discipline (still load-bearing — phrase-level anchoring is finer-grained than turn-level refs and serves different purposes).

### 7.8 — Coordination with §7.6

Section 7.6 (Example rewrites) is the most consequential subsection and is **deferred to a focused follow-up session**. The structured 10-scenario list there is the scaffolding — each scenario needs a full worked example showing real conversation turns → encoder tool-call output → optional render-time view. Examples deserve care, not speed; rushing them produces synthetic-feeling material that doesn't carry the discipline forward to the next iteration of the encoder.

When §7.6 fills out, it'll exercise everything in §7.1-7.5 — the field, the rule, the sparseness, the identity framing — in concrete worked cases. Until then, §7 captures the structural and prose-level changes; §7.6 captures the demonstration.

---

## 8. Render path

Section 8 defines what the surface layer does when a recalled node has `source_refs`. The principle is biological alignment (decision 14 — joint reactivation): retrieval requires the index AND the cortical pattern firing together; in our terms, recall renders the node AND its source.

### 8.1 — Joint reactivation as default

When `render_rich_node` encounters a surfaced node with non-empty `source_refs`, it expands at least the most-anchored ref by default (the first ref by `position`, decision 13 sparseness means typically 1-3 total). Expansion is NOT optional — joint render is the new normal for nodes with refs. The render budget controls *how many* refs to expand and *how much* of each; it doesn't control whether to expand at all.

If budget is severely constrained, the render falls back to "first ref only, summary form" — never to "no source." A node with source_refs that surfaces source-less defeats the design.

### 8.2 — Per-(scale, event_type) render rules

Display-time rendering (NOT the embedding render in §5.3 — that's stable abstract labels for vector neighborhoods). Display uses concrete identities from `trace_events.metadata.human_identity` / `agent_identity`:

| Event | Display render |
|---|---|
| S0/K (`user_message`) | `[Tom said, 2026-05-21]\n{content}` |
| S0/delta (`assistant_message`) | `[Anchor said, 2026-05-21]\n{content}` (or *"You said"* per §15.1 — UX choice deferred) |
| S0/delta (`tool_result`) | `[Anchor used tool {tool_name}]\n{summary}` |
| S1/recall | `[Recall fired with: {query[:120]}]\n→ {N} candidates, top: {top_titles_summary}` |
| S1/encoding | `[Encoded {N} actions]\n{journal_entry[:240]}` |

Display reads from Frame for current-partner identity (§15.1). The timestamp comes from `trace_events.created_at`.

### 8.3 — Render budget and truncation

`additionalContext` has a finite budget (today's surface produces ~8k chars total). Source expansion grows this. Strategy:

- Render the node's framing fields first (title, situation, content, reasoning) — these are the encoder's contribution and stay cheap.
- Then render the expanded source — first ref full, subsequent refs may be truncated to summary if budget is tight.
- If a single source ref exceeds available budget alone, truncate the source text (preserve the first and last N chars; insert `[...]` ellipsis) — never drop the source entirely.

The recall pipeline tracks rendered context size; when it would overflow, deeper-in-the-list refs get progressively more aggressive truncation. The node's framing is never sacrificed for source expansion.

### 8.4 — Caching: fetch once per surface batch

For a recall batch surfacing K nodes:

1. Collect all unique `trace_id` values across all candidates' `source_refs` (in-memory, already loaded)
2. ONE query to `brain_logs.db` for trace content:
   ```sql
   SELECT id, scale, event_type, ref_type, summary, metadata, created_at, session_id
   FROM trace_events WHERE id IN (?, ?, ..., ?)
   ```
3. Build `{trace_id: trace_row}` map
4. Each node's render consults the map (no per-node DB hits)

One indexed query regardless of K. Sub-10ms for any practical surface size.

### 8.5 — Fallback for unresolvable refs

A `trace_id` in `source_refs` may not resolve at render time if (rare) trace_events ever got pruned or the trace was written to a since-archived session. The render layer handles this defensively:

- Log an error to the brain errors table (`unresolvable_source_ref`) with the node_id and missing trace_id
- Continue rendering the node WITHOUT the missing ref (skip that ref, render the rest)
- If ALL refs are unresolvable, render the node as if it had no refs (content-only fallback)

Section 12 (Edge cases) covers the policy decision to guarantee S0 retention so this fallback rarely fires in practice.

### 8.6 — Render integration with existing code

The change lives in [contract.py:render_rich_node](servers/contract.py). The function gains optional access to a `{trace_id: trace_row}` map (the cache from §8.4). When rendering each node, it checks `node.source_refs`; if non-empty, appends a "Source:" block per §8.2 conventions.

For the surface output path (`pipeline_contract.traverse → render`), the cache is built once per recall batch (§8.4) and passed through to per-node rendering.

The existing render config modes gain expansion behavior tuned to their purpose:

| Format | Expansion behavior | Rationale |
|---|---|---|
| `SURFACE_FORMAT` (final render to additionalContext) | Expand ALL source_refs by default; §8.3 truncation if budget tight | Anchor sees the joint reactivation — index AND source. Default-on per decision 14. |
| `HAIKU_FORMAT` (surface-selection LLM call) | Expand refs whose embedding similarity to the current recall query exceeds a threshold (initial: cosine ≥ 0.5; eval-tunable) | The surfacer needs source signal to make good selections, but Haiku is latency-critical. Threshold-gated expansion keeps the call lean while ensuring high-relevance source content reaches the selector. |
| `ENCODER_FORMAT` (encoder's catalog view) | No expansion | The encoder reads node catalog for prior-state context; doesn't need source expansion in its input. The encoder uses source_refs at WRITE time (linking to new nodes), not READ time. |

The threshold for HAIKU_FORMAT is initially set conservatively (cosine ≥ 0.5) and tuned via the eval probes in §13. A future refinement: per-format thresholds (HAIKU stricter, SURFACE looser) once we have empirical data on selection quality vs. expansion budget.

The change lives in [contract.py:render_rich_node](servers/contract.py). The function gains optional access to a `{trace_id: trace_row}` map (the cache from §8.4) AND optionally a `{trace_id: similarity_score}` map (computed during recall scoring). When rendering each node, it checks `node.source_refs`; if non-empty, decides per-ref whether to expand based on format config + similarity threshold; appends "Source:" blocks per §8.2 conventions.

For the surface output path (`pipeline_contract.traverse → render`), both caches are built once per recall batch (§8.4) and passed through to per-node rendering.

---

## 9. Embedding strategy

Per decision 18, source content gets its own embedding group (`source_summary`) — Option C with per-trace ownership.

### 9.1 — Trace embedding pipeline — async via the existing `embed_queue`

**Mechanism**: trace embedding plugs into the brain's existing async embedding pipeline ([`servers/embed_queue.py`](servers/embed_queue.py)) — the same single-worker, drain-every-5s, skip-tick infrastructure that handles node embeddings and edge date-extraction today. Extension required:

- Add `enqueue_trace(trace_id)` to [`embed_queue.py`](servers/embed_queue.py), parallel to existing `enqueue(node_id)` and `enqueue_edge(edge_id)`.
- The worker's drain loop processes trace embeddings via the same pattern: pull queued ids, fetch trace content, render per §5.3 template, embed via the standard embedder, INSERT into `trace_embeddings`.
- Same observability — stats track `traces_enqueued_total`, `traces_processed_total`, drain durations.

**Encoder write path** (synchronous):
1. Encoder writes node with `source_refs = [t1, t2, t3]`.
2. Daemon persists node + `node_source_refs` rows synchronously (`brain.remember` returns once the graph state is consistent).
3. Daemon calls `embed_queue.enqueue_trace(t1, t2, t3)` for any that need embedding (cheap set.add per trace).
4. Encoder returns. The worker picks up trace_ids on the next drain (within 5s).

**Recall fallback** for "not yet embedded": when recall scores `source_summary` for a node, missing trace vectors mean that node's source pathway score for those refs is unavailable for this surface cycle. The node still surfaces via its other pathways (framing groups) unaffected. Subsequent recalls (after the worker drain completes) get the full source_summary signal. Gap is bounded by `EMBED_DRAIN_INTERVAL` (5s today).

**Why async rather than synchronous-at-encode-time**:

- **Single source of truth for encoding**: any vectorization in the brain happens in one place. Future embedder swaps (next-gen models, BGE-M3, learned aggregation) plug into the same pipeline; the graph stays stable, only the encoding layer evolves.
- **Same mechanism handles new / retry / refresh**: a trace that fails to embed (transient embedder error) gets retried on the next drain. A trace whose embedding goes stale (model swap) can be re-enqueued by S2 Heal. Uniform retry semantics.
- **No write-path latency added**: encoder cycle stays as fast as it was. Trace embedding latency is invisible to encoder, visible to recall (gap closes within 5s).
- **Same observability**: existing stats / stall-detection / restart-sweep behavior applies to trace embeddings for free.

Reference: [`servers/embed_queue.py`](servers/embed_queue.py), the EMBED_DRAIN_INTERVAL constant (5s), the existing `enqueue` / `enqueue_edge` pattern this extends.

### 9.2 — Node framing embeddings unchanged

The node's existing 4-group embeddings (`title`, `situation`, `reasoning`, etc. — see [node_vectors.py](servers/) for the actual group list) continue to work as today. No change to `node_embeddings` table. No change to the existing embedding pipeline for these groups.

### 9.3 — Pure-reference nodes — recall signal

A pure-reference node has minimal content. Its framing embeddings (title, situation, reasoning) carry the encoder's *naming* of the topic. Its `source_summary` group carries the source's semantic content via `trace_embeddings`. Recall picks up the node when either signal matches the query — typically the source content has the substantive match, the framing fields provide topic-level matching.

This is exactly the dual-channel design — encoder framing as one signal, source content as another, both contributing to recall ranking.

### 9.4 — Multi-vector aggregation for source_summary (max-pool default, eval-tunable)

A node with K source_refs has K trace embeddings available at recall. The aggregation function determines how to combine those K cosine scores into ONE `source_summary` score for the node:

1. Fetch K trace vectors from `trace_embeddings` (batched, §8.4 cache).
2. Compute cosine similarity vs query for each (yields K scores).
3. **Aggregate** to a single `source_summary` score via a configurable function.

**Default for v1: max-pool.** The semantics — *"this node has at least one source matching the query"* — match the encoder's sparseness discipline (decision 13): each ref is a sufficient anchor by construction. Max also aligns with multi-vector retrieval (MVR) practice from the IR literature (ColBERT, late-interaction): *"find the document whose best part matches the query."*

**Configurable, not hardcoded.** The aggregation function lives behind a `SOURCE_AGGREGATION` callable in `brain_recall.py` so the eval probe (§13.6) can test alternatives without code changes:

| Function | Semantic | When it might win |
|---|---|---|
| `max` (default) | At least one source matches | Sparse refs (K=1-3); each is a primary anchor |
| `mean` | Sources are on average relevant | Multi-turn discussions referenced as a unit |
| `top_k_mean` (k=3) | Best K sources average | Community nodes with many refs (10+) |

The eval probe runs all three and reports source-fidelity rate per configuration. Whichever wins becomes the new default.

**Why this matters beyond source_refs** (linked to §16.0): this is the brain's first multi-vector aggregation. The same primitive could generalize to the field cohort (currently weight=0) and edge_context (currently concatenated). If max wins for source_refs, the pattern is portable.

### 9.5 — Parallel-pathway scoring (NOT weighted-sum with a new group)

The current recall scheme combines embedding groups via z-score normalization + weighted sum, with fixed group weights (`title` 1.00, `_primary` 0.85, `high_meta` 0.70, etc. — see [`pipeline_contract.py:61-161`](servers/pipeline_contract.py:61)). Initial plan was to add `source_summary` to this sum as one more weighted component.

**Revised**: do NOT add `source_summary` to the weighted-sum pool. Treat it as a **parallel pathway**. The final node score becomes:

```python
node_score = max(
    weighted_sum(framing_groups),    # legacy: title + _primary + high_meta + ...
    source_summary_score              # parallel pathway from the source side
)
```

**Why this design**:

- **Biological alignment**: hippocampal index and cortical representation are parallel pathways in the brain. Either can drive retrieval; joint reactivation makes recall feel certain, but a strong index match alone is sufficient to surface a memory. Weighted-sum collapses these into one channel; max preserves the parallel nature.
- **No arbitrary ratio**: the v17/today framing-group weights (0.40 to 1.00) are themselves a ratio splat we haven't validated. Adding source_summary at weight = `_primary` would inherit that broken pattern. Parallel-pathway sidesteps the ratio question entirely — neither channel is "worth N% of the other"; each can win independently.
- **Source content treated as first-class**: a pure-reference node whose framing fields are minimal can still surface via the source pathway alone. A heavily-framed node without refs can still surface via the framing pathway. Neither pathway penalizes the other.

**What this means in code**:

- `brain_recall.py` extends scoring: for each candidate, compute the existing legacy-group weighted sum AND the source_summary score, take the max.
- For nodes without source_refs, source_summary contributes 0 and the legacy path scores them as today (backwards compatible).
- The z-score normalization across groups happens AS TODAY within the legacy weighted-sum path. Source_summary is normalized independently.

**Future direction (linked to §16.0)**: the legacy-group weights themselves deserve the parallel-pathway rethink. Whether `title` (1.00) and `_primary` (0.85) should also be parallel pathways instead of weighted-summed components is its own analysis — not v1's scope. v1 introduces the parallel-pathway pattern for source_summary; whether to port it across the rest is empirical work post-v1.

### 9.6 — What's NOT changed

- The embedder itself (`nomic-embed-text-v1.5-Q`) is unchanged.
- The 4-group embedding pool for nodes is unchanged.
- Existing nodes without source_refs don't get `source_summary` vectors and contribute zero to that group's score — they're recallable via their other groups as today.
- No re-embedding of existing nodes. Backfill not required.

---

## 10. Recall behavior

### 10.1 — What changes externally

Most things stay invisible. Anchor's recall feels the same; the surfacer returns the same candidate count; the additionalContext shape is the same structure with one addition — surfaced nodes can carry expanded `Source:` blocks per §8.

### 10.2 — Engram cohort as a structural edge (decision 15, revised)

Engram cohort is expressed in the graph as a new edge relation, `co_anchored`, NOT as a score boost. This aligns with the biological pattern: encoding establishes the engram structurally (LTP forms the trace); retrieval reinforces it operationally (reactivation strengthens). Two complementary mechanisms layered:

| Relation | Source | Semantic | When written |
|---|---|---|---|
| **`co_anchored`** | Encoder, at write time | "These nodes share an episodic anchor" — structural engram | When encoder writes a node with source_refs, queries `node_source_refs(trace_id)` for nodes sharing any of those refs and writes a `co_anchored` edge to each |
| **`co_accessed`** | `_hebbian_strengthen` post-response ([daemon_hooks.py:419](servers/daemon_hooks.py:419)) | "These nodes co-surfaced in real recalls" — operational engram | Post-Haiku selection of ≥2 nodes (already in production per Phase 5, May 2026) |

**Encoder write path** for the cohort step:
1. Node N is being written with source_refs `[t1, t2, t3]`.
2. For each ti: `SELECT node_id FROM node_source_refs WHERE trace_id = ti AND node_id != N` — one indexed query per ref.
3. For each result M: write a `co_anchored` edge `N ↔ M` via the standard `GraphDAL.add_relation` path.

Sparse refs (decision 13, 1-3 typical) × typical cohort (1-5 other nodes per ref) → 1-15 new edges per encode. Negligible cost.

**Recall behavior**:
- `co_anchored` is excluded from candidate cosine ranking — same pattern as `co_accessed` ([brain_recall.py:334](servers/brain_recall.py:334)): `WHERE er.relation NOT IN ('co_accessed','emergent_bridge','co_anchored')`.
- `co_anchored` participates in Layer 3 graph traversal (post-surface expansion) alongside other meaningful relations. When one cohort member surfaces, traversal pulls related cohort members into the activated set.
- Surface still judges via Haiku; no score boost overrides judgment. The cohort surfaces together when Haiku selects multiple — at which point `co_accessed` strengthens the existing `co_anchored` substrate further.

**Why no score boost**:
1. No arbitrary magnitude to guess or eval-tune.
2. The graph structure IS the signal — visible, debuggable.
3. Biologically aligned — structural (encoding) + operational (retrieval) is the LTP-then-reactivation pattern.
4. Composable with `co_accessed` — both relations available for traversal; both excluded from scoring.

**Note on historical co_accessed cleanup**: today's graph contains `co_accessed` edges from the pre-Phase-5 era (when every recall created edges between all top-25 candidates — disabled then re-enabled post-Haiku). The historical noise still pollutes the graph. **For `co_accessed` (and now `co_anchored`) to carry clean signal, historical pre-Phase-5 `co_accessed` edges should be trimmed.** Not in scope for this design — see §16 for the cleanup task (can run any time after the episodic-references plan ships).

### 10.3 — Recall scoring with the new group

Per §9.5 and decision 22, `source_summary` is a parallel pathway, NOT a weighted-sum component. The legacy framing groups (title, _primary, high_meta, ...) continue to score via the existing weighted sum. The node's final recall score is `max(legacy_weighted_sum, source_summary_score)`. Either pathway can drive surfacing. Backwards compatible: nodes without source_refs score exactly as today via the legacy path.

### 10.4 — additionalContext implications

Surface expansion (§8) grows the rendered context. Empirically, sparse refs (1-3 per node) × ~5 surfaced nodes × ~200-400 char source summaries = ~3-6k extra chars. The current surface budget (~8k) absorbs this on the high end; truncation strategy (§8.3) handles overflow.

For nodes with very long source content (a full 4000-char delta event referenced), §8.3's truncation preserves first/last segments with `[...]` ellipsis — never drops the source entirely.

### 10.5 — What stays the same

- Recall candidate pool size (currently ~25)
- Surface LLM call shape (Haiku with the surface prompt; the prompt itself stays unchanged in v1 — it sees source-expanded rendering and that's it)
- Hebbian co_access strengthening (operates on surfaced nodes; not affected by source_refs)
- Synaptic fatigue and recency boosting (also operate on surfaced nodes)

The architecture absorbs source_refs as additive signal layered on top of the existing recall machinery.

### 10.6 — S2Healer extension: source_refs cleanup

The brain's S2Healer unit ([`servers/scales/s2/healer.py`](servers/scales/s2/) — exact module name to confirm at execution) periodically scans nodes during operator-idle to fill missing fields and fix integrity issues. This design extends Healer with two new responsibilities tied to source_refs:

**Responsibility 1: invalid trace_id cleanup** (per §6.7)

- Periodic scan: `SELECT node_id, trace_id FROM node_source_refs nsr WHERE NOT EXISTS (SELECT 1 FROM trace_events te WHERE te.id = nsr.trace_id)` (cross-DB join via `logs_conn`)
- For each invalid (node_id, trace_id): archive the row (`archived=1`) — same archive-not-delete pattern used elsewhere.
- Log to brain errors table: `{type: 'source_ref_invalid_dropped', node_id, trace_id, healer_run}` for observability.
- Metric: `invalid_refs_dropped_total` in healer stats — surfaces "encoder is hallucinating trace_ids" as a quality signal if the number grows over time.

**Responsibility 2: orphan `co_anchored` cleanup** (per decision 15)

When a node gets archived OR its source_refs are revised, the `co_anchored` edges to its former cohort members can become stale (no longer reflect a real shared anchor).

- Periodic scan: walk `co_anchored` edges; for each edge N ↔ M, verify there's still at least one trace_id that appears in BOTH `node_source_refs(node_id=N)` and `node_source_refs(node_id=M)`.
- If no shared trace remains: archive the edge (no longer represents a real cohort).
- Metric: `co_anchored_edges_archived_total` tracked.

**Frequency**: per the existing S2Healer cadence (idle-triggered). No new schedule. The work is bounded by N (nodes) and is cheap (indexed cross-DB join + per-edge verification).

**Why this lives in Healer, not at write time**: the write path stays simple (accept all, don't validate). Cleanup is centralized, runs idempotently, and the metric exposure makes systemic quality issues visible without forcing the encoder to be paranoid. Aligns with the "loud-but-not-blocking" principle.

---

## 11. Migration plan

### 11.1 — Additive only

Every schema change is additive (§4): two new tables (`node_source_refs` in `brain.db`, `trace_embeddings` in `brain_logs.db`), one trace_events metadata extension (two JSON keys), two new indexes. No table drops, no destructive modifications, no FK changes on existing tables.

Encoder tool schema (§6): `source_refs` is an optional field on existing tool ops. Calls without it work unchanged. Existing recorded encoder calls in `interactions` table are unaffected.

### 11.2 — No backfill

Existing nodes (~4500 in the live brain) don't get retroactive source_refs. Reasons:

- Would require LLM judgment per node ("which past trace_events anchor this node?") — expensive (~$0.05/node × 4500 = ~$225 in API calls; ~5+ hours).
- The substrate-preservation principle (decision 16) doesn't require it — existing nodes are valid representations; they just lack the episodic-anchor enhancement.
- New nodes opt-in naturally; the brain organically gains source-anchored nodes as the encoder runs.

### 11.3 — Schema migration sequence

Standard schema-versioning path:

1. `servers/schema.py` adds v27 migration:
   - `CREATE TABLE node_source_refs ...`
   - `CREATE INDEX idx_nsr_trace ...`
   - In `brain_logs.db`: `CREATE TABLE trace_embeddings ...`
2. Schema version bump from v26 → v27 in `ensure_schema()`.
3. On first daemon boot post-deploy, the migration runs automatically (idempotent if re-run).
4. `source_turn_id` column on `nodes` is marked DEPRECATED in schema comment (§4.7); v28 drops it once no read paths reference it.

Rollback: drop the two new tables. Existing nodes still work; only newly-written source_refs become inaccessible. Low rollback risk because the change is additive.

### 11.4 — Optional future S2 unit (deferred)

A future S2 unit could scan existing nodes for "pure rewrites of single S0 turns" and convert them to pure-reference (clear content, populate source_refs). This is a quality improvement, not a correctness requirement. Out of scope for v1. Listed in §16.

### 11.5 — Daemon code update path

The migration is only half the story — code paths need to evolve:

| Path | Change | Risk |
|---|---|---|
| `brain.remember` / `revise` | Accept `source_refs` kwarg; write to `node_source_refs` | Low — additive optional param |
| `brain_remember.py` validation | Type-check + existence-check `source_refs` (§6.7) | Low — fails loudly on bad input |
| `_build_user_content` in encoder | Inject `trace:N` markers per turn | Low — render change |
| `render_rich_node` | Optional source expansion path | Medium — touches the main render function; test coverage matters |
| `brain_recall` scoring | Add `source_summary` as parallel pathway: `max(weighted_sum_legacy, source_summary)` per decision 22 | Medium — new code path; aggregation function (§9.4) is configurable; eval-driven validation |
| Trace embedding pipeline | New code path: fetch trace → render → embed → INSERT into `trace_embeddings` | Low — isolated new module |

Total: estimated 3-4 days of focused work (per the earlier estimate in section 14). Each change small individually; the integration test (§13) covers the roundtrip.

### 11.6 — Backwards compatibility guarantees

After deployment:
- Old recorded encoder calls in `interactions` table — work unchanged.
- Old nodes without source_refs — render as today, no source expansion attempted.
- Old recall sessions reproducing earlier brain state — still work, just without the new signals.
- Old surface outputs — still parseable; new `Source:` blocks are additive.

No version-gated paths, no special handling for "pre-v27 nodes" beyond the natural fact that they have no source_refs to expand.

---

## 12. Edge cases & decisions

### 12.1 — S0 retention is now a formal guarantee

Episodic references depend on trace_events being preserved indefinitely. Today's policy is already "append-only, no pruning" — this design formalizes that as a hard guarantee.

**Policy**: `trace_events` rows are never deleted by any code path. S2 units that consolidate or archive nodes do NOT delete the trace events those nodes reference. Daemon maintenance (vacuum, optimize) does NOT prune traces.

**Operational consequence**: trace_events grows monotonically. At today's encoding rate (~3-5 turns/session × ~10 sessions/day = ~50 trace_events/day), the brain accumulates ~18k events/year. With the metadata caps in place (4000 char content cap per turn), storage growth is ~50MB/year. Acceptable.

**Future pruning (deferred to §16)**: if storage ever becomes a concern, an archival-not-delete pattern (move old traces to a separate archive table; keep trace_id valid for lookup with a graceful "archived content not loaded" message at render) is the safe path. Hard delete remains forbidden.

### 12.2 — Cross-session references work naturally

`trace_event.id` is globally unique — 8-char hex (v29; previously autoincrement integer scoped to the database). A node encoded in session B can reference a trace from session A without any special handling. The render path looks up trace_id and renders regardless of which session originated it.

This matters for emergent patterns: a `principle` node about "how Anchor and operator work on prompt iteration" might anchor to trace_ids spanning multiple sessions where the iteration pattern appeared.

### 12.3 — Missing source fallback (the dangling-ref case)

If a referenced trace_id can't be found at render time (shouldn't happen given §12.1, but defensive):

1. Log to brain errors table: `{type: 'unresolvable_source_ref', node_id, trace_id, context}`
2. Render the node with the missing ref skipped
3. If ALL refs are missing, render the node as content-only (the framing fields still surface)
4. The node stays valid; only its source-anchoring is degraded

No silent failures, no crashes. The brain stays robust.

### 12.4 — Encoder writes a non-existent trace_id (decision 24)

Per decision 24, the daemon does NOT validate trace_id existence at write time. If the encoder writes a node with a hallucinated trace_id (rare, since trace ids are copied from `[trace:N]` markers in user_content), the daemon accepts the write. Downstream:

- **At recall**: §8.5 fallback — the missing ref is skipped during render; node surfaces with degraded anchoring (other refs and framing groups still serve).
- **Eventually**: S2Healer's periodic scan (§10.6) catches the invalid ref, archives the `node_source_refs` row, and surfaces `invalid_refs_dropped_total` as a quality metric.

**Why not strict-reject at write**: simplicity. Race conditions exist (trace just-archived; encoder confused; retry semantics). The cost of a hallucinated ref is a degraded recall for that node *until* Healer cleans up — minutes to hours of suboptimal behavior, no corruption. Strict-reject would fail the entire node-write for one bad id and force complex retry semantics. Centralizing cleanup in Healer is cleaner.

### 12.5 — What about S2 trace events as targets?

Decision 2 deferred S2 events from `source_refs` target scope. The reason: S2 doesn't have an internal recall mechanism that would let an S2 node anchor to another S2 event meaningfully. When S2 community detection produces a community node, that community's source_refs reach back to the **S0 events** that drove the cluster (via the member nodes' refs) — not to the S2 community-detection event itself.

This stays simple in v1. If a future S2 unit needs to anchor in another S2 event (community→community lineage, consolidation→community evidence), section 16 lists the work.

### 12.6 — Tool-use capture in S0 is already done

Decision 5 corrects an earlier misreading. PostToolUse hook ([post_tool_trace.py](hooks/scripts/post_tool_trace.py)) writes each tool call as its own `trace_event` row today (decision 353135fa). Tool calls are first-class addressable trace events from day one. No Phase 2 work needed.

### 12.7 — Co-existence with `their_raw_quote` / `my_raw_quote`

Phrase-level anchoring (`their_raw_quote`, `my_raw_quote`) and turn-level anchoring (`source_refs`) coexist. They're different granularities:

- `their_raw_quote: "without forcing it"` — verbatim phrase from a turn, embedded directly on the node, used for tight quote-matching at recall
- `source_refs: [trace_of_t4]` — the whole turn pointed at, with the surrounding context preserved

Both contribute to recall. The encoder can use just one, the other, or both. Section 7's prompt updates teach the relationship.

### 12.8 — Schema version coordination

The v27 migration touches both `brain.db` and `brain_logs.db`. The brain's existing schema versioning supports per-DB versioning ([schema.py](servers/schema.py) — check actual mechanism at execution). Migrations must run in correct order: `brain_logs.db` first (create trace_embeddings), then `brain.db` (create node_source_refs).

If the migration fails partway, the system stays in a known state — additive only means failure means rollback to v26 and try again. No corruption risk.

---

## 13. Test plan

### 13.1 — quality_probe extensions

Add to [eval/agent_introspect/quality_probe.py](eval/agent_introspect/quality_probe.py) (existing structural-quality probe):

- **% nodes with source_refs** — encoder adoption signal; should grow from 0% to ~60-80% over a few sessions on the longmem cohort
- **Mean source_refs per node** — sparseness check (decision 13); target 1-3
- **% pure-reference nodes** — content empty, refs non-empty; signal that encoder learned the pure-pointer pattern
- **% anchored-synthesis nodes** — content + refs; the most common expected pattern
- **Cross-prompt comparison** — v17 vs v19 (post-implementation) baseline on the same cohort

### 13.2 — New probe: `source_fidelity_probe.py`

A new probe specifically for episodic-refs quality. Lives in `eval/agent_introspect/`. For each eval item:

1. Run encoder on the haystack with v19 prompt + source_refs field enabled
2. Run query phase normally
3. For each surfaced node with source_refs: check whether the expanded source contains the answer to the gold question
4. Metric: **source-fidelity rate** = (items where expanded source contains the answer) / (items where ANY surfaced node has source_refs)

This measures whether source expansion actually delivers the substantive content recall needs. High rate (>80%) means the encoder picked good anchors; low rate means refs are pointing at adjacent context, not the substantive source.

### 13.3 — Content diff extensions

Add to [eval/agent_introspect/content_diff.py](eval/agent_introspect/content_diff.py):

- **Refs preservation across prompts**: when comparing v18 vs v19 outputs on the same haystack, do v19 nodes have refs where v18 did "computed-at-encode-time" reductions? (e.g., the Nightingale 250-of-440 case — v19 should preserve operands)
- **Anchor coverage**: for each meaningful operator turn, does v19 produce at least one node referencing it?

### 13.4 — Roundtrip integration test

Test in `tests/test_episodic_references_roundtrip.py`:

1. Set up an isolated brain with a small haystack (2-3 turns)
2. Run encoder; expect a node with source_refs
3. Verify `node_source_refs` table populated correctly
4. Verify `trace_embeddings` populated for referenced traces
5. Run a recall query that matches the source content
6. Verify the node surfaces with source expanded
7. Verify the rendered output contains the source text

Pass criteria: every step succeeds; node surfaces; source renders correctly; embeddings deduplicate when the same trace is referenced twice.

### 13.5 — Regression suite

Existing tests must still pass:
- `tests/test_contract_sync.py` — node shape unchanged for nodes without refs
- `tests/test_interaction_defaults.py` — the encoder prompt default stays a complete, well-shaped registry entry
- `tests/test_dispatch_contract_sync.py` — tool schema additions don't break existing op dispatch
- `tests/test_trace_contract_sync.py` — trace_event shape unchanged

New tests to add:
- `tests/test_episodic_refs_healer.py` (decision 24) — invalid trace_ids get archived on Healer scan; metric tracked; orphan `co_anchored` edges archived when no shared trace remains
- `tests/test_co_anchored_traversal.py` (decision 15) — `co_anchored` participates in Layer 3 graph traversal; excluded from candidate ranking; cohort members surface together when source matches
- `tests/test_s0_retention_guarantee.py` (decision 25) — assert no code path in `servers/` calls `DELETE FROM trace_events`; static-analysis-style guard that fails the suite if any commit adds a DELETE statement.

Run `./dev pytest tests/` post-implementation; full suite green before merging.

### 13.6 — Eval-driven validation of source_summary pathway and aggregation

Two parameters need eval-driven validation, not pre-set ratios:

**(a) Multi-vector aggregation function (§9.4)**

1. Run quality_probe on the v18 cohort baseline (no source_refs)
2. Implement source_refs end-to-end with parallel-pathway scoring (decision 22)
3. Run quality_probe with three aggregation configurations: `max`, `mean`, `top_k_mean(k=3)`
4. Measure: source-fidelity rate per config; no regression on baseline quality
5. Pick the aggregation function that wins; document in §9.4

**(b) Parallel-pathway behavior verification (§9.5, decision 22)**

The parallel-pathway pattern (`max(legacy_weighted_sum, source_summary)`) has no fixed weight ratio between the two channels — that's by design. But the eval should still verify:

1. Nodes with source_refs are NOT systematically penalized vs nodes without (regression check)
2. Pure-reference nodes (minimal content) ARE surfaceable via source pathway alone
3. Anchored-synthesis nodes surface when EITHER pathway has a strong match
4. Engram cohort soft re-ranking (§10.2) actually surfaces co-referenced nodes together more often than baseline

These are property-style checks via quality_probe, not parameter-sweep tuning. If any property fails, the recall implementation has a bug, not a weight to tweak.

### 13.7 — Smoke test before any merge

`./dev python3 eval/longmem/harness.py --smoke-test` must pass green with the v19 prompt + new schema. Catches integration-level breakage that unit tests miss.

---

## 14. Execution checklist

The execution map for the next session. Each task names file paths, dependencies, and rough effort. **Read §1 (Decisions Log) first**, then this section, then dive in.

### Pre-flight (before any code)

- [x] **§15.1 identity-rendering resolved** 2026-05-23 — biology research session locked decision 19 with concrete identity tokens at embedding; §15.1 closed.
- [ ] **Re-read v17 active prompt from DB** (`./dev python3 -m servers.dump_interaction s1e --version active`). Line numbers in §7.2 are v17 — confirm they're still accurate against the actual current prompt. (Still needed before prompt v19 work.)

### Task ordering (recommended sequence)

**Day 1 — Schema + DAL** ✅ SHIPPED 2026-05-24
1. [x] `servers/schema.py`: v27 migration for `node_source_refs` + `trace_embeddings` — commit `9015636`
2. [x] `servers/schema.py`: deprecation comment on `source_turn_id` — `9015636`
3. [x] `servers/dal.py`: extended `GraphDAL` with `add_source_refs` / `get_source_refs` / `get_nodes_referencing` — commit `8a52164`
4. [x] `servers/dal.py`: extended `TraceDAL` (no new class — "extend before create") with `store_embeddings` / `get_embeddings` / `find_unembedded` / `get_by_ids` (point/batch lookup, commit `d68bddc`)
5. [x] Daemon boot confirmed migration idempotent on existing brain — applied to live brain.db
6. [x] Test: `tests/test_schema_v27.py` (14 cases) + `tests/test_episodic_refs_dal.py` (23 cases) — all green

**Day 1-2 — Encoder I/O** (~1 day) — **not shipped yet; next session candidate**
7. [ ] `servers/scales/s1/encode.py::_build_user_content`: inject `[trace:N]` markers per turn (read trace_id from S0 trace fetch)
8. [ ] `servers/scales/s1/encoder_tools.py` (or wherever tool schemas live): add `source_refs` field to `remember_batch.nodes[].source_refs` + `revise_batch.revisions[].source_refs` + `brain_batch` remember/revise ops (§6.1, 6.2, 6.3)
9. [ ] `servers/brain_remember.py`: accept `source_refs` kwarg, validate (§6.7), persist via DAL
9b. [ ] **Engram cohort `co_anchored` writes (§10.2, decision 15)** — deferred to recall+eval block (consumer-side; without recall scoring against co_anchored these edges sit unused)
10. [ ] `servers/daemon_dispatch.py`: minimal validation per §6.7 (type check only — accept all trace_ids without existence verification; sparseness warning if >10 refs).
10b. [ ] `servers/brain_recall.py` graph traversal: include `co_anchored` — deferred to recall+eval block
10c. [ ] **S2Healer extension (§10.6, decision 24)** — deferred to recall+eval block
11. [ ] `servers/scales/s2/*` encoders: confirm they inherit the same tool surface
11b. **`get_traces` tool (§6.8, decision 23)** — PARTIAL ✅ 2026-05-24:
    - [x] `TraceDAL.get_by_ids(trace_ids)` shipped (commit `d68bddc`)
    - [x] `brain.get_trace(id)` / `brain.get_traces(ids)` shipped (commit `d68bddc`)
    - [ ] MCP wrapper + encoder-side tool surface still pending (own focused conversation)

**Day 2 — Trace embedding pipeline via embed_queue** ✅ SHIPPED 2026-05-24 (pull-reconciliation, not push-queue)
**Note**: Tom reframed the original push-queue design (`enqueue_trace`) into pull-reconciliation — the worker queries `find_unembedded(limit=5, scales=['s0'], ref_types=[...])` every tick. No queue state, restart-safe. Steps 12 / 14 effectively deleted; step 13 shipped as `_drain_trace_embeddings_once`.
12. [x] ~~`enqueue_trace`~~ — SUPERSEDED by pull-reconciliation; no push queue needed
13. [x] Worker drain phase added to `embed_queue._drain_trace_embeddings_once` — render per §5.3, embed batch, store via `TraceDAL.store_embeddings` (commit `7b5b845` + review fixes in `669ecee`)
14. [x] ~~Hook from `brain.remember`~~ — SUPERSEDED (no enqueue; worker reconciles)
15. [ ] Recall fallback for missing embeddings — deferred to recall+eval block (no consumer reads `trace_embeddings` yet)
16. [x] Test: `tests/test_episodic_refs_dal.py::TraceEmbeddingsDALTest` covers find_unembedded behavior; `tests/test_trace_embed_render.py` covers render templates

**Identity layer (I1-I3) — added to Phase A scope this session, ✅ SHIPPED 2026-05-24:**
- [x] `TraceDAL.set_identity` + `_stamp_identity` setdefault helper (commit `75075eb`)
- [x] `Brain.__init__` reads `BRAIN_OPERATOR_NAME` / `BRAIN_AGENT_NAME` from env via `daemon_config` helpers (`75075eb`)
- [x] `brain-env.sh` sources `~/.config/brain/env` so daemon launch path picks up identity (commit `65bf483`)
- [x] `_handle_trace_append` decodes JSON-string metadata (fixes pre-existing tool_result double-encode bug; `65bf483`)
- [x] Identity-unset signal moved from boot to write boundary — `TraceDAL._maybe_warn_identity_unset` (commit `987587f`)
- [x] **Historical identity migration** — all 57,546 historical trace_events backfilled (`scripts/migrate_trace_identity.py`, commit `5cff407`); also re-encoded 22,416 double-encoded legacy `tool_result` rows

**Day 2-3 — Render + recall** (~1 day)
15. [ ] `servers/contract.py::render_rich_node`: add `expand_source_refs` flag (default per format config); when set, fetch trace_events via batched query, render per §8.2, append to output
16. [ ] `servers/pipeline_contract.py::traverse` (or surface path): build the `{trace_id: trace_row}` cache once per recall batch (§8.4) and thread it to per-node rendering
17. [ ] `servers/brain_recall.py`: implement parallel-pathway scoring per decision 22 — `node_score = max(weighted_sum(framing_groups), source_summary_score)`. The source_summary score uses configurable aggregation (default `max`, see §9.4 — make `SOURCE_AGGREGATION` a module-level callable so the eval probe can test alternatives).
18. [ ] Engram cohort soft re-ranking (§10.2): post-selection boost for shared source_refs
19. [ ] Fallback path for unresolvable refs (§8.5, §12.3)

**Day 3 — Prompt v19** (~0.5 day, BLOCKING on §7.6 examples done)
20. [ ] Read v17 prompt from DB
21. [ ] Apply identity wording sweep (§7.3)
22. [ ] Insert new "Anchoring nodes in the substrate" subsection with §7.4 + §7.5 prose
23. [ ] Apply removals/modifications per §7.7
24. [ ] Insert §7.6 examples (from the focused follow-up session that filled them)
25. [ ] A/B the candidate as an override (`tests/interaction_override.py`) against the current default
26. [ ] Land the winner as `SYSTEM_PROMPT` in `servers/scales/s1/encoding_prompt.py` — the code default is the deployment
27. [ ] Same updates to S2 encoder prompts (community, consolidation, healer) — at minimum teach them the source_refs field and sparseness discipline

**Day 3-4 — Tests + validation** (~0.5-1 day)
28. [ ] `tests/test_episodic_references_roundtrip.py` (§13.4)
29. [ ] Extend `eval/agent_introspect/quality_probe.py` per §13.1
30. [ ] Build `eval/agent_introspect/source_fidelity_probe.py` per §13.2
31. [ ] Smoke test: `./dev python3 eval/longmem/harness.py --smoke-test` — green
32. [ ] Weight-tuning eval per §13.6: 3 configurations, pick best
33. [ ] Quality_probe on v18 vs v19 cohort comparison
34. [ ] Full `./dev pytest tests/` — all green

**Day 4 — Documentation + ship** (~0.5 day)
35. [ ] Update `CLAUDE.md` with the episodic-references concept and the source_refs field (brief — link to this doc)
36. [ ] Update `SKILL.md` if any operator-facing behavior changed (probably minimal)
37. [ ] Update `docs/RECALL-OVERVIEW.md` to mention source expansion at surface time
38. [ ] Update `docs/S2-DESIGN.md` to mention S2 encoders inherit source_refs
39. [ ] Commit + push

### Total estimated effort

3-4 days of focused work, sequenced. Each step is small individually; the integration test (§13.4) catches cross-step bugs.

### Open dependencies before starting

- ~~§15.1 (identity rendering)~~ — **RESOLVED 2026-05-23**.
- ~~Phase A substrate (steps 1-6, 13)~~ — **SHIPPED 2026-05-24** including the identity layer (I1-I3), historical migration, and pull-reconciliation embed worker.
- §7.6 (example rewrites) needs to be filled before step 24.
- Confirm the v17 prompt line numbers in §7.2 still match the active prompt at execution time.

### Status snapshot 2026-05-25 (mid-day, post Phase B substrate ship)

| Block | Status |
|---|---|
| Schema + DAL (steps 1-6) | ✅ shipped 2026-05-24 |
| Schema v29 — trace_id INTEGER → 8-char hex (brain-wide id consistency) | ✅ shipped 2026-05-25 |
| Embed worker (step 13, pull-reconciliation) | ✅ shipped 2026-05-24 |
| Identity layer (I1-I3, including historical migration of 57,672 rows) | ✅ shipped 2026-05-24 |
| Trace point-lookup API (MCP `get_trace` / `get_traces`) | ✅ shipped (string-typed post-v29) |
| Quality contract v3 (36 dims, Group 9 = D33-D36 example_authoring) | ✅ shipped 2026-05-25 |
| §7.6 wave-1 examples (7 examples, placeholder-clean) | ✅ shipped 2026-05-25 |
| Encoder source_refs WRITE path (MCP schemas + brain.remember/revise + dispatch validation, steps 7-10) | ✅ shipped 2026-05-25 |
| `[trace:<hex>]` marker injection in encoder input (`_build_user_content`) | ✅ shipped 2026-05-25 |
| Encoder prompt v22 — §7.4/§7.5 prose teaching encoder to USE source_refs | ⏳ next session — substrate-functional but inert until v22 |
| 3-way A/B eval (v22 vs v21 vs v19) before active production flip | ⏳ next session — discipline locked |
| `co_anchored` auto-edge in dispatch (decision 15) | ⏳ next session — substrate ready, ~30 min build |
| Render expansion at SURFACE_FORMAT (joint reactivation, steps 15-16) | ⏳ next-arc work |
| Recall consumers (`source_summary` scoring, S2Healer source_refs cleanup) | ⏳ recall+eval block, after render path closes |
| Wave 2 + wave 3 examples (shape diversity, domain breadth) | ⏳ deferred to S3 dynamic-selection arc |

**Reviewer follow-ups deferred from Phase B substrate ship**: F3 (GraphDAL commit-in-batch-mode audit, ~1 day), F6 (optional hex regex warning in `_validate_source_refs`, ~15 min), F7 (move `_SOURCE_REFS_SCHEMA` from `brain_mcp.py` to `contract.py` under new `JOIN_TABLE_FIELDS` category, ~45 min). See BACKLOG.md.

---

## 15. Open questions

These surfaced during the design session and need deeper conversation before execution starts. Each names the question, the current direction, and what's unresolved.

### 15.1 — Identity rendering — RESOLVED 2026-05-23

All four sub-decisions locked in decision 19. Summary:

- **Embedding**: concrete tokens (`Tom` / `Anchor`), not abstract slots.
- **Display**: asymmetric — first-person for Anchor, labeled for operator. Verbatim quotes + correction turns are carve-outs (stay labeled).
- **Frame→display**: pointer verbatim from `trace_events.metadata`; thin reconstructive frame when current partner differs from trace partner.
- **Multi-partner**: per-utterance binding, no participants list.

**Identity source — config file** (matches API-key single-source principle, decision `c39b6fdc`):
- `~/.config/brain/env` carries `BRAIN_OPERATOR_NAME` and `BRAIN_AGENT_NAME`.
- Daemon reads at brain construction, caches on `brain.operator_name` / `brain.agent_name`.
- `post_response_common` stamps each S0 trace event's metadata from cache.
- Missing keys → fail-loud at boot, no silent fallback to placeholder tokens (would corrupt embedding neighborhood).

Biology basis: research dive 2026-05-23 — concept cells (Quian Quiroga, Neuron 2026), self-referential processing (Northoff/Mitchell), per-utterance speaker binding (Mitchell &amp; Johnson 2009), constructive episodic memory (Schacter &amp; Addis 2007). Three independent research angles converged on the same architecture.

---

## 16. Future directions

Seven directions the episodic-references architecture enables but doesn't ship in v1. Each is its own design conversation when the time comes.

### 16.0 — Recall as cohesive-thought composition under budget

**The framing**. Recall today optimizes for "top-K relevant nodes": score each node against the query, take the highest-scoring K, truncate to budget. This treats retrieval as a ranking problem.

The real problem is **composition**: given a prompt or thought, assemble a cohesive set of pieces — nodes plus expanded sources — that compose into a coherent thought *within the budget* (today ~10k chars for additionalContext). Top-K-by-score doesn't optimize for this:

- Two top-scoring nodes might say the same thing (redundancy wastes budget).
- A moderate-scoring node might be the *bridge* that ties two strong ones into a coherent picture — and gets cut by the top-K threshold.
- Budget truncation drops pieces by position in the ranked list, not by how much they contribute to the assembled thought.
- Expanded sources (joint reactivation) consume budget per-node; some nodes are worth their expansion, others aren't, depending on what's already in the surface.

This is closer to a **structured prediction / combinatorial selection problem** than a pure ranking problem.

**Why this lands here, not in v1**. The v1 work (this design) introduces source_refs and multi-vector aggregation. Those are building blocks of the composition problem — but solving the composition problem is its own design effort.

**Components that the v1 design unlocks for the composition rethink**:

| v1 building block | What it makes possible at composition time |
|---|---|
| `source_refs` (decisions 1, 2) | Cross-node coherence signal — nodes sharing refs are about the same episode |
| Multi-vector aggregation for `source_summary` (§9.4) | The pattern generalizes: aggregating multiple vectors per group is a primitive that could apply to other groups (see below) |
| Engram cohort soft re-ranking (decision 15, §10.2) | First step toward coherence-aware selection — co-referenced nodes co-surface |
| Joint reactivation default (decision 14, §8.1) | Source expansion as a coherence primitive (an expanded source can pull related nodes via shared refs into focus) |
| `trace_event` metadata (decision 19) | Filter-based slicing of candidates (filter+embed hybrid, §16.3) — narrowing before composing |

**Where multi-vector aggregation generalizes** (the broader code adjustment Tom flagged):

The current brain has several spots where multiple vectors get collapsed into one — losing information that aggregation could preserve:

- **Field cohort** ([`pipeline_contract.py:120-160`](servers/pipeline_contract.py:120)) — per-field vectors (content, reasoning, user_quote, anchor_quote; situation is served by the dedicated `_situation` vector via fallback, not a field cohort) exist but are at **weight=0** in recall. Used only by the surface activation kernel. **Could be promoted** to participate in scoring via max-sim across per-field vectors — each field becomes its own pathway.
- **`edge_context`** ([`pipeline_contract.py:98-104`](servers/pipeline_contract.py:98)) — *all* of a node's edge descriptions concatenated into one vector at weight 0.55. A node with many edges produces a muddy vector. **Could be split** per-edge with aggregation (max or top-K) instead.
- **Community nodes with many member nodes** — the member nodes themselves aren't directly recall-anchored to the community node. A community node's source_summary could max-sim across its members' anchoring traces (a level of indirection past v1's scope).

If the source_refs aggregation experiment (§13.6 eval) shows max-sim is the right choice, the pattern is portable: apply it to the field cohort and edge_context, and we get parallel-pathway scoring at the field level — closer to the biological model of multiple engram-cell pathways co-firing.

**What composition-aware recall would look like** (sketch — needs its own design):

1. **Candidate pool with diversity-aware ranking**: top-N candidates selected not by pure relevance but by relevance × non-redundancy. (Maximal-marginal-relevance pattern from IR.)
2. **Coherence scoring**: pairwise signals between candidates (shared source_refs, shared edges, shared types) used to identify clusters of mutually-reinforcing nodes vs. isolated outliers.
3. **Budget-aware composition**: given a budget B, find the subset S that maximizes total coherent-relevance subject to `total_render_size(S) ≤ B`. Knapsack-style — NP-hard but well-approximated by greedy with coherence boost.
4. **Expansion choices part of the optimization**: for each node in S, decide whether to expand its source_refs (§8) based on whether expanded content adds value beyond what's already in S.

**What this would unlock**:
- A coherent surface instead of a top-K list (Anchor reads what *makes sense together*, not just what ranks highest individually).
- Budget used by composition logic, not by position truncation.
- Source_refs become first-class signals for "what's about the same episode" — coherence emerges from shared anchoring.
- The brain stops being "search engine for nodes" and starts being "thought assembler for the operator's question."

**When to do this**: not now. v1 ships the substrate (source_refs, multi-vector aggregation, engram cohort signal). The composition rethink builds on that substrate plus the eval data from §13.6. Expected: post-v1, after we have empirical evidence on how source_refs change recall quality at the node level — then we have the data to design the composition layer.

### 16.1 — Labile-state recall (biological reconsolidation)

When a node is recalled in a new context, biology says it becomes briefly labile and can be updated. An S2 unit could mark recently-surfaced nodes for re-evaluation against the current conversation — confirming, refining, or proposing supersession. Today Anchor's `revise()` is operator-initiated; this future direction adds a recall-triggered analog. The 2024 Neuron research on engram reconstruction is the substrate (Appendix A).

### 16.2 — Reconsolidation as source-comparison

Recall-triggered re-evaluation of semantic atoms against their source. An S2 unit could surface *"this node was encoded when source said X; source still says X — confirmed"* or *"source has been superseded — revise"*. Source_refs make this concrete: the source is right there, comparison is mechanical.

### 16.3 — Filter+embed hybrid recall

Query optionally extracts structured constraints (numbers, dates, tool names) and applies them to filter the candidate set BEFORE embedding scoring. Uses `trace_events.metadata` JSON for exact-match retrieval. Complements the embedding-only path. Useful for queries like *"what tools did I use on May 14?"* — exact-match metadata filtering is faster and more reliable than embedding similarity for these shapes.

### 16.4 — S2 referencing

An S2-scale recall mechanism would let S2 nodes anchor to other S2 events (community→community lineage, consolidation→community evidence, etc.). Decision 2 deferred this because S2 doesn't have its own recall path today. When that exists, source_refs extends naturally to target S2 events.

### 16.5 — Procedural memory as emergent property

As nodes with tool-use refs accumulate, S2 community detection will identify procedural patterns (recurring tool sequences for similar contexts). These emerge as nodes — typically `pattern` or `lesson` type — with source_refs spanning the tool-use episodes. No procedural module needed; the pattern surfaces through normal S2 consolidation. Decision 9 explicitly: experience anchors procedure, not the other way around.

### 16.6 — Backfill pass to retrofit pure-reference shape on existing nodes

A future S2 unit could scan existing nodes for "pure rewrites of single S0 turns" (low semantic divergence from one specific trace) and convert them to pure-reference (clear content, populate source_refs). Quality improvement, not correctness. ~5h LLM time on the current ~4500 nodes; one-time cost. Listed in §11.4.

### 16.7 — Archive-not-delete trace pruning

If storage ever becomes a concern (today: ~50MB/year growth — comfortable for years), an archive-not-delete pattern preserves trace_id validity while moving old trace content out of hot storage. Hard delete remains forbidden (it would dangle source_refs). §12.1.

### 16.8 — Trim historical pre-Phase-5 `co_accessed` edges

Today's graph contains `co_accessed` edges from the pre-Phase-5 era (when every recall created edges between all top-25 candidates, not just the post-Haiku selected ones). The noise undermines the new meaningful post-Haiku `co_accessed` mechanism plus the structural `co_anchored` signal (decision 15).

**The trim task** (runs any time after the episodic-references plan ships):
- Identify pre-Phase-5 `co_accessed` edges (by `created_at` < Phase 5 cutover date, or by lack of selection-source metadata if older edges don't carry it).
- Archive them (`archived=1`) rather than hard-delete (preserves the historical record while removing them from graph traversal).
- After trim, the remaining `co_accessed` edges represent only post-Haiku conscious selections — clean signal.
- After cleanup, `co_anchored` (decision 15) and post-Phase-5 `co_accessed` both carry meaningful signal; the graph layer becomes a real engram substrate rather than mixed signal + noise.

**Why deferred**: not blocking for the v1 episodic-references ship. The new `co_anchored` edges go in fresh; new post-Phase-5 `co_accessed` edges go in fresh. The historical noise affects existing graph traversal quality but doesn't break correctness. Run cleanup whenever — independent task, ~1-2h work.

---

## Appendix A — Reading list

External references that informed this design:

- CoALA framework — Cognitive Architectures for Language Agents (Sumers et al. 2023, updated 2025) — four memory types (working, episodic, semantic, procedural)
- "Memory in the Age of AI Agents" — survey, Dec 2025 (arxiv 2603.04740) — three orthogonal axes (form, function, dynamics)
- "Episodic Memory is the Missing Piece for Long-Term LLM Agents" — Feb 2026 position paper
- MemPalace (arxiv 2604.21284) — Wing/Room/Drawer pointer architecture; verbatim preservation
- MemMachine (arxiv 2604.04853) — ground-truth-preserving conversational episodes
- AriGraph (arxiv 2407.04363) — semantic + episodic knowledge graph
- Tulving & Squire — declarative (episodic + semantic) vs non-declarative memory taxonomy
- Hippocampus → cortex consolidation (standard model + reconsolidation work, PMC 5605913)

**Biology / neuroscience (added per the 2026-05-21 research dive)**:

- Hippocampal Indexing Theory — Teyler & Rudy (~2000); updating-the-index review (scite.ai); empirical validation [Concurrent feature-specific reactivation, PMC 8370760]; [The Hippocampal Engram as a Memory Index, PMC 6287299]
- Engrams (distributed, recruited at encoding, reactivated at retrieval): [Deconstruction of a memory engram, Nature Neuroscience 2026]; [Engram neurons: encoding to forgetting, Molecular Psychiatry]; [Engram Memory Encoding and Retrieval: Neurocomputational Perspective, arxiv 2506.01659]
- Complementary Learning Systems — McClelland, McNaughton & O'Reilly (1995); [Neural Network Model of CLS for Continual Learning, arxiv 2507.11393]
- Pattern separation (DG) + pattern completion (CA3): [CA3 Pattern Completion and DG Pattern Separation, PMC 3904133]; [Reassessing pattern separation in DG, PMC 3726960]
- Sharp-wave ripples + replay: [Large sharp-wave ripples promote consolidation, Neuron 2025]; [Sharp wave-ripple clusters enhance hippocampal-neocortical engagement, bioRxiv 2026]
- Schema theory + binding: [Schemas as scaffold for neocortical integration, PMC 9527246]; [Binding items to contexts via conjunctive representations, Nature Communications 2026]
- Reconsolidation — recall creates new engrams: [Reconstructing a new hippocampal engram for reconsolidation, Neuron 2024]; [Molecular mechanisms mediating engram retrievability, 2025]
