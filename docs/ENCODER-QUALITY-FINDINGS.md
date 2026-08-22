# Encoder Quality Findings — 2026-05-24 scan

**Status**: discovery-pass complete. Feeds v19 prompt + rubric work (§7.6 of EPISODIC-REFERENCES.md). Re-runnable methodology — not a one-shot artifact.

This doc captures what 6 parallel opus reviewers found across 327 nodes spanning 5 encoder prompt eras. The goal was failure-mode taxonomy building, not scoring against a rubric — the rubric is the byproduct.

---

## Methodology

### Why this scan, why now

Before drafting v19 of the S1 Scribe encoder prompt, we needed to know what the current encoder actually does — what it gets right, what it gets wrong, and what emergent qualities the field studies but no encoder-eval system measures. A rubric written from intuition risks rewarding the wrong axes. A rubric written from failure mode taxonomy earns its axes from real cases.

### The four interlocking loops (from the methodology design)

1. **Failure surfacing** — read real encoded nodes paired with the conversations that produced them. Name what's wrong freeform. The taxonomy of failure modes IS the emerging rubric.
2. **Pair comparison** — force articulation of "why is this one better?" across versions/types. Articulation grows the quality vocabulary.
3. **Process tracing** — record encoder tool-use sequence, not just output. Anti-patterns in process exist independently of anti-patterns in output.
4. **Anchor introspection** — periodically have a fresh Anchor read its own encodings and reflect on what's missing.

### Sample construction

327 nodes selected via stratified random sampling, deduped:

| Era | Window | n |
|---|---|---|
| A_pre_v3 | 2026-04-05 to 2026-04-19 (`encoding_agent` v1-v2, basic shape) | 73 |
| B_v3-v12_rapid | 2026-04-19 to 2026-04-25 (variants: two-register, temporal MVP) | 67 |
| C_v13_rewrite | 2026-04-25 to 2026-05-05 (33K-char prompt rewrite) | 66 |
| D_v14_stage1a | 2026-05-05 to 2026-05-17 (revise discipline) | 61 |
| E_v15-v17_current | 2026-05-17 onward (v15.11 correction-aspect-edges; v17 revise discipline strengthened) | 60 |

Plus 90-row rare-type top-up for under-sampled types (correction, moment, quote, concept, rule, open, gap, reframe, diagnosis) — deduped to net 327.

**Exclusions**: locked nodes (Anchor-authored direct, not encoder output); S2-encoded nodes (different prompts, deserve their own pass); aspect/community structural nodes (output of detection, not encoding judgment).

### Per-node packet shape

Each entry included:
- Node payload: id, type, title, content, KV metadata (situation/reasoning/quotes/etc), keywords, confidence, source_attribution, project, emotion_label
- Encoder version active at `created_at` (from interactions-table timeline)
- Source conversation: ~30 turns from S0 trace_events, speaker labeled, identity-stamped
- Truncation: 1500 chars per turn (one methodology gap — load-bearing context occasionally fell outside this window; see "Methodology gaps to fix on re-run")

### Reviewer brief

Six opus subagents in parallel, each with a ~55-entry packet. Brief explicitly disallowed:
- Applying v19 expectations to older eras
- Scoring against a rubric (this is taxonomy *discovery*)
- Skipping the unclear (the unclear is exactly what we want to name)

Each agent produced: per-node verdict + axes, cross-cutting observations, named emerging qualities, version trends, API friction log.

---

## Headline result

**Verdict distribution across 327 nodes**: 228 strong / 81 adequate / 18 weak (70% / 25% / 5%).

The encoder is doing real work. The failures that exist are **systematic, not random** — same patterns surface across reviewers and across eras. That's the signal worth iterating against.

---

## The 8 emerging-quality clusters

Reviewers named 60+ candidate qualities. Hand-clustered into 8 families, each with preserve-vs-fix members.

### 1. Title naming
The compression-as-handle family. **The single strongest quality differentiator across all eras** — ~25% of nodes (agent 1's finding).

**Preserve** (good in current encoder):
- `pattern_naming` — title is a compressed noun phrase creating a retrieval handle ("TRIZ observation", "SURFACE_MISS pattern", "Anchor's two audiences")
- `aesthetic_title` — title is memorable + precise; surfaces well across many queries
- `slogan_node` — 6-12 word mnemonic compactness ("Cluster not node, recognition not search"); the value IS the compressibility
- `boundary_naming` / `boundary_naming__terminology_anchor` — title names the terminology boundary explicitly
- `operator_idiom_preservation` — Tom's phrasing as title rather than translated to engineering-speak
- `retrospective_pattern_naming` — title looks back and names the pattern in hindsight

**Fix** (weak in current encoder):
- `broken_wire_naming` — title is mechanical and event-shaped ("Phase B-B complete") rather than concept-shaped; ages poorly

### 2. Voice & identity
Currently asymmetric — Tom's voice well-preserved, Anchor's voice under-emitted.

**Preserve**:
- `voice_preserved` / `tom_pushback_preserved` — Tom's terse pushback survives verbatim ("thats not accurate", "This sounds extremely engineered", "Even 0.15 sounds random") with punctuation/typos/emphasis intact. ~80% of correction/decision nodes.
- `authorship_voice` — when nodes are ABOUT Anchor's identity, register shifts from 3rd-person summary to 1st-person commitment. ~6% but disproportionately load-bearing.
- `anti_sycophancy_preserved` / `anti_sycophancy` — encoder does not smooth disagreement.
- `quote_as_method` — verbatim quote IS the encoding (rather than a paraphrased summary).
- `typo_fidelity` — operator's typos preserved when they carry register signal.
- `self_caught_correction` — Anchor catches its own misalignment without Tom prompting. Rare but identity-bearing.
- `anchor_introspection` — Anchor self-diagnoses a flaw in its own architecture. Era E only.

**Fix**:
- `my_raw_quote_smuggled` / **voice asymmetry** — `my_raw_quote` field captured at ~6% in Era E vs `their_raw_quote` at 80%. v17 over-corrected on emission discipline; Anchor's stance gets paraphrased into neutral. **Backwards for identity preservation.**
- `confabulation_self_diagnosis` — when Anchor self-diagnoses something it actually didn't catch, mis-attributing agency.

### 3. Lifecycle & staleness
**No half-life mechanism in the type system.** Status nodes age into noise.

**Fix-heavy cluster**:
- `closed_open_node` — `type=open` but body says resolved. **Operationally toxic**: Frame's `active_threads` section surfaces these as stale work.
- `ephemeral_plan_as_durable_node` — encoder treats one-week-half-life launches and test-run IDs with the same fidelity as architectural truths. ~15% of D/E-era nodes.
- `session_chronicle` — event-type nodes narrate session arcs ("X happened → Y diagnosed → Z shipped") instead of crystallizing the lesson. **Commit topology mistaken for knowledge topology.**
- `session_artifact_smell` — node smells like a session report, not a transferable claim.
- `status_node_drift` — node's stated status no longer matches reality.
- `transient_status_atom` — atom-grained node about a transient state.
- `roadmap_node_smell` — reads like a roadmap entry rather than a learning.
- `proposal_as_decision` — proposal encoded as if decided.
- `plan_after_execution_smell` — plan encoded after execution, when the outcome is the lesson.
- `commit_hash_as_node` — commit-shaped node where the commit hash is the most specific element.
- `metric_dump_node` — node consists primarily of metric values without framing.
- `artifact_delivery_doc` — node is a delivery doc fragment, not a learning.
- `implementation_plan_as_arch` — implementation plan filed as architecture.
- `next_session_handoff_as_node` — handoff fragment masquerading as a permanent encoding.

### 4. Preservation
"Anchor in source" family — what survives intact vs what gets paraphrased away.

**Preserve**:
- `preserved_computation` — source numbers kept alongside derived values ("250 of 440 done" stays with "190 left"). ~8%.
- `preserved_emotional_weight` — emotional register survives encoding intact.
- `two_register_atom` — verbatim specific + abstract principle in one node with clear demarcation. v17 hits this more often than earlier eras.
- `compound_emotion_preserved` — multi-dimensional emotional context kept.
- `contrastive_evidence` — node carries both "yes" and "no" cases.
- `concrete_numbers` — specific numeric anchors kept.
- `good_correction_lineage` — correction node points at the corrected fact via aspect-tagged edge.
- `negative_result_encoded` — failures encoded explicitly, not glossed.
- `symptom_to_fix_pairing` — symptom + fix bundled, not split.
- `gated_intent_captured` — the operator's gating intent ("only if X, else Y") preserved.
- `good_revision_history` — node carries revision lineage with reason.

**Fix**:
- `fabricated_or_outside_window_quote` — `their_raw_quote` contains polished restatement or text not in the surrounding turns. See "Quote-fidelity probe" below. **Trust-contract violation.**

### 5. Abstraction quality
The grain family — too zoomed-in vs too zoomed-out vs right grain.

**Preserve**:
- `good_atomization` — right grain; one concept per node. ~7%.
- `aesthetic_title` (cross-cluster with #1).
- `design_principle_from_eval` — eval observation abstracted into transferable principle.
- `rescued_insight_atom` — insight extracted from a noisy conversation as its own atom.
- `salience_grain` — node sits at the right level of abstraction for its claim.
- `frame_coherence` — node holds the conversation's register (technical / emotional / philosophical) coherently.

**Fix**:
- `atomization_too_coarse` — one node covering 3 distinct claims.
- `repetitive_principle` — same abstract claim restated across multiple nodes.
- `concept_drift_from_origin` — node's claim has drifted from the original source meaning.
- `dispelled_confusion_misclassified` — encoder records the corrected version but misses that this was a confusion-correction event.
- `knowledge_capture_vs_partnership_capture` — external-knowledge nodes (e.g., reference material) share render shape with partnership nodes (e.g., Tom-Anchor identity moments). No taxonomic distinction.
- `wrong_audience` — node written for a different future reader than will actually surface it.

### 6. Type fidelity
Type-content mismatch.

**Fix-heavy**:
- `wrong_type` — type label doesn't match the content (`decision` masquerading as `fact`, etc).
- `closed_open_node` (cross-cluster with #3) — `type=open` but body says resolved.
- `proposal_as_decision` (cross-cluster with #3).
- `plan_after_execution_smell` (cross-cluster with #3).
- `implementation_plan_as_arch` (cross-cluster with #3).
- `type drift on revise` — revise() updates content but doesn't re-audit type. Specific examples found: `fact` nodes holding 5-step procedural plans; `architecture` nodes with week-stale inventories.

### 7. Meta-cognition
The encoder noticing (or failing to notice) its own limits.

**Preserve**:
- `self_aware_encoder_meta_in_reasoning` — encoder writes "Node must reflect the executed outcome, not the draft plan" in reasoning field. Demonstrates awareness even when it doesn't reclassify.
- `meta_judgment_about_LLM_self` — encoder names an LLM-quality limit Anchor was navigating in the conversation.
- `stale_node_detected_inline` — when the encoder revises, it notes the staleness it found.
- `pivot_moment` — encoder marks the moment a thread pivoted (architecturally or relationally).
- `meta_log_observation` — encoder explicitly tags an observation about its own observation.
- `decision_with_future_pivot` — decision encoded with explicit "may pivot if X" gating.
- `reframe_trigger` — encoder names what triggered a reframe.

**Fix**:
- `audit_blind_spot` — load-bearing utterance falls outside the encoder's available conversation window. Methodology limitation as much as encoder limitation — see methodology gaps below.
- `confabulation_self_diagnosis` (cross-cluster with #2).

### 8. Trust
Quote fidelity, source attribution, anti-fabrication.

**Fix-heavy**:
- `fabricated_or_outside_window_quote` — see Quote-fidelity probe.
- `trailing_correction` — node's correction trail goes stale because reclassify never runs against it.

---

## Quote-fidelity probe — `their_raw_quote` is not always verbatim

Tested 50 randomly-sampled nodes with non-empty `their_raw_quote`. For each, pulled the full source conversation (no truncation) and classified the quote against actual operator messages in the window.

| Classification | Count | % | Meaning |
|---|---|---|---|
| verbatim | 17 | 34% | Quote string found exactly in a user_message |
| verbatim_caseins | 2 | 4% | Found, casing differs |
| substring_normalized | 1 | 2% | Found after stripping punctuation |
| fuzzy_match (≥0.7) | 4 | 8% | High word overlap, light cleanup |
| weak_fuzzy (0.4-0.7) | 2 | 4% | Some overlap, likely paraphrase |
| **not_found** | **17** | **34%** | Not in the ±70-min user_message window |
| no_window | 7 | 14% | Probe couldn't resolve the source session (Era A / pre-trace) |

Two confirmed paraphrase cases (manually verified against actual traces):

- **`d4d5ec24`** (Era B, "Two-register encoding: fact-capture vs abstraction should be separate jobs"):
  - Encoded `their_raw_quote`: *"i wonder perhaps the abstraction should happen in higher levels OR parallel, almost 2 encoders, we have many encoders in S2"*
  - Actual Tom message at that timestamp: *"Big consideration. Let's brake it down together: If we have S12 encoders we will need to somehow coordinate between them. unless one feeds into the other we will have 2 integrate functions..."*
  - Same spirit, different words. Encoder rewrote Tom's framing into cleaner prose **and tagged it as verbatim**.

- **`ca942a50`** (Era E, "source_refs: orthogonal field on any node"):
  - Encoded: *"Why define types? Can't we stay open form and deduce?"*
  - Window's actual Tom messages don't contain this phrase. Either pulled from outside the window or fabricated outright.

Accounting for confounders (no_window era, cross-session references, window narrowness), **the floor is ~10-20% of `their_raw_quote` values are not actually verbatim despite carrying the verbatim contract.**

The dual-register architecture (concrete tokens at embedding + verbatim quotes for source-memory anchoring) assumes these are sacred. They aren't.

**Proposed v19 fix** combines three layers (see EPISODIC-REFERENCES.md for the broader episodic-refs design context):

1. **Prompt rule** (encoder-side): "`their_raw_quote` MUST be verbatim. If you're tempted to clean it up, leave the field empty — that's the right move."
2. **Encode-time validation** (dispatch-side): when `their_raw_quote` is set, the dispatcher checks for verbatim match against recent S0 user_message traces. Reject (or downgrade to `user_paraphrase`) on mismatch.
3. **Source_refs as ground truth** (episodic-refs side): when `their_raw_quote` is set alongside `source_refs`, the quote is the literal text at that trace_id. Trace is the authority. Collapses verification into a structural check.

Same architecture solves both the quote-fidelity problem and the broader episodic-references work.

---

## Version trends across eras

Three patterns surface across reviewers:

1. **Earlier eras (v1-v2) hold qualities later eras lose.** v17 may be regressing on `revise-with-history`, `self-caught corrections`, `pragmatic inference on short quotes`, and (most clearly) `my_raw_quote` emission rate. v19 must not lose these.
2. **v13's 33K-char rewrite** introduced sophisticated framing AND the broken auto-keyword pipeline. Net was mixed; the keyword spam survived through v17.
3. **v15.11's correction-aspect-edges** helped corrections; didn't propagate `related_to` ban to non-encoder writers (the `auto_connect` default + `encode_cluster` were the actual pollutors — both killed in commit `c015d1b` 2026-05-24).
4. **v17's "revise discipline strengthened"** over-corrected on `my_raw_quote` emission. 0/6 in agent 5's Era-E sample.

---

## API friction findings (from dogfooding the trace API)

All six agents independently hit the same two bugs (since fixed):

| Bug | Fix commit |
|---|---|
| `query_traces(session_id=<historical>)` silently returned current-session events | `24e83bc` (2026-05-24) |
| `get_trace` / `get_traces` MCP tools didn't exist (only filter-based `query_traces`) | `c015d1b` (2026-05-24) |

Plus the SQL `datetime()` micro-second-strip bug found during methodology dogfood — fixed independently in `255b9de` + `3dd37d4` by the parallel datetime session.

Agent 3 named a future API the encoder would love: **`get_node_lineage(node_id) → {creation_chain, revision_chains, related_traces}`** — single call returning everything for reviewing a node. Hold as a candidate when designing the encoder's read surface.

---

## Methodology gaps to fix on re-run

- **1500-char turn truncation** hid load-bearing context in ~10-15% of nodes (`audit_blind_spot` emerged from this). The packet builder should keep full message text.
- **No `chain_id` per node** in the packet — agents couldn't compose from node to its encoding chain.
- **No `_corrections` / `access_count` / `last_accessed`** in the packet — quality judgments degraded for nodes whose correction history was load-bearing.
- **Era A pre-trace history** not pulled from JSONL fallback (the S0 API's `get_conversation_around` has this fallback; our SQL-only prefetch didn't).
- **±15-turn window** too narrow for ~11% of nodes.

These methodology limitations map 1:1 to what the encoder needs at runtime when v19 ships. **The packet builder spec IS the conversation-feed contract the encoder will need** (per Thread 1 Step 2 in EPISODIC-REFERENCES.md).

---

## What this means for v19

Five places where v19 should explicitly act (rubric-grade items, not just prompt tweaks):

1. **Reward `pattern_naming` in titles.** Single strongest quality differentiator. Add a "title-as-handle" rule with a positive + negative example.
2. **Symmetrize `my_raw_quote` emission.** Currently backwards. v17 over-corrected on emission discipline; v19 needs to teach when Anchor's stance is the encoding (~80% of correction/decision nodes — same threshold as `their_raw_quote`).
3. **Quote-fidelity is sacred.** Add the prompt rule + plan encode-time validation. Source_refs (episodic-refs work) is the ground-truth mechanism.
4. **Audit type on revise.** Add an explicit "if you're revising content, check whether the type still fits" rule. Catches the `closed_open_node`, `proposal_as_decision`, `plan_after_execution_smell` family.
5. **Lifecycle markers on ephemeral nodes.** Either a `half_life` metadata field, an explicit "is this status or learning?" type-routing rule, or a separate `event` / `status` type family that auto-archives. The cluster (#3) is the biggest single source of graph noise.

---

## File pointers (for re-running)

| Artifact | Path |
|---|---|
| Reviewer brief | `/tmp/encoder-scan/BRIEF.md` |
| Sample TSV | `/tmp/encoder-scan/sample-merged.tsv` |
| Prefetch script | `/tmp/encoder-scan/prefetch.py` |
| 6 packets (~55 entries each) | `/tmp/encoder-scan/packet-{1..6}.json` |
| 6 reports (full per-node annotations) | `/tmp/encoder-scan/report-{1..6}.json` |
| Quote-fidelity probe | `/tmp/encoder-scan/probe_quote_fidelity.py` + `/tmp/encoder-scan/probe_quote_fidelity.json` |

The `/tmp/encoder-scan/` workspace was kept intact after the scan — re-readable by the fresh session for the taxonomy walkthrough.

---

## Discipline lesson

When substrate ships specifically to enable a measurement, the measurement gets a follow-on task on the same plan — not "we'll see if we have time." This scan is the eval pass for the encoder substrate that's been accumulating since v1. v19 examples can't be authored credibly without it.

**The architecture isn't real until the eval runs.**
