# Encoder Journals — Design & Considerations

**Status:** Living design doc. **Phases 2, 3, and 5 SHIPPED + LIVE on `main`** — the journal-note contract is now the only path for both S2 LLM encoders (consolidation + community), base journal legacy retired, daemon restarted on v20. **Next: Phase 4 — S1 Scribe**, eval-gated on the Frozen Corpus (highest risk — live recall path). Started 2026-06-15. (Phase 4 reorder: community came before S1E per the "prove on cheap S2 first" rule — see the 2026-06-24 Phase-5 decision-log entry.)
**Scope:** The journal mechanism across all five encoders — S1 Scribe + S2 Consolidation / Community / Healer / Aspect.
**Reframe in one line:** the "journal" becomes a residue-only, **episodic** stream of notes — what the encoder's *mind* did, not its *hands* — each note a Δ of `integrate()`, stored in traces (retained, never pruned), read recency-bounded (last 3–5 runs), and mined later by a future S3.

**Status legend:** ✅ agreed (Tom-driven or endorsed) · 🔶 open fork (pending decision) · 💭 Anchor's proposal (awaiting Tom)

> Backing brain nodes are cited by id for future sessions. Decision Log at the bottom is the chronological record — never overwrite it, append.

---

## 0. TL;DR

Encoder journals today are **five independent reinventions**, mostly **inert** (no scale reads them to act) and **siloed** (no cross-encoder sharing; S1 is session-walled; the trace copy is write-only). The redesign:

1. **Separate three conflated objects** — *Recent moves* (trace-derived facts, → inline in S1 Scribe), *Frame/arc* (frame of mind, unchanged), *Journal* (residue).
2. Make the journal a **residue-only stream of notes** — never restate the trace — under **one shared contract** (single-source prompt + templatized output). Each note is a Δ: `(tag, subject, note)`.
3. Store notes **as episodes in traces**, retrieved by recency/subject through a **traces-module function** (not `recall_episodes` — that stays clean for Anchor). Notes are **retained like all traces** (the trace store is never pruned); cost is bounded by the **read** (last 3–5 runs), not by deletion.
4. **Complete the delta trace FIRST** (Phase 0) — today it doesn't capture all actions, so the journal can't shed "what I did" until it does.
5. **Touch S1 Scribe last**, eval-gated — highest-risk encoder.
6. A **future S3 unit** mines recurring notes into durable findings (and carries the confabulation safeguards). Deferred — not now.

---

## 0.5 Definitions ✅

- **run** — one `integrate(O,K)→Δ` cycle of an encoder (one invocation). The universal unit of continuity. (S1: a Stop-triggered encode; S2: an idle cycle.)
- **note** — a unit of journal residue, `(tag, subject, note)`. It **is Δ** — an output of the run's `integrate()`, written as a trace event. A run emits its objective-ops delta **plus** 0–few note-deltas; **multiple deltas per run is expected.**
- **boundary** — what scopes a continuity *read*. **S1 Scribe = the session** (conversation); **S2 units = the unit + a recency window.** Session is a query filter, **not** a storage partition. S2 has no session — two S2-community runs are *two runs, zero sessions*.
- **episodic, retained** — notes are episodes (time- and subject-anchored) in the trace substrate, which is **append-only and never pruned** (verified 2026-06-16: 104K trace rows, oldest from the brain's inception). So notes are **kept, not decayed** — like every other trace. Cost is bounded by the *read*, not by deletion. Interim consumer pre-S3 = **the operator** (dashboard + weekly manual pass); a future S3 mines the full retained history.
- **two reads, two windows** — **encoder continuity** reads the **last 3–5 runs** (rolling: enough for cross-run escalation, bounded so a 9th run never sees the 1st's note); the **operator + future miner** read the *full retained history* (nothing is pruned). Bounding happens at the *read*, not by deletion.
- **note = symptom** — a note is a *described symptom*, neutral: a good surprise or emergent pattern belongs as much as a friction. Not a complaint log.

---

## 1. Problem / current state ✅ (findings, this session)

### 1.1 Five reinventions, no shared contract — `4530ef2b`
Each encoder hand-writes its own journal block in its own prompt; the same ~four concepts get different names and inconsistent presence.

| Concept | S1 Scribe | Consolidation | Community | Healer | Aspect |
|---|---|---|---|---|---|
| what I did (restates trace) | ENCODED | CONSOLIDATED/EVOLVED/KEPT | ACCEPTED | HEALED | — |
| declined + why | SKIPPED | SKIPPED | REJECTED | SKIPPED | — |
| cross-cutting observation | — | OBSERVATIONS | — | PATTERNS | — |
| forward / watch | WATCHING | WATCHING | — | WATCHING | — |
| session handle | SESSION_CONTEXT | — | — | — | — |

Every journal **leads with the redundant part** (restating the trace). The valuable part (the why/experience) is the rarest and most inconsistently named. Healer's prompt asks for a journal its **code discards** (verified `healer_encoder.py:94`). Aspect has no journal instruction at all.

### 1.2 Inert — a beautiful diary no scale mines — `a30153ca`
The journals are readable, disciplined, and **completely inert**. "Temporal scout is spurious" appears in 58 journals; CATALOG_BLIND recurs every consolidation run; the seed-duplicate operator-ask is flagged "highest-leverage" repeatedly — **all recurred, none acted on.** A cleaner template just gives a cleaner diary. The missing piece is a **mining loop** (future S3).

### 1.3 Siloed + usage map — `754fffaf`, `8b926e8e`
| Encoder | Continuity feed (`brain_meta`) | Who actually reads it | Trace `journal_entry` |
|---|---|---|---|
| S1 Scribe | yes, **session-scoped** | next S1E *same session*; + Frame → boot + surfacer every turn; + `SESSION_CONTEXT` → recall | dashboard display only |
| Consolidation | yes, global | **only the next consolidation run** | dashboard display |
| Community | yes, global | **only the next community run** | dashboard display |
| Healer | **none** | **nobody** (prompt's promise is false) | dashboard display |
| Aspect | **none** | **nobody** | dashboard display |

**Only S1's journal touches the live system** — via the Frame, session-walled. The trace `journal_entry` is effectively **write-only**. **Zero cross-encoder flow** anywhere.

### 1.4 What each unit actually reports — live pull (2026-06-15/16) ✅
| Unit | Journal home | What it reports on | Genuine residue |
|---|---|---|---|
| **S1 Scribe** | LLM → `brain_meta` (session) | ENCODED / SKIPPED / WATCHING + scout rejections | *thin* — temporal-scout false-positives (58 journals) |
| **Consolidation** | LLM → `brain_meta` (global) | EVOLVED/KEPT + OBSERVATIONS + WATCHING | *real but inert* — locked-duplicate wall, seeder root-cause, operator unlock-asks |
| **Community** | LLM → `brain_meta` (global) | community lifecycle + drift rejections | *real but inert* — unnamed-drift threshold ratchet (1.5→1.9) |
| **Healer** | code → trace only | counter-string ("Healed N… Fields: question=N") | **zero** — LLM's `PATTERNS` discarded by code |
| **Aspect** | code → trace only | counter-string ("N classified…") | **zero** |

- **F1 — Residue spectrum.** 3 LLM units produce *some* residue buried in 80–90% trace-restatement; 2 code units produce *zero*.
- **F2 — Every LLM unit has a recurring, inert friction** (same disease, 3 instances): S1 scout · Consolidation locked-dup wall + seeder · Community drift ratchet. The proof the redesign earns its keep.
- **F3 — Healer discards its best signal.** Prompt asks for `PATTERNS`; code overwrites with a counter-string. Live data: every run `Fields: question=N` — nodes systematically created without a `question`; reported as a tally.
- **F4 — Cross-encoder blindness.** Dupe *creation* (S1 CATALOG_BLIND; seeder) and *cleanup* (Consolidation) are one phenomenon from two ends; only a cross-encoder view links cause to cost — siloed journals structurally can't.
- **F5 — Unmapped integrate() points.** S1R (surface) writes a *selection* trace, no residue channel; the seeder/boot injection has no journal — invisible except as downstream friction.

### 1.5 The delta trace does NOT yet capture all actions — verified 2026-06-16 ⚠️ (Phase-0 motivator)
Max action-fields recorded across each encoder's full run history:

| Encoder (runs) | created | revised | **connected (edges)** | archived | action_details | Verdict |
|---|---|---|---|---|---|---|
| S1E `encoding_run` (1137) | ✓ ≤14 | ✓ ≤5 | **✗ 0** | – | ✓ ≤4 | edges missing |
| Consolidation (582) | – | ✓ ≤5 | **✗ 0** | ✓ ≤5 | ✓ ≤3 | similar_to/supersedes edges missing |
| Community (3703) | ✓ ≤10 | ✓ ≤21 | **✗ 0** | ✓ 1 | ✓ ≤59 | **member-edges missing — its main action** |
| Healer (419) | ✗ 0 | ✗ 0 | ✗ 0 | ✗ 0 | **✗ 0** | **nothing structured — only a count** |
| Aspect (28) | ✗ 0 | ✗ 0 | ✗ 0 | ✗ 0 | **✗ 0** | **nothing structured — only a count** |

**Two gaps:** (1) **`connected` (edges) is empty for every encoder, all ~5,400 runs** — the edge rollup is unwired (raw ops may survive in `action_details` for the 3 `brain_batch` encoders; confirm). (2) **Healer & Aspect record zero structured actions** — only a count; their real work lives *only* in the journal counter-string.

**Consequence:** the journal carries "what I did" **because the trace doesn't.** Stripping it now would lose edge formation (everyone) and Healer/Aspect's entire change record. **Completing the trace is the precondition that makes the residue-cut safe → Phase 0.**

---

## 2. Core principles ✅

| # | Principle | Node |
|---|---|---|
| 2.1 | **The journal is the encoder encoding itself — residue, not a trace copy.** The trace records what the hands did; the journal is the only place for what the *mind* did. First rule is subtractive: never restate the trace. | `ce333a73` |
| 2.2 | **The note is Δ.** Objective Δ (machine ops) and subjective Δ (the note) are both outputs of `integrate()`; a run writes both. Divergence between them is a free **drift signal**. | `8e53a042` |
| 2.3 | **Report across the perspective gap; stay silent within it.** Each process sends up only the slice invisible to its consumer. | `74cdb51d` |
| 2.4 | **Two-test filter.** *Reconstruction:* could a future run rebuild this from the brain? If yes, drop it. *Successor:* would the next self/operator be worse off not knowing it? If no, drop it. | `a05dd8e6` |
| 2.5 | **Producer open, consumer structures** (TRIZ via aspects). Production stays fully open (`tag`); stable grouping is imposed downstream at consumption (clustering), never at write-time. | `1ed54807` |
| 2.6 | **Inert without a mining loop.** Atomized + linked is necessary but not sufficient — emergence needs a higher scale (future S3) reading recurring patterns. | `a30153ca` |
| 2.7 | **The journal is episodic and retained; bounded by the read, not by decay.** Notes are traces — kept, not pruned (the trace store is append-only, verified). Cost is bounded by the *read* (last 3–5 runs for continuity). The operator mines them weekly pre-S3; a future S3 consolidates recurring patterns into durable findings. | — |

---

## 3. The three objects — the de-confusion ✅

| Object | Before (today) | After |
|---|---|---|
| **Recent moves** — what the scribes already did | LLM writes `ENCODED:` (restates trace); blob dumped into Frame "Recent moves" (survived `ea81820`) | **Trace-derived, relocated inline into S1 Scribe** ("where things stand"). Leaves the Frame — slot removed *after* the inline version is built + tested (Q4). |
| **Frame / arc** — frame of mind, session arc | scribe writes `SESSION_CONTEXT:` → `session_context` → Frame "Current focus" → recall | **Unchanged ✅** — the one piece doing real downstream work stays put |
| **Journal** — why I declined, what fought back, what I'm unsure of, what to fix, what surprised me | `SKIPPED:`/`WATCHING:` in a session-walled `brain_meta` blob; read only by the next same-session scribe; never mined | **Residue-only notes** (each a Δ), subject-anchored, **episodic in traces**; read by next run (continuity, last 3–5), the operator (weekly), and a future miner; **decoupled from the Frame; retained, not decayed** |

**Key clarifications (verified):**
- **Frame ≠ journal.** Three distinct objects. **S1 Scribe does NOT consume the Frame** (`encode.py` never touches `build_frame`) — it *produces* the journal + arc; the Frame *renders* them.
- **The conflation lives in the Frame's render.** Once the journal is residue, the Frame must not dump it raw into "Recent moves" — that would pipe self-doubt into the waking prior (the "organs must not editorialize the self" hazard, §4.2). So the residue redesign **must cut the journal→Frame wire** — done by relocating Recent moves inline into S1E (Q4).
- **Durable cross-session intent rides the graph, not the journal** — it's Anchor, it persists: a lasting "do X" is an *open node*, surfaced by recall when relevant. The note is session-local working memory.

---

## 4. What the journal is for, and what to report

### 4.1 Purpose ✅
A **post-action review**: the why's and the experience — *what went wrong, questions that rose, things to fix, why it rejected, **what surprised me (good or bad)**.* Notes are **described symptoms**, neutral. It serves two reads of one episodic stream: **continuity** (next run, recency-bounded) and **mining** (future S3, wide window). 

### 4.2 Helpful / Noise / Hurtful ✅
**Noise wastes attention; hurtful corrupts identity.**
- **Helpful** — surprise / violated expectations; friction with my own inputs (stale catalog, spurious scout, dupes); **patterns crossing a threshold**; irreversible moves proportionate to consequence; tensions surfaced; gaps noticed-but-unfilled; honest "nothing happened"; **good symptoms, not only bad.**
- **Noise** — restating the objective Δ; per-instance routine rejections; mechanical confirmations (counts/timing/tokens); compliance narration; volume to fill a slot.
- **Hurtful** — **confabulated introspection** (fiction about the self); **laundered certainty**; **editorializing about who I am** (verdicts, not observations); **negative-only bias** (no "this is working" → depressive self-model); **crying wolf**; **self-grading loops** (K-changes from self-report with no external ground truth).

### 4.3 S1 vs S2 audiences ✅
- **S1 reports up to the operator, live** — its uncertainties are actionable *now* (only scale that can ask Tom while he's here).
- **S2 reports to the identity + laterally to S1** — structural emergence only visible whole-graph (holes, redundancy, drift), plus feedback on S1's behavior.

---

## 5. The journal note schema ✅ (Q1 resolved — no closed roles)

A note is a **Δ written as a trace event**: `(tag, subject, note)` — **`subject` + `note` required; `tag` encouraged, open.** (`79106f53` → `e6ec91e0` / `6b87e55d`.)

- **subject** *(required)* — what the note is *about*: a node id / cluster / tool / input-channel. The load-bearing field — an **indexed `ref_id`** that makes notes mineable (N notes on one subject = a hotspot) **and** a quality gate: *if you can't name what it's about, it isn't a note.*
- **note** *(required)* — the prose: the why, the friction, the doubt, the surprise. The insight lives here; atomize the index, never the insight.
- **tag** *(encouraged, open)* — one word for the *kind* of thing (`friction`, `doubt`, `surprise`, `hunch`). Open vocabulary; reused when it recurs. A clustering accelerant, not structural — if absent, the miner groups by `note`+`subject`.

**Storage shape:** each note is a trace event — `scale`, `chain_id`, `session_id` (S1) / `''` (S2), `ref_type: journal_note`, `ref_id: <subject>`, `metadata: {tag, note}`. It is **separate from the run's objective delta trace** (created/revised/…), which stays. Per run = 1 ops-delta + 0–few note-deltas. The old `journal_entry` blob retires; the encoder writes line-per-note, the write path splits them.

**Read = recency-bounded, never all-of-boundary.** Continuity feeds the **last 3–5 runs** within the boundary (rolling). **A 9th S1 run does NOT get the 1st run's note** — only the recent few; enough for cross-run escalation, not stale anchoring weight. The boundary stops cross-session bleed; *recency* bounds within it.

**Storage is retained, not decayed.** Notes are traces, and the trace store is **append-only — never pruned** (verified: 104K rows, oldest from inception; only `hook_errors`/`debug_log` telemetry are pruned). Cost is bounded by the *read* (recency, above), not deletion. Interim consumer = **the operator** (dashboard + weekly pass); a future S3 mines the full retained history. (If global trace volume ever needs a retention policy, that's a system-wide decision, not note-specific.)

**Why no roles** (resolved 2026-06-15): roles are a *consumer* concern. A closed role-set at production contradicts the emergence principle (`751e77a3`), invites slot-filling confabulation, and is unnecessary — grouping emerges downstream (clustering). Mineability comes from `subject` + clustering, not a role key.

**How production is guided without roles** — a bar to clear, not buckets to fill: **(1) stance** (a note to your next self about what the brain can't see); **(2) two tests** (§2.4); **(3) subject requirement** (grounds + filters noise); **(4) examples + contrast** (teach by demonstration). Plus two guards: **empty is good** and **never restate the trace**.

---

## 6. Architecture ✅

| Layer | What | Status |
|---|---|---|
| **L0 · Complete the delta trace** | Wire the `connected` rollup (all encoders); make Healer/Aspect emit structured `revised`/`created`. **Precondition** — the journal can't shed "what I did" until the trace captures it (§1.5). | Phase 0 |
| **L1 · Unify + relocate** | One source in `trace_contract.py`: note schema, two tests, prompt block, parser; notes stored as `journal_note` trace events in the central store. | core |
| **L2 · Atomize + link** | Notes are `(tag, subject, note)`; `subject`=indexed `ref_id` → the queryable cognition layer. | core |
| **L3 · Retrieve** | A **traces-module function** — `notes(subject= / encoder= / since=)` — used by S1 (continuity, prompt-assembly) and S2 (the future miner), both DAL-level. **`recall_episodes` stays clean** for Anchor (S0 conversation); journal notes are a separate door, so residue can't leak into Anchor's prior — the guard is *structural*. | core |
| **L4 · Mine (future S3)** | A future S3 unit reads the full retained history, consolidates recurring `(tag, subject)` clusters into durable **finding nodes** + operator-asks. **Confabulation handling lives here** (recurrence filter, trace cross-check, human-gated K-changes) — deferred with S3. **Not now.** | future |

**Q2 resolved → traces-canonical, episodic.** Notes live as episodes in traces; scope is a *read-time query* (session for S1, unit+recency for S2); `brain_meta` journal keys retire. A single mutable key could only ever express one boundary — append-only rows express all of them (and fix both the S1 session-wall and the parallel-leak the old key caused).

---

## 7. Prompt & guidance refinement ✅ (exact wording finalized at Phase 2/4)

### 7.1 The shared single-source block (identical for all five) — roles-free
> **Your review — a note to your next self about what the brain can't see on its own.**
>
> Two tests before any line:
> • *Reconstruction* — could a future run rebuild this by reading the brain? If yes, don't write it.
> • *Successor* — would your next self or the operator be worse off not knowing it? If no, don't write it.
>
> Anchor every note to **what it's about** — a node, a cluster, a tool, or an input you were handed. **If you can't name what it's about, it isn't a note.**
>
> Add **one word** for the kind of thing it is — your word, whatever fits. *(friction, doubt, surprise, dead-end — examples, not a list.)*
>
> Note what stood out — good or bad. **A clean run is an empty review** — never manufacture notes, and never restate what you did (that's the trace's job).
>
> ```
> friction · temporal date-scout · every candidate was a misread number, not a date — 3rd run; may be net-negative.
> doubt    · nodes a1b2/c3d4     · merged them but unsure the claims match; flag if it resurfaces.
> surprise · recall-ranking      · the IDF boost made an old node outrank a fresh one — unexpectedly right; worth watching.
> ```
*(Note the positive example: symptoms aren't only complaints.)*

### 7.2 S1 Scribe structure — the read/write split (prevents confusion)
```
## Where things stand        ← READ (trace-derived recent moves + last 3–5 runs' notes)
## Frame of mind             ← READ (the arc / what we're working on)
## When you're done, three things:
   1. Encode — the nodes and edges
   2. Arc — ONE line: what progressed this run      ← protects recall; never merged into the review
   3. Review — [the shared block, §7.1]
```
Removing the old `ENCODED:` instruction kills the pull toward restating the trace.

### 7.3 Per-encoder parameterization
Shared block constant; each *journaling* encoder supplies only its own **examples** + **subject vocabulary** (S1: nodes/turns · Consolidation: clusters/survivors · Community: communities). **Healer/Aspect are journal-exempt** — mechanical units with objective actions but no subjective residue; they get Phase 0 (structured actions) and no notes. One source, small per-encoder slot.

---

## 8. Implementation plan ✅

**Spine.** (a) **Complete the delta trace first** — the journal can't shed "what I did" until the trace captures it. (b) **Prove the contract on an S2 encoder** (idle, off the live path) before the expensive one. (c) **Port S1 Scribe last**, eval-gated, doing the residue-cut + inline-recent-moves + Frame-slot-removal *atomically* (replacement-before-removal). (d) **The miner is a future S3** — deferred.

| Phase | What | Verify-check | Risk |
|---|---|---|---|
| **0 · Complete the delta trace** ✅ **SHIPPED** (`7f43c2d`) | dispatch-authoritative `affected`; directional `edge_relation_revised`; Healer structured `revised`; Aspect `classifications` | §1.5 gaps closed | done |
| **1 · Baseline** | Frozen Corpus on *current* prompts; record `s1_encode_eval` + arc-production + journal samples | we know numerically what "not broken" is | none |
| **2 · Contract** ✅ **SHIPPED** (main `6aa0ef2`) | Expanded under the ease-in: `trace_contract.py` schema/parser/`## Review` block + `JOURNAL_CONTINUITY_RUNS`/`RESIDUE_REF_TYPES`; **read door** `brain.journal_notes` (composes `query_traces`; gained `ref_id`/`chain_suffix`/`exclude_ref_types`); **write door** `brain.write_journal_notes`+`extract_review_block`; recall guard; gating + dashboard residue-exclusion. **All no-op until a prompt emits `## Review`.** | targeted sweeps green; collect-only 1745/1749 clean | low (additive) |
| **3 · Prove on Consolidation** ← **NEXT** (open with `/code-review` high) | Plumbing already built in Phase 2 — now only: DORMANT Consolidation prompt with the §7.1 `## Review` block (#8); wire the encoder to call `write_journal_notes` + read predecessors via `journal_notes(unit='consolidation')` (#9); activate + sync (#10); dual-path (old journal stays) | well-formed residue-only notes; recency-bounded continuity; shape test | low (idle) |
| **4 · Port S1E** ⚠️ | Restructure prompt (§7.2): residue notes + inline trace-derived "Recent moves" + **remove Frame "Recent moves" slot** — atomically; DORMANT → A/B → activate → sync | eval gate (§8.1) **+ Frame no longer injects the journal** | **highest** |
| **5 · Port the rest** | Community under the note contract; **Healer/Aspect journal-exempt** (keep their Phase-0 structured actions, no notes); retire old journal code + Healer's dead-promise prompt text | the 3 journaling encoders emit unified notes; per-encoder shape tests | low |
| **6 · Mine (future S3)** | S3 unit consolidates recurring notes → finding nodes + operator-asks; **carries the confabulation safeguards** (recurrence filter, trace cross-check, human-gated K) | a known recurring pattern surfaces — **the temporal scout** | **deferred** |

### 8.1 S1E eval gate (Phase 4) — A/B old vs new prompt via `build_corpus --interaction-override` + `sweep.py`
| Dimension | Why it's the guardrail |
|---|---|
| **Arc still produced** (binary) | recall pipeline must not break — the #1 confusion failure |
| **Encode coverage** (ENCODE_MISS) | did it still capture what matters, or get distracted journaling? |
| **Recall-conditional pass rate** | downstream quality unchanged |
| **Notes residue-only** (no trace-restatement) | did the two tests take? |
| **Notes not over-produced** (empty on clean runs) | over-production guard (full confabulation handling is S3-deferred) |
| **Token cost** | journaling overhead stays cheap |

Activate S1E's new prompt **only if** arc-production holds and coverage doesn't regress.

---

## 9. Risks ✅ (the honest "why it might be an issue")

1. **Confabulation** — an introspection slot invites narration that wasn't the real reasoning. *Pre-S3 defense:* the two tests + empty-is-good (reduce surface) + the operator's weekly read. *The structural solve* — never act on a singleton (recurrence filter), cross-check factual notes against the trace, human-gate any K-change — **lives at the S3/mining layer and is deferred with it** (Tom: "confabulation is S3, not touching now"). Safe to defer because **pre-S3, notes are only *read*, never *acted on*** — a fabricated note can mislead a glance, not drive a change.
2. **Grading my own homework** — auto-tuning K from self-report has no external ground truth. *Mitigation:* objective trace authoritative; the miner (S3) is deferred and, when built, *proposes* not *applies*.
3. **Tag sprawl** — open tags fragment. *Mitigation:* semantic `(tag, subject)` clustering downstream + periodic consolidation.
4. **Atomization vs narrative** — a run's residue is sometimes a story. *Mitigation:* prose in `note`; notes can link.
5. **Scope creep** — the miner is the bet, not the fix. Properly **deferred to a future S3**; the value (cut the junk, fix the trace, continuity) lands in Phases 0–5. A clean run produces an *empty* review.

---

## 10. Open questions / pending decisions

- **Q1 — roles:** ✅ **RESOLVED:** no closed roles. `(tag, subject, note)`; roles emerge downstream. (§5)
- **Q2 — continuity store:** ✅ **RESOLVED:** traces-canonical, episodic; read = recency-bounded query (last 3–5 runs, session/unit boundary); **storage retained** (traces append-only, never pruned — operator is the interim miner); `brain_meta` keys retire; `recall_episodes` untouched. (§6)
- **Q3 — mining:** ✅ **RESOLVED:** a **future S3 unit**, deferred. When built: consolidate recurring notes → findings + operator-asks; proposes, never auto-applies K. **Confabulation handling lives here too** (recurrence filter, trace cross-check, human gate) — deferred with S3; pre-S3 notes are *read, not acted on*, so it's low-stakes. (§6 L4, §9.1)
- **Interim consumer (pre-S3):** ✅ **the operator** — dashboard visibility + a weekly manual pass over the notes. This is what makes Phases 1–5 standalone-valuable (and dissolves the "cleaner diary that forgets" worry: notes are retained *and* read).
- **Q4 — "Recent moves":** ✅ **RESOLVED:** relocate inline into S1 Scribe (trace-derived); replacement-before-removal. (§3)
- **Q5 — proving encoder:** ✅ **Consolidation** (richest residue, idle) — working choice (§8 Phase 3); revisit only if it proves awkward.
- **Q7 — S1R (decoder) in scope?** 🔶 **OPEN.** Anchor leans **out** — it selects, doesn't write to the graph; high-frequency; recall-failure residue is partly visible via `query_logs`. Note as a future consideration, not a phase.

---

## Appendix A — current-state reference

- **journal storage keys (today):** S1 `encoding_journal_{session_id}`; S2 `s2_consolidation_journal`, `s2_community_journal`; Healer/Aspect none. (All retire under the new design.)
- **delta trace:** `build_delta_metadata` (`trace_contract.py:265`), all 5 encoders; `journal_entry` first-class; validated per ref_type via `METADATA_REQUIRED_BY_REF_TYPE`. **Gap (§1.5):** `connected` rollup empty for all; Healer/Aspect emit no structured actions — Phase 0 fixes this.
- **Frame (post-`ea81820`):** 3 sections — *What I've learned* (`wisdom` aspect) / *Current focus* (`session_context_for`) / *Recent moves* (`get_recent_encoding_journal` — **unchanged; our removal target**). SKILL.md stance injected at boot, outside `[BRAIN]`. ~6,495→~1,500 chars. `_render_current_focus` / `_render_recent_moves` kept verbatim by the parallel stream — clean rebase point is `build_frame`'s `sections=[…]` list.
- **recall:** `recall_episodes` (BrainEpisodesMixin, shipped `dd41eaf`) — S0-scoped (`EAGER_TRACE_SCALES=('s0',)`), excludes by default; **stays clean for Anchor.** Journal notes get a separate traces-module `notes()` query.
- **prompt seeds (mirror production-active):** `encoding_prompt.py`, `consolidation_enrichment_prompt.py`, `community_enrichment_prompt.py`, `healer_prompt.py`, `aspect_prompt.py`.
- **eval:** `eval/s1_encode_eval.py`; Frozen Corpus `eval/longmem/{build_corpus,sweep}.py`; `eval/frame_replay.py`.
- **prompt discipline:** register DORMANT → (eval) → `set_interaction_active` → `./dev sync-prompts`.

## Decision Log

- **2026-06-15** — Session of origin. Agreed (✅): journal = residue not trace copy; three-object split (Recent moves / Frame-arc / Journal); arc pipeline untouched; journal pushed to central + continuity; unify under single source + templatized output; touch S1E last. Open (🔶): Q1–Q6. **Not executing this session** — doc is the carry-forward.
- **2026-06-15 — Q1 resolved:** dropped the closed role-set. Item schema `(tag, subject, note)` — `subject`+`note` required, `tag` encouraged/open. Roles are a consumer concern, assigned downstream by clustering, not picked at production. Guidance shifts from enumeration to bar-clearing. Backing: `6b87e55d`, `e6ec91e0`.
- **2026-06-15 — Empirical grounding added (§1.4):** live pull of all 5 encoders. Mapped Healer/Aspect (counter-strings, zero residue); confirmed the recurring-inert-friction triad.
- **2026-06-15 — Corrections logged:** (a) Frame stays session-bound — no cross-session `recent_moves` carry. (b) Drop "handoff" framing — it's Anchor, it persists via the graph; durable intent = open nodes surfaced by recall, not a Frame slot (Active threads removed in `ea81820`) and not the journal. (c) No resolution/lifecycle tracking — recency is enough. (d) Vocabulary: *run* (universal continuity unit); *boundary* (S1S=session, S2=unit+recency); session is a query filter, not a storage partition. (e) S2 today = 14KB rolling blob, char-budget trim.
- **2026-06-16 — Reconciled to the committed Frame (`ea81820`).** Frame now 3 sections; Operator/Partnership/Active-threads removed; SKILL.md stance at boot. "Recent moves" survived unchanged (still raw journal) — removal is ours, gated on S1E. **Q4 resolved:** relocate "Recent moves" inline into S1 Scribe; replacement-before-removal; old "decouple Frame first" folded into Phase 4.
- **2026-06-16 — Episodic model + Q2/Q3 closed + Phase 0 added (this revision).**
  - **The note is Δ** (output of `integrate()`); a run writes multiple deltas (ops + notes). Not a separate species.
  - **Q2 resolved → traces-canonical, episodic.** Notes are episodes; read = recency-bounded query (boundary = session/unit). *(Originally said "storage decays" — **superseded same-day**: traces are never pruned, so notes are **retained**; see the Risk-review entry below.)* **K-runs continuity** per integration function — a 9th run does NOT receive the 1st run's note. Retrieval is a **traces-module `notes()` function** used by S1/S2; `recall_episodes` stays clean for Anchor (structural guard against residue leaking into the prior).
  - **Q3 resolved → mining is a future S3 unit, deferred.** It consolidates recurring notes into durable findings; proposes, never auto-applies K.
  - **Notes are symptoms, neutral** — good surprises belong as much as frictions (added a positive example to §7.1).
  - **Journal-as-integrate-interface** noted as an emerging observation; not formalizing the interface now.
  - **Phase 0 added (the gate):** verified the delta trace does NOT capture all actions (§1.5) — `connected` rollup empty for all encoders; Healer/Aspect emit zero structured actions. The journal carries "what I did" *because the trace doesn't*. **Completing the trace is the precondition for the residue-cut.** Tom starts here next session.
- **2026-06-16 (cont.) — Risk-review resolutions (Anchor's open worries, answered).**
  - **Traces are NOT pruned** (verified: 104K rows, rowid #1 present, oldest 2026-04-05; only `hook_errors`/`debug_log` telemetry prune). **Correction:** notes are **retained, not decayed** — kept like all traces; cost bounded by the *read* (last 3–5 runs), not deletion. The "cleaner diary that forgets" worry dissolves — notes are retained *and* read.
  - **Interim consumer = the operator** (dashboard + weekly manual pass) until S3. Makes Phases 1–5 standalone-valuable; no need to gate them on S3.
  - **K = rolling last 3–5 runs** (was an unresolved tension between tight-continuity and cross-run-escalation).
  - **Not all integrate functions need a journal** — Healer/Aspect are **journal-exempt** (Phase 0 trace-completeness only, no notes). Dropped "all five emit notes."
  - **Phase 0 is diagnose-first** — find *why* `connected` is empty before writing the fix; a dedicated stream owns it.
  - **Confabulation is S3-territory, deferred** (Tom: not touching S3 now). Safe to defer because pre-S3 notes are read, not acted on. Pre-S3 defense = two tests + empty-is-good + operator's weekly eye.
- **2026-06-18 — Phase 0 verified shipped; Phase 2+3 decisions locked (this session).**
  - **Phase 0 confirmed in `main`** (`7f43c2d`, ancestor of HEAD `1ef1cb0`): dispatch-authoritative `affected`, directional `edge_relation_revised` events (`connected` retired), Healer structured `revised`, Aspect `classifications`. §1.5 gaps closed in code. Deferred tail (low pri): I3 (Healer should read dispatch's `affected` not rebuild `revised_ids`).
  - **Scope for this session = Phase 2 (Contract) + Phase 3 (Prove on Consolidation, dual-path).** S1E port, Frame-wire cut, and old-journal retirement stay Phases 4–5. The full Frozen-Corpus **baseline (§8 Phase 1)** is deferred to the Phase-4 S1E gate where it's actually consumed (cheap journal-sample capture instead, now).
  - **Run-identity (the gap §0.5/§5 glossed) — RESOLVED: seconds-stamp the S2 chain_id.** "Last 3–5 runs" continuity requires grouping note rows by run. S1 already groups cleanly (`s1e-{session}-{stop}`, unique per run). **S2 does NOT** — `chain_id = s2-{YYYYMMDD}-{unit}` ([base.py:152](../servers/scales/s2/base.py:152)) collapses every same-day run of a unit onto one id, so grouping → "last K *days*", not runs. **Fix: `s2-{YYYYMMDDHHMMSS}-{unit}` — ONE combined timestamp segment** (seconds; a unit can't run twice in a second under min-interval gating). Verified safe across all consumers: `_last_run_timestamp` suffix-`LIKE '%-{unit}'` intact; `insights_scanner._unit_slug_from_chain` `split('-',2)[2]` discards the date segment; `encoding.py` `[-1]`=unit; `s2_runs.py` uses chain_id opaquely (and the change **retires its `nearest_ok` same-day-collision workaround**, lines 85–87/479). `# clock-ok` like the existing line. **This SUPERSEDES the interim "add `run_id` to note metadata" idea** — rejected as a parallel field; the chain_id should *be* the run handle, and then `notes()` groups by `chain_id` uniformly across S1/S2 with no special-casing.
  - **Per-encoder continuity K = contract constant**, NOT interaction-tunable (Tom's call): `JOURNAL_CONTINUITY_RUNS = {'s1e': 5, 'consolidation': 3, 'community': 3}` (default 3–5). Bounds the *read* (last K note-bearing runs), never storage — append-only, retained (§2.7).
  - **Notes stay OUT of `recall()` / `recall_episodes()`** (Tom confirmed) — enforced *free* by `EAGER_TRACE_SCALES=('s0',)` ([embed_queue.py:43](../servers/embed_queue.py:43)): journal notes are s1/s2 scale → never embedded → unreachable by semantic recall (which searches node + s0-trace vectors). Locked with a guard test. Reachable only via the `notes()` door + dashboard. (If a future S3 wants semantic note-clustering, that's an explicit opt-in embed scope — deferred with S3; pre-S3 the miner clusters by `(tag, subject)` strings.)
  - **Storage shape confirmed (§5):** one `trace_events` row per note — `event_type='delta'`, `ref_type='journal_note'`, `ref_id=<subject>`, `metadata={tag, note}`, `session_id` set for S1 / `''` for S2; sits *beside* the run's ops-delta, never inside it.
  - **Write ergonomics:** encoder emits a `## Review` section, line-per-note `tag · subject · note`; single-source parser in `trace_contract.py` splits with `maxsplit=2` (a `·` in the prose is safe), `tag` optional, `subject`+`note` required, malformed line → loud-log + skip (rest kept).
- **2026-06-22 — Phase 2 SHIPPED end-to-end + landed to `main` (`6aa0ef2`).** The **ease-in reframe** (decision `2063d7c9`: ship all plumbing as a proven no-op, then flip the prompt) pulled the write door into Phase 2 — so **Phase 3 is now *only* the prompt flip + encoder wiring.**
  - **Built (all no-op until a prompt emits `## Review`):** seconds-stamped per-run S2 chain_id (`s2-{YYYYMMDDHHMMSS}-{unit}`, cached, UTC); `journal_note` ref_type + `{note,tag}` shape/builder (caps text, rejects empty) + shared `## Review` block + `tag·subject·note` parser + `JOURNAL_CONTINUITY_RUNS` + `RESIDUE_REF_TYPES`; **read door** `brain.journal_notes` composing `query_traces` (gained `ref_id` + `chain_suffix` + `exclude_ref_types`; `get_recent` extended likewise; shared `_like_suffix_param` LIKE-escape, which also closed the latent `community_detection` underscore-wildcard bug); **write door** `brain.write_journal_notes` + `extract_review_block` (fenced-only, per-note-isolated, loud); `idx_trace_ref_subject`.
  - **Consumer guards (review #1/#2):** `_last_run_timestamp` off raw SQL onto `query_traces(exclude_ref_types=RESIDUE_REF_TYPES)` — a notes-only run no longer reads as a completed integration; dashboard `_fetch_ok_deltas` excludes residue (constant **replicated** locally — disconnection contract forbids importing `servers.*`). Recall guard test (notes s1/s2 → never embedded → never in recall).
  - **Architecture corrections banked (Tom, this arc):** trace reads go through the public `query_traces` API, never TraceDAL/raw SQL; chain_id *is* the run handle (no parallel `run_id`); dashboard replicates constants, never imports `servers.*`; CLAUDE.md states what IS (the trace-chain line tightened to current-state only).
  - **Cross-stream:** merged the sibling's community-membership work cleanly (only `dal.py` overlapped — `GraphDAL.reconcile_community_membership` vs `TraceDAL.get_by_ref_type`, different regions, no conflict); acked.
  - **Verification:** targeted sweeps green across every changed surface (trace/dal/s2/journal/mcp/recall/community/dashboard); collect-only clean (1745/1749). The full-suite *run* hung ~85 min on a pre-existing unrelated network/LLM test (journal code adds no blocking calls) — flagged as separate infra.
  - **Code-review:** deferred to the **start of the next (prompt) session** (Tom's call) — `/code-review` high, before the live prompt flip. **Everything is merged to `main`, so the default diff is empty — point the review at the committed range, file-scoped** (keeps the sibling's community work out): `git diff 1ef1cb0 main -- servers/trace_contract.py servers/brain_recall.py servers/dal.py servers/scales/s2/base.py servers/schema.py dashboard/queries/s2_runs.py dashboard/queries/insights_scanner.py tests/test_journal_notes.py tests/test_trace_contract_sync.py tests/test_s2_community.py` (`1ef1cb0` = pre-journal base; `dal.py` also shows the sibling's `reconcile_community_membership` — out of scope).
  - **Next session:** `/code-review` → Phase 3 #8 (Consolidation prompt). S1E stays Phase 4, eval-gated.
- **2026-06-23 — Review run + write-door hardening landed (`f007b32`); review gate cleared.** Ran `/code-review` (a single briefed stream, Phase-3-leaned) on the journal diff. Verdict: substrate solid, principles hold; findings all on the write path under real prose. Fixed + landed F1 (silent drop when `## Review` present but fence missing/broken — now a distinct `status` + warning per case), F3 (leading markdown bullet poisoned the `tag` — parser strips it), F5 (summary previews the loud-capped note). `write_journal_notes` now returns `{written, malformed, status}` with a status for every case (`ok`/`empty_review`/`no_review_section`/`no_review_extracted`/`error`), warns on every drift/drop, and is failure-isolated (never breaks the encoder's run). `extract_review_block` returns `None` (no/broken fence) vs `''` (clean empty review). F2 (the `·` delimiter is a model-fidelity gamble) intentionally deferred — instrument the `journal_note_malformed` rate from Phase-3 run one, add fallback delimiters only if real data shows drift. **The review gate is cleared — next session opens directly on Phase 3 #8** (no re-review needed; the hardening was the review's own findings). Sibling streams: the Phase-2 `community.py:260` raw-DML guardrail RED was fixed independently (`reverent-noether` @ `acf0bb1`); coordinated the main-moved heads-up to the active community stream (`37a32ee9`).
- **2026-06-24 — Phase 3 SHIPPED to `main` (`dcaf30e`); daemon restart held for the community sibling's next restart.** Consolidation now writes residue review notes via the journal-note contract. All-code — no prompt re-registration. Decisions this arc:
  - **Eager, self-grounding block.** Dropped the "two tests" (reconstruction/successor) as over-correction against the old journal's restatement disease, not an evidenced need (the probe never validly showed they filter). The block uses only universal referents (no `brain`/`trace`/`operator`/identity tokens) → host-independent and testable in isolation. EAGER by intent (Tom): no value-filter gate, iterate from live. Folded in Tom's terse "we already log your actions; no need to rephrase; stay sharp" close. **Supersedes §7.1's earlier wording** (which had "next self / what the brain can't see" + the two tests).
  - **Inject, not bake.** Block single-sourced in `trace_contract.JOURNAL_REVIEW_INSTRUCTION`, injected at runtime via `IntegrationUnit._inject_review_block` (same pattern as `## Edge Families`): the registered prompt is transformed at runtime (strip legacy `## Encoding Journal`, relabel the continuity line, append the block), never re-registered — so it iterates in one place and community/S1E inherit the mechanism. Read side: `_load_journal_notes_prefix` (failure-isolated) + `render_journal_notes_prefix`.
  - **Probe lesson (Tom's catch).** `journal_review_probe.py` A/B'd examples-vs-none (V1 spec / V2 contrast / V3 examples; then a self-grounding V_sg vs Tom's terse V_sharp). It validly tested **mechanics** (parse, calibration, self-grounding-across-framings) but its **quality** numbers were invalid — synthetic thin run-descriptions ≠ the encoder's lived rich run (the "writing blind" note exposed it). Deleted after it served its purpose. Lesson banked: a probe whose *input* isn't production-faithful validates mechanics, not quality; and watch for rules that are fixes-to-past-trauma dressed as evidence.
  - **Code review (high effort, 3 finder angles + verify): 5 findings fixed** — dead marker-count `outcomes` (would silently report all-zeros now that the markers are gone), drift-silent-noop (transform now logs loud on a missing legacy anchor), read-path failure-isolation, stale `K_SOURCES`, transform altitude (extracted to `base._inject_review_block`). 145 tests green; merged clean over the sibling's community work (zero file overlap).
  - **Restart deferred** to the community sibling (`37a32ee9`) — code is committed to `main` (not WIP), so its restart safely brings Phase 3 live alongside its community deploy. **Next:** observe live (next idle consolidation cycle — `journal_note_malformed`/`no_review` rates, read real notes); then Phase 4 (S1E, eval-gated) / Phase 5 (community onto the contract + retire legacy).
- **2026-06-24 (cont.) — Phase 5 community LANDED to `main`; daemon restarted; v20 active.** Both S2 LLM encoders now on the journal-note contract; base journal legacy retired. Both `## Journal` (consolidation) and Phase-3 residue confirmed genuine in production first (the gate: `a31b384d` — first post-deploy consolidation run wrote 3 genuine notes, 0 malformed).
  - **Community port = body-edit, NOT pure inject** (`debf94c9`). Community's `## Journal` sat mid-prompt (YOUR ROLE after it, so it can't be stripped-to-end) AND was overloaded as the reject-rationale channel (`reject (journal line)` woven through 5+ places). So unlike consolidation's runtime-strip, it needed a **body-prompt revise (v20)**: delete `## Journal`, reframe `reject → no action`. Key: the skip is trace-recorded by `match_proposals_to_actions`→`record_rejections` — the journal line was pure narration (Tom: "why say anything about reject? … we can deduce from traces"). Code: `JOURNAL_MARKERS=()`, `K_SOURCES→journal_notes`, `_inject_review_block` (no `legacy_heading`), `_load_journal_notes_prefix`, **per-batch `write_journal_notes`** (community appends multi-batch `final_text`; a single post-loop write keys on the FIRST `## Review` fence → drops all but the first batch's notes). Validated on IsolatedBrain (`eval/sim_community_journal.py`, holding the reusable `make_v20` landing transform): 16 notes, 0 malformed, suppression intact (Δ+11 == skips), sane accept/reject split (`d5cbc19b`).
  - **Consolidation aligned (Phase-3 patch, `e58f6f03`).** Code-review found the **mirror bug**: consolidation overwrites `final_text` + single post-loop write → drops all but the LAST batch's notes (latent until `max_clusters_per_run` > batch size). Fixed to per-batch write → byte-identical journal path to community.
  - **Dead-code retirement (own commit `6cb8047`, Tom's call).** Both encoders on the note contract + Healer/Aspect never journaled + S1E uses its OWN `encode.py` `_save_journal` ⇒ `base.py`'s `_load_journal_prefix`/`_save_journal`/`_extract_journal_entry`/`journal_key`/`JOURNAL_*` attrs had ZERO callers → removed. (S1E's own journal + the `brain_meta` blobs retire with the S1E port.)
  - **Merge reconciliation with the S2-telemetry stream** (`silly-edison` `4893691c`, which landed `_sum_telemetry`+`elapsed_ms`+token fields on the same encoder regions). Combined: kept BOTH per-batch journal writes AND telemetry; dropped main's post-loop `_save_journal` (method deleted). 5 full-suite failures dissected: 4 were stale-base (silly-edison's own fixes `352f3fb`/`5c91053` my fork lacked → resolved by re-merge), 1 pre-existing flaky `test_fetch_tools` orphan-drop (data-dependent on the live-brain copy, not mine).
  - **Live deploy (verified).** FF `main`→branch → register+activate v20 (`anchor:journal_port_v20`) → MCP `restart` (loads fresh `servers/*` from main tree, etime 23s) → daemon healthy + v20 active confirmed. Pre-deploy: full suite 1809 passed; 270 targeted green. NOTE: maintenance-lock makes the daemon *skip startup* — never use it for a deploy-restart; the MCP `restart` tool is the path.
  - **Code-review (high, 3 finder angles + verify):** consolidation multi-batch (fixed), harness verdict read last-3-runs not this run's chain_id (`1db9c191`, fixed: filter by chain_id + `sections>0` vacuous-guard), landing-ordering coupling (discipline, not a bug). Refuted: chain_id stability, `journal_entry` readers, `K_SOURCES` validation, write failure-isolation.
  - **Open / next:** **observe live** community runs (residue quality + `journal_note_malformed`/`no_review` rates, like Phase 3); the community **drift-guard** (no `legacy_heading` ⇒ no warning if the prompt regresses to `## Journal`; backstop = `no_review_section` rate) **deferred** (Tom: "later maybe", `54f07d38`); **`community_size` bug** surfaced BY the journal's first community run (`49b34921` — sizes come out wrong after member-add revisions because the encoder can't see the prior count; fix = derive from the live `community_member` edge count). Then **Phase 4: S1E** — eval-gated on the Frozen Corpus, the highest-risk encoder (live recall path).
