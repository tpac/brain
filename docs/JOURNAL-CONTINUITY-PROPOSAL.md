# Encoder journal continuity: a shared working view

Design proposal, 2026-09-13. Subsequently approved for implementation by Tom.
Implemented in the isolated `codex/journal-continuity` worktree; not promoted
or deployed. The implementation keeps the existing call name,
`continuity(chain_id=...)`, rather than introducing the proposed
`begin_run`/`before_request` pair. The current contract is documented in
[TRACES-LAYER-DESIGN.md](TRACES-LAYER-DESIGN.md). The investigation and proposed
extensions below remain the decision record; reference syntax, explicit
reopening, and concurrent stale-write arbitration are not implemented.

Implementation validation: **298 tests and 12 subtests passed** across journal,
all encoder paths, Community membership, trace/API contracts, and ownership
guardrails. Eight captured history replays remain identical. On the 20:26
snapshot, the working view compacts 189 rows to 38 entries, renders 35 in 7,986
characters, and explicitly reports three omissions. The correct resolution
`3dfcd143` reaches the next request. No paid model evaluation or deployment has
run. Reproduction script and full rendered outputs:
`/private/tmp/journal-continuity-implementation/replay.py` and `replay.json`.

Review of `e1b50f5` covered placement, caller unification, cohesion, coupling,
and abstraction level. The owner boundaries require no restructuring. One
P2 selection leak was reproduced and corrected: an unchanged refresh could
admit old lifecycle rows from supporting history that the initial run window
or pin cap had excluded. The receipt now tracks admitted event IDs, extended
only by qualifying post-cursor lifecycle updates. Two regression tests cover
both exclusion paths and preservation of a newly admitted update.

The simplify pass extracted the shared complete-note renderer, removing the
working view's render-a-prefix/strip-its-header sequence. A deterministic
1,000-case differential comparison against `e1b50f5` preserved working text,
telemetry, and both historical render formats exactly. The full test selection
and eight captured replays above passed again after the correction.

**Recommendation:** extend `JournalBinding` with a run-scoped working view. Freeze the selection of private residue at the start of a run, but apply committed lifecycle changes to that selection before each subsequent request. Construct next-run continuity as compact state plus recent observations, rather than replaying every surviving assertion. Keep the historical journal append-only. All five encoders use this operation; none owns its own refresh policy.

This fixes the demonstrated handoff and repetition defects without making each batch consume every note written by its predecessors. It does not establish that an encoder's assertions are true, nor infer that two differently spelled subjects mean the same thing. An optional, shared reference extension below addresses targeting without fuzzy matching.

## Evidence and boundaries

Inspected revision: `3579bfa6819d68072fc94a480c48c11568bf16d8`, both this isolated worktree and production. Production `/Users/tpac/brain` was clean on `main`. Read-only daemon `ping` confirmed PID 30538, source `/Users/tpac/brain`, database directory `/Users/tpac/AgentsContext/brain`, loaded fingerprint `cb36cae55b922582`. No restart, live configuration change, journal repair, forced S2 run, or paid model evaluation occurred.

Starting evidence is in `/private/tmp/s2-postdeploy-audit/`: `continuity-investigation.md`, `continuity-events.json`, `continuity-replay.json`, `run.json`, `chain.json`, and `report.md`. I independently replayed all eight recorded cutoffs through the current `BrainTracesMixin.journal_notes` and renderer, using the 673 captured events in memory, with no Brain instance or database writes. Every row count, subject count, and character count matched. Live MCP point reads confirmed traces `3dfcd143`, `c9d09670`, and `ff6d0d98`.

| Before run | Rows | Subjects | Residue chars | Recorded residue + messages per batch |
|---|---:|---:|---:|---:|
| 15:44 UTC | 102 | 63 | 15,932 | 18,240 |
| 17:57 UTC | 164 | 57 | 25,463 | 27,771 |
| 20:26 UTC | 189 | 38 | 28,963 | 31,271 |

At 20:32:03, replay yields **zero** open rows for `482d2a1c_deploy_pending`; at 20:32:54, **one**. This isolates a working reader followed by a stale assertion. The original complete per-round input payloads were unavailable: historical input contents are reconstructed, while constant continuity lengths are recorded telemetry. This is not a model rerun or reconstruction of historical graph state. No counterfactual cost or quality improvement is measured here.

### Recovered rationale, checked against code

Brain nodes `2a81ea82`, `0e1a7303`, `7905e385`, `5d03ae49`, and `fdb13cc9` establish: self-grounding shared instructions; separate review, closure, and arc concerns; one parser and wire format across single-shot and loop encoders; caller-specific placement only. Current `journal.py` implements those boundaries.

Nodes `29de7994` and `243e39fa` describe open pins as one line plus a reader-derived count, and explicitly preserve later legitimate reopenings. Current code only deduplicates pins **outside** the recent-run window. Their claim of general one-line deduplication overstates the implementation. Their escalation-to-node language is also stale: current rendering hands long-lived items to Thalamus. Node `fe2c998d` and the archived journal design still describe Healer/Aspect as exempt; executable callers and guardrails prove that exemption is obsolete.

`cf7fe3e` refreshed Thalamus feedback while preserving frozen residue. Its parent already read `journal.continuity()` outside Community's loop. The bug predates that change. The comments correctly identify the danger of echoing fresh residue, but freeze lifecycle state along with content selection.

## Consumer map

Every modern residue read goes through `JournalBinding` → `brain.journal_notes` → `query_traces` → `TraceDAL`. Every normal Review write goes through binding harvest → `write_journal_notes` → `write_journal_note_rows`. Resolved/retire filter older matching notes at read time; open/still-open pin outside the recent window. The shared fetch is 200 events, not a character limit. K counts note-bearing chains after lifecycle filtering, not all completed integrations.

| Consumer | Scope and timing | Request shape and write timing | Limits and exposure |
|---|---|---|---|
| S1 Scribe, lived path | `s1` + full session ID; K=5. `encode._build_user_content` reads continuity once per encode in the dynamic user body. | One `run_llm_loop` per encode; rounds share conversation. Harvest after loop, including separate Arc accumulation. Next encode rereads. | No modern residue character cap. `ENCODING_AGENT.journal_max_chars=8000` applies to legacy blob only. Defaults include 20 messages, 5 rounds, 12,288 output tokens; these do not bound residue. **No demonstrated multi-batch freeze; shares history inflation and identity problems.** |
| S2 Community | `s2` + `community_detection`; K=3. Residue read before `_encode` loop; messages refreshed inside each batch. | Fresh `run_llm_loop` per batch. `_fold_batch_result` harvests immediately. Community graph refresh separately follows each batch. | Default 8 proposals/call; 48,000-char soft context target, 2 rounds, 32,768 output tokens. Singleton evidence may exceed target. `journal_max_chars=14000` unused. **Observed stale lifecycle handoff and repetition.** |
| S2 Consolidation | `s2` + `consolidation`; K=3. Same frozen residue/live messages split. | Fresh loop per cluster batch; same immediate fold/harvest owner. | 10 proposals/call, 2 rounds, 32,768 output tokens. Normal run cap is 10 clusters, but batching supports more. No total input-char target here; 14,000 journal setting unused. **Same latent handoff when multiple batches run; shares next-run defect.** |
| S2 Healer | `s2` + `healer`; default K=3. Same frozen residue/live messages split. | Multiple independent single-shot `_call_llm(..., journal=True)` requests. Shared call harvests and strips Review before JSON extraction; caller then validates/applies healings. | 10 nodes/call, up to 50/run, 4,096 output tokens. No residue or total input-char cap. **Same handoff in multi-batch runs, despite being single-shot per batch.** |
| S2 Aspect | `s2` + `aspect_integration`; default K=3. `continuity()` read once. | One single-shot `_call_llm(..., journal=True)` for proposals; harvest before JSON extraction and classification application. | 30 candidates/call, 8,192 output tokens. No residue char cap. **No subsequent batch in current implementation; shares next-run projection and identity behavior.** |

Verify effective interaction overrides when evaluating: table values describe code defaults, not a claim that every historical call used them. The live S1 bootstrap enables `BRAIN_S1E_LIVED_SEQUENCE=1`; the Python helper alone defaults off. Scribe's older blob branch remains callable by control/eval paths, reads/writes `encoding_journal_{session_id}`, and is outside the modern lifecycle mechanism. Preserve it for compatibility until separately retired; do not repurpose its 8,000-char setting silently.

Other relevant consumers/producers:

- S1 writes an `encoding-run-failure` journal row directly through attributed trace dispatch on failure. It uses the contract metadata builder, not a Review fence. Include it as an ordinary observation in the shared projection; do not force it through model parsing.
- `journal_notes(subject=...)` is the unfiltered, bounded hotspot history door. Generic trace tools, dashboard trace inspection, and eval scripts also inspect history. Preserve this observational API. `eval/sim_*_journal.py` and S1/LongMem entry points must opt into explicit history versus working-state reads where assertions depend on row counts.
- Thalamus owns tell/ask delivery, deduplication, settlement, expiry, and producer outcomes. Binding routes addressed lines, carries refused deliveries back as residue, and resolves matching producer items. Its producer view is already live per batch; contract limits are 10 rows and 2,500 chars. Do not reimplement that state machine in the journal.
- Arc → session context is a separate Scribe output (800-char accumulation cap), not journal-note lifecycle. Frame's prior, recall/surface, and integration run gating should not begin reading private residue. Gating and dashboard delta readers already exclude journal-note events from completed-operation deltas. No sixth journal-bound encoder was found in `servers/`.

## One owner per concern

| Concern | Existing owner and proposed extension |
|---|---|
| Scope, invocation lifetime, request preparation, harvest routing | `servers/scales/journal.py`: the binding owns a fresh run view and returns each request's continuity. |
| Event interpretation and compact projection | `servers/brain_traces.py`: add a working-view operation alongside the existing history door; one lifecycle reducer serves initial load and incremental refresh. |
| Grammar, tags, normalization, limits, render shapes, metadata | `servers/trace_contract.py`: extend shared contracts; no encoder-specific lifecycle prose or constants. |
| Ordered reads, cursors, paging, atomic append | `servers/dal_logs.py` (`TraceDAL`), reached through `brain_traces`; SQL remains here. |
| Request execution and per-batch completion | `IntegrationUnit` plus S1 `run_encoding`: call the binding before each independent request and harvest afterward. `runner.py` remains provider/transport machinery. |
| Messages to live people | Thalamus, unchanged; bind its producer view into the same per-request result. |

Legitimate differences remain data: S1 session versus S2 unit scope; K=5 versus K=3; S1 Arc opt-in; user-body placement; closure only for multi-round requests; batch packing and model limits. **Refresh, lifecycle matching, compaction, and overflow policy are shared behavior.**

## Proposed API and semantics

The following names are proposed, not implemented:

```python
# A binding is already attached to this encoder's identity.
journal.begin_run(chain_id)                 # reset even if unit instance is reused
context = journal.before_request()           # {text, request_id, stats}
# Caller places context.text in its established layout and runs the request.
payload = journal.harvest(text, chain_id, request_id=context.request_id)
```

`begin_run` obtains a `brain.journal_view(scope, k, policy)` snapshot with source event IDs, a stable read cursor, lifecycle state, and selected observations. `before_request` obtains committed changes through `brain.journal_changes(scope, after=cursor)`, applies them using the same projection, then renders residue plus current Thalamus outcomes. The existing `continuity()` can remain a compatibility convenience for a single fresh request, but production callers migrate together to the explicit lifecycle. `residue()` and `messages()` become composition details, not choices each batching loop makes.

The working view has two different responsibilities:

1. **Select private observations once per run.** Keep the baseline subject set and selected ordinary notes. New same-run subjects and new non-lifecycle prose are persisted but not added to later batch input. At next run they become eligible under the normal recency policy.
2. **Update the status of baseline subjects.** A committed resolve/retire removes their older selected notes and inserts one concise closure row, including reason and source trace. Retain that closure for the remainder of the invocation so a later batch knows *why* the old concern disappeared. Open/reopen transitions on baseline subjects update their status; don't append an ever-growing list of transition prose.

The baseline set includes eligible subjects omitted by rendering budget, so overflow does not make identity disappear. Select and render are separate steps. Transition reasons have a shared per-item cap; ordinary same-run notes stay excluded. A lifecycle event is an encoder assertion, not proof that the claimed deployment or graph operation succeeded—particularly important because Healer/Aspect harvest precedes applying their JSON payloads.

Use durable readback as authority, not the parser's `resolved` return list: the current writer can return parsed resolutions even if row persistence failed. `harvest` should return/record committed row IDs in its internal receipt; the public stripped-payload behavior remains compatible. A failed append must never make the next request believe an unrecorded resolution landed. Addressed routing keeps its existing independently isolated failure behavior.

### Next-run projection

Lifecycle reduction precedes rendering and budget selection:

- Keep one current lifecycle row per normalized subject/epoch, whether inside or outside K. Collapse repeated open/still-open assertions to the newest representative plus the distinct note-bearing-run count and first-seen information. A later resolution wins over earlier opens; a later open remains a reopening under the existing grammar.
- Keep distinct, recent **ordinary observations** under the existing recency window, removing exact duplicates. Do not blindly keep only one arbitrary note per subject: a node can have both a useful doubt and an unrelated friction. Lifecycle closure still retires older notes for that subject, matching today's contract.
- Keep recent closure summaries as context, even when no open remains. Older unresolved pins retain the existing separate cap. The appendix/history door exposes every original row.
- Preserve what `open_runs` actually means today: distinct runs **mentioning** the subject in the surviving epoch. It does not increase just because an unchanged pin is carried into another run. If elapsed-run age is wanted, introduce a separately named measure with its own evidence source; don't silently redefine ×N.

The current 200-event horizon can omit an old unresolved item or the closure that explains a newer assertion. Compaction alone cannot restore rows never fetched. Proposed reads use cursor paging with an explicit coverage flag and a contract-owned scan bound; run refresh must drain all new pages before advancing its cursor. If capped or failed, mark the result incomplete, retain the last known state, and log/trace the limitation. Do not claim complete historical state or durable “pinned forever” semantics. A full durable projection/index would be a separate extension if bounded read costs prove insufficient; it is not required to fix the observed in-run handoff.

One adjacent grammar trap needs explicit coverage: today's renderer emits `open ×N since ...` in the tag position, while `journal_key` only strips/casefolds and the parser preserves the echoed tag. A copied display head can therefore stop being a recognized lifecycle tag. In the proposed renderer, keep the machine tag exactly `open`, with persistence annotations outside the three-field head. If historical recovery is added, recognize only the exact shared renderer's suffix grammar in one contract helper; do not classify arbitrary tags beginning with “open” as lifecycle verbs. Keep the original authored tag in history. This is separate from subject typo recovery.

### Context budget

Add a shared `JOURNAL_VIEW_POLICY` in `trace_contract.py`, including K mapping, fetch/page bounds, pin cap, per-note cap, and residue character budget. Proposed initial residue ceiling: **8,000 chars for every modern binding**, to be calibrated before promotion. This is a new ceiling, not activation of Scribe's legacy setting. Existing message-view budget stays separate (at most 2,500 chars); report the sum to callers. Remove the two unused S2 `journal_max_chars` entries when migrating their callers.

Render complete rows, with lifecycle state and recent changes ahead of lower-priority observations, and reserve space for an omission notice with counts and coverage. Never cut a subject key in half. Bound closure reasons too; many transitions cannot bypass the cap. Retain omitted state internally and in history. It is impossible to show arbitrarily many open items in a fixed prompt; overflow must be visible and measurable, not presented as “nothing pending.” Community continues to own its 48,000-char soft target and complete singleton evidence; the new cap reduces journal pressure but does not guarantee every whole request fits.

### Scope, order, and concurrent requests

Use `(scale, session_id)` for S1 and `(scale, unit)` for S2, plus the invocation chain for working-view lifetime. Never key a mutable view globally by unit name alone. A new invocation clears run-local state; a failed invocation/retry can recover committed notes through the trace reader.

Current journal rows written together share a timestamp, and trace IDs are random. Timestamp-only pagination cannot safely resume inside such a group. Extend the DAL with an opaque ordered cursor using append position (SQLite rowid can serve internally while the journal is append-only); do not pretend lexical UUID order is chronology. Give legacy timestamp ties an explicit deterministic policy, with ambiguous contradictory same-append-batch lifecycle lines diagnosed rather than silently decided by query accident.

Current S2 is single-flight, and each encoder's batches are sequential. Per-request refresh is sufficient for that demonstrated path. If two requests for the **same scope** are permitted concurrently, freshness at read time alone cannot prevent stale writes. The write operation needs the request's observed subject revision and must diagnose conflicts against committed state under the existing logs write serializer. Persist the original assertion plus conflict metadata; don't let an assertion based on an obsolete revision replace newer current state. Different S1 sessions, or different S2 units, must never resolve each other's notes even when subjects match.

## Before and after examples

Observed quotes below are journal assertions from captured traces. Every proposed input/output is **hypothetical**, not a measured new model response.

### 1. Resolution within a multi-batch run

Observed chain `s2-20260913202631-community_detection`:

```text
20:32:02.774971, 3dfcd143
resolved · 482d2a1c_deploy_pending · 27ae615f confirms 3579bfa is the live
production HEAD (session + checkout both switched); bb69697+82e46b1 items
are subsumed — deploy is complete; retiring this watch

20:32:53.434600, c9d09670
open · 482d2a1c_deploy_pending · production brain deploy for bb69697+82e46b1
— prior resolved note cited 969f54c as of 09-13; no new confirmation this run
```

**Before:** both batches receive the run-start residue, including old deployment concern(s). The first resolution is stored but does not change the next input.

**After, hypothetical next input:**

```text
resolved · 482d2a1c_deploy_pending · Production 3579bfa confirmed;
bb69697+82e46b1 included. [trace:3dfcd143]
```

The old opens are absent. A newly written `friction · unrelated-input · ...` stays out until the next run. Expected—but model-evaluation-dependent—output is an empty Review for this subject, rather than another open. The projection fix cannot forbid a fresh model from making a false assertion anyway.

### 2. Next-run continuity and unchanged repeated opens

**Before:** next-run residue may carry repeated opens for the same key from all recent runs, plus a prior closure. At 20:26, 189 rows cover only 38 subjects.

Concrete repeated text appears in captured traces `79c46233`, `0baf1a5c`, and `fd1e0892`: subject `1dccd113_deploy_pending`, note “dbf040da handoff notes V3.4 pending production deploy.” These are three assertions from the same run, so their repetition adds neither a new subject nor another distinct-run count.

**After, hypothetical:** an unresolved subject repeated unchanged in five batches of one run and three distinct runs has one `open ×3` row, not fifteen lines. The count remains ×3 across batches. Distinct recent non-lifecycle observations are still eligible.

For the **actual historical** `3dfcd143 → c9d09670` sequence, compacting without changing lifecycle semantics must show one latest **open**, not magically declare the item resolved. That stale reopening already happened. The prospective handoff fix prevents its missing-input cause; it does not repair historical truth. Any repair would be a separately authorized, appended correction supported by evidence.

### 3. Legitimate reopening

Hypothetical new evidence after the confirmed deployment:

```text
open · 482d2a1c_deploy_pending · A new deployment check reports a rollback;
source trace NEW-EVIDENCE identifies the older running revision.
```

With the existing grammar, this starts a new epoch and must reach subsequent batches if the subject was in their baseline. Older epoch counts do not carry over. A different sentence alone is not machine-verifiable new evidence. Do not use text similarity or a permanent “resolved wins” rule to decide reality.

For stronger enforcement, an optional shared `reopen` verb can explicitly target the preceding closure and cite the new observation; see the identity extension below. That is a behavioral contract change requiring evaluation, not a hidden reader heuristic.

### 4. Mistyped or changed subject identity

Observed `ff6d0d98` resolves **`482a2a1c_deploy_pending`**, whereas the old open is **`482d2a1c_deploy_pending`**. Four original-key opens remain in the 15:45:54 replay. Refreshing more often does not connect them. `resolve_target` already recovers the old tag/subject inversion; it does not correct this typo.

**Recommended optional extension:** render an existing trace reference beside a state entry and accept that exact reference in the existing subject slot, e.g. `resolved · @<input-event-id> · why`. Contract parsing recognizes the reference; `brain_traces` resolves it to the canonical subject only within the binding scope and supplied request view. The model's original subject/reference is preserved in optional metadata. Existing text subjects and inversion recovery still work. A bad reference logs an unmatched-target diagnostic and never closes a nearby subject. No new identity table or fuzzy alias subsystem is needed.

Hypothetical: output `resolved · @OPEN-EVENT · confirmed` resolves the original subject even if descriptive prose spells its label incorrectly. A plain mistyped key still cannot be inferred safely. A genuinely new issue uses a new subject; a renamed label can keep targeting the old event identity. Reference support reduces copy errors, not semantic ambiguity. If adding explicit `reopen`, it targets `@CLOSURE-EVENT`, carries a nonempty reason, and preserves that causal link in metadata; presence of a citation is still not proof the cited evidence supports the claim.

### 5. Concurrent/session isolation

Hypothetical Scribe sessions A and B both have `open · release-question`. A resolves its own note. A's next invocation sees the closure; B still sees its own open. A reference to B's trace from A's output is rejected as out of scope. S2 Community's identically named note is likewise independent.

If same-scope requests R1/R2 both saw revision X, then R1 resolves it and R2 emits a stale open based on X, the optional revision-aware write check stores R2's assertion as conflicted history and leaves current state at R1's closure. R2 must first receive the closure to make an intentional reopening. Current sequential batch tests should not be mistaken for coverage of this race.

### 6. Two separate decisions

The provenance deployment (`bb69697`/`82e46b1`, subsequently included in `969f54c` and verified at `3579bfa`) was complete. This does **not** imply approval/promotion of S1E V3.4.

Captured `40e5417b` says `1dccd113_deploy_pending` is “V3.4 pending production deploy — same thread as 482d2a1c.” Other notes, including `1c20acef` and `e03b0975`, recognize uncertainty about whether these are separate decisions. Hypothetical corrected continuity holds a resolved provenance item and a distinct S1E promotion decision requiring its own evidence. The proposal neither grants that approval nor asserts the current status of later S1E variants.

## Alternatives and recommendation scope

| Alternative | Assessment |
|---|---|
| Reread full residue before every batch | Small change, fixes resolution visibility, but immediately echoes fresh private notes and can amplify them. Reject as the default. |
| Freeze everything and add a prompt instruction | Cannot supply unseen resolutions; today's instruction already says not to reassert. Insufficient. |
| Keep one ongoing model conversation across batches | Carries earlier outputs, but changes provider caching, context growth, request/retry behavior, and the meaning of independent batches. Too broad. |
| Apply only parsed resolutions in memory | Cheap, but diverges from durable state on failed writes and ignores external committed changes. Use committed trace events instead. |
| Deduplicate only at render time | Saves space but leaves state/matching inconsistent and fails the in-run handoff. Projection belongs before rendering. |
| Always prefer a resolution over later opens | Hides legitimate reopenings. Reject. |
| New journal database or global task ledger | Duplicates existing storage, scope, and lifecycle owners. Unnecessary for this defect. |
| Shared run view + compact projection | Recommended first implementation. Fixes the demonstrated boundary with existing grammar and owners. |

The smallest coherent release is shared run lifecycle, compact projection, coverage/overflow telemetry, and all five caller migrations. **Reference syntax, explicit reopening, and optimistic concurrency enforcement are a separately reviewable extension**, not prerequisites for sequential refresh. Until adopted, be candid: the first release cannot guarantee typo recovery or prevent every semantic reopening. Anti-drift means every encoder uses the same tested contract, not that all models become infallible.

## Migration and validation plan

No historical rewrite. Keep `journal_notes(subject=...)` and generic traces as history. Prefer an additive `journal_view` API so existing history tests and inspection tools keep their documented behavior. Migrate all modern production readers in the same reviewed change; preserve S1 control-arm behavior and Arc/Thalamus separation. Add optional event/cursor fields rather than making legacy rows invalid. Version working-view semantics in request telemetry. Update stale base/encoder comments and guardrail coverage in the same patch.

Meaningful deterministic checks before any paid evaluation:

1. Replay the captured corpus through old history and proposed projection. Original rows, IDs, order evidence, and subject history remain unchanged. Verify the exact `3dfcd143` closure reaches the next hypothetical request while new private notes do not. Verify the mistyped resolution does **not** close the original key.
2. Parameterize the same scenario over Community, Consolidation, Healer, Aspect, and Scribe adapters. First request sees an old open; harvest resolves it and emits a new private note; next independent request sees closure and fresh message outcomes, without that private note. For Aspect/Scribe, use a second invocation and verify new-run residue eligibility instead of inventing nonexistent in-run batches.
3. Exercise repeated opens inside K, outside K, and multiple batches on one chain; exact count semantics; legitimate reopen epoch reset; distinct ordinary notes; tag/subject inversion; failed/partial append; malformed fences; harvest before JSON extraction; refused addressed filing; empty Review; early return; retry and reused unit objects.
4. Exercise S1 A/B session isolation, S2 unit isolation, failed-run observations, interleaved same-scope events, equal timestamps, page boundaries, cursor advancement, truncated fetches, hard render ceilings, oversized subjects, and omission notices. If reference/revision support is included, add wrong-scope refs, unknown refs, renamed labels, stale-revision conflicts, and explicit reopen tests.
5. Strengthen `test_journal_binding_guardrail.py` beyond decorate/harvest token checks: every independent encoder request must pass through the shared preparation operation. Derive cases from the actual caller inventory; fail when a new caller lacks a case. No new per-encoder residue/messages concatenation or independent limit. Keep runner provider separation and current exception isolation.
6. Record selected/rendered/omitted rows, residue/message chars, coverage, cursor/revision, projection version, accepted lifecycle changes, and unmatched/conflicted targets with run/batch telemetry. These metrics expose growing state or a silently bypassed preparation path. Existing Community context-part accounting consumes the actual prepared text size.

Existing tests run here at the inspected revision: journal notes, lifecycle, component, binding guardrail, and Community decision context: **96 passed**; Scribe residue: **15 passed** (**111 total**). The original worktree `./dev` attempted a missing-runtime bootstrap and could not reach its download host; validation used the existing bundled Python 3.11 through `/Users/tpac/brain/dev` with this worktree as cwd. Fixtures use temporary brains; no second production writer was opened. Passing the current suite proves the present contracts, not the proposed fix.

**Another paid model evaluation is needed before promoting a behavior-changing release.** Deterministic tests can prove which input a batch receives and which events are reduced; they cannot show whether a model stops reopening stale concerns, retains useful residue, uses references correctly, or preserves graph-write quality after context changes. First run a bounded shared-block probe for grammar changes, then a small isolated multi-batch Community/Consolidation/Healer and multi-invocation Scribe/Aspect evaluation through the real entry points. Include accept and reject cases, a real evidence-backed reopening, the typo pair, and the deployment-versus-promotion distinction. Use `brain.run_s2()` for S2 activation and isolated starting states.

Record revision, isolated dataset, effective prompts/configs/model fingerprints, and per-request inputs. Compare lifecycle errors, retained useful observations, new-note echo, task correctness, graph/membership integrity where relevant, errors, and token costs. The S1 evaluation must include encoding/recall consequences, per project policy. Do not assume old simulator scripts still match current defaults. No paid call is authorized by this proposal, and no promotion or deployment should occur until Tom chooses the design and reviews the implementation/evidence.

## Source checkpoints

These anchors refer to the inspected revision; names are more durable than line numbers.

- [JournalBinding orchestration](../servers/scales/journal.py): `continuity`, `residue`, `messages`, `harvest`, `_route_addressed`.
- [Trace read/write owner](../servers/brain_traces.py): `journal_notes` at line 267; `write_journal_notes` at 403; row writer at 520.
- [Shared contract](../servers/trace_contract.py): normalization at 962; renderer at 1109; parser at 1215; inversion recovery at 1273; K mapping at 1459.
- [IntegrationUnit](../servers/scales/s2/base.py): binding at 204; per-batch harvest at 509; single-shot harvest near 623.
- [Community](../servers/scales/s2/community_encoder.py): frozen read at 453, per-batch composition at 479, fold at 526.
- [Consolidation](../servers/scales/s2/consolidation_encoder.py): frozen read at 131, composition at 166.
- [Healer](../servers/scales/s2/healer_encoder.py): frozen read at 87, composition at 98.
- [Aspect](../servers/scales/s2/aspect_encoder.py): one continuity call at 74.
- [Scribe wrapper](../servers/scales/s1/scribe.py) and [encode core](../servers/scales/s1/encode.py): binding passed at 80; continuity branch at 728, harvest at 284, legacy cap at 1317.
- [Current freshness test](../tests/test_community_decision_context.py): `test_second_batch_reads_new_message_but_not_same_run_residue` at 145; no existing-note resolution case.
- [Lifecycle tests](../tests/test_journal_lifecycle.py), [binding guardrail](../tests/test_journal_binding_guardrail.py), and [Scribe residue tests](../tests/test_s1e_residue.py): preserve historical semantics while adding working-view coverage.
