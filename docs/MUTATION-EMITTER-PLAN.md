# Mutation-Trace Emitter — Implementation Plan

**Rulings live in** [`MUTATION-EMITTER-DESIGN.md`](MUTATION-EMITTER-DESIGN.md). This doc is
the *work order* only — and the only doc in this set that carries line numbers, so
staleness is contained here.

> ## STATUS — steps 1-8 SHIPPED, MERGED, LIVE (2026-08-06)
>
> | step | state | commit |
> |---|---|---|
> | 1 — dispatch chokepoint | **DONE, live** | `537046d` (+ review fixes) |
> | 2 — append_batch atomicity | **DONE, live** | `5e024d9` |
> | (prereq) metadata-dict move | **DONE, live** | `053cfb0` — pure refactor, split out deliberately |
> | 3b-3d — emitter + registration | **DONE, live, DORMANT** | `c93b86d` |
> | 3e — enforce the revise pair | **DONE, live** | `e51727b` |
> | step-3 review fixes | **DONE** | `aa98604`, `50288fc` |
> | 4 — chokepoint hook + revise/revise_batch (+ batch revise accumulation) | **DONE, live** | `b5399b0` (+ `d2a2d9e` deploy finding) |
> | 5 — edge paths + DAL cleanups (`_emit_edge_revise_trace` deleted) | **DONE, live** | `7e74561` |
> | 6 — remember paths (`_emit_edge_traces` + `_infer_scale_and_chain` deleted) | **DONE, live** | `8ebec33` |
> | 7 — batch archive/absorb manifests + orphan property | **DONE, live** | `11d2d8d` |
> | 8 — archive cascade returns; inline trace DELETED; junk purge + hook:integrity rows | **DONE, live** | `b846106` (+ `782b8db` chain-collision fix) |
> | 9 — `bulk_archive_relations` primitive; dangling sweep commits + sweep trace rows | **DONE, live** | `a0a8562` |
> | 10 — S2 archive bypassers routed; healer dispatch hygiene; archive op survivor_id | **DONE** | `29f3c1f` (bulk — swept into a sibling's docs commit by an index-wide `git commit`, second occurrence of that incident class) + `c3b9c65` (review fixes) |
> | **11-12** | **OPEN — step 11 (dead code) small; step 12 parked by design. Dashboard display fixes still deferred.** | |
>
> **The emitter is the ONLY trace writer for mutations — zero legacy emitters remain.**
> Every mutation kind (node created/revised/archived/deleted, edge relations) flows
> from the chokepoint or, for the two non-dispatch paths (idle-maintenance junk purge,
> `health_check` hook:integrity archives), from a direct `emit_mutation_traces` call.
> The ORPHAN PROPERTY is pinned: a rolled-back brain_batch leaves zero trace rows of
> ANY kind. `TestOneWriterPin.ALLOWLIST` holds only `mutation_emitter.py`.
>
> **Citation stamp — step 9's line numbers are STALE.** Resolved 2026-08-04; steps 4-8
> have since rewritten `dispatch_write.py` and `dal_graph.py` (`delete_node_edges` now
> returns flipped pairs, not a count). **Re-grep every symbol; trust names, not numbers.**

**Every step must be production-correct alone, not merely green.** Merging auto-deploys
asynchronously: the daemon is launchd-pinned to the source checkout, and the next
`ensure_daemon()` or MCP health ping sees a fingerprint mismatch and issues
`launchctl kickstart -k` at an arbitrary moment after the merge. There is no manual gate
between steps.

**Test command:** `./dev pytest tests/`. Tier per step below; the **merge** runs the full
suite (SQL/DAL/dispatch blast radius — a `-k` filtered run misses the contract and
guardrail tests, which are named for the invariant, not the feature).

---

## Step 1 — The chokepoint (behavior-neutral refactor) — ✅ DONE (`537046d`)

> **Deviations from this plan, as built:**
> - `MUTATION_COMMANDS` was written and then **removed** — nothing referenced it until the
>   emitter, so it was speculative. It belongs in the step that uses it.
> - `_log_failed_batch_ops` moved to `dispatch_common.log_failed_batch_ops` (beside its
>   sibling `check_unknown_keys`), error key `encoder_batch_op_failed` → `batch_op_failed`,
>   context label now the call's `encoding_source` rather than a unit name.
> - A **fourth** caller was routed: `tests/test_connect_to_intra_batch.py`'s `_dispatch`
>   helper, whose docstring claimed "the same path MCP and hooks use" — true again now.
> - The mechanism had **zero** test coverage before being moved; a test was added and
>   negative-tested. Production-verified by a before/after probe: 0 → 1 `batch_op_failed`
>   rows on an identical failing batch.

**Goal.** One execution path for every dispatch command, with attribution resolved before
the handler runs. No traces change.

**Files.**

1. `servers/daemon_dispatch.py` — add `MUTATION_COMMANDS` frozenset (the 8 commands, §4 of
   the decision record) and `dispatch_command(brain, cmd, args, graph_changes)`:
   resolve entry → `check_unknown_keys` → resolve `session_id = caller_session(args)` and
   `chain_id = args.get('chain_id','')` → `entry.handler(...)` → return. **No emit yet.**
   Does **not** acquire the write lock — callers own it.
2. `servers/daemon_server.py:763-784` — keep the local `entry` lookup for *policy*
   (`is_write` → lock, `marks_dirty` → dirty flag); replace the execution body inside
   `_write` (`:777-781`) and the non-write branch (`:784`) with `dispatch_command(...)`.
   `caller_session` resolution at `:775` moves into the chokepoint; `_accumulate_touched`
   (`:789`) and the dirty flag stay in the daemon.
3. `servers/scales/s2/base.py:316-326` — replace the `entry`/`check_unknown_keys`/handler
   block inside `with brain.write_lock:` (`:323-324`) with `dispatch_command(...)`. Move
   the `_log_failed_batch_ops(brain, unit_name, cmd, result)` call (`:325`, currently
   **outside** the lock) into the chokepoint so the check generalizes to every caller
   including MCP — and so a writer is not called unlocked.
4. `tests/isolated_brain.py:218` — route `dispatch` through `dispatch_command`. This is the
   dispatch surface for the eval suites; it is why the chokepoint is registry-level.

**Not migrated, ruled explicitly.** 13 `eval/` scripts call `entry.handler(...)` directly
(`s1s_snapshot_replay.py:187`, `s1s_full_e2e.py:82`, `interview_encoder_probe.py:114`,
`encoding_agent_trace.py:210`, `encoding_prompt_eval.py:148`, `mcp_schema_gate.py:88`,
`s1s_ab_wiring_check.py:154`, `s1_encode_eval.py:128`, `s2_consolidation_eval.py:107`,
`s2_locked_probe.py:175`, `archive/encode_eval_v2.py:243`+`:245`, `longmem/replay.py:35`),
and 6 test files reference `_handle_*` directly (`test_brain_batch_op_contract.py`,
`test_edge_mutation_unified.py`, `test_project_provenance.py`,
`test_brain_batch_transaction.py`, `test_s1_scribe.py`, `test_revise_unified.py` — 104
reference lines). These keep working (the handlers are unchanged) but produce **no mutation
traces** post-flip. That is correct for eval replays, which must not pollute the trace
substrate they measure. Direct `_handle_*` import is **unsupported** for new code; the
negative grep-pin in step 3 enforces it for `servers/`.

**Tests.** `tests/test_daemon.py`, `tests/test_mcp_roundtrip.py`,
`tests/test_dispatch_contract_sync.py`, `tests/test_connect_to_intra_batch.py`,
`tests/test_s1_scribe.py`, plus a new assertion that `session_id` is non-empty for a
`remember` with `_caller_session` set (the pop-then-read regression pin — it will still fail
at this step because nothing consumes the resolved value yet; write it `@expectedFailure`
or defer it to step 6 and say which).

**Production-correct alone:** pure refactor; identical traces, identical returns.

---

## Step 2 — Harden `append_batch` (prerequisite for per-command rollups) — ✅ DONE (`5e024d9`)

> **Deviation:** added `rollback_unless_batched(conn)` to `db_backends/sqlite.py` as the
> mirror of `commit_unless_batched`, rather than inlining a `getattr(conn,'in_batch')` check
> — so both halves of the commit discipline live in one place.
>
> **Correction found while reviewing it:** the ROLLBACK is what makes the batch atomic, not
> the pre-flight validation. Verified by neutering each separately: without the rollback the
> mid-insert test fails; without the pre-flight, both tests still pass. Pre-flight is kept
> for what it actually does — fail before side effects. A comment claiming otherwise was
> corrected, and the test carries a note so nobody drops the rollback.

**Goal.** Make one-append-per-command safe *before* anything relies on it.

Today `TraceDAL.append_batch` (`servers/dal_logs.py:606`) validates **and** INSERTs in the
same `for ev in events:` loop and `raise`s mid-loop (`:626`); the commit at `:643` never
runs; `logs_conn` is default-isolation and nothing sets `logs_conn.in_batch`, so the
already-inserted prefix commits on the next unrelated logs write — **including the
emitter's own `_log_error`**. Today's one-append-per-mutation bounds the damage to a single
row; per-command batching would lose the rest of the command's traces *and* commit a partial
set that lies by omission. Worse: an unregistered ref_type raises on the **first** row
(`nodes.created` is first), producing zero mutation traces system-wide, visible only as
`_log_error` rows.

**Files.** `servers/dal_logs.py` — split the loop: validate **all** rows first, then insert;
roll back pending inserts on raise.

**Tests.** New: a batch with one invalid row inserts **nothing** and raises; a valid batch is
atomic. Existing: `tests/test_trace_contract_sync.py`, any test touching TraceDAL.

**Production-correct alone:** strictly hardens an existing path; no caller behavior change.

---

## Step 3 — Contract registration + the emitter module (dormant) — ✅ DONE (`053cfb0`, `c93b86d`, `e51727b`)

> **Deviations from this plan, as built:**
> - The metadata-dict move became its **own pure-refactor commit** (`053cfb0`) ahead of any
>   registration, so the `NameError` hazard was never in a diff that also added behavior.
>   Two facts learned by reading that shrank the job: `validate_trace_metadata` resolves the
>   dict at CALL time (so only the dict has the ordering constraint), and defining new shapes
>   above the dict sidesteps the trap entirely.
> - The emitter **filters** manifest rows to each builder's kwargs and logs unknown keys,
>   rather than splatting the row. Splatting means one spurious field raises and the
>   loud-wrap drops EVERY trace for that command. Same severity choice `check_unknown_keys`
>   makes. A test caught this on its first run.
> - The one-writer grep-pin landed **now**, with a shrinking allowlist, instead of waiting
>   for step 7 — so it guards the migration in flight. **Delete each allowlist entry as its
>   legacy site dies** (`tests/test_mutation_emitter.py::TestOneWriterPin`).
> - `3e` (enforcing the revise pair) revealed those shapes had been **declared but never
>   registered** — validation was dead for the two highest-volume mutation events. Verified
>   warning-silent three ways before landing, including a test that drives a real revise and
>   connect through dispatch and asserts stderr is clean (the only test that could notice,
>   since the warning is non-blocking).
> - Closed a **pre-existing** gap in passing: the dashboard's replicated `RESIDUE_REF_TYPES`
>   mirror had no pin at all. `test_dashboard_disconnection.py` now pins both mirrors.

**Goal.** Everything the emitter needs exists and is tested; nothing calls it.

**Files, in this order — the ordering is load-bearing.**

1. `servers/trace_contract.py` — **move `METADATA_REQUIRED_BY_REF_TYPE` (currently line
   1041) to below the last shape definition.** `REVISE_METADATA_SHAPE` is at **1215** and
   `EDGE_REVISE_METADATA_SHAPE` at **1259** — both *after* the dict. Registering them in the
   dict where it sits today is a module-level `NameError`; `trace_contract` is imported by
   `dal_logs`, so the import fails and **the daemon does not boot** on the next
   `launchctl kickstart`. Add a comment at the dict naming the constraint.
2. `servers/trace_contract.py` — `REF_TYPES` (`:61`): add `node_created`, `node_archived`,
   `node_deleted` at `("s0","delta")`, `("s1","delta")`, `("s2","delta")`.
   **This step is ADDITIVE ONLY.** Findings `E8` proposed pruning ~15 dead `REF_TYPES`
   entries in this same edit ("one edit, one review of the same table"). **Rejected**
   (2026-08-04): six of those entries are pinned by assertions in
   `tests/test_trace_system.py` and one has a live dashboard reader, so the prune means
   deleting contract-pinning test assertions — an edit that must be reviewed on its own
   merits, never as a footnote inside a commit whose headline is "add the emitter". It has
   zero dependency on the emitter (additions here, removals there). Logged in
   `docs/BACKLOG.md` (2026-08-04) with the verified inventory.
3. `servers/trace_contract.py` — new metadata shapes + builders for the three new types,
   placed with their siblings; then register all five in the relocated
   `METADATA_REQUIRED_BY_REF_TYPE`. **Verified warning-silent:** all current producers use
   the builders and already satisfy the two existing shapes, so registering them cannot
   fire on existing traffic.
4. `servers/trace_contract.py` — add `EMITTER_REF_TYPES` = the three new types **plus**
   `node_revised` and `edge_relation_revised` (decision record §6 — verified safe).
   **Not** added to `LLM_ENCODER_DELTA_REF_TYPES` (`:1091`) and **not** to
   `SAID_AND_DID_REF_TYPES` (`:198`) / `embed_queue.EAGER_TRACE_REF_TYPES` (`:46`).
5. `servers/scales/s2/base.py:291` (was `:331` before step 1 moved a helper out)
   `_last_run_timestamp` — exclude `EMITTER_REF_TYPES` the
   way `RESIDUE_REF_TYPES` rows are excluded. **Land the exclusion here, in the same step
   that registers the types** — arming the gate after traffic starts would let mutation rows
   re-arm the S2 idle gate in the interim.
6. `dashboard/queries/s2_runs.py` — mirror the exclusion list (`:31-61`) **and**
   `query_healer_runs` (`:488-539`, which filters no ref_type at all). Mirror, never import:
   `:23-24` states the servers-disconnection contract.
7. `servers/mutation_emitter.py` (new) — `emit_mutation_traces(brain, cmd, manifest, *,
   session_id, chain_id, encoding_source)`. Contents: the `brain.conn.in_transaction` gate
   (decision record §2.1); per-row scale/chain derivation from the row's own
   `encoding_source` (the `_infer_scale_and_chain` logic moves here, one copy); per-row
   completeness assertion reporting via `brain._log_error`; pre-flight
   `validate_trace_event` on every row; one `append_batch`; whole body loud-wrapped as
   `_log_error('mutation_trace_emit', ...)`, never raising into the caller.

**Tests.**

- **Runtime registration test** (new) — assert `validate_trace_event(scale, 'delta', rt)`
  for the cross-product of `EMITTER_REF_TYPES` × `('s0','s1','s2')`, plus a
  `METADATA_REQUIRED_BY_REF_TYPE` presence check. This replaces the design's reliance on
  `tests/test_trace_contract_sync.py`, whose extractor (`:56-85`) requires **literal quoted
  kwargs** — a table-driven emitter with a variable ref_type yields **zero** triples, so
  adding `mutation_emitter.py` to `TRACE_WRITER_FILES` (`:19-37`) would pass **vacuously**.
  (Same reason `_emit_edge_traces`' `append_batch` is invisible to that gate today.)
- **Negative grep-pin** (new, modelled on `tests/test_capture_grep_pin.py`, which exists in
  this tree) — no `_trace_dal.append` for mutation ref_types anywhere in `servers/` outside
  `mutation_emitter.py`. This is the pin that actually holds the "one emitter" invariant.
- Emitter unit tests against a hand-built manifest: row mapping, single `append_batch`,
  per-row scale routing (an `s2:consolidation` row and an `anchor` row in one manifest land
  on **different** scales and chains), empty-row silence, `in_transaction` → skip + error
  row, incomplete row → error row.
- Add the new DAL primitive to `tests/test_write_txn_discipline.py:221-226`
  (`BATCH_REACHABLE_WRITERS` is hardcoded).
- `dispatch_write.py` is missing from `TRACE_WRITER_FILES` — add it now (`brain_remember.py`
  is already there, `:22`).

**Production-correct alone:** nothing calls the emitter; registration is warning-silent;
the S2 gate exclusion is a no-op until rows exist.

---

## Steps 4-7 — Per-handler migration, one commit each, no double-write

**The mechanism.** `dispatch_command` emits **only if the handler returned a `mutations`
key**. A not-yet-converted handler yields no emit and its inline `_emit_*` still fires. Each
step below converts its handlers **and deletes their inline emits in the same commit**. No
reader ever sees two rows for one mutation, and every intermediate state is production-valid.

**Step 4 is the first step that writes a trace row.** `cp brain_logs.db
brain_logs.db.bak-{ts}` before it lands — steps 1-3 are invisible in production, step 4 is
not.

Each step's tests: the §6 pinned reader shapes against emitter-produced rows, plus that
step's own new-type coverage. Tier: targeted test files per step; full suite at merge.

### Step 4 — `revise` + `revise_batch` → `nodes.revised` — ✅ DONE

> **Deviations from this plan, as built:**
> - **The `dispatch_command` hook landed here** (this plan never named its step): identity
>   (`caller_session(args)`) and `chain_id` captured BEFORE the handler runs (handlers
>   mutate `args`); post-handler, `mutations` is **popped off the result** (never reaches
>   the agent's tool result) and emitted only when `result['ok']`. Capturing identity at
>   the chokepoint is the dispatch-level normalization the identity≠filter refactor left
>   open (brain id:90761bb9) — not a local patch.
> - **`co_anchored` → `noise` (Tom, 2026-08-04)**: the `brain.revise` /
>   `co_anchored_made` bullet below is SUPERSEDED — the refresh path stays deliberately
>   dark under the aspect coverage rule (see decision record §5). No pop-trap, no
>   `brain.revise` return change. `aspects_v1.json` (seed) edited in this step.
>   **Deploy finding:** heal REFUSED the addition on the live brain — the classifier had
>   already grown `co_anchored` into `temporal_sequence` in the WORKING copy, and noise
>   is exclusive. Required the supervised member-move (working-copy edit: out of
>   temporal_sequence, into noise + restart). Lesson: an aspect-membership check against
>   the SEED says nothing about classifier-grown working copies.
> - **The brain_batch revise branch converted NOW, not at step 7**: it splats `**r` into
>   agent-visible `results[]`, so leaving `mutations` on the sub-result was the 6M-char
>   class — instead the batch accumulates sub-manifests (`_accumulate_mutations`, the same
>   pop pattern as `affected`) and returns them top-level. Batch revises therefore emit
>   POST-COMMIT for the first time, and the rolled-back-batch → zero-traces acceptance
>   test landed here (revise path) rather than waiting for step 7.
> - **Direct-MCP revises now record `encoding_source='anchor'`** in trace metadata (the
>   chokepoint's command-level default) where the legacy emit recorded `''`. Scale and
>   chain fallback are unchanged; 'anchor' is the documented convention for unstamped
>   direct writes — truthful drift, noted for readers diffing old vs new rows.
> - **Review fix — the emitter now writes real summaries**: step 3 shipped it hardcoding
>   `summary=''`, which would have blanked the human line legacy rows carried ("revised 2
>   field(s): title, content") — all three step-4 reviewers flagged it. `MANIFEST_TRACE_MAP`
>   gained a summary column with legacy-identical formats for the revise pair and
>   `{verb} [type] title` for the three lifecycle kinds.
> - **Known narrowing, documented not built**: a brain_batch sub-op's OWN `chain_id` no
>   longer reaches its trace (the emitter's chain override is command-scoped; legacy honored
>   per-op). Unreachable today — no producer sets per-op `chain_id` — so no row-level
>   passthrough was added. If per-op chains become real, extend the manifest row and route
>   it in `build_events` alongside `encoding_source`.
> - **Tests migrated to the real door**: `test_revise_unified.py::TestTraceEvents` and
>   `test_mcp_roundtrip.py::_dispatch` drove handlers directly (the pre-step-1 idiom) and
>   went 0-trace when emission moved to the chokepoint — both now route
>   `dispatch_command`. New pins: manifest-never-in-result (single + batch), rolled-back
>   batch emits zero, batch revise emits exactly one post-commit.

- `servers/dispatch_write.py` — `_handle_revise` (`:597`), `_handle_revise_batch` (`:670`);
  delete `_emit_revise_trace` (`:185`) when both are converted.
- Per-row `encoding_source`: `:599` (`args.get('encoding_source','')`) and `:672`
  (`spec.get('encoding_source','') or top_encoding_source or ''`) become row fields.
- Emit gate is per row — a revise with empty deltas+warnings stays in `affected` and emits
  nothing (preserving `:610-611`'s unconditional `affected.revised`).
- ~~`brain.revise` returns `co_anchored_made`~~ — superseded by the `co_anchored` → `noise`
  ruling above.
- Pins: `tests/test_revise_unified.py:55`, `tests/test_project_provenance.py`.

### Step 5 — `connect` + `connect_batch` + `revise_edge` → `edges[]` — ✅ DONE

> **Deviations from this plan, as built (lens: clean the area, don't just add — Tom):**
> - **`_op_disconnect` converted HERE, not step 7** — its fabricated `0→1` flip is exactly
>   what `remove_relation`'s observed-truth return kills; leaving it two steps out of sync
>   with its own DAL primitive made no sense. brain_batch's manifest accumulator gained the
>   `edges` slot, and connect/disconnect sub-ops pop into it.
> - **`_emit_edge_revise_trace` deleted HERE, not step 7** — all four of its callers
>   (connect, connect_batch, revise_edge, disconnect) converted in this step, so the
>   same-commit deletion rule applies. `_emit_edge_traces` (connect_to/co_anchored) is now
>   the LAST legacy emit path; `_infer_scale_and_chain` survives only for it.
> - **Area cleanups ridden along**: the duplicated `enqueue_edge` try/except in
>   `add_relation`/`rename_relation` collapsed into `GraphDAL._enqueue_edge_embed`; the
>   duplicated birth-deltas lists in the INSERT/revive branches collapsed into one
>   `_birth_deltas` helper (where the new `archived: 1→0` revive delta lives once);
>   `_op_disconnect`'s separate `get_edge_id` lookup died with the fabricated flip.
> - **`revise_edge` envelope**: ops moved into `_revise_edge_ops` (envelope-agnostic);
>   the wrapper owns in_batch save/restore, rolls back on error AND on exception (a
>   mid-op raise must not leave uncommitted writes for the next batch's entry-flush to
>   silently commit), commits once (`# commit-ok` tagged for the txn-discipline pin).
> - **Row shaping unified**: one `_edge_row()` shaper in dispatch_write feeds all four
>   converted sites — builder-shaped by construction.
> - **Same `'' → 'anchor'` metadata drift as step 4, on the edge path**: an unstamped
>   `revise_edge` now records `encoding_source='anchor'` (chokepoint default) where legacy
>   wrote `''`. Only revise_edge — connect/connect_batch resolve `'anchor'` upstream,
>   disconnect resolves `'unknown'`. Scale/chain unchanged.
> - **Tests migrated to the real door** (`TestEdgeTraceEvents` → dispatch_command), the
>   revive pin now asserts the `archived: 1→0` delta explicitly; new pins:
>   disconnect-of-archived-relation emits nothing (the observed-truth headline),
>   edge results never carry `mutations`, rename+update = exactly one commit.

- `servers/dal_graph.py:1253` `remove_relation` — return
  `{edge_id, relation, flipped, deltas}` from the actual rowcount-checked UPDATE. Kills
  `_op_disconnect`'s fabricated `0→1` flip and its separate `get_edge_id` lookup.
- `servers/dal_graph.py` `add_relation` revive branch — add the **`archived: 1→0` delta**
  (nothing emits it today). **Leave `old=None`** on the revive branch: an empty `old` is the
  documented "just created" signal, and filling it would make a revive indistinguishable
  from an update for every reader of that convention. Amend the shape comment.
- `brain.revise_edge` echoes `source_id`/`target_id`/`warnings` (the handler already resolves
  endpoints, `brain_connections.py:254-255`). Its rename+update path is **two independent
  commits** (`dal_graph.py:1280` + `:1197`) — wrap in the `in_batch` save/restore envelope
  while touching it. `_handle_revise_edge` returns the dict verbatim (`:1167`) — pop.
- `servers/dispatch_write.py` — `_handle_connect` (`:1113`), `_handle_revise_edge` (`:1154`),
  `_handle_connect_batch` (`:1230`). Preserve the `'anchor'` edge default (`:1100`).
- Pins: `tests/test_edge_mutation_unified.py:81`, `tests/test_edge_relations.py`.

### Step 6 — `remember` + `remember_batch` → `nodes.created` + `edges[]` — ✅ DONE

> **Deviations from this plan, as built:**
> - **The last legacy emit family died HERE, not step 7**: converting remember's
>   connect_to path AND brain_batch's deferred connect_to pass left `_emit_edge_traces`
>   and `_infer_scale_and_chain` callerless — deleted same-commit, with the dead
>   `brain_today` import. `dispatch_write.py` no longer touches `brain._trace_dal` and
>   left `TestOneWriterPin.ALLOWLIST`. Step 7's remaining scope: archive/absorb manifest
>   accumulation + the orphan/absorb-unwind tests only.
> - **co_anchored: popped everywhere, traced nowhere** (noise ruling, step 4) — the
>   graph edge is still written; the two tests pinning the old tracing behavior now pin
>   the no-trace contract plus edge-still-written.
> - **`nodes.created` rows route per-spec `encoding_source`** in remember_batch (each
>   node's own creator decides its scale/chain); edge rows keep the top-level source,
>   bit-compatible with legacy's per-list resolution.
> - **"Un-skip the step-1 regression pin": no such skipped pin exists in the tree.**
>   The new `test_remember_emits_attributed_node_created_and_edge_traces` pins what it
>   described (session attribution through the chokepoint, impossible pre-step-6).
> - **Review hardening**: `_accumulate_mutations` is slot-agnostic (a step-7/8 archive
>   slot accumulates instead of silently dropping); stale co_anchored-tracing comments
>   corrected in brain_remember.py + trace_contract.py.

- `servers/dispatch_write.py` — `_handle_remember` (`:440`, `:447`),
  `_handle_remember_batch` (`:527`, `:535`).
- **`co_anchored` edges do NOT enter the manifest** (ruled at step 4: `co_anchored` →
  `noise`, decision record §5). Today remember's co_anchored edges get orphanable
  `edge_relation_revised` traces via `_emit_edge_traces` — under the coverage rule they
  STOP here, same treatment as `emergent_bridge`. Only `connect_to` edges ride `edges[]`.
- **This step fixes the pop-then-read bug** (decision record §1.3): session now comes from
  the chokepoint, resolved before `_pop_session_ctx` mutates `args`. Un-skip the step-1
  regression pin here.
- **`emergent_bridge` does NOT enter the manifest** — it is in the `noise` aspect, so the
  coverage rule excludes it (decision record §5; Tom 2026-08-04: *"co_accessed and emergent
  isnt interesting for now"*). No plumbing of `_bridge_at_store_time`'s return is needed;
  it keeps discarding its rows. Measured for the record: 23 created/week, 1,476 ever —
  `bridge_max_per_remember=2` is a ceiling, not the rate.
- Pins: `tests/test_mcp_roundtrip.py:614-644` (the d857e84d attribution pin).

### Step 7 — `brain_batch` + batch ops → merged manifest; all `_emit_*` deleted — ✅ DONE (`11d2d8d`)

> **Deviations from this plan, as built:**
> - **Absorb rows are attributed with `archived_by`** (the resolved actor), not the
>   op/batch `encoding_source` chain — review finding: absorb stamps its graph writes
>   with `archived_by` (which folds in op-level `archived_by`), so a row carrying the
>   command runner instead could contradict the graph and mis-route scale (an
>   `archived_by='s2:cleanup'` op's rows landing on s0). Pinned by
>   `test_absorb_rows_follow_op_level_archived_by`.
> - **The `_emit_*` helpers were already gone** — steps 5-6 deleted them ahead of
>   schedule; this step's scope was manifest accumulation + the orphan tests only.
> - **The orphan test was scoped to `EMITTER_REF_TYPES` at this step** — archive_node's
>   inline trace (a `tool_result` on a different connection) still escaped a rollback.
>   Step 8 widened it to zero rows of ANY kind, exactly the pairing the plan predicted.
> - The absorbed node's `nodes.archived` row landed at step 8 (when archive_node
>   started returning its cascade results); its inline trace kept coverage meanwhile.
> - **Flagged, not changed**: `_resolve_archived_by` falls back to `'unknown'` for an
>   unstamped batch op while the connect handler's rule is "unstamped IS anchor" —
>   pre-existing inconsistency; trace rows mirror the stored `'unknown'` truthfully.

- `servers/dispatch_write.py` — `_handle_brain_batch` accumulates sub-manifests the way it
  pops `affected` today (`:919`/`:941`/`:953` inheritance sites); the deferred connect_to
  pass (`:1070`) appends its rows; `_op_archive` (`:685`), `_op_absorb` (`:700`),
  `_op_disconnect` (`:729`) contribute rows. **`_op_archive`/`_op_absorb` splice `**r` into
  `results[]` (`:967`, `:973`)** — pop new keys in this commit.
- `brain.absorb` returns survivor-revise `deltas` (discarded at `brain_remember.py:625-630`)
  and migrated-edge rows (discarded at `:579`).
- Delete `_emit_edge_traces` (`:319`) and `_infer_scale_and_chain` (`:160`) — last callers
  gone. (`_emit_edge_revise_trace` already died at step 5 with its four callers.)
- **The orphan test is the point of this step**: a rolled-back `brain_batch` (`:1037`)
  produces **zero** mutation traces; an `absorb` savepoint unwind
  (`brain_remember.py:668`) produces zero traces for the unwound merge. Both are impossible
  to satisfy today for 10 of the 12 sites.
- Pins: `tests/test_absorb.py`, `tests/test_brain_batch_transaction.py`,
  `tests/test_brain_batch_op_contract.py`, `tests/test_absorbed_into_edge.py`.

**Note:** the `brain_logs.db` backup was already taken before step 4 (the first emitting
step) — this step needs no new one.

---

## Step 8 — Archive and hard-delete coverage — ✅ DONE

> **Deviations from this plan, as built:**
> - **`delete_node_edges` returns the flipped `[edge_id, relation]` pairs directly**
>   (list, not count) — count derives as `len()`; `archive_node` keeps the scalar
>   `edges_deleted` agent-visible alongside, and the collections (`edge_relations`,
>   `absorbed_into_edge`) are popped into the manifest by the dispatch ops. Landed
>   here rather than step 9 (the plan's "via step 9's change" cross-reference).
> - **`_archived_row` sets `encoding_source = archived_by`** — actor attribution keeps
>   all of one absorb's rows on one scale (same review finding as step 7's edge rows).
> - **Junk purge emits PER NODE, immediately after each cascade's commit** — review
>   finding: each cascade is individually durable, so a single post-loop emit would let
>   a mid-loop failure erase earlier nodes with zero record (and a hard delete's trace
>   is its only surviving record). Hooks bypass the dispatch chokepoint, so the purge
>   calls `emit_mutation_traces` directly.
> - **`health_check(auto_fix=True)` gained the same direct emit** (`brain_assembly.py`)
>   — NOT in any plan step: it archives stale context nodes at every session boot,
>   never crosses dispatch, and isn't in step 10's routing list, so deleting the inline
>   trace would have made `hook:integrity` archives permanently invisible. Rows land on
>   `maint-{YYYYMMDD}-integrity` at s0 — deliberately NOT the junk purge's
>   `maint-{YYYYMMDD}-mutation` (s2): one chain_id must never span two scales
>   (get_chains stamps chain scale from the first event). Step 10's scope is unchanged
>   (community/consolidation routing still pending — those keep coarse `heal_archive`
>   O-traces in the interim).
> - `tables_hit` gates `nodes_fts` on the virtual table actually existing (test DBs
>   lack it; the trace must not claim a statement that never ran).

- `servers/brain_remember.py` — `archive_node` returns its archived
  `(edge_id, relation)` pairs (via step 9's `delete_node_edges` change), the
  `absorbed_into` edge row (`:398-407`), and `vectors_deleted`; its inline trace emit
  (`:431-447`) is **deleted**, replaced by `node_archived` on the caller's real chain.
  Keep `title`/`type` on the row (decision record §4.1 — for a hard delete the trace is the
  only surviving record).
- `servers/dal_graph.py:877` `delete_node_edges` — return only rows the UPDATE **actually
  flipped**, as `(edge_id, relation)` pairs. Its SELECT (`:902-905`) returns all edges
  touching the node while the UPDATE (`:922-928`) excludes `archived=0` misses **and
  `exempt_relations`**; returning the SELECT list would claim the deliberately-exempted
  `absorbed_into` redirect edge was archived.
  **This is an agent-visible field change**: it feeds `brain_remember.py:385` → archive trace
  metadata (`:442`) and `archive_node`'s returned `edges_deleted` (`:456`), which
  `_op_archive` splices into `results[]`. Keep the old scalar count alongside the new
  collection; pop the collection into the manifest.
  Pins: `tests/test_absorbed_into_edge.py:166-190`, `tests/test_bg_writer.py:359`,
  `tests/test_surface_transitions.py:492`.
- `servers/brain_remember.py` — `delete_node_cascade` returns `{node_id, tables_hit}`.
- `servers/daemon_hooks.py:802-828` — the junk purge manifests one `node_deleted` row **per
  node** with `{node_id, type, title, deleted_by, encoding_source, reason}` (ruled by Tom
  2026-08-04, decision record §7 — not a count-only rollup). The `SELECT`s at `:804-818`
  already fetch `id, title`; `type` is `'vocabulary'` by construction. Chain: explicit
  `maint-{YYYYMMDD}-mutation` at s2.
- Measured 2026-08-04: 10 active `vocabulary` nodes brain-wide, **0** matching either purge
  strategy — the path is dormant, so a test must construct its own junk node rather than
  wait for the sweep.

---

## Step 9 — Bulk sweep rollups + the shared flip primitive — ✅ DONE

> **Deviations from this plan, as built (2026-08-06):**
> - **Preconditions honored**: `brain.db` backed up (`bak-20260807T022350Z` + WAL/SHM);
>   the ~0 rows/pass assumption re-measured live and CONFIRMED — zero relations below
>   the 0.1 prune threshold, zero `exemplifies` in the crossable band, and all 467
>   dangling candidates are exempt `absorbed_into` redirects (the Healer sweep is at
>   its fixed point). Pure correctness fix, as scoped.
> - **The primitive is `bulk_archive_relations`** (ruled name, node 482ef98e) and
>   `delete_node_edges` routes through it too — four callers, not the plan's three;
>   step 8 had built the SELECT-then-UPDATE inline, this step deduplicates it.
> - **decay-prune rows land at s0 on `maint-{date}-decay`**, not "at s2" as the plan
>   text said: the row's encoding_source mirrors the graph's `archived_by`
>   ('decay_pruned', unprefixed → s0, policy preserved exactly), and the s2 maint
>   chain string must not be reused across scales (one chain_id = one scale). Healer
>   sweep rows do land at s2 on `maint-{date}-mutation` (shared with the junk purge —
>   same scale, coherent).
> - **The Healer's sweep now runs under `brain.write_lock`** — it's a foreground
>   brain.conn write that now commits, and the emit contract requires the lock.
> - **Emit isolation (review)**: both sweep emits sit in their own inner try — the
>   sweep is committed by emit time, so a row-shaping failure degrades to missing
>   traces (logged as `healer_sweep_trace_emit` / `decay_prune_trace_emit`), never
>   clobbers the sweep's reported count or misattributes the error.
>   `edge_flip_rows` chunks its endpoint query at 500 (SQLite host-param cap).
> - Healer `actions` arithmetic unchanged — `edges_archived` is now sourced from the
>   dict return, so the plan's predicted `int + dict` TypeError never shipped.

- `servers/dal_graph.py` — one shared flip primitive with **explicit per-caller policy
  flags** (`null_embeddings`, `recompute_weight`, `exempt_relations`) and one return shape.
  Callers: `remove_relation` (`:1253`), `archive_dangling_edges` (`:438`), `decay_edges`'
  prune arm (`:955`). **Their policies differ and must be preserved exactly** — see decision
  record §5. Specifically: `archive_dangling_edges` keeps `null_embeddings=False`,
  `recompute_weight=False`, and its **load-bearing** `exempt_relations` (455 live
  `edge_relations` depend on it). The weight-**decay** UPDATE stays outside the primitive.
- `archive_dangling_edges` also **gains a commit** — it has none today, so any trace after
  it is orphanable by construction. Its selection is a correlated subquery *inside* the
  UPDATE, so returning edge ids requires restructuring to SELECT-then-UPDATE-by-ids.
- Its return type changes `int → dict`. **This crashes the Healer**: `s2/healer.py:81` does
  `encode_result.get('fields_written', 0) + edges_archived` → `int + dict` `TypeError` every
  cycle. Update that call site and `tests/test_edge_relations.py:435` in the same commit.
- **Per-edge rows through the same trace map — no rollup shape** (ruled 2026-08-03). The
  prune arm already collects `pruned_edge_ids` (`dal_graph.py:993-997`) and discards them;
  return them as manifest `edges[]` rows. `encoding_source='decay_pruned'` / `'s2:healer'`,
  on an explicit `maint-{YYYYMMDD}-mutation` chain at s2.
- **Sweep emission is `exemplifies` only.** Three relations decay
  (`brain_constants.py:EDGE_TYPES`, threshold `EDGE_PRUNE_THRESHOLD = 0.1` at `:282`) and the
  aspect coverage rule keeps exactly one: `co_accessed` (noise, 8,226 active) ✗,
  `emergent_bridge` (noise, 109) ✗, `exemplifies` (`validation_evidence`, 13) ✓.
- Measured live 2026-08-04: **0** relations below threshold, and 0 `exemplifies` in the
  crossable band → sweep emission is **~0 rows/pass**. This step is therefore a pure
  correctness fix (the missing commit, observed-truth returns, policy flags), **not** a
  coverage feature — scope it accordingly. Re-measure if it lands much later than the plan.
- **`cp brain.db brain.db.bak-{ts}` before this step** — it runs a bulk `UPDATE` on
  `brain.db` (CLAUDE.md, "Backup before destructive DB operations").

---

## Step 10 — Route the bypassers (moved from "step 0") — ✅ DONE

> **Deviations from this plan, as built (2026-08-07):**
> - **The archive op gained optional `survivor_id`** — NOT in the plan: the
>   superseded-handoff heal passes survivor lineage, and routing it through a
>   batch archive op without the field would have silently dropped the
>   `absorbed_into` redirect. Contract change ran both gates before merge:
>   `mcp_batch_probe` 40/40 across 8 dimensions, `mcp_schema_gate` PASSED.
>   `_op_archive` validates the survivor AT THE OP BOUNDARY (resolve short id,
>   exists + unarchived + not-self) because archive_node stores the pointer
>   before any existence check — a garbage agent-supplied survivor would have
>   returned ok=True with a dead pointer (review 2026-08-07). The redirect
>   edge row joins the manifest, same shape as absorb's.
> - **health_check NOT rerouted** (scope carve-out from step 8 — it has a
>   direct emit that must stay; brain node 38495efb).
> - **Healer**: `_make_dispatch` built on the ENCODER instance (one chain per
>   pass); `_store_fields` checks the dispatch result (rejected revise → 0,
>   was reporting N); the direct `brain.revise` fallback is deleted.
> - **Locked/critical pre-filtered in both heal sweeps** (review): a guarded
>   target would otherwise log `archive_guarded` + `batch_op_failed` every
>   cycle, forever.
> - **reconcile runs BEFORE the community delta write** so its backfill
>   (count + edge ids, `membership_reconciled`) rides THIS run's delta.
> - The step-3 dashboard-mirror precondition was already satisfied
>   (`s2_runs.py` filters `_NON_RUN_REF_TYPES`).

**Why this is now late, not first.** The original plan opened with Healer routing. Routing
buys **only the trace** — project provenance **cannot** be stamped on a revise by design
(`dispatch_write.py:554` "a revise never moves it"; `revise` routes to `_strip`,
`scales/dispatch.py:161-165`), so the original step-0 test ("field-fills produce
`node_revised` with project stamping") is **impossible**. With no emitter yet, this step
would change production write paths for zero observability gain. After step 7 it is
verifiable by its own test.

- `servers/scales/s2/healer_encoder.py` — build the dispatch closure **inside
  `HealerEncoder._make_dispatch()`**, not by handing the orchestrator's closure down
  (`s2/healer.py`). Every other encoder builds its own, so the closure's `run_chain_id`
  comes from the same instance that later calls `self.trace(...)`. `chain_id()` caches per
  instance at **seconds resolution**, and `Healer`/`HealerEncoder` are separate instances →
  handing it down yields `node_revised` on `s2-T1-healer` and `healer_generated` on
  `s2-T2-healer`: two chains for one pass, the phantom-run-card class `CHAIN_AWARE_WRITES`
  exists to prevent.
- Same file — **check the dispatch result**: `:340-348` returns `len(fields_to_write)`
  unconditionally, and `_handle_revise` returns `ok=False` rather than raising, so a
  rejected revise currently reports `fields_written=N`.
- `servers/scales/s2/community.py:253-256`, `consolidation_decoder.py:165-168`, `:234-237`
  — route direct `archive_node` calls through dispatch. There is **no standalone `archive`
  command** (`_op_archive` is batch-only, `dispatch_write.py:685`), so this means a
  `brain_batch` archive op — and it must use an **unguarded** `_make_encoder_dispatch()`.
  The consolidation encoder's closure carries `archive_guard=valid_archive_ids`
  (`consolidation_encoder.py:229-233`) which **drops** archives outside the cluster set and
  logs `s2_consolidation_out_of_scope_archive` (`s2/base.py:270`, re-resolved 2026-08-04).
  All three call sites
  target out-of-cluster nodes (orphan communities, superseded handoffs, dead communities) —
  reusing the guarded closure would stop the orphan heal working **while reporting success**.
- **Required test:** the orphan-community heal still archives; and `node_revised` rows join
  the **same** `chain_id` as the pass's `healer_generated` delta.
- **`reconcile_community_membership` → the community unit's own delta.** `dal_graph.py:485`,
  called from `community_encoder.py:233`, creates `community_member` edges and discards the
  results. It is direct-DAL and never touches dispatch, so the emitter cannot see it. Record
  it where the S2 story lives: count + edge ids in the community unit's existing delta
  (Tom 2026-08-04 — *"community members is interesting as part of S2community"*). The
  dispatch-routed `community_member` edges are covered separately by the emitter's one
  `noise` exception (decision record §5).
- Note: `dashboard/queries/s2_runs.py:query_healer_runs` filters no ref_type, so each
  field-fill would add a phantom card — the step-3 mirror fix must be in before this lands.

---

## Step 11 — Dead code (each verified at delete time)

| Candidate | Verdict |
|---|---|
| `servers/scales/s2/archive/reclassify.py` | **Hold.** Retired aspect-migration path, dormant, not in the coordinator's unit list — but `docs/BACKLOG.md:287` (+ the open list at `:17`) asks to verify it is wired into the S2 coordinator and run it once against the corpus, a **different** purpose. Close that backlog item explicitly or keep the file. Path drift: BACKLOG and `docs/S2-DESIGN.md:437` say `servers/scales/s2/reclassify.py`; the file is under `archive/`. |
| ~~`Brain.check_integrity` auto-fix arm~~ | **STRICKEN — the original entry would have deleted live boot behavior.** There is no `Brain.check_integrity` (verified: zero occurrences in the tree). The code is inside `Brain.health_check(session_id, auto_fix=True)` (`brain_assembly.py:397`, auto-fix arm `:441-450`), which runs at **every session boot** (`brain_voice.py:351`) and via the `health_check` dispatch command (`dispatch_ops.py:29`). It archives `context` nodes older than 14 days with `archived_by='hook:integrity'`. If this behavior should change, that is a separate decision with its own evidence. |
| `Brain.set_personal` (`brain_remember.py:2278-2312`) | **Delete** — zero callers, no dispatch command. Rationale corrected: the personal flag is **live** and scored in recall (`brain_recall.py:1869-1877`), and `revise` can already set `personal`/`personal_context`. The single capability lost is *auto-lock on `personal='fixed'`* — already bypassed today by `revise(personal='fixed')`. Precedent: `NodeDAL.unlock` deleted as dead (`node:0712ca78`). |

---

## Step 12 — First consumer (parked; do not start with steps 1-11)

Design `node:5efe5e02`, finding `node:30cf1bce`. **The original one-line claim is false** —
`GATHER_STREAMS` changes shape:

- `gather()` filters by `session_id` (`scales/s1/trace_links.py:369-374`) and unpacks a
  strict 2-tuple (`:371`), while encoder-originated writes carry `session_id=''` by
  construction (`apply_encoder_attribution` setdefaults only `encoding_source`/`chain_id`).
- `session_node_ids` parses ids via `_delta_ids(meta, 'created', 'revised')` (`:331-333`) —
  a per-node `node_created` row has no such metadata list key; ids come from `ref_id`.
- So the catalog-gap fix — the sole justification for `node_created` — would receive **zero
  rows**. Fix: either stamp `session_id` on S1-scope encoder writes at the chokepoint (a
  real addition, not free) or key the created stream by the `s1e-` chain. Pick one, and pick
  the scale: `node_created` is emitted at s0, s1 **and** s2 while the registry maps one
  `(ref_type, scale)` pair.
- **Parked invariants hold** (`node:c7d52ad0`): `NodeDAL.get_bulk` hydration stays the truth
  filter — a rolled-back id resolves to nothing and drops silently; coverage stays gated on
  the `encoding_run` success receipt. "A failure can neither hide turns nor conjure nodes."

---

## Gates, rollout, rollback

**Per-step tier:** targeted test files named in each step. **At merge:** `./dev pytest tests/`
(full suite — SQL/DAL/dispatch blast radius).

**Not required:** `eval/mcp_batch_probe.py`, `eval/mcp_schema_gate.py` — nothing touches
`BATCH_OP_SPECS` or any tool schema text. Re-check if that changes.

**Deploy.** `servers/*` only → **daemon restart** (`restart` MCP tool / `restart-daemon.sh`);
no `./redeploy.sh`. Step 3 also touches `dashboard/` → **restart the dashboard**
(`com.brain.dashboard` is a separate launchd service). Do **not** gate the deploy-restart
with the maintenance lock — it makes the daemon skip startup. Work lands on the branch, then
merges to `main` (what the daemon runs).

**Backups.** `cp brain_logs.db brain_logs.db.bak-{ts}` before **step 4** — the first step
that writes a trace row (steps 1-3 are production-invisible). `cp brain.db
brain.db.bak-{ts}` before **step 9** (bulk `UPDATE` on the graph DB).

**Production smoke check** (after step 7, and after each later emitting step):

1. `remember` a throwaway node via MCP.
2. `query_traces(ref_type='node_created')` → the row exists, with the real session and chain.
3. `query_logs` → no `mutation_trace_emit` / `mutation_trace_txn_open` error rows.
4. `brain_batch` an archive of the throwaway → `node_archived` on the caller's chain.
5. Confirm the S2 idle gate did not re-arm: an S2 unit's `_last_run_timestamp` is unchanged
   by the mutation rows.

**Rollback.** A revert + restart recovers behavior, but **wrong trace rows survive a revert**
and nothing in the repo prunes `trace_events`. Cleanup shape:

```sql
DELETE FROM trace_events
 WHERE ref_type IN ('node_created','node_archived','node_deleted')
   AND created_at > '{flip_ts}';
```

For `node_revised` / `edge_relation_revised` a revert restores the old emit path, so those
rows stay valid — only the three new types need pruning.
