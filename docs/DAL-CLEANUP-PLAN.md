# DAL Cleanup & Migration Plan

**Started:** 2026-05-30 · **Owner:** Anchor + Tom · **Status:** Phases 0–2 + 3a + F3-completion **merged to `main`** (merge `530d2f8`). **Structural F3 fix (#5) ✅ done** on worktree `dal-cleanup-2` (connection-bound batch flag — see below). Phase 3 in progress — 3a counts ✅, 3b TF-IDF ✅ (`_store_tfidf_vector`/`_rebuild_tfidf_index`→`TfIdfDAL`, dead `_tfidf_score` deleted; batch scorer read deferred to Phase 4); 3c node-writes/titles pending. Open: Phases 3c/4/5/6.

> **✅ Structural F3 fix (2026-05-30, branch `dal-cleanup-2`):** killed the
> by-convention `commit=not _batch_mode` fragility. Batch state now lives on the
> **connection** (`BatchAwareConnection.in_batch`, `db_backends/sqlite.py`); a
> single gate `commit_unless_batched(conn)` replaces every `self.conn.commit()`
> in `dal.py` (33 sites) + `Brain._maybe_commit()`. The `commit` kwarg is removed
> from the 6 GraphDAL writers and `brain._batch_mode` is deleted — one source of
> truth, nothing to forget. Owners (`_handle_brain_batch`, `recall_write_queue
> ._drain_once`) flip `conn.in_batch` in try/finally. Locked by
> `tests/test_write_txn_discipline.py` (behavior + source + signature + wiring
> contracts; verified they flag the pre-fix state). Full doc:
> `docs/WRITE-TXN-ISOLATION-ROOTFIX.md` → "Root fix — shipped (Option A)".

> **F3 code-review remediation (2026-05-30, `dal-cleanup-2`):** a high-effort
> segmented `/code-review` of the F3 commit surfaced a **pre-existing**
> concurrency gap (not an F3 regression): S2 maintenance calls `brain.set_config`
> **lock-free** on a pool thread (gating timestamps, failure counters, journals),
> concurrent with a client `brain_batch` holding `write_lock`+`in_batch=True` on
> the shared `brain.conn`. With `commit_unless_batched` now on `BrainMetaDAL.set`,
> the config INSERT folds into the batch txn and is lost on rollback. **Fixed:**
> `brain.set_config` now acquires `write_lock` (RLock, reentrant-safe) — it
> serializes against `brain_batch` (which resets `in_batch` before releasing the
> lock), so a non-owner thread never observes `in_batch`. Also: `log_communication`'s
> bare `self.conn.commit()` routed through `_maybe_commit()`; the guardrail widened
> to scan `brain.py`+`brain_*.py` (legit explicit-durability commits tagged
> `# commit-ok:`). **⚠ FLAGGED, not fixed (pre-existing, untouched by F3):**
> `Brain.save()` is also called **lock-free** in `_run_idle_maintenance`
> (daemon_server.py:831) and does an unconditional `self.conn.commit()` — same
> race class (could commit a concurrent client batch's partial state). Spawned as
> a separate follow-up task. Per-write `write_lock` (matching `set_config` and the
> encoder dispatch at `base.py:319`) is the fix, NOT wrapping all of maintenance
> (that would hold the lock across S2 LLM calls).

> **Code-review fixes (`0cd1c1d`, 2026-05-30):** an xhigh review found Phase-1's
> F3 fix was **incomplete** — 3 reachable GraphDAL writer calls still self-committed
> inside a batch (co_anchored auto-edges in remember+revise, the untyped `connect()`
> helper, and the bg-writer hebbian 'new edge' branch). All guarded now (+5 brain_batch
> regression/coverage tests). Also swept review cleanup: a missed consolidation_decoder
> straggler, the dead `from .dal import GraphDAL` imports Phase 2 left behind, and the
> stale count_locked/daemon status-count docs. **Lesson: the F3 by-convention guard is
> fragile — a structural fix (review finding #5) is still open as a follow-up.**

> **✅ Merge state (session end 2026-05-30):** `dal-cleanup` is **merged to `main`**
> (merge commit `530d2f8`) — Phases 0–2, 3a counts, and the F3 completion fix all
> landed; the co_anchored/connect()/hebbian atomicity gap is **closed on `main`**.
> Full suite green on `main` post-merge (1365/7/4). Clean auto-merge with the
> concurrent dashboard/self-channel work (only `brain_assembly.py` overlapped,
> additive both sides). Worktree `/Users/tpac/brain-dal-cleanup` retained on branch
> `dal-cleanup` — **`git merge --ff-only main` in it before resuming Phase 3b.**
>
> **Open for next session:** Phase 3b (TF-IDF→TfIdfDAL), 3c (node-writes/titles→NodeDAL),
> then Phases 4–6. (Structural F3 #5 is done — see the box above.) NOTE: the
> structural fix makes 3b/3c safer — once `remember` routes through
> `TfIdfDAL.store_tf_vector` / `NodeDAL.update_field`, those writers already gate
> on `conn.in_batch` (every dal.py commit was converted), so they're batch-atomic
> for free; no per-caller `commit=` plumbing needed.

Living tracker for resuming the stalled DAL migration. Update the **Status** lines
and the progress table as phases land. This doc is the single source of truth for
scope and progress — keep it current.

## Where this work lives (isolation)

Phases 1–6 run in a **dedicated git worktree** to avoid the shared-index
collisions that hit Phase 0 (a parallel session's `git add -A` swept the
staged work; recovered via reflog + patch). Setup:

- Worktree: `/Users/tpac/brain-dal-cleanup` · branch: `dal-cleanup` (based on `0bfba45`)
- `venv` is a symlink to the main repo's bundled venv (gitignored; never commit it)
- Commit per phase with **explicit paths** (`git commit -- <files>`), never bare `git commit`
- Phase 0 landed here as `9ca62fb` (the main-tree copy was reverted to keep main clean)
- **Merge-back:** `dal-cleanup` → `main` once Tom approves (end of run or per-phase),
  coordinating the moment since parallel streams commit to `main` continuously

---

## Why this exists

The DAL was designed for **incremental, table-at-a-time adoption** (locked decision
`7d14f588`; `dal.py` docstring: *"Direct self.conn.execute() calls continue to work
alongside the DAL"*). The original plan (`docs/archive/stale/DAL-MIGRATION-PLAN.md`)
defined **Step 0** = hold every DAL as an instance on `Brain`, then migrate callers.
**Step 0 was never finished and the migration stalled.** That produces two symptoms
that are *the same gap seen from opposite sides*:

| Symptom | Count | Root |
|---|---|---|
| Raw-SQL calls outside the DAL | ~301 total (~43 genuine violations after subtracting DDL/PRAGMA/maintenance) | callers never migrated |
| DAL methods with **zero callers** | ~40 | the methods callers *would* call were written but never adopted |

The clearest case: **`TfIdfDAL` is fully written, and `brain_remember.py` reimplements
it statement-for-statement inline.** The DAL method is "dead" *because* the violation
exists. Resume the migration → both disappear together.

**Verified facts (2026-05-30 audit):**
- All sampled dead methods have **0 callers repo-wide** (incl. `dashboard/`).
- `SessionStateDAL` is **never instantiated** anywhere; `session_state` table is live but accessed via raw SQL in `brain.py:680,738,768`.
- `_random_walk` (brain_connections.py:199) and `GraphDAL.get_random_walk_neighbors` are **both dead** — random-walk path retired.
- `GraphDAL.create_edge` — 0 callers, already `# DEPRECATED`.
- **68 ad-hoc DAL construction sites** in `servers/` for the 5 not-held DALs.
- F3 transaction bug confirmed: `connect_typed` (brain_connections.py:195) never passes `commit=False` → `add_relation` self-commits at dal.py:2777 inside batches.

---

## Principles (carry through every phase)

- **Incremental & stoppable.** Each phase ships independently, tests green, committed separately. We can stop after any phase and the tree is consistent.
- **DAL-first.** No new raw SQL outside `dal*.py` / `schema.py`. Phase 6 adds a guardrail test.
- **Loud by default.** No new silent `except`. De-silence as we touch.
- **Don't break the recall hot path.** Any change to `brain_recall.py` / surface / pipeline → run `eval/decode_funnel.py` before/after.
- **Backup before destructive DB ops.** This effort is mostly *code*, but any phase that touches data (none planned) → `cp brain.db brain.db.bak-{ts}` first.
- **Test integrity.** If a test fails, STOP — report expected vs actual, don't weaken it.

## Out of scope / avoid (parallel streams active)

Two other sessions are working the same repo. Per `git status` they touch
`signal.py`, `daemon_hooks.py`, `post_response_track.py`, `test_self_signal.py`,
`test_self_delivery.py`. To avoid collision:

- **No `SelfChannelDAL`** — `self_inflight` / `self_delivered` stay raw this effort (`signal.py` is theirs).
- **Touch `daemon_hooks.py` minimally** — only the ad-hoc DAL construction swap in Phase 2, coordinate first.

---

## Scope ledger

### Bugs (independent of the migration; fix where scheduled)

| ID | Bug | Location | Phase |
|---|---|---|---|
| B-F3 | GraphDAL writers self-commit → breaks `brain_batch` atomicity | dal.py `add_relation`/`remove_relation`/`delete_node_edges`/`decay_edges`/`add_source_refs`/`replace_source_refs`; brain_connections.py:195 | 1 (correctness) + **1b structural ✅** — `commit` gate moved to `conn.in_batch`; no kwarg to forget |
| B-SIG | `CachedVectorDAL.find_missing` drops `require_kv_keys_any` → latent `TypeError` | dal_vector_cached.py:179 vs dal.py:3197 | 0 |
| B-FTS | Two silent `except Exception` on a recall signal | Fts5DAL.search (dal.py:1710), delete (1730) | 0 |
| B-LCK | `count_locked` omits `archived=0`; 3 callers want it (semantics fork) | dal.py:1375 vs brain.py:1083, daemon_server.py:671, brain_assembly.py:346 | 0 |
| B-KEY | Neighbor-row key inconsistent (`target_id`/`node_id`/`id`) | dal.py get_random_walk_neighbors / get_well_connected / get_community_members | 0 (mostly resolved by deletions) / 6 |
| B-NAME | `MetaDAL` vs `MetadataDAL` naming trap | dal.py:1228 vs dal_metadata.py:68 | 0 |

### Dead methods — categorized

**Category A — delete now (Phase 0), no adoption value:**
- `GraphDAL.get_edge_count` (dup of `count_total` — keep `count_total`)
- `GraphDAL.get_well_connected`
- `GraphDAL.get_random_walk_neighbors` + `brain_connections._random_walk`
- `GraphDAL.create_edge` (already DEPRECATED)
- `NodeDAL.delete_for_node` (wrong owner — dup of `VectorDAL.delete_for_node` on `node_enrichments`)

**Category B — migration targets (adopt in Phases 3–4; raw SQL is the violation, DAL method is the fix). Do NOT delete:**
- All `TfIdfDAL`: `store_tf_vector`, `clear_all`, `get_doc_freq`, `get_node_terms`, `get_total_docs`, `delete_for_node`, `rebuild`
- `NodeDAL` writes: `update_field`, `update_confidence`, `set_critical`, `update_type`, `append_content`, `set_evolution_status`, `mark_accessed`, `delete`, `unlock`
- `NodeDAL` counts: `count`, `count_locked`, `count_by_type`; `GraphDAL.count_total`
- `MetadataDAL.get_all_by_key`
- `SessionStateDAL` (whole class) — resurrect + adopt for `brain.py` raw `session_state` SQL, OR consciously delete (decide in Phase 4)

**Category C — test-only / revisit Phase 6:**
- `MetaDAL.get_json`/`set_json`/`increment`, `MetadataDAL.delete_all`/`get_nodes_with_flag`/`clear_flag`/`field_coverage`, `GraphDAL.delete_node_edges`, `InteractionDAL.list_versions`, `Fts5DAL.rebuild`, `NodeDAL.get_all_for_reindex`/`get_all_with_titles` (latter may have a maintenance/CLI use — verify before deleting).

### Raw-SQL violations by subsystem (the migration work)

| Subsystem | Files | Target DAL | Phase |
|---|---|---|---|
| TF-IDF index | brain_remember.py:352-366, 400-480, 523-545 | TfIdfDAL (exists) | 3 |
| Node writes | brain_remember.py (UPDATE nodes …), :1359 content_summary | NodeDAL.update_field/etc. | 3 |
| Counts | brain.py:1071/1076/1081, daemon_server.py:670, brain_recall.py:1291, brain_assembly.py:346 | NodeDAL/GraphDAL counts | 3 |
| Node titles | brain_remember.py:312/1312, brain_connections.py:241 | NodeDAL.get_title | 3 |
| Metadata-KV reads | community_decoder.py:70/220/350, temporal_extraction.py:485 | MetadataDAL.get/get_field/get_all_by_key | 4 |
| N+1 loops | community_decoder.py:847 (embedding), :959 (created_at) | VectorDAL bulk / NodeDAL.get_bulk | 4 |
| trace_events (via DAL's own conn!) | conversation.py:161/173/210 | new TraceDAL methods | 4 |
| session_state | brain.py:680/738/768 | SessionStateDAL (resurrect) | 4 |
| edge relation rename | reclassify.py:92 | new `GraphDAL.rename_relation` | 4 |
| entity_dates (no DAL) | temporal_extraction.py:346/355/369, fetch_tools.py:577/592/485 | new EntityDatesDAL | 5 |

### Structure

- **Extract `SourceRefDAL`** from GraphDAL (`add_source_refs`/`replace_source_refs`/`get_source_refs`/`get_nodes_referencing`, dal.py:2924-3021 — operate on `node_source_refs`, not edges). Phase 5.
- **Cascade-delete path** — one `Brain.delete_node_cascade(node_id)` so `purge`/archive stop hand-rolling per-table SQL and the per-table `delete_for_node` methods get a real caller. Phase 5.
- Leave the GraphDAL **edge core** intact — its size is earned by the v22 two-table model.

---

## Phases

Legend: ☐ not started · ◐ in progress · ☑ done

### Phase 0 — Safe cleanup + cheap fixes  ☑ (2026-05-30)
**Landed:** Deleted Category-A dead code (`GraphDAL.get_edge_count`/`get_well_connected`/`get_random_walk_neighbors`/`create_edge`, `NodeDAL.delete_for_node`, `brain_connections._random_walk` + orphaned `import random`). Fixed B-FTS (de-silenced `Fts5DAL.search`/`delete`), B-SIG (`CachedVectorDAL.find_missing` now mirrors `require_kv_keys_any`), B-LCK (`count_locked(include_archived=False)` capability added; callers migrate in Phase 3), B-NAME (`MetaDAL`→**`BrainMetaDAL`**, chosen over `ConfigDAL` for `brain_meta` table-name alignment + `self._meta` coherence). Full suite: 1349 pass / 7 skip / **4 pre-existing fails** (recall fatigue+hub-dampening, confirmed red on baseline without these changes — NOT introduced here; flagged separately).
**Goal:** Remove confirmed-dead Category-A code and land the low-risk bug fixes. No behavior change.
**Work:**
1. Delete Category-A dead methods (get_edge_count, get_well_connected, get_random_walk_neighbors, _random_walk, create_edge, NodeDAL.delete_for_node) — verify 0 callers immediately before each delete.
2. B-FTS: de-silence `Fts5DAL.search`/`delete` (log to stderr, match the file's existing pattern at add_relation:2789).
3. B-SIG: mirror `require_kv_keys_any` into `CachedVectorDAL.find_missing`.
4. B-LCK: add `archived: bool = True` param to `NodeDAL.count_locked`, reconcile with the 3 callers' intent (they want `archived=0`).
5. B-NAME: rename `MetaDAL` → `ConfigDAL` (5 methods, 1 attr `_meta`); update all references.
**Verify:** `./dev pytest tests/` green; `./dev pytest tests/test_contract_sync.py tests/test_dispatch_contract_sync.py`.
**Stop point:** dead Category-A gone, 4 bug fixes in. Commit.

### Phase 1 — F3 transaction composition fix (correctness)  ☑ (2026-05-30)
**Landed (`fd9c313`):** `commit: bool = True` added to `remove_relation`, `add_source_refs`, `replace_source_refs`, `delete_node_edges`, `decay_edges` (mirroring `add_relation`); each `self.conn.commit()` guarded. Batch-context callers pass `commit=not _batch_mode`: `connect_typed` (the prime live leak), `dispatch_write` disconnect op, `brain_remember` source-ref writes. Added 4 regression tests (connect/disconnect/source-ref each COMMIT once inside a batch; connect rolls back fully on a later op's failure) — the existing commit-counter used only plain `remember` ops and never caught these. Full suite: 1353 pass / 7 skip / 4 deselected.
**Goal:** `brain_batch`'s all-or-nothing rollback becomes true. (Ref: `docs/WRITE-TXN-ISOLATION-ROOTFIX.md` Option A.)
**Work:**
1. Add `commit: bool = True` to `remove_relation`, `delete_node_edges`, `decay_edges`, `add_source_refs`, `replace_source_refs` (mirror `add_relation`).
2. Make `connect_typed` (brain_connections.py:195) pass `commit=False` when `self._batch_mode`.
3. Tests: per method, assert `conn.in_transaction is False` after a standalone call; assert a `brain_batch` wrapping each commits exactly once and rolls back fully on a mid-batch failure.
**Verify:** new txn tests green; existing batch tests green; reproduce-then-fix the "cannot start a transaction within a transaction" case (BACKLOG F3).
**Stop point:** correctness bug closed, interim guard can stay as belt-and-suspenders. Commit.

### Phase 2 — Repository aggregate (finish Step 0)  ☑ (2026-05-30)
**Landed (`d13d671` + cleanup `c1f9ceb`):** Held `self._nodes/_graph/_meta_kv/_fts/_tfidf` on Brain (foreground conn). Converted ~57 of the 68 explicit/alias sites (`self.conn`/`brain.conn`/`self.brain.conn` + the `brain_corrections`/`pipeline_contract` `conn=` aliases) to the held instances; removed all orphaned imports + 2 unused `conn` locals. **Residual (intentional):** `recall_write_queue:400` (`GraphDAL(conn_bg_writer)` — the documented bg exception); `daemon_hooks.py` ×3 (parallel-stream file — deferred to avoid merge conflict); `brain_recall`'s `_apply_filter(conn)` + `get_rich_node` bare-conn params and `surface_contract`'s `brain_conn` param (genuine function params, not held-instance candidates). Pure refactor — full suite 1353 pass / 7 skip / 4 deselected; import-smoke + DAL/pipeline subset green.
**Goal:** Hold all DALs on `Brain`, foreground-conn-bound. Replace the 68 ad-hoc construction sites. "Right connection by construction, not convention."
**Work:**
1. Add `self.nodes`, `self.graph`, `self.meta_kv` (MetadataDAL), `self.fts`, `self.tfidf` in `brain.py` `__init__` (foreground `self.conn`).
2. Replace ad-hoc `NodeDAL(self.conn)` / `GraphDAL(...)` / etc. with the held instances across the 68 sites — mechanical, one file at a time, tests after each.
3. Keep the **one** intentional `GraphDAL(conn_bg_writer)` in `recall_write_queue.py:400` as the documented exception (add a comment).
4. Coordinate `daemon_hooks.py` edits with the other stream.
**Verify:** `./dev pytest tests/`; `eval/decode_funnel.py` (recall hot path touched).
**Stop point:** zero ad-hoc construction except the documented bg_writer exception. Commit.

### Phase 3 — Migrate writes (stop writing raw SQL)  ◐ (3a, 3b done)
**3a — counts ✅ (`cfc8f02`):** `brain._get_{node,edge,locked}_count`, `brain_recall` brain-size, `brain_assembly` total_locked, `daemon_server` status counts → held DALs (`count`/`count_locked`/`count_total`/`count_by_type` adopted). daemon locked count gains the documented non-archived semantics.
**3b — TF-IDF ✅ (`dal-cleanup-2`):** `brain_remember._store_tfidf_vector` → `self._tfidf.store_tf_vector(node_id, tf)` (the verbatim-reimplemented dead class, now adopted); `_rebuild_tfidf_index` → `clear_all()` + per-node `store_tf_vector` wrapped in `conn.in_batch` (single commit preserved via save/restore + `_maybe_commit`); `brain.py` boot reindex check → `self._tfidf.get_total_docs()==0` + `self._nodes.count(archived=True)`. `_compute_tf`/`_tfidf_tokenize` stay (TF math, not DB). **Dead `_tfidf_score` (single-node, 0 callers) deleted.**
  - *Deferred to Phase 4 (reads):* `_batch_tfidf_scores` is a READ on the **recall hot path** (`brain_recall.py:884`) with subtle semantics (`total_docs = _get_node_count()` ≠ `get_total_docs()`; term-filtered node_vectors read with no exact DAL method). Migrating it needs a new `TfIdfDAL` read method + a `decode_funnel` before/after — belongs in the reads phase, not writes.
  - *Deferred to Phase 4:* `brain_assembly:433` stale-context count is a bespoke filtered `nodes` read (type+locked+archived+created_at), not a TF-IDF write.
**3c — node writes + titles (pending):** raw `UPDATE nodes …` → `NodeDAL.update_field`/etc.; `SELECT title …` → `NodeDAL.get_title`.
**Goal:** Adopt the Category-B *write* methods; remove the write violations.
**Work:** Route `brain_remember.py` TF-IDF block → `TfIdfDAL`; node UPDATEs → `NodeDAL.update_field`/etc.; the 3 `brain._get_*_count` + daemon_server counts → `NodeDAL`/`GraphDAL` counts; node-title reads → `NodeDAL.get_title`.
**Verify:** `./dev pytest tests/`; `eval/s1_encode_eval.py` (encode path touches remember); decode_funnel for counts on recall.
**Stop point:** `brain_remember.py` raw-SQL count drops from 37 toward the bespoke-only floor. Commit.

### Phase 4 — Migrate reads (stop reading raw SQL)  ☐
**Goal:** Adopt Category-B *read* methods + new read methods; fix the 2 N+1 loops.
**Work:** community_decoder/temporal metadata-KV reads → `MetadataDAL`; collapse the 2 N+1 loops (community_decoder.py:847/959) to bulk; add `TraceDAL` methods for conversation.py's 3 raw `trace_events` queries; decide SessionStateDAL (resurrect+adopt brain.py session_state, or delete); add `GraphDAL.rename_relation` for reclassify.py:92.
**Verify:** `./dev pytest tests/`; decode_funnel (community + recall touched).
**Stop point:** conversation.py no longer reaches into `_trace_dal.conn`; N+1 gone. Commit.

### Phase 5 — Missing DALs + structural extractions  ☐
**Goal:** Close the no-DAL subsystems and the god-class misfit.
**Work:** Build `EntityDatesDAL` (temporal_extraction write + fetch_tools read); extract `SourceRefDAL` from GraphDAL; add `Brain.delete_node_cascade` and route `purge`/archive through it.
**Verify:** `./dev pytest tests/`; temporal + recall-by-time paths.
**Stop point:** `entity_dates` and source-refs fully DAL'd; one cascade-delete path. Commit.

### Phase 6 — Lock it (guardrail)  ☐
**Goal:** Prevent regression; normalize remaining contracts.
**Work:** Add a contract test that fails on new raw `.execute(` outside `dal*.py`/`schema.py`/allowlisted maintenance files; normalize neighbor-row key → `id` (B-KEY) with a contract assertion; sweep Category-C test-only methods (delete or document); fold in BACKLOG P4.17 (`judge_output`→`surface_output` in `dal.py:get_user_turns`) if the file is open.
**Verify:** full suite; the new guardrail test fails on a planted violation.
**Stop point:** migration locked. Commit + archive this doc's predecessor reference.

---

## Progress

| Phase | Status | Commit | Notes |
|---|---|---|---|
| 0 — Safe cleanup + fixes | ☑ | 2026-05-30 | dead Category-A gone; B-FTS/SIG/LCK/NAME fixed; 0 new test fails |
| 1 — F3 correctness | ☑ | fd9c313 (+0cd1c1d) | writers+callers batch-aware; xhigh review later found 3 MISSED writers (co_anchored×2, connect(), hebbian) → completed in 0cd1c1d; 9 batch tests |
| 1b — F3 **structural** (#5) | ☑ | dal-cleanup-2 | conn-bound `in_batch` flag (`BatchAwareConnection`) + `commit_unless_batched`; `commit` kwarg + `_batch_mode` deleted; all 33 dal.py commits gated; guardrail `test_write_txn_discipline.py`. Kills the by-convention fragility |
| 2 — Repository aggregate | ☑ | d13d671, c1f9ceb | held 5 DALs; ~57/68 sites converted; residual = bg-writer + daemon_hooks + conn-params |
| 3 — Migrate writes | ◐ | cfc8f02, 0cd1c1d, dal-cleanup-2 | 3a counts ✅; 3b TF-IDF ✅ (store/rebuild→TfIdfDAL, dead _tfidf_score deleted, batch-scorer read→Phase 4); 3c node-writes/titles pending |
| 4 — Migrate reads | ☐ | — | |
| 5 — Missing DALs + extractions | ☐ | — | |
| 6 — Lock it | ☐ | — | |

## References
- `docs/archive/stale/DAL-MIGRATION-PLAN.md` — original plan (Step 0 never finished; anticipated F3)
- `docs/WRITE-TXN-ISOLATION-ROOTFIX.md` — F3 root-cause + Option A/B
- `docs/BACKLOG.md` — F3 (~1 day), P4.17
- Decision node `7d14f588` (locked) — DAL designed for incremental adoption
