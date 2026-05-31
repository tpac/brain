# DAL Cleanup & Migration Plan

**Started:** 2026-05-30 · **Owner:** Anchor + Tom · **Status:** Phases 0–2 + 3a on `main` (merge `530d2f8`). **Structural F3 fix (#5) + F3 review remediation + Phase 3 (3b, 3c) MERGED to `main` 2026-05-31 — merge commit `061cb40`** (5 commits `f91a1f5`/`8aedaaa`/`191bbc0`/`81a4b61`/`8b42e6d`; full suite 1374/7/4; clean --no-ff merge, parallel S2 stream's worktree work preserved). **Phase 3 + 4 COMPLETE** (Phase 4 on `main`: 9367509, 3595e02, 20f9276, 1e4e706, + `_batch_tfidf_scores` slice — SessionStateDAL resurrected & session_state fully swept, metadata-KV/trace_events/reclassify/N+1/tfidf reads all on the DAL, equivalence-tested). Open: Phase 5 (EntityDatesDAL, SourceRefDAL, delete_node_cascade), Phase 6 (raw-SQL guardrail), deferred 3c `update_fields` fork.

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
> `# commit-ok:`). **Also fixed (same race class, the spawned follow-up):**
> `Brain.save()` was likewise called **lock-free** in `_run_idle_maintenance`
> (daemon_server.py:831) with an unconditional `self.conn.commit()` that could
> commit a concurrent client batch's partial state. `save()` now wraps that
> commit in `self.write_lock` (RLock — the daemon autosave path that already
> holds it re-acquires safely; `logs_conn` stays outside, separate DB). Per-write
> lock granularity (matching `set_config` and the encoder dispatch at
> `base.py:319`), NOT wrapping all of maintenance. Locked by
> `TestSaveHoldsWriteLock`.

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
> ## ▶ Next session — start here (handoff 2026-05-31)
> **Done & on `main` (`081a634`):** Phases 0–3 + structural F3 fix (#5) + F3 review
> remediation. Full suite 1374/7/4. **Resume at Phase 4** (see the Phase 4 section
> below — line numbers re-audited 2026-05-31).
>
> **✅ Phase 4 underway (`9367509`):** the reserved `SessionStateDAL` decision is
> resolved — resurrected; the `session_state` table is now fully DAL-gated
> (brain.py ×3 + session_context.py save/load). Remaining Phase 4: metadata-KV
> reads → MetadataDAL, conversation.py trace_events → TraceDAL, reclassify
> rename_relation, and the recall-hot-path `_batch_tfidf_scores` (needs eval).
>
> **Audit gotchas found this session (don't trip on them):**
> - `eval/decode_funnel.py` (cited in CLAUDE.md + this doc as THE recall benchmark)
>   **was removed.** Use `eval/brain_recall_identity_eval.py` / `eval/surface_funnel.py`.
>   (CLAUDE.md line fixed this session.)
> - `fetch_tools.py` **no longer exists** — the Phase-5 `entity_dates` *reader* lived
>   there; re-grep `FROM entity_dates` to find the current reader.
> - `NodeDAL.purge` (dal.py:1537) is an **incomplete cascade** (misses `node_vectors`
>   + `node_source_refs`) — the Phase-5 `delete_node_cascade` must cover all 5 tables.
> - `_batch_tfidf_scores` (recall HOT PATH) was deferred from 3b → Phase 4; it has
>   two semantic traps (see Phase 4 #6) and needs a recall eval before/after.
> - The txn guardrail (`test_write_txn_discipline.py`) has two known coverage gaps →
>   Phase 6 #2 (aliased + compound-statement commits; tag `brain.py:122`).
>
> **Parallel-stream note:** the `/Users/tpac/brain` main worktree carries another
> stream's uncommitted **S2 absorb-op** WIP (`brain_remember.py` +127, `dispatch_write.py`,
> `contract.py`, `consolidation_enrichment_prompt.py`, untracked `S2-ABSORB-OP-DESIGN.md`
> / `s2_locked_*` / `test_absorb.py`). NOT part of this DAL work — leave it; commit Phase 4+
> with **explicit paths** (`git commit -- <files>`), never `git add -A`.
>
> **Working model that paid off:** the structural F3 fix means every `dal.py` writer
> now gates commits on `conn.in_batch` — so any NodeDAL/TfIdfDAL writer adopted in
> Phase 4+ is batch-atomic for free (no `commit=` plumbing). And do a `/code-review`
> on the connection/lock-touching segments — it caught a real pre-existing race this session.

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
- **Don't break the recall hot path.** Any change to `brain_recall.py` / surface / pipeline → run the recall eval before/after. ⚠ `eval/decode_funnel.py` (cited here + in CLAUDE.md) **no longer exists** — current recall evals: `eval/brain_recall_identity_eval.py` (see `eval/README.md`), `eval/surface_funnel.py`, `eval/decoding_suite.py`. (NOTE: the recall *read* hot path is read-only at SQLite; only `recall_write_queue`'s off-path drain is a write path — a pure commit-gating change there needs no recall eval.)
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
| session_state | brain.py:701/759/789 + session_context.py:219/228 | SessionStateDAL (resurrected) | 4 ✅ |
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

### Phase 3 — Migrate writes (stop writing raw SQL)  ☑ (3a, 3b, 3c done — on `main` via `061cb40`)
**3a — counts ✅ (`cfc8f02`):** `brain._get_{node,edge,locked}_count`, `brain_recall` brain-size, `brain_assembly` total_locked, `daemon_server` status counts → held DALs (`count`/`count_locked`/`count_total`/`count_by_type` adopted). daemon locked count gains the documented non-archived semantics.
**3b — TF-IDF ✅ (`dal-cleanup-2`):** `brain_remember._store_tfidf_vector` → `self._tfidf.store_tf_vector(node_id, tf)` (the verbatim-reimplemented dead class, now adopted); `_rebuild_tfidf_index` → `clear_all()` + per-node `store_tf_vector` wrapped in `conn.in_batch` (single commit preserved via save/restore + `_maybe_commit`); `brain.py` boot reindex check → `self._tfidf.get_total_docs()==0` + `self._nodes.count(archived=True)`. `_compute_tf`/`_tfidf_tokenize` stay (TF math, not DB). **Dead `_tfidf_score` (single-node, 0 callers) deleted.**
  - *Deferred to Phase 4 (reads):* `_batch_tfidf_scores` is a READ on the **recall hot path** (`brain_recall.py:884`) with subtle semantics (`total_docs = _get_node_count()` ≠ `get_total_docs()`; term-filtered node_vectors read with no exact DAL method). Migrating it needs a new `TfIdfDAL` read method + a `decode_funnel` before/after — belongs in the reads phase, not writes.
  - *Deferred to Phase 4:* `brain_assembly:433` stale-context count is a bespoke filtered `nodes` read (type+locked+archived+created_at), not a TF-IDF write.
**3c — node titles ✅ / writes mostly deferred (`dal-cleanup-2`):** the genuinely-clean wins done — `SELECT title FROM nodes WHERE id=?` → `self._nodes.get_title()` at `brain_connections._get_node_title` and `brain_remember` mark_critical (both fetch a real title; full-id exact ≡ get_title's prefix match; the dead bare `except:` in `_get_node_title` dropped). **The raw `UPDATE nodes` sites are NOT clean drop-ins and are deferred (migrating them naively = behavior change for marginal purity):**
  - `content_summary` backfill (brain_remember ~1270) deliberately omits the `updated_at` bump (it's a derived field; bumping pollutes recency-based recall) — `NodeDAL.update_field` always bumps. Leave raw.
  - archive node-UPDATE (brain_remember ~167) reuses a single `ts` shared with the archive audit metadata for consistency — `update_field`'s fresh `_now()` would desync them. Leave raw (it lives inside `archive_node`'s larger flow).
  - bg access-mark (recall_write_queue ~284) is a batched `executemany` multi-field UPDATE on `conn_bg_writer`. Specialized. Leave raw.
  - **FORK (deferred per cost):** the `revise` generic setter (brain_remember ~980, dynamic multi-field `UPDATE nodes SET %s`) and the personal-annotation writes (~1801/1806, 3-4 fields) are MULTI-field — they'd need a new `NodeDAL.update_fields(node_id, {col: val})` method. Real DAL-first value (revise is the most-trafficked node write) but it's a behavior-sensitive core path; a new method + careful migration is its own scoped task, not a cheap drop-in.
**Goal:** Adopt the Category-B *write* methods; remove the write violations.
**Work:** Route `brain_remember.py` TF-IDF block → `TfIdfDAL`; node UPDATEs → `NodeDAL.update_field`/etc.; the 3 `brain._get_*_count` + daemon_server counts → `NodeDAL`/`GraphDAL` counts; node-title reads → `NodeDAL.get_title`.
**Verify:** `./dev pytest tests/`; `eval/s1_encode_eval.py` (encode path touches remember); decode_funnel for counts on recall.
**Stop point:** `brain_remember.py` raw-SQL count drops from 37 toward the bespoke-only floor. Commit.

### Phase 4 — Migrate reads (stop reading raw SQL)  ☑ COMPLETE
**Goal:** Adopt Category-B *read* methods + new read methods; fix the N+1 loops; resolve the SessionStateDAL decision. **All items done (#1–#7); #7 decided-documented.**
**Work (line numbers re-audited 2026-05-31):**
1. **✅ DONE (`3595e02`) — metadata-KV reads → `MetadataDAL`**: `read_community_meta` → `get_field` (wraps handed conn); per-node loop → `self.brain._meta_kv.get(nid).items()`; drift-threshold load → `get_all_by_key`; `temporal_extraction.backfill_node_dates` → `MetadataDAL(conn).get(node_id)`. (NOTE: other raw `node_metadata_kv` reads exist beyond the plan's list — consolidation_encoder/decoder, healer_decoder subqueries, brain_reminders, dispatch_ops — mostly JOIN/subquery shapes; follow-up audit, not swept.)
2. **✅ DONE — N+1 collapsed**: `community_decoder._read_community_state` now bulk-fetches all nodes' community metadata in one `_meta_kv.get_fields_bulk(nids, COMMUNITY_METADATA_KEYS)` call (was one `get()` per node), narrowed to the keys it uses. (The other N+1 the original audit cited at :847/959 has shifted — re-grep before fixing if it resurfaces.)
3. **✅ DONE (`20f9276`) — conversation.py `trace_events` → `TraceDAL`**: added `latest_in_window(scale, ref_type, upper, lower)` + `find_by_metadata_substring(scale, ref_type, substring)`; the 3 reaches into `brain._trace_dal.conn` (`_find_encoding_session` ×2, `_from_traces_by_timestamp`) now call them. (`_resolve_node_timestamp`'s `nodes` reads are node-table, out of this item's scope.)
4. **✅ DONE (`9367509`) — `SessionStateDAL` resurrected (Tom's call: resurrect).** Added `ensure_default` (INSERT OR IGNORE — preserves a racing thread's state, unlike `set`'s upsert), `recently_updated`, `sessions_by_message_count`. Held logs-bound: `self._session_state = SessionStateDAL(self.logs_conn)`. Adopted the 3 brain.py sites (`get_or_create_session`/`present_streams`/`live_sessions`, `write_lock` kept at call sites) **and** `session_context.py` `save`/`load` (route through the DAL by wrapping the handed conn — SessionContext has no Brain ref). **Zero raw `session_state` SQL survives outside `dal.py`/`schema.py`** — the table is fully swept. New `test_session_state_dal.py` (6) locks the ensure_default-vs-set semantic.
5. **✅ DONE (`20f9276`) — `GraphDAL.rename_relation`** adopted at `reclassify.py` (no weight recompute — a rename changes neither weights nor active count); dropped the trailing raw `self.brain.conn.commit()` (redundant + would break atomicity inside a batch).
6. **✅ DONE — `_batch_tfidf_scores` routed through `TfIdfDAL`** (recall hot path). Added `TfIdfDAL.get_tf_vectors_for(terms, node_ids)` for the term+node-filtered `node_vectors` read; doc_freq via `get_doc_freq(term) **or 1**` (trap caught: inline defaulted absent-term df=**1**, but `get_doc_freq` returns 0 — `or 1` preserves the IDF). `total_docs = self._get_node_count()` kept (NOT `get_total_docs()`). **Verified by `tests/test_batch_tfidf_dal_equivalence.py`** — pins DAL-routed scores to a verbatim reference of the original raw-SQL logic (exact, places=12), incl. the absent-term edge. Unit equivalence chosen over an LLM eval (which can't detect small scoring drift); the plan's named `brain_recall_identity_eval.py` is a formatting/LLM A/B, not a scoring eval.
7. **DECIDED — leave documented (raw, accepted):** `brain_assembly.py:433` stale-context count (`type='context' AND locked=0 AND archived=0 AND created_at < ?`) + its paired auto-fix SELECT at `:446`. It's a health-check integrity audit (wall-clock-exempt, single caller). A `NodeDAL.count`/`get_ids` method this specific would be a single-caller method that *fragments* the DAL — the opposite of the cleanup's goal. Phase 6's raw-SQL guardrail targets DML (INSERT/UPDATE/DELETE), not SELECTs, and brain_assembly is a maintenance/audit file — so this stays raw by design.
**Verify:** `./dev pytest tests/`; recall eval for #6 (`eval/brain_recall_identity_eval.py` / `eval/surface_funnel.py` — NOT `decode_funnel.py`, which no longer exists).
**Stop point:** conversation.py no longer reaches into `_trace_dal.conn`; N+1 gone; SessionStateDAL decision resolved. Commit.

### Phase 5 — Missing DALs + structural extractions  ☐
**Goal:** Close the no-DAL subsystems and the god-class misfit.
**Work (re-audited 2026-05-31):**
1. **Build `EntityDatesDAL`** — write side: `temporal_extraction.py:331 write_entity_dates` (DELETE at :347, INSERT at :356/:370). **⚠ Read side moved:** the original plan cited `fetch_tools.py` for the `entity_dates` read — **`fetch_tools.py` no longer exists.** Re-locate the `recall_by_time` reader (grep `FROM entity_dates`) before building the DAL.
2. **Extract `SourceRefDAL`** from GraphDAL — 4 methods that operate on `node_source_refs`, not edges: `add_source_refs` (dal.py:2902), `replace_source_refs` (:2936), `get_source_refs` (:2972), `get_nodes_referencing` (:2983).
3. **`Brain.delete_node_cascade`** — centralize per-table deletes. **⚠ `NodeDAL.purge` (dal.py:1537) is an INCOMPLETE cascade** — it DELETEs `node_enrichments`/`node_metadata_kv`/`edges` but MISSES `node_vectors` and `node_source_refs` (orphan rows leak on purge). The cascade method must cover all 5 child tables; route `purge`/archive through it.
**Verify:** `./dev pytest tests/`; temporal + recall-by-time paths.
**Stop point:** `entity_dates` + source-refs fully DAL'd; one complete cascade-delete path. Commit.

### Phase 6 — Lock it (guardrail)  ◐ (in progress)
**Goal:** Prevent regression; normalize remaining contracts.
**Work:**
1. **✅ DONE — Raw-SQL guardrail** (`tests/test_raw_sql_guardrail.py`): a ratchet over per-file raw-DML counts in `servers/` (excl. `dal*`/`schema`). Baseline frozen at **40 sites / 11 files** (2026-05-31, categorized exception-vs-pending in `ALLOWED`); fails on any NEW raw INSERT/UPDATE/DELETE/REPLACE **and** when a migration drops a file below baseline (keeps the allowance honest). Carries a detector teeth-test so it can't go vacuous. Detection is literal-DML-after-`.execute(`; variable-assembled SQL is not caught (rare, different smell).
2. **Tighten the txn-discipline guardrail** (`test_write_txn_discipline.py`, from the F3 review): `TestBrainSelfCommitsAreMarked` currently matches `startswith('self.conn.commit()')` — it MISSES (a) aliased commits like `instance.conn.commit()` at `brain.py:122` (benign teardown flush in `clear_instances`, but should be tagged `# commit-ok:`), and (b) compound-statement commits (`foo; self.conn.commit()`). Switch to a regex on the code-portion of each line matching `\.conn\.commit\(\)` (catches aliases; `logs_conn.commit()` is excluded by the leading-dot requirement) with comment/docstring stripping + the `# commit-ok:` allowlist. Tag `brain.py:122` while there.
3. **Normalize neighbor-row key → `id`** (B-KEY) with a contract assertion.
4. Sweep Category-C test-only methods (delete or document); fold in BACKLOG P4.17 (`judge_output`→`surface_output` in `dal.py:get_user_turns`) if the file is open.
**Verify:** full suite; the new guardrail tests fail on a planted violation.
**Stop point:** migration locked. Commit + archive this doc's predecessor reference.

### Cosmetic / non-blocking (noticed, not scheduled)
- `_get_node_title` (brain_connections) now returns `node_id` instead of `''` for an empty-title node (`get_title(...) or node_id` short-circuits). Labeling-only, arguably an improvement; titles are effectively never empty. Leave unless it bites.

---

## Progress

| Phase | Status | Commit | Notes |
|---|---|---|---|
| 0 — Safe cleanup + fixes | ☑ | 2026-05-30 | dead Category-A gone; B-FTS/SIG/LCK/NAME fixed; 0 new test fails |
| 1 — F3 correctness | ☑ | fd9c313 (+0cd1c1d) | writers+callers batch-aware; xhigh review later found 3 MISSED writers (co_anchored×2, connect(), hebbian) → completed in 0cd1c1d; 9 batch tests |
| 1b — F3 **structural** (#5) | ☑ | dal-cleanup-2 | conn-bound `in_batch` flag (`BatchAwareConnection`) + `commit_unless_batched`; `commit` kwarg + `_batch_mode` deleted; all 33 dal.py commits gated; guardrail `test_write_txn_discipline.py`. Kills the by-convention fragility |
| 2 — Repository aggregate | ☑ | d13d671, c1f9ceb | held 5 DALs; ~57/68 sites converted; residual = bg-writer + daemon_hooks + conn-params |
| 3 — Migrate writes | ☑ | cfc8f02, 8aedaaa, 8b42e6d | 3a counts ✅; 3b TF-IDF→TfIdfDAL + dead _tfidf_score deleted ✅; 3c titles→get_title ✅ (risky multi-field UPDATEs deferred — see Phase 4/§Cosmetic). On `main` via `061cb40` |
| (review) — F3 lock-discipline remediation | ☑ | 191bbc0, 81a4b61 | code-review of F3 found a pre-existing race: S2 maintenance's `set_config`/`save()` committed `self.conn` lock-free; both now hold write_lock. Guardrail widened to brain*.py |
| 4 — Migrate reads | ☑ | 9367509, 3595e02, 20f9276, 1e4e706, +tfidf | ✅ SessionStateDAL (table fully swept); ✅ metadata-KV→MetadataDAL; ✅ conversation.py trace_events→TraceDAL; ✅ reclassify rename_relation; ✅ N+1 collapse; ✅ `_batch_tfidf_scores`→TfIdfDAL (equivalence-tested, no scoring drift); brain_assembly count decided-documented. **Phase 4 COMPLETE.** |
| 5 — Missing DALs + extractions | ☐ | — | EntityDatesDAL (⚠ fetch_tools.py gone — re-find reader); SourceRefDAL extract; delete_node_cascade (⚠ purge misses node_vectors+node_source_refs) |
| 6 — Lock it | ☐ | — | raw-SQL guardrail; tighten txn guardrail (alias/compound commits + tag brain.py:122); normalize neighbor key |

## References
- `docs/archive/stale/DAL-MIGRATION-PLAN.md` — original plan (Step 0 never finished; anticipated F3)
- `docs/WRITE-TXN-ISOLATION-ROOTFIX.md` — F3 root-cause + Option A/B
- `docs/BACKLOG.md` — F3 (~1 day), P4.17
- Decision node `7d14f588` (locked) — DAL designed for incremental adoption
