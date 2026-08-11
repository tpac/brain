# Vector / Index Deletion Consolidation — Investigation + Plan

**Status: SHIPPED 2026-07-17 (with amendments from the code-review pass).**
Investigated against main @ 96f9a1c; every file:line below was read in current code, not inferred.

## 0. Execution amendments (an 8-angle code review revised the plan before commit)

The review caught that this plan **endorsed a wrong premise** — "archive and cascade both
'stop the node being findable', so unify them through one `_deindex_node`." They are NOT the
same operation, and three parts of the plan changed at execution:

1. **Archive is SOFT delete, not symmetric with cascade.** §4.2's "archive now deletes tfidf"
   was reverted: `_deindex_node(node_id, include_tfidf=True)` — archive passes `False`, cascade
   defaults `True`. Deleting tfidf on archive inflated `doc_freq` (idf skew every archive —
   `TfIdfDAL.delete_for_node` never decrements) and stripped `include_archived` lexical
   reachability. Soft delete keeps the node row + tfidf; drops embeddings + FTS.
2. **De-index moved INSIDE the archive transaction (§4.3 reversed).** Post-commit de-index had
   a real escape (a de-index failure raised *after* `archived=1` committed → node archived but
   reported failed, trace lost). In-transaction is atomic (failure rolls the whole archive
   back), matches cascade's posture, and needs resync only on archive's own rollback.
3. **The archived-residue maintenance sweep (§8 step 6) was removed, not added.** Atomic archive
   leaves no crash-window to heal. Only a legacy FTS-orphan cleanup (`NOT IN nodes`) remains.

Also fixed by the review (correctness, all tests had passed over them): `VectorDAL.store`/
`store_batch` committed inside their exception-swallowing `try` (a failed COMMIT was silently
absorbed with the cache ahead of DB) → commit moved out; the dead FTS `except` in `_deindex_node`
(Fts5DAL self-logs) removed; `_absorb_unwind` resync moved to `finally`; the batch entry-flush
rollback gained a resync; a redundant `except: pass` around the already-total resync removed.
Lesson encoded: brain node 6a1af480 — a thorough plan's blind spot is its most-confident premise;
depth doesn't substitute for the adversarial pass. The body below is the ORIGINAL plan, retained
for the reasoning; read §0 for what actually shipped where they differ.

## 1. Why

A production bug shipped (fixed in 9095528) because node-embedding deletion had drifting
half-paths: `revise` raw-SQL-deleted only the affected `node_enrichments` rows but evicted the
node's ENTIRE vector set from the in-memory cache; the backfill (DB-truth) never restored the
survivors, so every revised/healed node went recall-invisible until daemon restart.

The fix repaired that one site. The disease is structural: **node de-indexing (enrichments,
tfidf, FTS5, in-memory cache) is hand-rolled at each caller**, each with its own subset of the
job, its own raw SQL, and its own cache-ordering comment. Operator directive: consolidate it.

## 2. Verified inventory — every site that deletes/invalidates node-derived index state

Substrates: `node_enrichments` (embedding vectors), `node_vectors`+`doc_freq` (tfidf), `nodes_fts`
(FTS5), `VectorCache` (in-memory mirror of enrichments), `embed_queue` (pending re-embeds).

| # | Site | File:line | enrichments | tfidf | FTS5 | cache | Discipline |
|---|------|-----------|-------------|-------|------|-------|------------|
| 1 | **revise invalidation** | `servers/brain_remember.py:1311-1344` | raw SQL, typed (`vectors_affected_by`) @1319-1322 | re-stored @1363 | re-upserted @1371 | `drop_node(node_id, vector_types=...)` @1338 (post-9095528) | **Two hand-mirrored calls** (raw SQL + typed cache drop) that must agree byte-for-byte; `AttributeError` swallow for plain-DAL parity @1340 |
| 2 | **archive_node** | `servers/brain_remember.py:189-368` | raw SQL, ALL rows, inside archive txn @296-299 | **not touched** | `_fts.delete` inside txn @304-313 | hand-sequenced POST-commit `drop_node(full_id)` @324-336, guarded by `hasattr` for plain DAL | Raw SQL + a 12-line comment documenting the cache-ahead-of-DB hazard the two-phase shape creates |
| 3 | **delete_node_cascade** | `servers/brain_remember.py:132-163` | `_vec_dal.delete_for_node` @150 | `_tfidf.delete_for_node` @151 | **MISSING** (see §5.1) | via `CachedVectorDAL.delete_for_node` @150 — but eagerly, inside the batch envelope (see §5.2) | DAL-routed — the near-model-citizen, with two real gaps |
| 4 | **absorb → archive** | `servers/brain_remember.py:545` | inherits #2, but inside absorb's SAVEPOINT with `in_batch=True` — the "post-commit" cache drop actually runs MID-transaction | | | eager | inherits #2's shape in the worst posture (see §5.2) |
| 5 | **brain_batch `archive`/`absorb`/`revise` ops** | `servers/dispatch_write.py:686-699, 701+, 762-1040` (envelope: `in_batch=True` @838, `BEGIN IMMEDIATE` @867, commit @1024, rollback @1032) | inherit #1/#2 | | | eager cache mutation mid-envelope; rollback @1032 does NOT resync cache | see §5.2 |
| 6 | **Orphan sweep** | `servers/dal_logs.py:186` (node_vectors), `:210` (enrichments), `:216` (doc_freq), called from idle maintenance `servers/daemon_hooks.py:849-854` with `graph_conn=brain.conn` | raw SQL, orphans only (`node_id NOT IN (SELECT id FROM nodes)`) | raw SQL, orphans only | **not touched** | **never invalidated** — but see §6.2: orphan rows were loaded into the cache at boot anyway | Raw SQL in a DAL file (LogsDAL) reaching into graph tables; cannot touch live nodes by construction |
| 7 | **DAL primitives** | `servers/dal.py:1100-1103` (`VectorDAL.delete_for_node` — node_id only, no types, no commit), `:553-558` (`TfIdfDAL.delete_for_node`, `commit_unless_batched`), `:647-658` (`Fts5DAL.delete`, no commit), `:560-564` (`TfIdfDAL.clear_all` — reindex path) | | | | | The primitives exist but don't cover the typed case, so callers hand-roll |
| 8 | **Cached decorator** | `servers/dal_vector_cached.py:98-103` (`delete_for_node` — full only), `:105-115` (`drop_node` — cache-only, typed post-9095528) | | | | | Two overlapping public methods; `drop_node`'s file-header docstring @19-21 is STALE ("the underlying SQL row stays") — archive has deleted the rows for a long time |
| 9 | **Cache primitive** | `servers/vector_cache.py:97-124` (`VectorCache.drop_node`, typed) | | | | | Correct since 9095528 |
| 10 | **embed_queue pending entries** | `servers/embed_queue.py:488-560` | n/a | | | | **No cleanup needed — verified benign**: drain calls `backfill_vectors(node_ids=batch)` whose `find_missing` filters `n.archived = 0` (`servers/dal.py:1036`) and scopes to existing ids; archived/deleted ids in the queue drain as no-ops |

Callers that funnel into the above (all inherit the consolidation for free — verified there is no
other archive/delete entry point in `servers/`, `scales/`, `hooks/`): brain_batch archive op
`dispatch_write.py:692`; S2 consolidation `scales/s2/consolidation_decoder.py:165`; S2 community
`scales/s2/community.py:255`; integrity hook `brain_assembly.py:448`; absorb
`brain_remember.py:545`; idle-maintenance vocab cleanup → `delete_node_cascade`
`daemon_hooks.py:758-783`.

Explicitly out of scope (different substrate, no node-vector coupling): edge-embedding
invalidation (`dal_graph.py:1204-1215` NULLs `edge_relations.embedding` + `enqueue_edge`) and
trace embeddings.

## 3. Read-path liveness audit (what makes residue safe — verified per path)

The archive-atomicity argument in §4.3 rests on this. Every recall-relevant read of the four
substrates filters liveness at READ time:

| Read path | Where | Filter |
|---|---|---|
| FTS5 candidate lane | `dal.py:619-623` | JOIN `nodes`, `archived = 0` (pinned by `tests/test_fts_archived_filter.py`) |
| tfidf keyword candidates | `dal.py:522-533` `get_nodes_matching_terms` | JOIN `nodes`, `archived = 0` |
| tfidf rescoring | `brain_remember.py:668` `_batch_tfidf_scores` | scores only the candidate ids it is GIVEN (already live); `total_docs` = `NodeDAL.count()` = `archived = 0` (`dal.py:329-335`, `brain.py:1238-1240`) |
| Cache-served scans | `dal_vector_cached.py:125-179` (`get_all_vectors`, `get_all_situations`, `get_all_with_context`) | mask via `_archived_ids()` / nodes-context join |
| Plain-DAL scans | `dal.py` (e.g. `get_all_situations` @991-995) | JOIN `nodes`, `archived = 0` |
| Backfill eligibility | `dal.py:1036` `find_missing` | `archived = 0` |
| LAF matrix | `dal.py:962-982` `vectors_since` JOINs `archived = 0`; deletions shrink `change_key` (count) → full rebuild (`recall_laf.py:488-497`) | bounded staleness until next `change_key` sync |

**Consequence 1 (answers the tfidf question from the brief):** tfidf residue on archived nodes
**cannot resurface them** — both read paths above filter. The archive-doesn't-clean-tfidf gap is
**cosmetic** (dead rows + storage), not a correctness bug.

**Consequence 2:** everything about de-indexing an *archived* node is hygiene, not correctness —
the `archived=1` flag, committed first, is the single source of truth. This is what licenses the
post-commit consolidation in §4.3.

## 4. The consolidation

### 4.1 ONE DAL method — `VectorDAL.delete_for_node(node_id, vector_types=None)`

Extend-before-create: no new name. Both implementations do the full correct job in one call.

```python
# servers/dal.py — VectorDAL
def delete_for_node(self, node_id: str, vector_types=None) -> int:
    """Delete a node's enrichment vectors — all rows, or only the given
    vector_types (revise invalidation). The ONE deletion path for
    node_enrichments; no caller raw-SQLs this table."""
    if vector_types is not None:
        vts = list(vector_types)
        if not vts:
            return 0
        ph = ','.join('?' * len(vts))
        self.conn.execute(
            'DELETE FROM node_enrichments WHERE node_id = ? AND vector_type IN (%s)' % ph,
            [node_id, *vts])
    else:
        self.conn.execute('DELETE FROM node_enrichments WHERE node_id = ?', (node_id,))
    n = self.conn.execute('SELECT changes()').fetchone()[0]
    commit_unless_batched(self.conn)   # NEW — joins the DAL-wide commit discipline
    return n
```

```python
# servers/dal_vector_cached.py — CachedVectorDAL
def delete_for_node(self, node_id: str, vector_types=None) -> int:
    """DB delete + cache drop, mirrored EXACTLY (same vector_types set) —
    the invariant whose violation was the 2026-07-17 healer-invisibility bug."""
    with self._sql_lock:
        n = self._inner.delete_for_node(node_id, vector_types=vector_types)
    self._cache.drop_node(node_id, vector_types=vector_types)
    return n
```

The DB-delete and cache-drop can no longer disagree: one parameter feeds both. Plain-vs-cached
parity is structural — same signature, same DB effect; cached adds the mirror. The revise-side
`AttributeError` dance (`brain_remember.py:1337-1344`) and archive-side `hasattr` guard
(`brain_remember.py:331`) both die.

Adding `commit_unless_batched` to `VectorDAL.delete_for_node` matches `TfIdfDAL.delete_for_node`
(already commits) and is a no-op inside every existing envelope (cascade, brain_batch set
`in_batch=True`); revise's trailing `_maybe_commit` becomes redundant at that site and is removed
with the raw SQL.

### 4.2 ONE brain-level helper — `_deindex_node(node_id)` for archive + cascade

**Decision: yes, share one helper** (the brief's question). Archive and cascade both mean "this
node must stop being findable through every index"; today they each cover a different subset
(archive misses tfidf, cascade misses FTS5) — exactly the drift class Tom named. The helper lives
in `BrainRememberMixin` (`brain_remember.py`), which already owns the node lifecycle; it stays
DAL-first — it orchestrates, each store keeps its own delete:

```python
def _deindex_node(self, node_id: str) -> int:
    """Remove a node from EVERY search index: enrichment vectors (+ in-memory
    cache), tfidf, FTS5. The single de-indexing path for archive_node and
    delete_node_cascade. Returns enrichment rows deleted (trace metadata).
    Composes with batch envelopes via commit_unless_batched."""
    vectors_deleted = self._vec_dal.delete_for_node(node_id)
    self._tfidf.delete_for_node(node_id)
    self._fts.delete(node_id)            # keeps its own try/except + loud log
    self._maybe_commit()                 # Fts5DAL.delete doesn't self-commit
    return vectors_deleted
```

- **`delete_node_cascade`** replaces lines 150-151 with `self._deindex_node(node_id)` — inside
  the existing envelope, unchanged atomicity, and **gains the missing FTS5 delete** (§5.1).
- **`archive_node`** drops steps 4-6 (raw SQL @296-299, FTS block @302-313, post-commit
  `drop_node` @324-336) and calls `self._deindex_node(full_id)` once, AFTER the archive
  transaction commits (§4.3). `vectors_deleted` for the trace event comes from the return value.
- **Revise does NOT use the helper** — it re-stores tfidf/FTS immediately after (they're not
  stale, only the typed vectors are) and calls the §4.1 primitive directly:

```python
# revise, replacing brain_remember.py:1317-1344
if invalidated_vectors:
    self._vec_dal.delete_for_node(node_id, vector_types=invalidated_vectors)
```

- The existing FTS5-table-absence handling (test DBs without FTS5, @304-306) moves into the
  helper unchanged (or is subsumed by `Fts5DAL.delete`'s existing try/except — decide at
  implementation; the loud-log stays either way).
- **Archive now deletes tfidf** (behavior change, deliberate): verified cosmetic-only today (§3),
  and keeping it bought nothing coherent — vectors and FTS were already deleted, and there is
  no unarchive path in the codebase to preserve it for (§6.3). Symmetry wins.

### 4.3 Archive atomicity — the two-phase shape dies; hygiene goes post-commit

Today: DB deletes INSIDE the archive txn + hand-ordered cache drop AFTER commit, with the
ordering hazard documented in a comment (@324-330). The two-call shape is the bug factory.

**Decision: one consolidated `_deindex_node(full_id)` call after the archive transaction
commits.** Argument (verified, not just inherited):

1. The archive txn commits `archived = 1` + edge soft-archive first. From that instant, every
   read path filters the node out (§3 table — each path checked individually). De-indexing is
   pure hygiene.
2. Crash/failure between commit and de-index leaves residue rows on an archived node — invisible
   to recall, cosmetic. Self-healing added in Step 6 (maintenance sweep). Compare the OLD failure
   mode the two-phase shape guarded against with comments: cache ahead of DB → live node
   invisible until restart. Residue-on-dead-node is strictly the better failure.
3. The half-archived-node hazard the current envelope comment (@254-262) worries about (archived
   but still in FTS) was ALREADY demoted to hygiene when FTS search gained the liveness JOIN
   (`dal.py:594-601` documents exactly this). The vector/FTS deletes no longer need to be inside
   the txn at all.
4. Composability: when archive runs inside an outer envelope (absorb SAVEPOINT, brain_batch op),
   "post-commit" is really "post-archive-body, inside the outer txn" — the DAL deletes gate their
   commits on `in_batch` and join the outer commit; the cache drop is eager. That eager drop
   inside a rollback-able envelope is the §5.2 hazard, healed once, centrally, by Step 5's
   reload-on-rollback — not by per-site ordering comments.

### 4.4 Rollback coherence — reload-on-rollback, not a deferred-drop queue

§5.2 documents the latent inverse bug: eager cache mutation inside a batch envelope + later
rollback ⇒ cache diverges from DB until restart (dropped rows the DB still has → invisible live
node; added rows the DB rolled back → ghost). Two candidate designs:

- **(a) Deferred cache mutations**: queue drops/adds on the connection while `in_batch`, flush on
  commit, discard on rollback. Correct, but new machinery on `BatchAwareConnection`, ordering
  semantics for interleaved add/drop of the same key, and every envelope owner must flush — a new
  drift surface, in the exact shape we're eliminating.
- **(b) Resync-on-rollback**: rollback is the rare, already-loud failure path. `CachedVectorDAL`
  already has the exact tool: `reload()` (`dal_vector_cached.py:67-71`) — one SELECT, ~200 ms,
  restores cache = DB unconditionally.

**Decision: (b).** Add `self._vec_dal.reload()` (hasattr-guarded — plain `VectorDAL` has no
cache to resync, and per loud-by-default we do NOT add a fake no-op `reload` to it) to the three
rollback sites:

1. `dispatch_write._handle_brain_batch` rollback (`dispatch_write.py:1032`) — covers revise /
   archive / absorb ops inside failed batches.
2. `delete_node_cascade` except-branch (`brain_remember.py:158-161`).
3. `_absorb_unwind` (`brain_remember.py:570-581`) — covers standalone absorb failure.

(The recall_write_queue drain envelope never deletes vectors — Hebbian/access writes only —
verified no cache mutation there; left alone.)

### 4.5 `CachedVectorDAL.drop_node` is removed from the public DAL surface

After the conversions, `drop_node`'s only caller was archive (now `delete_for_node`) — the
"mask cache, keep DB rows" semantic has ZERO production callers, and its docstring
(`dal_vector_cached.py:19-21`) already describes a design that no longer exists (archive has
deleted DB rows for a long time). Per clean-as-you-go: delete the method; `VectorCache.drop_node`
(`vector_cache.py:97`) survives as the internal cache primitive `delete_for_node` uses. Tests
that exercised `drop_node` directly (`test_dal_vector_cached.py:242-245, 294+`) convert to
`delete_for_node` semantics (DB rows gone AND cache rows gone).

### 4.6 Raw-SQL end state

Zero `DELETE FROM node_enrichments` outside the DAL layer:

- `servers/dal.py` — the one owning primitive (§4.1).
- `servers/dal_logs.py:210` — **kept, as a documented exception**: the orphan sweep is a DAL-file
  maintenance path, structurally unable to touch live nodes (`node_id NOT IN (SELECT id FROM
  nodes)`), and runs against a caller-supplied `graph_conn`. Routing it through VectorDAL would
  need a set-based orphan delete on VectorDAL for one janitorial caller — not worth the surface.
  Add a one-line comment marking it as the sanctioned exception.
- `brain_remember.py` drops from 2 raw enrichment-DELETEs to 0. The ratchet test
  (`tests/test_raw_sql_guardrail.py`) forces the `ALLOWED` baseline DOWN for
  `brain_remember.py` — the ratchet then permanently forbids regression.

## 5. Latent bugs found beyond the known inventory (fixed by this plan)

### 5.1 `delete_node_cascade` misses FTS5
`brain_remember.py:150-155` cleans enrichments, tfidf, kv, edges, source_refs, nodes — **not
`nodes_fts`**. There are no FTS triggers (verified: `schema.py` creates a plain
manually-maintained FTS5 table @924; the only upsert/delete callers are
`brain_remember.py:948,1371` and archive). A hard-deleted node leaves a permanent FTS row: it
can't surface (search JOINs `nodes`), but it violates the function's own "EVERY child-table row"
contract, and NOTHING ever cleans it — the orphan sweep doesn't cover `nodes_fts` either.
`tests/test_delete_node_cascade.py` ("clears every child table") does not pin `nodes_fts` — the
gap is invisible to the suite. Fixed by §4.2; test extended.

### 5.2 Eager cache mutation inside rollback-able envelopes
The inverse of the healer bug, still live today in three postures: (1) `CachedVectorDAL.
delete_for_node` inside `delete_node_cascade`'s envelope — a later cascade-step failure rolls the
DB back but the cache rows are gone; (2) brain_batch `revise`/`archive` ops mutate the cache
mid-envelope (`in_batch=True` @dispatch_write.py:838) — batch rollback @1032 leaves the cache
diverged; (3) absorb's SAVEPOINT — `archive_node`'s "post-commit" cache drop actually fires
mid-transaction because `prior=True` skips the commit but not step 6. All three: live node
invisible (or ghost vectors) until restart. Fixed centrally by §4.4.

### 5.3 Stale documentation at the seam
`dal_vector_cached.py:19-21` and `test_dal_vector_cached.py:7,243` describe archive as
"SQL row stays, cache masks" — archive deletes the rows. Drift artifacts of exactly the disease
being treated; corrected in passing (Step 4).

## 6. Adjacent findings — flagged, deliberately NOT in this change

6.1 **doc_freq inflation (real drift bug, separate fix):** `TfIdfDAL.store_tf_vector`
(`dal.py:535-551`) deletes the node's old `node_vectors` rows then INCREMENTS `doc_freq` for
every term — without ever decrementing the old contribution. Every revise inflates df for the
node's repeated terms; `delete_for_node` (@553) doesn't decrement either. IDF degrades
monotonically; the orphan sweep (`dal_logs.py:216`) only removes terms with zero remaining rows.
Out of scope (write-side tfidf, not deletion consolidation) — recommend a follow-up task.

6.2 **Cache boot-load includes orphan rows:** `_load_cache_from_db`
(`dal_vector_cached.py:58-65`) selects ALL enrichment rows — no `nodes` join — so legacy orphan
rows enter the cache and `get_all_vectors` serves them (its mask is archived-ids only). Wasted
scan work, and the orphan sweep's DB delete never reaches the cache. Harmless (downstream node
hydration drops them) but worth a one-line JOIN at load time in a follow-up.

6.3 **There is NO node unarchive/restore path** — verified: the only `archived = 0` writer in the
codebase is the edge_relations revive branch (`dal_graph.py:1174-1191`). A manual `archived=0`
flip today yields a node that (a) re-embeds only via the idle-maintenance full backfill scan
(`daemon_hooks.py:837` → `find_missing`, archived=0 ⇒ eligible — eventually, last_accessed-
ordered, batch 30), (b) is NEVER restored to FTS5 (nothing backfills FTS), (c) keeps tfidf today
/ loses it post-§4.2. If restore ever becomes a feature it must be `unarchive_node` =
flag flip + FTS upsert + tfidf store + `embed_queue.enqueue` — the inverse of `_deindex_node`.
Not built now (zero callers; speculative).

## 7. Test plan

**Existing pins (must stay green, some updated):**
- `tests/test_revise_invalidates_vectors.py` — field→vector-type mapping (12 tests) + the
  9095528 regression `test_revise_does_not_orphan_surviving_vectors_from_cache` (@173). Should
  pass UNCHANGED — the consolidation preserves revise's observable behavior exactly.
- `tests/test_vector_cache.py` — `VectorCache.drop_node` full + typed semantics. Unchanged.
- `tests/test_dal_vector_cached.py` — UPDATED: `test_drop_node_masks_cache_only` (@242) and
  `TestArchiveIntegration` (@294, pins "archive_node ALWAYS calls drop_node") convert to the
  `delete_for_node` contract: after archive, DB rows deleted AND cache rows gone.
- `tests/test_delete_node_cascade.py` — EXTENDED: pin `nodes_fts` cleanup (closes §5.1's test
  blind spot).
- `tests/test_fts_archived_filter.py`, `tests/test_batch_tfidf_dal_equivalence.py` — unchanged
  guards for the §3 liveness assumptions.
- `tests/test_raw_sql_guardrail.py` — `ALLOWED` baseline LOWERED for `brain_remember.py`
  (ratchet locks the consolidation in).

**New regression tests:**
1. `VectorDAL.delete_for_node(node_id, vector_types=[...])` typed deletion — plain DAL (rows for
   the given types gone, others intact, count returned).
2. Cached/plain parity, parametrized over `BRAIN_DISABLE_VECTOR_CACHE`: archive_node and revise
   leave identical DB state under both DAL wirings (the parity the old `hasattr`/`AttributeError`
   guards only hoped for).
3. Full-path archive de-index: after `brain.archive_node`, zero enrichment rows, zero
   `node_vectors` rows, no `nodes_fts` row, no cache rows, and cache-served + plain scans both
   exclude the node.
4. Rollback coherence: brain_batch of `[revise(node), <failing op>]` → after rollback, the
   node's vectors are still served from the cache (reload happened); same for a mid-cascade
   failure (inject a failing DAL step) and an unwound absorb.
5. `_deindex_node` batch-composability: called with `in_batch=True`, commits nothing (envelope
   owner commits).

**Tier:** SQL/DAL/dispatch change ⇒ full suite before merge (per repo discipline), not a
filtered `-k` run.

## 8. Execution steps (each small, reviewable, independently verified)

| Step | Change | Verify |
|---|---|---|
| 1 | `VectorDAL.delete_for_node` gains `vector_types=None` + `commit_unless_batched`; `CachedVectorDAL.delete_for_node` mirrors typed via `VectorCache.drop_node` | New tests 7.1/7.2 + `test_dal_vector_cached.py`, `test_vector_cache.py` |
| 2 | Revise converts: raw SQL + `drop_node` + `AttributeError` dance (`brain_remember.py:1317-1344`) → one `delete_for_node(node_id, vector_types=...)` | `test_revise_invalidates_vectors.py` green UNCHANGED; ratchet `ALLOWED` lowered |
| 3 | Add `_deindex_node`; `delete_node_cascade` routes through it (gains FTS5); `archive_node` drops steps 4-6 for one post-commit `_deindex_node(full_id)` (gains tfidf; trace `vectors_deleted` from return) | Extended `test_delete_node_cascade.py`; new test 7.3; updated `TestArchiveIntegration` |
| 4 | Remove public `CachedVectorDAL.drop_node`; fix stale docstrings (§5.3) | grep zero `drop_node` callers outside `vector_cache.py`/`dal_vector_cached.py` internals; suite green |
| 5 | Reload-on-rollback at the three envelope rollback sites (§4.4) | New test 7.4 |
| 6 | Maintenance sweep (`dal_logs.py` run_maintenance) extends to archived-node residue: `DELETE FROM node_enrichments / node_vectors / nodes_fts WHERE node_id IN (SELECT id FROM nodes WHERE archived = 1)` — heals any crash window from §4.3; comment `dal_logs.py:210` as the sanctioned raw-SQL exception | Unit test on `run_maintenance` counting archived-residue cleanup |
| 7 | Full suite + `eval/mcp_schema_gate.py` NOT required (no brain_batch schema change — verify no `BATCH_OP_SPECS` diff); commit | Full `./dev pytest tests/` |

**Deployment:** all changes are `servers/*` ⇒ **daemon restart** required to take effect
(`restart` MCP tool / `rebrain-daemon`); no `./redeploy.sh` needed (no hooks/brain_mcp/skill
changes). Note the running daemon still exhibits §5.2 until restarted.

## 9. Honesty — what was NOT verified

- **Static analysis only.** No tests were run, no daemon touched, no DB opened. All claims are
  from reading current code; line numbers are from main @ 96f9a1c.
- The "~200 ms" `reload()` cost is the docstring's figure (`dal_vector_cached.py:59`), not
  re-measured.
- LAF (`recall_laf.py`) was verified to filter archived at matrix-append (`vectors_since`) and to
  full-rebuild on deletion via `change_key`; the exact staleness window between an archive and
  the next `change_key` sync was not traced end-to-end. Bounded and pre-existing; unaffected by
  this plan.
- I did not enumerate every test-suite caller of `VectorDAL.delete_for_node` /
  `CachedVectorDAL.drop_node`; the signature extension is backward-compatible for the former, and
  Step 4's grep gates the latter's removal.
- Worktrees under `.claude/worktrees/` contain divergent copies of `brain_remember.py`; this plan
  targets the main tree only — merging streams should rebase onto it.
