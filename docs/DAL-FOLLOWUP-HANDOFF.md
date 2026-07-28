> **SUPERSEDED 2026-07-28 — absorbed into `docs/DAL-BOUNDARY-ARCH-PLAN.md`.**
> M1 (edge_exists direction bug) SHIPPED in DAL Step 0 (`77e5fcb`). ITEM 1
> (`NodeDAL.update_fields`) lives on as that plan's Step 8; H1 (filter_nodes
> metadata N+1) as its Step 3. The execution details below (leave-raw list,
> ratchet discipline) stay authoritative for those two items.

# DAL follow-up — execution brief (post-compaction)

Self-contained spec for two pieces left after the DAL cleanup arc (Phases 0–6, all
on `main`). Written 2026-05-31 before a context compaction. Everything needed to
execute is here — re-grep line numbers (they drift), but the substance is fixed.

## Environment / discipline (READ FIRST — bit me twice this session)

- **Worktree:** `/Users/tpac/brain/.claude/worktrees/youthful-panini-08eef6`, branch
  `claude/youthful-panini-08eef6`. `main` is in sync (0/0) as of this writing.
- **NEVER `cd /Users/tpac/brain` in Bash** — that's the MAIN repo, a different working
  tree; greps/imports there miss uncommitted worktree edits. Stay in the default
  (worktree) cwd. (This gotcha cost two verification cycles already.)
- **Run tests** (worktree venv has no pytest; use the main venv binary + bypass, from
  the worktree cwd):
  `BRAIN_ALLOW_ANY_PYTHON=1 /Users/tpac/brain/venv/bin/python -m pytest tests/... -q -p no:cacheprovider`
- **Baseline = 6 known reds, NOT ours** (other streams): `test_invalid_op_suppression`
  ×3 (absorb-op `closed_five`→six) + `test_mcp_roundtrip` ×3 (self-channel `self_send`).
  Success criterion: exactly those 6 fail, zero new. Full suite ≈ 7.5 min — run in
  background; don't run two at once (they thrash the embedder).
- **Per-slice commits, explicit paths, then FF main:**
  `git commit -F - -- <files>` then `git -C /Users/tpac/brain merge --ff-only "$(git rev-parse HEAD)"`.
  If FF fails (parallel stream advanced main): `git merge main --no-edit` in the
  worktree (clean — parallel work has been disjoint), then FF forward.
- A **full-suite gate is the real net** for cross-file changes — per-slice targeted
  tests missed a file once (SourceRefDAL → test_encoder_eval_probes tearDown). Don't
  call a slice done until the full gate is green.

---

## ITEM 1 — deferred 3c: `NodeDAL.update_fields` + migrate multi-field node UPDATEs

**Why deferred:** the clean-drop-in 3c writes were done; these are MULTI-field
`UPDATE nodes SET ...` writes needing a new method, and `revise` is the most-trafficked
node write (behavior-sensitive) — its own scoped task.

**Build:** `NodeDAL.update_fields(node_id, fields: dict[str,val], bump_updated_at=True)`
in `servers/dal.py` (near `update_field`). Dynamic `UPDATE nodes SET <cols> WHERE id=?`;
gate commit on `commit_unless_batched(self.conn)`. Default-bump `updated_at` (the revise
setter bumps it); allow opt-out for derived-field writes that must not bump recency.

**Migrate (current lines — re-grep `UPDATE nodes SET` in brain_remember.py):**
- `brain_remember.py:1200` — the `revise` generic setter (`UPDATE nodes SET %s` built
  from `set_parts`/`params`). The primary target. Route through `update_fields`. It
  bumps `updated_at` today — preserve that.
- `brain_remember.py:2021` + `:2026` — personal-annotation writes (`personal`,
  `personal_context`, [`locked`], `updated_at`). Two near-identical UPDATEs (locked vs
  not) — both → `update_fields`.

**Leave raw (documented reasons — do NOT migrate):**
- `:1490` `content_summary` backfill — deliberately omits the `updated_at` bump (derived
  field; bumping pollutes recency). `update_fields(..., bump_updated_at=False)` *could*
  take it later, but it's a single bespoke write — leave unless trivially clean.
- `:200` archive UPDATE — reuses one `ts` shared with archive audit metadata.
- `:427` access-mark increment (`access_count = access_count + ?`) — not a SET-value write.

**Guardrail:** removing the `:1200`/`:2021`/`:2026` raw UPDATEs drops
`brain_remember.py`'s raw-DML count. The ratchet (`tests/test_raw_sql_guardrail.py`,
`ALLOWED['brain_remember.py']` currently **11**) will FAIL on the drop — lower the number
to the new count (re-run the scan in the test to get it).

**Verify:** `test_revise*`, `test_remember*`, `eval/s1_encode_eval.py` path; full gate.

---

## ITEM 2 — DAL-audit efficiency wins H1+H2+H3 (recall HOT PATH — eval-gate it)

All three are behavior-preserving and the bulk methods already exist. Do as ONE slice;
gate with a recall equivalence check (like `_batch_tfidf_scores` got
`test_batch_tfidf_dal_equivalence.py`) — assert recall output unchanged before/after.

**H1 — `filter_nodes` metadata N+1** (`servers/brain_recall.py`): `_apply_filter`
(`:168`) → `_matches` (`:187`) calls `mdal.get_field(node['id'], key)` per node (`:193`)
inside the result loop. For a metadata-key filter that's N separate
`SELECT … node_metadata_kv`. Fix: before the loop, collect the filter keys NOT in node
columns, call `MetadataDAL.get_fields_bulk(all_node_ids, those_keys)` ONCE
(`dal_metadata.py` — exists, purpose-built), have `_matches` read the pre-fetched dict.
Semantics identical (missing key → falsy). Boot/Frame + per-turn path → high value.

**H2 — `get_naked_node` looped → `get_bulk`** (`servers/brain_recall.py`): looped sites
`:852` (tfidf seed hydration, ~50), `:1782` (embedding-only hydration), `:2065`, `:2146`
(FTS5). `NodeDAL.get_bulk(node_ids)` (`dal.py`) returns `{id: row}`, built for exactly
this. Replace each loop with one `get_bulk` + dict iteration; the `if node and not
node['archived']` filter maps onto dict iteration. NOTE: `:2023` is a SINGLE lookup
(in the `get_node` path) — verify it's not in a loop before touching; likely keep.

**H3 — kill per-call `SELECT * FROM nodes LIMIT 0`** (`servers/dal.py:1390` in
`get_naked_node`, `:1413` in `get_bulk`): both re-query column names every call. Use the
data query's own `cursor.description` instead (the `SELECT * … WHERE` cursor already has
it) — zero extra round-trip. Compounds with H2 (halves query count on the hydration
loops). Schema is process-stable; daemon restarts on migration.

**H4 (verify-first, optional):** `get_neighbors` vs `get_neighbors_bulk` in spread/
traverse (`brain_recall.py:~1999`, `pipeline_contract.py:~432`) — collapse ONLY if the
caller loops over multiple owners. Trace loop structure first.

**Verify:** recall eval (`eval/brain_recall_identity_eval.py` is formatting-only — use a
unit equivalence test on the affected methods + `eval/surface_funnel.py` against an
IsolatedBrain copy if you want end-to-end); full gate at the 6-red baseline.

---

## State at handoff
DAL arc Phases 0–6 complete on `main`. Two guardrails live: `test_raw_sql_guardrail.py`
(DML ratchet), `test_write_txn_discipline.py` (tokenize commit-discipline). The DAL-audit
full findings (H1–H4, M1–M2, L1–L3) are summarized in this session's final report; M1
(`edge_exists`≡`get_edge_id is not None`) and M2 (`add_relation` 2× node-exists → one IN)
are cheap medium wins if appetite remains after H1–H3.
