# DAL Boundary — Architecture Plan

## Scope

The DAL layer boundary: the 14 DAL classes across `servers/dal*.py`, every consumer that reaches
into them directly, duplication inside the DALs, and ops-layer code hand-rolling what a DAL method
already provides.

**Boundary traced (2026-07-28):** 5 DAL files + `db_backends/`, ~25 consumer files with direct
reach-ins across `servers/`, `scales/`, `hooks/`, `dashboard/`. Four parallel angle reviews
(layering/coupling, duplication/merges, placement/cohesion, altitude/instantiation), findings
verified against live code before inclusion. Consumer counts are production-only (`servers/`,
`hooks/`, `scripts/`, `dashboard/`); they are perishable — re-derive before acting.

**Prior-art status (recalled, verified):** the DAL cleanup arc (`docs/DAL-CLEANUP-PLAN.md`,
Phases 0–6) is COMPLETE. The traces-door audit (2026-07-11, `e8fcb61`) fixed 4 of 5 violations;
the fifth (daemon_hooks wake-envelope filter) is **deliberately parked** (id:c948eeb8) pending a
`get_conversation` signature widening — respected here, not reopened.
`docs/DAL-FOLLOWUP-HANDOFF.md`'s items are absorbed into this plan (ITEM 1 → Step 8; H1 → Step 3;
M1 → Step 0) — **mark that doc superseded when Step 0 lands.**

**Settled constraints respected throughout:** DAL-first (CLAUDE.md); no business logic in the DAL
(id:c8de37c6, "Dall shouldnt have business logic"); generic doors over bespoke DAL queries
(id:6e0279b7); S2 decoder raw SELECTs are house pattern; dashboard direct SQL is by design
(read-only observer); `recall_write_queue`/`temporal_extraction` DAL construction on
`conn_bg_writer` is the documented connection exception; trace `append*` writes from any file are
guardrail-sanctioned (`tests/test_traces_layer_guardrail.py` WRITE_OK).

**Cross-thread fences:**
- `docs/ASPECT-OWNERSHIP-ARCH-PLAN.md` Step 6 owns the `DEFAULT_EXCLUDED_RELATIONS` →
  aspects-derived-policy swap. This plan does NOT do it — but §"Handoff to aspects Step 6" below
  records the consumer sites Step 6's list misses and **one contradiction requiring an operator
  decision**.
- ~~The LAF/recall stream has an unmerged branch touching `hook_recall`.~~ **MERGED to main
  2026-07-28 (`27ff524`)** — the Steps 5–6 fence is lifted. Their fix still needs a daemon
  restart to go live; check `self_presence` before restarting under an active sibling. Line
  numbers in `brain_recall.py`/`recall_write_queue.py` cited below drifted with that merge —
  re-grep (standing rule).

---

## ⚠ RESOLVED 2026-07-28 — `community_member` is HIDDEN (standing decision applied)

The standing operator decision (id:49d734ad) governs flat READS: hide `community_member` from
default connection renders. Tom's same-day refinement: graph DYNAMICS (traverse, spread
activation, graph_expand) keep conducting through community edges — conduction is not
visibility. Shipped as two load-time policies on the registry: `structural_exclusions` (full
noise — reads; DAL swap still pending, aspects Step 6) and `traversal_exclusions` (noise −
community_member — LIVE at the three traversal sites since 2026-07-28). Original contradiction
record kept below for context.

## ~~⚠ OPERATOR DECISION NEEDED~~ — `community_member` visibility

Two committed artifacts disagree about one string, and the recorded decision was never implemented:

- **Settled decision id:49d734ad (2026-06, Tom's words: "Hide and align with exclusion list for
  now!"):** `community_member` should be excluded from default edge reads — noise-taxonomy
  cleanliness wins over marginal edge value. **Never implemented**: `DEFAULT_EXCLUDED_RELATIONS`
  is still the hardcoded 2-member set, `get_node` passes no exclusion, and `community_member`
  (7,237 edges) renders into every enrichment.
- **Aspects plan Step 6 (2026-07-25):** `structural_exclusions = relations_in(['noise']) −
  {community_member}` — i.e. KEEP it visible, with the `dal_graph.py:63` comment ("real thematic
  context") and `consolidation_decoder.py:783-786`'s dependence cited as the reason.

Whichever way Tom re-confirms, fix BOTH artifacts (revise id:49d734ad or amend aspects Step 6) so
exactly one intent survives. Until then neither this plan nor aspects Step 6 should touch the
default exclusion membership.

---

## Dependency summary

```
0 (correctness fixes) ── independent, do first
1 (interaction door)  ── independent, tiny
2 (same-conn dedup)   ── independent, mechanical
3 (hot-path N+1s)     ── independent (metadata halves); neighbor halves wait for 5
4 (MetadataDAL collapse + small merges) ── independent
5 (GraphDAL signature alignment) ──► 6 (brain_graph.py door)   [after LAF branch merges]
7 (policy out of DAL)  ── independent of 5/6; sub-items independent of each other
8 (ops raw-SQL → DAL sweep, incl. update_fields) ── independent; ratchet lowered per slice
9 (structural merges: Vector/Trace/GraphDAL archive unification) ── after 4; B4 is a behavior fix
10 (dal.py split + hygiene) ── LAST (moves everything; import churn)
```

Steps 0–4 are safe parallel-session material. 5→6 is one arc. 10 waits for content to settle.

---

## Step 0 — Correctness fixes — **SHIPPED 2026-07-28 (`77e5fcb`)**

(Landed with a refinement: two exclusion policies — see the resolved decision box above.)

**Problem.** Three verified defects found in passing:
1. `brain_connections.py:325-328` — bridge-creation existence check queries ONE direction
   (`source_id=? AND target_id=?`). Edges are stored single-direction (v22), so a pre-existing
   reversed edge is missed and a duplicate bridge is created. `GraphDAL.get_edge_id`
   (`dal_graph.py:125`) already checks both directions. (= DAL-FOLLOWUP-HANDOFF "M1".)
2. `brain_remember.py:~1699` `_compute_group_vectors` — dead (zero live callers; docstring says
   "no longer called") but loaded: raw `INSERT OR REPLACE INTO node_enrichments` that (a) bypasses
   the vector cache (written vectors invisible until restart) and (b) builds its PK as
   `{node_id}_{vector_type}` (single underscore) vs `VectorDAL.store`'s `%s__%s` — since the table
   has no unique on (node_id, vector_type), a future caller would DUPLICATE rows, not replace.
   Delete the function.
3. `dispatch_read.py:148-153` MCP `graph_expand` uses `brain_constants.EXCLUDED_EDGE_TYPES`
   (`{'emergent_bridge'}`, 1 member) while every other traversal uses
   `pipeline_contract.TRAVERSE_EXCLUDED_EDGES` (`{'co_accessed','emergent_bridge'}`) — so the MCP
   tool leaks `co_accessed` noise edges other paths filter. Minimal fix: point `dispatch_read` at
   `TRAVERSE_EXCLUDED_EDGES` (behavior-visible: graph_expand output loses co_accessed edges — the
   fix IS the point). Deeper unification is Step 6 + aspects Step 6.
4. `brain_recall.py:518` `get_interaction` — byte-identical copy of `brain.py:728`, unreachable
   (class body shadows mixin). Delete. (Verified via `def get_interaction` grep — exactly two.)

**Verification.** New test: bridge not created when reverse edge exists. `test_contract_sync`,
targeted recall tests for the graph_expand change. Grep proves the deletions.
**Blast radius.** Small; #3 changes MCP graph_expand output deliberately.
**Depends on.** None. **Respects.** id:6e0279b7.

## Step 1 — Finish the interaction-registry door — **SHIPPED 2026-07-28 (`58273cf`)**

**Problem.** The door is half-built: `brain.get_interaction`/`set_interaction_active`/
`list_interactions` exist but `register` has none, so callers reach past the wall —
`dispatch_observability.py:232,260,268` and `interaction_seed.py:288` hit `_interaction_dal`
directly, and two existing wrappers have ZERO production callers because dispatch bypasses them.
**Target state.** Add `brain.register_interaction(...)`; route the 4 sites through the doors.
Layering decided once, not per call site.
**Files.** `servers/brain.py` (+1 method), `dispatch_observability.py`, `interaction_seed.py`.
**Verification.** `test_prompt_sync.py`, `test_contract_sync.py`, interaction tests.
**Blast radius.** ~10 lines. **Depends on.** None.

## Step 2 — Same-connection duplicate DAL construction + stub-tolerance guards — **SHIPPED 2026-07-28 (`58273cf`)**

(Review at the commit boundary caught a NameError on the spread stored-embeddings branch —
fixtures never reach it, so targeted tests were blind; plus a missed brain-free test caller,
an unresolved-id batch lookup in muster, and a door-defaults divergence. All fixed in the
same commit. The session-DAL param threading altitude item is noted for Step 6's arc.)

**Problem.** Sites construct throwaway DALs on connections whose cached instance already exists
(`brain.py:254-262` holds one of each — the DAL-CLEANUP deferral reason "parallel-stream file"
has expired):
- `daemon_hooks.py:533` NodeDAL, `:792` GraphDAL, `:852` LogsDAL (all on brain's own conns)
- `brain_assembly.py:855` LogsDAL (self IS the Brain)
- `brain_recall.py:366,397,427` — **`Brain.get_node`, the canonical read door, builds 3 DALs per
  call** (+2 duplicate local imports); `self._nodes/_meta_kv/_graph` are bound to the same conn
- `surface_contract.py` `brain_conn` params (`_build_edge_coeffs:1025` etc.) — callers pass
  `(brain, brain.conn, ...)`; GraphDAL rebuilt per spread hop inside recall
- `session_context.py:271-279` — SessionStateDAL per save/load per autosave tick; callers already
  hold `brain._session_state`; change `save/load` to take the DAL
- `dispatch_ops.py:166,173` — `hasattr(brain,'_vec_dal'/'_logs_dal')` guards on attributes set
  unconditionally in `__init__`, wrapped in try/except anyway; drop guards, route `:173` through
  `brain.get_recent_errors` (pass `limit=20` — the door defaults 10, the DAL 20)
- Same family: `fetch_tools.py:765,770` + `scouts/muster.py:336-346` hasattr-on-Brain-methods
  guards (muster's per-id fallback branch is dead code reintroducing an N+1)
**Verification.** Full targeted: daemon, hooks, session, recall pipeline tests. Behavior-preserving.
**Blast radius.** ~15 files, mechanical. **Depends on.** None.
**Respects.** DAL-CLEANUP-PLAN residual list (its deferral reasons, now expired, are cited there).

## Step 3 — Hot-path N+1 batch fixes (eval-gated)

**Problem.** Verified per-call loops on hot paths where the bulk door already exists:
- `dispatch_read.py:21-22` — MCP `recall` enrichment calls `brain.get_node(id)` per result (≤8);
  the batch form is documented ("5 queries instead of N×4") and `_handle_get_nodes` already uses it
- `brain_recall.py:2226` — `_enrich_results` metadata `self._meta_kv.get(nid)` per node →
  `get_all_bulk` (used 30 lines away at `:397`)
- `brain_recall.py:199` — `_apply_filter._matches` `get_field` per node per key →
  `get_fields_bulk` hoisted above the loop (= handoff H1, still open, confirmed)
- `pipeline_contract.py:477-481` — `mdal.get(sid)` per seed → `get_all_bulk`
- `surface.py:55-60` + `encode.py:828-832` — `get_title` per id loops (≤20/turn) → one `get_bulk`
  (encode.py itself batches at `:936`)
- `community_decoder.py:75,517,520` — `read_community_meta` constructs MetadataDAL + 2 SELECTs per
  community per S2 run, while `:315` already bulk-fetches those very keys
  (`community_internal_fraction`/`maturity` ARE in COMMUNITY_METADATA_KEYS) then discards them.
  Widen the dict at `:343-350`; delete the helper.
- `consolidation_encoder.py:345` — `get_node` per cluster member (cold path; do while there)
NEIGHBOR halves of `_enrich_results` (`:2236`) and `pipeline_contract.traverse` (`:444-448`) are
NOT here — they need Step 5's `get_neighbors_bulk` signature work first.
**Verification.** Equivalence-gate: `eval/surface_funnel.py` + a recall-output-unchanged assertion
(model: `test_batch_tfidf_dal_equivalence.py`). Targeted: recall pipeline, community tests.
**Blast radius.** Hot path — behavior-preserving but eval-gated. **Depends on.** None.

## Step 4 — MetadataDAL collapse + small paired-method merges

**Problem.** Verified byte-level duplication with trivial merges:
- MetadataDAL: 5 read methods = one SQL shape (`get`≡`get_all_bulk([nid])`,
  `get_fields`≡`get_fields_bulk([nid],keys)`, etc. — grouping loops byte-identical); 3 write
  methods = one UPSERT. → one `_kv()` + one `_write()` + thin adapters. Net ~−40 lines.
  Preserve: `get_field` None-vs-{} semantics; writers deliberately don't commit (callers own txn).
- `NodeDAL.get_naked_node` → `get_bulk([nid]).get(nid)` (row-coercion block copy-pasted).
- `NodeDAL.archived_subset` → filter of `_live_status_bulk` (same query ± one predicate).
- `TfIdfDAL.store_tf_vector` first statement is verbatim `delete_for_node` — call it.
- `SourceRefDAL.add/replace_source_refs` → shared `_write_refs(*, replace)`; keep both public
  names (append-at-create vs replace-at-revise is contract-load-bearing).
- `NodeDAL.count(archived=)` — param sense is inverted vs sibling `count_locked(include_archived=)`;
  rename (2 call sites).
- `SessionStateDAL.set`/`ensure_default` → `_upsert(overwrite=)`; kill the dead datetime import.
- `InteractionDAL.get_active` fallback ≈ `get_version` — share the row builder.
- `find_missing`: canonical kwarg `require_kv_keys_any` has ZERO callers; the "misleading" alias
  `source_kv_keys` is the only one used. Pick one, drop the other from BOTH VectorDAL and
  CachedVectorDAL (their signatures must mirror — documented TypeError trap).
- `Fts5DAL.upsert` `_legacy_keywords` param — both callers pass `''`; drop it.
**Verification.** DAL component tests; `test_write_txn_discipline.py`; full suite before merge
(SQL/DAL tier). **Blast radius.** DAL-internal; adapters preserve semantics. **Depends on.** None.

## Step 5 — GraphDAL single/bulk signature alignment (reviewed step — defaults trap)

**Problem.** The single/bulk siblings have drifted into a trap:
- `get_neighbors` defaults `exclude_relations=None` → NO exclusion; `get_neighbors_bulk` defaults
  the SAME kwarg → `DEFAULT_EXCLUDED_RELATIONS`. Opposite meanings, same name. Bulk also lacks
  `limit`(per-owner)/`exclude_node_ids`/`content_preview_chars`, which is why
  `pipeline_contract.py:444` loops per seed (live N+1) and `_enrich_results:2236` can't batch.
- `get_community_members(cid, include_archived, require_active_member)` vs
  `get_members_bulk(cids, include_archived)` — bulk hardcodes `member.archived=0` AND adds a
  `c.type='community' AND c.archived=0` gate the single doesn't have; `community_decoder.py:339`
  loops per community because the bulk lacks `require_active_member=False`.
**Target state.** Align kwargs + defaults explicitly (a reviewed decision, not a silent merge);
single forms delegate to bulk; then batch the three blocked call sites (traverse loop,
`_enrich_results` neighbors — with `include_relations` pushed into SQL replacing the
fetch-3×-filter-in-Python idiom — and the community-members loop).
**Verification.** Recall equivalence gate + `test_community_structural.py`,
`test_community_membership_reconcile.py`, spread-activation tests. Full suite (DAL tier).
**Blast radius.** Recall + S2 candidate sets if a default flips silently — hence reviewed.
**Depends on.** Step 4 helps; **land after the LAF branch merges** (shared hot-path files).

## Step 6 — `brain_graph.py`: the graph read door + node-liveness doors

**Problem (the structural headline).** `brain_traces.py` exists as the ONE door for TraceDAL;
there is NO equivalent for GraphDAL or NodeDAL-liveness — `brain_connections.py` holds only
writes. So all ~16 graph reads/maintenance calls and ~12 liveness/resolution calls from scales,
dispatch, and hooks are FORCED reach-ins (nothing to call), including on the recall hot path.
Plus: short-ID resolution exists as NINE copies with FOUR divergent miss-semantics
(dispatch_common returns-input-on-miss; brain_traces prefix-fallback; surface.py zero-pad retry;
the rest raw `resolve_id if len<16`), and `Brain.get_node` holds a seventh copy internally.
**Target state.** `servers/brain_graph.py` (`BrainGraphMixin`) mirroring `brain_traces.py`:
reads `neighbors/communities_for/community_members(_bulk)/has_edge_between/
nodes_touched_by_relations/edge_counts_by_relation`; maintenance `decay_edges/
archive_dangling_edges/reconcile_community_membership/rename_relation` (delegation shims —
relocating their POLICY is Step 7). Node doors: `brain.archived_subset/resolve_live/resolve_id(s)`
— with ONE canonical miss/fallback semantics (design decision inside this step: recommend
`resolve_ids(list) -> {input: full|None}` owning the length threshold + zero-pad + prefix
fallbacks; callers choose their miss policy). `NodeDAL.resolve_ids` bulk method kills the
per-item LIKE round-trips (`dispatch_write` pays it per batch row). Statusline counts:
one `brain.store_counts()` replacing `daemon_server.py:883-886`'s four reach-ins + the three
near-duplicate private `_get_*_count` wrappers. `integrity_audit.py` gets
`brain.metadata_coverage(fields)`. `community_encoder`'s three live-id checks →
`NodeDAL.live_subset` (inverse of `archived_subset`, same query).
**The ratchet:** clone `tests/test_traces_layer_guardrail.py` → `test_graph_layer_guardrail.py`;
without it this door erodes like the interaction door did.
**Files.** New `servers/brain_graph.py`; ~28 call sites across scales/s1, scales/s2, dispatch_*,
daemon_hooks, pipeline_contract, integrity_audit.
**Verification.** Full suite (import surface + DAL tier). Guardrail test proves the fence.
**Blast radius.** Wide but pure delegation. **Depends on.** Step 5 (bulk signatures), LAF merge.
**Respects.** id:6e0279b7 (this is its graph twin); parked wake-envelope untouched.

## Step 7 — Policy out of the DAL (each sub-item independent)

**Problem.** Verified business logic living in storage code (violates id:c8de37c6):
- `dal_graph.py:485-586` `reconcile_community_membership` — a whole S2 healer (regex-parses
  member lists, "zero-edge case only" judgment, weight/description/encoding_source policy) →
  move judgment to the community unit; DAL keeps row primitives.
- `dal_graph.py:955-1022` `decay_edges` — half-life formula + prune threshold + EDGE_TYPES
  iteration → maintenance layer; DAL keeps `bulk_decay(relation, half_life)` +
  `archive_below(relation, threshold)`.
- `dal.py:616,622` — FTS `bm25(...,10.0,1.0)` title-weight ranking policy → pass weights from
  recall constants. `dal.py:660-675` `_sanitize_query` stop-words/8-term-cap → lexical layer.
- `dal.py:392-413` `filter_nodes` returns `{"error":...}` MCP payloads + caps limit at 200 →
  DAL raises/plain data; dispatch shapes errors.
- `dal_logs.py:139-248` `LogsDAL.run_maintenance` — retention policy + a cross-DB janitor that
  DELETEs six brain.db tables from the LOGS DAL on a passed-in conn, re-implementing
  `GraphDAL.hard_delete_node_edges` (the 17,982-orphan incident is this duplication's recorded
  cost) → maintenance module composes per-DAL primitives.
- `dal_logs.py:83-97` `log_hook_error` — CREATE TABLE per call + prune-to-200 → schema.py / maint.
- `dal_logs.py:1218-1374` `get_session_turns` — transcript assembly (role mapping, judge-output
  cross-ref, `'id8|title'` wire-format decode, ×2 windowing heuristic) belongs in brain_traces;
  DAL returns rows. The wire format needs a contract home either way.
- `dal_logs.py:1075-1204` — presence/turn definitions (live-types, wake-envelope exclusion)
  rebuilt 3×; extract shared predicate fragments (full restructure optional).
- `dal.py:1174,1186` EntityDatesDAL silent truncation → reject or caller trims.
- `ABSORB_EXCLUDED_RELATIONS` (`dal_graph.py:82`) — zero DAL consumers; lives next to `absorb`.
- Model-name literal `nomic-ai/...` as default in 4 DAL signatures → require explicit (config owns
  it; aligns with Tom's model-config direction, id:23a321af).
**Verification.** Per sub-item targeted tests; community/consolidation tests for the reconcile move;
`test_time_window_contract.py` for maintenance changes. **Depends on.** Step 6's shims make the
relocations mechanical but sub-items don't strictly depend on it.

## Step 8 — Ops-layer raw SQL → DAL sweep (incl. the open `update_fields` item)

**Problem.** Raw SELECT is unpoliced (`test_raw_sql_guardrail.py` matches DML only) and it shows.
Verified sites, each with the DAL target named:
- `brain_connections.py:66-71,111-114` edge-embedding backfill SELECT/UPDATE → GraphDAL
  (`find_unembedded_relations`/`store_relation_embedding` — TraceDAL already has this exact pair);
  falsifies `dal_graph.py:105` "ALL edge SQL lives here".
- `brain_connections.py:283-306` two-hop bridge SQL → GraphDAL.
- `recall_write_queue.py` hebbian strengthen UPDATEs + access-mark `executemany` →
  `GraphDAL.strengthen_co_access` / `NodeDAL.bump_access_batch` (the bg-conn exception covers the
  CONNECTION, not raw SQL — the file already constructs GraphDAL). Invariant updated by `d238f60`
  (2026-07-28): access marks touch `last_accessed` ONLY — "reads must never look like writes",
  no `updated_at` bump at all. That invariant moves into the DAL method's docstring.
- `brain_recall.py:336-346` degree cache UNION scan → `GraphDAL.degree_by_node(exclude_relations)`
  (aspects Step 6 swaps the literal; this moves the query).
- `brain_recall.py:275-278` rerank embedding fetch — **bypasses CachedVectorDAL entirely** (pays
  the SQL the cache exists to avoid) → route through the vector DAL.
- `brain_recall.py:665-690` hand-built `_situation` missing-vector scan → `VectorDAL.find_missing`
  + a `select_kv_field=` projection param (makes "eligible ⇔ yields text" one definition).
- `brain_assembly.py:247-315` boot reads (critical/locked×2/recent) → `NodeDAL.critical_nodes/
  locked_rows(limit,offset,with_content)/recent_ids`; the locked×2 pair is ONE query run twice
  (differing only in projection/paging); boot RANKING stays in brain_assembly as a passed sort
  spec. `brain_assembly.py:101-110` neighbor SQL → `get_neighbors_bulk`.
- `brain_reminders.py` — entire 37-line mixin is two raw LIKE queries → NodeDAL/MetadataDAL.
- `brain_remember.py:2380-2384` title LIKE → `NodeDAL.title_like`; `:1586-1598` summary backfill;
  `:1284-1290` old-value capture → `NodeDAL.get_fields(node_id, cols)` (whitelisted), pairing with
  **`NodeDAL.update_fields`** — build it and migrate the 3 multi-field UPDATEs exactly per
  `DAL-FOLLOWUP-HANDOFF.md` ITEM 1 (leave-raw list included there stays authoritative);
  `:1265-1267` re-SELECT of a row already fetched — reuse.
- `brain_recall.py:2373-2390` `_get_recent` half-raw/half-DAL → `NodeDAL.recent_ids` (shared with
  brain_assembly's copy — E6).
- `dispatch_observability.py:164-189` raw DELETEs on hook_errors/debug_log → `LogsDAL.clear(...)`
  (brain_mcp.py:1224 already states this rule for the same tables).
**Ratchet discipline.** Each slice lowers `ALLOWED` in `test_raw_sql_guardrail.py` in the same
commit. OPTIONAL (recommend): extend the ratchet to SELECTs once this step completes — that's the
mechanism that keeps Step 8 done.
**Verification.** Per-slice targeted + full suite at merge (SQL/DAL tier). Recall equivalence gate
for the brain_recall items. **Depends on.** None strictly; after Step 6 for the graph pieces.

## Step 9 — Structural merges inside the big DALs (equivalence-gated)

**Problem → target (verified duplications):**
- VectorDAL scan trio (`get_all_vectors`/`get_all_situations`/`vectors_since`) = one SQL shape →
  one `_scan(...)`; same collapse in the mirrored `vector_cache.py` trio. The 10-column node-context
  projection is copy-pasted between `dal.py:917-921` and `dal_vector_cached.py:251-254` → hoist
  `_NODE_CTX_COLS`/`_node_ctx_row()` (a dropped column today silently diverges cached vs uncached).
- `VectorDAL.store` → `store_batch([...])`; decide the None-embedding drift (store writes,
  batch skips) — currently a real semantic difference between siblings.
- TraceDAL: 4 event readers share `_event_where` but repeat the SELECT tail → `_select_events()`;
  keep per-door `hours` policy AT the door. `latest_in_window`/`find_by_metadata_substring` →
  one `first_event(...)` after widening `_event_where` (inclusive bounds, metadata-only LIKE).
- `GraphDAL` archive-a-relation exists as 4 drifted copies — **and `archive_dangling_edges` is
  the outlier that leaves stale embedding blobs + stale aggregate weight on archived relations**
  (the exact thing the other three's comments forbid). One `_archive_relations(where, ...)` —
  schedule as a BEHAVIOR FIX with its own test, not an incidental dedup.
  `delete_node_edges` chunks at 500; `hard_delete_node_edges` doesn't chunk at all (>999-edge
  node hits the SQLite bind limit on the hard path) — share the chunker.
**Verification.** Equivalence tests in the `test_batch_tfidf_dal_equivalence.py` mold for every
hot-path merge; `test_absorbed_into_edge.py` + `test_edge_relations.py` for the archive
unification. Full suite. **Depends on.** Step 4 (small merges land first).

## Step 10 — Split `dal.py`, then hygiene (LAST)

**Problem.** `dal.py` is a god-file by measurement, not vibe: 7 classes, ZERO intra-file calls,
no shared helpers beyond the baseline imports every sibling file has; churn concentrated
(since the 07-01 split: NodeDAL ×15 hunks, VectorDAL ×8, everything else ≤1). The one real
cross-class dependency points the WRONG way (`dal.py:34` imports edge-noise policy from
`dal_graph` so VectorDAL can mirror an edge filter — that's the seam, resolved in aspects Step 6 /
Step 7 here).
**Target state.** Split along churn: `dal_nodes.py` (NodeDAL+BrainMetaDAL), `dal_vectors.py`
(VectorDAL), `dal_search.py` (TfIdf+Fts5), `dal_episodic.py` (SourceRefDAL),
`dal_temporal.py` (EntityDatesDAL). `dal_logs.py` stays 4-classes-one-DB, but `TraceDAL`
(~970 lines) moves to its own file. Update `tests/test_raw_sql_guardrail.py` EXCLUDE list
(hardcodes DAL filenames) + ~15 import sites.
**Hygiene sweep (same session):** stale docstrings (`dal_metadata.py:4` claims validation that
doesn't exist; `_now()` "edge operations"), MetadataDAL's undocumented no-commit asymmetry
(document or align), dead module-level datetime imports, ~60 lines of "REMOVED on <date>"
tombstones (git carries history — per feedback_docs_current_state_only), the
`EDGE_CONTEXT_MIN_DESC_LENGTH` contract-constant home.
**Verification.** Full suite (import surface tier). **Depends on.** Steps 4/7/8/9 (content settles
before files move).

---

## Handoff to aspects-plan Step 6 (do NOT execute here)

Consumer sites Step 6's nine-site list misses, all breaking or silently changing when
`DEFAULT_EXCLUDED_RELATIONS` is deleted: `dal.py:34` + `dal.py:1087` (VectorDAL.find_missing
mirrors the exclusion — and VectorDAL has NO aspects handle, so the set must become a parameter,
changing VectorDAL **and** CachedVectorDAL signatures in lockstep — documented TypeError trap);
`dal_graph.py:787` (`get_edge_descriptions_for`); `brain_connections.py:46,58`;
`scripts/reembed_edges_drop_meaning.py:33`. Also: `get_neighbors` has NO default exclusion at all
(callers pass three different sets — see Step 0.3/Step 5); `get_node`→`get_connections_bulk`
currently passes NO exclusion (the id:49d734ad Phase-1 that never landed); and the
`community_member` contradiction at the top of this doc must be resolved by Tom first.

## Dropped — checked and rejected

- "Route `_handle_expand_graph`'s per-seed loop through `get_neighbors_bulk`" — the `seen` set
  grows across iterations; bulk changes dedup semantics. Not behavior-preserving; leave.
- "Delete zero-caller DAL methods" (`TfIdfDAL.get_node_terms`, `MetadataDAL.delete/set`,
  `VectorDAL.get_primary/get_for_node`, `CachedVectorDAL.reload`) — the Category-B precedent:
  zero-caller ≠ dead; `reload` is a documented escape hatch, `set` becomes free under Step 4's
  adapters. Verify intent per-method against brain history before ANY deletion; only
  `brain_recall.py:518`'s shadowed `get_interaction` is provably unreachable (Step 0).
- Re-opening the wake-envelope filter (id:c948eeb8) — parked by design, still parked.
- `GraphDAL.add_relation`→`embed_queue` coupling — documented, deliberate; the alternative
  (every caller remembers to enqueue) is worse.
- S2 decoder raw SELECTs, dashboard SQL, bg-writer DAL construction — sanctioned patterns.
