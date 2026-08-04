# Mutation-Trace Emitter — Decision Record (Traces Part 2)

**What this doc is.** The *why*: the problem evidence, the ruled properties, and the
corrected design rulings. Durable — it should not go stale.

**What it is not.** The work order. That lives in
[`MUTATION-EMITTER-PLAN.md`](MUTATION-EMITTER-PLAN.md), which is where line-number
citations live so staleness is contained to one file. Part 3 lives in
[`TRACES-PART3-CHARTER.md`](TRACES-PART3-CHARTER.md).

**Status:** design ruled and walked part-by-part with Tom on **2026-08-04**;
**nothing implemented, production untouched**.

History: the first draft (`2b7903c`) was reviewed by five scoped agents, which found the
design sound and the work order unsafe — two of its instructions would have damaged
production. All findings and evidence:
[`MUTATION-EMITTER-REVIEW-FINDINGS.md`](MUTATION-EMITTER-REVIEW-FINDINGS.md). This doc is the
rewrite; every citation was re-resolved against `bb7ed5a`.

Rulings Tom made during the 2026-08-04 walkthrough, all folded in below:
- Sweeps emit **per-edge rows**, never a rollup shape (§5) — keeps the emitter one-shaped.
- Hard deletes **are** recorded, per node, with title (§7).
- Coverage is expressed in **aspects**: semantic aspects traced, `noise` excluded, with
  `community_member` as the single explicit exception (§5).
- Eval/test dispatch sites stay traceless; `_log_failed_batch_ops` folds into the chokepoint (§4).
- The dead-`REF_TYPES` prune joins the same contract edit; s3/s4 entries deleted.

**Lineage.** Charter `node:98031b8e`; scope ruling `node:5fcb662c` (settled — manifest
+ additions + dead-code + touched-DAL unification); timing principle `node:17234b02`
(revised to conditional); citation discipline `node:1c3e211f`.
**Part 1** (shipped, sibling stream): the payload recorder — same disease, read side.
This is the write side.

---

## 1. Problem — verified inventory

### 1.1 Eleven hand-rolled emit sites in dispatch, one in brain, four disciplines

All line numbers below re-verified at `bb7ed5a`.

| # | Site | Emitted from | Trace | Timing (standalone) | Timing (in brain_batch) |
|---|------|-------------|-------|--------------------|------------------------|
| 1 | `dispatch_write.py:440` | `_handle_remember` connect_to | edge_relation_revised | post-commit | **orphanable** |
| 2 | `dispatch_write.py:447` | `_handle_remember` co_anchored | edge_relation_revised | post-commit | **orphanable** |
| 3 | `dispatch_write.py:527` | `_handle_remember_batch` connect_to | edge_relation_revised | post-commit | n/a (not a batch op) |
| 4 | `dispatch_write.py:535` | `_handle_remember_batch` co_anchored | edge_relation_revised | post-commit | n/a |
| 5 | `dispatch_write.py:597` | `_handle_revise` | node_revised | post-commit¹ | **orphanable** |
| 6 | `dispatch_write.py:670` | `_handle_revise_batch` | node_revised | post-commit | n/a |
| 7 | `dispatch_write.py:751` | `_op_disconnect` | edge_relation_revised | n/a (batch-only) | **orphanable, always** |
| 8 | `dispatch_write.py:1070` | `_handle_brain_batch` deferred connect_to | edge_relation_revised | — | post-commit (the only safe batch emit) |
| 9 | `dispatch_write.py:1113` | `_handle_connect` | edge_relation_revised | post-commit | **orphanable** |
| 10 | `dispatch_write.py:1154` | `_handle_revise_edge` | edge_relation_revised | post-commit | n/a |
| 11 | `dispatch_write.py:1230` | `_handle_connect_batch` | edge_relation_revised | post-commit | n/a |
| 12 | `brain_remember.py:432` | `archive_node` inline | tool_result on private `archive-{id8}` chain | post-commit | **orphanable** (via `_op_archive`/`_op_absorb`) |

¹ If the FTS upsert raises inside `revise`, the KV flush is skipped and the trace commits
against an open deferred transaction — the leak `_handle_brain_batch`'s entry-flush exists
to clean up (`dispatch_write.py:849`/`:852`).

**Root structural fact.** Traces commit **unconditionally and immediately** on
`brain_logs.db`: `TraceDAL` is bound to `logs_conn`, nothing anywhere sets
`logs_conn.in_batch`, so `commit_unless_batched` at `dal_logs.py:643` always commits. Zero
shared transaction state with `brain.db`. Any emit fired while `brain.conn` is inside the
batch envelope (`BEGIN IMMEDIATE` `dispatch_write.py:871` → commit `:1029`) produces a
durable trace for a graph write that can still roll back (`:1037`). The comment at
`dispatch_write.py:1024-1028` claiming traces are emitted "never before" the commit is
true for exactly one of the twelve sites (#8).

Secondary orphan window: `absorb`'s savepoint unwind (`brain_remember.py:668`
`ROLLBACK TO absorb_sp`) undoes the merge while `archive_node`'s inline trace stays
committed — orphanable without any whole-batch rollback.

### 1.2 Creates emit nothing

No `node_created` ref_type exists (`node:19b56d44`). A failed encode run's created nodes
leave no id record → the partial-run catalog gap (`node:30cf1bce`).

### 1.3 Attribution holes (the d857e84d class)

- The original d857e84d split is **fixed**, pinned by `tests/test_mcp_roundtrip.py:614-644`.
- **Live successor bug — pop-then-read:** `_handle_remember` pops both identity keys via
  `_pop_session_ctx` (which mutates `args` in place, `dispatch_common.py:73-81`), then
  reads `caller_session(args)` at `:437` → structurally `''`. Same in
  `_handle_remember_batch` (`:460` → `:522`). **Every co_anchored and connect_to edge trace
  from the remember handlers carries `session_id=''`.**
- `archive_node`'s inline trace: `session_id=''`, hardcoded off-chain `archive-{id8}`.
- Encoder writes carry no session by design — identity rides `encoding_source`.

### 1.4 Trace-invisible mutations

- **absorb**: survivor `revise()` emits nothing; migrated edges discard every
  `add_relation` result. A merge is invisible except the unattributed side-chain archive.
- **revise's co_anchored auto-edges** (`brain_remember.py:1546-1560`): results discarded at
  the source — structurally untraceable today.
- **`emergent_bridge` on every remember** — `brain_remember.py:1148` →
  `_bridge_at_store_time` (`:2262`), result discarded. Up to two untraced edges *per
  remember*, system-wide. (Missed by the first draft.)
- **`community_member` back-fill** — `dal_graph.py:485` `reconcile_community_membership`,
  results discarded; live caller `community_encoder.py:233`. (Missed by the first draft.)
- **Healer bypasses dispatch entirely**: `healer_encoder.py` falls back to
  `self.brain.revise(...)` because `self.dispatch is None`. Every field-fill: no trace.
- **Community/consolidation direct archives**: `community.py:253-256`,
  `consolidation_decoder.py:165-168`, `:234-237` → side-chain trace only.
- **Bulk edge sweeps**: `GraphDAL.archive_dangling_edges` (`dal_graph.py:438`) and
  `decay_edges` (`:955`) soft-archive real edges with no trace.
- **Junk-vocabulary purge** (`daemon_hooks.py:802-828`): idle maintenance **hard-deletes**
  via `delete_node_cascade` — irreversible, no trace. See §7 (open gate).

### 1.5 The manifest half-exists already

`_apply_connect_to` returns manifest-shaped rows (`brain_remember.py:2042-2045`);
`add_relation`/`connect_typed` return `{edge_id, created, revived_from_archive, updated,
deltas, warnings}` (`dal_graph.py:1118-1125`); `revise` returns `deltas`/`warnings`. The
handlers consume these locally to feed the hand-rolled emits **and throw them away**
(`connect` returns `{"connected": True}`, `dispatch_write.py:1129`). The design makes them
return the data instead of consuming it.

---

## 2. Ruled properties

1. **POST-COMMIT emission.** Stated asymmetry: a missing trace is recoverable from the DB;
   an orphaned trace lies about the graph. Prefer missing over lying. The miss window
   (crash between graph commit and emit) is accepted and loud-wrapped.
2. **Per-command rollups.** One `append_batch` (one logs-DB commit) per dispatched command,
   however many mutations it carried. Row *volume* is unchanged; only commit count collapses.
3. **Skeletal rows forever** (normal mode). Heavy payloads belong to the recorder (part 1).
4. **Never slows the write path.** After durability, off the recall path; a trace failure
   logs loudly and never raises into the caller.
5. **Attribution stamped once at the chokepoint** — kills the pop-then-read class (§1.3) by
   resolving session/chain **before** any handler mutates `args`.

### 2.1 The timing theorem is CONDITIONAL, and enforced

The first draft claimed post-commit holds "by construction." **It does not.** Handlers can
return with `brain.conn` mid-transaction, and the proof is in-tree:

- `_handle_brain_batch`'s entry-flush guard (`dispatch_write.py:840-866`) and its
  `brain_batch_stale_txn` error row exist *precisely because* upstream writes leak open
  deferred transactions.
- `MetadataKVDAL.set_many` (`dal_metadata.py:168-183`) does not commit, so a `revise`
  whose changed fields are all KV-resident is committed only by the TF-IDF or FTS call —
  both inside independent `try/except` arms that log and continue.
- `GraphDAL.archive_dangling_edges` (`dal_graph.py:438`) has **no commit call at all**
  (verified: no `commit` in the function body).

**Ruling.** State the theorem conditionally and *enforce* it at the chokepoint:

```python
if brain.conn.in_transaction:
    brain._log_error('mutation_trace_txn_open', ...)   # loud
    return                                             # skip: a miss, not a lie
```

Two lines. This turns the emitter from an assumer of durability into a **detector of the
transaction-leak class** — the most valuable thing it does beyond emitting. `node:17234b02`
carries this qualification.

---

## 3. The mutation manifest

Handlers return `mutations` as a **top-level sibling of `result`** — the same channel
`affected` already uses (`dispatch_write.py:303` builder; `runner.py:605-608` consumer).
**Verified agent-invisible:** the MCP layer forwards only `resp["result"]`
(`brain_mcp.py:1107`).

```python
"mutations": {
    "nodes": {
        "created":  [ {node_id, type, title, encoding_source} ],
        "revised":  [ {node_id, reason, encoding_source, deltas, warnings} ],
        "archived": [ {node_id, type, title, archived_by, encoding_source, reason,
                       edge_relations: [(edge_id, relation)], vectors_deleted} ],
        "deleted":  [ {node_id, type, title, deleted_by, encoding_source, reason} ],
    },
    "edges": [ {edge_id, source_id, target_id, relation, reason,
                encoding_source, deltas, warnings} ],
}
```

**Rulings on the shape:**

- **Every row carries its own `encoding_source`.** Non-negotiable: it is resolved per row
  today (`dispatch_write.py:672`; `_resolve_archived_by` at `:369` feeding `_op_archive`
  `:691` / `_op_absorb` `:709` / `_op_disconnect` `:741`; `:1100`'s `'anchor'` default;
  `:1156`), it is a **required** key in both metadata shapes, and `_infer_scale_and_chain`
  (`:160-182`) derives the trace **scale** from it. A single command-level value would send
  a mixed `brain_batch` — e.g. an op stamped `s2:consolidation` under an unstamped batch —
  to s0 on the date-fallback chain with `encoding_source:''`. Command-level value is the
  *fallback*; scale and chain derive **per row**. Preserve the asymmetric defaults
  (`'anchor'` for edges, `''` for node revises).
- **`affected` stays materialized, unchanged.** It crosses a process boundary (the runner
  reads it off the dispatch return) so it cannot become a helper over `mutations`. Keep
  `_affected()` and its three keys; hard deletes get **no** `affected` entry (matching the
  hardcoded 3-tuple at `daemon_server.py:823` and `runner.py:649-651`) and are trace-only.
- **The emit gate is per row, not per manifest.** `_handle_revise` returns
  `affected.revised=[node_id]` unconditionally while `_emit_revise_trace` returns early on
  empty deltas+warnings. Per-row gating preserves both: a no-delta revise stays in
  `affected` and emits no trace.
- **Edge rows are complete** — every field `build_edge_revise_metadata` needs, so the
  emitter never re-queries the graph. The emitter asserts row completeness and reports gaps
  via `brain._log_error` (the channel that reaches the error feed; `validate_trace_metadata`
  only warns to stderr, `dal_logs.py:578-587`).
- **Deltas are observed, not asserted** — from the DAL's actual before/after. This outlaws
  disconnect's fabricated `0→1` flip.
- **brain_batch accumulates.** Sub-manifests pop off sub-op results exactly the way
  `affected` does today, so nothing leaks into agent-facing `results[]`; the batch returns
  one merged manifest; the deferred connect_to pass appends its rows.

**The hazard the manifest does NOT protect against.** Adding fields to the *inner*
brain-method returns is agent-visible: `_handle_revise` returns `brain.revise`'s dict
verbatim (`dispatch_write.py:610`), `_handle_revise_edge` likewise (`:1167`), and
`_op_archive`/`_op_absorb` splice `**r` into `results[]` (`:967`, `:973`). New collections
(migrated-edge rows, survivor deltas, `co_anchored_made`) must be **popped into the
manifest in the same commit that adds them** — never left on the inner dict. This is the
6M-char incident class.

---

## 4. Placement: a registry-level `dispatch_command()`

**Ruled against** the first draft's "thin wrapper around handler invocation at
`daemon_dispatch`/`daemon_server`". `daemon_dispatch.py` is a **pure registry** — 130
lines, zero invocation sites — and there are **23 invoking call sites** in the tree
(verified): 3 in `servers/`, 7 in `tests/`, **13 in `eval/`**. A two-site wrapper strands
all of them, including `tests/isolated_brain.py:218` — the dispatch surface for the eval
suites — which makes §11's "pinned tests pass unchanged" unsatisfiable.

**Ruling.** `servers/daemon_dispatch.py` gains one execution function:

```python
MUTATION_COMMANDS = frozenset({
    'remember', 'remember_batch', 'revise', 'revise_batch',
    'connect', 'connect_batch', 'revise_edge', 'brain_batch',
})

def dispatch_command(brain, cmd, args, graph_changes):
    """Resolve → validate → attribute → run → emit. The caller owns the write lock."""
```

Three callers route through it: `daemon_server._dispatch` (`:763-784`),
`s2/base._make_encoder_dispatch` (`:316-326`), `tests/isolated_brain.dispatch` (`:218`).

**Why this placement is strictly better than a wrapper:**

1. **It covers the eval surface.** `isolated_brain` is one of the three.
2. **It makes the migration incremental with no double-write.** A handler that does not yet
   return a `mutations` key yields no emit → its existing inline `_emit_*` still fires.
   Convert one handler per commit; delete `_emit_*` when its last caller is gone. **No
   reader ever sees two rows for one mutation.**
3. **It is the natural home for the attribution fix.** `session_id` and `chain_id` resolve
   before the handler runs — property 5, for free, in one place.
4. **The lock stays the caller's.** Emission must happen **inside** the existing write-lock
   acquisition (`daemon_server._locked_exec` `:719`, `s2/base` `with brain.write_lock`
   `:323`), because an unlocked `logs_conn` write is exactly how another thread's
   `commit_unless_batched` commits a partial batch. `dispatch_command` therefore does *not*
   acquire the lock — it is called inside each site's existing envelope. Note
   `_log_failed_batch_ops` currently sits **outside** the lock at `s2/base.py:325`; do not
   copy that placement, and fold that check into the chokepoint (where it generalizes to
   every caller, MCP included).

**Not `WRITE_COMMANDS`.** `MUTATION_COMMANDS` is a new frozenset because
`WRITE_COMMANDS` (`scales/dispatch.py:82-87`) has ten members and **omits `revise_edge`** —
which is emit site #10, so gating on it would silently delete that trace. `COMMAND_TABLE`
`is_write=True` is ~22 commands. The mutation-carrying set is exactly the 8 above.

### 4.1 Trace mapping

| Manifest slot | ref_type | ref_id | shape |
|---|---|---|---|
| nodes.created | `node_created` (NEW) | node_id | NEW |
| nodes.revised | `node_revised` | node_id | `REVISE_METADATA_SHAPE` (unchanged) |
| nodes.archived | `node_archived` (NEW) | node_id | NEW |
| nodes.deleted | `node_deleted` (NEW) | node_id | NEW |
| edges[] | `edge_relation_revised` | `{edge_id}:{relation}` | `EDGE_REVISE_METADATA_SHAPE` (unchanged) |

New ref_types register at `("s0","delta")`, `("s1","delta")`, `("s2","delta")` — the three
scales the existing pair occupies.

**`node_archived` and the embedded lens — ruled.** Deleting `archive_node`'s inline emit is
not free: that row is `tool_result`, and `EAGER_TRACE_REF_TYPES = SAID_AND_DID_REF_TYPES`
includes `tool_result` (`trace_contract.py:198`, `embed_queue.py:46`) — so archives are
eagerly embedded and semantically reachable today (**1,203 rows** on `archive-%` chains,
measured live). Ruling: keep `title`/`type`/`vectors_deleted` on the archived and deleted
manifest rows (for a hard delete the trace is the only surviving record, so a bare id is
unresolvable), and **`node_archived` does not join the eager-embed set** — mutation traces
stay skeletal and query-reachable, per property 3. The semantic reachability of archives is
a deliberate, named loss; if it turns out to matter, adding one ref_type to
`SAID_AND_DID_REF_TYPES` restores it.

### 4.2 Chain naming — new types must not land on `-revise`

`trace_friendly.js:29` classifies any chain ending `-revise` as kind `revise`, and
`_revise()` (`:157-163`) adds `ev.ref_id` to the memory count for any ref_type other than
`edge_relation_revised`. Creations, archives and hard deletes on a `-revise` chain would
render as *"Refined N memories — Updated details on memories it already had."*

**Ruling:** the date-fallback chain for the three new types is
`{scale}-{YYYYMMDD}-mutation`; `node_revised` and `edge_relation_revised` keep
`{scale}-{YYYYMMDD}-revise` (bit-compat with `trace_friendly.js:29` and `_stop_of`'s
`None` on `-revise`, `trace_links.py:70-79`). Dashboard `_revise()` gains a ref_type branch.
Hook-origin sweeps (purge, decay, dangling) have no chain convention today and `_stop_of`
yields `None` for them — they pass an **explicit** chain `maint-{YYYYMMDD}-mutation` at
scale s2 (graph-scope maintenance, not conversational).

---

## 5. Coverage

**THE RULE (ruled by Tom 2026-08-04, in the brain's own vocabulary):**

> Trace every **node lifecycle** change, and every **relation whose aspect carries semantic
> claim**. The `noise` aspect is machinery and stays out.

Expressed in aspects rather than call sites, because that is the taxonomy the brain already
maintains for exactly this question — "what JOB does this kind of edge do?"

| | traced |
|---|---|
| **nodes** — created, revised, archived, deleted (hard) | ✓ all four |
| `identity_bearing`, `lesson_insight`, `wisdom` | ✓ |
| `correction_improvement` — corrects, supersedes, absorbed_into, reframes… | ✓ |
| `extension_refinement`, `explanation_causation`, `dependency_flow` | ✓ |
| `contradiction_conflict`, `validation_evidence`, `hierarchical_structure` | ✓ |
| `temporal_sequence`, `survivor_lineage`, `generic_relation` | ✓ |
| **`noise`** — co_accessed, emergent_bridge, dreamed_from, dream_observation… | ✗ machinery |

**Derived, not listed.** The exclusion set is `brain.aspects.noise.edge_relations`, so a new
machinery verb the S2 classifier files under `noise` is excluded automatically — no
hand-maintained list to rot. **Guard:** pin the `noise` membership in a test. If the
classifier ever mis-files a *meaningful* verb there, coverage disappears silently — and the
list already demonstrably drifts (it currently holds `temporal_sequence`,
`extension_refinement`, `validation_evidence` — aspect *names* an encoder emitted as relation
verbs).

**ONE exception: `community_member` is traced, and stays in `noise`.** Tom: *"community
members is interesting as part of S2community"* / *"co_accessed and emergent isnt interesting
for now."* The exception lives in the emitter, **not** in the taxonomy, because reclassifying
would break recall: the natural home (`hierarchical_structure`) carries
`structural_lineage: true` and `prompt_visible: true`, while `noise` has both false — so
reclassifying would make every `community_member` edge ride spread activation regardless of
text similarity, and would offer a system-owned relation to encoders as prompt vocabulary.

The two paths that create these edges are covered differently, matching "as part of
S2community":
- **Encoder `brain_batch` connect ops** → flow through dispatch → traced by the exception.
- **`reconcile_community_membership` back-fill** (`dal_graph.py:485`, called from
  `community_encoder.py:233`, results discarded) → direct DAL, never touches dispatch →
  recorded in the **community unit's own delta** (count + edge ids), where the S2 story lives.

**Sweeps.** Per-edge rows through the same trace map — never a rollup shape, which would
need a sixth `ref_id` convention and break the one-table property (§4.1). The data is already
in hand: `decay_edges` collects `pruned_edge_ids` (`dal_graph.py:993-997`) and discards them.
Only three relations decay at all, and after the aspect rule just **one** is traced:

| relation | aspect | traced | active | below threshold |
|---|---|---|---|---|
| `co_accessed` | noise | ✗ | 8,226 | 0 |
| `emergent_bridge` | noise | ✗ | 109 | 0 |
| `exemplifies` | validation_evidence | ✓ | 13 | 0 |

So sweep emission is ~**0 rows/pass** — which makes the shared-flip-primitive work in the
plan a pure correctness fix, not a coverage feature.

**The coverage claim, corrected.** The first draft said "zero edge-lifecycle changes
untraced" — false, and it contradicted its own Out table two paragraphs later. An absolute
claim that its own document falsifies is worse than a narrow true one, because the next
reader trusts it instead of checking. The honest claim: **every node lifecycle change, and
every semantically-claiming relation, reachable from a dispatch command or a named sweep.**

**Out (deliberate, documented).**

| Path | Why out |
|---|---|
| `enrich` / `store_enrichments` | Derived-data refresh; no node/edge lifecycle; already `NON_ATTRIBUTED_WRITES` |
| Hebbian `co_accessed` creation + access marks (`recall_write_queue.py:480-492`) | Infrastructure statistics, no judgment behind it, high volume; deliberately off-path |
| Vector/summary/temporal backfills, FTS rebuilds, schema migrations | Derived geometry / one-shot infra |
| Orphan-row deletes in `dal_logs.py:182-238` | Invariant restoration on rows whose nodes are already gone. **Note:** this is *not* "logs-DB janitorial" — it executes `DELETE FROM edge_relations` / `DELETE FROM edges` on a caller-supplied `graph_conn`. The exclusion holds; the first draft's label was wrong. |
| `seed_pack` boot writes | Once, on an empty brain; provenance already `anchor:seed` |

**The DAL unification (Tom's scope ask), corrected.** The three "copies of one operation"
are **three different operations**, differing on four axes: embedding NULL
(`remove_relation` yes / `archive_dangling_edges` **no** / `decay_edges` yes), aggregate
weight recompute (yes / **no** / yes), `exempt_relations` (no / **yes, load-bearing** /
no), commit gating (`commit_unless_batched` / **none** / `commit_unless_batched`).

Two live hazards if merged naively: standardizing on NULLing embeddings makes
`archive_dangling_edges` start destroying embeddings it does not touch today — a bulk
`UPDATE` on `brain.db` fired by the first idle cycle after deploy; and losing the exempt
clause severs **455 active `edge_relations`** whose endpoint node is archived (measured
live, 100% `absorbed_into` — every one alive *only* because of the exemption).

**Ruling:** one shared flip primitive with **explicit per-caller policy flags**
(`null_embeddings`, `recompute_weight`, `exempt_relations`), preserving all three
behaviors exactly. Unification of the *flip discipline and return shape*, not of policy.
The weight-**decay** UPDATE (thousands of rows/pass) stays outside the primitive.

---

## 6. Read-side preservation contract

- **ref_id formats**: `node_id` for node events; `{edge_id}:{relation}` for edge events.
  Pinned by `scripts/trace_node_reconstruct_20260612.py:109-122`, `trace_friendly.js:161`,
  `tests/test_revise_unified.py:55`, `tests/test_edge_mutation_unified.py:81`. No reader
  splits the composite; no reader JOINs trace ref_ids against nodes/edges.
- **Metadata keys**: `node_id/reason/encoding_source/deltas/warnings` and
  `edge_id/source_id/target_id/relation/...` — `trace_detail.js:33-45` shape-sniffs on
  `deltas` + (`node_id`|`edge_id`).
- **The empty-`old` convention is a signal — do not "fix" it.** The contract states an
  empty `old` in a delta means the field was just created; populated means update. Filling
  real old values into `add_relation`'s revive branch would make a revive
  indistinguishable from an update for every reader following that convention. The actual
  missing signal is the `archived: 1→0` delta, which nothing emits — **add that instead**,
  leave `old=None`, amend the shape comment.
- **`delete_node_edges` must return only rows the UPDATE flipped**, as `(edge_id, relation)`
  pairs. Its SELECT returns *all* edge ids touching the node while the UPDATE excludes
  `archived=0` misses **and `exempt_relations`** — returning the SELECT list would claim the
  deliberately-exempted `absorbed_into` redirect edge was archived. Pairs, not a flat list,
  because the flip is per `(edge_id, relation)` and that is the granularity every other
  edge event carries.
- **Dashboard is mirror-and-pin, never unify.** `dashboard/queries/s2_runs.py:23-24` states
  the servers-disconnection contract explicitly — the dashboard **cannot** import from
  `servers/`. The fix is a consistency *test* across trace_contract ↔ S2 gate ↔ dashboard
  mirror. Also `query_healer_runs` (`:488-539`) selects every delta on `chain_id LIKE
  '%healer%'` with no ref_type filter — it already emits phantom cards for `journal_note`,
  and Healer routing would add one per field-fill.
- **`EMITTER_REF_TYPES` covers the new types AND the existing pair.** Verified safe: each
  S2 unit writes its own delta ref_type (`aspect_classified`, `community_enriched`,
  `consolidated`, `healer_generated`), so excluding the pair from `_last_run_timestamp`
  (`s2/base.py:331`) cannot strand a unit at cold start. The broader reading is chosen
  because it *also* fixes the dashboard's mis-enrichment of revise rows.
- **Layering law**: all new reads go through `brain.query_traces` (`node:d1329a9f`). The
  `S1-ENCODER-ANCHOR-TOUCH.md:100-103` proposal to read via
  `brain._trace_dal.get_by_session_window` is a doc-level violation — do not inherit it.
- **Nothing prunes `trace_events`** (166,760 rows, 731MB logs DB). Volume impact is
  immaterial (`node_created` ≈ 415 rows/week ≈ 5% of weekly volume; `node_archived`
  replaces `tool_result` 1:1) but the absence of pruning is worth naming — part 1's per-kind
  retention is the pattern if it ever matters.

---

## 7. Purge observability — RULED (Tom, 2026-08-04)

**Hard deletes are recorded.** One `node_deleted` trace **per node** carrying
`{node_id, type, title, deleted_by, encoding_source, reason}`. Not a count-only rollup.

Tom: *"i think its worth saving somewhere hard deleted but we very rarely do that, its a
minor case."*

**Why per-node with title, and why this does not reopen `node:ca66f5bd`:**

- **Measured live 2026-08-04: the purge is dormant.** The brain holds **10** active
  `vocabulary` nodes total, and **0** match either purge strategy
  (`daemon_hooks.py:804-818`). There is no volume argument for a lossy shape.
- **For this purge's targets, the title *is* the content.** Strategy 2 selects single-word
  vocabulary nodes with `LENGTH(content) < 30`. Keeping `title` + `type` loses nothing
  meaningful about the deleted node, so the trace genuinely preserves what was erased.
- **No recall pollution.** `ca66f5bd`'s erase ruling exists because junk vocab *pollutes
  recall* — it matches everything in cosine and buries real results. A mutation trace is
  outside the recall path: not in `SAID_AND_DID_REF_TYPES`, not eagerly embedded, not in
  `CONVERSATIONAL_REF_TYPES`, reachable only by explicit `ref_type` query. Preserving the
  record there cannot reintroduce the problem the erasure solved.
- **Nothing prunes `trace_events`**, so the record is durable by default — which is what
  "saving somewhere" requires for an irreversible operation.

**If a future hard-delete path targets content-heavy nodes**, the body belongs in part 1's
recorder (a payload kind with the pointer in the trace row), not in trace metadata — that
preserves property 3 (skeletal rows). One `PAYLOAD_KIND_EXT` entry, decided then. Not built
now for a dormant path.

**Still not bundled:** whether the purge should hard-delete at all vs archive is a separate
decision with its own evidence. This design changes only its visibility.

---

## 8. Rulings inherited from the review — do not re-litigate

`node:5fcb662c` settled scope (manifest + additions + dead-code + touched-DAL unification);
manifest-vs-node-scope is closed. §1.1's twelve-site table, §1.3's pop-then-read diagnosis,
§2's five properties, the missing-beats-lying asymmetry, the one-emitter architecture, and
the ref_id contract were all independently confirmed. The MCP gate is exempt: nothing
touches `BATCH_OP_SPECS`, so `eval/mcp_batch_probe.py` and `eval/mcp_schema_gate.py` are
correctly not required.
