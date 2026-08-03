# Mutation-Trace Emitter — Design (Traces Part 2)

**Status:** design approved in-session 2026-08-03 (Tom: manifest scope + additions +
dead-code + touched-DAL unification). Implementation NOT started.
**Charter:** brain node `98031b8e`; `docs/TRACE-MODES-DESIGN.md` §"Part 2 charter".
**Part 1 (sibling):** the payload recorder (agent-I/O capture) — same disease, read
side. This doc is the write side: graph-mutation trace emission.
**Evidence base:** four-agent architecture review pass, 2026-08-03 (timing map,
manifest inventory, consumer map, attribution/boundary). Findings condensed into
§1/§7; file:line refs verified against this tree.

---

## 1. Problem — verified inventory

### 1.1 Twelve hand-rolled emit sites, four disciplines

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

¹ Edge case: if the FTS upsert raises inside `revise`, the KV flush at
`brain_remember.py:1448` is skipped and the trace commits against an open
deferred transaction (the leak `_handle_brain_batch`'s entry-flush exists to
clean up).

**Root structural fact:** traces commit **unconditionally and immediately** on
`brain_logs.db` — `TraceDAL` is bound to `logs_conn` (`brain.py:240`), nothing
anywhere sets `logs_conn.in_batch`, so `commit_unless_batched` at
`dal_logs.py:643` always commits. Zero shared transaction state with
`brain.db`. Any emit fired while `brain.conn` is inside the batch envelope
(`BEGIN IMMEDIATE` at `dispatch_write.py:871` → commit `:1029`) produces a
durable trace for a graph write that can still roll back (`:1037`). The comment
at `dispatch_write.py:1025-1028` claiming traces are emitted "never before"
the commit is true for exactly one of the twelve sites (#8).

Secondary orphan window: `absorb`'s savepoint unwind
(`brain_remember.py:668` `ROLLBACK TO absorb_sp`) undoes the merge while
`archive_node`'s inline trace (emitted at `:432` before the unwind decision)
stays committed — orphanable even without a whole-batch rollback.

### 1.2 Creates emit nothing

No `node_created` ref_type exists anywhere (finding `19b56d44`). A failed
encode run's created nodes leave no id record → the partial-run catalog gap
(`30cf1bce`, first consumer, §10).

### 1.3 Attribution holes (the d857e84d class)

- The original d857e84d split (revise/connect batch sub-ops unattributed) is
  **fixed** and pinned by `tests/test_mcp_roundtrip.py:614-644`.
- **Live successor bug — pop-then-read:** `_handle_remember` pops both identity
  keys via `_pop_session_ctx` at `dispatch_write.py:383` (it mutates args in
  place), then reads `sess = caller_session(args)` at `:437` → always `''`.
  Same in `_handle_remember_batch` (`:460` → `:522`). **Every co_anchored and
  connect_to edge trace emitted from the remember handlers carries
  `session_id=''`** — standalone and batched alike.
- `archive_node`'s inline trace: `session_id=''`, hardcoded off-chain
  `chain_id='archive-{id8}'` (`brain_remember.py:433`).
- Encoder-originated writes never carry session (by design — identity rides
  `encoding_source`; `apply_encoder_attribution` setdefaults
  `encoding_source`/`chain_id` only, `scales/s2/base.py:48-78`).

### 1.4 Trace-invisible mutations

- **absorb**: survivor `revise()` at `brain_remember.py:620` emits no trace
  (node_revised is dispatch-layer only); migrated edges (`:562-580`) discard
  every `add_relation` result. A merge is invisible except the unattributed
  side-chain archive trace.
- **revise's co_anchored auto-edges** (`brain_remember.py:1546-1560`): results
  discarded at the source — structurally untraceable today.
- **Healer in production bypasses dispatch entirely**:
  `healer_encoder.py:340-344` falls back to `self.brain.revise(...)` because
  `self.dispatch is None` (coordinator builds `Healer(brain)`; HealerEncoder
  never calls `_make_encoder_dispatch`). Every field-fill: no trace, no
  project stamp.
- **Community/consolidation direct archives**: `community.py:253-256`,
  `consolidation_decoder.py:165-168`, `:234-237` call `brain.archive_node`
  directly (side-chain trace only).
- **Bulk edge sweeps**: `GraphDAL.archive_dangling_edges`
  (`dal_graph.py:438-483`, healer invariant restorer) and
  `GraphDAL.decay_edges` (`dal_graph.py:955-1015`, idle maintenance decay +
  prune) soft-archive real edges with no trace.
- **Junk-vocabulary purge** (`daemon_hooks.py:802-828`): idle maintenance
  **hard-deletes** nodes via `delete_node_cascade` — irreversible, no trace,
  no archive. Highest-severity blind spot found.

### 1.5 The manifest half-exists already

`_apply_connect_to` already returns manifest-shaped edge rows
(`{src_id, target_id, relation, edge_id, deltas}`, `brain_remember.py:2042-2045`);
`add_relation`/`connect_typed` return `{edge_id, created, revived_from_archive,
updated, deltas, warnings}` (`dal_graph.py:1118-1125`); `revise` returns
`deltas`/`warnings`. The handlers **consume these locally to feed the
hand-rolled emits and throw them away** (`connect` returns `{"connected": True}`
and drops everything, `dispatch_write.py:1129`). The plan makes them return
the data instead of consuming it.

---

## 2. Ruled design properties

1. **POST-COMMIT emission**, stated asymmetry: a missing trace is recoverable
   from the DB; an orphaned trace lies about the graph. Prefer missing over
   lying. The miss window (crash between graph commit and emit) is accepted
   and loud-wrapped.
2. **Per-batch rollups, not per-op rows**: one `append_batch` (one logs-DB
   commit) per dispatched command, however many mutations it carried.
3. **Skeletal rows forever** (normal mode) — same row shapes as today; heavy
   payloads belong to the recorder (Part 1), not the emitter.
4. **Never slows the write path**: emission happens after durability, off the
   recall path entirely; a trace failure logs loudly and never raises into the
   caller.
5. **Attribution stamped once at the chokepoint** — kills the pop-then-read
   class (§1.3) by resolving session/chain **before** any handler mutates args.

---

## 3. The mutation manifest contract

Every write handler returns, at top level (sibling of `result`, invisible to
the agent — same treatment `affected` gets today):

```python
"mutations": {
    "nodes": {
        "created":  [ {"node_id": str, "type": str, "title": str} ],
        "revised":  [ {"node_id": str, "reason": str,
                       "deltas": [...], "warnings": [...]} ],
        "archived": [ {"node_id": str, "archived_by": str, "reason": str,
                       "edge_ids": [str]} ],          # edges archived with it
        "deleted":  [ {"node_id": str, "deleted_by": str, "reason": str} ],
    },
    "edges": [
        # one self-describing row per edge touched
        {"edge_id": str, "source_id": str, "target_id": str,
         "relation": str, "reason": str,
         "deltas": [...], "warnings": [...]}
    ],
}
```

Rules:

- **`affected` is derived, not duplicated.** The runner
  (`scales/runner.py:629`) and `_accumulate_touched`
  (`daemon_server.py:802-839`) keep reading id-lists; the dispatch layer
  derives `affected = {k: [r["node_id"] for r in mutations.nodes[k]] ...}` so
  neither consumer re-plumbs. (Whether `affected` remains materialized on the
  return or becomes a helper over `mutations` is an implementation choice;
  the two consumers' read shape is the contract.)
- **Edge rows are complete** — every field `build_edge_revise_metadata`
  needs (`trace_contract.py:1245`), so the emitter never re-derives endpoints
  or re-queries the graph.
- **Deltas are observed, not asserted.** A row's `deltas` come from the DAL's
  actual before/after (this outlaws disconnect's current fabricated
  `0→1` flip — §5).
- **brain_batch accumulates.** Sub-handlers' manifests are popped off sub-op
  results exactly the way `affected` is today (`dispatch_write.py:926` et al.)
  so nothing leaks into the agent-facing `results[]`; the batch returns one
  merged manifest; the deferred connect_to pass appends its rows.
- **Empty manifest = no trace** — idempotent re-connects (empty deltas) stay
  silent, as today (`dispatch_write.py:338-339` behavior preserved).

## 4. The emitter

One module: `servers/mutation_emitter.py`. One entry point:

```python
def emit_mutation_traces(brain, cmd, manifest, *, session_id, chain_id,
                         encoding_source) -> None
```

**Placement — the timing theorem.** The dispatcher never commits; every write
path is fully durable by the time its handler returns (standalone ops commit
inside the DAL mid-handler; brain_batch commits at `dispatch_write.py:1029`
before returning; a batch rollback re-raises out of the handler so the emitter
never runs). Therefore the emitter fires **in the dispatch layer, after
`entry.handler(...)` returns ok** — post-commit on every path by construction.
No per-site timing reasoning survives the migration. Concretely: a thin wrapper
around write-command handler invocation in `daemon_dispatch`/`daemon_server`'s
dispatch path (and the encoder dispatch closure, which calls the same
handlers — `scales/s2/base.py:281-289` — gets the same wrapper).

**Attribution.** `session_id = caller_session(args)` and
`chain_id = args.get('chain_id','')` resolved at the chokepoint **before** the
handler runs (handlers pop these keys). Scale/chain fallback logic
(`_infer_scale_and_chain`) moves into the emitter — one copy.

**Trace mapping.** From one manifest, the emitter builds:

| Manifest slot | ref_type | ref_id | metadata builder |
|---|---|---|---|
| nodes.created | `node_created` (NEW) | node_id | `build_node_created_metadata` (NEW: node_id, type, title, encoding_source) |
| nodes.revised | `node_revised` | node_id | `build_revise_metadata` (unchanged) |
| nodes.archived | `node_archived` (NEW) | node_id | `build_node_archived_metadata` (NEW: node_id, archived_by, reason, edge_ids) |
| nodes.deleted | `node_deleted` (NEW) | node_id | `build_node_deleted_metadata` (NEW) |
| edges[] | `edge_relation_revised` | `{edge_id}:{relation}` | `build_edge_revise_metadata` (unchanged) |

All rows for one command go out in **one `append_batch`**.

`node_archived` replaces site #12: `archive_node`'s inline trace is deleted;
archives land on the **caller's real chain** with real attribution instead of
the private `archive-{id8}` chain. (The dashboard's `-archive`/`archive-`
chain-kind branches in `trace_friendly.js:24-43` degrade gracefully — legacy
rows keep rendering; new rows render under their real chain.)

**Contract registration (required, or traces silently vanish):**
`validate_trace_event` raises on unregistered triples and the emitter's
loud-wrap would swallow it into `_log_error` — so registration is step one:

- `REF_TYPES` (`trace_contract.py:61-150`): add `node_created`,
  `node_archived`, `node_deleted` at `("s0","delta")`, `("s1","delta")`,
  `("s2","delta")` (same three scales the existing pair occupies).
- **Not** in `LLM_ENCODER_DELTA_REF_TYPES` (`:1065`) — no token telemetry.
- **Not** in eager-embed lists (`embed_queue.py:45-46`) — skeletal rows.
- S2 idle-gate: `s2/base.py:294-309` `_last_run_timestamp` counts any delta on
  a unit chain as a completed run. New ref_types must not re-arm it: exclude
  them the same way `RESIDUE_REF_TYPES` rows are excluded (add an
  `EMITTER_REF_TYPES` tuple to trace_contract and reference it there and in
  `dashboard/queries/s2_runs.py` — which currently hand-mirrors the residue
  list; unify that mirror while touching it).
- **Enforcement upgrade:** register `REVISE_METADATA_SHAPE`,
  `EDGE_REVISE_METADATA_SHAPE`, and the three new shapes in
  `METADATA_REQUIRED_BY_REF_TYPE` (`trace_contract.py:1015-1023`). Neither
  existing ref_type is enforced today — declared shapes, no-op validation.
  The d2eeab4 precedent (validate at the DAL chokepoint) already carries it.
- Add `servers/mutation_emitter.py` to `TRACE_WRITER_FILES` in
  `tests/test_trace_contract_sync.py:19-37` — `dispatch_write.py` was never
  scanned; the emitter must not inherit that blind spot.

**Absorbed check — per-op batch failures loud at the chokepoint.** Part 1's
step 2 shipped per-op `ok=False` batch-failure loud-logging at the *encoder*
dispatch only (`scales/s2/base.py::_log_failed_batch_ops`, landed bb7ed5a —
sibling stream's work). The emitter sees every caller's per-op results at the
chokepoint, so the check moves here: failed batch sub-ops loud-log for ALL
callers (MCP included), and the encoder-side special case is retired in the
flip (step 3).

**Failure semantics.** The emitter body is loud-wrapped as a unit
(`_log_error('mutation_trace_emit', ...)`); it never raises into the dispatch
return. A trace miss is recoverable (the graph is the truth); the error row is
the alarm. Registration errors (`ValueError` from `validate_trace_event`) are
programmer bugs and must fail tests, not production — the contract-sync test
covers every (ref_type × scale) the emitter can produce.

**Performance.** One `append_batch` per write command (≤ today's cost — today
a remember with edges does up to three separate logs commits). No reads added
to any hot path. Manifest assembly is O(mutations) dict work on data the
handlers already hold.

## 5. Return plumbing + touched-DAL unification

Changes at the brain/DAL layer so manifests carry observed truth. All are
additive-first (nothing consumes them until the flip, §8):

1. **`GraphDAL.remove_relation`** (`dal_graph.py:1253-1280`) — returns
   `{edge_id, relation, flipped: bool, deltas}` from the actual UPDATE
   (rowcount-checked). Kills `_op_disconnect`'s hand-built delta and its
   separate `get_edge_id` lookup; a disconnect of an already-archived relation
   stops emitting a false `0→1` flip.
2. **`GraphDAL.delete_node_edges`** (`dal_graph.py:877-891`) — returns the
   archived edge ids (SELECT-then-UPDATE; off hot path), feeding
   `nodes.archived[].edge_ids`.
3. **Unify the bulk sweeps on one primitive** —
   `GraphDAL.bulk_archive_relations(*, where..., archived_by) -> {count, edge_ids}`:
   `archive_dangling_edges` and `decay_edges`' prune arm become callers (their
   selection logic stays; the flip+return is shared). This is the
   touched-DAL unification Tom asked for: three copies of "soft-archive a set
   of edge_relations" (`remove_relation`, `archive_dangling_edges`,
   `decay_edges`) converge on one flip discipline with one return shape.
4. **`add_relation` revive branch honesty** (`dal_graph.py:1173-1191`) — it
   reads `old_desc/old_weight/old_es` and then reports `old=None`; return the
   real old values in the deltas. (Touched-function fix, no new callers.)
5. **`brain.revise` returns `co_anchored_made`** (mirror of `remember`'s
   `:1105-1113` collection) — closes the structurally-untraceable edge path
   (`:1546-1560`).
6. **`brain.revise_edge`** echoes `source_id`/`target_id`/`warnings`
   (handler already resolves endpoints; pass through,
   `brain_connections.py:254-255`). Note for the spec's test surface: its
   rename+update path is two independent commits (`dal_graph.py:1280` +
   `:1197`) — non-atomic; wrap in the `in_batch` save/restore envelope while
   touching it.
7. **`brain.absorb`** returns survivor-revise `deltas` (today discarded at
   `brain_remember.py:625-630`) and migrated-edge rows (today discarded at
   `:579`).
8. **`brain.archive_node`** returns its archived edge ids (via #2) and the
   `absorbed_into` edge row (`:398-407`); its inline trace emit (`:431-447`)
   is **deleted** in the flip.
9. **`brain.delete_node_cascade`** returns `{node_id, tables_hit}` so hard
   deletes can be manifested (`node_deleted`).

## 6. Coverage: what routes through the emitter

**In (v1):** all 9 write commands + 6 batch ops; S1 Scribe + community/
consolidation encoders (already routed via `_make_encoder_dispatch` →
COMMAND_TABLE handlers); **Healer re-routed** (prerequisite, §8 step 0);
community/consolidation direct `archive_node` calls re-routed through the
dispatch archive path; idle-maintenance junk purge (via `node_deleted`
manifest from `delete_node_cascade`); bulk sweeps (`decay_edges`,
`archive_dangling_edges`) emit **one rollup trace per pass** (count + edge ids
in metadata, `encoding_source='decay_pruned'`/`'s2:healer'`) — after this,
zero edge-lifecycle changes in the system are untraced.

**Out (deliberate, documented):**

| Path | Why out |
|---|---|
| `enrich` / `store_enrichments` | Derived-data refresh (`node_enrichments`), no node/edge lifecycle; already `NON_ATTRIBUTED_WRITES` (`scales/dispatch.py:96`) |
| Hebbian `co_accessed` edge creation + access marks (`recall_write_queue.py`) | Infrastructure statistics, no judgment behind it, high volume; deliberately off-path |
| Vector/summary/temporal backfills, FTS rebuilds, schema migrations | Derived geometry / one-shot infra |
| Logs-DB janitorial orphan deletes (`dal_logs.py:182-238`) | Invariant restoration on rows whose nodes are already gone |
| `seed_pack` boot writes | Once, on an empty brain, provenance already `anchor:seed`; traces add nothing recall would use |

**Separate policy flag for Tom (not bundled):** should the junk-vocabulary
purge hard-delete at all, vs archive? v1 makes it *visible* (`node_deleted`);
changing its behavior is a different decision.

## 7. Read-side preservation contract

What the emitter must keep bit-compatible (verified consumer map):

- **ref_id formats**: `node_id` for node events; `{edge_id}:{relation}` for
  edge events. Pinned by `scripts/trace_node_reconstruct_20260612.py:109-122`,
  `trace_friendly.js:161`, `tests/test_revise_unified.py:55`,
  `tests/test_edge_mutation_unified.py:81`. No reader splits the composite;
  equality-only lookups (`dal_logs.py:842-844`).
- **Metadata keys**: `node_id/reason/encoding_source/deltas/warnings` and
  `edge_id/source_id/target_id/relation/...` — `trace_detail.js:33-45` shape-
  sniffs on `deltas` + (`node_id`|`edge_id`); `_renderRevise` reads the rest.
- **Chain conventions**: caller-provided chain wins; date-fallback
  `{scale}-{YYYYMMDD}-revise` preserved (only reader:
  `trace_friendly.js:29`, presentation-only). `_stop_of`
  (`trace_links.py:70-79`) yields None on `-revise` chains — unchanged.
- **Event envelope**: `event_type='delta'`, scales s0/s1/s2 — unchanged for
  existing types.
- **Dashboard S2 runs query** (`dashboard/queries/s2_runs.py:31-61`) pulls all
  non-residue deltas on unit chains and mis-enriches revise rows today
  (pre-existing bug, same class the encoding view fixed at
  `encoding.py:154-166`). The `EMITTER_REF_TYPES` exclusion (§4) fixes it for
  new types and the existing pair while touching that file.
- **recall_episodes / embeds**: mutation traces stay out of
  `CONVERSATIONAL_REF_TYPES` and eager-embed — reachable by explicit
  `ref_type=` query only, as today.
- **Layering law**: all new reads go through `brain.query_traces` (correction
  `d1329a9f`). The `S1-ENCODER-ANCHOR-TOUCH.md:100-103` proposal to read via
  `brain._trace_dal.get_by_session_window` is a doc-level violation — do not
  inherit it; the first consumer reads via the `GATHER_STREAMS` registry
  (§10).

## 8. Migration plan (ordered; each step ships green alone)

- **Step 0 — Healer routing (prerequisite for honest coverage).**
  `HealerEncoder` gets the unit's `_make_encoder_dispatch` closure like every
  other encoder (`healer_encoder.py:340-344` fallback dies; coordinator wiring
  in `s2/healer.py:73`). Community/consolidation direct `archive_node` calls
  route through the dispatch archive path. Test: healer field-fills produce
  `node_revised` rows on the unit chain with project stamping.
- **Step 1 — Return plumbing** (§5, all nine items). Additive; existing emits
  untouched; unit tests on the new return shapes.
- **Step 2 — Emitter module + contract registration** (§4): ref_types,
  metadata shapes + enforcement entries, `EMITTER_REF_TYPES`,
  contract-sync scan list, the emitter itself with tests against a fake
  manifest. Nothing calls it in production yet.
- **Step 3 — The atomic flip (one commit).** Handlers return manifests;
  dispatch layer derives `affected` and calls the emitter post-return; ALL
  twelve emit sites + the three `_emit_*` helpers + `_infer_scale_and_chain`
  deleted from their current homes; brain_batch accumulates manifests;
  junk purge + bulk sweeps wired. Transition tests are the net: every pinned
  emission behavior (§7 test list) must pass unchanged, plus new-type tests.
- **Step 4 — Dead code** (§9, after per-item verify).
- **Step 5 — First consumer** (§10) — separate session, parked until 0-3 land.

**Gates:** full-suite tier (SQL/DAL/dispatch blast radius — not `-k` filtered),
own daemon-restart gate, `eval/mcp_batch_probe.py` + `eval/mcp_schema_gate.py`
NOT required (no MCP schema/description change — manifests are top-level,
agent-invisible; re-check this claim at flip time if any tool schema text
changes).

## 9. Dead-code appendix (each verified, per the Category-B rule)

| Candidate | Verdict | Evidence |
|---|---|---|
| `servers/scales/s2/archive/reclassify.py` | **Delete.** Retired aspect-migration path (memory: aspect live path = aspects_v1.json → AspectRegistry → AspectIntegration; migration RETIRED). Not in coordinator's unit list. | `reclassify.py:91-93` direct `rename_relation`, dormant |
| `Brain.check_integrity` auto-fix arm (`brain_assembly.py:446-450`) | **Delete** (zero callers; archive-with-`hook:integrity` path dead). Verify no external script calls it at delete time. | agent-verified zero callers |
| `Brain.set_personal` (`brain_remember.py:2278-2312`) | **Delete, with a flag**: it is the only code path that locks a node post-creation (`revise` IMMUTABLE includes `locked`, `brain_remember.py:1308`) — but it is unreachable (no dispatch command, zero callers), so the *capability* is already absent in practice. If post-creation locking is wanted, build it as a deliberate small feature (trace-emitting, through dispatch), not by reviving this personal-flag-era method. | zero callers; lock-door analysis this session |

## 10. First consumer (parked): partial-run catalog gap

Design `5efe5e02`, finding `30cf1bce`. After the emitter lands:

- `GATHER_STREAMS` (`scales/s1/trace_links.py:340-345`) gains one line:
  `'created': ('node_created', 's0'/'s1', ...)` — reads via
  `brain.query_traces` like every stream.
- `session_node_ids` (`:317-335`) unions the created-ids into the `encoded`
  set fed to `build_node_catalog`'s `extra_ids`.
- **Invariants (Tom probed explicitly, principle `c7d52ad0`):**
  `NodeDAL.get_bulk` hydration stays the truth filter — a rolled-back id
  resolves to nothing and drops silently; coverage (`encoded=` marks) stays
  gated on the `encoding_run` success receipt — catalog visibility never
  suppresses re-encoding. "A failure can neither hide turns nor conjure
  nodes."

## 11. Test surface

- **Contract:** emitter is the only mutation-trace writer (extend the
  grep-pin in `test_trace_contract_sync` to `dispatch_write.py` +
  `brain_remember.py` — no `_trace_dal.append` for mutation ref_types outside
  `mutation_emitter.py`); ref_type triples registered; metadata shapes
  enforced; `EMITTER_REF_TYPES` consistency (trace_contract ↔ s2 gate ↔
  dashboard query).
- **Component:** manifest derivation per handler (incl. empty-manifest
  silence); `remove_relation`/`bulk_archive_relations` observed deltas;
  revive-branch real old values; emitter row mapping + single append_batch;
  attribution stamping (session/chain resolved pre-handler — regression test
  for the pop-then-read bug).
- **Transition:** every §7-pinned reader shape against emitter-produced rows;
  brain_batch rollback → zero mutation traces (the orphan test — assert
  traces for a rolled-back batch do NOT exist); absorb savepoint unwind →
  zero traces for the unwound merge.
- **Cycle:** batch of mixed ops → one manifest → traces → `session_node_ids`
  union (post-consumer) sees created ids; failed batch → catalog unchanged.

## 12. Part 3 charter: the analysis substrate (own thread — chartered here, not specced)

Ruled by Tom 2026-08-03. The read-path + config half of trace observability —
sibling to part 1 (recorder: agent I/O) and part 2 (emitter: graph mutations).
Same pattern as part 2's chartering: inventory and rulings only; the spec is
its own design session. Scope additions here re-open nothing above.

**Gap A — LAF recall capture (the headline).** The LAF engine scores the ENTIRE
embedded corpus (~8k nodes × ~8 lanes) per pull (`recall_laf.py:841` —
"every node with ≥1 embedding view"); the recall O trace keeps a pipe-string
of 25 finals (`surface.py:775`) — the per-lane decomposition (`_laf_fields`)
is computed, attached in memory, and dropped. Post-hoc "why did X rank #14"
and production A/B of gain configs are impossible today (finding 5d9f1fd2).
Ruled design:

- **Trace row (bounded, always):** structured top-25 candidate shape (dict,
  not pipe-string — `RECALL_CANDIDATE_SHAPE` in the contract), plus stamps the
  O trace lacks: `recall_variant` (laf_v1 vs champion) and the `recall_laf`
  interaction version (the gain config that scored this pull), plus the
  payload pointer.
- **Payload file (recorder kind `recall_fields`, columnar):** one id array +
  parallel columns per lane — no id repetition, jq/pandas-native:
  `{query, gains, lane_cutoffs, node_ids[], final[], lanes{maxsim[], ...}}`.
- **Scope: the FIELD, not a top-K window.** Tom's correction 62b04f12 retired
  top-K cuts for LAF ("top 25 is very irrelevant now… 25 doesn't capture
  layering complexity") and the running-field architecture 87a6dae9 names
  cache-only-top-K as the anti-pattern — divisive normalization, reach, and
  inhibition all live in the below-pool mass, and the P2 walker trains on
  field-level data. Normal scope per pull: (a) the **full final field** — one
  activation per embedded node (~8k floats, ~30-50KB compact), every node's
  final score always answerable; (b) **per-lane decomposition for the
  ~500-node pre-floor field** (the measurement window 62b04f12 set) + each
  lane's cutoff at the field boundary — below it, absence is quantified.
  Debug scope: full dense per-lane over the whole corpus.
- **Cost (corrected shapes):** normal ~60-75KB/pull → ~20MB/day at ~300
  pulls, ~125MB at 7d / ~250MB at 14d retention. Debug full-dense
  ~300-450KB/pull → tuning windows only. Assembly is from `_laf_fields` +
  the field already in memory — single-digit ms, best-effort write, never on
  the scoring path (performance charter applies). Encoding (terse JSON
  columns vs binary sidecar) is a spec-session decision — recorder owner
  reviews the kind/format against the contract; one kind, one format.
- **Mechanism (already built — part 1; do not re-derive):** new kind = one
  `PAYLOAD_KIND_EXT` entry in trace_contract.py (gate config derives
  automatically, config overlay covers pre-existing brains). Writer:
  `brain.record_payload(chain_id, kind, content, seq=)` — call sites hold
  zero path/gate knowledge. Record at the surface TAIL (post-selection,
  never inside scoring); returned pointer goes in the trace row metadata —
  the `judge` kind shipped exactly this pattern, copy it. Reader rules
  (hard-won, part-1 review): pointer-in-trace-row, never glob payloads on
  polled endpoints (glob only on card expand); attempt ordinals sort with
  `payload_sort_key`, never `sorted()[-1]`; retention is per-kind config —
  if the P2 flywheel needs fields longer than the default 14d, raise the
  kind's retention or have the walker ingest before prune.
- **Separate switch (Tom, explicit):** `recall_fields` is its OWN per-kind
  gate block — `{enabled, scope: topk|full, k, retention_days}` — hot-
  flippable via the gate config without touching failed_run/round_payload.
  Doubly separated from LAF itself: `BRAIN_RECALL_VARIANT` gates whether LAF
  runs; this kind gates whether it's recorded.

**Gap B — runtime fingerprint + brain identity.** No trace today records which
code/config/brain produced it (boot heartbeat is summary-only, `brain.py:1020`;
schema migrations run silently). Ruled design: one `runtime_fingerprint` trace
per daemon boot carrying `{git_commit, git_dirty, branch, schema_version,
embedder, env_variants, active_interaction_versions{}, brain_id, instance_id,
arm_label, db_dir}`. Everything after joins by time. Identity scheme:
- `brain_id` — UUID minted once at brain creation (`brain_meta`; one backfill
  migration row). Copies INHERIT it — it's the lineage id.
- `instance_id` + `arm_label` — minted fresh at copy time
  (`IsolatedBrain.copy()` stamps both; harness names the arm). Cross-arm A/B
  becomes `GROUP BY arm`, not a file-path convention.
- git unavailable → `"unknown"` + loud log; never blocks boot.

**Gap C — config_changed trace (ruled: easy, do it).** `register_interaction` /
`set_interaction_active` / `set_config` are the most behavior-changing writes
in the system and emit nothing. One `config_changed` trace at those three
handlers puts every behavior flip on the timeline next to its effects — the
missing join for before/after A/B analysis.

**Gap D — engine-side folds (LAF-scoped, surfaced by the part-1 review).**
(a) `recall_laf` K-store reads are TTL-cached up to 60s with no invalidation —
a gain flip takes effect up to a minute after its `config_changed` timestamp,
skewing before/after joins exactly when Tom starts flipping gains. The
invalidation-hook pattern now exists at `brain.py:741` (built for
`trace_recording`); generalizing it is ~5 lines and belongs with Gap C.
(b) The `_laf_fields` forward-feed has sat unrouted since the 2026-07-15
handoff (64c506c4 deferred list) — Gap A's capture IS its routing; the spec
should close that deferred item explicitly, not leave a second half-wired
path.

**Ownership (Tom consolidated):** this thread owns the part-3 design; the
part-1 stream owns the capture engine and reviews the kind/format against the
recorder contract before implementation.

**Sequencing:** design-ready now; implementation in a fresh thread on top of
main ≥ bb7ed5a (part 1 step 2 landed — `record_round_fn` seam +
`build_round_payload` are the capture substrate `recall_fields` plugs into).
No dependency on part 2's emitter; Gap A's call site is `surface.py` (now
stable). New ref_types (`runtime_fingerprint`, `config_changed`) follow the
same REF_TYPES registration law as §4.

## 13. Rollout

`servers/*` only → daemon restart deploys it (no `./redeploy.sh` needed unless
MCP schema text changes — it shouldn't). Backup `brain_logs.db` schema
untouched (no migration — new ref_types are rows, not columns). Merge to main
per repo law; this branch rebases over the Part 1 step-2 migration stream
(1b8f05ca) if it lands first — no shared files beyond additive
`trace_contract.py` sections (their scope: capture-side builders; ours:
mutation shapes + `EMITTER_REF_TYPES`).
