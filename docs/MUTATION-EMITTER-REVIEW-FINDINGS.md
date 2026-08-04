# Mutation-Emitter Spec — Review Findings (2026-08-03)

**What this is.** `docs/MUTATION-EMITTER-DESIGN.md` (commit 2b7903c) was reviewed by five
independent scoped agents plus a full-document pass. This file is the complete findings
list. **The design is sound; the work order is not.** Rewrite the doc from this file.

**Read this first, then the design doc — not the reverse.** Sections of the doc marked
sound below should not be re-litigated; everything else changes.

> **ALL line numbers in the design doc AND in this file were resolved against a tree at
> `a25ca9b`, which is BEHIND main (`bb7ed5a`).** The branch was never rebased. Every
> citation touching `servers/scales/s2/base.py`, `servers/trace_contract.py`,
> `servers/scales/runner.py`, and `servers/scales/s1/surface.py` must be re-resolved after
> the rebase. Known drift: `s2/base.py` `_make_encoder_dispatch` 216→251, handler call
> 289→324, `_last_run_timestamp` 294→331.

---

## A. Production-harmful — the spec would cause damage if followed literally

**A1 — Registering the metadata shapes as instructed is a module-level `NameError` that
prevents the daemon from booting.**
Doc §4 says to add `REVISE_METADATA_SHAPE` / `EDGE_REVISE_METADATA_SHAPE` to
`METADATA_REQUIRED_BY_REF_TYPE` at `servers/trace_contract.py:1015`. Those shapes are
*defined* at `:1189` and `:1233` — after the dict. `trace_contract` is imported by
`dal_logs`, so the import fails and the daemon does not start on the next
`launchctl kickstart`. **Verified directly.**
*Correction:* move the dict below the shape definitions (or the shapes above the dict) and
state the ordering constraint in the step that makes the edit.

**A2 — §9 would delete live boot behavior under a function name that does not exist.**
There is no `Brain.check_integrity`. The code at `brain_assembly.py:446-450` lives inside
`Brain.health_check(session_id='boot', auto_fix=True)`, which runs at **every session
boot** (`servers/brain_voice.py:351`, inside `render_boot_v2`) and via the `health_check`
dispatch command (`servers/dispatch_ops.py:29`). It archives `context` nodes older than 14
days with `archived_by='hook:integrity'`. **Verified directly.**
*Correction:* strike the entry. The behavior is live; if it should change, that is a
separate decision with its own evidence.
*Process note:* the invented name propagated — two reviewers grepped it, found nothing, and
reached opposite wrong verdicts ("entirely dead, delete the method" vs "live, leave it").
A fabricated symbol is worse than a vague one.

---

## B. Design-changing — these alter rulings, not wording

**B1 — Emitter placement must be a registry-level function, not a wrapper at two call
sites.** (Two reviewers converged on this independently.)
The doc places the emitter "around write-command handler invocation in
`daemon_dispatch`/`daemon_server`'s dispatch path." `daemon_dispatch.py` contains no
invocation site at all (it is a pure registry), and there are ~23 `entry.handler(...)` /
direct `_handle_*` call sites — including `tests/isolated_brain.py:218` (the dispatch
surface for ~50 eval scripts) and 54 direct `_handle_*` calls across 5 test files. As
specced, all eval and most tests silently stop producing mutation traces, and the §7-pinned
tests (`test_mcp_roundtrip.py:614-644`, `test_revise_unified.py:55`,
`test_edge_mutation_unified.py:81`) fail — making §11's "pins pass unchanged" unsatisfiable.
*Correction:* add `dispatch_command(brain, cmd, args, graph_changes)` to
`servers/daemon_dispatch.py` (resolve entry → `check_unknown_keys` → resolve session/chain →
handler → emit) and route `daemon_server._dispatch`, `s2/base._make_encoder_dispatch`, and
`tests/isolated_brain.dispatch` through it. Declare direct `_handle_*` imports unsupported
post-flip and list the 5 test files as migration scope.
**Bonus: this makes the migration incremental with no double-write.** A handler that does
not yet return a `mutations` key yields an empty manifest → the wrapper emits nothing → its
existing inline `_emit_*` still fires. Convert one handler per commit; delete `_emit_*` when
the last caller is gone. No reader ever sees two rows for one mutation.

**B2 — Manifest rows need per-row `encoding_source`; the single command-level value is an
attribution *and scale* regression.**
`encoding_source` is resolved per-row today (`dispatch_write.py:672`, `:754` via
`_resolve_archived_by`, `:1100` with its `'anchor'` default, `:1215-1216`), it is a
**required** key in both metadata shapes, and `_infer_scale_and_chain` (`:160-182`) derives
the trace **scale** from it. A `brain_batch` carrying `{op:'archive',
encoding_source:'s2:consolidation'}` with no top-level source currently emits at scale s2 on
the unit chain; under the spec it lands at s0 on `s0-YYYYMMDD-revise` with
`encoding_source:''`.
*Correction:* add `encoding_source` to every manifest row; command-level value becomes the
fallback; derive scale and chain per row. Preserve the asymmetric defaults (`'anchor'` for
edges, `''` for node revises).

**B3 — "One `append_batch` per command" is a robustness regression as specced.**
`TraceDAL.append_batch` validates *and* INSERTs in the same loop and `raise`s mid-loop
(`dal_logs.py:620-644`); the commit at `:643` never runs; `logs_conn` is default-isolation
and nothing sets `logs_conn.in_batch`, so the already-inserted prefix commits on the next
unrelated logs write — including the emitter's own `_log_error`. One bad row therefore loses
the rest of the command's traces *and* commits a partial set that lies by omission. Today's
one-append-per-mutation bounds damage to a single row. Worse, an unregistered ref_type
(e.g. a rebase dropping the `REF_TYPES` hunk) raises on the first row — `nodes.created` is
first — producing **zero mutation traces system-wide**, visible only as `_log_error` rows.
*Correction:* pre-flight `validate_trace_event` on every row before touching the DAL; have
`append_batch` roll back its pending inserts on raise; keep the per-command batching only
after both.

**B4 — The timing theorem is conditional, not "by construction"; make it enforced.**
Two reviewers falsified it independently. Handlers *can* return with `brain.conn`
mid-transaction — the proof is in-tree: `_handle_brain_batch`'s entry-flush guard
(`dispatch_write.py:847-861`) and its `brain_batch_stale_txn` error row exist precisely
because upstream writes leak open deferred transactions. Mechanisms: `MetadataKVDAL.set_many`
(`dal_metadata.py:168-183`) does not commit, so a `revise` whose changed fields are all
KV-resident is committed only by the TF-IDF (`brain_remember.py:1439`) or FTS (`:1448`)
call — both inside independent `try/except` arms that log and continue. And
`GraphDAL.archive_dangling_edges` (`dal_graph.py:438-483`) has **no commit call at all**,
so a rollup trace after it is orphanable by construction.
*Correction:* state the theorem conditionally, and enforce it — at the chokepoint,
`if brain.conn.in_transaction:` loud-log and skip (a missing trace, per the stated
asymmetry). Two lines; converts the emitter from an assumer of durability into a detector of
the leak class. Also fix `archive_dangling_edges` to commit (§5 item 3).
*Brain node `17234b02` has been revised to carry this qualification.*

**B5 — Step 1 is not additive. Three concrete breaks.**
(a) `archive_dangling_edges` int→dict crashes the Healer: `healer.py:81` does
`encode_result.get('fields_written', 0) + edges_archived` → `int + dict` `TypeError` every
cycle; `tests/test_edge_relations.py:435` also breaks.
(b) `delete_node_edges` int→ids changes an **agent-visible** field: it feeds
`brain_remember.py:385` → archive trace metadata (`:442`) and `archive_node`'s returned
`edges_deleted` (`:456`), which `_op_archive` splices into `results[]`. `3` becomes
`['e1','e2','e3']`. Pinned by `test_absorbed_into_edge.py`, `test_bg_writer.py:359`,
`test_surface_transitions.py:492`.
(c) §5 items 5/7/8 leak into agent-facing payloads: `_handle_revise` returns `brain.revise`'s
dict verbatim (`dispatch_write.py:610`), `_handle_revise_edge` likewise (`:1167`), and
`_op_archive`/`_op_absorb` splice `**r` into `results[]` (`:967`, `:973`). Adding
`co_anchored_made`, survivor deltas, and migrated-edge rows puts edge lists in the model's
tool result — the same class as the 6M-char incident.
*Correction:* reword step 1 from "additive" to "shape-preserving; each item lands with its
callers updated." Keep old scalars alongside new collections, and require the handler to pop
new keys in the same commit that adds them; name the strip points.

**B6 — §5 item 3's "three copies of one operation" are three different operations.**
They differ on four axes: embedding NULL (`remove_relation` yes, `archive_dangling_edges`
**no**, `decay_edges` yes), aggregate weight recompute (yes / **no** / yes),
`exempt_relations` (no / **yes, load-bearing** / no), commit gating
(`commit_unless_batched` / **none** / `commit_unless_batched`).
Two live hazards: standardizing on NULLing embeddings makes `archive_dangling_edges` start
destroying embeddings it does not touch today (a bulk `UPDATE` on `brain.db` fired by the
first idle cycle after deploy); and losing the exempt clause severs **455 active
`edge_relations`** whose endpoint node is archived — measured live, 100% of them
`absorbed_into`, i.e. every one alive only because of the exemption.
Also: the selection in `archive_dangling_edges` is a correlated subquery *inside* the UPDATE
(`:467-482`), so returning edge ids requires restructuring to SELECT-then-UPDATE-by-ids; and
item 3 unifies only decay's *prune* arm, leaving the weight-decay UPDATE (`:981-989`,
thousands of rows/pass) outside — yet weight deltas are exactly what
`edge_relation_revised` records.
*Correction:* shared flip primitive with explicit per-caller policy flags
(`null_embeddings`, `recompute_weight`, `exempt_relations`), or drop the unification. Name
`tests/test_absorbed_into_edge.py:166-190` as a required gate. Add the `brain.db` backup
requirement (`CLAUDE.md:470`) to any step running a bulk UPDATE.

**B7 — §5 item 4's "revive branch honesty" fix would destroy a documented signal.**
The contract states "empty `old` in a delta means the field was just created; populated =
update" (`trace_contract.py:1226-1227`). Filling in real old values on the revive branch
makes a revive indistinguishable from an update for every reader following that convention
(and for `_renderRevise`'s old/new table).
*Correction:* the actual missing signal is the `archived: 1→0` delta, which nothing emits.
Add that; leave `old=None` and amend the shape comment. §7 should list the convention.

**B8 — §5 item 2 is a data-integrity trap.**
`delete_node_edges` (`dal_graph.py:877-932`) already does SELECT-then-UPDATE, but the SELECT
(`:902-905`) returns **all** edge ids touching the node while the UPDATE (`:922-928`)
excludes `archived=0` misses **and `exempt_relations`**. Returning the SELECT list would
claim the deliberately-exempted `absorbed_into` redirect edge was archived. Also the flip is
on `(edge_id, relation)` rows, so a flat `edge_ids` list loses the `:relation` granularity
every other edge event carries, and those relation archives get no `edge_relation_revised`
row at all.
*Correction:* return only rows the UPDATE actually flipped, as `(edge_id, relation)` pairs.

**B9 — §8 step 0's healer routing: the test is impossible and the wiring splits the chain.**
(a) Project provenance **cannot** be stamped on a revise by design — `dispatch_write.py:554`
("a revise never moves it"), `_stamp_session_project` yields `None` for a sessionless caller,
and `revise` routes to `_strip` (`scales/dispatch.py:161-165`). §1.4's "no project stamp"
claim about the Healer is wrong; routing buys the trace only.
(b) The doc names `s2/healer.py:73` (hand the orchestrator's closure to the encoder). Every
other encoder builds its own inside `_make_dispatch` so the closure's `run_chain_id` comes
from the same instance that later calls `self.trace(...)`. `chain_id()` caches per instance
at **seconds resolution**, and `Healer`/`HealerEncoder` are separate instances → `node_revised`
rows on `s2-T1-healer`, `healer_generated` on `s2-T2-healer`: two chains for one pass, the
phantom-run-card class `CHAIN_AWARE_WRITES` exists to prevent.
(c) The Healer ignores the dispatch return (`healer_encoder.py:340-348` returns
`len(fields_to_write)` unconditionally), and `_handle_revise` returns `ok=False` rather than
raising — so a rejected revise would report `fields_written=N`.
*Correction:* build the closure inside `HealerEncoder._make_dispatch()`; test that
`node_revised` rows join the **same** `chain_id` as the pass's `healer_generated` delta; drop
the project claim; check the dispatch result.

**B10 — §8 step 0's community/consolidation archive reroute would silently break the orphan
heal.** There is no standalone `archive` command — `_op_archive` is batch-only
(`dispatch_write.py:685-689`), so routing means a `brain_batch` archive op. And the
consolidation encoder's closure carries `archive_guard=valid_archive_ids`
(`consolidation_encoder.py:229-233`), which **drops** any archive whose node is outside the
cluster set and logs `s2_consolidation_out_of_scope_archive` (`base.py:264-277`). All three
call sites target out-of-cluster nodes: `consolidation_decoder.py:165-168` (orphan
communities), `:234-237` (superseded handoffs), `community.py:253-256` (dead communities).
Reusing the guarded closure stops the heal working while reporting success.
*Correction:* specify an **unguarded** `_make_encoder_dispatch()` for these, and add a
step-0 test that the orphan-community heal still archives.

**B11 — §10's first consumer cannot work as written, twice over.**
`gather()` filters by `session_id` (`trace_links.py:369-374`) and unpacks a strict 2-tuple
(`:371`), while encoder-originated writes carry `session_id=''` by construction
(`apply_encoder_attribution` setdefaults only `encoding_source`/`chain_id`). And
`session_node_ids` parses ids via `_delta_ids(meta,'created','revised')` (`:331-333`) — a
per-node `node_created` row's metadata has no such list key. So the catalog-gap fix, the sole
justification for `node_created`, receives **zero rows** even after the emitter lands.
*Correction:* either stamp `session_id` on S1-scope encoder writes at the chokepoint (a real
§4 addition, not free) or key the created stream by the `s1e-` chain; and specify that ids
come from `ref_id`, not a metadata list. Either way `GATHER_STREAMS` changes shape — the
"one-line addition" claim is false. Also pick the scale: `node_created` is emitted at s0, s1
and s2, and the registry maps one `(ref_type, scale)` pair.

**B12 — §6's coverage claim is false, and the inventory misses two live edge-creating
paths.**
Missing from §1.4 and both §6 lists:
- **`emergent_bridge` on every `remember`** — `brain_remember.py:1148` →
  `_bridge_at_store_time` (`:2262-2276`, up to `bridge_max_per_remember`=2) →
  `brain_connections.py:337` `connect_typed(...)`, result discarded. **Every remember in the
  system creates up to two untraced edges**, and §3's remember manifest has no slot for them.
- **`community_member` back-fill every community cycle** —
  `dal_graph.py:579` inside `reconcile_community_membership`, results discarded; live caller
  `community_encoder.py:233`.
Additionally the absolute claim ("zero edge-lifecycle changes untraced") contradicts §6's own
Out table, which excludes Hebbian `co_accessed` **creation** (`recall_write_queue.py:480-492`
creates `edges` rows and revives archived relations).
*Correction:* replace with "every edge mutation reachable from a dispatch command or a named
sweep." Inventory both paths above with an explicit In/Out ruling —
`reconcile_community_membership` is cheap to route (it already runs inside an encoder with a
dispatch closure).

**B13 — Deleting `archive_node`'s inline emit loses information and drops archives out of the
embedded lens.** That row is `tool_result` at s0, and
`EAGER_TRACE_REF_TYPES = SAID_AND_DID_REF_TYPES` includes `tool_result`
(`trace_contract.py:198`, `embed_queue.py:46`) — so archives are eagerly embedded and
semantically reachable today (**1,203 rows on `archive-%` chains**, measured live).
`node_archived` would not be. The shape also drops `title`, `type`, and `vectors_deleted`.
§7's "mutation traces stay out of eager-embed, as today" is false for this path.
*Correction:* keep `title`/`type` on the archived and deleted manifest rows (the junk purge
already has titles in hand — `daemon_hooks.py:816`); rule explicitly whether `node_archived`
joins the said+did lens. For hard deletes the trace is the only surviving record, so a bare
id is unresolvable.

**B14 — New rows land on a chain literally named `-revise` and render as "Refined memories."**
`trace_friendly.js:29` classifies any chain ending `-revise` as kind `revise`, and `_revise()`
(`:157-163`) adds `ev.ref_id` to the memory count for **any** ref_type other than
`edge_relation_revised`. Creations, archives and hard deletes would render as *"Refined N
memories — Updated details on memories it already had,"* and the archive card becomes
legacy-only. Hook-origin mutations (junk purge, sweeps) have no chain convention at all —
`-revise` would be a lie and `_stop_of` yields `None` for them.
*Correction:* rename the fallback chain (e.g. `{scale}-{YYYYMMDD}-mutation`) or branch
`_revise` on ref_type; define a chain convention for hook-origin mutations; add the
chain-kind branch. List `trace_friendly._revise` in §7 as a reader whose meaning changes.

**B15 — The registration gate the doc relies on cannot see the emitter.**
`tests/test_trace_contract_sync.py:56-85` extracts triples via regexes requiring **literal**
quoted kwargs. A table-driven emitter with a variable ref_type yields **zero** triples, so
adding `mutation_emitter.py` to `TRACE_WRITER_FILES` passes vacuously. (Same reason
`_emit_edge_traces`' `append_batch` is invisible today.) Note `brain_remember.py` is already
in that list (`:22`); only `dispatch_write.py` is missing.
*Correction:* make the gate a runtime test — assert `validate_trace_event(scale,'delta',rt)`
for the cross-product of `EMITTER_REF_TYPES` × `('s0','s1','s2')`, plus a
`METADATA_REQUIRED_BY_REF_TYPE` presence check — and add a **negative** grep-pin (no
`_trace_dal.append` for mutation ref_types outside the emitter), modelled on
`tests/test_capture_grep_pin.py` (exists at `bb7ed5a`, not in this tree).

**B16 — The "enforcement upgrade" is a *warning* upgrade, and the warning misses the error
feed.** `validate_trace_metadata` failures route to `_warn_metadata_invalid`
(`dal_logs.py:578-587`) which never blocks and writes only to stderr → daemon.log (TraceDAL
has no Brain reference). Element shape inside `deltas` is unchecked, and the builders default
optional fields to `''` — so a manifest missing `source_id`/`target_id` passes validation and
silently breaks §7's "reconstructable from the trace alone."
*Correction:* reword §4 honestly; have the emitter itself assert row completeness and report
via `brain._log_error` (the channel that reaches the error feed) — the one place a raise is
safe, since the body is loud-wrapped.
*Good news, verified:* registering the two shapes has **no** adverse effect on existing rows —
all current producers use the builders and satisfy the shapes. Step 2 is warning-silent.

**B17 — `affected`-from-manifest changes behavior for no-delta revises.**
`_handle_revise` returns `affected.revised=[node_id]` unconditionally (`:610-611`) while
`_emit_revise_trace` returns early on empty deltas+warnings (`:212-213`). "Empty manifest =
no trace" plus "affected is derived" collapses these: either `affected` loses no-op revises
(breaking `runner.py:629` and `_accumulate_touched`) or the trace gate must be per-row.
Also `affected` cannot be "a helper over `mutations`" — it crosses a process boundary and
must be materialized; and `nodes.deleted` has no `affected` counterpart (`_affected` builds 3
keys; `daemon_server.py:823` iterates a hardcoded 3-tuple), so hard deletes are trace-only.
*Correction:* rule all three explicitly.

**B18 — Two mislabels that would misdirect an audit.**
`dal_logs.py:182-238` is **not** "logs-DB janitorial" — it executes `DELETE FROM
edge_relations` / `DELETE FROM edges` on a caller-supplied `graph_conn`. The exclusion
rationale (endpoint nodes already gone) holds; the label doesn't.
"All 9 write commands" matches no set in the tree: `WRITE_COMMANDS`
(`scales/dispatch.py:82-87`) has 10 and **omits `revise_edge`** — which is emit site #10, so
gating the wrapper on that set silently deletes its trace. `COMMAND_TABLE` `is_write=True`
is ~22. The real mutation-carrying set is **8**: remember, remember_batch, revise,
revise_batch, connect, connect_batch, revise_edge, brain_batch. Name a new
`MUTATION_COMMANDS` frozenset; do not reuse `WRITE_COMMANDS` or `is_write`.

**B19 — The dashboard cannot import from `servers/`; "unify the mirror" is forbidden.**
`dashboard/queries/s2_runs.py:23-24` states the disconnection contract explicitly. Also
`query_healer_runs` (`:488-539`) selects **every** delta on `chain_id LIKE '%healer%'` with
no ref_type filter — it already emits phantom cards for `journal_note`, and after step 0
every healer field-fill would add one.
*Correction:* "mirror + pin" (a consistency test across trace_contract ↔ S2 gate ↔ dashboard
mirror), not "unify". Name `query_healer_runs` alongside `_fetch_ok_deltas`. Add "+ dashboard
restart" to the rollout (`com.brain.dashboard` is a separate launchd service).

**B20 — `EMITTER_REF_TYPES` scope must be stated, and it is safe.**
§4 says "new ref_types"; §7 says "new types **and** the existing pair" — these differ, and
the second changes `_last_run_timestamp` for consolidation, community and aspect integration.
*Verified safe:* each unit writes its own delta ref_type (`aspect_classified`,
`community_enriched`, `consolidated`, `healer_generated`), so excluding the pair cannot
strand a unit at cold start. Say which reading is intended and cite the verification.
Note step 0 arms the S2 gate before step 2 creates the exclusion — land it in step 0 or state
the interim scheduler effect.

**B21 — `runtime_fingerprint` at `(s0,K)` would be an unclassified conversational turn.**
`S0_CONVERSATIONAL_INCOMING` (`trace_contract.py:180-184`) is keyed by `(s0,K)` ref_type and
`CONVERSATIONAL_REF_TYPES` derives from it — which drives `turns_since_last_encode`,
`get_session_turns` and the eager-embed set. A new `(s0,K)` type must be classified there.
`config_changed` at `(s0,delta)` is uncontroversial. (Part-3 item.)

---

## C. Part-3 charter (§12) — arithmetic and mechanism corrections

**C1 — The cost model is wrong in both directions; the conclusion survives.**
- Recalls/day: **~28**, not ~300 (846 `recall` traces in 720h; one per user prompt, 1:1 with
  `surface_selected`/`additionalContext`). 10x overstated.
- Per-pull: as the sketched JSON columnar shape, **~175-200KB**, not 60-75KB (8208 short ids
  ≈ 90KB + 8208 4-decimal floats ≈ 66KB + 500×6 z ≈ 21KB). ~3x understated.
- Net: **~5MB/day JSON, ~2MB/day binary; ~70MB / ~30MB at 14d.** Affordable — but rule the
  encoding, and note sparse-lane relief is measured (`e43b4f79`: `pick` averages 84 non-zero
  nodes, `enc` 25) so 3 of 6 lanes should be index/value pairs.
- Corpus size confirmed exact: **8208** nodes, all with ≥1 embedding view.
- Caveat: engine invocations exceed surface tails (MCP `recall`/`recall_batch`, agentic fetch
  tools each score the field). Multiply if capture moves per-engine-call.

**C2 — Lane count is 6, not ~8:** `maxsim, pick, enc, idf, sit, proj`
(`recall_laf.py:754-762`). Moment slot lanes are dormant (`moment_K: 0`) and add 3 per
(side, slot) when activated.

**C3 — The "~500-node pre-floor field" does not exist.** Real pipeline: per-candidate noise
floor → sort → `scored_results[:limit]` where limit = 25 + 15 headroom = **40** → *then* the
relevance floor. Truncation precedes the floor, so the "pre-floor" set is already ≤40. The
500 is a **harness** concept from `62b04f12` whose enabling flag `full_field=True` was
proposed and **never built** (`full_field` appears nowhere in `servers/` or `eval/`).
*Correction:* specify "top-N of the sorted field, N configurable (500)" — honest about being
a window — or build `full_field` first.

**C4 — `_laf_fields` covers 50 nodes, not 500.** `telemetry_top_n` default 50
(`recall_laf.py:108`), built from `argsort(-s01)[:telemetry_top_n]` (`:953-961`); the live
K-store config carries only `{z_norm:'support'}`, so 50 is in force. The ruled per-lane
decomposition cannot be assembled from it. Also telemetry is assembled **inside** `scores()`
under the engine lock, so "never on the scoring path" is approximate.

**C5 — The "full final field" is not in scope at the claimed capture point.** `scores()`
returns a score map over all 8208 nodes into a local in `recall()`; the return dict carries
`_laf_fields` (`brain_recall.py:2156-2157`) and `_recall_mode` (`:2110`) but never the field.
At the surface tail the 8208-value field is already gone.
*Correction:* capture at the sorted `scored_results` immediately before the `[:limit]` cut
(`brain_recall.py:1920-1942`) — a real in-memory object — or add a return channel and name it.
Good news: `recall_variant` needs no derivation, `result['_recall_mode']` already carries
`laf_v1`/`embeddings_first` and is in scope at the tail.

**C6 — Tom's separate LAF switch is NOT a free config entry.** The shipped gate is
`{kinds: {kind: bool}, retention_days: <one global>}` (`trace_contract.py:391-397` at
`bb7ed5a`); per-kind retention does not exist, and `_payload_kind_enabled` returns
`bool(effective.get(kind))` — a dict value is truthy, so `{'enabled': False, ...}` would
**record anyway**. The ruled `{enabled, scope, retention_days}` block requires changing the
resolver, both named shapes, and the pruner — part-1's territory under the ownership split,
needing the capture-engine owner's sign-off.
*Also:* §12's gate vocabulary is stale — it says `{scope: topk|full, k}`; the brain node
(`2df35ee9`) says `{scope: field|full_dense}` with no `k`. The doc is the stale copy.
*Confirmed sound:* the `PAYLOAD_KIND_EXT` one-entry mechanism, `record_payload` writer,
pointer-in-trace-row, `payload_sort_key`, and the judge-kind precedent — all accurate at
`bb7ed5a`. A new kind defaults **off** in normal unless added to `_NORMAL_ON_KINDS`.

**C7 — Gap D(a): TTL claim true, hook claim needs re-checking post-rebase.** `CONFIG_TTL_S =
60.0` with no invalidation (`recall_laf.py:136`, `:379-381`) — a gain flip lands up to 60s
after its `config_changed` stamp: TRUE. The `brain.py:741` invalidation hook: **absent in this
tree**, present at `bb7ed5a` as `invalidate_trace_recording_cache` called from
`set_interaction_active`. Re-verify after rebase; it is a real pattern to copy, ~5-15 lines,
with two caveats: an out-of-process flip won't invalidate (TTL stays the backstop), and a
generic config-version counter is the better shape if more caches appear.

**C8 — Gap D(b) accurate:** `_laf_fields`' last touch is `brain_recall.py:2157`, zero
consumers; closing `64c506c4`'s deferred item via Gap A's capture is right.

---

## D. Confirmed sound — do not re-litigate

- **§1.1's twelve-site table**: every line ref verified (dispatch_write.py
  440/447/527/535/597/670/751/1070/1113/1154/1230 + brain_remember.py:432), batch envelope
  871→1029→1037, the comment at 1025-1028, footnote ¹.
- **§1.3's pop-then-read diagnosis**: `_pop_session_ctx` mutates `args` in place
  (`dispatch_common.py:73-81`), so `caller_session(args)` at `:437`/`:522` is structurally
  `''`.
- **§2's five ruled properties**, including the missing-beats-lying asymmetry.
- **The core architecture**: one emitter, manifest-driven, at a single chokepoint,
  post-commit. Grounded by `17234b02` (as revised) and `e307fd4c`.
- **§1.5's premise**: `_apply_connect_to`, `add_relation`/`connect_typed` and `revise`
  already return the needed rows and the handlers throw them away.
- **§4's contract-registration analysis**: the existing pair occupies exactly
  `(s0,delta)/(s1,delta)/(s2,delta)`; `METADATA_REQUIRED_BY_REF_TYPE` omits both, so
  validation is genuinely declared-but-dead.
- **§5 items 1, 5, 9** (`remove_relation` return, `revise`'s `co_anchored_made`,
  `delete_node_cascade` return) — accurate and caller-safe.
- **§7's ref_id contract**: complete and accurate. No reader splits the composite; no reader
  JOINs trace ref_ids against nodes/edges; per-edge trace **volume** is preserved by the
  design (only commit count collapses), so no volume-sensitive reader skews.
- **§9's `reclassify.py` verdict** — but see E2.
- **§10's parked invariants** (`get_bulk` hydration as truth filter, coverage gated on the
  `encoding_run` receipt) — correctly reasoned, correctly parked.
- **§6's `enrich` / Hebbian / `seed_pack` exclusions** — accurate.
- **The MCP-gate exemption**: nothing touches `BATCH_OP_SPECS`, so
  `test_brain_batch_op_contract.py` is unaffected and `eval/mcp_batch_probe.py` /
  `mcp_schema_gate.py` are correctly not required. The gate the doc *misses* is the runtime
  ref_type-registration test (B15) plus adding the new DAL primitive to
  `tests/test_write_txn_discipline.py:221-226` (`BATCH_REACHABLE_WRITERS` is hardcoded).
- **Trace volume impact is immaterial**: `node_created` ≈ 415 rows/week ≈ 5% of weekly trace
  volume; `node_archived` replaces `tool_result` 1:1. But note **nothing prunes
  `trace_events`** (166,760 rows, 731MB logs DB) — acknowledge it and point at part 1's
  per-kind retention pattern.

---

## E. Open items requiring a ruling or extra care

**E1 — OPEN RULING (Tom): purge observability.** Brain node `ca66f5bd` (live, unsuperseded)
rules that junk vocab should be **erased, not preserved with an audit trail**, and the doc
wrongly framed that hard delete as the highest-severity blind spot. The erase-the-node half
is settled. The open question is whether the *operation* should be observable — proposal: a
skeletal rollup ("purged N junk vocab nodes at T"), keeping no garbage content. `ca66f5bd`
has been revised to record this tension and to fix its stale mechanism reference
(`NodeDAL.purge()` no longer exists; the path is `brain.delete_node_cascade`).

**E2 — §9's `reclassify.py` deletion conflicts with an open backlog item.**
`docs/BACKLOG.md:287` (+ the open list at `:17`) asks to verify `reclassify.py` is wired into
the S2 coordinator and run it once against the corpus — a **different** purpose from the
retired aspect migration §9 cites. Step 4 must close that backlog item explicitly or keep the
file. Path drift: BACKLOG and `docs/S2-DESIGN.md:437` say `servers/scales/s2/reclassify.py`;
the file is at `servers/scales/s2/archive/reclassify.py`.

**E3 — `set_personal` verdict right, rationale wrong.** The personal flag is **live** and
scored in recall (`brain_recall.py:1869-1877`); `revise` can already set
`personal`/`personal_context`. The single capability lost by deleting `set_personal` is
*auto-lock on `personal='fixed'`* — and that invariant is already bypassed today by
`revise(personal='fixed')`. Precedent: `NodeDAL.unlock` was already deleted as dead
(`0712ca78`). Fix the rationale, keep the verdict.

**E4 — Emitter must run inside the existing write-lock acquisition.** Today every emit runs
inside `_locked_exec` (`daemon_server.py:719-733`) / `with brain.write_lock`
(`s2/base.py:288-289`), and the repo convention is that shared-`logs_conn` writes take
`write_lock` (`self_channel/signal.py:66-67`). Emitting after `_locked_exec` returns writes
`logs_conn` unlocked — which is exactly how another thread's `commit_unless_batched` commits
a partial batch (B3). Note main's `_log_failed_batch_ops` sits *outside* the lock — do not
copy that placement for a writer.

**E5 — "Each step ships green alone" must become "each step is production-correct alone."**
Merging auto-deploys asynchronously: the daemon is launchd-pinned to the source checkout, and
the next `ensure_daemon()` or MCP health ping sees a fingerprint mismatch and issues
`launchctl kickstart -k` at an arbitrary moment after the merge. There is no manual gate
between steps, so intermediate states must be behaviorally valid, not merely importable.

**E6 — Rollback story for the flip.** A revert + restart recovers behavior, but **wrong trace
rows are not cleaned up by a revert** and nothing in the repo prunes `trace_events`. Before
the flip: `cp brain_logs.db brain_logs.db.bak-{ts}`, and record the cleanup shape
(`DELETE FROM trace_events WHERE ref_type IN (...) AND created_at > '{flip_ts}'`).

**E8 — ADDED SCOPE for the contract step: ~15 registered ref_types have zero writers.**
(Found in the emission audit, not by the review agents — brain node `384ddae3`.) The
`REF_TYPES` table documents events that have never been emitted, which actively misleads
anyone reading it as a map of system behavior. Dead: the s2 legacy block
(`recall_quality_signal`, `confidence_adjust`, `kept_distinct`, `stale_nodes`,
`correction_chains`, `community_diff`, `graph_stats` — plus `evolved`, which has one stale
reference to check first) and **all of scale 3 and scale 4** (13 entries for scales with no
code). Prune the s2 legacy entries; either delete s3/s4 or move them out of the validated
table into a clearly-marked aspirational block. Do it in the SAME edit that registers
`node_created`/`node_archived`/`node_deleted` and fixes the dict/shape ordering (A1) — one
edit, one review, instead of a second pass over the same table.
**Do NOT confuse this with write-only traces:** Tom ruled `scout_input`/`scout_findings`
**stay** (node `57d30c1d`) — nested, otherwise-invisible work earns its trace precisely
because nothing else shows it. Never-written ≠ write-only.

**E7 — Doc-structure defects to fix in the rewrite.** §12 (~90 lines of *charter*) sits
inside the execution path between §11 and §13 and reads as backlog — move it to its own doc.
§8's step-3 pointer cites §7 where it means §11. §4's registration block and §8's step 3 are
checklists written as paragraphs — make them enumerated file lists. No step names a test
command ("full-suite tier" is not repo vocabulary; the repo form is `./dev pytest tests/`).
Missing operationally: a production smoke check (write a throwaway node → `query_traces
ref_type='node_created'`; `query_logs` for `mutation_trace_emit`), the rollback note, and
which restart form. Adopt one citation convention (`node:98031b8e` vs `@bb7ed5a`) —
`1b8f05ca` in §13 is a **session id**, not a commit.
