# Traces Part 3 — Analysis Substrate (charter)

**What this is.** A charter, not a spec: inventory and rulings only. The spec is its own
design session. Moved out of `MUTATION-EMITTER-DESIGN.md` because ~90 lines of charter
sitting inside an execution path reads as backlog and hides instructions.

**Status.** Design-ready; **no dependency on part 2's emitter**. Brain: `node:2df35ee9`.
Ruled by Tom 2026-08-03. Sibling to part 1 (recorder: agent I/O) and part 2 (emitter: graph
mutations) — this is the read-path + config half.

**Ownership.** This thread owns the part-3 design; the **part-1 stream owns the capture
engine** and reviews any new kind/format against the recorder contract before implementation.

> Citations resolved at `bb7ed5a`. Numbers below were re-measured, not relayed — the first
> draft's cost model was wrong in both directions.

---

## Gap A — LAF recall capture (the headline)

The LAF engine scores the **entire embedded corpus** per pull; the recall O trace keeps a
pipe-string of 25 finals; the per-lane decomposition is computed, attached in memory, and
dropped. Post-hoc "why did X rank #14" and production A/B of gain configs are impossible
today (`node:5d9f1fd2`).

### Corrected facts (the first draft got these wrong)

| Claim in the first draft | Verified truth |
|---|---|
| ~8 lanes | **6**: `maxsim, pick, enc, idf, sit, proj` (`recall_laf.py:92-103`). Moment slot lanes are dormant (`moment_K: 0`) and add 3 per (side, slot) when activated. Do not confuse with the **6 embedding views** `maxsim` reduces over (`:142`). |
| `_laf_fields` covers ~500 nodes | **50** — `telemetry_top_n` default 50 (`recall_laf.py:108`), applied at `:953`/`:958`. The live K-store config carries only `{z_norm:'support'}`, so 50 is in force. **The ruled per-lane decomposition cannot be assembled from it.** |
| a "~500-node pre-floor field" exists | **It does not.** Real pipeline: per-candidate noise floor → sort → `scored_results[:limit]` (`brain_recall.py:1942`) where limit = `max_candidates` 25 (`surface_contract.py:163`) + `seen_dedup_headroom` 15 (`:169`) = **40** → *then* the relevance floor. Truncation **precedes** the floor, so the "pre-floor" set is already ≤40. The 500 is a **harness** concept from `node:62b04f12` whose enabling flag `full_field=True` was proposed and **never built** (`full_field` appears nowhere in `servers/` or `eval/`). |
| ~300 recalls/day | **~28** (846 `recall` traces in 720h; one per user prompt, 1:1 with `surface_selected`). 10× overstated. |
| ~60-75KB/pull | **~175-200KB** as the sketched JSON columnar shape (8208 short ids ≈ 90KB + 8208 4-decimal floats ≈ 66KB + 500×6 z ≈ 21KB). ~3× understated. |
| — | Corpus size confirmed **exact: 8208** nodes with ≥1 embedding view. |

**Net cost, corrected: ~5MB/day JSON, ~2MB/day binary; ~70MB / ~30MB at 14d retention.**
The conclusion survives — affordable — but the *encoding* must be ruled, and sparse-lane
relief is measured (`node:e43b4f79`: `pick` averages 84 non-zero nodes, `enc` 25), so **3 of
6 lanes should be index/value pairs**, not dense arrays.

**Caveat that changes the arithmetic:** engine invocations exceed surface tails. MCP
`recall`/`recall_batch` and the agentic fetch tools each score the field. If capture moves
per-engine-call rather than per-surface-tail, multiply.

### Where to capture — corrected

`scores()` returns a score map over all 8208 nodes into a **local** in `recall()`; the return
dict carries `_laf_fields` (`brain_recall.py:2157`) and `_recall_mode` (`:2110`) but **never
the field**. At the surface tail the 8208-value field is already gone — so the first draft's
capture point cannot see what it proposed to record.

**Ruling:** capture at the sorted `scored_results` immediately before the `[:limit]` cut
(`brain_recall.py:1929-1942`) — a real in-memory object — or add an explicit return channel
and name it. Good news: `recall_variant` needs no derivation; `result['_recall_mode']`
already carries `laf_v1`/`embeddings_first` and is in scope at the tail.

Also note telemetry is assembled **inside** `scores()` under the engine lock, so "never on
the scoring path" is approximate — state the real cost.

### Ruled design (unchanged by the corrections)

- **Trace row (bounded, always):** structured top-25 candidate shape (a dict, not a
  pipe-string — `RECALL_CANDIDATE_SHAPE` in the contract), plus the stamps the O trace lacks:
  `recall_variant` and the `recall_laf` interaction version (the gain config that scored this
  pull), plus the payload pointer.
- **Payload file** (recorder kind `recall_fields`, columnar): one id array + parallel columns
  per lane — no id repetition, jq/pandas-native.
- **Scope: the FIELD, not a top-K window.** `node:62b04f12` retired top-K cuts for LAF and
  the running-field architecture `node:87a6dae9` names cache-only-top-K as the anti-pattern —
  divisive normalization, reach and inhibition all live in the below-pool mass, and the P2
  walker trains on field-level data. Because the "500-node pre-floor field" does not exist,
  the honest specification is **"top-N of the sorted field, N configurable (default 500)"** —
  named as a window — *or* build `full_field` first. Do not describe a window as a field.
- **Mechanism is already built (part 1) — do not re-derive.** A new kind is one
  `PAYLOAD_KIND_EXT` entry (`trace_contract.py:372`); writer is
  `brain.record_payload(chain_id, kind, content, seq=)`; the pointer goes in the trace row
  metadata; the `judge` kind shipped exactly this pattern. Reader rules (hard-won in part
  1's review): pointer-in-trace-row, never glob payloads on polled endpoints (glob only on
  card expand); attempt ordinals sort with `payload_sort_key`, never `sorted()[-1]`. A new
  kind defaults **off** in normal unless added to `_NORMAL_ON_KINDS`.

### The separate switch is NOT a free config entry — corrected

Tom ruled `recall_fields` gets its own hot-flippable gate block. The shipped gate is
`{kinds: {kind: bool}, retention_days: <ONE global>}` (`trace_contract.py:388-397`):

- **Per-kind retention does not exist.**
- `_payload_kind_enabled` (`brain_traces.py:806`) returns `bool(effective.get(kind))` — a
  dict value is **truthy**, so `{'enabled': False, ...}` would **record anyway**. This is a
  silent-failure trap, not a config addition.
- The ruled `{enabled, scope, retention_days}` block therefore requires changing the
  resolver, both named shapes, and the pruner — **part-1's territory** under the ownership
  split, needing the capture-engine owner's sign-off.
- Vocabulary: `node:2df35ee9` says `{scope: field|full_dense}` with **no `k`**. The first
  draft's `{scope: topk|full, k}` was the stale copy.

Doubly separated from LAF itself: `BRAIN_RECALL_VARIANT` gates whether LAF runs; this kind
gates whether it is recorded.

---

## Gap B — runtime fingerprint + brain identity

No trace records which code/config/brain produced it (the boot heartbeat is summary-only;
schema migrations run silently). Ruled: one `runtime_fingerprint` trace per daemon boot
carrying `{git_commit, git_dirty, branch, schema_version, embedder, env_variants,
active_interaction_versions{}, brain_id, instance_id, arm_label, db_dir}`. Everything after
joins by time.

- `brain_id` — UUID minted once at brain creation (`brain_meta`; one backfill migration row).
  Copies **inherit** it: it is the lineage id.
- `instance_id` + `arm_label` — minted fresh at copy time (`IsolatedBrain.copy()` stamps
  both; the harness names the arm). Cross-arm A/B becomes `GROUP BY arm`, not a file-path
  convention.
- git unavailable → `"unknown"` + loud log; never blocks boot.

**Registration constraint (new — missed by the first draft).** `runtime_fingerprint` at
`(s0, K)` would be an **unclassified conversational turn**: `S0_CONVERSATIONAL_INCOMING`
(`trace_contract.py:180-184`) is keyed by `(s0,K)` ref_type, and `CONVERSATIONAL_REF_TYPES`
derives from it — driving `turns_since_last_encode`, `get_session_turns`, and the eager-embed
set. Any new `(s0,K)` type **must** be classified there, or it silently becomes a
conversational turn and perturbs the Scribe's cadence. `config_changed` at `(s0,delta)` is
uncontroversial.

---

## Gap C — `config_changed` trace (ruled: easy, do it)

`register_interaction` / `set_interaction_active` / `set_config` are the most
behavior-changing writes in the system and emit nothing. One `config_changed` trace at those
three handlers puts every behavior flip on the timeline next to its effects — the missing
join for before/after A/B analysis.

---

## Gap D — engine-side folds (LAF-scoped)

**(a) TTL-cached gains with no invalidation.** `CONFIG_TTL_S = 60.0` (`recall_laf.py:136`,
checked at `:381`) with no invalidation hook — a gain flip lands **up to 60s after** its
`config_changed` stamp, skewing before/after joins exactly when Tom starts flipping gains.

The pattern to copy now exists: `invalidate_trace_recording_cache`
(`brain_traces.py:801`, called from `brain.py:745`), built for `trace_recording`. ~5-15
lines, with two caveats: an **out-of-process** flip won't invalidate (TTL stays the
backstop), and a generic config-version counter is the better shape if more caches appear.
Belongs with Gap C.

**(b) `_laf_fields` has sat unrouted** since the 2026-07-15 handoff (`node:64c506c4`'s
deferred list); its last touch is `brain_recall.py:2157` with **zero consumers**. Gap A's
capture **is** its routing — the spec should close that deferred item explicitly rather than
leave a second half-wired path.

---

## Sequencing

Implementation in a fresh thread on top of main ≥ `bb7ed5a` (part 1 step 2 landed — the
`record_round_fn` seam + `build_round_payload` are the capture substrate `recall_fields`
plugs into). Gap A's call site is `surface.py`/`brain_recall.py` (now stable). New ref_types
follow the same `REF_TYPES` registration law as part 2 — plus Gap B's
`S0_CONVERSATIONAL_INCOMING` classification requirement.
