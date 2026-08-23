# Eval Read Path → Brain/Traces API — Work Order

Route the longmem eval's reads through the brain API and the trace delta, delete the
hand-rolled SQL dumpers, and remove the drift class that broke one of them.

> ## STATUS — planned, step 0 verified
>
> | step | state |
> |---|---|
> | 0 — gate: `node_created` delta is live | **AMENDED** — live in production, absent in eval brains; see step 1 |
> | 1 — `artifacts.py` node/edge reads → brain API | **DONE** (with amendments below) |
> | 2 — collapse the duplicate dumpers | not started |
> | 3 — port the noun-retention metric | not started |
> | 4 — the `context_anchors` restore eval | not started, blocked on 1-3 |

**Each step runs cold in its own session.** Step 1's session runs
`/architecture-review` against this doc first — validate, form an opinion, then
execute. Do not execute this doc on trust; every code claim below is checkable
and several of my earlier claims in this arc turned out wrong.

---

## Why

The eval reads brain state by hand-written SQL instead of through the brain. That
produced three independent `dump_nodes` implementations:

| where | shape | state |
|---|---|---|
| `eval/longmem/artifacts.py:178` | 32 columns, incl. `keywords` | **broken** |
| `eval/longmem/ab_encode.py:60` | 5 cols + KV(`situation`,`event_time`), content→220 | diverged |
| ~~`eval/longmem/diff_encoding.py:19`~~ | 5 cols + KV(`situation`,`question`,`event_time`,`reasoning`), content→300 | **file deleted** 2026-08-22 (override migration Step 7) |

`ab_encode`'s docstring used to say it outright — *"(from diff_encoding, kept
local)"* — a declared copy that had since diverged from its source in both KV key
set and truncation limits. Both remaining dumpers N+1 query `node_metadata_kv` per
node in a Python loop.

(A fourth hand-rolled dumper of the same drift class exists outside this work
order's scope: `eval/oracle_audit/emb_bench/dump_pack.py` defines its own
`dump_edges(con)` against a raw sqlite connection. Steps 1–2 do not make the
migration complete; that one is a separate cut.)

`nodes.keywords` was dropped in schema v28. `artifacts.py` still selects it, so:

```
dump_nodes raises (no such column: keywords)
  → try/except writes an error file, returns
    → nodes.jsonl never written
      → load_artifacts: "Missing files become None — never raises"
        → analyzer.py:200  nodes = bundle.get('nodes') or []
          → node dimensions computed over []  →  reported as real numbers
```

Five modules consume that bundle (`analyzer`, `report`, `connect_ab`,
`pooled_review`, `harness`). Each would read zeros indistinguishable from
"measured, found nothing."

**Latent, not historical.** The in-tree reports are from April 2026; the column
was dropped around July. No published result was corrupted — the *next* run is
what breaks. Do not go looking for wrong numbers in old reports.

This is the failure mode the operator ruled against three months ago
(brain node `id:a00c6cd0`):

> we have traces for all of these things, its not about logging its about
> quality. thats why we have a brain (a seed in the case of the test)

The drift is the symptom; bypassing the brain API is the cause.

---

## The ruling

Split, decided by the operator:

- **B2 — delete the layer** for `nodes` and `edges`. The trace delta is not merely
  a replacement, it is *more correct* (see below).
- **B1 — keep the format, change the source** for `recall` and `meta`. Those are
  genuinely run-scoped artifacts, not brain state.

---

## Verified facts (step 0)

Checked against a **copy** of `brain_logs.db` — never open the live DB as a second
writer while the daemon runs.

- `node_created` is live: **639 rows, 2026-08-06 → 2026-08-14**. Shipped with the
  mutation emitter (`docs/MUTATION-EMITTER-PLAN.md` steps 1-8).
- It is a **`ref_type`**, not an `event_type`. `event_type` holds the lane
  (`O`/`K`/`delta`). Querying `event_type='node_created'` returns 0 and looks like
  the feature is dead — it is not. I made this mistake twice.
- Row shape: `scale='s1'`, `event_type='delta'`, `ref_id=<node id>`,
  `metadata={"node_id","type","title","encoding_source","reason", identities}`.
- Siblings in the same lane: `node_revised` 601, `encoding_run` 1636,
  `encoding_run_failed` 53, `node_archived` 1.
- **Limit:** the delta only exists from 2026-08-06. It cannot reanalyse older runs.
  Fine for new runs, which is all this needs.

**`query_traces` already covers the read — do not add a new owner function.**
`servers/brain_traces.py:113` supports `ref_type=` + `session_id=` +
`hours=None`, and flags truncation rather than silently clipping. The
"missing function is the finding" rule does not apply here; the function exists.

**`brain.get_node(node_id_or_ids)`** (`servers/brain_recall.py:349`) takes one or
many and is the canonical pull — it walks corrections and attaches `_corrections`.
Strictly better than the raw SQL it replaces.

**`analyzer.py` is already half-migrated.** It walks `traces` for `_walk_encoder`
and `_walk_scouts`, then reaches for `nodes.jsonl` for node content. B2 finishes a
migration already underway — cf. `id:24caccca`, half-migrated code accumulates.

---

## Steps

### 1 — `artifacts.py` node/edge reads → brain API

Replace the raw SQL in `dump_nodes` / `dump_edges` with:

- **which nodes this run created** → `query_traces(ref_type='node_created',
  session_id=…, hours=None)`, then `brain.get_node([ids])` for content.
- **edges** → the brain's edge read (`GraphDAL.get_connections_bulk` via the API,
  not directly — confirm the exposed method before writing).

This replaces the heuristic `encoding_source != 'anchor:seed'` — which *guesses*
which nodes a run produced — with the delta that *is* that set. Less code, better
answer. Note the two survive-check consumers need `id`, `title`, `type`,
`encoding_source`, plus full text/KV for `_node_text`; `get_node` returns all of it.

Per B2, prefer having the consumers read this directly over re-serialising to
JSONL. If the JSONL hop cannot be removed cleanly in one step, land the
re-sourcing first and delete the hop in step 2 — do not leave both paths alive
past step 2.

#### Step-1 amendments (found by the implementing session, 2026-08-16)

Four claims above were wrong or incomplete; the implementation follows the
amended versions:

1. **Step 0 verified the wrong database.** `node_created` is live in the
   *production* logs DB, but eval brains had **zero** rows (checked
   empirically across frozen corpus brains): `replay._make_local_dispatch`
   called `entry.handler()` directly, bypassing `dispatch_command` — the only
   caller of the mutation emitter. That bypass was deliberate (brain ruling
   `id:332d170a`, "eval sites stay traceless"), but it is superseded by the B2
   ruling, whose delta read cannot exist without emission — and its
   pollution rationale doesn't hold: the eval already writes the
   system-under-test's own traces (`encoding_run`, scouts, journal) via
   `trace_append`, and mutation traces are the same class of signal. Fixed:
   the local dispatch now routes through `dispatch_command` (mirroring
   `IsolatedBrain.dispatch`), which also restores `check_unknown_keys` and
   `log_failed_batch_ops` — production faithfulness the bypass had dropped.
   Old frozen corpora keep their already-written bundles; the delta exists in
   every corpus built after this change.
2. **Do not scope the delta by `session_id`.** Items ingest many haystack
   sessions and S2 units carry no session — a session-scoped query MISSES
   nodes. Per-item brains are fresh (`wipe=True` at every live call site), so
   the whole logs DB is the run: `query_traces(ref_type='node_created',
   hours=None)`, unscoped, loud on truncation.
3. **The consumer list was wrong.** `connect_ab` parses encoder tool-call
   payloads and `pooled_review` reads its brain directly via `filter_nodes` —
   neither touches `nodes.jsonl`/`edges.jsonl`. The real bundle consumers are
   `analyzer`, `report`, `structural_diff`, `run_diff` (and `harness` as the
   producer). `report._gold_in_brain_for_item` needed one adjustment: an
   empty delta is a real "encoder created nothing" answer, distinct from
   missing artifacts.
4. **The JSONL hop is not removable for `artifacts.py` — in step 2 or ever.**
   Its consumers are post-hoc: they read `reports/` after the per-item brain
   is deleted. The bundle *is* the durability layer (the module docstring's
   whole reason to exist). B2's "read directly" applies to the step-2 dumper
   (`ab_encode`), which analyzes a live in-process brain.

Semantic change, accepted under B2: `nodes.jsonl` was "snapshot including the
seed pack", it is now "the run's delta" — seed nodes and seed-pack edges no
longer inflate per-item counts (`structural_diff`/`run_diff` metrics now
measure what their docstrings claim). Noise relations (`co_accessed`,
`emergent_bridge`) are excluded from `edges.jsonl` by the canonical read.
Old bundles and new bundles are directly distinguishable by this.

Post-review widening: the node delta is `node_created` ∪ `node_revised`
(records carry `delta_op`) — without the revised half, gold the encoder
writes into a pre-existing seed node (revise, or absorb-into-seed-survivor)
would be invisible to the gold-bearing scans. `edges.jsonl` stays
created-nodes-only: a revised seed's pre-existing edges are not run behavior.

### 2 — collapse the duplicate dumpers

Delete `ab_encode.dump_nodes/dump_edges`; point it at the single reader from step 1.
**Half of this step landed for free:** `diff_encoding.py` was deleted whole on
2026-08-22 (override migration Step 7 — zero importers, and its `register_prompt`
mutated the production daemon), so only `ab_encode`'s copy remains and there is no
longer a divergence to reconcile. Its KV key set (2) and truncation (220/120) are
what step 1's reader must either match or deliberately supersede.

Also in scope, same class of rot:
- `eval/longmem/cost_summary.py:41` — `'scout_synthesis'`, a scout deleted from the
  codebase.
- `eval/longmem/analyzer.py:88` and `eval/longmem/structural_diff.py:70` — read the
  dead `keywords` field. (`structural_diff`'s `gold_keywords` is a *different*
  thing — local gold-matching vocabulary. Leave it.)

### 3 — port the noun-retention metric

`extract_proper_nouns()` + `noun_retention_pct` currently live in
`eval/s1s_ab_quality_analyzer.py` (last touched **2026-04-24**), which reads
preserved brains from `eval/reports/s1s_ab_smoke/{run}/brains/` — **a directory
that does not exist**. The metric code itself is clean of stale references; only
its harness and input are gone.

Move the two functions into `eval/longmem/analyzer.py` (maintained, on the live
corpus, wired to the A/B machinery). Do not resurrect the old harness.

### 4 — the `context_anchors` restore eval

Blocked on 1-3, because the point is to measure with an instrument we trust.

- `context_anchors` is an **encode**-side field. `sweep.py` is explicitly "Stage 2 —
  recall over a frozen corpus, zero encoding" and cannot see it.
- `build_corpus.py` **does** support `interaction_overrides={'s1_scout_facts': N}`
  (now `tests.interaction_override.override_interaction`). Two encode runs, v7 vs a
  DORMANT v8.
- Register v8 DORMANT, run both arms, compare noun retention, activate + `./dev
  sync-prompts` only if it holds.

**Seed-gate hazard (verified 2026-08-16, found by 17d9ae94 via 2ee7a900 —
do not re-land it):** on `ad9981c`, `interaction_seed.py:29` imports
`FACTS_OUTPUT_SCHEMA` and `:252` embeds it in `S1_SCOUT_FACTS_CONFIG_V1`
**by reference**. Adding `context_anchors` to that constant at registration
time would seed the DORMANT candidate as v1 on every fresh brain —
bypassing the eval gate `sync_prompts._fetch_active` guards on the template
channel, through the config channel. The correct shape: **code carries
ACTIVE, DB carries candidates.** Build the v8 schema at registration time
DB-only via `copy.deepcopy(FACTS_OUTPUT_SCHEMA)` — `{**...}` is a shallow
copy and the field sits four levels down
(`["properties"]["candidates"]["items"]["properties"]`), so spread-then-
assign mutates the shared constant and reopens the leak. Touch the constant
+ re-sync only at activation, as one step. Invariant:
`FACTS_OUTPUT_SCHEMA` always equals the ACTIVE version
(17d9ae94's `check_configs` enforces it).

Context for why this matters: seed_7 (`id:cc834325`) measured noun retention
regressing on every transcript, −1 to −18pt. `context_anchors` was the fix. seed_8
(`id:50018eb9`) shipped it and **never re-reported the axis**. It was then killed
silently by the Structured Outputs migration on 2026-05-17. This eval is the
measurement that was skipped.

---

## Scope guards

- **The corpus is clean — do not rebuild it.** `longmemeval_oracle.json`: 500 items,
  **0** duplicate `question_id`, **0** duplicate `question`. The 37 duplicate
  *answers* are short values (`2`, `4`, `$300`, `7 days`) across different
  questions. Expected, not corruption.
- **Leave `eval/laf/`, `eval/prompts/`, and `eval/reports/*.json` alone.** They are
  frozen captured artifacts. Their stale references are historically accurate.
- **Do not touch `docs/samples/`.** Both files are dated captures of a **v13**-era
  prompt and a real v13-era run; their content is true as history.
- Do not widen into the surface-side schema. `SURFACE_SELECTION_SCHEMA` has its own
  open finding (its shape is unguarded by any test) — that is a separate cut.
