# S2 `absorb` — a first-class lossless merge primitive

**Status:** ✅ SHIPPED + WIRED (2026-06-04). The `brain.absorb()` primitive (16 tests)
AND the consolidation encoder now use it: `s2_consolidation_enrichment` **v7 active**
+ decoder **lever A** (`_pre_classify` cross-type → `needs_judgment`) +
`suppression_relations` += {`corrects`, `supersedes`}, merged to main (`1277a57`).
K=3 ground-truth corpus: correct 10→15, under-merge 8→3, over-merge 1, lossless 86%.
Full result + the matched-pair finding (prompt + decoder must ship together):
`docs/archive/session-handoffs/S2-CONSOLIDATION-ABSORB-SESSION-HANDOFF.md` §0.
**Deferred:** the live real-pair merge (`96d2fdf8`/`426ae3cd`, needs operator unlock),
Track 1b (locked-refusal log→warning + alert), edge-direction fidelity (Known issue #2).
Lever A goes live in the running daemon on next **restart**.

---

> **✅ 2026-06-04 — the "NEXT SESSION" plan below is largely DONE.** Item 1 (wire
> consolidation → v7) and item 2 (preservation gate: `eval/absorb_preservation_probe.py`
> + `tests/test_absorb_preservation.py`) shipped; Known issue #1 (rejection-table SKIP
> mis-classification) is FIXED (commit `351bec6`). Remaining: item 3 (live merge), item 4
> (Track 1b), Known issue #2 (edge-direction). Section retained as the design record.

## ⚠️ NEXT SESSION — START HERE

**DONE (committed on main):**
- `brain.absorb(survivor_id, absorbed_id, content=, reason=, updates=, **field_kwargs, prune_edges=, drop_fields=)` in `servers/brain_remember.py` — lossless transfer-by-default + revise-shaped field overrides.
- `absorb` brain_batch op — `servers/contract.py` (`VALID_BATCH_OPS`), `servers/dispatch_write.py` (dispatch branch, revise-op-style field pass-through), `servers/brain_mcp.py` (schema enum + description). Any non-control key on the op is a survivor field override.
- `tests/test_absorb.py` (16 tests: every transfer dimension, guards, the brain_batch op path, atomicity) — green, + contract/dispatch sync.
- Track 1a (stop-the-churn prompt) shipped separately: `s2_consolidation_enrichment` **v6 active** (`714ee68`) — see `S2-CONSOLIDATION-LOCKED-ABSORB.md`.

**NEXT (the actual work):**
1. **Wire consolidation to emit `absorb`** — rewrite `consolidation_enrichment_prompt.py`: CONSOLIDATE/EVOLVE + the three edge-migration sections collapse into "use `absorb`". Register DORMANT → eval → activate → `./dev sync-prompts`. Consolidation is sacred → **eval-gated**.
2. **Build the preservation-probe gate** ("The gate" below) — prove `absorb` is lossless on a richly-populated fixture (refs + voice + emergent KV + access + edges) before the new prompt activates.
3. **Live lossless merge** of the real pair: operator unlocks `96d2fdf8` → `absorb` into `426ae3cd` → verify schema/roadmap/quote/refs/access all land. Safe now because `absorb` preserves them (the lock was the only thing guarding them — audit `988de522`).
4. **Track 1b (still open):** `archive_node` locked-refusal `_log_error`→`_log_warning` + an actionable once-per-pair "locked near-dupe → unlock to merge" alert.

**Pointers:** eval harnesses `eval/s2_locked_probe.py` (real-instance A/B + reasoning) + `eval/s2_locked_eval.py` (fixture A/B on existing dimensions). Brain nodes: decision `cb1cf256`, lossy-audit `988de522`, milestone `7af599d9`.

**Known issues — deferred from the 2026-05-31 code review, FIX DURING WIRING:**
1. **Rejection-table SKIP mis-classification (fires the moment consolidation emits `op:absorb`).** `absorb` writes no `similar_to`/`consolidated_into` edge, so a *successful* merge looks like a SKIP to `consolidation.py`'s suppression detector → it stamps a rejection fingerprint → the cluster is suppressed. When wiring the prompt, either (a) have the encoder also emit a `similar_to`-free suppression signal, or (b) teach the SKIP detector that an archived-absorbed (via `_sys_archived_survivor_id`) is a successful merge, not a SKIP. **Must be handled or merges silently stop happening.**
2. **Edge-direction fidelity (subtle, lower-frequency).** If the survivor already has an edge to the same neighbor in the *opposite* physical direction, `add_relation`→`get_edge_id` matches that pair and the migrated relation inherits the existing edge's direction (reversed). Also a bidirectional absorbed↔neighbor pair collapses to one `get_connections_bulk` entry. Needs a direction-aware migration (or an edge-model fix) — out of scope for the primitive, real for correctness.

The 2026-05-31 follow-up commit already fixed: atomicity (SAVEPOINT/rollback envelope), re-embed of filled fields (route through `revise`), `revise` error handling, edge-weight preservation, the already-archived guard, and the `_CONTROL` session_id/chain_id leak.

---

## Why

Consolidation's "survive-and-absorb" is currently *imperative and opt-in*: the
encoder rebuilds the survivor by hand — `revise(full merged content)` + one
`connect` per migrated edge + `archive(peer)`. **Everything it doesn't explicitly
re-emit is lost.** That's lossy by construction: the default is loss, preservation
is a thing the LLM has to remember to do.

### Evidence — the real-pair preservation audit (2026-05-30)

`426ae3cd` "absorbed" `96d2fdf8` (it couldn't archive it — locked — so both still
exist, a natural experiment). What actually survived:

| Dimension | On peer (96d2fdf8) | Preserved into survivor? |
|---|---|---|
| Content: "two living things" quote | yes | ✓ (textual) |
| Content: table schema (`parent_version`) | yes | ✗ LOST |
| Content: roadmap (s2/s3/s4) | yes | ✗ LOST |
| Voice: `user_raw_quote` (structured field) | yes | ✗ LOST |
| `source_refs` | n/a (pre-v29) | ✗ structurally never migrated (zero handling in `scales/s2/`) |
| `access_count` | 3999 | ✗ NOT merged (survivor 3542) |
| Edges | 10 | ~partial |

The probe also showed the baseline encoder emit **9 fiddly ops** for one merge and
still drop refs/access/voice. Imperative reconstruction can't be made reliably
lossless by prompt alone.

## The inversion

`absorb` makes preservation the **default** and loss an **explicit, audited
choice**. Transfer everything structurally; the encoder overrides only what it
deliberately changes ("save the cloning of all data s2consolidate didn't
hand-touch" — Tom).

```
{op: "absorb",
 survivor_id: "<canonical>",
 absorbed_id: "<redundant, must be archivable / unlocked>",
 # ANY survivor field override — SAME shape as revise (all optional):
 content: "<synthesis>", title: "...", confidence: 0.95, situation: "...",
 prune_edges: [...],   # optional: edges NOT to carry
 drop_fields: [...]}   # optional: KV NOT to auto-fill from absorbed
```
No field is mandatory: with none given, `absorb` does a pure structural transfer
(refs/edges/access/KV) and archives. The shipped Python signature mirrors this:
`absorb(survivor_id, absorbed_id, content=, reason=, updates=, **field_kwargs, prune_edges=, drop_fields=)`.

### Structural transfer policy (no LLM involvement)

| Dimension | Default | Override |
|---|---|---|
| `source_refs` | union(survivor, absorbed) | — |
| edges | re-point absorbed's externals → survivor, **dedup**, drop the intra-pair edge | `prune_edges` |
| `access_count` | **sum** (usage history is additive) | — |
| KV (situation, reasoning, voice quotes, emergent) | **fill where survivor lacks it**; survivor wins on conflict | `drop_fields` |
| content + ANY mutable field | survivor's own kept | caller overrides via `content` / `updates` dict / field kwargs (revise shape), applied through `revise()` AFTER the auto-transfers so the caller wins |
| `created_at` | keep survivor's | absorbed's recorded in provenance |
| then | `archive(absorbed, archived_survivor_id=survivor)` | — |

### Type constraints (subsumes the locked-absorb rule)

- `absorbed_id` **must be archivable** — never locked/critical. So a locked node
  can only ever be the **survivor** (you absorb *into* it). The "locked is always
  the survivor" rule stops being prompt-discipline and becomes a type error.
- Contradiction is out of scope: `absorb` is for redundancy. Supersession/
  correction routes to the correction path (operator escalation), unchanged.

## Implementation sites (✅ done unless noted)

The closed `brain_batch` vocabulary lives in `servers/contract.py` (not
`daemon_dispatch`), and the dispatcher is in `dispatch_write.py`:
- ✅ `servers/contract.py` — `VALID_BATCH_OPS += "absorb"` (the MCP schema enum AND the S2 rejection-table invalid-op detector derive from this frozenset)
- ✅ `servers/dispatch_write.py` — `_handle_brain_batch` `absorb` branch; every non-control key is forwarded to `absorb()`'s `updates` (revise-op style)
- ✅ `servers/brain_mcp.py` — `brain_batch` description documents `absorb` (and the stale "five ops" count is fixed)
- ✅ `servers/brain_remember.py` — `brain.absorb()` composes existing DAL: `get_source_refs`/`add_source_refs`, `get_connections_bulk`/`add_relation`, `MetadataDAL.get_all_bulk`/`set_many`, `revise` (field overrides + re-embed), `archive_node` (provenance). No commit kwargs — sub-writes self-gate on `conn.in_batch` (post-F3 idiom); ends with `_maybe_commit()`.
- ✅ `tests/test_absorb.py` — atomicity, each transfer dimension, dedup, locked-as-absorbed rejection, op-path
- ⬜ `servers/scales/s2/consolidation_enrichment_prompt.py` — rewrite CONSOLIDATE/EVOLVE + the three edge-migration sections down to "use `absorb`" (**NEXT SESSION**)

## The gate

A reusable **information-preservation probe** audits any `absorb` across all seven
dimensions on a richly-populated fixture (peer carrying source_refs, voice quotes,
an emergent KV field, real access_count). `absorb` must score lossless before it
activates, and before any live merge.

## Sequencing & coordination

1. ✅ DAL cleanup landed (`061cb40`) — `absorb` was re-applied fresh against the
   post-F3 `conn.in_batch` commit idiom (the old `_batch_mode`/`commit=` kwargs are gone).
2. ✅ Built op + DAL transfer + tests (`d3a0fa1`). ⬜ preservation-probe gate →
   prompt rewrite → eval (NEXT).
3. ⬜ **Live lossless merge:** operator unlocks `96d2fdf8` → `absorb` into
   `426ae3cd` → archive. Validates the whole pipeline end-to-end.

## Lineage

- Generalizes `a4a363aa` (edge migration at the encoder layer) — edges were the
  *first* dimension migrated; `absorb` migrates **all** of them, structurally.
- Implements `b19b2e5c` (survive-and-absorb) as a primitive instead of a
  decomposed op sequence.
- Makes `002ccc1e` (locked is always the absorb target) a type constraint.
