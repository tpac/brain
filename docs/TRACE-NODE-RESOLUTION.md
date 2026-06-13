# Historical-id → Live-node Resolution — the survivor-redirect contract

**Status:** design locked (2026-06-13). Implementation pending. Motivated by
the `spread_seed_no_vectors` incident class and the `discussed`-anchor
archived leak (brain nodes `232fa024`, `01c687bc`, `9c1f5b20`).

> **The rule, in one line:** a node id that came from *history* (a trace, a
> surface-output log, a prior S2 decision, an LLM re-selection) is **not** a
> live node id. Any path that turns such an id into a node MUST resolve it
> forward to the **live** node — returning the living descendant of what was
> referenced, or nothing. One centralized resolver; bespoke `resolve_id` /
> `get_bulk` on a history-sourced id is forbidden.

---

## 1. Why this exists

Traces are **immutable**; nodes are **mutable**. A `surface_selected` event
written weeks ago literally holds the node ids surfaced *that turn*. Since
then, S2 consolidation has absorbed duplicates (A merged into survivor B, A
archived). So every history→node path eventually dereferences an id whose
node is now archived. Two sub-cases:

1. **Absorbed** — A was merged into a live survivor B. The right answer is **B**
   (the living form). Today nothing follows the link, so the path either leaks
   the dead A or drops it (recall loss — "the thing we discussed 3 weeks ago"
   returns nothing, when its descendant is right there).
2. **True orphan** — A archived with no survivor. The right answer is **nothing**.

### What the merge actually leaves behind (verified 2026-06-13)

Audited the live brain, not assumptions:

- **340** archived nodes carry a survivor pointer — but only as a hidden
  metadata key `_sys_archived_survivor_id`, which is `_sys_`-prefixed (never
  rendered to LLMs), **not** the `resolved_by` column (that means something
  else — §5), and **read by no code path**.
- **No graph edge** records the merge. Sampled 8 archived-with-survivor nodes:
  7 had no edge to their survivor; the 1 that did was `co_anchored` (episodic,
  not a merge mark). Consolidation archives via `archive_node` directly
  (`consolidation_decoder.py:165`); the `brain_batch` `absorb` op deliberately
  *drops* the absorbed↔survivor intra-edge. Either way the survivor link never
  becomes traversable.
- **Chains are real**: `37cb583d` is the survivor of `d44bb207` *and* is itself
  absorbed into `28e92124` (A→B→C). Resolution must follow transitively.
- A tempting red herring: node `ddf366af` *does* have a `corrects` edge to its
  survivor `19ddb548` — but that edge's `encoding_source` is empty (made in
  conversation, two burial-mechanism nodes correcting each other), **not** a
  consolidation artifact. The general case has no such edge.

```
Trace (immutable, JSON id-list)         resolve_live walks the merge edge →
   │ "surfaced ddf366af"                  ┌──────────────────────────────┐
   ▼                                       ▼                              │
ddf366af ── ARCHIVED ──(absorbed_into)──▶ 19ddb548 ── ACTIVE ✓ (return)  │
                                          (and chains: B→C if B archived)─┘
```

The links are invisible — no trace↔node edge (it's an id-list), no
absorbed↔survivor edge — and that invisibility *is* the bug.

---

## 2. The contract (mandatory)

1. **History-sourced ids are resolved, never trusted.** Any id obtained from a
   trace, surface-output log, `source_refs` expansion, prior S2 decision trace,
   or LLM re-selection passes through the centralized resolver before it
   becomes a candidate, an edge endpoint, a render, or anything an LLM sees.
2. **Resolve forward, don't drop.** Absorbed → return the survivor (follow the
   chain to its live terminal). Only true orphans are dropped.
3. **Quiet by default.** A redirect or orphan-drop is normal retrieval, not an
   incident — like `recall` silently not returning archived nodes. The one loud
   ERROR is the **tripwire**: an archived id reaching an LLM-facing output
   *after* the resolver should have run — that's a producer that skipped the
   resolver (a code bug), should-never-fire.
4. **One resolver.** Centralized, the way `get_node()` is the one canonical rich
   pull. No bespoke resolution on history-sourced ids.

---

## 3. The survivor link is a first-class edge (the design)

**Decision (locked):** the absorbed→survivor link becomes a dedicated graph
edge, **not** a metadata column. New relation **`absorbed_into`** (source =
absorbed/dead node, target = survivor/live node), added to the
**`correction_improvement` aspect** in `aspects_v1.json`.

Why an edge in that aspect, not a column:

- **It shares the correction substrate's vocabulary** — but NOT "for free."
  CORRECTION (caught by `ae6e7d8d`, 2026-06-13): `correction_enrich` calls
  `get_connections_bulk(include_neighbor_archived=False)`, which forces
  `n1.archived=0 AND n2.archived=0`. An `absorbed_into` edge ALWAYS has an
  archived endpoint (the absorbed source), so the *standard* correction walk
  filters it out — it will NOT surface via plain `correction_enrich`. Surfacing
  it requires a reader that passes `include_neighbor_archived=True` for
  `absorbed_into` (the param already exists; it's just off by default). That
  reader is `resolve_live`'s edge-source (Phase 4) / a correction_enrich
  variant. So the aspect membership is the right *classification*, but the
  surfacing is a deliberate reader change, not automatic.
- **It's visible and traversable.** Spread activation can follow it; the
  dashboard can show it. A link nobody can see is what caused this incident.
- **It's Tom's "cluster with certain aspects"**: resolution finds the survivor
  by walking an aspect-tagged edge.
- **Dedicated relation, not generic `corrects`/`supersedes`.** `ddf366af` proved
  the ambiguity — a node can have several correctors but exactly one absorb
  survivor. `absorbed_into` says unambiguously "this is the node I became."

Writers (single source): both the `brain_batch` `absorb` op and the
consolidation archive path write the `absorbed_into` edge when archiving-via-
merge. The `absorb` op stops dropping that one intra-edge. `_sys_archived_survivor_id`
becomes redundant (kept as a write-time audit breadcrumb / backfill source).

`resolved_by` is **still off-limits** — it already means "the decision/rule that
resolved this open/task" (`schema.py:143`); a different concept.

---

## 4. The centralized primitive — `resolve_live()`

Home: `NodeDAL` (beside `archived_subset`, `get_bulk`), exposed on `brain`.

```
resolve_live(ids, *, aspects=None, on_orphan='drop', max_hops=8) -> ResolveResult
```

- **Walk the chain.** For each id: live → keep. Archived → follow the outgoing
  `absorbed_into` edge to the next node; repeat to a live terminal, an orphan
  (no `absorbed_into`), or a cycle / `max_hops` cap. Visited-set cycle-safe.
- **Dedup.** Many history ids collapse to one survivor — each live node once,
  first-seen order.
- **Orphans.** `on_orphan='drop'` (default) omits; `'mark'` returns them flagged
  for callers that need provenance (audit/dashboard).
- **Cluster-with-aspects (Q1, locked = bare default + opt-in).** `aspects=None`
  returns just the resolved live nodes (cheapest — this runs on the recall hot
  path). A consumer that feeds Anchor passes e.g.
  `aspects=['correction_improvement']` to get each survivor enriched with that
  aspect's neighbors, the same shape `get_node` attaches `_corrections`. Default
  bare keeps the hot path light; enrichment is the caller's explicit choice.

Returns `ResolveResult{ live, redirected: {input_id→survivor_id}, orphans }`.

---

## 5. Call-site map — every history→node path

Audited 2026-06-13. LIVE/LEAK rows = required migration to `resolve_live`.
SAFE rows already filter archived but should route through the resolver to gain
survivor-redirect (recall quality) under one contract.

| # | Site | file:line | id source | today | required |
|---|------|-----------|-----------|-------|----------|
| 1 | `recall_by_time` **discussed** | `scales/s1/fetch_tools.py` (anchor + source gate) | `surface_selected` traces (JSON ref_id) | drops archived + **loud ERROR** (this-session stopgap) | **resolve_live** — redirect to survivor; drop only orphans; de-noise |
| 2 | `recall_by_time` **event** | `fetch_tools.py:507` | `entity_dates` (DAL) | archived filtered (`n.archived=0`) ✓ | route through resolver for survivor-redirect |
| 3 | `_get_recently_surfaced` (surface dedup) | `scales/s1/surface.py:21-55` | `surface_selected` traces → `get_title` | **no liveness** — archived titles enter dedup set | resolve_live — track survivor, not dead id |
| 4 | Hebbian drain | `daemon_hooks.py:473-525` | surfaced-ids tmp file → `resolve_id` | archived dropped + loud (bc34734d) | resolve_live — strengthen edges on survivor |
| 5 | Surface outside-candidate recovery | `surface.py:716-741` (`_drop_archived_selected`) | Haiku id not in menu → `resolve_id` | archived dropped + loud (bc34734d) | resolve_live — absorbed id → its survivor (this is the `spread_seed_no_vectors` self-heal) |
| 6 | S1 Scribe catalog | `scales/s1/encode.py:462-475` | `recalled_raw` surface output → ids | **no liveness** | resolve_live — encoder sees live survivors, not dead ids it then tries to revise |
| 7 | S2 community decoder | `scales/s2/community_decoder.py:189` | own S2 traces (ref_id) | reads ids (suppression) | **deferred** (Q3) — audit only |
| 8 | S2 consolidation decoder | `scales/s2/consolidation_decoder.py:704` | own S2 traces (selected_ids) | reads ids | **deferred** (Q3) — audit only |

**Out of scope:** `source_refs` is *node → trace* provenance (node-anchored), not
a candidate feed. Listed to avoid confusion with the above.

---

## 6. Severity

- **Redirect performed** → no log (normal retrieval).
- **Orphan dropped** → optional low-severity/WARN, counted (informative during
  the post-incident recovery window).
- **Tripwire** (archived id reaches an LLM-facing output *after* resolution) →
  **ERROR, should-never-fire** — a producer bypassed `resolve_live`.

Supersedes the blanket "every archived sighting is a loud `_log_error`" stopgap:
loud belongs on the contract violation, not the routine redirect.

---

## 7. Decisions (locked 2026-06-13)

- **Q1 cluster-with-aspects:** bare live nodes default (`aspects=None`),
  enrichment opt-in. (Recall hot path stays light.)
- **Q2 survivor link:** first-class **`absorbed_into` edge** in the
  `correction_improvement` aspect (not a column, not `resolved_by`, not generic
  `corrects`). §3.
- **Q3 S2 decoders (#7/#8):** deferred — they read their own past suppression
  decisions; whether S2 should chase a moved target is an S2-semantics call,
  not load-bearing for recall. Audit, don't migrate now.

---

## 8. Implementation phases

**Status 2026-06-13:** Phase 3 primitive LANDED (resolve_live merged to main,
`20c2951`, read-only, 13 tests). Phase 2 backfill EXECUTED for the clean set
(216 `absorbed_into` edges written via the daemon, verified, 0 failures —
`eval/oracle_audit/backfill_absorbed_into.py`; brain.db backed up first).
The edges are LAID BUT INERT: `absorbed_into` is not yet in the
`correction_improvement` aspect (so `correction_enrich` doesn't walk them) and
`resolve_live` still reads the metadata pointer, not the edge — so there is NO
behavior change yet, and no leak (spread's `get_connections_bulk` defaults
`include_neighbor_archived=False`, so the archived endpoints don't surface).

Phase 1 CODE COMPLETE (`ae6e7d8d`, branch `claude/absorbed-into-ae6e7d8d`):
`absorbed_into` written in `archive_node` gated on `extra['survivor_id']` (single
chokepoint) + step-3 reaper exemption + aspect SEED + voice merge-append
(distinct `user_raw_quote`/`anchor_raw_quote` → appended, not survivor-wins-drop).
26 tests green. Pairs with the dangling-edge-reaper exemption (`4c971da`, main).
**Everything stays INERT until a deliberate daemon restart** (writers + reaper
exemption go live together; backfill must be RE-RUN after, since the old-code
Healer may have reaped the first 216).

Remaining = the **reader phase (was "Phase 4"), bundle these together**:
(i) add `absorbed_into` to the LIVE aspects working copy
`$BRAIN_DB_DIR/aspects_v1.json` (deliberate one-line, NOT left to S2
classifier — production mutation, supervised);
(ii) `resolve_live` edge-source: read the `absorbed_into` edge via
`include_neighbor_archived=True` (its seam already isolates the pointer read);
(iii) surface absorbed_into despite the archived endpoint — same
`include_neighbor_archived=True` query (the standard `correction_enrich` walk
filters it out — see §3 correction); (iv) migrate sites §5 #1–#6; (v) finish the
deferred backfill (~23 chains + clean the 94 stale stamps — verdict: one-time
recovery artifact, delete `_sys_archived_*` keys, no stamping bug).

1. **Edge + writers (live-wiring).** Add `absorbed_into` to
   `correction_improvement` in `aspects_v1.json` (this is what makes the 216
   backfilled edges live — `correction_enrich` starts surfacing them; restart
   to take effect).
   - **Writer chokepoint** (better than first draft — found by `ae6e7d8d`):
     centralize the edge-write inside `archive_node`, gated on
     `extra.get('survivor_id')`. That's the real convergence of the absorb op
     AND any consolidation merge — single source. Don't touch absorb's
     migration loop (its intra-edge drop stays; a fresh canonical
     `absorbed_into` is written instead). Direction: source = absorbed/dead,
     target = survivor/live.
   - **CRITICAL — exempt `absorbed_into` from edge-reaping (two sites).**
     `absorbed_into` intentionally bridges an archived node to a live one, so
     any sweep that archives edges-touching-archived-nodes will reap it and
     silently kill the redirect. Both must exempt it:
     (a) `archive_node` step-3 edge soft-archive (else B-absorbed-into-C
     re-archives the live A→B edge, breaking the chain) — `ae6e7d8d`'s slice;
     (b) `GraphDAL.archive_dangling_edges` (the Healer reaper) — **SHIPPED**
     `9711e04`-onward (`dal.py`, `AND er.relation != 'absorbed_into'`).
     Until the daemon restarts with (b), the live Healer still reaps the 216
     backfilled edges — so they are NOT durable until restart; **re-run the
     backfill after the wiring restart** (idempotent).
   - Fold in the **voice fix** here (independent of `bc34734d` — that was a
     stream/session id that filed a *finding*, never a commit; implement fresh):
     in absorb's metadata fill (`brain_remember.py` ~442-445), merge-append
     distinctive `user_raw_quote`/`anchor_raw_quote` from the absorbed peer
     instead of survivor-wins-drop; + a voice-preservation rule in
     `consolidation_enrichment_prompt` parallel to the numbers/dates rule.
2. **Backfill — remaining + census.** DONE: 216 clean 1-hop archived→live.
   TODO: ~23 multi-hop chains (survivor itself archived → run the
   resolve_live-based pass to terminal) and the **94 live-with-stale-stamp**
   clean (pending `ae6e7d8d`'s revert-artifact-vs-stamping-bug verdict; if a
   bug, fix the stamp-on-revert clear, don't just one-time-clean). Empirical breakdown — `resolve_live` run over all 340
   `_sys_archived_survivor_id` nodes (stream `ae6e7d8d`, 2026-06-13):
   - **225** archived → live survivor → backfill these into `absorbed_into` edges.
   - **94 LIVE nodes carrying a stale stamp** — reverted-archive artifacts from
     the June-12 recovery (archived, then restored, stamp never cleared). **The
     stamp is NOT a reliable archived-signal.** Backfill must filter
     `archived=1` only; these 94 stamps should be CLEARED. If archive-revert
     doesn't clear `_sys_archived_survivor_id`, that's a live bug minting stale
     stamps — fix it, don't just one-time-clean (under investigation by `ae6e7d8d`).
   - **21** orphans (no live terminal: missing target / archived-no-pointer /
     cycle) → accept loss, log.
   Net: backfill ~225, clean 94, log 21. `resolve_live` itself is robust to all
   three (it checks `archived` first, so a live-with-stale-stamp passes through
   live with no redirect).
3. **Primitive.** `NodeDAL.resolve_live()` + tests (chain A→B→C, cycle, orphan,
   dedup, aspect-cluster). Backed by the edge, `_sys_` key as fallback for any
   un-backfilled rows.
4. **Migrate** sites §5 #1–#6; replace this-session's drop-and-loud-gate in
   `recall_by_time` with quiet redirect. Tripwire (§6) stays as the post-
   resolution backstop. Audit #7/#8.
</content>
