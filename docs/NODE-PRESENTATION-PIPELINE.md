# Node Presentation Pipeline — unifying id → live → data → render

**Status:** design draft (2026-06-15). Planning a new task. Supersedes the
ad-hoc per-site liveness patches; **extends** `TRACE-NODE-RESOLUTION.md`
(`resolve_live` is stage 2 of the pipeline below). Motivated by Tom's
observation: *"we're duplicating other paths that are basically just different
formatters … the original meaning is really the same semantically."* Also closes
the long-open audit `71fbf649` ("find and merge recreated get_rich_node
equivalents") and the deferral `eaf63b60` ("land and migrate formatters as
follow-up").

> **The rule, one line:** turning node id(s) into text for an LLM is ONE
> operation with three ordered stages — **resolve-live → fetch → render**.
> Every LLM-facing consumer composes the *same* stages via one door,
> differing only by a format config and a fetch mode. No bespoke per-site
> resolve/fetch/render.

---

## 1. The problem — one semantic operation, fragmented concerns

"Show these node ids to an LLM" is implemented ad-hoc at ~a dozen sites. Each
site re-derives some subset of: make the id live (archived→survivor), fetch
its data, format it. Where a site skips a stage, that
stage's bug class appears there — which is exactly why the archived-id leak shows
up at *some* sites and not others. The leak is a symptom; the disease is that
the stages aren't a single pipeline.

The concerns, each with a canonical form AND scattered/bespoke instances:

### Concern 1 — ID resolution
Node ids are exact 8-char hex everywhere; there is no short→full resolution
step (the id-resolution unification, 2026-08-31 — brain node id:10a535fd).
The only tolerated recovery is surface's menu-scoped `_unique_prefix_match`
(uniqueness-required, logged) plus the leading-zero retry — both reconstruct
an exact id and announce themselves. A miss is the owning brain method's to
report; nothing pre-resolves ids at the dispatch layer.

### Concern 2 — Liveness (archived → survivor, or drop)
| Mechanism | Sites |
|---|---|
| `archived_subset` (drop archived) | `recall_write_queue.py:419` (hebbian), `fetch_tools.py:431` (edge endpoints), `fetch_tools.py:852` (execute_tool tripwire), `surface.py:614` (selection gate) |
| `resolve_live` (forward to survivor) | `fetch_tools.py:597` (discussed anchor) — **only caller** |
| `_drop_archived_selected` (selection gate) | `surface.py:794` |
| **MISSING liveness (the leaks)** | `surface.py:52` (dedup `get_title`), `encode.py:411` (encoder timeline `get_title`) |

### Concern 3 — Fetch (id → data)
| Path | Sites |
|---|---|
| `brain.get_node` (canonical rich, correction-enriched) | **26 callers / 12 files** — `fetch_tools.py`(7), `dispatch_read`(5), `surface_contract`(3), `community_encoder`(2), `encode_contract`(2), +1 each in `surface`, `healer_decoder`, `consolidation_encoder`, `scouts/muster`, `daemon_hooks`, `brain_recall`, `dal` |
| `get_title` (bespoke title-only) | `surface.py:52` (dedup), `encode.py:411` (encoder timeline), `brain_connections.py:254` (bridge, S2), `brain_remember.py:1507` (mark-critical existence check) |

### Concern 4 — Render (data → text)
Canonical: `render_rich_node(node, FORMAT)` — callers: `brain_mcp.py:867`,
`brain_voice.py:89`, `consolidation_encoder.py:367`, `community_encoder.py:431`,
`encode_contract.py:131`, `surface_contract.py:304/1796/1820`.

Format configs (~11): `GET_NODES_BALANCED_FORMAT`, `GET_NODES_COMPACT_FORMAT`
(`contract.py`); `SURFACE_ARC_FORMAT`, `SURFACE_FACT_FORMAT`,
`SURFACE_BACKGROUND_FORMAT`, `HAIKU_FORMAT` (`surface_contract.py`);
`S2CE_NODE_FORMAT`, `S2CE_COMMUNITY_FORMAT` (`community_contract.py`);
`CONSOLIDATION_NODE_FORMAT` (`consolidation_contract.py`); `S1_NODE_CONFIG`
(`encode_contract.py`); `MCP_FORMAT` (`brain_voice.py`).

**Bespoke inline render (the drift):**

| Site | Bespoke render | Stage(s) skipped |
|---|---|---|
| dedup block `surface.py:54` | `{id, title}` via `get_title` | resolve-live, render |
| encoder timeline `encode.py:412` | `'%s ("%s")'` via `get_title` | resolve-live, render |
| recall_by_time edge tier `fetch_tools.py:444` | `src[:8] → tgt[:8] : rel` | render (and embeds archived endpoint prefixes) |

*(Out of scope: `sel_detail`/`cand_detail` `[:8]` strings in `surface.py:482-496`
are TRACE/dashboard metadata, not LLM-facing — they don't go through the door.)*

---

## 2. How it connects to recall

Recall is the **producer** of node ids; the presentation pipeline is the
**consumer**. The crucial asymmetry (the `SEARCH vs FETCH-BY-ID` dichotomy,
`121bbbfb`):

```
 PRODUCERS                                         CONSUMER
 ─────────                                         ────────
 brain.recall() ───► 25 candidates (LIVE,  ─┐
   (cosine + FTS, archived=0 at SQL)         │
 fetch tools (topical/by_time/verbatim) ────┤
   results, execute_tool tripwire            ├──►  PRESENTATION PIPELINE
                                             │      resolve-live
 trace reads (surface_selected) ────────────┤        → fetch → render
   dedup / discussed / encoder timeline      │
   (HISTORY-sourced, STALE — no SQL filter) ─┘
```

- **Recall-sourced ids are already live** (filtered in SQL at the lane). They
  flow through the pipeline's resolve-live stage as a no-op passthrough.
- **History-sourced ids are stale** (live when the trace was written, archived
  since). The pipeline's resolve-live stage is the **equalizer** — it gives
  history ids the same liveness contract recall ids already have, so by the time
  anything renders, both streams are identical.

So the connection is: **the pipeline is the single output stage both recall
results and history-sourced ids converge into, and `resolve_live` is what makes
the two streams interchangeable.** Recall stays the producer; it does not need
to change. (This is also why baking liveness into `get_node`/`get_title` is
wrong — a fetch primitive can't tell a recall id from a history id; the
producer/door owns the policy.)

---

## 3. The design — one door, three composable stages

```
brain.present(ids, *, fmt, mode='rich', on_orphan='drop') -> List[str] | Dict[id,str]
    live = resolve_live(ids,           # stage 1: history→survivor, drop orphans
                        on_orphan)     #          (no-op for already-live recall ids)
    data = (get_node(live) if mode == 'rich'      # stage 2: fetch
            else get_titles_bulk(live))           #          lean = title-only
    return render(data, fmt)           # stage 3: render_rich_node per config
```

- **One entry point** for LLM-facing "show these ids." Callers choose `fmt`
  (existing configs) + `mode` (`rich` = full `get_node`; `lean` = a new batched
  `get_titles_bulk`, for {id,title} cases like the dedup block).
- **Stages stay individually public** (`resolve_live`, `get_node`,
  `render_rich_node`) for partial needs. The door *composes* them for the
  common case; it does not replace the primitives. (There is no short→full
  stage: node ids are exact 8-char hex — the id-resolution unification.)
- **`get_node` stays archived-inclusive** (explicit-id lookups / provenance
  unchanged). Liveness lives in the door's stage 2, not the fetch primitive.
- **`get_title` (per-id) is retired** as an LLM-facing fetch; lean mode uses
  `get_titles_bulk`. Non-LLM callers (`mark-critical` existence check, S2
  bridge) keep direct primitives.

**The invariant to lock:** a node id that will be shown to an LLM goes through
`present()`. New consumers pick a config + mode; they never hand-roll
resolve/fetch/render. That is what stops the next formatter from re-leaking.

---

## 4. Migration plan (phased by value/risk)

| Phase | Work | Value | Risk |
|---|---|---|---|
| **1** | Build `present()` + `get_titles_bulk` (lean) + tests. `resolve_live` already shipped; `get_node`/`render_rich_node` already exist. | foundation | low (additive) |
| **2** | Migrate the two leak sites — dedup block (`surface.py`) and encoder timeline (`encode.py`) — onto `present(..., mode='lean')`. **Closes the every-turn `surface_selected_archived` leak.** | **high** | low |
| **3** | Migrate `recall_by_time` edge tier to the door (kills the `src[:8]→tgt[:8]` inline render + the archived-endpoint-prefix residue). | med | med (edge shape) |
| **4** | Route already-canonical sites (fetch-tool results, surface candidates, S2 encoders) through `present()` for symmetry — they already render correctly, so this is consolidation not bugfix. | low | low |
Phases 1–2 are the leak fix done *right* (one door instead of N patches).
Phases 3–4 are pure de-fragmentation and can land incrementally. (A former
Phase 5 — unifying the short→full idiom — was resolved by subtraction: the
id-resolution unification deleted the idiom and its resolver outright.)

---

## 5. Open decisions (need Tom)

1. **Door granularity** — one `present()` composing all three stages (recommended), vs. mandating the sequence with separate calls at each site. Recommend: `present()` for LLM-facing; primitives stay public for partial use.
2. **Lean mode** — add `get_titles_bulk` for {id,title} cases (recommended, avoids paying `get_node` enrichment on a 20-id/turn dedup hint), vs. always-rich `get_node` (simpler, wasteful).
4. **`present()` home** — `Brain` method (alongside `get_node`) vs. `contract.py`/`pipeline_contract.py`. Recommend `Brain` (it orchestrates DAL stages, like `get_node` does).
5. **Eval gate** — recall quality is downstream; run `eval/surface_funnel.py` / frame replay before/after Phase 2 to confirm no regression (the dedup block feeds Haiku every turn).

---

## 6. Relationship to existing docs

- `TRACE-NODE-RESOLUTION.md` — defines stage 2 (`resolve_live`) and the
  history-id contract. This doc generalizes it: resolve-live is one stage of three, and the
  "8 sites" there are a subset of the consumers that route through `present()`.
- The format-config vocabulary (Concern 4) and `render_rich_node` are the
  `get_rich_node`/`format_node` unification (`3c3a3046`, `a18b7abf`). This
  finishes that arc by routing the last bespoke formatters onto it.
