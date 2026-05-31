# S2 `absorb` — a first-class lossless merge primitive

**Status:** design / spec. Not built. Blocked on the in-flight DAL cleanup
(`2be1295e`, write-path rewrite). Sequenced as Track 2 of the locked-absorb work.

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
 content: "<the deliberate synthesis>",   # ONLY mandatory hand-written field
 prune_edges: [...],                       # optional: edges NOT to carry
 drop_fields: [...]}                       # optional: KV NOT to carry
```

### Structural transfer policy (no LLM involvement)

| Dimension | Default | Override |
|---|---|---|
| `source_refs` | union(survivor, absorbed) | — |
| edges | re-point absorbed's externals → survivor, **dedup**, drop the intra-pair edge | `prune_edges` |
| `access_count` | **sum** (usage history is additive) | — |
| KV (situation, reasoning, voice quotes, emergent) | **fill where survivor lacks it**; survivor wins on conflict | `drop_fields` |
| content | explicit override (prose can't auto-merge) | required |
| `created_at` | keep survivor's | absorbed's recorded in provenance |
| then | `archive(absorbed, archived_survivor_id=survivor)` | — |

### Type constraints (subsumes the locked-absorb rule)

- `absorbed_id` **must be archivable** — never locked/critical. So a locked node
  can only ever be the **survivor** (you absorb *into* it). The "locked is always
  the survivor" rule stops being prompt-discipline and becomes a type error.
- Contradiction is out of scope: `absorb` is for redundancy. Supersession/
  correction routes to the correction path (operator escalation), unchanged.

## Implementation sites

Adding an op to the closed `brain_batch` vocabulary touches:
- `servers/daemon_dispatch.py` — `VALID_OPS` in `_handle_brain_batch` + dispatcher if/elif
- `servers/brain_mcp.py` — the `brain_batch` schema enum
- `servers/dal.py` (+ `brain_remember.py`) — the transfer logic: source_ref union,
  edge re-point/dedup, access_count sum, KV fill, then archive with survivor provenance
- `tests/` — atomicity, each transfer dimension, dedup, locked-as-absorbed rejection
- `servers/scales/s2/consolidation_enrichment_prompt.py` — rewrite CONSOLIDATE/
  EVOLVE + the three edge-migration sections down to "use `absorb`"

## The gate

A reusable **information-preservation probe** audits any `absorb` across all seven
dimensions on a richly-populated fixture (peer carrying source_refs, voice quotes,
an emergent KV field, real access_count). `absorb` must score lossless before it
activates, and before any live merge.

## Sequencing & coordination

1. **DAL cleanup lands first** (`2be1295e` — BatchAwareConnection F3 rewrites
   `dal.py`/`brain_remember.py`/dispatch). Building `absorb` concurrently collides.
2. Spec (this doc) → build op + DAL transfer + tests → preservation probe gate →
   prompt rewrite → re-eval.
3. **Live lossless merge:** unlock `96d2fdf8` → `absorb` into `426ae3cd` →
   archive. Validates the whole pipeline end-to-end.

## Lineage

- Generalizes `a4a363aa` (edge migration at the encoder layer) — edges were the
  *first* dimension migrated; `absorb` migrates **all** of them, structurally.
- Implements `b19b2e5c` (survive-and-absorb) as a primitive instead of a
  decomposed op sequence.
- Makes `002ccc1e` (locked is always the absorb target) a type constraint.
