# edge_context invalidation — restore the revise/edge symmetry

**Status 2026-09-13:** cause diagnosed and measured; the first fix was built, reviewed,
and **reverted** as the wrong layer. The principled fix (below) is approved and **not
started**. Backlog untouched.

| | |
|---|---|
| branch | `claude/episodic-role-errors-branches-9915a8` |
| address | the **branch**, not a head hash — hashes rot. `git log --oneline main..HEAD` for the arc; merge main first (it was 1 ahead of the branch point at handoff) |
| reverted attempt | `330e3c8`, reverted by `80fe7a2` — read it for what was tried |
| unrelated, keep | `9b9da13` (haiku_id_outside_candidates → warning) |
| deployed? | **no.** Daemon runs `/Users/tpac/brain` (main). Nothing here is live. |

## The bug

A node's `edge_context` vector is a blob of its own top-5 edge descriptions
(`ORDER BY weight DESC LIMIT 5`, `GraphDAL.get_edge_descriptions_for`). It is one of the
6 live LAF maxsim views at weight 0.55 — **the only lane that retrieves a node for what
its relationships say.** Per-relation vectors (`edge_relations.embedding`) are healthy but
feed `select_edges` and `_build_edge_coeffs`: presentation and spread, not retrieval.

Nothing refreshes `edge_context` when a node's edges change. A node is embedded *before*
its edges exist; `add_relation` enqueues the **edge**, never the endpoint **nodes**. All
three `embed_queue.enqueue()` call sites are node writes.

The coverage sweep cannot catch it: `vector_coverage_sweep` → `backfill_vectors` →
`find_missing` repairs **missing** rows, and a stale row is not a missing one.

**Measured on production (2026-09-13, pre-fix):**

| | |
|---|---|
| `edge_context` vectors | 9,387 |
| older than at least one qualifying relation | 5,593 (60%) |
| nodes genuinely stale (qualifying edge: real description, not `community_member`) | **3,519** |
| of those, would embed *different* text (sampled, untruncated rows only) | **75%** → ~2,600 |

**These figures grow.** Every new described edge stales two more endpoints — roughly 1,092
distinct endpoint nodes per week at current rates (2,670 edges/week). A larger number on
re-measure is the leak still running, not a regression and not a bad original measurement.

Caveat that cost a wrong number once: `node_enrichments.text` is `text[:500]` — a truncated
**debug copy**, not the embedded text. 35% of rows sit at that cap, so containment checks
against `text` are only valid on rows under 499 chars. The timestamp measure is unaffected.

## Root cause — a drift, not an oversight

Vector invalidation is a **brain-level** operation:
`revise()` → `vectors_affected_by(field)` → `_vec_dal.delete_for_node(...)`.

`revise_edge`'s own docstring claims the same contract — *"Mirrors revise()'s contract"*,
*"embedding is a brain-layer concern"* — then delegates the whole job down to
`add_relation` / `rename_relation` inside `GraphDAL`.

That was harmless while an edge only invalidated **its own** embedding, which is a column on
the row the DAL was already writing — storage-local, so the DAL could legitimately self-invalidate.

`edge_context` is the first vector where an **edge write invalidates a vector on a node**.
It crosses the entity boundary, and cross-entity invalidation needs the brain layer.
`GraphDAL.__init__` takes only a `conn` — no brain, no cache-aware `_vec_dal`.

## The two doors

`connect` / `connect_typed` / `revise_edge` (`brain_connections.py`) are the intended
brain-level edge doors. Five writers bypass them and call the DAL directly — all five write
descriptions long enough to feed `edge_context`, so all five need invalidation:

| site | edge written |
|---|---|
| `brain_remember.py:482` | `absorbed_into` on archive |
| `brain_remember.py:814` | edge migration during absorb |
| `brain_remember.py:1389` | `co_anchored` (source_refs siblings) |
| `brain_remember.py:1887` | `co_anchored` (second site) |
| `dal_graph.py:624` | `community_member` — needs its own ruling (excluded from edge_context, so arguably should NOT invalidate) |

## The decision

**Option 1 — restore the symmetry.** Invalidation moves into the brain-level edge doors;
the back doors get routed through them. Tom's call, taken in a dedicated session.

**Option 2 (declined) — the queue carries `stale_types`, worker deletes.** Works regardless of
door and is cheaper, but it accepts the two-door problem instead of fixing it. Critically:
**if Option 1 is landing, Option 2 must NOT also land** — two invalidation mechanisms for one
dependency is the exact drift being fixed. This is why `330e3c8` was reverted rather than kept
as a stopgap.

The backlog migration (a versioned `_migrate_vN` in `schema.py` deleting `edge_context` rows
older than a qualifying relation, letting the sweep rebuild them as missing) is
**shape-independent** — but land it *with* the cause fix, or the backlog re-accumulates.

## What Option 1 inherits from the review of the reverted attempt

Option 1 pre-solves the in-batch race, the `NOT IN ()` footgun, the per-sweep subquery cost,
and the shared-query special case. These remain live and are **not** solved by changing layer:

- **`rename_relation`** — a relation renamed across the `community_member` boundary changes
  eligibility. Deletion handles it cleanly (no timestamps involved); just remember the site.
- **Eligibility mirror** — invalidate only for relations that can actually feed the text
  (non-`community_member`, description > `EDGE_CONTEXT_MIN_DESC_LENGTH`). Otherwise every S2
  community pass re-embeds hundreds of nodes across all 8 vector groups for identical text.
- **Test coverage** — the reverted commit had no test that `add_relation` actually invalidates,
  and none that the sweep *repairs* a stale vector end-to-end. Both gaps should not survive.

## The durable generalization

`EMBEDDING_GROUPS` declares what a vector is **made of**; nothing declares who keeps it
**fresh**. `edge_context` declared `_edge_descriptions`, `vectors_affected_by` mapped it
correctly — and no write path ever called it, because the only caller iterates node kv fields.
A correct declaration with no consumer, and no test asserting it needed one.

Contrast `_emergent` (in `other_meta`): also has no consumer, but that is **deliberate and
pinned** by `test_revise_unknown_field_invalidates_nothing`. The codebase already knows how to
say "this field intentionally invalidates nothing" — `edge_context` just never said either thing.

**The rule worth keeping: any derived artifact whose source lives in a different entity than the
artifact needs an explicitly declared invalidation owner** — because no single write path
naturally owns both ends. kv-sourced vectors never drifted precisely because the node's write
path owns source and artifact together.

**Two layers to enforce it:**

1. **Declaration + contract test.** Add `invalidated_by` to each `EMBEDDING_GROUPS` entry
   (`'revise_kv'`, `'graph_edge_write'`, `'none'`). A test in `tests/test_pipeline_contract.py`
   asserts every group declares one, every named invalidator is reachable, and `'none'` is
   justified by a test. *This would have failed the day `edge_context` was added.*
2. **Runtime invariant probe.** "No vector older than its source" as a periodic guardrail that
   **fails loud** rather than silently repairing. This is the reverted predicate in its correct
   home: it was sound as *detection* and wrong as *repair* — buried inside a repair query it
   healed silently and told no one.

## Verify before trusting anything above

```bash
git -C . log --oneline main..HEAD          # the arc; 80fe7a2 is the revert, not the top
git -C . show 330e3c8 --stat               # the reverted attempt
grep -n "_graph.add_relation(\|graph_dal.add_relation(" servers/brain_remember.py
./dev python3 -m pytest tests/ -k "dal or vector or embed or recall or laf or graph" -q
```

Re-measure the backlog (read-only, via the daemon — never a second writer):

```
eval: brain.conn.execute("SELECT COUNT(DISTINCT ne.node_id) FROM node_enrichments ne WHERE ne.vector_type='edge_context' AND EXISTS (SELECT 1 FROM edges e JOIN edge_relations er ON er.edge_id=e.edge_id WHERE (e.source_id=ne.node_id OR e.target_id=ne.node_id) AND e.created_at > ne.created_at AND er.relation != 'community_member' AND length(COALESCE(er.description,'')) >= 10)").fetchall()
```

## Orientation check — answer these before editing

1. Why was `330e3c8` reverted rather than kept as a stopgap? (Answer: two mechanisms for one
   dependency. If you can't say why that matters, re-read *The decision*.)
2. Which layer will you put invalidation in, and what stops the four `brain_remember` back doors
   from bypassing it?
3. What is your ruling on `dal_graph.py:624` (`community_member`), and does it match the
   eligibility filter in `get_edge_descriptions_for`?
4. Does the backlog migration land in this change or a later one — and what happens if it lands
   first?

## Unrelated but blocking, as of 2026-09-13

`tests/test_deploy_contract.py::TestPublicTreeExport::test_live_tree_exports_clean` **fails on
main** — gate B finds `Tom` in `tests/test_surface_inject_render.py:81,146,221` and
`servers/scales/s1/surface_contract.py:824`, introduced by `92420aa`. Blocks the public-tree
export and the 5.x release path. Not this work-line's to fix; do not widen the allowlist to get
a green run.
