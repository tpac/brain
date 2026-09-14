# edge_context invalidation

`edge_context` embeds a node's top-k edge descriptions by edge weight (`GraphDAL.
get_edge_descriptions_for`; k is the `edge_context` interaction config's `top_k`, default 15, DB-
overridable and inspectable through the interaction tools) with noise-aspect relations excluded. It is
the one LAF lane (weight 0.55) that retrieves a node for what its relationships say. Its source lives
on **edges**; the artifact lives on the **node**.

## The mechanism

Vector invalidation is one brain-level path, `Brain.invalidate_source_fields(node_id, fields,
origin)` → `pipeline_contract.vectors_affected_by` → `_vec_dal.delete_for_node` → `embed_queue.
enqueue`. The embed_queue worker's backfill rebuilds the deleted rows as missing. Two callers:

| source of the text | who notices the change | declared as |
|---|---|---|
| node kv / columns | `revise()` — owns the field write and the delete | `invalidated_by: 'revise_kv'` |
| edge descriptions | `GraphDAL` reports the endpoint via `on_edge_text_changed`; `Brain._edge_text_changed` deletes | `invalidated_by: 'graph_edge_write'` |

`EMBEDDING_GROUPS` declares both halves for every group: `fields` (what the vector is made of)
and `invalidated_by` (who keeps it fresh). `tests/test_pipeline_contract.py` holds every group to
one owner and checks the edge-owned set equals `vectors_affected_by('_edge_descriptions')`.

**Why the DAL reports instead of a brain-level door guarding each writer.** Every edge write
passes through `GraphDAL` — `add_relation`, `rename_relation`, `bulk_archive_relations` (the one
soft-archive flip behind disconnect, node archive, and the dangling sweep) and
`hard_delete_node_edges` (the delete cascade). The DAL owns the text producer and its eligibility
rule, so it alone can say whether a write changed a node's edge text; it holds a hook set once in
`Brain.__init__` and never a brain reference. No caller has to remember anything: the direct
`add_relation` callers in `brain_remember.py` (absorb migration, `absorbed_into`, `co_anchored`) are
covered without a door of their own. A standalone `GraphDAL(conn)` reports to nobody and says so
once on stderr. Inside `bulk_archive_relations` the hook runs under a forced `in_batch` envelope so
the primitive stays commit-free and the vector delete rides the caller's commit.

**Eligibility** has one SQL expression, `dal_graph.edge_context_relation_sql(exclude_relations,
alias)` (description longer than `EDGE_CONTEXT_MIN_DESC_LENGTH`, relation not in the excluded set),
consumed by the producer, `VectorDAL.find_missing` and the write-side reports; `feeds_edge_context`
is its Python twin for rows already in hand. The excluded set is **policy, not the DAL's**: `Brain._edge_context_excluded()` derives it from the
noise aspect and is the one reader — installed on the DAL as a callable (never a copied set, because
the registry rebinds `structural_exclusions` on every adopt), passed to `find_missing` by the backfill
and by the daemon's boot re-queue. A write reports when it creates, revives,
edits or reweights a qualifying row, archives one, renames across the excluded set — or writes a row
that feeds nothing but moves the edge's aggregate weight while a described sibling rides the same
edge (the text is ranked by edge weight, so that reorders it). **Noise relations on their own never
invalidate**: an S2 community pass writing hundreds of `community_member` edges, or an encode writing
`co_anchored` to every sibling, re-embeds nothing.

## The backlog

Schema v33 (`_migrate_v33_edge_context_rebuild`) deletes every `edge_context` row: the text
definition changed with this version (noise excluded, top-15), and until it nothing invalidated the
lane on edge writes, so 4,196 of 9,443 production rows (2026-09-13) embedded an outdated snapshot
anyway. At daemon boot `_enqueue_vector_backfill_gaps` re-queues every node missing `edge_context`
that has a described edge, so the embed worker rebuilds them at its batch size and the coverage
sweep's `embed_coverage_gap` alarm keeps meaning "a writer bypassed the hooks".

## Verify

```bash
./dev pytest tests/test_edge_context_invalidation.py tests/test_pipeline_contract.py -q
```

Re-measure the backlog read-only through the daemon (should reach 0 shortly after the v33 deploy
and stay there):

```
eval: brain.conn.execute("SELECT COUNT(DISTINCT ne.node_id) FROM node_enrichments ne WHERE ne.vector_type='edge_context' AND EXISTS (SELECT 1 FROM edges e JOIN edge_relations er ON er.edge_id=e.edge_id WHERE (e.source_id=ne.node_id OR e.target_id=ne.node_id) AND er.relation != 'community_member' AND length(COALESCE(er.description,'')) > 10 AND (substr(er.created_at,1,19) > substr(ne.created_at,1,19) OR substr(er.archived_at,1,19) > substr(ne.created_at,1,19) OR substr(e.last_strengthened,1,19) > substr(ne.created_at,1,19)))").fetchone()
```

`node_enrichments.text` is `text[:500]`, a truncated debug copy; containment checks against it are
only valid on rows under 499 chars. The timestamp measure is unaffected.

## Open

- **Benchmark the new text.** Noise exclusion and top-15 change what the lane embeds; Tom ruled them
  in without a prior run. Compare `eval/brain_recall_identity_eval.py` before and after the rebuild.
- **Runtime invariant probe** — "no vector older than its source" as a periodic guardrail that
  fails loud. Detection and repair want different homes; this is the detection half, not built.
- The rule this bug earned: any derived artifact whose source lives on a different entity than the
  artifact needs a declared invalidation owner, because no single write path owns both ends.
