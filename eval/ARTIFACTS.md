# Eval Artifacts — Durable Per-Item Bundles for Deep Analysis

**Standard for every eval run, not optional.** When an eval finishes, you should be able to answer "what did the scout extract on turn 4?" or "at what rank did the fact-bearing node land in the top-25?" without re-running. Re-runs are expensive (wall time + API spend) and the brain state at failure-time is gone the moment the per-item brain.db gets cleaned. Artifacts are the durable record.

## Where it lives

- **Code:** `eval/longmem/artifacts.py` (dumper) + `eval/longmem/analyzer.py` (loader/diagnostics, future).
- **Outside `servers/`:** the brain has no eval-specific code. The dumper reads through brain's existing public surfaces (DAL connections) — read-only.
- **Output:** `eval/longmem/reports/{run_name}/items/{qid}/`

## Per-item bundle

After every item completes, the harness writes:

```
eval/longmem/reports/{run_name}/items/{qid}/
├── meta.json            ← qid, axis, dates, question, gold, haystack stats
├── interactions.jsonl   ← every interaction version this brain saw
├── traces.jsonl         ← every trace_event (S0/S1/S2/...) — full metadata
├── nodes.jsonl          ← every active node with full content + KV
├── edges.jsonl          ← every edge_relation with weights + descriptions
├── recall.json          ← query, top-N candidates with scores, selected/dropped, context, augmented evidence
└── result.json          ← harness result (mirror of streaming write)
```

Roughly 1 MB per item. ~200 ms extra runtime per item to dump. **Independent of `--keep_dbs`** — artifacts are kept by default; the per-item `brain.db` is still cleaned unless `--keep_dbs` is passed.

## File schemas

### `meta.json`
```json
{
  "run_name": "...", "qid": "...", "axis": "...",
  "question": "...", "gold": "...",
  "question_date": "YYYY-MM-DD",
  "haystack_dates": [...], "haystack_session_ids": [...],
  "haystack_turn_count": 22,
  "started_at": <epoch>
}
```

### `interactions.jsonl`
One row per interaction version present in the brain at item end. Each row:
```json
{"id": 1, "name": "s1e", "version": 14, "template": "...full prompt...",
 "parameters": {"model": "...", "max_tokens": 2000},
 "created_at": "...", "created_by": "seed", "parent_version": null}
```
Useful to know **which prompt versions** the encoder/surfacer ran with.

### `traces.jsonl`
One row per `trace_events` record in `brain_logs.db`. Captures every O / K / Δ event across **all scales**. Each row:
```json
{"id": 12, "chain_id": "s1e-...", "scale": "s1", "event_type": "delta",
 "ref_type": "encoding_actions", "ref_id": "...",
 "summary": "...", "metadata": {...full event metadata...},
 "session_id": "...", "interaction_id": 1, "created_at": "..."}
```
The `metadata` field is the rich one — for s1e events it holds tool calls + actions; for s1r events it holds candidates + selected; for s2 events it holds proposals/clusters/etc.

### `nodes.jsonl`
One row per active (non-archived) node, with **all** core columns plus a `kv` dict containing every `node_metadata_kv` entry for that node:
```json
{"id": "...", "type": "lesson", "title": "...", "content": "...",
 "keywords": "...", "activation": 1.0, "stability": 1.0, ...,
 "kv": {"situation": "...", "reasoning": "...", "user_raw_quote": "...", ...}}
```
This is the encoded knowledge graph at end-of-ingest. ENCODE_MISS analysis lives here.

### `edges.jsonl`
One row per active `edge_relations` row, joined with `edges` table + source/target node titles:
```json
{"edge_id": 5, "relation": "extends", "description": "...",
 "relation_weight": 0.6, "encoding_source": "encoder:sonnet",
 "source_id": "abc123...", "target_id": "def456...",
 "source_title": "...", "target_title": "...",
 "edge_weight": 0.6, "co_access_count": 1, "created_at": "..."}
```
Stage 1B: multiple relations per (source, target) pair appear as separate rows.

### `recall.json`
The query phase, end-to-end:
```json
{
  "query_session_id": "...",
  "query": "...",
  "candidates": [{"id": "...", "title": "...", "score": 0.78, "type": "..."}, ...],
  "candidate_count": 25,
  "selected": [...],            // surface output
  "dropped": [...],
  "context": "...",             // additionalContext that reached the answerer
  "context_chars": 3971,
  "classifier_evidence": {       // populated only on failures
    "gold_in_brain": {"found": ..., "matches": [...], "terms_used": [...]},
    "relevant_to_gold": [...],
    ...
  },
  "fact_node_ranks_in_candidates": {"<8charid>": 7, "<8charid>": -1},
  "answerer_response": {
    "hypothesis": "...", "abstained": false, "has_context": true,
    "tokens_in": ..., "tokens_out": ..., "elapsed_ms": ...
  }
}
```
**`fact_node_ranks_in_candidates` is the missing piece in the original classifier.** -1 means the fact-bearing node was NOT in the top-N candidates (true RECALL_MISS at the ranker layer). A positive integer means it WAS in candidates at that rank — if surface didn't pick it, that's a true SURFACE_MISS.

### `result.json`
Mirror of the harness's streaming `results_{run}.jsonl` row for this item. Top-level keys: `correct`, `failure_bucket`, `failure_reason`, timings, judge raw, etc.

## Loading + analyzing

```python
from eval.longmem.artifacts import load_artifacts, list_items

# All qids in a run
qids = list_items('eval_a_2026_05_10')

# Full bundle for one item
bundle = load_artifacts('eval_a_2026_05_10', '58470ed2')
print(bundle['meta']['question'])
print('nodes encoded:', len(bundle['nodes']))
print('candidates at query:', bundle['recall']['candidate_count'])
```

## What to look at when investigating a failure

| Suspected layer | Files to read | What to look for |
|---|---|---|
| Scout missed a fact | `traces.jsonl` (s1e events with `ref_type='scout_*'`) | What snippet each scout extracted on the relevant haystack turn; was the gold-bearing snippet present? |
| Encoder dropped a fact | `traces.jsonl` (s1e events with `event_type='delta'`) + `nodes.jsonl` | What actions did the encoder take? Is the gold fact in any node's title/content/KV? |
| Encoder transformed losing terms | `nodes.jsonl` semantic search vs gold | Look for paraphrases — the keyword scan misses these. |
| Recall ranker buried the right node | `recall.json` — `candidates` + `fact_node_ranks_in_candidates` | Was the fact-bearing node in top-25? At what rank? |
| Surface skipped the right candidate | `recall.json` — `candidates` (rank present) + `selected`/`dropped` | If the fact-node IS in candidates but NOT in selected, it's a surface failure. |
| Answerer failed despite right context | `recall.json` — `context` + `answerer_response` | Was the gold fact in the context string? Did the answerer abstain anyway? |

## Adding to other Scales (S0 / S2 / future)

The artifact schema is scale-agnostic. To extend:

1. **For S2 evals** (e.g. "did consolidation merge cluster X?"):
   ```python
   dumper.dump_nodes(brain, prefix='pre_unit')
   dumper.dump_edges(brain, prefix='pre_unit')
   # run unit
   dumper.dump_nodes(brain, prefix='post_unit')
   dumper.dump_edges(brain, prefix='post_unit')
   ```
   This produces `nodes_pre_unit.jsonl` etc. Diff them in the analyzer.

2. **For S0 evals** (per-conversation flow): use the existing single-checkpoint structure. `traces.jsonl` already captures S0 events.

3. **For new event types**: no schema change needed in the dumper — `traces.jsonl` records every `trace_events` row regardless of `scale` / `event_type`.

The dumper class is general; only the analyzer functions need to be axis-aware.

## Disk + retention

- ~1 MB per item × 50-item run = ~50 MB per run.
- Long-term retention: keep all runs. Disk is cheap relative to API spend on regenerating them.
- Cleanup tool: not needed yet. If artifacts dir grows >5 GB, archive old runs to a tarball.

## When to run with `--keep_dbs`

Pass `--keep_dbs` ONLY when you specifically want to re-run a query against the same brain (e.g. testing a new surface prompt against the same encoded state). For everything else the artifacts dumps are sufficient. Per-item `brain.db` is ~5 MB so 50 items × `--keep_dbs` = 250 MB extra; acceptable but not free.

## Failure-mode discipline

The dumper is wrapped in try/except in `run_item` — an artifact write failure must NEVER kill an eval. Failures get printed `[harness] artifacts dump failed (non-fatal)`. If you see this in production runs, fix the dumper, but don't let it abort a 30-min run.

## See also

- `eval/longmem/artifacts.py` — implementation
- `eval/longmem/harness.py:run_item` — call sites
- `eval/longmem/classifier.py` — failure bucket logic (artifact augments its evidence)
