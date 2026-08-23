# `eval.encoder_eval` — encoder quality eval infrastructure

**What this is**: a multi-version, multi-axis eval that measures what an
encoder writes — node shape, source_refs anchoring, edge structure, voice
balance, atomization, gold-fact presence — in addition to whether the
brain can answer a downstream question.

**Built for**: the v22 vs v21 vs v19 ship gate. Reusable for every future
encoder version. The infrastructure outlives the v22 decision.

## Why this exists (vs `eval/longmem/`)

`eval/longmem/` answers "did the encoded brain produce the right answer?"
That's necessary but doesn't tell us WHY. A brain can answer correctly with
bad encoding (lucky), or answer wrong with structurally good encoding
(downstream failure). Without probing the encoded structure directly, we
can't separate encoder regression from recall regression.

`eval/encoder_eval/` adds the structural layer:

| Layer | Question | Where it lives |
|---|---|---|
| Encoder structure | Did the encoder write nodes anchored to load-bearing turns with the right voice and edge shape? | `quality_probes.py` (THIS module) |
| Brain presence | Does the encoded brain contain the gold answer's atomic value? | `quality_probes.py:probe_brain_presence` |
| Downstream recall | Given the brain, can the surface+answerer pipeline answer the gold question? | reused from `eval/longmem/answerer.py` + `judge.py` |

Combined: a per-(version, item) row carries BOTH encoding-quality scores
AND answer correctness. Divergences are diagnostic.

## Module structure

```
eval/encoder_eval/
├── __init__.py
├── README.md            — this file
├── quality_probes.py    — 6 encoding-quality probes
├── harness.py           — per-(version, item) driver, composes longmem pieces
├── runner.py            — CLI entry with staged checkpointed execution
└── reports/             — output: timestamped run dirs with JSON + markdown
```

## The probes

Each probe takes (brain, item) and returns a JSON-shaped dict with the
metric value(s) plus evidence (node ids, snippets) for failure attachment.

| Probe | Measures | Failure dim mapping |
|---|---|---|
| `probe_brain_presence` | Does any node contain the gold answer's atomic value (numeric / date / proper noun)? Returns the closest node by string + cosine similarity. | answers-not-encoded |
| `probe_specificity_preservation` | Are numbers / ranges / exact phrases from the haystack preserved as-is in node content? Or smoothed to averages / paraphrased? | D5, D36 |
| `probe_source_refs_coverage` | What % of nodes carry `source_refs`? Distribution of refs/node. Sparseness violations (>5). Hex-format failures. | D25, D26, D33 |
| `probe_atomization_shape` | How many nodes per turn? Bundled (1 node covers 5 turns) vs atomized? | D3 |
| `probe_edge_structure` | Typed `connect_to` counts by relation. Auto `co_anchored` counts. Aspect distribution. `related_to` overuse. | D20-D24, D27 |
| `probe_voice_balance` | Per-node their_raw_quote / my_raw_quote presence. Symmetry on identity/correction-typed nodes. | D5, D7, D14 |

## The pipeline

Per (version, item) cell:

1. **Override the prompt** — `tests.interaction_override.interaction_override(
   brain, 's1e', template=...)` registers + activates the arm's prompt in the
   EVAL brain and clears the pointer on exit, even if the body raises. The
   arm's K is compared on `brain.get_interaction_stamp('s1e')['fingerprint']`.
2. **Fresh brain** — `eval/longmem/fresh_brain.create_fresh_eval_brain(qid)`
   makes an isolated per-item DB at `~/AgentsContext/brain-eval-{run}/{qid}/`.
3. **Replay haystack** — `eval/longmem/replay.replay_item(brain, ...)`
   ingests every turn; s1e fires every 5 turns and encodes.
4. **Quality probes** — run all 6 probes on the resulting brain. Each
   probe is read-only.
5. **Surface + answer + judge** — reused from longmem: surface call,
   answerer (junior-Anchor Sonnet), judge scores against gold answer.
6. **Seal result** — single dict per cell carrying all metrics + evidence.

## Staged checkpoints

Runner drives stages in sequence. Each stage = a batch of items. After
each stage:

- Aggregate per-version × per-axis × per-probe
- Evaluate `stop_conditions` (see below)
- Write incremental report
- Continue or HALT

Default stop conditions (configurable):

| Condition | Action |
|---|---|
| v22 hex-format error rate > 5% on any stage | HALT — encoder regression in source_refs format |
| v22 produces zero source_refs across any single-session axis | HALT — substrate teaching failed |
| v22 answer correctness < v19 by ≥10pp on any axis | HALT — answers regress |
| D27 engram_cohort: v22 co_anchored count zero across cohort items | HALT — co_anchored auto-edge broken |
| Cost > budget (env `ENCODER_EVAL_BUDGET_USD`) | HALT — cost guard |

Operator can override-and-continue with `--continue-on-stop`.

## Output

Each run writes `eval/encoder_eval/reports/{run_name}/`:

```
{run_name}/
├── config.json              — versions, corpus, items, stages, conditions
├── per_cell.jsonl           — one row per (version, item) cell, streamed
├── per_stage.json           — incremental aggregate at each checkpoint
├── final_report.md          — human-readable per-axis × per-version table
└── failure_attachments.md   — worst-5 nodes per regression dim with text
```

## CLI

```
./dev python3 -m eval.encoder_eval.runner \
    --versions 19,21,22 \
    --corpus realchat \
    --stages "0-0;1-2;3-5;6-9;10-14" \
    --run-name v22_gate_$(date +%Y%m%d_%H%M)
```

Stage `0-0` = single-item Stage A (mirrors the Stage A from the v22 eval
plan); `1-2` = identity-bearing pair Stage B; etc. Stage syntax is a
semicolon-separated list of inclusive Python-slice expressions.

## Reusable beyond v22

The infrastructure isn't v22-specific. Pass any `--versions` list, any
corpus, any stage shape. The probes are general; the report templates are
parametric. When v23/v24 lands, this is the gate it walks through too.
