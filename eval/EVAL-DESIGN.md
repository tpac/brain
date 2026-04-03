# Brain Evaluation Framework — Design Document

Created 2026-04-02. This is the single reference for what the eval measures and how.

## Philosophy

Encoding and decoding are a feedback loop — not independent systems. Bad decoding feeds wrong context to the encoder, which creates wrong connections, which degrades future decoding. The eval must test the LOOP, not just each half.

## Test Data

### Synthetic Conversations (`eval/corpus/`)
Existing: conv_001 (architecture), conv_002 (debugging), conv_003 (philosophy), conv_004 (art/design), conv_005 (emotions), conv_006 (product).

Each conversation has annotations in `eval/corpus/annotations/`:
- `key_facts`: Things that SHOULD be encoded (ground truth for encoding completeness)
- `queries`: Questions that should find specific nodes (ground truth for recall)
- `false_positives`: Questions that should return NOTHING from this conversation
- `cross_topic`: Questions from OTHER conversations that should NOT pull this conversation's nodes

### Real Conversations
Tom's actual conversation logs. Used for live-brain testing only (not fresh-brain).

## Modes

```
python3 eval/brain_eval.py --mode fresh        # Empty brain → encode → decode → measure loop
python3 eval/brain_eval.py --mode decode        # Current brain → run decode queries only
python3 eval/brain_eval.py --mode encode        # Encode conversations → measure encoding quality
python3 eval/brain_eval.py --mode loop          # Full loop: encode conv A, query, encode conv B, query, measure contamination
python3 eval/brain_eval.py --mode compare A B   # Compare two result files
```

## KPIs

### Group 1: Encoding Quality

| KPI | ID | What | Calculation | Target |
|---|---|---|---|---|
| Extraction completeness | `enc_completeness` | % of annotated key facts that have a matching node | LLM judge: "does any encoded node capture this fact?" | >80% |
| Over-encoding | `enc_overencoding` | Nodes created that don't map to any key fact | total_nodes - matched_nodes / total_nodes | <30% |
| Deduplication | `enc_dedup` | Semantically duplicate nodes in same run | Cosine sim >0.85 between nodes from same batch | 0 duplicates |
| Metadata richness | `enc_metadata` | % of new nodes with situation, reasoning, quotes | Count fields per node | >50% with situation, >70% with reasoning |
| Connection precision | `enc_conn_precision` | % of edges that make sense | LLM judge on sample | >70% |
| Connection coverage | `enc_conn_coverage` | Are related nodes connected? | For each pair of key facts that are related, is there an edge? | >50% |

### Group 2: Decoding Quality

| KPI | ID | What | Calculation | Target |
|---|---|---|---|---|
| Recall@8 | `dec_recall8` | Expected node in top 8 results | Binary per query, averaged | >70% |
| Recall@25 | `dec_recall25` | Expected node in top 25 | Binary per query, averaged | >85% |
| MRR | `dec_mrr` | Mean reciprocal rank of first relevant result | 1/rank averaged across queries | >0.3 |
| Precision@8 | `dec_precision8` | % of top 8 that are relevant | LLM judge per result | >50% |
| Hub concentration | `dec_hub_concentration` | Top-5 nodes as % of all recall slots | Sum(top5_counts) / total_slots | <25% |
| False positive rate | `dec_false_positive` | % of off-topic queries that return results | Queries with 0 relevant expected → did system return results? | <20% |
| Cross-topic contamination | `dec_contamination` | Topic A nodes appearing in Topic B queries | Count mismatched topic labels | <15% |
| Silence precision | `dec_silence` | When system returns empty, was it right? | Empty results on queries that SHOULD be empty | >80% |

### Group 3: Loop Quality

| KPI | ID | What | Calculation | Target |
|---|---|---|---|---|
| Encode-then-recall | `loop_encode_recall` | After encoding conv, can we find its facts? | Encode → query annotated queries → Recall@8 | >70% |
| Sequential contamination | `loop_contamination` | After encoding A then B, does B query pull A? | Encode A, encode B, run B queries, count A nodes in results | <15% |
| Connection utility | `loop_conn_utility` | Do encoding-created edges improve recall? | Compare recall with vs without new edges | >0 improvement |
| Revision accuracy | `loop_revision` | When encoder revises a node, did it improve? | Before/after content comparison via LLM judge | >70% improved |

## Result Storage

Results stored as JSON in `eval/results/`:
```
eval/results/2026-04-02T10-30-00_fresh_baseline.json
eval/results/2026-04-02T11-00-00_title_embedding.json
```

Each result file contains:
- Configuration hash (what code version, what parameters)
- All KPI values
- Per-query breakdown (for debugging)
- Timestamp

## Annotation Format

`eval/corpus/annotations/conv_001.json`:
```json
{
  "key_facts": [
    {"id": "kf1", "description": "Daemon switched from Unix socket to TCP", "importance": "high"},
    {"id": "kf2", "description": "Thread pool size = 1 to prevent SQLite deadlocks", "importance": "medium"}
  ],
  "queries": [
    {"query": "Why did we switch the daemon to TCP?", "expected_facts": ["kf1"], "expected_node_ids": ["daeb9fa6"]},
    {"query": "What is the thread pool size?", "expected_facts": ["kf2"]}
  ],
  "false_positives": [
    {"query": "How do I make pasta carbonara?", "expected_results": 0},
    {"query": "React hooks tutorial", "expected_results": 0}
  ],
  "cross_topic_queries": [
    {"query": "What was discussed about emotions?", "this_conv_should_NOT_appear": true}
  ]
}
```

## Embedding Architecture (implemented 2026-04-02)

4-group z-indexed multi-vector architecture. Each node gets 2-4 vectors:

| Group | Weight | Fields | Coverage |
|---|---|---|---|
| title | 1.00 | title only | 100% |
| blend | 0.85 | title + content | 100% |
| high_meta | 0.70 | situation + quotes | 28% (growing via retroactive enrichment) |
| other_meta | 0.40 | reasoning + correction_pattern + emergent | 29% (growing) |

Scoring: z-weight each vector (`weight × cosine_sim`), average the top 2. Requires two vectors to agree — prevents noisy single-field matches.

Emergent KV fields not in any explicit group auto-flow into `other_meta`.
Old enrichment types (question, anchor, bridge, keywords) participate with `other_meta` weight.

Defined in: `pipeline_contract.py` → `EMBEDDING_GROUPS`
Stored in: `node_enrichments` table (vector_type = group name)
Computed at: `brain_remember.py` → `_compute_group_vectors()`
Scored at: `brain_recall.py` → STEP 3.5

### Results (April 2, 2026)

| KPI | Before | After | Delta |
|---|---|---|---|
| R@25 | 76.3% | 84.7% | +8.4pts |
| Hub concentration@8 | 18.2% | 12.3% | -5.9pts |
| MRR | 52.7% | 54.5% | +1.8pts |
| R@8 | 66.1% | 66.1% | 0 (needs investigation — downstream scoring layers may dominate) |

## Implementation Files

```
eval/
  brain_eval.py          — Main runner (modes, CLI, orchestration)
  eval_kpis.py           — KPI calculation functions
  eval_annotations.py    — Annotation loader + ground truth matcher (planned)
  eval_report.py         — Result formatting, comparison, delta display (in brain_eval.py for now)
  corpus/
    annotations/         — Ground truth per conversation (planned)
  results/               — Timestamped result JSONs
```
