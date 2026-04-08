# Edge Selection Eval Results — April 7, 2026

## What Was Tested

6 edge selection strategies compared on 24 queries (18 single + 3 fatigue + 3 multi-turn) against hand-labeled golden sets from real brain.db data.

**Goal:** replace static `ORDER BY weight DESC LIMIT 3` with query-aware edge selection that shows different edges depending on what's being asked.

## Strategies Tested

| Strategy | Description | How Relevance Is Computed |
|----------|-------------|--------------------------|
| **A** | Static weight | `ORDER BY weight DESC` — current production |
| **B** | Node embedding | `cosine(query, stored_node_embedding) × weight × fatigue` |
| **C** | 70% node + 30% desc | `(0.7 × node_emb + 0.3 × desc_emb) × weight × fatigue` |
| **C2** | C + 3-msg blend | Same as C but query is blended from last 3 user messages |
| **D** | C2 + weight tiebreaker | Same as C2 but `relevance + weight × 0.01` instead of `relevance × weight` |
| **E** | Structural embed | `embed(edge_type + target_type + title + desc)`, weight tiebreaker |
| **F** | Description-first | `embed(description)` when desc exists, `embed(edge_type + target_type + title)` fallback |

## Final Results

| Strategy | Avg Precision | Avg Bad Rate | Zero-Bad | Wins/Losses vs A |
|----------|--------------|-------------|----------|-----------------|
| A (static weight) | 25.4% | 14.3% | 13/21 | baseline |
| B (node embedding) | 33.3% | 7.9% | 16/21 | 7/2 |
| C (70n+30d) | 38.1% | 7.9% | 16/21 | 9/1 |
| C2 (C + 3-msg) | 41.3% | 7.9% | 16/21 | 9/0 |
| **D (chosen)** | **42.9%** | **3.2%** | **19/21** | **11/3** |
| E (structural) | 39.7% | 6.3% | 17/21 | 10/3 |
| F (desc-first) | 42.8% | 6.3% | 17/21 | 10/3 |

**D was chosen** for production because:
- Highest precision (42.9%)
- Lowest bad edge rate (3.2%)
- Most zero-bad queries (19/21)
- Handles missing descriptions gracefully (pre-April edges have 5% coverage)

## Key Findings

### 1. Weight Is Meaningless for Selection
96% of edges are `related` type at weight 0.60. Weight carries zero information for distinguishing among them. Making weight a tiebreaker (strategies D, E, F) immediately improved precision by ~10pp and halved the bad edge rate.

**Action needed:** S2 should make weight dynamic based on outcomes. When that happens, restore weight's role in the scoring formula. See `WEIGHT_TIEBREAKER` constant in `surface_contract.py`.

### 2. Descriptions Are the Best Signal — When They Exist
Pre-April 1 edges: 5% have descriptions. Post-April: 63-67%. Strategy F (description-first) wins on well-described edges but collapses on undescribed ones. Strategy D's 70/30 blend handles the mixed coverage.

**Action needed:** Improve description coverage. Options:
- Fix encoder to always write descriptions on new edges
- Backfill descriptions on high-traffic old edges
- Allow multiple edges between same pair (new described edge alongside old undescribed)

### 3. Edge Types Are Uniformly `related`
The encoder defaults to `related` for almost everything. Edge type carries no signal. The descriptions contain the relationship information that should be in the type.

**Action needed:** Update encoder instructions to require specific edge types (`corrects`, `extends`, `produced`, `exemplifies`, etc.). Never use `related` or `related_to`.

### 4. Multi-Turn Context Helps Ambiguous Queries
"yes", "tell me more", "how does that connect?" — these queries have no topic signal alone. Blending with the previous 2-3 queries (0.6/0.3/0.1 weights) rescues edge selection. C2 vs C showed +3pp from multi-turn alone.

### 5. Fatigue Enables Rotation
Without fatigue: 3 unique edges across 5 runs (16% rotation, 5/5 identical).
With fatigue (K=0.25): 8-15 unique edges (31-67% rotation, 1/5 identical).

## Strategy D Implementation

Implemented in `servers/scales/s1/surface_contract.py:select_edges()`.

**Formula:** `score = relevance × fatigue_discount + weight × 0.01`

Where:
- `relevance = 0.7 × cosine(query, stored_node_embedding) + 0.3 × cosine(query, embed(description))` when description exists
- `relevance = cosine(query, stored_node_embedding)` when no description
- `fatigue_discount = 1 / (1 + 0.25 × surface_count)` — session-scoped
- `weight × 0.01` — tiebreaker only

**Wired in:** `daemon_hooks.py:hook_recall()` — runs `select_edges()` on each candidate's connections after `get_rich_node()` batch fetch. Uses query embedding from recall + prior 2 user messages for multi-turn blend.

**Constants (in surface_contract.py):**
```python
K_EDGE_FATIGUE = 0.25
EDGE_NODE_WEIGHT = 0.7
EDGE_DESC_WEIGHT = 0.3
WEIGHT_TIEBREAKER = 0.01
TURN_WEIGHTS = [0.6, 0.3, 0.1]
```

## What to Revisit

### When S2 is built:
- S2 adjusts edge weights based on outcomes → restore weight in scoring formula
- S2 manages cross-session edge fatigue persistence
- S2 community labels become a scoring signal (solves flat embedding space)

### When description coverage exceeds 80%:
- Re-run eval comparing D vs F
- F (description-first) should win when descriptions are comprehensive
- Switch from D to F if confirmed

### When encoder uses real edge types:
- Edge type becomes a scoring signal
- Consider: `cosine(query, embed(edge_type + description))` vs current blend
- Re-run eval with type-aware scoring

### When multi-edge support ships:
- Same node pair can have 2-3 edges with different types/descriptions
- Selection system naturally picks the most relevant relationship
- Re-run eval to measure improvement

## Test Queries

24 queries in `eval/edge_selection_eval.py:EDGE_QUERIES`:
- 13 disambiguation (same node, different intent)
- 2 correction chain
- 1 adversarial
- 3 fatigue rotation
- 2 new-node disambiguation
- 3 multi-turn ambiguous

Golden sets hand-labeled from real brain.db edge data. See `EDGE_SELECTION_EVAL_SPEC.md` for design rationale.

## How to Re-Run

```bash
# Full eval
BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 eval/edge_selection_eval.py

# Verbose with per-run fatigue details
python3 eval/edge_selection_eval.py --verbose

# Specific category
python3 eval/edge_selection_eval.py --category disambiguation

# Different edge limit
python3 eval/edge_selection_eval.py --edge-limit 5
```

Results saved to `eval/results/edge_selection_ab_latest.json`.
