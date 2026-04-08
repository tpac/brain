# Edge Selection Eval Spec

> **Status:** Strategy D implemented in production (April 7, 2026).
> See EDGE_SELECTION_RESULTS.md for full results and revisit criteria.

## Problem

Most edges have weight 0.60 — a 27-way tie. `ORDER BY weight DESC LIMIT 3` picks 3 arbitrary edges from the tie. The right edge for a given query could be #15 in the list. Current eval queries are too easy — any 3 edges look fine because the queries are broad enough.

## What This Eval Tests

1. **Relevance-aware edge selection** — does the system show query-relevant edges, not just high-weight ones?
2. **Fatigue rotation** — across repeated similar queries in one session, do different edges surface?
3. **Nuanced disambiguation** — same node, different queries, different edges needed

## KPIs

### Per-query KPIs
- **edge_precision**: of 3 edges shown, how many are in the golden "good edges" set?
- **edge_recall**: of the golden set edges, how many were shown?
- **bad_edge_rate**: of 3 edges shown, how many are in the "bad edges" set?

### Per-session (fatigue) KPIs
- **unique_edges_5_turns**: across 5 recalls of similar queries, how many unique edge targets surface?
- **rotation_rate**: what % of a node's edges have been shown after 5 turns?

### Aggregate KPIs
- **mean_edge_precision**: across all queries
- **mean_edge_recall**: across all queries
- **zero_bad_rate**: % of queries with no bad edges shown

## Test Categories

### Category 1: Disambiguation (same node, different intent)

Node `894795e3` "Rule: before writing code, ask where does this live" has 31 edges.

Query A: "What are Tom's architectural principles?"
- Good edges: architecture principle (supports), CLAUDE.md app-level rules, code violations to prevent
- Bad edges: SESSION_CONTEXT encoder, Distiller→Judge replacement

Query B: "How do Tom's rules relate to documentation?"
- Good edges: CLAUDE.md restructure, SKILL.md behavioral corrections, CLAUDE.md app-level
- Bad edges: dead file audit, architecture principle

Query C: "What does Tom's code quality rule connect to?"
- Good edges: dead file audit, code violations, architecture principle
- Bad edges: CLAUDE.md app-level, session handoff

Same node. Same weight. Different right answers. Static weight can't do this.

### Category 2: Correction Chain Discovery

Node `138ede9f` "Four-layer encoding problem" has 27 edges including corrections.

Query: "What corrections did Tom give about encoding depth?"
- Good edges: synthesizes edges (correction chains), Tom's correction patterns
- Bad edges: technical implementation edges (DAL, dispatch, pipeline)

Query: "How was the encoding architecture built?"
- Good edges: technical implementation edges
- Bad edges: correction/philosophical edges

### Category 3: Fatigue Rotation

Run 5 times with similar queries against node `1be3f985` "partner not user" (23 edges):
1. "How does Tom see the partnership?"
2. "What's the relationship model?"
3. "Tom and Anchor's working dynamic"
4. "How we collaborate"
5. "Partnership principles"

Target: unique_edges_5_turns >= 10 (out of 23 total)
Current expected: ~3-6 (same top 3 every time, minor Haiku variance)

### Category 4: Sparse Node Enrichment

Find nodes with < 100 chars content but 5+ edges. Query them.
The edges ARE the context — wrong edges = useless surface.

### Category 5: Cross-Node Coherence

Surface 3 nodes together. Do their edges form a narrative?
Query: "How did the surface pipeline evolve?"
Node A edges should connect to → architecture decisions
Node B edges should connect to → what changed
Node C edges should connect to → why it changed

## Golden Set Structure

```python
{
    "query": "...",
    "category": "disambiguation|correction|fatigue|sparse|coherence",
    "target_node": "node_id",  # the node we're evaluating edges for
    "good_edges": ["target_id_1", "target_id_2", ...],  # edges that SHOULD show
    "bad_edges": ["target_id_3", ...],  # edges that should NOT show
    "repeat_count": 1,  # >1 for fatigue tests
    "min_unique_edges": null,  # for fatigue tests
}
```

## Data Source

All golden edges hand-labeled from real brain.db edge data.
See `eval/surface_ab_eval.py` for the A/B testing infrastructure to build on.

## Architecture Decision

Edge selection happens in `get_rich_node()` which accepts:
- `query_vec` — for relevance scoring (cosine against edge target embeddings)
- `session` — for fatigue tracking (session-scoped rotation)

Formula: `edge_score = relevance × weight × (1 / (1 + K_edge × surface_count))`
- K_edge = 0.25 (in surface_contract.py)
- Node fatigue K = 10.0 / (1 + degree / 10.0) (in surface_contract.py, currently hardcoded in brain_recall.py)

## Baseline

Run eval with current system (static weight) first. Save results.
Then implement relevance + fatigue scoring. Compare.
The eval should be hard enough that the current system scores poorly.
