# S2 Community Detection — Final Design

## Pipeline Overview

```
1. LOAD      — Build weighted graph from brain nodes + edges
2. EMBED     — Embed unique edge relation texts → semantic edge weights
3. DETECT    — Overlapping community detection (SLPA via cdlib)
4. ASSEMBLE  — Build proposals with member data + edge type distribution
5. ENRICH    — ONE batched Haiku call: name, describe, or reject each proposal
6. DIFF      — Compare against existing community nodes
7. WRITE     — Create/revise/archive community nodes + sync edges
8. TRACE     — Record O/K/Δ for S3
```

No closed lists. No predefined dimensions. Edge types are open text — new types from tomorrow's encoder session automatically participate via embedding.

---

## Call Stack

```
hook_idle_maintenance()
  └→ CommunityDetection(brain).run()
       │
       ├→ _load_graph()
       │    ├→ Query non-archived, non-community nodes
       │    ├→ Query all edges (weight >= threshold)
       │    ├→ Deduplicate bidirectional edges
       │    └→ Return networkx.Graph (cdlib needs networkx, not igraph)
       │
       ├→ _compute_edge_weights()
       │    ├→ Collect unique relation strings from edges
       │    ├→ Embed each unique relation via brain.embedder
       │    ├→ Cache embeddings (relation_text → vector)
       │    ├→ For each edge: weight = original_weight × semantic_factor
       │    └→ semantic_factor derived from relation embedding richness
       │
       ├→ _detect_communities()
       │    ├→ Run SLPA(G, t=50, r=threshold) via cdlib
       │    ├→ Returns list of communities (overlapping — node can appear in multiple)
       │    └→ Filter: drop communities below min_size
       │
       ├→ _build_proposals()
       │    ├→ For each community:
       │    │    ├→ Collect member node titles, types, keywords
       │    │    ├→ Compute internal edge type distribution
       │    │    ├→ Collect edge descriptions (sample, not all)
       │    │    └→ Build proposal dict
       │    └→ Return list of proposal dicts
       │
       ├→ _enrich_proposals(proposals)
       │    ├→ Format ALL proposals into ONE Haiku prompt
       │    ├→ Call Haiku once (batched)
       │    ├→ Haiku returns per proposal:
       │    │    ├→ title (descriptive name)
       │    │    ├→ content (what this community is about)
       │    │    ├→ situation (when is it relevant)
       │    │    ├→ keywords (for recall)
       │    │    └→ OR reject with reason
       │    └→ Parse response, return enriched proposals
       │
       ├→ _diff_communities(enriched_proposals)
       │    ├→ Find existing community nodes (by encoding_source)
       │    ├→ Match new proposals to existing by member overlap
       │    └→ Return {new, updated, removed, unchanged}
       │
       ├→ _write_results(enriched_proposals, diff)
       │    ├→ Archive removed community nodes
       │    ├→ Create new community nodes (auto_connect=False)
       │    │    └→ Full node: type, title, content, keywords,
       │    │       situation, confidence, encoding_source, embeddings
       │    ├→ Revise updated community nodes
       │    └→ Sync bidirectional community_member edges per node
       │
       └→ trace() calls throughout:
            O: graph_structure (node/edge counts)
            K: community_partition (proposals + Haiku decisions)
            Δ: community_created/updated/removed/assignments
```

---

## Edge Embedding — The Open System

### Problem
Community detection algorithms see edges as (node_A, node_B, weight). They don't read edge labels. But our edges carry rich semantics: `relation='corrects_implementation_of'`, `description='this fixes the boot architecture violation'`.

### Solution
Embed the relation text. The embedding captures what KIND of relationship this is — without needing a predefined list of categories.

```python
def _compute_edge_weights(self, G):
    """Enhance edge weights with semantic signal from relation text."""
    
    # Collect unique relation strings
    unique_relations = set(G[u][v].get('relation', 'related') for u, v in G.edges())
    
    # Embed each unique relation (typically ~50-130 unique strings)
    # Uses daemon's already-loaded embedder — no extra memory
    relation_embeddings = {}
    for rel in unique_relations:
        relation_embeddings[rel] = brain.embedder.embed(rel)
    
    # Compute semantic richness factor:
    # - Generic relations ('related', 'co_accessed') get lower factor
    # - Specific relations ('corrects_implementation_of', 'extends') get higher
    # Method: compare each relation embedding to the "related" baseline.
    # Distance from baseline = specificity.
    baseline = relation_embeddings.get('related')
    for u, v in G.edges():
        rel = G[u][v].get('relation', 'related')
        original_weight = G[u][v].get('weight', 0.5)
        
        if baseline is not None:
            sim_to_baseline = cosine_similarity(relation_embeddings[rel], baseline)
            # More distant from "related" = more specific = higher factor
            specificity = 1.0 - sim_to_baseline  # 0 = generic, 1 = very specific
            semantic_factor = 0.5 + specificity  # range: 0.5 to 1.5
        else:
            semantic_factor = 1.0
        
        G[u][v]['weight'] = original_weight * semantic_factor
```

### Why this is open
- New edge type `validates_architecture_of` tomorrow → gets embedded automatically
- No list of "good" vs "bad" edge types
- The encoder creating more specific edge types naturally improves community detection
- Generic `related_to` edges get downweighted because they're close to the `related` baseline
- Specific `corrects_implementation_of` edges get upweighted because they're far from baseline

### Cache
Relation embeddings are cached across runs. Only new relation strings get embedded. With ~130 unique relations, the full cache is ~130 × 384 dims × 4 bytes = ~200KB.

---

## Overlapping Detection — SLPA

### Why SLPA
- Produces true overlapping communities (a node can belong to multiple)
- Works on weighted graphs
- Available in cdlib (well-maintained Python library)
- Bridge nodes naturally get multi-membership
- No predefined number of communities — emerges from the data

### How SLPA works
Each node starts with its own label. Over T iterations, each node "listens" to its neighbors' labels (weighted by edge weight) and adds the most frequent label to its memory. After T iterations, labels appearing below threshold `r` are pruned. Remaining labels = community memberships.

- `t=50`: iterations (more = more stable)
- `r`: retention threshold. Lower r = more overlap.
  - `r=0.1` on our brain: 155 communities, 396 multi-member nodes (23%)
  - `r=0.2`: 166 communities, 254 multi-member nodes (15%)

### Real brain results
```
Graph: 1719 nodes, 7468 edges (deduplicated)
SLPA r=0.1: 155 communities, 396 multi-member nodes
SLPA r=0.2: 166 communities, 254 multi-member nodes
```

Not all 155 communities get nodes — min_size filter and Haiku rejection will reduce this significantly.

---

## Batched Haiku Enrichment

### ONE call, all proposals

Instead of 26+ individual calls, format all proposals into one prompt. Haiku reads the batch and returns structured JSON.

```
System: You evaluate community proposals from a knowledge graph.
        For each proposal, either enrich it or reject it.

User:
PROPOSALS:

[1] 47 members
    Top types: decision(12), lesson(8), mechanism(5)
    Top keywords: hook, timeout, daemon, latency, async
    Internal edge types: co_accessed(45%), extends(20%), related(35%)
    Sample edges: "extends: 5s timeout extends the async hook pattern"
    Sample members:
      - "Stop hook timeout reduced to 5s"
      - "os._exit(0) in hook script — instant exit"
      - "Encoding write lock blocks recall — 58s Sonnet call"

[2] 92 members
    ...

Return JSON:
[
  {"id": 1, "action": "accept", "title": "...", "content": "...", 
   "situation": "...", "keywords": "..."},
  {"id": 2, "action": "reject", "reason": "..."},
  ...
]
```

### Learnable boundary
This prompt is registered as interaction `s2_community_enrichment`. S2 traces record proposals → Haiku decisions → final community nodes. Future prompt evolution can improve enrichment quality based on trace outcomes.

### Cost
- ~155 proposals × ~200 tokens each = ~31K input tokens
- Response: ~100 tokens per proposal = ~15K output tokens
- One Haiku call: ~$0.01-0.02
- Runs during idle, not time-critical

---

## Community Nodes — First-Class Citizens

Community nodes are regular nodes. Every field a knowledge node has, a community node has:

| Field | Source |
|-------|--------|
| `type` | `'community'` |
| `title` | Haiku enrichment |
| `content` | Haiku enrichment (what this cluster is about, why these nodes belong together) |
| `keywords` | Haiku enrichment |
| `situation` | Haiku enrichment (when is this community relevant for recall) |
| `confidence` | Average member confidence |
| `encoding_source` | `'s2:community_detection'` |
| `embeddings` | Generated via standard pipeline (title, blend, high_meta, other_meta) |
| `edges` | Bidirectional `community_member` edges to all members |

### Emergent characterization
Haiku sees the internal edge type distribution and names the community accordingly. A community dominated by `corrects`/`extends` edges gets described as an evolution cluster. A community dominated by `co_accessed` edges gets described as a usage pattern. The dimension is an output of enrichment, not an input to detection.

### No node_communities table
Community membership = edges. The `node_communities` table is dropped. Any query that needs "which community is node X in?" follows the `community_member` edges. Redistribution.py gets updated to query edges instead.

---

## Diff Strategy

### Matching new to existing
Overlapping communities don't have stable integer IDs across runs (SLPA is non-deterministic). Match by **member overlap**:
- For each new community, compute Jaccard similarity with each existing community node's members
- Match if Jaccard > 0.5 (majority of members overlap)
- Unmatched new → create
- Unmatched existing → archive

### Stability gate
If < N% of total memberships changed, skip the Haiku call and write. Avoids churning community nodes on every idle run.

---

## Selective Re-running

### Not every idle run needs full detection
Track `s2_community_last_run` and `s2_community_edge_count_at_last_run` in brain config.

**Full run** when:
- First run (no existing community nodes)
- Edge count changed by > 10% since last run
- Explicit trigger (manual or signal)

**Skip** when:
- Within cooldown period
- Edge count stable (< 10% change)

**Haiku call** only when:
- New communities detected (need naming)
- Existing community membership changed significantly (need re-description)
- Stable communities keep their existing enrichment

---

## Dependencies

| Package | Purpose | Size |
|---------|---------|------|
| `cdlib` | Overlapping community detection (SLPA) | Already installed |
| `igraph` | Graph operations (already have) | Already installed |
| `leidenalg` | Available as fallback for non-overlapping | Already installed |
| `networkx` | Required by cdlib | Installed with cdlib |

### Note on cdlib
cdlib pulls in matplotlib, pandas, seaborn as dependencies. These are heavy but only imported when specific algorithms are used. SLPA itself is lightweight. If the dependency footprint is a concern, we could vendor just the SLPA implementation (~200 lines).

---

## Challenges

### 1. SLPA non-determinism
SLPA uses random label propagation. Two runs on the same graph may produce slightly different communities. Mitigation: seed the random generator, use stability gate to avoid unnecessary churn.

### 2. Too many communities
SLPA r=0.1 produced 155 communities. After min_size filter and Haiku rejection, maybe 30-50 survive. But that's still a lot of community nodes + edges in the graph. Mitigation: aggressive min_size (5+), Haiku rejects noise clusters.

### 3. Haiku prompt size
155 proposals × 200 tokens = 31K tokens. Haiku's context is 200K, so this fits easily. But if the brain grows to 10K nodes, proposals could be larger. Mitigation: cap proposals at N (process largest/densest first), summarize member data more aggressively.

### 4. Edge embedding baseline
Using `related` as the baseline for specificity assumes `related` is generic. If `related` edges evolve to be more specific (better encoder), the baseline shifts. Mitigation: use the CENTROID of all relation embeddings as baseline instead of one specific relation.

### 5. networkx conversion
cdlib uses networkx. Our graph is in SQLite. We build a networkx graph each run. For 1719 nodes + 7468 edges, this takes <1 second. At 10K+ nodes, may need optimization.

### 6. Overlapping membership + recall
When S1R recalls a community node, it follows `community_member` edges. If a knowledge node belongs to 3 communities, those 3 community nodes compete for recall slots. Need to ensure community nodes don't crowd out knowledge nodes in the surface context. Mitigation: S1R can cap community nodes in surface results, or use community nodes for scoping only (not direct surfacing).

---

## Opportunities

### 1. Cross-project recall scoping
The immediate use case. When working on a different project, S1R can downweight communities that are brain-development-specific. Community `situation` field ("relevant when working on brain infrastructure") enables this.

### 2. Self-improving edge types
As the encoder creates more specific edge types, community detection automatically produces better-characterized communities. The semantic edge weighting rewards specificity — the system improves without changing community detection code.

### 3. Emergent dimension discovery
S3 reads community characterizations and discovers which dimensions exist. "I see 5 evolution communities, 12 topic communities, 3 operational communities." The dimensions aren't coded — they're observed from Haiku's descriptions.

### 4. Community-aware encoding
S1E could see "this conversation is about topics in community X" and prioritize encoding for gaps in that community. Future enhancement.

### 5. Boot context
"Your brain has N communities. Top active: [community titles]." Gives Anchor structural awareness of what the brain contains.

### 6. Dashboard visualization
Each community is a node. The graph tab already renders nodes and edges. Communities show up naturally as hub nodes with many connections.

---

## Files to Change

### New
- `servers/scales/s2/community.py` — rewrite with full pipeline
- `servers/scales/s2/community_contract.py` — rewrite with SLPA + enrichment config

### Modify
- `servers/scales/s2/base.py` — add shared utilities (graph loading, member edge sync)
- `servers/daemon_hooks.py` — already wired, may need Haiku client setup
- `servers/interaction_seed.py` — register `s2_community_enrichment` interaction
- `servers/redistribution.py` — query edges instead of node_communities table
- `servers/schema.py` — mark node_communities for removal
- `servers/trace_contract.py` — already done
- `build-plugin.sh` — already done
- `tests/test_s2_community.py` — rewrite for new pipeline

---

## Implementation Order

1. Remove node_communities table dependency (redistribution.py)
2. Rewrite community_contract.py with SLPA + enrichment config
3. Rewrite community.py with full pipeline (load → embed → detect → assemble → enrich → diff → write)
4. Register enrichment interaction
5. Write tests (synthetic + real brain eval)
6. Run on copy, evaluate results
7. Ship to production daemon
