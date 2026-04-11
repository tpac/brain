# S2 Community Detection — Design (Updated 2026-04-11)

## Status: SHIPPED TO PRODUCTION

114 communities live on production brain (cold start rerun 2026-04-11, 12 orphans self-healed, 1 duplicate merged).
Decoder/encoder split shipped. Dashboard 3D graph with legend click-to-focus. Merge detection with adaptive threshold for young brains. Community split NOT BUILT (roadmap).

## Architecture: S2CD / S2CE Decoder-Encoder Pair

Same pattern as S1R/S1E. Decoder finds structure, encoder characterizes it.

### S2CD — Community Decoder (algorithmic, <1s)

```
_decode()
├── _build_typed_adjacency()         → non-noise edges by family
├── _compute_pair_scores()           → z-scores by degree bucket
├── _seed_clusters()                 → z≥1.0 + direct edges
├── _validate_clusters()             → dissolve fragments, flag corridors
├── _compute_affinities()            → every node → {cluster: affinity}
├── _detect_cross_cutting()          → high-degree thin-spread nodes
├── _compute_orphan_affinities()     → embedding centroid matching
├── _analyze_ties()                  → overlap vs split signal
├── _build_proposals()               → community + affinity + drift proposals
└── Incremental: add_to_existing, drift detection, health updates
```

**No static thresholds** — all scoring is relative:
- Z-score pair scoring within degree buckets (adapts to graph density)
- Adaptive grow threshold (median of all affinities)
- Ratio-based overlap (secondary/primary ≥ 0.5)
- Cross-cutting detection (degree ≥ 15, top affinity < 35%)

**Incremental path**: placed nodes excluded from seeding. New nodes matched against existing communities. Drift detected when node's foreign affinity > home affinity × 1.5. Health updates when internal fraction degrades.

### S2CE — Community Encoder (Sonnet, agentic)

Uses `brain_batch` tool to create community nodes directly. Same `run_llm_loop` as S1E.

Prompt (v6, `s2_community_enrichment` interaction):
- "What pattern do these nodes reveal that no single node names?"
- "What would change how the next you approaches this area?"
- FLAT → RICH transformations (summary→insight, history→wisdom, technical→relational)
- Field guide: reads proposal data (int_frac, edge signature, timeline) as story signals

Tools: `brain_batch` (create + revise + connect), `get_nodes` (read).
Target: 1-2 rounds. Batched at 10 proposals per Sonnet call.

### Community Node Structure

Type `community`, encoding_source `s2:community_detection`.

**Visible to Anchor** (in render_rich_node):
- content: narrative with node references and dates
- situation: discriminating recall trigger
- community_narrative: 2-4 sentence arc
- community_key_decisions: "id: title" pairs
- community_open_questions: what's unresolved
- community_latest_development: most recent change
- community_maturity: forming/active/settled/corridor
- community_dominant_type: most common member type
- community_members: "id: title" pairs
- Open fields: community_learning_arc, community_tension, community_risk, etc.

**Hidden from Anchor** (machine-readable for S2/S3):
- community_internal_fraction, community_internal_edges, community_external_edges
- community_centroid, community_is_corridor, community_size
- community_growth_rate, community_run_count

**Source of truth**: `community_member` edges. Metadata is denormalized cache.

## Supporting Infrastructure

### S2 Edge Family Integration (`edge_families.py`)
- Classifies 224 edge relation types into 21 semantic families
- Sonnet classification with description samples
- Stored in `s2_edge_families` interaction (versioned, S3-editable)
- Used by decoder for relational signal analysis

### Shared S2 Base (`base.py`)
- `_has_new_traces()`, `_read_traces_since()`, `_last_run_timestamp()`
- `_call_llm()` — learnable prompt from interactions table
- `_extract_json()` — robust JSON parsing from LLM responses
- Reusable by all future S2 units (dedup, confidence, etc.)

### Trace Contract (updated)
- S2 O: `s1_delta`, `graph_structure`
- S2 K: `community_proposals`
- S2 delta: `community_enriched`, `community_created`, `recall_quality_signal`

### Eval Harness (`eval/s2_community_eval.py`)
- Scores communities across 10 dimensions
- Composite score, duplicate detection, dimension breakdown
- Latest eval: 0.983 composite on v6 prompt

## Bug Fixes Shipped

1. **Dispatch open fields bug** — `_handle_remember` and `_handle_remember_batch` silently dropped all fields not in contract whitelist. S1E's open fields (`assumed`, `reality`, `trigger`, etc.) were lost for months. Fixed: all fields pass through.

2. **Runner trace enrichment** — S1E delta traces now carry structured `{created, revised, connected}` metadata + full tool input. Previously only summaries.

3. **S1E community node exclusion** — community nodes excluded from S1E's node catalog. S2CE owns community membership, S1E encodes from conversation.

4. **Metadata render filtering** — structural community metrics (internal_fraction, centroid, etc.) hidden from Anchor's view. Only human-readable metadata shown.

5. **register_interaction MCP tool** — added for managing learnable boundaries from conversation.

## What's Next

### Immediate
- **Fatigue redesign** — message-distance-based fatigue replacing count-based. Doc: `docs/FATIGUE-REDESIGN.md`
- **Dashboard polish** — S2 decode/encode display in Live tab, scale filter
- **S2 Weaver/Healer** — connect orphan nodes (635 with no typed edges)

### Short-term
- **Incremental production runs** — idle hook triggers S2CD/S2CE on new S1 traces
- **Boot integration** — community summaries at session start
- **S1R integration** — community_key_decisions prioritized in graph expansion
- **Community merge** — detect and merge converging communities

### Medium-term
- **Other S2 units** — dedup, confidence recalibration, correction chain resolution
- **S3 foundation** — reads S2 traces, evaluates community quality over time
- **Progressive simulation eval** — replay brain history to validate incremental path

## Files Changed

| File | Change |
|------|--------|
| `servers/scales/s2/community.py` | Full rewrite — S2CD + S2CE pipeline |
| `servers/scales/s2/community_contract.py` | Config, metadata schema, node rendering |
| `servers/scales/s2/community_enrichment_prompt.py` | S2CE Sonnet prompt (v6) |
| `servers/scales/s2/edge_families.py` | Edge type classification unit |
| `servers/scales/s2/edge_families_v1.json` | Initial 21-family classification |
| `servers/scales/s2/base.py` | Shared S2 infrastructure |
| `servers/scales/runner.py` | Structured trace metadata + full tool input logging |
| `servers/scales/s1/encode_contract.py` | Exclude community nodes from S1E catalog |
| `servers/daemon_dispatch.py` | Open fields pass-through fix + register_interaction |
| `servers/daemon_hooks.py` | Simplified idle hook + edge family integration |
| `servers/interaction_seed.py` | S2CE + edge families interaction seeding |
| `servers/trace_contract.py` | S2 community ref_types |
| `servers/contract.py` | Metadata render filtering for community nodes |
| `servers/redistribution.py` | Edge-based community lookup |
| `servers/brain_mcp.py` | register_interaction MCP tool |
| `dashboard/brain_dashboard_standalone.py` | 3D graph, Decoding/Encoding tabs, scale filter |
| `eval/s2_community_eval.py` | Community quality eval harness |
| `scripts/recover_s1e_open_fields.py` | Metadata recovery script |
| `docs/FATIGUE-REDESIGN.md` | Fatigue redesign spec |
| `dashboard/Dashboard-nextwork.md` | Dashboard roadmap |
