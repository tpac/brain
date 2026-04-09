# Next Work — After Session 2026-04-08 (edge model v22 + S2 foundation)

## This Session

### Edge Model v22 — Multi-Relation + Single-Direction
- **Schema v22:** `edge_id` as PK on edges, `edge_relations` table with `(edge_id, relation)` PK
- **Single-direction storage:** 16,039 → 8,017 edges (mirrors deduplicated). Direction = source is actor.
- **Multi-relation:** One edge pair carries N typed relations. `add_relation()` adds without overwriting.
- **Deprecated columns removed:** `relation`, `edge_type`, `description`, `stability`, `decay_rate` dropped from edges
- **`encoding_source` on edge_relations:** tracks who created each relation (`encoder:sonnet`, `s2:relation_migration`, `hebbian`, etc.)
- **Direction-aware rendering:** "this extends X" vs "X extends this" — natural language for contextless LLMs
- **`auto_connect=False`** parameter on `remember()` — prevents Machine 1 noise for programmatic node creation

### LLM Relation Reclassification
- **2,621/2,643** generic `related` edges reclassified to specific types by Sonnet
- **120 unique relation types** emerged (extends, exemplifies, implements, addresses, validates, resolves, enables, contextualizes, explains, informs, supersedes, constrains, etc.)
- **22 edges kept as `related`** — Sonnet agreed they were genuinely generic
- **0 errors** across 53 batches
- **Provenance:** `encoding_source='s2:relation_migration'` on all reclassified edges

### S2 Infrastructure
- **`IntegrationUnit` base class** (`scales/s2/base.py`) — universal O/K/Δ contract for any scale
- **Community detection unit** (`scales/s2/community.py`) — Leiden-based, tested on real data (26 communities from 1719 nodes)
- **Reclassification unit** (`scales/s2/reclassify.py`) — Sonnet-based edge type classification
- **Trace contract updated:** S2="Graph", S3="Reasoning" with proper ref_types
- **`s2_community` interaction** registered in interaction_seed.py

### Proto-S2 Disabled
- `auto_heal()`, `dream()`, `consolidate()`, `auto_tune()`, `prompt_reflection()`, `auto_generate_self_reflection()` — all commented out from idle hook
- These were proto-S2 mechanisms. S2 integration units replace them.
- Code kept in `brain_evolution.py`, `brain_dreams.py` for reference

### Bug Fixes
- **`correction_enrich()` was querying wrong table** — `node_metadata` (empty) instead of `node_metadata_kv` (has data). Fixed.
- **Edge to nonexistent node** — `add_relation()` now validates both nodes exist, raises `ValueError`
- **`remember()` catches connection errors** — logs to error log instead of crashing
- **Ambiguous `target_id` column** in structural degree cache query — aliased with `e.`

### Pipeline Updates
- `pipeline_contract.py` — batch edge fetch reads `edge_relations` via `edge_id` JOIN, returns `direction` per connection
- `surface.py` — graph neighbor query uses `edge_relations` JOIN
- `surface_contract.py` — `correction_enrich()` uses metadata only (no hardcoded edge types)
- `daemon_dispatch.py` — get_neighbors handler updated
- `redistribution.py` — edge queries use `edge_relations` JOIN, both-direction
- `signal_producers.py` — edge type distribution reads `edge_relations`
- `brain_evolution.py` — deprecated column writes replaced with `add_relation()`
- `contract.py` — `render_rich_node()` direction-aware, multi-relation rendering
- `brain_connections.py` — `connect()`, `connect_typed()` delegate to `add_relation()`, bridge finder updated
- `dashboard/brain_dashboard_standalone.py` — all 5 query sites updated for v22

### Encoder Prompt Updated
- `connect_to` supports `relations` array: `[{relation: "extends", why: "builds on..."}, ...]`
- `brain_batch connect` op includes `description` field
- Emphasizes open text relation types, not a closed list

### Test Results
- **527 passing, 9 pre-existing failures** (trace_contract_sync, pipeline_contract, spread_activation)
- **18 new edge_relations tests** covering: migration, multi-relation, encoding output, query, Hebbian preservation, decay, cascade, backward compat, encoding_source tracking, direction, single-direction storage
- Tests updated for v22 schema across: test_core, test_format_node, test_mcp_roundtrip, test_s1_data_assembly, test_s2_community, test_surface_transitions, test_redistribution

### Deprecated / Removed
- `spread_activation()` — marked deprecated (legacy, `_traverse_graph()` is active)
- `stability` field — no longer written to edges
- `co_access_count` promotion to `related` — removed from auto_heal
- `STABILITY_BOOST`, `STABILITY_FLOOR_*` — marked deprecated in constants
- `node_communities` table — community membership is now edges, table scheduled for removal

---

## Next: S2 Community Detection (redesigned)

### Design (from docs/S2-COMMUNITY-DESIGN.md)
The community detection unit needs rewriting for the new edge model:

1. **Overlapping communities** via SLPA (cdlib) — nodes can belong to multiple communities
2. **Edge embedding** — embed relation text for semantic edge weighting. `related` baseline → specificity factor
3. **Batched Haiku enrichment** — ONE call with all proposals, Haiku names/describes/rejects each
4. **Community nodes as regular nodes** — type='community', full enrichment, bidirectional edges to members
5. **Emergent characterization** — community "dimension" emerges from internal edge type distribution, not predefined

### What's ready
- `edge_relations` has 120+ typed relations (from reclassification) — real data for characterization
- cdlib + leidenalg + igraph installed
- SLPA tested on real data: 155 communities, 396 multi-member nodes at r=0.1
- `IntegrationUnit` base class ready
- Daemon embedder available during idle hook (no extra memory)
- Anthropic client available for Haiku enrichment

### What needs building
- Rewrite `community.py` for SLPA overlapping detection
- Edge embedding for semantic weighting (embed relation text, compare to baseline)
- Haiku enrichment prompt + batched call
- Diff strategy with Jaccard similarity (overlapping communities don't have stable IDs)
- Community node lifecycle (create/revise/archive)

---

## Remaining Cleanup

### CLAUDE.md Stale References
Still has references to deleted code from earlier sessions. Same list as before:
- `store_exchange() → message_stream`, `dal_message_stream.py`, shim files
- Stop hook description mentions message_stream
- Encoding section references old file paths
- **NEW:** Edge model section needs rewrite for v22 (edge_id, single-direction, edge_relations)

### Duplicate Node Clusters
Still present (21 compaction + 16 session handoff). S2 dedup unit will handle.

### Dead Code Candidates
- `brain_evolution.py` — auto_heal, auto_tune, auto_discover disabled. Keep for S2 reference, mark for deletion after S2 dedup/confidence units are built.
- `brain_dreams.py` — dream, consolidate disabled. Same.
- `brain_consciousness.py` — priming system still active but separate from S2.
- `spread_activation()` in brain_recall.py — deprecated stub.
- `node_communities` table — drop after community detection uses edges.

### Edge Model Polish
- `select_edges()` Strategy D — could add relation type as scoring signal now that types are specific. Needs eval with new data.
- Edge rendering shows `related` as verb ("this related X") — reads poorly. Most `related` edges now reclassified, so this is minor.
- The 120 relation types should eventually converge to a smaller vocabulary as the encoder learns to use consistent types.

---

## S3/S4: Future Scales

### S3: Reasoning (periodic)
Reads S2 output (clusters, trajectories, landscapes). Connects clusters into larger patterns.
- **Status:** Architecture designed (docs/ARCHITECTURE-FRACTAL.md), not built

### S4: Growth (weekly)
External research, web search. Reads S3 traces.
- **Status:** Not built

---

## Infrastructure Notes

### Python 3.9 Ceiling
Still locked to onnxruntime 1.19.2. Monkey-patch works but fragile.

### Dependencies Added This Session
- `leidenalg` + `python-igraph` — community detection
- `cdlib` — overlapping community detection (pulls matplotlib, pandas, seaborn)

### Maintenance Mode
`touch /tmp/brain-maintenance-{uid}.lock` prevents daemon auto-restart. Used during v22 migration.

### Test Architecture
Tests organized by what they catch (same taxonomy). New: `test_edge_relations.py` (18 tests) for edge model.
