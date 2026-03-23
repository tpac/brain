# Embedding Migration to LLM Plan

**Date:** 2026-03-23
**Status:** Draft for review
**Context:** 12 recall experiments proved that encode-time LLM intelligence (V5 hybrid, NDCG 0.701) beats runtime sophistication (cross-encoder 0.518) with zero latency cost. The shift: move intelligence from the embedding model to an LLM at encode time. Arctic stays as the vector engine, but the LLM generates what gets embedded. The architecture is clear. Now we build it, clean the codebase, and make every fresh Claude session productive from turn 1.

---

## Evidence: 12 Conditions Tested (2026-03-23)

We tested every viable approach to improving recall. Here are all results against a 104-case golden dataset spanning 15 categories (semantic, natural_question, old_but_valid, correction_chain, keyword, mechanism, etc.):

| # | Approach | NDCG@10 | MRR | Passed | Latency | Verdict |
|---|---|---|---|---|---|---|
| 1 | **v1.5 baseline** | 0.204 | 0.202 | 34/104 | 106ms | Current production |
| 2 | Arctic v2.0 (newer model) | 0.198 | 0.197 | 33/104 | ~same | **Worse** — v2.0 regresses on our data |
| 3 | HyDE + TinyLlama (bare prompt) | 0.204 | 0.207 | 34/104 | 148ms | **No effect** — model hallucinated |
| 4 | HyDE + TinyLlama (rich context) | 0.204 | 0.204 | 35/104 | 105ms | **No effect** — ignored context |
| 5 | HyDE + Gemma 2B (bare prompt) | 0.204 | 0.207 | 34/104 | 129ms | **No effect** — generic output |
| 6 | HyDE + Gemma 2B (rich context) | 0.234 | 0.243 | 35/104 | 107ms | **Small gain** — first sign of life |
| 7 | v1.5 + MiniLM reranker (22M) | 0.232 | 0.241 | 35/104 | 131ms | **Small gain** — model too weak |
| 8 | v1.5 + bge-v2-m3 reranker (278M) | 0.494 | 0.510 | 61/104 | 4278ms | **Huge gain, way too slow** |
| 9 | v1.5 + gte-modernbert reranker (149M) | 0.518 | 0.533 | 61/104 | 2168ms | **Huge gain, still too slow** |
| 10 | v2.0 + gte-modernbert reranker | 0.522 | 0.533 | 62/104 | 2120ms | **v2.0 adds nothing on top of reranker** |
| 11 | **Multi-vec V2 (structured prompt)** | **0.704** | **0.706** | **91/104** | **~100ms** | **Best overall** |
| 12 | **Multi-vec V5 (hybrid)** | **0.701** | **0.697** | **93/104** | **~100ms** | **Best pass rate, chosen architecture** |

Also tested but not in main table:
- Multi-vec V1 (bare "list 3 questions"): NDCG 0.641, 89/104 — strong but V5 beats it
- Multi-vec V3 (motivational framing): NDCG 0.336, 51/104 — **disaster**, small models generate meta-commentary instead of answers when given motivational prompts
- Multi-vec V4 (anchor+bridge only): NDCG 0.598, 86/104 — good hit rate but weaker ranking than V5

### Why V5 Hybrid Wins

**The core problem:** Arctic embeds "Separate API + Web architecture" and "why did we separate backend from frontend" into vectors that land far apart. Same meaning, different words. Single-vector compression can't capture the equivalence.

**Why runtime approaches fail:**
- **Better embeddings (Arctic v2.0):** Same fundamental problem — one vector per document can't bridge vocabulary gaps
- **HyDE (query expansion):** Small local LLMs (1-2B) lack the knowledge to generate domain-relevant expansions. Even with 2750 chars of brain context, TinyLlama called Glo "an online electronics marketplace." They can't follow instructions well enough.
- **Cross-encoder reranking:** PROVES the quality ceiling exists (0.518 NDCG) — when a model sees query+document together, it understands "backend/frontend" = "API/Web." But 2-4 seconds per query is unacceptable for hooks that fire every message.

**Why V5 encode-time enrichment works:**
- At encode time, Claude (or a local LLM with structured prompt) generates: one question, one anchor phrase, one bridge sentence, shared keywords
- These get embedded alongside the original content — 4 extra vectors per node
- At recall time, "why separate backend from frontend" matches the question vector "why did we split backend and frontend" directly
- **Zero runtime cost** — the intelligence investment happens once at storage time
- NDCG 0.701, 93/104 passed — beats the cross-encoder's 0.518 without any latency penalty

**Why V5 over V2:**
- V2 (3 questions) scores NDCG 0.704 — slightly higher
- V5 (1 question + 1 anchor + 1 bridge + keywords) scores 0.701 with 93/104 passed vs 91/104
- V5 has broader coverage (higher pass rate) because anchors and bridges create embedding overlap with neighbors, making nodes findable from more angles
- V5 is more diverse — 4 different vector types vs 3 similar questions that may be redundant

---

## Phase 1: V5 Multi-Vector Encoding (THE core change)

**What:** When `remember()` is called, the brain recalls neighbors and asks Claude to generate enrichment vectors.

**Architecture:**
```
Claude calls remember(title, content, keywords, ...)
  → Brain receives it
  → Brain recalls 5 related nodes (embedding + edge neighbors)
  → Brain returns structured prompt to Claude:
      "I found these related memories: [neighbors with titles, types, keywords]
       New node: {title}
       Generate:
       Q: [one question a user would ask to find this]
       A: [3-5 word anchor using neighbor vocabulary]
       B: [one sentence bridge to most important neighbor]
       K: [5 keywords from neighbors that also apply]"
  → Claude fills in the form
  → Brain embeds Q, A, B, K alongside original content
  → Brain creates edges to neighbors
  → Brain adjusts neighbor confidence if warranted
```

**Files to change:**
| File | Change | Lines |
|---|---|---|
| `brain_remember.py` | Add `enrich_node()` — generates structured prompt, stores multi-vectors | ~80 |
| `brain_recall.py` | STEP 7: scan enrichment embeddings alongside originals, best-score-per-node | ~40 |
| `dal.py` | `EnrichmentDAL` — CRUD for `node_enrichments` table (node_id, vector_type, text, embedding) | ~60 |
| `schema.py` | Add `node_enrichments` table migration | ~15 |
| `brain_mcp.py` | Expose `enrich` tool so Claude can return the filled form | ~20 |
| `brain_constants.py` | V5 tuning params (ENRICHMENT_NEIGHBOR_COUNT=5, ANCHOR_MAX_WORDS=5) | ~10 |

**New table:**
```sql
CREATE TABLE node_enrichments (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    vector_type TEXT NOT NULL,  -- 'question', 'anchor', 'bridge', 'keywords'
    text TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);
CREATE INDEX idx_enrichments_node ON node_enrichments(node_id);
```

**Testing:**
- Run golden dataset before and after: NDCG must improve, not drop
- Benchmark script already exists: `tests/benchmark_multivec_encoding.py`
- Add enrichment integration test to `tests/test_core.py`

**Backfill:** One-time script to enrich existing ~400 brain-relevant nodes (post-Glo cleanup). Uses Gemma 2B with V2 structured prompt. ~7 minutes.

---

## Phase 2: Boot Experience (make fresh Claude SEE the brain)

**What:** Replace the stats-dump boot with a visceral experience that shows Claude what the brain IS and what happens when encoding is good vs bad.

**Current boot output:** 50+ lines of stats, rules, vocabulary, consciousness signals. Claude skims it.

**New boot structure:**
```
[BRAIN] The brain remembers so you don't have to.

STRUCTURE: 425 nodes, 8000 edges across 3 clusters:
  Recall pipeline (15 nodes, hub: recall_with_embeddings)
  Brain architecture (20 nodes, hub: v15 Architecture)
  Encoding lessons (12 nodes, hub: recall-before-encode)
  Weak bridges: [architecture ↔ lessons: only 2 edges]

LAST WIN: "why does recall miss old nodes?" → found "Embedding similarity floor"
  (encoded once, saved 3 sessions of rediscovery)

LAST LOSS: "how does temporal boosting work?" → found nothing
  (node existed but had no question vectors — invisible)

ENCODE GENEROUSLY. Every unencoded insight is lost at compaction.
The brain improves ONLY through what you store. More nodes = better recall.
Use remember() often. Use connect() to link related nodes.

{3-5 consciousness signals: tensions, reminders, fading knowledge}
```

**Files to change:**
| File | Change |
|---|---|
| `brain_surface.py` | `format_boot_context()` → new structure with win/loss/topology |
| `brain_voice.py` | `_operator_boot_summary()` → shorter, focused on tensions + reminders |
| `brain_consciousness.py` | Add `get_graph_topology()` — find clusters, hubs, weak bridges |
| `dal.py` | `get_recall_win_loss()` — query precision data for best/worst recent recalls |

**Key principle:** One adaptive system, not multiple guides. Each hook surfaces what's relevant for the current action. The boot shows the shape. The hooks fill in the details.

---

## Phase 3: Clean Architecture

### 3a: Remove Glo nodes from brain
465 Glo-specific nodes archived to `exports/glo_archive_2026-03-23.md`.
Delete nodes where `project='Glo'` EXCEPT any that are pure brain lessons (type=lesson, type=correction, type=divergence that reference encoding/recall/hooks).

### 3b: Flatten the mixin monolith
`brain.py` is 1886 lines — a God class that inherits from 11 mixins. The mixins are:

| File | Lines | Purpose | Status |
|---|---|---|---|
| `brain.py` | 1886 | God class + __init__ + core | REFACTOR: extract to thin orchestrator |
| `brain_absorb.py` | 282 | Transcript absorption | Keep as-is |
| `brain_connections.py` | 391 | Edge creation/management | Keep as-is |
| `brain_consciousness.py` | 1330 | 12 consciousness signals | SPLIT: signals vs analytics |
| `brain_constants.py` | 282 | Config constants | Keep as-is |
| `brain_dreams.py` | 401 | Dream generation | Keep as-is |
| `brain_engineering.py` | 1683 | Engineering memory types | REVIEW: overlap with remember? |
| `brain_evolution.py` | 2101 | Self-improvement, patterns | REVIEW: what's actually used? |
| `brain_precision.py` | 756 | Recall precision scoring | Keep as-is |
| `brain_recall.py` | 1281 | Recall pipeline | Keep — add V5 enrichment scan |
| `brain_remember.py` | 985 | Storage pipeline | Keep — add enrichment trigger |
| `brain_surface.py` | 946 | Format output for hooks | Keep — rewrite boot |
| `brain_vocabulary.py` | 168 | Vocab CRUD | Keep as-is |
| `brain_voice.py` | 1102 | Operator channel, hook output | Keep as-is |

**Target architecture:**
```
brain.py (thin orchestrator, <200 lines)
  ├── brain_recall.py (recall + enrichment scan)
  ├── brain_remember.py (store + enrichment trigger)
  ├── brain_connections.py (edges)
  ├── brain_consciousness.py (signals only, <800 lines)
  ├── brain_precision.py (scoring)
  ├── brain_vocabulary.py (vocab)
  ├── brain_voice.py (operator channel)
  ├── brain_surface.py (format for hooks)
  └── dal.py (ALL database access)
```

Remove or merge:
- `brain_evolution.py` (2101 lines) — audit what's used, extract useful methods, delete the rest
- `brain_engineering.py` (1683 lines) — most methods are convenience wrappers around remember(). Keep `remember_lesson`, `remember_impact`. Move rest to SKILL.md guidance.
- `brain_absorb.py` — rarely used, can stay but review
- `brain_dreams.py` — keep but review usage frequency

### 3c: DAL consolidation
`dal.py` is only 514 lines but many modules still do raw SQL. Audit all `cursor.execute()` calls outside dal.py and migrate them.

### 3d: Test cleanup
```
tests/
  golden_dataset.json          # Keep — 104 cases
  eval_runner.py               # Keep — NDCG/MRR scoring
  benchmark_multivec_encoding.py  # Keep — V5 benchmark
  benchmark_v4_v5.py           # Keep — comparison
  test_core.py                 # Keep — unit tests
  test_recall_quality.py       # Keep
  test_precision.py            # Keep

  # ARCHIVE (move to tests/archive/):
  benchmark_hyde_tinyllama.py   # Historical — HyDE failed
  benchmark_hyde_gemma.py       # Historical — HyDE failed
  benchmark_hyde_rich_context.py # Historical — HyDE marginal
  benchmark_v15_reranker.py     # Historical — reranker results
  bench_*.py                    # Old micro-benchmarks
```

### 3e: Hook cleanup
34 hook scripts (17 .sh + 17 .py). Each .sh is a thin wrapper that calls .py.
Audit: which hooks actually fire? (24h data shows post_response_track: 184, recall: 102, others single digits).

Low-value hooks to review:
- `idle-maintenance.sh` — fires during idle. Is consolidation actually helping?
- `worktree-context.sh` — 10 fires, keeps failing
- `config-change-host.sh` — 2 fires, niche

---

## Phase 4: Ripple Effect (after V5 is stable)

**What:** New information updates old nodes. When remember() creates a node:
1. Brain identifies which neighbors are validated/contradicted/extended
2. Validated neighbors: confidence +0.05, add edge
3. Contradicted neighbors: confidence -0.1, create correction edge, re-embed with new context
4. Extended neighbors: add new question vector from the new node's perspective

This keeps old nodes alive. Currently old_but_valid scores 0.000 in baseline — old nodes become invisible because they never get refreshed.

**Depends on:** Phase 1 (V5 enrichment infrastructure) and Phase 2 (Claude returns structured impact assessment).

---

## Phase 5: Session-Start Graph Topology

**What:** At boot, analyze the graph and tell Claude:
- Dense clusters (where knowledge is strong)
- Weak bridges (where connections are missing)
- Hub nodes (most-connected, likely important)
- Recently active areas vs stale areas

Claude naturally encodes to strengthen weak areas when it sees them.

---

## Phase 6: End-to-End Testing & Telemetry

**Problem:** Things fail silently. The daemon crashes and leaves a stale socket. Hooks swallow errors with `2>/dev/null`. Encoding happens but nobody knows if the enrichment vectors actually improved recall. The precision pipeline evaluates 5% of recalls. We fly blind.

### 6a: E2E Integration Tests

Full pipeline tests that exercise the real path, not unit-test mocks:

| Test | What it validates | How |
|---|---|---|
| **encode → recall roundtrip** | Store a node with V5 enrichment, recall it by question phrasing | Create node "API uses REST" → verify recall("what protocol does the API use") finds it |
| **enrichment generation** | V5 prompt returns parseable Q/A/B/K | Encode 10 diverse nodes, verify each gets 4 enrichment vectors in `node_enrichments` |
| **enrichment recall boost** | Enriched nodes rank higher than unenriched | Encode node with and without enrichment, same query, compare ranks |
| **ripple propagation** | New node updates neighbor confidence | Encode node that validates an existing node, verify confidence change |
| **boot context generation** | Boot output includes win/loss/topology | Call `format_boot_context()`, verify structure has required sections |
| **daemon lifecycle** | Start → encode → recall → crash → restart → recall still works | Kill daemon mid-session, verify auto-restart and data survives |
| **hook chain** | SessionStart → recall → encode → SessionEnd all fire | Simulate a session lifecycle, verify each hook produces output |
| **DB copy isolation** | Test agents don't corrupt live DB | Run benchmark against copy, verify live DB has zero changes |

**Test runner:** `tests/test_e2e.py` — uses real daemon, real DB (temp copy), real embedder. Not mocks. Run with `python3 tests/test_e2e.py`.

### 6b: Telemetry Dashboard

Every critical operation logs timing, success/failure, and output quality to `brain_telemetry` table:

```sql
CREATE TABLE brain_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    operation TEXT NOT NULL,     -- 'recall', 'remember', 'enrich', 'boot', 'hook_fire', 'ripple'
    duration_ms REAL,
    success INTEGER NOT NULL,   -- 1 or 0
    error_message TEXT,         -- NULL on success, full traceback on failure
    metadata TEXT               -- JSON: {query, result_count, ndcg, enrichment_count, ...}
);
CREATE INDEX idx_telemetry_op ON brain_telemetry(operation, timestamp);
CREATE INDEX idx_telemetry_fail ON brain_telemetry(success) WHERE success = 0;
```

**What gets logged:**

| Operation | Metrics logged | Failure means |
|---|---|---|
| `recall` | query, duration_ms, result_count, top_score, enrichment_hits (how many results came from enrichment vectors vs original) | Embedding failed, DB locked, zero results for non-empty query |
| `remember` | node_id, type, duration_ms, enrichment_generated (bool) | DB write failed, enrichment prompt failed |
| `enrich` | node_id, vectors_generated (0-4), prompt_tokens, parse_success | LLM unreachable, parse failure, zero usable vectors |
| `boot` | duration_ms, nodes_loaded, enrichments_loaded, topology_generated | Daemon unreachable, DB corrupt |
| `hook_fire` | hook_name, duration_ms, output_bytes | Hook script error, timeout, empty output |
| `ripple` | source_node, neighbors_updated, confidence_deltas | Ripple query failed, no neighbors found |
| `daemon_health` | uptime_s, memory_mb, socket_alive, embedder_loaded | Socket stale, OOM, embedder crashed |

**No silent failures rule — enforced by code:**
```python
def log_telemetry(operation, success, duration_ms, error=None, **metadata):
    """Every operation MUST call this. Wrap with decorator."""
    # Writes to brain_telemetry table
    # If success=0, ALSO writes to brain_logs.db (existing error log)
    # If success=0 AND operation in CRITICAL_OPS, surfaces as consciousness signal
```

**Decorator for automatic instrumentation:**
```python
@telemetry("recall")
def recall_with_embeddings(self, query, limit=10):
    ...  # existing code, unchanged
    # decorator auto-logs: duration, success/failure, result count
```

### 6c: Health Check Endpoint

The existing `health_check` MCP tool gets upgraded:

```json
{
  "status": "healthy",
  "uptime_s": 3600,
  "daemon_pid": 97675,
  "socket_alive": true,
  "embedder_loaded": true,
  "db_path": "/Users/tpac/AgentsContext/brain/brain.db",
  "db_size_mb": 45.2,
  "node_count": 425,
  "enrichment_count": 1200,
  "enrichment_coverage": "72%",
  "telemetry_24h": {
    "recalls": 102,
    "recall_failures": 0,
    "avg_recall_ms": 95,
    "enrichment_hit_rate": "45%",
    "remembers": 22,
    "enrichment_generations": 18,
    "enrichment_parse_failures": 2,
    "hook_fires": 310,
    "hook_failures": 3,
    "ripples": 15
  },
  "last_error": null,
  "stale_socket": false
}
```

### 6d: Regression Detection

After every code change to sacred systems, before committing:

1. **Golden dataset gate:** `python3 tests/eval_runner.py` — NDCG must not drop below 0.60 (post-V5 target)
2. **E2E gate:** `python3 tests/test_e2e.py` — all roundtrip tests pass
3. **Telemetry gate:** No new failure patterns in last 100 operations

**Pre-commit hook (optional but recommended):**
```bash
# .claude/hooks: PreToolUse(Edit) on servers/*.py
# Already exists (pre-edit-suggest.sh) — extend to include:
# "WARNING: This file is a sacred system. Run golden dataset before committing."
```

### 6e: Enrichment Quality Monitoring

Track whether enrichment vectors actually get used at recall time:

```python
# In recall pipeline, after scoring:
for result in results:
    if result.matched_via == 'enrichment':
        log_telemetry("recall", metadata={"enrichment_hit": True, "vector_type": result.vector_type})
    else:
        log_telemetry("recall", metadata={"enrichment_hit": False})
```

Surface as consciousness signal:
- "Enrichment hit rate: 45% of recalls matched via enrichment vectors (Q: 30%, A: 8%, B: 5%, K: 2%)"
- "3 nodes have enrichments that NEVER matched in 50+ recalls — review enrichment quality"
- "Enrichment coverage: 72% of nodes have enrichments. 120 nodes unenriched."

This closes the loop: we don't just generate enrichments — we KNOW if they work.

---

## Execution Order

| Step | Phase | Est. effort | Depends on |
|---|---|---|---|
| 1 | 3a: Clean Glo nodes | 30 min | Glo export (done) |
| 2 | 1: V5 multi-vector encoding | 4-6 hours | — |
| 3 | 6a: E2E tests for V5 | 2-3 hours | Step 2 |
| 4 | 1: Backfill existing nodes | 30 min | Step 3 passes |
| 5 | 6b: Telemetry table + decorator | 2-3 hours | Step 2 |
| 6 | 2: Boot experience rewrite | 2-3 hours | Steps 2, 5 |
| 7 | 3b-3e: Clean architecture | 6-8 hours | Steps 2-6 stable |
| 8 | 6c-6e: Health check + regression + monitoring | 3-4 hours | Steps 5, 7 |
| 9 | 4: Ripple effect | 4-6 hours | Steps 2, 6 |
| 10 | 5: Graph topology | 2-3 hours | Step 7 |

**Total:** ~27-37 hours across multiple sessions.

---

## Success Metrics

| Metric | Current | Target | How measured |
|---|---|---|---|
| Golden dataset NDCG | 0.204 | >0.60 | `tests/eval_runner.py` |
| Golden dataset pass rate | 34/104 | >80/104 | Same |
| Recall latency | 106ms | <150ms | Telemetry: avg_recall_ms |
| Enrichment coverage | 0% | >80% of brain nodes | `SELECT COUNT(*) FROM node_enrichments` |
| Enrichment hit rate | N/A | >30% of recalls use enrichment vectors | Telemetry |
| Silent failures (24h) | Unknown | 0 | `SELECT COUNT(*) FROM brain_telemetry WHERE success=0` |
| Hook failure rate | Unknown | <1% | Telemetry: hook_fires vs hook_failures |
| Boot context lines | 50+ | <20 | Count |
| brain.py lines | 1886 | <200 | wc -l |
| Total server code lines | 19,447 | <14,000 | wc -l servers/*.py |
| Fresh Claude encoding rate | Low (anecdotal) | Measurable via telemetry | remember ops per session |
| Node count (post-Glo cleanup) | 889 | ~425 | Brain stats |
| E2E test pass rate | N/A | 100% | `tests/test_e2e.py` |

---

## Non-Goals (explicitly out of scope)

- **ColBERT / per-token indexing** — overkill for <1000 nodes
- **External vector DB** — SQLite is fine at this scale
- **Fine-tuning Arctic** — V5 enrichment solves the problem without model changes
- **Larger local LLMs (7B+)** — Claude is already in the loop, no need for second large model
- **HTTP API** — Unix socket daemon works, keep it
