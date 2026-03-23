# Brain Components — Architecture Reference

**Last updated:** 2026-03-23
**Git version:** `67fae99` (Embedding Migration to LLM plan)

This document describes each component, its responsibility, key files, and how they connect. Written for a fresh Claude who needs to understand the system quickly.

---

## System Architecture

```
Claude Code → MCP server (brain_mcp.py, stdio) → daemon (Unix socket) → Brain + embedder
                                                                           ↓
                                                                    brain.db + brain_logs.db
```

The brain is a Python module behind a persistent daemon. The MCP server is a zero-dependency JSON-RPC relay. Claude Code starts it automatically.

---

## 1. MCP Server (`servers/brain_mcp.py`)

**Responsibility:** Thin stdio proxy. Receives JSON-RPC from Claude Code, forwards to daemon via Unix socket.
**Lines:** ~337
**Depends on:** Nothing (zero dependencies)
**Socket:** `/tmp/brain-daemon-{uid}.sock`

**Tools exposed:** recall, remember, connect, enrich, consciousness, context_boot, set_config, get_config, health_check, save, ping, eval, engineering_context

**Key rule:** This file should NEVER contain business logic. It's a relay.

---

## 2. Daemon (`servers/daemon.py`)

**Responsibility:** Persistent process that keeps the Brain + embedder in memory. Accepts commands via Unix socket.
**Lines:** ~967
**Depends on:** brain.py, embedder.py

**Lifecycle:**
- Started by `hooks/scripts/boot-brain.sh` at SessionStart
- Auto-timeout after 30 min idle
- Saves brain on shutdown, compaction, session end

**Key commands:** remember, recall, connect, enrich, eval, context_boot, consciousness, health_check

**Graph tracking:** Every remember/connect/enrich logs to `self.graph_changes` for session tracking.

---

## 3. Brain Core (`servers/brain.py`)

**Responsibility:** God class (being refactored). Inherits from 11 mixins. Owns the DB connections.
**Lines:** ~1886 (target: <200)
**Depends on:** All mixins, schema.py, dal.py, embedder.py

**Key attributes:**
- `self.conn` — brain.db (knowledge graph)
- `self.logs_conn` — brain_logs.db (operational data)
- `self._meta` — MetaDAL instance
- `self._logs_dal` — LogsDAL instance

---

## 4. Recall Pipeline (`servers/brain_recall.py`)

**Responsibility:** Find relevant nodes given a query. The most performance-critical path.
**Lines:** ~1281
**Sacred system:** Yes — benchmark before any changes.

**Pipeline steps:**
1. Vocabulary expansion (expand query with known vocab terms)
2. Embed query with Arctic v1.5
3. Brute-force cosine similarity against all node embeddings
4. **STEP 3.5 (v6):** Scan enrichment embeddings (Q/A/B/K vectors) — best score per node across primary + enrichments
5. Keyword recall as fallback
6. Unified scoring: blend embedding + keyword scores
7. Graph-augmented recall (1-hop typed neighbors)
8. Hydrate full node data

**Key constants (brain_constants.py):**
- `EMBEDDING_PRIMARY_WEIGHT = 0.90`
- `KEYWORD_FALLBACK_WEIGHT = 0.10`
- `GRAPH_AUGMENT_TOP_N = 5`
- `NEIGHBOR_DAMPEN = 0.6`

**Benchmark:** `tests/eval_runner.py` + `tests/golden_dataset.json` (104 cases)

---

## 5. Remember Pipeline (`servers/brain_remember.py`)

**Responsibility:** Store new nodes with embeddings, auto-connections, and enrichment prompts.
**Lines:** ~1080
**Sacred system:** Yes

**Pipeline:**
1. Validate & set confidence by type
2. Auto-extract keywords if not provided
3. Generate content summary
4. INSERT into nodes table
5. Build TF-IDF vector
6. Embed with Arctic v1.5 → store in node_embeddings
7. Auto-connect to recently accessed nodes
8. Emergent bridging
9. **v6:** Build enrichment prompt (find neighbors, format V5 template)
10. Return node ID + enrichment_prompt for Claude to fill in

**Enrichment flow (v6):**
```
remember() returns enrichment_prompt
  → Claude generates Q/A/B/K
  → Claude calls enrich(node_id, question=..., anchor=..., bridge=..., keywords=...)
  → store_enrichments() embeds each and stores in node_enrichments
```

---

## 6. Data Access Layer (`servers/dal.py`)

**Responsibility:** Database abstraction. Only module that should know table schemas.
**Lines:** ~700+
**Depends on:** sqlite3

**Classes:**
- `LogsDAL` — brain_logs.db: debug_log, access_log, recall_log, miss_log, staged_learnings
- `MetaDAL` — brain_meta table: key-value config
- `GraphDAL` — edges table: typed neighbor queries
- `EnrichmentDAL` — node_enrichments table: V5 multi-vector CRUD
- `TelemetryDAL` — brain_telemetry table (in brain_logs.db): operation logging

---

## 7. Schema (`servers/schema.py`)

**Responsibility:** Single source of truth for all table definitions. Auto-migration on startup.
**Lines:** ~969
**Version:** 16

**Two databases:**
- `brain.db` — TABLES dict: nodes, edges, node_embeddings, node_enrichments, etc.
- `brain_logs.db` — LOG_TABLES dict: debug_log, recall_log, brain_telemetry, etc.

**Migration:** `ensure_schema(conn)` creates missing tables, adds missing columns, creates indexes. No manual migration code elsewhere.

---

## 8. Embedder (`servers/embedder.py`)

**Responsibility:** Load and run Snowflake Arctic v1.5 embedding model.
**Lines:** ~367
**Sacred system:** Yes
**Model:** `model-package/brain_embedding/model/` (768d, ONNX via FastEmbed)

**Key functions:**
- `load_model()` — loads ONNX model from local path
- `embed(text)` → bytes (768-float blob)
- `cosine_similarity(vec_a, blob_b)` → float
- `is_ready()` → bool

**Known issue:** Python crashes occasionally during Arctic operations (Apple Silicon + ONNX). Being replaced — see PLAN.md.

---

## 9. Hooks (`hooks/scripts/`)

**34 files** (17 .sh + 17 .py). Each .sh is a thin wrapper calling its .py.

**High-fire hooks (24h data):**
| Hook | Fires | What it does |
|---|---|---|
| post_response_track | 184 | Captures session activity after each response |
| recall | 102 | Semantic recall before each Claude response |
| session_end | 9 | Session synthesis + save |
| worktree_context | 10 | Git context for worktrees |

**Key hooks:**
- `boot-brain.sh` — SessionStart: boots daemon, prints context
- `pre-response-recall.sh` — UserPromptSubmit: recall before response
- `pre-edit-suggest.sh` — PreToolUse(Edit|Write): surface rules before file edits
- `pre-compact-save.sh` — PreCompact: save brain before context loss
- `post-compact-reboot.sh` — PostCompact: re-boot after compaction

---

## 10. Constants (`servers/brain_constants.py`)

**Responsibility:** All tuning parameters in one place. Avoids circular imports.
**Lines:** ~300

**Key sections:**
- Decay half-lives by node type
- Confidence defaults by type
- Embedding/keyword weights
- Intent detection patterns
- Graph traversal params
- **V5 enrichment params** (ENRICHMENT_NEIGHBOR_COUNT, ENRICHMENT_PROMPT_TEMPLATE)
- Edge type definitions

---

## 11. Voice & Surface (`servers/brain_voice.py`, `servers/brain_surface.py`)

**Responsibility:** Format brain output for hooks and operator channel.

- `brain_voice.py` (1102 lines): Operator channel, `wrap_for_hook()`, `render_operator_prompt()`
- `brain_surface.py` (946 lines): Boot context, `format_boot_context()`

**Critical rule:** All operator-visible content goes through `wrap_for_hook()` into `additionalContext`. Never use `systemMessage` (dead channel).

---

## 12. Testing (`tests/`)

**Key files:**
| File | What | Cases |
|---|---|---|
| `golden_dataset.json` | Recall benchmark corpus | 104 cases, 15 categories |
| `eval_runner.py` | NDCG/MRR/precision scorer | Runs against any brain DB |
| `test_e2e_enrichment.py` | V5 pipeline tests | 38 tests |
| `test_core.py` | Unit tests | Core brain operations |
| `benchmark_multivec_encoding.py` | V5 variant comparison | V1/V2/V3 multi-vector |

**Baseline (before V5 enrichment):** NDCG 0.204, 34/104 passed
**Target (after V5 enrichment):** NDCG >0.60, >80/104 passed

---

## Database Tables (brain.db)

| Table | Purpose | Key columns |
|---|---|---|
| nodes | Knowledge graph nodes | id, type, title, content, keywords, confidence, locked |
| edges | Relationships between nodes | source_id, target_id, relation, weight |
| node_embeddings | Primary Arctic v1.5 vectors | node_id, embedding (768d blob) |
| **node_enrichments** | V5 multi-vector (Q/A/B/K) | node_id, vector_type, text, embedding |
| node_metadata | Rich encoding sidecar | node_id, reasoning, alternatives, correction_of |
| node_vectors | TF-IDF vectors | node_id, term, tf, tfidf |
| brain_meta | Key-value config | key, value |

## Database Tables (brain_logs.db)

| Table | Purpose |
|---|---|
| debug_log | Hook events, errors |
| recall_log | Recall queries + results |
| miss_log | Recall misses |
| **brain_telemetry** | Operation timing, success/failure |
| staged_learnings | Pending knowledge promotions |
