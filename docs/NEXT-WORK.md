# Next Work — After Session 2026-04-06 (cleanup + test overhaul)

## This Session: 20 commits

### API Cleanup
- Removed `record_divergence` + `learn_vocabulary` from MCP, dispatch, scales, SKILL.md
- Removed `produce_vocabulary_gap` signal producer
- Consolidated 7 inline SQL prefix queries into DAL methods (`resolve_id`, `get_title`)
- Fixed `get_interaction` version param (Brain.py override now accepts it)

### ONNX Runtime CPU Fix
- Identified Python 3.9 locks us to onnxruntime 1.19.2 (thread spin-wait bug)
- Monkey-patched fastembed to inject `session.intra_op.allow_spinning=0`
- threads=2 limit. Load time 1230ms→238ms, 0% idle CPU (was 200%)

### Test Suite Overhaul
- **Before:** 754 tests, 46 failing, 2 won't import, ~7 min
- **After:** 531 tests, 0 failing, 33s
- Deleted 11 test files, cut ~100 never-fail tests from test_core.py
- Added 67 new purposeful tests across 5 files:
  - `test_dal_nodes.py` (18) — DAL methods + signal queue
  - `test_format_node.py` (13) — contract.format_node()
  - `test_s1_data_assembly.py` (22) — S1 encode data assembly
  - `test_decode_transitions.py` (8) — decode pipeline wiring
  - `test_okd_cycle.py` (6) — fractal O/K/Δ loop property
- Brain.__init__(skip_embedder=True) for tests that don't need semantic search

### Database Cleanup
- brain_logs.db: 104MB → 6.1MB (VACUUM + 21 table drops + 20K fatigue row cleanup)
- brain.db: 66MB → 63MB (VACUUM + 7 table drops)
- Total tables: brain.db 31→24, brain_logs.db 23→9
- Maintenance mode mechanism (lock file prevents daemon auto-restart during DB ops)

### Schema Cleanup
- schema.py: removed 21 dead table definitions, added 3 missing (embedding_fidelity, node_communities, hook_errors)
- Fresh brain creates exactly what production has (no dead tables)
- `scripts/create_fresh_brain.py` — standalone seed script

### Code Cleanup
- Fatigue moved from per-node DB rows to SessionContext JSON blob (52K→1 row/session)
- Deleted render_boot v1 (325 lines dead code calling deleted methods)
- Deleted 3 shim files (encoding_agent.py, judge_contract.py, encoding_contract.py)
- Cleaned brain_surface.py: removed miss_log, staged_learnings refs + 4 dead functions
- Fixed node catalog regex to support typed-prefix IDs (con_xxxx)
- Fixed 'database is locked' in S1 recall traces (TraceDAL.append_batch)

---

## CLAUDE.md Stale References (fix next session)

CLAUDE.md has references to deleted code:
- `store_exchange() → message_stream` — message_stream deleted
- `dal_message_stream.py` — file deleted
- `encoding_agent.py`, `judge_contract.py`, `encoding_contract.py` — shim files deleted
- `recall_log` — table dropped, writes stopped
- Stop hook description mentions message_stream — should reference S0 traces
- Encoding section references `encoding_agent.py:_build_system_prompt()` — now `scales/s1/encode.py`

## Immediate Cleanup (before S2)

### 1. CLAUDE.md Update
Fix all stale references listed above. This is the developer guide — stale refs mislead future Anchors.

### 2. Duplicate Node Clusters
Signal queue flagged:
- 21 "Compaction boundary" duplicate nodes
- 16 "Session handoff" duplicate nodes
The encoder creates these every compaction/session-end. S2 should detect and consolidate, but we should understand why the encoder creates duplicates first.

### 3. Questionable brain.db Tables
4 tables with data but unclear active usage:
- `node_communities` (910 rows) — written by redistribution, read by dreams
- `bridge_proposals` (38 rows) — bridge edge candidates, may be stale
- `correction_traces` (17 rows) — correction lineage tracking
- `emotion_calibration` (2 rows) — emotion scoring calibration
Audit during S2 design — they may be inputs for graph maintenance.

### 4. Evolution API Dead Surface
`create_tension`, `create_hypothesis`, `create_aspiration`, `create_pattern`, `get_active_evolutions`, `confirm_evolution`, `dismiss_evolution` — methods exist in brain_evolution.py + dispatch handler, but no MCP tool, no hook, no encoder uses them. Either wire into S2 or delete.

### 5. brain_dreams.py / brain_consciousness.py Audit
These modules have idle-hook logic (dream, consolidate, heal) that partially overlaps with S2's scope. Before building S2, audit what exists and decide: keep as S2 primitives, refactor into S2, or delete.

---

## S2: Graph Maintenance Scale

### What S2 Does
Between-session graph operations that need the full picture. S1 encodes per-turn; S2 maintains the graph across sessions.

### S2 Capabilities (from architecture doc + brain recall)
1. **Dedup/Consolidation** — detect and merge duplicate nodes (like the 21 compaction boundary nodes)
2. **Correction Chain Resolution** — when node B corrects A, adjust confidence, potentially archive A
3. **Confidence Recalibration** — nodes that are recalled but never validated decay; frequently validated nodes grow
4. **Community Detection** — identify clusters, label them, surface to boot context
5. **Tool Use Learning** — read S0 tool traces to capture operational intelligence (workarounds, tool patterns, failure modes). Encode at S2, surface at PreToolUse hook.
6. **Evolution Management** — tensions, hypotheses, patterns lifecycle. The evolution API exists but needs consumers.

### S2 Architecture (mirrors S1)
```
scales/s2/
  encode.py          — S2 encode: graph maintenance agent
  encode_contract.py — S2 config, prompt, data assembly
```
Shared infrastructure already ready:
- `scales/dispatch.py` — TCP dispatch factory
- `scales/runner.py` — background thread lifecycle + LLM tool loop
- `contract.py:format_node()` — standard node format for all LLM consumers

### S2 Design Questions
- **Trigger:** Session end? Scheduled task? Signal-driven (when dupe count exceeds threshold)?
- **LLM or code?** Community detection is algorithmic (leidenalg). Dedup judgment needs LLM. Mix?
- **Review model:** S2 operates without Tom present. Stage changes for review via signal queue? Or auto-commit with confidence thresholds?
- **Scope per run:** Full graph scan or targeted (only nodes created since last S2 run)?

### Shared Infrastructure Needed (for S2 and beyond)
1. `_expand_and_enrich` — currently in scales/s1/recall.py, needed by MCP recall too. Move to shared.
2. Standardized scale interface: `detect() → select() → integrate() → commit() → trace()`
3. TraceDAL.append_batch already done (used by S1R, available for S2)
4. SessionContext.fatigue pattern — model for other session-scoped state S2 might need

---

## S3/S4: Future Scales (not yet designed)

### S3: Reasoning (periodic)
Abstract patterns, curiosity, uncertainty resolution. Reads S2 traces.
- Open questions accumulating → research needed
- Cross-project patterns → abstract insights
- **Status:** NOT BUILT, architecture doc has skeleton

### S4: Growth (weekly)
External research, web search. Reads S3 traces.
- Stale decisions → verify against current state of the world
- Research questions from S3 → web search + synthesis
- **Status:** NOT BUILT

---

## Infrastructure Notes

### Python 3.9 Ceiling
onnxruntime 1.19.2 is the last version supporting Python 3.9. Upgrading Python to 3.11+ would unlock onnxruntime 1.24.4 with native thread spinning fixes, memory improvements, and 5 major versions of bug fixes. The monkey-patch works but is fragile.

### Maintenance Mode
`touch /tmp/brain-maintenance-{uid}.lock` prevents daemon auto-restart. Use for VACUUM, schema changes, bulk deletes. Remove lock when done.

### Test Architecture
Tests organized by what they catch:
- **Contract tests** — layer sync, trace writes, pipeline shapes
- **Component tests** — DAL, format_node, scoring, signal queue, S1 data assembly
- **Transition tests** — wiring between pipeline stages
- **Cycle tests** — O/K/Δ loop property (Δ becomes next O)
- **Integration tests** — real data, full pipeline
New tests should follow this taxonomy.
