# Next Work — Open Items after Session 2026-04-05 (scales/ restructure)

## Done This Session

### Scales Architecture (fractal code structure)
- `scales/dispatch.py` — shared TCP dispatch + dispatch factory for all scale agents
- `scales/runner.py` — generic background thread lifecycle + LLM tool loop
- `scales/s1/recall.py` — S1R chain (judge, graph expand, correction enrich, traces)
- `scales/s1/recall_contract.py` — S1R config, judge prompt building, output formatting
- `scales/s1/encode.py` — S1E chain (gather, prompt, LLM loop, journal, context)
- `scales/s1/encode_contract.py` — S1E config, node formatting, catalog building
- Old files (`encoding_agent.py`, `judge_contract.py`, `encoding_contract.py`) → backward-compat shims
- daemon_hooks.py now S0-only (hooks + scale gates)

### Code Cleanup
- Extracted judge call from hook_recall god function (400→226 lines)
- Split pipeline_contract.py into per-boundary contracts
- Made session_id required in encoding agent (no silent fallback)
- Enriched MCP recall with corrections + graph expansion + metadata
- Removed recall_log INSERT writes (traces are single source of truth)
- Replaced recall_log_id with session-scoped recall_ref
- Removed message_stream fallback from encoding agent

---

## Still Open

### 1. S2 Session Encoder (HIGH — next major feature)
**Status:** Infrastructure ready. S2 copies S1 pattern.
**What exists:** scales/ directory, runner.py, dispatch.py, trace_contract has S2 ref_types
**What to build:**
- `scales/s2/encode.py` — session encoder (copies s1/encode.py pattern)
- `scales/s2/encode_contract.py` — S2 config
- Interaction seed: 'session_encoder' in interactions table
- SessionContext.s2_chain() method
- Encoding gate: counter % 15 in hook_post_response_track

### 2. Outcome Traces (HIGH — the learning signal)
**Problem:** Traces capture O/K/Δ but not outcomes. Without outcomes, no scale can learn whether its output served the target function.
**Action:** Add outcome events to trace system. Correction detection, future-recall tracking.

### 3. Dashboard Migration to Traces (MEDIUM)
**Problem:** Dashboard reads from recall_log (deprecated, no new writes) and tmp files.
**Action:** Migrate recall display to read from trace_events. Dashboard should observe the same data path as all other consumers.

### 4. Wire Code-Only Interactions to Config (MEDIUM)
**Problem:** 4 interactions (voice_surface, boot, pre_edit, signal_assembler) have config in DB but code reads hardcoded values.
**Action:** Wire remaining interactions to read from interactions table.

### 5. Daemon Stability (MEDIUM)
**Problem:** ONNX/FastEmbed memory leak. RSS grows to 1.1GB+.
**Proposed fix:** RSS watchdog (auto-restart at threshold).

### 6. brain_precision.py Cleanup (LOW)
**Problem:** 763-line module, only `get_precision_summary()` is called (by boot + self-reflection).
**Action:** Extract that one method into LogsDAL, delete the rest.

### 7. Signal Producers (LOW — moves to S2)
**Problem:** Signal producers (encoding gap, vocabulary gap, system health) run in S1 recall hook. They're S2 concerns.
**Action:** Move to S2 session encoder when built.

### 8. Shim Cleanup (LOW — after S2 ships)
**Action:** Remove backward-compat re-export shims once all imports updated:
- `encoding_agent.py` → `scales/s1/encode.py`
- `encoding_contract.py` → `scales/s1/encode_contract.py`
- `judge_contract.py` → `scales/s1/recall_contract.py`
