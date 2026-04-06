# Next Work — After Session 2026-04-05 (scales + cleanup)

## 27 commits this session

### Architecture: scales/ directory
- `scales/dispatch.py` — shared TCP dispatch + dispatch factory
- `scales/runner.py` — background thread lifecycle + generic LLM tool loop
- `scales/s1/recall.py` + `recall_contract.py` — S1R chain (judge, expand, traces)
- `scales/s1/encode.py` + `encode_contract.py` — S1E chain (gather, prompt, LLM loop)
- Old files → backward-compat re-export shims

### Database cleanup: 23 tables dropped
- brain_logs.db: 23 → 9 tables (recall_log, message_stream, access_log, suggest_log, health_log, brain_telemetry, curiosity_log, conflict_log, miss_log, eval_snapshots, tuning_log, pending_consolidation, staged_learnings, recall_gaps)
- brain.db: 32 → 23 tables (projects, project_maps, reasoning_chains, reasoning_steps, session_activity, summaries, prune_archive, version_history, suggest_metrics)

### Code cleanup: ~2,500+ lines removed
- brain_precision.py — entire module deleted (763 lines)
- dal_message_stream.py — entire module deleted (332 lines)
- 18 remember_*/create_* wrapper methods + 6 MCP tools removed
- TelemetryDAL class removed (~100 lines)
- access_log, recall_log, message_stream write paths removed
- get_engineering_context gutted (just nodes, no special structure)

### New capabilities
- MCP recall enriched (corrections, graph expansion, metadata)
- `brain_batch` — unified multi-op tool (remember + revise + connect)
- `connect_batch` — multiple edges in one call
- Dashboard reads from traces (not deprecated tables)
- Dashboard session selector + encoding tab from S1E traces

---

## Priority for Next Session

### 1. ENCODER REGRESSION (CRITICAL — quality has degraded)
The S1 encoding prompt needs serious attention. Tom identified these issues:

**Data assembly problems (scales/s1/encode.py `_build_user_content`):**
- Node catalog shows truncated content — used to be full rich nodes with edges, corrections
- Journal is reverse-ordered (Run #20 before Run #5)
- "Run #5", "Run #10" are stop counter values, not sequential — confusing
- max_messages may have regressed (should show 10 turns for context, fires every 5)

**Prompt problems (interactions table `encoding_agent` template):**
- Watching/Skipped/Encoded journal format not documented in prompt
- Session Context is 800 chars of pipe-separated noise — hard to parse
- ID formats inconsistent between prompt examples and actual data
- "BRAIN SURFACED" in timeline shows noise like "judge selected but no IDs parsed"

**Action:** Dedicated encoder prompt session. Read the actual prompt Sonnet receives (from /tmp/brain-encoding-prompt-*.json), compare against v3.2 intent, fix both the data assembly and the prompt text.

### 2. S2 Session Encoder (HIGH — infrastructure ready)
Everything needed: scales/ directory, runner.py, dispatch.py, trace_contract has S2 ref_types, brain_batch tool. Copy S1E pattern.

### 3. Outcome Traces (HIGH — the learning signal)
O/K/Δ traced but not outcomes. Without outcomes, no scale learns whether its Δ served the target function.

### 4. _expand_and_enrich shared (MEDIUM)
Currently in scales/s1/recall.py but MCP recall also uses it. Should be shared infrastructure — any consumer that surfaces nodes wants correction enrichment + graph expansion.

### 5. Remaining cleanup (LOW)
- brain_surface.py still references miss_log, staged_learnings (silently fails)
- brain_engineering.py dead methods (track_session_event, assess_session_health callers)
- Fatigue storage model (JSON blob instead of per-row in session_state — 52K→87 rows)
- VACUUM brain_logs.db to reclaim space (tables dropped but pages not freed)
- Shim files cleanup (encoding_agent.py, judge_contract.py, encoding_contract.py)
