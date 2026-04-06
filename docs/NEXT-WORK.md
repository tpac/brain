# Next Work — After Session 2026-04-05 (scales + cleanup)

## 30 commits this session

### Architecture: scales/ directory
- `scales/dispatch.py` — shared TCP dispatch + dispatch factory
- `scales/runner.py` — background thread lifecycle + generic LLM tool loop
- `scales/s1/recall.py` + `recall_contract.py` — S1R chain (judge, expand, traces)
- `scales/s1/encode.py` + `encode_contract.py` — S1E chain (gather, prompt, LLM loop)
- Old files → backward-compat re-export shims

### Database cleanup: 23 tables dropped
- brain_logs.db: 23 → 9 tables
- brain.db: 32 → 23 tables

### Code cleanup: ~2,500+ lines removed
- brain_precision.py deleted (763 lines)
- dal_message_stream.py deleted (332 lines)
- 18 remember_*/create_* wrapper methods + 6 MCP tools removed
- TelemetryDAL, dead DAL methods, stale table references cleaned

### New capabilities
- `brain_batch` — unified multi-op tool (remember + revise + connect in one call)
- `connect_batch` — multiple edges in one call
- MCP recall enriched (corrections, graph expansion, metadata)
- Dashboard reads from traces, session selector, encoding tab from S1E traces
- Dashboard moved to `dashboard/` (not part of brain)

### Key design decisions
- O/K/Δ is the complete formula — outcome is the next cycle's O, not a fourth element
- LLM reasoning IS the knowledge management system — architecture gives it the right data at the right time with the right questions
- Each scale optimizes its own interactions through traces — only the master prompt is human-curated
- Impact measurement is reasoning about traces, not metrics

---

## Priority for Next Session

### 1. ENCODER FIX (CRITICAL — quality has degraded)
The S1 encoding prompt and data assembly need serious attention:

**Data assembly (scales/s1/encode.py `_build_user_content`):**
- Show 15 messages (was 10, may have regressed to 5) — more context, same 5-stop trigger
- Node catalog must show full rich nodes with edges, corrections (currently truncated)
- Journal should be chronological, not reverse-ordered
- Stop counter values (Run #5, #10) need context — label as run sequence not raw counter

**Prompt (interactions table `encoding_agent` template):**
- Document watching/skipped/encoded journal format
- Fix ID format inconsistency between examples and actual data
- Clean up "BRAIN SURFACED" noise in timeline
- Session Context is 800 chars of pipe-separated noise — needs structure

### 2. Scale Architecture (revised)
Scales simplified from 5 to 4:
- **S0** — raw exchange (hooks observe, not control)
- **S1** — turn encoder. Fires every 5 stops, sees 10 turns (20 messages). Encodes knowledge. Already built.
- **S2** — graph maintenance between sessions. Community detection, consolidation, correction chains, dedup, confidence recalibration. **Also: tool use learning** — reads S0 tool traces to capture operational intelligence (workarounds, tool patterns, failure modes). Encode at S2, surface at PreToolUse. Partially built (idle hook has pieces).
- **S3** — abstract reasoning. Curiosity, uncertainty resolution, cross-project patterns.
- **S4** — external research, web search, growth.

S1 with wider observation window replaces the "session encoder" concept. No separate scale needed for journey arcs — S1 sees 10 turns.

### 3. Shared Infrastructure (MEDIUM)
- Move `_expand_and_enrich` from scales/s1/recall.py to shared (MCP recall uses it too)
- Standardize scale interface: observe → select → integrate → trace

### 4. Remaining Cleanup (LOW)
- brain_surface.py still has miss_log/staged_learnings SQL (silently fails)
- Fatigue storage: JSON blob instead of per-row (52K→87 rows in session_state)
- VACUUM brain_logs.db (tables dropped but space not reclaimed)
- Shim files cleanup
