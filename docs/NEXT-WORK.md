# Next Work — Open Items after Session 2026-04-03

## Done This Session (2026-04-03 cleanup)

### ✅ Dashboard v2
- Reads from `recall_log` + judge tmp files (single source of truth)
- `brain_dashboard.db` writes stopped (`log_hook_output` is no-op)
- Surface feed: source badges (HOOK/ANCHOR/INTERNAL), judge output display, Full Details
- Encoding tab: run cards with actual Sonnet prompt from tmp file
- Errors tab: source filter, time filter fixed (ISO format), badge for new errors only
- Status tab: Judge health replaces Dashboard DB status
- Encoding status indicator in stats bar

### ✅ Judge moved to daemon
- Haiku call runs inside `daemon_hooks.py hook_recall()`, not hook subprocess
- Eliminates 723ms cold import + hook timeout kills (was 85% failure rate)
- Thin client (`pre_response_recall.py`) reduced from 247 to 68 lines

### ✅ Encoder v3.2
- Node catalog: deduplicated rich nodes at top (full metadata, all KV, edges)
- Timeline: ID references, not repeated content
- Both roles get 2500 char display limit (shared learnings, not just Tom's words)
- Session context: additive journey with `|` separator (800 chars)
- `revise_batch()` added for encoder efficiency
- Encoder prompt updated with examples for both batch tools

### ✅ Decode-encode contamination fix
- `message_stream.recalled_node_ids`: judge-selected only (was all 25)
- `message_stream.judge_output`: exact additionalContext for encoder
- Encoder reads `judge_output` (curated) not `recalled_raw` (noisy)

### ✅ Revise replaces content
- Content is REPLACED on revise, not appended (brain was remembering wrong things)
- Old content saved to `revision_history` in metadata KV (last 5 versions)
- All metadata KV fields revisable, only `id`, `created_at`, `locked` immutable

### ✅ encoding_source convention
- Format: `category:process` (anchor, encoder:sonnet, idle:dreams, hook:compaction)
- Only `anchor` can create locked nodes
- Added to contract, schema, all callers

### ✅ Infrastructure
- `IsolatedBrain` test harness (tests/isolated_brain.py)
- 13 critical silent errors made loud
- PRECISION module deprecated (judge+encoder coupling replaces it)
- Truncation contracts centralized in PIPELINE
- Raw SQL moved to DAL (NodeDAL.purge, GraphDAL.count_total)
- Distiller dead code deleted (~90 lines)
- Daemon restart fixed (subprocess.Popen + os._exit, was silently failing)
- Boot prompt rewritten (recognition-first, not instruction-first)
- Plugin hooks.json synced with project settings (timeouts fixed)

---

## Still Open

### 1. MCP Skill Tool (HIGH — blocks Anchor's brain access)
**Problem:** Plugin disabled in user settings → MCP server not running → brain skill tools don't work.
**Solution ready:** Enable plugin, remove duplicate project hooks, use `--plugin-dir` for dev.
**Test:** Fresh session with `claude --plugin-dir /Users/tpac/brain`, run `/reload-plugins`.

### 2. Daemon Stability (HIGH — affects all sessions)
**Problem:** 200% CPU sustained for 30+ minutes. Diagnosed: ONNX Runtime / FastEmbed memory leak. RSS grows to 1.1GB+ over hours → swap thrashing.
**Proposed fix:** RSS watchdog (auto-restart at 1.5GB threshold). Not yet built.
**Alternative:** Replace FastEmbed with non-leaking embedding library.

### 3. Signal Producers (MEDIUM)
**Problem:** Stale signals accumulate. Queue cleared this session but producers still fire for conditions that no longer matter.
**Action:** Review each producer, disable outdated ones.

### 4. Encoder Revise Quality (MEDIUM — dedicated session)
**Problem:** 7% revision rate. Encoder overwhelmingly creates new nodes instead of revising existing ones. Now has full node catalog + `revise_batch()` — tools are ready, behavior needs tuning.
**Action:** Dedicated debugging session with eval comparison.

### 5. Daemon Reload MCP Tool (LOW)
**Problem:** WIP handler in daemon_dispatch.py, not wired to command table or MCP schema.
**Action:** Complete and test. Low priority since `hooks/scripts/restart-daemon.sh` works.

### 6. Fractal Integration Architecture (HIGH — S0/S1 infrastructure COMPLETE)
Architecture doc: `docs/ARCHITECTURE-FRACTAL.md`. Core: `integrate(O, K) → Δ` at every scale.

**BUILT this session:**
- Trace contract (`trace_contract.py`): S0-S4 scales, event types, ref types, validation at write time
- TraceDAL: 8 methods (append, get_chain, get_recent, get_chains, get_by_ref_type, get_outcomes, count_by, get_session_turns)
- 104 tests across 6 test suites (unit, contract sync, integration, S1 cycle, session context, interactions)
- SessionContext: per-request session identity, persisted in DB, survives daemon restarts
- Session_id flows from Claude Code hook args through thin clients to daemon
- S0 traces: K (user_message) + delta (assistant_message + tool_result), summary/metadata split
- S1 traces: O (candidates) + K (judge_selected) + delta (additionalContext + encoding_run)
- Interactions as learnable boundaries: 6 seeded (judge + encoding_agent with prompts, 4 code-only with config)
- Judge reads prompt from interactions table at runtime (learnable)
- Encoding agent reads prompt from interactions table at runtime (learnable)
- Trace events link to interaction_id for version comparison
- S0/S1 migration: encoding agent + judge context read from traces, message_stream stripped to escalation
- MCP tools: filter_nodes, query_logs, query_traces, query_outcomes, count_traces, list_interactions, get_interaction
- daemon_hooks.py refactored: 261→50 line main function + 5 clean helpers
- Single-writer rule: encoding agent writes via daemon TCP, not direct DB
- Dashboard: Traces tab, timezone fix, Status+Health merged

**REMAINING for next session:**
- Wire code-only boundaries to read config from interactions (voice, boot, pre_edit, assembler)
- Pass session_id in remaining thin clients (bash, host_check, session_end, compact, idle)
- Remove deprecated brain.session_id property
- Rename modules to scale positions (encoding_agent.py → s1_encoder.py)
- Outcome traces (the learning signal — not yet built)
- Verify encoding fires end-to-end with all fixes
- Dashboard recall display fix (broken — separate UI session)
- Encoding agent loads embedder unnecessarily (doubles memory/CPU)

**NOT BUILT (future sessions):**
- Session encoder (Scale 2)
- Sleep graph operations (Scale 3)
- Growth external research (Scale 4)
- Partnership signal tracking
- Encoding gap detection

### 7. Scale 1 Gaps (MEDIUM)
Remaining gaps for Scale 2 readiness:
1. Correction linking partially done (Layer 3.5 shipped, trace backfill pending)
2. No partnership signal tracking
3. No encoding gap detection (Scale 2 responsibility)
4. Tool interactions now captured via PostToolUse trace
5. Session patterns not fed to encoder

### 8. Sleep Cycle Scheduling (LOW — subsumed by Scale 3 design)
Redistribution, community detection, consolidation built but not scheduled.
Now part of the fractal architecture as Scale 3.

### 9. Non-critical Silent Errors (LOW)
136 silent `except: pass` in metrics.py, brain_engineering.py, migrations, CLI. Not urgent.
