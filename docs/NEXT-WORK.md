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

### 6. Sleep Cycle Scheduling (LOW — unchanged from previous session)
Redistribution, community detection, consolidation built but not scheduled.

### 7. Encoding Philosophy Shift (LOW — unchanged from previous session)
Encode journeys not just endpoints. Session context partially addresses this.

### 8. Non-critical Silent Errors (LOW)
136 silent `except: pass` in metrics.py, brain_engineering.py, migrations, CLI. Not urgent.
