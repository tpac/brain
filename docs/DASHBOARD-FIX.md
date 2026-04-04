# Dashboard Fix — IMPLEMENTED 2026-04-03

## Status: DONE

All items in this doc were implemented in the 2026-04-03 cleanup session (commits 027df0b through d0264b3).

## What Was Done

### 1. ✅ Dashboard reads from recall_log (single source of truth)
- Surface feed reads from `recall_log` in `brain_logs.db`, not `hook_log` in `brain_dashboard.db`
- Every recall (hook, MCP, internal) logged via `brain.recall()` with `source` column
- `/api/recalls` endpoint replaces `/api/hook-log`

### 2. ✅ Judge output stored in message_stream
- `judge_output` column added to `message_stream`
- Stop hook reads judge result from tmp file, stores in message_stream
- `recalled_node_ids` stores judge-selected IDs only (was: all 25 candidates)
- `recalled_raw` kept for debugging

### 3. ✅ brain_dashboard.db dependency removed
- `log_hook_output()` is now a no-op (deprecated)
- Dashboard reads from brain's own tables
- Dashboard DB status replaced with Haiku Judge status in Status tab

### 4. ✅ Dashboard shows exact additionalContext
- Short details = exact additionalContext (judge output)
- Full Details = exact judge prompt (from tmp file)
- Judge data flows async via `/tmp/brain-judge-result-{id}.json`
- Encoding tab shows actual encoder prompt (from `/tmp/brain-encoding-prompt-{counter}.json`)

## Architecture (current)

```
Brain (daemon + hooks) → does the work → writes to DBs + tmp files
Dashboard → reads from those same DBs + files → displays to operator
```

Dashboard is a passive observer. Never writes to brain data.
