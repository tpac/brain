# Session 2026-03-30 Handoff — Anchor to Next Anchor

## What This Session Built

Daemon stability, recall quality, encoding in production, short IDs, DAL migration of brain_recall.py. 8 commits.

### Don't Redo What's Done

1. **Daemon CPU spiral fixed** — Three root causes: watchdog thread leak (orphaned threads on timeout), SQLite single-connection deadlock (concurrent threads on shared conn), unbounded encoding agent threads. Fix: inline dispatch (no watchdog), pool=1 (serial), encoding runs inline in Stop hook with 60s timeout.

2. **Recall quality doubled** — 34%→65% top-3, 55%→79% top-8. Two changes: proportional title-match boost (TITLE_MATCH_BOOST=0.3), vocab nodes separated from primary results (surfaced as connectors with IDs, no forward traversal in graph walk). Pattern category: 33%→88%.

3. **Encoding agent works in production** — First time ever. Runs inline in Stop hook every 5th stop. 60s hook timeout, 55s thin client socket timeout. Profiled: brain ops ~300ms, rest is Sonnet API latency (3-10s per round, 2-3 rounds typical). Created real nodes: corrections, decisions, tensions.

4. **Short IDs** — 32-char UUIDs → 8-char hex. Zero collisions verified. All FKs migrated (85K rows brain.db, 88K brain_logs.db). `_generate_id()` returns `uuid4().hex[:8]`. `_resolve_id()` prefix matcher in daemon_dispatch.py for backward compat.

5. **brain_recall.py DAL migration** — 29 raw SQL → 4. Created: `EmbeddingDAL.get_all_with_context()`, `NodeDAL.get_metadata()`. Fixed: `LogsDAL.log_access()` had ghost columns (query, context) not in schema. `NodeDAL.mark_accessed()` was missing `updated_at`. Removed shadowed import that silently broke all recall. Pattern: search queries return IDs, hydration through `NodeDAL.get_node()`.

6. **Dashboard improvements** — Type filters on Surface/Encoding tabs, encoding countdown (1/5, 2/5..., ENCODING RAN), encoding_source on revised nodes, encoding agent profiling in log, newest-first sorting fixed.

7. **Launchd integration** — `com.brain.daemon.plist` with KeepAlive. Daemon clears `__pycache__` on start so launchd restarts always use latest code.

## Architecture After This Session

```
Pool=1 (serial, SQLite single-conn constraint)
  → All requests process one at a time
  → Encoding agent runs inline in Stop hook (not background thread)
  → No deadlocks, no thread leaks

DAL coverage:
  brain_recall.py:  29 → 4 raw SQL (DONE)
  brain_surface.py: 40 raw SQL (TODO)
  brain_remember.py: 42 raw SQL (TODO)
  brain_evolution.py: 102 raw SQL (lower priority)
  brain_engineering.py: 60 raw SQL (lower priority)

Cache layer pattern established:
  Search queries → return IDs only
  Hydration → NodeDAL.get_node() (single cache entry point)
  Embedding scan → EmbeddingDAL.get_all_with_context()
```

## Next Session: DAL Migration (surface + remember)

### brain_surface.py (40 raw SQL calls)

Quick wins with existing DAL methods:
- `NodeDAL.exists()` — lines 573, 603
- `NodeDAL.count_locked()` — line 343
- `NodeDAL.archive()` — line 608
- `NodeDAL.set_critical()` — several places

Missing DAL methods needed:
- `NodeDAL.get_critical_nodes(limit)` — used twice (lines 252, 802)
- `NodeDAL.get_locked_nodes(project, limit)` — lines 278-306
- `NodeDAL.get_recent_unvisited(limit)` — line 316
- `NodeDAL.get_typed_nodes(types, limit)` — lines 413, 873, 916, 932

### brain_remember.py (42 raw SQL calls)

Biggest issue: **TF-IDF duplication**. Lines 108-118 and 283-293 are IDENTICAL code. Both should use `TfIdfDAL.store_tf_vector()` which already exists.

Quick wins:
- `TfIdfDAL.delete_for_node()` — line 104
- `TfIdfDAL.store_tf_vector()` — lines 108, 283 (dedup)
- `EmbeddingDAL.store_embedding()` — lines 318, 436 (dedup)
- `EmbeddingDAL.get_embedding()` — line 475
- `NodeDAL.set_critical()` — line 803
- `NodeDAL.get_all_for_reindex()` — line 274

Missing:
- `NodeDAL.get_personal_nodes()` — lines 1137-1142 (used twice identically)

### Contract violations to fix
Several queries miss newer columns: `revised_at`, `encoding_source`, `content_summary`. Using `NodeDAL.get_node()` (returns all columns) fixes these automatically.

## Files Changed (8 commits)

| File | What |
|------|------|
| servers/daemon_server.py | Remove watchdog threads, HTTP MCP code, SO_REUSEPORT, pycache clearing |
| servers/daemon_config.py | Pool=1, constants |
| servers/daemon_dispatch.py | Thread count in ping, `_resolve_id()` prefix matcher |
| servers/daemon_hooks.py | Encoding inline, countdown output, profiling |
| servers/encoding_agent.py | Step profiling |
| servers/brain_recall.py | DAL migration (29→4), title boost, vocab separation |
| servers/brain_constants.py | TITLE_MATCH_BOOST |
| servers/brain_mcp.py | Vocab context in output |
| servers/brain_voice.py | (unchanged, vocab surfacing via MCP) |
| servers/brain_dashboard_standalone.py | Filters, sorting, encoding_source |
| servers/pipeline_contract.py | ID length 16→8, `format_node_header` |
| servers/embedder.py | numpy cosine similarity |
| servers/dal.py | `get_all_with_context()`, `get_metadata()`, fixed `log_access()`, `mark_accessed()` |
| servers/brain.py | `_generate_id()` → 8-char |
| hooks/scripts/post_response_track.py | 55s timeout |
| hooks/scripts/pre_response_recall.py | Vocab context append |
| .claude/settings.json | Stop hook 60s timeout |

## Numbers

- Daemon: 820% CPU → 0% idle
- Recall: 34%/55% → 65%/79% (top-3/top-8)
- Pattern category: 33% → 88%
- Recall latency: 326ms → 150ms (numpy cosine)
- Encoding: never worked → 6 actions/33s, 4 actions/18s
- IDs: 32 chars → 8 chars
- brain_recall.py raw SQL: 29 → 4
- HTTP MCP dead code: -350 lines
- Net lines: -400+ removed
