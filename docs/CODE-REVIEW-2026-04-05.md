# Code Review — 2026-04-05 (Post Fractal Infrastructure Build)

## What's Clean

### New files (built this session)
- `servers/trace_contract.py` — single purpose, pure data + one validation function. No dependencies.
- `servers/session_context.py` — clean object with clear interface. Save/load/chain ID generation.
- `servers/interaction_seed.py` — straightforward seed with v1 values from current code. Idempotent.

### Tests (6 files, 104 tests)
- Well-structured: one class per concern, clear test names, isolated via IsolatedBrain.
- `test_trace_contract_sync.py` scans production code against contract — catches drift automatically.
- `test_s1_cycle.py` tests the full recall→judge→encode→trace cycle end-to-end.
- `test_interactions_runtime.py` covers seeding, versioning, trace linkage.

### Refactored code
- `daemon_hooks.py:hook_post_response_track` — 6 numbered steps, each a clean function call.
- Helper functions extracted: `_read_recall_data`, `_hebbian_strengthen`, `_daemon_tcp_send`, `_make_encoding_dispatch`, `_run_encoding_agent`.

---

## What Needs Cleanup

### 1. `daemon_hooks.py:hook_recall` — god function (~350 lines)
**Problem:** The judge Haiku call is inlined: .env loading, API client creation, JSON parsing, recently-recalled dedup, graph expansion, correction enrichment, judge result file writing, S1 trace writes. All in one function.

**Fix:** Extract `_run_judge(brain, candidates, user_message, session_context, ...)` as a clean function like `_run_encoding_agent`. The recall hook becomes: recall → write candidates file → run judge → format output → write traces.

### 2. `pipeline_contract.py` — 800+ lines, 3 different boundaries
**Problem:** Contains judge prompt building, candidate formatting, voice output formatting, encoding agent config, truncation limits, and correction enrichment. Three different boundaries (judge, voice, encoding) share one file.

**Fix:** Split into:
- `contracts/judge_contract.py` — build_judge_prompt, format_candidate_for_judge, JUDGE config
- `contracts/voice_contract.py` — format_judge_output, format_node_for_encoder
- `contracts/encoding_contract.py` — ENCODING_AGENT config, timeline limits
- Keep `pipeline_contract.py` as a thin re-export for backward compat during transition

### 3. `brain.py` — scattered session/interaction surface
**Problem:** `get_interaction_config`, `get_interaction_prompt`, `get_interaction`, `get_or_create_session`, deprecated `session_id` property, `reset_session_activity` — 6 methods for 2 concerns (interactions + sessions) mixed with 50+ other methods.

**Fix:** These are already clean individually. The issue is density — brain.py is a mixin-composed class with 1000+ methods. Consider: interaction methods belong on a mixin (`BrainInteractionMixin`), session methods belong on another (`BrainSessionMixin`). Follow existing pattern — `brain_recall.py`, `brain_surface.py` etc. are already mixins.

### 4. `dal.py` — 10 DAL classes, 1000+ lines
**Problem:** LogsDAL, InteractionDAL, TraceDAL, SessionStateDAL, NodeDAL, EmbeddingDAL, TfIdfDAL, Fts5DAL, GraphDAL, EnrichmentDAL, TelemetryDAL — all in one file.

**Fix:** Split by database:
- `dal_brain.py` — NodeDAL, EmbeddingDAL, TfIdfDAL, Fts5DAL, GraphDAL, EnrichmentDAL
- `dal_logs.py` — LogsDAL, TraceDAL, InteractionDAL, SessionStateDAL, TelemetryDAL
- Keep `dal.py` as re-export for backward compat

### 5. Deprecated code still active
- `brain.session_id` property — marked deprecated but still read by recall fallbacks
- `recall_log` writes — marked deprecated but still writing (tmp file path dependency)
- `message_stream` stores 200-char content — escalation system reads it but could read from traces
- `brain_precision.py` — entire module deprecated but not deleted

### 6. `encoding_agent.py` — `brain.session_id` fallback
**Problem:** Line 72: `session_id = brain.session_id  # fallback for legacy callers only`. The comment says legacy but the guard means someone could call `run_encoding` without session_id and silently get the wrong value.

**Fix:** Make session_id required (no default). If a caller doesn't have it, that's a bug to surface, not silently absorb.

### 7. Tmp file inter-process communication
**Problem:** 5 tmp file paths hardcoded in daemon_hooks.py for passing data between hooks (candidates, judge selections, judge results, current stop). These use session_id in paths, creating stale files on session change.

**Fix:** Not urgent — this is the standard hook communication pattern in Claude Code. But could migrate to session_state table or a shared config for cleanliness. The current-stop tmp file is the weakest — SessionContext in DB is better.

---

## Priority for Next Session

1. **Extract judge call** from hook_recall (biggest win — makes the recall hook as clean as the stop hook)
2. **Split pipeline_contract.py** (reduces cognitive load for anyone touching judge vs voice vs encoding)
3. **Delete brain_precision.py** (dead code, no references)
4. **Make session_id required** in run_encoding (surface bugs instead of hiding them)

Everything else is nice-to-have — the system works, tests pass, the architecture is sound.
