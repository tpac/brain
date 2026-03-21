# Refactoring Targets

Last updated: 2026-03-21 (post v5.3.1 daemon consolidation)

## How to use this file

Start a new session. Say "open REFACTORING.md and pick one target." One target per session. Commit before compaction. Update this file when done.

---

## Critical — broken features

### 1. `mark_recall_used()` not implemented

**Status:** Completed (2026-03-21) — replaced with `brain_precision.py` module
**Files:** `servers/brain_precision.py` (NEW), `servers/brain_recall.py`, `servers/brain_engineering.py`, `servers/daemon_hooks.py`

The recall precision feedback loop was dead. Fixed by:
1. Created `servers/brain_precision.py` — `RecallPrecision` class owns all `recall_log` access
2. Three-signal evaluation model: Claude's response (stub), user followup (stub), explicit feedback (active)
3. Decoupled `_log_recall()` from `recall_with_embeddings()` — hooks call precision module explicitly
4. Wired into `hook_recall()` (two-turn followup) and `hook_post_response_track()` (response storage)
5. Consciousness reads `get_precision_summary()` instead of raw SQL
6. 24 tests in `tests/test_precision.py`

**Follow-up needed:** LLM evaluator to replace `pending_llm` stubs in evaluate_response/evaluate_followup.

### 2. `dal.py` log_recall() column mismatch

**Status:** Deprecated (2026-03-21) — superseded by `RecallPrecision.log_recall()`
**Files:** `servers/dal.py:141`

`dal.py`'s `log_recall()` uses wrong column names and is now bypassed. Marked deprecated with a comment pointing to `brain_precision.py`. No new callers should use it.

---

## High — dead code / abandoned features

### 3. Reasoning chains stub

**Status:** Not started
**Files:** `servers/brain_recall.py:517-524`, `servers/schema.py`

`reasoning_chains` and `reasoning_steps` tables exist in schema. `brain_recall.py` has empty stub: "reasoning methods not yet implemented, skipping for now." Tables are always empty.

**Decision needed:** Implement extraction or drop the tables and stub.

### 4. `tests/relearning.py` — 1679 lines obsolete

**Status:** Marked obsolete (2026-03-21)
**File:** `tests/relearning.py`

Early prototype for transcript replay. Duplicates hook logic that now lives in `daemon_hooks.py`. References `mark_recall_used()` which doesn't exist.

**Decision needed:** Delete entirely, or rewrite to use `daemon_hooks.py` functions.

### 5. `_STUB_consciousness_removed` in brain.py

**Status:** Not started
**File:** `servers/brain.py:834-910`

77-line stub marking where consciousness code was extracted to `brain_consciousness.py`. The real code is in the mixin now. Stub serves no purpose.

**To fix:** Delete the stub method.

---

## Medium — architectural debt

### 6. `log_consciousness_response()` possibly vestigial

**Status:** Needs investigation
**File:** `servers/brain_consciousness.py`

Only called in 2 places in `brain_evolution.py`. May not be active in current architecture.

### 7. `embedder.py` auto-install workaround

**Status:** Low priority
**File:** `servers/embedder.py:137`

TODO comment about removing auto-install once brain-embedding package is on PyPI.

---

## Completed

- [x] **Recall precision module** (2026-03-21) — `brain_precision.py` with three-signal evaluation model, 24 tests, hooks wired, consciousness reads summary
- [x] **dal.py log_recall deprecated** (2026-03-21) — superseded by `RecallPrecision.log_recall()`, wrong column names marked as deprecated
- [x] **Daemon consolidation** (v5.3.1, 2026-03-21) — 13 hooks centralized into `daemon_hooks.py`, ~3100 lines of duplicated code eliminated
- [x] **Recall precision signal gated** (2026-03-21) — `brain_engineering.py` now only fires recall precision signal when `useful > 0`, preventing false 0% alarms
- [x] **Stale 0% performance node deleted** (2026-03-21) — removed bogus consciousness node from live brain
