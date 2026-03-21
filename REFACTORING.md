# Refactoring Targets

Last updated: 2026-03-21 (post v5.3.1 daemon consolidation)

## How to use this file

Start a new session. Say "open REFACTORING.md and pick one target." One target per session. Commit before compaction. Update this file when done.

---

## Critical — broken features

### 1. `mark_recall_used()` not implemented

**Status:** Not started
**Files:** `servers/brain_recall.py`, `servers/brain_engineering.py`, `servers/daemon_hooks.py`

The recall precision feedback loop is dead:
- `recall_log` schema has `used_ids`, `used_count`, `precision_score` columns
- `recall_with_embeddings()` returns `_recall_log_id` with every result
- **Nothing ever writes back** to update the row
- The consciousness signal in `brain_engineering.py:1563` checks `used_count > 0` — always finds 0
- Signal generator currently gated (`if useful > 0`) as workaround so it doesn't fire false 0% alarms

**To fix:**
1. Add `Brain.mark_recall_used(recall_log_id, used_ids)` method
2. In `hook_post_response_track()`: compare recalled node IDs against response content, call mark with matches
3. Remove the `if useful > 0` gate in `brain_engineering.py`
4. Add test in `test_system.py`

### 2. `dal.py` log_recall() column mismatch

**Status:** Not started
**Files:** `servers/dal.py:141`, `servers/schema.py:742`, `servers/brain_recall.py:1165`

Two code paths write to `recall_log` with different column names:
- `dal.py:141` uses `result_ids` and `intent` (columns that **don't exist** in schema)
- `brain_recall.py:1165` uses `returned_ids` and `returned_count` (correct)

**To fix:** Align `dal.py` to match schema, or remove `dal.py`'s `log_recall()` entirely since `brain_recall.py` handles it.

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

- [x] **Daemon consolidation** (v5.3.1, 2026-03-21) — 13 hooks centralized into `daemon_hooks.py`, ~3100 lines of duplicated code eliminated
- [x] **Recall precision signal gated** (2026-03-21) — `brain_engineering.py` now only fires recall precision signal when `useful > 0`, preventing false 0% alarms
- [x] **Stale 0% performance node deleted** (2026-03-21) — removed bogus consciousness node from live brain
