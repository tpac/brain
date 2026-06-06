# Code review — self-channel messaging + trace contract (2026-06-05)

**Scope:** `git diff fecf9d9..694de91` (`servers/ tests/ dashboard/`) — the
self-channel per-message-TTL + trace-contract work + its follow-up fixes.
**Effort:** extra-high (`/code-review` xhigh) — 9 finder angles (6 subagents) →
self-verification of every substantive claim against code → gap sweep.
**Outcome:** 6 findings fixed (4 in commit `694de91`; #8 + #9 in a 2026-06-06
follow-up); 4 left open (real but defensible/latent/pre-existing) — listed below
for next session.

The diff is fundamentally sound: the high-risk mechanics all verified **correct**
(see "Verified correct" — don't re-investigate them).

---

## Fixed + deployed (commit `694de91`)

- **#1 — TTL config crash.** `signal._resolve_ttl_hours` wrapped `get_config` in
  an unguarded `float()`. `get_config` returns the *raw string* when its numeric
  auto-parse fails (a typo'd config), so `float("abc")` raised an uncaught
  `ValueError` out of every `send()`. Now: `try/except` → fall back to the
  documented default + `_log_error` loud. +regression test
  (`test_nonnumeric_ttl_config_falls_back_not_crash`).
- **#7 — dead `final_text` loud-truncation.** `runner.py:430` silently
  pre-truncated `final_text[:2000]` *before* `build_delta_metadata`, so the loud
  `_cap_text_loud` marker never fired (the drop was silent upstream). Now the
  runner passes full text; the builder caps the trace **loudly**.
  `_save_journal` / `_save_session_context` already self-cap → no behavior change
  for them.
- **#3 — double-validation / asymmetric logging.** `validate_trace_metadata` ran
  at BOTH the `trace_append` command handler AND `TraceDAL.append`. Removed the
  command-path copy → `TraceDAL.append`/`append_batch` is the single chokepoint.
  **Architectural note:** the DAL logs to **stderr→daemon.log, not the errors
  table** — on purpose. `TraceDAL` can't call `brain._log_error` mid-append
  because that commits, which would break `brain_batch` atomicity (same reason
  `_maybe_warn_identity_unset` uses stderr). See "#6-style follow-up" below if
  you want trace-metadata violations in the *queryable* dashboard surface.
- **#5 — stale docstring.** `present_streams` doc now says focus = latest
  *conversational* turn (user OR assistant per `CONVERSATIONAL_REF_TYPES`), not
  user_message-only (CR3 changed the behavior).

### Follow-up (2026-06-06) — the two "no silent failures" findings

- **#8 — `_add_column_if_missing` swallowed ALL exceptions.** (`schema.py`) The
  bare `except Exception: pass` hid genuine ALTER failures (locked DB, disk
  full), not just the expected duplicate-column case — a column that failed to
  add would break a feature at runtime with no trace. Now: `except
  sqlite3.OperationalError`, branch on `'duplicate column'` (the idempotent
  re-run signal stays silent — it isn't a failure), any other OperationalError
  prints LOUD to stderr→daemon.log, boot continues (matches every migration
  block in the file). Non-OperationalError exceptions propagate — a bad
  `col_type` is a dev-time bug that should fail loud.
- **#9 — duplicated loud-truncation helpers.** `_cap_text_loud` /
  `_cap_list_loud` (`trace_contract.py`) and the inlined body-cap in
  `_render_one` (`self_contract.py`) were two implementations of one "never a
  silent slice" contract (node 8178593a) that could drift. Extracted both
  primitives to a new `servers/loud_truncation.py` that neither domain owns
  (imports nothing from `servers` — no cycle); the differing markers are now a
  param. Marker output is byte-identical, so the marker-pinning tests
  (`test_trace_delta_shape`, `test_self_signal`) pass unchanged.

---

## Open — carryover for next session (ranked)

None are blocking; all shipped code is correct as-is. Pick up by value.

- **#6 (low-med, altitude) — `WAKE_ENVELOPE_MARKER` cross-system coupling.**
  The presence focus filter content-sniffs the literal `'<task-notification>'`
  ([dal.py active_sessions_by_turn] `NOT LIKE WAKE_ENVELOPE_MARKER||'%'`), but
  that string is produced by the *external* task-notification harness, not brain
  code. If the harness changes its envelope prefix, the filter silently stops
  matching and wake envelopes leak back into watchers' focus (the exact CR3 bug,
  regressed invisibly with no loud failure). **Deeper fix:** tag the wakeup
  ignition *structurally* at write time (a distinct ref_type or a metadata flag
  set when the heartbeat/ignite is recorded) instead of sniffing turn content.
- **#2 (low, latent) — bare-marker delta stored as `{identity}` dict.** A delta
  ref_type with `metadata=None` (an S2 early-out marker like "No clusters to
  process") passes validation (None→True), but `_stamp_identity(None)` then
  returns `{human_identity, agent_identity}` — so the stored row is a present
  dict that does NOT conform to `DELTA_METADATA_SHAPE`. Benign today (nothing
  re-validates stored rows); a future read-side shape check would flag every
  legitimate no-op marker. **Fix options:** `_stamp_identity` returns `None`
  unchanged for `None` input (cleanest — don't stamp identity onto a no-op
  marker); do NOT move validation after stamping (that would false-positive on
  bare markers).
- **#10 (low) — reap is idle-gated.** `reap_expired` only runs inside the idle
  S2-maintenance window. With the new **1h** broadcast TTL, a continuously-active
  (never-idle) operator leaves expired + legacy-NULL rows un-reaped far past TTL
  → `self_inflight` grows. Delivery stays correct (drain/peek/outbox exclude via
  `expires_at > now`); only the table-growth bound is defeated. **Fix:** run reap
  on a lightweight cadence independent of the idle window, or accept.
- **#11 (low, likely by-design) — outbox broadcast visibility shrinks.**
  `outbox()` now filters `expires_at > now`, so a broadcast vanishes from the
  sender's own `self_outbox` after ~1h (was 24h). Consistent with "broadcast is
  ephemeral," but a real reduction in sender-side delivery observability. Accept,
  or add a separate sent-log if sender observability matters.

**Informational / won't-fix:**
- `iso_after` (and `iso_now`/`iso_cutoff`) use `.isoformat()`, which omits
  `.ffffff` at microsecond==0 — the "always …ffffff+00:00" docstring invariant
  isn't literally guaranteed. Harmless for hours-scale TTL; a latent trap for any
  future exact-instant lex comparison. Pre-existing across the clock family.
- A directed self-message addressed to one's *own* session id is never delivered
  (`_PENDING_INBOX_SQL`'s `from_session != to_session` excludes it). Pre-existing
  filter, not changed here; arguably intentional.

---

## Verified correct (do NOT re-investigate)

The review checked these and cleared them — they are sound:

- **`active_sessions_by_turn` f-string SQL** — placeholder count + positional
  bind order verified correct (9 params, subquery-before-FROM textual order
  matches the tuple); no injection (only `?`-placeholder strings interpolated,
  never values); `WAKE_ENVELOPE_MARKER` LIKE has no unescaped wildcards.
- **`expires_at` NULL three-valued logic** — drain/peek/outbox exclude NULL
  (`NULL > x` is falsy), reap deletes NULL (`<= now OR IS NULL`). Symmetric: a
  legacy NULL row is never delivered AND always reaped — no stuck orphans.
  (`TestLegacyNullExpires`.)
- **`_pending_rows` dedup (CR5)** — drain and peek share one query + params;
  drain's consume-once (write_lock + `INSERT OR IGNORE` + self_delivered PK) and
  `ORDER BY created_at` are byte-identical to pre-dedup.
- **`_s0_trace`** — all four S0 appends map field-for-field; `ctx.session_id`
  is bound (the self_message append gains the session_id it previously dropped).
- **Removed symbols** — `DEFAULT_SIGNAL_TTL_HOURS` and `INFLIGHT_FIELDS` have
  zero dangling references anywhere.
- **`send()` new `expires_at` key** — additive; no caller unpacks by index or
  asserts an exact key set.
- **0-hour config override** — `set_config(..., 0)` → `"0"` (truthy) → `int("0")`
  → `iso_after(hours=0)` = send-time → strict `>` excludes on any later drain.
- **dashboard `expires_at`** — SELECT column maps to `r[7]` correctly; legacy
  NULL renders as JSON null (passive reader, tolerated).
