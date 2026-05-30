# Write-path transaction discipline — root fix

**Status:** interim guard SHIPPED; root fix DEFERRED to a future session.
**Relates to:** BACKLOG.md reviewer follow-up **F3** (this doc is F3's root-cause
analysis + reproduction + fix options).
**Surfaced:** 2026-05-29, by inserting S2 into the Frozen Corpus eval build
(`eval/longmem/build_corpus.py`).

## Symptom

During an S2 final-flush loop (multiple `run_s2` passes in quick succession),
the community encoder's `brain_batch` threw:

```
[s2ce] BATCH 1 FAILED: cannot start a transaction within a transaction
[s2ce] Encoder failed or incomplete - NOT stamping rejections
```

Two defects in one:
1. **The crash** — `brain_batch`'s explicit `BEGIN IMMEDIATE` ran while
   `self.conn` was already mid-transaction.
2. **Silent swallow** — the community encoder caught it below `run_s2` and
   returned `actions=0` with no exception, so the S2 Δ recorder logged
   `errors=0` and `brain_errors` got no row. Only stdout showed it. (A
   loud-by-default violation in `community_encoder.py:_encode`.)

It is **intermittent** — it only fires when the community encoder issues
`brain_batch` across multiple S2 passes in one flush, which depends on
LLM-driven proposal counts. Tiny 2-turn items reproduced it once and then
not again across reruns.

## Root cause

`self.conn` is opened with Python's **default deferred isolation**
(`sqlite3.connect(db_path, check_same_thread=False)` — no `isolation_level=None`,
[brain.py:168](../servers/brain.py:168)). Under deferred isolation Python
auto-issues `BEGIN` before any DML and you must `commit()`. The brain *also*
issues explicit `BEGIN IMMEDIATE` in `brain_batch` / queue drains. These two
transaction-control regimes coexist on the same connection.

The collision happens when a write path runs a DML on `self.conn` (Python
auto-`BEGIN`s) and **does not commit** — leaving the transaction open — and the
next `brain_batch` then issues `BEGIN IMMEDIATE` → *cannot start a transaction
within a transaction*. This is the same disease F3 names from the other side:
**GraphDAL write methods (`add_relation`, `delete_node_edges`, `remove_relation`,
archive paths) manage transactions inconsistently** — some commit inside a
batch (breaking F3's all-or-nothing rollback), some leave a deferred auto-BEGIN
open (this crash). The community encoder writes `community_member` edges via
`add_relation`, which is squarely in F3's named set.

Confirmed NOT the cause: concurrency (recall-write + embed queues use the
separate `conn_bg_writer`; a standalone eval Brain starts no `self.conn` writer
thread), `record_rejections` (commits, [rejection_table.py:210](../servers/scales/s2/rejection_table.py:210)),
recall/`_mark_accessed` (enqueue-only, read-only at SQLite on `self.conn`).

## Interim fix — SHIPPED

[dispatch_write.py:498](../servers/dispatch_write.py:498) — `_handle_brain_batch`
now enforces its own clean transaction boundary. If `self.conn.in_transaction`
is True at entry, it flushes the orphan, logs `brain_batch_stale_txn` loudly,
and begins clean. This converts the silent crash into a **loud, recovered
event** and removes the real corruption risk in the old code (a leaked txn's
writes were silently folding into the next batch's commit).

Mechanically verified: with the leaked-txn state simulated, `brain_batch`
recovers, the node persists, and the connection is clean afterward.

**The guard is now also the diagnostic.** The next time it fires — likely the
first real (20-item) corpus build — the `brain_batch_stale_txn` log plus the
surrounding stdout will name the op that ran immediately before, pinning the
exact leaking GraphDAL method.

## Root fix — two options for a future session

**Option A (recommended) — centralize write-path transaction discipline (this IS F3).**
Audit GraphDAL write methods so none commit or `BEGIN` internally; all defer to
the caller's envelope via `brain._maybe_commit()` (no-op while `_batch_mode`,
commits otherwise). Add a `commit: bool = True` flag where a method is called
both standalone and inside a batch. Targeted, preserves the current write model,
fixes both F3's atomicity symptom and this crash.

- Reproduce first: run `build_corpus.py` on the 20-item dev slice; the guard's
  loud log names the leaking method.
- Add a unit test that calls that method standalone and asserts
  `conn.in_transaction is False` afterward.
- Audit the full set (`add_relation`, `remove_relation`, `delete_node_edges`,
  archive paths) for the same pattern.

**Option B (defense-in-depth, only with benchmark) — `isolation_level=None`.**
Open `self.conn` / `conn_bg_writer` / `logs_conn` in autocommit and make
explicit `BEGIN IMMEDIATE` / `COMMIT` the *sole* transaction control. Eliminates
the Python auto-`BEGIN` class entirely. **Sacred write topology — do not ship
without** benchmarking recall (`eval/decode_funnel.py`) and encode
(`eval/s1_encode_eval.py`) latency before/after, because it changes commit
semantics on every write path. Bigger blast radius than A for a smaller code
diff.

**Recommendation:** do A (it's F3, and the guard's log will hand us the exact
method). Hold B as a separate, benchmarked hardening decision — the guard
already removes the user-visible failure, so there's no urgency to take the
blast radius.

## Also fix when doing this

`community_encoder.py:_encode` swallows batch failures without raising or
logging to `brain_errors` (returns `actions=0`). Make it `_log_error` on batch
failure so a broken S2 encode is loud, not inferred from a zero. (The eval's S2
Δ recorder counts `errors` from `run_s2` returns — it can't see failures
swallowed below that level.)
