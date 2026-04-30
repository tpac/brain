# Memory leak investigation — 2026-04-27 → 28

Pickup doc for the daemon memory + CPU investigation. Three independent
root causes were diagnosed and fixed; two pre-existing bugs were surfaced
and clearly characterized but not yet fixed.

---

## TL;DR

| Issue | Type | Status | Fix location |
|---|---|---|---|
| Daemon RSS ballooning to 4.6 GB | Code — ONNX `mem_pattern` cache thrashing on variable input shapes | ✅ Fixed | commit `615eeac` (bucketing in `servers/embedder.py`) |
| Idle daemon at 100% CPU | Config — `memory_watchdog.tracemalloc_enabled` left on after diagnosis | ✅ Fixed | DB config flipped to `false` |
| Self-referencing `community_member` edge | Data — `dec_ifp5` had `source_id == target_id` | ✅ Fixed | edge archived (backup at `brain.db.bak-pre-self-edge-fix-20260428-005631`) |
| fd leak in worker dispatch | Code — pre-existing, sockets not closed on worker hang | ❌ Open | `servers/daemon_server.py` accept loop + worker cleanup |
| onnxruntime 1.24 macOS lock contention | Upstream — fix in 1.25+ (PR #23278) | ❌ Open | wait for upgrade or set `intra_op_num_threads=1` |

The watchdog infrastructure (committed in `fffa6a1`) earned its keep: it's
the only reason all three closeable causes got caught. Without it, we'd
still be playing whack-a-mole.

---

## Original symptom

- Daemon process climbed from ~450 MB at startup to 4.6 GB over ~4 h of
  normal use
- Recall hook started timing out (`hook_recall: timeout`)
- Eventually the daemon would hit OS-level resource exhaustion and either
  crash, get killed, or spin
- Pattern recurred across multiple sessions; previous incidents had been
  blamed on different causes (encoder lock, fastembed thread spin)

Watchdog was added at the start of this investigation to capture the next
incident with timing data instead of after-the-fact guesswork.

---

## Diagnostic infrastructure built this session

### `servers/memory_watchdog.py` (commit `fffa6a1`)
Permanent opt-in module. Two layers, independently configurable:

- `memory_watchdog.enabled` — RSS sampling every N seconds (default 60).
  Logs `[mem] rss=X (delta) threads=N` to `daemon.log`. Flags >50 MB
  growth between samples with `⚠ growth`. Cheap.
- `memory_watchdog.tracemalloc_enabled` — heavier. Records stack traces
  for every Python allocation, dumps top-N allocators to
  `/tmp/brain-tracemalloc-{uid}-{ts}.txt` every M seconds. Use only
  during active investigation; **left on it pegs CPU at 100%** (see
  cause #2 below).

### `eval/verify_padding_safe.py`
Empirical check: does whitespace padding shift our embedder's output?
Required because Phase 2 of the leak fix (bucketing) hinges on the
assumption that trailing whitespace is semantically free.

Verdict: **safe**. 105 padding tests × 7 sample texts → cosine = 1.00000
in every case for spaces and newlines. Periods do shift (control case
that confirmed the test catches real shifts).

### `eval/verify_no_leak.py`
Reproduction test for the leak. 1000 mixed embeds (queries + edges +
content), watching RSS over time. Pre-fix: unbounded growth. Post-fix:
plateau at +396 MB after the first 100 embeds, zero further growth
through 900 more.

### `eval/analyze_baseline.py` (commit `d1fd1b2`)
Failure-pattern surfacer for `eval/full_suite.py` runs. Generates three
views: `failures_by_axis.md`, `scout_inspector.md`, `passes_vs_fails.md`.
Used to surface the "recall is the bottleneck, not encoding" insight in
the v9.5 baseline.

### Daemon stderr → `daemon.log` (commit `3d26a55`)
The `restart` command was redirecting the new daemon's stdout/stderr to
`/dev/null`. Every `Daemon started` line, every `[mem]` watchdog sample,
every encoding profile was being silently dropped after each restart.
Now they go to `daemon.log`.

This was masking everything; finding it was the unlock that let the
watchdog actually be useful.

---

## Root cause #1 — ONNX mem_pattern cache thrashing

### Diagnosis path

1. Watchdog caught a clean leak event 2026-04-27 ~01:02–01:04: 50 MB
   → 5.43 GB in 2 minutes, then plateaued at 4.61 GB.
2. Tracemalloc snapshots showed Python-tracked memory grew only +25 MB.
   The remaining 5.4 GB was C-extension memory (ONNX runtime).
3. Top Python-side growers — `embedder.py:287` (the `v.tobytes()` after
   embedding) and `surface_contract.py:920` (edge text format) — both
   grew by ~430 objects, exactly correlated.
4. 5.4 GB / 430 embedder calls ≈ 12.5 MB per call leaking into the
   ONNX arena.
5. Web search confirmed: onnxruntime's `mem_pattern` optimizer caches
   allocation patterns per distinct input tensor shape. With
   variable-length text inputs (every recall embedded different-length
   edge descriptions), the cache grew unboundedly. Documented in
   onnxruntime issues #11627, #22271, #11118, #9313, fastembed #222.

### Fix (commit `615eeac` — `servers/embedder.py`)

Pad each batch to one of a small fixed set of `(B, T)` shapes before
sending to the embedder. The mem_pattern cache then has at most ~12
entries (4 batch buckets × 3 length buckets) instead of thousands.

```python
_LEN_BUCKET_CHARS = (256, 1024, 4096)
_BATCH_BUCKETS = (1, 4, 16, 64)
```

All three embedder entry points (`embed_query`, `embed_document`,
`embed_batch`) go through `_bucket_pad`. No bypass paths.

Whitespace padding is semantically free for `nomic-embed-text-v1.5-Q`
(verified: 105 padding tests → cosine = 1.0).

### Verification

`eval/verify_no_leak.py` shows: 1000 mixed embeds → +396 MB once at the
start (arena warmup for the bucket shapes), then perfectly flat.
Throughput unchanged at ~37 embeds/sec.

### Caveat

Texts > 4096 chars get truncated to 4096 before embedding. Pre-fix the
model's native truncation kicked in (typically at ~8192 tokens). For our
brain this affects a small fraction of long-form content embeds. If
retrieval quality on long-form regresses, bump the largest bucket to
8192 and accept ~600 MB of warmup arena (still bounded).

---

## Root cause #2 — Tracemalloc left enabled after diagnosis

### Diagnosis

After Tom force-killed the wedged daemon, the freshly respawned daemon
hit **97.7% CPU on idle** within 2 minutes — no recalls, no encoding,
nothing. Just startup → 100% CPU.

`sample` showed all threads in `cond_wait`/`select`. No Python frames
were running. But `ps -o time=` confirmed 5 CPU-seconds in 5 wall-seconds
— really 100% of one core.

The daemon was loading the embedder (~500 MB allocs) + vector cache
(89 MB) at startup. Tracemalloc records a stack trace on every Python
allocation. Multiplied across millions of allocations during boot:
sustained 100% CPU just from stack-trace recording.

### Fix

Flipped `memory_watchdog.tracemalloc_enabled` from `true` → `false` in
`brain_meta`. Daemon respawned at 0% CPU, idle, listening normally.

### Lesson

This isn't a code regression — it's config drift. CLAUDE.md already
warned: *"Tracemalloc is heavier (slows allocation paths ~3x), so leave
it off until you actually need it."* I left it on after capturing the
leak data and forgot to flip it off.

**For next time:** treat tracemalloc as a "scope of investigation" tool.
Turn on, capture, turn off in the same session.

---

## Root cause #3 — Self-referencing `community_member` edge

### Discovery

Tom's hint: *"can it be in the db and not exactly in embed?"* — past
incidents had had DB-side root causes.

DB integrity scan turned up:
- 1 self-edge (`source_id == target_id`)
- 1 self-referencing `community_member` edge (same row)
- Both on node `dec_ifp5` ("Correction: dont hand user terminal commands
  — execute them yourself")

Almost certainly written by S2 community detection at some point — the
source/target got confused, leaving this decision claiming to be a
member of itself.

### Impact

Not the cause of the 100% CPU (`spread_activation` has a 3-hop safety
cap that prevents true infinite recursion). But every recall reaching
this node wastes hops on the self-cycle before the cap kicks in.

### Fix

```sql
UPDATE edge_relations
SET archived = 1, archived_at = strftime('%s','now'),
    archived_by = 'cleanup:self-member-bug'
WHERE edge_id = 'edg_06172bc1' AND archived = 0;
```

Backup: `~/AgentsContext/brain/brain.db.bak-pre-self-edge-fix-20260428-005631`.

### Follow-up worth considering

The S2 community detection pipeline shouldn't be capable of generating
self-member edges. Worth adding a guard at the encoder/dispatch layer:
reject `connect_to` writes where `source_id == target_id`.

---

## Pre-existing bugs surfaced (not yet fixed)

### fd leak in worker dispatch

When a recall worker hangs (e.g., on the onnxruntime macOS lock
contention below), the client times out at 60s and disconnects. The
daemon's worker holds the socket open forever — never closed.

Observed: 245 IPv4 sockets in CLOSED state held by daemon PID 43698,
hitting the 256-fd default limit. Once at the limit, every `accept()`
fails with `EMFILE` and the accept loop spins, logging the error and
retrying without backoff.

**Fix path** in `servers/daemon_server.py`:
1. Wrap worker dispatch in try/finally that always closes the client
   socket.
2. Add exponential backoff on `EMFILE` in the accept loop instead of
   tight retry.

Both small changes, ~30 min total work.

### onnxruntime 1.24 macOS lock contention

Stack: Python 3.11.11, onnxruntime **1.24.4**, fastembed 0.8.0.

Web search turned up that ORT thread synchronization on macOS is
specifically expensive, with up to 45% of inference time lost to lock
contention. The fix is upstream PR #23278 — landing in onnxruntime 1.25.

Workaround until then: set `intra_op_num_threads=1` and
`inter_op_num_threads=1` via the polyfill in `embedder.py`. Trade-off:
single-call latency ~1.3-1.5x slower, but eliminates the cond_wait
pile-up under concurrent load.

The polyfill we already have (`_install_fastembed_spin_polyfill`)
disables `allow_spinning` — that part is fine. The remaining contention
is at the cond_wait level which `allow_spinning=0` doesn't address.

Sources:
- [ORT issue #4251](https://github.com/microsoft/onnxruntime/issues/4251)
- [ORT issue #20354](https://github.com/microsoft/onnxruntime/issues/20354)
- [ORT PR #23278](https://github.com/microsoft/onnxruntime/pull/23278)
- [langchain issue #22898](https://github.com/langchain-ai/langchain/issues/22898)

---

## Commits this investigation

| Commit | What |
|---|---|
| `fffa6a1` | Memory watchdog (RSS sampling + tracemalloc, off by default) |
| `3d26a55` | Eval harness incremental writes + daemon restart stderr → daemon.log |
| `103e68e` | `eval/verify_padding_safe.py` — confirmed whitespace padding semantically free |
| `56b12ec` | `get_config` bool fix (was treating bool defaults as int and erroring on "true"/"false") |
| `e0cfa8a` | `full_suite` clobber prevention (don't overwrite other-phase .jsonls) |
| `615eeac` | **The leak fix**: bucket batch shapes in `embedder.py` |
| `d1fd1b2` | `eval/analyze_baseline.py` — failure-pattern surfacer |

Plus uncommitted DB writes:
- Self-member edge archived (`edg_06172bc1`)
- `memory_watchdog.tracemalloc_enabled` flipped to `false`

Plus backups created:
- `brain.db.bak-post-v9.5-baseline-20260427-003713` (177 MB, integrity ok)
- `brain.db.bak-pre-self-edge-fix-20260428-005631` (180 MB)

---

## How to verify on next session

1. **Idle CPU** — daemon process should be at 0% CPU when nothing's
   happening:
   ```bash
   DAEMON_PID=$(lsof -i :47203 | grep LISTEN | awk '{print $2}')
   T1=$(ps -p $DAEMON_PID -o time= | tr -d ' ')
   sleep 5
   T2=$(ps -p $DAEMON_PID -o time= | tr -d ' ')
   # T2 - T1 should be ~0 if daemon is idle
   ```

2. **RSS bounded under load** — run the leak repro:
   ```bash
   ./dev python3 eval/verify_no_leak.py
   ```
   Expect: RSS plateau at +396 MB, no growth past first 100 embeds.

3. **Watchdog samples in `daemon.log`** — should show steady RSS:
   ```bash
   grep "\[mem\]" ~/AgentsContext/brain/daemon.log | tail -20
   ```
   Look for `+0B` or small bounces, not `⚠ growth`.

4. **No more self-referencing edges**:
   ```bash
   sqlite3 ~/AgentsContext/brain/brain.db \
     "SELECT COUNT(*) FROM edges e
      JOIN edge_relations er ON er.edge_id = e.edge_id
      WHERE e.source_id = e.target_id AND er.archived = 0"
   ```
   Should return 0.

---

## Open work queue (priority order)

1. **fd leak in worker dispatch** — `daemon_server.py` accept-loop
   cleanup + EMFILE backoff. ~30 min.
2. **Add S2 guard against self-edges** — reject `connect_to` writes
   where `source_id == target_id` at the dispatch layer. Defensive
   correctness fix.
3. **Investigate the 5 rejected/failed abstention items** — `abs_pet_dog`
   0/3 and `abs_person_doctor` 0/3 in baseline_v9.5. Per
   `analyze_baseline` output, the answerer hallucinates content even
   when the surfacer correctly returns no context.
4. **multi-session regression** — 27% (broken Apr 25 run, no muster) →
   13% (v9.5 with muster). Recall surfaces less context for the failed
   queries (per `passes_vs_fails.md`). Likely related to the open
   fatigue + hub_dampening investigations (memories `3ebda2b9`,
   `c53d6a65`).
5. **Upgrade onnxruntime to 1.25 when available** — would address the
   macOS lock contention without needing the `intra_op_num_threads=1`
   workaround.

---

## Things to NOT forget

- The watchdog is opt-in but persistently configured. If you see the
  daemon at 100% CPU on idle, **first check
  `memory_watchdog.tracemalloc_enabled`** before anything else.
- Bucketing changes truncation behavior for >4096 char texts. If
  long-form recall quality drops, the largest bucket is the lever.
- Backup before any DB write. Always.
