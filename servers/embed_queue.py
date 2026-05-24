"""
brain — Embed Queue

Nodes that get written (remember/revise/connect) enqueue themselves here.
A single worker thread wakes every EMBED_DRAIN_INTERVAL seconds, drains the
set, and runs one pass of backfill_vectors() scoped to the drained ids.

Design constraints (from operator):
  - Single worker (no pool). One embed_batch call already uses ORT internally.
  - Skip-tick if the previous drain is still running. No overlapping drains.
  - Queue is in-memory. On daemon restart, S2 Heal sweeps up any gaps.
  - Latency bound: ~EMBED_DRAIN_INTERVAL seconds from write to indexed.

Exposed stats (via get_stats) feed embedder diagnostics so "my write isn't
indexed" is legible instead of silent.
"""

import sys
import threading
import time
from typing import Dict, Optional, Set

# Drain every N seconds if queue is non-empty. Empty queue = no-op, no work.
EMBED_DRAIN_INTERVAL = 5.0
# Max entities processed under a single write_lock acquisition. Lock is
# released between batches so concurrent writers can interleave. Set to
# handle the cold-start case (~20K entities at first boot) without
# holding the lock for the full drain.
DRAIN_BATCH_SIZE = 500
# Stall threshold — if pending work exists AND no drain has happened
# within STALL_THRESHOLD seconds, log loudly. Self-reported by the
# worker; covers "worker is alive but blocked / slow" not "worker is
# dead". Thread death is detected by daemon's thread-count watchdog.
STALL_THRESHOLD_S = EMBED_DRAIN_INTERVAL * 3

# Pull-reconciliation policy for trace embeddings (v27 episodic refs).
# Each tick: find up to N S0 trace events with no embedding yet, render
# per §5.3, embed in one batch, store. Newest-first so recent
# conversation is anchorable immediately; backfill of older traces
# works backwards on its own across subsequent ticks. No queue state;
# restart-safe — the LEFT JOIN finds whatever's still missing.
TRACE_DRAIN_LIMIT = 5
EAGER_TRACE_SCALES = ('s0',)
EAGER_TRACE_REF_TYPES = ('user_message', 'assistant_message', 'tool_result')
# Embed window — only consider traces newer than this. The architectural
# reason (not just a perf knob): traces older than v27/identity-stamping
# rollout have no human_identity / agent_identity in their metadata and
# would render with OPERATOR / ANCHOR sentinels. Decision 19 specifically
# keeps the embedding neighborhood concrete-token-only; embedding
# pre-stamping traces would land them in a different vector neighborhood
# from new traces. A 30-day window covers the source_refs use case (new
# encoder writes anchoring at recent traces) without touching the
# historical backlog.
TRACE_EMBED_WINDOW_DAYS = 30

# ─── State ───
_queue: Set[str] = set()           # node ids needing vector + date recomputation
_edge_queue: Set[str] = set()      # edge ids needing date recomputation
_lock = threading.Lock()
_drain_busy = threading.Lock()  # non-blocking; used for skip-tick semantics
_worker_started = False
_shutdown_requested = False
_stats = {
    'nodes_enqueued_total': 0,
    'edges_enqueued_total': 0,
    'drains_total': 0,
    'drains_skipped_busy': 0,
    'drains_skipped_empty': 0,
    'nodes_processed_total': 0,
    'edges_processed_total': 0,
    'vectors_written_total': 0,
    'temporal_intervals_written_total': 0,
    'traces_processed_total': 0,
    'traces_errors_total': 0,
    'traces_skipped_embedder_not_ready': 0,
    'last_trace_drain_at': None,
    'last_drain_at': None,   # epoch seconds
    'last_drain_took_ms': 0,
    'last_drain_size': 0,
    'worker_loop_errors_total': 0,
    'stalls_logged_total': 0,
}


def enqueue(node_id: str) -> None:
    """Mark a node as needing vector (re)computation. Cheap — set.add under lock."""
    if not node_id:
        return
    with _lock:
        _queue.add(node_id)
        _stats['nodes_enqueued_total'] += 1


def enqueue_edge(edge_id: str) -> None:
    """Mark an edge as needing date extraction. Cheap — set.add under lock.

    Edges only need date scanning (no embedding work today). Called from
    edge-write paths (GraphDAL.add_relation, edge revision).
    """
    if not edge_id:
        return
    with _lock:
        _edge_queue.add(edge_id)
        _stats['edges_enqueued_total'] += 1


def get_stats() -> dict:
    with _lock:
        return {**_stats,
                'queue_depth': len(_queue),
                'edge_queue_depth': len(_edge_queue),
                'interval_s': EMBED_DRAIN_INTERVAL}


def start(brain) -> None:
    """Start the single drain worker. Idempotent — safe to call multiple times."""
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
    t = threading.Thread(target=_worker_loop, args=(brain,),
                         name='embed-queue-drain', daemon=True)
    t.start()


def request_shutdown() -> None:
    """Signal the worker to exit cleanly at the next interval check."""
    global _shutdown_requested
    with _lock:
        _shutdown_requested = True


def _is_shutdown_requested() -> bool:
    with _lock:
        return _shutdown_requested


def _worker_loop(brain) -> None:
    """Worker thread body. One thread drives two queues every
    EMBED_DRAIN_INTERVAL seconds:

      1. `embed_queue` — node vectors + temporal date extraction.
         Heavier work; uses `_drain_busy` skip-tick semantics to avoid
         overlapping drains on cold-start (~20K entities at first boot).
      2. `recall_write_queue` — access marks + Hebbian co-access
         strengthening. Lightweight (atomic SQL +1s on
         `brain.conn_bg_writer`); always called per cycle since its
         own drain_once self-checks for empty queues.

    Top-level try/except is the load-bearing safety net for the
    "no silent errors" mandate — the worker thread MUST never die.
    Any unexpected exception is logged (origin `embed_worker_loop`)
    and the loop continues after a brief backoff.

    Shutdown: when `request_shutdown()` is called (either on
    `embed_queue` or on `recall_write_queue`), the loop exits at the
    next interval check.
    """
    # Import here so module-load order doesn't matter.
    from servers import recall_write_queue

    while True:
        try:
            time.sleep(EMBED_DRAIN_INTERVAL)

            if _is_shutdown_requested() or recall_write_queue.is_shutdown_requested():
                break

            # Liveness self-check — if pending work exists and last drain
            # was suspiciously long ago, log loudly. Helps surface a
            # blocked / slow worker without an external watchdog.
            _check_stall(brain, recall_write_queue)

            # Drain embed_queue if it has work. Skip-tick if previous
            # drain still running (cold-start defensive measure).
            with _lock:
                embed_pending = bool(_queue or _edge_queue)
            if embed_pending:
                if _drain_busy.acquire(blocking=False):
                    try:
                        _drain_once(brain)
                    except Exception as e:
                        try:
                            brain._log_error(
                                'embed_queue_drain', e,
                                'top-level embed drain caught')
                        except Exception as le:
                            print('[embed_queue] drain error: %s '
                                  '(log failed: %s)' % (e, le),
                                  file=sys.stderr)
                    finally:
                        _drain_busy.release()
                else:
                    with _lock:
                        _stats['drains_skipped_busy'] += 1
            else:
                with _lock:
                    _stats['drains_skipped_empty'] += 1

            # Pull-reconciliation: trace embeddings. Independent of
            # node/edge queues — writes to brain_logs.trace_embeddings
            # via its own connection. Caps per-tick work via
            # TRACE_DRAIN_LIMIT; runs on every tick so newly-written
            # S0 trace events get anchored within ~5s.
            try:
                _drain_trace_embeddings_once(brain)
            except Exception as e:
                try:
                    brain._log_error(
                        'embed_queue_trace_top', e,
                        'top-level trace embed drain caught')
                except Exception as le:
                    print('[embed_queue] trace embed error: %s '
                          '(log failed: %s)' % (e, le), file=sys.stderr)

            # Drain recall_write_queue (access + hebbian). Fast,
            # separate connection, separate transaction. drain_once is
            # contract-bound never to raise out — this try/except is
            # belt-and-suspenders.
            try:
                recall_write_queue.drain_once(brain)
            except Exception as e:
                try:
                    brain._log_error(
                        'recall_write_queue_drain', e,
                        'top-level recall_write drain caught — '
                        'drain_once is supposed to not raise')
                except Exception as le:
                    print('[embed_queue] recall_write drain error: %s '
                          '(log failed: %s)' % (e, le), file=sys.stderr)

        except Exception as e:
            # Worker thread MUST never die silently. Catch everything,
            # log loudly, sleep a beat, continue.
            with _lock:
                _stats['worker_loop_errors_total'] += 1
            try:
                brain._log_error(
                    'embed_worker_loop', e,
                    'worker loop caught unexpected exception — '
                    'continuing after backoff')
            except Exception as le:
                print('[embed_queue] worker loop fatal (log failed): '
                      'original=%s log=%s' % (e, le), file=sys.stderr)
            # Extra sleep on error so we don't spin if the cause is
            # immediate-and-repeating. time.sleep does not raise in
            # normal operation; if it does (signal handler reraises),
            # let it propagate — the OUTER while True will resume the
            # loop body at the next iteration.
            time.sleep(EMBED_DRAIN_INTERVAL)


def _check_stall(brain, recall_write_queue) -> None:
    """Log a stall warning if there's pending work AND last drain was
    longer than STALL_THRESHOLD_S ago. Self-reported by the worker —
    only fires when the worker thread is alive enough to execute this.
    For "worker thread dead entirely" detection, rely on the daemon's
    thread-count watchdog (separate signal).

    Log payload includes the write_lock holder (if TrackedRLock is in
    use) and the last drain's BEGIN IMMEDIATE wait time. These two
    signals usually answer "who's holding the WAL writer slot and for
    how long" — the root cause of most stalls.
    """
    try:
        embed_snap = get_stats()
        rwq_snap = recall_write_queue.get_stats()

        embed_depth = (embed_snap.get('queue_depth', 0)
                       + embed_snap.get('edge_queue_depth', 0))
        rwq_depth = (rwq_snap.get('access_queue_depth', 0)
                     + rwq_snap.get('hebbian_queue_depth', 0))
        total_depth = embed_depth + rwq_depth

        if total_depth == 0:
            return

        # "Last drain" = the most recent of either queue's last drain.
        embed_last = embed_snap.get('last_drain_at')
        rwq_last = rwq_snap.get('last_drain_at')
        last_drain_at = max(t for t in (embed_last, rwq_last) if t is not None) \
            if (embed_last or rwq_last) else None

        if last_drain_at is None:
            return  # never drained — first cycle, ignore

        age_s = time.time() - last_drain_at
        if age_s < STALL_THRESHOLD_S:
            return

        with _lock:
            _stats['stalls_logged_total'] += 1

        # Diagnostic enrichment: who holds write_lock right now, and
        # how long did the most recent BEGIN IMMEDIATE wait? Both are
        # best-effort — the snapshot() method only exists on
        # TrackedRLock and get_stats may not have the wait yet.
        holder_info = ''
        try:
            wl = getattr(brain, 'write_lock', None)
            if wl is not None and hasattr(wl, 'snapshot'):
                snap = wl.snapshot()
                if snap.get('holder'):
                    holder_info = ', write_lock_held_by=%s for %dms (depth=%d)' % (
                        snap['holder'], snap.get('held_for_ms') or 0,
                        snap.get('depth') or 0)
                else:
                    holder_info = ', write_lock=free'
        except Exception:
            pass

        last_wait = rwq_snap.get('last_begin_wait_ms', 0)
        wait_info = ', last_begin_wait=%dms' % last_wait if last_wait else ''

        try:
            brain._log_error(
                'bg_writer_worker_stalled',
                RuntimeError('no drain in %ds, embed_depth=%d rwq_depth=%d%s%s'
                             % (int(age_s), embed_depth, rwq_depth,
                                holder_info, wait_info)),
                'worker appears blocked or slow — investigate drain duration')
        except Exception as le:
            print('[embed_queue] stall log failed: %s' % le, file=sys.stderr)
    except Exception as e:
        # Stall check is best-effort observability — never raise.
        print('[embed_queue] _check_stall failed: %s' % e, file=sys.stderr)


def _render_trace_for_embedding(row: Dict) -> str:
    """Render a trace_event row to embedding text per docs/EPISODIC-REFERENCES.md §5.3.

    Concrete identity tokens at the embedding layer (revised decision
    19, biology-grounded): same individual's traces land in the same
    vector neighborhood regardless of role/context changes. Falls back
    to OPERATOR / ANCHOR sentinels if identity isn't configured in the
    trace metadata yet (so the pipeline keeps producing usable vectors
    on fresh installs).
    """
    meta = row.get('metadata') or {}
    ref_type = row.get('ref_type', '')
    human = meta.get('human_identity') or 'OPERATOR'
    agent = meta.get('agent_identity') or 'ANCHOR'
    # Prefer the longer metadata.content over the truncated summary;
    # falls back to summary when content isn't present (tool_result,
    # historical traces, etc.).
    content = meta.get('content') or row.get('summary') or ''
    if ref_type == 'user_message':
        return '%s: %s' % (human, content)
    if ref_type == 'assistant_message':
        return '%s: %s' % (agent, content)
    if ref_type == 'tool_result':
        tool = meta.get('tool') or 'tool'
        summary = row.get('summary') or ''
        return '%s via %s: %s' % (agent, tool, summary)
    # Unknown ref_type: best-effort tag so the embedder still gets
    # signal. Future ref_types (S1 recall, S1 encoding) can override
    # by adding a branch above; the design doc §5.3 has templates.
    return '%s: %s' % (ref_type or 'event', content)


def _drain_trace_embeddings_once(brain) -> None:
    """Pull-reconciliation tick: find recent S0 traces with no embedding,
    render → embed in one batch → store. Independent from node/edge
    drains (different table, different connection); runs even when
    those queues are empty. Skip-tick on prior overlap is unnecessary
    — TRACE_DRAIN_LIMIT caps per-tick work and the LEFT JOIN is cheap.
    """
    try:
        from datetime import datetime, timedelta, timezone
        since_iso = (datetime.now(timezone.utc)
                     - timedelta(days=TRACE_EMBED_WINDOW_DAYS)).isoformat()
        pending = brain._trace_dal.find_unembedded(
            limit=TRACE_DRAIN_LIMIT,
            scales=list(EAGER_TRACE_SCALES),
            ref_types=list(EAGER_TRACE_REF_TYPES),
            since=since_iso)
    except Exception as e:
        with _lock:
            _stats['traces_errors_total'] += 1
        try:
            brain._log_error('embed_queue_trace_find', e,
                             'find_unembedded raised')
        except Exception:
            pass
        return
    if not pending:
        with _lock:
            _stats['last_trace_drain_at'] = time.time()
        return

    try:
        from servers import embedder
        if not embedder.is_ready():
            # First few ticks during boot are expected — embedder
            # loads asynchronously. After that, "not ready" usually
            # means it failed to load and the worker would spin
            # forever silently. Track a per-process count and log
            # loudly when it persists.
            with _lock:
                _stats['traces_skipped_embedder_not_ready'] = (
                    _stats.get('traces_skipped_embedder_not_ready', 0) + 1)
                skips = _stats['traces_skipped_embedder_not_ready']
            # Loud once at the 5-tick mark (~25s), then every 50 ticks
            # (~4 min). Catches genuine boot grace; surfaces stuck state.
            if skips == 5 or (skips > 5 and skips % 50 == 0):
                try:
                    brain._log_error(
                        'embed_queue_trace_embedder_not_ready',
                        RuntimeError('embedder.is_ready() False after %d ticks' % skips),
                        'worker can\'t produce trace embeddings; check embedder.get_model_status()')
                except Exception:
                    pass
            return
        t0 = time.time()
        texts = [_render_trace_for_embedding(row) for row in pending]
        vectors = embedder.embed_batch(texts, kind='document')
        if not vectors or len(vectors) != len(pending):
            with _lock:
                _stats['traces_errors_total'] += 1
            try:
                brain._log_error(
                    'embed_queue_trace_embed_mismatch',
                    RuntimeError(
                        'embed_batch returned %d vectors for %d texts' %
                        (len(vectors) if vectors else 0, len(pending))),
                    'embedder returned partial or empty result; traces will be retried next tick')
            except Exception:
                pass
            return
        rows_to_store = []
        for row, vec, text in zip(pending, vectors, texts):
            if vec is not None:
                rows_to_store.append((row['id'], vec, text))
        if rows_to_store:
            model = embedder.get_config().get('model_name', 'unknown')
            n_written = brain._trace_dal.store_embeddings(
                rows_to_store, model=model)
            elapsed_ms = int((time.time() - t0) * 1000)
            with _lock:
                _stats['traces_processed_total'] += n_written
                _stats['last_trace_drain_at'] = time.time()
            print('[embed_queue] trace_embed %d in %dms (render+embed+store)' %
                  (n_written, elapsed_ms), flush=True)
    except Exception as e:
        with _lock:
            _stats['traces_errors_total'] += 1
        try:
            brain._log_error('embed_queue_trace_phase', e,
                             'render/embed/store failed')
        except Exception:
            pass


def _drain_once(brain) -> None:
    """Drain queues in BATCHES until empty.

    Holds brain.write_lock per-batch so cold-start (~20K entities at
    first boot) doesn't block other writers for the full drain. Each
    batch:
      1. backfill_vectors(node_ids) — writes node_enrichments embeddings
         (idempotent: vdal.find_missing skips already-vectorized nodes)
      2. backfill_entity_dates(node_ids, edge_ids) — writes entity_dates
         (idempotent: DELETE existing + INSERT extracted intervals)
    """
    t0 = time.time()
    total_nodes = 0
    total_edges = 0
    total_vectors = 0
    total_intervals = 0
    batches = 0

    while True:
        # Pull one batch out of the queues under the queue lock.
        with _lock:
            if not _queue and not _edge_queue:
                break
            node_batch = []
            edge_batch = []
            while _queue and len(node_batch) < DRAIN_BATCH_SIZE:
                node_batch.append(_queue.pop())
            remaining_budget = DRAIN_BATCH_SIZE - len(node_batch)
            while _edge_queue and remaining_budget > 0:
                edge_batch.append(_edge_queue.pop())
                remaining_budget -= 1
        if not node_batch and not edge_batch:
            break

        # ─── Vectors phase ─────────────────────────────────────────
        # Vectors stay on the primary connection (CachedVectorDAL holds
        # invalidation state keyed to self.conn — migrating to
        # conn_bg_writer is a separate refactor). The write_lock used to
        # wrap this whole call, which held the lock for the duration of
        # embed_batch — seconds to minutes of CPU work blocking every
        # other writer. backfill_vectors now self-locks only around the
        # DB write+commit per batch, so other writers get the slot
        # between vector types and between batches.
        if node_batch:
            try:
                result = brain.backfill_vectors(
                    batch_size=len(node_batch), node_ids=node_batch)
            except TypeError:
                result = brain.backfill_vectors(
                    batch_size=len(node_batch))
            total_vectors += sum(
                v for v in (result or {}).values() if isinstance(v, int))

        # ─── Temporal phase ────────────────────────────────────────
        # Outside brain.write_lock, on brain.conn_bg_writer. Single
        # BEGIN IMMEDIATE / COMMIT per batch. ROLLBACK + loud-log on
        # failure; the batch is dropped (no retry queue — matches the
        # loss-semantic contract). This is the architectural change
        # closing the lock cascade: foreground MCP writes via
        # `brain.conn` no longer compete with temporal extraction for
        # the WAL writer slot.
        try:
            from servers.temporal_extraction import backfill_entity_dates
            brain.conn_bg_writer.execute('BEGIN IMMEDIATE')
            try:
                stats = backfill_entity_dates(
                    brain, node_batch, edge_batch,
                    conn=brain.conn_bg_writer)
                total_intervals += int(stats.get('intervals_written', 0) or 0)
                brain.conn_bg_writer.commit()
            except Exception as inner:
                try:
                    brain.conn_bg_writer.rollback()
                except Exception as re:
                    try:
                        brain._log_error(
                            'bg_writer_drain_rollback_failed', re,
                            'rollback after temporal drain exception failed')
                    except Exception:
                        print('[embed_queue] rollback failed and log failed: '
                              '%s' % re, file=sys.stderr)
                raise inner
        except Exception as e:
            try:
                brain._log_error(
                    'bg_writer_drain_temporal', e,
                    'temporal batch dropped: nodes=%d edges=%d' %
                    (len(node_batch), len(edge_batch)))
            except Exception as le:
                print('[embed_queue] temporal extraction error: %s '
                      '(log failed: %s)' % (e, le), file=sys.stderr)

        total_nodes += len(node_batch)
        total_edges += len(edge_batch)
        batches += 1

    if batches == 0:
        # Empty tick — see matching note in recall_write_queue.drain_once.
        # Worker is alive, just had nothing to drain. Stamp last_drain_at
        # so the stall watchdog doesn't false-positive when a burst of
        # enqueues lands after a long idle period.
        with _lock:
            _stats['last_drain_at'] = t0
        return

    elapsed_ms = int((time.time() - t0) * 1000)
    with _lock:
        _stats['drains_total'] += 1
        _stats['nodes_processed_total'] += total_nodes
        _stats['edges_processed_total'] += total_edges
        _stats['vectors_written_total'] += total_vectors
        _stats['temporal_intervals_written_total'] += total_intervals
        _stats['last_drain_at'] = t0
        _stats['last_drain_took_ms'] = elapsed_ms
        _stats['last_drain_size'] = total_nodes + total_edges
    if batches > 1 or total_intervals > 0:
        print('[embed_queue] drained %d entities in %d batches (%dms): '
              'nodes=%d edges=%d vectors=%d intervals=%d'
              % (total_nodes + total_edges, batches, elapsed_ms,
                 total_nodes, total_edges, total_vectors, total_intervals),
              flush=True)
