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
from typing import Optional, Set

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
        try:
            brain._log_error(
                'bg_writer_worker_stalled',
                RuntimeError('no drain in %ds, embed_depth=%d rwq_depth=%d'
                             % (int(age_s), embed_depth, rwq_depth)),
                'worker appears blocked or slow — investigate drain duration')
        except Exception as le:
            print('[embed_queue] stall log failed: %s' % le, file=sys.stderr)
    except Exception as e:
        # Stall check is best-effort observability — never raise.
        print('[embed_queue] _check_stall failed: %s' % e, file=sys.stderr)


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
        # Vectors stay on the primary connection under brain.write_lock
        # for now — the cache layer (CachedVectorDAL) holds invalidation
        # state keyed to self.conn. Migrating vectors to conn_bg_writer
        # is a follow-on (Phase 4b candidate); the load-bearing fix here
        # is moving temporal extraction off the lock.
        with brain.write_lock:
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
