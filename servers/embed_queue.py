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

# ─── State ───
_queue: Set[str] = set()           # node ids needing vector + date recomputation
_edge_queue: Set[str] = set()      # edge ids needing date recomputation
_lock = threading.Lock()
_drain_busy = threading.Lock()  # non-blocking; used for skip-tick semantics
_worker_started = False
_stats = {
    'nodes_enqueued_total': 0,
    'edges_enqueued_total': 0,
    'drains_total': 0,
    'drains_skipped_busy': 0,
    'nodes_processed_total': 0,
    'edges_processed_total': 0,
    'vectors_written_total': 0,
    'temporal_intervals_written_total': 0,
    'last_drain_at': None,   # epoch seconds
    'last_drain_took_ms': 0,
    'last_drain_size': 0,
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


def _worker_loop(brain) -> None:
    while True:
        time.sleep(EMBED_DRAIN_INTERVAL)
        # Cheap pre-check — don't try to acquire drain lock if nothing to do.
        with _lock:
            if not _queue and not _edge_queue:
                continue
        # Skip-tick if a previous drain is still running.
        if not _drain_busy.acquire(blocking=False):
            with _lock:
                _stats['drains_skipped_busy'] += 1
            continue
        try:
            _drain_once(brain)
        except Exception as e:
            import sys
            print('[embed_queue] drain error: %s' % e, file=sys.stderr)
        finally:
            _drain_busy.release()


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

        # Process this batch under brain.write_lock. Released before the
        # next batch so concurrent writers can interleave.
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

            try:
                from servers.temporal_extraction import backfill_entity_dates
                stats = backfill_entity_dates(brain, node_batch, edge_batch)
                total_intervals += int(stats.get('intervals_written', 0) or 0)
                # write_entity_dates does INSERT/DELETE without committing —
                # owner of the lock commits at the end of the batch.
                brain.conn.commit()
            except Exception as e:
                import sys as _sys
                print('[embed_queue] temporal extraction error: %s' % e,
                      file=_sys.stderr)

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
