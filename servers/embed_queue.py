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

# ─── State ───
_queue: Set[str] = set()
_lock = threading.Lock()
_drain_busy = threading.Lock()  # non-blocking; used for skip-tick semantics
_worker_started = False
_stats = {
    'enqueued_total': 0,
    'drains_total': 0,
    'drains_skipped_busy': 0,
    'nodes_processed_total': 0,
    'vectors_written_total': 0,
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
        _stats['enqueued_total'] += 1


def get_stats() -> dict:
    with _lock:
        return {**_stats, 'queue_depth': len(_queue), 'interval_s': EMBED_DRAIN_INTERVAL}


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
            if not _queue:
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
    """Swap the queue and run one backfill pass scoped to the drained ids."""
    with _lock:
        drained: Set[str] = _queue.copy()
        _queue.clear()
    if not drained:
        return

    t0 = time.time()
    try:
        result = brain.backfill_vectors(batch_size=max(50, len(drained)),
                                        node_ids=drained)
    except TypeError:
        # backfill_vectors without node_ids param yet — fall back to full scan
        result = brain.backfill_vectors(batch_size=max(50, len(drained)))
    elapsed_ms = int((time.time() - t0) * 1000)

    vectors = sum(v for v in (result or {}).values() if isinstance(v, int))
    with _lock:
        _stats['drains_total'] += 1
        _stats['nodes_processed_total'] += len(drained)
        _stats['vectors_written_total'] += vectors
        _stats['last_drain_at'] = t0
        _stats['last_drain_took_ms'] = elapsed_ms
        _stats['last_drain_size'] = len(drained)
