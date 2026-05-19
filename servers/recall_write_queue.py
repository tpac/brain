"""
brain — Recall Write Queue

Deferred-write queue for recall-side bookkeeping that does NOT need realtime
persistence. The recall hot path enqueues; the embed_queue worker drains
every EMBED_DRAIN_INTERVAL seconds via `brain.conn_bg_writer`.

Two distinct signals ride on this queue:

  enqueue_access(node_id, session_id, ts)
    Recognition signal. Per-(node, session, drain-window) dedup'd by Dict
    semantics — the value held is the most recent timestamp seen. Drain
    produces ONE atomic UPDATE per unique (node, session) pair:
        access_count = access_count + 1
        activation   = MIN(1.0, activation + 0.1)
        last_accessed = ts (max within window)
    Multiple accesses to the same node from the same session within one
    drain window collapse to a single +1. Cross-session accesses to the
    same node still produce one increment per session.

  enqueue_hebbian_pairs(pairs, ts)
    Associative signal. Pairs are already resolved (caller did
    combinations() over Anchor's surface-layer picks — typically 3-5
    nodes, so ≤ C(5,2) = 10 pairs per recall). Drain strengthens (or
    creates) the `co_accessed` relation per pair via atomic SQL:
        weight = MIN(MAX_WEIGHT, weight + LEARNING_RATE * 0.5)

Design constraints (operator: Tom):
  - No silent errors. Every except logs via brain._log_error.
  - No read-modify-write where SQL can do +1 atomically.
  - Hot path is pure enqueue — no DB I/O.

Loss semantics (accepted by operator):
  - Daemon crash mid-drain: pending queue (≤ EMBED_DRAIN_INTERVAL seconds
    of recalls) lost. access_count is approximate. Hebbian re-fires next
    co-occurrence.
  - Transaction failure: ROLLBACK + drop the batch + log loudly. No retry
    queue (avoids poison-pill loops).

Stats (get_stats) feed dashboard observability.
"""

import sys
import threading
import time
from typing import Dict, Iterable, List, Tuple


# ─── Module-level state ─────────────────────────────────────────────
# All access guarded by `_lock`. Queues are mutated only via the public
# enqueue_* functions and _snapshot_and_clear().

# Access queue: maps (node_id, session_id) → most-recent timestamp string.
# Dict (not Set) so we preserve the latest `last_accessed` value without
# extra state. Within a drain window, multiple accesses to the same
# (node, session) collapse to one +1; the timestamp stored is the most
# recent the queue has seen.
_access_queue: Dict[Tuple[str, str], str] = {}

# Hebbian queue: list of (node_a, node_b, ts) tuples. NOT deduped at
# enqueue time — every entry is a separate co-occurrence event. If the
# same pair appears multiple times in one batch, the SQL drain processes
# each entry and the weight grows by N × delta (capped at MAX_WEIGHT).
_hebbian_queue: List[Tuple[str, str, str]] = []

_lock = threading.Lock()

# Shutdown flag — set by request_shutdown(), checked by the worker
# before starting a drain cycle. Allows clean exit from the worker loop.
_shutdown_requested = False

_stats = {
    'access_enqueued_total': 0,
    'hebbian_enqueued_total': 0,
    'drains_total': 0,
    'drains_skipped_empty': 0,
    'last_drain_at': None,            # epoch seconds
    'last_drain_took_ms': 0,
    'last_drain_size': 0,             # total items (access + hebbian) in batch
    'last_begin_wait_ms': 0,          # time spent in BEGIN IMMEDIATE (WAL slot wait)
    'access_drained_total': 0,
    'hebbian_pairs_drained_total': 0,
    'errors_total': 0,                # exceptions during drain (caught + logged)
    'rollbacks_total': 0,             # transactions that hit ROLLBACK
    'overlong_drains_total': 0,       # drains that took > _OVERLONG_THRESHOLD_MS
    'enqueue_errors_total': 0,        # exceptions during enqueue (lock contention)
}

# Threshold for the "drain took too long" warning emission (ms). Mirrors
# embed_queue's 10s overlong gate. Reviewed alongside drain interval if
# the queue ever changes producer/consumer ratio.
_OVERLONG_THRESHOLD_MS = 10_000

# Threshold for the "producer outpacing drain" warning emission. Sum of
# both queues. If we ever cross this, something pathological is happening
# upstream — emit a loud warning so we notice before the queue eats RAM.
_QUEUE_DEPTH_WARN = 5_000


# ─── Public API: enqueue ─────────────────────────────────────────────

def enqueue_access(node_id: str, session_id: str, ts: str) -> None:
    """Mark a node as accessed from a specific session at time `ts`.

    Cheap — dict update under lock. No DB I/O. Dedups multiple accesses
    to the same (node, session) within the current drain window down to
    one +1 increment, with last_accessed = the most recent ts seen.

    Caller MUST NOT touch the DB on the hot path. session_id of '' is
    coerced to 'unknown' (still tracked; access_count doesn't depend on
    attribution).
    """
    if not node_id:
        return
    sid = session_id or 'unknown'
    try:
        with _lock:
            key = (node_id, sid)
            existing = _access_queue.get(key)
            # ISO-8601 timestamp strings compare correctly as strings.
            if existing is None or ts > existing:
                _access_queue[key] = ts
            _stats['access_enqueued_total'] += 1
    except Exception as e:
        # Loud-by-default: we don't have a `brain` handle here for
        # _log_error, so stderr is the fallback. The caller (e.g.
        # brain_recall) can also wrap this and route through
        # brain._log_error for session attribution.
        with _lock:
            _stats['enqueue_errors_total'] += 1
        print('[recall_write_queue] enqueue_access failed: %s' % e,
              file=sys.stderr)


def enqueue_hebbian_pairs(pairs: Iterable[Tuple[str, str]], ts: str) -> None:
    """Enqueue Hebbian co-access events for a list of (node_a, node_b) pairs.

    Caller has already resolved the pair set — typically
    `combinations(surface_picks, 2)` where surface_picks is the 3-5
    nodes Anchor's surface layer consciously selected. Max C(5,2) = 10
    pairs per call.

    NOT deduped at enqueue time — if the same pair is enqueued by
    multiple recalls within one drain window, the SQL drain processes
    each entry and weight grows accordingly (capped at MAX_WEIGHT).

    Pairs are stored as-given (no sort/normalize). The drain's edge
    lookup checks both directions, so direction-insensitive in practice.
    Self-pairs (a == b) are silently dropped — defensive against caller
    bugs.
    """
    if not pairs:
        return
    try:
        with _lock:
            for a, b in pairs:
                if a and b and a != b:
                    _hebbian_queue.append((a, b, ts))
                    _stats['hebbian_enqueued_total'] += 1
    except Exception as e:
        with _lock:
            _stats['enqueue_errors_total'] += 1
        print('[recall_write_queue] enqueue_hebbian_pairs failed: %s' % e,
              file=sys.stderr)


# ─── Public API: introspection ───────────────────────────────────────

def queue_depth() -> int:
    """Total pending items across both queues. Cheap; for stall heartbeat."""
    with _lock:
        return len(_access_queue) + len(_hebbian_queue)


def get_stats() -> dict:
    """Snapshot of queue stats. Safe to call any time."""
    with _lock:
        return {
            **_stats,
            'access_queue_depth': len(_access_queue),
            'hebbian_queue_depth': len(_hebbian_queue),
            'overlong_threshold_ms': _OVERLONG_THRESHOLD_MS,
            'queue_depth_warn': _QUEUE_DEPTH_WARN,
            'shutdown_requested': _shutdown_requested,
        }


# ─── Public API: shutdown ────────────────────────────────────────────

def request_shutdown() -> None:
    """Signal the worker to skip subsequent drain cycles. Called from
    daemon shutdown for clean termination. Idempotent.
    """
    global _shutdown_requested
    with _lock:
        _shutdown_requested = True


def is_shutdown_requested() -> bool:
    with _lock:
        return _shutdown_requested


# ─── Internal: snapshot/swap ─────────────────────────────────────────

def _snapshot_and_clear() -> Tuple[Dict[Tuple[str, str], str],
                                   List[Tuple[str, str, str]]]:
    """Atomically swap queue contents to local snapshots and clear the
    module-level structures. Subsequent enqueues land in the fresh
    structures while the drain proceeds against the snapshot.

    Returns (access_snapshot, hebbian_snapshot).
    """
    global _access_queue, _hebbian_queue
    with _lock:
        access_snap = _access_queue
        hebbian_snap = _hebbian_queue
        _access_queue = {}
        _hebbian_queue = []
    return access_snap, hebbian_snap


# ─── Public API: drain ───────────────────────────────────────────────

def drain_once(brain) -> None:
    """Drain both queues against `brain.conn_bg_writer` in a single
    transaction. Called by the embed_queue worker once per cycle
    (wired in Phase 3).

    Empty queues are a no-op (counted in drains_skipped_empty).
    On any exception during the transaction: ROLLBACK + log via
    `bg_writer_batch_rollback` + drop the snapshot. The dropped batch
    is NOT re-queued — preserves the loss-semantic contract and avoids
    poison-pill loops (one corrupt pair could otherwise stall the
    queue forever).

    This function's contract is: never raise out — log and return.
    Top-level errors are caught here so the worker loop's outer guard
    is a backstop, not the primary safety net.
    """
    t0 = time.time()

    access_snap, hebbian_snap = _snapshot_and_clear()
    batch_size = len(access_snap) + len(hebbian_snap)

    if batch_size == 0:
        with _lock:
            _stats['drains_skipped_empty'] += 1
        return

    conn = brain.conn_bg_writer
    rolled_back = False
    begin_wait_ms = 0
    try:
        # BEGIN IMMEDIATE so we grab the WAL writer slot upfront; without
        # this SQLite auto-begins on first write and could deadlock on
        # busy_timeout mid-batch if a foreground write started first.
        # We measure this separately because stalls usually point here —
        # the WAL slot is contended, BEGIN IMMEDIATE waits, everything
        # downstream pays. The wait time is exported in stats so the
        # stall watchdog can attribute time accurately.
        _bi_t0 = time.time()
        conn.execute('BEGIN IMMEDIATE')
        begin_wait_ms = int((time.time() - _bi_t0) * 1000)

        if access_snap:
            # Atomic +1 per (node, session) pair via executemany. The
            # `archived = 0` filter makes the UPDATE a silent no-op on
            # archived nodes (race between enqueue and drain — intended,
            # not an error). Activation bump (0.1) matches
            # NodeDAL.mark_accessed's hardcoded boost.
            conn.executemany(
                'UPDATE nodes SET '
                '    access_count = access_count + 1, '
                '    activation = MIN(1.0, activation + 0.1), '
                '    recency_score = 1.0, '
                '    last_accessed = ?, '
                '    updated_at = ? '
                'WHERE id = ? AND archived = 0',
                [(ts, ts, nid) for (nid, _sid), ts in access_snap.items()])
            with _lock:
                _stats['access_drained_total'] += len(access_snap)

        if hebbian_snap:
            _apply_hebbian_pairs(brain, conn, hebbian_snap)
            with _lock:
                _stats['hebbian_pairs_drained_total'] += len(hebbian_snap)

        conn.commit()

    except Exception as e:
        # Primary path failed. Attempt rollback; either way, log loudly
        # and drop the batch.
        try:
            conn.rollback()
            rolled_back = True
        except Exception as re:
            # Double-fault. Best-effort log via _log_error; if that also
            # fails, stderr.
            try:
                brain._log_error('bg_writer_drain_rollback_failed', re,
                                 'rollback after primary drain exception failed')
            except Exception as le:
                print('[recall_write_queue] rollback failed AND _log_error '
                      'failed: rollback=%s log=%s' % (re, le),
                      file=sys.stderr)
        try:
            brain._log_error(
                'bg_writer_batch_rollback', e,
                'drain_once dropped batch: access=%d hebbian=%d' %
                (len(access_snap), len(hebbian_snap)))
        except Exception as le:
            print('[recall_write_queue] _log_error itself failed: drain=%s '
                  'log=%s' % (e, le), file=sys.stderr)
        with _lock:
            _stats['errors_total'] += 1
            if rolled_back:
                _stats['rollbacks_total'] += 1
        # NOTE: do NOT re-enqueue the snapshot. The data is intentionally
        # lost — preserves the loss contract and avoids poison-pill loops.

    took_ms = int((time.time() - t0) * 1000)
    with _lock:
        _stats['drains_total'] += 1
        _stats['last_drain_at'] = t0
        _stats['last_drain_took_ms'] = took_ms
        _stats['last_drain_size'] = batch_size
        _stats['last_begin_wait_ms'] = begin_wait_ms
        if took_ms > _OVERLONG_THRESHOLD_MS:
            _stats['overlong_drains_total'] += 1

    # Overlong warning — log AFTER stats are updated so observers see
    # the increment regardless of whether _log_error succeeds.
    if took_ms > _OVERLONG_THRESHOLD_MS:
        try:
            brain._log_error(
                'bg_writer_drain_overlong',
                RuntimeError('drain took %dms (>%dms threshold) — BEGIN IMMEDIATE waited %dms' %
                             (took_ms, _OVERLONG_THRESHOLD_MS, begin_wait_ms)),
                'access=%d hebbian=%d begin_wait_ms=%d' %
                (len(access_snap), len(hebbian_snap), begin_wait_ms))
        except Exception as le:
            print('[recall_write_queue] overlong log failed: %s' % le,
                  file=sys.stderr)

    depth_now = queue_depth()
    if depth_now > _QUEUE_DEPTH_WARN:
        try:
            brain._log_error(
                'bg_writer_queue_overflow',
                RuntimeError('post-drain queue depth %d > %d threshold' %
                             (depth_now, _QUEUE_DEPTH_WARN)),
                'producer outpacing drain — investigate')
        except Exception as le:
            print('[recall_write_queue] overflow log failed: %s' % le,
                  file=sys.stderr)


# ─── Internal: hebbian application ───────────────────────────────────

def _apply_hebbian_pairs(brain, conn,
                          hebbian_snap: List[Tuple[str, str, str]]) -> None:
    """Apply Hebbian strengthening for a list of (a, b, ts) triples.

    Per-pair logic:
      1. Look up edge_id via GraphDAL.get_edge_id (either direction).
      2. If edge exists: atomic UPDATE on edge_relations WHERE
         relation='co_accessed' AND archived=0. Cap weight at MAX_WEIGHT
         via SQL MIN(). If no co_accessed row exists on the edge yet,
         INSERT one via GraphDAL.add_relation (no auto-commit; respects
         the outer transaction).
      3. If no edge: GraphDAL.add_relation creates the edge AND the
         co_accessed relation with default weight. No segment filter —
         surface picks ARE the same context by definition (the segment
         filter that lived in the old `_hebbian_strengthen` was a proxy
         for "same context" that Anchor's surface selection makes
         redundant).

    Operates on `brain.conn_bg_writer`. We pass `commit=False` to
    GraphDAL.add_relation (which otherwise commits at the end of every
    upsert) so the outer BEGIN IMMEDIATE / COMMIT around the batch
    remains atomic. Without commit=False, a single add_relation mid-
    batch would commit earlier pairs and break the rollback contract.

    Per-pair exceptions are caught + logged but don't abort the batch.
    The outer transaction's success/failure is independent of any
    single pair. If a pair raises, it's lost (matches loss semantic).
    """
    from .brain_constants import LEARNING_RATE, MAX_WEIGHT, EDGE_TYPES
    from .dal import GraphDAL

    co_default_weight = EDGE_TYPES['co_accessed']['defaultWeight']
    delta = LEARNING_RATE * 0.5  # matches strengthen_relation's bump magnitude
    gdal = GraphDAL(conn)

    for a, b, ts in hebbian_snap:
        try:
            edge_id = gdal.get_edge_id(a, b)
            if edge_id:
                # Atomic strengthen — if a co_accessed row exists on this
                # edge, MIN(cap, weight + delta) in one statement. If not,
                # rowcount == 0 and we fall through to add_relation.
                #
                # Schema note (2026-05-18): `last_strengthened` lives on
                # the `edges` aggregate table, NOT `edge_relations`. The
                # initial draft of this UPDATE wrongly named it on the
                # relation row and every pair raised "no such column".
                # We now bump `edges.last_strengthened` in a sibling
                # statement after the relation update.
                cur = conn.execute(
                    'UPDATE edge_relations '
                    'SET weight = MIN(?, weight + ?) '
                    'WHERE edge_id = ? AND relation = ? AND archived = 0',
                    (MAX_WEIGHT, delta, edge_id, 'co_accessed'))
                if cur.rowcount > 0:
                    # Relation existed and was strengthened — also bump
                    # the edge-level last_strengthened so analytics can
                    # see "when did this co_accessed pair last fire".
                    conn.execute(
                        'UPDATE edges SET last_strengthened = ? '
                        'WHERE edge_id = ?',
                        (ts, edge_id))
                else:
                    # Edge exists but no active co_accessed relation
                    # (either never had one, or it's archived). Use
                    # add_relation — handles the archived-revive path
                    # too, plus correct edge_relations field defaults.
                    gdal.add_relation(
                        a, b, 'co_accessed',
                        description='hebbian co-access',
                        weight=co_default_weight,
                        encoding_source='recall:hebbian',
                        commit=False)
            else:
                # No physical edge between the pair. add_relation
                # creates both the edges row and the edge_relations row.
                gdal.add_relation(
                    a, b, 'co_accessed',
                    description='hebbian co-access',
                    weight=co_default_weight,
                    encoding_source='recall:hebbian')
        except Exception as e:
            try:
                brain._log_error(
                    'bg_writer_drain_hebbian', e,
                    'pair=%s,%s' % (a[:8], b[:8]))
            except Exception:
                print('[recall_write_queue] hebbian log failed for pair '
                      '%s,%s: %s' % (a[:8], b[:8], e), file=sys.stderr)
            with _lock:
                _stats['errors_total'] += 1


# ─── Test-only helpers ───────────────────────────────────────────────

def _clear_for_test() -> None:
    """Reset module state. Production code MUST NOT call this; it bypasses
    the normal enqueue → drain cycle and risks contention with the worker.
    """
    global _access_queue, _hebbian_queue, _shutdown_requested
    with _lock:
        _access_queue = {}
        _hebbian_queue = []
        _shutdown_requested = False
        for k in _stats:
            _stats[k] = 0 if isinstance(_stats[k], int) else None
