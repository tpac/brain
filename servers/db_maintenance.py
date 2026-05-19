"""Database maintenance scheduler — backend-agnostic.

Owns a background thread that fires checkpoint + optimize operations on
registered databases at fixed cadences. Backend-specific work (SQLite's
`PRAGMA wal_checkpoint`, `PRAGMA optimize`) is delegated to a backend
module that implements the BackendOps protocol. If the brain ever moves
off SQLite, swap the backend module and this scheduler stays as-is.

Scope, deliberately narrow for first cut:
- `checkpoint` every 5 min — flush WAL into DB and truncate the WAL file.
- `optimize` every 30 min — refresh query planner statistics.
- `stats` collected every checkpoint tick and logged for observability.

Out of scope for first cut (call out if you add them):
- `integrity_check` at boot (refuse to start on failure)
- `VACUUM` (manual via maintenance-lock file; locks the DB)
- `ANALYZE` daily (heavier than optimize, only useful on big shifts)

The thread is `daemon=True` so it never blocks shutdown. All operation
failures are caught and logged via brain._log_error — a single bad tick
must not kill the loop.

This module imports the backend lazily via `db_backends.current` so the
scheduler doesn't need to know which backend is active.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Protocol


class BackendOps(Protocol):
    """Interface every backend implements. Methods take a db identifier
    (path for SQLite, conn string for others) and return a dict of
    diagnostic info or raise on error."""

    def apply_pragmas(self, conn) -> None: ...
    def checkpoint(self, db_path: str) -> Dict: ...
    def optimize(self, db_path: str) -> Dict: ...
    def stats(self, db_path: str) -> Dict: ...


# Default cadences (seconds). Override per-DB via register() if needed.
_DEFAULT_CHECKPOINT_INTERVAL_S = 5 * 60
_DEFAULT_OPTIMIZE_INTERVAL_S = 30 * 60
# Worker wakes every ~30s and checks "is anything due". Short enough to
# stay responsive to shutdown; long enough that the scheduler is cheap.
_TICK_INTERVAL_S = 30.0


class DBMaintenance:
    """Background scheduler. One per daemon process.

    Usage:
        m = DBMaintenance(log_fn=daemon._log,
                          log_error_fn=brain._log_error)
        m.register('brain', brain.db_path)
        m.register('brain_logs', logs_db_path)
        m.start()
        ...
        m.stop()  # signals worker to exit at next tick
    """

    def __init__(self,
                 log_fn: Optional[Callable[[str], None]] = None,
                 log_error_fn: Optional[Callable[[str, Exception, str], None]] = None,
                 checkpoint_interval_s: float = _DEFAULT_CHECKPOINT_INTERVAL_S,
                 optimize_interval_s: float = _DEFAULT_OPTIMIZE_INTERVAL_S,
                 tick_interval_s: float = _TICK_INTERVAL_S):
        self._log = log_fn or (lambda m: print('[db_maintenance] %s' % m, flush=True))
        self._log_error = log_error_fn  # may be None during early boot
        self._checkpoint_interval_s = checkpoint_interval_s
        self._optimize_interval_s = optimize_interval_s
        self._tick_interval_s = tick_interval_s

        self._registered: List[Dict] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Resolve backend lazily so unit tests can monkey-patch.
        self._backend = None

    def register(self, name: str, db_path: str) -> None:
        """Add a database to the maintenance schedule.

        `name` is for log readability ('brain', 'brain_logs'). `db_path`
        is what the backend operates on. Subsequent registers with the
        same name replace the entry (idempotent boot)."""
        self._registered = [r for r in self._registered if r['name'] != name]
        self._registered.append({
            'name': name,
            'db_path': db_path,
            'last_checkpoint_at': 0.0,
            'last_optimize_at': 0.0,
        })

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='db-maintenance')
        self._thread.start()
        self._log('started — %d DBs registered, checkpoint=%ds optimize=%ds' % (
            len(self._registered),
            int(self._checkpoint_interval_s),
            int(self._optimize_interval_s)))

    def stop(self) -> None:
        self._running = False

    # ─── Internal ─────────────────────────────────────────────────────

    def _resolve_backend(self):
        if self._backend is None:
            from . import db_backends
            self._backend = db_backends.current
        return self._backend

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._tick_interval_s)
            if not self._running:
                break
            try:
                self._tick()
            except Exception as e:
                # Tick failures must never kill the loop.
                self._report_error('db_maintenance_tick', e, 'top-level tick caught')

    def _tick(self) -> None:
        now = time.time()
        backend = self._resolve_backend()
        for entry in self._registered:
            if now - entry['last_checkpoint_at'] >= self._checkpoint_interval_s:
                self._run_op(backend, entry, 'checkpoint', now)
            if now - entry['last_optimize_at'] >= self._optimize_interval_s:
                self._run_op(backend, entry, 'optimize', now)

    def _run_op(self, backend, entry: Dict, op_name: str, now: float) -> None:
        t0 = time.time()
        try:
            result = getattr(backend, op_name)(entry['db_path'])
            took_ms = int((time.time() - t0) * 1000)
            entry['last_%s_at' % op_name] = now
            self._log('%s %s ok in %dms: %s' % (
                op_name, entry['name'], took_ms, _short(result)))
        except Exception as e:
            took_ms = int((time.time() - t0) * 1000)
            self._report_error(
                'db_maintenance_%s' % op_name, e,
                'db=%s took=%dms' % (entry['name'], took_ms))

    def _report_error(self, origin: str, e: Exception, context: str) -> None:
        if self._log_error is not None:
            try:
                self._log_error(origin, e, context)
                return
            except Exception:
                pass
        # Fallback: stderr via the regular log function.
        self._log('ERROR %s: %s (%s)' % (origin, e, context))


def _short(d: Dict) -> str:
    """Compact dict repr for log lines — values truncated."""
    parts = []
    for k, v in (d or {}).items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + '...'
        parts.append('%s=%s' % (k, s))
    return ' '.join(parts)
