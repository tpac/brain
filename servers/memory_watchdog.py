"""Memory watchdog — periodic RSS sampling + optional tracemalloc snapshots.

Why this exists: the daemon process has leaked twice now (2026-04-26 grew
to 4.6 GB after ~4h uptime). When that happens we want diagnostics already
in place — not a scramble to bolt-on profiling against a process that's
already swelling. This module is permanent infrastructure, opt-in via
config, zero overhead when off.

Two layers, independently toggleable:

1. RSS sampling (`memory_watchdog.enabled`):
   Cheap. Every N seconds, log the process's resident set size and a
   diff vs the previous sample. Surfaces growth trends in brain.log
   without instrumenting allocations.

2. Tracemalloc snapshots (`memory_watchdog.tracemalloc_enabled`):
   More expensive — Python's `tracemalloc` tracks every allocation site.
   Snapshots are taken every M seconds and the top-N largest allocators
   are written to a snapshot file under /tmp. Two consecutive snapshots
   can be diffed offline to find growth points.

Config keys (all read at thread start; toggling at runtime requires
a daemon restart for now):

  memory_watchdog.enabled            bool   default False
  memory_watchdog.interval_seconds   int    default 60     (RSS sampling)
  memory_watchdog.tracemalloc_enabled bool  default False
  memory_watchdog.tracemalloc_seconds int   default 600    (10 min)
  memory_watchdog.tracemalloc_top_n  int    default 25
  memory_watchdog.snapshot_dir       str    default /tmp

The watchdog is a `daemon=True` background thread — it never blocks
shutdown. All errors are caught + logged via brain._log_error so a
profiling failure can never crash the daemon.
"""
from __future__ import annotations

import os
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Optional


def _rss_bytes() -> int:
    """Resident set size in bytes for the current process.

    Reads from /proc on Linux, ps on macOS. Returns 0 on failure rather
    than raising — a profiling failure must never affect the daemon.
    """
    # Linux fast path
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    kb = int(line.split()[1])
                    return kb * 1024
    except FileNotFoundError:
        pass  # macOS / non-Linux
    except Exception:
        pass
    # macOS / fallback: ps
    try:
        import subprocess
        out = subprocess.check_output(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())],
            timeout=2.0,
        ).decode().strip()
        return int(out) * 1024
    except Exception:
        return 0


def _human(n_bytes: int) -> str:
    if n_bytes < 1024:
        return '%dB' % n_bytes
    if n_bytes < 1024 * 1024:
        return '%.1fKB' % (n_bytes / 1024)
    if n_bytes < 1024 ** 3:
        return '%.1fMB' % (n_bytes / 1024 / 1024)
    return '%.2fGB' % (n_bytes / 1024 ** 3)


def _signed_human(delta_bytes: int) -> str:
    sign = '+' if delta_bytes >= 0 else '-'
    return sign + _human(abs(delta_bytes))


class MemoryWatchdog:
    """Background memory sampler. Spawn one per daemon at most."""

    def __init__(self, brain, log_fn=None):
        self.brain = brain
        self.log_fn = log_fn or (lambda msg: print('[mem] %s' % msg, flush=True))
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._tracemalloc_started = False
        self._last_rss = 0
        self._last_tracemalloc_at = 0.0

        # Snapshot config at start — runtime toggle requires restart.
        self.interval = max(10, int(brain.get_config(
            'memory_watchdog.interval_seconds', 60)))
        self.tracemalloc_enabled = bool(brain.get_config(
            'memory_watchdog.tracemalloc_enabled', False))
        self.tracemalloc_seconds = max(60, int(brain.get_config(
            'memory_watchdog.tracemalloc_seconds', 600)))
        self.tracemalloc_top_n = max(5, int(brain.get_config(
            'memory_watchdog.tracemalloc_top_n', 25)))
        self.snapshot_dir = str(brain.get_config(
            'memory_watchdog.snapshot_dir', '/tmp'))

    @classmethod
    def maybe_start(cls, brain, log_fn=None) -> Optional['MemoryWatchdog']:
        """Start the watchdog iff `memory_watchdog.enabled` is True.

        Returns the running watchdog or None if disabled. Idempotent:
        second call from same daemon returns None (the existing
        watchdog stays). Safe to call from daemon startup unconditionally.
        """
        if not brain.get_config('memory_watchdog.enabled', False):
            return None
        wd = cls(brain, log_fn=log_fn)
        wd.start()
        return wd

    def start(self):
        if self.running:
            return
        self.running = True
        if self.tracemalloc_enabled and not tracemalloc.is_tracing():
            tracemalloc.start(25)  # 25 = stack frame depth captured
            self._tracemalloc_started = True
            self.log_fn('tracemalloc started (%d-frame depth, top %d, every %ds)' % (
                25, self.tracemalloc_top_n, self.tracemalloc_seconds))
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='memory-watchdog')
        self._thread.start()
        self._last_rss = _rss_bytes()
        self.log_fn('started — interval=%ds, baseline RSS=%s' % (
            self.interval, _human(self._last_rss)))

    def stop(self):
        """Stop the watchdog. Tracemalloc tracing also stops."""
        self.running = False
        if self._tracemalloc_started:
            try:
                tracemalloc.stop()
            except Exception:
                pass
            self._tracemalloc_started = False

    def _loop(self):
        while self.running:
            try:
                time.sleep(self.interval)
                self._sample_rss()
                if self.tracemalloc_enabled and (
                        time.time() - self._last_tracemalloc_at
                        >= self.tracemalloc_seconds):
                    self._snapshot_tracemalloc()
                    self._last_tracemalloc_at = time.time()
            except Exception as e:
                # Profiling must never crash the daemon. Log and continue.
                try:
                    self.brain._log_error(
                        'memory_watchdog_loop', e,
                        'sampling failed; loop continues')
                except Exception:
                    pass

    def _sample_rss(self):
        rss = _rss_bytes()
        if rss == 0:
            return  # couldn't read, skip
        delta = rss - self._last_rss
        self._last_rss = rss
        threads = threading.active_count()
        # Always log absolute; flag growth above 50MB since last sample.
        flag = ' ⚠ growth' if delta > 50 * 1024 * 1024 else ''
        self.log_fn('rss=%s (%s vs prev) threads=%d%s' % (
            _human(rss), _signed_human(delta), threads, flag))

    def _snapshot_tracemalloc(self):
        try:
            snapshot = tracemalloc.take_snapshot()
        except Exception as e:
            self.brain._log_error(
                'memory_watchdog_snapshot', e, 'take_snapshot failed')
            return
        # Group by lineno (most actionable for finding the leak source).
        top = snapshot.statistics('lineno')[:self.tracemalloc_top_n]
        ts = time.strftime('%Y%m%d-%H%M%S')
        path = Path(self.snapshot_dir) / f'brain-tracemalloc-{os.getuid()}-{ts}.txt'
        try:
            with open(path, 'w') as f:
                f.write('Tracemalloc snapshot at %s\n' % ts)
                f.write('PID: %d  RSS: %s\n' % (os.getpid(), _human(_rss_bytes())))
                f.write('Top %d allocators (by current size):\n\n' % len(top))
                for i, stat in enumerate(top, 1):
                    f.write('#%d %s\n' % (i, stat))
                f.write('\n--- traceback for #1 ---\n')
                if top:
                    for line in top[0].traceback.format():
                        f.write(line + '\n')
            self.log_fn('tracemalloc snapshot → %s (%d entries)' % (
                path, len(top)))
        except Exception as e:
            self.brain._log_error(
                'memory_watchdog_snapshot_write', e,
                'failed writing snapshot to %s' % path)


def get_watchdog_status(brain) -> dict:
    """Diagnostic — read config + current RSS without instantiating.

    Safe to call from the dispatch thread; doesn't start anything.
    """
    return {
        'enabled': bool(brain.get_config('memory_watchdog.enabled', False)),
        'interval_seconds': int(brain.get_config(
            'memory_watchdog.interval_seconds', 60)),
        'tracemalloc_enabled': bool(brain.get_config(
            'memory_watchdog.tracemalloc_enabled', False)),
        'tracemalloc_seconds': int(brain.get_config(
            'memory_watchdog.tracemalloc_seconds', 600)),
        'snapshot_dir': str(brain.get_config(
            'memory_watchdog.snapshot_dir', '/tmp')),
        'current_rss_bytes': _rss_bytes(),
        'current_rss_human': _human(_rss_bytes()),
        'tracemalloc_active': tracemalloc.is_tracing(),
    }
