"""Benchmark: VectorDAL (SQL) vs CachedVectorDAL (in-memory).

Measures the two operations that dominate recall:
  1. get_all_with_context — primary vectors + node context
  2. get_all_vectors      — enrichment vectors scan

Single-thread (cold-cache cost) and 7-concurrent (the spin scenario we hit).

Run against production brain.db via BRAIN_DB env:
  BRAIN_DB=~/AgentsContext/brain/brain.db python3 tests/bench_vector_cache.py
"""
import os
import sqlite3
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.dal import VectorDAL
from servers.dal_vector_cached import CachedVectorDAL


def _connect(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def _time_ms(fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1000.0


def bench_single(conn, n_iter=20):
    """Single-thread repeated calls. Reveals per-call latency."""
    plain = VectorDAL(conn)
    cached = CachedVectorDAL(conn)

    def _run(dal, method):
        times = []
        for _ in range(n_iter):
            times.append(_time_ms(getattr(dal, method)))
        return times

    results = {}
    for method in ['get_all_with_context', 'get_all_vectors', 'get_all_situations']:
        plain_t = _run(plain, method)
        cached_t = _run(cached, method)
        results[method] = {
            'plain_median_ms':  statistics.median(plain_t),
            'cached_median_ms': statistics.median(cached_t),
            'plain_p95_ms':     sorted(plain_t)[int(len(plain_t) * 0.95)],
            'cached_p95_ms':    sorted(cached_t)[int(len(cached_t) * 0.95)],
            'speedup':          statistics.median(plain_t) / max(statistics.median(cached_t), 0.001),
        }
    return results


def bench_concurrent(db_path, n_threads=7, n_iter_per_thread=10):
    """N threads hammering the hot read method. Simulates the pre_edit
    hook storm we hit on 2026-04-18 (7 concurrent readers thrashing
    SQLite's page cache).

    Plain path uses per-thread connections — Python's sqlite3 module on
    macOS ARM will SIGSEGV if multiple threads call execute() on one
    connection object, even with check_same_thread=False. The daemon
    avoids this via single-writer dispatch; the benchmark simulates the
    read-concurrency that WAL mode was designed for.

    Cached path keeps the single shared cache (the whole point — one
    in-memory matrix) and relies on CachedVectorDAL's own _sql_lock for
    the small nodes-table JOIN.
    """
    cached = CachedVectorDAL(_connect(db_path))

    def plain_run(method):
        times = []
        def worker():
            tc = _connect(db_path)
            try:
                dal = VectorDAL(tc)
                for _ in range(n_iter_per_thread):
                    times.append(_time_ms(getattr(dal, method)))
            finally:
                tc.close()
        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        return times, (time.perf_counter() - t0) * 1000.0

    def cached_run(method):
        times = []
        def worker():
            for _ in range(n_iter_per_thread):
                times.append(_time_ms(getattr(cached, method)))
        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        return times, (time.perf_counter() - t0) * 1000.0

    results = {}
    for method in ['get_all_with_context']:
        plain_t, plain_total = plain_run(method)
        cached_t, cached_total = cached_run(method)
        results[method] = {
            'plain_total_ms':     plain_total,
            'cached_total_ms':    cached_total,
            'plain_p50_call_ms':  statistics.median(plain_t),
            'cached_p50_call_ms': statistics.median(cached_t),
            'plain_p95_call_ms':  sorted(plain_t)[int(len(plain_t) * 0.95)],
            'cached_p95_call_ms': sorted(cached_t)[int(len(cached_t) * 0.95)],
            'total_speedup':      plain_total / max(cached_total, 0.001),
        }
    return results


def main():
    db_path = os.path.expanduser(
        os.environ.get('BRAIN_DB', '~/AgentsContext/brain/brain.db'))
    if not os.path.exists(db_path):
        print(f'DB not found: {db_path}', file=sys.stderr)
        sys.exit(1)
    print(f'DB: {db_path}')

    conn = _connect(db_path)
    v_count = conn.execute(
        'SELECT COUNT(*) FROM node_enrichments WHERE embedding IS NOT NULL'
    ).fetchone()[0]
    n_count = conn.execute('SELECT COUNT(*) FROM nodes WHERE archived=0').fetchone()[0]
    print(f'Active nodes: {n_count}   Vectors: {v_count}\n')

    print('=' * 78)
    print('SINGLE-THREAD (20 iterations, median + p95)')
    print('=' * 78)
    single = bench_single(conn, n_iter=20)
    print(f'{"method":<28}  {"plain p50":>10}  {"cached p50":>10}  {"plain p95":>10}  {"cached p95":>10}  {"speedup":>8}')
    print('-' * 96)
    for method, r in single.items():
        print(f'{method:<28}  {r["plain_median_ms"]:>9.1f}ms  '
              f'{r["cached_median_ms"]:>9.1f}ms  '
              f'{r["plain_p95_ms"]:>9.1f}ms  '
              f'{r["cached_p95_ms"]:>9.1f}ms  '
              f'{r["speedup"]:>7.1f}x')

    print()
    print('=' * 78)
    print('7 CONCURRENT THREADS × 10 calls each (70 total calls per variant)')
    print('=' * 78)
    concurrent = bench_concurrent(db_path, n_threads=7, n_iter_per_thread=10)
    print(f'{"method":<28}  {"plain total":>12}  {"cached total":>12}  {"plain p95":>10}  {"cached p95":>10}  {"speedup":>8}')
    print('-' * 96)
    for method, r in concurrent.items():
        print(f'{method:<28}  {r["plain_total_ms"]:>11.1f}ms  '
              f'{r["cached_total_ms"]:>11.1f}ms  '
              f'{r["plain_p95_call_ms"]:>9.1f}ms  '
              f'{r["cached_p95_call_ms"]:>9.1f}ms  '
              f'{r["total_speedup"]:>7.1f}x')


if __name__ == '__main__':
    main()
