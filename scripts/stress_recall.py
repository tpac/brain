#!/usr/bin/env python3
"""Stress test brain.recall + hook_pre_edit via daemon TCP.

Fires many parallel calls to surface concurrent-recall behavior under
realistic conditions:
  - Mix of identical queries (exercises result cache + single-flight)
  - Mix of unique queries (exercises commit-batching, no cache hit)
  - Measures p50/p95/p99 latency, success rate
  - Tracks daemon CPU/RSS before/during/after
  - Counts cache hits / dispatch hits in returned timings

Usage:
  ./dev python3 scripts/stress_recall.py
  ./dev python3 scripts/stress_recall.py --workers 16 --iterations 50
  ./dev python3 scripts/stress_recall.py --pre-edit  # stress pre_edit instead
"""

import argparse
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from servers.daemon_client import send_command


REPEAT_QUERIES = [
    "editing servers/brain_recall.py",
    "editing servers/scales/s1/surface_contract.py",
    "editing CLAUDE.md",
    "git status",
]

UNIQUE_QUERY_TEMPLATE = "stress test query {i} {nonce}"


def daemon_proc():
    out = subprocess.check_output(['launchctl', 'list'], text=True)
    for line in out.splitlines():
        if 'com.brain.daemon' in line:
            parts = line.split()
            if parts and parts[0] != '-':
                return int(parts[0])
    return None


def daemon_stats(pid):
    if not pid:
        return None
    try:
        out = subprocess.check_output(
            ['ps', '-p', str(pid), '-o', '%cpu,rss,etime'],
            text=True, stderr=subprocess.DEVNULL).strip().splitlines()
        if len(out) < 2:
            return None
        cpu, rss_kb, etime = out[1].split(None, 2)
        return {'cpu_pct': float(cpu), 'rss_mb': int(rss_kb) / 1024.0, 'etime': etime}
    except Exception:
        return None


def call_recall(query, limit=15, timeout=20.0):
    t0 = time.time()
    res = send_command('recall', {'query': query, 'limit': limit}, timeout=timeout)
    elapsed_ms = (time.time() - t0) * 1000
    ok = bool(res.get('ok'))
    err = res.get('error', '') if not ok else ''
    nresults = 0
    if ok:
        result = res.get('result') or {}
        nresults = len(result.get('results') or [])
    return {'ok': ok, 'ms': elapsed_ms, 'err': err, 'n': nresults}


def call_pre_edit(file, timeout=20.0):
    t0 = time.time()
    res = send_command('hook_pre_edit', {'filename': file, 'tool_name': 'Edit'},
                       timeout=timeout)
    elapsed_ms = (time.time() - t0) * 1000
    ok = bool(res.get('ok'))
    err = res.get('error', '') if not ok else ''
    return {'ok': ok, 'ms': elapsed_ms, 'err': err}


def run_burst(callable_fn, jobs, workers, label):
    results = []
    pid = daemon_proc()
    before = daemon_stats(pid)
    print('\n=== %s ===' % label)
    print('  jobs=%d workers=%d  daemon: PID=%s CPU=%.1f%% RSS=%.0fMB' % (
        len(jobs), workers, pid,
        (before or {}).get('cpu_pct', 0),
        (before or {}).get('rss_mb', 0)))

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(callable_fn, *args) for args in jobs]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({'ok': False, 'ms': -1, 'err': 'thread_exc:%s' % e[:80]})
    t_total = time.time() - t_start

    after = daemon_stats(pid)
    succ = [r for r in results if r['ok']]
    fail = [r for r in results if not r['ok']]
    times = sorted(r['ms'] for r in succ)

    if times:
        p50 = times[len(times)//2]
        p95 = times[int(len(times)*0.95)] if len(times) >= 20 else times[-1]
        p99 = times[int(len(times)*0.99)] if len(times) >= 100 else times[-1]
        avg = statistics.mean(times)
    else:
        p50 = p95 = p99 = avg = 0

    print('  total: %.2fs  ok=%d  fail=%d' % (t_total, len(succ), len(fail)))
    print('  latency ms:  avg=%.0f  p50=%.0f  p95=%.0f  p99=%.0f  min=%.0f  max=%.0f' % (
        avg, p50, p95, p99, times[0] if times else 0, times[-1] if times else 0))
    print('  daemon after: CPU=%.1f%%  RSS=%.0fMB  Δ=%+.0fMB' % (
        (after or {}).get('cpu_pct', 0),
        (after or {}).get('rss_mb', 0),
        ((after or {}).get('rss_mb', 0)) - ((before or {}).get('rss_mb', 0))))
    if fail:
        # Show first 3 unique error messages
        seen_errs = set()
        for r in fail:
            if r['err'] not in seen_errs:
                print('  err: %s' % (r['err'][:120]))
                seen_errs.add(r['err'])
                if len(seen_errs) >= 3:
                    break

    return {
        'total_s': t_total, 'ok': len(succ), 'fail': len(fail),
        'avg_ms': avg, 'p50_ms': p50, 'p95_ms': p95, 'p99_ms': p99,
        'before': before, 'after': after,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument('--workers', type=int, default=8,
                        help='concurrent thread count (default 8)')
    parser.add_argument('--iterations', type=int, default=20,
                        help='calls per phase (default 20)')
    parser.add_argument('--pre-edit', action='store_true',
                        help='stress pre_edit instead of recall')
    args = parser.parse_args()

    if args.pre_edit:
        # Phase: same file repeated (cache should kick in)
        same_file_jobs = [('servers/brain_recall.py',)] * args.iterations
        run_burst(call_pre_edit, same_file_jobs, args.workers,
                  'pre_edit — same file × %d (cache hit territory)' % args.iterations)

        # Phase: unique files (no cache hits)
        unique_jobs = [('test/file_%d.py' % i,) for i in range(args.iterations)]
        run_burst(call_pre_edit, unique_jobs, args.workers,
                  'pre_edit — unique files × %d (cold path)' % args.iterations)

        # Phase: mixed (some repeat, some unique)
        mixed_jobs = []
        for i in range(args.iterations):
            if i % 3 == 0:
                mixed_jobs.append(('test/file_%d.py' % i,))
            else:
                mixed_jobs.append(('servers/brain_recall.py',))
        run_burst(call_pre_edit, mixed_jobs, args.workers,
                  'pre_edit — mixed × %d' % args.iterations)
        return 0

    # Recall phases
    # Phase 1: identical queries → cache + single-flight should kick in
    repeat_jobs = []
    for i in range(args.iterations):
        repeat_jobs.append((REPEAT_QUERIES[i % len(REPEAT_QUERIES)],))
    run_burst(call_recall, repeat_jobs, args.workers,
              'recall — repeat queries × %d (cache + single-flight)' % args.iterations)

    # Phase 2: unique queries → exercises commit-batching, no cache hit
    nonce = int(time.time())
    unique_jobs = [(UNIQUE_QUERY_TEMPLATE.format(i=i, nonce=nonce),)
                   for i in range(args.iterations)]
    run_burst(call_recall, unique_jobs, args.workers,
              'recall — unique queries × %d (cold)' % args.iterations)

    # Phase 3: mixed
    mixed_jobs = []
    for i in range(args.iterations):
        if i % 2 == 0:
            mixed_jobs.append((REPEAT_QUERIES[i % len(REPEAT_QUERIES)],))
        else:
            mixed_jobs.append((UNIQUE_QUERY_TEMPLATE.format(i=i, nonce=nonce + 1),))
    run_burst(call_recall, mixed_jobs, args.workers,
              'recall — mixed × %d (50/50)' % args.iterations)

    print('\nDone. Watch for: low fail rate, p95 < a few seconds, daemon CPU back near 0% after.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
