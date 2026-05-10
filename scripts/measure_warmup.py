"""Measure warmup vs first-recall cost on an isolated copy of production.

Compares two paths against an isolated brain (copied from live brain.db):

  A. Cold first-recall — fresh Brain, no warmup, recall immediately.
     Captures the user-visible "first prompt after daemon boot" cost.

  B. Warm first-recall — fresh Brain, run Brain.warm_up(), then recall.
     Captures the cost remaining after warmup paid the first-call tax.

For both, also report a steady-state recall (the second one) so we can
see the per-call floor.

Run:
    ./dev python3 scripts/measure_warmup.py

Reports a table: phase | wall_ms | rss_delta_mb. RSS is sampled before
and after each phase, so we can see which step balloons memory and
whether warmup actually shifts the cost off the recall path.
"""
from __future__ import annotations
import os
import sys
import time
import gc

# Path setup — run as `./dev python3 scripts/measure_warmup.py`
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _rss_mb() -> float:
    """Resident set size in MB."""
    try:
        # Linux fast path
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    # macOS
    try:
        import subprocess
        out = subprocess.check_output(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())]).strip()
        return int(out) / 1024.0
    except Exception:
        return 0.0


def _phase(label: str, fn):
    """Run fn, return (wall_ms, rss_delta_mb, result)."""
    gc.collect()
    rss_before = _rss_mb()
    t0 = time.monotonic()
    result = fn()
    wall_ms = int((time.monotonic() - t0) * 1000)
    rss_after = _rss_mb()
    delta = rss_after - rss_before
    print(f"  {label:35s}  wall={wall_ms:6d}ms  rss={rss_after:7.1f}MB  Δ={delta:+7.1f}MB")
    return wall_ms, delta, result


def _run(use_warmup: bool, query: str = 'whats next ?'):
    """Build a fresh Brain on the isolated copy, optionally warm, then recall."""
    from tests.isolated_brain import IsolatedBrain

    print(f"\n=== {'WARM' if use_warmup else 'COLD'} run ===")
    with IsolatedBrain() as env:
        brain = env.brain  # Already constructed by IsolatedBrain
        # Note: IsolatedBrain's __enter__ has already loaded the embedder
        # and run schema checks. We're measuring incremental warmup +
        # first-recall cost from here, which mirrors the daemon's state
        # right after boot announces ready.
        print(f"  baseline                         rss={_rss_mb():7.1f}MB")

        if use_warmup:
            _phase('warm_up()', brain.warm_up)

        _phase('first recall',
               lambda: brain.recall(query=query, limit=10))
        _phase('second recall (steady state)',
               lambda: brain.recall(query=query, limit=10))


def main():
    print("Measuring warmup vs cold first-recall on an isolated copy of brain.db.")
    print("Each run is a fresh Brain → fresh recall.")
    _run(use_warmup=False)
    _run(use_warmup=True)


if __name__ == '__main__':
    main()
