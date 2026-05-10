"""Bench cold vs warm Haiku first-call latency.

Question: how much does pre-importing the SDK, constructing the client at
boot, and pinging models.retrieve + a 1-token messages.create save on the
first "real" surface call the user pays for?

Why subprocesses: lazy-imports, httpx pool init, DNS, and TLS handshake
are process-scoped — once paid, they persist for the lifetime of the
process. Measuring cold vs warm in the same process would only measure
the cold leg. Each mode below runs in a fresh subprocess so cold is
genuinely cold.

Modes (selected via --mode):
  cold — no warmup. Measure the user's first messages.create() call as if
         the daemon had just spawned and the user prompted immediately.
         This is what the brain pays today on a fresh daemon.
  warm — warmup at "boot": import anthropic, construct client, call
         models.retrieve, call messages.create(max_tokens=1). THEN
         measure a second messages.create() as the "user's first call".
         Reports both the warmup cost (off the user's critical path)
         and the user-visible call cost.

Output is one JSON line per run, so the runner can parse multiple
samples cleanly.

Run:
    ./dev python3 scripts/bench_anthropic_warmup.py
        # spawns several cold + warm subprocesses, prints comparison

    ./dev python3 scripts/bench_anthropic_warmup.py --mode cold
        # one cold subprocess, prints JSON to stdout

    ./dev python3 scripts/bench_anthropic_warmup.py --mode warm
        # one warm subprocess, prints JSON to stdout
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# What the surface call actually looks like in production. Mirrored from
# servers/scales/s1/surface.py:_call_surface — same model, same shape, same
# max_tokens magnitude. The system block is short and arbitrary; we're
# measuring connection cost, not real surface output.
MODEL = 'claude-haiku-4-5'
WARMUP_PING_MSG = [{'role': 'user', 'content': '.'}]
USER_LIKE_MSG = [{
    'role': 'user',
    'content': (
        # ~50 tokens to mimic a real surface user message length without
        # actually doing surface logic. The body is irrelevant — we want
        # the request shape to be representative, not the answer to be
        # useful. Token cost: ~$0.0001 per call.
        'Pick the most relevant items from the list and return JSON: '
        '{"selected": ["a", "b"]}.'
    ),
}]


def _bench_cold() -> dict:
    """Cold: import on demand, construct, call. Time the messages.create()."""
    t_import_0 = time.monotonic()
    # Import is intentionally inside this function so it counts toward cold.
    from servers.scales.dispatch import load_env
    load_env()
    import anthropic
    t_import = time.monotonic() - t_import_0

    t_client_0 = time.monotonic()
    client = anthropic.Anthropic()
    t_client = time.monotonic() - t_client_0

    # The measurement: first user-visible Haiku call.
    t_call_0 = time.monotonic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=USER_LIKE_MSG,
    )
    t_call = time.monotonic() - t_call_0

    return {
        'mode': 'cold',
        'import_ms': int(t_import * 1000),
        'client_ctor_ms': int(t_client * 1000),
        'first_call_ms': int(t_call * 1000),
        'user_visible_ms': int((t_import + t_client + t_call) * 1000),
        'request_id': resp._request_id,
    }


def _bench_warm() -> dict:
    """Warm: pay import + client + retrieve + ping at 'boot'. Time the
    user's FIRST real call after that."""
    # ── "Boot" phase: paid before the user prompts ──
    t_import_0 = time.monotonic()
    from servers.scales.dispatch import load_env
    load_env()
    import anthropic
    t_import = time.monotonic() - t_import_0

    t_client_0 = time.monotonic()
    client = anthropic.Anthropic()
    t_client = time.monotonic() - t_client_0

    # Free TLS + httpx pool warmup. Doesn't bill.
    t_retrieve_0 = time.monotonic()
    client.models.retrieve(MODEL)
    t_retrieve = time.monotonic() - t_retrieve_0

    # Inference path warmup. ~$0.001.
    t_ping_0 = time.monotonic()
    client.messages.create(model=MODEL, max_tokens=1, messages=WARMUP_PING_MSG)
    t_ping = time.monotonic() - t_ping_0

    boot_total = t_import + t_client + t_retrieve + t_ping

    # ── User phase: this is what the user perceives ──
    t_call_0 = time.monotonic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=USER_LIKE_MSG,
    )
    t_call = time.monotonic() - t_call_0

    return {
        'mode': 'warm',
        'import_ms': int(t_import * 1000),
        'client_ctor_ms': int(t_client * 1000),
        'models_retrieve_ms': int(t_retrieve * 1000),
        'warmup_ping_ms': int(t_ping * 1000),
        'boot_total_ms': int(boot_total * 1000),
        'first_call_ms': int(t_call * 1000),
        'user_visible_ms': int(t_call * 1000),
        'request_id': resp._request_id,
    }


def _run_subprocess_mode(mode: str) -> dict:
    """Spawn a fresh Python process running this script with --mode=<mode>.

    Returns the parsed JSON line. A fresh process is essential — sharing
    a process across modes pollutes the cold case with the warm case's
    state.
    """
    proc = subprocess.run(
        [sys.executable, __file__, '--mode', mode],
        capture_output=True, text=True, env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f'{mode} subprocess failed (rc={proc.returncode}):\n'
            f'STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}'
        )
    # The subprocess prints JSON on the last non-empty line.
    last = next((l for l in reversed(proc.stdout.strip().split('\n')) if l), '')
    return json.loads(last)


def _print_comparison(cold_runs: list[dict], warm_runs: list[dict]) -> None:
    """Side-by-side. Median over runs to dampen network jitter."""

    def med(rs: list[dict], k: str) -> int:
        return int(statistics.median(r[k] for r in rs if k in r))

    print('\n' + '=' * 72)
    print('Results — median across runs (ms)')
    print('=' * 72)
    print(f"  Cold runs: {len(cold_runs)}    Warm runs: {len(warm_runs)}")
    print()
    print(f"  {'phase':<22}{'cold':>12}{'warm':>12}{'delta':>14}")
    print(f"  {'-'*22}{'-'*12:>12}{'-'*12:>12}{'-'*14:>14}")

    cols = [
        ('import anthropic', 'import_ms', True),
        ('Anthropic() ctor', 'client_ctor_ms', True),
        ('models.retrieve', 'models_retrieve_ms', False),
        ('warmup ping (1tok)', 'warmup_ping_ms', False),
        ('first user call', 'first_call_ms', True),
    ]
    for label, key, in_both in cols:
        cold_v = med(cold_runs, key) if in_both else None
        warm_v = med(warm_runs, key)
        cold_str = f'{cold_v:>10}ms' if cold_v is not None else f"{'—':>12}"
        warm_str = f'{warm_v:>10}ms'
        if cold_v is not None:
            delta = warm_v - cold_v
            sign = '+' if delta >= 0 else ''
            delta_str = f'{sign}{delta:>10}ms'
        else:
            delta_str = f"{'(boot)':>14}"
        print(f"  {label:<22}{cold_str:>12}{warm_str:>12}{delta_str:>14}")

    print()
    cold_uv = med(cold_runs, 'user_visible_ms')
    warm_uv = med(warm_runs, 'user_visible_ms')
    saved = cold_uv - warm_uv
    pct = (saved / cold_uv * 100) if cold_uv else 0.0
    print(f"  USER-VISIBLE LATENCY:  cold={cold_uv}ms  warm={warm_uv}ms  "
          f"saved={saved}ms ({pct:.0f}%)")
    print(f"  Boot cost (off user's path): "
          f"{med(warm_runs, 'boot_total_ms')}ms median")
    print('=' * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('cold', 'warm'), default=None,
                        help='If set, run one bench and emit JSON. Else, '
                             'run the full A/B (multiple subprocesses).')
    parser.add_argument('--runs', type=int, default=3,
                        help='Samples per mode for the A/B (default 3).')
    args = parser.parse_args()

    if args.mode == 'cold':
        print(json.dumps(_bench_cold()))
        return
    if args.mode == 'warm':
        print(json.dumps(_bench_warm()))
        return

    print(f"Spawning {args.runs} cold + {args.runs} warm subprocesses...")
    cold_runs, warm_runs = [], []
    # Interleave so transient network conditions affect both modes equally.
    for i in range(args.runs):
        print(f"  cold run {i+1}/{args.runs}...", flush=True)
        cold_runs.append(_run_subprocess_mode('cold'))
        print(f"  warm run {i+1}/{args.runs}...", flush=True)
        warm_runs.append(_run_subprocess_mode('warm'))

    _print_comparison(cold_runs, warm_runs)


if __name__ == '__main__':
    main()
