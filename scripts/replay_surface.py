"""Replay the S1 surface Haiku call standalone, with full timing visibility.

Why this exists: the hook_recall PhaseTimer just measured 40 seconds inside
the `surface` phase on a single user prompt. The surface phase wraps a SINGLE
Haiku messages.create() call (no tool loop). Either:
  (a) Haiku itself returned the response slowly,
  (b) the SDK retried internally (default max_retries=2),
  (c) max_tokens=600 was hit and the model spent its whole budget,
  (d) something is wrong with the request shape.

Two reconstruction modes:

  --judge-result PATH    Apples-to-apples. Reads the EXACT prompt the daemon
                         sent from the dashboard file written by surface.py
                         (`/tmp/brain-judge-result-{recall_ref}.json`). The
                         file contains `surface_prompt` = system + '---' +
                         user_content, so we can split and replay byte-
                         identical to what the daemon issued.

  --candidates PATH      Approximation. Rebuilds the user_content via
                         build_surface_prompt from the candidates file.
                         Misses ctx.get_frame(brain), which is built fresh
                         per call — produces a smaller payload than what
                         the daemon actually sends. Use only when the
                         dashboard file is missing.

Default: latest dashboard file (judge-result mode).

Run:
    ./dev python3 scripts/replay_surface.py
        # latest dashboard file — apples-to-apples replay

    ./dev python3 scripts/replay_surface.py --judge-result PATH
        # specific dashboard file (e.g. for the 46s recall)

    ./dev python3 scripts/replay_surface.py --stream
        # also do a streaming run for TTFT and per-token timing

    ./dev python3 scripts/replay_surface.py --no-retry
        # Anthropic(max_retries=0) so SDK retries surface as errors
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.dispatch import load_env  # noqa: E402
load_env()

import anthropic  # noqa: E402
from servers.scales.s1.surface_contract import (  # noqa: E402
    SURFACE_MODEL, build_surface_prompt,
)


def _latest_candidates_file() -> str:
    paths = glob.glob('/tmp/brain-*-recall-candidates.json')
    paths = [p for p in paths if os.path.getsize(p) > 1000]  # skip stub files
    if not paths:
        raise SystemExit('No candidates file found in /tmp/')
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]


def _latest_judge_result_file() -> str:
    paths = glob.glob('/tmp/brain-judge-result-*.json')
    paths = [p for p in paths if os.path.getsize(p) > 1000]  # skip stub files
    if not paths:
        raise SystemExit('No judge-result file found in /tmp/')
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]


def _build_request_from_judge_result(path: str) -> tuple[str, str, int, dict]:
    """Read the dashboard surface-result file and split system/user.

    surface.py:_write_surface_result_file writes the exact prompt sent to
    Haiku as `surface_prompt = system + '\\n\\n---\\n\\n' + user_content`.
    Splitting on the literal '\\n\\n---\\n\\n' separator gives us byte-
    identical inputs to what the daemon issued. max_tokens isn't stored
    in the file — pull from SURFACE['max_tokens'] (the contract default).
    """
    from servers.scales.s1.surface_contract import SURFACE
    with open(path) as f:
        d = json.load(f)
    surface_prompt = d.get('surface_prompt', '')
    if not surface_prompt:
        raise SystemExit(f'{path}: missing surface_prompt key')

    sep = '\n\n---\n\n'
    if sep not in surface_prompt:
        raise SystemExit(
            f'{path}: surface_prompt has no system/user separator. '
            'May be from before the system-block split.')
    system_block, user_content = surface_prompt.split(sep, 1)

    debug = {
        'judge_result_file': path,
        'recall_ref': d.get('recall_ref'),
        'system_chars': len(system_block),
        'user_chars': len(user_content),
        'output_chars_in_file': len(d.get('surface_output', '') or ''),
        'max_tokens': SURFACE['max_tokens'],
    }
    return system_block, user_content, SURFACE['max_tokens'], debug


def _load_surface_template() -> str:
    logs_db = os.path.join(
        os.environ.get('BRAIN_DB_DIR') or
        os.path.expanduser('~/AgentsContext/brain'),
        'brain_logs.db')
    con = sqlite3.connect(f'file:{logs_db}?mode=ro', uri=True)
    cur = con.cursor()
    row = cur.execute(
        "SELECT template FROM interactions WHERE name='surface' "
        "ORDER BY version DESC LIMIT 1").fetchone()
    con.close()
    if not row or not row[0]:
        raise SystemExit("No 'surface' interaction in brain_logs.db")
    return row[0]


def _build_request(candidates_file: str) -> tuple[str, str, int, dict]:
    """Returns (system_block, user_content, max_tokens, debug)."""
    with open(candidates_file) as f:
        d = json.load(f)

    template = _load_surface_template()
    user_content, max_tokens = build_surface_prompt(
        d['candidates'], d.get('user_message', ''),
        recent_messages=d.get('recent_messages', []),
        recently_recalled=[],
        retrieval_stats=None,
        intent=None,
        frame=d.get('session_context', ''))

    debug = {
        'candidates_file': candidates_file,
        'n_candidates': len(d.get('candidates', [])),
        'user_message_len': len(d.get('user_message', '') or ''),
        'frame_len': len(d.get('session_context', '') or ''),
        'recent_messages': len(d.get('recent_messages', [])),
        'system_chars': len(template),
        'user_chars': len(user_content),
        'max_tokens': max_tokens,
    }
    return template, user_content, max_tokens, debug


def _run_blocking(client, system, user, max_tokens):
    print('\n--- BLOCKING run ---')
    t0 = time.monotonic()
    resp = client.messages.create(
        model=SURFACE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{'role': 'user', 'content': user}])
    wall = time.monotonic() - t0
    u = resp.usage
    print(f'  wall:                  {wall*1000:8.0f} ms')
    print(f'  input_tokens:          {u.input_tokens:8d}')
    print(f'  cache_read_tokens:     {getattr(u, "cache_read_input_tokens", 0):8d}')
    print(f'  cache_creation_tokens: {getattr(u, "cache_creation_input_tokens", 0):8d}')
    print(f'  output_tokens:         {u.output_tokens:8d}'
          f'  (max_tokens={max_tokens})')
    print(f'  stop_reason:           {resp.stop_reason}')
    print(f'  request_id:            {resp._request_id}')
    if u.output_tokens >= max_tokens - 5:
        print('  ⚠ HIT MAX_TOKENS — response truncated. '
              'Output time is dominated by the model running out '
              'its full budget.')
    print(f'  output preview: {resp.content[0].text[:240]!r}')
    return wall, resp


def _run_streaming(client, system, user, max_tokens):
    print('\n--- STREAMING run ---')
    t0 = time.monotonic()
    ttft = None
    n_text_deltas = 0
    n_chars = 0
    last_chunk = t0
    chunk_intervals = []
    final_message = None
    with client.messages.stream(
        model=SURFACE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{'role': 'user', 'content': user}],
    ) as stream:
        for event in stream:
            now = time.monotonic()
            if event.type == 'content_block_delta':
                if ttft is None:
                    ttft = now - t0
                if hasattr(event.delta, 'text'):
                    n_text_deltas += 1
                    n_chars += len(event.delta.text or '')
                    chunk_intervals.append(now - last_chunk)
                    last_chunk = now
        final_message = stream.get_final_message()
    wall = time.monotonic() - t0
    u = final_message.usage
    print(f'  wall:                  {wall*1000:8.0f} ms')
    print(f'  TTFT:                  {(ttft or 0)*1000:8.0f} ms')
    print(f'  text deltas:           {n_text_deltas:8d}')
    print(f'  output chars streamed: {n_chars:8d}')
    if chunk_intervals:
        avg = sum(chunk_intervals) / len(chunk_intervals)
        peak = max(chunk_intervals)
        print(f'  avg inter-chunk gap:   {avg*1000:8.1f} ms')
        print(f'  max inter-chunk gap:   {peak*1000:8.1f} ms'
              + ('  ⚠ stall' if peak > 1.0 else ''))
    print(f'  output_tokens:         {u.output_tokens:8d}'
          f'  (max_tokens={max_tokens})')
    print(f'  stop_reason:           {final_message.stop_reason}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge-result', default=None,
                        help='Path to dashboard /tmp/brain-judge-result-*.json. '
                             'Default: latest. Replays byte-identical input.')
    parser.add_argument('--candidates', default=None,
                        help='Approximation mode: rebuild from candidates file. '
                             'Misses ctx.get_frame(brain). Use only when '
                             'judge-result file is missing.')
    parser.add_argument('--stream', action='store_true',
                        help='Also do a streaming run (reports TTFT + chunk gaps).')
    parser.add_argument('--no-retry', action='store_true',
                        help='max_retries=0 so SDK retries become errors '
                             '(useful to detect retry storms).')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of blocking runs (default 1).')
    args = parser.parse_args()

    if args.candidates:
        system, user, max_tokens, debug = _build_request(args.candidates)
        debug['mode'] = 'candidates (approximation — missing real Frame)'
    else:
        path = args.judge_result or _latest_judge_result_file()
        system, user, max_tokens, debug = _build_request_from_judge_result(path)
        debug['mode'] = 'judge-result (apples-to-apples)'

    print('=' * 72)
    print('Reconstructed surface request')
    print('=' * 72)
    for k, v in debug.items():
        print(f'  {k}: {v}')

    kwargs = {}
    if args.no_retry:
        kwargs['max_retries'] = 0
        print('  max_retries=0  (SDK retries will surface as errors)')
    client = anthropic.Anthropic(**kwargs)

    for i in range(args.runs):
        if args.runs > 1:
            print(f'\n=== run {i+1}/{args.runs} ===')
        _run_blocking(client, system, user, max_tokens)

    if args.stream:
        _run_streaming(client, system, user, max_tokens)


if __name__ == '__main__':
    main()
