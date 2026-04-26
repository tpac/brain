"""Full end-to-end test: v14 prompt + muster + S2 + recall + answerer.

Purpose (Tom's ask, session end 2026-04-24):
  - Real end-to-end run with the full S1+S2 pipeline
  - "Resolving with agent the recalls — like it should be"
  - Detailed internal capture so we can analyze across sessions

What this does differently from eval/s1s_ab_wiring_check.py:
  - Uses eval/longmem/replay.py for the full pipeline (S1R + S0 + S1E
    + S2 at every 2 encodings + final S2 flush + backfill_vectors)
  - Registers v14 prompt in each fresh brain before replay
  - Muster runs unconditionally (architectural default since the v13 ship)
  - Single arm (v14 + muster), no A/B — the question this answers is
    "how does the full new stack perform", not "is B better than A"
  - Preserves ALL internals (brain DB, logs DB, traces, preserved
    brain dirs under ~/AgentsContext/brain-eval-{run_name}/{qid}/)

Usage:
    ./dev python3 eval/s1s_full_e2e.py                  # 5 axes × 3 runs serial
    ./dev python3 eval/s1s_full_e2e.py --items 5        # 5 items only
    ./dev python3 eval/s1s_full_e2e.py --run-name foo   # named run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Items per axis — deterministic picks, same as wiring_check
LONGMEM_PICKS = [
    ("info_extraction",   "6f9b354f"),
    ("multi_session",     "edced276"),
    ("temporal",          "5e1b23de"),
    ("knowledge_update",  "10e09553"),
    ("abstention",        "0862e8bf_abs"),
]


def _load_env():
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                k, v = k.strip(), v.strip()
                if v:
                    os.environ[k] = v


def run_item(item: Dict[str, Any], run_name: str, arm: str = 'B') -> Dict[str, Any]:
    """Full e2e on one longmem item. Preserves brain at
    ~/AgentsContext/brain-eval-{run_name}/{qid}-r{variance_idx}/."""
    from eval.longmem.fresh_brain import create_fresh_eval_brain, per_item_brain_dir
    from eval.longmem.replay import replay_item, query_brain
    from eval.longmem.answerer import answer_question
    from eval.longmem.judge import judge_one
    from eval.longmem.classifier import classify_failure
    from eval.s1s_v13_prompt import extract_v13_prompt
    from servers.daemon_dispatch import COMMAND_TABLE

    qid = item['question_id']
    axis = item.get('_axis', 'unknown')
    variance_idx = item.get('_variance_idx')
    prefix = f'[{axis}:{qid}-r{variance_idx}]' if variance_idx is not None else f'[{axis}:{qid}]'
    brain_dir = per_item_brain_dir(qid, run_name=run_name, variance_idx=variance_idx)
    brain = create_fresh_eval_brain(path=brain_dir, wipe=True)

    def dispatch(cmd, args=None):
        entry = COMMAND_TABLE.get(cmd)
        if not entry:
            return {"ok": False, "error": f"unknown: {cmd}"}
        return entry.handler(brain, args or {}, [])

    # Arm B: register v14 + enable muster
    registered_version = None
    if arm == 'B':
        v14 = extract_v13_prompt()
        reg = dispatch('register_interaction', {
            'name': 's1e', 'template': v14,
            'parameters': '', 'created_by': 'full_e2e',
        })
        if not reg.get('ok'):
            raise RuntimeError(f'{prefix} register_interaction failed: {reg}')
        registered_version = (reg.get('result') or {}).get('version')

    err_before = brain.logs_conn.execute(
        "SELECT COUNT(*) FROM debug_log WHERE event_type='error'").fetchone()[0]

    # Drive full S1+S2 pipeline
    t0 = time.time()
    session_id = f'full_e2e-{qid}'
    ingest_stats = replay_item(
        brain, session_id, item['haystack_sessions'],
        haystack_dates=item.get('haystack_dates'),
        log_prefix=prefix)
    ingest_ms = int((time.time() - t0) * 1000)

    # Query + answer + judge
    q_result = query_brain(brain, item['question'], item.get('question_date'))
    a_result = answer_question(item['question'], q_result['additional_context'],
                               item.get('question_date'))
    j_result = judge_one(item['question'], item['answer'], a_result['hypothesis'])

    # Classification for diagnostics
    failure_info = {}
    if not j_result['correct']:
        failure_info = classify_failure(
            brain, item['question'], item['answer'], a_result['hypothesis'],
            q_result['query_session_id'], a_result['has_context'],
            a_result['abstained'])

    # Capture detailed internal state
    err_after = brain.logs_conn.execute(
        "SELECT COUNT(*) FROM debug_log WHERE event_type='error'").fetchone()[0]
    n_new_errors = err_after - err_before

    # Count node/edge volumes
    n_nodes = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE archived=0 "
        "AND encoding_source NOT IN ('anchor:seed','hook:compaction','hook:integrity')"
    ).fetchone()[0]
    n_edges = brain.conn.execute(
        "SELECT COUNT(*) FROM edges e "
        "JOIN edge_relations er ON er.edge_id = e.edge_id "
        "WHERE er.encoding_source NOT LIKE 'anchor:%' "
        "AND er.encoding_source NOT LIKE 'hook:%'"
    ).fetchone()[0]

    # Count scout events
    scout_events = brain.logs_conn.execute(
        "SELECT COUNT(*) FROM trace_events WHERE ref_type IN "
        "('scout_input','scout_findings')").fetchone()[0]

    # Close brain (keep dir) + force GC. Each item creates a new Brain with
    # its own SQLite conns, vector cache, trace buffers; without explicit
    # cleanup the parent process accumulates unbounded across longmem items
    # (per-item brains + their cached vectors stay reachable until cycle
    # collector runs). ProcessPoolExecutor isolates by process so this only
    # matters when the harness runs serial / inside another worker.
    try:
        brain.save()
        brain.close()
    except Exception:
        pass
    del brain
    import gc
    gc.collect()

    return {
        'qid': qid,
        'axis': axis,
        'arm': arm,
        'registered_version': registered_version,
        'question': item['question'],
        'gold': str(item['answer'])[:400],
        'hypothesis': a_result['hypothesis'][:500],
        'correct': j_result['correct'],
        'abstained': a_result['abstained'],
        'has_context': a_result['has_context'],
        'judge_raw': j_result['raw'][:200],
        'ingest_stats': ingest_stats,
        'ingest_ms': ingest_ms,
        'query_s1r_ms': q_result['s1r_ms'],
        'additional_context_chars': len(q_result.get('additional_context') or ''),
        'answer_ms': a_result['elapsed_ms'],
        'answer_tokens_in': a_result.get('tokens_in', 0),
        'answer_tokens_out': a_result.get('tokens_out', 0),
        'n_nodes_created': n_nodes,
        'n_edges_created': n_edges,
        'n_scout_events': scout_events,
        'n_new_errors': n_new_errors,
        'brain_dir': brain_dir,
        **failure_info,
    }


def _load_items(n_per_axis: int) -> List[Dict[str, Any]]:
    data = json.loads((ROOT / "eval" / "longmem" / "data" /
                       "longmemeval_oracle.json").read_text(encoding="utf-8"))
    items = []
    for axis, qid in LONGMEM_PICKS:
        for rec in data:
            if rec['question_id'] == qid:
                for idx in range(n_per_axis):
                    copy = dict(rec)
                    copy['_axis'] = axis
                    copy['_variance_idx'] = idx
                    items.append(copy)
                break
    return items


def main():
    parser = argparse.ArgumentParser(description='Full E2E — v14 + muster + S2')
    parser.add_argument('--n-per-axis', type=int, default=3,
                        help='variance runs per axis (default 3)')
    parser.add_argument('--items', type=int, default=None,
                        help='cap total items (for debugging)')
    parser.add_argument('--run-name', default=None,
                        help='run name (default timestamp)')
    parser.add_argument('--workers', type=int, default=6,
                        help='Parallel worker count (default 6)')
    parser.add_argument('--arm', default='B', choices=['A', 'B'],
                        help='A = no muster + prod v12 prompt; B = v14 + muster')
    args = parser.parse_args()

    _load_env()

    # Muster is now architecturally unconditional in run_encoding(); the
    # `--arm` flag only controls which prompt gets registered (v12 default
    # vs v14+SPLIT). Arm A no longer disables scouts — that distinction
    # would require plumbing a muster_enabled kwarg through replay_item.

    run_name = args.run_name or f'full_e2e_{time.strftime("%Y%m%d_%H%M%S")}'
    reports_dir = ROOT / 'eval' / 'reports' / 's1s_full_e2e'
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_dir = reports_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    items = _load_items(args.n_per_axis)
    if args.items:
        items = items[:args.items]

    print(f'[e2e] run_name: {run_name}')
    print(f'[e2e] arm: {args.arm}  (muster is now architecturally unconditional)')
    print(f'[e2e] items: {len(items)} ({args.n_per_axis} per axis × '
          f'{len(set((i["_axis"]) for i in items))} axes)')
    print(f'[e2e] brains: ~/AgentsContext/brain-eval-{run_name}/{{qid}}/')
    print(f'[e2e] reports: {run_dir}')
    print()

    results: List[Dict[str, Any]] = []
    t_start = time.time()
    if args.workers <= 1 or len(items) == 1:
        for i, item in enumerate(items):
            print(f'[e2e] === item {i+1}/{len(items)}: '
                  f'{item["_axis"]} / {item["question_id"]} ===')
            try:
                r = run_item(item, run_name, arm=args.arm)
                mark = '✓' if r['correct'] else '✗'
                print(f'[e2e] {mark}  nodes={r["n_nodes_created"]} '
                      f'edges={r["n_edges_created"]} scout_events={r["n_scout_events"]} '
                      f'errors={r["n_new_errors"]} '
                      f'query_ctx={r["additional_context_chars"]}chars')
                print(f'[e2e]    H: {r["hypothesis"][:120]}')
                results.append(r)
            except Exception as e:
                tb = traceback.format_exc()
                print(f'[e2e] FAILED: {e}')
                results.append({
                    'qid': item['question_id'], 'axis': item['_axis'],
                    'error': str(e), 'traceback': tb[-2000:],
                })
            (run_dir / 'results.jsonl').write_text(
                '\n'.join(json.dumps(r, default=str) for r in results) + '\n',
                encoding='utf-8')
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f'[e2e] running {len(items)} items across {args.workers} workers')
        print(f'[e2e] muster runs unconditionally in each worker via encode.run_encoding()')
        by_idx: Dict[int, Dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_item, item, run_name, args.arm): i
                for i, item in enumerate(items)
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                item = items[i]
                try:
                    r = fut.result()
                    by_idx[i] = r
                    mark = '✓' if r.get('correct') else '✗'
                    print(f'[e2e] {mark} {done+1}/{len(items)}: '
                          f'{r["axis"]} / {r["qid"]}  nodes={r.get("n_nodes_created","?")} '
                          f'scout_events={r.get("n_scout_events","?")} '
                          f'errors={r.get("n_new_errors","?")}', flush=True)
                except Exception as e:
                    tb = traceback.format_exc()
                    by_idx[i] = {
                        'qid': item['question_id'], 'axis': item['_axis'],
                        'error': str(e), 'traceback': tb[-2000:],
                    }
                    print(f'[e2e] ✗ {done+1}/{len(items)}: {item["question_id"]} FAILED: {e}',
                          flush=True)
                done += 1
                # Incremental save
                ordered = [by_idx[k] for k in sorted(by_idx.keys())]
                (run_dir / 'results.jsonl').write_text(
                    '\n'.join(json.dumps(r, default=str) for r in ordered) + '\n',
                    encoding='utf-8')
        results = [by_idx[k] for k in sorted(by_idx.keys())]

    total_s = int(time.time() - t_start)
    print()
    print(f'[e2e] all done in {total_s}s ({total_s//60}m {total_s%60}s)')

    # Final summary
    graded = [r for r in results if 'correct' in r]
    if graded:
        from collections import defaultdict
        by_axis = defaultdict(lambda: {'n': 0, 'correct': 0})
        for r in graded:
            by_axis[r['axis']]['n'] += 1
            if r['correct']:
                by_axis[r['axis']]['correct'] += 1
        total_correct = sum(d['correct'] for d in by_axis.values())
        total_n = sum(d['n'] for d in by_axis.values())

        print()
        print('=== FULL E2E RESULTS ===')
        print(f'{"AXIS":<20} {"CORRECT":>10}')
        print('-' * 32)
        for axis in ['info_extraction', 'knowledge_update', 'temporal',
                     'multi_session', 'abstention']:
            d = by_axis.get(axis, {'n': 0, 'correct': 0})
            if d['n']:
                print(f'{axis:<20}  {d["correct"]}/{d["n"]}')
        print('-' * 32)
        print(f'{"TOTAL":<20}  {total_correct}/{total_n} '
              f'({100*total_correct/max(total_n,1):.0f}%)')

    (run_dir / 'results.json').write_text(
        json.dumps({
            'run_name': run_name,
            'arm': args.arm,
            'elapsed_s': total_s,
            'n_items': len(results),
            'results': results,
        }, indent=2, default=str), encoding='utf-8')

    print(f'\n[e2e] results: {run_dir}/results.json')
    print(f'[e2e] per-item brains preserved at brain_dir in each result')


if __name__ == '__main__':
    main()
