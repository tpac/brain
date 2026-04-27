"""Full broader eval suite — guards against single-benchmark optimization
by running diverse axes in parallel and aggregating metrics.

Three sub-suites, all isolated from the live brain:

  1. longmem_broad        30 items × 3 variance = 90 jobs
                          (5 items × 6 dataset categories)
  2. abstention_battery   5 synthesized abstention queries × 3 variance = 15 jobs
                          (entity-type diversity across pet/person/place/item)
  3. snapshot_replay      3 different real conversations replayed against the
                          Apr 19 prod snapshot (production-fidelity encoding)

Production safety:
  - All eval brains under ~/AgentsContext/brain-eval-{run_name}/...
  - Snapshot replays use snapshot copies; never touch live brain.db
  - ProcessPoolExecutor isolates env vars per worker
  - Daemon stays untouched

Usage:
  ./dev python3 eval/full_suite.py --run-name preflight_2026-04-26 \\
      --longmem-workers 25 --abstention-workers 10 \\
      --snapshot ~/AgentsContext/brain/brain.db.bak-pre-situation-migration
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────
# Picks — deterministic so reruns are comparable. Hand-picked to span
# entity types and conversation styles.
# ─────────────────────────────────────────────────────────────────────────

LONGMEM_BROAD_PICKS = {
    'temporal-reasoning': ['0bb5a684', '08f4fc43', '2c63a862', '2a1811e2', 'bbf86515'],
    'multi-session': ['0a995998', '6d550036', 'b5ef892d', 'e831120c', '3a704032'],
    'knowledge-update': ['6a1eabeb', '6aeb4375', '830ce83f', '852ce960', '945e3d21'],
    'single-session-preference': ['8a2466db', '06878be2', '75832dbd', '0edc2aef', '35a27287'],
    'single-session-assistant': ['7161e7e2', 'c4f10528', '89527b6b', 'e9327a54', '4c36ccef'],
    'single-session-user': ['e47becba', '118b2229', '51a45a95', '58bf7951', '1e043500'],
}

# Abstention battery — synthetic queries against 0862e8bf source (cat-care
# conversation). Each query asks about an entity NOT in the source so the
# brain must abstain — but ideally surfaces adjacent context (Luna). Spans
# pet/person/place/item axes so failure modes don't all collapse to "pet".
ABSTENTION_BATTERY = [
    {
        '_qid_suffix': 'abs_pet_hamster',
        'question': 'What is the name of my hamster?',
        'answer': "You did not mention this. You mentioned your cat Luna but not a hamster.",
    },
    {
        '_qid_suffix': 'abs_pet_dog',
        'question': "What's my dog's name?",
        'answer': "You did not mention a dog. You mentioned your cat Luna but no dog.",
    },
    {
        '_qid_suffix': 'abs_person_doctor',
        'question': 'What is my doctor\'s name?',
        'answer': "You did not mention a doctor. You mentioned your vet at the January 15 follow-up.",
    },
    {
        '_qid_suffix': 'abs_place_city',
        'question': 'What city did I visit recently?',
        'answer': "You did not mention any city visits. You discussed your cat Luna's care.",
    },
    {
        '_qid_suffix': 'abs_item_car',
        'question': 'What kind of car do I drive?',
        'answer': "You did not mention any car. You discussed your cat Luna's digestive and litter situation.",
    },
]

SNAPSHOT_REPLAY_CONVERSATIONS = [
    '71857713-2390-414d-9d51-1ef1de652d90',
    'eba17631-1caf-4f2c-a4ef-245a132f1862',
    'fd829e08-35b9-408b-9ea6-d50cc9e19aec',
]

DEFAULT_SNAPSHOT = os.path.expanduser(
    '~/AgentsContext/brain/brain.db.bak-pre-situation-migration')


# ─────────────────────────────────────────────────────────────────────────
# Build longmem items + abstention items
# ─────────────────────────────────────────────────────────────────────────


def _load_longmem_dataset() -> List[Dict[str, Any]]:
    return json.loads((ROOT / 'eval' / 'longmem' / 'data' /
                       'longmemeval_oracle.json').read_text())


def build_longmem_broad_items(variance: int) -> List[Dict[str, Any]]:
    dataset = _load_longmem_dataset()
    by_qid = {it['question_id']: it for it in dataset}
    items = []
    for axis, qids in LONGMEM_BROAD_PICKS.items():
        for qid in qids:
            base = by_qid.get(qid)
            if not base:
                print(f'[full_suite] WARN missing qid {qid} in dataset', flush=True)
                continue
            for v in range(variance):
                copy = dict(base)
                copy['_axis'] = axis
                copy['_variance_idx'] = v
                items.append(copy)
    return items


def build_abstention_items(variance: int) -> List[Dict[str, Any]]:
    dataset = _load_longmem_dataset()
    base = next(it for it in dataset if it['question_id'] == '0862e8bf')
    items = []
    for entry in ABSTENTION_BATTERY:
        for v in range(variance):
            copy = dict(base)
            copy['question_id'] = '0862e8bf_' + entry['_qid_suffix']
            copy['question'] = entry['question']
            copy['answer'] = entry['answer']
            copy['_axis'] = 'abstention'
            copy['_variance_idx'] = v
            items.append(copy)
    return items


# ─────────────────────────────────────────────────────────────────────────
# Pool worker — must be top-level for pickling
# ─────────────────────────────────────────────────────────────────────────


def _pool_worker(args):
    item, run_name = args
    try:
        from eval.s1s_full_e2e import run_item
        return run_item(item, run_name=run_name)
    except Exception as e:
        import traceback
        return {
            'qid': item.get('question_id', '?'),
            'axis': item.get('_axis', '?'),
            '_variance_idx': item.get('_variance_idx'),
            'error': str(e),
            'traceback': traceback.format_exc()[-2000:],
        }


def run_pool(items: List[Dict[str, Any]], run_name: str,
             workers: int, label: str) -> List[Dict[str, Any]]:
    print(f'\n[{label}] starting {len(items)} jobs with {workers} workers',
          flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_pool_worker, (it, run_name)): it for it in items}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            mark = '✓' if r.get('correct') else ('!' if r.get('error') else '✗')
            print(f'[{label}] {done:>3d}/{len(items)} {mark} '
                  f'{r.get("axis", "?"):26s} {r.get("qid", "?"):30s} '
                  f'{int(r.get("query_s1r_ms") or 0):>5d}ms', flush=True)
    elapsed = time.time() - t0
    correct = sum(1 for r in results if r.get('correct'))
    errored = sum(1 for r in results if r.get('error'))
    print(f'[{label}] done — {correct}/{len(items)} correct, {errored} errors '
          f'({elapsed:.1f}s)', flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────
# Snapshot replays — separate subprocesses so each loads its own embedder
# ─────────────────────────────────────────────────────────────────────────


def launch_snapshot_replays(snapshot: str, conversations: List[str],
                             run_name: str) -> List[subprocess.Popen]:
    procs = []
    for conv_id in conversations:
        conv_path = os.path.expanduser(
            f'~/.claude/projects/-Users-tpac-brain/{conv_id}.jsonl')
        if not os.path.exists(conv_path):
            print(f'[snapshot] WARN conversation not found: {conv_path}',
                  flush=True)
            continue
        sub_run_name = f'{run_name}__{conv_id[:8]}'
        log_path = f'/tmp/full_suite_{sub_run_name}.log'
        cmd = [
            './dev', 'python3', str(ROOT / 'eval' / 's1s_snapshot_replay.py'),
            '--snapshot', snapshot,
            '--conversation', conv_path,
            '--run-name', sub_run_name,
        ]
        print(f'[snapshot] launching: {conv_id[:8]} → {log_path}', flush=True)
        f = open(log_path, 'w')
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        procs.append((conv_id, p, log_path, sub_run_name))
    return procs


def collect_snapshot_replays(procs) -> List[Dict[str, Any]]:
    results = []
    for conv_id, p, log_path, sub_run_name in procs:
        rc = p.wait()
        summary_path = (ROOT / 'eval' / 'reports' / 'snapshot_replay' /
                        sub_run_name / 'summary.json')
        if rc != 0 or not summary_path.exists():
            results.append({
                'conv_id': conv_id, 'sub_run_name': sub_run_name,
                'rc': rc, 'log_path': log_path, 'summary': None,
                'error': f'replay failed (rc={rc}); see {log_path}',
            })
            print(f'[snapshot] FAIL {conv_id[:8]}: rc={rc}, log={log_path}',
                  flush=True)
            continue
        s = json.loads(summary_path.read_text())
        results.append({
            'conv_id': conv_id, 'sub_run_name': sub_run_name,
            'summary': s, 'log_path': log_path,
        })
        v14 = s.get('replay_v14') or {}
        v12 = s.get('production_v12') or {}
        print(f'[snapshot] DONE {conv_id[:8]}: v12 cycles={v12.get("total_cycles")} '
              f'vs v14 cycles={v14.get("total_cycles")} '
              f'(new_nodes={v14.get("new_nodes")}, edges={v14.get("new_edges")})',
              flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────


def aggregate(longmem_results, abstention_results, snapshot_results,
              report_dir: Path, run_name: str, total_elapsed: float):
    # Per-axis counts
    def axis_counts(results):
        counts = {}
        for r in results:
            ax = r.get('axis', '?')
            counts.setdefault(ax, {'total': 0, 'correct': 0, 'errored': 0,
                                    'no_context': 0})
            counts[ax]['total'] += 1
            if r.get('correct'):
                counts[ax]['correct'] += 1
            if r.get('error'):
                counts[ax]['errored'] += 1
            if not r.get('has_context'):
                counts[ax]['no_context'] += 1
        return counts

    longmem_axes = axis_counts(longmem_results)
    abstention_axes = axis_counts(abstention_results)

    summary = {
        'run_name': run_name,
        'total_elapsed_s': round(total_elapsed, 1),
        'longmem_broad': {
            'total_items': len(longmem_results),
            'total_correct': sum(1 for r in longmem_results if r.get('correct')),
            'by_axis': longmem_axes,
        },
        'abstention_battery': {
            'total_items': len(abstention_results),
            'total_correct': sum(1 for r in abstention_results if r.get('correct')),
            'by_axis': abstention_axes,
            'no_context_count': sum(1 for r in abstention_results if not r.get('has_context')),
        },
        'snapshot_replay': {
            'count': len(snapshot_results),
            'replays': [
                {
                    'conv_id': r['conv_id'],
                    'v12_cycles': (r.get('summary') or {}).get('production_v12', {}).get('total_cycles'),
                    'v14_cycles': (r.get('summary') or {}).get('replay_v14', {}).get('total_cycles'),
                    'v14_nodes': (r.get('summary') or {}).get('replay_v14', {}).get('new_nodes'),
                    'v14_edges': (r.get('summary') or {}).get('replay_v14', {}).get('new_edges'),
                    'error': r.get('error'),
                } for r in snapshot_results
            ],
        },
    }

    # Save full — but DON'T clobber files that aren't in the current run.
    # Bug caught 2026-04-27: an --skip-longmem rerun called aggregate() with
    # an empty longmem_results list, which overwrote the prior run's 90-item
    # longmem_results.jsonl with an empty file. Now: only write the .jsonl
    # for a phase if that phase actually has results in this invocation.
    # The summary.json/.md still reflect what THIS invocation produced.
    (report_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    if longmem_results:
        (report_dir / 'longmem_results.jsonl').write_text(
            '\n'.join(json.dumps(r, default=str) for r in longmem_results) + '\n')
    if abstention_results:
        (report_dir / 'abstention_results.jsonl').write_text(
            '\n'.join(json.dumps(r, default=str) for r in abstention_results) + '\n')

    # Markdown report
    lines = [f'# Full Suite Report — {run_name}', '',
             f'Total elapsed: {total_elapsed:.1f}s',
             f'Wall: {total_elapsed/60:.1f} min', '',
             '## Longmem broad', '',
             f"Total: {summary['longmem_broad']['total_correct']}/{summary['longmem_broad']['total_items']}",
             '', '| Axis | Correct | No-context | Errors |', '|---|---|---|---|']
    for ax, c in sorted(longmem_axes.items()):
        lines.append(f"| {ax} | {c['correct']}/{c['total']} | {c['no_context']} | {c['errored']} |")
    lines += ['', '## Abstention battery', '',
              f"Total: {summary['abstention_battery']['total_correct']}/{summary['abstention_battery']['total_items']}",
              f"No-context (brain surfaced nothing): {summary['abstention_battery']['no_context_count']}",
              '', '| Variation | Correct | No-context |', '|---|---|---|']
    # Per-variation breakdown
    by_qid = {}
    for r in abstention_results:
        qid = r.get('qid', '?').replace('0862e8bf_', '')
        by_qid.setdefault(qid, {'total': 0, 'correct': 0, 'no_context': 0})
        by_qid[qid]['total'] += 1
        if r.get('correct'): by_qid[qid]['correct'] += 1
        if not r.get('has_context'): by_qid[qid]['no_context'] += 1
    for qid, c in sorted(by_qid.items()):
        lines.append(f"| {qid} | {c['correct']}/{c['total']} | {c['no_context']} |")

    lines += ['', '## Snapshot replay', '',
              '| Conversation | v12 cycles | v14 cycles | v14 nodes | v14 edges |',
              '|---|---|---|---|---|']
    for r in summary['snapshot_replay']['replays']:
        cid = r['conv_id'][:8]
        v12c = r.get('v12_cycles') or '-'
        v14c = r.get('v14_cycles') or '-'
        nodes = r.get('v14_nodes') or '-'
        edges = r.get('v14_edges') or '-'
        err = r.get('error') or ''
        lines.append(f"| {cid} | {v12c} | {v14c} | {nodes} | {edges} | {err}")

    (report_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(f'\n[full_suite] reports written to {report_dir}', flush=True)


# ─────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default=time.strftime('full_%Y%m%d_%H%M%S'))
    parser.add_argument('--variance', type=int, default=3)
    # Worker defaults dropped from 25/10 to 15/5 after the 2026-04-25 run
    # hit Sonnet rate-limits with 25+10+3 concurrent. Empirically 15+5
    # serial holds steady. Override via flags if rate limits have grown.
    parser.add_argument('--longmem-workers', type=int, default=15)
    parser.add_argument('--abstention-workers', type=int, default=5)
    parser.add_argument('--snapshot', default=DEFAULT_SNAPSHOT)
    # Snapshot lane is OFF by default. It compares v12 (gone) against v14
    # (live), which is no longer a useful question. It was also the
    # heaviest contention contributor — it died in two of three runs on
    # 2026-04-25. Pass --enable-snapshot to opt in for archival comparison.
    parser.add_argument('--enable-snapshot', action='store_true',
                        help='Run snapshot replay lane (v12 baseline). '
                             'Off by default — heavy + low signal post-v14.')
    parser.add_argument('--skip-longmem', action='store_true')
    parser.add_argument('--skip-abstention', action='store_true')
    args = parser.parse_args()

    report_dir = ROOT / 'eval' / 'reports' / 'full_suite' / args.run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f'[full_suite] run_name: {args.run_name}', flush=True)
    print(f'[full_suite] reports → {report_dir}', flush=True)
    print(f'[full_suite] workers: longmem={args.longmem_workers} '
          f'abstention={args.abstention_workers} '
          f'snapshot={"on" if args.enable_snapshot else "off"}', flush=True)

    t_start = time.time()
    # Incremental aggregation — write summary.json/md after EACH phase so
    # a kill mid-run preserves whatever finished. Matches what's actually
    # there: the 2026-04-25 run had 90 longmem results in the log but
    # zero summary files because aggregation only ran at the end.
    longmem_results: List[Dict[str, Any]] = []
    abstention_results: List[Dict[str, Any]] = []
    snapshot_results: List[Dict[str, Any]] = []

    def _checkpoint(label: str):
        elapsed = time.time() - t_start
        try:
            aggregate(longmem_results, abstention_results, snapshot_results,
                      report_dir, args.run_name, elapsed)
            print(f'[full_suite] checkpoint after {label}: '
                  f'summary.json/md written ({elapsed/60:.1f} min in)',
                  flush=True)
        except Exception as e:
            # Aggregation failures must not lose finished work.
            print(f'[full_suite] checkpoint after {label} FAILED: {e}',
                  flush=True)

    # Phase 1: longmem broad. Heaviest, runs first so we get the
    # most-informative number even if killed mid-suite.
    if not args.skip_longmem:
        items = build_longmem_broad_items(args.variance)
        longmem_results = run_pool(items, args.run_name, args.longmem_workers,
                                    'longmem_broad')
        _checkpoint('longmem')

    # Phase 2: abstention battery — sequential after longmem to avoid
    # API contention. ~15 min, sized to recover even if interrupted.
    if not args.skip_abstention:
        items = build_abstention_items(args.variance)
        abstention_results = run_pool(items, args.run_name,
                                       args.abstention_workers, 'abstention')
        _checkpoint('abstention')

    # Phase 3 (opt-in): snapshot replays. Launch + collect SEQUENTIALLY,
    # not concurrently with the pools, since we know that hit rate-limits.
    if args.enable_snapshot:
        snapshot_procs = launch_snapshot_replays(
            args.snapshot, SNAPSHOT_REPLAY_CONVERSATIONS, args.run_name)
        if snapshot_procs:
            print('\n[full_suite] waiting for snapshot replays to finish...',
                  flush=True)
            snapshot_results = collect_snapshot_replays(snapshot_procs)
            _checkpoint('snapshot')

    total_elapsed = time.time() - t_start
    # Final write — same as the last checkpoint, but with definitive elapsed.
    aggregate(longmem_results, abstention_results, snapshot_results,
              report_dir, args.run_name, total_elapsed)

    # Print final
    print('\n' + '=' * 70)
    print(f'FULL SUITE — {args.run_name} ({total_elapsed/60:.1f} min)')
    print('=' * 70)
    if longmem_results:
        c = sum(1 for r in longmem_results if r.get('correct'))
        print(f'Longmem broad:        {c}/{len(longmem_results)}')
    if abstention_results:
        c = sum(1 for r in abstention_results if r.get('correct'))
        nc = sum(1 for r in abstention_results if not r.get('has_context'))
        print(f'Abstention battery:   {c}/{len(abstention_results)} '
              f'({nc} with no-context — Haiku rejected adjacent)')
    if snapshot_results:
        ok = sum(1 for r in snapshot_results if not r.get('error'))
        print(f'Snapshot replays:     {ok}/{len(snapshot_results)} succeeded')
    print(f'\nReport: {report_dir}/summary.md')


if __name__ == '__main__':
    main()
