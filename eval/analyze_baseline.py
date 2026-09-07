"""Analyze a baseline_v*.5 run — surface failure patterns for fix prioritization.

Built after the 2026-04-27 baseline showed multi-session 13%, single-session-
preference 27% — both regressions vs the broken Apr 25 run. The flat
`longmem_results.jsonl` lists 90 items but doesn't help you SEE the patterns.
This script produces two views, both written under the run's report dir:

  failures_by_axis.md  — for each weak axis, every failed item with
                          question/gold/hypothesis side-by-side, sorted
                          to make pattern detection eyeball-fast
  passes_vs_fails.md   — for each axis, contrast the items that passed
                          against the items that failed: nodes created,
                          edges created, context length

Usage:
    ./dev python3 eval/analyze_baseline.py baseline_v9.5
    ./dev python3 eval/analyze_baseline.py baseline_v9.5 --axis multi-session

Outputs go to eval/reports/full_suite/<run_name>/analysis/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_results(run_dir: Path) -> list:
    """Load the per-item results from longmem_results.jsonl."""
    path = run_dir / 'longmem_results.jsonl'
    if not path.exists():
        raise SystemExit(f'No results file at {path}')
    items = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _by_axis(items: list) -> dict:
    """Group items by axis, sorted: failures first, then passes."""
    out: dict = defaultdict(list)
    for it in items:
        out[it.get('axis', 'unknown')].append(it)
    for axis in out:
        # Failures first (correct=False), then passes (correct=True)
        out[axis].sort(key=lambda x: (bool(x.get('correct')), x.get('qid', '')))
    return dict(out)


def _truncate(s: str, n: int = 400) -> str:
    if not s:
        return '(empty)'
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + '…'


def render_failures_by_axis(items: list, only_axis: str | None = None) -> str:
    """For each axis, dump every failed item with question/gold/hypothesis."""
    grouped = _by_axis(items)
    lines = ['# Failures by axis', '',
             f'Total items: {len(items)}',
             f'Total failures: {sum(1 for x in items if not x.get("correct"))}',
             '']

    # Summary table
    lines += ['## Summary', '',
              '| Axis | Pass | Fail | Rate |',
              '|---|---:|---:|---:|']
    for axis in sorted(grouped):
        if only_axis and axis != only_axis:
            continue
        pass_n = sum(1 for x in grouped[axis] if x.get('correct'))
        fail_n = sum(1 for x in grouped[axis] if not x.get('correct'))
        total = pass_n + fail_n
        rate = (100 * pass_n / total) if total else 0
        lines.append(f'| {axis} | {pass_n} | {fail_n} | {rate:.0f}% |')
    lines.append('')

    for axis in sorted(grouped):
        if only_axis and axis != only_axis:
            continue
        fails = [x for x in grouped[axis] if not x.get('correct')]
        if not fails:
            continue
        lines += [f'## {axis}: {len(fails)} failures', '']
        for it in fails:
            qid = it.get('qid', '?')
            var = it.get('_variance_idx')
            tag = f'{qid}-r{var}' if var is not None else qid
            bucket = it.get('failure_bucket') or '(unbucketed)'
            lines += [
                f'### {tag} — bucket: `{bucket}`',
                '',
                '**Question:**',
                f'> {_truncate(it.get("question", ""), 600)}',
                '',
                '**Gold answer:**',
                f'> {_truncate(it.get("gold", ""), 600)}',
                '',
                '**Hypothesis:**',
                f'> {_truncate(it.get("hypothesis", ""), 600)}',
                '',
                '**Failure reason (judge):**',
                f'> {_truncate(it.get("failure_reason", "") or it.get("judge_raw", ""), 400)}',
                '',
                f'- nodes_created={it.get("n_nodes_created")} '
                f'edges_created={it.get("n_edges_created")} '
                f'errors={it.get("n_new_errors")}',
                f'- has_context={it.get("has_context")} '
                f'abstained={it.get("abstained")} '
                f'context_chars={it.get("additional_context_chars")}',
                f'- brain_dir: `{it.get("brain_dir")}`',
                '',
                '---',
                '',
            ]
    return '\n'.join(lines)


def render_passes_vs_fails(items: list) -> str:
    """For each axis, compare passes and fails on quantitative dimensions."""
    grouped = _by_axis(items)
    lines = ['# Passes vs Fails — quantitative comparison', '',
             'Helps identify whether failures correlate with specific '
             'patterns in encoding/recall behavior.',
             '']
    lines += ['| Axis | Status | N | nodes/item | edges/item | ctx_chars | s1r_ms |',
              '|---|---|---:|---:|---:|---:|---:|']
    for axis in sorted(grouped):
        for status_label, status_filter in [('pass', True), ('fail', False)]:
            cohort = [x for x in grouped[axis] if bool(x.get('correct')) == status_filter]
            if not cohort:
                continue
            n = len(cohort)
            avg_nodes = sum(x.get('n_nodes_created', 0) or 0 for x in cohort) / n
            avg_edges = sum(x.get('n_edges_created', 0) or 0 for x in cohort) / n
            avg_ctx = sum(x.get('additional_context_chars', 0) or 0 for x in cohort) / n
            avg_s1r = sum(x.get('query_s1r_ms', 0) or 0 for x in cohort) / n
            lines.append(
                f'| {axis} | {status_label} | {n} | {avg_nodes:.1f} | '
                f'{avg_edges:.1f} | {avg_ctx:.0f} | '
                f'{avg_s1r:.0f} |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('run_name',
                        help='Run directory name under eval/reports/full_suite/')
    parser.add_argument('--axis', default=None,
                        help='Focus on a single axis (e.g. multi-session)')
    args = parser.parse_args()

    run_dir = ROOT / 'eval' / 'reports' / 'full_suite' / args.run_name
    if not run_dir.is_dir():
        raise SystemExit(f'No run dir at {run_dir}')

    items = _load_results(run_dir)
    out_dir = run_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[analyze] {len(items)} items in {args.run_name}', flush=True)

    print('[analyze] writing failures_by_axis.md...', flush=True)
    (out_dir / 'failures_by_axis.md').write_text(
        render_failures_by_axis(items, only_axis=args.axis), encoding='utf-8')

    print('[analyze] writing passes_vs_fails.md...', flush=True)
    (out_dir / 'passes_vs_fails.md').write_text(
        render_passes_vs_fails(items), encoding='utf-8')

    print(f'[analyze] done. Reports in {out_dir}/', flush=True)
    for name in ['failures_by_axis.md', 'passes_vs_fails.md']:
        p = out_dir / name
        if p.exists():
            print(f'  {p}  ({p.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
