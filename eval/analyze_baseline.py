"""Analyze a baseline_v*.5 run — surface failure patterns for fix prioritization.

Built after the 2026-04-27 baseline showed multi-session 13%, single-session-
preference 27% — both regressions vs the broken Apr 25 run. The flat
`longmem_results.jsonl` lists 90 items but doesn't help you SEE the patterns.
This script produces three views, all written under the run's report dir:

  failures_by_axis.md  — for each weak axis, every failed item with
                          question/gold/hypothesis side-by-side, sorted
                          to make pattern detection eyeball-fast
  scout_inspector.md   — for failed items, what did the scouts (quote /
                          temporal / facts) actually produce? Reads each
                          item's per-item brain_logs.db trace events for
                          'scout_findings' rows. Tells you whether the
                          regression is "scouts wrong" vs "encoder
                          ignored correct scouts"
  passes_vs_fails.md   — for each axis, contrast the items that passed
                          against the items that failed: nodes created,
                          edges created, scout events, context length

Usage:
    ./dev python3 eval/analyze_baseline.py baseline_v9.5
    ./dev python3 eval/analyze_baseline.py baseline_v9.5 --axis multi-session
    ./dev python3 eval/analyze_baseline.py baseline_v9.5 --only-failures

Outputs go to eval/reports/full_suite/<run_name>/analysis/.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
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
                f'scout_events={it.get("n_scout_events")} '
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


def _read_scout_findings(brain_dir: str) -> list:
    """Pull scout_findings trace events from a per-item brain_logs.db.

    Returns list of dicts: {scout, summary, metadata}.
    """
    db = Path(brain_dir) / 'brain_logs.db'
    if not db.exists():
        return []
    out = []
    try:
        c = sqlite3.connect(str(db))
        # Get the encoding chain's scout findings rows.
        rows = c.execute("""
            SELECT chain_id, scale, event_type, ref_type, summary, metadata, created_at
            FROM trace_events
            WHERE ref_type IN ('scout_findings', 'scout_input')
            ORDER BY created_at
        """).fetchall()
        for chain_id, scale, evt, ref_type, summary, metadata, created_at in rows:
            try:
                meta = json.loads(metadata) if metadata else {}
            except Exception:
                meta = {}
            out.append({
                'chain_id': chain_id,
                'event_type': evt,
                'ref_type': ref_type,
                'summary': summary or '',
                'scout_name': meta.get('scout_name') or meta.get('scout'),
                'metadata': meta,
            })
        c.close()
    except Exception as e:
        out.append({'error': f'Failed to read {db}: {e}'})
    return out


def render_scout_inspector(items: list, only_axis: str | None = None,
                            only_failures: bool = True) -> str:
    """For each (failed) item, show what each scout reported."""
    grouped = _by_axis(items)
    lines = ['# Scout inspector', '',
             'For each item, what did the muster scouts (quote / temporal / '
             'facts) actually produce?',
             'Tells us whether failures are "scouts wrong" or "encoder '
             'ignored correct scouts".',
             '']

    for axis in sorted(grouped):
        if only_axis and axis != only_axis:
            continue
        targets = grouped[axis]
        if only_failures:
            targets = [x for x in targets if not x.get('correct')]
        if not targets:
            continue
        lines += [f'## {axis}', '']
        for it in targets:
            qid = it.get('qid', '?')
            var = it.get('_variance_idx')
            tag = f'{qid}-r{var}' if var is not None else qid
            mark = '✓' if it.get('correct') else '✗'
            lines += [f'### {mark} {tag}', '',
                      f'Q: {_truncate(it.get("question", ""), 200)}',
                      '']
            findings = _read_scout_findings(it.get('brain_dir', ''))
            if not findings:
                lines.append('(no scout traces — brain_dir missing or no scouts ran)')
                lines.append('')
                continue
            # Group by chain_id (one chain per encoding cycle)
            by_chain: dict = defaultdict(list)
            for f in findings:
                by_chain[f.get('chain_id', '?')].append(f)
            for chain_id in sorted(by_chain):
                lines.append(f'**chain `{chain_id}`:**')
                for f in by_chain[chain_id]:
                    name = f.get('scout_name') or '?'
                    rt = f.get('ref_type', '?')
                    summary = _truncate(f.get('summary', ''), 250)
                    lines.append(f'- [{rt}] {name}: {summary}')
                lines.append('')
            lines.append('---')
            lines.append('')
    return '\n'.join(lines)


def render_passes_vs_fails(items: list) -> str:
    """For each axis, compare passes and fails on quantitative dimensions."""
    grouped = _by_axis(items)
    lines = ['# Passes vs Fails — quantitative comparison', '',
             'Helps identify whether failures correlate with specific '
             'patterns in encoding/recall behavior.',
             '']
    lines += ['| Axis | Status | N | nodes/item | edges/item | scouts/item | ctx_chars | s1r_ms |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for axis in sorted(grouped):
        for status_label, status_filter in [('pass', True), ('fail', False)]:
            cohort = [x for x in grouped[axis] if bool(x.get('correct')) == status_filter]
            if not cohort:
                continue
            n = len(cohort)
            avg_nodes = sum(x.get('n_nodes_created', 0) or 0 for x in cohort) / n
            avg_edges = sum(x.get('n_edges_created', 0) or 0 for x in cohort) / n
            avg_scouts = sum(x.get('n_scout_events', 0) or 0 for x in cohort) / n
            avg_ctx = sum(x.get('additional_context_chars', 0) or 0 for x in cohort) / n
            avg_s1r = sum(x.get('query_s1r_ms', 0) or 0 for x in cohort) / n
            lines.append(
                f'| {axis} | {status_label} | {n} | {avg_nodes:.1f} | '
                f'{avg_edges:.1f} | {avg_scouts:.1f} | {avg_ctx:.0f} | '
                f'{avg_s1r:.0f} |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('run_name',
                        help='Run directory name under eval/reports/full_suite/')
    parser.add_argument('--axis', default=None,
                        help='Focus on a single axis (e.g. multi-session)')
    parser.add_argument('--only-failures', action='store_true', default=True,
                        help='Scout inspector only shows failed items (default true)')
    parser.add_argument('--all-items', action='store_true',
                        help='Scout inspector shows passes too (overrides --only-failures)')
    args = parser.parse_args()

    run_dir = ROOT / 'eval' / 'reports' / 'full_suite' / args.run_name
    if not run_dir.is_dir():
        raise SystemExit(f'No run dir at {run_dir}')

    items = _load_results(run_dir)
    out_dir = run_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    only_failures = not args.all_items

    print(f'[analyze] {len(items)} items in {args.run_name}', flush=True)

    print('[analyze] writing failures_by_axis.md...', flush=True)
    (out_dir / 'failures_by_axis.md').write_text(
        render_failures_by_axis(items, only_axis=args.axis), encoding='utf-8')

    print('[analyze] writing scout_inspector.md...', flush=True)
    (out_dir / 'scout_inspector.md').write_text(
        render_scout_inspector(items, only_axis=args.axis,
                                only_failures=only_failures),
        encoding='utf-8')

    print('[analyze] writing passes_vs_fails.md...', flush=True)
    (out_dir / 'passes_vs_fails.md').write_text(
        render_passes_vs_fails(items), encoding='utf-8')

    print(f'[analyze] done. Reports in {out_dir}/', flush=True)
    for name in ['failures_by_axis.md', 'scout_inspector.md', 'passes_vs_fails.md']:
        p = out_dir / name
        if p.exists():
            print(f'  {p}  ({p.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
