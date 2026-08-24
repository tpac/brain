"""Consolidated A/B comparison across two eval arms.

Runs run_diff (verdict + behavioral signals) + structural_diff (encoder
graph shape) + cost_summary (tokens, latency, dollar cost) on both arms,
diffs the per-arm cost summaries, and emits one SUMMARY.md that links to
the detailed reports.

This is the report that answers "should we ship the new bundle":
  - verdict (pass rate, per-axis, failure buckets)
  - encoding quality (event_time emission, facts, edges)
  - performance (latency p50/p90)
  - cost ($ per item, cohort total)
  - decision-criteria pass/fail per the ship rubric

USE
    ./dev python3 eval/longmem/compare_arms.py \\
        ab_armA_v14_v4_<TS>  ab_armB_v15_11_v5_<TS> \\
        --labels v14+v4,v15.11+v5_agentic \\
        --out-dir eval/longmem/reports/ab_compare_<TS>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.longmem.cost_summary import summarize as cost_summarize


def _pct(a: float, b: float) -> str:
    """Render a delta as percentage if both nonzero, else delta."""
    if not a:
        return f'+∞' if b else '0'
    delta_pct = ((b - a) / a) * 100
    return f'{delta_pct:+.1f}%'


def _delta_str(a: float, b: float, unit: str = '') -> str:
    return f'{b - a:+.2f}{unit}'


def render_summary(run_a: str, run_b: str, label_a: str, label_b: str,
                    rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]],
                    agg_a: Dict[str, Any], agg_b: Dict[str, Any],
                    out_dir: Path) -> str:
    lines = [f'# A/B comparison — `{label_a}` vs `{label_b}`', '']
    lines.append(f'**Arm A:** `{run_a}` ({label_a})')
    lines.append(f'**Arm B:** `{run_b}` ({label_b})')
    lines.append('')
    lines.append('Detailed reports:')
    lines.append(f'- [run_diff.md](run_diff.md) — verdict + behavioral signals + per-item movement')
    lines.append(f'- [structural_diff.md](structural_diff.md) — encoder graph structure')
    lines.append(f'- [cost_armA.md](cost_armA.md) / [cost_armB.md](cost_armB.md) — token + cost + latency')
    lines.append('')

    # Headline
    lines.append('## Headline')
    lines.append('')
    lines.append('| | A | B | Δ |')
    lines.append('|---|---:|---:|---:|')
    pa, pb = agg_a.get('pass_rate', 0), agg_b.get('pass_rate', 0)
    pna, pnb = agg_a.get('pass_count', 0), agg_b.get('pass_count', 0)
    ca, cb = agg_a.get('total_cost_usd', 0), agg_b.get('total_cost_usd', 0)
    p50a, p50b = agg_a.get('p50_item_ms', 0), agg_b.get('p50_item_ms', 0)
    p90a, p90b = agg_a.get('p90_item_ms', 0), agg_b.get('p90_item_ms', 0)
    n_a = agg_a.get('n_items', 0)
    n_b = agg_b.get('n_items', 0)
    lines.append(f'| Items completed | {n_a} | {n_b} | {n_b - n_a:+d} |')
    lines.append(f'| Pass rate | {pa:.1%} ({pna}/{n_a}) | {pb:.1%} ({pnb}/{n_b}) | {(pb-pa)*100:+.1f}pp |')
    # A zero on the A side means UNMEASURED (see the rubric guard below) —
    # render it as such here too, or the reader meets a confident $0.00
    # two sections before the caveat.
    _ca = f'${ca:.2f}' if ca else 'unmeasured'
    _dc = f'${cb-ca:+.2f} ({_pct(ca, cb)})' if ca else '—'
    lines.append(f'| Total cohort cost (USD) | {_ca} | ${cb:.2f} | {_dc} |')
    _p50a = f'{p50a/1000:.1f}' if p50a else 'unmeasured'
    _d50 = f'{(p50b-p50a)/1000:+.1f}s ({_pct(p50a, p50b)})' if p50a else '—'
    lines.append(f'| Wall time p50 (s) | {_p50a} | {p50b/1000:.1f} | {_d50} |')
    _p90a = f'{p90a/1000:.1f}' if p90a else 'unmeasured'
    _d90 = f'{(p90b-p90a)/1000:+.1f}s ({_pct(p90a, p90b)})' if p90a else '—'
    lines.append(f'| Wall time p90 (s) | {_p90a} | {p90b/1000:.1f} | {_d90} |')
    lines.append('')

    # Cost decomposition
    lines.append('## Cost decomposition')
    lines.append('')
    lines.append('| Component | A | B | Δ |')
    lines.append('|---|---:|---:|---:|')
    for key, label in [('enc_cost_usd', 'Encoder cost'),
                        ('ans_cost_usd', 'Answerer cost')]:
        va, vb = agg_a.get(key, 0), agg_b.get(key, 0)
        lines.append(f'| {label} | ${va:.3f} | ${vb:.3f} | ${vb-va:+.3f} |')
    for key, label in [('enc_calls', 'Encoder calls'),
                        ('enc_actions', 'Encoder write actions'),
                        ('enc_fresh_in', 'Encoder fresh-in tokens'),
                        ('enc_tokens_out', 'Encoder output tokens')]:
        va, vb = agg_a.get(key, 0), agg_b.get(key, 0)
        lines.append(f'| {label} | {va:,} | {vb:,} | {vb-va:+,} |')
    lines.append('')

    # Per-item side by side (sorted by qid)
    by_qid_a = {r['qid']: r for r in rows_a}
    by_qid_b = {r['qid']: r for r in rows_b}
    common = sorted(set(by_qid_a) & set(by_qid_b))
    lines.append('## Per-item side by side')
    lines.append('')
    lines.append('| qid | axis | A ✓/✗ | B ✓/✗ | A cost | B cost | A wall | B wall |')
    lines.append('|---|---|:---:|:---:|---:|---:|---:|---:|')
    for qid in common:
        a, b = by_qid_a[qid], by_qid_b[qid]
        a_m = '✓' if a['correct'] else '✗'
        b_m = '✓' if b['correct'] else '✗'
        lines.append(
            f"| `{qid}` | {a.get('axis') or '-'} | {a_m} | {b_m} | "
            f"${a['item_total_cost_usd']:.3f} | ${b['item_total_cost_usd']:.3f} | "
            f"{a['total_item_ms']/1000:.1f}s | {b['total_item_ms']/1000:.1f}s |"
        )
    lines.append('')

    # Decision criteria (the rubric)
    lines.append('## Decision criteria')
    lines.append('')
    lines.append('| Criterion | Threshold | A | B | Pass? |')
    lines.append('|---|---|---:|---:|:---:|')
    # 1. Aggregate pass rate
    pass_rate_ok = (pb - pa) >= -0.05  # B no worse than A by >5pp
    lines.append(f'| Aggregate pass rate | B ≥ A − 5pp | {pa:.1%} | {pb:.1%} | '
                 f'{"✓" if pass_rate_ok else "✗"} |')
    # 4. Cost — a zero A cost means UNMEASURED (e.g. no eval log parsed),
    # not free: render the row as unmeasured instead of a vacuous pass.
    if ca:
        cost_ok = cb <= 2 * ca
        lines.append(f'| Cohort cost | B ≤ 2× A | ${ca:.2f} | ${cb:.2f} | '
                     f'{"✓" if cost_ok else "✗"} |')
    else:
        lines.append(f'| Cohort cost | B ≤ 2× A | unmeasured | ${cb:.2f} | ? |')
    # 5. Latency — same rule.
    if p90a:
        latency_ok = p90b <= 1.5 * p90a
        lines.append(f'| p90 latency | B ≤ 1.5× A p90 | {p90a/1000:.1f}s | '
                     f'{p90b/1000:.1f}s | {"✓" if latency_ok else "✗"} |')
    else:
        lines.append(f'| p90 latency | B ≤ 1.5× A p90 | unmeasured | '
                     f'{p90b/1000:.1f}s | ? |')
    lines.append('')
    lines.append('(Per-axis + encoder-structure criteria are in '
                 '[run_diff.md](run_diff.md) and [structural_diff.md](structural_diff.md).)')
    lines.append('')

    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_a', help='Arm A run name (baseline)')
    p.add_argument('run_b', help='Arm B run name (candidate)')
    p.add_argument('--labels', default='A,B', help='comma-separated A,B labels')
    p.add_argument('--out-dir', required=True, help='directory to write reports into')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_a, label_b = [s.strip() for s in args.labels.split(',')[:2]] or ['A', 'B']

    # 1. cost_summary per arm
    rows_a, agg_a = cost_summarize(args.run_a)
    rows_b, agg_b = cost_summarize(args.run_b)
    (out_dir / 'cost_armA.md').write_text(
        __import__('eval.longmem.cost_summary', fromlist=['render_md']).render_md(args.run_a, rows_a, agg_a))
    (out_dir / 'cost_armB.md').write_text(
        __import__('eval.longmem.cost_summary', fromlist=['render_md']).render_md(args.run_b, rows_b, agg_b))
    print(f'wrote {out_dir/"cost_armA.md"} + cost_armB.md', flush=True)

    # 2. run_diff (verdict + behavioral)
    diff_path = out_dir / 'run_diff.md'
    rc = subprocess.run([
        './dev', 'python3', 'eval/longmem/run_diff.py',
        args.run_a, args.run_b, '--out', str(diff_path)
    ], cwd=str(ROOT), check=False)
    if rc.returncode == 0:
        print(f'wrote {diff_path}', flush=True)
    else:
        print(f'WARN: run_diff failed rc={rc.returncode}', flush=True)

    # 3. structural_diff (encoder graph shape)
    struct_path = out_dir / 'structural_diff.md'
    rc = subprocess.run([
        './dev', 'python3', 'eval/longmem/structural_diff.py',
        args.run_a, args.run_b, '--labels', args.labels,
        '--out', str(struct_path)
    ], cwd=str(ROOT), check=False)
    if rc.returncode == 0:
        print(f'wrote {struct_path}', flush=True)

    # 4. SUMMARY.md (top-level consolidated)
    summary = render_summary(args.run_a, args.run_b, label_a, label_b,
                              rows_a, rows_b, agg_a, agg_b, out_dir)
    summary_path = out_dir / 'SUMMARY.md'
    summary_path.write_text(summary)
    print(f'wrote {summary_path}', flush=True)


if __name__ == '__main__':
    main()
