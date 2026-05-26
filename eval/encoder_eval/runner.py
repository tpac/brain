"""runner — CLI entry for staged multi-version encoder eval.

Usage:
    ./dev python3 -m eval.encoder_eval.runner \\
        --versions 19,21,22 \\
        --corpus realchat \\
        --stages 0-0,1-2,3-5,6-9,10-14 \\
        --run-name v22_gate_$(date +%Y%m%d_%H%M)

  --versions       comma-separated s1e versions (must be already registered)
  --corpus         realchat | longmem
  --stages         comma-separated inclusive Python slices "i-j[,k-l,...]"
                   Each stage is one checkpoint boundary.
  --run-name       output directory under eval/encoder_eval/reports/
  --skip-probes    optional comma-separated probe names to skip
  --continue-on-stop  don't halt when a stop condition fires
  --dry-run        materialize templates + print plan, run nothing
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_versions(arg: str) -> List[int]:
    out = []
    for tok in arg.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"--versions tokens must be integers; got {tok!r}")
        out.append(int(tok))
    if not out:
        raise ValueError("--versions cannot be empty")
    return out


def _parse_stages(arg: str, n_items_available: int) -> List[Dict[str, Any]]:
    """Parse "0-0,1-2,3-5" → [{name: 'stage_A', indices: [0]},
                                {name: 'stage_B', indices: [1,2]}, ...]"""
    stages = []
    letters = 'ABCDEFGHIJKLMNO'
    for i, tok in enumerate(arg.split(',')):
        tok = tok.strip()
        if not tok:
            continue
        if '-' not in tok:
            raise ValueError(f"stage range must be 'i-j'; got {tok!r}")
        lo, hi = tok.split('-', 1)
        lo, hi = int(lo), int(hi)
        if lo > hi or lo < 0 or hi >= n_items_available:
            raise ValueError(
                f"stage range {tok!r} out of bounds for n_items={n_items_available}")
        idx = list(range(lo, hi + 1))
        name = f"stage_{letters[i] if i < len(letters) else i}"
        stages.append({'name': name, 'indices': idx, 'range': (lo, hi)})
    return stages


def _load_corpus(corpus_name: str) -> List[Dict[str, Any]]:
    if corpus_name == 'realchat':
        path = ROOT / 'eval' / 'longmem' / 'data' / 'realchat_oracle.json'
    elif corpus_name == 'longmem':
        path = ROOT / 'eval' / 'longmem' / 'data' / 'longmemeval_oracle.json'
    else:
        raise ValueError(f"unknown corpus: {corpus_name}")
    with open(path) as f:
        return json.load(f)


def _normalize_item(item: Dict[str, Any], idx: int,
                     corpus_name: str) -> Dict[str, Any]:
    """Some corpora are missing question_id — synthesize one for tracking."""
    if not item.get('question_id'):
        # realchat oracle has no question_id; synthesize from index + question hash
        import hashlib
        h = hashlib.md5(item.get('question', '').encode()).hexdigest()[:8]
        item['question_id'] = f"{corpus_name}_{idx:03d}_{h}"
    if not item.get('question_type') and not item['question_id'].endswith('_abs'):
        item['question_type'] = 'single-session-user'  # fallback
    return item


def main():
    parser = argparse.ArgumentParser(description='Multi-version encoder quality eval')
    parser.add_argument('--versions', required=True,
                        help='Comma-separated s1e versions, e.g. "19,21,22"')
    parser.add_argument('--corpus', default='realchat',
                        choices=['realchat', 'longmem'])
    parser.add_argument('--stages', default='0-0',
                        help='Comma-separated inclusive item ranges, e.g. "0-0,1-2". '
                             'Ignored when --stratify is set.')
    parser.add_argument('--stratify', type=int, default=0, metavar='PER_AXIS',
                        help='Stratified sample mode: pick N items per axis. '
                             'Overrides --stages. Each axis becomes its own '
                             'stage so checkpoint stops fire at axis boundaries.')
    parser.add_argument('--stratify-seed', type=int, default=42,
                        help='RNG seed for stratified sampling (reproducible).')
    parser.add_argument('--run-name', default=None,
                        help='Output directory name under eval/encoder_eval/reports/')
    parser.add_argument('--skip-probes', default='',
                        help='Comma-separated probe names to skip')
    parser.add_argument('--continue-on-stop', action='store_true',
                        help='Do not halt when stop conditions fire')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the plan; do not run')
    args = parser.parse_args()

    versions = _parse_versions(args.versions)
    raw_items = _load_corpus(args.corpus)
    items = [_normalize_item(it, i, args.corpus) for i, it in enumerate(raw_items)]

    if args.stratify > 0:
        from eval.longmem.harness import _item_axis
        import random
        per_axis = args.stratify
        rng = random.Random(args.stratify_seed)
        # Group by axis, prefer smaller items (faster encoding)
        by_axis: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            ax = _item_axis(it)
            by_axis.setdefault(ax, []).append(it)
        stages_spec = []
        letters = 'ABCDEFGHIJKLMN'
        for i, axis in enumerate(sorted(by_axis.keys())):
            pool = by_axis[axis]
            # Sort by total turn count ascending, take bottom-half-shuffled
            pool_sorted = sorted(pool, key=lambda it: sum(
                len(s) for s in (it.get('haystack_sessions') or [])))
            pool_bot = pool_sorted[: max(per_axis * 4, len(pool_sorted) // 2)]
            rng.shuffle(pool_bot)
            picked = pool_bot[:per_axis]
            if not picked:
                continue
            stages_spec.append({
                'name': f"stage_{letters[i] if i < len(letters) else i}_{axis}",
                'indices': [items.index(p) for p in picked],
                'range': (axis, len(picked)),
            })
    else:
        stages_spec = _parse_stages(args.stages, len(items))

    run_name = args.run_name or f"encoder_eval_{datetime.utcnow():%Y%m%d_%H%M%S}"
    out_dir = ROOT / 'eval' / 'encoder_eval' / 'reports' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_probes = [s.strip() for s in args.skip_probes.split(',') if s.strip()]

    # ── Plan summary ────────────────────────────────────────────
    print(f"\n[encoder_eval] PLAN", file=sys.stderr)
    print(f"  versions  : {versions}", file=sys.stderr)
    print(f"  corpus    : {args.corpus} ({len(items)} items total)",
          file=sys.stderr)
    print(f"  stages    : {len(stages_spec)}", file=sys.stderr)
    for s in stages_spec:
        r = s['range']
        if isinstance(r[0], int):
            scope = f"items[{r[0]}..{r[1]}]"
        else:
            scope = f"axis={r[0]} (n={r[1]})"
        print(f"    {s['name']}: {scope} ({len(s['indices'])} items)",
              file=sys.stderr)
    print(f"  out_dir   : {out_dir}", file=sys.stderr)
    if skip_probes:
        print(f"  skip_probes: {skip_probes}", file=sys.stderr)
    print(f"  versions × items = {len(versions) * sum(len(s['indices']) for s in stages_spec)} cells",
          file=sys.stderr)

    if args.dry_run:
        print("[encoder_eval] dry-run; exiting before any encoding.",
              file=sys.stderr)
        return 0

    # Write config
    config = {
        'run_name': run_name,
        'started_at': datetime.utcnow().isoformat() + 'Z',
        'versions': versions,
        'corpus': args.corpus,
        'stages': [{'name': s['name'], 'range': s['range'],
                    'indices': s['indices']} for s in stages_spec],
        'skip_probes': skip_probes,
        'continue_on_stop': args.continue_on_stop,
        'argv': sys.argv,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Materialize stage payloads
    stage_payloads = [
        {'name': s['name'], 'items': [items[i] for i in s['indices']]}
        for s in stages_spec
    ]

    # Run
    from eval.encoder_eval.harness import run_staged
    t0 = time.time()
    summary = run_staged(
        versions=versions, stages=stage_payloads, run_name=run_name,
        out_dir=out_dir, skip_probes=skip_probes,
        continue_on_stop=args.continue_on_stop)
    wall_ms = int((time.time() - t0) * 1000)
    summary['wall_ms'] = wall_ms
    summary['ended_at'] = datetime.utcnow().isoformat() + 'Z'

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Final report
    from eval.encoder_eval.report import render_report
    report_path = out_dir / 'final_report.md'
    render_report(out_dir / 'per_cell.jsonl', report_path)

    print(f"\n[encoder_eval] DONE — {summary['n_cells']} cells, "
          f"{wall_ms/1000:.1f}s", file=sys.stderr)
    print(f"[encoder_eval] report: {report_path}", file=sys.stderr)
    if summary['halted']:
        print(f"[encoder_eval] HALTED — see halt_reasons in summary.json",
              file=sys.stderr)
        return 2  # distinct exit code for halt
    return 0


if __name__ == '__main__':
    sys.exit(main())
