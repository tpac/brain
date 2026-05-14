"""Cost + latency summary for an eval run.

Walks the eval log and per-item artifacts to compute:

  - Per-item token usage by component (encoder/scouts/surface/answerer)
  - Per-item dollar cost using model rates
  - Per-item wall time + latency breakdown
  - Cohort aggregates + percentiles

Two arms can be compared by running this on each run name and diffing.

USE
    ./dev python3 eval/longmem/cost_summary.py ab_armA_v14_v4_<TS> \\
        --out eval/longmem/reports/<dir>/cost_armA.md
"""
import argparse
import json
import re
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─── Model rates (per million tokens, USD) — 2026-05 pricing ───────────────

RATES = {
    'sonnet': {'in': 3.0, 'out': 15.0, 'cache_read': 0.30, 'cache_write': 3.75},
    'haiku':  {'in': 0.80, 'out': 4.0, 'cache_read': 0.08, 'cache_write': 1.0},
}

# What model each component uses (matches the runtime config)
COMPONENT_MODEL = {
    'encoder': 'sonnet',         # claude-sonnet-4-6
    'scout_temporal': None,      # algorithmic, no LLM
    'scout_quote': 'haiku',      # claude-haiku-4-5
    'scout_facts': 'haiku',
    'scout_synthesis': 'haiku',
    'surface': 'haiku',
    'answerer': 'haiku',
    'judge': 'sonnet',           # the longmem judge (post-answer correctness check)
}


# ─── Encoder log line parser ──────────────────────────────────────────────

RE_S1E_TOKENS = re.compile(
    r'\[s1e\] Rounds: (?P<rounds>\d+) \| Actions: (?P<actions>\d+) '
    r'\(writes: (?P<writes>\d+), reads: (?P<reads>\d+)\) \| '
    r'Tokens: (?P<fresh>\d+) fresh / (?P<cread>\d+) cached-read / '
    r'(?P<cwrite>\d+) cached-write / (?P<out>\d+) out'
    r'(?: \| hit=(?P<hit>\d+)%)?'
    r'(?: \| Profile: (?P<profile>[^\n]+))?'
)

RE_ITEM_START = re.compile(r'\[harness\] item (?P<n>\d+)/\d+ qid=(?P<qid>\S+)')
RE_S1E_DONE = re.compile(r'\[s1e\] done\. (?P<rounds>\d+) rounds, (?P<actions>\d+) actions\. PROFILE: (?P<prof>[^\n]+)')


def _cost_sonnet_or_haiku(fresh: int, out: int, cache_read: int, cache_write: int,
                           model: str) -> float:
    """Compute USD cost from a single LLM call's token usage."""
    r = RATES[model]
    return (
        (fresh * r['in']) / 1_000_000 +
        (out * r['out']) / 1_000_000 +
        (cache_read * r['cache_read']) / 1_000_000 +
        (cache_write * r['cache_write']) / 1_000_000
    )


def parse_eval_log(log_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Walk the eval stdout log and group per-item s1e firings.

    Returns: {qid: [{rounds, actions, fresh, cached_read, cached_write, out,
                     hit, latencies_ms: [...]}, ...]}
    """
    by_qid: Dict[str, List[Dict[str, Any]]] = {}
    current_qid: Optional[str] = None
    if not log_path.exists():
        return by_qid
    for line in log_path.read_text().splitlines():
        m_start = RE_ITEM_START.search(line)
        if m_start:
            current_qid = m_start.group('qid')
            by_qid.setdefault(current_qid, [])
            continue
        m_tok = RE_S1E_TOKENS.search(line)
        if m_tok and current_qid:
            row = {
                'rounds': int(m_tok.group('rounds')),
                'actions': int(m_tok.group('actions')),
                'writes': int(m_tok.group('writes')),
                'reads': int(m_tok.group('reads')),
                'fresh_in': int(m_tok.group('fresh')),
                'cached_read': int(m_tok.group('cread')),
                'cached_write': int(m_tok.group('cwrite')),
                'tokens_out': int(m_tok.group('out')),
                'hit_pct': int(m_tok.group('hit') or 0),
                'profile': m_tok.group('profile') or '',
            }
            # Parse per-round latencies from profile
            row['llm_ms'] = sum(
                int(x) for x in re.findall(r'llm_r\d+=(\d+)ms', row['profile'])
            )
            by_qid[current_qid].append(row)
    return by_qid


# ─── Per-item rollup ──────────────────────────────────────────────────────

def per_item_costs(run_name: str, encoder_by_qid: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Compose cost + latency for every completed item."""
    reports_root = ROOT / 'eval' / 'longmem' / 'reports' / run_name
    items_dir = reports_root / 'items'
    out: List[Dict[str, Any]] = []
    if not items_dir.exists():
        return out
    for qid in sorted(os.listdir(items_dir)):
        d = items_dir / qid
        result_path = d / 'result.json'
        if not result_path.exists():
            continue  # in-flight
        result = json.loads(result_path.read_text())
        meta = json.loads((d / 'meta.json').read_text()) if (d / 'meta.json').exists() else {}
        recall = json.loads((d / 'recall.json').read_text()) if (d / 'recall.json').exists() else {}
        ans = recall.get('answerer_response') or {}

        # Encoder rollup
        enc_calls = encoder_by_qid.get(qid, [])
        enc = {
            'calls': len(enc_calls),
            'rounds': sum(c['rounds'] for c in enc_calls),
            'actions': sum(c['actions'] for c in enc_calls),
            'fresh_in': sum(c['fresh_in'] for c in enc_calls),
            'cached_read': sum(c['cached_read'] for c in enc_calls),
            'cached_write': sum(c['cached_write'] for c in enc_calls),
            'tokens_out': sum(c['tokens_out'] for c in enc_calls),
            'llm_ms': sum(c['llm_ms'] for c in enc_calls),
        }
        enc['cost'] = _cost_sonnet_or_haiku(
            enc['fresh_in'], enc['tokens_out'],
            enc['cached_read'], enc['cached_write'], 'sonnet')

        # Answerer (Haiku — fresh-only billing)
        ans_in = ans.get('tokens_in', 0)
        ans_out = ans.get('tokens_out', 0)
        ans_cost = _cost_sonnet_or_haiku(ans_in, ans_out, 0, 0, 'haiku')
        ans_ms = ans.get('elapsed_ms', 0)

        # Total wall (per the harness result)
        total_ms = result.get('total_item_ms') or (
            result.get('ingest_ms', 0) + result.get('s1r_ms', 0) + ans_ms)

        out.append({
            'qid': qid,
            'axis': meta.get('axis') or result.get('axis'),
            'correct': bool(result.get('correct')),
            'bucket': result.get('failure_bucket'),
            'enc_calls': enc['calls'],
            'enc_rounds': enc['rounds'],
            'enc_actions': enc['actions'],
            'enc_fresh_in': enc['fresh_in'],
            'enc_cached_read': enc['cached_read'],
            'enc_cached_write': enc['cached_write'],
            'enc_tokens_out': enc['tokens_out'],
            'enc_llm_ms': enc['llm_ms'],
            'enc_cost_usd': enc['cost'],
            'ans_tokens_in': ans_in,
            'ans_tokens_out': ans_out,
            'ans_cost_usd': ans_cost,
            'ans_ms': ans_ms,
            'total_item_ms': total_ms,
            'item_total_cost_usd': enc['cost'] + ans_cost,
        })
    return out


# ─── Cohort aggregate + percentiles ───────────────────────────────────────

def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def cohort_aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return {
        'n_items': len(rows),
        'pass_count': sum(1 for r in rows if r['correct']),
        'pass_rate': sum(1 for r in rows if r['correct']) / len(rows),
        'total_cost_usd': sum(r['item_total_cost_usd'] for r in rows),
        'enc_cost_usd': sum(r['enc_cost_usd'] for r in rows),
        'ans_cost_usd': sum(r['ans_cost_usd'] for r in rows),
        'enc_fresh_in': sum(r['enc_fresh_in'] for r in rows),
        'enc_tokens_out': sum(r['enc_tokens_out'] for r in rows),
        'enc_cached_read': sum(r['enc_cached_read'] for r in rows),
        'enc_calls': sum(r['enc_calls'] for r in rows),
        'enc_actions': sum(r['enc_actions'] for r in rows),
        'p50_item_ms': _percentile([r['total_item_ms'] for r in rows], 50),
        'p90_item_ms': _percentile([r['total_item_ms'] for r in rows], 90),
        'max_item_ms': max(r['total_item_ms'] for r in rows),
        'p50_enc_ms_per_call': _percentile(
            [r['enc_llm_ms'] / r['enc_calls'] for r in rows if r['enc_calls']], 50),
        'p90_enc_ms_per_call': _percentile(
            [r['enc_llm_ms'] / r['enc_calls'] for r in rows if r['enc_calls']], 90),
    }


# ─── Reporting ────────────────────────────────────────────────────────────

def render_md(run_name: str, rows: List[Dict[str, Any]], agg: Dict[str, Any]) -> str:
    lines = [f'# Cost + perf summary — `{run_name}`', '']
    lines.append(f'**Items completed:** {agg["n_items"]}')
    lines.append(f'**Pass rate:** {agg["pass_count"]}/{agg["n_items"]} ({agg["pass_rate"]:.1%})')
    lines.append(f'**Total cohort cost:** ${agg["total_cost_usd"]:.2f} '
                 f'(encoder ${agg["enc_cost_usd"]:.2f} + answerer ${agg["ans_cost_usd"]:.2f})')
    lines.append('')

    lines.append('## Cohort aggregate')
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---:|')
    lines.append(f'| Encoder calls (sum) | {agg["enc_calls"]} |')
    lines.append(f'| Encoder actions (sum) | {agg["enc_actions"]} |')
    lines.append(f'| Encoder fresh-in tokens | {agg["enc_fresh_in"]:,} |')
    lines.append(f'| Encoder cached-read tokens | {agg["enc_cached_read"]:,} |')
    lines.append(f'| Encoder output tokens | {agg["enc_tokens_out"]:,} |')
    lines.append(f'| Total wall — p50 / p90 / max | '
                 f'{agg["p50_item_ms"]/1000:.1f}s / {agg["p90_item_ms"]/1000:.1f}s / '
                 f'{agg["max_item_ms"]/1000:.1f}s |')
    lines.append(f'| Encoder per-call — p50 / p90 | '
                 f'{agg["p50_enc_ms_per_call"]/1000:.1f}s / '
                 f'{agg["p90_enc_ms_per_call"]/1000:.1f}s |')
    lines.append('')

    lines.append('## Per-item')
    lines.append('')
    lines.append('| qid | axis | ✓/✗ | enc calls | enc tok (in/out) | enc cost | total wall | item cost |')
    lines.append('|---|---|:---:|---:|---|---:|---:|---:|')
    for r in sorted(rows, key=lambda x: (x.get('axis') or '', x['qid'])):
        mark = '✓' if r['correct'] else '✗'
        toks = f"{r['enc_fresh_in']:,}/{r['enc_tokens_out']:,}"
        lines.append(
            f"| `{r['qid']}` | {r['axis'] or '-'} | {mark} | "
            f"{r['enc_calls']} | {toks} | ${r['enc_cost_usd']:.3f} | "
            f"{r['total_item_ms']/1000:.1f}s | ${r['item_total_cost_usd']:.3f} |"
        )
    lines.append('')
    return '\n'.join(lines)


# ─── Main ────────────────────────────────────────────────────────────────

def summarize(run_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    log_path = ROOT / 'eval' / 'longmem' / 'reports' / '_compare_logs' / f'{run_name}.log'
    encoder_by_qid = parse_eval_log(log_path)
    rows = per_item_costs(run_name, encoder_by_qid)
    agg = cohort_aggregate(rows)
    return rows, agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_names', nargs='+', help='one or more run names to summarize')
    p.add_argument('--out', default=None,
                   help='markdown output path (if multiple runs, suffix with run name)')
    args = p.parse_args()

    for run_name in args.run_names:
        rows, agg = summarize(run_name)
        if not rows:
            print(f'[cost] {run_name}: no completed items', flush=True)
            continue
        md = render_md(run_name, rows, agg)
        if args.out:
            out_path = Path(args.out)
            if len(args.run_names) > 1:
                stem = out_path.stem + '_' + run_name
                out_path = out_path.with_name(stem + out_path.suffix)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md)
            print(f'wrote {out_path}', flush=True)
        else:
            print(md)


if __name__ == '__main__':
    main()
