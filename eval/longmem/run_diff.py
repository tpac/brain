"""Side-by-side diff of two eval runs — same items, different prompt versions.

Built specifically for the v14 → v15.3 comparison but generic across any
two runs that share qids. Uses the artifacts subsystem (Phase 1) — pulls
per-item bundles to compute behavioral signals beyond pass/fail.

Outputs:
  - eval/longmem/reports/diff_{run_a}_vs_{run_b}/comparison.md
    Markdown report with per-item table, per-axis shifts, bucket
    distribution shift, behavioral signal counts (my_raw_quote
    usage, scout-handoff usage, open-node creation, etc.)

USE
    ./dev python3 eval/longmem/run_diff.py \\
        eval_a_2026_05_10 \\
        eval_a_v15_3_2026_05_10
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.longmem.artifacts import load_artifacts, list_items


# ─── behavioral signal probes ────────────────────────────────────────

def _count_nodes_with_field(nodes, field_name):
    """Count nodes whose `kv` dict contains a non-empty value for field_name."""
    n = 0
    for node in nodes or []:
        kv = node.get('kv') or {}
        if kv.get(field_name):
            n += 1
    return n


def _count_open_nodes(nodes):
    """Count `type='open'` nodes — direct signal of live-contradiction encoding."""
    return sum(1 for n in (nodes or []) if n.get('type') == 'open')


def _count_third_party_quotes(nodes):
    """Count `type='quote'` nodes that have neither their_raw_quote nor
    my_raw_quote populated — these are third-party verbatim
    preservation, the v15.2/v15.3 new behavior."""
    n = 0
    for node in nodes or []:
        if node.get('type') != 'quote':
            continue
        kv = node.get('kv') or {}
        if not kv.get('their_raw_quote') and not kv.get('my_raw_quote'):
            n += 1
    return n


def _interaction_used(interactions, name):
    """Return (max_version, template_chars) for the named interaction in
    a per-item brain — confirms which prompt the encoder actually saw.
    """
    rows = [i for i in (interactions or []) if i.get('name') == name]
    if not rows:
        return (None, 0)
    latest = max(rows, key=lambda r: r.get('version', 0))
    return (latest.get('version'), len(latest.get('template') or ''))


def _summarize_tool_trace(recall):
    """Count agentic-surface tool invocations from recall.json tool_trace.

    tool_trace shape (from servers/scales/s1/surface.py::_call_surface_agentic):
      [{round, stop_reason, tool_calls: [{tool, args, result_count, ...}, ...]}, ...]

    Returns dict: {tool_name: count, '_total': n, '_rounds': r,
                   '_variant': surface_variant}. v4 surface emits an empty
    list; v5 agentic emits one entry per round.
    """
    out = {'_total': 0, '_rounds': 0,
           '_variant': (recall or {}).get('surface_variant', '')}
    for round_entry in (recall or {}).get('tool_trace') or []:
        if not isinstance(round_entry, dict):
            continue
        out['_rounds'] += 1
        for tc in round_entry.get('tool_calls') or []:
            name = (tc.get('tool') if isinstance(tc, dict) else None) or '?'
            out[name] = out.get(name, 0) + 1
            out['_total'] += 1
    return out


def _gather_item_signals(bundle):
    """Per-item behavioral signal sample."""
    nodes = bundle.get('nodes') or []
    interactions = bundle.get('interactions') or []
    recall = bundle.get('recall') or {}
    s1e_v, s1e_chars = _interaction_used(interactions, 's1e')
    surface_v, _ = _interaction_used(interactions, 'surface')
    return {
        'node_count': len(nodes),
        'with_their_raw_quote': _count_nodes_with_field(nodes, 'their_raw_quote'),
        'with_my_raw_quote': _count_nodes_with_field(nodes, 'my_raw_quote'),
        'with_entity_field': _count_nodes_with_field(nodes, 'entity'),  # scout handoff
        'with_event_time': _count_nodes_with_field(nodes, 'event_time'),  # L4 + v15.8 temporal
        'open_nodes': _count_open_nodes(nodes),
        'third_party_quote_nodes': _count_third_party_quotes(nodes),
        's1e_version': s1e_v,
        's1e_chars': s1e_chars,
        'surface_version': surface_v,
        'surface_tools': _summarize_tool_trace(recall),
    }


# ─── diff core ───────────────────────────────────────────────────────

def _load_results_jsonl(run_name: str) -> dict:
    """Fallback for runs without per-item artifacts (pre-Phase-1).

    Reads eval/longmem/reports/results_{run}.jsonl and returns
    {qid: result_dict}. Used when artifact bundles aren't available.
    """
    p = Path(__file__).resolve().parent / 'reports' / f'results_{run_name}.jsonl'
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            qid = r.get('question_id')
            if qid:
                out[qid] = r
        except Exception:
            continue
    return out


def diff_runs(run_a: str, run_b: str) -> dict:
    """Compute side-by-side comparison + behavioral signals.

    Falls back to results.jsonl for runs without per-item artifacts.
    Behavioral signals only computable for runs with artifact bundles.
    """
    qids_a_artifacts = set(list_items(run_a))
    qids_b_artifacts = set(list_items(run_b))
    results_a = _load_results_jsonl(run_a)
    results_b = _load_results_jsonl(run_b)

    qids_a = qids_a_artifacts | set(results_a.keys())
    qids_b = qids_b_artifacts | set(results_b.keys())
    common = sorted(qids_a & qids_b)
    only_a = sorted(qids_a - qids_b)
    only_b = sorted(qids_b - qids_a)

    rows = []
    a_passes = b_passes = 0
    by_axis_a = defaultdict(lambda: {'pass': 0, 'fail': 0})
    by_axis_b = defaultdict(lambda: {'pass': 0, 'fail': 0})
    bucket_a = Counter()
    bucket_b = Counter()
    signals_a = []
    signals_b = []

    for qid in common:
        # Try artifacts first; fall back to results.jsonl for runs without them.
        ba = load_artifacts(run_a, qid) if qid in qids_a_artifacts else {}
        bb = load_artifacts(run_b, qid) if qid in qids_b_artifacts else {}

        ra = ba.get('result') or results_a.get(qid) or {}
        rb = bb.get('result') or results_b.get(qid) or {}

        a_correct = bool(ra.get('correct'))
        b_correct = bool(rb.get('correct'))
        axis = (
            (ba.get('meta') or {}).get('axis')
            or (bb.get('meta') or {}).get('axis')
            or ra.get('axis')
            or rb.get('axis')
            or '?'
        )

        if a_correct: a_passes += 1
        if b_correct: b_passes += 1

        by_axis_a[axis]['pass' if a_correct else 'fail'] += 1
        by_axis_b[axis]['pass' if b_correct else 'fail'] += 1

        if not a_correct and ra.get('failure_bucket'):
            bucket_a[ra['failure_bucket']] += 1
        if not b_correct and rb.get('failure_bucket'):
            bucket_b[rb['failure_bucket']] += 1

        sig_a = _gather_item_signals(ba)
        sig_b = _gather_item_signals(bb)
        signals_a.append(sig_a)
        signals_b.append(sig_b)

        movement = (
            'unchanged_pass' if a_correct and b_correct else
            'unchanged_fail' if not a_correct and not b_correct else
            'fail→pass' if not a_correct and b_correct else
            'pass→fail'
        )

        rows.append({
            'qid': qid,
            'axis': axis,
            'a_correct': a_correct,
            'b_correct': b_correct,
            'movement': movement,
            'a_bucket': ra.get('failure_bucket'),
            'b_bucket': rb.get('failure_bucket'),
            'a_nodes': sig_a['node_count'],
            'b_nodes': sig_b['node_count'],
            'a_anchor_quotes': sig_a['with_my_raw_quote'],
            'b_anchor_quotes': sig_b['with_my_raw_quote'],
            'a_open_nodes': sig_a['open_nodes'],
            'b_open_nodes': sig_b['open_nodes'],
            'a_third_party_quotes': sig_a['third_party_quote_nodes'],
            'b_third_party_quotes': sig_b['third_party_quote_nodes'],
            'a_entity_nodes': sig_a['with_entity_field'],
            'b_entity_nodes': sig_b['with_entity_field'],
            'a_event_time': sig_a['with_event_time'],
            'b_event_time': sig_b['with_event_time'],
            'a_surface_tools': sig_a.get('surface_tools') or {},
            'b_surface_tools': sig_b.get('surface_tools') or {},
            'b_s1e_version': sig_b['s1e_version'],
            'b_s1e_chars': sig_b['s1e_chars'],
        })

    movement_counts = Counter(r['movement'] for r in rows)

    # Aggregate signal sums — skip non-numeric fields (e.g. s1e_version str/None)
    def _sum(field, lst):
        return sum(s[field] for s in lst if isinstance(s.get(field), (int, float)))

    return {
        'run_a': run_a,
        'run_b': run_b,
        'common_count': len(common),
        'only_a': only_a,
        'only_b': only_b,
        'a_passes': a_passes,
        'b_passes': b_passes,
        'a_pass_rate': a_passes / len(common) if common else 0,
        'b_pass_rate': b_passes / len(common) if common else 0,
        'by_axis_a': dict(by_axis_a),
        'by_axis_b': dict(by_axis_b),
        'bucket_a': dict(bucket_a),
        'bucket_b': dict(bucket_b),
        'movement_counts': dict(movement_counts),
        'rows': rows,
        'signal_totals': {
            'a': {f: _sum(f, signals_a) for f in signals_a[0]} if signals_a else {},
            'b': {f: _sum(f, signals_b) for f in signals_b[0]} if signals_b else {},
        },
    }


# ─── markdown rendering ──────────────────────────────────────────────

def render_md(diff: dict) -> str:
    a, b = diff['run_a'], diff['run_b']
    lines = []
    lines.append(f"# Run diff — {a} vs {b}")
    lines.append('')
    lines.append(f"**A:** {a}")
    lines.append(f"**B:** {b}")
    lines.append(f"**Common items:** {diff['common_count']}")
    if diff['only_a']:
        lines.append(f"**Only in A:** {len(diff['only_a'])} ({', '.join(diff['only_a'][:5])}…)")
    if diff['only_b']:
        lines.append(f"**Only in B:** {len(diff['only_b'])} ({', '.join(diff['only_b'][:5])}…)")
    lines.append('')

    # Headline pass rates
    lines.append('## Headline')
    lines.append('')
    lines.append('| | Pass rate | Pass count |')
    lines.append('|---|---:|---:|')
    lines.append(f"| A ({a}) | {diff['a_pass_rate']:.1%} | {diff['a_passes']}/{diff['common_count']} |")
    lines.append(f"| B ({b}) | {diff['b_pass_rate']:.1%} | {diff['b_passes']}/{diff['common_count']} |")
    delta = diff['b_pass_rate'] - diff['a_pass_rate']
    lines.append(f"| **Δ** | **{delta:+.1%}** | **{diff['b_passes'] - diff['a_passes']:+d}** |")
    lines.append('')

    # Movement counts
    lines.append('## Movement')
    lines.append('')
    lines.append('| Movement | Count |')
    lines.append('|---|---:|')
    for m in ['unchanged_pass', 'unchanged_fail', 'fail→pass', 'pass→fail']:
        lines.append(f"| `{m}` | {diff['movement_counts'].get(m, 0)} |")
    lines.append('')

    # Per-axis pass rate
    lines.append('## Per-axis pass rate')
    lines.append('')
    lines.append('| Axis | A pass | B pass | Δ |')
    lines.append('|---|---:|---:|---:|')
    axes = sorted(set(diff['by_axis_a']) | set(diff['by_axis_b']))
    for axis in axes:
        a_d = diff['by_axis_a'].get(axis, {'pass': 0, 'fail': 0})
        b_d = diff['by_axis_b'].get(axis, {'pass': 0, 'fail': 0})
        a_total = a_d['pass'] + a_d['fail']
        b_total = b_d['pass'] + b_d['fail']
        a_rate = a_d['pass'] / a_total if a_total else 0
        b_rate = b_d['pass'] / b_total if b_total else 0
        lines.append(f"| {axis} | {a_rate:.0%} ({a_d['pass']}/{a_total}) | "
                     f"{b_rate:.0%} ({b_d['pass']}/{b_total}) | {b_rate-a_rate:+.0%} |")
    lines.append('')

    # Bucket distribution
    lines.append('## Failure bucket distribution')
    lines.append('')
    lines.append('| Bucket | A | B | Δ |')
    lines.append('|---|---:|---:|---:|')
    buckets = sorted(set(diff['bucket_a']) | set(diff['bucket_b']))
    for bucket in buckets:
        a_n = diff['bucket_a'].get(bucket, 0)
        b_n = diff['bucket_b'].get(bucket, 0)
        lines.append(f"| `{bucket}` | {a_n} | {b_n} | {b_n-a_n:+d} |")
    lines.append('')

    # Behavioral signals — total across the cohort
    lines.append('## Behavioral signal totals (cohort-wide)')
    lines.append('')
    lines.append('| Signal | A total | B total | Δ | What this measures |')
    lines.append('|---|---:|---:|---:|---|')
    sa, sb = diff['signal_totals']['a'], diff['signal_totals']['b']
    explanations = {
        'node_count': 'Total nodes encoded',
        'with_their_raw_quote': 'Nodes preserving operator voice',
        'with_my_raw_quote': 'Nodes preserving Anchor voice (v15+ behavior)',
        'with_entity_field': 'Nodes with entity field — scout-handoff signal',
        'with_event_time': 'Nodes with event_time kv — temporal anchor (v15.8 + L4)',
        'open_nodes': 'Open-type nodes — live-contradiction encoding (v15+)',
        'third_party_quote_nodes': 'Quote nodes with no participant attribution — third-party verbatim (v15.2+)',
    }
    for k in ['node_count', 'with_their_raw_quote', 'with_my_raw_quote',
              'with_entity_field', 'with_event_time', 'open_nodes',
              'third_party_quote_nodes']:
        a_v = sa.get(k, 0)
        b_v = sb.get(k, 0)
        delta = b_v - a_v
        lines.append(f"| {k} | {a_v} | {b_v} | {delta:+d} | {explanations[k]} |")
    lines.append('')

    # Agentic-surface tool usage (per-arm). v4 surface is single-shot (no
    # tool_trace); v5 agentic emits tool_use blocks recorded per item.
    tool_counts_a = Counter()
    tool_counts_b = Counter()
    items_with_tools_a = items_with_tools_b = 0
    variants_a = Counter()
    variants_b = Counter()
    for r in diff['rows']:
        sa_t = r.get('a_surface_tools') or {}
        sb_t = r.get('b_surface_tools') or {}
        for k, v in sa_t.items():
            if k.startswith('_'): continue
            tool_counts_a[k] += v
        for k, v in sb_t.items():
            if k.startswith('_'): continue
            tool_counts_b[k] += v
        if sa_t.get('_total', 0) > 0:
            items_with_tools_a += 1
        if sb_t.get('_total', 0) > 0:
            items_with_tools_b += 1
        if sa_t.get('_variant'):
            variants_a[sa_t['_variant']] += 1
        if sb_t.get('_variant'):
            variants_b[sb_t['_variant']] += 1

    lines.append('## Surface tool usage')
    lines.append('')
    lines.append('| | A | B |')
    lines.append('|---|---|---|')
    lines.append(f"| Surface variant(s) | {dict(variants_a) or '-'} | {dict(variants_b) or '-'} |")
    lines.append(f"| Items where surface invoked tools | {items_with_tools_a}/{diff['common_count']} | {items_with_tools_b}/{diff['common_count']} |")
    lines.append(f"| Total tool calls | {sum(tool_counts_a.values())} | {sum(tool_counts_b.values())} |")
    lines.append('')
    if tool_counts_a or tool_counts_b:
        lines.append('| Tool | A calls | B calls |')
        lines.append('|---|---:|---:|')
        for tool in sorted(set(tool_counts_a) | set(tool_counts_b)):
            lines.append(f"| `{tool}` | {tool_counts_a.get(tool, 0)} | {tool_counts_b.get(tool, 0)} |")
        lines.append('')

    # Movement-specific item lists
    lines.append('## Items that flipped')
    lines.append('')
    flipped_to_pass = [r for r in diff['rows'] if r['movement'] == 'fail→pass']
    flipped_to_fail = [r for r in diff['rows'] if r['movement'] == 'pass→fail']

    lines.append(f"### `fail→pass` ({len(flipped_to_pass)})")
    lines.append('')
    if flipped_to_pass:
        lines.append('| qid | axis | A bucket | B nodes | B open | B anchor-quotes |')
        lines.append('|---|---|---|---:|---:|---:|')
        for r in flipped_to_pass:
            lines.append(f"| `{r['qid']}` | {r['axis']} | `{r['a_bucket']}` | "
                         f"{r['b_nodes']} | {r['b_open_nodes']} | {r['b_anchor_quotes']} |")
    else:
        lines.append('_(none)_')
    lines.append('')

    lines.append(f"### `pass→fail` ({len(flipped_to_fail)})")
    lines.append('')
    if flipped_to_fail:
        lines.append('| qid | axis | B bucket | B nodes |')
        lines.append('|---|---|---|---:|')
        for r in flipped_to_fail:
            lines.append(f"| `{r['qid']}` | {r['axis']} | `{r['b_bucket']}` | {r['b_nodes']} |")
    else:
        lines.append('_(none — no regression)_')
    lines.append('')

    # Full per-item table (collapsible)
    lines.append('## All items (full table)')
    lines.append('')
    lines.append('| qid | axis | A | B | move | A bucket | B bucket | A nodes | B nodes |')
    lines.append('|---|---|:---:|:---:|---|---|---|---:|---:|')
    for r in diff['rows']:
        a_mark = '✓' if r['a_correct'] else '✗'
        b_mark = '✓' if r['b_correct'] else '✗'
        lines.append(f"| `{r['qid']}` | {r['axis']} | {a_mark} | {b_mark} | "
                     f"{r['movement']} | `{r['a_bucket'] or '-'}` | "
                     f"`{r['b_bucket'] or '-'}` | {r['a_nodes']} | {r['b_nodes']} |")
    lines.append('')

    # B-run prompt sanity
    if diff['rows']:
        b_versions = Counter(r['b_s1e_version'] for r in diff['rows'])
        b_chars = Counter(r['b_s1e_chars'] for r in diff['rows'])
        lines.append('## Sanity: which s1e prompt did B actually run?')
        lines.append('')
        lines.append(f"- s1e versions seen in B: {dict(b_versions)}")
        lines.append(f"- s1e prompt char counts: {dict(b_chars)}")
        lines.append('')

    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_a')
    p.add_argument('run_b')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    diff = diff_runs(args.run_a, args.run_b)

    out_dir = Path(__file__).resolve().parent / 'reports' / f'diff_{args.run_a}_vs_{args.run_b}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out or str(out_dir / 'comparison.md')

    Path(out).write_text(render_md(diff))
    json_out = out.replace('.md', '.json')
    Path(json_out).write_text(json.dumps(diff, indent=2, default=str))

    print(f"diff: {diff['common_count']} common items")
    print(f"A pass rate: {diff['a_pass_rate']:.1%} → B pass rate: {diff['b_pass_rate']:.1%}")
    print(f"  fail→pass: {diff['movement_counts'].get('fail→pass', 0)}")
    print(f"  pass→fail: {diff['movement_counts'].get('pass→fail', 0)}")
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
