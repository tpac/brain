"""Per-item agentic surface tool-trace summary.

Reads recall.json from each item bundle and emits the per-round tool
invocation sequence (which tool, what args, how many results).

tool_trace shape from servers/scales/s1/surface.py::_call_surface_agentic:
  [{round, stop_reason, tool_calls: [{tool, args, result_count, latency_ms, error}, ...]}, ...]

USE
    ./dev python3 eval/longmem/tool_trace_summary.py compare_candidate_2026_05_13_HHMMSS
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.longmem.artifacts import list_items, load_artifacts


def _short_args(call):
    """Compact rendering of one tool_call's args."""
    args = call.get('args') if isinstance(call, dict) else None
    if not isinstance(args, dict):
        return json.dumps(args, default=str)[:100] if args is not None else ''
    parts = []
    for k in ('query', 'queries', 'topic', 'node_id', 'date', 'aspect',
              'limit', 'k', 'days'):
        if k in args:
            v = args[k]
            if isinstance(v, str):
                v_show = v if len(v) < 60 else v[:57] + '...'
                parts.append(f'{k}="{v_show}"')
            else:
                parts.append(f'{k}={json.dumps(v, default=str)[:60]}')
    for k, v in args.items():
        if k not in ('query', 'queries', 'topic', 'node_id', 'date',
                     'aspect', 'limit', 'k', 'days'):
            v_show = json.dumps(v, default=str)
            v_show = v_show if len(v_show) < 60 else v_show[:57] + '...'
            parts.append(f'{k}={v_show}')
    return ', '.join(parts)


def _result_summary(call):
    """Render result_count + error from one tool_call."""
    if not isinstance(call, dict):
        return ''
    n = call.get('result_count')
    err = call.get('error')
    lat = call.get('latency_ms')
    parts = []
    if n is not None:
        parts.append(f'→ {n} results')
    if lat:
        parts.append(f'{lat}ms')
    if err:
        parts.append(f'ERROR={err}')
    return ' '.join(parts)


def summarize_run(run_name: str) -> str:
    out = []
    qids = list_items(run_name)

    items_with_tools = 0
    total_calls = 0
    total_rounds = 0
    tool_counts = {}
    body = []

    for qid in qids:
        bundle = load_artifacts(run_name, qid)
        meta = bundle.get('meta') or {}
        result = bundle.get('result') or {}
        recall = bundle.get('recall') or {}
        rounds = recall.get('tool_trace') or []

        # Flatten tool_calls across rounds for counts
        flat_calls = []
        for r in rounds:
            if isinstance(r, dict):
                for tc in r.get('tool_calls') or []:
                    flat_calls.append(tc)

        n_calls = len(flat_calls)
        n_rounds = len(rounds)
        if n_calls > 0:
            items_with_tools += 1
        total_calls += n_calls
        total_rounds += n_rounds

        for tc in flat_calls:
            tname = (tc.get('tool') if isinstance(tc, dict) else None) or '?'
            tool_counts[tname] = tool_counts.get(tname, 0) + 1

        verdict = 'PASS' if result.get('correct') else 'FAIL'
        axis = meta.get('axis', '?')
        variant = recall.get('surface_variant', '')
        body.append(f'## `{qid}` ({axis}) — {verdict} — variant={variant or "—"} '
                    f'— {n_rounds} rounds, {n_calls} tool calls')
        body.append('')
        body.append(f'**Question:** {meta.get("question", "?")}')
        body.append('')
        if n_calls == 0 and n_rounds <= 1:
            if bundle.get('recall') is None:
                body.append('_⚠ recall.json missing — tool trace unavailable '
                            '(capture state unknown, not a behavioral finding)_')
            elif n_rounds == 0 and (variant or '').startswith('v5'):
                # The agentic surface ALWAYS writes a round record, even on a
                # no-fire run (sweep.py's probe-fidelity invariant) — an empty
                # trace under v5 means capture broke, not "chose no tools".
                body.append('_⚠ tool_trace EMPTY under the agentic variant — '
                            'capture broke; not a behavioral finding_')
            else:
                body.append('_(no tool calls — surface chose to answer with retrieval alone)_')
            body.append('')
            continue
        for ri, r in enumerate(rounds):
            stop_r = r.get('stop_reason', '?') if isinstance(r, dict) else '?'
            calls = (r.get('tool_calls') or []) if isinstance(r, dict) else []
            body.append(f'  **Round {ri}** (stop_reason={stop_r}, {len(calls)} call{"s" if len(calls)!=1 else ""}):')
            for i, tc in enumerate(calls, start=1):
                tool = tc.get('tool', '?') if isinstance(tc, dict) else '?'
                args = _short_args(tc)
                res = _result_summary(tc)
                body.append(f'    {i}. **`{tool}`**({args}) {res}')
            body.append('')

    header = []
    header.append(f'# Tool trace summary — {run_name}')
    header.append('')
    header.append(f'**Items:** {len(qids)}')
    header.append(f'**Items with tool use:** {items_with_tools}/{len(qids)}')
    header.append(f'**Total rounds across cohort:** {total_rounds}')
    header.append(f'**Total tool calls:** {total_calls}')
    if tool_counts:
        header.append(f'**Per-tool counts:** {tool_counts}')
    header.append('')
    return '\n'.join(header + body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_name')
    p.add_argument('--out', default=None,
                   help='Markdown output path (default: stdout)')
    args = p.parse_args()
    md = summarize_run(args.run_name)
    if args.out:
        Path(args.out).write_text(md)
        print(f'wrote {args.out}')
    else:
        print(md)


if __name__ == '__main__':
    main()
