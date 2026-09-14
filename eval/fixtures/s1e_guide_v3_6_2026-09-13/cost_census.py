"""Tokens, rounds and tool use per arm — what a cell cost and how the encoder spent its turns.

Reads every window's result.json (the metered usage of the encoder loop: input, output,
cache_read, cache_write) and calls.json (the tool sequence), plus downstream/summary.json
(answer + judge tokens) and the judge log's running totals when present. Prices are the
list prices passed on the command line, so the estimate is reproducible and its assumptions
are visible; the token counts are the record.

    ./dev python3 cost_census.py <root> [<root>...] --out <dir> --label <cell> \
        [--price-in 3 --price-out 15 --price-cache-read 0.30 --price-cache-write 3.75]   # $ per 1M tokens
"""
import argparse, collections, json, re
from pathlib import Path

WRITE = {'remember', 'remember_batch', 'revise', 'revise_batch', 'brain_batch', 'connect', 'connect_batch', 'revise_edge'}


def windows(root):
    root = Path(root)
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(('seed', 'process', 'blind', 'downstream', 'coverage', 'preflight', 'whole', 'crashed'))):
        for rep_dir in sorted(arm_dir.glob('repeat*')):
            for corpus_dir in sorted(p for p in rep_dir.iterdir() if p.is_dir()):
                for win in sorted(corpus_dir.glob('window*'), key=lambda p: int(p.name[6:])):
                    if (win / 'result.json').exists() and (win / 'calls.json').exists():
                        yield arm_dir.name, rep_dir.name, corpus_dir.name, win


def judge_totals(root):
    """The judge log's last running total: 'calls so far N, in X out Y'."""
    for log in Path(root).glob('content_quality_judge.log'):
        last = None
        for line in log.read_text().splitlines():
            m = re.search(r'calls so far (\d+), in ([\d,]+) out ([\d,]+)', line)
            if m:
                last = {'calls': int(m.group(1)), 'in': int(m.group(2).replace(',', '')), 'out': int(m.group(3).replace(',', ''))}
        return last
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--label', default='cell')
    ap.add_argument('--price-in', type=float, default=3.0); ap.add_argument('--price-out', type=float, default=15.0)
    ap.add_argument('--price-cache-read', type=float, default=0.30); ap.add_argument('--price-cache-write', type=float, default=3.75)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per = collections.OrderedDict()
    for root in a.roots:
        for arm, rep, corpus, win in windows(root):
            s = per.setdefault(arm, {'windows': 0, 'rounds': 0, 'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0,
                                     'tools': collections.Counter(), 'calls': 0, 'writes': 0, 'reads': 0, 'sequences': set()})
            result = json.loads((win / 'result.json').read_text()); calls = json.loads((win / 'calls.json').read_text())
            u = result.get('usage', {})
            s['windows'] += 1; s['rounds'] += result['result'].get('rounds') or 0
            for k in ('input', 'output', 'cache_read', 'cache_write'):
                s[k] += u.get(k, 0)
            for c in calls:
                t = c['call']['tool']; s['tools'][t] += 1; s['calls'] += 1
                s['writes' if t in WRITE else 'reads'] += 1
            s['sequences'].add((rep, corpus))

    def dollars(s):
        return (s['input'] * a.price_in + s['output'] * a.price_out + s['cache_read'] * a.price_cache_read + s['cache_write'] * a.price_cache_write) / 1e6

    lines = [f'# Cost and tools — {a.label}', '',
             f'Encoder loop, metered per window (Sonnet 4.6). Prices assumed per 1M tokens: in ${a.price_in}, out ${a.price_out}, cache read ${a.price_cache_read}, cache write ${a.price_cache_write} — an estimate; the token counts are the record.', '',
             '| Arm | sequences | windows | rounds / window | input | output | cache read | cache write | est. $ | tool calls / window | writes / reads | tools used |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    total = collections.Counter()
    for arm, s in per.items():
        lines.append(f"| {arm} | {len(s['sequences'])} | {s['windows']} | {s['rounds'] / max(s['windows'], 1):.2f} | {s['input']:,} | {s['output']:,} | {s['cache_read']:,} | {s['cache_write']:,} | {dollars(s):.2f} | {s['calls'] / max(s['windows'], 1):.2f} | {s['writes']} / {s['reads']} | {', '.join(f'{k} {v}' for k, v in s['tools'].most_common())} |")
        for k in ('input', 'output', 'cache_read', 'cache_write', 'windows', 'calls'):
            total[k] += s[k]
        total['dollars'] += dollars(s)
    lines.append(f"| **all arms** | | {total['windows']} | | {total['input']:,} | {total['output']:,} | {total['cache_read']:,} | {total['cache_write']:,} | **{total['dollars']:.2f}** | | | |")
    extras = []
    for root in a.roots:
        summary = Path(root) / 'downstream' / 'summary.json'
        if summary.exists():
            rows = json.loads(summary.read_text())['rows']
            ai = sum(r['usage']['answer_in'] for r in rows); ao = sum(r['usage']['answer_out'] for r in rows)
            ji = sum(r['usage']['judge_in'] for r in rows); jo = sum(r['usage']['judge_out'] for r in rows)
            extras.append(f"| downstream ({Path(root).name}) | {len(rows)} answers | in {ai + ji:,} / out {ao + jo:,} | {((ai + ji) * a.price_in + (ao + jo) * a.price_out) / 1e6:.2f} |")
        jt = judge_totals(root)
        if jt:
            extras.append(f"| per-node judge ({Path(root).name}) | {jt['calls']} calls | in {jt['in']:,} / out {jt['out']:,} | {(jt['in'] * a.price_in + jt['out'] * a.price_out) / 1e6:.2f} |")
    if extras:
        lines += ['', '| Instrument (Sonnet 4.6) | calls | tokens | est. $ |', '|---|---:|---|---:|'] + extras
    lines += ['', 'Blind reviewers (Opus 4.8 agents) are metered by the harness, not by these files; record their reported token totals in the results document by hand.']
    (out / f'COST-TOOLS-{a.label}.md').write_text('\n'.join(lines) + '\n')
    (out / f'cost_tools_{a.label}.json').write_text(json.dumps({arm: {**{k: v for k, v in s.items() if k not in ('tools', 'sequences')}, 'tools': dict(s['tools']), 'sequences': len(s['sequences'])} for arm, s in per.items()}, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
