"""Surface-redundancy census — how many retrieval surfaces carry each node's load-bearing values.

For every FINAL node of every sequence (last window's nodes_after): extract the load-bearing
values of the node — numbers/times (not bare years), ISO and month-day dates, and
proper-noun tokens (capitalized words not at sentence start, not role words) — from the
union of title, content and quotes. For each value, count which of five surfaces carry it:
title, content, quote (their/my_raw_quote), situation, question. Report per arm the
distribution of surfaces-per-value, the share of values findable only through content, the
share of nodes whose title carries at least one of the node's values, and per-field value
shares. No model calls; descriptive. The instrument X1 (three paths to one fact) needs.

Usage: surface_redundancy.py <root> [<root>...] --out <dir> --label <cell>
"""
import argparse, json, re, collections
from pathlib import Path

MONTH = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?'
NUM_RE = re.compile(r'(?<![A-Za-z0-9-])\d[\d,.:]*(?:\s?(?:%|am|pm|km|kg|lb|oz|min|hrs?|hours?|minutes?|seconds?|coins?|jars?|bunches?|plants?|points?|years?|weeks?|days?))?(?![A-Za-z0-9-])', re.I)
DATE_RE = re.compile(r'\b\d{4}-\d\d(?:-\d\d)?\b|\b' + MONTH + r' \d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?\b|\b\d{1,2} ' + MONTH + r'(?: \d{4})?\b')
CAP_RE = re.compile(r"(?<![.!?:]\s)(?<!^)(?<![\"“'‘(\[])\b[A-Z][a-z]{2,}(?:['’][a-z]+)?\b")
ROLE = {'User', 'Partner', 'Collector', 'They', 'The', 'She', 'Her', 'His', 'Their', 'Other', 'Side', 'Assistant', 'Encoder', 'Anchor', 'Prior', 'Earlier', 'Later', 'Current', 'When', 'This', 'That', 'These', 'Those', 'What', 'How', 'Why', 'Where', 'Who', 'Which', 'Not', 'And', 'But', 'For', 'With', 'From', 'Also', 'Both', 'Each', 'Then', 'Than', 'Into', 'Over', 'After', 'Before', 'During', 'While', 'Because', 'Since', 'Until', 'Still', 'Yet', 'Confirmed', 'Updated', 'Plan', 'Plans', 'Decision', 'Fact', 'Note', 'Status', 'Turn', 'Window', 'Session'}
SURF = ['title', 'content', 'quote', 'situation', 'question']


def field(n, k):
    if k in n and n[k] not in (None, ''): return n[k]
    return (n.get('_metadata') or {}).get(k)


def values_of(text):
    t = str(text or '')
    vals = set()
    for m in DATE_RE.findall(t): vals.add(m.strip())
    for m in NUM_RE.findall(t):
        v = m.strip().rstrip('.,:')
        if not v or (v.isdigit() and 1900 <= int(v) <= 2100) or v in ('1', '2', '3', 'one'): continue
        vals.add(v)
    for m in CAP_RE.findall(t):
        if m not in ROLE: vals.add(m)
    return vals


def carries(text, v):
    core = v.split()[0] if ' ' in v and not DATE_RE.fullmatch(v) else v   # "12 bunches" -> "12" is enough
    return re.search(r'(?<![A-Za-z0-9])' + re.escape(core) + r'(?![A-Za-z0-9])', str(text or ''), re.I) is not None


def final_nodes(root):
    root = Path(root)
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(('seed', 'process', 'blind', 'downstream', 'coverage', 'preflight', 'whole'))):
        for rep_dir in sorted(arm_dir.glob('repeat*')):
            for corpus_dir in sorted(p for p in rep_dir.iterdir() if p.is_dir() and not p.name.endswith('.crashed-window2')):
                wins = sorted(corpus_dir.glob('window*/nodes_after.json'), key=lambda p: int(p.parent.name[6:]))
                if wins:
                    yield arm_dir.name, int(rep_dir.name[6:]), corpus_dir.name, json.loads(wins[-1].read_text())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('roots', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--label', default='cell')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_arm = collections.OrderedDict(); examples = collections.defaultdict(list)
    for root in a.roots:
        for arm, rep, corpus, nodes in final_nodes(root):
            s = per_arm.setdefault(arm, {'nodes': 0, 'nodes_with_values': 0, 'values': 0, 'dist': collections.Counter(), 'content_only': 0, 'title_has_value': 0, 'surface_share': collections.Counter()})
            for nid, n in nodes.items():
                s['nodes'] += 1
                quote = ' '.join(str(field(n, q) or '') for q in ('their_raw_quote', 'my_raw_quote'))
                texts = {'title': field(n, 'title'), 'content': field(n, 'content'), 'quote': quote, 'situation': field(n, 'situation'), 'question': field(n, 'question')}
                vals = values_of(texts['title']) | values_of(texts['content']) | values_of(quote)
                if not vals: continue
                s['nodes_with_values'] += 1
                title_hit = False
                for v in vals:
                    where = [k for k in SURF if texts[k] and carries(texts[k], v)]
                    s['values'] += 1; s['dist'][min(len(where), 4)] += 1
                    for k in where: s['surface_share'][k] += 1
                    if where == ['content']: s['content_only'] += 1
                    if 'title' in where: title_hit = True
                    if len(where) == 1 and len(examples[arm]) < 12: examples[arm].append(f"{corpus} r{rep} `{nid}` `{v}` only in {where[0]} — “{str(texts['title'])[:90]}”")
                s['title_has_value'] += title_hit
    lines = [f'# Surface redundancy — {a.label}', '', 'Load-bearing values (numbers, dates, proper nouns) per final node and how many of title / content / quote / situation / question carry each. Descriptive; the instrument for X1.', '',
             '| Arm | Nodes | with values | values | on 1 surface | 2 | 3 | 4+ | content-only | mean surfaces/value | nodes whose title carries a value | title share | content | quote | situation | question |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, s in per_arm.items():
        v = s['values'] or 1; d = s['dist']
        mean = sum(k * c for k, c in d.items()) / v
        lines.append(f"| {arm} | {s['nodes']} | {s['nodes_with_values']} | {s['values']} | {d[1] / v:.0%} | {d[2] / v:.0%} | {d[3] / v:.0%} | {d[4] / v:.0%} | {s['content_only'] / v:.0%} | {mean:.2f} | {s['title_has_value'] / (s['nodes_with_values'] or 1):.0%} | " + ' | '.join(f"{s['surface_share'][k] / v:.0%}" for k in SURF) + ' |')
    lines += ['', '## Single-surface values, examples', ''] + [f'- **{arm}**: ' + e for arm in examples for e in examples[arm][:6]]
    (out / f'SURFACE-REDUNDANCY-{a.label}.md').write_text('\n'.join(lines) + '\n')
    (out / f'surface_redundancy_{a.label}.json').write_text(json.dumps({arm: {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in s.items()} for arm, s in per_arm.items()}, indent=1))
    print('\n'.join(lines[4:6 + len(per_arm)]))


if __name__ == '__main__':
    main()
