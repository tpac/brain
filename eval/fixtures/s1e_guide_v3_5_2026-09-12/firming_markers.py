"""Firming-up census — certainty markers in memory that the source never used.

For every final node (last window's nodes_after) and each of title, content, situation,
question, reasoning, thought: find status-upgrading markers (confirmed, committed, decided,
measured, verified, established, finalized, adopted, agreed, completed, launched,
implemented, always, never, definitely, guaranteed, proven) and hedges (may, might, not yet,
unconfirmed, leaning, considering, tentative, unclear, possibly, plans to, intends to).
A marker whose stem appears nowhere in the corpus source (all windows, both voices) is
UNSOURCED — the encoder supplied the certainty. Per arm: markers per 100 nodes, unsourced
share (reasoning reported apart — 'confirms the transition' there is provenance language), hedges per 100 nodes, and a reading queue of unsourced instances. No model calls;
the queue feeds the reviewer (blind dimension 4), it is not a verdict.

Usage: firming_markers.py <root> [<root>...] --out <dir> --label <cell>
"""
import argparse, json, re, collections
from pathlib import Path

MARKERS = {'confirm': r'confirm(?:ed|s|ation)?', 'commit': r'committ?(?:ed|s|ment)', 'decide': r'decid(?:ed|es|ion)', 'measure': r'measur(?:ed|ement)', 'verify': r'verif(?:ied|ication)',
           'establish': r'establish(?:ed|es)', 'finalize': r'finali[sz](?:ed|es)', 'adopt': r'adopt(?:ed|s|ion)', 'agree': r'agree(?:d|s|ment)', 'complete': r'complet(?:ed|es|ion)',
           'launch': r'launch(?:ed|es)', 'implement': r'implement(?:ed|s|ation)', 'always': r'always', 'never': r'never', 'definitely': r'definite(?:ly)?', 'guarantee': r'guarantee[ds]?', 'proven': r'prov(?:en|ed)\b'}
HEDGES = r"\b(?:may|might|not yet|unconfirmed|leaning|considering|tentative(?:ly)?|unclear|possibly|plans? to|intends? to|no(?:t)? (?:yet )?reported|has not|have not|remains? open|unresolved|estimate[sd]?)\b"
FIELDS = ['title', 'content', 'situation', 'question', 'reasoning', 'thought']


def field(n, k):
    if k in n and n[k] not in (None, ''): return n[k]
    return (n.get('_metadata') or {}).get(k)


def source_text(root, corpus):
    p = Path(root) / (corpus + '.json')
    if not p.exists(): return ''
    fx = json.loads(p.read_text())
    return ' '.join(str(t.get(k) or '') for w in fx.get('windows', []) for t in w.get('turns', []) for k in ('other', 'me'))


def final_nodes(root):
    root = Path(root)
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(('seed', 'process', 'blind', 'downstream', 'coverage', 'preflight', 'whole'))):
        for rep_dir in sorted(arm_dir.glob('repeat*')):
            for corpus_dir in sorted(p for p in rep_dir.iterdir() if p.is_dir() and not p.name.endswith('.crashed-window2')):
                wins = sorted(corpus_dir.glob('window*/nodes_after.json'), key=lambda p: int(p.parent.name[6:]))
                if wins: yield arm_dir.name, int(rep_dir.name[6:]), corpus_dir.name, json.loads(wins[-1].read_text())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('roots', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--label', default='cell')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_arm = collections.OrderedDict(); queue = []; sources = {}
    for root in a.roots:
        for arm, rep, corpus, nodes in final_nodes(root):
            src = sources.setdefault((root, corpus), source_text(root, corpus).lower())
            s = per_arm.setdefault(arm, {'nodes': 0, 'markers': 0, 'unsourced': 0, 'unsourced_surface': 0, 'hedges': 0, 'nodes_with_marker': 0, 'nodes_with_unsourced': 0, 'nodes_with_unsourced_surface': 0, 'by_marker': collections.Counter(), 'unsourced_by_marker': collections.Counter(), 'by_field': collections.Counter()})
            for nid, n in nodes.items():
                s['nodes'] += 1; had, had_un = False, False
                for f in FIELDS:
                    val = str(field(n, f) or '')
                    if not val: continue
                    s['hedges'] += len(re.findall(HEDGES, val, re.I))
                    for name, rx in MARKERS.items():
                        for m in re.finditer(r'\b' + rx, val, re.I):
                            s['markers'] += 1; s['by_marker'][name] += 1; s['by_field'][f] += 1; had = True
                            if not re.search(r'\b' + rx, src, re.I):
                                s['unsourced'] += 1; s['unsourced_by_marker'][name] += 1
                                if f != 'reasoning': s['unsourced_surface'] += 1; had_un = True
                                if len(queue) < 400:
                                    lo = max(0, m.start() - 70); queue.append({'arm': arm, 'repeat': rep, 'corpus': corpus, 'node_id': nid, 'field': f, 'marker': name, 'snippet': re.sub(r'\s+', ' ', val[lo:m.end() + 70])})
                s['nodes_with_marker'] += had; s['nodes_with_unsourced_surface'] += had_un
    lines = [f'# Firming-up markers — {a.label}', '', 'Certainty markers in memory fields against the corpus source; UNSOURCED = the stem appears nowhere in the source, so the encoder supplied the certainty. Reading queue for the scope reviewer, not a verdict.', '',
             '| Arm | Nodes | markers /100 nodes | unsourced, all fields | unsourced in title/content/situation/question | nodes carrying one (surface) | hedges /100 nodes | top markers (unsourced) | fields |', '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for arm, s in per_arm.items():
        n = s['nodes'] or 1
        top = ', '.join(f"{k} {v}({s['unsourced_by_marker'][k]})" for k, v in s['by_marker'].most_common(5))
        flds = ', '.join(f'{k} {v}' for k, v in s['by_field'].most_common())
        lines.append(f"| {arm} | {s['nodes']} | {100 * s['markers'] / n:.1f} | {s['unsourced']} ({s['unsourced'] / (s['markers'] or 1):.0%}) | {s['unsourced_surface']} | {s['nodes_with_unsourced_surface']} ({s['nodes_with_unsourced_surface'] / n:.0%}) | {100 * s['hedges'] / n:.1f} | {top} | {flds} |")
    surf = [q for q in queue if q['field'] != 'reasoning']
    lines += ['', f'## Queue — unsourced markers on title/content/situation/question ({len(surf)} shown; reasoning instances are provenance language and sit in the JSON only)', ''] + [f"- **{q['arm']}** r{q['repeat']} {q['corpus']} `{q['node_id']}` {q['field']} [{q['marker']}]: “{q['snippet']}”" for q in surf]
    (out / f'FIRMING-MARKERS-{a.label}.md').write_text('\n'.join(lines) + '\n')
    (out / f'firming_markers_{a.label}.json').write_text(json.dumps({'per_arm': {arm: {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in s.items()} for arm, s in per_arm.items()}, 'queue': queue}, indent=1))
    print('\n'.join(lines[4:6 + len(per_arm)])); print(f'queue {len(queue)} → {out}')


if __name__ == '__main__':
    main()
