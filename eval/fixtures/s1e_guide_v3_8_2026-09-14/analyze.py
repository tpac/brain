"""Descriptive census, readable whole-memory packets and a cross-arm table for the V3.7 cell.

Reuses the shared field/edge/quote logic of the V3.1 census; differs in keying
on each sequence's LAST window (transfer items have two or three) and in
gathering the arms from their own result folders. No model calls; counts
describe outputs and never stand for quality — the author's source-based
review reads the packets.

    ./dev python3 eval/fixtures/s1e_guide_v3_7_2026-09-14/analyze.py --cell carriers
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('_census', ROOT / 'eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/quality_inventory.py')
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)
FIELDS, TEXT_FIELDS, field, words, get_edges, authored_fields, summary = (
    census.FIELDS, census.TEXT_FIELDS, census.field, census.words, census.get_edges, census.authored_fields, census.summary)
REFINE = ROOT / 'eval/results/s1e_v37_carriers_2026-09-14'
REGRESSION = ROOT / 'eval/results/s1e_v37_regression_v36_2026-09-14'
SAVED36 = ROOT / 'eval/results/s1e_v36_refine_2026-09-13'
ARMS = ('v3_6_full', 'v3_7_advice', 'v3_7_quote')


def _pinned_arms(out):
    """The arms a cell ran, from its manifest — the regression's rung is named after the refine cell is read."""
    return list(json.loads((out / 'manifest.json').read_text())['arms']) if (out / 'manifest.json').exists() else []


CELLS = {
    'carriers': {'out': REFINE, 'fixtures': REFINE, 'arms': [(a, REFINE) for a in ARMS]},
    # the shipping rung(s) beside the saved brains on the same six corpora: v3_4_live and v3_6_full from the V3.6 cell
    'regression_v36': {'out': REGRESSION, 'fixtures': REGRESSION, 'arms': [(a, REGRESSION) for a in _pinned_arms(REGRESSION)]
                       + [(a, SAVED36) for a in ('v3_4_live', 'v3_6_full')]},
}
ARC_RE = re.compile(r'## Arc\s*\n```[^\n]*\n(.*?)\n```', re.S)


def read(path):
    return json.loads(Path(path).read_text())


def sequences(cell):
    for arm, location in CELLS[cell]['arms']:
        for repeat_dir in sorted((location / arm).glob('repeat*')):
            for folder in sorted(p for p in repeat_dir.iterdir() if p.is_dir()):
                windows = sorted(folder.glob('window*'), key=lambda p: int(p.name[6:]))
                if windows and (windows[-1] / 'nodes_after.json').exists():
                    yield arm, repeat_dir.name, folder.name, folder, windows


def node_text(node):
    parts = [str(field(node, k) or '') for k in TEXT_FIELDS]
    return ' '.join(parts)


def spotlight_creative(nodes, continuity_text):
    """Heuristic flags for the development source; the author verifies each by reading."""
    texts = {nid: node_text(n).lower() for nid, n in nodes.items()}
    def has(pattern):
        return [nid for nid, t in texts.items() if re.search(pattern, t)]
    return {
        'priority_first_candidates': has(r'(start with|first step|first prototype|build .{0,30}first|prioriti[sz]|begin with|starting point)'),
        'graph_temperature_first': has(r'(start with|first|prioriti[sz]|begin with|starting point)[^.]{0,120}(graph|temperature)|(graph|temperature)[^.]{0,120}(start with|first step|first build|prioriti[sz]|starting point)'),
        'bursts_detail': has(r'burst'),
        'dormant_not_bad': has(r"(cold|dormant)[^.]{0,80}(not (necessarily )?(bad|wrong|stale)|isn.t (necessarily )?bad|just dormant)"),
        'audio_off_by_default_strengthened': has(r'(non-negotiable|off[- ]by[- ]default|must be off|mandatory)'),
        'd3_agreed': has(r'd3[^.]{0,80}(agreed|decided|confirmed|settled)|(agreed|decided|confirmed|settled)[^.]{0,80}d3'),
        'arc_mentions_first_priority': bool(re.search(r'(first|priorit|start with)', continuity_text.lower())),
    }


def arc_lines(windows):
    out = []
    for w in windows:
        final = read(w / 'result.json')['result'].get('final_text', '') or ''
        m = ARC_RE.search(final)
        out.append((m.group(1).strip() if m else '').replace('\n', ' '))
    return out


def analyze(cell):
    conf = CELLS[cell]
    out = conf['out']
    packets = out / 'whole_memory_review'
    packets.mkdir(exist_ok=True)
    reports, windows_exec, revisions = [], [], []
    for arm, repeat, corpus, folder, windows in sequences(cell):
        fixture = read(conf['fixtures'] / (corpus + '.json'))
        sources = {role: '\n'.join(t[role] for w in fixture['windows'] for t in w['turns']) for role in ('other', 'me')}
        nodes = {k: v for k, v in read(windows[-1] / 'nodes_after.json').items() if isinstance(v, dict) and v.get('title')}
        authored = authored_fields(folder)
        adjacency = defaultdict(set)
        edges, edge_words = [], []
        for nid, node in nodes.items():
            for e in get_edges(node):
                edges.append((nid, e['target_id'], e['relation'], e['description']))
                edge_words.append(words(e['description']))
                if e['target_id'] in nodes:
                    adjacency[nid].add(e['target_id']); adjacency[e['target_id']].add(nid)
        isolates = sorted(nid for nid in nodes if not adjacency[nid])
        field_words = defaultdict(list); quote_mismatch = 0; quotes = 0
        for nid, node in nodes.items():
            for k in FIELDS:
                v = field(node, k)
                if v not in (None, '', [], {}):
                    field_words[k].append(words(v))
            for key, role in (('their_raw_quote', 'other'), ('my_raw_quote', 'me')):
                q = field(node, key)
                if isinstance(q, str) and q:
                    quotes += 1
                    if ' '.join(q.split()) not in ' '.join(sources[role].split()):
                        quote_mismatch += 1
        usage = Counter(); exec_rows = []
        continuity_text = ''
        for wn, w in enumerate(windows, 1):
            before, after = read(w / 'nodes_before.json'), read(w / 'nodes_after.json')
            result = read(w / 'result.json'); calls = read(w / 'calls.json')
            usage.update(result['usage'])
            row = {'arm': arm, 'repeat': repeat, 'corpus': corpus, 'window': wn, 'created': len(set(after) - set(before)),
                   'nodes_after': len(after), 'rounds': result['result'].get('rounds'),
                   'tool_sequence': [c['call']['tool'] for c in calls], 'output_tokens': result['result'].get('output_tokens'),
                   'truncations': result['result'].get('truncations'), 'error': result['result'].get('error'),
                   'final_done': (result['result'].get('final_text') or '').rstrip().endswith('DONE'),
                   'partial_failures': sum(1 for c in calls if isinstance(c['result'], dict) and isinstance(c['result'].get('result'), dict)
                                           and (c['result']['result'].get('failed') or c['result']['result'].get('connect_to_failures')))}
            exec_rows.append(row); windows_exec.append(row)
            for nid in set(before) & set(after):
                if not (isinstance(before[nid], dict) and isinstance(after[nid], dict)):
                    continue
                changes = {k: {'before': field(before[nid], k), 'after': field(after[nid], k)}
                           for k in FIELDS if field(before[nid], k) != field(after[nid], k)}
                if changes:
                    revisions.append({'arm': arm, 'repeat': repeat, 'corpus': corpus, 'window': wn, 'node_id': nid, 'changes': changes})
            if (w / 'next_continuity.txt').exists():
                continuity_text = (w / 'next_continuity.txt').read_text()
        arcs = arc_lines(windows)
        report = {'arm': arm, 'repeat': repeat, 'corpus': corpus, 'windows': len(windows), 'nodes': len(nodes),
                  'node_words': sum(sum(v) for v in field_words.values()),
                  'field_nodes': {k: len(v) for k, v in field_words.items()}, 'field_words': {k: sum(v) for k, v in field_words.items()},
                  'thought_nodes': len(field_words.get('thought', [])), 'relations': len(edges), 'edge_words': sum(edge_words),
                  'isolates': isolates, 'quotes': quotes, 'quote_mismatch': quote_mismatch,
                  'revised_existing_nodes': len({r['node_id'] for r in revisions if r['arm'] == arm and r['repeat'] == repeat and r['corpus'] == corpus}),
                  'field_change_events': dict(Counter(k for r in revisions if r['arm'] == arm and r['repeat'] == repeat and r['corpus'] == corpus for k in r['changes'])),
                  'explicitly_authored_field_counts': dict(Counter(k for nid in nodes for k in authored[nid])),
                  'usage': dict(usage), 'arc_lines': arcs, 'arc_final_chars': len(continuity_text.split('Session arc:')[-1]) if 'Session arc:' in continuity_text else 0,
                  'errors': [r for r in exec_rows if r['error'] or r['truncations'] or not r['final_done'] or r['partial_failures']]}
        if corpus == 'creative_design':
            report['spotlight'] = spotlight_creative(nodes, continuity_text)
        reports.append(report)
        packet = [f'# {arm} / {repeat} / {corpus}: whole saved memory', '', f'source: {fixture["source_id"]} ({fixture["source_kind"]})', '']
        for nid, node in nodes.items():
            packet.append(f'## {nid}: {node["title"]} [{node.get("type")}]')
            for k in FIELDS:
                v = field(node, k)
                if v not in (None, '', [], {}):
                    packet.append(k + ': ' + (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)))
            for e in get_edges(node):
                packet.append(f'edge {e["relation"]} -> {e["target_id"]} ({e["target_title"]}): {e["description"]}')
            packet.append('')
        for wn, w in enumerate(windows, 1):
            final = read(w / 'result.json')['result'].get('final_text', '') or ''
            packet.extend([f'# Window {wn} closing text', final, ''])
            if (w / 'next_continuity.txt').exists():
                packet.extend([f'# Window {wn} carried continuity', (w / 'next_continuity.txt').read_text(), ''])
        (packets / f'{arm}_{repeat}_{corpus}.md').write_text('\n'.join(packet) + '\n')
    table = defaultdict(lambda: Counter())
    for r in reports:
        t = table[(r['corpus'], r['arm'])]
        t['sequences'] += 1; t['nodes'] += r['nodes']; t['node_words'] += r['node_words']; t['thought_nodes'] += r['thought_nodes']
        t['relations'] += r['relations']; t['edge_words'] += r['edge_words']; t['isolates'] += len(r['isolates'])
        t['revised_existing_nodes'] += r['revised_existing_nodes']; t['output_tokens'] += r['usage'].get('output', 0)
        t['quote_mismatch'] += r['quote_mismatch']; t['quotes'] += r['quotes']; t['errors'] += len(r['errors'])
        t['arcs_nonempty'] += sum(1 for a in r['arc_lines'] if a)
    lines = ['| corpus | arm | seqs | nodes | node words | thoughts | relations | edge words | isolates | revised existing | quotes (mismatch) | arcs nonempty | output tokens | errors |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (corpus, arm), t in sorted(table.items()):
        lines.append(f"| {corpus} | {arm} | {t['sequences']} | {t['nodes']} | {t['node_words']} | {t['thought_nodes']} | {t['relations']} | {t['edge_words']} | {t['isolates']} | {t['revised_existing_nodes']} | {t['quotes']} ({t['quote_mismatch']}) | {t['arcs_nonempty']} | {t['output_tokens']} | {t['errors']} |")
    (out / 'analysis_inventory.json').write_text(json.dumps({'sequences': reports, 'windows': windows_exec, 'field_revisions': revisions,
                                                             'table': [dict(corpus=c, arm=a, **t) for (c, a), t in sorted(table.items())]}, indent=2, ensure_ascii=False, default=str) + '\n')
    (out / 'ANALYSIS-TABLE.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    for r in reports:
        if r.get('spotlight'):
            print(r['arm'], r['repeat'], 'spotlight', json.dumps({k: (v if isinstance(v, bool) else len(v)) for k, v in r['spotlight'].items()}))
        if r['errors']:
            print('ERRORS', r['arm'], r['repeat'], r['corpus'], r['errors'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', choices=list(CELLS), required=True)
    analyze(parser.parse_args().cell)
