"""Reuse the established census; expose readable whole memories and deltas."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'eval/results/s1e_production_comparison_2026-09-11'
PRIOR = ROOT / 'eval/results/s1e_v32_semantic_sanity_2026-09-10'


def read(path):
    return json.loads(path.read_text())


def main():
    if not (OUT / 'completion.json').exists():
        raise RuntimeError('Wait for all nine production encodes before the final census')
    spec = importlib.util.spec_from_file_location('_quality_census', ROOT / 'eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/quality_inventory.py')
    census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(census)
    census.OUT = OUT
    census.main()
    prod, cand = read(OUT / 'quality_inventory.json'), read(PRIOR / 'quality_inventory.json')
    reports = prod['sequences'] + [s for s in cand['sequences'] if s['arm'] == 'v3_2_titles']
    compact = []
    for s in reports:
        compact.append({k: s[k] for k in ('arm', 'repeat', 'nodes', 'node_text_words', 'fields',
            'explicitly_authored_field_counts', 'edges', 'edge_words', 'operations', 'usage', 'actual_revised_nodes')})
    windows, revisions, compatibility = [], [], []
    packets = OUT / 'whole_memory_review'
    packets.mkdir(exist_ok=True)
    for arm, location in [('production_deployed', OUT), ('v3_2_titles', PRIOR)]:
        for repeat in range(1, 4):
            folder = location / arm / f'repeat{repeat}' / 'creative_design'
            packet = [f'# {arm} / repeat{repeat}: whole saved memory', '']
            for nid, node in read(folder / 'window3/nodes_after.json').items():
                packet.append(f'## {nid}: {node["title"]} [{node.get("type")}]')
                for field in census.FIELDS:
                    value = census.field(node, field)
                    if value not in (None, '', [], {}):
                        packet.append(field + ': ' + (value if isinstance(value, str) else json.dumps(value)))
                for edge in census.get_edges(node):
                    packet.append(f'edge {edge["relation"]} -> {edge["target_id"]} ({edge["target_title"]}): {edge["description"]}')
                packet.append('')
            for wn in range(1, 4):
                d = folder / f'window{wn}'
                before, after = read(d / 'nodes_before.json'), read(d / 'nodes_after.json')
                result, calls = read(d / 'result.json')['result'], read(d / 'calls.json')
                windows.append({'arm': arm, 'repeat': repeat, 'window': wn, 'created': len(set(after)-set(before)),
                    'nodes_after': len(after), 'rounds': result['rounds'], 'tool_sequence': [c['call']['tool'] for c in calls],
                    'output_tokens': result['output_tokens'], 'truncations': result.get('truncations'),
                    'error': result.get('error'), 'final_done': result.get('final_text', '').rstrip().endswith('DONE')})
                for nid in set(before) & set(after):
                    changes = {key: {'before': census.field(before[nid],key), 'after': census.field(after[nid],key)}
                               for key in census.FIELDS if census.field(before[nid],key) != census.field(after[nid],key)}
                    if changes:
                        revisions.append({'arm': arm, 'repeat': repeat, 'window': wn, 'node_id': nid, 'changes': changes})
                if arm == 'production_deployed':
                    for call in calls:
                        args, name = call['call']['args'], call['call']['tool']
                        operations = args.get('operations', []) if name == 'brain_batch' else args.get('revisions', []) if name == 'revise_batch' else []
                        for op in operations:
                            if name == 'brain_batch' and op.get('op') != 'revise':
                                continue
                            wider = [k for k in census.TEXT_FIELDS if k in op and not isinstance(op[k], str)]
                            if op.get('connect_to') or wider:
                                compatibility.append({'repeat': repeat, 'window': wn, 'op': op, 'wider_fields': wider})
                packet.extend([f'# Window {wn} actual continuity', (d / 'next_continuity.txt').read_text(), ''])
            (packets / f'{arm}_repeat{repeat}.md').write_text('\n'.join(packet)+'\n')
    (OUT / 'comparison_inventory.json').write_text(json.dumps({'method': 'Same descriptive census; author reviews semantic value in whole-memory packets.',
        'sequences': compact, 'windows': windows, 'field_revisions': revisions,
        'production_used_candidate_only_revision_shapes': compatibility}, indent=2, ensure_ascii=False)+'\n')
    print('ending_errors', [w for w in windows if w['error'] or w['truncations'] or not w['final_done']])
    print('production_wider_api_uses', len(compatibility))
    print('field_changes', dict(Counter((r['arm'], k) for r in revisions for k in r['changes'])))


if __name__ == '__main__':
    main()
