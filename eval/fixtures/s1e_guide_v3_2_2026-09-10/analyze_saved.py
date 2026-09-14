"""Describe saved V3.2 outputs alongside the existing V3.1 inventory.

No model or database access. Reuse the original word/field census so metrics
are comparable; preserve every prior result and packet unchanged.
"""
from collections import Counter
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'eval/results/s1e_v32_semantic_sanity_2026-09-10'
PRIOR = ROOT / 'eval/results/s1e_v31_cross_corpus_2026-09-08'


def read(path):
    return json.loads(path.read_text())


def main():
    if not (OUT / 'completion.json').exists():
        raise RuntimeError('The cell must complete before the final census')
    spec = importlib.util.spec_from_file_location('_quality_inventory',
        ROOT / 'eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/quality_inventory.py')
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    inventory.OUT = OUT
    inventory.main()
    baseline = read(PRIOR / 'quality_inventory.json')
    candidate = read(OUT / 'quality_inventory.json')
    reports = [s for s in baseline['sequences']
               if s['arm'] == 'v3_1_titles' and s['corpus'] == 'creative_design']
    reports += candidate['sequences']
    windows, revisions = [], []
    for arm, location in [('v3_1_titles', PRIOR), ('v3_2_titles', OUT)]:
        for repeat in range(1, 4):
            sequence = location / arm / ('repeat' + str(repeat)) / 'creative_design'
            for w in range(1, 4):
                folder = sequence / ('window' + str(w))
                before, after = read(folder / 'nodes_before.json'), read(folder / 'nodes_after.json')
                result, calls = read(folder / 'result.json')['result'], read(folder / 'calls.json')
                windows.append({'arm': arm, 'repeat': repeat, 'window': w,
                    'created': len(set(after) - set(before)), 'nodes_after': len(after),
                    'rounds': result['rounds'], 'tool_sequence': [c['call']['tool'] for c in calls],
                    'output_tokens': result['output_tokens'], 'truncations': result.get('truncations'),
                    'error': result.get('error'), 'final_done': result.get('final_text', '').rstrip().endswith('DONE'),
                    'path': str(folder.relative_to(ROOT))})
                for nid in set(before) & set(after):
                    changes = {key: {'before': inventory.field(before[nid], key),
                                     'after': inventory.field(after[nid], key)}
                               for key in inventory.FIELDS
                               if inventory.field(before[nid], key) != inventory.field(after[nid], key)}
                    if changes:
                        revisions.append({'arm': arm, 'repeat': repeat, 'window': w,
                                          'node_id': nid, 'changes': changes,
                                          'path': str(folder.relative_to(ROOT))})
    value = {'method': 'Descriptive saved-artifact census; no automatic semantic score.',
             'sequences': reports, 'windows': windows, 'field_revisions': revisions}
    (OUT / 'comparison_inventory.json').write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    for s in reports:
        print(s['arm'], s['repeat'], 'nodes', s['nodes'], 'words', s['node_text_words']['total'],
              'revised', s['actual_revised_nodes'], 'authored_fields', s['explicitly_authored_field_counts'])
    print('candidate_ending_errors', [w for w in windows if w['arm'] == 'v3_2_titles'
                                     and (w['error'] or w['truncations'] or not w['final_done'])])
    print('field_change_events', dict(Counter((r['arm'], key) for r in revisions for key in r['changes'])))


if __name__ == '__main__':
    main()
