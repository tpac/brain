"""Read saved cross-corpus artifacts. No model calls, database access or scoring.

Counts are observations; DEEP-REVIEW.md supplies source-based interpretation.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'eval/results/s1e_v31_cross_corpus_2026-09-08'
FIELDS = ('type', 'title', 'content', 'situation', 'question', 'reasoning',
          'thought', 'their_raw_quote', 'my_raw_quote', 'event_time',
          'evolution_status', 'confidence')
ARMS = ('v3_titles', 'v3_titles_new_tools', 'v3_1_titles')


def read(path):
    return json.loads(path.read_text())


def fields(node):
    return {k: node.get(k, node.get('_metadata', {}).get(k)) for k in FIELDS}


def rel(path):
    return str(path.relative_to(ROOT))


def main():
    summary = read(OUT / 'quality_summary.json')
    advice = {(s['arm'], s['repeat']): set(s['node_ids'])
              for s in summary['standalone_advice']}
    windows, sequences, revisions = [], [], []
    for arm in ARMS:
        for repeat in ('repeat1', 'repeat2', 'repeat3'):
            for corpus in ('creative_design', 'longmem_unseen'):
                sequence = OUT / arm / repeat / corpus
                created, advice_created, counts = [], [], Counter()
                for w in range(1, 4):
                    folder = sequence / f'window{w}'
                    before = read(folder / 'nodes_before.json')
                    after = read(folder / 'nodes_after.json')
                    result = read(folder / 'result.json')['result']
                    new_ids = sorted(set(after) - set(before))
                    created.append(len(new_ids))
                    advice_created.append(len(set(new_ids) & advice.get((arm, repeat), set())))
                    record = dict(arm=arm, repeat=repeat, corpus=corpus, window=w,
                                  path=rel(folder), before_nodes=len(before),
                                  after_nodes=len(after), created_ids=new_ids,
                                  created_count=len(new_ids), rounds=result['rounds'],
                                  write_actions=result['write_actions'],
                                  output_tokens=result['output_tokens'],
                                  truncations=result.get('truncations', []),
                                  error=result.get('error'),
                                  captured_requests=len(list(folder.glob('round*.json'))),
                                  final_has_done='DONE' in result.get('final_text', ''))
                    windows.append(record)
                    for nid in sorted(set(before) & set(after)):
                        old, new = fields(before[nid]), fields(after[nid])
                        delta = {k: {'before': old[k], 'after': new[k]}
                                 for k in FIELDS if old[k] != new[k]}
                        if delta:
                            counts.update(delta.keys())
                            revisions.append(dict(arm=arm, repeat=repeat, corpus=corpus,
                                                  window=w, node_id=nid, changes=delta,
                                                  path=rel(folder)))
                sequences.append(dict(arm=arm, repeat=repeat, corpus=corpus,
                                      created_per_window=created, final_nodes=len(after),
                                      standalone_advice_created_per_window=(advice_created
                                          if corpus == 'longmem_unseen' else None),
                                      changed_field_events=dict(counts),
                                      initial_request_sha256=hashlib.sha256(
                                          (sequence / 'window1/round000.json').read_bytes()).hexdigest()))

    selected = [
        ('v3_1_titles', 'repeat3', 'creative_design', 'ecf214b5'),
        ('v3_1_titles', 'repeat1', 'creative_design', '3eb6c5ea'),
        ('v3_titles', 'repeat1', 'longmem_unseen', '4e2e53e8'),
        ('v3_titles', 'repeat3', 'longmem_unseen', 'b9958b7d'),
        ('v3_titles_new_tools', 'repeat3', 'creative_design', '3b9bcdab'),
        ('v3_1_titles', 'repeat1', 'creative_design', '5063c0cf'),
        ('v3_titles_new_tools', 'repeat2', 'longmem_unseen', '09470da3'),
        ('v3_1_titles', 'repeat2', 'creative_design', '9c606c81'),
        ('v3_1_titles', 'repeat3', 'creative_design', 'fc1c577e'),
        ('v3_titles', 'repeat1', 'creative_design', '4078d1c7'),
    ]
    lifecycles = []
    for arm, repeat, corpus, nid in selected:
        stages = []
        for w in range(1, 4):
            folder = OUT / arm / repeat / corpus / f'window{w}'
            node = read(folder / 'nodes_after.json').get(nid)
            if node is None:
                continue
            result = read(folder / 'result.json')['result']
            stages.append(dict(window=w, path=rel(folder), node=node,
                               mentions=[line for text in result['round_texts']
                                         for line in text.splitlines() if nid in line],
                               final_text=result.get('final_text', '')))
        lifecycles.append(dict(arm=arm, repeat=repeat, corpus=corpus,
                               node_id=nid, stages=stages))

    dossier = dict(
        method='Read-only saved-artifact analysis. No composite quality score. '
               'Standalone advice IDs are the prior hand-reviewed classification.',
        windows=windows, sequences=sequences, actual_field_revisions=revisions,
        selected_lifecycles=lifecycles)
    (OUT / 'deep_evidence.json').write_text(json.dumps(dossier, indent=2, ensure_ascii=False) + '\n')
    for s in sequences:
        if s['corpus'] == 'longmem_unseen':
            print(s['arm'], s['repeat'], 'created', s['created_per_window'],
                  'final', s['final_nodes'], 'standalone advice created',
                  s['standalone_advice_created_per_window'])
    print('windows', len(windows), 'zero-create', sum(w['created_count'] == 0 for w in windows),
          'truncated', sum(bool(w['truncations']) for w in windows),
          'errors', sum(bool(w['error']) for w in windows))


if __name__ == '__main__':
    main()
