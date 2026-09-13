"""Retrieved-subset answer test over the kept final brains of the transfer cell.

Whole-memory review credits everything the graph holds; a future reader gets
only what recall returns. For every (arm, repeat, LongMemEval corpus) this
opens the kept final brain in an isolated copy, runs the real recall on the
item's question, renders the top hits the way a reader would see them, asks
Sonnet 4.6 to answer from those memories alone, and judges the answer against
the gold. It also records whether any retrieved node carries the gold string,
so retrieval reach and answer synthesis can be read apart.

    ./dev python3 eval/fixtures/s1e_guide_v3_3_2026-09-11/downstream.py --run
"""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CELLS = {'transfer': ROOT / 'eval/results/s1e_v34_transfer_2026-09-12', 'regression': ROOT / 'eval/results/s1e_v34_regression_2026-09-12'}
OUT = CELLS['transfer']
DOWN = OUT / 'downstream'
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
MODEL = 'claude-sonnet-4-6'
TOP = 5
ANSWER_SYSTEM = ('You answer a question using ONLY the memory notes supplied. Each note is what an assistant '
                 'stored from earlier conversations with this user. If the notes do not contain the answer, say '
                 '"The memories do not contain this." Answer in one or two sentences; state the specific value, '
                 'name or date when the notes hold it, and quote the note phrase you relied on.')
JUDGE_SYSTEM = ('You grade an answer against a reference. Reply on the first line with exactly CORRECT, PARTIAL '
                'or INCORRECT — CORRECT when the response conveys the reference answer (paraphrase and extra '
                'detail are fine; a response admitting the memories lack the answer is INCORRECT); PARTIAL when '
                'it conveys part of a multi-part reference. Then one sentence of justification.')


def render(node):
    """What a reader gets: every populated retrieval surface of the hit, plus its edges by title."""
    meta = node.get('_metadata') or node.get('metadata') or {}
    def field(key):
        return node.get(key) or meta.get(key)
    lines = [f"[{node.get('type')}] {node.get('title')} (id:{str(node.get('id'))[:8]})"]
    for key in ('content', 'situation', 'question', 'reasoning', 'thought', 'their_raw_quote', 'my_raw_quote', 'event_time'):
        value = field(key)
        if value:
            lines.append(f'  {key}: {value}')
    for edge in node.get('connections') or []:
        for rel in edge.get('relations') or [edge]:
            if rel.get('relation') in ('co_anchored', 'community_member'):
                continue
            lines.append(f"  edge {rel.get('relation')} -> {edge.get('title')}: {rel.get('description') or ''}")
    return '\n'.join(lines)


def call(client, system, user, max_tokens=500):
    for attempt in range(4):
        try:
            resp = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                                          messages=[{'role': 'user', 'content': user}])
            return ''.join(b.text for b in resp.content if hasattr(b, 'text')), resp.usage.input_tokens, resp.usage.output_tokens
        except Exception as error:  # transient API errors
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def one(client, arm, repeat, corpus, fixture):
    from isolated_brain import IsolatedBrain
    kept = OUT / arm / f'repeat{repeat}' / corpus / 'final_brain'
    gold = fixture['prior_gold']
    question, answer = gold['question'], str(gold['answer'])
    with IsolatedBrain(production_dir=str(kept), cleanup=True, load_env=True) as env:
        result = env.brain.recall(question, limit=TOP, mark_accessed=False)
        lean = result.get('results') or []
        rich = env.brain.get_node([str(n['id']) for n in lean]) if lean else {}
        hits = [rich.get(str(n['id'])) or rich.get(str(n['id'])[:8]) or n for n in lean]  # canonical nodes, recall order
        rendered = [render(n) for n in hits]
        total_nodes = len(env.brain.get_node(json.loads((OUT / arm / f'repeat{repeat}' / corpus / 'final_brain.json').read_text())['node_ids']) or {})
    contains = any(answer.lower() in (json.dumps(n, ensure_ascii=False, default=str)).lower() for n in hits) if answer else None
    user = ('MEMORY NOTES (top %d by recall):\n\n%s\n\nQUESTION (asked on %s): %s' % (
        TOP, '\n\n---\n\n'.join(rendered) or '(recall returned nothing)', gold.get('question_date'), question))
    reply, ai, ao = call(client, ANSWER_SYSTEM, user)
    verdict, ji, jo = call(client, JUDGE_SYSTEM, f'Question: {question}\nReference answer: {answer}\nResponse: {reply}', 200)
    return {'arm': arm, 'repeat': repeat, 'corpus': corpus, 'question_type': gold.get('question_type'),
            'question': question, 'gold': answer, 'retrieved': [{'id': str(n.get('id'))[:8], 'title': n.get('title')} for n in hits],
            'retrieved_render': rendered, 'gold_string_in_retrieved': contains, 'final_nodes': total_nodes,
            'answer': reply, 'judge': verdict.strip(), 'verdict': verdict.strip().split()[0].strip('.:').upper() if verdict.strip() else 'NONE',
            'usage': {'answer_in': ai, 'answer_out': ao, 'judge_in': ji, 'judge_out': jo}}


def run():
    import anthropic
    from isolated_brain import _load_env
    _load_env()
    client = anthropic.Anthropic()
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if json.loads((OUT / 'completion.json').read_text())['status'] != 'complete':
        raise RuntimeError('Transfer cell not complete')
    DOWN.mkdir(exist_ok=True)
    rows = []
    for corpus in sorted(manifest['corpora']):
        fixture = json.loads((OUT / (corpus + '.json')).read_text())
        if 'question' not in fixture['prior_gold']:
            continue
        for arm in manifest['arms']:
            for repeat in range(1, 4):
                path = DOWN / f'{arm}_repeat{repeat}_{corpus}.json'
                if path.exists():
                    rows.append(json.loads(path.read_text())); continue
                row = one(client, arm, repeat, corpus, fixture)
                path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + '\n')
                rows.append(row)
                print(arm, repeat, corpus, row['verdict'], 'gold_in_retrieved', row['gold_string_in_retrieved'], flush=True)
    summary = {}
    for row in rows:
        key = f"{row['arm']}|{row['corpus']}"
        s = summary.setdefault(key, {'arm': row['arm'], 'corpus': row['corpus'], 'type': row['question_type'], 'CORRECT': 0, 'PARTIAL': 0, 'INCORRECT': 0, 'gold_in_retrieved': 0, 'repeats': 0})
        s['repeats'] += 1
        s[row['verdict'] if row['verdict'] in s else 'INCORRECT'] += 1
        s['gold_in_retrieved'] += 1 if row['gold_string_in_retrieved'] else 0
    (DOWN / 'summary.json').write_text(json.dumps({'model': MODEL, 'top_k': TOP, 'rows': rows, 'summary': sorted(summary.values(), key=lambda s: (s['corpus'], s['arm']))}, indent=2, ensure_ascii=False) + '\n')
    for s in sorted(summary.values(), key=lambda s: (s['corpus'], s['arm'])):
        print(f"{s['corpus']:14} {s['arm']:20} correct {s['CORRECT']}/{s['repeats']} partial {s['PARTIAL']} gold_in_retrieved {s['gold_in_retrieved']}/{s['repeats']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--cell', choices=list(CELLS), default='transfer')
    args = parser.parse_args()
    OUT = CELLS[args.cell]; DOWN = OUT / 'downstream'
    if args.run:
        run()
    else:
        parser.error('--run')
