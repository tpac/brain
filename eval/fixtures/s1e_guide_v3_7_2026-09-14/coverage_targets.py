"""Objective coverage against a synthetic corpus's own ground truth.

For every (arm, repeat) final memory of a synthetic corpus that carries `ground_truth`
{encode_targets, decode_queries}: (1) a Sonnet 4.6 judge reads the memory's nodes (title,
type, content, situation) and, for each encode target, says whether it is held as its OWN
node, FOLDED inside another node, or ABSENT, citing ids; (2) each decode query runs through
the real recall door (top 5) and a Sonnet answer is judged against the expected topics.
Usage: coverage_targets.py --cell <results dir> --corpus <name> --arms a,b,c [--corpus-json path]
Writes <cell>/coverage/<corpus>.json and prints a table."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
MODEL = 'claude-sonnet-4-6'
JUDGE = ('You audit a memory store against a list of knowledge targets. For EACH target, reply with one line: '
         '<target index>: OWN <node id> | FOLDED <node id> | ABSENT — then a short reason. OWN means a node whose title or '
         'main claim is that target; FOLDED means the knowledge is present only inside a node about something else '
         '(content, situation or reasoning); ABSENT means no node carries it. Judge the substance, not the exact wording.')
ANSWER = ('You answer a question using ONLY the memory notes supplied. If the notes do not contain the answer, say "The memories do not contain this." '
          'Answer in one or two sentences and quote the note phrase you relied on.')
GRADE = ('You grade an answer against expected topics. Reply on the first line with COVERED if the answer conveys every expected topic, '
         'PARTIAL if some, MISSED if none or if it says the memories lack it. Then one sentence.')

def call(client, system, user, max_tokens=700):
    for attempt in range(4):
        try:
            r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system, messages=[{'role': 'user', 'content': user}])
            return ''.join(b.text for b in r.content if hasattr(b, 'text'))
        except Exception:
            if attempt == 3: raise
            time.sleep(5 * (attempt + 1))

def field(n, k): return n.get(k) if n.get(k) not in (None, '') else (n.get('_metadata') or {}).get(k)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cell', required=True); ap.add_argument('--corpus', required=True); ap.add_argument('--arms', required=True); ap.add_argument('--corpus-json')
    a = ap.parse_args(); cell = Path(a.cell).resolve(); arms = a.arms.split(',')
    import anthropic
    from isolated_brain import IsolatedBrain, _load_env
    _load_env(); client = anthropic.Anthropic()
    gt = json.loads(Path(a.corpus_json or (ROOT / 'eval/corpus' / f'{a.corpus}.json')).read_text())['ground_truth']
    targets, queries = gt['encode_targets'], gt['decode_queries']
    out_dir = cell / 'coverage'; out_dir.mkdir(exist_ok=True); rows = []
    for arm in arms:
        for rep in (1, 2, 3):
            seq = cell / arm / f'repeat{rep}' / a.corpus
            wins = sorted(seq.glob('window*/nodes_after.json'), key=lambda p: int(p.parent.name[6:]))
            if not wins: continue
            nodes = {k: v for k, v in json.loads(wins[-1].read_text()).items() if isinstance(v, dict) and v.get('title')}
            dump = '\n\n'.join(f"[{k}] ({field(n,'type')}) {field(n,'title')}\n  content: {field(n,'content')}\n  situation: {field(n,'situation') or ''}" for k, n in nodes.items())
            tlist = '\n'.join(f"{i+1}. ({t['type']}) {t['topic']}" for i, t in enumerate(targets))
            verdict = call(client, JUDGE, f'TARGETS:\n{tlist}\n\nMEMORY ({len(nodes)} nodes):\n{dump}')
            per_target = []
            for i, t in enumerate(targets):
                line = next((l for l in verdict.splitlines() if l.strip().startswith(f'{i+1}:') or l.strip().startswith(f'{i+1}.')), '')
                status = 'OWN' if 'OWN' in line.upper()[:40] else 'FOLDED' if 'FOLDED' in line.upper()[:40] else 'ABSENT' if 'ABSENT' in line.upper()[:40] else '?'
                per_target.append({'target': t['topic'], 'type': t['type'], 'status': status, 'line': line.strip()})
            qrows = []
            kept = seq / 'final_brain'
            if kept.exists() and (seq / 'final_brain.json').exists():
                with IsolatedBrain(production_dir=str(kept), cleanup=True, load_env=True) as env:
                    for q in queries:
                        res = env.brain.recall(q['query'], limit=5, mark_accessed=False); lean = res.get('results') or []
                        rich = env.brain.get_node([str(n['id']) for n in lean]) if lean else {}
                        hits = [rich.get(str(n['id'])) or rich.get(str(n['id'])[:8]) or n for n in lean]
                        notes = '\n\n---\n\n'.join(f"[{field(n,'type')}] {field(n,'title')}\n  content: {field(n,'content')}\n  situation: {field(n,'situation') or ''}\n  quote: {field(n,'their_raw_quote') or field(n,'my_raw_quote') or ''}" for n in hits) or '(recall returned nothing)'
                        ans = call(client, ANSWER, f'MEMORY NOTES (top 5 by recall):\n\n{notes}\n\nQUESTION: {q["query"]}', 400)
                        grade = call(client, GRADE, f"Question: {q['query']}\nExpected topics: {q['expected_topics']}\nAnswer: {ans}", 150)
                        qrows.append({'query': q['query'], 'expected': q['expected_topics'], 'retrieved': [str(field(n, 'title'))[:80] for n in hits], 'answer': ans, 'grade': grade.strip().split()[0].strip('.:').upper() if grade.strip() else '?'})
            rows.append({'arm': arm, 'repeat': rep, 'nodes': len(nodes), 'targets': per_target, 'judge_raw': verdict, 'queries': qrows})
            print(arm, rep, 'nodes', len(nodes), 'targets:', ' '.join(p['status'] for p in per_target), '| queries:', ' '.join(q['grade'] for q in qrows), flush=True)
    (out_dir / f'{a.corpus}.json').write_text(json.dumps({'corpus': a.corpus, 'targets': targets, 'queries': queries, 'rows': rows}, indent=1, ensure_ascii=False))
    print('\n| arm | repeat | nodes | ' + ' | '.join(t['topic'][:34] for t in targets) + ' | ' + ' | '.join(q['query'][:30] for q in queries) + ' |')
    for r in rows: print(f"| {r['arm']} | {r['repeat']} | {r['nodes']} | " + ' | '.join(p['status'] for p in r['targets']) + ' | ' + ' | '.join(q['grade'] for q in r['queries']) + ' |')

if __name__ == '__main__': main()
