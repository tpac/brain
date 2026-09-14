"""Save a blind reviewer's final message verbatim from its agent transcript (JSONL) into the cell's
verdicts folder, with a header that records the pack and that the key stayed sealed.
Usage: save_verdict.py <agent.output jsonl> <verdicts/{corpus}_repeat{n}.md> <corpus> <n>"""
import json, sys, pathlib
src, dst, corpus, rep = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
last = None
for line in src.read_text().splitlines():
    try: o = json.loads(line)
    except Exception: continue
    m = o.get('message') or {}
    if o.get('type') == 'assistant' or m.get('role') == 'assistant':
        texts = [b.get('text', '') for b in (m.get('content') or []) if isinstance(b, dict) and b.get('type') == 'text']
        if texts and ''.join(texts).strip(): last = ''.join(texts)
if not last: raise SystemExit('no assistant text found in ' + str(src))
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(f'# Blind verdict — {corpus} repeat {rep} (reviewer saw A/B/C only; verbatim, unsealed key in ../key.json)\n\n' + last.strip() + '\n')
print(dst, len(last), 'chars')
