"""Replay one captured encoder request and collect what the model does — no brain, no isolation.

A captured round payload (`payloads/<date>/s1e-ingest-<stop>-<n>/000-round_payload.json`, or a
cell window's `round000.json`) holds the exact request the encoder made: model, effort, system,
messages, tools. Replaying it asks the same model the same question and records the reply's text
and tool_use blocks — the cheapest way to probe "what does the encoder do on this exact window"
and to try prompt variations against one failure (swap `--system` for a candidate template's
assembled system prompt; swap `--gist` to edit the gist text inside the user content).

  ./dev python3 replay_payload.py <round_payload.json> [--repeats 3] [--system assembled.txt]
      [--sub 'OLD=>NEW' ...]   # exact-once text substitutions inside the user content (e.g. the gist)

Prints, per repeat: the text preamble (lists), every tool_use name with a one-line gist of its
input (op counts, node titles, revise targets), and whether the reply ended the run (no tool call).
Cost: one model call per repeat at the captured prompt size (~$0.05–0.15 on Sonnet 4.6).
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]


def gist_of_tool_use(name, inp):
    ops = inp.get('operations') or inp.get('nodes') or inp.get('revisions') or inp.get('edges') or []
    out = []
    for o in ops if isinstance(ops, list) else []:
        if not isinstance(o, dict):
            continue
        op = o.get('op') or ('revise' if 'node_id' in o and 'title' not in o and 'type' not in o else name.replace('_batch', ''))
        tgt = o.get('node_id') or o.get('title') or o.get('source_id') or ''
        fields = [k for k in o if k not in ('op', 'node_id', 'reason', 'type', 'connect_to')]
        out.append(f"{op}:{str(tgt)[:60]}[{','.join(fields)[:80]}]")
    if name in ('get_nodes', 'recall_batch'):
        out.append(str(inp)[:160])
    return ' | '.join(out) or str(inp)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('payload'); ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--system', help='file whose text replaces the captured system prompt')
    ap.add_argument('--sub', action='append', default=[], help="exact-once substitution 'OLD=>NEW' in the user content")
    ap.add_argument('--sub-file', nargs=2, action='append', default=[], metavar=('OLD_FILE', 'NEW_FILE'),
                    help='exact-once substitution in the user content, old and new text read from files (a gist arm)')
    ap.add_argument('--dump', help='directory: write each reply as repeatN.json (text blocks and full tool_use inputs) for scoring')
    ap.add_argument('--max-tokens', type=int, default=8000)
    ap.add_argument('--no-effort', action='store_true', help="drop the captured effort (the runtime sends output_config.effort; the V3.7 probe sent none)")
    a = ap.parse_args()
    p = json.loads(Path(a.payload).read_text())
    system = Path(a.system).read_text() if a.system else p['system']
    messages = json.loads(json.dumps(p['messages']))
    subs = [tuple(s.split('=>', 1)) for s in a.sub] + [(Path(o).read_text(), Path(n).read_text()) for o, n in a.sub_file]
    for old, new in subs:
        hits = 0
        for m in messages:
            if m['role'] != 'user':
                continue
            if isinstance(m['content'], str):
                hits += m['content'].count(old); m['content'] = m['content'].replace(old, new)
            else:
                for b in m['content']:
                    if b.get('type') == 'text':
                        hits += b['text'].count(old); b['text'] = b['text'].replace(old, new)
        if hits != 1:
            raise SystemExit(f'substitution {old[:40]!r} matched {hits} times, expected 1')
    # The capture stores tool NAMES; the schemas are the runtime's (encode._get_tool_schemas on this
    # checkout — the same shapes the run used, descriptions possibly newer; freeze-time schemas live in
    # the fixture's tools_live.json if the exact bytes matter).
    tools = p['tools']
    if tools and isinstance(tools[0], str):
        from servers.scales.s1 import encode
        by_name = {t['name']: t for t in encode._get_tool_schemas()}
        tools = [by_name[n] for n in p['tools'] if n in by_name]
    import anthropic
    from isolated_brain import _load_env
    _load_env(); client = anthropic.Anthropic()
    # The runtime rides `effort` as output_config (servers/scales/runner.py); mirror it so the probe is the run.
    extra = {} if (a.no_effort or not p.get('effort')) else {'output_config': {'effort': p['effort']}}
    print(f"model={p['model']} system={len(system):,} chars tools={[t['name'] for t in tools]} messages={len(messages)} effort={extra.get('output_config', {}).get('effort') or 'API default'}")
    for r in range(a.repeats):
        resp = client.messages.create(model=p['model'], max_tokens=a.max_tokens, system=system, messages=messages, tools=tools, **extra)
        texts = [b.text for b in resp.content if b.type == 'text']
        uses = [(b.name, b.input) for b in resp.content if b.type == 'tool_use']
        print(f"\n=== repeat {r + 1}: stop={resp.stop_reason} tool_calls={len(uses)} out_tokens={resp.usage.output_tokens}")
        if a.dump:
            Path(a.dump).mkdir(parents=True, exist_ok=True)
            (Path(a.dump) / f'repeat{r + 1}.json').write_text(json.dumps(
                {'stop_reason': resp.stop_reason, 'output_tokens': resp.usage.output_tokens, 'texts': texts,
                 'tool_uses': [{'name': n, 'input': i} for n, i in uses]}, indent=1, ensure_ascii=False))
        for t in texts:
            print('  TEXT:', t[:1500].replace('\n', '\n        '))
        for name, inp in uses:
            print(f'  TOOL_USE {name}: {gist_of_tool_use(name, inp)[:600]}')
        if not uses:
            print('  (no tool call — this reply would END the run with zero writes)')


if __name__ == '__main__':
    main()
