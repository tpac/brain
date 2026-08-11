"""connect_to A/B over captured scribe prompts — the Option D gate.

Replays frozen pool60 encoding-prompt captures (real conversation + catalog +
journal inputs) against two s1e prompt versions and scores ONLY the emitted
tool calls — no brain, no dispatch, no answer phase. The capture is
self-contained: a valid catalog-target id must appear in that capture's own
rendered catalog, so resolution is checkable offline.

Per connect_to entry, the target's `title` slot is classified:
  id_ok           hex id present in the capture's catalog
  id_bad          hex id NOT in the capture's catalog (would fail loudly)
  placeholder     literal `<...>` leaked from the prompt examples
  sibling_title   matches another node created in the same call (Pass 1 path)
  catalog_title   matches a catalog title (old Pass 3 path — what D retires)
  title_from_input  exact title visible elsewhere in the input (edge lines,
                  scout notes) — real node, resolves via Pass 3 in production
  unresolved      matches nothing visible — hub synthesis / confabulation

Reads are stubbed identically across arms (no brain to serve them); the run
stops at the first round containing a write tool, which is the scored round.

Usage:
  ./dev python3 eval/longmem/connect_ab.py --arms 30,31 --items 3          # smoke
  ./dev python3 eval/longmem/connect_ab.py --arms 30,31 --items 40
  ./dev python3 eval/longmem/connect_ab.py --arms 30,31                    # all 172
"""
import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CAPTURE_GLOB = os.path.expanduser(
    '~/AgentsContext/eval-corpus/0a9baa/pooled/brain-encoding-prompt-*.json')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'reports', 'connect_ab')
MODEL = 'claude-sonnet-4-6'
MAX_ROUNDS = 3
WRITE_TOOLS = {'remember_batch', 'revise_batch', 'brain_batch', 'connect_batch'}
HEX_RE = re.compile(r'^[0-9a-fA-F]{8,}$')
# Any id: occurrence in the input counts as copyable — the encoder sees ids
# in catalog headers, edge lines, and scout/journal listings alike, and all
# of them resolve at Pass 0. A header-only pattern false-flagged real copies.
CATALOG_ID_RE = re.compile(r'id:([0-9a-f]{6,8})\b')
CATALOG_TITLE_RE = re.compile(r'\[\w+\] "([^"]+)" \(id:')


def fetch_template(version):
    from servers.daemon_client import send_command
    r = send_command('get_interaction', {'name': 's1e', 'version': version})
    if not r.get('ok'):
        raise RuntimeError('cannot fetch s1e v%d: %s' % (version, r))
    res = r['result']
    params = res.get('parameters') or '{}'
    if isinstance(params, str):
        params = json.loads(params or '{}')
    return res['template'], params.get('effort')


def tool_schemas():
    from servers import brain_mcp
    names = {'remember_batch', 'revise_batch', 'brain_batch',
             'connect_batch', 'recall_batch', 'get_nodes'}
    return [{'name': t['name'], 'description': t['description'],
             'input_schema': t['inputSchema']}
            for t in brain_mcp.TOOLS if t['name'] in names]


def extract_connect_entries(tool_call):
    """Yield (node_title, connect_entry) plus the set of created titles."""
    inp = tool_call['input']
    nodes = []
    if tool_call['name'] == 'brain_batch':
        nodes = [op for op in (inp.get('operations') or [])
                 if isinstance(op, dict) and op.get('op') == 'remember']
    elif tool_call['name'] == 'remember_batch':
        nodes = [n for n in (inp.get('nodes') or []) if isinstance(n, dict)]
    created = {str(n.get('title') or '').strip().lower() for n in nodes}
    entries = []
    for n in nodes:
        for e in (n.get('connect_to') or []):
            if isinstance(e, dict):
                entries.append((n.get('title'), e))
    return created, entries


def classify(target, created_titles, catalog_ids, catalog_titles):
    t = str(target or '').strip()
    if t.startswith('<'):
        return 'placeholder'
    if HEX_RE.fullmatch(t):
        tl = t.lower()
        return 'id_ok' if any(cid.startswith(tl[:6]) and tl.startswith(cid[:6])
                              or cid == tl[:len(cid)] or tl == cid
                              for cid in catalog_ids) else 'id_bad'
    tl = t.lower()
    if tl in created_titles:
        return 'sibling_title'
    if tl in catalog_titles:
        return 'catalog_title'
    return 'unresolved'


def run_item(client, path, system_prompt, effort, tools):
    cap = json.load(open(path))
    catalog_ids = {m.lower() for m in CATALOG_ID_RE.findall(cap['user_content'])}
    catalog_titles = {t.strip().lower()
                      for t in CATALOG_TITLE_RE.findall(cap['user_content'])}
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': cap['user_preamble'],
         'cache_control': {'type': 'ephemeral'}},
        {'type': 'text', 'text': cap['user_content']},
    ]}]
    kwargs = dict(
        model=MODEL, max_tokens=8000,
        system=[{'type': 'text', 'text': system_prompt,
                 'cache_control': {'type': 'ephemeral'}}],
        tools=tools, messages=messages)
    if effort:
        kwargs['output_config'] = {'effort': effort}

    input_text = cap['user_content']
    rows, usage = [], {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0}
    for rnd in range(1, MAX_ROUNDS + 1):
        resp = client.messages.create(**kwargs)
        u = resp.usage
        usage['input'] += u.input_tokens
        usage['output'] += u.output_tokens
        usage['cache_read'] += getattr(u, 'cache_read_input_tokens', 0) or 0
        usage['cache_write'] += getattr(u, 'cache_creation_input_tokens', 0) or 0
        calls = [{'name': b.name, 'input': b.input, 'id': b.id}
                 for b in resp.content if b.type == 'tool_use']
        writes = [c for c in calls if c['name'] in WRITE_TOOLS]
        if writes:
            parsed = [extract_connect_entries(c) for c in writes]
            round_created = set().union(*(p[0] for p in parsed))
            for created, entries in parsed:
                for node_title, e in entries:
                    cls = classify(e.get('title'), created,
                                   catalog_ids, catalog_titles)
                    if (cls == 'unresolved'
                            and str(e.get('title') or '').strip().lower()
                            in round_created):
                        # Created by a DIFFERENT call this round: not a
                        # sibling (Pass 1 scope is one call), not in the
                        # rendered catalog — production limps through FTS.
                        cls = 'sibling_cross_call'
                    elif (cls == 'unresolved'
                            and str(e.get('title') or '').strip() in input_text):
                        # Exact title visible in a non-catalog input surface
                        # (edge line, scout note) — a real node; resolves via
                        # Pass 3 in production. Not confabulation.
                        cls = 'title_from_input'
                    rows.append({
                        'node': node_title, 'target': e.get('title'),
                        'relation': e.get('relation'), 'class': cls})
            rec = {'rows': rows, 'rounds': rnd, 'usage': usage,
                   'write_calls': len(writes), 'status': 'ok',
                   'created_titles': sorted(round_created)}
            if os.environ.get('CONNECT_AB_DUMP'):
                rec['raw_writes'] = [{'name': c['name'], 'input': c['input']}
                                     for c in writes]
            return rec
        if not calls or resp.stop_reason != 'tool_use':
            return {'rows': [], 'rounds': rnd, 'usage': usage,
                    'write_calls': 0, 'status': 'no_write'}
        # Reads before writes: stub identically for both arms, continue.
        messages.append({'role': 'assistant', 'content': resp.content})
        messages.append({'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': c['id'],
             'content': '(replay: result unavailable — proceed with the '
                        'catalog and timeline you already have)'}
            for c in calls]})
        kwargs['messages'] = messages
    return {'rows': [], 'rounds': MAX_ROUNDS, 'usage': usage,
            'write_calls': 0, 'status': 'no_write_by_max_rounds'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='30,31')
    ap.add_argument('--items', type=int, default=0, help='0 = all')
    ap.add_argument('--label', default=time.strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--only', default='',
                    help='comma-separated filename substrings to select items')
    args = ap.parse_args()

    import threading
    from concurrent.futures import ThreadPoolExecutor
    import anthropic
    client = anthropic.Anthropic()
    paths = sorted(glob.glob(CAPTURE_GLOB))
    if args.only:
        frags = [f for f in args.only.split(',') if f]
        paths = [p for p in paths if any(f in os.path.basename(p) for f in frags)]
    if args.items:
        step = max(1, len(paths) // args.items)
        paths = paths[::step][:args.items]
    tools = tool_schemas()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, '%s.jsonl' % args.label)

    arms = [int(a) for a in args.arms.split(',')]
    templates = {v: fetch_template(v) for v in arms}
    print('items=%d arms=%s workers=%d out=%s' % (
        len(paths), arms, args.workers, out_path))

    out = open(out_path, 'a')
    lock = threading.Lock()
    done = [0]
    total = len(paths) * len(arms)

    def one(path, v):
        qid = os.path.basename(path)
        tmpl, effort = templates[v]
        try:
            r = run_item(client, path, tmpl, effort, tools)
        except Exception as e:
            r = {'rows': [], 'status': 'error: %s' % e,
                 'rounds': 0, 'usage': {}, 'write_calls': 0}
        counts = {}
        for row in r['rows']:
            counts[row['class']] = counts.get(row['class'], 0) + 1
        with lock:
            out.write(json.dumps({'item': qid, 'arm': v, **r}) + '\n')
            out.flush()
            done[0] += 1
            print('[%d/%d] v%d %s %s edges=%d %s' % (
                done[0], total, v, qid[:48], r['status'],
                len(r['rows']), counts or ''), flush=True)

    # Prime each arm's system-prompt cache with one serial call, then fan
    # out — a cold parallel first wave would pay cache_write per worker.
    for v in arms:
        one(paths[0], v)
    rest = [(p, v) for p in paths[1:] for v in arms]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda t: one(*t), rest))
    out.close()

    # Summary
    per_arm = {v: {} for v in arms}
    edges = {v: 0 for v in arms}
    for line in open(out_path):
        rec = json.loads(line)
        v = rec['arm']
        if v not in per_arm:
            continue
        for row in rec['rows']:
            per_arm[v][row['class']] = per_arm[v].get(row['class'], 0) + 1
            edges[v] += 1
    print('\n=== summary (%s) ===' % out_path)
    for v in arms:
        print('arm v%d: %d edges  %s' % (v, edges[v], per_arm[v]))


if __name__ == '__main__':
    main()
