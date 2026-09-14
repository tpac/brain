"""Stale-surface detector over saved encodes — did a revise leave the old value where the
encoder was not looking? No model calls; a READING QUEUE, not a verdict (id:3cdb33b3).

For every successful revise op in a window (brain_batch revise / revise_batch), derive the
OLD-VALUE TOKENS the op replaced:
  swap fields  — value-class tokens (numbers, times, ISO/month dates, capitalized names) in
                 `old` and not in `new`; for SHORT swaps (old <= 8 words) every removed word
                 of 4+ chars counts too ("twice", "weekly", "Kauai")
  whole values — the same, against the field's value in nodes_before
  edge whys    — connect_to why swaps, same rule
Then read the node's END-OF-WINDOW state (nodes_after) and report every OTHER surface that
still carries one of those tokens: title, content, situation, question, reasoning, thought,
their_raw_quote, my_raw_quote, and every edge description on the node, both directions.
Content and reasoning are record fields (history and provenance may live there, E17); hits there are reported apart from the retrieval and delivered surfaces (title, situation, question, thought, quotes, edge descriptions).

Also listed for hand review: revise ops that moved a date in title/content without touching
event_time, and every event_time change (old -> new) with the node's quote beside it.

Usage: stale_surfaces.py <root> [<root>...] --out <dir> --label <cell>
"""
import argparse, json, re, sys, collections
from pathlib import Path

CONTROL = {'op', 'node_id', 'reason', 'connect_to', 'source_refs'}
SURFACES = ['title', 'content', 'situation', 'question', 'reasoning', 'thought', 'their_raw_quote', 'my_raw_quote']
MONTH = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?'
VALUE_RE = re.compile(r'\b\d[\d,.:/-]*\b|\b' + MONTH + r' \d{1,2}\b|\b\d{1,2} ' + MONTH + r'\b|(?<![.!?]\s)\b[A-Z][a-z]{2,}\b')
ROLE_WORDS = {'User', 'Partner', 'Collector', 'They', 'The', 'She', 'Her', 'His', 'Their', 'Other', 'Side', 'Assistant', 'Encoder', 'Anchor', 'Sam', 'Mira', 'Prior', 'Earlier', 'Later', 'Current', 'Confirmed', 'Updated', 'Plan', 'Plans'}
STOP = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they', 'their', 'them', 'than', 'then', 'into', 'onto', 'when', 'what', 'which', 'while', 'about', 'after', 'before', 'still', 'also', 'both', 'each', 'more', 'most', 'some', 'such', 'only', 'over', 'under', 'very', 'will', 'would', 'could', 'should', 'does', 'said', 'says', 'user', 'other', 'side', 'first', 'there', 'where', 'these', 'those', 'being', 'through', 'during', 'since', 'until', 'because', 'between', 'within', 'without', 'against', 'same', 'like', 'just', 'made', 'make', 'take', 'took', 'used', 'using', 'update', 'updated', 'noted', 'note'}


def field(node, key):
    if key in node and node[key] not in (None, ''): return node[key]
    return (node.get('_metadata') or {}).get(key)


def is_swap(v): return isinstance(v, dict) and set(v) == {'old', 'new'}
def is_swaps(v): return isinstance(v, list) and v and all(is_swap(x) for x in v)
def words(s): return re.findall(r"[A-Za-z0-9][A-Za-z0-9,.:/'-]*", str(s or ''))


def old_tokens(old, new, field_name=None):
    """Tokens that name the OLD value. Numbers and dates always; capitalized names and short-swap
    words only when the replaced text sits in a claim field (title, content, situation, question) —
    a quote, reasoning or thought replaced wholesale is a different sentence, not a changed value."""
    o, n = str(old or ''), str(new or '')
    claim_field = field_name in (None, 'title', 'content', 'situation', 'question', 'why')
    out = set()
    for m in VALUE_RE.finditer(o):
        t = m.group().strip('.,:;-')
        if len(t) < 2 or (t.isdigit() and len(t) == 1): continue
        is_number = bool(re.match(r'^\d', t))
        if not is_number and not claim_field: continue
        if m.start() == 0 and t[0].isupper(): continue          # sentence-initial capital, not a name
        if t in ROLE_WORDS or t.lower() in STOP: continue
        if not carries(n, t): out.add(t)
    if claim_field and len(words(o)) <= 8:
        for w in words(o):
            w2 = w.strip('.,:;-')
            if len(w2) >= 4 and w2.lower() not in STOP and w2 not in ROLE_WORDS and not carries(n, w2):
                out.add(w2)
    return out


def carries(text, tok):
    return re.search(r'(?<![A-Za-z0-9])' + re.escape(tok) + r'(?![A-Za-z0-9])', str(text or ''), re.I) is not None


def edge_texts(node):
    out = []
    for e in node.get('connections') or []:
        for r in e.get('relations') or [e]:
            if r.get('relation') in ('co_anchored', 'community_member'): continue
            out.append((e.get('direction', '?'), e.get('id'), r.get('relation'), r.get('description') or ''))
    return out


def revise_ops(calls):
    for ci, c in enumerate(calls):
        tool = c['call']['tool']; args = c['call']['args']; res = c['result']
        rr = res.get('result', res) if isinstance(res, dict) else {}
        results = rr.get('results', []) if isinstance(rr, dict) else []
        if tool == 'brain_batch':
            ops = args.get('operations', []); idx = [i for i, o in enumerate(ops) if o.get('op') == 'revise']
        elif tool == 'revise_batch':
            ops = args.get('revisions', []); idx = list(range(len(ops)))
        else:
            continue
        by_index = {r.get('index', i): r for i, r in enumerate(results)}
        for i in idx:
            if not isinstance(ops[i], dict): continue
            yield ci, tool, ops[i], by_index.get(i, {})


def scan_window(win, arm, rep, corpus, wnum):
    calls = json.loads((win / 'calls.json').read_text())
    before = json.loads((win / 'nodes_before.json').read_text()); after = json.loads((win / 'nodes_after.json').read_text())
    rows, date_flags, et_changes = [], [], []
    for ci, tool, op, res in revise_ops(calls):
        ok = bool(res.get('ok')) or res.get('status') == 'revised'
        nid = op.get('node_id')
        if not ok or nid not in after: continue
        op = dict(op)
        if op.get('content_edits') is not None and op.get('content') is None: op['content'] = op.pop('content_edits')
        bnode, anode = before.get(nid, {}), after[nid]
        toks = {}  # token -> field it was removed from
        touched = set()
        for f, v in op.items():
            if f in CONTROL or v is None: continue
            touched.add(f)
            if f in ('event_time', 'type', 'confidence', 'evolution_status', 'emotion', 'emotion_label'): continue
            if is_swap(v): pairs = [(v['old'], v['new'])]
            elif is_swaps(v): pairs = [(s['old'], s['new']) for s in v]
            else: pairs = [(field(bnode, f), v)]
            for o, n in pairs:
                for t in old_tokens(o, n, f): toks.setdefault(t, f)
        for e in op.get('connect_to') or []:
            if isinstance(e, dict) and is_swap(e.get('why')):
                for t in old_tokens(e['why']['old'], e['why']['new'], 'why'): toks.setdefault(t, 'why')
        if not toks: continue
        # a date moved in text but event_time was not walked
        if any(re.search(r'\d{4}-\d\d-\d\d|\b' + MONTH + r' \d{1,2}\b|\b\d{1,2} ' + MONTH, str(op.get(f) or '')) for f in ('title', 'content') if op.get(f) is not None) and 'event_time' not in touched and field(bnode, 'event_time'):
            date_flags.append({'arm': arm, 'repeat': rep, 'corpus': corpus, 'window': wnum, 'node_id': nid, 'event_time': field(bnode, 'event_time'), 'title_after': field(anode, 'title')})
        stale = []
        for s in SURFACES:
            val = field(anode, s)
            if val in (None, ''): continue
            for t in toks:
                if carries(val, t):
                    stale.append({'surface': s, 'token': t, 'removed_from': toks[t], 'snippet': re.sub(r'\s+', ' ', str(val))[:160]})
        for direction, tid, rel, desc in edge_texts(anode):
            for t in toks:
                if carries(desc, t):
                    stale.append({'surface': f'edge:{direction}:{rel}->{tid}', 'token': t, 'removed_from': toks[t], 'snippet': desc[:160]})
        # tokens that only ever lived in the field they were removed from produce no hit; report what remains
        rows.append({'arm': arm, 'repeat': rep, 'corpus': corpus, 'window': wnum, 'call': ci, 'tool': tool, 'node_id': nid,
                     'old_tokens': sorted(toks), 'fields_touched': sorted(touched), 'n_surfaces_touched': len(touched),
                     'stale_retrieval': [h for h in stale if h['surface'] not in ('content', 'reasoning')],
                     'stale_content_record': [h for h in stale if h['surface'] in ('content', 'reasoning')]})
    for nid, anode in after.items():
        b = before.get(nid)
        if b and field(b, 'event_time') and field(anode, 'event_time') and field(b, 'event_time') != field(anode, 'event_time'):
            et_changes.append({'arm': arm, 'repeat': rep, 'corpus': corpus, 'window': wnum, 'node_id': nid, 'old': field(b, 'event_time'), 'new': field(anode, 'event_time'),
                               'title_after': field(anode, 'title'), 'their_raw_quote': (field(anode, 'their_raw_quote') or '')[:120]})
    return rows, date_flags, et_changes


def discover(roots):
    for root in roots:
        root = Path(root)
        for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(('seed', 'process', 'blind', 'downstream', 'coverage', 'preflight', 'whole'))):
            for rep_dir in sorted(arm_dir.glob('repeat*')):
                for corpus_dir in sorted(p for p in rep_dir.iterdir() if p.is_dir() and not p.name.endswith('.crashed-window2')):
                    for win in sorted(corpus_dir.glob('window*')):
                        if (win / 'calls.json').exists() and (win / 'nodes_after.json').exists():
                            yield arm_dir.name, int(rep_dir.name[6:]), corpus_dir.name, int(win.name[6:]), win


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('roots', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--label', default='cell')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, date_flags, et_changes = [], [], []
    for arm, rep, corpus, wnum, win in discover(a.roots):
        r, d, e = scan_window(win, arm, rep, corpus, wnum); rows += r; date_flags += d; et_changes += e
    per_arm = collections.OrderedDict()
    for r in rows:
        s = per_arm.setdefault(r['arm'], collections.Counter())
        s['ops_with_old_value'] += 1
        s['ops_leaving_retrieval_stale'] += bool(r['stale_retrieval'])
        s['ops_keeping_old_in_content'] += bool(r['stale_content_record'])
        for h in r['stale_retrieval']:
            s['stale_' + ('edge' if h['surface'].startswith('edge:') else h['surface'])] += 1
        s['surfaces_touched_total'] += r['n_surfaces_touched']
    lines = [f'# Stale surfaces after revise — {a.label}', '', 'Reading queue, not a verdict: every hit is a token the op removed from one field that still sits in another retrieval surface of the same node at the end of the window. Content hits are listed apart (content is the record field).', '',
             '| Arm | Revise ops replacing a value | …leaving it in a retrieval/delivered surface | …keeping it in content/reasoning | stale title | situation | question | reasoning | thought | quote | edge why | mean surfaces touched/op |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, s in per_arm.items():
        n = s['ops_with_old_value'] or 1
        lines.append(f"| {arm} | {s['ops_with_old_value']} | {s['ops_leaving_retrieval_stale']} | {s['ops_keeping_old_in_content']} | {s['stale_title']} | {s['stale_situation']} | {s['stale_question']} | {s['stale_reasoning']} | {s['stale_thought']} | {s['stale_their_raw_quote'] + s['stale_my_raw_quote']} | {s['stale_edge']} | {s['surfaces_touched_total'] / n:.2f} |")
    lines += ['', f'## Queue — ops leaving a retrieval surface stale ({sum(1 for r in rows if r["stale_retrieval"])})', '']
    for r in rows:
        if not r['stale_retrieval']: continue
        lines.append(f"- **{r['arm']}** r{r['repeat']} {r['corpus']} w{r['window']} `{r['node_id']}` removed {r['old_tokens']} from {sorted(set(h['removed_from'] for h in r['stale_retrieval']))}; touched {r['fields_touched']}")
        for h in r['stale_retrieval'][:6]:
            lines.append(f"    - {h['surface']} still has `{h['token']}`: “{h['snippet']}”")
    lines += ['', f'## Dates moved in text with event_time untouched ({len(date_flags)})', ''] + [f"- **{d['arm']}** r{d['repeat']} {d['corpus']} w{d['window']} `{d['node_id']}` event_time {d['event_time']} — “{d['title_after']}”" for d in date_flags]
    lines += ['', f'## event_time changes ({len(et_changes)})', ''] + [f"- **{e['arm']}** r{e['repeat']} {e['corpus']} w{e['window']} `{e['node_id']}` {e['old']} → {e['new']} — “{e['title_after']}” | quote: “{e['their_raw_quote']}”" for e in et_changes]
    (out / f'STALE-SURFACES-{a.label}.md').write_text('\n'.join(lines) + '\n')
    (out / f'stale_surfaces_{a.label}.json').write_text(json.dumps({'per_arm': per_arm, 'rows': rows, 'date_flags': date_flags, 'event_time_changes': et_changes}, indent=1, default=str))
    print('\n'.join(lines[:6 + len(per_arm)]))
    print(f'queue: {sum(1 for r in rows if r["stale_retrieval"])} ops; date flags {len(date_flags)}; event_time changes {len(et_changes)}; written to {out}')


if __name__ == '__main__':
    main()
