"""Sweep every revise operation in saved encodes and check the infrastructure did
what it promises (contract.REVISE_RULE, brain_remember.revise):

  shape       — value vs swap per field, swap counts, connect_to on revise and the
                key it used, tool (brain_batch / revise_batch)
  outcome     — ok / refused, error class (old not found — and whether that `old`
                lives in another field of the same node —, ambiguous, no-op,
                no stored value, node not found, shape), warnings
  landing     — for every successful op, the field's final value in the window's
                nodes_after equals the composition of the window's successful ops
                on it (swap applied exactly once, bare value replaced); fields no
                op touched are byte-identical before and after (preservation);
                the returned deltas match the actual before/after; revised_at bumps
                only when title or content changed
  repair      — a refused op's intended change landed later in the same window
  edges       — connect_to entries on revise exist in nodes_after with the
                relation and the requested why
  vectors     — in the sequence's final brain.db, every embedding whose source
                field changed carries the CURRENT text (single-field vectors
                compared exactly; blends by containment); fields with text but
                no vector row are counted as missing

Usage: ./dev python3 revise_sweep.py <root> [<root> ...] --out <dir> --label <cell>
Roots hold <arm>/repeat<n>/<corpus>/window*/{calls,nodes_before,nodes_after}.json
and optionally <arm>/repeat<n>/<corpus>/final_brain/brain.db.
"""
import argparse, json, re, sqlite3, sys
from collections import Counter, defaultdict
from pathlib import Path

REVISABLE = ('title', 'content', 'situation', 'question', 'reasoning', 'thought', 'their_raw_quote', 'my_raw_quote',
             'event_time', 'confidence', 'type', 'evolution_status', 'emotion', 'emotion_label', 'correction_pattern', 'source_context', 'source_refs')
TEXT = ('title', 'content', 'situation', 'question', 'reasoning', 'thought', 'their_raw_quote', 'my_raw_quote', 'correction_pattern', 'source_context')
SYSTEM = {'updated_at', 'revised_at', 'access_count', 'last_accessed', 'recency_score', 'activation', 'stability', 'connections', 'content_summary', 'communities', '_metadata', 'emotion_source'}
CONTROL = {'op', 'node_id', 'reason', 'connect_to', 'encoding_source', 'chain_id', 'session_id'}

def field(node, key):
    if node is None: return None
    if key in node and node[key] not in (None, ''): return node[key]
    return (node.get('_metadata') or {}).get(key)

def is_swap(v): return isinstance(v, dict) and set(v) == {'old', 'new'}
def is_swaps(v): return isinstance(v, list) and v and all(is_swap(x) for x in v)

def apply(current, v):
    """Mirror contract.apply_swaps; returns (value, error)."""
    if is_swap(v) or is_swaps(v):
        cur = current if isinstance(current, str) else ''
        for i, e in enumerate(v if isinstance(v, list) else [v]):
            n = cur.count(e['old'])
            if n != 1: return None, ('not_found' if n == 0 else 'ambiguous', i)
            cur = cur.replace(e['old'], e['new'], 1)
        return cur, None
    return v, None

def classify_error(err, op, before_node):
    e = str(err or '')
    if 'not found in the node' in e:
        m = re.match(r'(\w+) swap\[(\d+)\]', e); sub = ''
        if m and before_node is not None:
            f, i = m.group(1), int(m.group(2)); v = op.get(f); sw = (v if isinstance(v, list) else [v])[i] if v else None
            if sw and isinstance(sw, dict):
                for g in TEXT:
                    if g != f and isinstance(field(before_node, g), str) and sw['old'] in field(before_node, g): sub = f':old_lives_in_{g}'; break
                else:
                    if not sub: sub = ':old_not_in_any_field'
        return 'old_not_found' + sub
    if 'matches' in e and 'places' in e: return 'ambiguous'
    if 'identical' in e: return 'noop_identical'
    if 'no stored value' in e: return 'no_stored_value'
    if 'Node not found' in e: return 'node_not_found'
    if 'mutually exclusive' in e or 'bare value' in e or 'must be a string' in e or 'must be {old' in e: return 'shape'
    if 'No updates' in e: return 'empty'
    return 'other'

def revise_ops(calls):
    for ci, c in enumerate(calls):
        tool = c['call']['tool']; args = c['call']['args']; res = c['result']
        rr = res.get('result', res) if isinstance(res, dict) else {}
        results = rr.get('results', []) if isinstance(rr, dict) else []
        if tool == 'brain_batch':
            ops = args.get('operations', [])
            idx = [i for i, o in enumerate(ops) if o.get('op') == 'revise']
        elif tool == 'revise_batch':
            ops = args.get('revisions', []); idx = list(range(len(ops)))
        else:
            continue
        by_index = {r.get('index', i): r for i, r in enumerate(results)}
        for i in idx:
            yield ci, tool, ops[i], by_index.get(i, {})

def sweep_window(win, arm, rep, corpus, wnum):
    calls = json.loads((win / 'calls.json').read_text())
    before = json.loads((win / 'nodes_before.json').read_text()); after = json.loads((win / 'nodes_after.json').read_text())
    rows = []; touched = defaultdict(lambda: defaultdict(list)); failed_intents = []
    running = {}  # node -> field -> running value
    for ci, tool, op, res in revise_ops(calls):
        nid = op.get('node_id'); bnode = before.get(nid); ok = bool(res.get('ok')) or res.get('status') == 'revised'
        op = dict(op)
        if op.get('content_edits') is not None and op.get('content') is None:
            op['content'] = op.pop('content_edits')  # tool alias: content_edits == content: [swaps]
        fields = {f: ('swaps:%d' % len(op[f]) if is_swaps(op[f]) else 'swap' if is_swap(op[f]) else 'value') for f in op if f not in CONTROL and op[f] is not None and f != 'source_refs'}
        ct = op.get('connect_to'); ct_keys = sorted({k for e in (ct or []) if isinstance(e, dict) for k in e}) if ct else []
        row = {'arm': arm, 'repeat': rep, 'corpus': corpus, 'window': wnum, 'call': ci, 'tool': tool, 'node_id': nid, 'fields': fields,
               'shape': 'value_only' if fields and all(v == 'value' for v in fields.values()) else 'swap_only' if fields and all(v != 'value' for v in fields.values()) else 'mixed' if fields else 'edges_only',
               'connect_to': len(ct) if ct else 0, 'connect_to_keys': ct_keys, 'ok': ok, 'error': res.get('error'), 'error_class': None if ok else classify_error(res.get('error'), op, bnode),
               'warnings': res.get('warnings') or (res.get('result') or {}).get('warnings') if isinstance(res.get('result'), dict) else res.get('warnings'),
               'node_in_before': nid in before, 'checks': {}}
        deltas = res.get('deltas') or (res.get('result') or {}).get('deltas') if isinstance(res.get('result'), dict) else res.get('deltas')
        if ok and nid in before:
            for f, kind in fields.items():
                cur = running.get(nid, {}).get(f, field(bnode, f))
                val, err = apply(cur, op[f])
                if err: row['checks'][f'compose_error:{f}'] = err[0]; continue
                running.setdefault(nid, {})[f] = val; touched[nid][f].append(val)
            # deltas vs actual
            if isinstance(deltas, list):
                for d in deltas:
                    f = d.get('field'); exp_new = running.get(nid, {}).get(f)
                    if f in fields and exp_new is not None and str(d.get('new')) != str(exp_new) and not (is_swap(op[f]) or is_swaps(op[f])):
                        row['checks'][f'delta_new_mismatch:{f}'] = True
                row['deltas_fields'] = sorted({d.get('field') for d in deltas if isinstance(d, dict)})
        elif not ok and nid in before:
            for f, kind in fields.items():
                if kind != 'value':
                    for e in (op[f] if isinstance(op[f], list) else [op[f]]):
                        failed_intents.append((nid, f, e['new'], ci))
                else:
                    failed_intents.append((nid, f, op[f], ci))
        rows.append(row)
    # landing + preservation per node revised successfully this window
    landing = []
    for nid, fl in touched.items():
        a = after.get(nid); b = before.get(nid)
        if a is None: landing.append({'node_id': nid, 'missing_after': True}); continue
        rec = {'node_id': nid, 'arm': arm, 'repeat': rep, 'corpus': corpus, 'window': wnum, 'landed': {}, 'preserved': True, 'preservation_violations': [], 'revised_at_ok': True}
        for f, vals in fl.items():
            exp = vals[-1]; got = field(a, f)
            rec['landed'][f] = (str(got).strip() == str(exp).strip()) if isinstance(exp, str) else (got == exp or str(got) == str(exp))
        for f in REVISABLE:
            if f in fl: continue
            bv, av = field(b, f), field(a, f)
            if (bv or None) != (av or None) and f not in ('confidence',):
                # a remember-time system default may differ (e.g. emotion auto); only flag text/time fields
                if f in TEXT or f in ('event_time', 'type', 'evolution_status'):
                    rec['preserved'] = False; rec['preservation_violations'].append(f)
        claim_changed = any(f in fl for f in ('title', 'content'))
        ra_b, ra_a = b.get('revised_at'), a.get('revised_at')
        if claim_changed and not ra_a: rec['revised_at_ok'] = False
        if not claim_changed and ra_a != ra_b: rec['revised_at_ok'] = False
        landing.append(rec)
    # repair: failed intents that landed anyway
    repairs = []
    for nid, f, new, ci in failed_intents:
        a = after.get(nid); got = field(a, f) if a else None
        landed = isinstance(got, str) and isinstance(new, str) and new.strip() in got
        later_ok = any(r['ok'] and r['node_id'] == nid and f in r['fields'] and r['call'] > ci for r in rows)
        repairs.append({'node_id': nid, 'field': f, 'landed_by_window_end': bool(landed), 'later_successful_op_on_field': later_ok, 'arm': arm, 'repeat': rep, 'corpus': corpus, 'window': wnum})
    # connect_to on revise → edge present after
    edges = []
    for ci, tool, op, res in revise_ops(calls):
        ct = op.get('connect_to'); nid = op.get('node_id')
        if not ct or not (res.get('ok') or res.get('status') == 'revised'): continue
        a = after.get(nid) or {}
        out = {}
        for e in a.get('connections') or []:  # either direction: a revise connect_to may ride an edge that points INTO this node
            for r in e.get('relations') or [e]:
                out[(e['id'], r.get('relation'))] = r.get('description') or ''
        for entry in ct:
            if not isinstance(entry, dict): continue
            tgt = entry.get('target') or entry.get('title'); rel = entry.get('relation'); why = entry.get('why') or entry.get('description')
            rel_v = rel['new'] if is_swap(rel) else rel
            cand = {k: v for k, v in out.items() if k[0] == tgt and (rel_v is None or k[1] == rel_v)}
            want = why['new'] if is_swap(why) else (why if isinstance(why, str) else None)
            ok = bool(cand) and (want is None or any(want.strip() in v for v in cand.values()))
            ctr = res.get('connect_to_result') or (res.get('result') or {}).get('connect_to_result') if isinstance(res.get('result'), dict) else res.get('connect_to_result')
            edges.append({'node_id': nid, 'target': tgt, 'relation': rel_v, 'why_kind': 'swap' if is_swap(why) else 'value' if why else 'none', 'edge_present_with_why': ok, 'tool_reported': (json.dumps(ctr)[:200] if ctr else None), 'arm': arm, 'repeat': rep, 'corpus': corpus, 'window': wnum})
    return rows, landing, repairs, edges

BLENDS = {'_primary': ('title', 'content'), 'high_meta': ('situation', 'their_raw_quote', 'my_raw_quote'), 'other_meta': ('reasoning', 'correction_pattern', 'source_context')}
SINGLE = {'title': 'title', 'content': 'content', 'question': 'question', 'reasoning': 'reasoning', 'their_raw_quote': 'their_raw_quote', 'my_raw_quote': 'my_raw_quote'}
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
try:
    from servers.pipeline_contract import vectors_affected_by
except Exception:  # pragma: no cover — the sweep still runs without the timestamp lane
    vectors_affected_by = None

def _norm(s): return ' '.join(str(s or '').split())
def _prefix_ok(stored, current):
    """The embedder may truncate long text: stale only if the current text does not START with the stored text (or vice versa for a shorter current)."""
    a, b = _norm(stored), _norm(current)
    if not a: return None
    n = min(len(a), len(b))
    return a[:n] == b[:n]

def vector_check(db, last_changed):
    """last_changed: node_id -> fields changed by the node's LAST successful revise (from deltas).
    Two lanes: text — stored embedding text is (a prefix of) the current field text; time — every vector
    whose source field changed in the last revise was re-created at or after the node's updated_at."""
    if not db.exists(): return None
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    out = {'nodes_checked': 0, 'vectors_checked': 0, 'text_stale': [], 'missing': [], 'not_refreshed': [], 'refresh_checked': 0}
    for nid, changed in last_changed.items():
        row = con.execute('select title, content, updated_at, revised_at from nodes where id=?', (nid,)).fetchone()
        if not row: continue
        kv = dict(con.execute('select key, value from node_metadata_kv where node_id=?', (nid,)).fetchall())
        vals = {'title': row[0] or '', 'content': row[1] or ''}
        for k in ('situation', 'question', 'reasoning', 'their_raw_quote', 'my_raw_quote', 'correction_pattern', 'source_context'):
            v = kv.get(k)
            if isinstance(v, str) and v.startswith('"'):
                try: v = json.loads(v)
                except Exception: pass
            vals[k] = v or ''
        enr = {vt: (text or '', created) for vt, text, created in con.execute('select vector_type, text, created_at from node_enrichments where node_id=?', (nid,))}
        out['nodes_checked'] += 1
        for vt, f in SINGLE.items():
            if vals.get(f):
                if vt not in enr: out['missing'].append((nid, vt)); continue
                ok = _prefix_ok(enr[vt][0], vals[f])
                if ok is not None:
                    out['vectors_checked'] += 1
                    if not ok: out['text_stale'].append((nid, vt))
        for vt, fs in BLENDS.items():
            present = [vals[f] for f in fs if vals.get(f)]
            if not present: continue
            if vt not in enr: out['missing'].append((nid, vt)); continue
            txt = _norm(enr[vt][0])
            if txt:
                out['vectors_checked'] += 1
                if not all((_norm(p) in txt) or txt.endswith(_norm(p)[: max(len(txt) - txt.find(_norm(p)[:40]), 1)]) or (_norm(p)[:60] in txt) for p in present): out['text_stale'].append((nid, vt))
        if vectors_affected_by and changed:
            affected = set()
            for f in changed: affected |= set(vectors_affected_by(f))
            for vt in sorted(affected):
                if vt not in enr:
                    out['missing'].append((nid, vt)); continue
                out['refresh_checked'] += 1
                if enr[vt][1] < row[2]: out['not_refreshed'].append((nid, vt))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('roots', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--label', default='cell')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows_all, landing_all, repairs_all, edges_all, vectors_all = [], [], [], [], {}
    for root in [Path(r).resolve() for r in a.roots]:
        seqs = defaultdict(list)
        for win in sorted(root.glob('*/repeat*/*/window*'), key=lambda p: (str(p.parent), int(p.name[6:]))):
            if not (win / 'calls.json').exists(): continue
            arm, rep, corpus = win.parts[-4], win.parts[-3], win.parts[-2]
            r, l, rp, e = sweep_window(win, arm, rep, corpus, int(win.name[6:]))
            rows_all += r; landing_all += l; repairs_all += rp; edges_all += e
            for x in r:
                if x['ok']: seqs[(arm, rep, corpus)].append((x['node_id'], x.get('deltas_fields') or list(x['fields'])))
        for (arm, rep, corpus), pairs in seqs.items():
            last = {}
            for nid, flds in pairs: last[nid] = [f for f in flds if f != 'source_refs']
            vc = vector_check(root / arm / rep / corpus / 'final_brain' / 'brain.db', last)
            if vc is not None: vectors_all[f'{arm}|{rep}|{corpus}'] = vc
    json.dump({'label': a.label, 'ops': rows_all, 'landing': landing_all, 'repairs': repairs_all, 'connect_to_on_revise': edges_all, 'vectors': vectors_all}, open(out / f'revise_sweep_{a.label}.json', 'w'), indent=1, default=str)
    arms = [x for x in ('production_deployed', 'v3_2_titles', 'v3_3_titles') if any(r['arm'] == x for r in rows_all)] + sorted({r['arm'] for r in rows_all} - {'production_deployed', 'v3_2_titles', 'v3_3_titles'})
    names = {'production_deployed': 'Production', 'v3_2_titles': 'V3.2', 'v3_3_titles': 'V3.3'}
    L = [f'| Revise sweep ({a.label}) | ' + ' | '.join(names.get(x, x) for x in arms) + ' |', '|---|' + '---:|' * len(arms)]
    def row(label, fn): L.append(f'| {label} | ' + ' | '.join(str(fn(x)) for x in arms) + ' |')
    R = lambda x: [r for r in rows_all if r['arm'] == x]
    row('Revise ops sent (brain_batch / revise_batch)', lambda x: f"{len(R(x))} ({sum(1 for r in R(x) if r['tool']=='brain_batch')} / {sum(1 for r in R(x) if r['tool']=='revise_batch')})")
    row('Refused', lambda x: sum(1 for r in R(x) if not r['ok']))
    row('Shape: value-only / swap-only / mixed / edges-only', lambda x: ' / '.join(str(sum(1 for r in R(x) if r['shape']==s)) for s in ('value_only','swap_only','mixed','edges_only')))
    row('Ops using a swap on any field', lambda x: sum(1 for r in R(x) if any(v != 'value' for v in r['fields'].values())))
    row('Fields touched (successful ops)', lambda x: ', '.join(f'{k} {v}' for k, v in Counter(f for r in R(x) if r['ok'] for f in r['fields']).most_common()))
    row('Swap targets by field (all ops)', lambda x: ', '.join(f'{k} {v}' for k, v in Counter(f for r in R(x) for f, kind in r['fields'].items() if kind != 'value').most_common()) or '—')
    row('connect_to on revise: ops / keys used', lambda x: f"{sum(1 for r in R(x) if r['connect_to'])} / {dict(Counter(k for r in R(x) if r['connect_to'] for k in r['connect_to_keys']))}")
    row('Error classes', lambda x: ', '.join(f'{k} {v}' for k, v in Counter(r['error_class'] for r in R(x) if not r['ok']).most_common()) or '—')
    Ld = lambda x: [l for l in landing_all if l.get('arm') == x]
    row('Nodes revised (node × window) / all touched fields landed as composed', lambda x: f"{len(Ld(x))} / {sum(1 for l in Ld(x) if l.get('landed') and all(l['landed'].values()))}")
    row('Fields landed / not landed', lambda x: f"{sum(sum(1 for v in l.get('landed', {}).values() if v) for l in Ld(x))} / {sum(sum(1 for v in l.get('landed', {}).values() if not v) for l in Ld(x))}")
    row('Untouched fields preserved (nodes) / violations', lambda x: f"{sum(1 for l in Ld(x) if l.get('preserved'))} / {dict(Counter(f for l in Ld(x) for f in l.get('preservation_violations', [])))}")
    row('revised_at semantics respected (nodes)', lambda x: f"{sum(1 for l in Ld(x) if l.get('revised_at_ok'))} of {len(Ld(x))}")
    row('Delta mismatches / compose errors flagged', lambda x: f"{sum(1 for r in R(x) for k in r['checks'] if k.startswith('delta'))} / {sum(1 for r in R(x) for k in r['checks'] if k.startswith('compose'))}")
    Rp = lambda x: [p for p in repairs_all if p['arm'] == x]
    row('Refused intents: total / landed by window end / later op on same field', lambda x: f"{len(Rp(x))} / {sum(1 for p in Rp(x) if p['landed_by_window_end'])} / {sum(1 for p in Rp(x) if p['later_successful_op_on_field'])}")
    E = lambda x: [e for e in edges_all if e['arm'] == x]
    row('connect_to on revise entries / edge present with requested why', lambda x: f"{len(E(x))} / {sum(1 for e in E(x) if e['edge_present_with_why'])}")
    V = lambda x: [v for k, v in vectors_all.items() if k.startswith(x + '|')]
    row('Vectors of revised nodes: nodes / text-checked / text stale / missing', lambda x: f"{sum(v['nodes_checked'] for v in V(x))} / {sum(v['vectors_checked'] for v in V(x))} / {sum(len(v['text_stale']) for v in V(x))} / {sum(len(v['missing']) for v in V(x))}" if V(x) else 'no final_brain')
    row('Vectors affected by the last revise: checked / not refreshed after it', lambda x: f"{sum(v['refresh_checked'] for v in V(x))} / {sum(len(v['not_refreshed']) for v in V(x))}" if V(x) else 'no final_brain')
    (out / f'REVISE-SWEEP-{a.label}.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
    # detail lists worth reading
    print('\nrefused ops:')
    for r in rows_all:
        if not r['ok']: print(f"  {r['arm']} {r['repeat']} {r['corpus']} w{r['window']} c{r['call']} {r['tool']} {r['node_id']} fields={r['fields']} -> {r['error_class']}")
    print('\nlanding failures / preservation violations / revised_at issues:')
    for l in landing_all:
        if l.get('missing_after') or (l.get('landed') and not all(l['landed'].values())) or not l.get('preserved', True) or not l.get('revised_at_ok', True):
            print('  ', {k: v for k, v in l.items() if k in ('arm','repeat','corpus','window','node_id','landed','preservation_violations','revised_at_ok','missing_after')})
    print('\nconnect_to on revise entries not found with their why:')
    for e in edges_all:
        if not e['edge_present_with_why']: print('  ', e)
    print('\ntext-stale / not-refreshed / missing vectors:')
    for k, v in vectors_all.items():
        if v['text_stale'] or v['missing'] or v['not_refreshed']: print('  ', k, 'text_stale', v['text_stale'], 'not_refreshed', v['not_refreshed'], 'missing', v['missing'])

if __name__ == '__main__':
    main()
