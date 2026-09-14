"""Node-quality census over saved encodes — what a node carries into a live turn.

Reads final-window `nodes_after.json` per sequence (arm / repeat / corpus) under
one or more result roots and measures, per arm and per corpus:

  types and aspects   — distinct types, entropy, share of the top type, aspect
                        shares from aspects_v1.json (identity_bearing,
                        lesson_insight, episodic_anchor, active_thread,
                        correction_improvement, noise, unmapped)
  field fill          — every text field, event_time (with precision), thought,
                        quotes, confidence spread, non-neutral emotion
  content             — words per field; specificity per 100 content words
                        (numbers, dates, capitalized tokens, quoted strings);
                        turn-coordinate and encoder-clock leakage
  retrieval surface   — situation in trigger register and not a title restatement;
                        question phrased as a question and not from the encoder's
                        side; title specificity; near-twin pairs (title / content
                        token Jaccard)
  edges               — relations, generic-relation share, description words,
                        descriptions that restate the endpoint titles, duplicate
                        pairs, degree, isolates, hub concentration, components
  recall lanes        — how many nodes feed each lane recall reads: title (all),
                        _situation, question, high_meta (quotes), edge_context,
                        event_time (temporal); Frame candidates (identity_bearing
                        + lesson_insight), open threads (active_thread),
                        correction-aspect edges

Descriptive census, not a score. Usage:
  ./dev python3 quality_census.py <root> [<root> ...] --out <dir> [--label transfer]
Each root holds <arm>/repeat<n>/<corpus>/window*/nodes_after.json; the corpus
source for quote checks is <root>/<corpus>.json when present.
"""
import argparse, json, math, re, statistics, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASPECTS = json.loads((ROOT / 'servers/scales/s2/aspects_v1.json').read_text())
TYPE_ASPECT = {}
for name, spec in ASPECTS.items():
    if name.startswith('_') or not isinstance(spec, dict): continue
    for t in spec.get('node_types', []): TYPE_ASPECT.setdefault(t, name)
GENERIC_RELATIONS = set(ASPECTS.get('generic_relation', {}).get('edge_relations', [])) | {'related', 'related_to'}
CORRECTION_RELATIONS = set(ASPECTS.get('correction_improvement', {}).get('edge_relations', []))
TEXT_FIELDS = ('title', 'content', 'situation', 'question', 'reasoning', 'thought', 'their_raw_quote', 'my_raw_quote')
TRIGGER_OPENERS = ('when', 'before', 'if', 'at ', 'during', 'whenever', 'any time', 'anytime', 'after', 'on ', 'while', 'pre-action', 'each time')
QUESTION_OPENERS = ('what', 'which', 'who', 'when', 'where', 'why', 'how', 'did', 'does', 'do ', 'is ', 'are ', 'was ', 'has ', 'have ', 'should', 'can ', 'could', 'will')

def field(node, key):
    if key in node and node[key] not in (None, ''): return node[key]
    md = node.get('_metadata') or {}
    return md.get(key)

def words(v): return len(str(v).split()) if v not in (None, '') else 0
def toks(s): return set(re.findall(r"[a-z0-9]{3,}", str(s).lower()))
def jaccard(a, b): return len(a & b) / len(a | b) if (a | b) else 0.0
def spec_hits(text):
    t = str(text)
    return {'numbers': len(re.findall(r'\b\d[\d,.:/-]*\b', t)),
            'dates': len(re.findall(r'\b(?:20\d\d|19\d\d)(?:-\d\d(?:-\d\d)?)?\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d', t)),
            'capitalized': len(re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}", t)),
            'quoted': len(re.findall(r"[\"“'‘][^\"”'’]{3,}[\"”'’]", t))}

def edges_of(node):
    out = []
    for e in node.get('connections') or []:
        if e.get('direction') != 'outgoing': continue
        for r in e.get('relations') or [e]:
            rel = r.get('relation')
            if rel in ('co_anchored', 'community_member'): continue
            out.append({'target_id': e['id'], 'target_title': e.get('title', ''), 'relation': rel, 'description': r.get('description') or ''})
    return out

def sources_for(root, corpus):
    p = root / f'{corpus}.json'
    if not p.exists(): return None
    fx = json.loads(p.read_text())
    clocks = [w.get('now', '')[:10] for w in fx['windows']]
    return {'other': '\n'.join(t.get('other', '') for w in fx['windows'] for t in w['turns']),
            'me': '\n'.join(t.get('me', '') for w in fx['windows'] for t in w['turns']), 'clocks': clocks}

def quote_class(q, src):
    ws = lambda s: ' '.join(s.split())
    if ws(q) in ws(src): return 'verbatim'
    bare = lambda s: re.sub(r"[\"'‘’“”`]", '', ws(s).lower().replace('—', '-').replace('–', '-').replace('…', '...'))
    if bare(q) in bare(src): return 'style_only'
    parts = [p for p in re.split(r'\s*(?:\.\.\.|…)\s*', q) if p.strip()]
    if len(parts) > 1 and all(bare(p) in bare(src) for p in parts): return 'ellipsis_splice'
    return 'not_in_source'

def census_sequence(nodes, src):
    n = len(nodes); ids = set(nodes)
    c = Counter(); fw = defaultdict(list); spec = Counter(); conf = []; types = Counter()
    rel_vocab = Counter(); edge_words = []; degree = Counter(); pairs = Counter(); quotes = Counter()
    edge_restate = 0; edges_total = 0; generic = 0; correction_edges = 0; adjacency = defaultdict(set)
    et_precision = Counter(); et_outside = 0
    for nid, node in nodes.items():
        t = node.get('type') or 'untyped'; types[t] += 1
        c[f'aspect:{TYPE_ASPECT.get(t, "unmapped")}'] += 1
        for k in TEXT_FIELDS:
            v = field(node, k)
            if v not in (None, ''):
                c[f'has:{k}'] += 1; fw[k].append(words(v))
        content = field(node, 'content') or ''
        cw = max(words(content), 1)
        for k, v in spec_hits(content).items(): spec[k] += v
        spec['content_words'] += cw
        title = field(node, 'title') or ''
        th = spec_hits(title)
        if th['numbers'] + th['dates'] + th['capitalized'] + th['quoted'] > 0: c['title_specific'] += 1
        if words(title) < 4: c['title_short'] += 1
        sit = field(node, 'situation') or ''
        if sit:
            s = sit.strip().lower()
            if s.startswith(TRIGGER_OPENERS): c['situation_trigger_register'] += 1
            if jaccard(toks(sit), toks(title)) >= 0.5: c['situation_restates_title'] += 1
        q = field(node, 'question') or ''
        if q:
            ql = q.strip().lower()
            if ql.startswith(QUESTION_OPENERS) and q.strip().endswith('?'): c['question_is_question'] += 1
            if re.search(r"\b(did|do|have|had|should|what|which) i\b|\bi (recommend|suggest|offer|gave|told|said)", ql): c['question_encoder_side'] += 1
        reasoning = field(node, 'reasoning') or ''
        src_years = {k[:4] for k in (src or {}).get('clocks', []) if k}
        for k in ('reasoning', 'content', 'situation'):
            v = field(node, k) or ''
            if re.search(r'\bturns? \d', str(v), re.I): c[f'turn_coord:{k}'] += 1
            if src_years and any(y not in src_years for y in re.findall(r'\b(20\d\d)-\d\d', str(v))): c[f'clock_leak:{k}'] += 1
        lanes = 1 + sum(1 for k in ('situation', 'question', 'event_time') if field(node, k)) + (1 if (field(node, 'their_raw_quote') or field(node, 'my_raw_quote')) else 0) + (1 if edges_of(node) else 0)
        c[f'lanes:{lanes}'] += 1; c['lanes_total'] += lanes
        if reasoning and re.search(r'\b(turn \d|said|stated|disclosed|confirmed|reported|20\d\d-\d\d)', reasoning, re.I): c['reasoning_cites_provenance'] += 1
        cf = node.get('confidence')
        if isinstance(cf, (int, float)): conf.append(float(cf))
        if node.get('emotion') not in (None, 0, 0.0): c['emotion_nonzero'] += 1
        if (node.get('emotion_label') or 'neutral') != 'neutral': c['emotion_labelled'] += 1
        if node.get('evolution_status'): c['has:evolution_status'] += 1
        et = field(node, 'event_time')
        if et:
            c['has:event_time'] += 1
            s = str(et)
            et_precision['datetime' if 'T' in s else 'date' if len(s) >= 10 else 'month' if len(s) == 7 else 'year'] += 1
            if src and src.get('clocks'):
                yrs = {k[:4] for k in src['clocks'] if k}
                if s[:4] not in yrs: et_outside += 1
        for key, role in (('their_raw_quote', 'other'), ('my_raw_quote', 'me')):
            qv = field(node, key)
            if isinstance(qv, str) and qv:
                quotes['total'] += 1
                if src: quotes[quote_class(qv, src[role])] += 1
        seen_targets = Counter()
        for e in edges_of(node):
            edges_total += 1; rel_vocab[e['relation']] += 1; edge_words.append(words(e['description']))
            if e['relation'] in GENERIC_RELATIONS: generic += 1
            if e['relation'] in CORRECTION_RELATIONS: correction_edges += 1
            seen_targets[e['target_id']] += 1
            dt = toks(e['description']); tt = toks(title) | toks(e['target_title'])
            if dt and len(dt & tt) / len(dt) >= 0.6: edge_restate += 1
            degree[nid] += 1
            if e['target_id'] in ids:
                degree[e['target_id']] += 1; adjacency[nid].add(e['target_id']); adjacency[e['target_id']].add(nid)
        pairs['dup_pairs'] += sum(1 for v in seen_targets.values() if v > 1)
    isolates = sum(1 for nid in nodes if not adjacency[nid])
    seen = set(); components = 0
    for nid in nodes:
        if nid in seen: continue
        components += 1; stack = [nid]
        while stack:
            x = stack.pop()
            if x in seen: continue
            seen.add(x); stack.extend(adjacency[x] - seen)
    twins = 0
    for a, b in combinations(nodes.values(), 2):
        if jaccard(toks(field(a, 'title')), toks(field(b, 'title'))) >= 0.5 or jaccard(toks(field(a, 'content')), toks(field(b, 'content'))) >= 0.45: twins += 1
    ent = -sum((v / n) * math.log2(v / n) for v in types.values()) if n else 0.0
    return {'nodes': n, 'types': dict(types), 'type_entropy_bits': round(ent, 2), 'top_type_share': round(max(types.values()) / n, 2) if n else 0,
            'counts': dict(c), 'field_words': {k: sum(v) for k, v in fw.items()}, 'field_word_medians': {k: statistics.median(v) for k, v in fw.items()},
            'specificity': dict(spec), 'confidence': conf, 'event_time_precision': dict(et_precision), 'event_time_year_outside_source': et_outside,
            'quotes': dict(quotes), 'edges': edges_total, 'relations': dict(rel_vocab), 'generic_relations': generic, 'correction_edges': correction_edges,
            'edge_words': sum(edge_words), 'edge_restates_titles': edge_restate, 'dup_edge_pairs': pairs['dup_pairs'], 'isolates': isolates, 'components': components,
            'max_degree': max(degree.values()) if degree else 0, 'near_twin_pairs': twins}

def discover(roots):
    for root in roots:
        for na in sorted(root.glob('*/repeat*/*/window*/nodes_after.json')):
            arm, rep, corpus = na.parts[-5], na.parts[-4], na.parts[-3]
            yield root, arm, rep, corpus, na

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('roots', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--label', default='cell')
    a = ap.parse_args()
    roots = [Path(r).resolve() for r in a.roots]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    latest = {}
    for root, arm, rep, corpus, na in discover(roots):
        w = int(na.parent.name[6:])
        key = (arm, rep, corpus)
        if key not in latest or w > latest[key][0]: latest[key] = (w, na, root)
    per_seq = []
    for (arm, rep, corpus), (w, na, root) in sorted(latest.items()):
        nodes = {k: v for k, v in json.loads(na.read_text()).items() if isinstance(v, dict) and v.get('title')}
        src = sources_for(root, corpus)
        r = census_sequence(nodes, src); r.update({'arm': arm, 'repeat': rep, 'corpus': corpus, 'final_window': w}); per_seq.append(r)
    # aggregate per arm and per (arm, corpus)
    def agg(rows):
        out = {'sequences': len(rows), 'nodes': sum(r['nodes'] for r in rows), 'types': Counter(), 'counts': Counter(), 'field_words': Counter(), 'specificity': Counter(),
               'confidence': [], 'event_time_precision': Counter(), 'event_time_year_outside_source': 0, 'quotes': Counter(), 'edges': 0, 'relations': Counter(),
               'generic_relations': 0, 'correction_edges': 0, 'edge_words': 0, 'edge_restates_titles': 0, 'dup_edge_pairs': 0, 'isolates': 0, 'components': 0, 'near_twin_pairs': 0, 'max_degree': []}
        for r in rows:
            for k in ('types', 'counts', 'field_words', 'specificity', 'event_time_precision', 'quotes', 'relations'): out[k].update(r[k])
            for k in ('edges', 'generic_relations', 'correction_edges', 'edge_words', 'edge_restates_titles', 'dup_edge_pairs', 'isolates', 'components', 'near_twin_pairs', 'event_time_year_outside_source'): out[k] += r[k]
            out['confidence'] += r['confidence']; out['max_degree'].append(r['max_degree'])
        n = out['nodes'] or 1
        ent = -sum((v / n) * math.log2(v / n) for v in out['types'].values()) if out['nodes'] else 0.0
        out['distinct_types'] = len(out['types']); out['type_entropy_bits'] = round(ent, 2); out['top_type_share'] = round(max(out['types'].values()) / n, 2) if out['types'] else 0
        cs = out['confidence']
        out['confidence_summary'] = {'n': len(cs), 'mean': round(statistics.mean(cs), 3) if cs else None, 'stdev': round(statistics.pstdev(cs), 3) if len(cs) > 1 else 0, 'distinct': len(set(cs))}
        for k in ('types', 'counts', 'field_words', 'specificity', 'event_time_precision', 'quotes', 'relations'): out[k] = dict(out[k])
        del out['confidence']
        return out
    arms = sorted({r['arm'] for r in per_seq})
    report = {'label': a.label, 'roots': [str(r) for r in roots], 'per_sequence': per_seq,
              'per_arm': {arm: agg([r for r in per_seq if r['arm'] == arm]) for arm in arms},
              'per_arm_corpus': {f'{arm}|{c}': agg([r for r in per_seq if r['arm'] == arm and r['corpus'] == c]) for arm in arms for c in sorted({r['corpus'] for r in per_seq})}}
    (out / f'quality_census_{a.label}.json').write_text(json.dumps(report, indent=1, default=str))
    # markdown table
    A = report['per_arm']; names = {'production_deployed': 'Production', 'v3_2_titles': 'V3.2', 'v3_3_titles': 'V3.3'}
    cols = [x for x in ('production_deployed', 'v3_2_titles', 'v3_3_titles') if x in A] + [x for x in arms if x not in names]
    L = [f'| Measure ({a.label}; totals over {A[cols[0]]["sequences"]} sequences per arm) | ' + ' | '.join(names.get(x, x) for x in cols) + ' |', '|---|' + '---:|' * len(cols)]
    def row(label, fn): L.append(f'| {label} | ' + ' | '.join(str(fn(A[x])) for x in cols) + ' |')
    pct = lambda r, k, base='nodes': f"{r['counts'].get(k, 0)} ({round(100 * r['counts'].get(k, 0) / max(r[base] if base == 'nodes' else r['counts'].get(base, 0), 1))}%)"
    row('Final nodes', lambda r: r['nodes'])
    row('Distinct types / entropy (bits) / top-type share', lambda r: f"{r['distinct_types']} / {r['type_entropy_bits']} / {r['top_type_share']}")
    for asp in ('identity_bearing', 'lesson_insight', 'episodic_anchor', 'active_thread', 'correction_improvement', 'noise', 'unmapped'):
        row(f'Aspect {asp}', lambda r, asp=asp: pct(r, f'aspect:{asp}'))
    for k in ('situation', 'question', 'reasoning', 'thought', 'their_raw_quote', 'my_raw_quote', 'event_time', 'evolution_status'):
        row(f'Nodes with {k}', lambda r, k=k: pct(r, f'has:{k}'))
    row('event_time precision date / month / datetime', lambda r: f"{r['event_time_precision'].get('date', 0)} / {r['event_time_precision'].get('month', 0)} / {r['event_time_precision'].get('datetime', 0)}")
    row('event_time year outside the source clock', lambda r: r['event_time_year_outside_source'])
    row('Words: content / situation / reasoning / thought (totals)', lambda r: ' / '.join(str(r['field_words'].get(k, 0)) for k in ('content', 'situation', 'reasoning', 'thought')))
    row('Content specificity per 100 words: numbers / dates / capitalized / quoted', lambda r: ' / '.join(f"{round(100 * r['specificity'].get(k, 0) / max(r['specificity'].get('content_words', 1), 1), 1)}" for k in ('numbers', 'dates', 'capitalized', 'quoted')))
    row('Titles with a name, number or quoted term', lambda r: pct(r, 'title_specific'))
    row('Situation in trigger register', lambda r: pct(r, 'situation_trigger_register', 'has:situation'))
    row('Situation restating its title (Jaccard ≥ .5)', lambda r: pct(r, 'situation_restates_title', 'has:situation'))
    row('Question phrased as a question', lambda r: pct(r, 'question_is_question', 'has:question'))
    row("Question from the encoder's side ('did I recommend')", lambda r: pct(r, 'question_encoder_side', 'has:question'))
    row('Reasoning citing provenance (turn, said, date)', lambda r: pct(r, 'reasoning_cites_provenance', 'has:reasoning'))
    row('Turn coordinates in reasoning / content / situation', lambda r: ' / '.join(str(r['counts'].get(f'turn_coord:{k}', 0)) for k in ('reasoning', 'content', 'situation')))
    row("Dates from outside the source's years in reasoning / content (encoder clock leak)", lambda r: ' / '.join(str(r['counts'].get(f'clock_leak:{k}', 0)) for k in ('reasoning', 'content')))
    row('Recall lanes per node (title + situation + question + event_time + quotes + edges): mean / nodes with ≤3', lambda r: f"{round(r['counts'].get('lanes_total', 0) / max(r['nodes'], 1), 2)} / {sum(r['counts'].get(f'lanes:{i}', 0) for i in (1, 2, 3))}")
    row('Confidence: distinct values / stdev', lambda r: f"{r['confidence_summary']['distinct']} / {r['confidence_summary']['stdev']}")
    row('Emotion non-zero / labelled', lambda r: f"{r['counts'].get('emotion_nonzero', 0)} / {r['counts'].get('emotion_labelled', 0)}")
    row('Quotes: total / verbatim / style-only / marked splice / not in source', lambda r: ' / '.join(str(r['quotes'].get(k, 0)) for k in ('total', 'verbatim', 'style_only', 'ellipsis_splice', 'not_in_source')))
    row('Edges / distinct relations / generic-relation edges', lambda r: f"{r['edges']} / {len(r['relations'])} / {r['generic_relations']}")
    row('Edge words / descriptions restating endpoint titles / duplicate pairs', lambda r: f"{r['edge_words']} / {r['edge_restates_titles']} / {r['dup_edge_pairs']}")
    row('Correction-aspect edges (walked on recall)', lambda r: r['correction_edges'])
    row('Isolates / components / max degree (median per sequence)', lambda r: f"{r['isolates']} / {r['components']} / {statistics.median(r['max_degree']) if r['max_degree'] else 0}")
    row('Near-twin pairs (title Jaccard ≥ .5 or content ≥ .45)', lambda r: r['near_twin_pairs'])
    (out / f'QUALITY-CENSUS-{a.label}.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f"\nwritten: {out / f'quality_census_{a.label}.json'}, {out / f'QUALITY-CENSUS-{a.label}.md'}")

if __name__ == '__main__':
    main()
