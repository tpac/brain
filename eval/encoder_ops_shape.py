#!/usr/bin/env python3
"""Deep encode-shape metrics over intercepted-op dumps — a port of
eval/longmem/corpus_shape.py + eval/s1s_ab_quality_analyzer.py metrics to the
op layer (those read frozen brains; arm-F runs never execute their writes).

USE
    ./dev python3 eval/encoder_ops_shape.py <dir holding ops*/ dumps from encoder_prompt_ab.py --dump-ops>
The 2026-09 A/B dumps live outside git at
/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/ (private conversation content).

Debt, named: this belongs inside encoder_prompt_ab.score_shape so every run
gets it for free (extend before creating); kept standalone so the metric
definitions survive the session that ported them. See docs/S1E-CHECKLIST.md E22.

Adds the metrics those tools list as NOT computable from a brain:
content_edits vs full-content rewrite, connect_to vs standalone connect,
op mix per run. Specificity retention uses the frozen payload's <timeline>
as the source text.
"""
import glob, json, math, os, re, sys
from collections import Counter
from statistics import mean

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()  # dir holding ops*/ dumps
P = '/Users/tpac/AgentsContext/s1e-field-coverage-gold/payloads'
PAYLOAD = {  # chain → capture
    's1e-61608651-16': P + '/2026-08-31/s1e-61608651-16/000-prompt.md',
    's1e-831149ce-5':  P + '/2026-09-01/s1e-831149ce-5/000-prompt.md',
    's1e-5076cdc2-17': P + '/2026-08-31/s1e-5076cdc2-17/000-prompt.md',
    's1e-67015b88-17': P + '/2026-08-31/s1e-67015b88-17/000-prompt.md',
    's1e-17d9ae94-44': P + '/2026-08-17/s1e-17d9ae94-44/000-prompt.md',
}
ARMS = [  # (label, glob) — order = table order
    ('v41',          'ops/*-v41/*.json'),   ('v41 rep',      'ops/*-v41-rep/*.json'),
    ('v42',          'ops/*-v42/*.json'),   ('v42 rep',      'ops/*-v42-rep/*.json'),
    ('v41+gistA',    'ops3/*-v41gist/*.json'),
    ('ideas+gistA',  'ops3/*-ideasgist/*.json'),
    ('ideas',        'ops3/*-ideas/*.json'),
    ('v41+gistB',    'ops4/*-v41gistB/*.json'),
    ('v9+gistB',     'ops4/*-v9gistB/*.json'),
]
MERGE = {'v41 rep': 'v41', 'v42 rep': 'v42'}
GENERIC = {'related', 'related_to'}
RESCUE = {'similar_to', 'corrects', 'supersedes', 'refines', 'grounds', 'contradicts'}
ABSTRACT = {'principle', 'lesson', 'mechanism', 'pattern', 'rule', 'architecture', 'insight'}
HEX8 = re.compile(r'^[0-9a-f]{8}$')
NOUN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b")
NUM = re.compile(r"\b(\d+(?:[.,]\d+)?)\b")
STOP = set("I The A An This That These Those Here There When Where Why How What Who My Your Our Their His Her Its Just So But And Or If Then Yes No Okay OK Thanks Thank Sure Now Some Any All Each Every Many Most Several Few Can Could Would Should Will Shall May Might Must Do Does Did Have Has Had Is Are Was Were Be Been Being Let Go See Look Good Great Nice".split())

_src_cache = {}
def source_tokens(chain):
    if chain in _src_cache: return _src_cache[chain]
    t = open(PAYLOAD[chain]).read()
    m = re.search(r'<timeline[^>]*>(.*)</timeline>', t, re.S)
    tl = m.group(1) if m else ''
    tl = re.sub(r'<[^>]+>', ' ', tl)  # strip tags; keep spoken text + action cues
    nouns = {p for p in (x.group(1) for x in NOUN.finditer(tl))
             if ' ' in p or (len(p.split()[0]) > 2 and p.split()[0] not in STOP)}
    nums = set(NUM.findall(tl))
    _src_cache[chain] = (nouns, nums); return _src_cache[chain]

def ops_of(d):
    for w in d['writes']:
        a = w.get('args') or {}
        for key in ('operations', 'nodes', 'revisions', 'connections'):
            v = a.get(key)
            if isinstance(v, list):
                for op in v: yield op
                break
        else:
            if a: yield a

def kind(op):
    if op.get('op'): return op['op']
    if op.get('node_id'): return 'revise'
    if op.get('source_id') and op.get('target_id'): return 'connect'
    return 'remember'

def score_run(path):
    d = json.load(open(path)); chain = os.path.basename(path).split('-F-')[0]
    ops = list(ops_of(d))
    rem = [o for o in ops if kind(o) == 'remember']
    rev = [o for o in ops if kind(o) in ('revise', 'absorb')]
    con = [o for o in ops if kind(o) == 'connect']
    r = Counter()
    r['runs'] = 1; r['creates'] = len(rem); r['revises'] = len(rev); r['connects'] = len(con)
    r['archives'] = sum(1 for o in ops if kind(o) in ('archive', 'disconnect'))
    # field population on creates
    for f in ('situation', 'reasoning', 'question', 'thought', 'event_time', 'their_raw_quote', 'my_raw_quote'):
        r['f_' + f] = sum(1 for o in rem if str(o.get(f) or '').strip())
    r['f_source_refs'] = sum(1 for o in rem if o.get('source_refs'))
    refs = [x for o in rem for x in (o.get('source_refs') or [])]
    r['refs_total'] = len(refs); r['refs_hex8'] = sum(1 for x in refs if HEX8.match(str(x)))
    r['f_emotion_pair'] = sum(1 for o in rem if o.get('emotion') not in (None, 0, 0.0) and str(o.get('emotion_label') or '').lower() not in ('', 'neutral', 'none'))
    r['f_confidence_key'] = sum(1 for o in rem if isinstance(o.get('confidence'), str))
    r['content_chars'] = sum(len(str(o.get('content') or '')) for o in rem)
    r['title_chars'] = sum(len(str(o.get('title') or '')) for o in rem)
    r['title_over100'] = sum(1 for o in rem if len(str(o.get('title') or '')) > 100)
    types = Counter(str(o.get('type') or '') for o in rem)
    r['types_distinct'] = len(types)
    r['abstract_nodes'] = sum(1 for o in rem if o.get('type') in ABSTRACT)
    r['abstract_with_grounds'] = sum(1 for o in rem if o.get('type') in ABSTRACT and any((c.get('relation') == 'grounds') for c in (o.get('connect_to') or [])))
    # edges: connect_to on creates + standalone connects
    whys, rels = [], Counter()
    deg = [len(o.get('connect_to') or []) for o in rem]
    id_t = title_t = 0
    for o in rem:
        for c in (o.get('connect_to') or []):
            t = str(c.get('title') or '').strip()
            if HEX8.match(t): id_t += 1
            else: title_t += 1
            rel = str(c.get('relation') or ''); rels[rel] += 1
            ws = [c.get('why')] + [x.get('why') for x in (c.get('relations') or []) if isinstance(x, dict)]
            whys += [len(str(w)) for w in ws if w is not None]
    for o in con:
        rels[str(o.get('relation') or '')] += 1
        whys.append(len(str(o.get('description') or '')))
    r['edges'] = len(whys); r['why_chars'] = sum(whys)
    r['why_band'] = sum(1 for L in whys if 120 <= L <= 180); r['why_thin'] = sum(1 for L in whys if L < 80); r['why_empty'] = sum(1 for L in whys if L == 0)
    r['generic'] = sum(rels[x] for x in GENERIC); r['rescue'] = sum(rels[x] for x in RESCUE)
    r['ct_id'] = id_t; r['ct_title'] = title_t
    r['deg0_2'] = sum(1 for x in deg if x <= 2); r['deg0'] = sum(1 for x in deg if x == 0)
    # relation entropy (bits)
    tot = sum(rels.values())
    r['_rels'] = rels
    # revise op shape
    r['rev_content_edits'] = sum(1 for o in rev if o.get('content_edits'))
    r['rev_full_content'] = sum(1 for o in rev if o.get('content') and not o.get('content_edits'))
    r['rev_title'] = sum(1 for o in rev if o.get('title')); r['rev_situation'] = sum(1 for o in rev if o.get('situation'))
    r['rev_question'] = sum(1 for o in rev if o.get('question')); r['rev_reasoning'] = sum(1 for o in rev if o.get('reasoning'))
    r['rev_evolution'] = sum(1 for o in rev if o.get('evolution_status')); r['rev_type'] = sum(1 for o in rev if o.get('type'))
    r['rev_fields_mean_num'] = sum(len([k for k in o if o.get(k) and k not in ('op', 'node_id', 'reason')]) for o in rev)
    # specificity retention vs the timeline
    nouns, nums = source_tokens(chain)
    blob = ' '.join(' '.join(str(o.get(k) or '') for k in ('title', 'content', 'situation', 'reasoning', 'their_raw_quote', 'my_raw_quote')) for o in rem + rev)
    blob += ' '.join(str(e.get('new') or '') for o in rev for e in (o.get('content_edits') or []) if isinstance(e, dict))
    r['src_nouns'] = len(nouns); r['nouns_kept'] = sum(1 for p in nouns if p in blob)
    r['src_nums'] = len(nums); r['nums_kept'] = sum(1 for p in nums if p in blob)
    return r

def pct(a, b): return '  -' if not b else '%3.0f%%' % (100.0 * a / b)
def main():
    arms = {}
    for label, pat in ARMS:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            arms.setdefault(MERGE.get(label, label), []).append(score_run(p))
    order = [l for l in dict.fromkeys(MERGE.get(l, l) for l, _ in ARMS) if l in arms]
    def agg(rs):
        c = Counter()
        rels = Counter()
        for r in rs:
            rels.update(r['_rels'])
            c.update({k: v for k, v in r.items() if k != '_rels'})
        c['_rels'] = rels; return c
    A = {l: agg(arms[l]) for l in order}
    def row(name, fn):
        print('%-34s' % name + ''.join('%11s' % fn(A[l]) for l in order))
    print('%-34s' % '' + ''.join('%11s' % l for l in order))
    print('=' * (34 + 11 * len(order)))
    row('runs', lambda c: c['runs'])
    row('creates / run', lambda c: '%.1f' % (c['creates'] / c['runs']))
    row('revises / run', lambda c: '%.1f' % (c['revises'] / c['runs']))
    row('standalone connects / run', lambda c: '%.1f' % (c['connects'] / c['runs']))
    row('revise share of writes', lambda c: pct(c['revises'], c['creates'] + c['revises']))
    print('-- creates: field population')
    for f in ('situation', 'reasoning', 'question', 'thought', 'event_time', 'their_raw_quote', 'my_raw_quote', 'source_refs', 'emotion_pair', 'confidence_key'):
        row('  ' + f, lambda c, f=f: pct(c['f_' + f], c['creates']))
    row('  source_refs hex8-conforming', lambda c: pct(c['refs_hex8'], c['refs_total']))
    print('-- creates: content / title / types')
    row('  content chars mean', lambda c: '%d' % (c['content_chars'] / max(1, c['creates'])))
    row('  title chars mean', lambda c: '%d' % (c['title_chars'] / max(1, c['creates'])))
    row('  titles > 100 chars', lambda c: pct(c['title_over100'], c['creates']))
    row('  distinct types (sum over runs)', lambda c: c['types_distinct'])
    row('  abstract types w/ grounds edge', lambda c: pct(c['abstract_with_grounds'], c['abstract_nodes']) + ' of %d' % c['abstract_nodes'])
    print('-- edges (encoder-written)')
    row('  edges / create', lambda c: '%.2f' % (c['edges'] / max(1, c['creates'])))
    row('  degree 0 (connect_to)', lambda c: pct(c['deg0'], c['creates']))
    row('  degree 0-2 (connect_to)', lambda c: pct(c['deg0_2'], c['creates']))
    row('  why chars mean', lambda c: '%d' % (c['why_chars'] / max(1, c['edges'])))
    row('  why in 120-180 band', lambda c: pct(c['why_band'], c['edges']))
    row('  why thin (<80)', lambda c: pct(c['why_thin'], c['edges']))
    row('  why empty', lambda c: c['why_empty'])
    row('  generic relations', lambda c: c['generic'])
    row('  rescue verbs', lambda c: pct(c['rescue'], c['edges']))
    row('  catalog targets (id-form)', lambda c: pct(c['ct_id'], c['ct_id'] + c['ct_title']))
    def entropy(c):
        tot = sum(c['_rels'].values());
        return '%.2f' % -sum((v / tot) * math.log2(v / tot) for v in c['_rels'].values()) if tot else '-'
    row('  relation entropy (bits)', entropy)
    row('  distinct relations', lambda c: len(c['_rels']))
    print('-- revises: op shape (not computable from a brain)')
    row('  content_edits (patch)', lambda c: pct(c['rev_content_edits'], c['revises']))
    row('  full content rewrite', lambda c: pct(c['rev_full_content'], c['revises']))
    for f in ('title', 'situation', 'question', 'reasoning', 'type', 'evolution'):
        row('  touches ' + f, lambda c, f=f: pct(c['rev_' + f], c['revises']))
    row('  fields per revise (mean)', lambda c: '%.1f' % (c['rev_fields_mean_num'] / max(1, c['revises'])))
    print('-- specificity retention (timeline → nodes)')
    row('  proper nouns kept', lambda c: pct(c['nouns_kept'], c['src_nouns']))
    row('  numbers kept', lambda c: pct(c['nums_kept'], c['src_nums']))

if __name__ == '__main__':
    main()
