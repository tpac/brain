import json, glob, collections, re, sys
rows = []
for f in glob.glob('content_quality_carriers_v37*.jsonl'):
    for l in open(f):
        if l.strip(): rows.append(json.loads(l))
seen = set(); uniq = []
for r in rows:
    k = (r['arm'], r['repeat'], r['corpus'], r['id'])
    if k in seen: continue
    seen.add(k); uniq.append(r)
QM = re.compile(r'quote', re.I)
print(f"{'arm':12}{'rep':4} nodes  generic  owner✗  overcl  mixed  fabr  ev_wrong  quote-mismatch(evidence)  method: n owner✗ overcl+mixed  parse✗")
tot = collections.defaultdict(collections.Counter)
for arm in ('v3_6_full','v3_7_advice','v3_7_quote'):
    for rep in (1,2,3):
        rs = [r for r in uniq if r['arm']==arm and int(str(r['repeat']).replace('repeat',''))==rep]
        if not rs: continue
        c = collections.Counter(); c['n'] = len(rs)
        for r in rs:
            v = r['verdict']
            if not v: c['parse'] += 1; continue
            c['generic'] += v.get('value')=='generic_advice'
            c['owner'] += v.get('owner_ok') is False
            c['over'] += v.get('fidelity')=='overclaimed'; c['mixed'] += v.get('fidelity')=='mixed'; c['fab'] += v.get('fidelity')=='fabricated_element'
            c['ev'] += (v.get('fields') or {}).get('event_time')=='wrong'
            ev = str(v.get('fidelity_evidence',''))+' '+str(v.get('defect',''))
            if QM.search(ev) and re.search(r'(old|earlier|pre-|previous|stale|contradict|mismatch|outdated|superseded)', ev, re.I): c['qmis'] += 1
            if r.get('type')=='method':
                c['m_n'] += 1; c['m_owner'] += v.get('owner_ok') is False; c['m_over'] += v.get('fidelity') in ('overclaimed','mixed')
        for k,vv in c.items(): tot[arm][k] += vv
        print(f"{arm:12}{rep:<4} {c['n']:5}  {c['generic']:7}  {c['owner']:6}  {c['over']:6}  {c['mixed']:5}  {c['fab']:4}  {c['ev']:8}  {c['qmis']:24}  {c['m_n']:3} {c['m_owner']:6} {c['m_over']:12}  {c['parse']:6}")
    t = tot[arm]
    if t['n']:
        n = t['n']-t['parse'] or 1
        print(f"{arm:12}{'all':4} {t['n']:5}  {t['generic']:7} ({100*t['generic']/n:.0f}%)  {t['owner']:6} ({100*t['owner']/n:.0f}%)  {t['over']:6} ({100*t['over']/n:.0f}%)  {t['mixed']:5} ({100*t['mixed']/n:.0f}%)  {t['fab']:4} ({100*t['fab']/n:.0f}%)  {t['ev']:8} ({100*t['ev']/n:.0f}%)  {t['qmis']:5}  method {t['m_n']} owner✗ {t['m_owner']} over+mixed {t['m_over']}\n")
if len(sys.argv) > 1:
    arm = sys.argv[1]
    print(f'\n== {arm}: generic_advice nodes'); [print(f"  r{r['repeat']} {r['corpus'][3:12]} [{r['type']}] {r['title'][:80]} :: {str(r['verdict'].get('defect',''))[:120]}") for r in uniq if r['arm']==arm and r['verdict'] and r['verdict'].get('value')=='generic_advice']
    print(f'\n== {arm}: method nodes owner_ok false'); [print(f"  r{r['repeat']} {r['corpus'][3:12]} {r['title'][:80]} :: {str(r['verdict'].get('fidelity_evidence',''))[:130]}") for r in uniq if r['arm']==arm and r['verdict'] and r['type']=='method' and r['verdict'].get('owner_ok') is False]
    print(f'\n== {arm}: quote-mismatch evidence'); [print(f"  r{r['repeat']} {r['corpus'][3:12]} {r['title'][:70]} :: {(str(r['verdict'].get('fidelity_evidence',''))+' | '+str(r['verdict'].get('defect','')))[:200]}") for r in uniq if r['arm']==arm and r['verdict'] and QM.search(str(r['verdict'].get('fidelity_evidence',''))+str(r['verdict'].get('defect','')))]
