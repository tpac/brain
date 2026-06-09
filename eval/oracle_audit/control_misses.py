#!/usr/bin/env python3
"""SHOW THE MISSES — exactly which essential gold nodes Control fails to retrieve, by band:
  Band A: NOT in top-25  -> the hard ~17% (recall genuinely can't surface it)
  Band B: in top-25 but NOT in top-5 -> retrieved but not near the top (surfacer/spread territory)
Time-scope fix applied (limit=200 for episodic). Usage: ./dev python3 eval/oracle_audit/control_misses.py
"""
import sys, json, re
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
TIMED = {'episode'}


def cut(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


with IsolatedBrain() as env:
    b = env.brain

    def title(n8):
        r = b.conn.execute("SELECT type,title FROM nodes WHERE id LIKE ?", (n8 + '%',)).fetchone()
        return (r[0][:4] + ': ' + r[1][:58]) if r else '?'

    bandA, bandB = [], []
    for q in QS:
        ess = q.get('gold_essential', [])
        if not ess:
            continue
        c = cut(q); elig = None
        if c:
            elig = {x[0][:8] for x in b.conn.execute(
                "SELECT id FROM nodes WHERE created_at<=? AND COALESCE(archived,0)=0", (c,)).fetchall()}
        filt = {"created_at": {"lte": c}} if c else None
        lim = 200 if c else 25
        try:
            out = b.recall(query=q['query'], limit=lim, filter=filt)
        except Exception:
            out = b.recall(query=q['query'], limit=lim)
        ids = [(r.get('id') or r.get('node_id'))[:8]
               for r in (out.get('results', []) if isinstance(out, dict) else out or [])]
        if elig is not None:
            ids = [n for n in ids if n in elig]
            ess = [e for e in ess if e[:8] in elig]
        t5, t25 = set(ids[:5]), set(ids[:25])
        for e in ess:
            e8 = e[:8]
            if e8 not in t25:
                bandA.append((q['id'], q['mode'], e8, title(e8)))
            elif e8 not in t5:
                bandB.append((q['id'], q['mode'], e8, title(e8)))

    print("=" * 80)
    print("BAND A — essential MISSING from top-25 entirely (the hard ~17%%): %d nodes" % len(bandA))
    for qid, mode, n, t in bandA:
        print("  #%-4s %-8s %s  %s" % (qid, mode, n, t))
    print("\nBAND B — in top-25 but NOT in top-5 (retrieved, just not near top): %d nodes" % len(bandB))
    for qid, mode, n, t in bandB:
        print("  #%-4s %-8s %s  %s" % (qid, mode, n, t))
    from collections import Counter
    print("\nBand A (hard miss) by mode:", dict(Counter(x[1] for x in bandA)))
