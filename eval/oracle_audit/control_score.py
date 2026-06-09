#!/usr/bin/env python3
"""CONTROL SCORE (recall-only, no LLM) — scores Control recall against the CURRENT corpus gold.
Use after gold edits to get the trustworthy fail-rate without re-judging (which would re-roll gold).
Time-scoped for episodes. Daemon-safe (IsolatedBrain). Usage: ./dev python3 eval/oracle_audit/control_score.py
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
    fails = 0; tot = 0; in5 = 0; in25 = 0; permode = {}
    for q in QS:
        ess = q.get('gold_essential', [])
        if not ess:
            continue
        c = cut(q); elig = None
        if c:
            elig = {x[0][:8] for x in b.conn.execute(
                "SELECT id FROM nodes WHERE created_at<=? AND COALESCE(archived,0)=0", (c,)).fetchall()}
        filt = {"created_at": {"lte": c}} if c else None
        _lim = 200 if c else 25   # recall POST-filters created_at (after scoring), so a time-scoped
        try:                       # query needs a larger limit to leave a real top-25 of ELIGIBLE nodes
            out = b.recall(query=q['query'], limit=_lim, filter=filt)
        except Exception:
            out = b.recall(query=q['query'], limit=_lim)
        ids = [(r.get('id') or r.get('node_id'))[:8]
               for r in (out.get('results', []) if isinstance(out, dict) else out or [])]
        if elig is not None:
            ids = [n for n in ids if n in elig]
            ess = [e for e in ess if e[:8] in elig]   # only gold that existed at the moment counts
        if not ess:
            continue
        t5, t25 = set(ids[:5]), set(ids[:25])
        e5 = sum(1 for e in ess if e[:8] in t5)
        e25 = sum(1 for e in ess if e[:8] in t25)
        tot += len(ess); in5 += e5; in25 += e25
        f = any(e[:8] not in t5 for e in ess)
        if f:
            fails += 1
        permode[q['mode']] = permode.get(q['mode'], 0) + (1 if f else 0)
        print("#%-4s %-8s ess=%d in5=%d in25=%d %s" % (q['id'], q['mode'], len(ess), e5, e25, 'FAIL' if f else ''))
    n = sum(1 for q in QS if q.get('gold_essential'))
    print("\nCONTROL FAILS (>=1 essential missing from top-5): %d/%d" % (fails, n))
    print("essential coverage — top-5: %d/%d (%.0f%%)   top-25: %d/%d (%.0f%%)"
          % (in5, tot, 100.0 * in5 / max(tot, 1), in25, tot, 100.0 * in25 / max(tot, 1)))
    print("fails by mode:", permode)
