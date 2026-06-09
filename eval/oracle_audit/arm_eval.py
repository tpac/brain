#!/usr/bin/env python3
"""ARM EVAL — answer-presence Control vs A (vs B/C later) on recall_corpus_v2.

Pass criterion (Tom 2026-06-08): the answer is recoverable from the SELECTED node set.
Deterministic proxy: a node_gold_primary (where the answer lives) appears in top-K.
  pass@5  ~ what reaches awareness (surfacer feeds on the top of the pool)
  pass@25 ~ the retrieval pool the surfacer chooses from
(An LLM answer-judge reading the selected nodes + the `answer` field is the fidelity upgrade if
the deterministic signal is ambiguous — not needed unless gold-presence under-counts.)

Daemon-safe (IsolatedBrain). Cache cleared between arm flips (cache key is not arm-keyed).
Extensible: add 'B','C' to ARMS once their fusion lands. Usage: ./dev python3 eval/oracle_audit/arm_eval.py
"""
import os, sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/recall_corpus_v2.json'))['queries']
ARMS = ['control', 'A']


def _clear(b):
    if hasattr(b, '_recall_cache'):
        try: b._recall_cache.clear()
        except Exception: pass


def topset(b, query, n=25):
    out = b.recall(query=query, limit=n)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    return [(r.get('id') or r.get('node_id'))[:8] for r in res]


def present(gold, ids):
    s = set(ids)
    return any(g[:8] in s for g in gold)


def rank_first(gold, ids):
    gp = {g[:8] for g in gold}
    for i, x in enumerate(ids, 1):
        if x in gp:
            return i
    return None


with IsolatedBrain() as env:
    b = env.brain
    rows = []
    for q in CORPUS:
        gold = q.get('node_gold_primary', [])
        per = {}
        for arm in ARMS:
            if arm == 'control':
                os.environ.pop('BRAIN_RECALL_ARM', None)
            else:
                os.environ['BRAIN_RECALL_ARM'] = arm
            _clear(b)
            ids = topset(b, q['query_rich'], 25)
            per[arm] = (present(gold, ids[:5]), present(gold, ids[:25]), rank_first(gold, ids))
        os.environ.pop('BRAIN_RECALL_ARM', None)
        rows.append((q['id'], q['kind'], q.get('discriminates'), gold, per))

    print("%-5s %-15s %-6s | ctrl 5/25 rk | A 5/25 rk | rank lift" % ("id", "kind", "disc"))
    print("-" * 74)
    agg = {a: [0, 0] for a in ARMS}
    ranks = {a: [] for a in ARMS}
    for qid, kind, disc, gold, per in rows:
        for arm in ARMS:
            agg[arm][0] += int(per[arm][0]); agg[arm][1] += int(per[arm][1])
            if per[arm][2]:
                ranks[arm].append(per[arm][2])
        c, a = per['control'], per['A']
        yn = lambda t: "%s/%s" % ('Y' if t[0] else '.', 'Y' if t[1] else '.')
        cr = str(c[2]) if c[2] else '-'
        ar = str(a[2]) if a[2] else '-'
        lift = ('%+d' % (c[2] - a[2])) if (c[2] and a[2]) else ''
        print("%-5s %-15s %-6s | %-7s %-4s | %-6s %-3s | %s"
              % (qid, kind[:15], str(disc)[:6], yn(c), cr, yn(a), ar, lift))
    n = len(rows)
    print("-" * 74)
    for arm in ARMS:
        rr = ranks[arm]; mean = (sum(rr) / len(rr)) if rr else 0
        print("  %-8s pass@5=%d/%d pass@25=%d/%d  mean rank-of-first-gold=%.1f (n=%d)"
              % (arm, agg[arm][0], n, agg[arm][1], n, mean, len(rr)))
    lifted = [(qid, per['control'][2], per['A'][2]) for qid, k, dsc, g, per in rows
              if per['control'][2] and per['A'][2] and per['A'][2] < per['control'][2]]
    print("  arm A improved gold RANK on:", [(q, '%d->%d' % (cr, ar)) for q, cr, ar in lifted] or 'none')
    drops = [qid for qid, k, dsc, g, per in rows if k == 'control' and per['control'][0] and not per['A'][0]]
    print("  control answers DROPPED @5 by A:", drops or 'none')
