#!/usr/bin/env python3
"""A/B: BRAIN_ENRICH_SCORE arms on the control corpus (recall-only, no LLM, daemon-safe).

Tests the STEP 3.5 enrichment-scoring change (top-2-avg burial fix):
  avg2     — current production (avg of top-2 weighted vectors)
  max      — best vector wins (never buries a strong single-field match)
  maxbonus — best vector + 0.25 * second (full-strength best + agreement bonus)

One IsolatedBrain (one DB copy). Flips the env var + busts the recall cache between
arms so arms can't serve each other's cached results. Time-scopes episode questions
exactly like control_score.py. Usage: ./dev python3 eval/oracle_audit/ab_enrich_score.py
"""
import os, sys, json, re
# Import the repo this script physically lives in (worktree-safe — hardcoding
# /Users/tpac/brain would silently import the MAIN checkout, not the worktree).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
TIMED = {'episode'}
ARMS = ['avg2', 'max', 'maxbonus']


def cut(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


def bust_cache(b):
    if hasattr(b, '_recall_cache') and isinstance(b._recall_cache, dict):
        b._recall_cache.clear()


def run_arm(b, arm, elig_cache):
    os.environ['BRAIN_ENRICH_SCORE'] = arm
    bust_cache(b)
    rows = []   # per-query: (id, mode, n_ess, e5, e25, fail)
    fails = tot = in5 = in25 = 0
    permode = {}
    for q in QS:
        ess = q.get('gold_essential', [])
        if not ess:
            continue
        c = cut(q)
        elig = elig_cache.get(q['id'])
        filt = {"created_at": {"lte": c}} if c else None
        _lim = 200 if c else 25
        try:
            out = b.recall(query=q['query'], limit=_lim, filter=filt)
        except Exception:
            out = b.recall(query=q['query'], limit=_lim)
        ids = [(r.get('id') or r.get('node_id'))[:8]
               for r in (out.get('results', []) if isinstance(out, dict) else out or [])]
        e_arm = list(ess)
        if elig is not None:
            ids = [n for n in ids if n in elig]
            e_arm = [e for e in ess if e[:8] in elig]
        if not e_arm:
            continue
        t5, t25 = set(ids[:5]), set(ids[:25])
        e5 = sum(1 for e in e_arm if e[:8] in t5)
        e25 = sum(1 for e in e_arm if e[:8] in t25)
        fail = any(e[:8] not in t5 for e in e_arm)
        tot += len(e_arm); in5 += e5; in25 += e25
        fails += 1 if fail else 0
        permode[q['mode']] = permode.get(q['mode'], 0) + (1 if fail else 0)
        rows.append((q['id'], q['mode'], len(e_arm), e5, e25, fail))
    os.environ.pop('BRAIN_ENRICH_SCORE', None)
    return {'rows': rows, 'fails': fails, 'tot': tot, 'in5': in5, 'in25': in25,
            'permode': permode, 'n': len(rows)}


with IsolatedBrain() as env:
    b = env.brain
    # Precompute eligibility sets once (time-scoped episodes) — identical across arms.
    elig_cache = {}
    for q in QS:
        c = cut(q)
        if c:
            elig_cache[q['id']] = {x[0][:8] for x in b.conn.execute(
                "SELECT id FROM nodes WHERE created_at<=? AND COALESCE(archived,0)=0",
                (c,)).fetchall()}

    results = {arm: run_arm(b, arm, elig_cache) for arm in ARMS}

    # ---- Summary table ----
    print("\n=== BRAIN_ENRICH_SCORE A/B — control corpus (%d scored queries) ===" % results['avg2']['n'])
    print("%-9s  %-9s  %-9s  %-9s" % ("arm", "fails/N", "top5", "top25"))
    for arm in ARMS:
        r = results[arm]
        print("%-9s  %-9s  %-9s  %-9s" % (
            arm,
            "%d/%d" % (r['fails'], r['n']),
            "%d/%d (%.0f%%)" % (r['in5'], r['tot'], 100.0 * r['in5'] / max(r['tot'], 1)),
            "%d/%d (%.0f%%)" % (r['in25'], r['tot'], 100.0 * r['in25'] / max(r['tot'], 1)),
        ))

    print("\n--- fails by mode ---")
    modes = ['trigger', 'topic', 'heavy', 'remote', 'episode']
    print("%-9s  %s" % ("arm", "  ".join("%-8s" % m for m in modes)))
    for arm in ARMS:
        pm = results[arm]['permode']
        print("%-9s  %s" % (arm, "  ".join("%-8d" % pm.get(m, 0) for m in modes)))

    # ---- Per-query flips vs avg2 baseline ----
    base = {r[0]: r for r in results['avg2']['rows']}
    print("\n--- per-query changes vs avg2 (e5 = essentials in top-5) ---")
    any_flip = False
    for arm in ['max', 'maxbonus']:
        for r in results[arm]['rows']:
            qid, mode, ness, e5, e25, fail = r
            b0 = base.get(qid)
            if b0 and (e5 != b0[3] or e25 != b0[4]):
                any_flip = True
                arrow = "↑" if e5 > b0[3] else "↓"
                print("  [%s] #%-4s %-8s  e5 %d→%d %s  e25 %d→%d  (of %d ess)"
                      % (arm, qid, mode, b0[3], e5, arrow, b0[4], e25, ness))
    if not any_flip:
        print("  (no top-5/top-25 changes — enrichment scoring didn't move any gold node)")
