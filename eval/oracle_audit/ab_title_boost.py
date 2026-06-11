#!/usr/bin/env python3
"""A/B: BRAIN_TITLE_BOOST arms on the control corpus (recall-only, no LLM, daemon-safe).

Arms (see knob comment in brain_recall.py STEP 6):
  add  — production: raw whitespace terms + substring containment + flat additive
  idf  — punctuation-stripped terms weighted by rarity across node titles
  idf2 — idf + tokenization (keeps ex.co) + stopword floor + word-boundary match

Scoreboard is helpful-aware (TO1/TO4/TO6 lesson, ab_topic_decomp.py): an essential
displaced by a gold_helpful node is a SOFT displacement, not a regression. Metrics:
  fails     — ≥1 essential out of top-5 (the strict original metric)
  hardfails — ≥1 essential out of top-5 AND ≥1 junk (non-gold) node in top-5
  top5/top25 essential coverage, and mean top-5 gold density (ess∪helpful)

One IsolatedBrain. Flips env + busts the recall cache between arms. Also prints the
EP5 targeted rank check. Usage: ./dev python3 eval/oracle_audit/ab_title_boost.py"""
import os, sys, json, re, time
# Import the repo this script physically lives in (worktree-safe).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
TIMED = {'episode'}
ENV = 'BRAIN_TITLE_BOOST'
ARMS = ['add', 'idf', 'idf2']

EP5_QUERY = "what did we do on the last session we worked on ex.co?"
EP5_TARGETS = ['b3b6ce2a', 'dabb3078', 'b8b8370b', '7b14f270', '8359cf1d']


def cut(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


def bust(b):
    if hasattr(b, '_recall_cache') and isinstance(b._recall_cache, dict):
        b._recall_cache.clear()


def run_arm(b, arm, elig_cache):
    os.environ[ENV] = arm
    bust(b)
    t0 = time.time()
    rows = []
    fails = hardfails = tot = in5 = in25 = 0
    golddense = []
    permode, permode_hard = {}, {}
    for q in QS:
        ess = q.get('gold_essential', [])
        if not ess:
            continue
        gold_all = {g[:8] for g in ess} | {g[:8] for g in q.get('gold_helpful', [])}
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
        junk5 = sum(1 for n in ids[:5] if n not in gold_all)
        fail = any(e[:8] not in t5 for e in e_arm)
        hard = fail and junk5 > 0
        tot += len(e_arm); in5 += e5; in25 += e25
        fails += 1 if fail else 0
        hardfails += 1 if hard else 0
        golddense.append((len(ids[:5]) - junk5) / max(len(ids[:5]), 1))
        permode[q['mode']] = permode.get(q['mode'], 0) + (1 if fail else 0)
        permode_hard[q['mode']] = permode_hard.get(q['mode'], 0) + (1 if hard else 0)
        rows.append((q['id'], q['mode'], len(e_arm), e5, e25, fail, hard, junk5))
    os.environ.pop(ENV, None)
    return {'rows': rows, 'fails': fails, 'hardfails': hardfails, 'tot': tot,
            'in5': in5, 'in25': in25, 'permode': permode, 'permode_hard': permode_hard,
            'n': len(rows), 'golddense': sum(golddense) / max(len(golddense), 1),
            'wall_s': time.time() - t0}


def ep5_ranks(b, arm):
    os.environ[ENV] = arm
    bust(b)
    out = b.recall(query=EP5_QUERY, limit=300)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    ids = [(r.get('id') or r.get('node_id'))[:8] for r in res]
    os.environ.pop(ENV, None)
    return {t: (ids.index(t[:8]) + 1) if t[:8] in ids else None for t in EP5_TARGETS}


with IsolatedBrain() as env:
    b = env.brain
    elig_cache = {}
    for q in QS:
        c = cut(q)
        if c:
            elig_cache[q['id']] = {x[0][:8] for x in b.conn.execute(
                "SELECT id FROM nodes WHERE created_at<=? AND COALESCE(archived,0)=0",
                (c,)).fetchall()}

    results = {arm: run_arm(b, arm, elig_cache) for arm in ARMS}

    print("\n=== BRAIN_TITLE_BOOST A/B — control corpus (%d scored queries) ===" % results['add']['n'])
    print("%-5s  %-9s  %-10s  %-13s  %-13s  %-9s  %-7s" % (
        "arm", "fails/N", "HARDfails", "top5", "top25", "gold-dens5", "wall"))
    for arm in ARMS:
        r = results[arm]
        print("%-5s  %-9s  %-10s  %-13s  %-13s  %-9s  %.0fs" % (
            arm, "%d/%d" % (r['fails'], r['n']), "%d/%d" % (r['hardfails'], r['n']),
            "%d/%d (%.0f%%)" % (r['in5'], r['tot'], 100.0 * r['in5'] / max(r['tot'], 1)),
            "%d/%d (%.0f%%)" % (r['in25'], r['tot'], 100.0 * r['in25'] / max(r['tot'], 1)),
            "%.0f%%" % (100.0 * r['golddense']), r['wall_s']))

    modes = ['trigger', 'topic', 'heavy', 'remote', 'episode']
    print("\n--- fails by mode (strict / hard) ---")
    print("%-5s  %s" % ("arm", "  ".join("%-9s" % m for m in modes)))
    for arm in ARMS:
        pm, ph = results[arm]['permode'], results[arm]['permode_hard']
        print("%-5s  %s" % (arm, "  ".join("%-9s" % ("%d/%d" % (pm.get(m, 0), ph.get(m, 0)))
                                            for m in modes)))

    base = {r[0]: r for r in results['add']['rows']}
    print("\n--- per-query changes vs add (e5 essentials in top-5; j5 junk in top-5) ---")
    any_flip = False
    for arm in ['idf', 'idf2']:
        for r in results[arm]['rows']:
            qid, mode, ness, e5, e25, fail, hard, junk5 = r
            b0 = base.get(qid)
            if b0 and (e5 != b0[3] or e25 != b0[4] or junk5 != b0[7]):
                any_flip = True
                arrow = "↑" if e5 > b0[3] else ("↓" if e5 < b0[3] else "·")
                print("  [%-4s] #%-4s %-8s  e5 %d→%d %s  e25 %d→%d  j5 %d→%d  (of %d ess)%s"
                      % (arm, qid, mode, b0[3], e5, arrow, b0[4], e25, b0[7], junk5,
                         ness, "  HARD" if hard else ""))
    if not any_flip:
        print("  (no changes)")

    print("\n--- EP5 targeted rank check (untimed; — = absent from top-300 pool) ---")
    print("%-5s  %s" % ("arm", "  ".join("%-10s" % t for t in EP5_TARGETS)))
    for arm in ARMS:
        pos = ep5_ranks(b, arm)
        print("%-5s  %s" % (arm, "  ".join("%-10s" % (pos[t] if pos[t] else '—') for t in EP5_TARGETS)))
