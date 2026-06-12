#!/usr/bin/env python3
"""TOPIC-REGRESSION DECOMPOSITION — why does the idf title boost lose TO1/TO4/TO5/TO6
golds that the flat 'add' boost kept? For each query, for each gold essential:
rank + emb_sim + replicated boost under BOTH arms, with per-term contributions.
Then the top-8 under idf with their boosts, marking golds — shows who displaced whom
and whether the regression is the gold's boost SHRINKING (denominator dilution) or
competitors GAINING. Daemon-safe (IsolatedBrain)."""
import os, sys, json, math, string
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers.brain_constants import TITLE_MATCH_BOOST  # noqa: E402

QIDS = ['TO1', 'TO4', 'TO5', 'TO6']
QS = [q for q in json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
      if q['id'] in QIDS]
ENV = 'BRAIN_TITLE_BOOST'


def bust(b):
    if hasattr(b, '_recall_cache') and isinstance(b._recall_cache, dict):
        b._recall_cache.clear()


def rank_map(b, query, arm, limit=200):
    os.environ[ENV] = arm
    bust(b)
    out = b.recall(query=query, limit=limit)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    os.environ.pop(ENV, None)
    rows = []
    for i, r in enumerate(res, 1):
        rows.append(((r.get('id') or r.get('node_id') or '')[:8], i,
                     r.get('embedding_similarity'), r.get('title') or ''))
    return rows


with IsolatedBrain() as env:
    b = env.brain
    # df over all node titles (same basis the idf mode uses)
    titles_l = [r[0].lower() for r in b.conn.execute(
        "SELECT title FROM nodes WHERE COALESCE(archived,0)=0 AND title IS NOT NULL").fetchall()]
    n_titles = max(len(titles_l), 1)

    def idf_of(term):
        df = sum(1 for t in titles_l if term in t)
        return math.log((n_titles + 1) / (df + 1)), df

    for q in QS:
        query = q['query']
        raw_terms = set(query.lower().split())
        clean_terms = {t.strip(string.punctuation) for t in raw_terms} - {''}
        idf = {t: idf_of(t) for t in clean_terms}
        idf_total = sum(w for w, _ in idf.values()) or 1.0

        print("\n" + "=" * 90)
        print("%s | %s" % (q['id'], query))
        print("  idf weights (term: idf, df):  " + "  ".join(
            "%s:%.2f(df%d)" % (t, w, df) for t, (w, df) in sorted(idf.items(), key=lambda x: -x[1][0])))

        def add_boost(title):
            t = title.lower()
            hits = [x for x in raw_terms if x in t]
            return (len(hits) / len(raw_terms)) * TITLE_MATCH_BOOST, hits

        def idf_boost(title):
            t = title.lower()
            hits = [x for x in clean_terms if x in t]
            return (sum(idf[x][0] for x in hits) / idf_total) * TITLE_MATCH_BOOST, hits

        arms = {arm: rank_map(b, query, arm) for arm in ('add', 'idf')}
        pos = {arm: {nid: (rk, sim, ti) for nid, rk, sim, ti in rows}
               for arm, rows in arms.items()}

        print("  -- gold essentials --")
        for g in q['gold_essential']:
            g8 = g[:8]
            a = pos['add'].get(g8)
            i = pos['idf'].get(g8)
            ti = (a or i or (None, None, ''))[2]
            ab, ah = add_boost(ti)
            ib, ih = idf_boost(ti)
            print("  %s  rank add:%s → idf:%s   emb=%s" % (
                g8, a[0] if a else '—', i[0] if i else '—',
                ('%.3f' % a[1]) if a and a[1] is not None else '?'))
            print("        title: %s" % ti[:70])
            print("        add boost=%.3f (hits: %s)" % (ab, ",".join(sorted(ah)) or '-'))
            print("        idf boost=%.3f (hits: %s)" % (ib, ",".join(sorted(ih)) or '-'))

        print("  -- top-8 under idf (gold marked ★) --")
        gold8 = {g[:8] for g in q['gold_essential']}
        for nid, rk, sim, ti in arms['idf'][:8]:
            ib, ih = idf_boost(ti)
            ab, _ = add_boost(ti)
            mark = '★' if nid in gold8 else ' '
            print("   %s r%-3d %s emb=%-5s idf_boost=%.3f add_boost=%.3f  %s" % (
                mark, rk, nid, ('%.3f' % sim) if sim is not None else '—', ib, ab, ti[:48]))
