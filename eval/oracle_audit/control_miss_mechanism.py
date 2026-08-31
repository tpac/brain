#!/usr/bin/env python3
"""PER-MISS MECHANISM — for each essential node Control misses from top-25, which retrieval PROCESS
would have surfaced it (given the query)? Tests, per (query, missed-node):
  rawrank   = node's rank in raw dense cosine of the FULL query (how buried in pure embedding)
  fts       = keyword/bm25 (porter) finds it in top-50?           -> LEXICAL lane
  graph     = 1/2 hops from a node Control DID surface in top-5?  -> SPREAD / traversal
  anchor    = best rank when a SINGLE query term is embedded alone -> ENTITY-ANCHOR / query-decomp
              (if a lone term ranks it high but the full query buries it = query DILUTION)
  temporal  = days between node.created_at and the question's moment -> BY-TIME / episodic lane
Time-scope fix applied. Usage: ./dev python3 eval/oracle_audit/control_miss_mechanism.py
"""
import sys, json, re
import numpy as np
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers import embedder                      # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
TIMED = {'episode'}
STOP = set("the a an of to in on for and or is are was what how do we did you your i my it that this "
           "about again over them have has on's so but not your at as with can could".split())


def cut(q):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', q.get('moment', ''))
    return (m.group(1) + 'T23:59:59+00:00') if (m and q['mode'] in TIMED) else None


with IsolatedBrain() as env:
    b = env.brain
    model = embedder.stats.get('model_name') or None
    vmap = {}
    for r in b._vec_dal.get_all_vectors(vector_types=['_primary'], model=model):
        e = r['embedding']
        if e:
            v = np.frombuffer(e, dtype=np.float32); n = np.linalg.norm(v)
            if n:
                vmap[r['node_id'][:8]] = v / n
    allids = list(vmap.keys())
    allmat = np.vstack([vmap[i] for i in allids])

    def qvec(t):
        v = np.frombuffer(embedder.embed_query(t), dtype=np.float32); n = np.linalg.norm(v)
        return v / (n or 1.0)

    def rawrank(n8, qv):
        if n8 not in vmap:
            return None
        sN = float(vmap[n8] @ qv)
        return int((allmat @ qv > sN).sum()) + 1

    def meta(n8):
        r = b.conn.execute("SELECT title,created_at FROM nodes WHERE id = ?", (n8,)).fetchone()
        return (r[0][:46], (r[1] or '')[:10]) if r else ('?', '?')

    def neigh(n8):
        rows = b.conn.execute("SELECT target_id FROM edges WHERE source_id = ? "
                              "UNION SELECT source_id FROM edges WHERE target_id = ?",
                              (n8, n8)).fetchall()
        return {x[0][:8] for x in rows}

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
        cids = [(r.get('id') or r.get('node_id'))[:8]
                for r in (out.get('results', []) if isinstance(out, dict) else out or [])]
        if elig is not None:
            cids = [n for n in cids if n in elig]; ess = [e for e in ess if e[:8] in elig]
        t25 = set(cids[:25]); top5 = cids[:5]
        miss = [e[:8] for e in ess if e[:8] not in t25]
        if not miss:
            continue
        qv = qvec(q['query'])
        ftsQ = {n[:8] for n, _ in b._fts.search_scored(q['query'], 50)}
        hop1 = set()
        for s in top5:
            hop1 |= neigh(s)
        hop2 = set()
        for n in list(hop1)[:80]:
            hop2 |= neigh(n)
        words = [w for w in re.findall(r"[a-z][a-z._-]+", q['query'].lower()) if w not in STOP and len(w) > 2]
        mdate = (cut(q) or '')[:10]
        print("\n#%-4s [%s] \"%s\"" % (q['id'], q['mode'], q['query'][:60]))
        for n8 in miss:
            title, cr = meta(n8)
            rr = rawrank(n8, qv)
            best = (None, 10 ** 9)
            for w in set(words):
                r = rawrank(n8, qvec(w))
                if r and r < best[1]:
                    best = (w, r)
            g = '1hop' if n8 in hop1 else ('2hop' if n8 in hop2 else '-')
            print("   %s %-46s\n        rawrank=%-4s fts=%s graph=%-4s  anchor='%s'@%s  created=%s"
                  % (n8, title, rr, ('Y' if n8 in ftsQ else 'n'), g, best[0], best[1], cr))
