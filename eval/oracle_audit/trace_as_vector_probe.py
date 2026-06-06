#!/usr/bin/env python3
"""TRACE-AS-VECTOR probe (Tom's synthesis idea) — isolated copy.

Idea: episodic recall finds the matched trace ("Your last EX.CO session was..."), which is
PURE un-diluted ex.co content. Use THAT trace as a second query vector for NODE retrieval —
HyDE with a real retrieved document instead of a hallucinated one. Does it surface the ex.co
NODES the diluted query buried at rank 36-59?

  baseline:        embed_query(#11)            -> cosine vs nodes -> best ex.co NODE rank
  trace-as-vector: embed_query(matched trace)  -> cosine vs nodes -> best ex.co NODE rank
  centroid:        mean of top-5 trace vectors -> cosine vs nodes -> best ex.co NODE rank

Never touches live. Usage: ./dev python3 eval/oracle_audit/trace_as_vector_probe.py
"""
import os, sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
KNOWN_EXCO = {'e62cc595','dabb3078','af92b2cb','30d88dd0','b3bda662','5fe121db',
              '8359cf1d','5410f4be','ef2f3276','41d31ca5','671d1f22','598d78a8'}

with IsolatedBrain() as env:
    brain = env.brain
    from servers import embedder
    lc = brain.logs_conn
    model = embedder.stats.get('model_name') or ''
    # node vectors
    nrows = brain._vec_dal.get_all_vectors(vector_types=['_primary'], model=model or None)
    nvecs = [(r['node_id'], r['embedding']) for r in nrows if r['embedding']]
    # trace vectors
    trows = lc.execute("SELECT trace_id, vector, text FROM trace_embeddings").fetchall()

    q11 = next(it['prompt'] for it in CORPUS if it['rank'] == 11)

    def best_exco_node(qvec):
        scored = sorted(((embedder.cosine_similarity(qvec, b), nid) for nid, b in nvecs), key=lambda x: -x[0])
        for i, (_, nid) in enumerate(scored[:300], 1):
            if nid[:8] in KNOWN_EXCO:
                return i, scored[:3]
        return None, scored[:3]

    def titles(top3):
        out = []
        for _, nid in top3:
            t = brain.conn.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
            out.append((nid[:8], (t[0] if t else '')[:34]))
        return out

    print("\n=== TRACE-AS-VECTOR probe (#11) ===")
    # baseline: query as vector
    qv = embedder.embed_query(q11)
    br, _ = best_exco_node(qv)
    print("  baseline  embed_query(#11) -> best ex.co NODE rank = %s" % br)

    # rank traces, take top-6
    tscored = sorted(((embedder.cosine_similarity(qv, v), tid, txt) for tid, v, txt in trows), key=lambda x: -x[0])[:6]
    print("\n  -- each matched trace used AS the node-query vector --")
    centroid_vecs = []
    for cos, tid, txt in tscored:
        tv = embedder.embed_query((txt or '')[:500])
        centroid_vecs.append(tv)
        r, top3 = best_exco_node(tv)
        echo = '(query-echo)' if 'what did we do on the last session' in (txt or '').lower() else ''
        print("   trace %s tcos=%.3f  best ex.co NODE rank=%s %s" % (tid, cos, r, echo))
        print("       text: %s" % (txt or '').replace('\n', ' ')[:70])
        print("       its top-3 nodes: %s" % titles(top3))

    # centroid of top-5 trace vectors
    vs = [embedder._blob_to_vec(b) for b in centroid_vecs[:5]]
    dim = len(vs[0])
    mean = [sum(v[i] for v in vs) / len(vs) for i in range(dim)]
    mblob = embedder._vec_to_blob(mean)
    cr, ctop = best_exco_node(mblob)
    print("\n  centroid(top-5 traces) -> best ex.co NODE rank = %s" % cr)
    print("       centroid top-3 nodes: %s" % titles(ctop))
