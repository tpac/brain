#!/usr/bin/env python3
"""SCORE DECOMPOSITION — verify against the REAL pipeline, not a reimplementation.

Tom's worry: am I imagining the "raw cosine is fine, the boost buries it" story?
This localizes the #11 burial to a NAMED pipeline component using the REAL recall's own
returned numbers (IsolatedBrain runs the actual brain_recall.py on a DB copy):

  raw_cos   = embed_query(#11) . node._primary           (transparent computation, mine)
  z_emb     = result['embedding_similarity']             (pipeline's z-weighted multi-vector score)
  final     = result['effective_activation']             (pipeline's final blended score: +title-boost etc.)

For ex.co nodes vs the top session nodes, show the RANK under each. Whichever column ex.co
falls through is the culprit. If raw_cos ALSO buries ex.co -> my "rank 3" claim was imagined.

PASS (defined before running): burial localizes to ONE component that raw_cos doesn't share.
Never touches live. Usage: ./dev python3 eval/oracle_audit/score_decomp_probe.py
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
    q11 = next(it['prompt'] for it in CORPUS if it['rank'] == 11)

    # raw _primary cosine over ALL nodes (transparent) -> rank for every node
    model = embedder.stats.get('model_name') or ''
    nrows = brain._vec_dal.get_all_vectors(vector_types=['_primary'], model=model or None)
    qv = embedder.embed_query(q11)
    raw = sorted(((embedder.cosine_similarity(qv, r['embedding']), r['node_id']) for r in nrows if r['embedding']),
                 key=lambda x: -x[0])
    raw_rank = {nid[:8]: i for i, (_, nid) in enumerate(raw, 1)}
    raw_cos = {nid[:8]: c for c, nid in raw}

    # REAL pipeline recall (the actual brain_recall.py code, on the copy)
    if hasattr(brain, '_recall_cache'):
        try: brain._recall_cache.clear()
        except Exception: pass
    out = brain.recall(query=q11, limit=100)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    # the pipeline's own returned scores
    cand = []
    for i, r in enumerate(res, 1):
        nid = (r.get('id') or r.get('node_id') or '')[:8]
        cand.append((nid, r.get('embedding_similarity'), r.get('effective_activation'), i))
    z_rank = {nid: i for i, (nid, _, _, _) in enumerate(sorted(cand, key=lambda x: -(x[1] or 0)), 1)}
    final_rank = {nid: r for nid, _, _, r in cand}   # recall returns in final-score order
    zval = {nid: z for nid, z, _, _ in cand}
    fval = {nid: f for nid, _, f, _ in cand}

    def title(nid):
        t = brain.conn.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
        return (t[0] if t else '')[:34]

    def row(nid):
        return "  %s %s  raw_cos=%.3f r%-4s | z_emb=%-6s r%-4s | final=%-6s r%-4s  %s" % (
            nid, '[EXCO]' if nid in KNOWN_EXCO else '      ',
            raw_cos.get(nid, 0), raw_rank.get(nid, '?'),
            ('%.3f' % zval[nid]) if nid in zval and zval[nid] is not None else '--',
            z_rank.get(nid, '>100'),
            ('%.3f' % fval[nid]) if nid in fval and fval[nid] is not None else '--',
            final_rank.get(nid, '>100'),
            title(nid))

    print("\n=== SCORE DECOMPOSITION (#11) — real pipeline numbers ===")
    print("  columns: raw_cos (mine) | z_emb (pipeline z-weighted) | final (pipeline blended)")
    print("\n  -- EX.CO nodes (by raw-cosine rank) --")
    exco_sorted = sorted([n for n in KNOWN_EXCO], key=lambda n: raw_rank.get(n, 99999))
    for nid in exco_sorted[:8]:
        print(row(nid))
    print("\n  -- top-8 by FINAL score (what actually surfaces) --")
    for nid, _, _, _ in sorted(cand, key=lambda x: x[3])[:8]:
        print(row(nid))

    # verdict: where does the best ex.co node fall through?
    be = exco_sorted[0]
    print("\n  BEST EX.CO node %s: raw_cos rank=%s -> z_emb rank=%s -> final rank=%s"
          % (be, raw_rank.get(be), z_rank.get(be, '>100'), final_rank.get(be, '>100')))
    print("  (which arrow drops it = the culprit component; if raw_cos rank is already bad, the 'raw cosine is fine' claim was imagined)")
