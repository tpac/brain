#!/usr/bin/env python3
"""PER-VECTOR DECOMPOSITION for EP5 ("last session on ex.co"). The mechanism question:
is a buried node buried because (a) ALL its vectors have low cosine to the query
(episodic-vs-semantic mismatch — an embedding problem no avg/max formula fixes), or
(b) it has ONE high-cosine vector that the top-2-avg drags down (a formula problem)?

For each target node, dumps every vector_type's cosine, its group weight, the weighted
score, then avg2 vs max final, and whether either beats the raw _primary (the STEP 3.5 gate).
Also shows where the node lands in the real recall. Daemon-safe (IsolatedBrain)."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402
from servers import embedder                      # noqa: E402
from servers.pipeline_contract import get_group_weight  # noqa: E402

QUERY = "what did we do on the last session we worked on ex.co?"
TARGETS = {
    'b3b6ce2a': 'gold/surfaced (r10)',
    'dabb3078': 'gold/buried (r92)',
    '8359cf1d': 'handoff flagship (absent)',
    'b8b8370b': 'helper: episodic-trace recall',
}


def cos(qv, blob):
    return embedder.cosine_similarity(qv, blob) if blob else None


with IsolatedBrain() as env:
    b = env.brain
    qv = embedder.embed_query(QUERY)
    model = embedder.stats.get('model_name') or None

    # raw _primary rank over ALL nodes (the transparent baseline)
    prim = sorted(((cos(qv, r['embedding']), r['node_id'][:8])
                   for r in b._vec_dal.get_all_vectors(vector_types=['_primary'], model=model)
                   if r['embedding']), key=lambda x: -(x[0] or 0))
    prim_rank = {nid: i for i, (_, nid) in enumerate(prim, 1)}
    prim_cos = {nid: c for c, nid in prim}
    print("TOTAL nodes with _primary: %d" % len(prim))
    print("top-5 raw _primary cosine: %s"
          % ", ".join("%s=%.3f" % (nid, c) for c, nid in prim[:5]))

    # all vectors for the target nodes
    vecs = {}   # nid8 -> {vtype: blob}
    for r in b._vec_dal.get_all_vectors(model=model):
        nid8 = r['node_id'][:8]
        if nid8 in TARGETS and r['embedding']:
            vecs.setdefault(nid8, {})[r['vector_type']] = r['embedding']

    for nid, label in TARGETS.items():
        print("\n=== %s  (%s) ===" % (nid, label))
        print("  raw _primary cosine = %.3f  (rank %s of %d)"
              % (prim_cos.get(nid, 0), prim_rank.get(nid, '?'), len(prim)))
        vt = vecs.get(nid, {})
        scored = []
        for vtype, blob in vt.items():
            c = cos(qv, blob)
            w = get_group_weight(vtype)
            scored.append((w * (c or 0), c, w, vtype))
        scored.sort(reverse=True)
        print("  per-vector (cosine × weight = weighted), sorted by weighted:")
        for wscore, c, w, vtype in scored:
            print("    %-16s cos=%.3f  w=%.2f  weighted=%.3f" % (vtype, c or 0, w, wscore))
        if scored:
            top1 = scored[0][0]
            top2 = scored[1][0] if len(scored) > 1 else scored[0][0]
            avg2 = (top1 + top2) / 2 if len(scored) > 1 else top1
            mx = top1
            rawp = prim_cos.get(nid, 0)
            print("  → avg2 final = %.3f   max final = %.3f   (raw _primary = %.3f)"
                  % (avg2, mx, rawp))
            print("  → gate (final > raw_primary, so enrichment overwrites):  avg2=%s  max=%s"
                  % (avg2 > rawp, mx > rawp))
            print("  → best single vector: %s (cos %.3f). Is the node even NEAR the query? %s"
                  % (scored[0][3], scored[0][1] or 0,
                     "YES" if (scored[0][1] or 0) >= 0.55 else "NO — low cosine across all vectors"))
