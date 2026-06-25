#!/usr/bin/env python3
"""What does spread_activation RETURN, given a seed set? Controlled 1-seed and
2-seed runs, dumping the literal output structure (not the flood). Read-only.
Usage: ./dev python3 eval/oracle_audit/spread_output_shape.py
"""
import os, sys
import numpy as np
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain          # noqa: E402
from servers import embedder                            # noqa: E402
from servers.scales.s1.surface import _graph_expand     # noqa: E402


def s(x):
    return (x or '')[:8]


with IsolatedBrain() as env:
    b = env.brain

    def title(nid):
        r = b.conn.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
        return r[0] if r else '?'

    # Get two real, valid seed ids via recall on two different topics.
    qv = np.frombuffer(embedder.embed_query("when did EX.CO first come up"), dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) or 1.0)
    out = b.recall(query="when did EX.CO first come up", limit=5)
    res = out.get('results', []) if isinstance(out, dict) else out
    ids = [(r.get('id') or r.get('node_id')) for r in res]
    seedA, seedB = ids[0], ids[1]

    for label, seeds in (("ONE SEED", [seedA]), ("TWO SEEDS", [seedA, seedB])):
        print("\n" + "=" * 88)
        print("%s:" % label)
        for sd in seeds:
            print("   seed %s  %s" % (s(sd), title(sd)[:60]))
        print("=" * 88)

        r = _graph_expand(b, seeds, query_vec=qv)

        print("\nRETURN KEYS: %s" % sorted(r.keys()))
        na = r.get('node_activation', {})
        fa = r.get('field_activation', {})
        tr = r.get('trace', [])
        rn = r.get('rich_nodes', {})
        cv = r.get('convergence', {})

        print("  node_activation : %d entries   (type: {node_id: float})" % len(na))
        print("  field_activation: %d entries   (type: {node_id: {field: float}})" % len(fa))
        print("  rich_nodes      : %d entries" % len(rn))
        print("  convergence     : %d entries" % len(cv))
        print("  trace           : %d hop-steps" % len(tr))

        print("\n  → is activation attributed PER-SEED? key sample: %r" %
              (list(na.keys())[:1]))
        print("    (a flat {node_id: float} map — seeds are MERGED, no per-seed provenance)")

        print("\n  seed's OWN entry in the output:")
        for sd in seeds:
            print("    %s act=%.3f  field_activation=%s" %
                  (s(sd), na.get(sd, float('nan')),
                   {k: round(v, 2) for k, v in sorted(fa.get(sd, {}).items(),
                                                       key=lambda kv: -kv[1])[:4]}))

        print("\n  trace (the wave this seed-set produced):")
        for st in tr:
            print("    hop %d: new=%-4s considered=%-5s transmitted=%-5s max_act=%.3f" %
                  (st.get('step', -1), st.get('new_nodes', '?'),
                   st.get('edges_considered', '?'), st.get('edges_transmitted', '?'),
                   st.get('max_act', 0.0)))
