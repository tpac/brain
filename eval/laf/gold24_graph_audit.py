#!/usr/bin/env python3
"""Standalone audit of the rebuilt graph operator (relational_reinstatement) vs the old blunt spread.

Gate: does weighting edges by MEANING (cos(cue, edge.why)) instead of the uncalibrated 0.5 stored
weight clear the 2% floor the old graph1hop hit? Reports edge-embedding coverage + each operator's
standalone need-collapsed gold hit@5/@25, against the seed (maxsim) it spreads from.

Run (daemon maintenance-locked): ./dev python3 eval/laf/gold24_graph_audit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field,
    build_adjacency, graph_spread, build_edge_conductance, relational_reinstatement,
)
from gold24_diagnostic import load_cues                              # noqa: E402
from gold24_field_audit import need_hit, ranks                       # noqa: E402


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        N = len(master)
        ca = np.array([dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall()).get(n, "")
                       for n in master])
        adj = build_adjacency(brain, idx)                       # old: stored-weight, undirected
        edges = build_edge_conductance(brain, idx)             # new: per-edge description vectors

        # edge-embedding coverage of the non-noise graph
        n_new = edges[0].size
        n_old = adj[0].size
        print("edges: noise-excluded pairs(old adj)=%d | with description-embedding(new)=%d (%.0f%% have a why-vector)"
              % (n_old, n_new, 100 * n_new / (n_old or 1)))

        rows = {k: [0, 0] for k in (
            "maxsim (seed)", "OLD graph1hop (blunt)",
            "relational (continuous seed,1hop)", "relational (sparse top25,1hop)",
            "relational (sparse top25,2hop)", "maxsim + relational(sparse,1hop)")}
        nc = 0
        cond_stats = []   # (mean, std, frac>0.6) of cue↔edge-why conductance, per cue

        def z(x, elig):
            m = elig & np.isfinite(x); o = np.zeros(N)
            if int(m.sum()) > 2 and np.std(x[m]) > 1e-9:
                o[m] = (x[m] - x[m].mean()) / x[m].std()
            return o

        for c in cues:
            qv = query_vec(c["query"])
            if qv is None:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))
            seed_cont = np.clip(ms, 0.0, None)
            # sparse seed: keep only top-25 maxsim nodes (concentrate the spread)
            seed_sparse = np.zeros(N)
            top = np.argsort(-np.where(elig & np.isfinite(ms), ms, -np.inf))[:25]
            seed_sparse[top] = np.clip(ms[top], 0.0, None)

            old = graph_spread((ms > np.nanpercentile(ms[np.isfinite(ms)], 90)).astype(np.float64), adj, hops=1)
            r_cont = relational_reinstatement(qv, seed_cont, edges, N, hops=1)
            r_sp1 = relational_reinstatement(qv, seed_sparse, edges, N, hops=1)
            r_sp2 = relational_reinstatement(qv, seed_sparse, edges, N, hops=2)
            combo = z(ms, elig) + 0.5 * z(r_sp1, elig)

            # conductance distribution this cue (is edge-why itself flat?)
            cond = np.clip(edges[2] @ qv, 0.0, None) if edges[2].shape[0] else np.zeros(1)
            cond_stats.append((float(cond.mean()), float(cond.std()), float(np.mean(cond > 0.6))))

            nc += 1
            for name, vec in (("maxsim (seed)", ms), ("OLD graph1hop (blunt)", old),
                              ("relational (continuous seed,1hop)", r_cont),
                              ("relational (sparse top25,1hop)", r_sp1),
                              ("relational (sparse top25,2hop)", r_sp2),
                              ("maxsim + relational(sparse,1hop)", combo)):
                rk = ranks(vec, elig, master)
                m5 = need_hit(rk, c, 5); m25 = need_hit(rk, c, 25)
                if m5 is not None:
                    rows[name][0] += m5; rows[name][1] += m25

        print("\n  %-36s %-8s %-8s" % ("operator (standalone, need-collapsed)", "hit@5", "hit@25"))
        for name, (h5, h25) in rows.items():
            print("  %-36s %-8s %-8s" % (name, "%.0f%%" % (100 * h5 / nc), "%.0f%%" % (100 * h25 / nc)))
        cm = np.mean([s[0] for s in cond_stats]); csd = np.mean([s[1] for s in cond_stats])
        chi = np.mean([s[2] for s in cond_stats])
        print("\n  edge-why conductance cos(cue, edge.why): mean=%.3f std=%.3f  frac>0.6=%.1f%%" %
              (cm, csd, 100 * chi))
        print("  (if mean~0.5 / std small / frac>0.6 ~0 → the edge-why embedding is ALSO flat → weak steering,")
        print("   i.e. the operator is correct but the flat embedder caps it, same wall as node cosine.)")


if __name__ == "__main__":
    main()
