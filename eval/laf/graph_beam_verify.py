#!/usr/bin/env python3
"""CHECK-YOURSELF for the graph beam: is the 11%@25 reach REAL, or a scoring fluke?

The seedless probe said the beam reaches ~11% of far gold vs cosine's 3% floor — but that's
~9 needs on N=24, and this thread has been burned by artifact numbers before. So don't trust
the aggregate: trace the concrete rescues. For the winning config (edge+node, τ=0.7, K_max=2,
2 hops), for every gold node the beam REACHES that maxsim buried (own maxsim rank > 25), print:

    the cue · the rescued gold (+ its maxsim rank) · the SEED it came from · the connecting
    edge's relation + why-text + cos(cue, edge.why)

If the paths are coherent (the edge `why` really is the need, the seed really is cue-relevant),
the reach is real. If they're random, it's noise and the aggregate was over-read.

Run (daemon maintenance-locked): ./dev python3 eval/laf/graph_beam_verify.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field,
    build_edge_conductance, edge_cos, created_at_array,
)
from gold24_diagnostic import load_cues                              # noqa: E402

SEED_K, K_MAX, HOPS, FLOOR, TAU = 25, 2, 2, 0.3, 0.7


def snip(s, n=90):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        N = len(master)
        ca = created_at_array(brain, master)
        title = dict(brain.conn.execute("SELECT id, title FROM nodes").fetchall())
        edges = build_edge_conductance(brain, idx)
        src, dst, emat, rels = edges.src, edges.dst, edges.emat, edges.rels

        # edge lookup: (row_a, row_b) -> list of (edge_index) — for path display
        pair_edges = defaultdict(list)
        for e in range(src.shape[0]):
            pair_edges[(int(src[e]), int(dst[e]))].append(e)
            pair_edges[(int(dst[e]), int(src[e]))].append(e)

        # neighbour index with edge index (so we can recover relation + why per hop)
        nbr = defaultdict(list)
        for e in range(src.shape[0]):
            nbr[int(src[e])].append((int(dst[e]), e))
            nbr[int(dst[e])].append((int(src[e]), e))

        def edge_desc(e):
            """(relation, description-text) for edge index e — fetched for display only."""
            a, b = master[int(src[e])], master[int(dst[e])]
            row = brain.conn.execute(
                "SELECT er.relation, er.description FROM edge_relations er JOIN edges ed "
                "ON ed.edge_id = er.edge_id WHERE ((ed.source_id=? AND ed.target_id=?) OR "
                "(ed.source_id=? AND ed.target_id=?)) AND er.relation=? LIMIT 1",
                (a, b, b, a, rels[e])).fetchone()
            return (rels[e], row[1] if row else "")

        total_rescued = 0
        cues_with_rescue = 0
        for c in cues:
            qv = query_vec(c["query"])
            if qv is None or not c["needs"]:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))
            ecos = edge_cos(edges, qv, cutoff=c["cutoff"])   # cutoff-masked (no future edges)
            # maxsim full-universe rank map (to identify what maxsim BURIED)
            ms_rank = {master[i]: r + 1 for r, i in
                       enumerate(np.argsort(-np.where(elig & np.isfinite(ms), ms, -np.inf)))}
            seeds = [int(i) for i in np.argsort(-np.where(elig & np.isfinite(ms), ms, -np.inf))[:SEED_K]]
            seed_set = set(seeds)

            # run the beam, remembering how each node was first reached (seed, edge, hop)
            reached_via = {}    # node_row -> (from_row, edge_index, hop)
            frontier = list(seeds)
            seen = set(seeds)
            for hop in range(1, HOPS + 1):
                nxt = []
                for i in frontier:
                    es = sorted(nbr.get(i, []), key=lambda x: -ecos[x[1]])
                    if not es:
                        continue
                    best = ecos[es[0][1]]
                    thr = max(FLOOR, TAU * best)
                    for j, e in [(j, e) for j, e in es if ecos[e] >= thr][:K_MAX]:
                        if j not in reached_via and j not in seed_set:
                            reached_via[j] = (i, e, hop)
                        if j not in seen:
                            seen.add(j); nxt.append(j)
                frontier = nxt

            # rescued gold = a need's gold node that the beam REACHED and maxsim buried (>25)
            rescues = []
            for need, nids in c["needs"].items():
                for g in nids:
                    gi = idx.get(g)
                    if gi is None or gi not in reached_via:
                        continue
                    if (ms_rank.get(g) or 10**9) > 25:      # maxsim buried it
                        frm, e, hop = reached_via[gi]
                        rel, desc = edge_desc(e)
                        rescues.append((need, g, ms_rank.get(g), master[frm], rel, desc, float(ecos[e]), hop))
            if not rescues:
                continue
            cues_with_rescue += 1
            total_rescued += len(rescues)
            print("\n▶ cue [%s]: %s" % (c["id"], snip(c["query"], 120)))
            for need, g, gr, frm, rel, desc, ec, hop in rescues:
                print("   RESCUED %s  \"%s\"  (maxsim rank %s, %d-hop)"
                      % (g[:8], snip(title.get(g, "?"), 70), gr, hop))
                print("     ← seed %s \"%s\"" % (frm[:8], snip(title.get(frm, "?"), 60)))
                print("       --[%s  cos=%.2f]→ why: %s" % (rel, ec, snip(desc, 100)))

        print("\n" + "=" * 70)
        print("TOTAL: %d far-gold nodes rescued by the beam across %d/24 cues."
              % (total_rescued, cues_with_rescue))
        print("(far = maxsim buried it past rank 25; rescued = the beam reached it via a typed edge)")


if __name__ == "__main__":
    main()
