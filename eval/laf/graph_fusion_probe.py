#!/usr/bin/env python3
"""FUSION test: how should the graph beam combine with the maxsim base? (§18.21, Tom 2026-06-30)

The seedless probe found the graph reaches 11% of far gold ALONE, but only 8% once SUMMED with
maxsim — cosine re-buries the low-cosine far nodes the graph just rescued. So the fusion METHOD,
not the operator, is what's leaking the reach. This fixes the verified beam (edge+node, τ=0.7,
K_max=2, 2 hops) and varies ONLY the fusion:

  sum   z(ms) + g·z(beam)            — current. additive/OR pool, but base magnitude can dominate.
  max   max(z(ms), g·z(beam))        — a node rides the STRONGER of its cosine or graph score.
  RRF   1/(k+rank_ms) + 1/(k+rank_beam)  — rank-union: a far node maxsim ranks 1120 but the beam
                                          ranks 3 gets the beam's rank credit, so cosine can't
                                          demote it. The "separate lane" fusion, done cleanly.

Full-universe (NOT seedless) need-collapsed hit@5/@25 + brought/lost vs maxsim — this is the
ranking Anchor actually sees. Reused beam/helpers from graph_beam_probe (one field build).

Run (daemon maintenance-locked): ./dev python3 eval/laf/graph_fusion_probe.py
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
    build_edge_conductance, edge_cos, created_at_array,
)
from laf_metrics import zscore, ranks, best_ranks, need_hit, need_bl  # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402
from graph_beam_probe import beam, build_neighbors, c_edge_plus_node, SEED_K, TAU  # noqa: E402

RRF_K = 60


def rank_among(scores, mask, master):
    """{id: rank} ranking ONLY the masked nodes by score desc (others absent)."""
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return {}
    order = idxs[np.argsort(-scores[idxs])]
    return {master[int(i)]: r + 1 for r, i in enumerate(order)}


def fuse_sum(ms, bm, elig, N, g):
    return zscore(ms, elig, N) + g * zscore(bm, elig, N)


def fuse_max(ms, bm, elig, N, g):
    return np.maximum(zscore(ms, elig, N), g * zscore(bm, elig, N))


def fuse_rrf(ms, bm, elig, N, idx, master):
    """Rank-union: every eligible node gets maxsim's RRF term; reached nodes ALSO get the
    beam's — so a far node the beam ranks high survives cosine's demotion."""
    ms_r = rank_among(ms, elig & np.isfinite(ms), master)
    bm_r = rank_among(bm, elig & (bm > 1e-12), master)
    sc = np.where(elig, 0.0, -np.inf)
    for nid, r in ms_r.items():
        sc[idx[nid]] += 1.0 / (RRF_K + r)
    for nid, r in bm_r.items():
        sc[idx[nid]] += 1.0 / (RRF_K + r)
    return sc


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        N = len(master)
        ca = created_at_array(brain, master)
        edges = build_edge_conductance(brain, idx)

        # per-cue precompute: ms (seed + node_cos), beam activation, eligibility, base need-rank
        per = {}
        base_ref = {}
        for c in cues:
            qv = query_vec(c["query"])
            if qv is None or not c["needs"]:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))
            ecos = edge_cos(edges, qv, cutoff=c["cutoff"])   # cutoff-masked (no future edges)
            nbr = build_neighbors(edges, ecos)
            top = np.argsort(-np.where(elig & np.isfinite(ms), ms, -np.inf))[:SEED_K]
            bm = beam([int(i) for i in top], nbr, ms, c_edge_plus_node, TAU, N)
            per[c["id"]] = {"elig": elig, "ms": ms, "bm": bm, "needs": c["needs"]}
            base_ref[c["id"]] = best_ranks(ranks(ms, elig, master), c["needs"])
        nc = len(per) or 1

        CONFIGS = [
            ("maxsim (base)",  lambda p: p["ms"]),
            ("sum g=0.3",      lambda p: fuse_sum(p["ms"], p["bm"], p["elig"], N, 0.3)),
            ("sum g=0.5",      lambda p: fuse_sum(p["ms"], p["bm"], p["elig"], N, 0.5)),
            ("max g=0.5",      lambda p: fuse_max(p["ms"], p["bm"], p["elig"], N, 0.5)),
            ("max g=1.0",      lambda p: fuse_max(p["ms"], p["bm"], p["elig"], N, 1.0)),
            ("RRF k=60",       lambda p: fuse_rrf(p["ms"], p["bm"], p["elig"], N, idx, master)),
        ]

        print("FUSION test — beam=edge+node τ=%.2f, %d cues, full universe\n" % (TAU, nc))
        print("  %-16s %-7s %-7s | %-8s %-6s" % ("fusion", "hit@5", "hit@25", "brought", "lost"))
        for name, fn in CONFIGS:
            h5 = h25 = 0; brought = lost = 0
            for c in cues:
                p = per.get(c["id"])
                if p is None:
                    continue
                sc = fn(p)
                h5 += need_hit(sc, p["elig"], master, p["needs"], 5)
                h25 += need_hit(sc, p["elig"], master, p["needs"], 25)
                if name != "maxsim (base)":
                    b, l = need_bl(sc, p["elig"], master, p["needs"], base_ref[c["id"]])
                    brought += b; lost += l
            extra = "—        —" if name == "maxsim (base)" else "+%-7d −%d" % (brought, lost)
            print("  %-16s %-7s %-7s | %s"
                  % (name, "%.0f%%" % (100*h5/nc), "%.0f%%" % (100*h25/nc), extra))


if __name__ == "__main__":
    main()
