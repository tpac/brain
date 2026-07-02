#!/usr/bin/env python3
"""LAYER CONTRIBUTION ablation — maxsim / graph / temporal, with the REBUILT graph (Tom 2026-06-30).

The old ablation (gold24_diagnostic --ablate) ran the INERT graph_spread + degenerate temporal in
the settling engine — graph contributed exactly 0 (full == −graph). This redoes the same
maxsim/graph/temporal variation with the VERIFIED beam graph (edge+node, τ=0.7, K_max=2, 2 hops,
additive sum g=0.5) so we see what each layer actually contributes now:

    maxsim              z(ms)
    + graph             z(ms) + 0.5·z(beam)
    + temporal          z(ms) + 0.3·z(temporal)         # von-Restorff distinctiveness (query-indep)
    + both              z(ms) + 0.5·z(beam) + 0.3·z(temporal)

Full-universe need-collapsed hit@5/@25 + brought/lost @25 vs maxsim base — the marginal
contribution of each layer on the ranking Anchor actually sees.

Run (daemon maintenance-locked): ./dev python3 eval/laf/graph_layers_probe.py
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
    temporal_distinctiveness, parse_days,
)
from laf_metrics import zscore, ranks, best_ranks, need_hit, need_bl  # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402
from graph_beam_probe import beam, build_neighbors, c_edge_plus_node, SEED_K, TAU  # noqa: E402

G_GRAPH = 0.5
G_TEMP = 0.3
TEMP_WINDOW = 7.0


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        N = len(master)
        ca = created_at_array(brain, master)
        days = parse_days(list(ca))
        edges = build_edge_conductance(brain, idx)

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
            tv = temporal_distinctiveness(days, elig, TEMP_WINDOW)
            zms, zbm, ztv = zscore(ms, elig, N), zscore(bm, elig, N), zscore(tv, elig, N)
            per[c["id"]] = {"elig": elig, "zms": zms, "zbm": zbm, "ztv": ztv, "needs": c["needs"]}
            base_ref[c["id"]] = best_ranks(ranks(ms, elig, master), c["needs"])
        nc = len(per) or 1

        CONFIGS = [
            ("maxsim (base)",      lambda p: p["zms"]),
            ("+ graph",            lambda p: p["zms"] + G_GRAPH * p["zbm"]),
            ("+ temporal",         lambda p: p["zms"] + G_TEMP * p["ztv"]),
            ("+ both",             lambda p: p["zms"] + G_GRAPH * p["zbm"] + G_TEMP * p["ztv"]),
        ]

        print("LAYER CONTRIBUTION — rebuilt beam graph (g=%.1f) + temporal (g=%.1f), %d cues, full universe\n"
              % (G_GRAPH, G_TEMP, nc))
        print("  %-16s %-7s %-7s | %-8s %-6s" % ("config", "hit@5", "hit@25", "brought", "lost"))
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

        # temporal liveness check on this corpus (the degeneracy question)
        p0 = next(iter(per.values()))
        tv_live = "varies" if float(np.std(p0["ztv"][p0["elig"]])) > 1e-6 else "CONSTANT/degenerate"
        print("\n  temporal field on this corpus: %s" % tv_live)


if __name__ == "__main__":
    main()
