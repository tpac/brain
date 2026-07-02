#!/usr/bin/env python3
"""EPISODIC layer measurement — the three roles, standalone + in-stack (Tom 2026-07-01).

The substrate (the unified nodes_for_traces join) is now on main's trace_links.py
(33 tests green). This measures the three episodic layers in the SAME frame as the graph
ablation (need-collapsed, full universe, 24-cue honest gold, brought/lost vs maxsim base):

    pick+   nodes Haiku surfaced AND selected at similar past moments      (+act)
    enc+    nodes created/revised at/after similar past moments            (+act)
    drop−   ÷prevalence: repeatedly offered, consistently NOT selected     (−inhibit)

Seeded from recall_episodes(cue, older_than=cutoff, scale='s0') — similar past MOMENTS,
±1-turn window (beat single-turn in the spike audit). Configs:

    maxsim (base)        z(ms)
    + graph              z(ms) + 0.5·z(beam)                    ← the verified stack so far
    pick standalone      z(pick) alone (what does the role see by itself?)
    enc  standalone      z(enc) alone
    + pick / + enc       marginal on maxsim
    + drop               maxsim − 0.3·z(drop)  (inhibition: does it clear space or kill gold?)
    + episodic (3)       maxsim + 0.5·pick + 0.3·enc − 0.3·drop
    full stack           maxsim + graph + all three

Run (daemon maintenance-locked): ./dev python3 eval/laf/episodic_layers_probe.py
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
from episodic_ops import (                                             # noqa: E402
    episodic_roles, episodic_encoded, episodic_picked, episodic_dropped,
)
from gold24_diagnostic import load_cues                              # noqa: E402
from graph_beam_probe import beam, build_neighbors, c_edge_plus_node, SEED_K, TAU  # noqa: E402

G_GRAPH = 0.5
EPI_WINDOW = ("window", 1)          # ±1-turn moment (spike audit: beats single-turn)


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

        per = {}
        base_ref = {}
        n_epi_empty = 0
        epi_cand = []                     # nonzero counts per cue (coverage stat)
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

            recs = episodic_roles(brain, c["query"], c["cutoff"], window=EPI_WINDOW)
            if not recs:
                n_epi_empty += 1
            pick = episodic_picked(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            enc = episodic_encoded(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            drop = episodic_dropped(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            epi_cand.append(int(np.sum((pick > 0) | (enc > 0) | (drop > 0))))

            per[c["id"]] = {
                "elig": elig, "needs": c["needs"],
                "zms": zscore(ms, elig, N), "zbm": zscore(bm, elig, N),
                "zpick": zscore(pick, elig, N), "zenc": zscore(enc, elig, N),
                "zdrop": zscore(drop, elig, N),
            }
            base_ref[c["id"]] = best_ranks(ranks(ms, elig, master), c["needs"])
        nc = len(per) or 1

        CONFIGS = [
            ("maxsim (base)",   lambda p: p["zms"]),
            ("+ graph",         lambda p: p["zms"] + G_GRAPH * p["zbm"]),
            ("pick standalone", lambda p: p["zpick"]),
            ("enc standalone",  lambda p: p["zenc"]),
            ("+ pick",          lambda p: p["zms"] + 0.5 * p["zpick"]),
            ("+ enc",           lambda p: p["zms"] + 0.3 * p["zenc"]),
            ("+ drop (inhib)",  lambda p: p["zms"] - 0.3 * p["zdrop"]),
            ("+ pick+enc",      lambda p: p["zms"] + 0.5 * p["zpick"] + 0.3 * p["zenc"]),
            ("+ episodic (3)",  lambda p: p["zms"] + 0.5 * p["zpick"] + 0.3 * p["zenc"] - 0.3 * p["zdrop"]),
            ("full − drop",     lambda p: p["zms"] + G_GRAPH * p["zbm"]
                                          + 0.5 * p["zpick"] + 0.3 * p["zenc"]),
            ("full stack",      lambda p: p["zms"] + G_GRAPH * p["zbm"]
                                          + 0.5 * p["zpick"] + 0.3 * p["zenc"] - 0.3 * p["zdrop"]),
        ]

        print("EPISODIC layers — ±1-turn moments, %d cues (%d with no episodes; "
              "median touched nodes/cue %d), full universe\n"
              % (nc, n_epi_empty, int(np.median(epi_cand)) if epi_cand else 0))
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


if __name__ == "__main__":
    main()
