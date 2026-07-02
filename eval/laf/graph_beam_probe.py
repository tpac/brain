#!/usr/bin/env python3
"""Graph BEAM operator — Tom's ranked-walk shape, swept on the 24-cue lens-independent gold.

The rebuilt graph operator (`relational_reinstatement`) is a diffusion SUM: it pushes
source_activation × edge_conductance to every neighbour and adds it up (÷degree hub-damp).
Tom's shape (2026-06-30) is different — a ranked BEAM walk:

    seed   = the max-cosine nodes (top-K maxsim)
    hop    = rank a node's edges by cos(cue, edge.why); follow those within a RATIO of the
             best edge (adaptive fan-out, not a fixed top-5), capped at K_max; 2 hops.
    score  = each reached node's activation = COMBINE(edge_cos, node_cos)

Two things this tests that the diffusion can't:
  • adaptive fan-out — a node with one dominant edge follows ~1; a node with several
    comparably-strong edges follows several; a node with only weak edges follows ZERO
    (cost-natural: quiet nodes stay quiet). No arbitrary `5`.
  • the COMBINE — how the reached node is scored decides reach-vs-rerank:
      edge·node   suppresses the low-node-cos far nodes → KILLS reach (thesis: loses @25)
      edge+node   a strong edge carries a far node; node is a sanity nudge (thesis: best @25)
      edge-only   pure path score
      edge>node   gate — admit only when the path beats the node's own cosine (pure reach)
      ratio       edge/(node+c) — rewards "graph rescued it" (strong edge, weak node)

REUSE (Tom): maxsim is computed once and is BOTH the seed ranking AND the node_cos term;
edge `why` vectors are pre-stored, so edge cosine is one matmul. No re-embedding.

Run (daemon maintenance-locked): ./dev python3 eval/laf/graph_beam_probe.py
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
from laf_metrics import zscore, ranks, best_ranks                     # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402

SEED_K = 25          # "the max-cosine nodes" = top-K maxsim
K_MAX = 2            # Tom: cap the fan-out at 2 per hop
HOPS = 2
FLOOR = 0.3          # absolute edge-cos floor: below this, an edge is never followed
TAU = 0.7            # relative fan-out threshold (proven inert at K_MAX=2; kept as the shared constant)


# ── combine functions: (edge_cos, node_cos) → reached-node activation ──
def c_edge_plus_node(e, n):  return 0.7 * e + 0.3 * n      # headline thesis
def c_edge_times_node(e, n): return e * n                  # thesis: kills reach
def c_edge_only(e, n):       return e
def c_edge_gate(e, n):       return e if e > n else 0.0    # admit only if path beats node
def c_ratio(e, n):           return e / (n + 0.5)          # bounded "graph rescued it"

COMBINES = {
    "edge+node": c_edge_plus_node, "edge*node": c_edge_times_node,
    "edge-only": c_edge_only, "edge>node": c_edge_gate, "ratio": c_ratio,
}


def build_neighbors(edges, ecos):
    """Undirected neighbour index: node_row → [(neighbour_row, edge_cos), ...].
    Rebuilt per cue because `ecos` is per-cue (cutoff-masked cue↔edge cosine)."""
    nbr = defaultdict(list)
    for e in range(edges.src.shape[0]):
        s, t, c = int(edges.src[e]), int(edges.dst[e]), float(ecos[e])
        nbr[s].append((t, c))
        nbr[t].append((s, c))
    return nbr


def beam(seeds, nbr, node_cos, combine, tau, n):
    """One ranked-beam walk → [n] activation. Reached-node score = max over paths of
    combine(reaching_edge_cos, node_cos)."""
    act = np.zeros(n, dtype=np.float64)
    frontier = list(seeds)
    seen = set(seeds)
    for _ in range(HOPS):
        nxt = []
        for i in frontier:
            edges = sorted(nbr.get(i, []), key=lambda x: -x[1])
            if not edges:
                continue
            best = edges[0][1]
            thr = max(FLOOR, tau * best)
            followed = [(j, c) for j, c in edges if c >= thr][:K_MAX]
            for j, c in followed:
                s = combine(c, float(node_cos[j]))
                if s > act[j]:
                    act[j] = s
                if j not in seen:
                    seen.add(j); nxt.append(j)
        frontier = nxt
    return act


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
        print("beam probe — %d cues, %d nodes, %d typed edges w/ why-vec | seed_k=%d K_max=%d hops=%d floor=%.2f"
              % (len(cues), N, edges.src.shape[0], SEED_K, K_MAX, HOPS, FLOOR))

        # per-cue precompute (reused across all configs): qv, maxsim, edge_cos, neighbours
        pc = {}
        for c in cues:
            qv = query_vec(c["query"])
            if qv is None or not c["needs"]:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))   # seed ranking AND node_cos
            ecos = edge_cos(edges, qv, cutoff=c["cutoff"])     # cutoff-masked (no future edges)
            top = np.argsort(-np.where(elig & np.isfinite(ms), ms, -np.inf))[:SEED_K]
            pc[c["id"]] = {"elig": elig, "ms": ms, "seeds": [int(i) for i in top],
                           "nbr": build_neighbors(edges, ecos)}

        # ── SEEDS REMOVED: strip the maxsim top-SEED_K base from the ranking universe,
        #    so we measure ONLY reach BEYOND the base — does the graph surface gold cosine
        #    didn't already have? (Tom, 2026-06-30.) A need whose gold is ALL in the seed
        #    set becomes unhittable by construction — correct: seed-covered ≠ graph's job.
        nc = sum(1 for c in cues if pc.get(c["id"])) or 1
        seed_mask = {}
        n_addr = 0        # needs with ≥1 gold OUTSIDE the seed set (the addressable universe)
        n_need = 0
        for c in cues:
            p = pc.get(c["id"])
            if p is None:
                continue
            m = np.zeros(N, dtype=bool); m[p["seeds"]] = True
            seed_mask[c["id"]] = m
            for nids in c["needs"].values():
                n_need += 1
                if any((idx.get(g) is not None and not m[idx[g]]) for g in nids):
                    n_addr += 1

        def h(scores, universe, needs, k):
            rk = ranks(scores, universe, master)
            br = best_ranks(rk, needs)
            return sum(1 for r in br.values() if r and r <= k) / (len(br) or 1)

        print("\n  SEEDS REMOVED — reach BEYOND the maxsim top-%d base (%d/%d needs addressable, "
              "rest are seed-covered):" % (SEED_K, n_addr, n_need))
        print("  %-16s %-9s %-9s | %-11s %-11s"
              % ("config", "beam@5", "beam@25", "ms+beam@5", "ms+beam@25"))

        # reference: maxsim's OWN ranking of the non-seed nodes (does plain cosine rank far gold?)
        r5 = r25 = 0
        for c in cues:
            p = pc.get(c["id"])
            if p is None:
                continue
            sl = p["elig"] & ~seed_mask[c["id"]]
            r5 += h(p["ms"], sl, c["needs"], 5); r25 += h(p["ms"], sl, c["needs"], 25)
        print("  %-16s %-9s %-9s | %-11s %-11s"
              % ("maxsim-seedless", "—", "—", "%.0f%%" % (100*r5/nc), "%.0f%%" % (100*r25/nc)))

        # sweep combine (τ dropped — proven inert at K_max=2, cap dominates the ratio)
        for cname, cfn in COMBINES.items():
            tau = 0.7
            b5 = b25 = m5 = m25 = 0
            for c in cues:
                p = pc.get(c["id"])
                if p is None:
                    continue
                sl = p["elig"] & ~seed_mask[c["id"]]
                bm = beam(p["seeds"], p["nbr"], p["ms"], cfn, tau, N)
                b5 += h(bm, sl, c["needs"], 5); b25 += h(bm, sl, c["needs"], 25)
                vec = zscore(p["ms"], p["elig"], N) + 0.5 * zscore(bm, p["elig"], N)
                m5 += h(vec, sl, c["needs"], 5); m25 += h(vec, sl, c["needs"], 25)
            print("  %-16s %-9s %-9s | %-11s %-11s"
                  % (cname, "%.0f%%" % (100*b5/nc), "%.0f%%" % (100*b25/nc),
                     "%.0f%%" % (100*m5/nc), "%.0f%%" % (100*m25/nc)))


if __name__ == "__main__":
    main()
