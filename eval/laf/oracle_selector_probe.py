#!/usr/bin/env python3
"""PER-CUE ORACLE SELECTOR — the size of the prize competition is playing for (Tom 2026-07-01).

The static z-sum composes reach (@25) but degrades precision (@5): graph+episodic overlap
destructively (16→10@5). The union measurement (53%@5 across cues vs 19% single) says the
prize is PER-CUE selection. This measures it directly: if a perfect selector picked the best
ARM per cue, what @5/@25 falls out — vs the best static config?

Arms = the realistic selector menu (each a full fused ranking, not a raw signal):
    base            z(ms)
    +graph          z(ms) + 0.5·z(beam)
    +pick+enc       z(ms) + 0.5·z(pick) + 0.3·z(enc)
    +all            z(ms) + 0.5·z(beam) + 0.5·z(pick) + 0.3·z(enc)

Oracle = per cue, the arm maximizing need-hit@5 (tie → higher @25, then base-first order —
a stability bias toward the simpler arm). Also reported:
  • the routing distribution (how often each arm wins, how often it wins UNIQUELY by a real
    margin) — what a selector must learn;
  • honesty caveat: a 4-arm per-cue max on N=23 inflates by chance; the UNIQUE-win count is
    the part a selector can actually capture.

Run (daemon maintenance-locked): ./dev python3 eval/laf/oracle_selector_probe.py
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field,
    build_edge_conductance, edge_cos, created_at_array,
)
from laf_metrics import zscore, ranks, best_ranks, need_hit           # noqa: E402
from episodic_ops import episodic_roles, episodic_encoded, episodic_picked  # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402
from graph_beam_probe import beam, build_neighbors, c_edge_plus_node, SEED_K, TAU  # noqa: E402

EPI_WINDOW = ("window", 1)

ARMS = [
    ("base",       lambda p: p["zms"]),
    ("+graph",     lambda p: p["zms"] + 0.5 * p["zbm"]),
    ("+pick+enc",  lambda p: p["zms"] + 0.5 * p["zpick"] + 0.3 * p["zenc"]),
    ("+all",       lambda p: p["zms"] + 0.5 * p["zbm"] + 0.5 * p["zpick"] + 0.3 * p["zenc"]),
]


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
        for c in cues:
            qv = query_vec(c["query"])
            if qv is None or not c["needs"]:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))
            ecos = edge_cos(edges, qv, cutoff=c["cutoff"])
            nbr = build_neighbors(edges, ecos)
            top = np.argsort(-np.where(elig & np.isfinite(ms), ms, -np.inf))[:SEED_K]
            bm = beam([int(i) for i in top], nbr, ms, c_edge_plus_node, TAU, N)
            recs = episodic_roles(brain, c["query"], c["cutoff"], window=EPI_WINDOW)
            pick = episodic_picked(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            enc = episodic_encoded(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            per[c["id"]] = {
                "elig": elig, "needs": c["needs"],
                "zms": zscore(ms, elig, N), "zbm": zscore(bm, elig, N),
                "zpick": zscore(pick, elig, N), "zenc": zscore(enc, elig, N),
            }
        nc = len(per) or 1

        # per-cue per-arm scores
        cue_arm = {}          # cue_id -> [(arm, h5, h25), ...]
        for cid, p in per.items():
            rows = []
            for name, fn in ARMS:
                sc = fn(p)
                rows.append((name,
                             need_hit(sc, p["elig"], master, p["needs"], 5),
                             need_hit(sc, p["elig"], master, p["needs"], 25)))
            cue_arm[cid] = rows

        # static arms (the no-selector baselines)
        print("PER-CUE ORACLE SELECTOR — %d cues, 4 arms\n" % nc)
        print("  %-22s %-7s %-7s" % ("config", "hit@5", "hit@25"))
        for i, (name, _fn) in enumerate(ARMS):
            h5 = sum(cue_arm[cid][i][1] for cid in cue_arm) / nc
            h25 = sum(cue_arm[cid][i][2] for cid in cue_arm) / nc
            print("  %-22s %-7s %-7s" % ("static " + name,
                                         "%.0f%%" % (100 * h5), "%.0f%%" % (100 * h25)))

        # oracle: per cue argmax h5 (tie → h25, then earlier arm = simpler)
        o5 = o25 = 0.0
        wins = Counter()
        unique_wins = Counter()      # sole argmax with a REAL margin (> 0, not tie-broken)
        for cid, rows in cue_arm.items():
            best_i = max(range(len(rows)), key=lambda i: (rows[i][1], rows[i][2], -i))
            name, h5, h25 = rows[best_i]
            o5 += h5; o25 += h25
            wins[name] += 1
            top5 = max(r[1] for r in rows)
            if sum(1 for r in rows if r[1] == top5) == 1 and \
               top5 > sorted((r[1] for r in rows), reverse=True)[1]:
                unique_wins[name] += 1
        print("  %-22s %-7s %-7s   ← the selection prize"
              % ("ORACLE (per-cue best)", "%.0f%%" % (100 * o5 / nc), "%.0f%%" % (100 * o25 / nc)))

        print("\n  routing distribution (which arm the oracle picks):")
        for name, _ in ARMS:
            print("    %-12s picked %2d/%d   unique-win-by-margin %d"
                  % (name, wins.get(name, 0), nc, unique_wins.get(name, 0)))
        n_uniq = sum(unique_wins.values())
        print("\n  honesty: %d/%d cues have a UNIQUE best arm by real margin — that slice is the"
              % (n_uniq, nc))
        print("  capturable prize; the rest of the oracle lift is tie-breaking luck at N=%d." % nc)


if __name__ == "__main__":
    main()
