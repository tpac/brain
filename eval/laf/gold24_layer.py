#!/usr/bin/env python3
"""The LAYERING test — one summed activation field, do the verified operators beat MaxSim? (§18.21)

All three fields now pass the standalone gate: MaxSim (base), relational_reinstatement (graph,
sparse-2hop seed, cos(cue,edge.why) conductance), and the episodic three-way (picked+/encoded+/
dropped−, ±1-turn moment). This composes them as ONE field (Tom's model — NOT find/rank):

    base = Σ_k gain_k · zscore(op_k)         # +maxsim +graph +picked +encoded  − dropped

and measures not just hit-rate but the two ways an overlapping operator earns its place:
  • REACH BROUGHT  — needs met@25 by the stack that MaxSim-alone missed
  • REINFORCEMENT  — needs both reach, but the stack ranks higher (raised)
  • REGRESSIONS    — needs MaxSim met that the stack lost (the cost of layering)

Run (daemon maintenance-locked): ./dev python3 eval/laf/gold24_layer.py
Out: eval/laf/gold24_layer.md  (+ console)
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
    build_edge_conductance, relational_reinstatement, created_at_array,
)
from episodic_ops import (                                            # noqa: E402
    episodic_roles, episodic_encoded, episodic_picked, episodic_dropped,
)
from gold24_diagnostic import load_cues                              # noqa: E402
from gold24_field_audit import ranks                                  # noqa: E402

OUT_MD = os.path.join(os.path.dirname(__file__), "gold24_layer.md")
EPI_WINDOW = ("window", 1)        # ±1-turn moment (beat single-turn in the audit)

# config → gains {op: gain}; maxsim is the base, dropped is subtracted
CONFIGS = {
    "maxsim (base)":        {"ms": 1.0},
    "+graph":               {"ms": 1.0, "graph": 0.5},
    "+episodic":            {"ms": 1.0, "pick": 0.5, "enc": 0.3, "drop": -0.3},
    "+both (full)":         {"ms": 1.0, "graph": 0.5, "pick": 0.5, "enc": 0.3, "drop": -0.3},
    "+both (lighter aux)":  {"ms": 1.0, "graph": 0.3, "pick": 0.3, "enc": 0.2, "drop": -0.2},
}


def z(x, elig, n):
    m = elig & np.isfinite(x)
    o = np.zeros(n)
    if int(m.sum()) > 2 and np.std(x[m]) > 1e-9:
        o[m] = (x[m] - x[m].mean()) / x[m].std()
    return o


def best_ranks(rank_map, cue):
    """{need: best (min) rank over its gold nodes}; None if none ranked."""
    out = {}
    for need, nids in cue["needs"].items():
        rs = [rank_map.get(nm) for nm in nids if rank_map.get(nm) is not None]
        out[need] = min(rs) if rs else None
    return out


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

        # per-cue operator vectors (z-scored), computed once
        per_cue = {}
        n_epi_empty = 0
        for c in cues:
            qv = query_vec(c["query"])
            if qv is None:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))
            seed = np.zeros(N)
            top = np.argsort(-np.where(elig & np.isfinite(ms), ms, -np.inf))[:25]
            seed[top] = np.clip(ms[top], 0.0, None)
            rel = relational_reinstatement(qv, seed, edges, N, hops=2, cutoff=c["cutoff"])
            recs = episodic_roles(brain, c["query"], c["cutoff"], window=EPI_WINDOW)
            if not recs:
                n_epi_empty += 1
            epick = episodic_picked(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            eenc = episodic_encoded(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            edrop = episodic_dropped(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            per_cue[c["id"]] = {
                "elig": elig,
                "ms": z(ms, elig, N), "graph": z(rel, elig, N),
                "pick": z(epick, elig, N), "enc": z(eenc, elig, N), "drop": z(edrop, elig, N),
            }

        # score each config; need-collapsed hit + the reach/reinforcement decomposition vs maxsim
        results = {}
        base_ranks = {}      # cue -> {need: rank} under maxsim alone (the reference)
        for name, gains in CONFIGS.items():
            h5 = h25 = nc = 0
            brought = lost = reinforced = shared = 0
            for c in cues:
                pc = per_cue.get(c["id"])
                if pc is None or not c["needs"]:
                    continue
                vec = sum(g * pc[op] for op, g in gains.items())
                rk = ranks(vec, pc["elig"], master)
                br = best_ranks(rk, c)
                nc += 1
                hit5 = sum(1 for r in br.values() if r and r <= 5) / len(br)
                hit25 = sum(1 for r in br.values() if r and r <= 25) / len(br)
                h5 += hit5; h25 += hit25
                if name == "maxsim (base)":
                    base_ranks[c["id"]] = br
                else:
                    ref = base_ranks.get(c["id"], {})
                    for need, r in br.items():
                        r0 = ref.get(need)
                        in_now = r is not None and r <= 25
                        in_ref = r0 is not None and r0 <= 25
                        if in_now and not in_ref:
                            brought += 1
                        elif in_ref and not in_now:
                            lost += 1
                        elif in_now and in_ref:
                            shared += 1
                            if r < r0:
                                reinforced += 1
            results[name] = {"h5": h5 / nc, "h25": h25 / nc,
                             "brought": brought, "lost": lost,
                             "reinforced": reinforced, "shared": shared}

        out = []
        def p(s=""):
            print(s); out.append(s)
        p("LAYERING — one summed field (z-scored ops), %d cues | episodic empty: %d | window=%s"
          % (len(per_cue), n_epi_empty, EPI_WINDOW))
        p("\n  %-22s %-7s %-7s | %-9s %-7s %-9s %s" %
          ("config", "hit@5", "hit@25", "brought", "lost", "reinforced", "(vs maxsim @25)"))
        for name in CONFIGS:
            r = results[name]
            extra = "" if name == "maxsim (base)" else "+%d  −%d  ↑%d/%d" % (
                r["brought"], r["lost"], r["reinforced"], r["shared"])
            p("  %-22s %-7s %-7s | %s" %
              (name, "%.0f%%" % (100 * r["h5"]), "%.0f%%" % (100 * r["h25"]), extra))
        p("\n  brought = needs the stack reaches@25 that maxsim missed · lost = maxsim reached, stack dropped")
        p("  reinforced = needs both reach but the stack ranks higher (raised) — the overlap-still-has-value test")
        open(OUT_MD, "w").write("# LAF layering test (§18.21)\n\n```\n" + "\n".join(out) + "\n```\n")
        p("\n  → %s" % os.path.relpath(OUT_MD))


if __name__ == "__main__":
    main()
