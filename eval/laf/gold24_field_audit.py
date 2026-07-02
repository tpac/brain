#!/usr/bin/env python3
"""Per-field HEALTH AUDIT — every activation field must stand on its own before we layer (§18.21).

Tom's gate (2026-06-30): "make sure each activation field we already have is built properly and
returns results we can stand behind by itself — no broken stuff inside or wrong assumptions."
Layering on a broken field (graph is inert, temporal is degenerate, MaxSim has a nanmax-enrichment
bias per 794c137a) is building on sand. So before any interaction analysis, audit each field ALONE.

For each field this reports:
  • COVERAGE   — % of nodes that actually have a vector (catches the edge_context-dead class).
  • LIVENESS   — is the field non-constant / input-dependent (catches recency=1.000 degeneracy).
  • INVARIANT  — field-specific correctness (MaxSim ≥ its groups; primary == raw-byte cosine).
  • STANDALONE — the field's OWN need-collapsed gold hit@5/@25 on the 24-cue lens-independent gold.
  • BIAS FLAGS — the known wrong-assumptions: MaxSim↔enrichment correlation; temporal distinctiveness%.

Run (daemon maintenance-locked): ./dev python3 eval/laf/gold24_field_audit.py
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
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field, primary_field,
    build_adjacency, graph_spread, parse_days, temporal_distinctiveness,
)
from gold24_diagnostic import load_cues                              # noqa: E402
from gold24_matrix import episodic_field                            # noqa: E402


def need_hit(rank_of_node, cue, k):
    """need-collapsed hit@k for one cue given {node_id: rank}."""
    needs = defaultdict(list)
    for nid in cue["ess"]:
        nd = next((n for n, ids in cue["needs"].items() if nid in ids), nid)
        needs[nd].append(nid)
    if not needs:
        return None
    met = sum(1 for nids in needs.values()
              if any((rank_of_node.get(n) or 1e9) <= k for n in nids))
    return met / len(needs)


def ranks(scores, eligible, master):
    s = np.where(eligible & np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-s)
    return {master[i]: r + 1 for r, i in enumerate(order)}


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        adj = build_adjacency(brain, idx)
        ca = np.array([dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall()).get(n, "")
                       for n in master])
        days = parse_days(list(ca))
        N = len(master)

        # n_groups present per node (enrichment) — for the MaxSim bias check
        n_groups = np.zeros(N)
        for vt in MAXSIM_GROUPS:
            n_groups += np.isfinite(mats[vt][:, 0]).astype(float)

        FIELDS = ["_primary", "maxsim"] + list(MAXSIM_GROUPS) + ["graph1hop", "temporal", "episodic"]

        # ---- per-cue field score vectors (compute once) ----
        per_cue = {}
        for c in cues:
            qv = query_vec(c["query"])
            elig = (ca != "") & (ca <= c["cutoff"])
            if qv is None:
                continue
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))
            vecs = {"_primary": primary_field(qv, mats), "maxsim": ms}
            for vt in MAXSIM_GROUPS:
                vecs[vt] = mats[vt] @ qv
            seeds = (ms > np.nanpercentile(ms[np.isfinite(ms)], 90)).astype(np.float64)
            vecs["graph1hop"] = graph_spread(seeds, adj, hops=1)
            vecs["temporal"] = temporal_distinctiveness(days, elig, 7.0)
            vecs["episodic"] = episodic_field(brain, c["query"], c["cutoff"], idx, N)
            per_cue[c["id"]] = (vecs, elig)

        print("field audit — %d cues, %d embedded nodes\n" % (len(per_cue), N))
        print("  %-12s %-9s %-7s %-8s %-9s %s" %
              ("field", "coverage", "hit@5", "hit@25", "live?", "flags"))

        for f in FIELDS:
            # coverage
            if f in MAXSIM_GROUPS:
                cov = float(np.mean(np.isfinite(mats[f][:, 0])))
            elif f == "_primary":
                cov = float(np.mean(np.isfinite(mats["_primary"][:, 0])))
            else:
                cov = float("nan")   # derived field — coverage is of its inputs
            # standalone need-collapsed hit@k + liveness + bias accumulation
            h5 = h25 = nc = 0
            const_hits = 0
            ms_corr = []
            for c in cues:
                pc = per_cue.get(c["id"])
                if pc is None:
                    continue
                vecs, elig = pc
                rk = ranks(vecs[f], elig, master)
                m5 = need_hit(rk, c, 5); m25 = need_hit(rk, c, 25)
                if m5 is not None:
                    h5 += m5; h25 += m25; nc += 1
                v = vecs[f][elig & np.isfinite(vecs[f])]
                if v.size and float(np.std(v)) < 1e-9:
                    const_hits += 1
                if f == "maxsim":
                    # enrichment bias: corr(maxsim score, n_groups) over eligible nodes
                    mask = elig & np.isfinite(vecs[f])
                    if int(mask.sum()) > 10:
                        a = vecs[f][mask]; b = n_groups[mask]
                        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                            ms_corr.append(float(np.corrcoef(a, b)[0, 1]))
            h5 /= (nc or 1); h25 /= (nc or 1)

            flags = []
            if not np.isnan(cov) and cov < 0.5:
                flags.append("LOW-COVERAGE(%.0f%%)" % (100 * cov))
            if const_hits:
                flags.append("CONSTANT@%d-cues" % const_hits)
            if f == "temporal":
                # distinctiveness: fraction of eligible nodes that are temporally isolated (value high)
                pc = next(iter(per_cue.values()))
                tv = pc[0]["temporal"]; te = pc[1]
                vals = tv[te & (tv > 0)]
                distinct = float(np.mean(vals > 0.5)) if vals.size else 0.0
                flags.append("query-INDEP node-prior")
                flags.append("distinct=%.0f%%%s" % (100 * distinct,
                             " ⚠DEGENERATE" if distinct < 0.05 else ""))
            if f == "maxsim" and ms_corr:
                mc = float(np.mean(ms_corr))
                flags.append("enrich-corr=%.2f%s" % (mc, " ⚠BIAS" if mc > 0.3 else ""))
            if f == "graph1hop":
                # is it inert? share of nonzero activation
                pc = next(iter(per_cue.values()))
                gv = pc[0]["graph1hop"]
                nz = float(np.mean(gv > 1e-9))
                flags.append("nonzero=%.0f%%" % (100 * nz))
            if f == "episodic":
                pc = next(iter(per_cue.values()))
                ev = pc[0]["episodic"]
                flags.append("candidates~%d" % int(np.sum(ev > 0)))

            covs = "%.0f%%" % (100 * cov) if not np.isnan(cov) else "(derived)"
            live = "OK" if not const_hits else "DEAD"
            print("  %-12s %-9s %-7s %-8s %-9s %s" %
                  (f, covs, "%.0f%%" % (100 * h5), "%.0f%%" % (100 * h25), live, " ".join(flags)))

        # invariants
        print("\n  invariants:")
        c0 = cues[0]; vecs0, _ = per_cue[c0["id"]]
        viol = int(np.sum(vecs0["maxsim"] < np.nanmax(
            np.stack([vecs0[vt] for vt in MAXSIM_GROUPS]), axis=0) - 1e-5))
        print("    MaxSim >= max(groups) per node: %s (%d violations)"
              % ("OK" if viol == 0 else "FAIL", viol))
        qv0 = query_vec(c0["query"])
        recompute = mats["_primary"] @ qv0
        d = float(np.nanmax(np.abs(recompute - vecs0["_primary"])))
        print("    _primary == raw-byte cosine recompute: max|Δ|=%.2e %s" % (d, "OK" if d < 1e-4 else "FAIL"))


if __name__ == "__main__":
    main()
