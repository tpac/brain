#!/usr/bin/env python3
"""MaxSim decomposition — break the nanmax into per-view lanes and measure (Tom, 2026-07-02).

Tom's question: situation-in-maxsim vs situation-as-own-layer differ — so should maxsim
itself be broken into separate layers? Three composition semantics over the SAME 6 views
(title, _primary, high_meta, other_meta, edge_context, question):

  nanmax(raw)  — the shipped operator: per node, best RAW cosine across views. Selection
                 semantics; biased toward hot-distribution views (the nanmax-enrichment bias).
  max(z)       — selection semantics, but each view z-scored first (distribution bias removed).
  sum(z)       — evidence accumulation: views VOTE; agreement stacks. Fully trainable
                 (each view its own gain; equal gains here — P3 fits them).

Each view also standalone (which views carry signal alone), and the winner slotted into the
full laf_v1 stack (uncapped pick/enc + idf + sit) vs the nanmax-based reference.

Run: ./dev python3 eval/laf/maxsim_decomp.py
Out: eval/laf/maxsim_decomp.md
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from episodic_ops import episodic_roles, episodic_encoded, episodic_picked  # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field,
)
from laf_metrics import zscore, ranks, best_ranks, need_hit, need_bl  # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402
from composition_probe import (                                       # noqa: E402
    UncappedEpisodes, fts_lane, idf_lane, build_title_tokens,
    build_situation_matrix, EPI_WINDOW,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "maxsim_decomp.md")


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        groups = list(MAXSIM_GROUPS)
        master, idx, mats = build_field_matrices(brain, model, groups)
        N = len(master)
        ca_rows = dict(brain._fts.conn.execute(
            "SELECT id, created_at FROM nodes").fetchall())
        ca = np.array([ca_rows.get(nid, "") or "" for nid in master])
        title_tok = build_title_tokens(brain, idx)
        sit_M, _ = build_situation_matrix(brain, idx, N, model)
        scan = UncappedEpisodes(brain)

        per, base_ref = {}, {}
        for c in cues:
            qv = query_vec(c["query"])
            if qv is None or not c["needs"]:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            view_raw = {vt: mats[vt] @ qv for vt in groups}       # NaN where absent
            view_z = {vt: zscore(view_raw[vt], elig, N) for vt in groups}
            ms = maxsim_field(qv, mats, groups)

            real = brain.recall_episodes
            brain.recall_episodes = scan
            try:
                recs = episodic_roles(brain, c["query"], c["cutoff"], window=EPI_WINDOW)
            finally:
                brain.recall_episodes = real
            pick = episodic_picked(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            enc = episodic_encoded(brain, c["query"], c["cutoff"], idx, N, _records=recs)
            idfv = idf_lane(c["query"], title_tok, N)
            sit = np.zeros(N)
            if sit_M is not None:
                s = sit_M @ qv
                sit = np.where(np.isfinite(s), s, 0.0)

            zstack = np.stack([view_z[vt] for vt in groups])       # [6, N], zeros where absent
            per[c["id"]] = {
                "elig": elig, "needs": c["needs"],
                "zms": zscore(ms, elig, N),
                "maxz": zstack.max(axis=0),
                "sumz": zstack.sum(axis=0),
                "views": view_z,
                "zpick": zscore(pick, elig, N), "zenc": zscore(enc, elig, N),
                "zidf": zscore(idfv, elig, N), "zsit": zscore(sit, elig, N),
            }
            base_ref[c["id"]] = best_ranks(ranks(ms, elig, master), c["needs"])
        nc = len(per) or 1

        def lanes(p):
            return 0.5 * p["zpick"] + 0.3 * p["zenc"] + 0.5 * p["zidf"] + 0.5 * p["zsit"]

        CONFIGS = [("view %s alone" % vt,
                    (lambda v: lambda p: p["views"][v])(vt)) for vt in groups]
        CONFIGS += [
            ("nanmax(raw)  [shipped]", lambda p: p["zms"]),
            ("max(z)",                 lambda p: zscore(p["maxz"], p["elig"], N)),
            ("sum(z)",                 lambda p: p["sumz"]),
            ("laf_v1 ref (nanmax)",    lambda p: p["zms"] + lanes(p)),
            ("laf_v1 max(z)",          lambda p: zscore(p["maxz"], p["elig"], N) + lanes(p)),
            ("laf_v1 sum(z)",          lambda p: zscore(p["sumz"], p["elig"], N) + lanes(p)),
        ]

        lines = ["# MaxSim decomposition — nanmax vs max(z) vs sum(z) over the 6 views",
                 "", "%d cues · %d nodes" % (nc, N), "",
                 "| config | need@5 | need@25 | brought | lost |", "|---|---|---|---|---|"]
        print("\n  %-24s %-7s %-8s | %-8s %s" % ("config", "need@5", "need@25", "brought", "lost"))
        for name, fn in CONFIGS:
            h5 = h25 = brought = lost = 0
            for cid, p in per.items():
                sc = fn(p)
                h5 += need_hit(sc, p["elig"], master, p["needs"], 5)
                h25 += need_hit(sc, p["elig"], master, p["needs"], 25)
                b, l = need_bl(sc, p["elig"], master, p["needs"], base_ref[cid])
                brought += b
                lost += l
            row = (name, "%.0f%%" % (100*h5/nc), "%.0f%%" % (100*h25/nc), brought, lost)
            print("  %-24s %-7s %-8s | +%-7d −%d" % row)
            lines.append("| %s | %s | %s | +%d | −%d |" % row)

        with open(OUT_MD, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\nwrote %s" % OUT_MD)


if __name__ == "__main__":
    main()
