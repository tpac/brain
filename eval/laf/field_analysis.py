#!/usr/bin/env python3
"""Field-interaction analysis — what the composition results IMPLY (Tom, 2026-07-02).

Four questions, one instrumented pass over the 24-cue gold:

  A. CAP → EXPLICIT RECENCY. The newest-500 cap helped @5 by accident (recency prior at
     moment selection). Structurally it shouldn't exist (Tom). Measure the principled
     replacement: uncapped full-history scan + e^(−ρ·Δdays) decay applied AT SELECTION
     (UncappedEpisodes rho arms) — does explicit recency recover/beat capped 16/28?

  B. FTS DRILL. FTS floods on average but should shine on rare-identifier prompts
     (names/places/companies), mostly operator-side (Tom). Per cue: source, the query's
     max title-idf token (rare-identifier proxy), fts-alone needs@25, unique reach vs
     maxsim. Answers: does the corpus even CONTAIN good FTS candidates?

  C. SIT DRILL. Where exactly does the situation lane pay? Per cue: stack_c+sit vs
     stack_c need@5/@25 deltas + needs sit-alone reaches @25 that maxsim misses.

  D. RESIDUAL — the missing-field question, answered verbally. Needs NO lane reaches
     @25 (best rank per lane, each lane ranked ALONE) — dumped with cue source + need
     text so the miss themes are readable, plus per-lane UNIQUE reach (which lanes earn
     slots). The residual texts are the spec for the next activation field.

Run: ./dev python3 eval/laf/field_analysis.py
Out: eval/laf/field_analysis.md + field_analysis.json
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from servers.brain_constants import _TITLE_BOOST_STOPWORDS            # noqa: E402
import episodic_ops                                                   # noqa: E402
from episodic_ops import episodic_roles, episodic_encoded, episodic_picked  # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field,
)
from laf_metrics import zscore, ranks, best_ranks, need_hit           # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402
from composition_probe import (                                       # noqa: E402
    UncappedEpisodes, fts_lane, idf_lane, build_title_tokens,
    build_situation_matrix, _IDF_TOK, EPI_WINDOW,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "field_analysis.md")
OUT_JSON = os.path.join(HERE, "field_analysis.json")
RHOS = (0.01, 0.05, 0.15)            # per-day decay arms (0.01 = SYNAPSE's value)


def epi_pair(brain, cue, scanner, idx, N):
    """(pick, enc) vectors with `scanner` standing in for recall_episodes."""
    real = brain.recall_episodes
    brain.recall_episodes = scanner
    try:
        recs = episodic_roles(brain, cue["query"], cue["cutoff"], window=EPI_WINDOW)
    finally:
        brain.recall_episodes = real
    return (episodic_picked(brain, cue["query"], cue["cutoff"], idx, N, _records=recs),
            episodic_encoded(brain, cue["query"], cue["cutoff"], idx, N, _records=recs))


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        N = len(master)
        ca_rows = dict(brain._fts.conn.execute(
            "SELECT id, created_at FROM nodes").fetchall())
        ca = np.array([ca_rows.get(nid, "") or "" for nid in master])
        title_tok = build_title_tokens(brain, idx)
        sit_M, _ = build_situation_matrix(brain, idx, N, model)
        base_scan = UncappedEpisodes(brain)
        decay_scans = {r: UncappedEpisodes(brain, rho=r, _share=base_scan) for r in RHOS}

        # global title-idf table (for the rare-identifier proxy in the FTS drill)
        n_titles = max(len(title_tok), 1)
        df = defaultdict(int)
        for ts in title_tok.values():
            for t in ts:
                df[t] += 1

        LANES = ["ms", "pick_c", "enc_c", "pick_u", "enc_u", "fts", "idf", "sit"]
        per, drill = {}, []
        for c in cues:
            qv = query_vec(c["query"])
            if qv is None or not c["needs"]:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))
            recs_c = episodic_roles(brain, c["query"], c["cutoff"], window=EPI_WINDOW)
            pick_c = episodic_picked(brain, c["query"], c["cutoff"], idx, N, _records=recs_c)
            enc_c = episodic_encoded(brain, c["query"], c["cutoff"], idx, N, _records=recs_c)
            pick_u, enc_u = epi_pair(brain, c, base_scan, idx, N)
            decay_pairs = {r: epi_pair(brain, c, s, idx, N)
                           for r, s in decay_scans.items()}
            fts = fts_lane(brain, c["query"], idx, N)
            idfv = idf_lane(c["query"], title_tok, N)
            sit = np.zeros(N)
            if sit_M is not None:
                s = sit_M @ qv
                sit = np.where(np.isfinite(s), s, 0.0)

            raw = {"ms": ms, "pick_c": pick_c, "enc_c": enc_c,
                   "pick_u": pick_u, "enc_u": enc_u,
                   "fts": fts, "idf": idfv, "sit": sit}
            z = {k: zscore(v, elig, N) for k, v in raw.items()}
            for r, (pu, eu) in decay_pairs.items():
                z["pick_d%.2f" % r] = zscore(pu, elig, N)
                z["enc_d%.2f" % r] = zscore(eu, elig, N)

            # per-lane best rank per need (each lane ranked ALONE)
            lane_best = {k: best_ranks(ranks(raw[k], elig, master), c["needs"])
                         for k in LANES}
            q_tokens = {t for t in _IDF_TOK.findall(c["query"].lower())
                        if len(t) >= 2 and t not in _TITLE_BOOST_STOPWORDS}
            tok_idf = sorted(((math.log((n_titles + 1) / (df.get(t, 0) + 1)), t)
                              for t in q_tokens), reverse=True)
            per[c["id"]] = {"elig": elig, "needs": c["needs"], "z": z}
            drill.append({
                "cue": c["id"], "source": c["source"], "qt": c["query_type"],
                "query": c["query"][:200], "lane_best": lane_best,
                "top_idf_tokens": [(t, round(v, 2)) for v, t in tok_idf[:3]],
            })
        nc = len(per) or 1

        # ── A. decay arms ──
        def stack(p, pk, ek, extra=()):
            s = p["z"]["ms"] + 0.5 * p["z"][pk] + 0.3 * p["z"][ek]
            for g, k in extra:
                s = s + g * p["z"][k]
            return s

        arms = [("stack capped (ref)", "pick_c", "enc_c"),
                ("stack uncapped", "pick_u", "enc_u")]
        arms += [("uncapped + decay ρ=%.2f" % r, "pick_d%.2f" % r, "enc_d%.2f" % r)
                 for r in RHOS]
        a_rows = []
        for name, pk, ek in arms:
            for extra, suff in (((), ""), ((((0.5, "idf")), (0.5, "sit")), " +idf+sit")):
                h5 = h25 = 0
                for cid, p in per.items():
                    sc = stack(p, pk, ek, extra)
                    h5 += need_hit(sc, p["elig"], master, p["needs"], 5)
                    h25 += need_hit(sc, p["elig"], master, p["needs"], 25)
                a_rows.append((name + suff, 100 * h5 / nc, 100 * h25 / nc))

        # ── D. residual + unique reach ──
        residual, unique = [], defaultdict(list)
        for d in drill:
            for need, nids in per[d["cue"]]["needs"].items():
                bests = {k: d["lane_best"][k].get(need) for k in LANES}
                reach = {k for k, r in bests.items() if r is not None and r <= 25}
                if not reach:
                    residual.append({"cue": d["cue"], "source": d["source"],
                                     "qt": d["qt"], "need": need,
                                     "best_any": min((r for r in bests.values()
                                                      if r is not None), default=None),
                                     "query": d["query"]})
                elif len(reach) == 1:
                    unique[next(iter(reach))].append((d["cue"], need[:90]))

        # ── write report ──
        L = ["# Field-interaction analysis (24-cue gold)", ""]
        L += ["## A. Cap → explicit recency (decay at moment selection)", "",
              "| arm | need@5 | need@25 |", "|---|---|---|"]
        L += ["| %s | %.0f%% | %.0f%% |" % r for r in a_rows]
        L += ["", "## B. FTS drill — where lexical should shine", "",
              "| cue | source | type | top idf tokens (rarity) | fts needs@25 | ms needs@25 |",
              "|---|---|---|---|---|---|"]
        for d in sorted(drill, key=lambda x: -max((v for _, v in
                        [(t, v) for t, v in x["top_idf_tokens"]]), default=0)):
            f25 = sum(1 for r in d["lane_best"]["fts"].values()
                      if r is not None and r <= 25)
            m25 = sum(1 for r in d["lane_best"]["ms"].values()
                      if r is not None and r <= 25)
            L.append("| %s | %s | %s | %s | %d/%d | %d/%d |" % (
                d["cue"], d["source"], d["qt"],
                ", ".join("%s(%.1f)" % t for t in d["top_idf_tokens"]),
                f25, len(d["lane_best"]["fts"]), m25, len(d["lane_best"]["ms"])))
        L += ["", "## C. Situation drill — per-cue effect of +sit on stack_c", "",
              "| cue | source | Δ@5 | Δ@25 |", "|---|---|---|---|"]
        for cid, p in per.items():
            b = stack(p, "pick_c", "enc_c")
            s = stack(p, "pick_c", "enc_c", ((0.5, "sit"),))
            d5 = need_hit(s, p["elig"], master, p["needs"], 5) - \
                need_hit(b, p["elig"], master, p["needs"], 5)
            d25 = need_hit(s, p["elig"], master, p["needs"], 25) - \
                need_hit(b, p["elig"], master, p["needs"], 25)
            if abs(d5) > 1e-9 or abs(d25) > 1e-9:
                src = next(d["source"] for d in drill if d["cue"] == cid)
                L.append("| %s | %s | %+.2f | %+.2f |" % (cid, src, d5, d25))
        L += ["", "## D. Unique reach per lane (needs ONLY that lane gets @25)", ""]
        for k in LANES:
            L.append("- **%s**: %d unique" % (k, len(unique[k])))
            for cid, nd in unique[k][:6]:
                L.append("    - %s — %s" % (cid, nd))
        L += ["", "## E. RESIDUAL — needs NO lane reaches @25 (%d)" % len(residual),
              "", "The verbatim spec for missing fields:", ""]
        for r in residual:
            L.append("- **%s** (%s/%s, best rank anywhere: %s)" % (
                r["cue"], r["source"], r["qt"], r["best_any"]))
            L.append("  - need: %s" % r["need"][:220])
            L.append("  - cue: %s" % r["query"][:150].replace("\n", " "))
        out = "\n".join(L) + "\n"
        with open(OUT_MD, "w") as f:
            f.write(out)
        with open(OUT_JSON, "w") as f:
            json.dump({"drill": drill, "residual": residual,
                       "unique": {k: v for k, v in unique.items()},
                       "arms": a_rows}, f, indent=1, default=str)
        print(out)
        print("wrote %s + .json" % OUT_MD)


if __name__ == "__main__":
    main()
