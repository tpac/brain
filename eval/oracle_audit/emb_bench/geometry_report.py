#!/usr/bin/env python3
"""Cross-model vector-behavior report — compares geometry.py scorecards.

Loads results/geometry_<key>.json (+ _rank.npz) for two or more models and
renders one markdown scorecard: per-substrate geometry deltas, ranking arms,
proposition bands, view redundancy, edge-lane conductance, gold block, and a
cross-model rank-agreement section (Jaccard@25 + Spearman over top-100 union
per query — how DIFFERENTLY the models see the same corpus).

Run: ./dev python3 eval/oracle_audit/emb_bench/geometry_report.py nomic_q gte_modernbert
Out: results/geometry_compare_<a>_vs_<b>.md (also printed)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def load(key):
    doc = json.load(open(os.path.join(RESULTS, f"geometry_{key}.json")))
    npz = np.load(os.path.join(RESULTS, f"geometry_{key}_rank.npz"), allow_pickle=False)
    return doc, npz


def rank_agreement(npz_a, npz_b, arm="top_raw", sample=600, seed=20260807):
    """Per-query Jaccard@25 and Spearman over the union of both top-100s.
    Requires identical query order (same pack — enforced by pack hash)."""
    ta, tb = npz_a[arm], npz_b[arm]
    assert len(ta) == len(tb), "rank stores differ in query count — different packs?"
    rng = np.random.default_rng(seed)
    qs = rng.choice(len(ta), min(sample, len(ta)), replace=False)
    jac, rho = [], []
    for q in qs:
        a, b = ta[q], tb[q]
        a = a[a >= 0]
        b = b[b >= 0]
        if len(a) < 25 or len(b) < 25:
            continue
        sa, sb = set(a[:25].tolist()), set(b[:25].tolist())
        jac.append(len(sa & sb) / len(sa | sb))
        union = np.array(sorted(set(a.tolist()) | set(b.tolist())))
        pos_a = {nid: r for r, nid in enumerate(a)}
        pos_b = {nid: r for r, nid in enumerate(b)}
        cap = len(a) + 1
        ra = np.array([pos_a.get(n, cap) for n in union], dtype=np.float64)
        rb = np.array([pos_b.get(n, cap) for n in union], dtype=np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt((ra**2).sum() * (rb**2).sum())
        if denom > 0:
            rho.append(float((ra * rb).sum() / denom))
    return {
        "n_queries": len(jac),
        "jaccard_at_25_mean": round(float(np.mean(jac)), 4),
        "spearman_top100_union_mean": round(float(np.mean(rho)), 4),
    }


def table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main(keys):
    docs = {}
    npzs = {}
    for k in keys:
        docs[k], npzs[k] = load(k)

    hashes = {json.dumps(d["pack_meta"]["sha256_16"], sort_keys=True) for d in docs.values()}
    pack_warning = "" if len(hashes) == 1 else (
        "\n> **WARNING: models ran on DIFFERENT packs — numbers are not comparable.**\n")

    L = []
    L.append(f"# Vector-behavior scorecard — {' vs '.join(keys)}\n")
    L.append(f"Pack: {docs[keys[0]]['pack_meta']['dumped_at']} · "
             f"{docs[keys[0]]['pack_meta']['n_door1_cues']} Door-1 cues · "
             f"{docs[keys[0]]['pack_meta']['n_door1_with_gold']} with gold"
             + pack_warning)

    # A: geometry per substrate
    L.append("\n## A · Space geometry (label-free)\n")
    subs = list(docs[keys[0]]["geometry"].keys())
    for sub in subs:
        rows = []
        for k in keys:
            g = docs[k]["geometry"][sub]
            rows.append([k, g["n"], g["random_pair_cos_mean"],
                         g["random_pair_cos_sigma"],
                         g["random_pair_cos_sigma_centred"],
                         g["mean_vector_norm_ratio"], g["pc1_share"],
                         g["participation_ratio"]])
        L.append(f"**{sub}**\n")
        L.append(table(rows, ["model", "n", "aniso(cos-rand)", "σ raw",
                              "σ centred", "‖mean‖ ratio", "PC1", "PR(eff dims)"]))
        L.append("")

    # B+E: ranking arms + gold
    L.append("\n## B · Ranking behavior + E · Door-1 gold (node_primary, pure cosine)\n")
    rows = []
    for k in keys:
        for arm in ("raw", "centred"):
            r = docs[k]["ranking"][arm]
            g = r["gold"]
            rows.append([k, arm, r["n_queries"], r["mean_spread_1_25"],
                         r["mean_margin_5"], r["mean_margin_25"],
                         r["mean_top25_sigma"], r["hub_top1pct_share"],
                         g["median_rank"], f"{g['hit5']:.0%}", f"{g['hit25']:.0%}",
                         g["mean_gold_margin_vs_top"], g["mean_gold_z"]])
    L.append(table(rows, ["model", "arm", "Q", "spread1-25", "margin@5",
                          "margin@25", "top25σ", "hub1%share",
                          "gold med", "hit@5", "hit@25", "g-margin", "g-z"]))

    # C: proposition bands
    L.append("\n## C · Proposition bands (same-topic discrimination)\n")
    rows = []
    for k in keys:
        for arm in ("raw", "centred"):
            p = docs[k]["probes"][arm]
            rows.append([k, arm,
                         f"{p['correction_pairs']['mean']} ±{p['correction_pairs']['sigma']}",
                         f"{p['community_sibling_pairs']['mean']} ±{p['community_sibling_pairs']['sigma']}",
                         f"{p['random_pairs']['mean']} ±{p['random_pairs']['sigma']}",
                         p["band_gap_sibling_vs_random"], p["within_topic_sigma"]])
    L.append(table(rows, ["model", "arm", "correction pairs", "topic siblings",
                          "random", "topic−random gap", "within-topic σ"]))

    # D: redundancy
    L.append("\n## D · Multi-view redundancy (MaxSim degeneration check)\n")
    rows = []
    for k in keys:
        r = docs[k]["redundancy"]
        rows.append([k,
                     f"{r['title_vs_primary']['mean']} ±{r['title_vs_primary']['sigma']}",
                     f"{r['situation_vs_primary']['mean']} ±{r['situation_vs_primary']['sigma']}",
                     f"{r['question_vs_primary']['mean']} ±{r['question_vs_primary']['sigma']}"])
    L.append(table(rows, ["model", "title↔primary", "situation↔primary", "question↔primary"]))

    # F: edge lane
    L.append("\n## F · Edge-lane conductance (query → edge-why cosine)\n")
    rows = []
    for k in keys:
        for arm in ("raw", "centred"):
            e = docs[k]["edge_lane"][arm]
            rows.append([k, arm, e["mean"], e["sigma"], f"{e['frac_above_06']:.1%}"])
    L.append(table(rows, ["model", "arm", "mean", "σ", ">0.6"]))

    # cross-model rank agreement
    if len(keys) == 2:
        L.append("\n## Cross-model rank agreement (how differently they see the corpus)\n")
        rows = []
        for arm in ("top_raw", "top_centred"):
            agree = rank_agreement(npzs[keys[0]], npzs[keys[1]], arm)
            rows.append([arm.replace("top_", ""), agree["n_queries"],
                         agree["jaccard_at_25_mean"],
                         agree["spearman_top100_union_mean"]])
        L.append(table(rows, ["arm", "Q", "Jaccard@25", "Spearman(top-100 ∪)"]))

    # cost
    L.append("\n## G · Cost\n")
    rows = []
    for k in keys:
        c = docs[k]["cost"]
        rows.append([k, c["load_ms"], c["embed_s"].get("node_primary"),
                     c["latency_ms"], c["peak_rss_mb"]])
    L.append(table(rows, ["model", "load ms", "primary embed s", "1-query ms", "peak RSS MB"]))
    L.append("")

    text = "\n".join(L)
    out = os.path.join(RESULTS, f"geometry_compare_{'_vs_'.join(keys)}.md")
    open(out, "w").write(text)
    print(text)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["nomic_q", "gte_modernbert"])
