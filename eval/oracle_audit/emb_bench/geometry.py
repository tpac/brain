#!/usr/bin/env python3
"""Vector-behavior worker — the flatness battery, per embedding model.

Where bench.py asks "does the model hit the golds", this asks "what does the
model's vector space LOOK like" on every substrate the LAF engine embeds.
Model-keyed and subprocess-isolated exactly like bench.py, so any future
embedder is one registry entry + one run away from a full geometry scorecard.

Blocks (per model, one results/geometry_<key>.json):
  A geometry     per substrate: anisotropy (random-pair cosine raw/centered),
                 mean-vector norm ratio, PC variance shares, participation ratio
  B ranking      Door-1 queries vs node `_primary`, pure cosine, as-of masked:
                 spread, decision-boundary margins (@5/@25), top-25 sigma,
                 hubness (top-25 k-occurrence concentration); raw AND centered
                 arms; top-100 ids per query -> geometry_<key>_rank.npz for
                 cross-model rank comparison (geometry_report.py)
  C probes       graph-derived proposition bands: correction pairs vs
                 community-sibling pairs vs random pairs (same topic /
                 different proposition is the diagnosed nomic gap)
  D redundancy   multi-view collapse: cos(title,primary), cos(situation,primary)
                 per node — high mean means MaxSim degenerates to single-vector
  E gold         Door-1 golds: rank, hit@5/25, gold margin, gold z — raw and
                 centered arms (extends the walker centering finding cross-model)
  F edge_lane    cos(query, edge-why) conductance distribution vs the measured
                 nomic baseline (mean 0.499, 14% > 0.6) — raw and centered
  G cost         load ms, per-substrate embed s, single-query latency, peak RSS

Substrate pack from dump_pack.py (/tmp/emb_bench). Run:
  ./dev python3 eval/oracle_audit/emb_bench/geometry.py nomic_q
"""
import json
import os
import resource
import sys
import time

import numpy as np

import bench  # sibling: MODELS registry, backends, register_custom, spin polyfill

DATA = bench.DATA
OUTDIR = bench.OUTDIR
SEED = 20260807
RANDOM_PAIRS = 20000
EDGE_SAMPLE = 8000
RANK_STORE_K = 100


# ── block A: label-free geometry ─────────────────────────────────


def geometry_block(vecs, rng):
    """Anisotropy + spectrum stats for one substrate's (N, d) unit matrix."""
    n, d = vecs.shape
    i = rng.integers(0, n, RANDOM_PAIRS)
    j = rng.integers(0, n, RANDOM_PAIRS)
    keep = i != j
    pair_cos = np.einsum("ij,ij->i", vecs[i[keep]], vecs[j[keep]])

    mean_vec = vecs.mean(axis=0)
    centred = vecs - mean_vec
    cnorm = np.linalg.norm(centred, axis=1, keepdims=True)
    cnorm[cnorm == 0] = 1.0
    cunit = centred / cnorm
    pair_cos_c = np.einsum("ij,ij->i", cunit[i[keep]], cunit[j[keep]])

    # spectrum on the centred matrix (population structure, not the offset)
    sv = np.linalg.svd(centred, compute_uv=False)
    var = sv**2
    share = var / var.sum()
    participation = float((var.sum() ** 2) / (var**2).sum())

    return {
        "n": int(n), "dim": int(d),
        "random_pair_cos_mean": round(float(pair_cos.mean()), 4),
        "random_pair_cos_sigma": round(float(pair_cos.std()), 4),
        "random_pair_cos_mean_centred": round(float(pair_cos_c.mean()), 4),
        "random_pair_cos_sigma_centred": round(float(pair_cos_c.std()), 4),
        "mean_vector_norm_ratio": round(
            float(np.linalg.norm(mean_vec) / np.linalg.norm(vecs, axis=1).mean()), 4),
        "pc1_share": round(float(share[0]), 4),
        "pc8_share": round(float(share[:8].sum()), 4),
        "participation_ratio": round(participation, 1),
    }


def _center_unit(vecs, mean_vec):
    c = vecs - mean_vec
    n = np.linalg.norm(c, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return c / n


# ── blocks B+E: ranking behavior + gold, one masked pass ─────────


def ranking_and_gold(node_vecs, node_ids, created, cues, cue_vecs,
                     gold_ids, arm_name):
    """One pass over Door-1 cues: per-query ranking stats, hubness, gold
    metrics, and stored top-K for cross-model comparison."""
    id_index = {nid: k for k, nid in enumerate(node_ids)}
    per_q, gold_rows = [], []
    occurrence = np.zeros(len(node_ids), dtype=np.int64)
    top_store = np.full((len(cues), RANK_STORE_K), -1, dtype=np.int32)

    for qi, cue in enumerate(cues):
        sims = node_vecs @ cue_vecs[qi]
        mask = created <= cue["ts"]
        idx = np.where(mask)[0]
        if len(idx) < 30:
            continue
        order = idx[np.argsort(-sims[idx])]
        s = sims[order]
        top_store[qi, : min(RANK_STORE_K, len(order))] = order[:RANK_STORE_K]
        occurrence[order[:25]] += 1
        per_q.append({
            "spread_1_25": float(s[0] - s[24]),
            "margin_5": float(s[4] - s[5]),
            "margin_25": float(s[24] - s[25]) if len(s) > 25 else 0.0,
            "top25_sigma": float(np.std(s[:25])),
        })

        gid = gold_ids.get(cue["key"])
        gi = id_index.get(gid) if gid else None
        if gi is not None and mask[gi]:
            rank = int(np.where(order == gi)[0][0]) + 1
            cand = sims[idx]
            mu, sd = float(cand.mean()), float(cand.std()) or 1e-9
            gold_rows.append({
                "rank": rank,
                "gold_margin_vs_top": float(sims[gi] - s[0]),
                "gold_z": (float(sims[gi]) - mu) / sd,
            })

    def col(rows, k):
        return np.array([r[k] for r in rows]) if rows else np.array([0.0])

    ranks = col(gold_rows, "rank")
    block = {
        "arm": arm_name,
        "n_queries": len(per_q),
        "mean_spread_1_25": round(float(col(per_q, "spread_1_25").mean()), 4),
        "mean_margin_5": round(float(col(per_q, "margin_5").mean()), 5),
        "mean_margin_25": round(float(col(per_q, "margin_25").mean()), 5),
        "mean_top25_sigma": round(float(col(per_q, "top25_sigma").mean()), 4),
        "hub_top1pct_share": round(float(
            np.sort(occurrence)[::-1][: max(1, len(node_ids) // 100)].sum()
            / max(occurrence.sum(), 1)), 4),
        "gold": {
            "n": len(gold_rows),
            "median_rank": int(np.median(ranks)) if gold_rows else None,
            "hit5": round(float((ranks <= 5).mean()), 4),
            "hit25": round(float((ranks <= 25).mean()), 4),
            "mean_gold_margin_vs_top": round(float(col(gold_rows, "gold_margin_vs_top").mean()), 4),
            "mean_gold_z": round(float(col(gold_rows, "gold_z").mean()), 3),
        },
    }
    return block, top_store


# ── block C: proposition bands from graph structure ──────────────


def probe_block(vecs, node_ids, pairs, rng):
    idx = {nid: k for k, nid in enumerate(node_ids)}

    def band(pair_list):
        rows = [(idx[a], idx[b]) for a, b in pair_list if a in idx and b in idx]
        if not rows:
            return {"n": 0}
        a, b = zip(*rows)
        cos = np.einsum("ij,ij->i", vecs[list(a)], vecs[list(b)])
        return {"n": len(rows), "mean": round(float(cos.mean()), 4),
                "sigma": round(float(cos.std()), 4)}

    n = len(node_ids)
    ri = rng.integers(0, n, 5000)
    rj = rng.integers(0, n, 5000)
    keep = ri != rj
    rand_cos = np.einsum("ij,ij->i", vecs[ri[keep]], vecs[rj[keep]])

    correction = band(pairs["correction_pairs"])
    sibling = band(pairs["community_sibling_pairs"])
    rand = {"n": int(keep.sum()), "mean": round(float(rand_cos.mean()), 4),
            "sigma": round(float(rand_cos.std()), 4)}
    return {
        "correction_pairs": correction,
        "community_sibling_pairs": sibling,
        "random_pairs": rand,
        # topic signal: same-topic pairs above random
        "band_gap_sibling_vs_random": round(sibling.get("mean", 0) - rand["mean"], 4),
        # proposition texture: spread WITHIN same-topic pairs — flat = all
        # topical siblings equidistant, no within-topic structure to rank on
        "within_topic_sigma": sibling.get("sigma"),
    }


# ── main ─────────────────────────────────────────────────────────


def run(key):
    bench.register_custom(key)
    hf_name, doc_prefix, query_prefix, backend_kind = bench.MODELS[key]
    rng = np.random.default_rng(SEED)

    nodes = json.load(open(os.path.join(DATA, "nodes.json")))
    edges = json.load(open(os.path.join(DATA, "edges.json")))
    episodic = json.load(open(os.path.join(DATA, "episodic.json")))
    door1 = json.load(open(os.path.join(DATA, "door1_cues.json")))

    node_ids = [n["id"] for n in nodes]
    created = np.array([n["created_at"] for n in nodes])
    gold_ids = {c["key"]: c["gold_id"] for c in door1 if c.get("gold_id")}

    t0 = time.time()
    Backend = (bench.STBackend if backend_kind == "st"
               or os.environ.get("BENCH_BACKEND") == "st" else bench.FastEmbedBackend)
    model = Backend(hf_name)
    load_ms = round((time.time() - t0) * 1000)

    # ── substrate texts ──
    edge_whys = [r["description"] for r in edges["relations"] if r["description"].strip()]
    if len(edge_whys) > EDGE_SAMPLE:
        edge_whys = list(rng.choice(edge_whys, EDGE_SAMPLE, replace=False))
    substrates_doc = {
        "node_primary": ["%s %s" % (n["title"], n["content"]) for n in nodes],
        "node_title": [n["title"] for n in nodes],
        "node_situation": [n["situation"] for n in nodes if n.get("situation")],
        "node_question": [n["question"] for n in nodes if n.get("question")],
        "edge_why": edge_whys,
        "episodic": [e["text"] for e in episodic],
    }

    # Substrate vector cache — keyed by (model, substrate, pack content hash).
    # Metric tweaks in this file re-run in seconds instead of re-embedding.
    pack_meta = json.load(open(os.path.join(DATA, "pack_meta.json")))
    pack_tag = pack_meta["sha256_16"]["nodes.json"][:8]

    def embed_cached(name, texts, prefix):
        cache = os.path.join(OUTDIR, f"vecs_{key}_{name}_{pack_tag}.npz")
        if os.path.exists(cache):
            v = np.load(cache)["v"]
            if len(v) == len(texts):
                embed_s[name] = 0.0
                print(f"  {name}: {len(texts)} from cache", flush=True)
                return v
        t = time.time()
        v = model.embed(texts, prefix)
        embed_s[name] = round(time.time() - t, 1)
        np.savez_compressed(cache, v=v)
        print(f"  embedded {name}: {len(texts)} in {embed_s[name]}s", flush=True)
        return v

    embed_s, vecs = {}, {}
    for name, texts in substrates_doc.items():
        vecs[name] = embed_cached(name, texts, doc_prefix)
    qvecs = embed_cached("door1_queries", [c["query"] for c in door1], query_prefix)

    lat = []
    for c in door1[:15]:
        ts = time.time()
        model.embed([c["query"]], query_prefix)
        lat.append((time.time() - ts) * 1000)

    # ── A: geometry per substrate ──
    geometry = {name: geometry_block(v, rng) for name, v in vecs.items()}
    geometry["door1_queries"] = geometry_block(qvecs, rng)

    # ── B+E: ranking + gold, raw and centred arms ──
    prim = vecs["node_primary"]
    rank_raw, top_raw = ranking_and_gold(
        prim, node_ids, created, door1, qvecs, gold_ids, "raw")
    node_mean = prim.mean(axis=0)
    query_mean = qvecs.mean(axis=0)
    rank_cen, top_cen = ranking_and_gold(
        _center_unit(prim, node_mean), node_ids, created, door1,
        _center_unit(qvecs, query_mean), gold_ids, "centred")

    # ── C: proposition bands (raw + centred — flatness hides structure) ──
    probes = {
        "raw": probe_block(prim, node_ids, edges, rng),
        "centred": probe_block(_center_unit(prim, node_mean), node_ids, edges, rng),
    }

    # ── D: multi-view redundancy ──
    def view_redundancy(view_key):
        # vecs[node_<view>] was embedded over nodes having the field, in node
        # order — so row k of the view matrix pairs with the k-th such node.
        prim_idx = [k for k, n in enumerate(nodes) if n.get(view_key)]
        if not prim_idx:
            return {"n": 0}
        cos = np.einsum("ij,ij->i", prim[prim_idx], vecs[f"node_{view_key}"])
        return {"n": len(prim_idx), "mean": round(float(cos.mean()), 4),
                "sigma": round(float(cos.std()), 4)}

    redundancy = {
        "title_vs_primary": view_redundancy("title"),
        "situation_vs_primary": view_redundancy("situation"),
        "question_vs_primary": view_redundancy("question"),
    }

    # ── F: edge-lane conductance (query -> edge-why cosine) ──
    ew = vecs["edge_why"]
    qsub = qvecs[rng.choice(len(qvecs), min(400, len(qvecs)), replace=False)]
    cond = (qsub @ ew.T).ravel()
    ew_mean = ew.mean(axis=0)
    cond_c = (_center_unit(qsub, query_mean) @ _center_unit(ew, ew_mean).T).ravel()
    edge_lane = {
        "raw": {"mean": round(float(cond.mean()), 4),
                "sigma": round(float(cond.std()), 4),
                "frac_above_06": round(float((cond > 0.6).mean()), 4)},
        "centred": {"mean": round(float(cond_c.mean()), 4),
                    "sigma": round(float(cond_c.std()), 4),
                    "frac_above_06": round(float((cond_c > 0.6).mean()), 4)},
    }

    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))
    out = {
        "key": key, "hf_name": hf_name,
        "doc_prefix": doc_prefix, "query_prefix": query_prefix,
        "pack_meta": json.load(open(os.path.join(DATA, "pack_meta.json"))),
        "geometry": geometry,
        "ranking": {"raw": rank_raw, "centred": rank_cen},
        "probes": probes,
        "redundancy": redundancy,
        "edge_lane": edge_lane,
        "cost": {"load_ms": load_ms, "embed_s": embed_s,
                 "latency_ms": round(float(np.median(lat)), 1),
                 "peak_rss_mb": peak_rss_mb},
    }
    json.dump(out, open(os.path.join(OUTDIR, f"geometry_{key}.json"), "w"), indent=1)
    np.savez_compressed(
        os.path.join(OUTDIR, f"geometry_{key}_rank.npz"),
        top_raw=top_raw, top_centred=top_cen,
        node_ids=np.array(node_ids), query_keys=np.array([c["key"] for c in door1]))

    g = geometry["node_primary"]
    r = rank_raw
    print(f"[{key}] primary: aniso={g['random_pair_cos_mean']} "
          f"PR={g['participation_ratio']} | spread={r['mean_spread_1_25']} "
          f"margin5={r['mean_margin_5']} | gold med_rank={r['gold']['median_rank']} "
          f"hit5={r['gold']['hit5']:.0%} gz={r['gold']['mean_gold_z']} "
          f"| {peak_rss_mb}MB")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "nomic_q")
