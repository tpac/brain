#!/usr/bin/env python3
"""Stage-2 reranker A/B — does a small cross-encoder lift hit@5 by reordering nomic's top-K?

Baseline  = nomic cosine ORDER of the top-K candidates.
Treatment = same K candidates, REORDERED by a cross-encoder (joint query-doc attention).

Pure precision test: hit@K is identical across arms (same K nodes) — only the ORDER
changes, so hit@1/@5, recall@5, nDCG@5 move. Ceiling for hit@5 is the baseline's hit@K
(reranking can at best sort every in-pool gold to the top).

Runs entirely in fastembed ONNX-CPU (no torch). Node text truncated to the reranker's
512-token limit (~RERANK_CHARS). Reuses bench.py for the nomic path + score_one.

Run: BENCH_THREADS=4 nice -n 10 .../python rerank_ab.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # FastEmbedBackend, score_one, MODELS, polyfill, _normalize

DATA = "/tmp/emb_bench"
K = int(os.environ.get("RERANK_K", "25"))
RERANK_CHARS = 1800                       # ~450 tok, safe under the 512 cap
RERANKER = os.environ.get("RERANKER", "BAAI/bge-reranker-base")
OUT = os.path.join(HERE, "results", "rerank_ab.json")


def agg(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return sum(v) / len(v) if v else 0.0


def main():
    nodes = json.load(open(f"{DATA}/nodes.json"))
    cues = json.load(open(f"{DATA}/cues.json"))
    ids = [n["id"] for n in nodes]
    created = np.array([n["created_at"] for n in nodes])
    docs = ["%s %s" % (n["title"], n["content"]) for n in nodes]
    id2doc = dict(zip(ids, docs))

    # ── nomic candidate generation (same path as the validated baseline) ──
    hf, dp, qp, _ = bench.MODELS["nomic_q"]
    print(f"embedding {len(nodes)} nodes + {len(cues)} cues with nomic-Q …", flush=True)
    m = bench.FastEmbedBackend(hf)
    node_vecs = m.embed(docs, dp)
    cue_vecs = m.embed([c["query"] for c in cues], qp)

    # ── reranker (fastembed ONNX-CPU) ──
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    print(f"loading reranker {RERANKER} …", flush=True)
    ce = TextCrossEncoder(model_name=RERANKER)

    base, rr = [], []
    lat = []
    for i, c in enumerate(cues):
        sims = node_vecs @ cue_vecs[i]
        idx = np.where(created <= c["cutoff"])[0]
        order = idx[np.argsort(-sims[idx])][:K]
        top_ids = [ids[j] for j in order]                       # nomic order
        ess, helpful = c["gold_essential"], c.get("gold_helpful", [])
        base.append({**bench.score_one(top_ids, ess, helpful),
                     "source": c["source"]})

        rdocs = [id2doc[t][:RERANK_CHARS] for t in top_ids]
        t0 = time.time()
        scores = np.array(list(ce.rerank(c["query"], rdocs)))   # aligned to rdocs
        lat.append(time.time() - t0)
        rr_ids = [top_ids[j] for j in np.argsort(-scores)]
        rr.append({**bench.score_one(rr_ids, ess, helpful),
                   "source": c["source"]})

    def report(label, rows):
        print(f"  {label:26s} n={len(rows):3d} | hit@1 {agg(rows,'hit1_ess'):.0%}  "
              f"hit@5 {agg(rows,'hit5_ess'):.0%}  hit@25 {agg(rows,'hit25_ess'):.0%}  "
              f"recall@5 {agg(rows,'recall5_ess'):.0%}  nDCG@5 {agg(rows,'ndcg5'):.2f}")

    print(f"\n{'='*86}\nRERANK A/B — nomic top-{K} reordered by {RERANKER} | 73-cue endo gold\n{'='*86}")
    report("nomic order (baseline)", base)
    report("+ reranker", rr)
    for src in ("operator_msg", "anchor_turn"):
        print(f"  -- {src} --")
        report("    nomic", [r for r in base if r["source"] == src])
        report("    + reranker", [r for r in rr if r["source"] == src])
    print(f"\n  rerank latency/cue: median {np.median(lat)*1000:.0f}ms, max {np.max(lat)*1000:.0f}ms (K={K}, {len(cues)} cues)")
    print(f"  (hit@25 must match across arms — same K candidates, only order changes)")

    json.dump({"baseline": base, "rerank": rr, "reranker": RERANKER, "K": K,
               "lat_ms_median": float(np.median(lat) * 1000)}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
