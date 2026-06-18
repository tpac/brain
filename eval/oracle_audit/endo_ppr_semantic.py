#!/usr/bin/env python3
"""RE-RUN the PPR arm on the NOISE-FILTERED SEMANTIC subgraph (Tom's catch, 2026-06-18).

endo_ppr_ab.py built P from `SELECT source_id,target_id,weight FROM edges` -- ALL edges, no
aspect filter. But ~71% of edge_relations are structural/usage NOISE (co_accessed 32%,
community_member 14%, related_to 13%, related 6%, co_anchored 3%, emergent_bridge 3%) --
which production EXCLUDES from traversal (co_accessed: d1d1a90c) and which amplify hubs. So
the original PPR diffused over mostly usage-noise -> the 'PPR net-negative' verdict is confounded.

This builds the SEMANTIC subgraph (exclude the `noise` aspect + usage/structural relations;
keep extends/grounds/corrects/implements/refines/depends_on/...) and runs PPR over it vs the
full(noisy) graph vs cosine_cue, alpha-swept, by source. Settles whether the typed semantic
topology -- which is encoder-assigned, NOT cosine-derived, so it can carry signal the flat
embedding lacks (the HippoRAG property) -- helps once the noise is removed.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_ppr_semantic.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tests.isolated_brain import IsolatedBrain
from servers import embedder
from endo_baseline_recall import load_corpus, score_one

TOPK_SEED, K_ITERS = 50, 10

def build_P(triples, idx, N):
    er, ec, ew = [], [], []
    for s, t, w in triples:
        if s in idx and t in idx and s != t:
            i, j = idx[s], idx[t]; wt = float(w) if (w and w > 0) else 1.0
            er += [i, j]; ec += [j, i]; ew += [wt, wt]
    er, ec = np.array(er), np.array(ec); ew = np.array(ew, dtype=np.float64)
    deg = np.zeros(N); np.add.at(deg, er, ew)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return er, ec, ew * dinv[er] * dinv[ec], len(er) // 2

def ppr(seed, er, ec, pval, N, alpha, k=K_ITERS):
    s = seed.astype(np.float64); r = (1 - alpha) * s
    for _ in range(k):
        pr = np.zeros(N); np.add.at(pr, er, pval * r[ec]); r = (1 - alpha) * s + alpha * pr
    return r

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn = env.brain.conn; corpus = load_corpus()
    rows = conn.execute("""SELECT n.id, n.created_at, e.embedding FROM node_enrichments e
        JOIN nodes n ON n.id=e.node_id WHERE e.vector_type='_primary' AND n.archived=0""").fetchall()
    ids = [r[0] for r in rows]; idx = {n: i for i, n in enumerate(ids)}
    created = np.array([r[1] or "" for r in rows]); V = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows]); N = len(ids)

    # noise/usage/structural exclude set -- canonical noise aspect + the known usage/structural relations
    noise = set()
    try:
        na = env.brain.aspects.by_name('noise')
        if na:
            noise = set(na.edge_relations)
    except Exception as e:
        print("aspect lookup failed:", e)
    EXCLUDE = noise | {"co_accessed", "community_member", "co_anchored", "related", "related_to", "emergent_bridge"}
    print("noise aspect relations:", sorted(noise))
    print("excluding (semantic graph):", sorted(EXCLUDE))

    allrows = conn.execute("""SELECT e.source_id, e.target_id, er.relation, COALESCE(er.weight, 1.0)
        FROM edge_relations er JOIN edges e ON e.edge_id = er.edge_id""").fetchall()
    full = [(s, t, w) for s, t, rel, w in allrows]
    sem = [(s, t, w) for s, t, rel, w in allrows if rel not in EXCLUDE]
    print(f"edge_relations: {len(allrows)} total | {len(sem)} semantic kept | {len(allrows)-len(sem)} excluded")

    erf, ecf, pf, nf = build_P(full, idx, N)
    ers, ecs, ps, ns = build_P(sem, idx, N)
    print(f"full(noisy) graph: {nf} undirected edges | SEMANTIC graph: {ns} undirected edges")

    cue_b = embedder.embed_batch([c["query"] for c in corpus], kind="query")
    cache = []
    for c, b in zip(corpus, cue_b):
        elig = created < c["cutoff"]; sc = V @ np.frombuffer(b, dtype=np.float32)
        s = np.where(elig, sc, -np.inf); top = np.argsort(-s)[:TOPK_SEED]
        seed = np.zeros(N); seed[top] = np.clip(sc[top], 0, None); z = seed.sum(); seed = seed / z if z > 0 else seed
        cache.append((c, elig, sc, seed))

    def rank(score, elig, k=120):
        s = np.where(elig, score, -np.inf)
        return [ids[j] for j in np.argsort(-s)[:k] if np.isfinite(s[j])]

    def evalset(rankfn, label):
        h5, r5, h25 = [], [], []; bysrc = {"anchor_turn": [], "operator_msg": []}
        for c, elig, sc, seed in cache:
            m = score_one(rankfn(c, elig, sc, seed), c["gold_essential"], c.get("gold_helpful", []))
            h5.append(m["hit5_ess"]); r5.append(m["recall5_ess"] or 0); h25.append(m["hit25_ess"])
            bysrc[c["source"]].append(m["hit5_ess"])
        print(f"  {label:34s} hit@5 {np.mean(h5):.0%}  hit@25 {np.mean(h25):.0%}  recall@5 {np.mean(r5):.0%}"
              f"  (anchor {np.mean(bysrc['anchor_turn']):.0%} / op {np.mean(bysrc['operator_msg']):.0%})")

    print("\n=== cosine_cue  vs  PPR full(noisy)  vs  PPR SEMANTIC-only ===")
    evalset(lambda c, elig, sc, seed: rank(sc, elig), "cosine_cue (no graph)")
    for a in (0.2, 0.35, 0.5):
        evalset(lambda c, elig, sc, seed, a=a: rank(ppr(seed, erf, ecf, pf, N, a), elig), f"PPR full(noisy) a={a}")
    for a in (0.2, 0.35, 0.5):
        evalset(lambda c, elig, sc, seed, a=a: rank(ppr(seed, ers, ecs, ps, N, a), elig), f"PPR SEMANTIC a={a}")
