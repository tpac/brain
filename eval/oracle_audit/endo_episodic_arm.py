#!/usr/bin/env python3
"""STAGE 3b — EPISODIC arm: cue -> similar past S0 traces -> their nodes, with the
episodic similarity score PRESERVED into the ranking (Tom's design question).

Two-hop, score-preserving:
  hop1: tcos_t   = cosine(cue, trace_t)      # how cue-similar each PAST moment is
  hop2: ncos(t,n)= cosine(trace_t, node_n)   # how trace-relevant each node is
  node episodic score = AGG_t [ tcos_t * ncos(t,n) ]   # the episodic score is NOT dropped

Both aggregations, because they encode DIFFERENT signals:
  MAX — strongest single moment wins (what the dormant trace-chain lane does)
  SUM — convergence: a node pointed at by MANY cue-similar moments scores higher
        (the signal MAX drops — the episodic analog of PPR convergence-as-vote)

Noise control is the score itself: a weakly-cue-similar moment (low tcos) contributes
proportionally little mass, so its nodes can't pollute. Top-T trace cutoff bounds it.

Clean test: episodic was NOT a gold-discovery lens, so scoring on the current corpus
is unbiased for this arm. Same scorer/corpus as baseline (incl hit@25), by source.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_episodic_arm.py [T]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tests.isolated_brain import IsolatedBrain
from servers import embedder
from endo_baseline_recall import load_corpus, score_corpus

T = int(sys.argv[1]) if len(sys.argv) > 1 else 20      # top past traces per cue
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "endo_corpus")

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn, logs = env.brain.conn, env.brain.logs_conn
    corpus = load_corpus()

    # node matrix (non-archived, primary vector)
    nrows = conn.execute(
        """SELECT n.id, n.created_at, e.embedding FROM node_enrichments e
           JOIN nodes n ON n.id=e.node_id WHERE e.vector_type='_primary' AND n.archived=0""").fetchall()
    ids = [r[0] for r in nrows]
    ncreated = np.array([r[1] or "" for r in nrows])
    V = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in nrows])
    N = len(ids)

    # S0 trace vectors (the episodic substrate) — user/assistant only (tool_result is recall-echo poison)
    trows = logs.execute(
        """SELECT te.vector, ev.created_at FROM trace_embeddings te
           JOIN trace_events ev ON ev.id=te.trace_id
           WHERE ev.scale='s0' AND ev.ref_type IN ('user_message','assistant_message')
             AND te.vector IS NOT NULL""").fetchall()
    TV = np.vstack([np.frombuffer(r[0], dtype=np.float32) for r in trows])
    tcreated = np.array([r[1] or "" for r in trows])
    print(f"nodes={N}  s0-traces={len(TV)}  corpus={len(corpus)}  T={T}")

    # precompute per-cue score vectors (cue-cosine + episodic MAX/SUM)
    cue_b = embedder.embed_batch([c["query"] for c in corpus], kind="query")
    cache = {}
    for c, cb in zip(corpus, cue_b):
        elig = ncreated < c["cutoff"]
        cv = np.frombuffer(cb, dtype=np.float32) if cb else np.zeros(V.shape[1], np.float32)
        sc_cue = V @ cv
        tcos = TV @ cv                                       # hop1: cue -> traces
        tcos = np.where(tcreated < c["cutoff"], tcos, -np.inf)   # only PAST moments
        top = np.argsort(-tcos)[:T]
        top = top[np.isfinite(tcos[top])]
        if len(top):
            NV = V @ TV[top].T                               # hop2: (N, T) node-trace cosines
            w = NV * tcos[top]                               # preserve the episodic score (weight each column)
            ep_max = w.max(axis=1)
            ep_sum = w.sum(axis=1)
        else:
            ep_max = ep_sum = np.zeros(N)
        cache[c["id"]] = dict(elig=elig, sc_cue=sc_cue, ep_max=ep_max, ep_sum=ep_sum)

    def rank_by(score, elig, k=120):
        s = np.where(elig, score, -np.inf)
        return [ids[j] for j in np.argsort(-s)[:k] if np.isfinite(s[j])]

    def r_cue(c):   d = cache[c["id"]]; return rank_by(d["sc_cue"], d["elig"])
    def r_epmax(c): d = cache[c["id"]]; return rank_by(d["ep_max"], d["elig"])
    def r_epsum(c): d = cache[c["id"]]; return rank_by(d["ep_sum"], d["elig"])
    def r_rrf(c, kk=60):                                      # RRF fuse cue-cosine + episodic_sum
        d = cache[c["id"]]
        sc = {}
        for rank, nid in enumerate(rank_by(d["sc_cue"], d["elig"])):
            sc[nid] = sc.get(nid, 0) + 1.0 / (kk + rank + 1)
        for rank, nid in enumerate(rank_by(d["ep_sum"], d["elig"])):
            sc[nid] = sc.get(nid, 0) + 1.0 / (kk + rank + 1)
        return [nid for nid, _ in sorted(sc.items(), key=lambda x: -x[1])[:120]]

    cc = score_corpus(r_cue,   corpus, arm="cosine_cue (semantic baseline)")
    em = score_corpus(r_epmax, corpus, arm="episodic_MAX (strongest moment)")
    es = score_corpus(r_epsum, corpus, arm="episodic_SUM (convergence over moments)")
    rr = score_corpus(r_rrf,   corpus, arm="RRF(cosine_cue + episodic_SUM)")

    # delta: does episodic find gold cosine_cue misses?
    def hit5(scored):
        return {s["id"] for s in scored if s["hit5_ess"]}
    base5 = hit5(cc)
    print()
    for label, sc in (("episodic_MAX", em), ("episodic_SUM", es), ("RRF", rr)):
        h = hit5(sc)
        print(f"  {label:14s} vs cosine_cue: newly-hit@5 {len(h - base5)}, lost {len(base5 - h)}, "
              f"net {len(h) - len(base5):+d}")

    # cleanest test of a NEW lens: gold the SEMANTIC lenses (cos_cue/cos_next) missed -> fts-only gold
    def h5(ranked, gold):
        return 1 if set(ranked[:5]) & set(gold) else 0
    deb = []
    for c in corpus:
        g = [x for x in c["gold_essential"]
             if "cos_cue" not in c["gold_lens"].get(x, []) and "cos_next" not in c["gold_lens"].get(x, [])]
        if g:
            deb.append((c, g))
    if deb:
        ep = np.mean([h5(r_epsum(c), g) for c, g in deb])
        cu = np.mean([h5(r_cue(c), g) for c, g in deb])
        print(f"\n  semantic-lens-missed gold (fts-only, n={len(deb)}): "
              f"episodic_SUM hit@5 {ep:.0%}  vs  cosine_cue {cu:.0%}")
