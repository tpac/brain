#!/usr/bin/env python3
"""STAGE 3c — PREDICTIVE-EPISODIC arm (Tom's idea): seed from the FUTURE of similar PAST moments.

The next-move ORACLE (seed from the actual next turn) scored ~2x cue-cosine but isn't
realizable — we don't have the future. Tom's insight: we DO have the future of PAST
episodes. Find past moments similar to NOW, then seed from what came AFTER them — the
followup of a past analog predicts the followup of now.

  hop1: tcos_t      = cosine(cue, past_trace_t)            # past moments like now
  hop2: ncos(f_t,n) = cosine(FOLLOWUP(trace_t), node)      # nodes near that moment's FUTURE
  node score = AGG_t [ tcos_t * ncos(f_t, n) ]             # both t and f_t must be < cutoff (no leak)

Realizable, non-parametric next-move predictor via episodic analogy (the SR in spirit).
Unlike content-episodic (which collapsed into cosine by transitivity), the followup f_t is
a DIFFERENT turn than the cue/trace, so it can point where cue-cosine can't reach.

Floor = cosine_cue (~22%); ceiling = oracle cosine_next (~42%, lens-primed). Where this
lands shows how much of the oracle headroom analogical prediction captures.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_followup_arm.py [T]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tests.isolated_brain import IsolatedBrain
from servers import embedder
from endo_baseline_recall import load_corpus, score_corpus

T = int(sys.argv[1]) if len(sys.argv) > 1 else 20
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "endo_corpus")

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn, logs = env.brain.conn, env.brain.logs_conn
    corpus = load_corpus()

    nrows = conn.execute("""SELECT n.id, n.created_at, e.embedding FROM node_enrichments e
        JOIN nodes n ON n.id=e.node_id WHERE e.vector_type='_primary' AND n.archived=0""").fetchall()
    ids = [r[0] for r in nrows]; ncreated = np.array([r[1] or "" for r in nrows])
    V = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in nrows]); N = len(ids)

    # S0 traces ordered per session -> followup mapping (next trace in same session)
    trows = logs.execute("""SELECT te.vector, ev.created_at, ev.session_id FROM trace_embeddings te
        JOIN trace_events ev ON ev.id=te.trace_id
        WHERE ev.scale='s0' AND ev.ref_type IN ('user_message','assistant_message') AND te.vector IS NOT NULL
        ORDER BY ev.session_id, ev.created_at""").fetchall()
    TV = np.vstack([np.frombuffer(r[0], dtype=np.float32) for r in trows])
    tcreated = np.array([r[1] or "" for r in trows]); tsess = [r[2] for r in trows]
    fidx = np.full(len(trows), -1, dtype=int)
    for i in range(len(trows) - 1):
        if tsess[i] == tsess[i + 1]:
            fidx[i] = i + 1
    FV = np.zeros_like(TV)
    fcr = np.array([""] * len(trows), dtype=object)
    for i in range(len(trows)):
        if fidx[i] >= 0:
            FV[i] = TV[fidx[i]]; fcr[i] = tcreated[fidx[i]]
    has_f = fidx >= 0
    print(f"nodes={N} s0-traces={len(TV)} with-followup={int(has_f.sum())} corpus={len(corpus)} T={T}")

    cue_b = embedder.embed_batch([c["query"] for c in corpus], kind="query")
    nxt_b = embedder.embed_batch([c["next_move"] for c in corpus], kind="query")   # oracle ref
    cache = {}
    for c, cb, nb in zip(corpus, cue_b, nxt_b):
        elig = ncreated < c["cutoff"]
        cv = np.frombuffer(cb, dtype=np.float32) if cb else np.zeros(V.shape[1], np.float32)
        nv = np.frombuffer(nb, dtype=np.float32) if nb else np.zeros(V.shape[1], np.float32)
        sc_cue = V @ cv; sc_next = V @ nv
        tcos = TV @ cv
        valid = (tcreated < c["cutoff"]) & has_f & (fcr < c["cutoff"]) & (fcr != "")
        tcos = np.where(valid, tcos, -np.inf)
        top = np.argsort(-tcos)[:T]; top = top[np.isfinite(tcos[top])]
        if len(top):
            NVf = V @ FV[top].T                      # nodes near the FOLLOWUPS (the past's future)
            w = NVf * tcos[top]
            fu_sum = w.sum(axis=1); fu_max = w.max(axis=1)
        else:
            fu_sum = fu_max = np.zeros(N)
        cache[c["id"]] = dict(elig=elig, sc_cue=sc_cue, sc_next=sc_next, fu_sum=fu_sum, fu_max=fu_max)

    def rb(score, elig, k=120):
        s = np.where(elig, score, -np.inf)
        return [ids[j] for j in np.argsort(-s)[:k] if np.isfinite(s[j])]

    def r_cue(c):   d = cache[c["id"]]; return rb(d["sc_cue"], d["elig"])
    def r_fusum(c): d = cache[c["id"]]; return rb(d["fu_sum"], d["elig"])
    def r_fumax(c): d = cache[c["id"]]; return rb(d["fu_max"], d["elig"])
    def r_next(c):  d = cache[c["id"]]; return rb(d["sc_next"], d["elig"])
    def r_rrf(c, kk=60):
        d = cache[c["id"]]; sc = {}
        for rank, nid in enumerate(rb(d["sc_cue"], d["elig"])): sc[nid] = sc.get(nid, 0) + 1.0 / (kk + rank + 1)
        for rank, nid in enumerate(rb(d["fu_sum"], d["elig"])): sc[nid] = sc.get(nid, 0) + 1.0 / (kk + rank + 1)
        return [nid for nid, _ in sorted(sc.items(), key=lambda x: -x[1])[:120]]

    cc = score_corpus(r_cue,   corpus, arm="cosine_cue (FLOOR)")
    fs = score_corpus(r_fusum, corpus, arm="followup_SUM (predict via past analogs)")
    fm = score_corpus(r_fumax, corpus, arm="followup_MAX")
    rr = score_corpus(r_rrf,   corpus, arm="RRF(cue + followup_SUM)")
    cn = score_corpus(r_next,  corpus, arm="cosine_next (ORACLE ceiling, lens-primed)")

    def h5(s):
        return {x["id"] for x in s if x["hit5_ess"]}
    base = h5(cc)
    print()
    for label, sc in (("followup_SUM", fs), ("followup_MAX", fm), ("RRF", rr)):
        h = h5(sc)
        print(f"  {label:14s} vs cosine_cue: newly-hit@5 {len(h - base)}, lost {len(base - h)}, net {len(h) - len(base):+d}")
