#!/usr/bin/env python3
"""Test the underutilized `question` field as a recall signal (Tom's idea).

Every node carries a `question` vector ("what question does this node answer") in
node_enrichments -- but recall scores against `_primary` (title+content), NOT question.
Hypothesis: for cue-far (Disease B) cases -- cue is the QUESTION, gold is the ANSWER -- the
gold's `question` field is closer to the cue than its content, because the encoder already
distilled what the node answers. So cue<->question cosine could be the REALIZABLE Disease-B
bridge the analogical-followup arm wasn't (no future needed; the vector already exists).

Compares cosine(cue,_primary) vs cosine(cue,question) vs RRF, scored against gold, broken
out by source AND by baseline bucket (hit/buried/far). The FAR bucket is the test.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_question_field.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tests.isolated_brain import IsolatedBrain
from servers import embedder
from endo_baseline_recall import load_corpus, score_one

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn = env.brain.conn; corpus = load_corpus()
    rows = conn.execute("""SELECT n.id, n.created_at, p.embedding, q.embedding
        FROM nodes n
        JOIN node_enrichments p ON p.node_id=n.id AND p.vector_type='_primary'
        JOIN node_enrichments q ON q.node_id=n.id AND q.vector_type='question'
        WHERE n.archived=0""").fetchall()
    ids = [r[0] for r in rows]; pos = {nid: i for i, nid in enumerate(ids)}
    created = np.array([r[1] or "" for r in rows])
    Vp = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    Vq = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    N = len(ids)
    print(f"nodes with both _primary + question vectors: {N}")

    cue_b = embedder.embed_batch([c["query"] for c in corpus], kind="query")

    def rank(score, elig, k=120):
        s = np.where(elig, score, -np.inf)
        return [ids[j] for j in np.argsort(-s)[:k] if np.isfinite(s[j])]

    def rrf(ra, rb, kk=60):
        sc = {}
        for r, nid in enumerate(ra): sc[nid] = sc.get(nid, 0) + 1.0/(kk+r+1)
        for r, nid in enumerate(rb): sc[nid] = sc.get(nid, 0) + 1.0/(kk+r+1)
        return [nid for nid, _ in sorted(sc.items(), key=lambda x: -x[1])[:120]]

    rec = []
    for c, b in zip(corpus, cue_b):
        cv = np.frombuffer(b, dtype=np.float32); elig = created < c["cutoff"]
        scp = np.where(elig, Vp @ cv, -np.inf); scq = np.where(elig, Vq @ cv, -np.inf)
        rp, rq = rank(scp, elig), rank(scq, elig); rr = rrf(rp, rq)
        order_p = np.argsort(-scp)
        rankp = {ids[order_p[i]]: i+1 for i in range(len(order_p))}
        bestp = min((rankp[g] for g in c["gold_essential"] if g in rankp), default=99999)
        bucket = "hit" if bestp <= 5 else "buried" if bestp <= 25 else "far"
        rec.append(dict(src=c["source"], bucket=bucket,
                        mp=score_one(rp, c["gold_essential"], c.get("gold_helpful", [])),
                        mq=score_one(rq, c["gold_essential"], c.get("gold_helpful", [])),
                        mr=score_one(rr, c["gold_essential"], c.get("gold_helpful", [])),
                        gp_cos=[float(Vp[pos[g]] @ cv) for g in c["gold_essential"] if g in pos],
                        gq_cos=[float(Vq[pos[g]] @ cv) for g in c["gold_essential"] if g in pos]))

    def agg(rows, arm): return np.mean([r[arm]["hit5_ess"] for r in rows]) if rows else 0
    def agg25(rows, arm): return np.mean([r[arm]["hit25_ess"] for r in rows]) if rows else 0
    def show(rows, label):
        if rows:
            print(f"  {label:18s} n={len(rows):2d} | hit@5  primary {agg(rows,'mp'):.0%}  question {agg(rows,'mq'):.0%}  "
                  f"RRF {agg(rows,'mr'):.0%}  | hit@25  primary {agg25(rows,'mp'):.0%}  question {agg25(rows,'mq'):.0%}")

    print("\n=== cosine(cue, _primary)  vs  cosine(cue, question)  vs  RRF(both) ===")
    show(rec, "ALL")
    for s in ("anchor_turn", "operator_msg"):
        show([r for r in rec if r["src"] == s], s)
    for bk in ("hit", "buried", "far"):
        show([r for r in rec if r["bucket"] == bk], f"bucket={bk}")

    allgp = [x for r in rec for x in r["gp_cos"]]; allgq = [x for r in rec for x in r["gq_cos"]]
    print(f"\n  gold↔cue cosine (ALL):  primary {np.mean(allgp):.3f}   question {np.mean(allgq):.3f}")
    far = [r for r in rec if r["bucket"] == "far"]
    fgp = [x for r in far for x in r["gp_cos"]]; fgq = [x for r in far for x in r["gq_cos"]]
    if fgp:
        print(f"  gold↔cue cosine (FAR):  primary {np.mean(fgp):.3f}   question {np.mean(fgq):.3f}"
              f"   (question higher ⇒ the `question` field bridges Disease B)")
