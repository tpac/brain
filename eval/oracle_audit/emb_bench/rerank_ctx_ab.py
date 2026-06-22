#!/usr/bin/env python3
"""Multi-anchor rerank A/B on OPERATOR cues — does enriching the rerank query with the
PREVIOUS ANCHOR TURN flip the operator-cue regression?

Candidate set is unchanged (operator prompt's nomic top-25). Only the RERANK QUERY varies:
  A  baseline           = nomic cosine order (no rerank)
  B  rerank op-only      = rerank query = operator prompt           (reproduces −3 regression)
  C  rerank op+prevanchor = rerank query = prev_anchor_tail + operator prompt   (the test)

Prev anchor turn = latest assistant_message in the cue's session before its ts, read
read-only from brain_logs.db (session+ts joined from candidates.json by cand_id).
fastembed ONNX-CPU (no torch). 512-tok reranker → query budgeted so the doc keeps room.
"""
import json, os, sys, sqlite3
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench

DATA = "/tmp/emb_bench"
EC = "/Users/tpac/brain/eval/oracle_audit/endo_corpus"
K = 25
RERANKER = os.environ.get("RERANKER", "BAAI/bge-reranker-base")
Q_PREV_CHARS, Q_OP_CHARS, DOC_CHARS = 700, 500, 1200
OUT = os.path.join(HERE, "results", "rerank_ctx_ab.json")


def agg(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return sum(v) / len(v) if v else 0.0


def main():
    nodes = json.load(open(f"{DATA}/nodes.json"))
    cues = json.load(open(f"{DATA}/cues.json"))
    cands = {c["cand_id"]: c for c in json.load(open(f"{EC}/candidates.json"))}
    ids = [n["id"] for n in nodes]
    created = np.array([n["created_at"] for n in nodes])
    docs = ["%s %s" % (n["title"], n["content"]) for n in nodes]
    id2doc = dict(zip(ids, docs))
    op_cues = [c for c in cues if c["source"] == "operator_msg"]

    DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
    con = sqlite3.connect(f"file:{os.path.join(DBDIR, 'brain_logs.db')}?mode=ro", uri=True)

    def prev_anchor(session, ts):
        if not session or not ts:
            return ""
        row = con.execute(
            "SELECT metadata FROM trace_events WHERE session_id=? AND ref_type='assistant_message' "
            "AND created_at<? ORDER BY created_at DESC LIMIT 1", (session, ts)).fetchone()
        if not row:
            return ""
        try:
            return json.loads(row[0]).get("content") or ""
        except Exception:
            return ""

    hf, dp, qp, _ = bench.MODELS["nomic_q"]
    print(f"embedding nodes + {len(op_cues)} operator cues with nomic-Q …", flush=True)
    m = bench.FastEmbedBackend(hf)
    node_vecs = m.embed(docs, dp)
    cue_vecs = m.embed([c["query"] for c in op_cues], qp)

    from fastembed.rerank.cross_encoder import TextCrossEncoder
    print(f"loading reranker {RERANKER} …", flush=True)
    ce = TextCrossEncoder(model_name=RERANKER)

    arms = {"A baseline (nomic)": [], "B rerank op-only": [], "C rerank op+prevanchor": []}
    n_noprev = 0
    for i, c in enumerate(op_cues):
        sims = node_vecs @ cue_vecs[i]
        idx = np.where(created <= c["cutoff"])[0]
        order = idx[np.argsort(-sims[idx])][:K]
        top_ids = [ids[j] for j in order]
        ess, helpful = c["gold_essential"], c.get("gold_helpful", [])
        rdocs = [id2doc[t][:DOC_CHARS] for t in top_ids]

        arms["A baseline (nomic)"].append(bench.score_one(top_ids, ess, helpful))

        sB = np.array(list(ce.rerank(c["query"][:Q_OP_CHARS], rdocs)))
        arms["B rerank op-only"].append(
            bench.score_one([top_ids[j] for j in np.argsort(-sB)], ess, helpful))

        cand = cands.get(c["id"], {})
        pa = prev_anchor(cand.get("session"), cand.get("ts"))
        if not pa:
            n_noprev += 1
        q = ((pa[-Q_PREV_CHARS:] + "\n") if pa else "") + c["query"][:Q_OP_CHARS]
        sC = np.array(list(ce.rerank(q, rdocs)))
        arms["C rerank op+prevanchor"].append(
            bench.score_one([top_ids[j] for j in np.argsort(-sC)], ess, helpful))
    con.close()

    print(f"\n{'='*94}\nOPERATOR-cue rerank A/B (n={len(op_cues)}) | {RERANKER} | "
          f"{n_noprev} cues had no prior anchor turn\n{'='*94}")
    for lbl, S in arms.items():
        print(f"  {lbl:26s} hit@1 {agg(S,'hit1_ess'):.0%}  hit@5 {agg(S,'hit5_ess'):.0%}  "
              f"hit@25 {agg(S,'hit25_ess'):.0%}  recall@5 {agg(S,'recall5_ess'):.0%}  nDCG@5 {agg(S,'ndcg5'):.2f}")
    print("  (C vs B = pure effect of adding the prior anchor turn to the rerank query)")
    json.dump({k: v for k, v in arms.items()} | {"reranker": RERANKER, "n_noprev": n_noprev},
              open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
