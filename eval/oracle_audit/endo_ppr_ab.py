#!/usr/bin/env python3
"""STAGE 3 — PPR/SR arm A: A/B vs baseline on the frozen endo gold corpus.

Tests the two levers the Step-2 lens finding separated (51% of gold is cosine-near
the next_move, NOT the cue):
  (a) the OPERATOR — geodesic graph-reach: does PPR diffusing from the cue-seed
      recover gold that flat cue-cosine can't?  [arm ppr_cue]   -> manual-recall case
  (b) the SEED — multi-anchor: does seeding from richer state help?
        - ppr_cue+traj : realizable (cue + recent session trajectory)  -> in-conversation case
        - cosine_next / ppr_cue+next : the ORACLE ceiling (seed knows the next move)

PPR: r = (1-alpha)s + alpha P r , P = D^-1/2 W D^-1/2 (symmetric-normalized typed-edge
adjacency; symmetric-norm is the principled hub damper). Pure-numpy sparse matvec.
Operator validation: alpha sweep — as alpha->1, ranking must collapse toward hub
centrality (the known failure). Same corpus + same scorer (incl hit@25) as baseline
=> apples-to-apples. Per Tom: every arm broken out by source.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_ppr_ab.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))          # import sibling scorer
from tests.isolated_brain import IsolatedBrain
from servers import embedder
from endo_baseline_recall import load_corpus, score_corpus, score_one, make_baseline_ranker

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "endo_corpus")
TOPK_SEED, K_ITERS = 50, 10
ALPHA = 0.5
sess_of = {c["cand_id"]: c.get("session") for c in json.load(open(f"{OUT}/candidates.json"))}

def seed_from(sc, elig, topk=TOPK_SEED):
    """Top-K positive-cosine seed over ELIGIBLE nodes, normalized to sum 1."""
    s = np.where(elig, sc, -np.inf)
    top = np.argsort(-s)[:topk]
    out = np.zeros(len(sc))
    out[top] = np.clip(sc[top], 0, None)
    z = out.sum()
    return out / z if z > 0 else out

def ppr(seed, rows, cols, pval, N, alpha=ALPHA, k=K_ITERS):
    s = seed.astype(np.float64)
    r = (1 - alpha) * s
    for _ in range(k):
        pr = np.zeros(N)
        np.add.at(pr, rows, pval * r[cols])            # P @ r (sparse matvec)
        r = (1 - alpha) * s + alpha * pr
    return r

def rank_by(score, elig, k=120):
    s = np.where(elig, score, -np.inf)
    order = np.argsort(-s)[:k]
    return [order, s]

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn, logs = env.brain.conn, env.brain.logs_conn
    corpus = load_corpus()
    print(f"corpus: {len(corpus)} cues")

    # node matrix
    nrows = conn.execute(
        """SELECT n.id, n.created_at, e.embedding FROM node_enrichments e
           JOIN nodes n ON n.id = e.node_id
           WHERE e.vector_type='_primary' AND n.archived=0""").fetchall()
    ids = [r[0] for r in nrows]
    idx = {nid: i for i, nid in enumerate(ids)}
    created = np.array([r[1] or "" for r in nrows])
    V = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in nrows])
    N = len(ids)

    # symmetric-normalized weighted adjacency over the node set
    er, ec, ew = [], [], []
    for s, t, w in conn.execute("SELECT source_id, target_id, weight FROM edges"):
        if s in idx and t in idx and s != t:
            i, j = idx[s], idx[t]
            wt = float(w) if (w and w > 0) else 1.0
            er += [i, j]; ec += [j, i]; ew += [wt, wt]          # symmetric
    er, ec, ew = np.array(er), np.array(ec), np.array(ew, dtype=np.float64)
    deg = np.zeros(N); np.add.at(deg, er, ew)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    pval = ew * dinv[er] * dinv[ec]                              # D^-1/2 W D^-1/2
    print(f"graph: {N} nodes, {len(er)//2} undirected edges, "
          f"{int((deg==0).sum())} orphans (no edges)")

    # ── per-cue cache: embeddings + seeds (embed once, reuse across arms/alpha) ──
    def traj_text(cid, cutoff, n=4):
        sid = sess_of.get(cid)
        if not sid:
            return ""
        rows = logs.execute(
            """SELECT metadata FROM trace_events WHERE session_id=? AND ref_type='assistant_message'
               AND created_at < ? ORDER BY created_at DESC LIMIT ?""", (sid, cutoff, n)).fetchall()
        out = []
        for r in rows:
            try:
                out.append(json.loads(r[0]).get("content") or "")
            except Exception:
                pass
        return " \n ".join(reversed(out))[:2000]

    cue_b = embedder.embed_batch([c["query"] for c in corpus], kind="query")
    nxt_b = embedder.embed_batch([c["next_move"] for c in corpus], kind="query")
    trj_b = embedder.embed_batch([traj_text(c["id"], c["cutoff"]) or " " for c in corpus], kind="query")

    cache = {}
    for c, cb, nb, tb in zip(corpus, cue_b, nxt_b, trj_b):
        elig = created < c["cutoff"]
        sc_cue = (V @ np.frombuffer(cb, dtype=np.float32)) if cb else np.zeros(N)
        sc_nxt = (V @ np.frombuffer(nb, dtype=np.float32)) if nb else np.zeros(N)
        sc_trj = (V @ np.frombuffer(tb, dtype=np.float32)) if tb else np.zeros(N)
        cache[c["id"]] = dict(elig=elig, sc_cue=sc_cue, sc_nxt=sc_nxt, sc_trj=sc_trj)

    def ids_from(order_s):
        order, s = order_s
        return [ids[j] for j in order if np.isfinite(s[j])]

    # ── rankers ──
    def r_cosine_cue(c):                       # pure cue cosine (no blend, no graph)
        d = cache[c["id"]]; return ids_from(rank_by(d["sc_cue"], d["elig"]))
    def r_cosine_next(c):                      # ORACLE seed ceiling: cosine on next_move
        d = cache[c["id"]]; return ids_from(rank_by(d["sc_nxt"], d["elig"]))
    def r_ppr_cue(c, alpha=ALPHA):             # (a) operator: geodesic from cue-seed
        d = cache[c["id"]]
        r = ppr(seed_from(d["sc_cue"], d["elig"]), er, ec, pval, N, alpha)
        return ids_from(rank_by(r, d["elig"]))
    def r_ppr_cue_traj(c):                     # (b) realizable: cue + recent trajectory
        d = cache[c["id"]]
        seed = 0.5 * seed_from(d["sc_cue"], d["elig"]) + 0.5 * seed_from(d["sc_trj"], d["elig"])
        r = ppr(seed, er, ec, pval, N)
        return ids_from(rank_by(r, d["elig"]))
    def r_ppr_cue_next(c):                     # (b) oracle: cue + next_move seed, diffused
        d = cache[c["id"]]
        seed = 0.5 * seed_from(d["sc_cue"], d["elig"]) + 0.5 * seed_from(d["sc_nxt"], d["elig"])
        r = ppr(seed, er, ec, pval, N)
        return ids_from(rank_by(r, d["elig"]))

    # ── alpha sweep (operator validation): hit@5 should peak low and decay as alpha->1 ──
    print("\n── alpha sweep (ppr_cue) — expect decay toward hub-centrality as alpha→1 ──")
    for a in (0.2, 0.35, 0.5, 0.7, 0.85, 0.95):
        h5, r5 = [], []
        for c in corpus:
            m = score_one(r_ppr_cue(c, alpha=a), c["gold_essential"], c.get("gold_helpful", []))
            h5.append(m["hit5_ess"]); r5.append(m["recall5_ess"] or 0)
        print(f"  alpha={a:.2f}: hit@5 {np.mean(h5):.0%}  recall@5 {np.mean(r5):.0%}")

    # ── arms (same scorer, same corpus) ──
    base = score_corpus(make_baseline_ranker(env.brain), corpus, arm="BASELINE (brain.recall, cue-blend)")
    cc   = score_corpus(r_cosine_cue,  corpus, arm="cosine_cue (pure cue cosine, no graph)")
    pc   = score_corpus(r_ppr_cue,     corpus, arm="(a) ppr_cue  — operator, geodesic from cue")
    pct  = score_corpus(r_ppr_cue_traj, corpus, arm="(b) ppr_cue+traj — realizable multi-anchor")
    cn   = score_corpus(r_cosine_next, corpus, arm="ORACLE cosine_next (seed = next_move)")
    pcn  = score_corpus(r_ppr_cue_next, corpus, arm="ORACLE ppr_cue+next (seed = cue+next, diffused)")

    # ── delta: which cues each arm newly lands in top-5 vs baseline ──
    def hit5(scored):
        return {s["id"] for s in scored if s["hit5_ess"]}
    b5 = hit5(base)
    for label, sc in (("ppr_cue", pc), ("ppr_cue+traj", pct), ("cosine_next", cn)):
        h = hit5(sc)
        print(f"\n  {label}: newly-hit@5 vs baseline = {len(h - b5)}, lost = {len(b5 - h)}, "
              f"net {len(h) - len(b5):+d}")
