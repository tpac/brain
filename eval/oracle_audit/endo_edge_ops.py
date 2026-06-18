#!/usr/bin/env python3
"""THIN TEST -- edges as per-pair OPERATION-SELECTORS within the retrieved set (Tom's
inhibition/reinforce idea). NOT a global PPR diffusion (that failed). Take the top-K cosine
pool, look at the typed edges AMONG those nodes, and let each edge's aspect pick the op:
  SUPPRESS target  -- correction_improvement / contradiction_conflict (corrects/supersedes/
                      contradicts): the target is stale/opposed -> demote it (the 'blocker')
  REINFORCE target -- extension/explanation/dependency/validation/hierarchical (extends/
                      grounds/depends_on): convergence -> boost the foundational target
  DEDUP            -- similar_to: near-duplicate -> demote the lower-cosine one
Run separately (suppress+dedup / reinforce / both) so we see WHICH bio-operation sharpens.
Scored vs cosine baseline, by source + bucket (buried = Disease A is the test). Reports avg
edges-in-pool (the sparsity check -- typed edges are ~rare).

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_edge_ops.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tests.isolated_brain import IsolatedBrain
from servers import embedder
from endo_baseline_recall import load_corpus, score_one

POOL = 120
SUP, BOOST, DEDUP = 0.3, 0.12, 0.5   # tunable: suppress x0.3, reinforce +0.12, dedup x0.5

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn = env.brain.conn; corpus = load_corpus()
    rows = conn.execute("""SELECT n.id, n.created_at, e.embedding FROM node_enrichments e
        JOIN nodes n ON n.id=e.node_id WHERE e.vector_type='_primary' AND n.archived=0""").fetchall()
    ids = [r[0] for r in rows]; created = np.array([r[1] or "" for r in rows])
    V = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows]); N = len(ids)

    _opcache = {}
    def op_of(rel):
        if rel not in _opcache:
            if rel == "similar_to":
                _opcache[rel] = "dedup"
            else:
                a = env.brain.aspects.by_edge_relation(rel); an = a.name if a else ""
                _opcache[rel] = ("suppress" if an in ("correction_improvement", "contradiction_conflict")
                                 else "reinforce" if an in ("extension_refinement", "explanation_causation",
                                                            "dependency_flow", "validation_evidence",
                                                            "hierarchical_structure")
                                 else None)
        return _opcache[rel]

    cue_b = embedder.embed_batch([c["query"] for c in corpus], kind="query")
    recs = []
    for c, b in zip(corpus, cue_b):
        cv = np.frombuffer(b, dtype=np.float32); elig = created < c["cutoff"]
        sc = np.where(elig, V @ cv, -np.inf)
        order = [j for j in np.argsort(-sc)[:POOL] if np.isfinite(sc[j])]
        poolids = [ids[j] for j in order]
        vals = np.array([sc[j] for j in order]); lo, hi = vals.min(), vals.max()
        nb = {ids[j]: float((sc[j] - lo) / (hi - lo + 1e-9)) for j in order}
        ph = ",".join("?" * len(poolids))
        erows = conn.execute(
            f"""SELECT e.source_id, e.target_id, er.relation FROM edge_relations er
                JOIN edges e ON e.edge_id = er.edge_id
                WHERE e.source_id IN ({ph}) AND e.target_id IN ({ph})""",
            poolids + poolids).fetchall()

        def adjust(use_sup, use_rein):
            s = dict(nb)
            for src, tgt, rel in erows:
                if src not in s or tgt not in s:
                    continue
                o = op_of(rel)
                if o == "suppress" and use_sup:
                    s[tgt] *= SUP
                elif o == "reinforce" and use_rein:
                    s[tgt] += BOOST
                elif o == "dedup" and use_sup:
                    lo_id = tgt if nb[tgt] < nb[src] else src
                    s[lo_id] *= DEDUP
            return [k for k, _ in sorted(s.items(), key=lambda x: -x[1])]

        base_rank = [k for k, _ in sorted(nb.items(), key=lambda x: -x[1])]
        best = min((base_rank.index(g) + 1 for g in c["gold_essential"] if g in nb), default=99999)
        bucket = "hit" if best <= 5 else "buried" if best <= 25 else "far"
        arms = {"cosine": base_rank, "suppress+dedup": adjust(True, False),
                "reinforce": adjust(False, True), "both": adjust(True, True)}
        rec = {"source": c["source"], "bucket": bucket, "n_edges": len(erows), "arms": {}}
        for name, r in arms.items():
            rec["arms"][name] = score_one(r, c["gold_essential"], c.get("gold_helpful", []))
        recs.append(rec)

    def report(rs, title):
        print(f"\n=== {title} (n={len(rs)}) ===  avg typed-edges in pool: {np.mean([r['n_edges'] for r in rs]):.1f}")
        for name in ("cosine", "suppress+dedup", "reinforce", "both"):
            def h(k, sub=None):
                ms = [r["arms"][name][k] for r in rs if sub is None or r["bucket"] == sub]
                return np.mean(ms) if ms else 0.0
            print(f"  {name:16s} hit@5 {h('hit5_ess'):4.0%}  hit@25 {h('hit25_ess'):4.0%} | "
                  f"hit {h('hit5_ess','hit'):3.0%} buried {h('hit5_ess','buried'):4.0%} far {h('hit5_ess','far'):3.0%}")
    report(recs, "ALL")
    report([r for r in recs if r["source"] == "operator_msg"], "OPERATOR")
    report([r for r in recs if r["source"] == "anchor_turn"], "ANCHOR")
