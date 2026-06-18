#!/usr/bin/env python3
"""LANE-FUSION lab (Tom's framework): every signal is a LANE (a per-node score over the
same node set); fuse them additively / multiplicatively / hard-AND. Graph-LEVEL fusion
(combine whole rankings) -- the cheap probe before node/edge-level signed PPR.

Simple test to get LEADS: does gating (multiplication / AND) sharpen where additive cosine
can't -- especially Disease A (the 'buried' bucket: gold in pool, ranked 6-25)?

Lanes are pluggable (extend LANES): cos = primary cosine, fts = lexical match (the
non-cosine signal), ques = question-field cosine. Each normalized [0,1] per cue over
eligible nodes. Fusions are pluggable (extend FUSIONS). Scored by source + by primary-bucket.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_lane_fusion.py
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
    rows = conn.execute("""SELECT n.id, n.created_at, p.embedding, q.embedding FROM nodes n
        JOIN node_enrichments p ON p.node_id=n.id AND p.vector_type='_primary'
        JOIN node_enrichments q ON q.node_id=n.id AND q.vector_type='question'
        WHERE n.archived=0""").fetchall()
    ids = [r[0] for r in rows]; pos = {n: i for i, n in enumerate(ids)}
    created = np.array([r[1] or "" for r in rows]); N = len(ids)
    Vp = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    Vq = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])

    # ---- ASPECT INVENTORY: documents what endo_ppr_semantic.py's GRAPH filter excludes.
    # NOTE: THIS lane-fusion lab uses NO edges (cos/fts/ques are score-lanes, not graphs),
    # so NO aspect filter applies here. This only documents the PPR-run's graph filter. ----
    PPR_EXCLUDE = set(env.brain.aspects.by_name('noise').edge_relations) | {
        "co_accessed", "community_member", "co_anchored", "related", "related_to", "emergent_bridge"}
    print("=== aspect inventory (top edge relations -> aspect, PPR-graph exclude/keep) ===")
    for rel, n in conn.execute("SELECT relation, COUNT(*) FROM edge_relations GROUP BY relation "
                               "ORDER BY COUNT(*) DESC LIMIT 22"):
        asp = env.brain.aspects.by_edge_relation(rel)
        print(f"  {rel:22s} {n:6d}  aspect={(asp.name if asp else '(unclassified)'):22s} "
              f"{'EXCLUDED' if rel in PPR_EXCLUDE else 'kept'}")
    try:
        names = [a.get('name') if isinstance(a, dict) else getattr(a, 'name', str(a))
                 for a in env.brain.aspects.all_with_counts()]
        print(f"  ALL aspects in taxonomy: {names}")
    except Exception as e:
        print("  aspect list error:", e)
    print(f"  PPR-graph EXCLUDE set: {sorted(PPR_EXCLUDE)}\n")

    cue_b = embedder.embed_batch([c["query"] for c in corpus], kind="query")

    def norm(x, elig):                       # min-max over eligible -> [0,1]; ineligible -> -1
        v = np.where(elig, x, np.nan); lo = np.nanmin(v); hi = np.nanmax(v)
        n = (v - lo) / (hi - lo + 1e-9)
        return np.where(elig, n, -1.0)

    # ---- LANES (pluggable) ----
    percue = []
    for c, b in zip(corpus, cue_b):
        elig = created < c["cutoff"]; cv = np.frombuffer(b, dtype=np.float32)
        cos = Vp @ cv; quesv = Vq @ cv
        fts = np.zeros(N)
        for r, nid in enumerate(env.brain._fts.search(c["query"], limit=50)):
            if nid in pos:
                fts[pos[nid]] = 1.0 / (r + 1)
        lanes = dict(cos=norm(cos, elig), fts=norm(fts, elig), ques=norm(quesv, elig))
        order = np.argsort(-np.where(elig, cos, -np.inf))
        rk = {ids[order[i]]: i + 1 for i in range(len(order))}
        best = min((rk[g] for g in c["gold_essential"] if g in rk), default=99999)
        bucket = "hit" if best <= 5 else "buried" if best <= 25 else "far"
        percue.append((c, elig, lanes, bucket))

    def P(x):                                # clamp lane to [0,1] (ineligible -1 -> 0)
        return np.maximum(x, 0.0)

    # ---- FUSIONS (pluggable) ----
    FUSIONS = {
        "cos (baseline)":            lambda L: L["cos"],
        "add  cos+fts":              lambda L: L["cos"] + P(L["fts"]),
        "add  cos+fts+ques":         lambda L: L["cos"] + P(L["fts"]) + P(L["ques"]),
        "MULT cos*(1+fts)":          lambda L: P(L["cos"]) * (1 + P(L["fts"])),
        "MULT cos*(1+fts)*(1+ques)": lambda L: P(L["cos"]) * (1 + P(L["fts"])) * (1 + P(L["ques"])),
        "AND  cos*fts (hard gate)":  lambda L: P(L["cos"]) * P(L["fts"]),
    }

    def rank_ids(score, elig, k=120):
        s = np.where(elig, score, -np.inf)
        return [ids[j] for j in np.argsort(-s)[:k] if np.isfinite(s[j]) and s[j] > -1]

    recs = []
    for c, elig, lanes, bucket in percue:
        arms = {name: score_one(rank_ids(fn(lanes), elig), c["gold_essential"], c.get("gold_helpful", []))
                for name, fn in FUSIONS.items()}
        recs.append({"source": c["source"], "bucket": bucket, "arms": arms})

    def report(rows, title):
        print(f"\n=== {title}  (n={len(rows)}) ===")
        print(f"  {'fusion':28s} hit@5 hit@25 | hit  buried  far")
        for name in FUSIONS:
            def h(k, sub=None):
                ms = [r["arms"][name][k] for r in rows if sub is None or r["bucket"] == sub]
                return np.mean(ms) if ms else 0.0
            print(f"  {name:28s} {h('hit5_ess'):4.0%}  {h('hit25_ess'):4.0%} | "
                  f"{h('hit5_ess','hit'):4.0%} {h('hit5_ess','buried'):5.0%} {h('hit5_ess','far'):4.0%}")

    report(recs, "ALL")
    report([r for r in recs if r["source"] == "operator_msg"], "OPERATOR only")
    report([r for r in recs if r["source"] == "anchor_turn"], "ANCHOR only")
    nb = {s: {b: sum(1 for r in recs if r["source"] == s and r["bucket"] == b) for b in ("hit", "buried", "far")}
          for s in ("operator_msg", "anchor_turn")}
    print(f"\n  bucket counts: operator {nb['operator_msg']} | anchor {nb['anchor_turn']}")
    print("  LEAD: buried = Disease A; far = Disease B.")
