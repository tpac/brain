#!/usr/bin/env python3
"""MISS ANALYSIS — where did the gold rank under pure cue-cosine, and WHY (the math)?

Per essential-gold node: its rank + cosine under cue-cosine, the cosine of the 5th-ranked
node (the bar it had to beat), the gap, and whether FTS (lexical) would have found it.
Buckets the misses and surfaces the flat-space signature: gold that is RELEVANT and
cosine-near (~0.6) but indistinguishable from non-gold noise (~0.6-0.65), so it can't
rank top-5 no matter the ranking algorithm.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_miss_analysis.py
"""
import json, os, sys
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.isolated_brain import IsolatedBrain
from servers import embedder

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "endo_corpus")
corpus = json.load(open(f"{OUT}/endo_gold_corpus.json"))

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn = env.brain.conn
    rows = conn.execute("""SELECT n.id, n.title, n.created_at, e.embedding FROM node_enrichments e
        JOIN nodes n ON n.id=e.node_id WHERE e.vector_type='_primary' AND n.archived=0""").fetchall()
    ids = [r[0] for r in rows]; idx = {n: i for i, n in enumerate(ids)}
    titles = {r[0]: (r[1] or "")[:68] for r in rows}
    created = np.array([r[2] or "" for r in rows])
    V = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])

    cue_b = embedder.embed_batch([c["query"] for c in corpus], kind="query")
    recs = []
    for c, cb in zip(corpus, cue_b):
        elig = created < c["cutoff"]
        cv = np.frombuffer(cb, dtype=np.float32)
        sc = np.where(elig, V @ cv, -np.inf)
        order = np.argsort(-sc)
        rank_of = {ids[order[r]]: r + 1 for r in range(len(order)) if np.isfinite(sc[order[r]])}
        cos_of = {ids[j]: float(sc[j]) for j in range(len(ids)) if np.isfinite(sc[j])}
        rank5 = float(sc[order[4]])
        fts = set(env.brain._fts.search(c["query"], limit=25))
        for g in c["gold_essential"]:
            if g not in idx:
                continue
            recs.append(dict(cue=c["id"], qt=c["query_type"], gold=g,
                             rank=rank_of.get(g, 99999), gcos=cos_of.get(g, 0.0),
                             rank5=rank5, gap=rank5 - cos_of.get(g, 0.0),
                             fts=g in fts,
                             top=[(titles[ids[order[r]]], float(sc[order[r]])) for r in range(3)]))

    def bucket(r):
        return "top5(HIT)" if r <= 5 else "buried 6-25" if r <= 25 else "deep 26-120" if r <= 120 else "far 120+"
    bc = Counter(bucket(r["rank"]) for r in recs)
    n = len(recs)
    print(f"=== {n} essential-gold instances across {len(corpus)} cues — cue-cosine ranks ===")
    for k in ("top5(HIT)", "buried 6-25", "deep 26-120", "far 120+"):
        print(f"  {k:14s} {bc.get(k,0):3d}  ({100*bc.get(k,0)//n}%)")

    miss = [r for r in recs if r["rank"] > 5]
    gc = sorted(r["gcos"] for r in miss)
    print(f"\n-- of {len(miss)} MISSES (gold not in top-5) --")
    print(f"  gold cosine: min/median/max = {gc[0]:.3f} / {gc[len(gc)//2]:.3f} / {gc[-1]:.3f}")
    print(f"  bar (rank-5 cosine), median = {sorted(r['rank5'] for r in miss)[len(miss)//2]:.3f}")
    near = sum(1 for r in miss if r["gap"] <= 0.03)
    print(f"  FLAT-SPACE SIGNATURE — gold within 0.03 cosine of the top-5 bar: "
          f"{near}/{len(miss)} ({100*near//len(miss)}%)  (relevant, but cosine can't rank it)")
    big = sum(1 for r in miss if r["gap"] > 0.06)
    print(f"  gold >0.06 below the bar (genuinely cosine-far): {big}/{len(miss)} ({100*big//len(miss)}%)")
    ftss = sum(1 for r in miss if r["fts"])
    print(f"  missed gold that FTS (lexical) WOULD have found: {ftss}/{len(miss)} ({100*ftss//len(miss)}%)")

    print("\n-- sample misses: gold (rank, cos) vs the top-3 that beat it (the flat-space picture) --")
    for r in sorted(miss, key=lambda x: x["gap"])[:8]:
        print(f"\n[{r['cue']}] {r['qt']}  gold rank={r['rank']} cos={r['gcos']:.3f}  "
              f"(bar {r['rank5']:.3f}, gap {r['gap']:+.3f})  fts_would_find={r['fts']}")
        print(f"   GOLD: {titles[r['gold']]}")
        for t, s in r["top"]:
            print(f"   top:  {s:.3f}  {t}")
