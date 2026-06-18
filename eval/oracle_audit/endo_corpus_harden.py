#!/usr/bin/env python3
"""STAGE 2.5 — HARDEN the corpus/measurement before trusting the seed-lever result.

Tom: bad experience trusting results then finding the test was corrupted/wrong.
Three checks, all from EXISTING data (+ cheap cosine, NO recall, NO API spend):

  1. MECHANICAL INTEGRITY — is the test corrupted? Every essential-gold node must be
     created strictly BEFORE the cue (no cutoff leak), non-archived (recall could
     return it), and present in the db. Catches the §12c artifact-1 class directly.
  2. DE-BIAS the seed lever — the oracle was lens-primed (gold partly DISCOVERED via
     the cos_next lens). Re-measure cue-vs-next cosine hit@5 on gold NOT discovered
     via cos_next. If next still beats cue there, the lever is real, not priming.
  3. HIGH-CONFIDENCE slice — does the lever hold on conf=high cues (least
     hindsight-biased)?

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_corpus_harden.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.isolated_brain import IsolatedBrain
from servers import embedder

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "endo_corpus")
corpus = json.load(open(f"{OUT}/endo_gold_corpus.json"))

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn = env.brain.conn
    rows = conn.execute(
        """SELECT n.id, n.created_at, n.archived, e.embedding
           FROM nodes n LEFT JOIN node_enrichments e
             ON e.node_id = n.id AND e.vector_type='_primary'""").fetchall()
    info = {r[0]: {"created": r[1] or "", "archived": r[2], "emb": r[3]} for r in rows}
    nz = [(r[0], r[1] or "", r[3]) for r in rows if r[2] == 0 and r[3]]
    ids = [x[0] for x in nz]
    created = np.array([x[1] for x in nz])
    V = np.vstack([np.frombuffer(x[2], dtype=np.float32) for x in nz])

    # ---- 1. mechanical integrity ----
    print("== 1. MECHANICAL INTEGRITY (is the test corrupted?) ==")
    leak = arch = missing = 0
    leak_ex = []
    for c in corpus:
        for g in c["gold_essential"]:
            gi = info.get(g)
            if not gi:
                missing += 1; continue
            if gi["created"] >= c["cutoff"]:
                leak += 1; leak_ex.append((c["id"], g, gi["created"][:10], c["cutoff"][:10]))
            if gi["archived"]:
                arch += 1
    print(f"  essential-gold created >= cutoff (LEAK): {leak}" + (f"  e.g. {leak_ex[:3]}" if leak else ""))
    print(f"  essential-gold archived (unrecallable):  {arch}")
    print(f"  essential-gold missing from db:          {missing}")
    print(f"  -> {'CLEAN' if leak == arch == missing == 0 else '*** PROBLEM ***'}")

    # ---- cosine arms (cue vs next), eligible-masked ----
    cb = embedder.embed_batch([c["query"] for c in corpus], kind="query")
    nb = embedder.embed_batch([c["next_move"] for c in corpus], kind="query")

    def rank(blob, elig, k=25):
        sc = V @ np.frombuffer(blob, dtype=np.float32)
        s = np.where(elig, sc, -np.inf)
        return [ids[j] for j in np.argsort(-s)[:k]]

    per = []
    for c, cbl, nbl in zip(corpus, cb, nb):
        elig = created < c["cutoff"]
        per.append({"c": c, "cue": rank(cbl, elig), "next": rank(nbl, elig)})

    def h5(ranked, gold):
        return 1 if set(ranked[:5]) & set(gold) else 0

    def measure(items, goldfn, label):
        cue = [h5(p["cue"], goldfn(p["c"])) for p in items if goldfn(p["c"])]
        nxt = [h5(p["next"], goldfn(p["c"])) for p in items if goldfn(p["c"])]
        if cue:
            print(f"  {label:44s} n={len(cue):3d} | cue {np.mean(cue):.0%}  next {np.mean(nxt):.0%}  "
                  f"(next-cue {np.mean(nxt)-np.mean(cue):+.0%})")

    print("\n== 2. DE-BIAS: cue-vs-next hit@5 partitioned by gold discovery lens ==")
    measure(per, lambda c: c["gold_essential"], "ALL essential gold")
    measure(per, lambda c: [g for g in c["gold_essential"] if "cos_next" in c["gold_lens"].get(g, [])],
            "gold found via cos_next (PRIMED for next)")
    measure(per, lambda c: [g for g in c["gold_essential"] if "cos_next" not in c["gold_lens"].get(g, [])],
            "gold NOT via cos_next (DE-BIASED)")

    print("\n== 3. HIGH-CONFIDENCE slice ==")
    hi = [p for p in per if p["c"]["confidence"] == "high"]
    measure(hi, lambda c: c["gold_essential"], "conf=high, all essential gold")
    measure(hi, lambda c: [g for g in c["gold_essential"] if "cos_next" not in c["gold_lens"].get(g, [])],
            "conf=high, de-biased gold")
    print(f"\n  (conf split: high={sum(1 for c in corpus if c['confidence']=='high')}, "
          f"medium={sum(1 for c in corpus if c['confidence']=='medium')})")
