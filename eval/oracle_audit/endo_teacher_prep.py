#!/usr/bin/env python3
"""STAGE 1b-ii — teacher PREP: pick the teacher set + build an UNBIASED candidate
union per cue. Offline, no API spend (local embedder + FTS only).

Selection: balanced anchor_turn / operator_msg, stratified across aged-coverage
(harvest the high stratum + a mid/low calibration tail) so the teacher verdicts
ALSO validate that coverage predicts endo-worthiness.

Candidate union (decoupled from the baseline cosine-on-cue path so gold isn't
gold-by-construction): cosine-on-cue ∪ cosine-on-next_move ∪ FTS-on-cue, all
filtered to nodes created strictly before the cue's cutoff. The teacher picks
gold from this union; Step-2 baseline measures cosine-on-cue as a SUBSET → an
honest miss. lens/cov tags are kept for analysis but HIDDEN from the teacher.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_teacher_prep.py [n_per_source]
"""
import json, os, sys
import numpy as np
from datetime import date
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.isolated_brain import IsolatedBrain
from servers import embedder

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "endo_corpus")
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 50    # per source -> ~100 total
K_CUE, K_NEXT, K_FTS, SNIP = 18, 18, 14, 350
AGE_MIN = 1

cov = [c for c in json.load(open(os.path.join(OUT, "coverage.json")))
       if c.get("coverage") and c.get("top")]

def dd(s):
    return date.fromisoformat(s[:10])

for c in cov:
    cut = dd(c["cutoff"])
    aged = [t["cos"] for t in c["top"] if (cut - dd(t["created_at"])).days >= AGE_MIN]
    c["_cov_aged"] = aged[0] if aged else 0.0       # top is cos-desc -> first aged = best

def strided(grp, k):
    if len(grp) <= k:
        return grp
    step = len(grp) / k
    return [grp[int(i * step)] for i in range(k)]

def select(src, n):
    pool = sorted([c for c in cov if c["source"] == src], key=lambda c: -c["_cov_aged"])
    if len(pool) <= n:
        return pool
    n_hi = int(n * 0.6); n_mid = int(n * 0.25); n_lo = n - n_hi - n_mid
    third = max(1, len(pool) // 3)
    return (strided(pool[:third], n_hi) +
            strided(pool[third:2 * third], n_mid) +
            strided(pool[2 * third:], n_lo))

sel = select("anchor_turn", N_PER) + select("operator_msg", N_PER)
print(f"selected {len(sel)} cues ({N_PER}/source)")

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn = env.brain.conn
    rows = conn.execute(
        """SELECT n.id, n.title, n.type, n.created_at, n.content, e.embedding
           FROM node_enrichments e JOIN nodes n ON n.id = e.node_id
           WHERE e.vector_type='_primary' AND n.archived=0""").fetchall()
    ids = [r[0] for r in rows]
    meta = {r[0]: {"title": r[1] or "", "type": r[2] or "",
                   "created_at": r[3] or "", "content": r[4] or ""} for r in rows}
    created = np.array([r[3] or "" for r in rows])
    V = np.vstack([np.frombuffer(r[5], dtype=np.float32) for r in rows])

    nm_blobs = embedder.embed_batch([c["next_move"] for c in sel], kind="query")

    out = []
    for c, nb in zip(sel, nm_blobs):
        cutoff = c["cutoff"]
        cand = {}                                       # id -> set(lens)
        for t in c["top"][:K_CUE]:                      # lens 1: cosine-on-cue
            cand.setdefault(t["id"], set()).add("cos_cue")
        if nb:                                          # lens 2: cosine-on-next_move
            s = V @ np.frombuffer(nb, dtype=np.float32)
            s = np.where(created < cutoff, s, -np.inf)
            for j in np.argsort(-s)[:K_NEXT]:
                if np.isfinite(s[j]):
                    cand.setdefault(ids[j], set()).add("cos_next")
        for nid in env.brain._fts.search(c["cue_text"], limit=K_FTS):   # lens 3: FTS-on-cue
            if nid in meta and meta[nid]["created_at"] < cutoff:
                cand.setdefault(nid, set()).add("fts")

        cands = []
        for nid, lenses in cand.items():
            m = meta.get(nid)
            if not m:
                continue
            cands.append({"id": nid, "type": m["type"], "title": m["title"][:120],
                          "created_at": m["created_at"][:10],
                          "lens": sorted(lenses), "snippet": (m["content"] or "")[:SNIP]})
        out.append({
            "cand_id": c["cand_id"], "source": c["source"], "cutoff": cutoff,
            "cov_aged": round(c["_cov_aged"], 4),
            "cue_text": c["cue_text"], "next_move": c["next_move"],
            "candidates": cands,
        })

json.dump(out, open(os.path.join(OUT, "teacher_input.json"), "w"), indent=1)
ncand = [len(o["candidates"]) for o in out]
from collections import Counter
print(f"wrote teacher_input.json — {len(out)} cues, candidates/cue min/median/max = "
      f"{min(ncand)}/{sorted(ncand)[len(ncand)//2]}/{max(ncand)}")
lens_ct = Counter(l for o in out for cc in o["candidates"] for l in cc["lens"])
print(f"lens coverage (candidate-appearances): {dict(lens_ct)}")
print(f"by source: {dict(Counter(o['source'] for o in out))}")

# echo one cue's candidate union for eyeball
ex = next((o for o in out if o["source"] == "operator_msg" and o["cov_aged"] >= 0.75), out[0])
print(f"\n--- sample teacher_input [{ex['cand_id']}] cov_aged={ex['cov_aged']} ---")
print(f"CUE: {ex['cue_text'][:200]}")
print(f"NEXT: {ex['next_move'][:200]}")
print(f"{len(ex['candidates'])} candidates:")
for cc in ex["candidates"][:10]:
    print(f"  [{cc['id']}] ({cc['type']}, {cc['created_at']}, {'+'.join(cc['lens'])}) {cc['title'][:70]}")
