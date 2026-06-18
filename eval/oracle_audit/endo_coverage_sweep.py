#!/usr/bin/env python3
"""STAGE 1b-i — offline PRIOR-COVERAGE sweep over endo candidate cues.

For each cue, score how dense the PRE-CUTOFF node neighborhood is in pure cosine
space — the cheap proxy for "does a forgotten move-changer plausibly exist?"

Decoupled from the recall pipeline ON PURPOSE: ranking cues by the blend we're
evaluating would be circular — a real endo cue that recall *misses* would score
low and get dropped, hiding exactly the failure we want to measure. Pure cosine
to node_enrichments._primary (the canonical full-coverage vector) is the neutral
stratifier; the recall pipeline is what gets tested LATER against this corpus.

Local compute only — embeds cues with the local nomic model, ZERO API spend.
Runs against an IsolatedBrain COPY (never live). Persists durable output under
eval/oracle_audit/endo_corpus/ so the corpus + coverage are reusable.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_coverage_sweep.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.isolated_brain import IsolatedBrain
from servers import embedder

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "endo_corpus")
os.makedirs(OUT, exist_ok=True)
CAND = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "candidates.json")
K = 30
THRESHOLDS = [0.6, 0.65, 0.7, 0.75]

cues = json.load(open(CAND))
print(f"loaded {len(cues)} candidate cues from {CAND}")

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    if not embedder.is_ready():
        sys.exit("embedder failed to load — aborting")

    conn = env.brain.conn
    rows = conn.execute(
        """SELECT n.id, n.title, n.type, n.created_at, e.embedding
           FROM node_enrichments e JOIN nodes n ON n.id = e.node_id
           WHERE e.vector_type='_primary' AND n.archived=0""").fetchall()
    ids = [r[0] for r in rows]
    titles = [r[1] or "" for r in rows]
    types = [r[2] or "" for r in rows]
    created = np.array([r[3] or "" for r in rows])               # ISO-T, lexically comparable
    V = np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows])   # (N,768), pre-L2-normalized
    print(f"loaded {len(ids)} non-archived node vectors (dim={V.shape[1]})")

    blobs = embedder.embed_batch([c["cue_text"] for c in cues], kind="query")
    if len(blobs) != len(cues):
        print(f"WARN: embed returned {len(blobs)} blobs for {len(cues)} cues", file=sys.stderr)

    out = []
    for c, blob in zip(cues, blobs):
        c2 = dict(c)
        if not blob:
            c2["coverage"] = None
            out.append(c2)
            continue
        cv = np.frombuffer(blob, dtype=np.float32)
        scores = V @ cv                                          # cosine (both pre-normalized)
        elig = created < c["cutoff"]                             # strictly-before-cutoff nodes
        n_elig = int(elig.sum())
        s = np.where(elig, scores, -np.inf)
        k = min(K, n_elig)
        order = np.argsort(-s)[:k]
        top = [{"id": ids[j], "type": types[j], "title": titles[j][:90],
                "cos": round(float(scores[j]), 4), "created_at": str(created[j])[:10]}
               for j in order]
        fin = s[np.isfinite(s)]
        feats = {
            "n_eligible": n_elig,
            "cov_max":   round(float(top[0]["cos"]), 4) if top else 0.0,
            "cov_mean5": round(float(np.mean([t["cos"] for t in top[:5]])), 4) if top else 0.0,
            "cov_mean10": round(float(np.mean([t["cos"] for t in top[:10]])), 4) if top else 0.0,
        }
        for th in THRESHOLDS:
            feats[f"cnt_{int(th*100)}"] = int((fin >= th).sum())
        c2["coverage"] = feats
        c2["top"] = top
        out.append(c2)

with open(os.path.join(OUT, "coverage.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"wrote coverage -> {os.path.join(OUT, 'coverage.json')}")

# ── distribution report (this is what sizes the teacher pass) ──
def pctiles(vals):
    a = np.array(sorted(vals))
    return {p: round(float(np.percentile(a, p)), 3) for p in (10, 25, 50, 75, 90, 95)} if len(a) else {}

scored = [c for c in out if c.get("coverage")]
print(f"\nscored {len(scored)}/{len(out)} cues (rest had no cue embedding)")
for src in ("anchor_turn", "operator_msg", "ALL"):
    sub = [c for c in scored if src == "ALL" or c["source"] == src]
    if not sub:
        continue
    mx = [c["coverage"]["cov_max"] for c in sub]
    m5 = [c["coverage"]["cov_mean5"] for c in sub]
    print(f"\n[{src}] n={len(sub)}")
    print(f"  cov_max   pctiles: {pctiles(mx)}")
    print(f"  cov_mean5 pctiles: {pctiles(m5)}")
    for th in THRESHOLDS:
        print(f"  cues w/ cov_max>={th}: {sum(1 for v in mx if v >= th)}")
