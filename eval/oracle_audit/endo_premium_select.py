#!/usr/bin/env python3
"""Select 10 PREMIUM-corpus seed cues (5 operator + 5 anchor) spanning the failure
modes (hit / within-cluster-buried / cue-far) and query types, and write a rich
dossier per cue for the qualitative Opus-agent deep dive.

Each dossier: cue + next_move + cutoff, current teacher gold (+titles), the broad
candidate union (from teacher_input, +titles+lens), the baseline cue-cosine gold
rank + bucket, cov_aged. The agents then recall MORE, read traces before/after the
cue, examine node fields, and analyze "what would have surfaced this".

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_premium_select.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.isolated_brain import IsolatedBrain
from servers import embedder

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "endo_corpus")
corpus = json.load(open(f"{OUT}/endo_gold_corpus.json"))
tinput = {o["cand_id"]: o for o in json.load(open(f"{OUT}/teacher_input.json"))}
cmeta = {c["cand_id"]: c for c in json.load(open(f"{OUT}/candidates.json"))}   # session + cue_trace_id

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn = env.brain.conn
    rows = conn.execute("""SELECT n.id, n.title, n.created_at, e.embedding FROM node_enrichments e
        JOIN nodes n ON n.id=e.node_id WHERE e.vector_type='_primary' AND n.archived=0""").fetchall()
    ids = [r[0] for r in rows]; pos = {nid: i for i, nid in enumerate(ids)}
    titles = {r[0]: (r[1] or "")[:90] for r in rows}
    created = np.array([r[2] or "" for r in rows])
    V = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])

    cb = embedder.embed_batch([c["query"] for c in corpus], kind="query")
    for c, b in zip(corpus, cb):
        elig = created < c["cutoff"]
        sc = np.where(elig, V @ np.frombuffer(b, dtype=np.float32), -np.inf)
        rankpos = {nid: r + 1 for r, nid in enumerate(np.argsort(-sc))}  # node_idx rank
        gr = [rankpos[pos[g]] for g in c["gold_essential"] if g in pos]
        c["_best_rank"] = min(gr) if gr else 99999
        c["_bucket"] = "hit" if c["_best_rank"] <= 5 else "buried" if c["_best_rank"] <= 25 else "far"

def pick(src):
    pool = [c for c in corpus if c["source"] == src]
    chosen = []; seen_q = set()
    for target in ["hit", "buried", "far", "buried", "far"]:    # span the failure modes
        cands = [c for c in pool if c["_bucket"] == target and c not in chosen]
        cands.sort(key=lambda c: (c["query_type"] in seen_q, c["_best_rank"]))  # prefer new qtype
        if cands:
            chosen.append(cands[0]); seen_q.add(cands[0]["query_type"])
    for c in sorted(pool, key=lambda c: c["_best_rank"]):        # backfill to 5
        if len(chosen) >= 5:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen[:5]

sel = pick("operator_msg") + pick("anchor_turn")
doss = []
for c in sel:
    cands = tinput.get(c["id"], {}).get("candidates", [])
    doss.append({
        "cue_id": c["id"], "source": c["source"], "query_type": c["query_type"],
        "session": cmeta.get(c["id"], {}).get("session"),
        "cue_trace_id": cmeta.get(c["id"], {}).get("cue_trace_id"),
        "baseline_gold_rank": c["_best_rank"], "bucket": c["_bucket"], "cov_aged": c.get("cov_aged"),
        "cutoff": c["cutoff"], "cue_text": c["query"], "next_move": c["next_move"],
        "current_gold_essential": [{"id": g, "title": titles.get(g, "?")} for g in c["gold_essential"]],
        "current_gold_helpful": [{"id": g, "title": titles.get(g, "?")} for g in c["gold_helpful"]],
        "teacher_why": c["teacher_why"],
        "candidate_union": [{"id": x["id"], "title": x["title"], "lens": x["lens"], "type": x["type"]}
                            for x in cands],
    })
json.dump(doss, open(f"{OUT}/premium_seeds.json", "w"), indent=1)
print(f"wrote {len(doss)} premium dossiers -> premium_seeds.json\n")
for d in doss:
    g = d["current_gold_essential"][0]["title"][:48] if d["current_gold_essential"] else "-"
    print(f"  [{d['cue_id']:17s}] {d['source'][:8]:8s} {d['query_type']:12s} "
          f"rank={str(d['baseline_gold_rank']):>5} ({d['bucket']:6s})  gold: {g}")
