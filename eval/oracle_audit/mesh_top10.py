#!/usr/bin/env python3
"""Mesh the two disjoint draws into the final audit corpus (meshed_top10.json).
Selection was hand-curated across both draws; this records it reproducibly.
See docs/ORACLE-AUDIT-SPEC.md §11."""
import json, os

OUTDIR = "/Users/tpac/brain/.claude/worktrees/frosty-feistel-90c7a9/eval/oracle_audit"
b1 = {o["i"]: o for o in json.load(open(f"{OUTDIR}/sample_30.json"))}      # seed 4
b2 = {o["i"]: o for o in json.load(open(f"{OUTDIR}/sample_seed7.json"))}   # seed 7

picks = [
    ("B1", 13, "learned-preference / scar"),
    ("B1", 20, "cross-project (EX.CO)"),
    ("B2", 9,  "prior-decision (dreams paused)"),
    ("B2", 27, "recurring-incident recall"),
    ("B2", 21, "identity / emotional continuity"),
    ("B1", 16, "prior-decision recall"),
    ("B1", 1,  "prior-research recall"),
    ("B1", 17, "design recall"),
    ("B1", 24, "convention recall"),
    ("B2", 4,  "deprecation recall"),
]
out = []
for rank, (src, idx, cls) in enumerate(picks, 1):
    o = (b1 if src == "B1" else b2)[idx]
    out.append({
        "rank": rank, "src": f"{src}-{idx}", "class": cls,
        "trace_id": o["trace_id"], "s0_chain": o["s0_chain"],
        "recall_chain": o["recall_chain"], "session_id": o["session_id"],
        "created_at": o["created_at"], "prompt": o["prompt"],
    })

with open(f"{OUTDIR}/meshed_top10.json", "w") as f:
    json.dump(out, f, indent=2)
for o in out:
    print(f"{o['rank']:>2} {o['src']:<6} {o['created_at'][:10]} {o['class']:<32} | {o['prompt'][:70]}")
print("\nwrote meshed_top10.json")
