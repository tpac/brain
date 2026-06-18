#!/usr/bin/env python3
"""Faithful endo recall — runs against an IsolatedBrain COPY (never the live db).
Fixes the two recall artifacts the MCP tool can't:
  - FATIGUE: each cue gets a UNIQUE session_id → no cross-cue fatigue dampening.
  - CUTOFF: large over-fetch + created_at<=cutoff → cue-era survivors aren't
    truncated away by today's future nodes stealing top-K slots.
Dumps top-K per cue (no LLM, no Anthropic spend) for inline strict-bar judging.

Run: ./dev python3 eval/oracle_audit/endo_clean_recall.py [n_cues]
"""
import json, os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.isolated_brain import IsolatedBrain

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SRC = "/tmp/endo_baseline"
OUT = "/tmp/endo_clean"
os.makedirs(OUT, exist_ok=True)
LIMIT = 120  # over-fetch so the cutoff strip leaves real cue-era survivors

cue_files = sorted(glob.glob(os.path.join(SRC, "cue_*.json")))[:N]

with IsolatedBrain() as env:
    brain = env.brain
    for cf in cue_files:
        rec = json.load(open(cf))
        i = os.path.basename(cf).replace("cue_", "").replace(".json", "")
        try:
            res = brain.recall(
                query=rec["cue_text"],
                filter={"created_at": {"lte": rec["cutoff"]}},
                limit=LIMIT,
                session_id=f"endo-eval-{i}",   # fresh session => zero fatigue carryover
            )
            results = res.get("results", []) if isinstance(res, dict) else []
        except Exception as e:
            results = []
            print(f"cue {i}: recall error {e}", file=sys.stderr)

        top = []
        for r in results[:12]:
            top.append({
                "id": r.get("id"),
                "type": r.get("type"),
                "title": (r.get("title") or "")[:90],
                "created_at": (r.get("created_at") or "")[:10],
            })
        out = {
            "cue_id": rec["cue_id"],
            "cutoff": rec["cutoff"][:10],
            "cue_text": rec["cue_text"],
            "next_move": rec["next_move"],
            "returned": len(results),
            "top": top,
        }
        json.dump(out, open(os.path.join(OUT, f"cue_{i}.json"), "w"), indent=1)
        print(f"cue {i}: {len(results)} returned (clean), top-1: {top[0]['title'] if top else '—'}")

print(f"\nwrote clean recalls to {OUT}")
