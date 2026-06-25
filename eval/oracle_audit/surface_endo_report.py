#!/usr/bin/env python3
"""Aggregate endo-recall verdicts into the cosine-arm baseline.
hit@K = the Haiku-free top-K contained a forgotten + would-change-the-move node.

Run: ./dev python3 eval/oracle_audit/surface_endo_report.py [dir]
  expects <dir>/verdicts.json (list of endo verdict objects)
"""
import json, os, sys
from collections import Counter

DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/endo_baseline"
V = json.load(open(os.path.join(DIR, "verdicts.json")))
N = len(V)
def pct(a, b): return f"{100*a/b:4.1f}%" if b else "  n/a"

print(f"\n=== ENDO-RECALL BASELINE (cosine arm) — {N} cues ===\n")

conf = Counter(v.get("confidence") for v in V)
print("confidence:", dict(conf))

h1 = sum(1 for v in V if v.get("hit_in_top1"))
h2 = sum(1 for v in V if v.get("hit_in_top2"))
h5 = sum(1 for v in V if v.get("hit_in_top5"))
print("\n-- HIT RATE: Haiku-free top-K holds a forgotten + would-change node --")
print(f"  hit@1 : {h1:3d}  {pct(h1, N)}   <- the 1-node nudge")
print(f"  hit@2 : {h2:3d}  {pct(h2, N)}   <- the 1-2 node nudge (design target)")
print(f"  hit@5 : {h5:3d}  {pct(h5, N)}   <- reachable in the shallow pool")

# where the gold lands
ranks = [v.get("first_hit_rank") for v in V if v.get("first_hit_rank")]
none = sum(1 for v in V if not v.get("first_hit_rank"))
print("\n-- first_hit_rank (where the gold lands) --")
rc = Counter(ranks)
for r in sorted(rc):
    print(f"    rank {r:2d}: {rc[r]:3d}  {pct(rc[r], N)}")
print(f"    no hit in returned set: {none}  {pct(none, N)}")
if ranks:
    import statistics as st
    print(f"    mean first_hit_rank (when hit): {st.mean(ranks):.2f}")

# surface thinness
rcount = [v.get("returned_count", 0) for v in V]
empty = sum(1 for c in rcount if c == 0)
print("\n-- surface thinness (nodes recall returned post floor+cutoff) --")
import statistics as st
print(f"    mean returned: {st.mean(rcount):.1f}   empty (0 returned): {empty} ({pct(empty,N)})")

# conditional: among cues where recall returned >=1
nonempty = [v for v in V if v.get("returned_count", 0) > 0]
if nonempty:
    c1 = sum(1 for v in nonempty if v.get("hit_in_top1"))
    c2 = sum(1 for v in nonempty if v.get("hit_in_top2"))
    print(f"\n-- conditional on non-empty recall (n={len(nonempty)}) --")
    print(f"    hit@1: {pct(c1, len(nonempty))}   hit@2: {pct(c2, len(nonempty))}")

print("\n-- sample hits (cosine-endo got it) --")
for v in [v for v in V if v.get("hit_in_top1")][:8]:
    print(f"    {v['cue_id'][:16]:16s} r{v.get('first_hit_rank')}  {v.get('note','')[:90]}")
print("\n-- sample misses (nothing forgotten+would-change in top-K) --")
for v in [v for v in V if not v.get("first_hit_rank")][:8]:
    print(f"    {v['cue_id'][:16]:16s} ret={v.get('returned_count')}  {v.get('note','')[:85]}")
print()
