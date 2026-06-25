#!/usr/bin/env python3
"""Aggregate the teacher-quality verdicts into the recall-quality baseline.
Joins verdicts (from the workflow) to the sample's flags for breakdowns.

Run: ./dev python3 eval/oracle_audit/surface_quality_report.py [dir]
  expects <dir>/verdicts.json (list of verdict objects) + <dir>/flags.json
"""
import json, os, sys
from collections import Counter, defaultdict

DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/teacher_baseline"
verdicts = json.load(open(os.path.join(DIR, "verdicts.json")))
flags = json.load(open(os.path.join(DIR, "flags.json")))

N = len(verdicts)
def pct(a, b): return f"{100*a/b:4.1f}%" if b else "  n/a"
def dist(key, items=verdicts):
    c = Counter(v.get(key) for v in items)
    return c

def show(title, c, denom=None):
    denom = denom or sum(c.values())
    print(f"  {title}")
    for k, n in c.most_common():
        print(f"    {str(k):28s} {n:3d}  {pct(n, denom)}")

print(f"\n=== RECALL-QUALITY BASELINE — {N} teacher verdicts ===\n")

print("confidence:")
show("", dist("confidence"))

# headline
sm = dist("served_move")
served = sm.get("served_well", 0)
served_partial = served + sm.get("partial", 0)
irrelevant = sm.get("recall_irrelevant_to_turn", 0)
needed = N - irrelevant  # turns that actually needed memory
print("\n-- HEADLINE: did recall serve the move? --")
show("", sm)
print(f"  served_well                 : {pct(served, N)} of all turns")
print(f"  served_well+partial         : {pct(served_partial, N)} of all turns")
print(f"  served_well (memory-needed) : {pct(served, needed)}  (excl. {irrelevant} recall-irrelevant turns)")

print("\n-- pick quality --")
show("", dist("picks_quality"))
dropped = [v for v in verdicts if v.get("picks_quality") == "better_candidate_dropped"]
print(f"  better_candidate_dropped: {len(dropped)} ({pct(len(dropped), N)})")

def by_group(group_fn, label):
    print(f"\n-- served_move by {label} --")
    groups = defaultdict(list)
    for v in verdicts:
        g = group_fn(v)
        if g is not None:
            groups[g].append(v)
    for g, items in sorted(groups.items(), key=lambda x: -len(x[1])):
        sw = sum(1 for v in items if v["served_move"] == "served_well")
        sp = sum(1 for v in items if v["served_move"] in ("served_well", "partial"))
        print(f"    {str(g):28s} n={len(items):3d}  served_well {pct(sw,len(items))}  +partial {pct(sp,len(items))}")

by_group(lambda v: v.get("query_type"), "query_type")
by_group(lambda v: flags.get(v["turn_id"], {}).get("provenance"), "pick provenance")
by_group(lambda v: "contested(top1 dropped)" if flags.get(v["turn_id"], {}).get("contested") else "uncontested", "structural contested-flag")

print("\n-- sample dropped-better notes (selection left value in the pool) --")
for v in dropped[:12]:
    print(f"    {v['turn_id']:18s} drop={v.get('dropped_better_id')}  {v.get('note','')[:90]}")

print("\n-- a few 'did_not_serve' / 'mostly_noise' notes --")
bad = [v for v in verdicts if v["served_move"] == "did_not_serve" or v["picks_quality"] == "mostly_noise"]
for v in bad[:12]:
    print(f"    {v['turn_id']:18s} {v['served_move']}/{v['picks_quality']}  {v.get('note','')[:80]}")
print()
