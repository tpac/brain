#!/usr/bin/env python3
"""STAGE 1b-iv — FREEZE the endo gold corpus from teacher verdicts + validate.

Keeps endo_worthy cues that named gold, joins back cue/cutoff/candidates, writes
the frozen corpus (control_corpus-shaped, two-tier gold) that Step-2 baseline and
Step-3 PPR both score against.

Reports (per Tom: anchor vs operator ALWAYS broken out separately):
  - endo_worthy RATE by source
  - endo_worthy RATE by aged-coverage tertile — does coverage predict worthiness?
    (validates the stratifier — a flat gradient means cov was not the right filter)
  - which LENS each essential-gold came from → the baseline ceiling: gold reachable
    only via the next_move / FTS lenses is gold the cosine-on-cue baseline MUST miss.

Run: ./dev python3 eval/oracle_audit/endo_corpus_freeze.py
"""
import json, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "endo_corpus")
inp = {o["cand_id"]: o for o in json.load(open(f"{OUT}/teacher_input.json"))}
V = [v for v in json.load(open(f"{OUT}/teacher_verdicts.json")) if not v.get("error")]
print(f"{len(V)} non-error verdicts")

def rate(items):
    n = len(items); ew = sum(1 for v in items if v["endo_worthy"])
    return f"{ew}/{n} = {100*ew/n:.0f}%" if n else "n/a"

print("\n-- endo_worthy RATE by source --")
for src in ("anchor_turn", "operator_msg"):
    print(f"  {src:13s} {rate([v for v in V if v['source'] == src])}")

print("\n-- endo_worthy RATE by aged-coverage tertile (validates the stratifier) --")
covs = sorted(v["cov_aged"] for v in V if v.get("cov_aged") is not None)
if covs:
    t1, t2 = np.percentile(covs, [33, 66])
    def band(v):
        c = v.get("cov_aged") or 0
        return "hi" if c > t2 else ("mid" if c > t1 else "lo")
    for b in ("lo", "mid", "hi"):
        print(f"  cov-{b:3s} {rate([v for v in V if band(v) == b])}")

print("\n-- endo_worthy cues by query_type --")
for qt, ct in Counter(v["query_type"] for v in V if v["endo_worthy"]).most_common():
    print(f"  {qt:13s} {ct}")

# consistency: endo_worthy but no gold named (contradictory — log, don't freeze)
no_gold = [v for v in V if v["endo_worthy"] and not (v["gold_essential"] or v["gold_helpful"])]
if no_gold:
    print(f"\n  NOTE: {len(no_gold)} endo_worthy verdicts named NO gold (dropped): "
          f"{[v['cand_id'] for v in no_gold]}")

# ---- freeze ----
corpus = []
for v in V:
    if not v["endo_worthy"] or not (v["gold_essential"] or v["gold_helpful"]):
        continue
    o = inp[v["cand_id"]]
    lens_of = {c["id"]: c["lens"] for c in o["candidates"]}
    corpus.append({
        "id": v["cand_id"], "source": v["source"], "cutoff": o["cutoff"],
        "query": o["cue_text"], "next_move": o["next_move"],
        "query_type": v["query_type"], "confidence": v["confidence"],
        "gold_essential": v["gold_essential"], "gold_helpful": v["gold_helpful"],
        "teacher_why": v["why"], "cov_aged": v.get("cov_aged"),
        "gold_lens": {g: lens_of.get(g, []) for g in v["gold_essential"] + v["gold_helpful"]},
    })

json.dump(corpus, open(f"{OUT}/endo_gold_corpus.json", "w"), indent=1)
print(f"\n=== FROZEN endo gold corpus -> endo_gold_corpus.json : {len(corpus)} cues ===")
print(f"  by source: {dict(Counter(c['source'] for c in corpus))}")
print(f"  by query_type: {dict(Counter(c['query_type'] for c in corpus))}")

# baseline ceiling: how reachable is essential gold via cosine-on-cue (the baseline path)?
ess_lens = Counter()
for c in corpus:
    for g in c["gold_essential"]:
        for l in c["gold_lens"].get(g, []):
            ess_lens[l] += 1
cue_reach = sum(1 for c in corpus
                if c["gold_essential"]
                and all("cos_cue" in c["gold_lens"].get(g, []) for g in c["gold_essential"]))
with_ess = sum(1 for c in corpus if c["gold_essential"])
print(f"  essential-gold lens appearances: {dict(ess_lens)}")
print(f"  cues where ALL essential gold is cosine-on-cue-reachable: {cue_reach}/{with_ess}")
print(f"    -> the rest need next_move/FTS lenses: cosine-on-cue baseline will structurally MISS them")
