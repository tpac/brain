#!/usr/bin/env python3
"""STAGE 1b-i report — stratify endo cues by AGED prior-coverage.

cov_max alone is confounded for anchor_turn cues: Anchor's own prose is trivially
cosine-near nodes the encoder wrote FROM Anchor's prose (self-similar style /
same-session echo). A FORGOTTEN move-changer is by definition OLDER than the cue.
So we add an age-gap: cov_max_aged = the highest-cosine prior at least AGE_MIN days
older than the cue. That's the real "a forgotten prior plausibly exists" signal.

Read-only over endo_corpus/coverage.json. No embedder, no spend.
Run: ./dev python3 eval/oracle_audit/endo_coverage_report.py [age_min_days]
"""
import json, os, sys
from datetime import date
import numpy as np

AGE_MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 1
HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.path.join(HERE, "endo_corpus", "coverage.json")
cues = [c for c in json.load(open(COV)) if c.get("coverage") and c.get("top")]

def d(s):
    return date.fromisoformat(s[:10])

for c in cues:
    cut = d(c["cutoff"])
    gaps = [(cut - d(t["created_at"])).days for t in c["top"]]
    c["_top1_gap"] = gaps[0]
    aged = [(t["cos"], g) for t, g in zip(c["top"], gaps) if g >= AGE_MIN]
    c["_cov_max_aged"] = round(aged[0][0], 4) if aged else 0.0   # top is cos-desc -> first aged = best aged
    c["_aged_gap"] = aged[0][1] if aged else None

def pct(vals):
    a = np.array(sorted(vals))
    return {p: round(float(np.percentile(a, p)), 3) for p in (10, 25, 50, 75, 90)} if len(a) else {}

print(f"AGE_MIN = {AGE_MIN} day(s); {len(cues)} scored cues\n")
for src in ("anchor_turn", "operator_msg", "ALL"):
    sub = [c for c in cues if src == "ALL" or c["source"] == src]
    if not sub:
        continue
    same_day = sum(1 for c in sub if c["_top1_gap"] < AGE_MIN)
    aged_mx = [c["_cov_max_aged"] for c in sub]
    print(f"[{src}] n={len(sub)}")
    print(f"  top-1 prior is SAME-DAY/echo (gap<{AGE_MIN}d): {same_day} ({100*same_day//len(sub)}%)")
    print(f"  cov_max_AGED pctiles: {pct(aged_mx)}")
    for th in (0.7, 0.75, 0.8):
        print(f"    cues w/ cov_max_aged>={th}: {sum(1 for v in aged_mx if v >= th)}")
    print()

# the endo-worthy candidate pool: aged prior exists AND is strong
POOL_TH = 0.75
pool = [c for c in cues if c["_cov_max_aged"] >= POOL_TH]
from collections import Counter
print(f"=== endo-worthy candidate POOL (cov_max_aged >= {POOL_TH}): {len(pool)} ===")
print(f"  by source: {dict(Counter(c['source'] for c in pool))}")

print("\n--- sample: highest aged-coverage cues (the teacher-worthy end) ---")
pool.sort(key=lambda c: -c["_cov_max_aged"])
for c in pool[:6]:
    aged_top = next((t for t, g in zip(c["top"], [(d(c['cutoff'])-d(t['created_at'])).days for t in c["top"]]) if g >= AGE_MIN), None)
    print(f"\n[{c['cand_id']}] cov_aged={c['_cov_max_aged']} gap={c['_aged_gap']}d  src={c['source']}  cutoff={c['cutoff'][:10]}")
    print(f"  CUE : {c['cue_text'][:160]}")
    print(f"  NEXT: {c['next_move'][:200]}")
    if aged_top:
        print(f"  TOP AGED PRIOR: [{aged_top['type']}] {aged_top['title']}  (cos={aged_top['cos']}, {aged_top['created_at']})")

print("\n--- sample: LOW aged-coverage cues (likely correct-silence / no prior) ---")
low = sorted(cues, key=lambda c: c["_cov_max_aged"])[:4]
for c in low:
    print(f"\n[{c['cand_id']}] cov_aged={c['_cov_max_aged']}  src={c['source']}  cutoff={c['cutoff'][:10]}")
    print(f"  CUE : {c['cue_text'][:160]}")
    print(f"  NEXT: {c['next_move'][:160]}")
