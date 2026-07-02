#!/usr/bin/env python3
"""P1 SHIP GATE — laf_v1 vs champion through the REAL brain.recall() (§19 P1).

The probes measured the composition; this measures the SHIPPED PATH: both arms
call the actual `brain.recall(query, limit=25)` on the same IsolatedBrain copy,
toggling BRAIN_RECALL_VARIANT per call. Whatever wiring, floors, fatigue,
hydration or telemetry does to the ranking is IN the number.

Fairness under no-cutoff-masking: the live path can't mask post-cutoff nodes,
so BOTH arms rank today's full brain; scoring then drops results created after
the cue's cutoff and rank-compresses IDENTICALLY in both arms. The comparison
is fair; absolute numbers are not directly comparable to the probe tables
(different eligibility mechanics — the probes masked INSIDE scoring).

Cache note: recall's 5s-TTL dedup key does NOT include the variant flag (in
production the flag is fixed per daemon process, so it can't alias there) —
the gate clears the result cache before every call so arms can't cross.

Reports: need@5 / need@25 per arm + per-cue win/loss table + warm latency
p50/p95 per arm (first flag-on call builds the engine caches — reported as
cold separately).

Run: ./dev python3 eval/laf/p1_gate.py
Out: eval/laf/p1_gate.md
"""
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from laf_metrics import need_hit_at                                   # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "p1_gate.md")
LIMIT = 25
PASSES = 2            # pass 1 warms caches; pass 2 supplies the latency numbers


def clear_cache(brain):
    if getattr(brain, "_recall_cache", None):
        with brain._recall_cache_lock:
            brain._recall_cache.clear()


def recall_arm(brain, query, variant):
    """One recall through the real path under the given variant. Returns
    (ranked [(node_id, created_at)], recall_mode, seconds)."""
    prev = os.environ.get("BRAIN_RECALL_VARIANT")
    try:
        if variant:
            os.environ["BRAIN_RECALL_VARIANT"] = variant
        else:
            os.environ.pop("BRAIN_RECALL_VARIANT", None)
        clear_cache(brain)
        t0 = time.perf_counter()
        res = brain.recall(query=query, limit=LIMIT)
        dt = time.perf_counter() - t0
        ranked = [(r["id"], r.get("created_at") or "") for r in res["results"]]
        return ranked, res.get("_recall_mode"), dt
    finally:
        if prev is None:
            os.environ.pop("BRAIN_RECALL_VARIANT", None)
        else:
            os.environ["BRAIN_RECALL_VARIANT"] = prev


def eligible_rank_map(ranked, cutoff):
    """{node_id: 1-based rank} over results created ≤ cutoff (rank-compressed
    identically for both arms — the fairness mechanic)."""
    out, r = {}, 0
    for nid, created in ranked:
        if created and created > cutoff:
            continue
        r += 1
        out[nid] = r
    return out


def pct(vals, q):
    return float(np.percentile(np.asarray(vals, dtype=float), q)) if vals else 0.0


def main():
    cues = load_cues()
    arms = {"champion": None, "laf_v1": "laf_v1"}
    per = defaultdict(dict)          # cue → arm → (h5, h25)
    lat = defaultdict(list)          # arm → warm seconds
    modes = defaultdict(set)
    cold = {}

    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        for p in range(PASSES):
            for arm, variant in arms.items():
                for c in cues:
                    if not c["needs"]:
                        continue
                    ranked, mode, dt = recall_arm(brain, c["query"], variant)
                    modes[arm].add(mode)
                    if p == 0 and arm == "laf_v1" and arm not in cold:
                        cold[arm] = dt          # includes engine cache build
                    if p == PASSES - 1:
                        lat[arm].append(dt)
                        rm = eligible_rank_map(ranked, c["cutoff"])
                        per[c["id"]][arm] = (
                            need_hit_at(rm, c["needs"], 5) or 0.0,
                            need_hit_at(rm, c["needs"], 25) or 0.0)

    nc = len(per) or 1
    lines = ["# P1 ship gate — laf_v1 vs champion through the real recall path",
             "",
             "%d cues · limit=%d · IsolatedBrain · both arms rank today's full "
             "brain, post-cutoff results dropped identically at scoring" % (nc, LIMIT),
             "",
             "| arm | need@5 | need@25 | warm p50 | warm p95 | mode |",
             "|---|---|---|---|---|---|"]
    print()
    totals = {}
    for arm in arms:
        h5 = sum(per[c][arm][0] for c in per) / nc
        h25 = sum(per[c][arm][1] for c in per) / nc
        totals[arm] = (h5, h25)
        row = (arm, "%.0f%%" % (100 * h5), "%.0f%%" % (100 * h25),
               "%.0fms" % (1000 * pct(lat[arm], 50)),
               "%.0fms" % (1000 * pct(lat[arm], 95)),
               "/".join(sorted(str(m) for m in modes[arm])))
        lines.append("| %s | %s | %s | %s | %s | %s |" % row)
        print("  %-9s need@5 %-5s need@25 %-5s | p50 %s p95 %s | %s" % row)
    lines += ["",
              "laf_v1 first call (engine cache build): %.0fms" %
              (1000 * cold.get("laf_v1", 0)),
              "",
              "## Per-cue (need@5 / need@25)",
              "",
              "| cue | champion | laf_v1 | Δ@5 | Δ@25 |",
              "|---|---|---|---|---|"]
    for cid in sorted(per):
        ch, lf = per[cid]["champion"], per[cid]["laf_v1"]
        lines.append("| %s | %.2f / %.2f | %.2f / %.2f | %+.2f | %+.2f |"
                     % (cid, ch[0], ch[1], lf[0], lf[1],
                        lf[0] - ch[0], lf[1] - ch[1]))
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote %s" % OUT_MD)


if __name__ == "__main__":
    main()
