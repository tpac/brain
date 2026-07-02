#!/usr/bin/env python3
"""P1 latency probe — what the episodic field costs per query on the live path (§19 P1).

The §19 P1 gate needs the episodic pull cost measured FIRST: if cheap, laf_v1 ships whole
(maxsim + pick+enc episodic); if expensive, split shape (maxsim-only on the MCP path, full
stack on the hook path). This probe decomposes the per-query cost of the WINNING stack's
episodic side on the 24 frozen gold cues, against IsolatedBrain (never a second Brain() on
the live DB), with production `brain.recall()` as the comparator the overhead is judged
against.

Per-query stages instrumented (monkeypatch timers, code paths unchanged):
  embed        embedder.embed_query — the cue embedding recall_episodes does internally
  ep_scan      TraceDAL.filter_event_vectors — SQL join + ≤500 trace-vector fetch + cosine
  ep_hydrate   TraceDAL.get_by_ids — top-K episode record hydration
  gather       per-session 3-stream trace pulls (the predicted dominant term)
  join         nodes_for_traces — the stop-keyed role join
  vec          episodic_encoded + episodic_picked over the master index (_records reuse;
               dropped is OUT of the winner by direct ablation)

Conditions (7-day-default shape deliberately NOT tested — it's a wiring trap to avoid,
not a candidate: Tom, 2026-07-02):
  A eval-faithful      older_than = per-cue gold cutoff (ties latency to the 16/28 numbers)
  C live-full-history  older_than = now (the shipping shape; NOTE filter_event_vectors is
                       ORDER BY created_at DESC LIMIT 500, so "full history" = newest 500
                       embedded s0 conversational traces — a coverage ceiling, not a knob)

Also measured: production brain.recall(limit=25) per cue (baseline), per-query maxsim_field
(the other half of the winner — expected trivial), one-time build costs (matrices, embedder
load) reported separately from the hot path.

Run: ./dev python3 eval/laf/latency_probe.py        (daemon may add cpu noise; repeats+medians)
Out: eval/laf/latency_probe.md
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.isolated_brain import IsolatedBrain                        # noqa: E402
from servers import embedder                                          # noqa: E402
from servers.clock import iso_now                                     # noqa: E402
import episodic_ops                                                   # noqa: E402
from episodic_ops import (                                            # noqa: E402
    episodic_roles, episodic_encoded, episodic_picked,
)
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, build_field_matrices, maxsim_field, query_vec,
)
from gold24_diagnostic import load_cues                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "latency_probe.md")
REPEATS = 3
BASELINE_LIMIT = 25          # the hook path's candidate pull size


# ───────────────────────── instrumentation ─────────────────────────
class Acc:
    """Per-call stage accumulator — reset per measured query."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = defaultdict(float)      # stage → seconds (summed within one query)
        self.n = defaultdict(int)        # stage → call count / row tallies


ACC = Acc()


def _timed(stage, fn, rows_of=None):
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        ACC.t[stage] += time.perf_counter() - t0
        ACC.n[stage] += 1
        if rows_of is not None:
            ACC.n[stage + "_rows"] += rows_of(r)
        return r
    return wrapper


def instrument(brain):
    """Patch timers around the episodic stack's real call sites. Paths unchanged."""
    embedder.embed_query = _timed("embed", embedder.embed_query)
    episodic_ops.gather = _timed(
        "gather", episodic_ops.gather,
        rows_of=lambda st: sum(len(v) for v in st.values()))
    episodic_ops.nodes_for_traces = _timed("join", episodic_ops.nodes_for_traces)
    brain._trace_dal.filter_event_vectors = _timed(
        "ep_scan", brain._trace_dal.filter_event_vectors, rows_of=len)
    brain._trace_dal.get_by_ids = _timed(
        "ep_hydrate", brain._trace_dal.get_by_ids, rows_of=len)


# ───────────────────────── stats helpers ─────────────────────────
def pct(vals, q):
    return float(np.percentile(np.asarray(vals, dtype=float), q)) if vals else 0.0


def ms(x):
    return "%.0f" % (x * 1000.0)


def summarize(rows, key):
    """Per-cue median of `key` (damp repeat noise), then p50/p95 across cues."""
    per_cue = defaultdict(list)
    for r in rows:
        per_cue[r["cue"]].append(r[key])
    medians = [float(np.median(v)) for v in per_cue.values()]
    return pct(medians, 50), pct(medians, 95)


# ───────────────────────── the probe ─────────────────────────
def run():
    cues = load_cues()
    print("cues: %d" % len(cues))
    now = iso_now()

    with IsolatedBrain() as env:
        brain = env.brain

        # one-time costs, kept OUT of hot-path numbers
        t0 = time.perf_counter()
        brain.recall(query="warm", limit=1)          # loads embedder + recall caches
        t_warm = time.perf_counter() - t0
        t0 = time.perf_counter()
        master, idx, mats = build_field_matrices(brain, None, MAXSIM_GROUPS)
        t_mats = time.perf_counter() - t0
        n = len(master)
        print("warmup %.1fs · matrices %.1fs (%d nodes)" % (t_warm, t_mats, n))

        instrument(brain)
        episodic_roles(brain, "warmup episodic pull", now)   # warm trace-side caches

        conditions = {
            "A_eval_cutoff": lambda c: c["cutoff"],
            "C_live_fullhist": lambda c: now,
        }
        ep_rows, base_rows, maxsim_rows = [], [], []

        for cond, cut_of in conditions.items():
            for c in cues:
                for rep in range(REPEATS):
                    ACC.reset()
                    t0 = time.perf_counter()
                    records = episodic_roles(brain, c["query"], cut_of(c))
                    t_roles = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    episodic_encoded(brain, c["query"], cut_of(c), idx, n,
                                     _records=records)
                    episodic_picked(brain, c["query"], cut_of(c), idx, n,
                                    _records=records)
                    t_vec = time.perf_counter() - t0
                    ep_rows.append({
                        "cue": c["id"], "cond": cond, "rep": rep,
                        "total": t_roles + t_vec, "roles": t_roles, "vec": t_vec,
                        "embed": ACC.t["embed"], "ep_scan": ACC.t["ep_scan"],
                        "ep_hydrate": ACC.t["ep_hydrate"],
                        "gather": ACC.t["gather"], "join": ACC.t["join"],
                        "sessions": ACC.n["gather"],
                        "rows": ACC.n["gather_rows"],
                        "scanned": ACC.n["ep_scan_rows"],
                        "moments": len(records),
                    })
                print("  %s %s: total %s ms (gather %s ms, %d sessions, %d rows)"
                      % (cond, c["id"], ms(ep_rows[-1]["total"]),
                         ms(ep_rows[-1]["gather"]), ep_rows[-1]["sessions"],
                         ep_rows[-1]["rows"]))

        # baseline: production recall, same cues (embed timer still active — fine,
        # we only use the end-to-end number here). recall has a 5s-TTL result
        # cache — clear it between repeats or the median collapses to the
        # cached fast path.
        for c in cues:
            for rep in range(REPEATS):
                if getattr(brain, "_recall_cache", None):
                    with brain._recall_cache_lock:
                        brain._recall_cache.clear()
                t0 = time.perf_counter()
                brain.recall(query=c["query"], limit=BASELINE_LIMIT)
                base_rows.append({"cue": c["id"], "rep": rep,
                                  "total": time.perf_counter() - t0})

        # maxsim per query (the winner's other half)
        for c in cues:
            qv = query_vec(c["query"])
            for rep in range(REPEATS):
                t0 = time.perf_counter()
                maxsim_field(qv, mats, MAXSIM_GROUPS)
                maxsim_rows.append({"cue": c["id"], "rep": rep,
                                    "total": time.perf_counter() - t0})

    report(cues, ep_rows, base_rows, maxsim_rows, t_warm, t_mats, n)


def report(cues, ep_rows, base_rows, maxsim_rows, t_warm, t_mats, n):
    lines = []
    w = lines.append
    w("# P1 latency probe — episodic field on the live path")
    w("")
    w("%d cues × %d repeats · IsolatedBrain (%d master nodes) · one-time: warmup %.1fs, "
      "field matrices %.1fs" % (len(cues), REPEATS, n, t_warm, t_mats))
    w("")
    w("## Stage decomposition (per-cue medians → p50/p95 across cues, ms)")
    w("")
    w("| condition | total | embed | ep_scan | ep_hydrate | gather | join | vec | "
      "sessions p50 | rows p50 | moments p50 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in ("A_eval_cutoff", "C_live_fullhist"):
        rows = [r for r in ep_rows if r["cond"] == cond]
        cells = ["%s/%s" % (ms(summarize(rows, k)[0]), ms(summarize(rows, k)[1]))
                 for k in ("total", "embed", "ep_scan", "ep_hydrate",
                           "gather", "join", "vec")]
        sess = pct([float(np.median([x["sessions"] for x in rows
                    if x["cue"] == c])) for c in {r["cue"] for r in rows}], 50)
        rws = pct([float(np.median([x["rows"] for x in rows
                   if x["cue"] == c])) for c in {r["cue"] for r in rows}], 50)
        mom = pct([float(np.median([x["moments"] for x in rows
                   if x["cue"] == c])) for c in {r["cue"] for r in rows}], 50)
        w("| %s | %s | %.0f | %.0f | %.0f |" % (cond, " | ".join(cells),
                                                sess, rws, mom))
    b50, b95 = summarize(base_rows, "total")
    m50, m95 = summarize(maxsim_rows, "total")
    w("")
    w("## Comparators")
    w("")
    w("| path | p50 | p95 |")
    w("|---|---|---|")
    w("| production brain.recall(limit=%d) | %s | %s |" % (BASELINE_LIMIT,
                                                           ms(b50), ms(b95)))
    w("| maxsim_field (winner's other half) | %s | %s |" % (ms(m50), ms(m95)))
    w("")
    empties = sorted({r["cue"] for r in ep_rows
                      if r["cond"] == "C_live_fullhist" and r["moments"] == 0})
    w("Cues with ZERO moments on the live shape (empty ≠ truth — honesty note): %s"
      % (", ".join(empties) or "none"))
    w("")
    w("## Per-cue detail — C_live_fullhist (median of %d repeats, ms)" % REPEATS)
    w("")
    w("| cue | total | gather | sessions | rows | moments |")
    w("|---|---|---|---|---|---|")
    for cid in sorted({r["cue"] for r in ep_rows}):
        rows = [r for r in ep_rows if r["cond"] == "C_live_fullhist"
                and r["cue"] == cid]
        med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
        w("| %s | %s | %s | %.0f | %.0f | %.0f |"
          % (cid, ms(med("total")), ms(med("gather")),
             med("sessions"), med("rows"), med("moments")))
    out = "\n".join(lines) + "\n"
    with open(OUT_MD, "w") as f:
        f.write(out)
    print("\n" + out.split("## Per-cue detail")[0])
    print("wrote %s" % OUT_MD)


if __name__ == "__main__":
    run()
