#!/usr/bin/env python3
"""The gold × signal MATRIX — the reusable LAF instrument (§18.21).

ONE primitive, two jobs (Tom, 2026-06-29):
  • compare LAF VARIATIONS — roll the matrix up into a tier×rank-band scorecard per ranker.
  • reverse-engineer NEW fields FROM golds — drill the matrix per gold: which signal (if any)
    reaches it, set-cover the reachable golds, and surface the residual no column reaches
    (the encode/embedder gold = the target a NEW activation field must light up).

Why a matrix, not a scorecard: a scorecard aggregates away the per-gold detail that IS the
raw material for inventing a field. The matrix keeps every (cue, gold-node, signal) cell, so:
  - a LAF variation = a re-weighting of columns (cheap, no re-embed — substrate built once),
  - a new field = a new column that covers golds the current columns miss (unique-reach/set-cover).
Same shape as reverse-regression / reach_matrix / the Q4 operator-bank (8bcc8c96), now on the
LENS-INDEPENDENT four-tier gold (Gold+/Gold/Silver+/Silver) instead of the circular corpus.

Run (daemon maintenance-locked — 2nd embedder contends):
  ./dev python3 eval/laf/gold24_matrix.py
Out: eval/laf/gold24_matrix.json  (the persisted matrix, for offline reverse-engineering)
   + eval/laf/gold24_matrix.md    (scorecard + per-gold drill)  + console summary
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, maxsim_field, primary_field,
    build_adjacency, graph_spread,
)
from field_recall import FieldEngine, LAFConfig, ranked_ids           # noqa: E402
from gold24_diagnostic import load_cues, ndcg5, TIER_GAIN             # noqa: E402
from servers.scales.s1.trace_links import gather, nodes_for_traces    # noqa: E402


def episodic_field(brain, query, cutoff, idx, n):
    """Episodic reach: cue → similar PAST conversations (recall_episodes, older_than=cutoff)
    → the nodes surfaced/encoded in those conversations (nodes_for_traces) → per-node activation
    = the best (max) cosine score of an episode it was linked to. Query-DEPENDENT, a different
    reach path than cosine (cue↔conversation, not cue↔node). Verified live (gold24_verify)."""
    vec = np.zeros(n, dtype=np.float64)
    ep = brain.recall_episodes(query=query, older_than=cutoff, scale="s0", limit=15)
    episodes = ep.get("episodes", []) if isinstance(ep, dict) else []
    by_sess = {}
    for e in episodes:
        by_sess.setdefault(e.get("session_id"), []).append(e)
    for sess, eps in by_sess.items():
        if not sess:
            continue
        surf, enc = gather(brain, sess)
        links = nodes_for_traces(surf, enc, eps)
        score_by_tid = {e["id"]: float(e.get("_score") or 0.0) for e in eps}
        for tid, link in links.items():
            s = score_by_tid.get(tid, 0.0)
            for node in set(link.get("surfaced", []) + link.get("encoded", [])):
                i = idx.get(node)
                if i is not None and s > vec[i]:
                    vec[i] = s
    return vec

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "gold24_matrix.json")
OUT_MD = os.path.join(HERE, "gold24_matrix.md")

BANDS = [("≤5", 1, 5), ("6–25", 6, 25), ("26–120", 26, 120), ("unreach", 121, 10**9)]
TIERS = [("gold_plus", "Gold+"), ("gold", "Gold"), ("silver_plus", "Silver+"), ("silver", "Silver")]


def band_of(rank):
    if rank is None:
        return "unreach"
    for name, lo, hi in BANDS:
        if lo <= rank <= hi:
            return name
    return "unreach"


def zscore(x, mask):
    out = np.zeros(len(x), dtype=np.float64)
    m = mask & np.isfinite(x)
    if int(np.sum(m)) < 2:
        return out
    mu, sd = float(np.mean(x[m])), float(np.std(x[m]))
    if sd > 1e-9:
        out[m] = (x[m] - mu) / sd
    return out


def ranks_from_scores(scores, eligible, master_idx_of):
    """{row_index: rank} via one argsort; ineligible/NaN pushed to the bottom."""
    s = np.where(eligible & np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-s)
    rank = {int(i): r + 1 for r, i in enumerate(order)}
    return rank


def main():
    cues = load_cues()
    # node → its tier (for matrix rows) ; collect all gold+silver rows we care about
    print("gold24 matrix — %d cues" % len(cues))

    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        eng = FieldEngine(brain, model, cfg=LAFConfig(scale=8.0))
        master, idx, mats = eng.master, eng.idx, eng.mats
        adj = build_adjacency(brain, idx)
        ca = np.array([dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall()).get(n, "")
                       for n in master])
        FIELDS = list(MAXSIM_GROUPS)

        # ── base signals (matrix columns) computed per cue over the FULL field ──
        # plus the cheap variations (z-combos) and the settled LAF; pipeline is rank-only.
        SIGNALS = ["primary", "maxsim", "episodic", "graph1hop", "temporal"] + FIELDS
        VARIANTS = ["pipeline", "primary", "maxsim", "maxsim+temporal", "maxsim+graph",
                    "maxsim+episodic", "LAF-settled"]

        matrix = {}          # cue_id → node_id → {tier, form, source, qt, need, lens, signals:{rank,z}}
        variant_ranks = defaultdict(dict)   # variant → cue_id → {node_id: rank}

        for c in cues:
            qv = query_vec(c["query"])
            elig = (ca != "") & (ca <= c["cutoff"])
            cue_rows = {}

            # full-field signal vectors
            if qv is not None:
                prim = primary_field(qv, mats)
                ms = maxsim_field(qv, mats, FIELDS)
                seeds = (ms > np.nanpercentile(ms[np.isfinite(ms)], 90)).astype(np.float64)
                g1 = graph_spread(seeds, adj, hops=1)
                temp = eng._z(np.zeros(len(master)), elig)  # placeholder; real temporal below
                from operators import temporal_distinctiveness  # local import; node-prior
                temp = temporal_distinctiveness(eng.days, elig, eng.cfg.window_days)
                field_vecs = {vt: (mats[vt] @ qv) for vt in FIELDS}
                epi = episodic_field(brain, c["query"], c["cutoff"], idx, len(master))
                vecs = {"primary": prim, "maxsim": ms, "episodic": epi,
                        "graph1hop": g1, "temporal": temp, **field_vecs}
                z = {s: zscore(vecs[s], elig) for s in SIGNALS}
                rk = {s: ranks_from_scores(vecs[s], elig, idx) for s in SIGNALS}

                # variants (cheap re-weights of z-scored signals; LAF = settled)
                vr = {}
                vr["primary"] = rk["primary"]
                vr["maxsim"] = rk["maxsim"]
                vr["maxsim+temporal"] = ranks_from_scores(z["maxsim"] + 0.3 * z["temporal"], elig, idx)
                vr["maxsim+graph"] = ranks_from_scores(z["maxsim"] + 0.5 * z["graph1hop"], elig, idx)
                vr["maxsim+episodic"] = ranks_from_scores(z["maxsim"] + 0.5 * z["episodic"], elig, idx)
                a_laf, _ = eng.recall(qv, elig)
                laf_order = {nid: i + 1 for i, nid in enumerate(ranked_ids(a_laf, master))}
            else:
                vecs = z = rk = {}; vr = {}; laf_order = {}

            # pipeline (production recall, top-120)
            if qv is not None:
                res = brain.recall(query=c["query"], filter={"created_at": {"lte": c["cutoff"]}},
                                   limit=120, session_id="mtx-%s" % c["id"])
                pipe = {r.get("id"): i + 1 for i, r in
                        enumerate(res.get("results", []) if isinstance(res, dict) else [])}
            else:
                pipe = {}

            # ── fill the matrix rows for this cue (gold+silver nodes) ──
            for nid in (c["ess"] | c["helpful"]):
                i = idx.get(nid)
                sig = {}
                for s in SIGNALS:
                    r = rk.get(s, {}).get(i) if i is not None else None
                    zz = float(z[s][i]) if (i is not None and s in z) else None
                    sig[s] = {"rank": r, "z": round(zz, 3) if zz is not None else None}
                row = {
                    "tier": c["tier_of"].get(nid),
                    "form": c["form_of"].get(nid, ""),
                    "source": c["source"], "query_type": c["query_type"],
                    "need": next((nd for nd, ids in c["needs"].items() if nid in ids), None),
                    "lens": c["lens"].get(nid, []),
                    "in_master": i is not None,
                    "signals": sig,
                    "pipeline_rank": pipe.get(nid),
                }
                cue_rows[nid] = row
            matrix[c["id"]] = cue_rows

            # variant ranks for the scorecard roll-up (over gold+silver rows)
            allnodes = c["ess"] | c["helpful"]
            for v in VARIANTS:
                if v == "pipeline":
                    variant_ranks[v][c["id"]] = {nid: pipe.get(nid) for nid in allnodes}
                elif v == "LAF-settled":
                    variant_ranks[v][c["id"]] = {nid: laf_order.get(nid) for nid in allnodes}
                else:
                    src = vr.get(v, {})
                    variant_ranks[v][c["id"]] = {nid: (src.get(idx.get(nid)) if idx.get(nid) is not None else None)
                                                 for nid in allnodes}

    json.dump({"cues": {c["id"]: {"source": c["source"], "query_type": c["query_type"],
                                  "cutoff": c["cutoff"], "n_encode_gaps": c["n_encode_gaps"]}
                        for c in cues},
               "matrix": matrix, "signals": SIGNALS}, open(OUT_JSON, "w"), indent=1)
    report(cues, matrix, variant_ranks, VARIANTS, SIGNALS)


# ───────────────────────────── views ─────────────────────────────
def report(cues, matrix, variant_ranks, VARIANTS, SIGNALS):
    out = []
    def p(s=""):
        print(s); out.append(s)

    tier_nodes = {t: [] for t, _ in TIERS}      # (cue,node) per tier
    need_index = defaultdict(lambda: defaultdict(set))  # cue → need → node_ids (essential needs)
    for c in cues:
        for nid, row in matrix[c["id"]].items():
            if row["tier"]:
                tier_nodes[row["tier"]].append((c["id"], nid))
            if row["tier"] in ("gold_plus", "gold") and row["need"]:
                need_index[c["id"]][row["need"]].add(nid)

    # ===== VIEW A — expanded scorecard: tier × rank-band, per variant =====
    p("\n================  VIEW A — SCORECARD (tier × rank-band, per variant)  ================")
    for v in VARIANTS:
        p("\n  %s" % v)
        p("    %-9s %5s  %5s %5s %6s %7s" % ("tier", "n", "≤5", "6–25", "26–120", "unreach"))
        for t, tl in TIERS:
            rows = tier_nodes[t]
            if not rows:
                continue
            bands = Counter(band_of(variant_ranks[v][cid].get(nid)) for cid, nid in rows)
            n = len(rows)
            p("    %-9s %5d  %4.0f%% %4.0f%% %5.0f%% %6.0f%%" % (
                tl, n, 100*bands["≤5"]/n, 100*bands["6–25"]/n, 100*bands["26–120"]/n, 100*bands["unreach"]/n))
        # need-collapsed: a need met@5 by ANY essential node, and by a GOLD+ node specifically
        any5 = gp5 = total = 0
        for c in cues:
            for need, nids in need_index[c["id"]].items():
                total += 1
                if any((variant_ranks[v][c["id"]].get(n) or 1e9) <= 5 for n in nids):
                    any5 += 1
                gp = {n for n in nids if matrix[c["id"]][n]["tier"] == "gold_plus"}
                if gp and any((variant_ranks[v][c["id"]].get(n) or 1e9) <= 5 for n in gp):
                    gp5 += 1
        p("    need-met@5: any=%.0f%%  gold+only=%.0f%%   (of %d essential needs)"
          % (100*any5/(total or 1), 100*gp5/(total or 1), total))

    # ===== form × source breakouts (essential nodes, best variant = LAF-settled) =====
    p("\n================  VIEW A2 — FORM / SOURCE breakouts (LAF-settled, essential nodes)  ================")
    v = "LAF-settled"
    for axis, key in (("form", "form"), ("source", "source")):
        p("\n  by %s:" % axis)
        groups = defaultdict(list)
        for c in cues:
            for nid, row in matrix[c["id"]].items():
                if row["tier"] in ("gold_plus", "gold"):
                    groups[row.get(key) or "—"].append((c["id"], nid))
        p("    %-12s %5s  %5s %5s %6s %7s" % (axis, "n", "≤5", "6–25", "26–120", "unreach"))
        for g in sorted(groups, key=lambda k: -len(groups[k])):
            rows = groups[g]; n = len(rows)
            bands = Counter(band_of(variant_ranks[v][cid].get(nid)) for cid, nid in rows)
            p("    %-12s %5d  %4.0f%% %4.0f%% %5.0f%% %6.0f%%" % (
                g, n, 100*bands["≤5"]/n, 100*bands["6–25"]/n, 100*bands["26–120"]/n, 100*bands["unreach"]/n))

    # ===== VIEW B — reverse-engineer: per-signal unique reach + set-cover + residual =====
    p("\n================  VIEW B — REVERSE-ENGINEER (per essential need)  ================")
    # for each essential need: under EACH signal, best rank over its nodes; reach@25 = any node ≤25
    sig_reach = {s: set() for s in SIGNALS}       # signal → set of (cue,need) it reaches@25
    all_needs = set()
    residual = []                                  # needs NO signal reaches@25 (the new-field target)
    for c in cues:
        for need, nids in need_index[c["id"]].items():
            key = (c["id"], need)
            all_needs.add(key)
            reached_by = []
            for s in SIGNALS:
                best = min((matrix[c["id"]][n]["signals"][s]["rank"] or 1e9) for n in nids)
                if best <= 25:
                    sig_reach[s].add(key); reached_by.append(s)
            if not reached_by:
                residual.append((c["id"], need, sorted({matrix[c["id"]][n]["form"] for n in nids if matrix[c["id"]][n]["form"]})))
    p("  per-signal reach@25 (of %d essential needs) + UNIQUE (only signal that reaches it):" % len(all_needs))
    p("    %-14s %7s %7s" % ("signal", "reach", "unique"))
    for s in sorted(SIGNALS, key=lambda s: -len(sig_reach[s])):
        uniq = sum(1 for k in sig_reach[s] if all(k not in sig_reach[o] for o in SIGNALS if o != s))
        p("    %-14s %6d  %6d" % (s, len(sig_reach[s]), uniq))
    covered = set().union(*sig_reach.values()) if sig_reach else set()
    p("  UNION reach@25 (any signal): %d/%d = %.0f%%   |   RESIDUAL (no signal reaches): %d"
      % (len(covered), len(all_needs), 100*len(covered)/(len(all_needs) or 1), len(residual)))
    # greedy set-cover: minimal signals to cover the reachable needs
    remaining = set(covered); chosen = []
    pool = {s: set(sig_reach[s]) for s in SIGNALS}
    while remaining:
        best_s = max(pool, key=lambda s: len(pool[s] & remaining))
        gain = pool[best_s] & remaining
        if not gain:
            break
        chosen.append((best_s, len(gain))); remaining -= gain
    p("  greedy SET-COVER of the reachable needs: " + " → ".join("%s(+%d)" % (s, g) for s, g in chosen))
    p("\n  RESIDUAL needs (no current signal reaches @25 — the NEW-FIELD targets, %d):" % len(residual))
    for cid, need, forms in residual[:12]:
        p("    [%s|%s] %s" % (cid, "/".join(forms) or "—", (need[:88] + "…") if len(need) > 88 else need))
    if len(residual) > 12:
        p("    … +%d more (full list in %s)" % (len(residual) - 12, os.path.relpath(OUT_JSON)))

    write_md(out)
    p("\n  → matrix persisted: %s   |   views: %s" % (os.path.relpath(OUT_JSON), os.path.relpath(OUT_MD)))


def write_md(console):
    open(OUT_MD, "w").write(
        "# LAF gold × signal matrix — scorecard + reverse-engineer views (§18.21)\n\n"
        "Generated by `eval/laf/gold24_matrix.py`. Matrix primitive persisted to "
        "`gold24_matrix.json` (per cue × gold/silver node × signal: rank + z). VIEW A = compare "
        "variations (tier × rank-band). VIEW B = reverse-engineer (per-signal unique reach, "
        "set-cover, and the residual needs no signal reaches = the targets a NEW activation "
        "field must light up).\n\n```\n" + "\n".join(console) + "\n```\n")


if __name__ == "__main__":
    main()
