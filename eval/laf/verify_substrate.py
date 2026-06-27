#!/usr/bin/env python3
"""LAF substrate + operator verification gate — prove every component is ALIVE and
FAITHFUL before any LAF number is trusted.

Tom's mandate (2026-06-25): the episodic-0-on-IsolatedBrain bug cost a whole session of
false iteration because a dead component was invisible to the top-level metric. On run 1
this gate already caught a second one — edge_context, a 0.55-weighted scoring group, has
NEVER produced a row in production (its backfill handler was never implemented). So:
no LAF result is believable until every input + operator passes LIVENESS · INPUT-DEPENDENCE
(varies with input — catches the recency=1.000 constant class) · INVARIANT (obeys its math)
· GROUND-TRUTH (matches reality) · FAITHFULNESS (the copy behaves like production).

The baseline (19%/33%) is a SUSPECT, not an anchor — re-derived here from verified
components; ground truth bottoms out in REALITY (cosine recomputed from raw embedding
bytes, gold checked against the nodes table), never a prior number.

Run:
  ./dev python3 eval/laf/verify_substrate.py            # self-contained core + MaxSim
  ./dev python3 eval/laf/verify_substrate.py --live     # + live-daemon faithfulness (next)
  ./dev python3 eval/laf/verify_substrate.py --haiku     # + Haiku qualitative audit (next)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))            # project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))  # endo harness

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from endo_baseline_recall import (                                    # noqa: E402
    load_corpus, make_baseline_ranker, score_one, LIMIT,
)
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, KNOWN_DEAD_GROUPS, query_vec,
    build_field_matrices, maxsim_field, primary_field,
    build_adjacency, graph_spread,
    parse_days, temporal_distinctiveness,
)

SAMPLE_CUES = 6
COS_TOLERANCE = 0.02     # recall rounds embedding_similarity to ~3dp
NODE_FLOOR = 4000
NEG = -1e9               # rank-out sentinel for ineligible/absent nodes


class Gate:
    def __init__(self):
        self.fails, self.warns, self.passes = [], [], []

    def check(self, name, ok, detail="", warn=False):
        tag = "PASS" if ok else ("WARN" if warn else "FAIL")
        print("  [%s] %s%s" % (tag, name, (" — " + detail) if detail else ""), flush=True)
        (self.passes if ok else (self.warns if warn else self.fails)).append(name)
        return ok

    def info(self, msg):
        print("       %s" % msg, flush=True)


# ───────────────────────── tiers ─────────────────────────

def t0_substrate(g, brain, env):
    print("\n== T0 · substrate liveness ==")
    n = env.node_count()
    g.check("nodes copied", n >= NODE_FLOOR, "%d non-archived (floor %d)" % (n, NODE_FLOOR))
    ready = embedder.is_ready()
    if not ready:
        try:
            brain.recall(query="warm the embedder", limit=1)
            ready = embedder.is_ready()
        except Exception:
            pass
    g.check("embedder ready", ready, "model=%s" % (embedder.stats.get("model_name") or "?"))
    qv = query_vec("does the embedder produce a real vector") if ready else None
    g.check("embed_query returns a real vector", qv is not None, "dim=%d" % (len(qv) if qv is not None else 0))
    return embedder.stats.get("model_name") or ""


def t1_gold_integrity(g, brain, corpus):
    print("\n== T1 · corpus + gold integrity (the episodic / gold-rot class) ==")
    gold_ids = sorted({gid for c in corpus for gid in c.get("gold_essential", [])})
    g.check("corpus loaded", len(corpus) == 73, "%d cues, %d distinct essential-gold" % (len(corpus), len(gold_ids)))
    ph = ",".join("?" * len(gold_ids))
    rows = brain.conn.execute(
        "SELECT id, archived, created_at FROM nodes WHERE id IN (%s)" % ph, gold_ids).fetchall()
    found = {r[0]: {"archived": r[1], "created_at": r[2]} for r in rows}
    missing = [gid for gid in gold_ids if gid not in found]
    archived = [gid for gid, v in found.items() if v["archived"]]
    g.check("every essential-gold exists in the copy", not missing, "%d missing: %s" % (len(missing), missing[:6]))
    g.check("no essential-gold is archived", not archived, "%d archived: %s" % (len(archived), archived[:6]))
    bad_time = [(c["id"], gid) for c in corpus for gid in c.get("gold_essential", [])
                if found.get(gid, {}).get("created_at") and found[gid]["created_at"] > c["cutoff"]]
    g.check("every essential-gold predates its cue cutoff", not bad_time,
            "%d post-date cutoff: %s" % (len(bad_time), bad_time[:4]))
    return gold_ids


def t2_embedding_liveness(g, brain, mats, idx, gold_ids):
    print("\n== T2 · embedding liveness per field-group ==")
    n_total = brain.conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
    for vt in MAXSIM_GROUPS:
        present = int(np.sum(~np.isnan(mats[vt][:, 0])))
        cov = present / max(n_total, 1)
        floor = 0.5 if vt in ("_primary", "title") else 0.001
        g.check("group %-12s alive" % vt, present > 0 and cov >= floor,
                "coverage %.0f%% (%d/%d)" % (cov * 100, present, n_total))
    for vt in sorted(KNOWN_DEAD_GROUPS):
        # explicit, documented exclusion — NOT a silent green. Proceeding without it
        # per Tom's call (2026-06-25); fix tracked in a separate task.
        g.check("group %-12s (known-dead, excluded from MaxSim)" % vt, False,
                "0 rows in prod — _edge_descriptions backfill unimplemented; fix task spawned", warn=True)
    gold_no_prim = [gid for gid in gold_ids
                    if gid not in idx or np.isnan(mats["_primary"][idx[gid], 0])]
    g.check("every essential-gold has a _primary vector", not gold_no_prim,
            "%d gold cosine-unreachable: %s" % (len(gold_no_prim), gold_no_prim[:6]), warn=bool(gold_no_prim))


def t3_recompute(g, brain, corpus, mats, idx):
    print("\n== T3 · independent recomputation (don't trust the pipeline's self-report) ==")
    diffs, checked, no_vec = [], 0, 0
    for c in corpus[:SAMPLE_CUES]:
        qv = query_vec(c["query"])
        if qv is None:
            continue
        res = brain.recall(query=c["query"], filter={"created_at": {"lte": c["cutoff"]}},
                           limit=LIMIT, session_id="verify-recompute-%s" % c["id"])
        for r in res.get("results", []):
            rep, nid = r.get("embedding_similarity"), r.get("id")
            if rep is None:
                continue
            row = idx.get(nid)
            if row is None or np.isnan(mats["_primary"][row, 0]):
                no_vec += 1
                continue
            diffs.append(abs(float(rep) - float(np.dot(qv, mats["_primary"][row]))))
            checked += 1
    if diffs:
        med, mx = float(np.median(diffs)), float(np.max(diffs))
        g.check("reported cosine == recomputed-from-raw-bytes", med <= COS_TOLERANCE,
                "n=%d  median|Δ|=%.4f  max|Δ|=%.4f  (tol %.2f)" % (checked, med, mx, COS_TOLERANCE))
        if no_vec:
            g.info("%d returned nodes had a similarity but no _primary vector (keyword/fts candidates)" % no_vec)
    else:
        g.check("reported cosine == recomputed", False, "no comparable pairs")


def t5_baseline(g, brain, corpus):
    print("\n== T5 · baseline re-derivation (the suspect, re-measured on verified substrate) ==")
    ranker = make_baseline_ranker(brain)
    h5 = h25 = miss = 0
    for c in corpus:
        m = score_one(ranker(c), c["gold_essential"], c.get("gold_helpful", []))
        h5 += m["hit5_ess"]; h25 += m["hit25_ess"]; miss += (m["best_ess_rank"] is None)
    n = len(corpus)
    p5, p25 = h5 / n, h25 / n
    g.info("re-derived BASELINE: hit@5 %.0f%%  hit@25 %.0f%%  (%d/%d gold-not-in-pool)" % (p5*100, p25*100, miss, n))
    near = abs(p5 - 0.19) <= 0.02 and abs(p25 - 0.33) <= 0.02
    g.check("re-derived baseline reproduces recorded 19%/33% (±2pp — HARD FAIL)", near,
            "got %.0f%%/%.0f%% — %s" % (p5*100, p25*100,
            "reproducible" if near else "DIVERGES → substrate is NOT faithful; no LAF number is trustworthy"))


def t6_maxsim(g, brain, corpus, master, idx, mats):
    print("\n== T6 · operator: MaxSim-cosine (liveness · input-dependence · invariant · A/B) ==")
    # invariant: MaxSim ≥ _primary everywhere (max includes _primary)
    qv = query_vec(corpus[0]["query"])
    ms, pr = maxsim_field(qv, mats, MAXSIM_GROUPS), primary_field(qv, mats)
    mask = ~np.isnan(pr)
    viol = int(np.sum(ms[mask] < pr[mask] - 1e-5))
    g.check("MaxSim ≥ _primary (max includes the primary group)", viol == 0,
            "%d violations of the invariant" % viol)
    g.check("MaxSim is live (non-empty field)", ms.size > 0 and np.any(~np.isnan(ms)), "%d nodes scored" % ms.size)
    # input-dependence
    qa = query_vec("daemon recovery and launchd restart")
    qb = query_vec("identity, partnership, and what it means to be Anchor")
    ta = {master[i] for i in np.argsort(-np.nan_to_num(maxsim_field(qa, mats, MAXSIM_GROUPS), nan=NEG))[:10]}
    tb = {master[i] for i in np.argsort(-np.nan_to_num(maxsim_field(qb, mats, MAXSIM_GROUPS), nan=NEG))[:10]}
    g.check("MaxSim is input-dependent (≠ constant)", ta != tb, "two unrelated queries share %d/10 top" % len(ta & tb))

    # A/B signal: MaxSim-ranking vs _primary-ranking, full-field, cutoff-filtered.
    _ca = dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall())   # one query, not N
    ca = np.array([_ca.get(nid) or "" for nid in master])
    qv_cache = {c["id"]: query_vec(c["query"]) for c in corpus}                      # embed each cue once
    def hits(score_fn):
        h5 = h25 = 0
        for c in corpus:
            q = qv_cache[c["id"]]
            if q is None:
                continue                                          # unembeddable cue → miss
            s = np.nan_to_num(score_fn(q), nan=NEG)
            s = np.where((ca != "") & (ca <= c["cutoff"]), s, NEG)   # cutoff eligibility (empty excluded)
            order = np.argsort(-s)
            gold = {idx[gid] for gid in c["gold_essential"] if gid in idx}
            h5 += int(bool(gold & set(order[:5].tolist())))
            h25 += int(bool(gold & set(order[:25].tolist())))
        n = len(corpus)
        return h5 / n, h25 / n
    ms5, ms25 = hits(lambda q: maxsim_field(q, mats, MAXSIM_GROUPS))
    pr5, pr25 = hits(lambda q: primary_field(q, mats))
    g.info("full-field A/B (cutoff-filtered, no fatigue/pipeline):")
    g.info("  _primary-only  hit@5 %.0f%%  hit@25 %.0f%%   (recorded raw _primary: 21%%/37%%)" % (pr5*100, pr25*100))
    g.info("  MaxSim(%d grp) hit@5 %.0f%%  hit@25 %.0f%%   (Δ@5 %+.0fpp  Δ@25 %+.0fpp vs _primary)"
           % (len(MAXSIM_GROUPS), ms5*100, ms25*100, (ms5-pr5)*100, (ms25-pr25)*100))
    # Reach Δ is INFORMATIONAL, not a gate: it's only meaningful under field add/remove
    # ablation, not as a single-operator pass/fail. Raw MaxSim diluting @25 is the expected
    # result that makes the separation/recurrence layer load-bearing — measured, not gated.


def t7_graph_spread(g, brain, master, idx):
    print("\n== T7 · operator: typed-graph-spread (liveness · seed-locality · input-dependence) ==")
    adj = build_adjacency(brain, idx)
    src, dst, w, degree = adj
    g.check("typed adjacency non-empty", src.size > 0, "%d typed undirected edges" % src.size)
    connected = int(np.sum(degree > 0))
    g.check("graph covers the field", connected > 0,
            "%d/%d nodes have ≥1 typed edge (%.0f%%)" % (connected, len(master), 100*connected/max(len(master), 1)))
    # seed-locality: one hop from the most-connected node lights its neighbors
    seed_i = int(np.argmax(degree))
    a = np.zeros(len(master), dtype=np.float32); a[seed_i] = 1.0
    out = graph_spread(a, adj, hops=1)
    nbrs = list(set(dst[src == seed_i].tolist()) | set(src[dst == seed_i].tolist()))
    nbr_lit = bool(nbrs) and all(out[j] > 0 for j in nbrs[:30])
    g.check("spread lights the seed's neighbors", nbr_lit and out.sum() > 0,
            "%d neighbors lit, sum(out)=%.3f" % (len(nbrs), float(out.sum())))
    # invariant: a degree-0 node stays dark
    dark = np.where(degree == 0)[0]
    if dark.size:
        g.check("disconnected node stays dark after spread", float(out[int(dark[0])]) == 0.0,
                "%d isolated nodes" % int(dark.size))
    # input-dependence
    seed_j = int(np.argsort(-degree)[5])
    b = np.zeros(len(master), dtype=np.float32); b[seed_j] = 1.0
    out2 = graph_spread(b, adj, hops=1)
    g.check("spread is input-dependent (≠ constant)", not np.allclose(out, out2),
            "two different seeds give different fields")
    return adj


def t8_temporal(g, brain, master):
    print("\n== T8 · operator: temporal-distinctiveness (liveness · INDEPENDENT direction · degeneracy) ==")
    WINDOW = 7.0
    ca_map = dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall())
    days = parse_days([ca_map.get(nid, "") for nid in master])
    elig = ~np.isnan(days)
    dist = temporal_distinctiveness(days, elig, WINDOW)
    nz = dist[dist > 0]
    g.check("temporal produces values", nz.size > 0, "%d nodes scored" % nz.size)

    # Direction check, INDEPENDENT (not 1/dist, which is circular with the operator): recount
    # each node's neighbours straight from `days`, then confirm the node the operator ranks
    # most-isolated genuinely has fewer real neighbours than the one it ranks most-crowded.
    valid = days[elig]
    def real_neighbours(i):
        return int(np.sum(np.abs(valid - days[i]) <= WINDOW)) - 1        # −1 excludes self
    i_iso = int(np.argmax(dist))
    i_crowd = int(np.argmin(np.where(dist > 0, dist, np.inf)))
    nb_iso, nb_crowd = real_neighbours(i_iso), real_neighbours(i_crowd)
    g.check("operator's 'isolated' node independently HAS fewer neighbours (von-Restorff direction)",
            nb_iso < nb_crowd,
            "isolated %d real neighbours  vs crowd %d  (recounted from created_at, not from dist)"
            % (nb_iso, nb_crowd))

    # DEGENERACY gate — the check the old `std > 1e-6` missed. von-Restorff needs ISOLATED nodes
    # to exist; on a burst-created corpus every node sits in a dense temporal crowd, so the field
    # is functionally constant and the `_z` step reinflates that micro-variance into a pure
    # query-independent NOISE prior. An operator that separates ~nothing must not be trusted to
    # move the fused score, even though it is technically "non-constant" and "live".
    frac_distinctive = float(np.mean(nz >= 1.0 / 3.0)) if nz.size else 0.0   # ≥1/3 ⇔ ≤2 neighbours
    cv = float(nz.std() / nz.mean()) if nz.size and nz.mean() > 1e-12 else 0.0
    g.check("temporal is NON-DEGENERATE (separates nodes, not z-amplified noise)",
            frac_distinctive >= 0.01,
            "%.1f%% of nodes distinctive (≤2 neighbours), CV=%.3f, range=[%.4f, %.4f]%s"
            % (frac_distinctive * 100, cv, float(nz.min()), float(nz.max()),
               "" if frac_distinctive >= 0.01 else
               " — DEGENERATE on this corpus: contributes NOISE, do NOT trust its fused lift"),
            warn=frac_distinctive < 0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--haiku", action="store_true")
    args = ap.parse_args()

    print("LAF substrate + operator verification — every component proves itself first.")
    print("MaxSim field-groups (edge_context excluded, known-dead): %s" % ", ".join(MAXSIM_GROUPS))

    g = Gate()
    corpus = load_corpus()
    with IsolatedBrain() as env:
        brain = env.brain
        model = t0_substrate(g, brain, env)
        gold_ids = t1_gold_integrity(g, brain, corpus)
        master, idx, mats = build_field_matrices(brain, model, MAXSIM_GROUPS)
        t2_embedding_liveness(g, brain, mats, idx, gold_ids)
        t3_recompute(g, brain, corpus, mats, idx)
        t5_baseline(g, brain, corpus)
        t6_maxsim(g, brain, corpus, master, idx, mats)
        t7_graph_spread(g, brain, master, idx)
        t8_temporal(g, brain, master)
        if args.live or args.haiku:
            print("\n(--live / --haiku gates land next, once the operator set is complete)")

    print("\n" + "=" * 64)
    print("VERIFICATION: %d pass · %d warn · %d FAIL" % (len(g.passes), len(g.warns), len(g.fails)))
    if g.fails:
        print("FAILED: %s" % ", ".join(g.fails))
        print("→ NOT trustworthy. Do not run LAF experiments until green.")
        sys.exit(1)
    if g.warns:
        print("WARN: %s" % ", ".join(g.warns))
    print("→ verified (warnings are known, documented exclusions).")


if __name__ == "__main__":
    main()
