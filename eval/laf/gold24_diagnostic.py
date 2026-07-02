#!/usr/bin/env python3
"""LAF A/B + miss-diagnostic on the lens-independent 24-cue gold (§18.20 → §18.21).

Two jobs in one pass, both demanded by Tom (2026-06-29):
  1. RE-BASELINE the A/B on the NEW four-tier gold (Gold+/Gold/Silver+/Silver), need-collapsed
     + tier-graded — the old 21/37 numbers were on the CIRCULAR corpus and are not comparable.
  2. SOPHISTICATED ANALYSIS — per-need: did it surface, and if not WHY (reach vs rank), under
     which signal would it have, what FORM (redirect/ground/enrich) was it, and how did the
     blind judge find it (lens_tags) — so we can troubleshoot misses and characterize wins.

Reuses the existing instruments (extend, don't rebuild): operators.py (field matrices, MaxSim,
primary, typed-graph-spread), field_recall.FieldEngine (the settling LAF), IsolatedBrain.

Gold/cue source:
  frozen_gold_24.json   — tiers per cue: {node_id, form, need}; cutoff; source; query_type
  frozen_cards_24.json  — per-judge essential[{node_id, lens_tags, ...}] + encode_gaps + issues
  moments.json          — the clean conversation-window cue text (the query recall fires on)

Run (daemon maintenance-locked — 2nd embedder contends):
  ./dev python3 eval/laf/gold24_diagnostic.py
Out: eval/laf/gold24_diagnostic.md  (+ console summary)
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field, primary_field,
    build_adjacency, graph_spread,
)
from field_recall import FieldEngine, LAFConfig, ranked_ids           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REMINT = os.path.join(HERE, "..", "oracle_audit", "gold_remint")
GOLD = os.path.join(REMINT, "frozen_gold_24.json")
CARDS = os.path.join(REMINT, "frozen_cards_24.json")
MOMENTS = os.path.join(REMINT, "moments.json")
OUT_MD = os.path.join(HERE, "gold24_diagnostic.md")

KS = (1, 5, 10, 25)
TOP5, BURIED_MAX = 5, 120          # ≤5 hit · 6..120 buried(rank-miss) · >120/None reach-miss
TIER_GAIN = {"gold_plus": 3.0, "gold": 2.0, "silver_plus": 1.5, "silver": 1.0}


# ───────────────────────────── load + adapt ─────────────────────────────
def load_cues():
    gold = json.load(open(GOLD))
    cards = json.load(open(CARDS))
    moments = {m["cue_id"]: m for m in json.load(open(MOMENTS))}
    cues = []
    for cid, g in gold.items():
        mom = moments.get(cid)
        if not mom:
            continue
        tiers = g["tiers"]
        # essential = gold_plus ∪ gold ; helpful = silver_plus ∪ silver
        gplus = {it["node_id"] for it in tiers.get("gold_plus", [])}
        gold_ids = {it["node_id"] for it in tiers.get("gold", [])}
        ess = gplus | gold_ids
        sp = {it["node_id"] for it in tiers.get("silver_plus", [])}
        s = {it["node_id"] for it in tiers.get("silver", [])}
        helpful = sp | s
        # node → its best tier (for nDCG gain) ; node → form ; need → {node_ids}
        tier_of, form_of = {}, {}
        needs = defaultdict(set)
        for t in ("silver", "silver_plus", "gold", "gold_plus"):   # later overwrites → best tier wins
            for it in tiers.get(t, []):
                tier_of[it["node_id"]] = t
                if t in ("gold", "gold_plus"):
                    form_of[it["node_id"]] = it.get("form") or ""
                    needs[(it.get("need") or it["node_id"]).strip()].add(it["node_id"])
        # lens_tags per node from BOTH judge cards (essential + silver)
        lens = defaultdict(set)
        card = cards.get(cid, {})
        n_gaps = 0
        for jk in ("a", "b"):
            j = card.get(jk) or {}
            for fld in ("essential", "silver"):
                for it in j.get(fld, []) or []:
                    for lt in it.get("lens_tags", []) or []:
                        lens[it["node_id"]].add(lt)
            n_gaps += len(j.get("encode_gaps", []) or [])
        cues.append({
            "id": cid, "query": mom["cue"]["text"], "cutoff": g["cutoff"],
            "source": g["source"], "query_type": g.get("query_type"),
            "ess": ess, "gplus": gplus, "gold": gold_ids, "helpful": helpful,
            "tier_of": tier_of, "form_of": form_of, "needs": dict(needs),
            "lens": {k: sorted(v) for k, v in lens.items()}, "n_encode_gaps": n_gaps,
        })
    return cues


# ───────────────────────────── ranking helpers ─────────────────────────────
def rank_of(scores, eligible, i):
    """1-based rank of node-row i among eligible nodes by score desc (None if i absent/ineligible)."""
    if i is None or not eligible[i]:
        return None
    s = np.where(eligible & np.isfinite(scores), scores, -np.inf)
    v = s[i]
    if not np.isfinite(v):
        return None
    return int(np.sum(s > v)) + 1


def ranks_for_set(scores, eligible, idx, node_ids):
    """{node_id: rank} for the nodes that have a row + are eligible."""
    out = {}
    for nid in node_ids:
        i = idx.get(nid)
        r = rank_of(scores, eligible, i) if i is not None else None
        out[nid] = r
    return out


def hit_at(ranks, k):
    return 1 if any(r is not None and r <= k for r in ranks.values()) else 0


def ndcg5(ranked_list_ids, tier_of):
    rel = [TIER_GAIN.get(tier_of.get(nid), 0.0) for nid in ranked_list_ids[:5]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(tier_of.values(), key=lambda t: -TIER_GAIN.get(t, 0))[:5]
    idcg = sum(TIER_GAIN.get(t, 0.0) / math.log2(i + 2) for i, t in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


# ───────────────────────────── ablation (which field contributes) ─────────────────────────────
def cue_scores(c, ranked):
    """Tier/need-aware metrics for one cue from a ranked id list (capped lists ok)."""
    pos = {nid: i + 1 for i, nid in enumerate(ranked)}
    rk = {n: pos.get(n) for n in c["ess"]}
    rk_gp = {n: pos.get(n) for n in c["gplus"]}
    need_hits = sum(1 for nids in c["needs"].values() if any((pos.get(n) or 1e9) <= 5 for n in nids))
    return {"hit5": hit_at(rk, 5), "hit25": hit_at(rk, 25), "hit5_gp": hit_at(rk_gp, 5),
            "need_r5": need_hits / (len(c["needs"]) or 1), "ndcg5": ndcg5(ranked, c["tier_of"]),
            "source": c["source"]}


def run_ablation(eng, cues, master, ca, brain):
    """Per-operator toggle on the NEW gold → which activation field drives LAF.
    Reference baselines (pipeline = current production recall, raw _primary cosine) on top."""
    pre = {}
    for c in cues:
        pre[c["id"]] = (query_vec(c["query"]), (ca != "") & (ca <= c["cutoff"]))

    def agg(rows, k):
        v = [r[k] for r in rows]
        return sum(v) / len(v) if v else 0.0

    # ---- reference baselines ----
    refs = {"pipeline (current recall)": [], "raw _primary (cosine)": []}
    for c in cues:
        qv, elig = pre[c["id"]]
        if qv is None:
            refs["pipeline (current recall)"].append(cue_scores(c, []))
            refs["raw _primary (cosine)"].append(cue_scores(c, []))
            continue
        res = brain.recall(query=c["query"], filter={"created_at": {"lte": c["cutoff"]}},
                           limit=BURIED_MAX, session_id="abl-%s" % c["id"])
        pipe = [r.get("id") for r in (res.get("results", []) if isinstance(res, dict) else [])]
        refs["pipeline (current recall)"].append(cue_scores(c, pipe))
        prim = primary_field(qv, eng.mats)
        pr = [master[i] for i in np.argsort(-np.where(elig & np.isfinite(prim), prim, -np.inf))]
        refs["raw _primary (cosine)"].append(cue_scores(c, pr))

    # ---- LAF operator ablation (the settling pipeline, gains toggled) ----
    configs = [("LAF full (ms+temp+graph)", 1.0, 0.3, 0.5), ("LAF − graph (ms+temp)", 1.0, 0.3, 0.0),
               ("LAF − temporal (ms+graph)", 1.0, 0.0, 0.5), ("LAF maxsim-only (ms)", 1.0, 0.0, 0.0)]
    eng.cfg.scale = 8.0
    laf = {}
    for name, gm, gt, gg in configs:
        eng.cfg.gain_maxsim, eng.cfg.gain_temporal, eng.cfg.gain_graph = gm, gt, gg
        rows = []
        for c in cues:
            qv, elig = pre[c["id"]]
            rows.append(cue_scores(c, ranked_ids(eng.recall(qv, elig)[0], master)) if qv is not None
                        else cue_scores(c, []))
        laf[name] = rows

    print("\n================  ABLATION on the lens-independent 24-cue gold  ================")
    print("  which activation field drives LAF? (settling pipeline, one operator toggled at a time)")
    print("  %-28s %-7s %-7s %-8s %-8s %-7s" % ("config", "hit@5", "hit@25", "need@5", "gold+@5", "nDCG"))
    for name, rows in list(refs.items()):
        print("  %-28s %-7s %-7s %-8s %-8s %-7.2f" % (
            name, "%.0f%%" % (100 * agg(rows, "hit5")), "%.0f%%" % (100 * agg(rows, "hit25")),
            "%.0f%%" % (100 * agg(rows, "need_r5")), "%.0f%%" % (100 * agg(rows, "hit5_gp")), agg(rows, "ndcg5")))
    print("  " + "-" * 70)
    for name, rows in laf.items():
        print("  %-28s %-7s %-7s %-8s %-8s %-7.2f" % (
            name, "%.0f%%" % (100 * agg(rows, "hit5")), "%.0f%%" % (100 * agg(rows, "hit25")),
            "%.0f%%" % (100 * agg(rows, "need_r5")), "%.0f%%" % (100 * agg(rows, "hit5_gp")), agg(rows, "ndcg5")))
    print("\n  reading: compare LAF full vs '− temporal' (does temporal help or is it the burst-corpus")
    print("  artifact?) and vs 'maxsim-only' (does graph-spread add anything on a converged field?).")


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true",
                    help="per-operator toggle (which field drives LAF) instead of the full diagnostic")
    args = ap.parse_args()
    cues = load_cues()
    print("gold24 diagnostic — %d cues" % len(cues))
    print("essential nodes total: %d (gold+ %d / gold %d)  | needs total: %d"
          % (sum(len(c["ess"]) for c in cues), sum(len(c["gplus"]) for c in cues),
             sum(len(c["gold"]) for c in cues), sum(len(c["needs"]) for c in cues)))

    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        eng = FieldEngine(brain, model, cfg=LAFConfig(scale=8.0))   # converged settings (§18.18.1)
        master, idx, mats = eng.master, eng.idx, eng.mats
        adj = build_adjacency(brain, idx)
        ca = np.array([dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall()).get(n, "")
                       for n in master])

        if args.ablate:
            run_ablation(eng, cues, master, ca, brain)
            return

        # signals we rank under, per cue. "oracle" = best-field (ceiling, hindsight).
        FIELDS = list(MAXSIM_GROUPS)
        rows = []           # per-need diagnostic records
        ab = defaultdict(list)   # ranker → list of per-cue metric dicts

        for c in cues:
            qv = query_vec(c["query"])
            elig = (ca != "") & (ca <= c["cutoff"])
            cue_metrics = {}

            if qv is None:                       # unembeddable cue → all-miss (keeps denom honest)
                for rk in ("pipeline", "primary", "maxsim", "laf", "oracle"):
                    ab[rk].append({"hit5": 0, "hit25": 0, "hit5_gp": 0, "need_r5": 0.0, "ndcg5": 0.0,
                                   "source": c["source"], "qt": c["query_type"]})
                continue

            # ----- field/operator score vectors over the full master -----
            prim = primary_field(qv, mats)
            ms = maxsim_field(qv, mats, FIELDS)
            per_field = {vt: (mats[vt] @ qv) for vt in FIELDS}
            a_laf, _ = eng.recall(qv, elig)
            g1 = graph_spread((ms > np.nanpercentile(ms[np.isfinite(ms)], 90)).astype(np.float64),
                              adj, hops=1)        # 1-hop reach from the top-decile MaxSim seeds

            # oracle (best-field) ranked list: each node by its max field cosine = ms
            laf_ranked = ranked_ids(a_laf, master)
            prim_ranked = [master[i] for i in np.argsort(-np.where(elig & np.isfinite(prim), prim, -np.inf))]
            ms_ranked = [master[i] for i in np.argsort(-np.where(elig & np.isfinite(ms), ms, -np.inf))]

            # pipeline = production brain.recall (top-120)
            res = brain.recall(query=c["query"], filter={"created_at": {"lte": c["cutoff"]}},
                               limit=BURIED_MAX, session_id="g24-%s" % c["id"])
            pipe_ids = [r.get("id") for r in (res.get("results", []) if isinstance(res, dict) else [])]
            pipe_pos = {nid: i + 1 for i, nid in enumerate(pipe_ids)}

            # ----- A/B cue-level metrics per ranker -----
            def cue_metric(ranked_or_pos, is_pos=False):
                if is_pos:
                    rk = {nid: ranked_or_pos.get(nid) for nid in c["ess"]}
                    rk_gp = {nid: ranked_or_pos.get(nid) for nid in c["gplus"]}
                    ranked_list = pipe_ids
                else:
                    pos = {nid: i + 1 for i, nid in enumerate(ranked_or_pos)}
                    rk = {nid: pos.get(nid) for nid in c["ess"]}
                    rk_gp = {nid: pos.get(nid) for nid in c["gplus"]}
                    ranked_list = ranked_or_pos
                need_hits = sum(1 for nids in c["needs"].values()
                                if any((rk.get(n) or 1e9) <= 5 for n in nids))
                return {"hit5": hit_at(rk, 5), "hit25": hit_at(rk, 25), "hit5_gp": hit_at(rk_gp, 5),
                        "need_r5": need_hits / (len(c["needs"]) or 1), "ndcg5": ndcg5(ranked_list, c["tier_of"]),
                        "source": c["source"], "qt": c["query_type"]}

            ab["pipeline"].append(cue_metric(pipe_pos, is_pos=True))
            ab["primary"].append(cue_metric(prim_ranked))
            ab["maxsim"].append(cue_metric(ms_ranked))
            ab["laf"].append(cue_metric(laf_ranked))
            ab["oracle"].append(cue_metric(ms_ranked))   # ms == per-node max-field == best-field oracle

            # ----- per-NEED diagnostic (realizable: best of primary/maxsim/laf) -----
            laf_pos = {nid: i + 1 for i, nid in enumerate(laf_ranked)}
            prim_pos = {nid: i + 1 for i, nid in enumerate(prim_ranked)}
            ms_pos = {nid: i + 1 for i, nid in enumerate(ms_ranked)}
            for need, nids in c["needs"].items():
                best = {"rank": None, "via": None, "node": None}
                per_node = {}
                for nid in nids:
                    i = idx.get(nid)
                    in_master = i is not None
                    sig = {
                        "primary": prim_pos.get(nid), "maxsim": ms_pos.get(nid),
                        "laf": laf_pos.get(nid), "pipeline": pipe_pos.get(nid),
                    }
                    # best realizable (exclude pipeline — it's the production ref, capped at 120)
                    for via in ("primary", "maxsim", "laf"):
                        r = sig[via]
                        if r is not None and (best["rank"] is None or r < best["rank"]):
                            best = {"rank": r, "via": via, "node": nid}
                    # which single field ranks this node best (recovery lever)
                    field_ranks = {vt: rank_of(per_field[vt], elig, i) for vt in FIELDS} if in_master else {}
                    fr = {k: v for k, v in field_ranks.items() if v is not None}
                    best_field = min(fr, key=fr.get) if fr else None
                    per_node[nid] = {"in_master": in_master, "sig": sig,
                                     "best_field": best_field, "best_field_rank": fr.get(best_field),
                                     "g1_rank": rank_of(g1, elig, i) if in_master else None,
                                     "lens": c["lens"].get(nid, [])}
                # classify the need by its best realizable rank
                br = best["rank"]
                any_master = any(per_node[n]["in_master"] for n in nids)
                if br is not None and br <= TOP5:
                    cls = "hit"
                elif br is not None and br <= BURIED_MAX:
                    cls = "buried"            # rank-miss (recoverable)
                elif not any_master:
                    cls = "reach_novec"       # no embedding at all
                else:
                    cls = "reach_flat"        # exists+embeds but cosine-far
                # form (take the gold/gold+ form of any node in the need)
                form = next((c["form_of"].get(n) for n in nids if c["form_of"].get(n)), "") or "—"
                tier = "gold_plus" if (nids & c["gplus"]) else "gold"
                # lens class: cosine-reachable vs structural-only
                all_lens = set().union(*[set(per_node[n]["lens"]) for n in nids]) if nids else set()
                cosine_lens = all_lens & {"cos_cue", "cos_outcome", "fts"}
                lens_cls = "cosine" if cosine_lens else ("structural" if all_lens else "—")
                # robustness: how many realizable signals put it ≤5
                nsig5 = 0
                for via in ("primary", "maxsim", "laf"):
                    if any((per_node[n]["sig"][via] or 1e9) <= 5 for n in nids):
                        nsig5 += 1
                rows.append({
                    "cue": c["id"], "source": c["source"], "qt": c["query_type"],
                    "need": need, "nodes": sorted(nids), "tier": tier, "form": form,
                    "class": cls, "best_rank": br, "best_via": best["via"],
                    "lens_cls": lens_cls, "lens": sorted(all_lens), "nsig5": nsig5,
                    "recover_field": (per_node[best["node"]]["best_field"] if best["node"] else None),
                    "g1": any(per_node[n]["g1_rank"] is not None and per_node[n]["g1_rank"] <= TOP5 for n in nids),
                })

    report(cues, ab, rows)


# ───────────────────────────── reporting ─────────────────────────────
def _mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return sum(v) / len(v) if v else 0.0


def report(cues, ab, rows):
    out = []
    def p(s=""):
        print(s); out.append(s)

    n = len(cues)
    p("\n================  A/B BASELINE on the lens-independent 24-cue gold  ================")
    p("  (need_r5 = needs-met@5, need-collapsed · hit5_gp = Gold+ hit@5 · nDCG@5 tier-graded)")
    p("  %-10s %-8s %-8s %-9s %-9s %-8s" % ("ranker", "hit@5", "hit@25", "need@5", "gold+@5", "nDCG@5"))
    order = ["pipeline", "primary", "maxsim", "laf", "oracle"]
    label = {"pipeline": "pipeline", "primary": "raw _primary", "maxsim": "MaxSim-6grp",
             "laf": "LAF settle", "oracle": "best-field*"}
    for rk in order:
        r = ab[rk]
        p("  %-10s %-8s %-8s %-9s %-9s %-8.2f"
          % (label[rk], "%.0f%%" % (100 * _mean(r, "hit5")), "%.0f%%" % (100 * _mean(r, "hit25")),
             "%.0f%%" % (100 * _mean(r, "need_r5")), "%.0f%%" % (100 * _mean(r, "hit5_gp")),
             _mean(r, "ndcg5")))
    p("  * best-field is an ORACLE (per-node max field, hindsight) = the ceiling, not deployable.")

    # by source
    p("\n  -- hit@5 (essential) by source --")
    for src in ("anchor_turn", "operator_msg"):
        seg = {rk: [m for m in ab[rk] if m["source"] == src] for rk in order}
        cells = "  ".join("%s %.0f%%" % (label[rk][:8], 100 * _mean(seg[rk], "hit5")) for rk in order)
        p("  %-13s n=%d  %s" % (src, len(seg["primary"]), cells))

    # ---- the miss-diagnostic ----
    total = len(rows)
    cc = Counter(r["class"] for r in rows)
    p("\n================  MISS DIAGNOSTIC  (per essential need, n=%d)  ================" % total)
    p("  outcome (best of primary/maxsim/laf):")
    for cls, lab in (("hit", "HIT ≤5"), ("buried", "BURIED 6–120 (rank-miss, recoverable)"),
                     ("reach_flat", "REACH-MISS: embeds but cosine-far"),
                     ("reach_novec", "REACH-MISS: no vector (encode/embed absent)")):
        p("    %-38s %3d  (%.0f%%)" % (lab, cc.get(cls, 0), 100 * cc.get(cls, 0) / (total or 1)))

    # form × outcome — THE headline (does recall miss redirect more than ground?)
    p("\n  FORM × outcome  (the redirect-vs-ground hypothesis):")
    p("    %-10s %5s %6s %6s %6s   %s" % ("form", "n", "hit", "buried", "reach", "hit%"))
    byform = defaultdict(list)
    for r in rows:
        byform[r["form"]].append(r)
    for form in sorted(byform, key=lambda f: -len(byform[f])):
        rs = byform[form]
        h = sum(1 for r in rs if r["class"] == "hit")
        b = sum(1 for r in rs if r["class"] == "buried")
        rm = sum(1 for r in rs if r["class"].startswith("reach"))
        p("    %-10s %5d %6d %6d %6d   %.0f%%" % (form, len(rs), h, b, rm, 100 * h / len(rs)))

    # lens-class × outcome — does structural-only correlate with reach-miss?
    p("\n  JUDGE LENS × outcome  (how the blind judge found it → what reaches it):")
    p("    %-12s %5s %6s %6s %6s" % ("lens", "n", "hit", "buried", "reach"))
    bylens = defaultdict(list)
    for r in rows:
        bylens[r["lens_cls"]].append(r)
    for lc in ("cosine", "structural", "—"):
        rs = bylens.get(lc, [])
        if not rs:
            continue
        h = sum(1 for r in rs if r["class"] == "hit")
        b = sum(1 for r in rs if r["class"] == "buried")
        rm = sum(1 for r in rs if r["class"].startswith("reach"))
        p("    %-12s %5d %6d %6d %6d" % (lc, len(rs), h, b, rm))

    # recoverable buried: which field would pull it to top-5
    buried = [r for r in rows if r["class"] == "buried"]
    if buried:
        rf = Counter(r["recover_field"] for r in buried if r["recover_field"])
        g1n = sum(1 for r in buried if r["g1"])
        p("\n  RANK-MISS recovery (%d buried needs): best single field that ranks them —" % len(buried))
        p("    " + ", ".join("%s×%d" % (f, n) for f, n in rf.most_common()))
        p("    graph-1hop reaches ≤5: %d/%d buried" % (g1n, len(buried)))

    # encode-gap residual (judge-flagged, read-side-unreachable)
    gaps = sum(c["n_encode_gaps"] for c in cues)
    p("\n  ENCODE-GAP residual (judge-flagged needs with NO node): %d across %d cues"
      % (gaps, sum(1 for c in cues if c["n_encode_gaps"])))

    write_markdown(cues, ab, rows, out, order, label)
    p("\n  → full per-cue detail written to %s" % os.path.relpath(OUT_MD))


def write_markdown(cues, ab, rows, console, order, label):
    L = ["# LAF A/B + miss-diagnostic — lens-independent 24-cue gold (§18.21)\n",
         "Auto-generated by `eval/laf/gold24_diagnostic.py`. New gold = §18.20 four-tier "
         "(Gold+/Gold/Silver+/Silver), need-collapsed, tier-graded. **Not comparable to the old "
         "circular-corpus 21/37** (broader gold: three-form helpfulness, no availability subtraction).\n",
         "## Summary (console)\n", "```\n" + "\n".join(console) + "\n```\n",
         "## Per-cue needs\n"]
    byc = defaultdict(list)
    for r in rows:
        byc[r["cue"]].append(r)
    for c in cues:
        rs = byc.get(c["id"], [])
        L.append("### `%s`  (%s · %s)\n" % (c["id"], c["source"], c["query_type"]))
        L.append("> %s\n" % (c["query"][:240].replace("\n", " ") + ("…" if len(c["query"]) > 240 else "")))
        if not rs:
            L.append("_(no essential needs)_\n"); continue
        L.append("| need | tier | form | outcome | best rank (via) | lens | recover |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rs:
            L.append("| %s | %s | %s | **%s** | %s%s | %s | %s |" % (
                (r["need"][:70] + "…") if len(r["need"]) > 70 else r["need"],
                "G+" if r["tier"] == "gold_plus" else "G", r["form"], r["class"],
                r["best_rank"] if r["best_rank"] is not None else "—",
                " (%s)" % r["best_via"] if r["best_via"] else "",
                "+".join(r["lens"]) or "—",
                (r["recover_field"] or "") if r["class"] == "buried" else ""))
        L.append("")
    open(OUT_MD, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
