#!/usr/bin/env python3
"""Survivor-credit A/B — what the absorbed-id drop in the episodic lanes actually costs.

THE BUG (finding id:ad74941e): the pick/enc lanes harvest node ids from PAST surface and
encode traces, then map each to a live-matrix row. S2 consolidation absorbs A into live
survivor B and archives A, so A has no row: the lookup returns None and that moment's
evidence is dropped on the floor. The survivor never inherits the activation history its
own content earned, and the loss compounds with every consolidation cycle (340 archived-
with-survivor nodes in 2026-06, 687 today).

THE FIX (servers/recall_laf.py:role_rows): one batched `brain.resolve_live` walk over the
UNRESOLVED harvested ids maps each dead id to its survivor's row.

WHY THIS PROBE AND NOT composition_probe ALONE — three reasons the grid can't answer it:
  1. BOTH ARMS IN ONE RUN. The lanes are built twice from the SAME role records, the same
     matrices, the same brain; only the id→row map differs. A pre-run/post-run pair would
     also carry corpus drift and IsolatedBrain copy-time differences.
  2. THE SHIPPED COMPOSITION. composition_probe's grid has no row for what actually ships
     (gain_maxsim 1.0 · pick .5 · enc .3 · idf .5 · sit .5) and z-scores at kind='current'
     while production runs the K-store's z_norm='support'. Both normalizers are reported
     here so the verdict isn't an artifact of that choice.
  3. N=24. One cue is 4pp of need@k, so need@k alone cannot resolve a small effect. The
     GOLD-RANK DELTAS are the sensitive instrument: every gold node's rank under each arm.

Reads three views, because "did the overall metric move" is the least informative of them:
  · footprint     — how many harvested ids were dead, how many found a live survivor
  · affected cues — credit-touched (the fix fired) and gold-survivor (it could plausibly
                    move the metric: a gold node is itself the survivor of an absorbed one)
  · movement      — need@5/@25 overall and on each subset, plus per-gold rank deltas

Run: ./dev python3 eval/laf/survivor_credit_probe.py
Out: eval/laf/survivor_credit_probe.md
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from servers.recall_laf import DEFAULT_CONFIG, role_rows              # noqa: E402
from episodic_ops import episodic_roles                               # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field,
)
from composition_probe import build_title_tokens, build_situation_matrix, idf_lane  # noqa: E402
from laf_metrics import zscore, ranks, best_ranks, need_hit_at, brought_lost  # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "survivor_credit_probe.md")

EPI_WINDOW = ("window", DEFAULT_CONFIG["window_turns"])   # production ±1-turn moments
G = DEFAULT_CONFIG                                        # the shipped gains, verbatim
Z_KINDS = ("support", "current")                          # production K-store, then the
                                                          # historical probe normalizer


def lane(records, role, rows, n):
    """[n] activation vector: per node, the best similar-moment score in `role`.

    The production `_episodic_vectors` inner loop, with the id→row map injected —
    that map IS the arm, so the two arms differ in nothing else.
    """
    vec = np.zeros(n, dtype=np.float64)
    for r in records:
        s = r["score"]
        for node in set(r[role]):
            i = rows.get(node)
            if i is not None and s > vec[i]:
                vec[i] = s
    return vec


def plain_rows(ids, idx):
    """The PRE-fix id→row map: live-matrix lookup only, dead ids simply absent."""
    out = {}
    for nid in set(ids):
        i = idx.get(nid)
        if i is not None:
            out[nid] = i
    return out


def survivor_map(brain):
    """{survivor_id: {absorbed ids}} — the reverse of the `_sys_archived_survivor_id`
    pointer, for the gold-survivor subset (which gold nodes stand to inherit history)."""
    rev = defaultdict(set)
    for dead, surv in brain._nodes.conn.execute(
            "SELECT node_id, value FROM node_metadata_kv "
            "WHERE key = '_sys_archived_survivor_id'").fetchall():
        if surv:
            rev[surv].add(dead)
    return rev


def score(p, kind, pick, enc):
    """The SHIPPED laf_v1 composition over one cue's lanes, at normalizer `kind`."""
    n, elig = p["n"], p["elig"]
    return (G["gain_maxsim"] * zscore(p["ms"], elig, n, kind)
            + G["gain_pick"] * zscore(pick, elig, n, kind)
            + G["gain_enc"] * zscore(enc, elig, n, kind)
            + G["gain_idf"] * zscore(p["idf"], elig, n, kind)
            + G["gain_sit"] * zscore(p["sit"], elig, n, kind))


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        n = len(master)
        ca_rows = dict(brain._fts.conn.execute(
            "SELECT id, created_at FROM nodes").fetchall())
        ca = np.array([ca_rows.get(nid, "") or "" for nid in master])
        title_tok = build_title_tokens(brain, idx)
        sit_M, _ = build_situation_matrix(brain, idx, n, model)
        rev = survivor_map(brain)
        print("master %d · archived-with-survivor %d" % (n, sum(len(v) for v in rev.values())))

        # Occurrence counts answer "how much evidence moves"; DISTINCT node counts
        # answer "how many nodes are involved" — an id picked in 5 cues is 5
        # occurrences but 1 node, and conflating them overstates the blast radius.
        per = {}
        n_harvest = n_dead = n_credited = 0
        uniq_harvest, uniq_dead, uniq_credited, uniq_survivors = set(), set(), set(), set()
        live_ids = set(idx)
        archived_kinds = defaultdict(set)
        for c in cues:
            qv = query_vec(c["query"])
            if qv is None or not c["needs"]:
                continue
            records = episodic_roles(brain, c["query"], c["cutoff"], window=EPI_WINDOW)
            harvested = {nid for r in records for nid in r["picked"] + r["encoded"]}
            off = plain_rows(harvested, idx)
            on, _ = role_rows(brain, harvested, idx.get)
            n_harvest += len(harvested)
            n_dead += len(harvested) - len(off)
            n_credited += len(on) - len(off)
            dead_ids = harvested - set(off)
            uniq_harvest |= harvested
            uniq_dead |= dead_ids
            uniq_credited |= set(on) - set(off)
            uniq_survivors |= {master[r] for d, r in on.items() if d not in off}

            sit = np.zeros(n)
            if sit_M is not None:
                s = sit_M @ qv
                sit = np.where(np.isfinite(s), s, 0.0)
            gold = {nid for nids in c["needs"].values() for nid in nids}
            # The decisive check: did any restored credit land ON a gold node for THIS
            # cue? Zero here would mean a null INSTRUMENT (the corpus holds no case the
            # bug hurts), not a null EFFECT — a distinction need@k cannot make.
            gold_rows = {idx[g] for g in gold if g in idx}
            gold_credited = {d for d, row in on.items()
                             if d not in off and row in gold_rows}
            per[c["id"]] = {
                "n": n, "needs": c["needs"], "gold": gold,
                "elig": (ca != "") & (ca <= c["cutoff"]),
                "ms": maxsim_field(qv, mats, list(MAXSIM_GROUPS)),
                "idf": idf_lane(c["query"], title_tok, n),
                "sit": sit,
                "pick_off": lane(records, "picked", off, n),
                "enc_off": lane(records, "encoded", off, n),
                "pick_on": lane(records, "picked", on, n),
                "enc_on": lane(records, "encoded", on, n),
                # the fix FIRED on this cue: some dead id was credited to a live row
                "touched": len(on) > len(off),
                # a gold node is itself the survivor of an absorbed node — where
                # inherited history could plausibly move this cue's metric
                "gold_surv": bool(gold & set(rev)),
                # restored credit actually landed on this cue's gold — where it MUST
                "gold_credited": len(gold_credited),
            }

        # Why is each dead id not a live row? One batched classification pass.
        status = brain._nodes._live_status_bulk(uniq_dead)
        pointers = brain._nodes._survivor_pointers_bulk(uniq_dead)
        for d in uniq_dead:
            st = status.get(d)
            archived_kinds["absent from nodes table" if st is None else
                           "archived + survivor pointer" if pointers.get(d) else
                           "archived, retired (no survivor)"].add(d)

        lines = ["# Survivor-credit A/B — absorbed-id drop in the LAF episodic lanes",
                 "",
                 "%d cues · master %d nodes · shipped laf_v1 gains "
                 "(maxsim %.1f · pick %.1f · enc %.1f · idf %.1f · sit %.1f), ±%d-turn moments"
                 % (len(per), n, G["gain_maxsim"], G["gain_pick"], G["gain_enc"],
                    G["gain_idf"], G["gain_sit"], G["window_turns"]),
                 "",
                 "## Footprint — did the fix fire at all?",
                 "",
                 "| harvested role ids | dead (no live row) | credited to a survivor | "
                 "cues touched | cues with a survivor gold node | credits landing ON gold |",
                 "|---|---|---|---|---|---|",
                 "| %d | %d (%.1f%%) | %d (%.1f%%) | %d/%d | %d/%d | %d (in %d/%d cues) |"
                 % (n_harvest, n_dead, 100.0 * n_dead / max(n_harvest, 1),
                    n_credited, 100.0 * n_credited / max(n_harvest, 1),
                    sum(1 for p in per.values() if p["touched"]), len(per),
                    sum(1 for p in per.values() if p["gold_surv"]), len(per),
                    sum(p["gold_credited"] for p in per.values()),
                    sum(1 for p in per.values() if p["gold_credited"]), len(per)),
                 "",
                 "`credits landing ON gold` is the load-bearing column: zero would mean the "
                 "corpus holds no case this bug hurts — a null INSTRUMENT, not a null effect.",
                 "",
                 "Those are OCCURRENCES (an id harvested by 5 cues counts 5×) — the honest "
                 "measure of how much evidence moves. The DISTINCT nodes behind them:",
                 "",
                 "| distinct harvested | distinct dead | distinct credited | distinct survivors "
                 "receiving credit |",
                 "|---|---|---|---|",
                 "| %d | %d (%.1f%%) | %d | %d |"
                 % (len(uniq_harvest), len(uniq_dead),
                    100.0 * len(uniq_dead) / max(len(uniq_harvest), 1),
                    len(uniq_credited), len(uniq_survivors)),
                 "",
                 "### Why each dead id had no live row", "",
                 "| reason | distinct ids |", "|---|---|"]
        lines += ["| %s | %d |" % (k, len(v))
                  for k, v in sorted(archived_kinds.items(), key=lambda kv: -len(kv[1]))]

        subsets = [("ALL", lambda p: True),
                   ("credit-touched", lambda p: p["touched"]),
                   ("gold-survivor", lambda p: p["gold_surv"]),
                   ("gold-credited", lambda p: p["gold_credited"] > 0)]
        for kind in Z_KINDS:
            tag = "production K-store" if kind == "support" else "historical probe default"
            lines += ["", "## need@k — z_norm='%s' (%s)" % (kind, tag), "",
                      "| subset | cues | need@5 OFF → ON | need@25 OFF → ON | brought | lost |",
                      "|---|---|---|---|---|---|"]
            for name, keep in subsets:
                sel = [p for p in per.values() if keep(p)]
                if not sel:
                    lines.append("| %s | 0 | — | — | — | — |" % name)
                    continue
                h5 = [0.0, 0.0]
                h25 = [0.0, 0.0]
                brought = lost = 0
                for p in sel:
                    r_off = ranks(score(p, kind, p["pick_off"], p["enc_off"]),
                                  p["elig"], master)
                    r_on = ranks(score(p, kind, p["pick_on"], p["enc_on"]),
                                 p["elig"], master)
                    for j, rk in enumerate((r_off, r_on)):
                        h5[j] += need_hit_at(rk, p["needs"], 5) or 0.0
                        h25[j] += need_hit_at(rk, p["needs"], 25) or 0.0
                    b, l = brought_lost(r_on, p["needs"], best_ranks(r_off, p["needs"]))
                    brought += b
                    lost += l
                m = len(sel)
                lines.append("| %s | %d | %.0f%% → %.0f%% (%+.1fpp) | %.0f%% → %.0f%% "
                             "(%+.1fpp) | +%d | −%d |"
                             % (name, m, 100*h5[0]/m, 100*h5[1]/m,
                                100*(h5[1]-h5[0])/m, 100*h25[0]/m, 100*h25[1]/m,
                                100*(h25[1]-h25[0])/m, brought, lost))

        # ── the sensitive instrument: per-gold rank movement (N=24 can't resolve need@k) ──
        # Split by DEPTH: a gold node sitting at rank 2300 moving 34 places is noise in a
        # reach-miss; the only movement that can change what recall returns is near the top.
        NEAR = 50
        lines += ["", "## Gold-rank movement — every gold node, both arms", "",
                  "`near-top` = gold reaching rank ≤ %d in either arm (the only band where a "
                  "move can change what recall surfaces); `tail` is everything deeper." % NEAR,
                  "",
                  "| z_norm | band | gold nodes | improved | worsened | unchanged | "
                  "median Δrank | best Δ | worst Δ |",
                  "|---|---|---|---|---|---|---|---|---|"]
        movers, crossings = {}, {}
        for kind in Z_KINDS:
            bands, moved, cross = defaultdict(list), [], []
            for cid, p in per.items():
                r_off = ranks(score(p, kind, p["pick_off"], p["enc_off"]), p["elig"], master)
                r_on = ranks(score(p, kind, p["pick_on"], p["enc_on"]), p["elig"], master)
                for g in p["gold"]:
                    a, b = r_off.get(g), r_on.get(g)
                    if a is None or b is None:
                        continue
                    bands["near-top" if min(a, b) <= NEAR else "tail"].append(a - b)
                    if a != b:
                        moved.append((a - b, cid, g, a, b))
                # k-boundary crossings: the ONLY movement need@k can see
                b_off = best_ranks(r_off, p["needs"])
                b_on = best_ranks(r_on, p["needs"])
                for need, ro in b_off.items():
                    rn = b_on.get(need)
                    for k in (5, 25):
                        was = ro is not None and ro <= k
                        now = rn is not None and rn <= k
                        if was != now:
                            cross.append((k, "gained" if now else "LOST", cid,
                                          need[:44], ro, rn))
            for band in ("near-top", "tail"):
                d = np.array(bands.get(band) or [])
                if not len(d):
                    continue
                lines.append("| %s | %s | %d | %d | %d | %d | %+.0f | %+d | %+d |"
                             % (kind, band, len(d), int((d > 0).sum()), int((d < 0).sum()),
                                int((d == 0).sum()), float(np.median(d)),
                                int(d.max()), int(d.min())))
            movers[kind] = sorted(moved, key=lambda t: -abs(t[0]))[:12]
            crossings[kind] = cross

        for kind in Z_KINDS:
            cross = crossings.get(kind) or []
            lines += ["", "### need@k boundary crossings — z_norm='%s'" % kind, ""]
            if not cross:
                lines.append("None — no need changed side at k=5 or k=25. "
                             "The lanes moved; nothing crossed.")
                continue
            lines += ["| k | direction | cue | need | OFF rank | ON rank |",
                      "|---|---|---|---|---|---|"]
            lines += ["| %d | %s | %s | %s | %s | %s |"
                      % (k, d, cid, nd, ro if ro is not None else "—",
                         rn if rn is not None else "—")
                      for k, d, cid, nd, ro, rn in sorted(cross)]

        for kind, rows_ in movers.items():
            if not rows_:
                continue
            lines += ["", "### Largest gold-rank moves — z_norm='%s'" % kind, "",
                      "| Δrank | cue | gold node | OFF | ON |", "|---|---|---|---|---|"]
            lines += ["| %+d | %s | %s | %d | %d |" % (dl, cid, g, a, b)
                      for dl, cid, g, a, b in rows_]

        out = "\n".join(lines) + "\n"
        print(out)
        with open(OUT_MD, "w") as f:
            f.write(out)
        print("wrote %s" % OUT_MD)


if __name__ == "__main__":
    main()
