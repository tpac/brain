#!/usr/bin/env python3
"""laf_v1 COMPOSITION probe — production's lexical/situation channels as lanes + uncapped episodic.

Step-3 pre-wiring measurement (Tom, 2026-07-02). The measured winner (z(ms)+0.5·z(pick)
+0.3·z(enc), ±1-turn, 16/28) is BARE — production recall composes channels the stack lacks.
Before wiring BRAIN_RECALL_VARIANT=laf_v1, measure each candidate lane on the 24-cue honest
gold so the SHIPPED composition is chosen on evidence, not parity anxiety:

  fts    — BM25 over nodes_fts (title×10 + content), production's lexical DISCOVERY channel.
           Actual bm25 magnitudes (not rank-decay): probe-side SQL mirroring Fts5DAL.search
           with the score column exposed (shipping needs a search_scored DAL door).
  idf    — production idf2 title boost verbatim (same tokenizer regex, same stopword floor,
           same word-boundary log-idf), production's lexical RANK channel. Separate lane from
           fts DELIBERATELY: same law (rare-token discrimination), different support
           (title vs full text), different shape (capped bump vs graded bm25) — two gains,
           the P3 fit decides their mix (Tom's call, 2026-07-02).
  sit    — cos(query, _situation enrichment vectors): production STEP 3.5b as a lane.
           Prompt-cued today (production parity — situation_vec defaults to query_vec);
           kept OUT of maxsim's nanmax so P4 can re-cue it from φ(situation-state) later.
  epi_u  — pick+enc with the newest-500 scan cap ELIMINATED (Tom's directive): one SQL pull
           of ALL embedded conversational-s0 trace vectors before the cutoff, matrixized
           cosine (the 'optimize later' done now), same top-15 moments / ±1-turn windows.
           recall_episodes is monkeypatched so episodic_ops' role-join runs UNCHANGED.

Anchor row reproduces the capped 16/28 stack so every delta is attributable. New lanes enter
at static g=0.5 (the fusion probe's winner). Also reports the uncapped scan's per-query
latency (the number behind 'we can later optimize') and capped↔uncapped moment overlap.

Run: ./dev python3 eval/laf/composition_probe.py
Out: eval/laf/composition_probe.md
"""
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from servers.brain_constants import _TITLE_BOOST_STOPWORDS            # noqa: E402
from servers.trace_contract import CONVERSATIONAL_REF_TYPES           # noqa: E402
import episodic_ops                                                   # noqa: E402
from episodic_ops import (                                            # noqa: E402
    episodic_roles, episodic_encoded, episodic_picked, DEFAULT_TOP_MOMENTS,
)
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field, unit,
)
from laf_metrics import zscore, ranks, best_ranks, need_hit, need_bl  # noqa: E402
from gold24_diagnostic import load_cues                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "composition_probe.md")

EPI_WINDOW = ("window", 1)          # ±1-turn moments (the measured winner's window)
G = 0.5                              # static gain for each NEW lane (fusion probe winner)
_IDF_TOK = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")   # idf2's tokenizer, verbatim


# ───────────────────────── uncapped episodic seeding ─────────────────────────
class UncappedEpisodes:
    """recall_episodes stand-in: full-history matrixized scan, no newest-500 cap.

    Pulls ALL embedded conversational-s0 trace vectors once (blobs → one [T×768]
    matrix, rows already L2-normalized by fastembed so dot=cosine), then per query
    masks by created_at < cutoff and takes top-K. Same return shape as
    recall_episodes so episodic_ops' moment-dedup/role-join runs unchanged.
    """

    def __init__(self, brain):
        ph = ",".join("?" * len(CONVERSATIONAL_REF_TYPES))
        rows = brain._trace_dal.conn.execute(
            "SELECT te.chain_id, te.session_id, te.created_at, tem.vector "
            "FROM trace_events te JOIN trace_embeddings tem ON tem.trace_id = te.id "
            "WHERE te.scale='s0' AND te.ref_type IN (%s)" % ph,
            list(CONVERSATIONAL_REF_TYPES)).fetchall()
        rows = [r for r in rows if r[3]]
        self.meta = [(r[0], r[1]) for r in rows]
        self.created = np.array([r[2] for r in rows])
        self.M = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        self.timings = []

    def __call__(self, query=None, older_than=None, scale="s0", limit=None, **_):
        t0 = time.perf_counter()
        qv = unit(embedder.embed_query(query))
        limit = limit or DEFAULT_TOP_MOMENTS
        mask = self.created < older_than if older_than else np.ones(len(self.meta), bool)
        sims = np.where(mask, self.M @ qv, -np.inf)
        top = np.argsort(-sims)[:limit]
        eps = [{"chain_id": self.meta[i][0], "session_id": self.meta[i][1],
                "_score": float(sims[i])} for i in top if np.isfinite(sims[i])]
        self.timings.append(time.perf_counter() - t0)
        return {"episodes": eps, "ranked_by": "relevance", "truncated": False}


# ───────────────────────── production-channel lanes ─────────────────────────
def fts_lane(brain, query, idx, n):
    """BM25 scores over nodes_fts (title×10/content×1, live nodes) — Fts5DAL.search's
    exact query with the bm25 value exposed (negated: SQLite bm25 is lower=better)."""
    vec = np.zeros(n, dtype=np.float64)
    safe = brain._fts._sanitize_query(query)
    if not safe:
        return vec
    try:
        rows = brain._fts.conn.execute(
            "SELECT nodes_fts.node_id, -bm25(nodes_fts, 0, 10.0, 1.0) "
            "FROM nodes_fts JOIN nodes ON nodes.id = nodes_fts.node_id "
            "WHERE nodes_fts MATCH ? AND nodes.archived = 0 "
            "ORDER BY bm25(nodes_fts, 0, 10.0, 1.0) LIMIT 500", (safe,)).fetchall()
    except Exception:
        return vec
    for nid, s in rows:
        i = idx.get(nid)
        if i is not None and s > vec[i]:
            vec[i] = s
    return vec


def build_title_tokens(brain, idx):
    """{row → frozenset(title tokens)} with idf2's tokenizer, over live master nodes."""
    out = {}
    for nid, title in brain._fts.conn.execute(
            "SELECT id, title FROM nodes WHERE archived = 0").fetchall():
        i = idx.get(nid)
        if i is not None and title:
            out[i] = frozenset(_IDF_TOK.findall(title.lower()))
    return out


def idf_lane(query, title_tok, n):
    """Production idf2 title boost as a lane: per node, Σ log-idf of the query's
    rare tokens found in its title, normalized by the query's total idf mass."""
    vec = np.zeros(n, dtype=np.float64)
    q_tokens = {t for t in _IDF_TOK.findall(query.lower())
                if len(t) >= 2 and t not in _TITLE_BOOST_STOPWORDS}
    if not q_tokens or not title_tok:
        return vec
    n_titles = max(len(title_tok), 1)
    idf = {}
    for t in q_tokens:
        df = sum(1 for ts in title_tok.values() if t in ts)
        idf[t] = math.log((n_titles + 1) / (df + 1))
    total = sum(idf.values()) or 1.0
    for i, ts in title_tok.items():
        m = sum(idf[t] for t in q_tokens if t in ts)
        if m > 0:
            vec[i] = m / total
    return vec


def build_situation_matrix(brain, idx, n, model):
    """[n×dim] matrix of _situation enrichment vectors (NaN rows where absent)."""
    rows = brain._vec_dal.get_all_situations(model=model or None)
    dim = None
    for r in rows:
        uv = unit(r.get("situation_embedding") or r.get("embedding"))
        if uv is not None:
            dim = len(uv)
            break
    if dim is None:
        return None, 0
    M = np.full((n, dim), np.nan, dtype=np.float32)
    covered = 0
    for r in rows:
        i = idx.get(r["node_id"])
        uv = unit(r.get("situation_embedding") or r.get("embedding"))
        if i is not None and uv is not None:
            M[i] = uv
            covered += 1
    return M, covered


# ───────────────────────── the probe ─────────────────────────
def pct(vals, q):
    return float(np.percentile(np.asarray(vals, dtype=float), q)) if vals else 0.0


def main():
    cues = load_cues()
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        master, idx, mats = build_field_matrices(brain, model, list(MAXSIM_GROUPS))
        N = len(master)
        ca_rows = dict(brain._fts.conn.execute(
            "SELECT id, created_at FROM nodes").fetchall())
        ca = np.array([ca_rows.get(nid, "") or "" for nid in master])
        title_tok = build_title_tokens(brain, idx)
        sit_M, sit_cov = build_situation_matrix(brain, idx, N, model)
        uncapped = UncappedEpisodes(brain)
        print("master %d · situation coverage %d · trace matrix %s"
              % (N, sit_cov, uncapped.M.shape))

        per, base_ref = {}, {}
        overlap = []
        for c in cues:
            qv = query_vec(c["query"])
            if qv is None or not c["needs"]:
                continue
            elig = (ca != "") & (ca <= c["cutoff"])
            ms = maxsim_field(qv, mats, list(MAXSIM_GROUPS))

            # capped episodic (the measured winner's shape — the anchor)
            rc = episodic_roles(brain, c["query"], c["cutoff"], window=EPI_WINDOW)
            pick_c = episodic_picked(brain, c["query"], c["cutoff"], idx, N, _records=rc)
            enc_c = episodic_encoded(brain, c["query"], c["cutoff"], idx, N, _records=rc)

            # uncapped episodic (full-history scan) — same role-join, patched seeding
            real = brain.recall_episodes
            brain.recall_episodes = uncapped
            try:
                ru = episodic_roles(brain, c["query"], c["cutoff"], window=EPI_WINDOW)
            finally:
                brain.recall_episodes = real
            pick_u = episodic_picked(brain, c["query"], c["cutoff"], idx, N, _records=ru)
            enc_u = episodic_encoded(brain, c["query"], c["cutoff"], idx, N, _records=ru)
            overlap.append(_moment_overlap(rc, ru))

            fts = fts_lane(brain, c["query"], idx, N)
            idfv = idf_lane(c["query"], title_tok, N)
            sit = np.zeros(N)
            if sit_M is not None:
                s = sit_M @ qv
                sit = np.where(np.isfinite(s), s, 0.0)

            per[c["id"]] = {
                "elig": elig, "needs": c["needs"],
                "zms": zscore(ms, elig, N),
                "zpc": zscore(pick_c, elig, N), "zec": zscore(enc_c, elig, N),
                "zpu": zscore(pick_u, elig, N), "zeu": zscore(enc_u, elig, N),
                "zfts": zscore(fts, elig, N), "zidf": zscore(idfv, elig, N),
                "zsit": zscore(sit, elig, N),
            }
            base_ref[c["id"]] = best_ranks(ranks(ms, elig, master), c["needs"])
        nc = len(per) or 1

        CONFIGS = [
            ("maxsim (base)",       lambda p: p["zms"]),
            ("stack capped (ref)",  lambda p: p["zms"] + 0.5*p["zpc"] + 0.3*p["zec"]),
            ("stack UNCAPPED",      lambda p: p["zms"] + 0.5*p["zpu"] + 0.3*p["zeu"]),
            ("fts standalone",      lambda p: p["zfts"]),
            ("idf standalone",      lambda p: p["zidf"]),
            ("sit standalone",      lambda p: p["zsit"]),
            ("+ fts",               lambda p: p["zms"] + G*p["zfts"]),
            ("+ idf",               lambda p: p["zms"] + G*p["zidf"]),
            ("+ sit",               lambda p: p["zms"] + G*p["zsit"]),
            ("stack_u + fts",       lambda p: p["zms"] + 0.5*p["zpu"] + 0.3*p["zeu"]
                                              + G*p["zfts"]),
            ("stack_u + idf",       lambda p: p["zms"] + 0.5*p["zpu"] + 0.3*p["zeu"]
                                              + G*p["zidf"]),
            ("stack_u + sit",       lambda p: p["zms"] + 0.5*p["zpu"] + 0.3*p["zeu"]
                                              + G*p["zsit"]),
            ("stack_u + fts+idf",   lambda p: p["zms"] + 0.5*p["zpu"] + 0.3*p["zeu"]
                                              + G*p["zfts"] + G*p["zidf"]),
            ("stack_u + all three", lambda p: p["zms"] + 0.5*p["zpu"] + 0.3*p["zeu"]
                                              + G*p["zfts"] + G*p["zidf"] + G*p["zsit"]),
            # the capped stack is the @5 winner — the marginal question is what
            # ADDS to it (the first run only combined lanes with stack_u)
            ("stack_c + fts",       lambda p: p["zms"] + 0.5*p["zpc"] + 0.3*p["zec"]
                                              + G*p["zfts"]),
            ("stack_c + idf",       lambda p: p["zms"] + 0.5*p["zpc"] + 0.3*p["zec"]
                                              + G*p["zidf"]),
            ("stack_c + sit",       lambda p: p["zms"] + 0.5*p["zpc"] + 0.3*p["zec"]
                                              + G*p["zsit"]),
            ("stack_c + idf+sit",   lambda p: p["zms"] + 0.5*p["zpc"] + 0.3*p["zec"]
                                              + G*p["zidf"] + G*p["zsit"]),
            # half-gain arms — is fts/sit salvageable at lower influence?
            ("stack_c + fts@.25",   lambda p: p["zms"] + 0.5*p["zpc"] + 0.3*p["zec"]
                                              + 0.25*p["zfts"]),
            ("stack_c + sit@.25",   lambda p: p["zms"] + 0.5*p["zpc"] + 0.3*p["zec"]
                                              + 0.25*p["zsit"]),
            ("stack_c + idf@.25",   lambda p: p["zms"] + 0.5*p["zpc"] + 0.3*p["zec"]
                                              + 0.25*p["zidf"]),
            # two episodic WINDOWS as two lanes — recency field + history field
            ("stack_c + u-lanes@.25", lambda p: p["zms"] + 0.5*p["zpc"] + 0.3*p["zec"]
                                                + 0.25*p["zpu"] + 0.15*p["zeu"]),
        ]

        lines = ["# laf_v1 composition probe — production channels as lanes + uncapped episodic",
                 "",
                 "%d cues · master %d · situation coverage %d nodes · trace matrix %d×%d"
                 % (nc, N, sit_cov, *uncapped.M.shape),
                 "",
                 "| config | need@5 | need@25 | brought | lost |",
                 "|---|---|---|---|---|"]
        print("\n  %-20s %-7s %-8s | %-8s %s" % ("config", "need@5", "need@25", "brought", "lost"))
        for name, fn in CONFIGS:
            h5 = h25 = brought = lost = 0
            for c in cues:
                p = per.get(c["id"])
                if p is None:
                    continue
                sc = fn(p)
                h5 += need_hit(sc, p["elig"], master, p["needs"], 5)
                h25 += need_hit(sc, p["elig"], master, p["needs"], 25)
                if name != "maxsim (base)":
                    b, l = need_bl(sc, p["elig"], master, p["needs"], base_ref[c["id"]])
                    brought += b
                    lost += l
            row = (name, "%.0f%%" % (100*h5/nc), "%.0f%%" % (100*h25/nc), brought, lost)
            print("  %-20s %-7s %-8s | +%-7d −%d" % row)
            lines.append("| %s | %s | %s | +%d | −%d |" % row)

        lat = uncapped.timings
        lines += ["",
                  "Uncapped scan latency: p50 %.0fms / p95 %.0fms per query (%d calls, %d traces)"
                  % (1000*pct(lat, 50), 1000*pct(lat, 95), len(lat), uncapped.M.shape[0]),
                  "Capped↔uncapped moment overlap (of %d): p50 %.1f" %
                  (DEFAULT_TOP_MOMENTS, pct(overlap, 50))]
        print("\n".join(lines[-3:]))
        with open(OUT_MD, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote %s" % OUT_MD)


def _moment_overlap(rc, ru):
    """How many of the capped run's moments the uncapped run also seeded from —
    keyed by the role fingerprint (same picked/encoded sets = same moment)."""
    key = lambda r: (tuple(r["picked"]), tuple(r["encoded"]))  # noqa: E731
    return len({key(r) for r in rc} & {key(r) for r in ru})


if __name__ == "__main__":
    main()
