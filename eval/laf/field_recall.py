#!/usr/bin/env python3
"""LAF settling recall — the recurrent activation-field engine (the experimental recall stub).

This is the function all LAF experiments run through. The mechanism (docs §18 + node 15d62a95):

    static seed   s = Σ_k gain_k · zscore(op_k)        # operator-experts at LOGIT level,
                                                        # standardized so no field hijacks the sum
    settle        a_0      = α-entmax(s)
                  a_{t+1}  = α-entmax( s + gain_g · zscore(graph_spread(a_t)) )   # the recurrence
                  until ||a_{t+1} − a_t||_1 < eps  or  max_iters
    readout       the nonzero support of the settled a, ranked by activation

Why recurrent: α-entmax is monotonic, so it can't change a STATIC ranking — its leverage is
(a) the sparse commit/readout and (b) gating the recurrence: only confident (non-zero) nodes
spread, so graph-spread redeems MaxSim's reach without re-flooding the field with noise.

Operators (all verified by verify_substrate.py before this is trusted):
  - MaxSim-cosine over field-groups  (query↔node, query-dependent)
  - temporal-distinctiveness         (node-prior, query-independent)
  - typed-graph-spread               (the recurrent operator)

Run the A/B (after the gate is green):
  ./dev python3 eval/laf/field_recall.py
  ./dev python3 eval/laf/field_recall.py --alpha 1.5 --gain-graph 0.5 --gain-temporal 0.3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle_audit"))

from tests.isolated_brain import IsolatedBrain                       # noqa: E402
from servers import embedder                                          # noqa: E402
from endo_baseline_recall import load_corpus, make_baseline_ranker, score_one  # noqa: E402
from operators import (                                               # noqa: E402
    MAXSIM_GROUPS, query_vec, build_field_matrices, maxsim_field, primary_field,
    build_adjacency, graph_spread, parse_days, temporal_distinctiveness,
)

# True -inf, not a large finite sentinel: alpha_entmax excludes non-finite scores from its
# bisection bracket. A finite sentinel (-1e30) poisons the bracket's ptp → bisection precision
# collapses → entmax returns near-uniform regardless of temperature (the bug the sweep exposed).
NEG = -np.inf


class LAFConfig:
    def __init__(self, alpha=1.5, scale=10.0, gain_maxsim=1.0, gain_temporal=0.3, gain_graph=0.5,
                 hops=1, window_days=7.0, max_iters=20, eps=1e-3):
        self.alpha = alpha
        self.scale = scale          # logit temperature — z-scored ops are unit-variance, so
                                    # entmax needs the spread amplified to actually go sparse
                                    # (without it, support ≈ whole field and the recurrence is inert)
        self.gain_maxsim = gain_maxsim
        self.gain_temporal = gain_temporal
        self.gain_graph = gain_graph
        self.hops = hops
        self.window_days = window_days
        self.max_iters = max_iters
        self.eps = eps


def alpha_entmax(scores, alpha=1.5, iters=50):
    """Tsallis α-entmax over a 1-D score vector → a SPARSE prob dist (sums to 1).

    p_i = [ (α-1)·z_i − τ ]_+ ^ (1/(α-1)), τ chosen so Σ p = 1 (bisection on τ).
    α=1 → softmax (dense); α=2 → sparsemax; 1.5 sits between. −inf scores → exact 0.
    Hand-rolled (no entmax dep). Shift-equivariant, so we subtract max(z) for stability.
    """
    if alpha <= 1.0:                      # α=1 IS softmax; 1/(α-1) is singular — guard the limit
        return softmax(np.asarray(scores, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    fin = np.isfinite(s)
    out = np.zeros_like(s)
    if not fin.any():
        return out
    am1 = alpha - 1.0
    z = s - np.max(s[fin])
    zz = am1 * z
    zz[~fin] = -np.inf
    hi = float(np.max(zz[fin]))                       # τ=hi → Σp≈0
    lo = hi - (float(np.ptp(zz[fin])) + 1.0)          # τ=lo → Σp>1
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = np.clip(zz - mid, 0.0, None)
        v[~fin] = 0.0
        if float(np.sum(v ** (1.0 / am1))) > 1.0:
            lo = mid
        else:
            hi = mid
    v = np.clip(zz - 0.5 * (lo + hi), 0.0, None)
    v[~fin] = 0.0
    p = v ** (1.0 / am1)
    tot = p.sum()
    return (p / tot) if tot > 0 else out


def softmax(x):
    """Masked softmax: −inf → 0. The CONTRACTIVE in-loop nonlinearity (modern Hopfield):
    keeps the field dense+bounded each step so the recurrence settles instead of collapsing."""
    fin = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if not fin.any():
        return out
    e = np.zeros_like(out)
    e[fin] = np.exp(x[fin] - float(np.max(x[fin])))
    s = float(e.sum())
    return (e / s) if s > 0 else out


class FieldEngine:
    """Holds the precomputed substrate (field matrices, adjacency, node times) and runs
    one settling recall per (query, eligibility). Build once, recall many."""

    def __init__(self, brain, model, groups=MAXSIM_GROUPS, cfg=None):
        self.cfg = cfg or LAFConfig()
        self.groups = groups
        self.master, self.idx, self.mats = build_field_matrices(brain, model, groups)
        self.adj = build_adjacency(brain, self.idx)
        ca = dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall())
        self.days = parse_days([ca.get(nid, "") for nid in self.master])

    @staticmethod
    def _z(x, mask):
        """Standardize x over the masked, finite entries → commensurate scale; 0 elsewhere."""
        out = np.zeros(len(x), dtype=np.float64)
        m = mask & np.isfinite(x)
        if int(np.sum(m)) < 2:
            return out
        mu, sd = float(np.mean(x[m])), float(np.std(x[m]))
        if sd > 1e-9:
            out[m] = (x[m] - mu) / sd
        return out

    def recall(self, qv, eligible):
        cfg = self.cfg
        ms = maxsim_field(qv, self.mats, self.groups)                        # query-dependent
        temp = temporal_distinctiveness(self.days, eligible, cfg.window_days)   # node-prior
        base = cfg.gain_maxsim * self._z(ms, eligible) + cfg.gain_temporal * self._z(temp, eligible)
        neg = ~eligible

        def sm(logit):
            v = logit.copy()
            v[neg] = -np.inf
            return softmax(v)

        # Contractive settling (the Hopfield form): softmax in-loop keeps the field dense+bounded,
        # and the spread is RAW (no in-loop z-score) so the loop can't manufacture a fresh outlier
        # every step. This is what makes it converge instead of collapsing to a wandering single node.
        a = sm(cfg.scale * base)
        iters_run, converged = 0, False
        for t in range(cfg.max_iters):
            spread = graph_spread(a, self.adj, hops=cfg.hops)
            a_new = sm(cfg.scale * (base + cfg.gain_graph * spread))
            iters_run = t + 1
            if float(np.sum(np.abs(a_new - a))) < cfg.eps:
                a = a_new
                converged = True
                break
            a = a_new
        # Commit readout: the SPARSE projection lives ONLY here, on the settled field — it sets the
        # recall-SET size, not the dynamics. hit@k is ranked by the settled dense field `a`.
        final = cfg.scale * (base + cfg.gain_graph * graph_spread(a, self.adj, hops=cfg.hops))
        final[neg] = -np.inf
        commit = alpha_entmax(final, cfg.alpha)
        diag = {"iters": iters_run, "converged": converged,
                "support": int(np.sum(commit > 0)), "support_soft": int(np.sum(a > 1e-6))}
        return a, diag


def ranked_ids(a, master):
    order = np.argsort(-a)
    return [master[i] for i in order]


def run_corpus(eng, corpus, qvs, eligs):
    """One full-corpus pass at eng.cfg. Returns aggregate hit-rate + settling stats.
    hit@k is over ALL cues (an unembeddable cue counts as a miss — no query, no recall),
    so the denominator stays comparable to the baselines; iters/support average over the
    cues actually run."""
    h5 = h25 = 0
    it = conv = supp = skipped = scored = 0
    n = len(corpus)
    for c in corpus:
        qv = qvs[c["id"]]
        if qv is None:
            skipped += 1
            continue
        a, diag = eng.recall(qv, eligs[c["id"]])
        m = score_one(ranked_ids(a, eng.master), c["gold_essential"], c.get("gold_helpful", []))
        h5 += m["hit5_ess"]; h25 += m["hit25_ess"]
        it += diag["iters"]; conv += int(diag["converged"]); supp += diag["support"]
        scored += 1
    d = scored or 1
    return {"h5": h5/n, "h25": h25/n, "iters": it/d, "conv": conv, "supp": supp/d,
            "n": n, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--gain-maxsim", type=float, default=1.0)
    ap.add_argument("--gain-temporal", type=float, default=0.3)
    ap.add_argument("--gain-graph", type=float, default=0.5)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--sweep", action="store_true", help="sweep logit temperature to find the sparsity regime")
    ap.add_argument("--ablate", action="store_true", help="toggle each operator's gain → marginal contribution")
    args = ap.parse_args()
    cfg = LAFConfig(alpha=args.alpha, scale=args.scale, gain_maxsim=args.gain_maxsim,
                    gain_temporal=args.gain_temporal, gain_graph=args.gain_graph,
                    hops=args.hops, max_iters=args.max_iters)

    print("LAF settling recall — A/B vs baselines")
    print("cfg: alpha=%.2f gains(ms=%.2f temp=%.2f graph=%.2f) hops=%d max_iters=%d"
          % (cfg.alpha, cfg.gain_maxsim, cfg.gain_temporal, cfg.gain_graph, cfg.hops, cfg.max_iters))

    corpus = load_corpus()
    n_zero = sum(1 for c in corpus if not c.get("gold_essential"))
    print("corpus: %d cues (%d with essential-gold, %d zero-gold = structural misses on hit@k_ess)"
          % (len(corpus), len(corpus) - n_zero, n_zero))
    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query="warm", limit=1)
        model = embedder.stats.get("model_name") or ""
        eng = FieldEngine(brain, model, cfg=cfg)
        master = eng.master
        _ca = dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall())
        ca = np.array([_ca.get(n, "") for n in master])
        # precompute query vectors + eligibility once (so a sweep doesn't re-embed).
        # eligibility EXCLUDES empty created_at: "" <= cutoff is lexicographically true, which
        # would silently admit undated nodes into every cue's pool.
        qvs = {c["id"]: query_vec(c["query"]) for c in corpus}
        eligs = {c["id"]: (ca != "") & (ca <= c["cutoff"]) for c in corpus}

        print("\n  baselines:  pipeline 19%/33%   raw _primary 21%/37%   (hit@5 / hit@25)")
        if args.ablate:
            # hit@k is τ-invariant (softmax monotonic), so fix τ; vary only the operator gains.
            eng.cfg.scale = 8.0
            configs = [("full", 1.0, 0.3, 0.5), ("− graph", 1.0, 0.3, 0.0),
                       ("− temporal", 1.0, 0.0, 0.5), ("maxsim only", 1.0, 0.0, 0.0)]
            print("  %-13s %-9s %-7s %-8s %-8s" % ("config", "support", "iters", "hit@5", "hit@25"))
            for name, gm, gt, gg in configs:
                eng.cfg.gain_maxsim, eng.cfg.gain_temporal, eng.cfg.gain_graph = gm, gt, gg
                r = run_corpus(eng, corpus, qvs, eligs)
                print("  %-13s %-9.0f %-7.1f %-8s %-8s"
                      % (name, r["supp"], r["iters"], "%.0f%%" % (100*r["h5"]), "%.0f%%" % (100*r["h25"])))
        else:
            scales = [1, 2, 4, 8, 16, 32] if args.sweep else [cfg.scale]
            print("  %-7s %-9s %-7s %-9s %-8s %-8s" % ("scale", "support", "iters", "conv", "hit@5", "hit@25"))
            for s in scales:
                eng.cfg.scale = s
                r = run_corpus(eng, corpus, qvs, eligs)
                print("  %-7.0f %-9.0f %-7.1f %-9s %-8s %-8s"
                      % (s, r["supp"], r["iters"], "%d/%d" % (r["conv"], r["n"]),
                         "%.0f%%" % (100*r["h5"]), "%.0f%%" % (100*r["h25"])))


if __name__ == "__main__":
    main()
