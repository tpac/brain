"""THE DECISIVE TEST (path "b", 2026-07-18): does a per-MESSAGE-routed
composition beat the fixed additive z-blend on held-out gold-reach@5 + soft_r,
or does the fixed z-blend's per-candidate lane competition already capture the
routing? And what is the ORACLE ceiling — if even a perfect per-turn router
barely beats the blend, path "a" (keep both lanes, fixed gains) wins regardless
of any signal's quality.

Baseline: the fixed-gain fitted blend = definitive_fit's S_content (maxsim/sit/
idf × op0..8 + anchor1..8, + M_e_f). Decomposed per candidate into
  cur_part  = Σ w·z(lane, op j0)          # this message
  hist_part = Σ w·z(lane, op/anchor j>=1) # the moment stack
  me_part   = w·fatigue
score_blend = cur_part + hist_part + me_part.

Routed composition tilts the blend per turn by a recall-time signal s
(standardized): score = cur_part·(1+β·s) + hist_part·(1-β·s) + me_part
             = score_blend + β·s·(cur_part - hist_part).
β is fit on TRAIN folds (grid, maximizing reach@5); s ∈ {cur_maxz, rel_conf}.
β=0 recovers the blend, so any lift is pure routing value.

Oracle router (ceiling, uses the gold — unrealizable): per turn pick the
cur/hist weighting from a menu that ranks the gold best. Measures the HEADROOM
a perfect router could ever capture over the blend.

Protocol: group(session) 5-fold CV — fit blend gains + β on 4 folds, evaluate
on the held fold, pool. Works identically on live and pool60 (pool60 has no era
split). Metric: gold-reach@5 (pool-resident re-rank, rank<5 among the ~25
candidates — NOT out-of-pool reach) + soft_r (pooled corr score vs soft_max).

Run: ./dev python3 eval/laf/walker/router_decisive_probe.py
     WALKER_OUT_DIR=.../0a9baa/walker ./dev python3 .../router_decisive_probe.py
"""
import sys
from pathlib import Path

import numpy as np

from walker_db import open_walker, WALKER_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import _zscore                              # noqa: E402
from q1_sweep import load, gate_provenance                          # noqa: E402
from p3_fit import fit_logistic                                     # noqa: E402
from definitive_fit import turn_features, FEATURES                 # noqa: E402
from episodic_roles import K_MAX                                    # noqa: E402

SOFT_MARGIN = 0.10                # matches definitive_fit.pairs_soft
BETAS = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
# oracle cur/hist weighting menu (blend, cur-lean, hist-lean, cur-only, hist-only)
MENU = [(1.0, 1.0), (1.5, 0.5), (0.5, 1.5), (1.0, 0.0), (0.0, 1.0)]


def pairs_soft_folds(fold_feats):
    """Soft-usage winner-loser diff rows over an EXPLICIT turn list (the CV
    train folds) — unlike definitive_fit.pairs_soft it does NOT filter td.val,
    because the fold membership IS the train/eval split (pool60 is all-val)."""
    rows = []
    for td, X in fold_feats:
        fin = np.flatnonzero(np.isfinite(td.soft))
        if len(fin) < 2:
            continue
        s = td.soft[fin]
        d = s[:, None] - s[None, :]
        wi, li = np.nonzero(d >= SOFT_MARGIN)
        if len(wi):
            rows.append(X[fin[wi]] - X[fin[li]])
    return np.concatenate(rows) if rows else np.empty((0, len(FEATURES)))


def col_roles():
    """content column index -> role; returns (cur_cols, hist_cols, me_col)."""
    content = [i for i, f in enumerate(FEATURES)
               if not f.startswith(('pick·', 'enc·'))]
    cur, hist, me = [], [], None
    for i in content:
        f = FEATURES[i]
        if f == 'M_e_f':
            me = i
        elif int(f.split('·')[1].lstrip('opanchor')) == 0:
            cur.append(i)
        else:
            hist.append(i)
    return content, cur, hist, me


SIG_NAMES = ['curmaxz', 'relconf', 'hasq']    # realizable recall-time signals


def turn_signals(td, hasq):
    """dict of recall-time routing signals for a turn (no gold used)."""
    nc = len(td.cands)
    zc = _zscore(td.op['maxsim'][:, 0], nc)
    h = np.concatenate([td.op['maxsim'][:, 1:K_MAX + 1],
                        td.anchor['maxsim'][:, 1:K_MAX + 1]], axis=1)
    with np.errstate(all='ignore'):
        hl = np.where(np.all(np.isnan(h), axis=1), np.nan, np.nanmax(h, axis=1))
    zh = _zscore(hl, nc)
    cmz, hmz = float(zc.max()), float(zh.max())
    return {'curmaxz': cmz, 'relconf': cmz - hmz,
            'hasq': float(hasq.get(td.key, 0))}


def gold_of(td, hi):
    if not np.isfinite(td.soft).any():
        return None
    g = int(np.nanargmax(td.soft))
    return g if td.soft[g] >= hi else None


def reach5(rank):
    return float(np.mean([r < 5 for r in rank]))


def soft_r(scores, softs):
    x = np.concatenate(scores)
    y = np.concatenate(softs)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1])


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    hasq = {}
    for sess, epoch, seq, hq in walker.execute(
            "SELECT session_id, epoch, seq, has_question FROM turns"):
        hasq[(sess, epoch, seq)] = int(hq or 0)
    walker.close()
    feats = [(td, turn_features(td)) for td in turns]
    _, cur_cols, hist_cols, me_col = col_roles()

    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)

    # group(session) 5-fold
    sessions = sorted({td.key[0] for td in turns})
    rng = np.random.RandomState(0)
    rng.shuffle(sessions)
    fold_of = {s: i % 5 for i, s in enumerate(sessions)}

    route_keys = ['route_' + s for s in SIG_NAMES]
    R = {k: [] for k in (['blend', 'cur_only', 'hist_only', 'oracle']
                         + route_keys)}
    S = {k: ([], []) for k in R}          # (scores, softs) for soft_r
    beta_used = {k: [] for k in route_keys}
    n_gold = 0
    oracle_changed = 0

    for f in range(5):
        train = [(td, X) for td, X in feats if fold_of[td.key[0]] != f]
        held = [(td, X) for td, X in feats if fold_of[td.key[0]] == f]
        # fit S_content blend gains on train soft pairs
        content = cur_cols + hist_cols + [me_col]
        D = pairs_soft_folds(train)
        w = fit_logistic(D[:, content])
        wmap = dict(zip(content, w))
        wcur = np.array([wmap[c] for c in cur_cols])
        whist = np.array([wmap[c] for c in hist_cols])
        wme = wmap[me_col]

        def decompose(X):
            return (X[:, cur_cols] @ wcur, X[:, hist_cols] @ whist,
                    X[:, me_col] * wme)

        # signal train stats for per-fold standardization (train golds only)
        train_sig = {s: [] for s in SIG_NAMES}
        for td, X in train:
            if gold_of(td, hi) is None:
                continue
            sg = turn_signals(td, hasq)
            for s in SIG_NAMES:
                train_sig[s].append(sg[s])
        mu = {s: float(np.mean(v)) for s, v in train_sig.items()}
        sd = {s: float(np.std(v) or 1.0) for s, v in train_sig.items()}

        def std_sig(td, name):
            return (turn_signals(td, hasq)[name] - mu[name]) / sd[name]

        def routed(cur, car, me, td, name, b):
            s = std_sig(td, name)
            return cur * (1 + b * s) + car * (1 - b * s) + me

        # fit routing β per signal on TRAIN reach@5 (β=0 ≡ blend, so any pick
        # ≠0 must EARN it on train; noise picks ~0 / flip sign across folds)
        betas = {}
        for name in SIG_NAMES:
            best_b, best_r = 0.0, -1
            for b in BETAS:
                ranks = []
                for td, X in train:
                    g = gold_of(td, hi)
                    if g is None:
                        continue
                    cur, car, me = decompose(X)
                    sc = routed(cur, car, me, td, name, b)
                    ranks.append(int((sc > sc[g]).sum()))
                r = reach5(ranks) if ranks else -1
                if r > best_r:
                    best_r, best_b = r, b
            betas[name] = best_b
            beta_used['route_' + name].append(best_b)

        for td, X in held:
            g = gold_of(td, hi)
            if g is None:
                continue
            n_gold += 1
            cur, car, me = decompose(X)
            blend = cur + car + me
            variants = {'blend': blend, 'cur_only': cur, 'hist_only': car}
            for name in SIG_NAMES:
                variants['route_' + name] = routed(cur, car, me, td, name,
                                                   betas[name])
            for k, sc in variants.items():
                R[k].append(int((sc > sc[g]).sum()))
                S[k][0].append(sc)
                S[k][1].append(td.soft)
            # oracle: best menu weighting per turn (ceiling on cur/hist routing)
            best = min(int(((cur * a + car * b + me) >
                            (cur * a + car * b + me)[g]).sum())
                       for a, b in MENU)
            R['oracle'].append(best)
            oracle_changed += int(best < int((blend > blend[g]).sum()))
            S['oracle'][0].append(blend)      # soft_r ill-defined for oracle
            S['oracle'][1].append(td.soft)

    print('CORPUS %s' % WALKER_DIR)
    print('gold turns (soft>=90th=%.2f): %d · group(session) 5-fold CV' %
          (hi, n_gold))
    for name in SIG_NAMES:
        print('β chosen per fold — %-8s %s' % (name, beta_used['route_' + name]))
    print('\n%-16s  reach@5   Δvs blend   soft_r' % 'composition')
    base = reach5(R['blend'])
    order = ['blend', 'cur_only', 'hist_only'] + route_keys + ['oracle']
    for k in order:
        r5 = reach5(R[k])
        sr = soft_r(S[k][0], S[k][1])
        tag = ''
        if k == 'blend':
            tag = '  <- fixed z-blend baseline'
        elif k == 'oracle':
            tag = '  <- CEILING (uses gold)'
        print('  %-14s  %5.1f%%   %+5.1f pp    %+.3f%s'
              % (k, 100 * r5, 100 * (r5 - base), sr, tag))
    print('\noracle changed the gold rank vs blend on %d/%d turns (%.0f%%) — '
          'the routing headroom' % (oracle_changed, n_gold,
                                    100 * oracle_changed / n_gold))
    return 0


if __name__ == '__main__':
    sys.exit(main())
