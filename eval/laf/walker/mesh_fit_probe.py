"""Phase 3 of the mesh work: FITTED mesh formulas + the readout ROUTER,
through the composition-gate discipline (session-grouped 5-fold CV), on the
field cache. The question Phase 2 left open: single readouts are weak
(best AUC 0.554) — does a FITTED readout combination, applied as a mesh
weight, capture any of the field-level oracle headroom (+5.6pp 2-way)?

Arms:
  M_full        static flat mesh F0 + M_h                    [baseline]
  fitted-lin4   w·[F0,F1,A1,F2] fitted on train soft pairs   [best linear]
  router-soft   w0(r)·F0 + (1-w0(r))·M_h, w0 = σ(β·readouts) fitted on
                train-fold oracle labels                     [fields vote]
  router-hard   choose F0 vs M_full by router, ONLY in the disagreement
                region (ov_F0_F1 below train median), else M_full
  oracle-2way / oracle-5way ceilings

Oracle labels for the router fit: side=1 iff F0 ranks gold ≥5 ranks better
than M_full (or crosses @5) — meaningful differences only, not rank jitter.

Metrics: full-field gold reach@5/@25 · pool@5 · soft_r (cand-restricted) ·
churn vs M_full (@5 gained/lost).

Run: ./dev python3 eval/laf/walker/mesh_fit_probe.py   (pool60 via env)
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from p3_fit import fit_logistic                                     # noqa: E402
from field_mesh_probe import relu, conc, topset, wsum, gold_rank    # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'
GAMMA = 0.5
FOLDS = 5
SOFT_MARGIN = 0.10
ROUTER_FEATS = ('peak_F0', 'peak_Mh', 'conc_F0', 'conc_Mh', 'ov_F0_F1')


def newton_logistic(X, y, lam=1.0, iters=40):
    """Plain L2 logistic (with intercept) for the router fit."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -35, 35)))
        g = Xb.T @ (y - p) - lam * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + lam * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w


def router_prob(w, X):
    Xb = np.column_stack([np.ones(len(X)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -35, 35)))


class Turn:
    __slots__ = ('key', 'sess', 'row', 'gr', 'cand_rows', 'soft', 'fields',
                 'mh', 'mfull', 'ro')

    def __init__(self, t, fields, S):
        self.key, self.row = tuple(t['key']), t['row']
        self.sess = t['key'][0]
        self.cand_rows = t['cand_rows']
        self.gr = t['cand_rows'][t['gold_i']]
        self.soft = np.array([np.nan if x is None else x
                              for x in t['soft']])
        F = fields[t['row']].astype(np.float32)
        f0, f1, a1, f2 = (F[S['op0']], F[S['op1']], F[S['anchor1']],
                          F[S['op2']])
        self.fields = (f0, f1, a1, f2)
        self.mh = wsum([(GAMMA, f1), (GAMMA, a1), (GAMMA ** 2, f2)])
        self.mfull = wsum([(1.0, f0), (1.0, self.mh)])
        f1ok = f1 is not None and not np.isnan(f1).all()
        self.ro = None
        if self.mh is not None:
            self.ro = {
                'peak_F0': float(np.nanmax(f0)),
                'peak_Mh': float(np.nanmax(self.mh)),
                'conc_F0': conc(f0), 'conc_Mh': conc(self.mh),
                'ov_F0_F1': (len(topset(f0, 100) & topset(f1, 100)) / 100
                             if f1ok else 0.0),
            }


def cand_feats(t):
    """[n_cand × 4] field values at candidate rows (NaN-safe)."""
    out = np.full((len(t.cand_rows), 4), np.nan)
    for fi, f in enumerate(t.fields):
        if f is None:
            continue
        for ci, r in enumerate(t.cand_rows):
            if r >= 0 and np.isfinite(f[r]):
                out[ci, fi] = f[r]
    return np.where(np.isfinite(out), out, 0.0)


def soft_pairs(turns, keys):
    rows = []
    for t in turns:
        if t.key not in keys or not np.isfinite(t.soft).any():
            continue
        X = cand_feats(t)
        fin = np.flatnonzero(np.isfinite(t.soft))
        if len(fin) < 2:
            continue
        s = t.soft[fin]
        wi, li = np.nonzero((s[:, None] - s[None, :]) >= SOFT_MARGIN)
        if len(wi):
            rows.append(X[fin[wi]] - X[fin[li]])
    return np.concatenate(rows) if rows else np.zeros((0, 4))


def oracle_label(t):
    """1 = F0 meaningfully better than M_full at the gold, 0 = reverse,
    None = tie/jitter (excluded from the router fit)."""
    r0 = gold_rank(t.fields[0], t.gr)
    rM = gold_rank(t.mfull, t.gr) if t.mfull is not None else None
    if r0 is None or rM is None:
        return None
    if abs(r0 - rM) >= 5 or (r0 <= 5) != (rM <= 5):
        return int(r0 < rM)
    return None


def evaluate(turns, keys, score_of, base_of=None):
    """Full-field reach + pool@5 + soft_r (+ churn vs base_of)."""
    r5 = r25 = n = 0
    pool5, sx, sy = [], [], []
    gained = lost = 0
    for t in turns:
        if t.key not in keys:
            continue
        f = score_of(t)
        if f is None:
            continue
        rk = gold_rank(f, t.gr)
        if rk is None:
            continue
        n += 1
        r5 += int(rk <= 5)
        r25 += int(rk <= 25)
        crows = [r for r in t.cand_rows if r >= 0]
        cs = np.array([f[r] if np.isfinite(f[r]) else -np.inf
                       for r in crows])
        if t.gr in crows and np.isfinite(f[t.gr]):
            pool5.append(int((cs > f[t.gr]).sum()) < 5)
        m = [i for i, r in enumerate(t.cand_rows)
             if r >= 0 and np.isfinite(f[r]) and np.isfinite(t.soft[i])]
        if len(m) > 2:
            sx.append(np.array([f[t.cand_rows[i]] for i in m]))
            sy.append(t.soft[m])
        if base_of is not None:
            b = base_of(t)
            rb = gold_rank(b, t.gr) if b is not None else None
            if rb is not None:
                gained += int(rk <= 5 < rb if rb > 5 else False)
                lost += int(rb <= 5 < rk if rk > 5 else False)
    soft_r = float(np.corrcoef(np.concatenate(sx),
                               np.concatenate(sy))[0, 1]) if sx else np.nan
    return {'n': n, 'r5': 100 * r5 / max(1, n), 'r25': 100 * r25 / max(1, n),
            'pool5': 100 * np.mean(pool5) if pool5 else np.nan,
            'soft_r': soft_r, 'gained': gained, 'lost': lost}


def main():
    idx = json.loads(INDEX.read_text())
    fields = np.load(CACHE, mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    turns = [Turn(t, fields, S) for t in idx['turns']
             if not t.get('skipped')]
    turns = [t for t in turns if t.gr >= 0 and t.ro is not None]
    sess = sorted({t.sess for t in turns})
    fold_of = {s: i % FOLDS for i, s in enumerate(sess)}
    print('turns %d · sessions %d' % (len(turns), len(sess)))

    # per-fold fits
    lin_w, rt_w, rt_mu, rt_sd, ov_med = {}, {}, {}, {}, {}
    for f in range(FOLDS):
        tr = {t.key for t in turns if fold_of[t.sess] != f}
        D = soft_pairs(turns, tr)
        lin_w[f] = fit_logistic(D) if len(D) >= 10 else None
        X, y = [], []
        for t in turns:
            if t.key not in tr:
                continue
            lab = oracle_label(t)
            if lab is None:
                continue
            X.append([t.ro[k] for k in ROUTER_FEATS])
            y.append(lab)
        X, y = np.array(X), np.array(y, dtype=float)
        m = np.isfinite(X).all(axis=1)
        X, y = X[m], y[m]
        rt_mu[f], rt_sd[f] = X.mean(0), X.std(0) + 1e-9
        rt_w[f] = newton_logistic((X - rt_mu[f]) / rt_sd[f], y)
        ov_med[f] = float(np.median(
            [t.ro['ov_F0_F1'] for t in turns if t.key in tr]))
        print('  fold %d: %d soft pairs · %d oracle labels '
              '(base %.2f) · router w %s'
              % (f, len(D), len(y), y.mean(),
                 np.round(rt_w[f], 2).tolist()))

    def fold(t):
        return fold_of[t.sess]

    def s_mfull(t):
        return t.mfull

    def s_lin(t):
        w = lin_w[fold(t)]
        if w is None:
            return None
        return wsum(list(zip(w, t.fields)))

    def w0_of(t):
        f = fold(t)
        x = np.array([[t.ro[k] for k in ROUTER_FEATS]])
        if not np.isfinite(x).all():
            return None
        return float(router_prob(rt_w[f], (x - rt_mu[f]) / rt_sd[f])[0])

    def s_router_soft(t):
        w0 = w0_of(t)
        if w0 is None or t.mh is None:
            return t.mfull
        return wsum([(w0, t.fields[0]), (1 - w0, t.mh)])

    def s_router_hard(t):
        w0 = w0_of(t)
        if w0 is None or t.ro['ov_F0_F1'] >= ov_med[fold(t)]:
            return t.mfull                    # agreement region: blend
        return t.fields[0] if w0 > 0.5 else t.mfull

    def s_oracle2(t):
        r0 = gold_rank(t.fields[0], t.gr)
        rM = gold_rank(t.mfull, t.gr) if t.mfull is not None else None
        if rM is None:
            return t.fields[0]
        if r0 is None:
            return t.mfull
        return t.fields[0] if r0 < rM else t.mfull

    def s_oracle5(t):
        best, bf = None, None
        for f in (t.fields[0], t.fields[1], t.fields[2], t.mh, t.mfull):
            if f is None:
                continue
            r = gold_rank(f, t.gr)
            if r is not None and (best is None or r < best):
                best, bf = r, f
        return bf

    allkeys = {t.key for t in turns}
    arms = [('M_full (static)', s_mfull), ('fitted-lin4', s_lin),
            ('router-soft', s_router_soft), ('router-hard', s_router_hard),
            ('oracle-2way', s_oracle2), ('oracle-5way', s_oracle5)]
    print('\narm               n      reach@5  reach@25  pool@5   soft_r'
          '   Δ@5 vs M_full (gain/loss)')
    for name, fn in arms:
        m = evaluate(turns, allkeys, fn, base_of=s_mfull)
        print('  %-15s %5d   %5.1f%%   %5.1f%%   %5.1f%%   %+.3f'
              '   +%d/-%d' % (name, m['n'], m['r5'], m['r25'],
                              m['pool5'], m['soft_r'],
                              m['gained'], m['lost']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
