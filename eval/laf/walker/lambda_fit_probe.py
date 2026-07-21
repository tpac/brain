"""The λ-derivation hunt, step 2: do the (weak, |ρ|≤0.16) per-layer
readouts COMPOSE into usable λ tracking? Ridge λ̂(readouts) under the
composition-gate discipline (session-grouped 5-fold CV), scored by the
graduation metric: full-field gold reach@5 of the λ̂-mixed field vs the
train-fitted static λ, against the per-turn oracle-λ ceiling.

Arms (per eval fold, aggregated):
  static-λ      best single λ on the TRAIN fold           [the fair baseline]
  λ̂-ridge      clip01(ridge on z-scored msg-0 readouts, fitted on train
                decisive turns' λ*-mid)                    [the derivation]
  λ̂-conc1      1-feature version: conc_sit only (the strongest single
                readout) — how much does the full vector add?
  oracle-λ      per-turn best λ                            [ceiling]

Machinery: readouts via layer_readout_probe.msg0_readouts (THE definition),
λ ranks via lambda_probe.lambda_star (grid + single-λ̂ calls) — never
re-implement either (the wsum rule).

Run:    ./dev python3 eval/laf/walker/lambda_fit_probe.py
Pool60: WALKER_OUT_DIR=~/AgentsContext/eval-corpus/0a9baa/walker ... (same)
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn, lambda_star, plateau_of, GRID           # noqa: E402
from layer_readout_probe import lane_z, msg0_readouts                # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
LANE_CACHE = OUT_DIR / 'lane_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'
FOLDS = 5
RIDGE_LAM = 3.0
FEATS = ('conc_sit', 'conc_maxsim', 'conc_idf', 'conc_pick', 'conc_enc',
         'peak_maxsim', 'peak_sit', 'peak_idf', 'peak_pick', 'peak_enc',
         'conc_F0', 'peak_F0', 'conc_Mh', 'peak_Mh', 'ov_F0_F1',
         'shr_pick', 'shr_sit', 'shr_idf', 'agr_cos_epi', 'agr_cos_sit',
         'disagree')


def ridge(X, y, lam=RIDGE_LAM):
    Xb = np.column_stack([np.ones(len(X)), X])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[0, 0] -= lam                    # don't penalize the intercept
    return np.linalg.solve(A, Xb.T @ y)


def main():
    idx = json.loads(INDEX.read_text())
    fields = np.load(CACHE, mmap_mode='r')
    lanes_mm = np.load(LANE_CACHE, mmap_mode='r')
    slots, lanes = idx['slots'], idx['lanes']
    S = {s: i for i, s in enumerate(slots)}
    n = idx['n_nodes']

    # one pass: readouts + λ-grid ranks per turn
    rows = []
    for t in idx['turns']:
        if t.get('skipped'):
            continue
        tt = Turn(t, fields, S)
        if tt.gr < 0 or tt.ro is None or tt.mh is None \
                or np.isnan(tt.fields[0]).all():
            continue
        f0z, mhz = zn(tt.fields[0]), zn(tt.mh)
        ranks = lambda_star(f0z, mhz, tt.gr)
        if not ranks:
            continue
        lo, hi, mid = plateau_of(ranks)
        L = lanes_mm[t['row']].astype(np.float32)
        zf = {}
        for si, sl in enumerate(slots):
            mx = L[si, lanes.index('maxsim')]
            if np.isnan(mx).all():
                continue
            alive = np.isfinite(mx)
            for li, ln in enumerate(lanes):
                zf[(sl, ln)] = lane_z(L[si, li], ln, alive, n)
        ro = msg0_readouts(tt, zf, lanes)
        rows.append({'sess': tt.sess, 'gr': tt.gr, 'f0z': f0z, 'mhz': mhz,
                     'ranks': ranks, 'mid': mid,
                     'decisive': hi - lo <= 0.5,
                     'x': np.array([ro.get(k, np.nan) for k in FEATS])})
    print('turns %d (decisive %d)'
          % (len(rows), sum(r['decisive'] for r in rows)))

    sess = sorted({r['sess'] for r in rows})
    fold_of = {s: i % FOLDS for i, s in enumerate(sess)}
    ci = FEATS.index('conc_sit')

    def rank_at(r, lam):
        rk = lambda_star(r['f0z'], r['mhz'], r['gr'],
                         grid=np.array([round(float(lam), 4)]))
        return min(rk.values()) if rk else None

    hits = {'static-λ': 0, 'λ̂-ridge': 0, 'λ̂-conc1': 0, 'oracle-λ': 0}
    n_eval, churn = 0, {'λ̂-ridge': [0, 0], 'λ̂-conc1': [0, 0]}
    lam_static_all = []
    for f in range(FOLDS):
        tr = [r for r in rows if fold_of[r['sess']] != f]
        ev = [r for r in rows if fold_of[r['sess']] == f]
        if not tr or not ev:
            continue
        # static λ fitted on train (reach@5-argmax over the grid)
        stat = max(GRID, key=lambda l: sum(
            r['ranks'].get(l, 10 ** 9) <= 5 for r in tr))
        lam_static_all.append(float(stat))
        # ridge fit on train decisive turns
        trd = [r for r in tr if r['decisive']]
        X = np.stack([r['x'] for r in trd])
        y = np.array([r['mid'] for r in trd])
        mu = np.nanmean(X, axis=0)
        Xi = np.where(np.isfinite(X), X, mu)
        sd = Xi.std(axis=0) + 1e-9
        w = ridge((Xi - mu) / sd, y)
        Xc = Xi[:, [ci]]
        muc, sdc = mu[[ci]], sd[[ci]]
        wc = ridge((Xc - muc) / sdc, y)

        for r in ev:
            rs = r['ranks'].get(stat)
            xi = np.where(np.isfinite(r['x']), r['x'], mu)
            lam_hat = float(np.clip(
                w[0] + (xi - mu) / sd @ w[1:], 0.0, 1.0))
            lam_c = float(np.clip(
                wc[0] + (xi[[ci]] - muc) / sdc @ wc[1:], 0.0, 1.0))
            rr = rank_at(r, lam_hat)
            rc = rank_at(r, lam_c)
            if rs is None or rr is None or rc is None:
                continue
            n_eval += 1
            hits['static-λ'] += int(rs <= 5)
            hits['λ̂-ridge'] += int(rr <= 5)
            hits['λ̂-conc1'] += int(rc <= 5)
            hits['oracle-λ'] += int(min(r['ranks'].values()) <= 5)
            churn['λ̂-ridge'][0] += int(rr <= 5 < rs)
            churn['λ̂-ridge'][1] += int(rs <= 5 < rr)
            churn['λ̂-conc1'][0] += int(rc <= 5 < rs)
            churn['λ̂-conc1'][1] += int(rs <= 5 < rc)
        print('  fold %d: static λ=%.2f · ridge top w: %s'
              % (f, stat, ', '.join(
                  '%s %+0.2f' % (FEATS[i], w[1 + i])
                  for i in np.argsort(-np.abs(w[1:]))[:5])))

    print('\narm         reach@5 (n=%d)   Δ@5 vs static (gain/loss)'
          % n_eval)
    for k in ('static-λ', 'λ̂-ridge', 'λ̂-conc1', 'oracle-λ'):
        c = churn.get(k)
        print('  %-9s  %5.1f%%           %s'
              % (k, 100 * hits[k] / max(1, n_eval),
                 '+%d/-%d' % tuple(c) if c else '—'))
    print('static λ per fold: %s' % lam_static_all)
    return 0


if __name__ == '__main__':
    sys.exit(main())
