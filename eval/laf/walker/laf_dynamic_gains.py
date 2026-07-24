"""Dynamic gains — can per-message features set the lane gains?

Tom's ask: a smart regression with as many params as possible to learn how
to set gains DYNAMICALLY from the cue + lane shapes + graph shapes + their
interactions. The formulation:

    score(node) = Σ_lane [ w_L0 + Σ_j w_Lj·φ_j ] · z_lane(node)

The gain on each lane is a linear readout of the per-message features φ
(lane peaks/gaps/supports, graph support/convergence). Because φ is
turn-constant, it survives within-turn pairwise differencing ONLY as a GATE
on lane-z — a bilinear logistic. Fitted coefficients ARE production gains
(deploy as K-store values, zero code) with the φ-gate as the new part.

THE VERDICT BAR (the cheap λ-router failed this twice — 701d86f8/88414714;
a richer feature set is a legitimate stronger test, not a reopening):
  1. session-grouped K-fold CV — held-out ONLY, never in-sample
  2. beat best-FIXED gains (same logistic, no φ) — not shipped, not shuffle
  3. SHUFFLE control — φ permuted across turns; if COND ≈ shuffle, φ is noise
  4. ECHO ablation — drop pick/enc (+their φ); if the win collapses, it echoed
Metrics: held-out pooled picked-AUC + soft-AUC; + reach@5 on the reach
substrate applying the learned conditional gains vs best-fixed.

Read-only. Run:  ./dev python3 eval/laf/walker/laf_dynamic_gains.py
"""
import json
import sys

import numpy as np

from walker_db import OUT_DIR

sys.path.insert(0, str(OUT_DIR))
from p3_fit import fit_logistic                                     # noqa: E402
from soft_usage import auc                                          # noqa: E402
from lambda_probe import zn                                         # noqa: E402
from miss_anatomy import rank_in                                    # noqa: E402
import laf_lane_audit as A                                         # noqa: E402

LANES6 = ('maxsim', 'sit', 'idf', 'pick', 'enc', 'graph')
FOLDS = 5
LAM = 1.0
SOFT_MARGIN = 0.10
REPORT = OUT_DIR / 'laf_dynamic_gains.md'

# per-message features φ (observable at inference — NO gold peeking)
PHI = ('maxsim_peak', 'maxsim_gap', 'maxsim_std', 'sit_peak', 'idf_peak',
       'log_idf_sup', 'log_pick_sup', 'log_enc_sup',
       'log_graph_sup', 'graph_peak', 'graph_conv2', 'graph_maxconv')


def top2gap(z):
    fin = z[np.isfinite(z)]
    if fin.size < 2:
        return 0.0
    s = np.sort(fin)[::-1]
    return float(s[0] - s[1])


def phi_raw(t):
    zl = t['zl']
    def pk(ln):
        z = zl[ln]
        return float(np.nanmax(z)) if np.isfinite(z).any() else 0.0
    def sup(ln):
        return int(np.sum(np.abs(zl[ln]) > 1e-9))
    return np.array([
        pk('maxsim'), top2gap(zl['maxsim']),
        float(np.nanstd(zl['maxsim'][np.isfinite(zl['maxsim'])])),
        pk('sit'), pk('idf'),
        np.log1p(sup('idf')), np.log1p(sup('pick')), np.log1p(sup('enc')),
        np.log1p(t['graph_support']),
        float(np.nanmax(t['graph_z'])) if np.isfinite(t['graph_z']).any() else 0.0,
        float(t['graph_n_conv2']), float(t['graph_max_conv']),
    ])


def lane_at(t, ln):
    """Lane z at the turn's candidate rows (absent cand row → 0)."""
    z = t['graph_z'] if ln == 'graph' else t['zl'][ln]
    cr = t['cand_rows']
    out = np.zeros(len(cr))
    ok = cr >= 0
    out[ok] = np.where(np.isfinite(z[cr[ok]]), z[cr[ok]], 0.0)
    return out


def cand_matrix(t, lanes, phi_std, conditional):
    """[n_cand × dim]: lane-z (+ lane-z×φ if conditional)."""
    cols = [lane_at(t, ln) for ln in lanes]
    if conditional:
        for ln in lanes:
            zc = lane_at(t, ln)
            for j in range(len(PHI)):
                cols.append(zc * phi_std[j])
    return np.column_stack(cols)


def pairs(t, X, target):
    if target == 'picked':
        si = np.flatnonzero(t['sel'])
        di = np.flatnonzero(~t['sel'])
        if len(si) and len(di):
            return (X[si][:, None, :] - X[di][None, :, :]).reshape(-1, X.shape[1])
        return None
    fin = np.flatnonzero(np.isfinite(t['soft']))
    if len(fin) < 2:
        return None
    s = t['soft'][fin]
    wi, li = np.nonzero((s[:, None] - s[None, :]) >= SOFT_MARGIN)
    if len(wi):
        return X[fin[wi]] - X[fin[li]]
    return None


def pooled_auc(turns_te, w, matgen, target):
    sel, drp = [], []
    for t in turns_te:
        X = matgen(t)
        s = X @ w
        if target == 'picked':
            if t['sel'].any() and not t['sel'].all():
                sel.append(s[t['sel']])
                drp.append(s[~t['sel']])
        else:
            fin = np.flatnonzero(np.isfinite(t['soft']))
            if len(fin) < 2:
                continue
            sv = t['soft'][fin]
            wi, li = np.nonzero((sv[:, None] - sv[None, :]) >= SOFT_MARGIN)
            if len(wi):
                sel.append((s[fin[wi]]))
                drp.append((s[fin[li]]))
    if not sel:
        return None
    return auc(np.concatenate(sel), np.concatenate(drp))


def cv_auc(turns, lanes, conditional, target, shuffle=False, seed=42):
    """Session-grouped K-fold held-out pooled AUC. φ standardized on TRAIN
    fold only; shuffle permutes φ rows across TRAIN+TEST turns (kills the
    message↔candidate link while preserving φ's marginal distribution)."""
    sess = sorted({t['sess'] for t in turns})
    rng = np.random.default_rng(seed)
    fold_of = {s: i % FOLDS for i, s in enumerate(sess)}
    phi_all = {id(t): phi_raw(t) for t in turns}
    aucs = []
    for f in range(FOLDS):
        tr = [t for t in turns if fold_of[t['sess']] != f]
        te = [t for t in turns if fold_of[t['sess']] == f]
        P = np.array([phi_all[id(t)] for t in tr])
        mu, sd = P.mean(0), P.std(0) + 1e-9
        phi_map = {}
        pool = [phi_all[id(t)] for t in (tr + te)]
        if shuffle:
            perm = rng.permutation(len(pool))
            pool = [pool[i] for i in perm]
        for t, praw in zip(tr + te, pool):
            phi_map[id(t)] = (praw - mu) / sd
        def matgen(t, _l=lanes, _c=conditional):
            return cand_matrix(t, _l, phi_map[id(t)], _c)
        D = [pairs(t, matgen(t), target) for t in tr]
        D = np.concatenate([d for d in D if d is not None])
        w = fit_logistic(D, lam=LAM)
        a = pooled_auc(te, w, matgen, target)
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs))


def fit_full_conditional(turns, lanes, target):
    """Fit on ALL turns (φ standardized globally) → gains readout for the
    reach application. In-sample; used only to derive gain_L(φ), which is
    then applied on the reach substrate as a separate honest measurement."""
    P = np.array([phi_raw(t) for t in turns])
    mu, sd = P.mean(0), P.std(0) + 1e-9
    phi_map = {id(t): (phi_raw(t) - mu) / sd for t in turns}
    def matgen(t):
        return cand_matrix(t, lanes, phi_map[id(t)], True)
    D = np.concatenate([d for d in (pairs(t, matgen(t), target)
                                    for t in turns) if d is not None])
    w = fit_logistic(D, lam=LAM)
    return w, mu, sd


def reach_dynamic(turns, w, mu, sd, lanes):
    """reach@5 applying per-turn conditional gains g_L(φ) on the reach
    substrate (all nodes). w layout: [lane0..k] then [lane×φ] blocks."""
    nl = len(lanes)
    w0 = w[:nl]
    wj = w[nl:].reshape(nl, len(PHI))
    h = tot = 0
    for t in turns:
        phi = (phi_raw(t) - mu) / sd
        gains = {ln: float(w0[i] + wj[i] @ phi) for i, ln in enumerate(lanes)}
        f = np.zeros_like(t['graph_z'])
        for ln in lanes:
            z = t['graph_z'] if ln == 'graph' else t['zl'][ln]
            f = f + gains[ln] * z
        f[~t['alive']] = np.nan
        mix = A.LAM * zn(f) + (1 - A.LAM) * zn(t['mh'])
        r = rank_in(mix, t['gr'])
        if r is None:
            continue
        tot += 1
        h += int(r <= 5)
    return 100.0 * h / tot if tot else 0.0


def main():
    turns, n = A.build()
    L = ['# Dynamic gains — conditional-gain regression (held-out)', '',
         'n=%d clean valids · φ = %d per-message features (%s) · '
         'model score=Σ_lane[w0+Σ w_j φ_j]·z_lane · L2 λ=%.1f · %d-fold '
         'session-CV' % (len(turns), len(PHI), ', '.join(PHI), LAM, FOLDS),
         '', 'VERDICT BAR: COND must beat best-FIXED **held-out** AND beat '
         'SHUFFLE; echo-ablation (drop pick/enc) must not collapse the win.',
         '']

    ECHO = tuple(ln for ln in LANES6 if ln not in ('pick', 'enc'))
    L += ['## Held-out pooled AUC (session-CV)', '',
          '| model | lanes | picked-AUC | soft-AUC |', '|---|---|---|---|']
    rows = [
        ('best-FIXED', LANES6, False, False),
        ('CONDITIONAL', LANES6, True, False),
        ('COND · SHUFFLE φ', LANES6, True, True),
        ('COND · echo-ablate', ECHO, True, False),
        ('FIXED · echo-ablate', ECHO, False, False),
    ]
    res = {}
    for name, lanes, cond, shuf in rows:
        pa = cv_auc(turns, lanes, cond, 'picked', shuffle=shuf)
        sa = cv_auc(turns, lanes, cond, 'soft', shuffle=shuf)
        res[name] = (pa, sa)
        L.append('| %s | %s | %.4f | %.4f |'
                 % (name, len(lanes), pa, sa))
        print('%-22s picked-AUC %.4f  soft-AUC %.4f' % (name, pa, sa))
    L.append('')

    # reach@5 application: best-fixed vs conditional gains on reach substrate
    base_all, _ = A.reach(turns, A.GAINS, 0.0)
    w_s, mu_s, sd_s = fit_full_conditional(turns, LANES6, 'soft')
    r_dyn_soft = reach_dynamic(turns, w_s, mu_s, sd_s, LANES6)
    w_p, mu_p, sd_p = fit_full_conditional(turns, LANES6, 'picked')
    r_dyn_pick = reach_dynamic(turns, w_p, mu_p, sd_p, LANES6)
    L += ['## reach@5 — conditional gains applied on reach substrate',
          '(in-sample gains; a ceiling read, not held-out reach)', '',
          '| composition | reach@5 |', '|---|---|',
          '| shipped fixed gains | %.1f%% |' % base_all,
          '| conditional (soft-fit) | %.1f%% |' % r_dyn_soft,
          '| conditional (picked-fit) | %.1f%% |' % r_dyn_pick, '']

    # verdict
    fp, fs = res['best-FIXED']
    cp, cs = res['CONDITIONAL']
    sp, ss = res['COND · SHUFFLE φ']
    beats_fixed = (cp - fp > 0.015) or (cs - fs > 0.015)
    beats_shuf = (cp - sp > 0.01) and (cs - ss > 0.01)
    verdict = ('DYNAMIC GAINS REAL — COND beats FIXED and SHUFFLE held-out'
               if beats_fixed and beats_shuf else
               'NULL — conditional ≈ fixed/shuffle held-out; rich features do '
               'NOT capture generalizable dynamic gains (extends the router '
               'null 701d86f8 to a large feature set)')
    L += ['## Verdict', '',
          '- COND vs FIXED (held-out): picked %+.4f · soft %+.4f'
          % (cp - fp, cs - fs),
          '- COND vs SHUFFLE (held-out): picked %+.4f · soft %+.4f'
          % (cp - sp, cs - ss),
          '- echo-ablate COND: picked %.4f soft %.4f (vs full COND %.4f/%.4f)'
          % (*res['COND · echo-ablate'], cp, cs),
          '', '**%s**' % verdict, '']

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
