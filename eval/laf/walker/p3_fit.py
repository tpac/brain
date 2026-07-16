"""P3.1 — fit the composition on the walker table (§20.13, pre-registered).

FEATURES (30 lane-slots + M_e = 31; rank_in_pool STRUCTURALLY EXCLUDED —
features are built from lane tensors + the fatigue table only, production's
own output never enters):
  {maxsim, sit, idf, pick, enc} × {j0-op, j1-op, j1-anchor, j2-op, j2-anchor,
  tail} where tail = Σ_{j=3..8} 0.5^(j−3)·(op_j ⊕ anchor_j) (nansum; all-NaN
  stays NaN), + M_e fatigue value f (RAW — production subtracts δ·f in
  z-space, so the fitted coefficient IS −δ).

SUBSTRATE (P3.0 verdict): 'current' — each lane-slot is production _zscore
over the turn's pool; NaN → 0 (neutral). Fitted coefficients are therefore
gains in the production sense (score = Σ gain·z) — the §19 promise: deploy
as recall_laf K-store VALUES, zero code.

MODEL: within-turn pairwise logistic (Bradley-Terry on selected−dropped
diffs), L2 λ=1.0 (pre-declared; sensitivity over {0.1, 1, 10} reported),
Newton-Raphson (no scipy in the venv — 31-dim Hessian is trivial). The
intercept cancels in the pairwise formulation — reported as 0 by
construction, not fitted.

SPLIT: train April–May (ts < 2026-06-01), validate June+ — the Q1 era split.

FIT ARMS (all pre-registered):
  A  full31 · picked      — THE fit (primary target)
  B  full31 · soft-usage  — secondary target, separate fit, compared to A
  C  full31 minus pick/enc · picked — THE ECHO ABLATION (mandatory): if
     content-only collapses, the learned signal was echo and we say so
  D  j0-only · picked     — P3a deliverable (production scores j=0 today)
  E  j0-only minus pick/enc · picked — echo read on the P3a gains

Run:  ./dev python3 eval/laf/walker/p3_fit.py
Out:  p3_fit.md, p3_fit.json (coefficients + metrics — P3.2's input)
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import WALKER_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import _zscore                             # noqa: E402
from q1_sweep import GAINS, load, gate_provenance, configs, \
    evaluate                                                        # noqa: E402
from soft_usage import auc                                          # noqa: E402
from episodic_roles import K_MAX                                    # noqa: E402

LANES = ('maxsim', 'sit', 'idf', 'pick', 'enc')
SLOTS = ('j0-op', 'j1-op', 'j1-anchor', 'j2-op', 'j2-anchor', 'tail')
TAIL_GAMMA = 0.5
LAMBDA = 1.0
LAMBDA_SENS = (0.1, 1.0, 10.0)
SOFT_MARGIN = 0.10            # min soft_max gap to mint a soft-target pair
WINNER = 'K1-exp0.5-turnsum-zsum-opanchor-me0'
REPORT = WALKER_DIR / 'p3_fit.md'
OUT = WALKER_DIR / 'p3_fit.json'

FEATURES = ['%s·%s' % (ln, sl) for ln in LANES for sl in SLOTS] + ['M_e_f']


def slot_raw(td, ln, slot):
    """One lane-slot raw vector for a turn (NaN = missing)."""
    if slot == 'j0-op':
        return td.op[ln][:, 0]
    if slot == 'j1-op':
        return td.op[ln][:, 1]
    if slot == 'j1-anchor':
        return td.anchor[ln][:, 1]
    if slot == 'j2-op':
        return td.op[ln][:, 2]
    if slot == 'j2-anchor':
        return td.anchor[ln][:, 2]
    g = TAIL_GAMMA ** np.arange(K_MAX + 1 - 3, dtype=float)
    m = np.concatenate([td.op[ln][:, 3:] * g, td.anchor[ln][:, 3:] * g],
                       axis=1)
    with np.errstate(all='ignore'):
        v = np.nansum(m, axis=1)
    v[np.all(np.isnan(m), axis=1)] = np.nan
    return v


def turn_features(td):
    """[n_cand × 31] — 'current'-substrate z per lane-slot + raw fatigue."""
    nc = len(td.cands)
    cols = []
    for ln in LANES:
        for sl in SLOTS:
            cols.append(_zscore(slot_raw(td, ln, sl), nc))
    cols.append(td.fat)
    return np.column_stack(cols)


def build(turns):
    """Per-turn feature matrices + masks, once."""
    return [(td, turn_features(td)) for td in turns]


def pairs_picked(feats, train_only):
    """Selected−dropped diff rows, within turn."""
    rows = []
    for td, X in feats:
        if train_only is not None and td.val == train_only:
            continue
        si = np.flatnonzero(td.sel)
        di = np.flatnonzero(~td.sel)
        if not len(si) or not len(di):
            continue
        rows.append((X[si][:, None, :] - X[di][None, :, :]).reshape(
            -1, X.shape[1]))
    return np.concatenate(rows) if rows else np.empty((0, len(FEATURES)))


def pairs_soft(feats, train_only):
    """Soft-usage target: within-turn pairs with a ≥SOFT_MARGIN soft gap,
    diff oriented winner−loser."""
    rows = []
    for td, X in feats:
        if train_only is not None and td.val == train_only:
            continue
        fin = np.flatnonzero(np.isfinite(td.soft))
        if len(fin) < 2:
            continue
        s = td.soft[fin]
        d = s[:, None] - s[None, :]
        wi, li = np.nonzero(d >= SOFT_MARGIN)
        if not len(wi):
            continue
        rows.append(X[fin[wi]] - X[fin[li]])
    return np.concatenate(rows) if rows else np.empty((0, len(FEATURES)))


def fit_logistic(D, lam=LAMBDA, iters=40):
    """Bradley-Terry MLE: maximize Σ log σ(D·w) − (λ/2)|w|², Newton."""
    k = D.shape[1]
    w = np.zeros(k)
    for _ in range(iters):
        z = D @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        g = D.T @ (1.0 - p) - lam * w
        H = (D * (p * (1.0 - p))[:, None]).T @ D + lam * np.eye(k)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.abs(step).max() < 1e-9:
            break
    return w


def model_auc(feats, w, cols, subset):
    """Pooled sel-vs-drop AUC of s = X[:,cols]·w over turns in subset."""
    sel, drp = [], []
    for td, X in feats:
        if not subset(td):
            continue
        s = X[:, cols] @ w
        sel.append(s[td.sel])
        drp.append(s[~td.sel])
    if not sel:
        return None
    return auc(np.concatenate(sel), np.concatenate(drp))


def model_soft_r(feats, w, cols):
    xs, ys = [], []
    for td, X in feats:
        s = X[:, cols] @ w
        m = np.isfinite(td.soft) & np.isfinite(s)
        if m.sum() > 2:
            xs.append(s[m])
            ys.append(td.soft[m])
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(np.corrcoef(x, y)[0, 1])


ARMS = {
    'A_full_picked': {'target': 'picked',
                      'cols': list(range(len(FEATURES)))},
    'B_full_soft': {'target': 'soft',
                    'cols': list(range(len(FEATURES)))},
    'C_ablate_echo': {'target': 'picked',
                      'cols': [i for i, f in enumerate(FEATURES)
                               if not f.startswith(('pick·', 'enc·'))]},
    'D_j0_only': {'target': 'picked',
                  'cols': [i for i, f in enumerate(FEATURES)
                           if f.endswith('j0-op') or f == 'M_e_f']},
    'E_j0_ablate': {'target': 'picked',
                    'cols': [i for i, f in enumerate(FEATURES)
                             if (f.endswith('j0-op') or f == 'M_e_f')
                             and not f.startswith(('pick·', 'enc·'))]},
    'F_soft_ablate': {'target': 'soft',
                      'cols': [i for i, f in enumerate(FEATURES)
                               if not f.startswith(('pick·', 'enc·'))]},
    'G_j0_soft': {'target': 'soft',
                  'cols': [i for i, f in enumerate(FEATURES)
                           if f.endswith('j0-op') or f == 'M_e_f']},
    'H_j0_soft_ablate': {'target': 'soft',
                         'cols': [i for i, f in enumerate(FEATURES)
                                  if (f.endswith('j0-op') or f == 'M_e_f')
                                  and not f.startswith(('pick·', 'enc·'))]},
}


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    walker.close()
    print('building features over %d turns...' % len(turns))
    feats = build(turns)

    D_pick = pairs_picked(feats, train_only=False)     # train = not val
    D_soft = pairs_soft(feats, train_only=False)
    print('train pairs: picked %d · soft %d' % (len(D_pick), len(D_soft)))

    lines = ['# p3_fit — the composition fit (§20.13 P3.1)', '',
             '- substrate: current (P3.0 verdict); features: 30 lane-slots + '
             'M_e_f; intercept ≡ 0 (cancels in pairwise diffs)',
             '- train April–May / validate June+; pairs: picked %d, soft %d '
             '(margin %.2f)' % (len(D_pick), len(D_soft), SOFT_MARGIN), '']

    results = {}
    for name, spec in ARMS.items():
        D = D_pick if spec['target'] == 'picked' else D_soft
        cols = spec['cols']
        w = fit_logistic(D[:, cols])
        m = {
            'auc_val': model_auc(feats, w, cols, lambda td: td.val),
            'auc_train': model_auc(feats, w, cols, lambda td: not td.val),
            'auc_val_normal': model_auc(
                feats, w, cols, lambda td: td.val and not td.flagged),
            'auc_val_flagged': model_auc(
                feats, w, cols, lambda td: td.val and td.flagged),
            'soft_r': model_soft_r(feats, w, cols),
            'coef': {FEATURES[c]: round(float(v), 4)
                     for c, v in zip(cols, w)},
        }
        results[name] = m
        print('%s: val AUC %.4f' % (name, m['auc_val']))

    # λ sensitivity on the primary arm (train fit → val AUC)
    sens = {}
    colsA = ARMS['A_full_picked']['cols']
    for lam in LAMBDA_SENS:
        wl = fit_logistic(D_pick[:, colsA], lam=lam)
        sens[str(lam)] = round(model_auc(feats, wl, colsA,
                                         lambda td: td.val), 4)

    # static references on the same pooled-val metric — including a
    # content-only static (pick/enc gains zeroed) so the C ablation is
    # compared against a reference that carries no echo either
    k0 = evaluate(turns, configs()[0])
    win = evaluate(turns, next(c for c in configs() if c['name'] == WINNER))
    g_content = {ln: (0.0 if ln in ('pick', 'enc') else GAINS[ln])
                 for ln in GAINS}
    from q1_sweep import score_turn, weights as q1_weights
    k0_cfg = configs()[0]
    w0 = q1_weights(k0_cfg)
    sel, drp = [], []
    for td in turns:
        if not td.val:
            continue
        s = score_turn(td, k0_cfg, w0, gains=g_content)
        sel.append(s[td.sel])
        drp.append(s[~td.sel])
    k0_content_val = auc(np.concatenate(sel), np.concatenate(drp))

    lines.append('## metrics (June+ validation, pooled sel-vs-drop AUC)')
    lines.append('| arm | val AUC | train AUC | val normal | val flagged | '
                 'soft_r |')
    lines.append('|---|---|---|---|---|---|')
    for name, m in results.items():
        lines.append('| %s | %.4f | %.4f | %.4f | %.4f | %.3f |'
                     % (name, m['auc_val'], m['auc_train'],
                        m['auc_val_normal'], m['auc_val_flagged'],
                        m['soft_r']))
    lines.append('| K0-static | %.4f |  |  |  | %.3f |'
                 % (k0['auc_val'], k0.get('soft_r', 0)))
    lines.append('| winner-static | %.4f |  |  |  | %.3f |'
                 % (win['auc_val'], win.get('soft_r', 0)))
    lines.append('| K0-static content-only (pick/enc=0) | %.4f |  |  |  |  |'
                 % k0_content_val)
    lines.append('')
    lines.append('- λ sensitivity (arm A, val AUC): %s' % json.dumps(sens))
    lines.append('')

    lines.append('## coefficients — arm A (full, picked), z-space gains')
    lines.append('| lane | ' + ' | '.join(SLOTS) + ' |')
    lines.append('|' + '---|' * (len(SLOTS) + 1))
    cA = results['A_full_picked']['coef']
    for ln in LANES:
        lines.append('| %s | ' % ln + ' | '.join(
            '%+.3f' % cA['%s·%s' % (ln, sl)] for sl in SLOTS) + ' |')
    lines.append('- M_e_f: %+.4f (−δ in production terms)' % cA['M_e_f'])
    lines.append('')
    lines.append('## P3a candidates (j0-only fitted gains)')
    for arm in ('D_j0_only', 'G_j0_soft', 'H_j0_soft_ablate'):
        lines.append('- **%s**: %s' % (arm, ' · '.join(
            '%s %+.3f' % (f.replace('·j0-op', ''), v)
            for f, v in results[arm]['coef'].items())))
    lines.append('')
    lines.append('## echo read')
    dA = results['A_full_picked']['auc_val'] - k0['auc_val']
    dC = results['C_ablate_echo']['auc_val'] - k0['auc_val']
    lines.append('- full fit ΔAUC vs K0-static: %+.4f; content-only (no '
                 'pick/enc): %+.4f — echo share of the fit gain: %.0f%%'
                 % (dA, dC, 100 * (1 - dC / dA) if dA else 0))

    OUT.write_text(json.dumps({'results': results, 'lambda_sens': sens,
                               'features': FEATURES,
                               'arms': {k: v['cols'] for k, v in ARMS.items()},
                               'k0_static_val': k0['auc_val'],
                               'winner_static_val': win['auc_val'],
                               'k0_content_val': k0_content_val}))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
