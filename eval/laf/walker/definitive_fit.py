"""The definitive fit — full lane × slot × side factorization at K=8.

Consolidates the arc's findings into ONE deployable composition:
anchor-side dominance, depth past K=2, op-history ≈ 0, redundancy —
all become fitted weights instead of narrative. Runs on walker v7.

FEATURES: 5 lanes × (op j0..8 + anchor j1..8) = 85 + M_e_f = 86.
TARGET: soft-usage pairs (picked pairs reported for reference only —
the echo verdict stands). SPLIT: fit April–May, evaluate June+ (the
corrected discipline; soft_r val-gated).

ARMS:
  S_full    soft · all 86                     — THE candidate
  S_content soft · minus pick/enc (52)        — content-only variant
  P_full    picked · all 86                   — reference (echo read)
REFERENCES on identical val metrics: K0-static, winner-static,
F_soft_ablate (30-feat, corrected p3_fit.json), fitted-17 per-message.

Out:  definitive_fit.md, definitive_fit.json (gains keyed for K-store)
Run:  ./dev python3 eval/laf/walker/definitive_fit.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import _zscore                              # noqa: E402
from q1_sweep import GAINS, load, gate_provenance, configs, \
    evaluate                                                        # noqa: E402
from p3_fit import fit_logistic, LANES                              # noqa: E402
from soft_usage import auc                                          # noqa: E402
from episodic_roles import K_MAX                                    # noqa: E402

SOFT_MARGIN = 0.10
WINNER = 'K1-exp0.5-turnsum-zsum-opanchor-me0'
REPORT = OUT_DIR / 'definitive_fit.md'
OUT = OUT_DIR / 'definitive_fit.json'

SLOTS = [('op', j) for j in range(K_MAX + 1)] + \
        [('anchor', j) for j in range(1, K_MAX + 1)]
FEATURES = ['%s·%s%d' % (ln, sd, j) for ln in LANES
            for sd, j in SLOTS] + ['M_e_f']


def turn_features(td):
    nc = len(td.cands)
    cols = []
    for ln in LANES:
        for sd, j in SLOTS:
            raw = (td.op if sd == 'op' else td.anchor)[ln][:, j]
            cols.append(_zscore(raw, nc))
    cols.append(td.fat)
    return np.column_stack(cols)


def pairs_soft(feats):
    rows = []
    for td, X in feats:
        if td.val:
            continue
        fin = np.flatnonzero(np.isfinite(td.soft))
        if len(fin) < 2:
            continue
        s = td.soft[fin]
        d = s[:, None] - s[None, :]
        wi, li = np.nonzero(d >= SOFT_MARGIN)
        if len(wi):
            rows.append(X[fin[wi]] - X[fin[li]])
    return np.concatenate(rows)


def pairs_picked(feats):
    rows = []
    for td, X in feats:
        if td.val:
            continue
        si = np.flatnonzero(td.sel)
        di = np.flatnonzero(~td.sel)
        if len(si) and len(di):
            rows.append((X[si][:, None, :] - X[di][None, :, :])
                        .reshape(-1, X.shape[1]))
    return np.concatenate(rows)


def val_metrics(feats, w, cols):
    sel, drp, sx, sy = [], [], [], []
    top1 = nturn = 0
    for td, X in feats:
        if not td.val:
            continue
        s = X[:, cols] @ w
        if td.sel.any() and not td.sel.all():
            sel.append(s[td.sel])
            drp.append(s[~td.sel])
            nturn += 1
            top1 += int(td.sel[int(np.argmax(s))])
        m = np.isfinite(td.soft) & np.isfinite(s)
        if m.sum() > 2:
            sx.append(s[m])
            sy.append(td.soft[m])
    return {'auc': auc(np.concatenate(sel), np.concatenate(drp)),
            'soft_r': float(np.corrcoef(np.concatenate(sx),
                                        np.concatenate(sy))[0, 1]),
            'sel_at_1': top1 / nturn}


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    walker.close()
    print('features over %d turns (%d cols)...'
          % (len(turns), len(FEATURES)))
    feats = [(td, turn_features(td)) for td in turns]

    D_soft = pairs_soft(feats)
    D_pick = pairs_picked(feats)
    print('train pairs: soft %d · picked %d' % (len(D_soft), len(D_pick)))

    all_cols = list(range(len(FEATURES)))
    content_cols = [i for i, f in enumerate(FEATURES)
                    if not f.startswith(('pick·', 'enc·'))]
    arms = {
        'S_full': (D_soft, all_cols),
        'S_content': (D_soft, content_cols),
        'P_full': (D_pick, all_cols),
    }
    results, weights_out = {}, {}
    for name, (D, cols) in arms.items():
        w = fit_logistic(D[:, cols])
        results[name] = val_metrics(feats, w, cols)
        weights_out[name] = {FEATURES[c]: round(float(v), 4)
                             for c, v in zip(cols, w)}
        print(name, results[name])

    # references on the identical val slice
    k0 = evaluate([td for td in turns], configs()[0])
    win = evaluate([td for td in turns],
                   next(c for c in configs() if c['name'] == WINNER))

    lines = ['# definitive_fit — lane × slot × side @ K=8 (walker v7)',
             '',
             '- fit April–May (%d soft / %d picked pairs) · eval June+ · '
             'soft_r val-gated' % (len(D_soft), len(D_pick)), '',
             '| arm | val AUC | val soft_r | sel@1 |',
             '|---|---|---|---|']
    for name, m in results.items():
        lines.append('| %s | %.4f | %.3f | %.3f |'
                     % (name, m['auc'], m['soft_r'], m['sel_at_1']))
    lines.append('| K0-static | %.4f | %.3f | — |'
                 % (k0['auc_val'], k0.get('soft_r', 0)))
    lines.append('| winner-static | %.4f | %.3f | — |'
                 % (win['auc_val'], win.get('soft_r', 0)))
    lines.append('')

    # the candidate's weight matrix, rendered lane × slot
    lines.append('## S_full weights (z-space gains)')
    lines.append('| lane | ' + ' | '.join('%s%d' % (sd[0], j)
                                          for sd, j in SLOTS) + ' |')
    lines.append('|' + '---|' * (len(SLOTS) + 1))
    wf = weights_out['S_full']
    for ln in LANES:
        lines.append('| %s | ' % ln + ' | '.join(
            '%+.2f' % wf['%s·%s%d' % (ln, sd, j)]
            for sd, j in SLOTS) + ' |')
    lines.append('- M_e_f: %+.3f' % wf['M_e_f'])
    lines.append('')
    # side/depth marginals — the narrative check
    for ln in ('maxsim',):
        op_sum = sum(wf['%s·op%d' % (ln, j)] for j in range(1, K_MAX + 1))
        an_sum = sum(wf['%s·anchor%d' % (ln, j)]
                     for j in range(1, K_MAX + 1))
        lines.append('- %s history mass: op %+.2f · anchor %+.2f '
                     '(j0 %+.2f)' % (ln, op_sum, an_sum,
                                     wf['%s·op0' % ln]))

    OUT.write_text(json.dumps({'results': results,
                               'weights': weights_out,
                               'features': FEATURES}, indent=1))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
