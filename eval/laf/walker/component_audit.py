"""Leg A — the component audit (lane × slot contribution matrix).

The moment is NOT an extra signal: the full lane stack fires on every
message slot (j0, j1-op, j1-anchor, ...) and the per-message activations
mesh with decayed weight. The audit unit is therefore the (lane, slot)
CELL, read three ways:

  rows     — a lane's contribution across the whole moment
  columns  — a slot's contribution across all lanes
  cells    — suspect drill-downs (pick-j1 echo, sit-j2-anchor, idf negatives)

Every unit gets two arms under BOTH label systems (picked pairs = judge
echo-prone; soft-usage pairs = what the answer actually needed):

  removed — refit the full model without the unit; Δ vs the full fit
  alone   — history units: j0-stack + unit, Δ vs the j0-only fit;
            j0/lane units: the unit by itself (single-signal power)

Plus the shipped-dials view: winner-static / K0-static with each lane's
gain zeroed (no refit — what each dial does at current settings).

Noise: turn-level paired bootstrap on the val slice; deltas inside the
band are reported as noise, never as findings (gold-24 lesson).

Run:  ./dev python3 eval/laf/walker/component_audit.py
Out:  component_audit.md, component_audit.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import WALKER_DIR, open_walker, open_brain_ro

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import GAINS, load, gate_provenance, configs, score_turn, \
    weights                                                         # noqa: E402
from p3_fit import LANES, SLOTS, FEATURES, build, pairs_picked, \
    pairs_soft, fit_logistic, model_auc, model_soft_r               # noqa: E402
from soft_usage import auc                                          # noqa: E402

WINNER = 'K1-exp0.5-turnsum-zsum-opanchor-me0'
N_BOOT = 500
SEED = 20260716
REPORT = WALKER_DIR / 'component_audit.md'
OUT = WALKER_DIR / 'component_audit.json'

J0_COLS = [i for i, f in enumerate(FEATURES) if f.endswith('j0-op')]
ALL_COLS = list(range(len(FEATURES)))


def units():
    """(name, kind, cols) — rows, columns, then the 31 cells."""
    out = []
    for ln in LANES:
        out.append(('lane:%s' % ln, 'row',
                    [i for i, f in enumerate(FEATURES)
                     if f.startswith(ln + '·')]))
    for sl in SLOTS:
        out.append(('slot:%s' % sl, 'col',
                    [i for i, f in enumerate(FEATURES)
                     if f.endswith('·' + sl)]))
    out.append(('M_e_f', 'cell', [FEATURES.index('M_e_f')]))
    for i, f in enumerate(FEATURES[:-1]):
        out.append(('cell:%s' % f, 'cell', [i]))
    return out


def fit_eval(D, feats, cols, target):
    w = fit_logistic(D[:, cols])
    return {
        'auc_val': model_auc(feats, w, cols, lambda td: td.val),
        'auc_val_normal': model_auc(feats, w, cols,
                                    lambda td: td.val and not td.flagged),
        'auc_val_flagged': model_auc(feats, w, cols,
                                     lambda td: td.val and td.flagged),
        'soft_r': model_soft_r(feats, w, cols),
        'w': w,
    }


def per_turn_scores(feats, w, cols):
    """Per val-turn (sel, drp, soft_x, soft_y) arrays for bootstrapping."""
    rows = []
    for td, X in feats:
        if not td.val:
            continue
        s = X[:, cols] @ w
        m = np.isfinite(td.soft) & np.isfinite(s)
        rows.append((s[td.sel], s[~td.sel],
                     s[m] if m.sum() > 2 else np.empty(0),
                     td.soft[m] if m.sum() > 2 else np.empty(0)))
    return rows


def boot_stats(rows_a, rows_b, n_boot=N_BOOT, seed=SEED):
    """Paired turn-level bootstrap of (ΔAUC, Δsoft_r) between two models
    scored on the same val turns. Returns (d_auc mean±sd, d_soft mean±sd)."""
    rng = np.random.default_rng(seed)
    n = len(rows_a)
    d_auc, d_soft = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats = []
        for rows in (rows_a, rows_b):
            sel = np.concatenate([rows[i][0] for i in idx])
            drp = np.concatenate([rows[i][1] for i in idx])
            sx = np.concatenate([rows[i][2] for i in idx])
            sy = np.concatenate([rows[i][3] for i in idx])
            r = (float(np.corrcoef(sx, sy)[0, 1])
                 if len(sx) > 2 and np.std(sx) > 0 else 0.0)
            stats.append((auc(sel, drp), r))
        d_auc.append(stats[0][0] - stats[1][0])
        d_soft.append(stats[0][1] - stats[1][1])
    return (float(np.mean(d_auc)), float(np.std(d_auc)),
            float(np.mean(d_soft)), float(np.std(d_soft)))


def eval_static(turns, cfg, gains):
    """q1 evaluate() with a gains override (val slice + soft_r only)."""
    w = weights(cfg)
    sel, drp, sx, sy = [], [], [], []
    for td in turns:
        s = score_turn(td, cfg, w, gains=gains)
        if td.val:
            sel.append(s[td.sel])
            drp.append(s[~td.sel])
        m = np.isfinite(td.soft) & np.isfinite(s)
        if m.sum() > 2:
            sx.append(s[m])
            sy.append(td.soft[m])
    return {'auc_val': auc(np.concatenate(sel), np.concatenate(drp)),
            'soft_r': float(np.corrcoef(np.concatenate(sx),
                                        np.concatenate(sy))[0, 1])}


def verdict(d_auc_p, sd_p, d_soft, sd_s):
    """Classify a REMOVED delta pair (full − removed): positive = the unit
    was contributing on that axis."""
    sig_p = d_auc_p > 2 * sd_p
    sig_s = d_soft > 2 * sd_s
    neg_p = d_auc_p < -2 * sd_p
    neg_s = d_soft < -2 * sd_s
    if sig_p and sig_s:
        return 'REAL (both axes)'
    if sig_p and not sig_s:
        return 'ECHO-LEANING (picked only)'
    if sig_s and not sig_p:
        return 'QUALITY-ONLY (soft only)'
    if neg_p or neg_s:
        return 'HARMFUL (improves when removed)'
    return 'noise / dead weight'


def titles_for(node_ids):
    b = open_brain_ro()
    q = ','.join('?' * len(node_ids))
    t = dict(b.execute(
        'SELECT id, title FROM nodes WHERE id IN (%s)' % q,
        list(node_ids)).fetchall())
    b.close()
    return t


def eyeball(feats, w_full, cols_full, w_abl, cols_abl, unit_name, lines):
    """The 2 val turns where removing the unit churns the top-5 most:
    side-by-side top-8, ✓ = picked, soft value if labeled."""
    churn = []
    for k, (td, X) in enumerate(feats):
        if not td.val or len(td.cands) < 8:
            continue
        top_f = set(np.argsort(-(X[:, cols_full] @ w_full))[:5])
        top_a = set(np.argsort(-(X[:, cols_abl] @ w_abl))[:5])
        churn.append((len(top_f - top_a), k))
    churn.sort(reverse=True)
    for _, k in churn[:2]:
        td, X = feats[k]
        sf = X[:, cols_full] @ w_full
        sa = X[:, cols_abl] @ w_abl
        tmap = titles_for(td.cands)
        lines.append('')
        lines.append('### eyeball · %s · %s/%s/%s' % (unit_name, *td.key))
        lines.append('| # | with unit | without unit |')
        lines.append('|---|---|---|')
        of, oa = np.argsort(-sf)[:8], np.argsort(-sa)[:8]

        def fmt(i):
            t = (tmap.get(td.cands[i]) or td.cands[i])[:55]
            mark = ' ✓' if td.sel[i] else ''
            soft = (' s=%.2f' % td.soft[i]
                    if np.isfinite(td.soft[i]) else '')
            return '%s%s%s' % (t, mark, soft)
        for r in range(8):
            lines.append('| %d | %s | %s |' % (r + 1, fmt(of[r]), fmt(oa[r])))


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    walker.close()
    print('features over %d turns...' % len(turns))
    feats = build(turns)
    D_pick = pairs_picked(feats, train_only=False)
    D_soft = pairs_soft(feats, train_only=False)
    print('train pairs: picked %d · soft %d' % (len(D_pick), len(D_soft)))

    lines = ['# component_audit — Leg A: lane × slot contributions', '',
             '- unit = (lane, slot) cell; moment = the full stack fired '
             'per message slot, meshed (turn-meshing pin)',
             '- removed = refit full-minus-unit, Δ vs full fit; '
             'alone = j0-stack + unit (history) or unit-only (j0/lane)',
             '- both targets: picked (echo-prone) + soft-usage '
             '(answer-need); paired turn-bootstrap ×%d, band = 2σ'
             % N_BOOT, '']

    # ---- reference fits
    refs, ptrows = {}, {}
    for tgt, D in (('picked', D_pick), ('soft', D_soft)):
        refs[('full', tgt)] = fit_eval(D, feats, ALL_COLS, tgt)
        refs[('j0', tgt)] = fit_eval(D, feats, J0_COLS + [len(FEATURES) - 1],
                                     tgt)
        for k in ('full', 'j0'):
            r = refs[(k, tgt)]
            cols = ALL_COLS if k == 'full' else J0_COLS + [len(FEATURES) - 1]
            ptrows[(k, tgt)] = per_turn_scores(feats, r['w'], cols)
    lines.append('## reference fits (June+ val)')
    lines.append('| ref | target | val AUC | soft_r |')
    lines.append('|---|---|---|---|')
    for (k, tgt), r in refs.items():
        lines.append('| %s | %s | %.4f | %.3f |'
                     % (k, tgt, r['auc_val'], r['soft_r']))
    lines.append('')

    # ---- static dials view (shipped compositions, lane gain zeroed)
    lines.append('## shipped dials — winner-static & K0-static, lane → 0')
    lines.append('| lane zeroed | winner ΔAUC val | winner Δsoft_r | '
                 'K0 ΔAUC val | K0 Δsoft_r |')
    lines.append('|---|---|---|---|---|')
    cfg_k0 = configs()[0]
    cfg_win = next(c for c in configs() if c['name'] == WINNER)
    base_w = eval_static(turns, cfg_win, dict(GAINS))
    base_k = eval_static(turns, cfg_k0, dict(GAINS))
    static = {}
    for ln in LANES:
        g = {k: (0.0 if k == ln else v) for k, v in GAINS.items()}
        ew = eval_static(turns, cfg_win, g)
        ek = eval_static(turns, cfg_k0, g)
        static[ln] = {'winner_d_auc': ew['auc_val'] - base_w['auc_val'],
                      'winner_d_soft': ew['soft_r'] - base_w['soft_r'],
                      'k0_d_auc': ek['auc_val'] - base_k['auc_val'],
                      'k0_d_soft': ek['soft_r'] - base_k['soft_r']}
        s = static[ln]
        lines.append('| %s | %+.4f | %+.3f | %+.4f | %+.3f |'
                     % (ln, -s['winner_d_auc'], -s['winner_d_soft'],
                        -s['k0_d_auc'], -s['k0_d_soft']))
        print('static %s done' % ln)
    lines.append('')
    lines.append('(positive = the lane contributes at shipped dials — '
                 'zeroing it costs that much)')
    lines.append('')

    # ---- fitted ablation loop
    results = {}
    n_feat = len(FEATURES)
    for name, kind, cols in units():
        u = {'kind': kind}
        for tgt, D in (('picked', D_pick), ('soft', D_soft)):
            rem_cols = [c for c in ALL_COLS if c not in cols]
            rem = fit_eval(D, feats, rem_cols, tgt)
            rows_rem = per_turn_scores(feats, rem['w'], rem_cols)
            da, sda, ds, sds = boot_stats(ptrows[('full', tgt)], rows_rem)
            u[tgt] = {
                'removed_auc': rem['auc_val'],
                'd_auc': refs[('full', tgt)]['auc_val'] - rem['auc_val'],
                'd_auc_sd': sda,
                'removed_soft_r': rem['soft_r'],
                'd_soft': refs[('full', tgt)]['soft_r'] - rem['soft_r'],
                'd_soft_sd': sds,
            }
            # alone arm
            in_j0 = all(c in J0_COLS or c == n_feat - 1 for c in cols)
            alone_cols = (cols if kind == 'row' or in_j0
                          else sorted(set(J0_COLS + cols)))
            al = fit_eval(D, feats, alone_cols, tgt)
            u[tgt]['alone_auc'] = al['auc_val']
            u[tgt]['alone_soft_r'] = al['soft_r']
            u[tgt]['alone_base'] = ('none' if kind == 'row' or in_j0
                                    else 'j0')
        u['verdict'] = verdict(u['picked']['d_auc'], u['picked']['d_auc_sd'],
                               u['soft']['d_soft'], u['soft']['d_soft_sd'])
        results[name] = u
        print('%-22s removed: dAUC %+ .4f (sd %.4f) dsoft %+ .3f  -> %s'
              % (name, u['picked']['d_auc'], u['picked']['d_auc_sd'],
                 u['soft']['d_soft'], u['verdict']))

    # ---- report tables
    def table(kind, title):
        lines.append('## %s' % title)
        lines.append('| unit | removed: ΔAUC±sd (picked) | removed: '
                     'Δsoft_r±sd | alone AUC (picked) | alone soft_r | '
                     'alone base | verdict |')
        lines.append('|---|---|---|---|---|---|---|')
        for name, u in results.items():
            if u['kind'] != kind:
                continue
            lines.append(
                '| %s | %+.4f ± %.4f | %+.3f ± %.3f | %.4f | %.3f | %s '
                '| %s |'
                % (name, u['picked']['d_auc'], u['picked']['d_auc_sd'],
                   u['soft']['d_soft'], u['soft']['d_soft_sd'],
                   u['picked']['alone_auc'], u['soft']['alone_soft_r'],
                   u['picked']['alone_base'], u['verdict']))
        lines.append('')

    table('row', 'rows — lane across the whole moment')
    table('col', 'columns — slot across all lanes')

    lines.append('## cells — removed ΔAUC (picked) / Δsoft_r (soft) grid')
    lines.append('| lane | ' + ' | '.join(SLOTS) + ' |')
    lines.append('|' + '---|' * (len(SLOTS) + 1))
    for ln in LANES:
        row = []
        for sl in SLOTS:
            u = results['cell:%s·%s' % (ln, sl)]
            row.append('%+.3f / %+.3f'
                       % (u['picked']['d_auc'], u['soft']['d_soft']))
        lines.append('| %s | %s |' % (ln, ' | '.join(row)))
    u = results['M_e_f']
    lines.append('- M_e_f: %+.4f / %+.3f — %s'
                 % (u['picked']['d_auc'], u['soft']['d_soft'], u['verdict']))
    lines.append('')

    # ---- eyeballs on the two most load-bearing rows by |d_soft|
    top_rows = sorted(
        (n for n, u in results.items() if u['kind'] == 'row'),
        key=lambda n: -abs(results[n]['soft']['d_soft']))[:2]
    for name in top_rows:
        cols = [i for i, f in enumerate(FEATURES)
                if f.startswith(name.split(':')[1] + '·')]
        rem_cols = [c for c in ALL_COLS if c not in cols]
        w_full = refs[('full', 'soft')]['w']
        w_abl = fit_logistic(D_soft[:, rem_cols])
        eyeball(feats, w_full, ALL_COLS, w_abl, rem_cols, name, lines)
    lines.append('')

    OUT.write_text(json.dumps(
        {'results': {k: {t: {kk: vv for kk, vv in v[t].items()
                             if kk != 'w'}
                         for t in ('picked', 'soft')} | {
                             'kind': v['kind'], 'verdict': v['verdict']}
                     for k, v in results.items()},
         'static': static,
         'refs': {('%s_%s' % k): {'auc_val': r['auc_val'],
                                  'soft_r': r['soft_r']}
                  for k, r in refs.items()}}, indent=1))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
