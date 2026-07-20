"""Gated mesh probe — Tom's insight (2026-07-17): a pivot is apparent
from the graph itself; the mesh should read field agreement, not add
blindly.

DETECTOR (label-free, cue-side only): agreement between the current
message's composed field F0 and each history field Fj —
  spearman: rank correlation over the pool
  jaccard : top-q set overlap
Graph-native variant (edge-connectivity between top sets) noted for the
full-field rig; pool candidates are too few edges for it here.

STEP 1 — validate the detector BEFORE gating on it:
  condition (moment − K0) target-rank delta and soft_r on agreement
  quartiles. The detector is real only if low-agreement turns are where
  the moment hurts. Also: the four named eyeball MOMENT-HURT cases must
  land in the low-agreement half.

STEP 2 — gated mesh arms (only meaningful if step 1 passes):
  linear      : Σ w_j·F_j (control)
  fitted      : per-message fitted weights (the reweighting champion —
                the gate must beat THIS, not just linear; renorm lesson)
  gate-cont   : w_j ← w_j · max(0, ρ_j)      (agreement-scaled)
  gate-hard   : history zeroed when mean ρ < THRESH
  gate-fitted : fitted weights × continuous gate

Run:  ./dev python3 eval/laf/walker/gated_mesh.py
Out:  gated_mesh.md, gated_mesh.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import load, gate_provenance, configs, weights        # noqa: E402
from moment_grids import cfg_for, message_fields, eval_arm          # noqa: E402
from p3_fit import fit_logistic                                     # noqa: E402
from soft_usage import auc                                          # noqa: E402

CFG = cfg_for(8, 0.7)             # deep-K base: gating has room to act
THRESH = 0.10                     # hard-gate cutoff on mean agreement
TOPQ = 5                          # jaccard top-q
SOFT_MARGIN = 0.10
HURT_CASES = {('ad249ee4', 1, 6), ('124cf35a', 0, 18),
              ('c2244e8e', 0, 16), ('9ec0b4e8', 0, 10)}
REPORT = OUT_DIR / 'gated_mesh.md'
OUT = OUT_DIR / 'gated_mesh.json'


def spearman(a, b):
    fin = np.isfinite(a) & np.isfinite(b)
    if fin.sum() < 4:
        return np.nan
    ra = np.argsort(np.argsort(a[fin])).astype(float)
    rb = np.argsort(np.argsort(b[fin])).astype(float)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def agreements(F):
    """Per-history-column agreement with F0: (rho[j], jaccard[j])."""
    f0 = F[:, 0]
    rhos, jacs = [], []
    top0 = set(np.argsort(-np.where(np.isfinite(f0), f0, -np.inf))[:TOPQ])
    for col in range(1, F.shape[1]):
        fj = F[:, col]
        if not np.isfinite(fj).any():
            rhos.append(np.nan)
            jacs.append(np.nan)
            continue
        rhos.append(spearman(f0, fj))
        topj = set(np.argsort(-np.where(np.isfinite(fj), fj,
                                        -np.inf))[:TOPQ])
        jacs.append(len(top0 & topj) / len(top0 | topj))
    return np.array(rhos), np.array(jacs)


def mesh_gated(F, ww, rhos, mode, w_fit=None):
    nc, n_msg = F.shape
    w = np.array(ww, dtype=float) if w_fit is None else w_fit.copy()
    if mode == 'gate-cont' or mode == 'gate-fitted':
        for col in range(1, n_msg):
            r = rhos[col - 1]
            w[col] *= max(0.0, r) if np.isfinite(r) else 0.0
    elif mode == 'gate-hard':
        fin = rhos[np.isfinite(rhos)]
        if fin.size and fin.mean() < THRESH:
            w[1:] = 0.0
    with np.errstate(all='ignore'):
        v = np.nansum(F * w, axis=1)
    v[np.all(np.isnan(F), axis=1)] = np.nan
    return v


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    walker.close()

    w = weights(CFG)
    cache, agree = {}, {}
    for td in turns:
        F, ww = message_fields(td, CFG, w)
        cache[td.key] = (F, ww)
        agree[td.key] = agreements(F)
    lines = ['# gated_mesh — field-agreement pivot detector + gated mesh',
             '', '- base %s · gate = spearman(F0, Fj) over the pool · '
             'hard-gate thresh %.2f' % (CFG['name'], THRESH), '']

    # ---------- step 1: detector validation
    cfg_k0 = configs()[0]
    w0 = weights(cfg_k0)
    rows = []
    for td in turns:
        if not td.val or not np.isfinite(td.soft).any():
            continue
        F, ww = cache[td.key]
        rhos, _ = agree[td.key]
        fin = rhos[np.isfinite(rhos)]
        if not fin.size:
            continue
        from moment_grids import mesh
        s1 = mesh(F, ww, 'linear')
        F0 = F[:, :1]
        s0 = mesh(F0, ww[:1], 'linear')
        tgt = int(np.nanargmax(td.soft))
        if td.soft[tgt] < 0.5:
            continue

        def rank_of(s, i):
            o = np.argsort(-np.where(np.isfinite(s), s, -np.inf))
            return int(np.where(o == i)[0][0]) + 1
        rows.append({'key': td.key, 'rho': float(fin.mean()),
                     'd_rank': rank_of(s1, tgt) - rank_of(s0, tgt)})
    rho_all = np.array([r['rho'] for r in rows])
    qs = np.percentile(rho_all, [25, 50, 75])
    lines.append('## step 1 — detector validation '
                 '(Δ target-rank = moment − j0-only; negative = moment '
                 'helped)')
    lines.append('| agreement quartile | mean ρ | n | mean Δrank | '
                 '%turns moment hurt (Δ>+2) | %helped (Δ<−2) |')
    lines.append('|---|---|---|---|---|---|')
    det = []
    for b in range(4):
        lo = -np.inf if b == 0 else qs[b - 1]
        hi = np.inf if b == 3 else qs[b]
        sub = [r for r in rows if lo < r['rho'] <= hi]
        d = np.array([r['d_rank'] for r in sub])
        det.append({'q': b + 1, 'mean_rho': float(np.mean(
            [r['rho'] for r in sub])), 'n': len(sub),
            'mean_d': float(d.mean()),
            'hurt': float((d > 2).mean()), 'helped': float((d < -2).mean())})
        lines.append('| Q%d | %.3f | %d | %+.2f | %.0f%% | %.0f%% |'
                     % (b + 1, det[-1]['mean_rho'], len(sub), d.mean(),
                        100 * det[-1]['hurt'], 100 * det[-1]['helped']))
    lines.append('')
    named = [(r['key'], r['rho'],
              float((rho_all < r['rho']).mean()))
             for r in rows if (r['key'][0][:8],) + r['key'][1:]
             in HURT_CASES]
    lines.append('- named eyeball MOMENT-HURT cases (ρ percentile in '
                 'population): %s'
                 % ', '.join('%s→ρ=%.2f (p%d)' % (k[0][:8], rho, int(100 * p))
                             for k, rho, p in named))
    lines.append('')
    print('step 1 done')

    # ---------- step 2: gated arms
    # fitted per-message weights (reweighting champion, train-only)
    n_msg = next(iter(cache.values()))[0].shape[1]
    drows = []
    for td in turns:
        if td.val:
            continue
        F, _ = cache[td.key]
        X = np.where(np.isfinite(F), F, 0.0)
        fin = np.flatnonzero(np.isfinite(td.soft))
        if len(fin) < 2:
            continue
        s = td.soft[fin]
        d = s[:, None] - s[None, :]
        wi, li = np.nonzero(d >= SOFT_MARGIN)
        if len(wi):
            drows.append(X[fin[wi]] - X[fin[li]])
    w_fit = fit_logistic(np.concatenate(drows))

    def arm(mode, use_fit=False):
        def score(td):
            F, ww = cache[td.key]
            rhos, _ = agree[td.key]
            return mesh_gated(F, ww, rhos, mode,
                              w_fit=w_fit if use_fit else None)
        return eval_arm(turns, CFG, w, score_fn=score)

    lines.append('## step 2 — gated mesh vs linear vs fitted '
                 '(val turns)')
    lines.append('| arm | sel@1 | sel-in-5 | AUC | soft_r |')
    lines.append('|---|---|---|---|---|')
    results = {}
    for name, mode, use_fit in (
            ('linear', 'linear', False),
            ('fitted (reweight champion)', 'linear', True),
            ('gate-cont', 'gate-cont', False),
            ('gate-hard', 'gate-hard', False),
            ('gate-fitted', 'gate-fitted', True)):
        m = arm(mode, use_fit)
        results[name] = m
        lines.append('| %s | %.3f | %.3f | %.4f | %.3f |'
                     % (name, m['sel_at_1'], m['sel_in_5'], m['auc'],
                        m['soft_r']))
        print('arm %s done' % name)
    lines.append('')
    lines.append('- coverage: turns with no history score identically '
                 'across all arms by construction (gate acts on w[1:]).')

    OUT.write_text(json.dumps({'detector': det, 'named_hurt': [
        {'key': list(k), 'rho': r, 'pct': p} for k, r, p in named],
        'arms': results}, indent=1))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
