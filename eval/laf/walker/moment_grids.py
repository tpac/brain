"""Tom's four questions (2026-07-17) — stop-cue recall, deeper K,
settling, decay factor. All pool-restricted walker math.

1. BOTH SIDES — op-cue (labels at t) vs stop-cue: score turn t's pool
   with j0 dropped and decay re-anchored at j1 (my response = freshest
   message; the operator hasn't spoken yet). Stop-cue labels are the
   SAME turn's selections/usage — at stop time those are one step in
   the future, so this measures anticipatory (endo) recall with zero
   new columns.
2. K grid {1,2,3,4,5,8} × 4. exp decay γ {0.3,0.5,0.7,0.9} — judged on
   the quality axis (soft_r) alongside placement, both cue sides.
3. SETTLING — nonlinear mesh variants vs the linear control:
   renorm (z after each message add), sparsify (per-message top-q
   survive), softmax mesh (temperature sharpening per message).

Out:  moment_grids.md, moment_grids.json
Run:  ./dev python3 eval/laf/walker/moment_grids.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import WALKER_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import (GAINS, load, gate_provenance, configs,  # noqa: E402
                      score_turn, weights, stack_messages)
from servers.recall_laf import _zscore                        # noqa: E402
from soft_usage import auc                                    # noqa: E402

REPORT = WALKER_DIR / 'moment_grids.md'
OUT = WALKER_DIR / 'moment_grids.json'
K_GRID = (1, 2, 3, 4, 5, 8)
G_GRID = (0.3, 0.5, 0.7, 0.9)
SPARSIFY_KEEP = 8            # per-message survivors in a ~25 pool
SOFTMAX_T = 0.5


def cfg_for(K, gamma):
    return {'name': 'K%d-exp%s' % (K, gamma), 'K': K,
            'decay': ('exp', gamma), 'comp': 'turnsum', 'agg': 'zsum',
            'texts': 'op+anchor', 'me': ('off', 0.0)}


def stop_weights(cfg):
    """Decay re-anchored at j1: w = [0, 1, γ, γ², ...] — j0 (the future
    operator message) carries nothing at stop time."""
    K = cfg['K']
    g = cfg['decay'][1]
    w = np.zeros(K + 1)
    for j in range(1, K + 1):
        w[j] = g ** (j - 1)
    return w


def message_fields(td, cfg, w):
    """[n_cand × M] per-message composed z-fields (zsum inner step),
    plus the message weights — the mesh input, exposed so settling
    variants can vary the mesh itself."""
    nc = len(td.cands)
    mats, ww = {}, None
    for ln in GAINS:
        mats[ln], ww = stack_messages(td.op[ln], td.anchor[ln], w, cfg)
    n_msg = next(iter(mats.values())).shape[1]
    F = np.full((nc, n_msg), np.nan)
    for col in range(n_msg):
        acc = np.zeros(nc)
        any_data = np.zeros(nc, dtype=bool)
        for ln, g in GAINS.items():
            x = mats[ln][:, col]
            acc += g * _zscore(x, nc)
            any_data |= np.isfinite(x)
        F[:, col] = np.where(any_data, acc, np.nan)
    return F, ww


def mesh(F, ww, mode):
    """Mesh per-message fields → one field. 'linear' ≡ compose turnsum."""
    nc, n_msg = F.shape
    if mode == 'linear':
        with np.errstate(all='ignore'):
            v = np.nansum(F * ww, axis=1)
        v[np.all(np.isnan(F), axis=1)] = np.nan
        return v
    if mode == 'renorm':                      # A ← z(A + w_j·F_j)
        A = np.zeros(nc)
        touched = np.zeros(nc, dtype=bool)
        for col in range(n_msg):
            f = F[:, col]
            fin = np.isfinite(f)
            if not fin.any():
                continue
            A = A + np.where(fin, ww[col] * f, 0.0)
            A = _zscore(A, nc)
            touched |= fin
        return np.where(touched, A, np.nan)
    if mode == 'sparsify':                    # per-message top-q survive
        Fs = np.full_like(F, np.nan)
        for col in range(n_msg):
            f = F[:, col]
            fin = np.isfinite(f)
            if not fin.any():
                continue
            k = min(SPARSIFY_KEEP, int(fin.sum()))
            thr = np.sort(f[fin])[-k]
            Fs[:, col] = np.where(f >= thr, f, 0.0)
            Fs[~fin, col] = np.nan
        return mesh(Fs, ww, 'linear')
    if mode == 'softmax':                     # sharpen per message
        Fs = np.full_like(F, np.nan)
        for col in range(n_msg):
            f = F[:, col]
            fin = np.isfinite(f)
            if not fin.any():
                continue
            e = np.exp((f[fin] - np.nanmax(f)) / SOFTMAX_T)
            p = np.zeros_like(f)
            p[fin] = e / e.sum()
            Fs[:, col] = _zscore(np.where(fin, p, np.nan), nc)
            Fs[~fin, col] = np.nan
        return mesh(Fs, ww, 'linear')
    raise ValueError(mode)


def eval_arm(turns, cfg, w, mode='linear', score_fn=None):
    """Placement + quality on val turns for one (cfg, weights, mesh)."""
    top1 = nsel = nturn = 0
    top5 = 0
    sx, sy, sel_p, drp_p = [], [], [], []
    for td in turns:
        if not td.val:
            continue
        if score_fn is not None:
            s = score_fn(td)
        elif mode == 'linear':
            s = score_turn(td, cfg, w)
        else:
            F, ww = message_fields(td, cfg, w)
            s = mesh(F, ww, mode)
        fin = s[np.isfinite(s)]
        if fin.size == 0 or np.ptp(fin) == 0.0:
            # all-missing OR all-tied: an uninformative field. Ranking a
            # constant would credit insertion order — which is production
            # rank_in_pool order, i.e. pure label echo on the stop side
            # (2026-07-17 audit finding 2: 139/1440 stop turns, sel@1
            # 0.561 by echo alone). Skip from placement AND soft.
            continue
        s = np.where(np.isfinite(s), s, -np.inf)
        if td.sel.any() and not td.sel.all():
            order = np.argsort(-s)
            ranks = np.empty(len(s), dtype=int)
            ranks[order] = np.arange(1, len(s) + 1)
            rsel = np.sort(ranks[td.sel])
            nturn += 1
            nsel += len(rsel)
            top1 += int(rsel[0] == 1)
            top5 += int((rsel <= 5).sum())
            sel_p.append(s[td.sel])
            drp_p.append(s[~td.sel])
        m = np.isfinite(td.soft) & np.isfinite(s)
        if m.sum() > 2:
            sx.append(s[m])
            sy.append(td.soft[m])
    return {'sel_at_1': top1 / nturn, 'sel_in_5': top5 / nsel,
            'auc': auc(np.concatenate(sel_p), np.concatenate(drp_p)),
            'soft_r': float(np.corrcoef(np.concatenate(sx),
                                        np.concatenate(sy))[0, 1]),
            'n_turns': nturn}


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    walker.close()
    lines = ['# moment_grids — stop-cue, K×γ, settling', '']
    out = {}

    # -------- 1+2+4: K × γ grid, both cue sides
    k0 = configs()[0]
    base = eval_arm(turns, k0, weights(k0))
    lines.append('## K × γ grid — op-cue (labels: this turn) vs stop-cue '
                 '(j0 dropped, labels: next moment)')
    lines.append('- op-cue K0 baseline: sel@1 %.3f · sel-in-5 %.3f · AUC '
                 '%.4f · soft_r %.3f' % (base['sel_at_1'], base['sel_in_5'],
                                         base['auc'], base['soft_r']))
    lines.append('')
    lines.append('| K | γ | op sel@1 | op soft_r | op AUC | stop sel@1 | '
                 'stop soft_r | stop AUC |')
    lines.append('|---|---|---|---|---|---|---|---|')
    grid = {}
    for K in K_GRID:
        for g in G_GRID:
            cfg = cfg_for(K, g)
            op_m = eval_arm(turns, cfg, weights(cfg))
            st_m = eval_arm(turns, cfg, stop_weights(cfg))
            grid['K%d-g%s' % (K, g)] = {'op': op_m, 'stop': st_m}
            lines.append('| %d | %s | %.3f | %.3f | %.4f | %.3f | %.3f | '
                         '%.4f |' % (K, g, op_m['sel_at_1'], op_m['soft_r'],
                                     op_m['auc'], st_m['sel_at_1'],
                                     st_m['soft_r'], st_m['auc']))
        print('K=%d done' % K)
    out['grid'] = grid
    out['k0'] = base
    lines.append('')
    lines.append('- stop-cue = anticipatory recall at my stop event: the '
                 'freshest cue is my just-finished response (j1-anchor), '
                 'the operator message that generated these labels is '
                 'excluded. K0 has no stop-cue analog (nothing to cue on).')
    lines.append('')

    # -------- 3: settling probes on the best-K op cfg + stop side
    best_key = max(grid, key=lambda k: grid[k]['op']['soft_r'])
    Kb, gb = best_key.replace('K', '').split('-g')
    cfg_b = cfg_for(int(Kb), float(gb))
    lines.append('## settling probes — mesh variants on %s' % best_key)
    lines.append('| mesh | side | sel@1 | sel-in-5 | AUC | soft_r |')
    lines.append('|---|---|---|---|---|---|')
    settle = {}
    for mode in ('linear', 'renorm', 'sparsify', 'softmax'):
        for side, w in (('op', weights(cfg_b)),
                        ('stop', stop_weights(cfg_b))):
            m = eval_arm(turns, cfg_b, w, mode=mode)
            settle['%s-%s' % (mode, side)] = m
            lines.append('| %s | %s | %.3f | %.3f | %.4f | %.3f |'
                         % (mode, side, m['sel_at_1'], m['sel_in_5'],
                            m['auc'], m['soft_r']))
            print('settle %s-%s done' % (mode, side))
    out['settling'] = settle
    lines.append('')
    lines.append('- renorm: A ← z(A + w_j·F_j) after each message; '
                 'sparsify: per-message top-%d survive, rest → 0; '
                 'softmax: per-message T=%.1f sharpening. linear = '
                 'production mesh (control).' % (SPARSIFY_KEEP, SOFTMAX_T))

    OUT.write_text(json.dumps(out, indent=1))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
