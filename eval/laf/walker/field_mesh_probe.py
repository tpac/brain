"""Phase 2 of the mesh work: field-level arms, the FIELD-LEVEL ORACLE, and
the READOUT MENU — all computed from field_cache.npy (free re-runs).

Arms (gold rank over the full alive field):
  F0 / F1(op-1) / A1(anchor-1) / M_h (history-only moment, exp0.5) /
  M_full (flat linear mesh = F0 + M_h — the static blend baseline) /
  turnmax mesh / oracle-5way / oracle-2way {F0 vs M_full} (router headroom).

Readout menu (Tom's numbering, 2026-07-20):
  1 align(F0, M_h)          — msg0 vs the Moment
  2 align(F1, M_h_excl1)    — msg-1 vs the Moment built without it
  3 the difference of 1 and 2 (and ratio)
  4 overlap/containment/align between F0 and F1
  5 post-mesh conc(M_full) + unary conc/peak per field
Readout-tracks-oracle: AUC of each readout predicting the 2-way oracle side.

Run: ./dev python3 eval/laf/walker/field_mesh_probe.py [--smoke]
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR

CACHE = OUT_DIR / 'field_cache.npy'          # DATA artifact — env-honoring
INDEX = OUT_DIR / 'field_cache_index.json'
GAMMA = 0.5                      # Q1-winner exp decay
TOPK_SRC = 100                   # containment source set
TOPK_TGT = 500                   # containment target region
REACH = (5, 25)


def relu(f):
    return np.where(np.isfinite(f) & (f > 0), f, 0.0)


def topset(f, k):
    fin = np.where(np.isfinite(f), f, -np.inf)
    if not np.isfinite(fin).any():
        return set()
    k = min(k, int(np.isfinite(f).sum()))
    return set(np.argpartition(-fin, k - 1)[:k].tolist())


def align(f, g):
    a, b = relu(f), relu(g)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else np.nan


def conc(f, k=50):
    a = relu(f)
    tot = a.sum()
    if tot < 1e-9:
        return np.nan
    k = min(k, len(a))
    return float(np.sort(a)[-k:].sum() / tot)


def containment(f, g):
    """share of f's top-100 that lies inside g's top-500 region."""
    ts, tg = topset(f, TOPK_SRC), topset(g, TOPK_TGT)
    return len(ts & tg) / len(ts) if ts else np.nan


def gold_rank(f, gr):
    if gr < 0 or not np.isfinite(f[gr]):
        return None
    return int((np.where(np.isfinite(f), f, -np.inf) > f[gr]).sum()) + 1


def wsum(parts):
    """weighted nansum of (weight, field) — absent (all-NaN) parts drop;
    returns None if nothing present. Nodes finite in NO part (dead at the
    turn's as_of — NaN in every slot) stay NaN: zero-filling them made
    them score-0 COMPETITORS in composite ranks while raw single-slot
    fields correctly excluded them (2026-07-20 review BLOCKER — biased
    every F0-vs-Moment delta toward F0)."""
    acc, present = None, None
    for w, f in parts:
        if f is None or np.isnan(f).all():
            continue
        fin = np.isfinite(f)
        x = w * np.where(fin, f, 0.0)
        acc = x if acc is None else acc + x
        present = fin if present is None else (present | fin)
    if acc is None:
        return None
    acc[~present] = np.nan
    return acc


def main():
    smoke = '--smoke' in sys.argv
    sfx = '.smoke' if smoke else ''
    idx = json.loads(Path(str(INDEX) + sfx).read_text())
    fields = np.load(str(CACHE) + sfx, mmap_mode='r')
    slots = idx['slots']
    S = {s: i for i, s in enumerate(slots)}
    turns = [t for t in idx['turns'] if not t.get('skipped')]
    print('turns %d · slots %s · n_nodes %d'
          % (len(turns), slots, idx['n_nodes']))

    arm_names = ['F0', 'F1(op-1)', 'A1(anchor-1)', 'M_h(hist)',
                 'M_full(blend)', 'turnmax', 'oracle-5way', 'oracle-2way']
    ranks = {a: [] for a in arm_names}
    pool_hits = {a: [] for a in arm_names}
    readouts, side_labels = [], []
    win_of = {a: 0 for a in ('F0', 'F1', 'A1', 'M_h', 'M_full')}

    for t in turns:
        F = fields[t['row']].astype(np.float32)
        f0, f1, a1, f2 = (F[S['op0']], F[S['op1']], F[S['anchor1']],
                          F[S['op2']])
        gr = t['cand_rows'][t['gold_i']]
        if gr < 0:
            continue
        mh = wsum([(GAMMA, f1), (GAMMA, a1), (GAMMA ** 2, f2)])
        mh_ex1 = wsum([(GAMMA, a1), (GAMMA ** 2, f2)])
        mfull = wsum([(1.0, f0), (1.0, mh)])
        tmax = None
        parts = [(1.0, f0), (GAMMA, f1), (GAMMA, a1), (GAMMA ** 2, f2)]
        pres = [w * np.where(np.isfinite(f), f, -np.inf)
                for w, f in parts if f is not None and not np.isnan(f).all()]
        if pres:
            tmax = np.max(np.stack(pres), axis=0)
            tmax[~np.isfinite(tmax)] = np.nan

        base = {'F0': f0, 'F1': f1, 'A1': a1, 'M_h': mh, 'M_full': mfull}
        rk = {k: gold_rank(f, gr) if f is not None else None
              for k, f in base.items()}
        rk['turnmax'] = gold_rank(tmax, gr) if tmax is not None else None
        cand = {k: (rk[k] or 10 ** 9) for k in base}
        rk['oracle-5way'] = min(cand.values())
        rk['oracle-2way'] = min(cand['F0'], cand['M_full'])
        best = min(cand, key=cand.get)
        if cand[best] < 10 ** 9:
            win_of[best] += 1

        for arm, key in zip(arm_names, ['F0', 'F1', 'A1', 'M_h', 'M_full',
                                        'turnmax', 'oracle-5way',
                                        'oracle-2way']):
            r = rk[key]
            if r is not None and r < 10 ** 9:
                ranks[arm].append(r)
        # pool-restricted hit@5 for comparability
        crows = [r for r in t['cand_rows'] if r >= 0]
        for arm, key in zip(arm_names, ['F0', 'F1', 'A1', 'M_h', 'M_full',
                                        'turnmax', 'oracle-5way',
                                        'oracle-2way']):
            f = base.get(key)
            if key == 'turnmax':
                f = tmax
            if key.startswith('oracle') or f is None or np.isnan(f).all():
                continue
            cs = f[crows]
            if not np.isfinite(cs[crows.index(gr)] if gr in crows else
                               np.nan):
                continue
            pool_hits[arm].append(
                int((np.where(np.isfinite(cs), cs, -np.inf)
                     > f[gr]).sum()) < 5)

        # ---- readout menu (only where both sides exist)
        if mh is None or np.isnan(f0).all():
            continue
        r0, rM = rk['F0'] or 10 ** 9, cand['M_full']
        if r0 == rM:
            side = None
        else:
            side = int(r0 < rM)          # 1 = msg0 side wins
        ro = {
            'a_F0_Mh': align(f0, mh),                              # menu 1
            'a_F1_Mhx': (align(f1, mh_ex1)
                         if f1 is not None and mh_ex1 is not None
                         and not np.isnan(f1).all() else np.nan),  # menu 2
            'peak_F0': float(np.nanmax(f0)),
            'peak_Mh': float(np.nanmax(mh)),
            'conc_F0': conc(f0), 'conc_Mh': conc(mh),
            'ov_F0_F1': (len(topset(f0, 100) & topset(f1, 100)) / 100
                         if f1 is not None and not np.isnan(f1).all()
                         else np.nan),                              # menu 4
            'cont_F0_in_F1': (containment(f0, f1)
                              if f1 is not None
                              and not np.isnan(f1).all() else np.nan),
            'cont_F1_in_F0': (containment(f1, f0)
                              if f1 is not None
                              and not np.isnan(f1).all() else np.nan),
            'a_F0_F1': (align(f0, f1) if f1 is not None
                        and not np.isnan(f1).all() else np.nan),
            'conc_Mfull': conc(mfull),                              # menu 5
            'cont_F0_in_Mh': containment(f0, mh),
        }
        ro['d_align'] = ro['a_F0_Mh'] - ro['a_F1_Mhx']              # menu 3
        ro['d_peak'] = ro['peak_F0'] - ro['peak_Mh']
        ro['d_conc'] = ro['conc_F0'] - ro['conc_Mh']
        readouts.append(ro)
        side_labels.append(side)

    print('\n== field-level arms (gold rank over full alive field) ==')
    print('arm             n     reach@5   reach@25   median')
    for a in arm_names:
        r = np.array(ranks[a])
        ph = np.array(pool_hits[a]) if pool_hits[a] else None
        print('  %-14s %4d   %5.1f%%    %5.1f%%     %5.0f%s'
              % (a, len(r), 100 * (r <= 5).mean(), 100 * (r <= 25).mean(),
                 np.median(r),
                 ('   (pool@5 %.1f%%)' % (100 * ph.mean())
                  if ph is not None else '')))
    tot = sum(win_of.values())
    print('\n5-way oracle winner shares: '
          + ' · '.join('%s %.0f%%' % (k, 100 * v / max(1, tot))
                       for k, v in win_of.items()))

    # ---- readout-tracks-oracle
    lab = np.array([s for s in side_labels if s is not None])
    keep = [i for i, s in enumerate(side_labels) if s is not None]
    print('\n== readouts → 2-way oracle side (1 = msg0 wins; n=%d, '
          'base rate %.2f) ==' % (len(lab), lab.mean()))
    from soft_usage import auc
    rows = []
    for k in readouts[0]:
        v = np.array([readouts[i][k] for i in keep])
        m = np.isfinite(v)
        if m.sum() < 20 or lab[m].std() < 1e-9:
            continue
        a = auc(v[m & (lab == 1)], v[m & (lab == 0)])
        rows.append((abs(a - 0.5), k, a, int(m.sum())))
    for _, k, a, nn in sorted(rows, reverse=True):
        print('  %-16s AUC %.3f  (n=%d)' % (k, a, nn))
    return 0


if __name__ == '__main__':
    sys.exit(main())
