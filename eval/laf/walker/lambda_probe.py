"""The λ* oracle — first measurement of Tom's integrate-function reframe
(9cba610e, 2026-07-20): msg 0 is an UPDATE EVENT on the standing Moment;
the object is the per-event update gain, not a choice between fields.

score(λ) = (1−λ)·z(M_h) + λ·z(F0)   over alive nodes, λ ∈ [0,1] grid.
Both fields z-normalized first so λ is a pure mixture dial, comparable
across turns.

Outputs:
  1. fixed-λ curve — reach@5 per static λ (the 1-param re-baseline of the
     flat Moment; answers 'is 1:1 F0+M_h even the right static mix?')
  2. oracle-λ reach — per-turn best λ (the recursive-form ceiling)
  3. λ* plateau distribution — how much per-event plasticity varies
  4. readouts → λ* Spearman on decisive turns (which readout tracks the
     update gain — the derivation target)

Run: ./dev python3 eval/laf/walker/lambda_probe.py    (pool60 via env)
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from mesh_fit_probe import Turn                                     # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def zn(f):
    m = np.isfinite(f)
    o = np.full_like(f, np.nan)
    if m.sum() > 2 and np.std(f[m]) > 1e-9:
        o[m] = (f[m] - f[m].mean()) / f[m].std()
    return o


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def lambda_star(f0z, mhz, gr, grid=GRID):
    """Per-turn λ oracle: gold rank per λ over the z-mixed field.
    THE λ* machinery — layer_readout_probe imports this; never re-implement
    (the wsum rule). Known small endpoint understatement at λ∈{0,1} from
    inf·0 — interior clean. Returns {λ: rank} (empty if gold unscorable)."""
    both = np.isfinite(f0z) & np.isfinite(mhz)
    f0w = np.where(both, f0z, np.where(np.isfinite(f0z), f0z, -np.inf))
    mhw = np.where(both, mhz, np.where(np.isfinite(mhz), mhz, -np.inf))
    ranks = {}
    for l in grid:
        s = (1 - l) * mhw + l * f0w
        if not np.isfinite(s[gr]):
            continue
        ranks[l] = int((s > s[gr]).sum()) + 1
    return ranks


def plateau_of(ranks):
    """(lo, hi, mid) of the λ*-plateau — the flat argmin region."""
    best = min(ranks.values())
    pl = [l for l, r in ranks.items() if r == best]
    return min(pl), max(pl), float(np.median(pl))


def main():
    idx = json.loads(INDEX.read_text())
    fields = np.load(CACHE, mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    turns = []
    for t in idx['turns']:
        if t.get('skipped'):
            continue
        tt = Turn(t, fields, S)
        if tt.gr >= 0 and tt.ro is not None and tt.mh is not None \
                and not np.isnan(tt.fields[0]).all():
            turns.append(tt)
    print('turns with standing field + msg0: %d' % len(turns))

    per_l_hits = {l: 0 for l in GRID}
    lam_mid, lam_lo, lam_hi, oracle_hit = [], [], [], 0
    ros, strong = [], []
    s_hits = {l: 0 for l in GRID}
    s_oracle = 0
    for tt in turns:
        ranks = lambda_star(zn(tt.fields[0]), zn(tt.mh), tt.gr)
        if not ranks:
            continue
        lo, hi, mid = plateau_of(ranks)
        lam_lo.append(lo)
        lam_hi.append(hi)
        lam_mid.append(mid)
        oracle_hit += int(min(ranks.values()) <= 5)
        for l, r in ranks.items():
            per_l_hits[l] += int(r <= 5)
        if tt.strong:               # gold-tier: soft-gold ∩ Haiku-picked
            s_oracle += int(min(ranks.values()) <= 5)
            for l, r in ranks.items():
                s_hits[l] += int(r <= 5)
        strong.append(tt.strong)
        ros.append(tt.ro)

    n = len(lam_mid)
    print('\n== 1. fixed-λ curve (reach@5, %d turns) ==' % n)
    row = ['  λ    '] + ['%.2f' % l for l in GRID[::2]]
    print(' '.join(row))
    print('  hit%  ' + ' '.join('%4.1f' % (100 * per_l_hits[l] / n)
                                for l in GRID[::2]))
    best_static = max(GRID, key=lambda l: per_l_hits[l])
    print('  best static λ = %.2f → %.1f%% · oracle-λ → %.1f%% '
          '(headroom %.1fpp)'
          % (best_static, 100 * per_l_hits[best_static] / n,
             100 * oracle_hit / n,
             100 * (oracle_hit - per_l_hits[best_static]) / n))
    ns = sum(strong)
    if ns:
        bs = max(GRID, key=lambda l: s_hits[l])
        print('  gold-strong tier (soft ∩ Haiku-picked, n=%d): static '
              'λ=%.2f → %.1f%% · oracle-λ → %.1f%% (headroom %.1fpp)'
              % (ns, bs, 100 * s_hits[bs] / ns, 100 * s_oracle / ns,
                 100 * (s_oracle - s_hits[bs]) / ns))

    lo, hi, mid = np.array(lam_lo), np.array(lam_hi), np.array(lam_mid)
    width = hi - lo
    decisive = width <= 0.5
    print('\n== 2. λ* plateau distribution ==')
    print('  plateau includes λ=0 (pure Moment): %.0f%% · includes λ=1 '
          '(pure msg0): %.0f%% · interior-only: %.0f%%'
          % (100 * (lo == 0).mean(), 100 * (hi == 1).mean(),
             100 * ((lo > 0) & (hi < 1)).mean()))
    print('  decisive turns (plateau width <= 0.5): %d (%.0f%%) · their '
          'λ*-mid histogram:' % (decisive.sum(), 100 * decisive.mean()))
    hh, edges = np.histogram(mid[decisive], bins=np.arange(0, 1.1, 0.1))
    print('  ' + ' '.join('%.1f:%d' % (e, c)
                          for e, c in zip(edges[:-1], hh)))

    print('\n== 3. readouts → λ* (Spearman, decisive turns n=%d) =='
          % int(decisive.sum()))
    for k in ros[0]:
        v = np.array([r[k] for r in ros])
        print('  %-12s %+0.3f' % (k, spearman(v[decisive], mid[decisive])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
